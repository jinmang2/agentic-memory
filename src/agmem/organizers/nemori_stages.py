"""Assembled Nemori pipeline stages (spec §2.3).

``PerMessageBoundary`` is the first stage extracted out of ``NemoriOrganizer``:
the online, per-message boundary detector (the paper's v1 f_theta formalism)
that decides whether the newest buffered message opens a new episode. It is
v1-equivalent with the logic previously inlined in ``NemoriOrganizer.on_message``
— same prompt, same schema, same thresholds, same buffer-splitting semantics.
``BatchPartitioner``, ``EpisodeMerger``, and the semantic integrators
(``AppendIntegrator``/``DedupIdReuseIntegrator``/``ThreeWayIntegrator``) round
out the remaining stages so the organizer becomes a thin composition of
independently testable stages.
"""

from __future__ import annotations

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, new_id
from agmem.organizers.base import OrganizerContext

BOUNDARY_SCHEMA = {
    "type": "object",
    "properties": {
        "boundary": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["boundary", "confidence"],
}

# Condensed from Nemori's segmentation criteria (BATCH_SEGMENTATION_PROMPT),
# recast for our online per-message mode (the paper v1 formalism).
BOUNDARY_PROMPT = """Decide whether the NEWEST message starts a new episode
(a different topic, intent, or time context) relative to the buffered
conversation. Be strict — high sensitivity to shifts. Signals:
- topic change (a different subject or activity)
- intent transition (e.g. information request to decision, discussion to
  casual chat)
- temporal markers ("earlier", "before", "by the way", "oh right", "also",
  or a gap of 30+ minutes between message timestamps)
- structural signals (explicit transition phrases like "changing topics",
  "speaking of which", "quick question", or a concluding statement
  indicating the current topic is finished)
- content relatedness below ~30%
Episodes work best with 2-15 messages. When in doubt, split.

Buffered conversation:
{buffer}

Newest message:
{message}

Return JSON: {{"boundary": true/false, "confidence": 0.0-1.0}}"""


def _fmt(episode: Episode) -> str:
    ts = episode.meta.get("date") or episode.timestamp.isoformat()
    return f"[{ts}] {episode.role}: {episode.content}"


class PerMessageBoundary:
    """Online per-message boundary detector (paper v1 f_theta formalism).

    ``push`` decides, for the current buffer, whether to cut one or more
    segments off the front and what remains buffered. The LLM is not injected
    at construction time — it comes from the shared ``ctx`` passed to each
    call, consistent with the rest of the pipeline's stages.
    """

    def __init__(
        self,
        confidence: float = 0.7,
        buffer_min: int = 2,
        buffer_max: int = 25,
    ) -> None:
        """``confidence`` is the boundary-verdict threshold (sigma); below
        ``buffer_min`` messages ``push`` never calls the LLM; at
        ``buffer_max`` the whole buffer is force-cut as one segment."""
        self.confidence = confidence  # paper sigma_boundary=0.7
        self.buffer_min = buffer_min  # repo config buffer_size_min=2 (not in paper v1)
        self.buffer_max = buffer_max  # paper beta_max=25

    def push(
        self, buffer: list[Episode], ctx: OrganizerContext
    ) -> tuple[list[list[Episode]], list[Episode]]:
        """Return (segments to flush, remaining buffer) for the given buffer."""
        if len(buffer) < self.buffer_min:
            return [], buffer
        if len(buffer) >= self.buffer_max:
            return [buffer], []

        verdict = ctx.llm.call(
            "extract",
            BOUNDARY_PROMPT.format(
                buffer="\n".join(_fmt(e) for e in buffer[:-1]),
                message=_fmt(buffer[-1]),
            ),
            BOUNDARY_SCHEMA,
            required_keys=("boundary", "confidence"),
            phase="segment",
        )
        if verdict is None:
            return [], buffer  # drop counted upstream; treat as no boundary
        if verdict.get("boundary") and float(verdict.get("confidence", 0.0)) >= self.confidence:
            # The newest message opened the next topic: it stays buffered.
            return [buffer[:-1]], [buffer[-1]]
        return [], buffer

    def flush(self, buffer: list[Episode], ctx: OrganizerContext) -> list[list[Episode]]:
        """Flush whatever remains in the buffer as the final segment(s)."""
        return [buffer] if buffer else []


BATCH_SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "episodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "indices": {"type": "array", "items": {"type": "integer"}},
                    "topic": {"type": "string"},
                },
                "required": ["indices"],
            },
        }
    },
    "required": ["episodes"],
}

# Condensed from upstream BATCH_SEGMENTATION_PROMPT (v4 Local Message
# Partitioning P_par): topic independence, 2-15 messages, when in doubt
# split, relevance < ~30% cut; indexed batch in, index groups out.
BATCH_SEGMENT_PROMPT = """Partition this conversation into topically coherent
episodes. Each episode centers on ONE core topic or event. Signals for a cut:
explicit topic shifts, intent transitions, temporal markers ("earlier",
"by the way", 30+ minute gaps), structural signals (transition phrases,
concluding statements), content relatedness below ~30%. Episodes work best
with 2-15 messages; when in doubt, split. Every message index must appear in
exactly one episode, in order.

Messages (indexed):
{messages}

Return JSON: {{"episodes": [{{"indices": [0, 1, ...], "topic": "..."}}]}}"""


class BatchPartitioner:
    """v4 §3.2.1 Local Message Partitioning / upstream batch segmentation.

    Buffers until ``window`` then LLM-partitions the whole buffer. flush()
    on a tail shorter than the window stores it as one segment without an
    LLM call — upstream's single-'conversation'-group path below
    batch_threshold."""

    def __init__(self, window: int = 20, buffer_min: int = 2, chunk_max: int = 80) -> None:
        """``window`` messages accumulate before ``push`` triggers a
        partition call; ``chunk_max`` caps how many messages go into one LLM
        partitioning call, splitting larger buffers into sequential chunks.
        ``buffer_min`` is accepted for interface parity with
        `PerMessageBoundary` but unused here (batch mode has no per-message
        minimum)."""
        self.window = window
        self.buffer_min = buffer_min
        self.chunk_max = chunk_max  # upstream: >80msg batches are chunked

    def push(
        self, buffer: list[Episode], ctx: OrganizerContext
    ) -> tuple[list[list[Episode]], list[Episode]]:
        """Returns ``([], buffer)`` unchanged until ``window`` is reached;
        once reached, partitions the WHOLE buffer via `_partition` and
        always returns an empty remaining buffer (unlike
        `PerMessageBoundary`, which may keep a tail message buffered)."""
        if len(buffer) < self.window:
            return [], buffer
        return self._partition(buffer, ctx), []

    def flush(self, buffer: list[Episode], ctx: OrganizerContext) -> list[list[Episode]]:
        """Empty buffer -> ``[]``. Below ``window`` -> the whole buffer as
        one segment, no LLM call (upstream's single-group short-batch path).
        At or above ``window`` -> partitions via `_partition` same as
        `push`."""
        if not buffer:
            return []
        if len(buffer) < self.window:
            return [buffer]
        return self._partition(buffer, ctx)

    def _partition(
        self, buffer: list[Episode], ctx: OrganizerContext
    ) -> list[list[Episode]]:
        segments: list[list[Episode]] = []
        for start in range(0, len(buffer), self.chunk_max):
            chunk = buffer[start : start + self.chunk_max]
            indexed = "\n".join(f"[{i}] {_fmt(e)}" for i, e in enumerate(chunk))
            result = ctx.llm.call(
                "extract",
                BATCH_SEGMENT_PROMPT.format(messages=indexed),
                BATCH_SEGMENT_SCHEMA,
                required_keys=("episodes",),
                phase="segment",
            )
            groups = (result or {}).get("episodes") or []
            covered: set[int] = set()
            for g in groups:
                indexes = sorted(
                    i
                    for i in g.get("indices", [])
                    if isinstance(i, int) and 0 <= i < len(chunk) and i not in covered
                )
                if not indexes:
                    continue
                covered.update(indexes)
                segments.append([chunk[i] for i in indexes])
            leftover = [chunk[i] for i in range(len(chunk)) if i not in covered]
            if leftover:  # LLM 실패/누락 인덱스 — 세그먼트를 잃지 않는다 (프로젝트 원칙)
                segments.append(leftover)
        return segments


MERGE_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["merge", "new"]},
        "target_index": {"type": "integer"},
    },
    "required": ["decision"],
}
# EPISODE_SCHEMA look-alike, defined here instead of nemori.py to avoid a
# circular import (EpisodeMerger needs it, nemori.py imports EpisodeMerger).
MERGE_CONTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "narrative": {"type": "string"},
        "timestamp": {"type": "string"},
    },
    "required": ["title", "narrative"],
}

# Condensed from upstream MERGE_DECISION; the >1h ban line is injected only
# when time_gap_hours is set (upstream preset) — the v4 paper has no time
# constraint (verified 2026-07-18, docs/research/write-path-lifecycle-survey.md §1.2).
MERGE_DECISION_PROMPT = """A new episodic memory arrived. Decide whether it
describes the SAME event as one of the candidate episodes and should be
merged into it, or is a distinct episode.
Merge only when they cover the same underlying event or activity thread.
{time_gap_rule}
New episode:
title: {title}
narrative: {narrative}
timestamp: {timestamp}

Candidates (indexed):
{candidates}

Return JSON: {{"decision": "merge" or "new", "target_index": <candidate index, only when merging>}}"""

TIME_GAP_RULE = (
    "Do NOT merge if they are separated by significant time gaps "
    "(>{hours} hour(s)) between their timestamps."
)

# Condensed from upstream MERGE_CONTENT: synthesize without duplication,
# chronological flow, keep participants/decisions/emotions/outcomes,
# earliest timestamp wins.
MERGE_CONTENT_PROMPT = """Merge these two episodic memories about the same event
into ONE. Synthesize without duplication, preserve chronological event flow,
retain all critical details (participants, decisions, emotions, outcomes).
Use the EARLIEST timestamp.

Episode A:
title: {title_a}
narrative: {narrative_a}
timestamp: {ts_a}

Episode B:
title: {title_b}
narrative: {narrative_b}
timestamp: {ts_b}

Return JSON: {{"title": "...", "narrative": "...", "timestamp": "ISO"}}"""


class EpisodeMerger:
    """v4 §3.2.3 Episode-level merging / upstream merger.py.

    Called right after episode narration (v4 Alg.1 order — narrate, then
    merge-or-new, then predict-calibrate). Looks up nearby episodes in the
    vector index, asks the LLM whether the new episode is the same event as
    one of them, and if so asks a second LLM call to synthesize the merged
    title/narrative. Any failure along the way (no candidates, LLM declines,
    LLM call fails) returns ``None`` so the caller falls back to a plain ADD
    — a segment is never lost to a merge attempt.
    """

    def __init__(
        self,
        top_k: int = 5,
        similarity: float | None = None,
        time_gap_hours: float | None = None,
    ) -> None:
        """``top_k`` bounds the candidate-episode neighborhood searched.
        ``similarity=None`` means no cosine floor is applied before asking
        the LLM (v4 paper has none); ``time_gap_hours=None`` means no
        merge-ban-by-elapsed-time rule is injected into the merge-decision
        prompt (see the field comments for each default's provenance)."""
        self.top_k = top_k
        self.similarity = similarity  # upstream 0.85; v4 paper doesn't specify one (None)
        self.time_gap_hours = time_gap_hours  # upstream 1.0; v4 paper has no gap ban (None)

    def merge_or_none(
        self,
        title: str,
        narrative: str,
        episode_timestamp: str,
        source_ids: list[str],
        ctx: OrganizerContext,
        exclude_ids: set[str] | None = None,
    ) -> tuple[list[MemoryOp], str, str, str] | None:
        """On success: ``(ops, merged_id, merged_title, merged_narrative)``
        where ``ops`` is a MERGE op for the new synthesized episode plus an
        INVALIDATE for the absorbed one. Returns ``None`` on every failure
        path (see class docstring) — the caller must then fall back to a
        plain ADD. ``exclude_ids`` removes candidates already merged away
        earlier in the same batch (e.g. `NemoriOrganizer`'s within-call
        supersession guard, I1)."""
        # exclude_ids drops episodes already merged-away earlier in this same
        # on_message batch (I1) — same within-pass guard as ThreeWayIntegrator.
        query_embedding = ctx.embedder.embed([f"{title}\n{narrative}"])[0]
        hits = ctx.vector_store.search(
            query_embedding, k=self.top_k, memory_type="episodes", namespace=ctx.namespace
        )
        if self.similarity is not None:
            hits = [(hit_id, s) for hit_id, s in hits if s >= self.similarity]
        candidates = [
            c
            for c in ctx.doc_store.get_items([h[0] for h in hits], "episodes")
            if not c.get("invalid_at") and (exclude_ids is None or c["id"] not in exclude_ids)
        ]
        if not candidates:
            return None
        gap_rule = (
            TIME_GAP_RULE.format(hours=self.time_gap_hours) if self.time_gap_hours else ""
        )
        candidate_text = "\n".join(
            f"[{i}] title: {c.get('title', '')} | timestamp: {c.get('timestamp', '')}\n"
            f"    narrative: {c.get('content', '')}"
            for i, c in enumerate(candidates)
        )
        verdict = ctx.llm.call(
            "distill",
            MERGE_DECISION_PROMPT.format(
                time_gap_rule=gap_rule,
                title=title,
                narrative=narrative,
                timestamp=episode_timestamp,
                candidates=candidate_text,
            ),
            MERGE_DECISION_SCHEMA,
            required_keys=("decision",),
            phase="merge",
        )
        if not verdict or verdict.get("decision") != "merge":
            return None
        target_index = verdict.get("target_index")
        if not isinstance(target_index, int) or not 0 <= target_index < len(candidates):
            return None
        old = candidates[target_index]
        merged = ctx.llm.call(
            "distill",
            MERGE_CONTENT_PROMPT.format(
                title_a=old.get("title", ""),
                narrative_a=old.get("content", ""),
                ts_a=old.get("timestamp", ""),
                title_b=title,
                narrative_b=narrative,
                ts_b=episode_timestamp,
            ),
            MERGE_CONTENT_SCHEMA,
            required_keys=("title", "narrative"),
            phase="merge",
        )
        if merged is None:
            return None  # 병합 실패 → 호출측이 일반 ADD로 저장 (세그먼트 불손실)
        merged_id = new_id()
        merged_title = str(merged.get("title", "")).strip() or title
        merged_narrative = str(merged.get("narrative", "")).strip() or narrative
        merged_timestamp = str(merged.get("timestamp", "")).strip() or min(
            str(old.get("timestamp", "")), str(episode_timestamp)
        )
        ops = [
            MemoryOp(
                op=OpType.MERGE,
                target_type="episodes",
                target_id=merged_id,
                payload={
                    "id": merged_id,
                    "title": merged_title,
                    "content": merged_narrative,
                    "timestamp": merged_timestamp,
                    "supersedes": [old["id"]],
                    "source_episode_ids": list(old.get("source_episode_ids", []))
                    + list(source_ids),
                    "embedding_text": f"{merged_title}\n{merged_narrative}",
                },
            ),
            MemoryOp(
                op=OpType.INVALIDATE,
                target_type="episodes",
                target_id=old["id"],
                payload={"reason": "merged", "superseded_by": merged_id},
            ),
        ]
        return ops, merged_id, merged_title, merged_narrative


INTEGRATE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["new", "merge", "conflict"]},
        "target_indexes": {"type": "array", "items": {"type": "integer"}},
        "statement": {"type": "string"},
    },
    "required": ["decision"],
}

# v4 §3.3.3 P_con condensed: new=distinct, merge=supersede with unified
# statement, conflict=the new insight invalidates the old entries.
INTEGRATE_PROMPT = """A new knowledge statement arrived. Compare it with the
existing similar statements and decide:
- "new": genuinely distinct knowledge -> keep all
- "merge": same knowledge (possibly partial overlaps) -> produce ONE unified
  statement superseding the indexed existing ones
- "conflict": the new statement invalidates/corrects the indexed existing
  ones -> they must be replaced

New statement: {fact}

Existing (indexed):
{existing}

Return JSON: {{"decision": "new"|"merge"|"conflict",
"target_indexes": [affected indexes],
"statement": "unified or corrected statement (merge/conflict only)"}}"""


class AppendIntegrator:
    """Current default: every distilled fact becomes its own semantic ADD.
    No dedup, no LLM call — this is the repo's pre-v4 main path, kept as the
    baseline so switching semantic_integration doesn't change existing runs."""

    def integrate(
        self,
        fact: str,
        episode_id: str,
        source_ids: list[str],
        ctx: OrganizerContext,
        exclude_ids: set[str] | None = None,
    ) -> list[MemoryOp]:
        """Always returns exactly one ADD op for ``fact`` — never merges,
        never fails. ``exclude_ids`` is accepted for interface parity with
        the other integrators but has no effect here."""
        # exclude_ids is part of the common integrator contract (I1) but has no
        # effect here — Append never searches for candidates.
        fact_id = new_id()
        return [
            MemoryOp(
                op=OpType.ADD,
                target_type="semantic",
                target_id=fact_id,
                payload={
                    "id": fact_id,
                    "content": fact,
                    "episode_id": episode_id,
                    "source_episode_ids": list(source_ids),
                    "embedding_text": fact,
                },
            )
        ]


class DedupIdReuseIntegrator:
    """PR#19 semantics: top-1 embedding match >= threshold reuses the id —
    latest content wins, provenance re-pointed. No LLM call."""

    def __init__(self, threshold: float = 0.85) -> None:
        """``threshold`` is the min cosine similarity for the top-1 match to
        be treated as the same fact and reused (PR#19 semantics)."""
        self.threshold = threshold

    def integrate(
        self,
        fact: str,
        episode_id: str,
        source_ids: list[str],
        ctx: OrganizerContext,
        exclude_ids: set[str] | None = None,
    ) -> list[MemoryOp]:
        """If the nearest existing semantic item scores >= ``threshold``,
        returns one UPDATE op that overwrites its content/provenance
        (id reused, latest content wins). Otherwise falls back to
        `AppendIntegrator` and returns a plain ADD. ``exclude_ids`` removes
        ids from consideration before the top-1 check."""
        query_embedding = ctx.embedder.embed([fact])[0]
        hits = [
            (hit_id, s)
            for hit_id, s in ctx.vector_store.search(
                query_embedding, k=1, memory_type="semantic", namespace=ctx.namespace
            )
            if exclude_ids is None or hit_id not in exclude_ids
        ]
        if hits and hits[0][1] >= self.threshold:
            return [
                MemoryOp(
                    op=OpType.UPDATE,
                    target_type="semantic",
                    target_id=hits[0][0],
                    payload={
                        "content": fact,
                        "episode_id": episode_id,
                        "source_episode_ids": list(source_ids),
                        "embedding_text": fact,
                    },
                )
            ]
        return AppendIntegrator().integrate(fact, episode_id, source_ids, ctx)


class ThreeWayIntegrator:
    """v4 §3.3.3 P_con: tau-filter the vector neighborhood, then ask the LLM
    to decide new/merge/conflict over the top-K survivors. Any failure path
    (no candidates past tau, LLM call fails, decision=new, no valid target
    indexes) falls back to a plain AppendIntegrator ADD — a fact is never
    lost to a failed integration attempt."""

    def __init__(self, top_k: int = 5, tau: float = 0.70) -> None:
        """``top_k`` bounds the vector-neighborhood search; ``tau`` is the
        minimum cosine similarity a hit must clear to become an LLM
        candidate at all (v4 §3.3.3)."""
        self.top_k = top_k
        self.tau = tau

    def integrate(
        self,
        fact: str,
        episode_id: str,
        source_ids: list[str],
        ctx: OrganizerContext,
        exclude_ids: set[str] | None = None,
        phase: str = "integrate",
    ) -> list[MemoryOp]:
        """Returns a plain ADD (`AppendIntegrator` fallback) on every failure
        path (see class docstring). On "merge"/"conflict", returns a list
        starting with the new head op (MERGE or ADD respectively) followed
        by one INVALIDATE per superseded target. ``phase`` is passed through
        to `ctx.llm.call` only to distinguish inline calls from
        `SemanticOfflineConsolidator`'s deferred reuse of this method in
        budget/logging breakdowns — it does not change the decision logic.
        ``exclude_ids`` removes ids from the candidate search."""
        # ``phase`` distinguishes the same integration code running inline
        # (llm3way, phase="integrate") from SemanticOfflineConsolidator's
        # deferred pass over the shared implementation (phase="consolidate").
        query_embedding = ctx.embedder.embed([fact])[0]
        hits = [
            (hit_id, s)
            for hit_id, s in ctx.vector_store.search(
                query_embedding, k=self.top_k, memory_type="semantic", namespace=ctx.namespace
            )
            if s >= self.tau and (exclude_ids is None or hit_id not in exclude_ids)
        ]
        candidates = [
            c
            for c in ctx.doc_store.get_items([h[0] for h in hits], "semantic")
            if not c.get("invalid_at")
        ]
        add = AppendIntegrator().integrate(fact, episode_id, source_ids, ctx)
        if not candidates:
            return add
        existing = "\n".join(f"[{i}] {c.get('content', '')}" for i, c in enumerate(candidates))
        verdict = ctx.llm.call(
            "distill",
            INTEGRATE_PROMPT.format(fact=fact, existing=existing),
            INTEGRATE_SCHEMA,
            required_keys=("decision",),
            phase=phase,
        )
        if verdict is None or verdict.get("decision") == "new":
            return add
        indexes = [
            i
            for i in verdict.get("target_indexes", [])
            if isinstance(i, int) and 0 <= i < len(candidates)
        ]
        if not indexes:
            return add
        statement = str(verdict.get("statement", "")).strip() or fact
        targets = [candidates[i]["id"] for i in indexes]
        fact_id = new_id()
        if verdict["decision"] == "merge":
            head = MemoryOp(
                op=OpType.MERGE,
                target_type="semantic",
                target_id=fact_id,
                payload={
                    "id": fact_id,
                    "content": statement,
                    "episode_id": episode_id,
                    "source_episode_ids": list(source_ids),
                    "supersedes": targets,
                    "embedding_text": statement,
                },
            )
        else:  # conflict — 신규가 구 항목들을 대체 (spec §2.3)
            head = MemoryOp(
                op=OpType.ADD,
                target_type="semantic",
                target_id=fact_id,
                payload={
                    "id": fact_id,
                    "content": statement,
                    "episode_id": episode_id,
                    "source_episode_ids": list(source_ids),
                    "embedding_text": statement,
                },
            )
        return [head] + [
            MemoryOp(
                op=OpType.INVALIDATE,
                target_type="semantic",
                target_id=target_id,
                payload={"reason": verdict["decision"], "superseded_by": fact_id},
            )
            for target_id in targets
        ]


class SemanticOfflineConsolidator:
    """Deferred three-way consolidation over facts accumulated since the
    cursor — our addition (absent from both the paper and upstream; the
    LightMem/MOOM dual-phase pattern, spec §2.3). Own outputs carry
    consolidated=True and are skipped on later passes, so repeated calls
    converge instead of re-judging their own merges."""

    def __init__(self, top_k: int = 5, tau: float = 0.70, scan_limit: int = 10000) -> None:
        """``top_k``/``tau`` are forwarded to the inner `ThreeWayIntegrator`.
        ``scan_limit`` bounds one `run()` call's `ops_since` page — see
        `run` for what happens when a scan hits this limit."""
        self._inner = ThreeWayIntegrator(top_k=top_k, tau=tau)
        # scan_limit mirrors the doc store's ops_since page size; exposed so a
        # test can force truncation with a tiny value (review I2).
        self.scan_limit = scan_limit

    def run(self, organizer, ctx: OrganizerContext) -> list[MemoryOp]:
        """Re-judges every not-yet-consolidated semantic ADD this
        ``organizer`` produced since its cursor, via `ThreeWayIntegrator`
        (own merge/conflict outputs marked ``consolidated=True`` so they
        aren't re-judged on a later pass). Returns ``[]`` with no cursor
        advance when there is nothing new since the cursor; otherwise
        returns the accumulated ADD/MERGE/INVALIDATE ops plus exactly one
        trailing cursor-advance op. If the scan hit ``scan_limit``, the
        cursor advances only to the last row actually scanned (not to
        `last_seq()`), so a future call resumes rather than skipping the
        unscanned tail."""
        cursor = organizer.read_cursor(ctx)
        rows = ctx.doc_store.ops_since(cursor, target_type="semantic", limit=self.scan_limit)
        # Cursor advance must respect ops_since's limit truncation (review I2):
        # if the semantic log since the cursor filled scan_limit, the tail is
        # cut off here, so advance only to the last row we actually scanned and
        # let the next call resume from there. Jumping to last_seq() would step
        # the cursor past the unscanned facts, dropping them from consolidation
        # forever. With no truncation, advance to last_seq() so trailing
        # non-semantic ops aren't re-scanned next pass (original behavior).
        if len(rows) >= self.scan_limit:
            end = rows[-1][0]
        else:
            end = ctx.doc_store.last_seq()
        if end <= cursor:
            return []
        ops: list[MemoryOp] = []
        # Ops accumulated so far in *this* run() are only applied to
        # doc_store/vector_store after run() returns (facade appends the
        # whole batch atomically), so a fact merged-away earlier in this same
        # pass still looks perfectly live to ctx.doc_store/ctx.vector_store
        # for every later iteration of this loop.
        # Without tracking that here, two mutually-similar facts queued since
        # the cursor would each independently earn their own merge head
        # against the not-yet-updated store — both survive live, and since
        # MERGE-typed outputs are never re-judged (op.op is not OpType.ADD
        # above), the duplication becomes permanent (review Critical-1).
        superseded_this_pass: set[str] = set()
        for _seq, op in rows:
            if op.op is not OpType.ADD or op.payload.get("consolidated"):
                continue
            if op.actor != organizer.name:
                continue  # only re-judge this organizer's own semantic ADDs
            if op.target_id in superseded_this_pass:
                continue  # already absorbed by another merge earlier in this pass
            current = ctx.doc_store.get_items([op.target_id], "semantic")
            if not current or current[0].get("invalid_at"):
                continue  # already superseded by this pass or inline
            fact = str(current[0].get("content", ""))
            # exclude_ids drops the fact's own item, plus everything already
            # superseded earlier in this pass, from ThreeWay's candidate
            # search — otherwise vector_store.search reliably returns the
            # fact's own top-1 neighbor (Task 9 signature change, spec §2.3
            # note), and
            # would also happily re-offer an already-absorbed duplicate as a
            # fresh merge candidate.
            out = self._inner.integrate(
                fact,
                current[0].get("episode_id", ""),
                current[0].get("source_episode_ids", []),
                ctx,
                exclude_ids={op.target_id} | superseded_this_pass,
                phase="consolidate",
            )
            if not any(o.op is OpType.INVALIDATE for o in out):
                continue  # decision=new / no candidates -> keep the stored original
            produced_new = [o for o in out if o.op in (OpType.ADD, OpType.MERGE)]
            for o in produced_new:
                o.payload["consolidated"] = True
            # the original fact is superseded by the consolidated statement too
            head_id = produced_new[0].target_id if produced_new else ""
            out.append(
                MemoryOp(
                    op=OpType.INVALIDATE,
                    target_type="semantic",
                    target_id=op.target_id,
                    payload={"reason": "consolidated", "superseded_by": head_id},
                )
            )
            superseded_this_pass.update(
                o.target_id for o in out if o.op is OpType.INVALIDATE
            )
            ops.extend(out)
        ops.append(organizer.cursor_op(end))
        return ops
