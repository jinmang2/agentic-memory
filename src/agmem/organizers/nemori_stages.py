"""Assembled Nemori pipeline stages (spec §2.3).

``PerMessageBoundary`` is the first stage extracted out of ``NemoriOrganizer``:
the online, per-message boundary detector (the paper's v1 f_theta formalism)
that decides whether the newest buffered message opens a new episode. It is
v1-equivalent with the logic previously inlined in ``NemoriOrganizer.on_message``
— same prompt, same schema, same thresholds, same buffer-splitting semantics.
Later tasks will add the remaining stages (``BatchPartitioner``,
``EpisodeMerger``, and the semantic integrators) alongside this one so the
organizer becomes a thin composition of independently testable stages.
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


def _fmt(ep: Episode) -> str:
    ts = ep.meta.get("date") or ep.timestamp.isoformat()
    return f"[{ts}] {ep.role}: {ep.content}"


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
        self.window = window
        self.buffer_min = buffer_min
        self.chunk_max = chunk_max  # upstream: >80msg batches are chunked

    def push(
        self, buffer: list[Episode], ctx: OrganizerContext
    ) -> tuple[list[list[Episode]], list[Episode]]:
        if len(buffer) < self.window:
            return [], buffer
        return self._partition(buffer, ctx), []

    def flush(self, buffer: list[Episode], ctx: OrganizerContext) -> list[list[Episode]]:
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
            )
            groups = (result or {}).get("episodes") or []
            covered: set[int] = set()
            for g in groups:
                idxs = sorted(
                    i
                    for i in g.get("indices", [])
                    if isinstance(i, int) and 0 <= i < len(chunk) and i not in covered
                )
                if not idxs:
                    continue
                covered.update(idxs)
                segments.append([chunk[i] for i in idxs])
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
        self.top_k = top_k
        self.similarity = similarity  # upstream 0.85; v4 paper doesn't specify one (None)
        self.time_gap_hours = time_gap_hours  # upstream 1.0; v4 paper has no gap ban (None)

    def merge_or_none(
        self,
        title: str,
        narrative: str,
        ep_ts: str,
        source_ids: list[str],
        ctx: OrganizerContext,
    ) -> tuple[list[MemoryOp], str, str, str] | None:
        emb = ctx.embedder.embed([f"{title}\n{narrative}"])[0]
        hits = ctx.vec.search(
            emb, k=self.top_k, memory_type="episodes", namespace=ctx.namespace
        )
        if self.similarity is not None:
            hits = [(hid, s) for hid, s in hits if s >= self.similarity]
        cands = [
            c
            for c in ctx.doc.get_items([h[0] for h in hits], "episodes")
            if not c.get("invalid_at")
        ]
        if not cands:
            return None
        gap_rule = (
            TIME_GAP_RULE.format(hours=self.time_gap_hours) if self.time_gap_hours else ""
        )
        cand_text = "\n".join(
            f"[{i}] title: {c.get('title', '')} | timestamp: {c.get('timestamp', '')}\n"
            f"    narrative: {c.get('content', '')}"
            for i, c in enumerate(cands)
        )
        verdict = ctx.llm.call(
            "distill",
            MERGE_DECISION_PROMPT.format(
                time_gap_rule=gap_rule,
                title=title,
                narrative=narrative,
                timestamp=ep_ts,
                candidates=cand_text,
            ),
            MERGE_DECISION_SCHEMA,
            required_keys=("decision",),
        )
        if not verdict or verdict.get("decision") != "merge":
            return None
        idx = verdict.get("target_index")
        if not isinstance(idx, int) or not 0 <= idx < len(cands):
            return None
        old = cands[idx]
        merged = ctx.llm.call(
            "distill",
            MERGE_CONTENT_PROMPT.format(
                title_a=old.get("title", ""),
                narrative_a=old.get("content", ""),
                ts_a=old.get("timestamp", ""),
                title_b=title,
                narrative_b=narrative,
                ts_b=ep_ts,
            ),
            MERGE_CONTENT_SCHEMA,
            required_keys=("title", "narrative"),
        )
        if merged is None:
            return None  # 병합 실패 → 호출측이 일반 ADD로 저장 (세그먼트 불손실)
        merged_id = new_id()
        m_title = str(merged.get("title", "")).strip() or title
        m_narr = str(merged.get("narrative", "")).strip() or narrative
        m_ts = str(merged.get("timestamp", "")).strip() or min(
            str(old.get("timestamp", "")), str(ep_ts)
        )
        ops = [
            MemoryOp(
                op=OpType.MERGE,
                target_type="episodes",
                target_id=merged_id,
                payload={
                    "id": merged_id,
                    "title": m_title,
                    "content": m_narr,
                    "timestamp": m_ts,
                    "supersedes": [old["id"]],
                    "source_episode_ids": list(old.get("source_episode_ids", []))
                    + list(source_ids),
                    "embedding_text": f"{m_title}\n{m_narr}",
                },
            ),
            MemoryOp(
                op=OpType.INVALIDATE,
                target_type="episodes",
                target_id=old["id"],
                payload={"reason": "merged", "superseded_by": merged_id},
            ),
        ]
        return ops, merged_id, m_title, m_narr
