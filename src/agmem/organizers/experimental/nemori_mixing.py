"""Nemori 'our-mixing' semantic stages — NOT in the paper or upstream.

``ThreeWayIntegrator`` (Nemori ``semantic_integration="llm3way"``) and
``SemanticOfflineConsolidator`` (``consolidation="semantic_offline"``) are
compositions we added on top of Nemori's faithful v1/v4 pipeline; neither the
paper nor the upstream repo carries them. They live here so the fidelity
boundary is explicit and Nemori's core stays paper-faithful — selected only via
those explicit non-default presets. ``AppendIntegrator``/``DedupIdReuseIntegrator``
remain in ``nemori_stages`` as the v1/v4 baseline (spec
2026-07-21-organizer-experimental-split).
"""

from __future__ import annotations

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import new_id
from agmem.organizers.base import OrganizerContext
from agmem.organizers.nemori_stages import AppendIntegrator

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
                query_embedding,
                k=self.top_k,
                memory_type="semantic",
                namespace=ctx.namespace,
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
            superseded_this_pass.update(o.target_id for o in out if o.op is OpType.INVALIDATE)
            ops.extend(out)
        ops.append(organizer.cursor_op(end))
        return ops
