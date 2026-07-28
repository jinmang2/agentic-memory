"""G-Memory organizer (arXiv:2506.07398, NeurIPS'25) — compact port.

Trajectory memory + reward-shaped insight rules with periodic LLM
finetune (ADD/EDIT/REMOVE ops on the rule list — upstream parses these
from free text with a regex; we get them as structured JSON).

Deviations (documented per docs/research/g-memory.md):
- The query graph (upstream: networkx + pickle sidecar) is item-payload
  adjacency instead: ``on_task_end`` links the new task to existing
  trajectories at similarity >= 0.7 among the top-10 candidates
  (``TaskLayer.add_task_node``'s constants) via ``task_edges``, and the
  1-hop read expansion (paper Eq.(5)) plus task-association insight recall
  (Eq.(6); upstream ``positive_correlation_tasks``, count-scored as
  ``_find_related_insights`` does) is ``TaskGraphExpansion`` in
  retrieval/steps.py — everything rides the op log, no sidecar.
- Upstream ``retrieve_memory``'s per-successful-trajectory LLM importance
  rerank (``generative_task_user_prompt``, one call per candidate) is NOT
  ported — a read-side LLM cost the embedding ranking stands in for
  (docs/16 session 5).
- FINCH cluster-merge is deferred; the rule cap is enforced upstream-style
  by suppressing ADD when full + soft REMOVE (-3) + score<=0 pruning.
Score semantics follow the official code (round-5): ADD init 2, EDIT/AGREE
+1, REMOVE soft -1 (-3 full), prune at <=0 (clear_insights); backward
reward +1/-2 applies to insights served since the last backward
(on_retrieval cache).
Provenance, stated precisely because the previous wording was wrong: this
is NOT a clean-room reimplementation. ``github.com/bingreeky/GMemory`` was
cloned and read as the primary reference (round-5,
docs/research/round5/gmemory-verify-report.md cites commit ``7b581c5``),
and it had to be — the paper and the code disagree about the central
mechanism (§4.3 describes a summarisation function J plus a supporting
query set, the code implements Reflexion-style critique finetune with
ADD/EDIT/REMOVE/AGREE, backward reward and FINCH), and this port follows
the CODE. The score constants two lines up are read off that code.

That repository carries NO license file (checked again 2026-07-27), so
nothing there is granted for reuse by default. What is reproduced here is
behaviour and numeric constants rather than source text, which is the
same footing as every other port in this package — but the earlier
"clean-room ... from the paper" claim asserted an independence that the
audit trail contradicts, and an inaccurate provenance note is worse than
none.
"""

from __future__ import annotations

import json
import logging

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import new_id
from agmem.organizers.base import Organizer, OrganizerContext

logger = logging.getLogger("agmem.organizers.gmemory")

SPARSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "key_steps": {"type": "array", "items": {"type": "string"}},
        "mistakes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["key_steps"],
}

FINETUNE_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "maxItems": 4,  # upstream: at most 4 ops per prompt
            "items": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["ADD", "EDIT", "REMOVE", "AGREE"],
                    },
                    "id": {"type": "string"},
                    "rule": {"type": "string"},
                },
                "required": ["op"],
            },
        }
    },
    "required": ["operations"],
}

PROJECT_SCHEMA = {
    "type": "object",
    "properties": {"insights": {"type": "array", "items": {"type": "string"}}},
    "required": ["insights"],
}

SPARSIFY_PROMPT = """Condense this multi-agent task trajectory: keep only the decisive steps
(prune failed detours into "mistakes").

Task: {task}
Outcome: {outcome}
Trajectory:
{trajectory}

Return JSON: {{"key_steps": ["...", ...], "mistakes": ["...", ...]}}"""

FINETUNE_PROMPT = """You maintain a list of general insight rules for solving tasks.
Compare recent successful and failed trajectories against the current rules and
propose operations: ADD a new rule, EDIT an existing rule (give its id),
AGREE with a rule that recent evidence supports (give its id), or REMOVE a rule
that recent evidence contradicts (give its id).
Do at most 4 operations, and each existing rule can get at most 1 operation.
Write each rule in the form "XXX, because XXX".

Current rules:
{rules}

Recent trajectories:
{trajectories}

Return JSON: {{"operations": [{{"op": "ADD", "rule": "..."}},
{{"op": "EDIT", "id": "<rule id>", "rule": "..."}},
{{"op": "AGREE", "id": "<rule id>"}}, {{"op": "REMOVE", "id": "<rule id>"}}]}}"""

PROJECT_PROMPT = """Rewrite these general insights so they are directly actionable for the
agent role "{role}" (drop insights irrelevant to that role).

Insights:
{insights}

Return JSON: {{"insights": ["role-tailored insight", ...]}}"""


class GMemoryOrganizer(Organizer):
    """G-Memory trajectory + insight-rule organizer (arXiv:2506.07398; see
    module docstring for the query-graph approximation and score-semantics
    deviations from the official code). Writes only via returned MemoryOps;
    reads (task-similar trajectories/insights) go through ``ctx.doc_store``
    / ``ctx.vector_store``."""

    name = "gmemory"

    produces = ("strategies",)

    def __init__(self, finetune_every: int = 5, insight_max: int = 10) -> None:
        """``finetune_every``: run the insight-rule finetune LLM call once
        per this many ``on_task_end`` calls. ``insight_max``: cap on
        insights fetched into one finetune prompt — reaching it suppresses
        further ADD operations (upstream full-list behavior, round-5
        §2.2)."""
        self.finetune_every = finetune_every
        self.insight_max = insight_max
        self._task_count = 0
        # upstream insights_cache: ids served since the last on_feedback —
        # reward applies only to insights the agent actually saw (round-5 W-4)
        self._served: set[str] = set()

    def on_retrieval(
        self, hits: list[tuple[str, str, float]], ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Records served insight ids into ``_served`` for ``on_feedback``
        reward attribution (round-5 W-4); always returns [] — no store
        writes, no LLM calls."""
        self._served.update(i for i, mt, _ in hits if mt == "strategies")
        return []

    def on_task_end(
        self, trajectory: list[dict], outcome: str, task: str, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Always emits one ADD trajectory op (mechanical fallback to a
        truncated raw trajectory when no LLM is configured or sparsify
        fails); every ``finetune_every`` calls, additionally runs
        ``_finetune_insights`` and appends its ADD/UPDATE/DELETE ops."""
        traj_text = "\n".join(json.dumps(s, ensure_ascii=False, default=str) for s in trajectory)[
            :6000
        ]
        self._task_count += 1
        ops: list[MemoryOp] = []

        if ctx.llm is None:
            logger.warning("gmemory: no LLM — storing mechanical trajectory (explicit degradation)")
            key_steps, mistakes = [traj_text[:1000]], []
        else:
            result = ctx.llm.call(
                "distill",
                SPARSIFY_PROMPT.format(task=task, outcome=outcome, trajectory=traj_text),
                SPARSIFY_SCHEMA,
                required_keys=("key_steps",),
            )
            if result is None:
                key_steps, mistakes = [traj_text[:1000]], []
            else:
                key_steps = [str(s) for s in result.get("key_steps", [])]
                mistakes = [str(m) for m in result.get("mistakes", [])]

        traj_id = new_id()
        content = "\n".join(key_steps) + ("\nMistakes: " + "; ".join(mistakes) if mistakes else "")

        # Query-graph edges (paper Eq.(9); upstream TaskLayer.add_task_node):
        # link the new task to existing trajectories at similarity >= 0.7
        # among the top-10 candidates, undirected — each neighbor gains the
        # back-edge via UPDATE. A repeat of an already-stored task title adds
        # no edges, mirroring upstream's early return when the task text is
        # already a node. The read-side 1-hop expansion over these edges is
        # ``TaskGraphExpansion`` (retrieval/steps.py).
        task_edges: list[str] = []
        edge_updates: list[MemoryOp] = []
        hit_scores = dict(
            ctx.vector_store.search(
                ctx.embedder.embed([task[:2000]])[0],
                k=10,
                memory_type="strategies",
                namespace=ctx.namespace,
            )
        )
        if hit_scores:
            neighbors = [
                n
                for n in ctx.doc_store.get_items(list(hit_scores), "strategies")
                if n.get("kind") == "trajectory" and not n.get("deleted")
            ]
            if not any(n.get("title") == task[:80] for n in neighbors):
                for n in neighbors:
                    # 0.7 is upstream TaskLayer.similarity_threshold
                    if hit_scores.get(n["id"], 0.0) < 0.7:
                        continue
                    task_edges.append(n["id"])
                    edge_updates.append(
                        MemoryOp(
                            op=OpType.UPDATE,
                            target_type="strategies",
                            target_id=n["id"],
                            payload={"task_edges": [*n.get("task_edges", []), traj_id]},
                        )
                    )

        ops.append(
            MemoryOp(
                op=OpType.ADD,
                target_type="strategies",
                target_id=traj_id,
                payload={
                    "id": traj_id,
                    "title": task[:80],
                    "content": content,
                    "outcome": outcome,
                    "kind": "trajectory",
                    "score": 1.0 if outcome == "success" else -2.0,
                    "task_edges": task_edges,
                    "embedding_text": f"{task}\n{content}"[:2000],
                },
            )
        )
        ops.extend(edge_updates)

        if ctx.llm is not None and self._task_count % self.finetune_every == 0:
            ops.extend(self._finetune_insights(task, ctx))
        return ops

    def _fetch(self, ctx: OrganizerContext, query: str, kind: str, k: int) -> list[dict]:
        query_embedding = ctx.embedder.embed([query])[0]
        hits = ctx.vector_store.search(
            query_embedding, k=k * 3, memory_type="strategies", namespace=ctx.namespace
        )
        items = ctx.doc_store.get_items([h[0] for h in hits], "strategies")
        return [i for i in items if i.get("kind") == kind and not i.get("deleted")][:k]

    def _finetune_insights(self, task: str, ctx: OrganizerContext) -> list[MemoryOp]:
        insights = self._fetch(ctx, task, "insight", self.insight_max)
        trajectories = self._fetch(ctx, task, "trajectory", 10)
        # Task titles this finetune round saw — recorded on every touched
        # insight as upstream does (`relative_tasks` in _finetune_insights:
        # ADD/EDIT/AGREE extend positive_correlation_tasks, REMOVE extends
        # negative). TaskGraphExpansion's Eq.(6) recall reads the positive set.
        relative_tasks = sorted({str(t.get("title", "")) for t in trajectories} - {""})
        result = ctx.llm.call(
            "distill",
            FINETUNE_PROMPT.format(
                rules="\n".join(
                    f"- id={i['id']} (score={i.get('score', 0)}) {i['content']}" for i in insights
                )
                or "(none)",
                trajectories="\n".join(
                    f"- [{t.get('outcome')}] {t.get('title')}: {t.get('content', '')[:300]}"
                    for t in trajectories
                ),
            ),
            FINETUNE_SCHEMA,
            required_keys=("operations",),
        )
        if result is None:
            return []

        # Upstream score semantics (round-5 §2.2): ADD starts at 2, EDIT and
        # AGREE reinforce (+1), REMOVE is SOFT (-1; -3 when the list is
        # full). Actual deletion happens only when a score reaches <= 0
        # (upstream clear_insights), here and after backward reward.
        valid = {i["id"]: i for i in insights}
        scores = {i["id"]: float(i.get("score", 0)) for i in insights}
        touched: set[str] = set()  # each existing rule: at most 1 operation
        ops: list[MemoryOp] = []
        n_insights = len(insights)
        list_full = n_insights >= self.insight_max
        for raw in result.get("operations", [])[:4]:
            op, rule_id, rule = (
                raw.get("op"),
                raw.get("id"),
                str(raw.get("rule", "")).strip(),
            )
            if op == "ADD" and rule:
                if list_full:
                    continue  # upstream suppresses ADD when the list is full
                insight_id = new_id()
                ops.append(
                    MemoryOp(
                        op=OpType.ADD,
                        target_type="strategies",
                        target_id=insight_id,
                        payload={
                            "id": insight_id,
                            "title": rule[:60],
                            "content": rule,
                            "kind": "insight",
                            "score": 2.0,
                            "positive_correlation_tasks": relative_tasks,
                            "negative_correlation_tasks": [],
                            "embedding_text": rule,
                        },
                    )
                )
                n_insights += 1
                continue
            if rule_id not in valid or rule_id in touched:
                continue  # hallucinated or double-touched ids emit nothing
            touched.add(rule_id)
            if op == "EDIT" and rule:
                scores[rule_id] += 1.0
                ops.append(
                    MemoryOp(
                        op=OpType.UPDATE,
                        target_type="strategies",
                        target_id=rule_id,
                        payload={
                            "content": rule,
                            "score": scores[rule_id],
                            "positive_correlation_tasks": sorted(
                                set(valid[rule_id].get("positive_correlation_tasks", []))
                                | set(relative_tasks)
                            ),
                            "embedding_text": rule,
                        },
                    )
                )
            elif op == "AGREE":
                scores[rule_id] += 1.0
                ops.append(
                    MemoryOp(
                        op=OpType.UPDATE,
                        target_type="strategies",
                        target_id=rule_id,
                        payload={
                            "score": scores[rule_id],
                            "positive_correlation_tasks": sorted(
                                set(valid[rule_id].get("positive_correlation_tasks", []))
                                | set(relative_tasks)
                            ),
                        },
                    )
                )
            elif op == "REMOVE":
                scores[rule_id] -= 3.0 if list_full else 1.0
                ops.append(
                    MemoryOp(
                        op=OpType.UPDATE,
                        target_type="strategies",
                        target_id=rule_id,
                        payload={
                            "score": scores[rule_id],
                            "negative_correlation_tasks": sorted(
                                set(valid[rule_id].get("negative_correlation_tasks", []))
                                | set(relative_tasks)
                            ),
                        },
                    )
                )

        # prune: any insight whose score dropped to <= 0 is deleted
        for rule_id, score in scores.items():
            if score <= 0:
                ops.append(
                    MemoryOp(
                        op=OpType.DELETE,
                        target_type="strategies",
                        target_id=rule_id,
                        payload={"reason": "score_pruned"},
                    )
                )
        return ops

    def project_insights(self, role: str, insights: list[str], ctx: OrganizerContext) -> list[str]:
        """Role-specific insight rewriting (multi-agent injection path)."""
        if ctx.llm is None or not insights:
            return insights
        result = ctx.llm.call(
            "distill",
            PROJECT_PROMPT.format(role=role, insights="\n".join(f"- {i}" for i in insights)),
            PROJECT_SCHEMA,
            required_keys=("insights",),
        )
        return [str(i) for i in result["insights"]] if result else insights

    def on_feedback(
        self, memory_ids: list[str], helpful: bool, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Reward shaping on served insights (+1 success / -2 failure),
        followed by upstream clear_insights: score <= 0 is pruned. Applies
        only to items served since the last feedback (``self._served``).

        This was previously a ``backward(insight_items, reward)`` method with no
        caller anywhere, while the facade's ``report_feedback`` reimplemented the
        same rule *without* the ``_served`` gate — so the round-5 W-4 fix
        ("reward applies only to insights the agent actually saw") lived in dead
        code, and ``_served`` was never cleared, growing for the process
        lifetime. Reaching the rule through the organizer hook makes the live
        path and the fidelity fix the same code."""
        reward = 1.0 if helpful else -2.0
        insight_items = ctx.doc_store.get_items(list(memory_ids), "strategies")
        ops: list[MemoryOp] = []
        for i in insight_items:
            if i.get("deleted"):
                continue
            if self._served and i["id"] not in self._served:
                continue
            new_score = float(i.get("score", 0)) + reward
            ops.append(
                MemoryOp(
                    op=OpType.UPDATE,
                    target_type="strategies",
                    target_id=i["id"],
                    payload={"score": new_score},
                )
            )
            if new_score <= 0 and i.get("kind") == "insight":
                ops.append(
                    MemoryOp(
                        op=OpType.DELETE,
                        target_type="strategies",
                        target_id=i["id"],
                        payload={"reason": "score_pruned"},
                    )
                )
        self._served.clear()
        return ops
