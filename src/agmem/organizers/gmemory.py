"""G-Memory organizer (arXiv:2506.07398, NeurIPS'25) — compact port.

Trajectory memory + reward-shaped insight rules with periodic LLM
finetune (ADD/EDIT/REMOVE ops on the rule list — upstream parses these
from free text with a regex; we get them as structured JSON).

Deviations (documented per docs/research/g-memory.md):
- The query graph (networkx, k-hop over task similarity) is approximated
  by embedding retrieval — same recall role, no pickle sidecar. TODO:
  optional SqliteGraphStore-backed task graph.
- FINCH cluster-merge is deferred; the rule cap is enforced by dropping
  the lowest-score insight instead.
- No official license upstream: this is a clean-room reimplementation
  from the paper + published research notes.
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
    "properties": {"key_steps": {"type": "array", "items": {"type": "string"}},
                   "mistakes": {"type": "array", "items": {"type": "string"}}},
    "required": ["key_steps"],
}

FINETUNE_SCHEMA = {
    "type": "object",
    "properties": {"operations": {
        "type": "array", "maxItems": 6,
        "items": {"type": "object",
                  "properties": {"op": {"type": "string",
                                        "enum": ["ADD", "EDIT", "REMOVE"]},
                                 "id": {"type": "string"},
                                 "rule": {"type": "string"}},
                  "required": ["op"]}}},
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
propose operations: ADD a new rule, EDIT an existing rule (give its id), or
REMOVE a rule that recent evidence contradicts (give its id). Propose few, high-value ops.

Current rules:
{rules}

Recent trajectories:
{trajectories}

Return JSON: {{"operations": [{{"op": "ADD", "rule": "..."}},
{{"op": "EDIT", "id": "<rule id>", "rule": "..."}}, {{"op": "REMOVE", "id": "<rule id>"}}]}}"""

PROJECT_PROMPT = """Rewrite these general insights so they are directly actionable for the
agent role "{role}" (drop insights irrelevant to that role).

Insights:
{insights}

Return JSON: {{"insights": ["role-tailored insight", ...]}}"""


class GMemoryOrganizer(Organizer):
    name = "gmemory"

    def __init__(self, finetune_every: int = 5, insight_max: int = 10) -> None:
        self.finetune_every = finetune_every
        self.insight_max = insight_max
        self._task_count = 0

    def on_task_end(self, trajectory: list[dict], outcome: str,
                    task: str, ctx: OrganizerContext) -> list[MemoryOp]:
        traj_text = "\n".join(json.dumps(s, ensure_ascii=False, default=str)
                              for s in trajectory)[:6000]
        self._task_count += 1
        ops: list[MemoryOp] = []

        if ctx.llm is None:
            logger.warning("gmemory: no LLM — storing mechanical trajectory (explicit degradation)")
            key_steps, mistakes = [traj_text[:1000]], []
        else:
            result = ctx.llm.call("distill",
                                  SPARSIFY_PROMPT.format(task=task, outcome=outcome,
                                                         trajectory=traj_text),
                                  SPARSIFY_SCHEMA, required_keys=("key_steps",))
            if result is None:
                key_steps, mistakes = [traj_text[:1000]], []
            else:
                key_steps = [str(s) for s in result.get("key_steps", [])]
                mistakes = [str(m) for m in result.get("mistakes", [])]

        traj_id = new_id()
        content = "\n".join(key_steps) + (
            "\nMistakes: " + "; ".join(mistakes) if mistakes else "")
        ops.append(MemoryOp(
            op=OpType.ADD, target_type="strategies", target_id=traj_id,
            payload={"id": traj_id, "title": task[:80], "content": content,
                     "outcome": outcome, "kind": "trajectory",
                     "score": 1.0 if outcome == "success" else -2.0,
                     "embedding_text": f"{task}\n{content}"[:2000]},
        ))

        if ctx.llm is not None and self._task_count % self.finetune_every == 0:
            ops.extend(self._finetune_insights(task, ctx))
        return ops

    def _fetch(self, ctx: OrganizerContext, query: str, kind: str, k: int) -> list[dict]:
        emb = ctx.embedder.embed([query])[0]
        hits = ctx.vec.search(emb, k=k * 3, memory_type="strategies",
                              namespace=ctx.namespace)
        items = ctx.doc.get_items([h[0] for h in hits], "strategies")
        return [i for i in items if i.get("kind") == kind and not i.get("deleted")][:k]

    def _finetune_insights(self, task: str, ctx: OrganizerContext) -> list[MemoryOp]:
        insights = self._fetch(ctx, task, "insight", self.insight_max)
        trajectories = self._fetch(ctx, task, "trajectory", 10)
        result = ctx.llm.call(
            "distill",
            FINETUNE_PROMPT.format(
                rules="\n".join(f'- id={i["id"]} (score={i.get("score", 0)}) {i["content"]}'
                                for i in insights) or "(none)",
                trajectories="\n".join(
                    f'- [{t.get("outcome")}] {t.get("title")}: {t.get("content", "")[:300]}'
                    for t in trajectories)),
            FINETUNE_SCHEMA, required_keys=("operations",))
        if result is None:
            return []

        valid = {i["id"]: i for i in insights}
        ops: list[MemoryOp] = []
        n_insights = len(insights)
        for raw in result.get("operations", [])[:6]:
            op, rid, rule = raw.get("op"), raw.get("id"), str(raw.get("rule", "")).strip()
            if op == "ADD" and rule:
                iid = new_id()
                ops.append(MemoryOp(
                    op=OpType.ADD, target_type="strategies", target_id=iid,
                    payload={"id": iid, "title": rule[:60], "content": rule,
                             "kind": "insight", "score": 0.0, "embedding_text": rule}))
                n_insights += 1
            elif op == "EDIT" and rid in valid and rule:
                ops.append(MemoryOp(op=OpType.UPDATE, target_type="strategies",
                                    target_id=rid,
                                    payload={"content": rule, "embedding_text": rule}))
            elif op == "REMOVE" and rid in valid:
                ops.append(MemoryOp(op=OpType.DELETE, target_type="strategies",
                                    target_id=rid, payload={"reason": "finetune_remove"}))
                n_insights -= 1
            # hallucinated ids fall through silently-visible: nothing emitted

        if n_insights > self.insight_max and insights:
            worst = min(insights, key=lambda i: i.get("score", 0))
            ops.append(MemoryOp(op=OpType.DELETE, target_type="strategies",
                                target_id=worst["id"], payload={"reason": "insight_cap"}))
        return ops

    def project_insights(self, role: str, insights: list[str],
                         ctx: OrganizerContext) -> list[str]:
        """Role-specific insight rewriting (multi-agent injection path)."""
        if ctx.llm is None or not insights:
            return insights
        result = ctx.llm.call("distill",
                              PROJECT_PROMPT.format(role=role,
                                                    insights="\n".join(f"- {i}" for i in insights)),
                              PROJECT_SCHEMA, required_keys=("insights",))
        return [str(i) for i in result["insights"]] if result else insights

    def backward(self, insight_items: list[dict], reward: float) -> list[MemoryOp]:
        """Reward shaping on retrieved insights (+1 success / -2 failure)."""
        return [MemoryOp(op=OpType.UPDATE, target_type="strategies",
                         target_id=i["id"],
                         payload={"score": float(i.get("score", 0)) + reward})
                for i in insight_items]
