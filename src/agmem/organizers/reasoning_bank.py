"""ReasoningBank organizer (arXiv:2509.25140, ICLR'26).

on_task_end: (optional self-judge) -> distill up to 3 strategy items from
the trajectory — success AND failure both teach — then append-only ADD.
No pruning/merging by design (the paper's deliberate simplification).

LLM roles used: ``judge`` (binary success call, t=0.0 in the paper),
``distill`` (extraction, t=1.0 in the paper — we keep role-level config).
"""

from __future__ import annotations

import json
import logging

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, StrategyItem, new_id
from agmem.organizers.base import Organizer, OrganizerContext

logger = logging.getLogger("agmem.organizers.reasoning_bank")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"success": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["success"],
}

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["title", "description", "content"],
            },
        }
    },
    "required": ["items"],
}

JUDGE_PROMPT = """Judge whether the agent successfully completed the task.

Task: {task}

Trajectory:
{trajectory}

Return JSON: {{"success": true/false, "reason": "one sentence"}}"""

# Condensed from the paper's SUCCESSFUL_SI / FAILED_SI instructions.
EXTRACT_SUCCESS_PROMPT = """The agent SUCCEEDED at this task. First think about WHY the approach \
worked, then distill up to 3 transferable strategy items.

Rules:
- Do NOT embed literal product names, queries, or task-specific strings.
- Prefer concrete executable procedures over abstract principles.
- description must state WHEN to apply (and when not to). No duplicates.

Task: {task}
Trajectory:
{trajectory}

Return JSON: {{"items": [{{"title": "...", "description": "...", "content": "1-5 sentences"}}]}}"""

EXTRACT_FAILURE_PROMPT = """The agent FAILED at this task. Reflect on WHY it failed, then distill \
up to 3 preventative lessons or recovery procedures as strategy items.

Rules:
- Do NOT embed literal product names, queries, or task-specific strings.
- Prefer concrete executable procedures over abstract principles.
- description must state WHEN to apply (and when not to). No duplicates.

Task: {task}
Trajectory:
{trajectory}

Return JSON: {{"items": [{{"title": "...", "description": "...", "content": "1-5 sentences"}}]}}"""


def _format_trajectory(trajectory: list[dict], max_chars: int = 6000) -> str:
    text = "\n".join(json.dumps(step, ensure_ascii=False, default=str) for step in trajectory)
    if len(text) > max_chars:
        # keep head and tail — failures usually surface at the end
        half = max_chars // 2
        text = text[:half] + "\n...[truncated]...\n" + text[-half:]
    return text


class ReasoningBankOrganizer(Organizer):
    name = "reasoning_bank"

    def __init__(self, max_items: int = 3, self_judge: bool = True) -> None:
        self.max_items = max_items
        self.self_judge = self_judge

    def on_task_end(self, trajectory: list[dict], outcome: str,
                    task: str, ctx: OrganizerContext) -> list[MemoryOp]:
        if ctx.llm is None:
            logger.warning("reasoning_bank: no LLM configured — skipping distillation "
                           "(explicit skip, task=%.60s)", task)
            return []

        traj_text = _format_trajectory(trajectory)

        if outcome not in ("success", "failure") and self.self_judge:
            verdict = ctx.llm.call(
                "judge", JUDGE_PROMPT.format(task=task, trajectory=traj_text),
                JUDGE_SCHEMA, required_keys=("success",),
            )
            if verdict is None:
                return []  # drop already counted by StructuredCaller
            outcome = "success" if verdict["success"] else "failure"

        prompt_tpl = EXTRACT_SUCCESS_PROMPT if outcome == "success" else EXTRACT_FAILURE_PROMPT
        result = ctx.llm.call(
            "distill", prompt_tpl.format(task=task, trajectory=traj_text),
            EXTRACT_SCHEMA, required_keys=("items",),
        )
        if result is None or not isinstance(result.get("items"), list):
            return []

        ops: list[MemoryOp] = []
        for raw in result["items"][: self.max_items]:
            if not all(isinstance(raw.get(f), str) and raw.get(f)
                       for f in ("title", "description", "content")):
                continue  # field-level fallback: keep valid items, skip broken ones
            item = StrategyItem(
                title=raw["title"], description=raw["description"],
                content=raw["content"], outcome=outcome, namespace=ctx.namespace,
            )
            ops.append(MemoryOp(
                op=OpType.ADD, target_type="strategies", target_id=item.id,
                payload={
                    "id": item.id, "title": item.title, "description": item.description,
                    "content": item.content, "outcome": outcome,
                    "embedding_text": item.embedding_text(),
                },
            ))
        return ops
