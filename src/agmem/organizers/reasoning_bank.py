"""ReasoningBank organizer (arXiv:2509.25140, ICLR'26).

on_task_end: (optional self-judge) -> distill up to 3 strategy items from
the trajectory — success AND failure both teach — then append-only ADD.
No pruning/merging by design (the paper's deliberate simplification).
The judge's reason is appended to the trajectory before extraction, as
upstream appends autoeval thoughts ("The task succeeded/failed because:").
Each task also stores an EXPERIENCE record (task query + its item ids):
retrieving memory_type "experiences" with k=1 reproduces upstream's
top-1-experience injection (the pipeline expands it to the member items);
retrieving "strategies" directly is the item-level convenience mode.

LLM roles used: ``judge`` (binary success call, t=0.0 in the paper),
``distill`` (extraction, t=1.0 in the paper — set per agent-bench config).
The SI rules ride in the system message, as upstream's one_step_chat does;
``persona`` prepends a domain persona ("You are an expert in web
navigation.") when the benchmark has one.
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
# Injected as the SYSTEM message (upstream one_step_chat(system_msg=SI)).
EXTRACT_SUCCESS_SI = """The agent SUCCEEDED at this task. First think about WHY the approach \
worked, then distill at most 3 transferable strategy items.

Rules:
- Do NOT embed literal product names, queries, or task-specific strings.
- Prefer concrete executable procedures over abstract principles.
- description is ONE sentence stating WHEN to apply (and when not to).
- content is 1-3 sentences describing the insight. No duplicates.

Respond with a single JSON object:
{"items": [{"title": "...", "description": "...", "content": "..."}]}"""

EXTRACT_FAILURE_SI = """The agent FAILED at this task. Reflect on WHY it failed, then distill \
at most 3 preventative lessons or recovery procedures as strategy items.

Rules:
- Do NOT embed literal product names, queries, or task-specific strings.
- Prefer concrete executable procedures over abstract principles.
- description is ONE sentence stating WHEN to apply (and when not to).
- content is 1-3 sentences describing the insight. No duplicates.

Respond with a single JSON object:
{"items": [{"title": "...", "description": "...", "content": "..."}]}"""

# User message: query + trajectory, as upstream induce_memory formats it.
EXTRACT_USER_TEMPLATE = """**Query:** {task}

**Trajectory:**
{trajectory}"""


def _format_trajectory(trajectory: list[dict], max_chars: int = 60000) -> str:
    # upstream feeds the full trajectory untruncated; the (raised) cap is a
    # context-overflow guard only — round-5 (f)
    text = "\n".join(json.dumps(step, ensure_ascii=False, default=str) for step in trajectory)
    if len(text) > max_chars:
        # keep head and tail — failures usually surface at the end
        half = max_chars // 2
        text = text[:half] + "\n...[truncated]...\n" + text[-half:]
    return text


class ReasoningBankOrganizer(Organizer):
    """ReasoningBank (see module docstring for the paper/upstream mapping)."""

    name = "reasoning_bank"

    def __init__(
        self, max_items: int = 3, self_judge: bool = True, persona: str | None = None
    ) -> None:
        """`self_judge=True` runs the judge role when `outcome` isn't already
        "success"/"failure"; set False to always trust the caller-supplied `outcome`
        and skip that LLM call. `persona` is prepended to the extraction system
        message verbatim when set (see module docstring)."""
        self.max_items = max_items
        self.self_judge = self_judge
        self.persona = persona  # e.g. "You are an expert in web navigation."

    def on_task_end(
        self, trajectory: list[dict], outcome: str, task: str, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Returns `[]` without calling the LLM if `ctx.llm` is unset, if the judge
        or extraction call drops (see `StructuredCaller`), or if extraction returns
        no items — never raises on LLM failure. Items missing a required field are
        skipped individually rather than failing the whole batch."""
        if ctx.llm is None:
            logger.warning(
                "reasoning_bank: no LLM configured — skipping distillation "
                "(explicit skip, task=%.60s)",
                task,
            )
            return []

        traj_text = _format_trajectory(trajectory)
        reason = ""

        if outcome not in ("success", "failure") and self.self_judge:
            verdict = ctx.llm.call(
                "judge",
                JUDGE_PROMPT.format(task=task, trajectory=traj_text),
                JUDGE_SCHEMA,
                required_keys=("success",),
            )
            if verdict is None:
                return []  # drop already counted by StructuredCaller
            outcome = "success" if verdict["success"] else "failure"
            reason = str(verdict.get("reason", "")).strip()

        if reason:
            # upstream appends autoeval thoughts to the trajectory so the
            # extractor reflects on the actual cause (round-5 (c))
            status = "succeeded" if outcome == "success" else "failed"
            traj_text += f"\n\nThe task {status} because: {reason}"

        si = EXTRACT_SUCCESS_SI if outcome == "success" else EXTRACT_FAILURE_SI
        if self.persona:
            si = f"{self.persona}\n\n{si}"
        result = ctx.llm.call(
            "distill",
            EXTRACT_USER_TEMPLATE.format(task=task, trajectory=traj_text),
            EXTRACT_SCHEMA,
            required_keys=("items",),
            system=si,
        )
        if result is None or not isinstance(result.get("items"), list):
            return []

        ops: list[MemoryOp] = []
        item_ids: list[str] = []
        for raw in result["items"][: self.max_items]:
            if not all(
                isinstance(raw.get(f), str) and raw.get(f)
                for f in ("title", "description", "content")
            ):
                continue  # field-level fallback: keep valid items, skip broken ones
            item = StrategyItem(
                title=raw["title"],
                description=raw["description"],
                content=raw["content"],
                outcome=outcome,
                namespace=ctx.namespace,
            )
            ops.append(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="strategies",
                    target_id=item.id,
                    payload={
                        "id": item.id,
                        "title": item.title,
                        "description": item.description,
                        "content": item.content,
                        "outcome": outcome,
                        "embedding_text": item.embedding_text(),
                    },
                )
            )
            item_ids.append(item.id)

        if item_ids:
            # experience record: the retrieval unit upstream actually uses —
            # task-query embedding, expanded to its member items at read time
            experience_id = new_id()
            ops.append(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="experiences",
                    target_id=experience_id,
                    payload={
                        "id": experience_id,
                        "task": task,
                        "outcome": outcome,
                        "item_ids": item_ids,
                        "embedding_text": task,
                    },
                )
            )
        return ops
