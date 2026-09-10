from collections.abc import Sequence
from typing import Literal

from agmem.core.ops import MemoryOp, OpType

DistillStatus = Literal["started", "completed", "skipped", "partial", "failed"]


def outcome_status(ops: Sequence[MemoryOp]) -> DistillStatus:
    statuses = {op.payload.get("distill_status") for op in ops if op.op == OpType.NOOP}
    produced = any(op.op in (OpType.ADD, OpType.MERGE) for op in ops)
    if "partial" in statuses or ("failed" in statuses and produced):
        return "partial"
    if "failed" in statuses:
        return "failed"
    if "skipped" in statuses and not produced and "completed" not in statuses:
        return "skipped"
    return "completed"
