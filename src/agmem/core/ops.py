"""Memory operations and the append-only evolution log.

Every mutation any organizer makes is expressed as a ``MemoryOp`` and
recorded before being applied. This gives all seven methodologies one
audit trail (ACE delta ops, G-Memory rule ops, Zep invalidation, A-Mem
evolution) and makes state reproducible by replay.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from agmem.core.types import utcnow


class OpType(str, Enum):
    """The mutation kinds every organizer's output is normalized into."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    DELETE = "DELETE"
    INVALIDATE = "INVALIDATE"  # bi-temporal: never physically remove
    LINK = "LINK"
    TAG = "TAG"


@dataclass
class MemoryOp:
    """The only channel through which an organizer may mutate memory state.

    Organizers never write to a store directly (docs/04 §2) — they return
    `MemoryOp`s, which `AgenticMemory` appends to the evolution log and then
    applies. `target_type` must be one of `MEMORY_TYPES`; `payload` shape is
    op- and target-type-specific (interpreted by `AgenticMemory._apply_one`).
    """

    op: OpType
    target_type: str  # one of MEMORY_TYPES
    target_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    actor: str = "system"  # organizer name that produced the op
    t_transaction: datetime = field(default_factory=utcnow)

    def to_json(self) -> str:
        """Serialize for the evolution log; `op`/`t_transaction` become strings."""
        d = asdict(self)
        d["op"] = self.op.value
        d["t_transaction"] = self.t_transaction.isoformat()
        return json.dumps(d, ensure_ascii=False, default=str)

    @classmethod
    def from_row(
        cls,
        op: str,
        target_type: str,
        target_id: str,
        payload: str,
        actor: str,
        t_transaction: str,
    ) -> "MemoryOp":
        """Inverse of `to_json`; `payload` may be `""`/`None` (empty dict then)."""
        return cls(
            op=OpType(op),
            target_type=target_type,
            target_id=target_id,
            payload=json.loads(payload) if payload else {},
            actor=actor,
            t_transaction=datetime.fromisoformat(t_transaction),
        )


class EvolutionLog(Protocol):
    """Append-only op log. Implemented by the doc store."""

    def append(self, ops: list[MemoryOp]) -> None:
        """Persist before the caller applies the ops (log-ahead, docs/12 §3.2)."""
        ...

    def tail(self, n: int = 20) -> list[MemoryOp]:
        """Most recent `n` ops, oldest-first (so replay order is preserved)."""
        ...

    def count(self) -> int: ...
