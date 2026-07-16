"""Organizer plugin interface — one methodology per plugin.

Organizers never touch stores directly. They receive read access via the
context and return ``MemoryOp`` lists; the facade logs and applies them.
That keeps methodology code decoupled from storage and makes every
mutation auditable/replayable (docs/04 §2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agmem.core.ops import MemoryOp
from agmem.core.types import Episode
from agmem.embed.base import Embedder
from agmem.stores.base import DocStore, VectorStore


@dataclass
class OrganizerContext:
    doc: DocStore
    vec: VectorStore
    embedder: Embedder
    namespace: str
    llm: Any | None = None  # role-routing LLM client; None when no endpoint


class Organizer:
    """Base class. Subclasses override the hooks they care about."""

    name = "base"

    def on_message(self, ep: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        return []

    def on_task_end(self, trajectory: list[dict], outcome: str,
                    task: str, ctx: OrganizerContext) -> list[MemoryOp]:
        return []

    def warm_start(self, corpus: list[Episode], ctx: OrganizerContext) -> list[MemoryOp]:
        """Default warm start: replay the corpus through on_message."""
        ops: list[MemoryOp] = []
        for ep in corpus:
            ops.extend(self.on_message(ep, ctx))
        return ops
