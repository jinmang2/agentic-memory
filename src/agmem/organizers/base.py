"""Organizer plugin interface — one methodology per plugin.

Organizers never touch stores directly. They receive read access via the
context and return ``MemoryOp`` lists; the facade logs and applies them.
That keeps methodology code decoupled from storage and makes every
mutation auditable/replayable (docs/04 §2).

Lifecycle hooks (spec §1):
- ``on_message``, ``on_task_end``, ``on_retrieval``: entry points
- ``on_memory_event``: chaining hook for subscribed organizers
- ``consolidate``: deferred management pass with cursor recovery
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode
from agmem.embed.base import Embedder
from agmem.stores.base import DocStore, VectorStore


@dataclass
class MemoryEvent:
    """One applied ADD/UPDATE/MERGE, delivered to subscribed organizers.

    ``supersedes`` rides only on MERGE and lists same-type ids the merge
    absorbed — the atomic channel managers use to retire derived state
    (spec §1.2); INVALIDATE/DELETE ops are never propagated as events."""

    source: str
    op: OpType
    target_type: str
    target_id: str
    payload: dict
    supersedes: tuple[str, ...] = ()


@dataclass
class OrganizerContext:
    """Read-only handles an organizer hook needs; never mutated by hooks (docs/04 §2).

    ``doc_store``/``vector_store``/``graph_store`` are the facade's own store instances —
    hooks read from them but express writes only as returned ``MemoryOp``s. ``embedder``
    is shared so organizer-computed embeddings stay consistent with retrieval's."""

    doc_store: DocStore
    vector_store: VectorStore
    embedder: Embedder
    namespace: str
    llm: Any | None = None  # role-routing LLM client; None when no endpoint
    graph_store: Any | None = None  # shared graph store (Zep/G-Memory), data_dir-persistent


class Organizer:
    """Base class: subclasses override the hooks they care about.

    Hooks only read via ``ctx.*_store``; they never write directly — mutations are
    expressed as returned ``MemoryOp`` lists, which the facade logs (append-only) before
    applying (docs/04 §2). Unoverridden hooks are no-ops (return [])."""

    name = "base"
    consumes: tuple[str, ...] = ()

    def on_message(self, episode: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        """Called once per stored episode; the raw episode is already durable/searchable
        by this point (write-then-organize order, docs/04 §2)."""
        return []

    def on_task_end(
        self, trajectory: list[dict], outcome: str, task: str, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Called once per completed task; the facade never persists the full
        ``trajectory`` itself, so this hook is the only place methodologies see it."""
        return []

    def on_retrieval(
        self, hits: list[tuple[str, str, float]], ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Read->write feedback: called after every search with the served
        (item_id, memory_type, score) triples. Restores the upstream loops
        the round-5 audit found missing — MemoryOS visit-heat (N_visit),
        G-Memory served-insight cache for backward reward. Must be cheap:
        no LLM calls here."""
        return []

    def on_memory_event(self, ev: MemoryEvent, ctx: OrganizerContext) -> list[MemoryOp]:
        """Chaining hook: another organizer's applied output, if subscribed
        via ``consumes``. Runs inline (same dispatch as on_message); returned
        ops are applied but NOT re-propagated (depth=1)."""
        return []

    def consolidate(self, ctx: OrganizerContext) -> list[MemoryOp]:
        """Deferred management pass — only via AgenticMemory.consolidate().
        Implementations resume from read_cursor() and end their batch with
        cursor_op(new_seq) so progress survives restarts (spec §1.4)."""
        return []

    def read_cursor(self, ctx: OrganizerContext) -> int:
        """Read this organizer's consolidate cursor seq (0 if unset).

        The cursor id is scoped to ``self.name`` only, so two instances of the
        same organizer class in one memory would share (and clobber) one
        cursor; and ``get_items`` is not namespace-filtered, so a doc store
        shared across namespaces could collide too. Both are harmless in the
        current configs (one instance per class; per-namespace db files)."""
        items = ctx.doc_store.get_items([f"consolidate:{self.name}"], "state")
        return int(items[0].get("seq", 0)) if items else 0

    def cursor_op(self, seq: int) -> MemoryOp:
        """Emit the cursor-advance op (see read_cursor for the name/namespace
        scope constraint — same ``consolidate:{self.name}`` id)."""
        return MemoryOp(
            op=OpType.UPDATE,
            target_type="state",
            target_id=f"consolidate:{self.name}",
            payload={"seq": seq},
        )

    def warm_start(self, corpus: list[Episode], ctx: OrganizerContext) -> list[MemoryOp]:
        """Default warm start: replay the corpus through on_message."""
        ops: list[MemoryOp] = []
        for episode in corpus:
            ops.extend(self.on_message(episode, ctx))
        return ops
