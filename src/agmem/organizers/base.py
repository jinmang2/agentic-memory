"""Organizer plugin interface — one methodology per plugin.

Organizers never touch stores directly. They receive read access via the
context and return ``MemoryOp`` lists; the facade logs and applies them.
That keeps methodology code decoupled from storage and makes every
mutation auditable/replayable (docs/04 §2).

Lifecycle hooks (spec §1):
- ``on_message``, ``on_task_end``, ``on_retrieval``, ``on_feedback``: entry points
- ``on_memory_event``: chaining hook for subscribed organizers
- ``consolidate``: deferred management pass with cursor recovery
- ``flush_buffer``: end-of-ingestion drain for organizers that buffer
- ``retire``, ``patch_unit``: derived-state upkeep under chained composition

Every hook is declared on ``Organizer`` with a no-op default, so callers
invoke them unconditionally instead of probing with ``getattr``/``hasattr``.
Where a caller must distinguish "the subclass really implements this" from
"the base no-op ran" — ``ChainedConsumer``'s custom-vs-generic retirement
split is the only such case — use ``overrides()`` rather than attribute
presence.
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
    # Memory types this organizer's ops write. Declarative counterpart to
    # ``consumes``: it drives ``AgenticMemory.default_memory_types`` so callers
    # that don't name types (MCP, ad-hoc use) search what the active
    # methodology actually produced. It does NOT gate the read-path steps —
    # those stay keyed on the memory type alone, so items written straight to a
    # store still get them (retrieval/steps.py).
    # ORDER IS THE READ ORDER and is load-bearing: an expansion step dedups
    # against ids already in the bundle, so a type another type's step pulls in
    # must be listed first (see ZepGraphOrganizer: facts before entities).
    produces: tuple[str, ...] = ()

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

    def on_feedback(
        self, memory_ids: list[str], helpful: bool, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Usage outcome for previously served memories, from
        ``AgenticMemory.report_feedback``.

        Feedback semantics are methodology-owned, so they live here rather than
        in the facade: ACE counts helpful/harmful per bullet, G-Memory shapes
        insight scores by reward. The facade used to branch on ``target_type``
        itself, which meant a ReasoningBank strategy item — whose paper is
        deliberately append-only, with no feedback loop at all — silently
        received G-Memory's +1/-2 because the two share the ``strategies`` type.
        Fanning out to organizers instead makes "who owns this rule" the same
        question as "which organizer is active".

        ``memory_ids`` may name items this organizer knows nothing about;
        implementations filter and ignore the rest."""
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

    def flush_buffer(self, ctx: OrganizerContext) -> list[MemoryOp]:
        """End-of-ingestion drain, called by ``AgenticMemory.flush()``.

        Organizers that hold a buffer (Nemori's segment buffer, MemoryOS's STM)
        override this to emit whatever a boundary/capacity trigger would never
        reach, so a partial tail is not stranded. Buffer-less organizers keep
        the no-op."""
        return []

    def retire(self, superseded: set[str]) -> list[MemoryOp]:
        """Retire derived state whose source units were absorbed by a MERGE.

        Only reached under chained composition (``ChainedConsumer``), which
        falls back to a generic 1:1 INVALIDATE when this hook is not overridden
        — so overriding it means "I own my retirement policy" (MemoryOS
        invalidates a page only once *all* its sources are gone)."""
        return []

    def patch_unit(self, unit: Episode) -> None:
        """Revise a not-yet-consolidated buffered unit in place.

        Only reached under chained composition, when an upstream organizer
        UPDATEs a unit this one has buffered but not yet turned into a derived
        memory. The no-op default means the derived item is left stale —
        documented staleness (spec §3), not an error."""

    # ---- consolidate cursor -------------------------------------------------

    @property
    def cursor_key(self) -> str:
        """Doc-store id of this organizer's consolidate cursor.

        Defaults to the organizer name; ``AgenticMemory`` overwrites the
        backing ``_cursor_scope`` with ``name#idx`` when one memory holds
        several instances of the same organizer (they would otherwise share
        and clobber one cursor). Single-instance configs keep the bare name, so
        cursors persisted before instance scoping existed still resolve.
        ``get_items`` is not namespace-filtered, so a doc store shared across
        namespaces could still collide — harmless with per-namespace db files."""
        return f"consolidate:{self._cursor_scope or self.name}"

    _cursor_scope: str | None = None

    def read_cursor(self, ctx: OrganizerContext) -> int:
        """Read this organizer's consolidate cursor seq (0 if unset)."""
        items = ctx.doc_store.get_items([self.cursor_key], "state")
        return int(items[0].get("seq", 0)) if items else 0

    def cursor_op(self, seq: int) -> MemoryOp:
        """Emit the cursor-advance op for ``cursor_key``.

        ADD, not UPDATE: the cursor's whole state is ``seq``, so a full replace
        is exactly right, and it works on the first advance when no cursor row
        exists yet. It used to be an UPDATE, which only worked because
        ``_apply_one`` treated UPDATE as an upsert — bookkeeping quietly widening
        the meaning of an op every methodology also uses. Reading it back is
        unchanged (``read_cursor`` sees the same ``{"seq": n}`` item)."""
        return MemoryOp(
            op=OpType.ADD,
            target_type="state",
            target_id=self.cursor_key,
            payload={"seq": seq},
        )

    def warm_start(self, corpus: list[Episode], ctx: OrganizerContext) -> list[MemoryOp]:
        """Default warm start: replay the corpus through on_message."""
        ops: list[MemoryOp] = []
        for episode in corpus:
            ops.extend(self.on_message(episode, ctx))
        return ops


def overrides(organizer: Organizer, hook: str) -> bool:
    """True when ``organizer``'s class really implements ``hook``.

    Since every hook now has a no-op default, ``hasattr`` can no longer answer
    "did the subclass opt in?" — this compares the bound function against
    ``Organizer``'s to tell a real implementation from the inherited no-op."""
    return getattr(type(organizer), hook, None) is not getattr(Organizer, hook, None)
