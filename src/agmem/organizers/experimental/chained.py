"""Cross-organizer chaining (Task 12/13) — an experimental composition.

A ``ChainedConsumer`` wraps a paper-faithful organizer and feeds it another
organizer's *episodes* instead of the raw message stream, so methodologies can
be stacked (Nemori episodes -> A-Mem notes, or -> MemoryOS pages). This has no
counterpart in any of the source papers; it lives here so the wrapped
organizers stay messages-only and pure.

The seam is deliberately narrow and lossy: the upstream ``MemoryEvent`` is
flattened to an ``Episode`` carrying only ``content`` + ``timestamp`` (the
upstream ``title``/``source_episode_ids``/``embedding_text`` are dropped), then
fed to the wrapped organizer's ordinary ``on_message``. Wrapped organizers may
opt into two lifecycle hooks:

- ``retire(superseded_ids) -> ops``: custom derived-state retirement (MemoryOS
  invalidates a page only once all its source units are gone). When absent, the
  consumer INVALIDATEs the single item it tracked 1:1 for that source (A-Mem
  notes).
- ``patch_unit(unit)``: in-place UPDATE of a not-yet-consolidated unit
  (MemoryOS STM). When absent, an UPDATE event leaves the derived item stale
  rather than re-ingesting (documented staleness, spec §3).
"""

from __future__ import annotations

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode
from agmem.organizers.base import MemoryEvent, Organizer, OrganizerContext


class ChainedConsumer(Organizer):
    """Wraps ``wrapped`` and drives it from another organizer's ``source_type``
    events. ``name`` mirrors the wrapped organizer's so applied ops keep the
    same actor attribution; the raw message stream is ignored (input arrives
    via ``on_memory_event`` only)."""

    def __init__(self, wrapped: Organizer, source_type: str = "episodes") -> None:
        self.wrapped = wrapped
        self.name = wrapped.name
        self.consumes = (source_type,)
        # source event id -> (produced_id, produced_type); only used for the
        # generic 1:1 retire path (wrapped organizers without their own retire).
        self._produced: dict[str, tuple[str, str]] = {}

    def on_message(self, episode: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        """No-op: a chained consumer never reads the raw stream (its input is
        another organizer's episodes, delivered via ``on_memory_event``)."""
        return []

    def on_memory_event(self, ev: MemoryEvent, ctx: OrganizerContext) -> list[MemoryOp]:
        """Retire derived state for any superseded sources first, then either
        patch (UPDATE) or feed (ADD/MERGE) the flattened unit to the wrapped
        organizer's ``on_message`` — mirroring what the wrapped organizers'
        former ``input="episodes"`` branches did, so behavior is unchanged."""
        ops: list[MemoryOp] = []
        if ev.supersedes:
            ops.extend(self._retire(set(ev.supersedes)))
        unit = Episode(
            content=str(ev.payload.get("content", "")),
            role="episode",
            id=ev.target_id,
            namespace=ctx.namespace,
            meta={"date": ev.payload.get("timestamp", "")},
        )
        if ev.op is OpType.UPDATE:
            patch = getattr(self.wrapped, "patch_unit", None)
            if patch is not None:
                patch(unit)
            return ops  # no re-ingest on UPDATE: documented staleness (spec §3)
        produced = self.wrapped.on_message(unit, ctx)  # ADD / MERGE feed
        if produced and not hasattr(self.wrapped, "retire"):
            head = produced[0]  # first op is the primary ADD of the derived item
            self._produced[ev.target_id] = (head.target_id, head.target_type)
        return ops + produced

    def _retire(self, superseded: set[str]) -> list[MemoryOp]:
        retire = getattr(self.wrapped, "retire", None)
        if retire is not None:
            return retire(superseded)
        ops: list[MemoryOp] = []
        for source_id in superseded:
            produced = self._produced.pop(source_id, None)
            if produced:
                ops.append(
                    MemoryOp(
                        op=OpType.INVALIDATE,
                        target_type=produced[1],
                        target_id=produced[0],
                        payload={"reason": "episode_superseded"},
                    )
                )
        return ops
