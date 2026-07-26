"""Cross-organizer chaining (Task 12/13) — an experimental composition.

A ``ChainedConsumer`` wraps a paper-faithful organizer and feeds it another
organizer's output instead of the raw message stream, so methodologies can be
stacked (Nemori episodes -> A-Mem notes, or -> MemoryOS pages). The stacking
adapter itself has no counterpart in the source papers; it lives here so the
wrapped organizers stay messages-only and pure.

One configuration IS a published experiment: Nemori v4 §Table 7 feeds Nemori's
distilled knowledge K to A-MEM/MemoryOS *in place of raw messages* and reports
45-64% less storage with core scores +1.9%~+6.1%. That is
``ChainedConsumer(AMemOrganizer(), "semantic")``. The paper does not say at what
granularity K arrives, and it matters:

- per fact (``batch_key=None``): one note per distilled fact. The literal
  reading, but A-Mem's Ps1 then analyzes a single atomic sentence, so the
  keywords/context it produces are near-vacuous.
- per episode (``batch_key="episode_id"``): the facts distilled from one Nemori
  episode become one note. Keeps a note ≈ an event, and provenance points at the
  upstream episode rather than at an individual fact.

Both are wired so measurement can discriminate: only one of them can land in the
paper's 45-64% storage band.

Caveat for either granularity: Nemori's calibration prompt bans time/date from
semantic statements, so K carries no timestamp and the derived notes fall back to
the ingest wall clock. Feeding K instead of messages therefore gives up A-Mem's
per-note conversation time — relevant when reading temporal-category results.

The seam is deliberately narrow and lossy: the upstream ``MemoryEvent`` is
flattened to an ``Episode`` carrying only ``content`` + ``timestamp`` (the
upstream ``title``/``source_episode_ids``/``embedding_text`` are dropped), then
fed to the wrapped organizer's ordinary ``on_message``. Wrapped organizers may
opt into two lifecycle hooks (both declared on ``Organizer`` as no-ops, so
"opting in" means overriding them — detected via ``base.overrides``):

- ``retire(superseded_ids) -> ops``: custom derived-state retirement (MemoryOS
  invalidates a page only once all its source units are gone). Not overridden ->
  the consumer INVALIDATEs the single item it tracked 1:1 for that source
  (A-Mem notes).
- ``patch_unit(unit)``: in-place UPDATE of a not-yet-consolidated unit
  (MemoryOS STM). Not overridden -> an UPDATE event leaves the derived item
  stale rather than re-ingesting (documented staleness, spec §3).

Known gap: ``flush_buffer`` drains this adapter's own pending batch but is
deliberately NOT forwarded to the wrapped organizer, so a chained MemoryOS's
partial STM tail is still stranded at ``AgenticMemory.flush()``. Forwarding it
would change the measured ``nemori_memoryos`` numbers, so it stays a separate
decision rather than a refactor side effect.
"""

from __future__ import annotations

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode
from agmem.organizers.base import MemoryEvent, Organizer, OrganizerContext, overrides


class ChainedConsumer(Organizer):
    """Wraps ``wrapped`` and drives it from another organizer's ``source_type``
    events. ``name`` mirrors the wrapped organizer's so applied ops keep the
    same actor attribution; the raw message stream is ignored (input arrives
    via ``on_memory_event`` only)."""

    def __init__(
        self,
        wrapped: Organizer,
        source_type: str = "episodes",
        batch_key: str | None = None,
    ) -> None:
        """``batch_key`` names a source payload field whose value groups
        consecutive events into one downstream unit (``"episode_id"`` for
        Nemori's semantic facts). ``None`` feeds every event separately, which is
        the original behavior."""
        self.wrapped = wrapped
        self.name = wrapped.name
        self.consumes = (source_type,)
        # the wrapper writes exactly what the wrapped organizer writes, so it
        # must carry the same read-type declaration (default_memory_types)
        self.produces = wrapped.produces
        self.batch_key = batch_key
        # batch value -> (produced_id, produced_type), and batch value -> the
        # source ids still alive in it. Only used for the generic retire path
        # (wrapped organizers without their own retire). Unbatched mode uses the
        # event id as its own batch value, so the "retire once every member is
        # superseded" rule degenerates to the original 1:1 rule.
        self._produced: dict[str, tuple[str, str]] = {}
        self._members: dict[str, set[str]] = {}
        self._batch_of: dict[str, str] = {}
        # accumulating batch (batched mode only): source id -> content
        self._pending: dict[str, str] = {}
        self._pending_key: str | None = None
        self._pending_date: str = ""

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
        content = str(ev.payload.get("content", ""))
        date = str(ev.payload.get("timestamp", ""))
        if ev.op is OpType.UPDATE:
            if ev.target_id in self._pending:  # still accumulating -> revise in place
                self._pending[ev.target_id] = content
                return ops
            self.wrapped.patch_unit(self._unit(ev.target_id, content, date, ctx))
            return ops  # no re-ingest on UPDATE: documented staleness (spec §3)

        if self.batch_key is None:
            return ops + self._feed(ev.target_id, {ev.target_id: content}, date, ctx)

        # batched: a change in the key value closes the previous batch
        key = str(ev.payload.get(self.batch_key) or ev.target_id)
        if self._pending and key != self._pending_key:
            ops.extend(self._flush_pending(ctx))
        if not self._pending:
            self._pending_key, self._pending_date = key, date
        self._pending[ev.target_id] = content
        return ops

    def flush_buffer(self, ctx: OrganizerContext) -> list[MemoryOp]:
        """Feed the last accumulating batch, which has no following event to
        close it. NOT forwarded to the wrapped organizer — see the module
        docstring's known gap."""
        return self._flush_pending(ctx) if self._pending else []

    # ---- internals ----------------------------------------------------------

    def _unit(self, unit_id: str, content: str, date: str, ctx: OrganizerContext) -> Episode:
        return Episode(
            content=content,
            role="episode",
            id=unit_id,
            namespace=ctx.namespace,
            meta={"date": date},
        )

    def _flush_pending(self, ctx: OrganizerContext) -> list[MemoryOp]:
        pending, key, date = self._pending, self._pending_key or "", self._pending_date
        self._pending, self._pending_key, self._pending_date = {}, None, ""
        return self._feed(key, pending, date, ctx)

    def _feed(
        self, key: str, members: dict[str, str], date: str, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Hand one unit (a single event, or a whole batch joined) to the wrapped
        organizer's ordinary ``on_message`` and record what it produced."""
        if not members:
            return []
        produced = self.wrapped.on_message(
            self._unit(key, "\n".join(members.values()), date, ctx), ctx
        )
        if produced and not overrides(self.wrapped, "retire"):
            head = produced[0]  # first op is the primary ADD of the derived item
            self._produced[key] = (head.target_id, head.target_type)
            self._members[key] = set(members)
            for source_id in members:
                self._batch_of[source_id] = key
        return produced

    def _retire(self, superseded: set[str]) -> list[MemoryOp]:
        if overrides(self.wrapped, "retire"):
            return self.wrapped.retire(superseded)  # organizer owns its policy
        ops: list[MemoryOp] = []
        for source_id in superseded:
            self._pending.pop(source_id, None)  # not fed downstream yet
            key = self._batch_of.pop(source_id, None)
            if key is None:
                continue
            members = self._members.get(key)
            if members is None:
                continue
            members.discard(source_id)
            if members:
                continue  # partially absorbed: the derived item still has sources
            self._members.pop(key, None)
            produced = self._produced.pop(key, None)
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
