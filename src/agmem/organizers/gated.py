"""Apply a write-admission policy to any message-driven organizer.

``policies/`` claims its members are cross-cutting — orthogonal to mechanism
choice, in survey arXiv:2603.07670's terms. A gate reachable only through one
mechanism's constructor argument would not honour that claim, and it would also
make the mechanism import the policy. This wrapper is what makes the claim true:
one implementation, no ``admission=`` parameter on any organizer, and
``organizers/amem/`` has no idea policies exist.

    AdmissionGated(AMemOrganizer(), AdmissionGate())

This module sits at the ``organizers/`` root rather than in a package because the
root is the plugin *framework* (``base.py``'s contract, the registry, this
adapter) while every methodology owns a package. ``AdmissionGated`` is an
``Organizer`` but not a methodology: it implements no paper, it composes one with
a policy. See docs/04 §1.2.

Rejected messages never reach the wrapped organizer, so its per-message LLM work
is skipped entirely — 2 calls for A-Mem (Ps1 + Ps2/Ps3), 1 for Zep-graph's entity
extraction. That saving is the entire point of the admission papers, and it only
exists if the gate runs *before* the mechanism.

Applicability is real but not universal, and the limits are properties of the
host's lifecycle rather than of this wrapper:

- **Fits**: ``amem``, ``zep_graph``, ``memoryos`` — per-message organizers that
  spend LLM calls (or capacity) on each turn, so a veto is a real saving.
- **Pointless**: ``passthrough`` — its ``on_message`` returns ``[]`` and the
  facade has already stored/indexed the raw episode before any organizer runs
  (``memory.py`` write-then-organize order). Gating it changes nothing at all.
- **Changes the mechanism, not just its cost**: ``nemori`` buffers messages and
  detects episode boundaries over the stream, so dropping messages mid-stream
  alters segmentation itself — the thing the paper is reproducing. Wrapping it is
  allowed, but it is an ablation, not a cost optimisation, and must be reported
  as one.
- **Not applicable**: ``ace``, ``gmemory``, ``reasoning_bank`` declare no
  ``on_message`` at all — they are task-driven (``on_task_end``). An
  episode-keyed admission gate has nothing to decide about a trajectory. A
  *different* policy would (Memory Worth's discard/suppression governs retrieved
  units regardless of how they were written), which is why this limit belongs to
  admission specifically and not to ``policies/``.

Known gap, in the same spirit as ``ChainedConsumer``'s documented one: under
``ChainedConsumer(AdmissionGated(x))``, ``base.overrides()`` inspects this
wrapper rather than ``x``, so a wrapped organizer with its own ``retire`` policy
(MemoryOS) would be routed through the chained generic retirement path instead.
The admission papers all target the direct message path, so that composition is
out of scope rather than supported; ``consumes`` is mirrored so the combination
at least does not silently mis-subscribe.
"""

from __future__ import annotations

from agmem.core.ops import MemoryOp
from agmem.core.types import Episode
from agmem.organizers.base import MemoryEvent, Organizer, OrganizerContext
from agmem.policies.admission import AdmissionDecision, AdmissionGate


class AdmissionGated(Organizer):
    """``wrapped``, with ``gate`` vetoing messages before they reach it.

    ``name``/``produces``/``consumes`` mirror the wrapped organizer so op actor
    attribution, ``default_memory_types``, and event subscription are unchanged —
    the gate must be invisible to everything except the admit/reject decision."""

    def __init__(self, wrapped: Organizer, gate: AdmissionGate) -> None:
        self.wrapped = wrapped
        self.gate = gate
        self.name = wrapped.name
        self.produces = wrapped.produces
        self.consumes = wrapped.consumes
        # Mirrored for the same reason as the three above: `AgenticMemory.bulk_ingest`
        # routes on this flag, and the class default (False) on the wrapper sent a
        # gated zep_graph down the batched fast path the flag exists to keep it off —
        # its on_message reads the store, so batching shows it the full corpus where
        # per-message ingest shows a prefix.
        self.observes_store_on_message = wrapped.observes_store_on_message
        # A gate left at novelty_types=None (unset) compares N against the
        # host's own output types. Without this defaulting, wrapping any
        # non-A-Mem organizer would fall back to A-Mem's ("notes",), search a
        # type the host never writes, and silently reintroduce upstream defect
        # (a)'s shape — N pinned at 1.0. An explicit novelty_types from the
        # caller (even ("notes",)) is honored untouched.
        if gate.novelty_types is None:
            gate.novelty_types = wrapped.produces
        # Every decision, for artifact capture: the admit rate and feature
        # distribution are the measurement this wiring exists to produce, and
        # `gate.stats` alone cannot say *which* turns were dropped.
        self.decisions: list[AdmissionDecision] = []

    # -- the one hook that is actually gated ---------------------------------

    def on_message(self, episode: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        """Reject -> ``[]`` and the wrapped organizer never sees the message."""
        decision = self.gate.decide(episode, ctx)
        self.decisions.append(decision)
        if not decision.admit:
            return []
        return self.wrapped.on_message(episode, ctx)

    def warm_start(self, corpus: list[Episode], ctx: OrganizerContext) -> list[MemoryOp]:
        """Filter the corpus, then warm-start the wrapped organizer on what was
        admitted.

        Filtering up front rather than relying on the gated ``on_message`` is
        required, not stylistic: ``NemoriOrganizer`` and ``MemoryOSOrganizer``
        both override ``warm_start`` with their own replay, so a gate that only
        sat in ``on_message`` would be bypassed for exactly those two.

        Known divergence between the two ingest paths: every decision here runs
        before ``wrapped.warm_start`` writes anything, so on a fresh store no
        host memories exist during any decision and N is pinned at 1.0 for the
        whole corpus, whereas ``on_message`` scores N live turn-by-turn — the
        two paths implement different gates. Latent, not a corrupted
        measurement: the LoCoMo bench ingests via ``add_message`` (the
        on_message path), so measured runs are unaffected."""
        admitted = []
        for episode in corpus:
            decision = self.gate.decide(episode, ctx)
            self.decisions.append(decision)
            if decision.admit:
                admitted.append(episode)
        return self.wrapped.warm_start(admitted, ctx)

    # -- everything else is pass-through ------------------------------------

    def on_task_end(
        self, trajectory: list[dict], outcome: str, task: str, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Ungated: admission scores a candidate memory against a transcript, and
        a trajectory is neither. Task-driven organizers therefore run untouched
        even when wrapped."""
        return self.wrapped.on_task_end(trajectory, outcome, task, ctx)

    def on_retrieval(
        self, hits: list[tuple[str, str, float]], ctx: OrganizerContext
    ) -> list[MemoryOp]:
        return self.wrapped.on_retrieval(hits, ctx)

    def on_feedback(
        self, memory_ids: list[str], helpful: bool, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        return self.wrapped.on_feedback(memory_ids, helpful, ctx)

    def on_memory_event(self, ev: MemoryEvent, ctx: OrganizerContext) -> list[MemoryOp]:
        return self.wrapped.on_memory_event(ev, ctx)

    def consolidate(self, ctx: OrganizerContext) -> list[MemoryOp]:
        return self.wrapped.consolidate(ctx)

    def flush_buffer(self, ctx: OrganizerContext) -> list[MemoryOp]:
        return self.wrapped.flush_buffer(ctx)

    def recent_context(self) -> str:
        """Forwarded, not gated: admission governs writes, and this is a read
        channel. Benches call it on the instance they hold — this wrapper — so
        the base class's "" here silently dropped a gated MemoryOS's verbatim
        STM injection at QA."""
        return self.wrapped.recent_context()

    def retire(self, superseded: set[str]) -> list[MemoryOp]:
        return self.wrapped.retire(superseded)

    def patch_unit(self, unit: Episode) -> None:
        self.wrapped.patch_unit(unit)

    @property
    def _cursor_scope(self) -> str | None:  # type: ignore[override]
        """The consolidate cursor belongs to the wrapped organizer.

        ``AgenticMemory`` stamps ``_cursor_scope`` on the instance it holds — this
        wrapper — when several organizers share a name. Forwarding it keeps the
        wrapped organizer's ``cursor_key`` (and so its persisted progress) the
        same whether or not a gate is attached."""
        return self.wrapped._cursor_scope

    @_cursor_scope.setter
    def _cursor_scope(self, value: str | None) -> None:
        self.wrapped._cursor_scope = value
