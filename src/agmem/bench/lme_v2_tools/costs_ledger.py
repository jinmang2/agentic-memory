"""Deterministic aggregation and budget replay for cost reports."""

from __future__ import annotations

from pathlib import Path

from .budget import BudgetDecision, BudgetState, BudgetStatus, ReservationRequest
from .costs_types import ComponentTotal, CostComponent, CostInputError, UsageEvent


def component_totals(
    events: tuple[UsageEvent, ...],
    expected: tuple[CostComponent, ...],
    zero_components: tuple[CostComponent, ...],
) -> dict[CostComponent, ComponentTotal]:
    """Aggregate usage rows without treating absent expected components as zero."""
    totals: dict[CostComponent, ComponentTotal] = {}
    for component in expected:
        bucket = tuple(event for event in events if event.component == component)
        totals[component] = _component_total(bucket, component in zero_components)
    return totals


def total_cost(
    totals: dict[CostComponent, ComponentTotal], expected: tuple[CostComponent, ...]
) -> int | None:
    """Return known total cost or None when any expected component is unpriced."""
    return _sum_optional(tuple(totals[component].cost_micro_usd for component in expected))


def retry_cost(events: tuple[UsageEvent, ...]) -> int | None:
    """Return retry-only subtotal; retry rows also stay in their component."""
    costs = tuple(event.cost_micro_usd for event in events if event.is_retry)
    return 0 if not costs else _sum_optional(costs)


def missing_coverage(
    totals: dict[CostComponent, ComponentTotal], expected: tuple[CostComponent, ...]
) -> tuple[CostComponent, ...]:
    """Return expected components whose cost is still unknown."""
    return tuple(component for component in expected if totals[component].cost_micro_usd is None)


def replay_budget(
    limit_micro_usd: int | None,
    reservations: tuple[ReservationRequest, ...],
    events: tuple[UsageEvent, ...],
) -> tuple[tuple[BudgetDecision, ...], BudgetState]:
    """Replay offline reservation decisions with in-flight reservations included."""
    costs = {event.event_id: event.cost_micro_usd for event in events}
    _reject_duplicate_settlements(reservations)
    settlement_ids = frozenset(
        request.settlement_event_id
        for request in reservations
        if request.settlement_event_id is not None
    )
    state = _initial_budget_state(limit_micro_usd, events, settlement_ids)
    decisions: list[BudgetDecision] = []
    for request in reservations:
        settlement_cost = None
        if request.settlement_event_id is not None:
            if request.settlement_event_id not in costs:
                raise CostInputError(
                    Path(request.reservation_id), "settlement_event_id has no event"
                )
            settlement_cost = costs[request.settlement_event_id]
        state, decision = state.apply(request, settlement_cost)
        decisions.append(decision)
    return tuple(decisions), state


def _reject_duplicate_settlements(reservations: tuple[ReservationRequest, ...]) -> None:
    seen: set[str] = set()
    for request in reservations:
        event_id = request.settlement_event_id
        if event_id is None:
            continue
        if event_id in seen:
            raise CostInputError(
                Path(request.reservation_id), f"duplicate settlement_event_id {event_id!r}"
            )
        seen.add(event_id)


def final_decision(
    state: BudgetState, known_total: int | None, missing: tuple[CostComponent, ...]
) -> BudgetDecision:
    """Summarize offline report status without claiming live execution approval."""
    if not missing or state.blocked_over_micro_usd is not None:
        return state.decide_total(known_total)
    return BudgetDecision(
        status=BudgetStatus.UNKNOWN,
        reason="expected component cost coverage is missing",
        reservation_id=None,
        limit_micro_usd=state.limit_micro_usd,
        estimate_micro_usd=None,
        spent_micro_usd=state.spent_micro_usd,
        reserved_micro_usd=state.reserved_micro_usd,
        over_micro_usd=None,
    )


def _initial_budget_state(
    limit_micro_usd: int | None,
    events: tuple[UsageEvent, ...],
    settlement_ids: frozenset[str],
) -> BudgetState:
    initial_costs = tuple(
        event.cost_micro_usd for event in events if event.event_id not in settlement_ids
    )
    spent = _sum_optional(initial_costs)
    return BudgetState(
        limit_micro_usd=limit_micro_usd,
        spent_micro_usd=0 if spent is None else spent,
        reserved_micro_usd=0,
        has_unknown=spent is None,
    )


def _component_total(events: tuple[UsageEvent, ...], zero_confirmed: bool) -> ComponentTotal:
    if not events and zero_confirmed:
        return ComponentTotal(events=0, calls=0, tokens_in=0, tokens_out=0, cost_micro_usd=0)
    if not events:
        return ComponentTotal(
            events=0, calls=0, tokens_in=None, tokens_out=None, cost_micro_usd=None
        )
    return ComponentTotal(
        events=len(events),
        calls=sum(event.calls for event in events),
        tokens_in=_sum_optional(tuple(event.tokens_in for event in events)),
        tokens_out=_sum_optional(tuple(event.tokens_out for event in events)),
        cost_micro_usd=_sum_optional(tuple(event.cost_micro_usd for event in events)),
    )


def _sum_optional(values: tuple[int | None, ...]) -> int | None:
    total = 0
    for value in values:
        if value is None:
            return None
        total += value
    return total
