"""Public cost report API for the offline LME-V2 preparation tools."""

from __future__ import annotations

from pathlib import Path

from .costs_ledger import (
    component_totals,
    final_decision,
    missing_coverage,
    replay_budget,
    retry_cost,
    total_cost,
)
from .costs_parse import read_cost_fixture
from .costs_types import COST_SCHEMA_VERSION, INTEGRATION_POINTS, CostInputError, CostReport


def cost_report(path: Path) -> CostReport:
    """Build a deterministic offline report from a strict JSON fixture."""
    expected, zero_components, events, reservations, budget_limit = read_cost_fixture(path)
    totals = component_totals(events, expected, zero_components)
    known_total = total_cost(totals, expected)
    missing = missing_coverage(totals, expected)
    decisions, budget_state = replay_budget(budget_limit, reservations, events)
    return CostReport(
        schema_version=COST_SCHEMA_VERSION,
        expected_components=tuple(component.value for component in expected),
        zero_cost_components=tuple(component.value for component in zero_components),
        components={component.value: totals[component] for component in expected},
        total_micro_usd=known_total,
        retry_cost_micro_usd=retry_cost(events),
        missing_coverage=tuple(component.value for component in missing),
        reservations=decisions,
        decision=final_decision(budget_state, known_total, missing),
        live_cap_guarantee=False,
        integration_points=INTEGRATION_POINTS,
    )


__all__ = ["CostInputError", "CostReport", "cost_report"]
