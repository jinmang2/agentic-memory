from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import pytest

from agmem.bench.lme_v2_tools.costs import CostInputError, cost_report


class JsonDumpable(Protocol):
    pass


def _fixture(tmp_path: Path, payload: JsonDumpable) -> Path:
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_reservation_refuses_call_when_in_flight_budget_is_insufficient(tmp_path: Path) -> None:
    # Given: one reservation is already in flight under a tight offline cap.
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "budget_micro_usd": 100,
            "expected_components": [],
            "events": [],
            "reservations": [
                {"reservation_id": "reader-q1", "estimate_micro_usd": 60},
                {"reservation_id": "reader-q2", "estimate_micro_usd": 50},
            ],
        },
    )

    # When: the offline budget log is replayed.
    report = cost_report(path)

    # Then: the second request is refused before it can spend.
    assert report.reservations[0].status == "approved"
    assert report.reservations[1].status == "blocked"
    assert report.reservations[1].over_micro_usd == 10


def test_reservation_settlement_releases_estimate_and_counts_actual_event_once(
    tmp_path: Path,
) -> None:
    # Given: a reserved call settles against a lower actual provider event.
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "budget_micro_usd": 100,
            "expected_components": ["reader"],
            "events": [
                {
                    "event_id": "reader-q1-a1",
                    "component": "reader",
                    "cost_micro_usd": 45,
                }
            ],
            "reservations": [
                {
                    "reservation_id": "reader-q1",
                    "estimate_micro_usd": 60,
                    "settlement_event_id": "reader-q1-a1",
                },
                {"reservation_id": "reader-q2", "estimate_micro_usd": 55},
            ],
        },
    )

    # When: settlement and the next preflight are replayed in order.
    report = cost_report(path)

    # Then: the second reservation can use the released 15 microUSD.
    assert [item.status for item in report.reservations] == ["settled", "approved"]
    assert report.reservations[0].spent_micro_usd == 45
    assert report.total_micro_usd == 45
    assert report.decision.status == "approved"


def test_unknown_reservation_blocks_subsequent_budget_decisions(tmp_path: Path) -> None:
    # Given: the next call has no reliable estimate.
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "budget_micro_usd": 100,
            "expected_components": [],
            "events": [],
            "reservations": [
                {"reservation_id": "reader-q1", "estimate_micro_usd": None},
                {"reservation_id": "reader-q2", "estimate_micro_usd": 1},
            ],
        },
    )

    # When: the offline budget log is replayed.
    report = cost_report(path)

    # Then: both the unknown call and later calls are not approved.
    assert report.reservations[0].status == "unknown"
    assert report.reservations[1].status == "unknown"
    assert report.decision.status == "unknown"


def test_settlement_breach_counts_pending_reservations_in_final_decision(
    tmp_path: Path,
) -> None:
    # Given: one pending reservation remains when another call settles above estimate.
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "budget_micro_usd": 100,
            "expected_components": ["reader"],
            "events": [
                {
                    "event_id": "reader-q2-a1",
                    "component": "reader",
                    "cost_micro_usd": 80,
                }
            ],
            "reservations": [
                {"reservation_id": "reader-q1", "estimate_micro_usd": 60},
                {
                    "reservation_id": "reader-q2",
                    "estimate_micro_usd": 40,
                    "settlement_event_id": "reader-q2-a1",
                },
            ],
        },
    )

    # When: the budget replay reaches final report state.
    report = cost_report(path)

    # Then: billed total alone is not approved while pending exposure remains.
    assert report.total_micro_usd == 80
    assert report.reservations[1].status == "breached"
    assert report.reservations[1].over_micro_usd == 40
    assert report.decision.status == "blocked"
    assert report.decision.over_micro_usd == 40


def test_reservation_ids_are_deduped_or_rejected_on_conflict(tmp_path: Path) -> None:
    # Given: replay emits one identical reservation twice and one conflicting duplicate.
    duplicate = {"reservation_id": "reader-q1", "estimate_micro_usd": 1}
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "expected_components": [],
            "events": [],
            "reservations": [
                duplicate,
                duplicate,
                {"reservation_id": "reader-q1", "estimate_micro_usd": 2},
            ],
        },
    )

    # When / Then: the conflicting replay id is rejected.
    with pytest.raises(CostInputError, match="conflicting reservation_id"):
        cost_report(path)


def test_reservations_reject_duplicate_settlement_event_ownership(tmp_path: Path) -> None:
    # Given: two reservation ids try to settle against the same provider event.
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "expected_components": ["reader"],
            "events": [
                {
                    "event_id": "reader-q1-a1",
                    "component": "reader",
                    "cost_micro_usd": 5,
                }
            ],
            "reservations": [
                {
                    "reservation_id": "reader-q1",
                    "estimate_micro_usd": 5,
                    "settlement_event_id": "reader-q1-a1",
                },
                {
                    "reservation_id": "reader-q1-replay",
                    "estimate_micro_usd": 5,
                    "settlement_event_id": "reader-q1-a1",
                },
            ],
        },
    )

    # When / Then: budget replay refuses the doublecounting ownership claim.
    with pytest.raises(CostInputError, match="duplicate settlement_event_id"):
        cost_report(path)


def test_final_decision_keeps_prior_refused_reservation(tmp_path: Path) -> None:
    # Given: a settled event fits known total but a later preflight was refused.
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "budget_micro_usd": 100,
            "expected_components": ["reader"],
            "events": [
                {
                    "event_id": "reader-q1",
                    "component": "reader",
                    "cost_micro_usd": 80,
                }
            ],
            "reservations": [
                {
                    "reservation_id": "r1",
                    "estimate_micro_usd": 40,
                    "settlement_event_id": "reader-q1",
                },
                {"reservation_id": "r2", "estimate_micro_usd": 60},
            ],
        },
    )

    # When: final report status is computed after replaying the refusal.
    report = cost_report(path)

    # Then: final status keeps the earlier preflight block.
    assert [item.status for item in report.reservations] == ["settled", "blocked"]
    assert report.total_micro_usd == 80
    assert report.decision.status == "blocked"
    assert report.decision.over_micro_usd == 40
