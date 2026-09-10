from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import pytest

from agmem.bench.lme_v2_tools.costs import CostInputError, cost_report


class JsonDumpable(Protocol):
    pass


def _fixture(tmp_path: Path, payload: JsonDumpable) -> Path:
    path = tmp_path / "costs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cost_report_deduplicates_identical_source_events(tmp_path: Path) -> None:
    # Given: a provider trace replayed the same source event twice.
    event = {
        "event_id": "reader-q1-a1",
        "component": "reader",
        "tokens_in": 10,
        "tokens_out": 2,
        "cost_micro_usd": 7,
    }
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "budget_micro_usd": 7,
            "expected_components": ["reader"],
            "events": [event, event],
        },
    )

    # When: the offline ledger is built from the fixture.
    report = cost_report(path)

    # Then: the repeated source row is counted once and equality passes.
    assert report.components["reader"].events == 1
    assert report.components["reader"].cost_micro_usd == 7
    assert report.total_micro_usd == 7
    assert report.decision.status == "approved"


def test_cost_report_rejects_conflicting_source_event_ids(tmp_path: Path) -> None:
    # Given: two rows claim the same source event id but disagree on usage.
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "expected_components": ["embedding"],
            "events": [
                {"event_id": "embed-1", "component": "embedding", "tokens_in": 3},
                {"event_id": "embed-1", "component": "embedding", "tokens_in": 4},
            ],
        },
    )

    # When / Then: the ledger rejects the ambiguous accounting input.
    with pytest.raises(CostInputError, match="conflicting event_id"):
        cost_report(path)


def test_cost_report_counts_retry_cost_once_and_separately(tmp_path: Path) -> None:
    # Given: a reader call succeeds only after one paid retry.
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "budget_micro_usd": 18,
            "expected_components": ["reader"],
            "events": [
                {
                    "event_id": "reader-q1-a1",
                    "component": "reader",
                    "tokens_in": 10,
                    "tokens_out": 2,
                    "cost_micro_usd": 8,
                },
                {
                    "event_id": "reader-q1-a2",
                    "component": "reader",
                    "base_event_id": "reader-q1-a1",
                    "is_retry": True,
                    "tokens_in": 10,
                    "tokens_out": 3,
                    "cost_micro_usd": 10,
                },
            ],
        },
    )

    # When: the report aggregates component and retry views.
    report = cost_report(path)

    # Then: total includes both paid attempts, and retry subtotal includes only the retry row.
    assert report.components["reader"].cost_micro_usd == 18
    assert report.retry_cost_micro_usd == 10
    assert report.total_micro_usd == 18
    assert report.decision.status == "approved"


def test_cost_report_keeps_unknown_usage_and_price_unknown(tmp_path: Path) -> None:
    # Given: retrieval ran but its provider did not emit a priced usage row.
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "budget_micro_usd": 1,
            "expected_components": ["retrieval"],
            "events": [
                {
                    "event_id": "retrieval-q1",
                    "component": "retrieval",
                    "tokens_in": None,
                    "tokens_out": 0,
                    "cost_micro_usd": None,
                }
            ],
        },
    )

    # When: the ledger totals costs.
    report = cost_report(path)

    # Then: unknown fields are not coerced to zero and budget cannot be proven.
    assert report.components["retrieval"].tokens_in is None
    assert report.components["retrieval"].cost_micro_usd is None
    assert report.total_micro_usd is None
    assert report.missing_coverage == ("retrieval",)
    assert report.decision.status == "unknown"


def test_cost_report_blocks_budget_overrun(tmp_path: Path) -> None:
    # Given: known costs exceed the offline budget cap.
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "budget_micro_usd": 9,
            "expected_components": ["judge"],
            "events": [
                {
                    "event_id": "judge-q1",
                    "component": "judge",
                    "tokens_in": 4,
                    "tokens_out": 1,
                    "cost_micro_usd": 10,
                }
            ],
        },
    )

    # When: the report compares the known total to the cap.
    report = cost_report(path)

    # Then: the offline decision is blocked.
    assert report.decision.status == "blocked"
    assert report.decision.over_micro_usd == 1


def test_cost_report_rejects_malformed_events(tmp_path: Path) -> None:
    # Given: a fixture uses a boolean where a finite non-negative integer is required.
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "expected_components": ["write"],
            "events": [
                {
                    "event_id": "write-1",
                    "component": "write",
                    "cost_micro_usd": True,
                }
            ],
        },
    )

    # When / Then: malformed accounting input fails at the boundary.
    with pytest.raises(CostInputError, match="cost_micro_usd"):
        cost_report(path)


def test_cost_report_is_asdict_serializable(tmp_path: Path) -> None:
    # Given: a minimal valid fixture.
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "expected_components": ["write"],
            "events": [
                {
                    "event_id": "write-1",
                    "component": "write",
                    "cost_micro_usd": 3,
                }
            ],
        },
    )

    # When: a caller asks for the report shape intended for a future CLI.
    report_dict = asdict(cost_report(path))

    # Then: the output is JSON-compatible without custom encoders.
    assert report_dict["schema_version"] == 1
    assert report_dict["components"]["write"]["cost_micro_usd"] == 3


def test_cost_report_defaults_to_full_component_coverage(tmp_path: Path) -> None:
    # Given: only reader cost is present and the fixture does not narrow scope.
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "budget_micro_usd": 7,
            "events": [
                {
                    "event_id": "reader-q1",
                    "component": "reader",
                    "cost_micro_usd": 7,
                }
            ],
        },
    )

    # When: the report applies default whole-study coverage.
    report = cost_report(path)

    # Then: missing components prevent approval.
    assert report.missing_coverage == ("retrieval", "write", "embedding", "judge")
    assert report.decision.status == "unknown"


def test_cost_report_allows_explicit_zero_cost_components(tmp_path: Path) -> None:
    # Given: a scoped fixture marks retrieval as measured and free.
    path = _fixture(
        tmp_path,
        {
            "schema_version": 1,
            "budget_micro_usd": 0,
            "expected_components": ["retrieval"],
            "zero_cost_components": ["retrieval"],
            "events": [],
        },
    )

    # When: the report builds component totals.
    report = cost_report(path)

    # Then: explicit zero coverage is treated as known zero.
    assert report.components["retrieval"].cost_micro_usd == 0
    assert report.total_micro_usd == 0
    assert report.decision.status == "approved"
