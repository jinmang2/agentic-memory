from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from agmem.bench.lme_v2_tools.audit import AuditError, audit_root


def _row(
    question_id: str,
    *,
    score: bool,
    category: str = "static",
    qtype: str = "static-environment",
    abstention: bool = False,
    unknown: bool = False,
) -> dict[str, bool | int | str]:
    return {
        "index": 0,
        "question_id": question_id,
        "question_type": qtype,
        "category": category,
        "is_abstention_problem": abstention,
        "score_bool": score,
        "score": 1 if score else 0,
        "is_unknown": unknown,
    }


def _write_run(path: Path, rows: list[dict[str, bool | int | str]]) -> None:
    correct = sum(1 for row in rows if row["score_bool"])
    non_abs = [row for row in rows if not row["is_abstention_problem"]]
    abs_rows = [row for row in rows if row["is_abstention_problem"]]
    non_abs_correct = sum(1 for row in non_abs if row["score_bool"] and not row["is_unknown"])
    abs_correct = sum(1 for row in abs_rows if row["score_bool"] and not row["is_unknown"])
    by_category = {}
    for category in {str(row["category"]) for row in non_abs}:
        cat_rows = [row for row in non_abs if row["category"] == category]
        by_category[category] = {
            "count": len(cat_rows),
            "pct_correct": sum(1 for row in cat_rows if row["score_bool"] and not row["is_unknown"])
            / len(cat_rows),
        }
    abs_by_category = {}
    for category in {str(row["category"]) for row in abs_rows}:
        cat_rows = [row for row in abs_rows if row["category"] == category]
        abs_by_category[category] = {
            "count": len(cat_rows),
            "pct_correct": sum(1 for row in cat_rows if row["score_bool"] and not row["is_unknown"])
            / len(cat_rows),
        }
    path.mkdir()
    (path / "aggregated_metrics.json").write_text(
        json.dumps(
            {
                "overall": {
                    "overall_full_set": correct / len(rows),
                    "overall_non_abstention_only": non_abs_correct / len(non_abs)
                    if non_abs
                    else 0.0,
                    "overall_abstention_only": abs_correct / len(abs_rows) if abs_rows else 0.0,
                    "count_all_questions": len(rows),
                    "count_non_abstention": len(non_abs),
                    "count_abstention": len(abs_rows),
                },
                "non_abstention_by_category": by_category,
                "abstention_by_category": abs_by_category,
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0},
                "memory_context": {"avg_final_tokens": 0},
                "memory_query": {
                    "avg_seconds": 0.0,
                    "p50_seconds": 0.0,
                    "p95_seconds": 0.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "per_question.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_audit_root_reports_categories_and_strict_paired_tests(tmp_path: Path) -> None:
    # Given two complete runs over the same question set.
    _write_run(
        tmp_path / "arm_a",
        [
            _row("q1", score=True),
            _row("q2", score=False, category="dynamic", qtype="dynamic-environment"),
            _row("q3", score=True, category="static-abs", qtype="static-abs", abstention=True),
        ],
    )
    _write_run(
        tmp_path / "arm_b",
        [
            _row("q1", score=False),
            _row("q2", score=True, category="dynamic", qtype="dynamic-environment"),
            _row("q3", score=True, category="static-abs", qtype="static-abs", abstention=True),
        ],
    )
    (tmp_path / "unfinished").mkdir()
    # When auditing the root.
    report = audit_root(tmp_path)
    # Then category summaries, skipped directories, and exact sign tests are serializable.
    assert asdict(report)["skipped_dirs"] == ["unfinished"]
    arm_a = report.arms[0]
    assert arm_a.name == "arm_a"
    assert arm_a.n == 3 and arm_a.correct == 2 and arm_a.abstention_n == 1
    assert arm_a.categories["static"].correct == 1
    assert arm_a.categories["dynamic"].non_abstention_n == 1
    assert arm_a.categories["static-abs"].abstention_n == 1
    test = report.sign_tests[0]
    assert test.a == "arm_a" and test.b == "arm_b"
    assert test.only_a == 1 and test.only_b == 1 and test.p_two_sided == 1.0
    assert test.by_category["dynamic"].only_b == 1


def test_duplicate_and_blank_question_ids_are_rejected(tmp_path: Path) -> None:
    # Given a run with an invalid join key.
    _write_run(tmp_path / "arm", [_row("q1", score=True), _row("q1", score=False)])
    # When auditing it.
    with pytest.raises(AuditError, match="duplicate question_id"):
        audit_root(tmp_path)


def test_bool_is_not_accepted_as_an_integer_or_score_count(tmp_path: Path) -> None:
    # Given aggregate counts that would pass Python's bool-is-int coercion.
    _write_run(tmp_path / "arm", [_row("q1", score=True)])
    agg = json.loads((tmp_path / "arm" / "aggregated_metrics.json").read_text())
    agg["overall"]["count_all_questions"] = True
    (tmp_path / "arm" / "aggregated_metrics.json").write_text(json.dumps(agg), encoding="utf-8")
    # When auditing it.
    with pytest.raises(AuditError, match="count_all_questions"):
        audit_root(tmp_path)


def test_nan_aggregate_ratios_are_rejected(tmp_path: Path) -> None:
    # Given an aggregate ratio that JSON accepts but statistics cannot interpret.
    _write_run(tmp_path / "arm", [_row("q1", score=True)])
    aggregate = json.loads((tmp_path / "arm" / "aggregated_metrics.json").read_text())
    aggregate["overall"]["overall_full_set"] = float("nan")
    (tmp_path / "arm" / "aggregated_metrics.json").write_text(
        json.dumps(aggregate), encoding="utf-8"
    )
    # When auditing it.
    with pytest.raises(AuditError, match="overall_full_set"):
        audit_root(tmp_path)


def test_score_disagreement_is_rejected(tmp_path: Path) -> None:
    # Given a row with contradictory score fields.
    _write_run(tmp_path / "arm", [_row("q1", score=True)])
    rows = (tmp_path / "arm" / "per_question.jsonl").read_text().splitlines()
    row = json.loads(rows[0])
    row["score"] = 0
    (tmp_path / "arm" / "per_question.jsonl").write_text(json.dumps(row), encoding="utf-8")
    # When auditing it.
    with pytest.raises(AuditError, match="score must equal"):
        audit_root(tmp_path)


def test_nonbinary_score_is_rejected_even_when_score_bool_is_true(tmp_path: Path) -> None:
    # Given a score value outside upstream's 0.0/1.0 contract.
    _write_run(tmp_path / "arm", [_row("q1", score=True)])
    rows = (tmp_path / "arm" / "per_question.jsonl").read_text().splitlines()
    row = json.loads(rows[0])
    row["score"] = 2.0
    (tmp_path / "arm" / "per_question.jsonl").write_text(json.dumps(row), encoding="utf-8")
    # When auditing it.
    with pytest.raises(AuditError, match="score must equal"):
        audit_root(tmp_path)


def test_aggregate_overall_mismatch_is_rejected(tmp_path: Path) -> None:
    # Given an aggregate file whose reported accuracy does not match per-question rows.
    _write_run(tmp_path / "arm", [_row("q1", score=True), _row("q2", score=False)])
    agg = json.loads((tmp_path / "arm" / "aggregated_metrics.json").read_text())
    agg["overall"]["overall_full_set"] = 1.0
    (tmp_path / "arm" / "aggregated_metrics.json").write_text(json.dumps(agg), encoding="utf-8")
    # When auditing it.
    with pytest.raises(AuditError, match="overall_full_set"):
        audit_root(tmp_path)


def test_category_pct_correct_follows_upstream_unknown_semantics(tmp_path: Path) -> None:
    # Given upstream's category breakdown excludes unknown answers from correct_count.
    _write_run(tmp_path / "arm", [_row("q1", score=True, unknown=True)])
    # When auditing it.
    report = audit_root(tmp_path)
    # Then overall still follows score_bool while aggregate category pct remains valid.
    assert report.arms[0].correct == 1
    assert report.arms[0].categories["static"].unknown == 1


def test_required_category_pct_correct_cannot_be_missing(tmp_path: Path) -> None:
    # Given a non-empty category aggregate without the required pct_correct field.
    _write_run(tmp_path / "arm", [_row("q1", score=True)])
    aggregate = json.loads((tmp_path / "arm" / "aggregated_metrics.json").read_text())
    del aggregate["non_abstention_by_category"]["static"]["pct_correct"]
    (tmp_path / "arm" / "aggregated_metrics.json").write_text(
        json.dumps(aggregate), encoding="utf-8"
    )
    # When auditing it.
    with pytest.raises(AuditError, match="pct_correct"):
        audit_root(tmp_path)


def test_empty_canonical_categories_in_aggregate_are_allowed(tmp_path: Path) -> None:
    # Given a subset run whose aggregate includes upstream's zero-count category buckets.
    _write_run(tmp_path / "arm", [_row("q1", score=True)])
    agg = json.loads((tmp_path / "arm" / "aggregated_metrics.json").read_text())
    agg["non_abstention_by_category"]["dynamic"] = {"count": 0, "pct_correct": None}
    agg["abstention_by_category"]["static-abs"] = {"count": 0, "pct_correct": None}
    (tmp_path / "arm" / "aggregated_metrics.json").write_text(json.dumps(agg), encoding="utf-8")
    # When auditing it.
    report = audit_root(tmp_path)
    # Then the actual row category still drives the denominator summary.
    assert report.arms[0].categories["static"].n == 1


def test_paired_comparisons_require_equal_ids_and_stable_metadata(tmp_path: Path) -> None:
    # Given two runs whose question IDs match but category metadata drifted.
    _write_run(tmp_path / "arm_a", [_row("q1", score=True, category="static")])
    _write_run(tmp_path / "arm_b", [_row("q1", score=False, category="dynamic")])
    # When auditing paired comparisons.
    with pytest.raises(AuditError, match="metadata mismatch"):
        audit_root(tmp_path)


def test_partial_child_run_is_not_silently_ignored(tmp_path: Path) -> None:
    # Given a child directory containing only one result artifact.
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "per_question.jsonl").write_text("", encoding="utf-8")
    # When auditing the root.
    with pytest.raises(AuditError, match="incomplete run"):
        audit_root(tmp_path)
