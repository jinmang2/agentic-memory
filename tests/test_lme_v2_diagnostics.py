from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from agmem.bench.lme_v2_tools.diagnostics import diagnose_root
from agmem.bench.lme_v2_tools.types import AuditError


def _row(
    question_id: str,
    *,
    score: bool,
    category: str = "static",
    context: list[dict[str, str]] | None = None,
    metadata: dict[str, bool | int | str | None] | None = None,
) -> dict[str, bool | int | str | list[dict[str, str]] | dict[str, bool | int | str | None]]:
    row: dict[str, bool | int | str | list[dict[str, str]] | dict[str, bool | int | str | None]] = {
        "question_id": question_id,
        "question_type": f"{category}-environment",
        "category": category,
        "is_abstention_problem": False,
        "score_bool": score,
        "score": 1 if score else 0,
        "is_unknown": False,
    }
    if context is not None:
        row["memory_context"] = context
    if metadata is not None:
        row["memory_post_query_metadata"] = metadata
    return row


def _write_run(
    path: Path,
    rows: list[
        dict[str, bool | int | str | list[dict[str, str]] | dict[str, bool | int | str | None]]
    ],
    *,
    max_steps: int | None = 3,
) -> None:
    path.mkdir()
    correct = sum(1 for row in rows if row["score_bool"])
    by_category: dict[str, dict[str, float | int]] = {}
    for category in {str(row["category"]) for row in rows}:
        bucket = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "count": len(bucket),
            "pct_correct": sum(1 for row in bucket if row["score_bool"]) / len(bucket),
        }
    (path / "aggregated_metrics.json").write_text(
        json.dumps(
            {
                "overall": {
                    "overall_full_set": correct / len(rows),
                    "overall_non_abstention_only": correct / len(rows),
                    "overall_abstention_only": 0.0,
                    "count_all_questions": len(rows),
                    "count_non_abstention": len(rows),
                    "count_abstention": 0,
                },
                "non_abstention_by_category": by_category,
                "abstention_by_category": {},
            }
        ),
        encoding="utf-8",
    )
    (path / "per_question.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    if max_steps is not None:
        (path / "runtime_inputs").mkdir()
        (path / "runtime_inputs" / "memory_config.json").write_text(
            json.dumps({"memory_params": {"max_steps": max_steps}}),
            encoding="utf-8",
        )


def test_diagnose_root_reports_per_question_evidence_and_overlap_counts(tmp_path: Path) -> None:
    # Given: two validated runs with aligned IDs and diagnostic fields.
    _write_run(
        tmp_path / "arm_a",
        [
            _row(
                "q1",
                score=False,
                category="dynamic",
                context=[],
                metadata={"steps": 3, "degraded": "no LLM configured"},
            ),
            _row(
                "q2",
                score=True,
                category="static",
                context=[{"type": "text", "value": "kept private"}],
                metadata={"steps": 1, "degraded": None},
            ),
        ],
    )
    _write_run(
        tmp_path / "arm_b",
        [
            _row("q1", score=True, category="dynamic"),
            _row("q2", score=False, category="static"),
        ],
        max_steps=None,
    )
    # When: diagnosing offline artifacts.
    report = diagnose_root(tmp_path)
    payload = asdict(report)
    # Then: raw context text is not exported, and absent fields remain explicit.
    first = payload["questions"][0]
    assert first["arm"] == "arm_a"
    assert first["question_id"] == "q1"
    assert first["signals"] == {
        "step_cap_hit": True,
        "step_cap_reason": "cap_exhausted_degraded",
        "empty_retrieval_context": True,
        "degraded": True,
        "answer_support_present": None,
    }
    assert first["evidence_fields"] == [
        "per_question.memory_context",
        "per_question.memory_post_query_metadata.degraded",
        "per_question.memory_post_query_metadata.steps",
        "runtime_inputs.memory_config.memory_params.max_steps",
    ]
    assert "kept private" not in json.dumps(payload)
    arm_counts = payload["by_arm"]["arm_a"]
    assert arm_counts["all_three"]["incorrect"] == 1
    assert arm_counts["observed"]["all_three"]["incorrect"] == 1
    assert arm_counts["exclusive"]["all_three"]["incorrect"] == 1
    assert arm_counts["step_cap"]["incorrect"] == 1
    assert arm_counts["missing_evidence"]["correct"] == 0
    assert payload["by_category"]["dynamic"]["all_three"]["incorrect"] == 1
    assert payload["by_arm_category"]["arm_a"]["dynamic"]["all_three"]["incorrect"] == 1
    assert "by_category pools question-arm rows" in payload["note"]
    assert payload["questions"][2]["signals"]["step_cap_hit"] is None
    assert payload["questions"][2]["signals"]["empty_retrieval_context"] is None


def test_diagnose_root_rejects_malformed_diagnostic_fields(tmp_path: Path) -> None:
    # Given: a validated run whose diagnostic metadata has the wrong shape.
    _write_run(
        tmp_path / "arm",
        [_row("q1", score=False, metadata={"steps": "three", "degraded": None})],
    )
    # When / Then: diagnostic parsing fails instead of silently inventing evidence.
    with pytest.raises(AuditError, match="memory_post_query_metadata.steps"):
        diagnose_root(tmp_path)


def test_diagnose_root_reports_step_cap_unknown_without_config(tmp_path: Path) -> None:
    # Given: steps were captured but no run config proves the configured cap.
    _write_run(
        tmp_path / "arm",
        [
            _row(
                "q1",
                score=False,
                context=[],
                metadata={"steps": 8},
            )
        ],
        max_steps=None,
    )
    # When: diagnosing the row.
    report = diagnose_root(tmp_path)
    payload = asdict(report)
    # Then: the cap signal is unknown, while context emptiness stays observed.
    question = payload["questions"][0]
    assert question["signals"]["step_cap_hit"] is None
    assert question["signals"]["step_cap_reason"] == "missing_configured_max_steps"
    assert question["signals"]["empty_retrieval_context"] is True
    assert question["signals"]["degraded"] is None
    assert question["evidence_fields"] == [
        "per_question.memory_context",
        "per_question.memory_post_query_metadata.steps",
    ]
    assert payload["by_arm"]["arm"]["observed"]["step_cap"] == {"correct": 0, "incorrect": 0}
    assert payload["by_arm"]["arm"]["exclusive"]["unknown"] == {"correct": 0, "incorrect": 1}


def test_diagnose_root_classifies_step_cap_from_producer_step_semantics(
    tmp_path: Path,
) -> None:
    # Given: rows representing normal final, forced final, forced failure, and an ambiguous edge.
    _write_run(
        tmp_path / "arm",
        [
            _row("q1", score=True, metadata={"steps": 2, "degraded": None}),
            _row("q2", score=False, metadata={"steps": 3, "degraded": None}),
            _row("q3", score=False, metadata={"steps": 2, "degraded": "max_steps"}),
            _row("q4", score=False, metadata={"steps": 2}),
        ],
        max_steps=2,
    )
    # When: diagnosing cap status.
    payload = asdict(diagnose_root(tmp_path))
    by_id = {question["question_id"]: question["signals"] for question in payload["questions"]}
    # Then: equality alone is not treated as cap exhaustion.
    assert by_id["q1"]["step_cap_hit"] is False
    assert by_id["q1"]["step_cap_reason"] == "final_within_last_allowed_step"
    assert by_id["q2"]["step_cap_hit"] is True
    assert by_id["q2"]["step_cap_reason"] == "forced_final_after_cap"
    assert by_id["q3"]["step_cap_hit"] is True
    assert by_id["q3"]["step_cap_reason"] == "cap_exhausted_degraded"
    assert by_id["q4"]["step_cap_hit"] is None
    assert by_id["q4"]["step_cap_reason"] == "missing_degraded_at_cap_boundary"
    assert payload["by_arm"]["arm"]["step_cap"] == {"correct": 0, "incorrect": 2}
    assert payload["by_arm"]["arm"]["observed"]["step_cap"] == {"correct": 1, "incorrect": 2}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("steps", -1, "memory_post_query_metadata.steps"),
        ("memory_context_token_count", -1, "memory_context_token_count"),
    ],
)
def test_diagnose_root_rejects_negative_counts(
    tmp_path: Path, field: str, value: int, message: str
) -> None:
    # Given: a diagnostic count has an impossible negative value.
    row = _row("q1", score=False, metadata={"degraded": None})
    if field == "steps":
        row["memory_post_query_metadata"] = {"steps": value, "degraded": None}
    else:
        row[field] = value
    _write_run(tmp_path / "arm", [row])
    # When / Then: the boundary parser rejects it.
    with pytest.raises(AuditError, match=message):
        diagnose_root(tmp_path)


def test_diagnose_root_rejects_non_positive_configured_max_steps(tmp_path: Path) -> None:
    # Given: the runtime config records an impossible exploration cap.
    _write_run(tmp_path / "arm", [_row("q1", score=False, metadata={"steps": 0})], max_steps=1)
    config_path = tmp_path / "arm" / "runtime_inputs" / "memory_config.json"
    config_path.write_text(
        json.dumps({"memory_params": {"max_steps": 0}}),
        encoding="utf-8",
    )
    # When / Then: the configured cap is rejected.
    with pytest.raises(AuditError, match="max_steps"):
        diagnose_root(tmp_path)


def test_diagnose_root_uses_strict_audit_join_before_diagnosis(tmp_path: Path) -> None:
    # Given: two runs with non-pairable question sets.
    _write_run(tmp_path / "arm_a", [_row("q1", score=True)])
    _write_run(tmp_path / "arm_b", [_row("q2", score=True)])
    # When / Then: the existing audit guard rejects the comparison.
    with pytest.raises(AuditError, match="question_id set mismatch"):
        diagnose_root(tmp_path)


def test_diagnose_root_preserves_duplicate_json_key_guard(tmp_path: Path) -> None:
    # Given: a result row with duplicate keys in the diagnostic payload.
    _write_run(tmp_path / "arm", [_row("q1", score=True)])
    row = (
        '{"question_id":"q1","question_type":"static-environment","category":"static",'
        '"is_abstention_problem":false,"score_bool":true,"score":1,"is_unknown":false,'
        '"memory_post_query_metadata":{"steps":1,"steps":2}}\n'
    )
    (tmp_path / "arm" / "per_question.jsonl").write_text(row, encoding="utf-8")
    # When / Then: shared JSON parsing catches it.
    with pytest.raises(AuditError, match="duplicate JSON key"):
        diagnose_root(tmp_path)
