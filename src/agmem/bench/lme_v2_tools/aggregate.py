"""Aggregate-metric consistency checks for LongMemEval-V2 audit inputs.

The saved aggregate file is treated as a redundant checksum over per-question
rows; this module verifies counts and ratios while allowing upstream's empty
category buckets on subset runs.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path

from .json_io import JsonValue as Json
from .types import AuditError, QuestionResult

EPSILON = 1e-12


def check_aggregate(path: Path, aggregate: dict[str, Json], rows: list[QuestionResult]) -> None:
    """Reject aggregate files that disagree with parsed per-question rows."""

    overall = _mapping(path, aggregate, "overall")
    correct = sum(1 for row in rows if row.correct)
    abstention = sum(1 for row in rows if row.meta.is_abstention)
    non_abstention = len(rows) - abstention
    _expect_int(path, overall, "count_all_questions", len(rows))
    _expect_int(path, overall, "count_non_abstention", non_abstention)
    _expect_int(path, overall, "count_abstention", abstention)
    _expect_ratio(path, overall, "overall_full_set", correct, len(rows))
    _expect_ratio(
        path,
        overall,
        "overall_non_abstention_only",
        sum(1 for row in rows if row.correct and not row.unknown and not row.meta.is_abstention),
        non_abstention,
    )
    _expect_ratio(
        path,
        overall,
        "overall_abstention_only",
        sum(1 for row in rows if row.correct and not row.unknown and row.meta.is_abstention),
        abstention,
    )
    _check_category_block(path, aggregate, "non_abstention_by_category", rows, abstention=False)
    _check_category_block(path, aggregate, "abstention_by_category", rows, abstention=True)


def _check_category_block(
    path: Path,
    aggregate: dict[str, Json],
    key: str,
    rows: list[QuestionResult],
    *,
    abstention: bool,
) -> None:
    block = _mapping(path, aggregate, key)
    expected = sorted({row.meta.category for row in rows if row.meta.is_abstention is abstention})
    missing = sorted(set(expected) - set(block))
    if missing:
        raise AuditError(path, f"{key} missing categories: {missing}")
    _check_extra_empty_categories(path, key, block, expected)
    for category in expected:
        metrics = _mapping(path, block, category)
        bucket = [
            row
            for row in rows
            if row.meta.category == category and row.meta.is_abstention is abstention
        ]
        _expect_int(path, metrics, f"{key}.{category}.count", len(bucket), actual_key="count")
        _expect_ratio(
            path,
            metrics,
            f"{key}.{category}.pct_correct",
            sum(1 for row in bucket if row.correct and not row.unknown),
            len(bucket),
            actual_key="pct_correct",
        )


def _check_extra_empty_categories(
    path: Path, key: str, block: dict[str, Json], expected: list[str]
) -> None:
    for category in sorted(set(block) - set(expected)):
        metrics = _mapping(path, block, category)
        _expect_int(path, metrics, f"{key}.{category}.count", 0, actual_key="count")
        pct_correct = metrics.get("pct_correct")
        if type(pct_correct) is bool:
            raise AuditError(path, f"{key}.{category}.pct_correct must be null for empty category")
        if pct_correct is not None and pct_correct != 0:
            raise AuditError(path, f"{key}.{category}.pct_correct must be null for empty category")


def _mapping(path: Path, row_payload: dict[str, Json], key: str) -> dict[str, Json]:
    value = row_payload.get(key)
    if not isinstance(value, dict):
        raise AuditError(path, f"{key} must be an object")
    return value


def _expect_int(
    path: Path,
    row_payload: dict[str, Json],
    label: str,
    expected: int,
    *,
    actual_key: str | None = None,
) -> None:
    key = label if actual_key is None else actual_key
    value = row_payload.get(key)
    if type(value) is not int:
        raise AuditError(path, f"{label} must be an integer")
    if value != expected:
        raise AuditError(path, f"{label} mismatch: expected {expected}, got {value}")


def _expect_ratio(
    path: Path,
    row_payload: dict[str, Json],
    label: str,
    correct: int,
    total: int,
    *,
    actual_key: str | None = None,
) -> None:
    key = label if actual_key is None else actual_key
    value = row_payload.get(key)
    if total == 0 and value is None:
        return
    if not isinstance(value, int | float) or type(value) is bool or not isfinite(value):
        raise AuditError(path, f"{label} must be numeric")
    expected = correct / total if total else 0.0
    if abs(float(value) - expected) > EPSILON:
        raise AuditError(path, f"{label} mismatch: expected {expected}, got {value}")
