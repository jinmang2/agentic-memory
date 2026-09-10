"""Strict parsers for saved LongMemEval-V2 result artifacts.

The audit CLI uses this module as the trust boundary for offline benchmark
outputs: malformed rows, lossy type coercions, incomplete files, and aggregate
drift fail before paired comparisons are computed.
"""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path

from .json_io import JsonInputError, parse_json
from .json_io import JsonValue as Json
from .types import AuditError, CategorySummary, LoadedRun, QuestionMeta, QuestionResult


def has_result_pair(run_dir: Path) -> bool:
    """Report whether a directory has both files required for a finished run."""

    return (run_dir / "aggregated_metrics.json").exists() and (
        run_dir / "per_question.jsonl"
    ).exists()


def has_partial_result(run_dir: Path) -> bool:
    """Report whether a directory looks like an unfinished benchmark output."""

    files = [
        (run_dir / "aggregated_metrics.json").exists(),
        (run_dir / "per_question.jsonl").exists(),
    ]
    return any(files) and not all(files)


def load_run(run_dir: Path) -> LoadedRun:
    """Parse one explicit run directory and reject unfinished or inconsistent artifacts."""

    from .aggregate import check_aggregate

    if has_partial_result(run_dir):
        raise AuditError(
            run_dir, "incomplete run: expected aggregated_metrics.json and per_question.jsonl"
        )
    if not has_result_pair(run_dir):
        raise AuditError(run_dir, "missing result artifacts")
    aggregate = _read_json_mapping(run_dir / "aggregated_metrics.json")
    rows = _read_rows(run_dir / "per_question.jsonl")
    check_aggregate(run_dir / "aggregated_metrics.json", aggregate, rows)
    return _build_run(run_dir, rows)


def _read_json_mapping(path: Path) -> dict[str, Json]:
    try:
        payload: Json = parse_json(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditError(path, f"malformed JSON at line {exc.lineno} column {exc.colno}") from exc
    except JsonInputError as exc:
        raise AuditError(path, str(exc)) from exc
    if not isinstance(payload, dict):
        raise AuditError(path, "expected a JSON object")
    return payload


def _read_rows(path: Path) -> list[QuestionResult]:
    rows: list[QuestionResult] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                raise AuditError(path, f"blank JSONL line {line_no}")
            try:
                payload: Json = parse_json(line)
            except json.JSONDecodeError as exc:
                raise AuditError(
                    path, f"malformed JSONL line {line_no}: column {exc.colno}"
                ) from exc
            except JsonInputError as exc:
                raise AuditError(path, f"line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise AuditError(path, f"line {line_no}: expected object")
            row = _parse_row(path, line_no, payload)
            question_id = row.meta.question_id
            if question_id in seen:
                raise AuditError(path, f"duplicate question_id {question_id!r}")
            seen.add(question_id)
            rows.append(row)
    if not rows:
        raise AuditError(path, "empty per_question.jsonl")
    return rows


def _parse_row(path: Path, line_no: int, row_payload: dict[str, Json]) -> QuestionResult:
    question_id = _string_field(path, line_no, row_payload, "question_id")
    question_type = _string_field(path, line_no, row_payload, "question_type")
    category = _string_field(path, line_no, row_payload, "category")
    correct = _bool_field(path, line_no, row_payload, "score_bool")
    _check_score_field(path, line_no, row_payload, correct)
    return QuestionResult(
        meta=QuestionMeta(
            question_id=question_id,
            question_type=question_type,
            category=category,
            is_abstention=_bool_field(path, line_no, row_payload, "is_abstention_problem"),
        ),
        correct=correct,
        unknown=_bool_field(path, line_no, row_payload, "is_unknown"),
    )


def _string_field(path: Path, line_no: int, row_payload: dict[str, Json], key: str) -> str:
    value = row_payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AuditError(path, f"line {line_no}: {key} must be a nonblank string")
    return value


def _bool_field(path: Path, line_no: int, row_payload: dict[str, Json], key: str) -> bool:
    value = row_payload.get(key)
    if type(value) is not bool:
        raise AuditError(path, f"line {line_no}: {key} must be a boolean")
    return value


def _check_score_field(
    path: Path, line_no: int, row_payload: dict[str, Json], correct: bool
) -> None:
    if "score" not in row_payload:
        return
    score = row_payload["score"]
    if not isinstance(score, int | float) or type(score) is bool or not isfinite(score):
        raise AuditError(path, f"line {line_no}: score must be numeric")
    expected = 1.0 if correct else 0.0
    if float(score) != expected:
        raise AuditError(path, f"line {line_no}: score must equal {expected}")


def _build_run(run_dir: Path, rows: list[QuestionResult]) -> LoadedRun:
    scores = {row.meta.question_id: row.correct for row in rows}
    metadata = {row.meta.question_id: row.meta for row in rows}
    categories = _summarize_categories(rows)
    return LoadedRun(
        name=run_dir.name,
        path=run_dir,
        n=len(rows),
        correct=sum(1 for row in rows if row.correct),
        unknown=sum(1 for row in rows if row.unknown),
        abstention_n=sum(1 for row in rows if row.meta.is_abstention),
        non_abstention_n=sum(1 for row in rows if not row.meta.is_abstention),
        categories=categories,
        scores=scores,
        metadata=metadata,
    )


def _summarize_categories(rows: list[QuestionResult]) -> dict[str, CategorySummary]:
    categories: dict[str, CategorySummary] = {}
    for category in sorted({row.meta.category for row in rows}):
        bucket = [row for row in rows if row.meta.category == category]
        categories[category] = CategorySummary(
            n=len(bucket),
            correct=sum(1 for row in bucket if row.correct),
            unknown=sum(1 for row in bucket if row.unknown),
            abstention_n=sum(1 for row in bucket if row.meta.is_abstention),
            non_abstention_n=sum(1 for row in bucket if not row.meta.is_abstention),
        )
    return categories
