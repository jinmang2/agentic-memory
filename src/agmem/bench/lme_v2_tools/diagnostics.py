"""Offline failure diagnostics for validated LongMemEval-V2 runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .audit import audit_root
from .diagnostics_counts import DiagnosticSignals, OverlapCounts, overlap_counts
from .json_io import JsonInputError, JsonValue, parse_json
from .results import has_partial_result, has_result_pair, load_run
from .types import AuditError, LoadedRun


@dataclass(frozen=True, slots=True)
class QuestionDiagnosis:
    """Per-question diagnostic row with provenance field names only."""

    arm: str
    question_id: str
    category: str
    correct: bool
    signals: DiagnosticSignals
    evidence_fields: list[str]


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """Top-level report returned by diagnose_root and serializable via asdict."""

    root: str
    questions: list[QuestionDiagnosis]
    by_arm: dict[str, OverlapCounts]
    by_category: dict[str, OverlapCounts]
    by_arm_category: dict[str, dict[str, OverlapCounts]]
    note: str


@dataclass(frozen=True, slots=True)
class DiagnosticRun:
    run: LoadedRun
    max_steps: int | None
    fields: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class StepCapSignal:
    hit: bool | None
    reason: str


def diagnose_root(root: Path) -> DiagnosticReport:
    """Diagnose saved run artifacts without model calls or prompt/context export."""

    audit_root(root)
    runs = _load_runs(root)
    questions = [
        _diagnose_question(diagnostic_run, question_id)
        for diagnostic_run in runs
        for question_id in sorted(diagnostic_run.run.scores)
    ]
    return DiagnosticReport(
        root=str(root),
        questions=questions,
        by_arm=_counts_by_key(questions, key="arm"),
        by_category=_counts_by_key(questions, key="category"),
        by_arm_category=_counts_by_arm_category(questions),
        note=(
            "Counts are overlaps in saved artifacts only; they are not causal "
            "failure labels. by_category pools question-arm rows across arms. "
            "answer_support_present is unknown without human labels."
        ),
    )


def _load_runs(root: Path) -> list[DiagnosticRun]:
    if has_result_pair(root) or has_partial_result(root):
        return [_diagnostic_run(root)]
    runs = [
        _diagnostic_run(child)
        for child in sorted(candidate for candidate in root.iterdir() if candidate.is_dir())
        if has_result_pair(child)
    ]
    if not runs:
        raise AuditError(root, "no complete runs found")
    return runs


def _diagnostic_run(path: Path) -> DiagnosticRun:
    return DiagnosticRun(
        run=load_run(path),
        max_steps=_max_steps(path),
        fields=_read_per_question_fields(path / "per_question.jsonl"),
    )


def _max_steps(path: Path) -> int | None:
    config_path = path / "runtime_inputs" / "memory_config.json"
    if not config_path.exists():
        return None
    payload = _read_json_object(config_path)
    memory_params = payload.get("memory_params")
    if not isinstance(memory_params, dict):
        return None
    value = memory_params.get("max_steps")
    if value is None:
        return None
    if type(value) is not int:
        raise AuditError(config_path, "memory_params.max_steps must be an integer")
    if value <= 0:
        raise AuditError(config_path, "memory_params.max_steps must be positive")
    return value


def _read_per_question_fields(path: Path) -> dict[str, JsonValue]:
    rows: dict[str, JsonValue] = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            try:
                payload = parse_json(line)
            except json.JSONDecodeError as exc:
                raise AuditError(
                    path, f"malformed JSONL line {line_no}: column {exc.colno}"
                ) from exc
            except JsonInputError as exc:
                raise AuditError(path, f"line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise AuditError(path, f"line {line_no}: expected object")
            question_id = payload.get("question_id")
            if not isinstance(question_id, str) or not question_id.strip():
                raise AuditError(path, f"line {line_no}: question_id must be a nonblank string")
            rows[question_id] = payload
    return rows


def _diagnose_question(diagnostic_run: DiagnosticRun, question_id: str) -> QuestionDiagnosis:
    row = diagnostic_run.fields.get(question_id)
    if not isinstance(row, dict):
        raise AuditError(diagnostic_run.run.path, f"missing diagnostic row for {question_id!r}")
    fields: list[str] = []
    degraded = _degraded(diagnostic_run.run.path, row, fields)
    step_cap = _step_cap_signal(diagnostic_run, row, fields, degraded)
    return QuestionDiagnosis(
        arm=diagnostic_run.run.name,
        question_id=question_id,
        category=diagnostic_run.run.metadata[question_id].category,
        correct=diagnostic_run.run.scores[question_id],
        signals=DiagnosticSignals(
            step_cap_hit=step_cap.hit,
            step_cap_reason=step_cap.reason,
            empty_retrieval_context=_empty_context(diagnostic_run.run.path, row, fields),
            degraded=degraded,
            answer_support_present=None,
        ),
        evidence_fields=sorted(fields),
    )


def _step_cap_signal(
    diagnostic_run: DiagnosticRun,
    row: dict[str, JsonValue],
    fields: list[str],
    degraded: bool | None,
) -> StepCapSignal:
    metadata = row.get("memory_post_query_metadata")
    if metadata is None:
        return StepCapSignal(hit=None, reason="missing_metadata")
    if not isinstance(metadata, dict):
        raise AuditError(diagnostic_run.run.path, "memory_post_query_metadata must be an object")
    steps = metadata.get("steps")
    if steps is None:
        return StepCapSignal(hit=None, reason="missing_steps")
    if type(steps) is not int:
        raise AuditError(
            diagnostic_run.run.path, "memory_post_query_metadata.steps must be an integer"
        )
    fields.append("per_question.memory_post_query_metadata.steps")
    if steps < 0:
        raise AuditError(
            diagnostic_run.run.path, "memory_post_query_metadata.steps must be non-negative"
        )
    if diagnostic_run.max_steps is None:
        return StepCapSignal(hit=None, reason="missing_configured_max_steps")
    fields.append("runtime_inputs.memory_config.memory_params.max_steps")
    if steps > diagnostic_run.max_steps:
        return StepCapSignal(hit=True, reason="forced_final_after_cap")
    if steps < diagnostic_run.max_steps:
        return StepCapSignal(hit=False, reason="below_configured_max_steps")
    if degraded is True:
        return StepCapSignal(hit=True, reason="cap_exhausted_degraded")
    if degraded is False:
        return StepCapSignal(hit=False, reason="final_within_last_allowed_step")
    return StepCapSignal(hit=None, reason="missing_degraded_at_cap_boundary")


def _empty_context(path: Path, row: dict[str, JsonValue], fields: list[str]) -> bool | None:
    context = row.get("memory_context")
    if context is None:
        tokens = row.get("memory_context_token_count")
        if tokens is None:
            return None
        if type(tokens) is not int:
            raise AuditError(path, "memory_context_token_count must be an integer")
        if tokens < 0:
            raise AuditError(path, "memory_context_token_count must be non-negative")
        fields.append("per_question.memory_context_token_count")
        return tokens == 0
    if not isinstance(context, list):
        raise AuditError(path, "memory_context must be a list")
    fields.append("per_question.memory_context")
    return len(context) == 0


def _degraded(path: Path, row: dict[str, JsonValue], fields: list[str]) -> bool | None:
    metadata = row.get("memory_post_query_metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise AuditError(path, "memory_post_query_metadata must be an object")
    if "degraded" not in metadata:
        return None
    value = metadata["degraded"]
    if value is None:
        fields.append("per_question.memory_post_query_metadata.degraded")
        return False
    if not isinstance(value, str):
        raise AuditError(path, "memory_post_query_metadata.degraded must be a string or null")
    fields.append("per_question.memory_post_query_metadata.degraded")
    return bool(value.strip())


def _read_json_object(path: Path) -> dict[str, JsonValue]:
    try:
        payload = parse_json(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditError(path, f"malformed JSON at line {exc.lineno} column {exc.colno}") from exc
    except JsonInputError as exc:
        raise AuditError(path, str(exc)) from exc
    if not isinstance(payload, dict):
        raise AuditError(path, "expected a JSON object")
    return payload


def _counts_by_key(questions: list[QuestionDiagnosis], *, key: str) -> dict[str, OverlapCounts]:
    names = sorted({getattr(question, key) for question in questions})
    return {
        name: overlap_counts([question for question in questions if getattr(question, key) == name])
        for name in names
    }


def _counts_by_arm_category(
    questions: list[QuestionDiagnosis],
) -> dict[str, dict[str, OverlapCounts]]:
    arms = sorted({question.arm for question in questions})
    return {
        arm: _counts_by_key(
            [question for question in questions if question.arm == arm],
            key="category",
        )
        for arm in arms
    }
