from __future__ import annotations

from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

from agmem.bench.coding_eval.manifest import COST_COMPONENTS, Manifest, PlannedJob
from agmem.bench.lme_v2_tools.json_io import JsonObject, parse_json

RESULT_KEYS = frozenset(
    {
        "job_id",
        "task_id",
        "arm",
        "repeat",
        "success",
        "re_explanation_required",
        "correction_required",
        "harmful_regression",
        "latency_ms",
        "cost",
    }
)


@dataclass(frozen=True, slots=True)
class ResultsValidation:
    errors: tuple[str, ...]


def validate_results(path: Path, manifest: Manifest) -> ResultsValidation:
    try:
        raw = parse_json(path.expanduser().resolve().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ResultsValidation(errors=(f"results not found: {path}",))
    except (JSONDecodeError, TypeError, ValueError) as error:
        return ResultsValidation(errors=(f"invalid results JSON: {error}",))
    if not isinstance(raw, list):
        return ResultsValidation(errors=("results root must be a list",))
    expected = {job.job_id: job for job in manifest.jobs}
    seen: set[str] = set()
    errors: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            errors.append("result row must be an object")
            continue
        row_errors = _row_errors(item, expected)
        errors.extend(row_errors)
        job_id = item.get("job_id")
        if isinstance(job_id, str) and job_id in expected:
            seen.add(job_id)
    missing = sorted(set(expected) - seen)
    extra_count = len(raw) - len(seen)
    if missing:
        errors.append(f"missing result for job: {missing[0]}")
    if extra_count > 0 and not missing:
        errors.append("results contain duplicate or unknown jobs")
    return ResultsValidation(errors=tuple(errors))


def _row_errors(raw: JsonObject, expected: dict[str, PlannedJob]) -> tuple[str, ...]:
    unknown = sorted(set(raw) - RESULT_KEYS)
    if unknown:
        return (f"unknown result keys: {', '.join(unknown)}",)
    job_id = raw.get("job_id")
    if not isinstance(job_id, str) or job_id not in expected:
        return ("unknown result job_id",)
    job = expected[job_id]
    errors: list[str] = []
    if raw.get("task_id") != job.task_id:
        errors.append(f"task_id mismatch for job: {job_id}")
    if raw.get("arm") != job.arm:
        errors.append(f"arm mismatch for job: {job_id}")
    if raw.get("repeat") != job.repeat:
        errors.append(f"repeat mismatch for job: {job_id}")
    errors.extend(_nullable_bool(raw, key) for key in _BOOL_KEYS if _nullable_bool(raw, key))
    latency = raw.get("latency_ms")
    if latency is not None and (
        isinstance(latency, bool) or not isinstance(latency, int) or latency < 0
    ):
        errors.append("latency_ms must be null or a non-negative integer")
    cost = raw.get("cost")
    if not isinstance(cost, dict):
        errors.append("cost must be an object")
    else:
        errors.extend(_cost_errors(cost))
    return tuple(errors)


_BOOL_KEYS = (
    "success",
    "re_explanation_required",
    "correction_required",
    "harmful_regression",
)


def _nullable_bool(raw: JsonObject, key: str) -> str:
    value = raw.get(key)
    if value is not None and not isinstance(value, bool):
        return f"{key} must be null or boolean"
    return ""


def _cost_errors(raw: JsonObject) -> tuple[str, ...]:
    unknown = sorted(set(raw) - set(COST_COMPONENTS))
    if unknown:
        return (f"unknown cost keys: {', '.join(unknown)}",)
    errors: list[str] = []
    for key in COST_COMPONENTS:
        value = raw.get(key)
        if value is None:
            continue
        if key.endswith("_tokens"):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"{key} must be null or a non-negative integer")
        elif isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            errors.append(f"{key} must be null or a non-negative number")
    return tuple(errors)
