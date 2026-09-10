"""Offline audit report for prepared LongMemEval-V2 result directories.

Callers use :func:`audit_root` before interpreting benchmark deltas. The report
is JSON-serializable with ``dataclasses.asdict`` and refuses partial joins that
would make paired comparisons look cleaner than the saved artifacts justify.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from .results import has_partial_result, has_result_pair, load_run
from .types import AuditError, CategorySummary, LoadedRun

__all__ = [
    "ArmAudit",
    "AuditError",
    "AuditReport",
    "PairedAudit",
    "audit_root",
    "exact_sign_p_value",
]


@dataclass(frozen=True, slots=True)
class ArmAudit:
    """Public per-arm counts after strict result artifact validation."""

    name: str
    n: int
    correct: int
    accuracy: float
    unknown: int
    abstention_n: int
    non_abstention_n: int
    categories: dict[str, CategorySummary]


@dataclass(frozen=True, slots=True)
class PairedAudit:
    """Exact sign-test result for two arms over one fixed question set."""

    a: str
    b: str
    both: int
    only_a: int
    only_b: int
    neither: int
    discordant: int
    p_two_sided: float
    by_category: dict[str, PairedAudit]


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Top-level audit output suitable for CLI JSON rendering."""

    root: str
    arms: list[ArmAudit]
    sign_tests: list[PairedAudit]
    skipped_dirs: list[str]


def audit_root(root: Path) -> AuditReport:
    """Audit a run root without model calls and fail on invalid paired comparisons."""

    if not root.exists():
        raise AuditError(root, "path does not exist")
    if not root.is_dir():
        raise AuditError(root, "expected a directory")
    runs, skipped_dirs = _load_runs(root)
    if not runs:
        raise AuditError(root, "no complete runs found")
    _require_pairable_runs(runs)
    return AuditReport(
        root=str(root),
        arms=[_arm_audit(run) for run in runs],
        sign_tests=[
            _paired_audit(first, second, category=None) for first, second in combinations(runs, 2)
        ],
        skipped_dirs=skipped_dirs,
    )


def exact_sign_p_value(only_first: int, only_second: int) -> float:
    """Compute the exact two-sided sign-test p-value for discordant paired wins."""

    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    smaller_side = min(only_first, only_second)
    combinations_at_or_below_tail: int = 0
    for index in range(smaller_side + 1):
        combinations_at_or_below_tail += _combination_count(discordant, index)
    p_value = float(2 * combinations_at_or_below_tail) / float(2**discordant)
    return min(1.0, p_value)


def _combination_count(total: int, chosen: int) -> int:
    chosen = min(chosen, total - chosen)
    numerator = 1
    denominator = 1
    for offset in range(1, chosen + 1):
        numerator *= total - chosen + offset
        denominator *= offset
    return numerator // denominator


def _load_runs(root: Path) -> tuple[list[LoadedRun], list[str]]:
    if has_result_pair(root) or has_partial_result(root):
        return [load_run(root)], []
    runs: list[LoadedRun] = []
    skipped_dirs: list[str] = []
    for child in sorted(candidate for candidate in root.iterdir() if candidate.is_dir()):
        if has_result_pair(child):
            runs.append(load_run(child))
        elif has_partial_result(child):
            raise AuditError(
                child, "incomplete run: expected aggregated_metrics.json and per_question.jsonl"
            )
        else:
            skipped_dirs.append(child.name)
    return runs, skipped_dirs


def _require_pairable_runs(runs: list[LoadedRun]) -> None:
    baseline = runs[0]
    baseline_ids = set(baseline.scores)
    for run in runs[1:]:
        run_ids = set(run.scores)
        if run_ids != baseline_ids:
            missing = sorted(baseline_ids - run_ids)[:5]
            extra = sorted(run_ids - baseline_ids)[:5]
            raise AuditError(
                run.path, f"question_id set mismatch: missing={missing}, extra={extra}"
            )
        for question_id in sorted(baseline_ids):
            if run.metadata[question_id] != baseline.metadata[question_id]:
                raise AuditError(run.path, f"metadata mismatch for question_id {question_id!r}")


def _arm_audit(run: LoadedRun) -> ArmAudit:
    return ArmAudit(
        name=run.name,
        n=run.n,
        correct=run.correct,
        accuracy=run.correct / run.n,
        unknown=run.unknown,
        abstention_n=run.abstention_n,
        non_abstention_n=run.non_abstention_n,
        categories=run.categories,
    )


def _paired_audit(first: LoadedRun, second: LoadedRun, *, category: str | None) -> PairedAudit:
    question_ids = _question_ids_for(first, category)
    both = sum(
        1
        for question_id in question_ids
        if first.scores[question_id] and second.scores[question_id]
    )
    only_first = sum(
        1
        for question_id in question_ids
        if first.scores[question_id] and not second.scores[question_id]
    )
    only_second = sum(
        1
        for question_id in question_ids
        if second.scores[question_id] and not first.scores[question_id]
    )
    discordant = only_first + only_second
    categories = sorted({first.metadata[question_id].category for question_id in question_ids})
    by_category = (
        {
            category_name: _paired_audit(first, second, category=category_name)
            for category_name in categories
        }
        if category is None
        else {}
    )
    return PairedAudit(
        a=first.name,
        b=second.name,
        both=both,
        only_a=only_first,
        only_b=only_second,
        neither=len(question_ids) - both - discordant,
        discordant=discordant,
        p_two_sided=exact_sign_p_value(only_first, only_second),
        by_category=by_category,
    )


def _question_ids_for(run: LoadedRun, category: str | None) -> list[str]:
    if category is None:
        return sorted(run.scores)
    return sorted(
        question_id
        for question_id, metadata in run.metadata.items()
        if metadata.category == category
    )
