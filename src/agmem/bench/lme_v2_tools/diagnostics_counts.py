"""Count helpers for LongMemEval-V2 diagnostic reports."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class DiagnosticSignals:
    """Evidence flags extracted without reading prompt, gold, or context payloads."""

    step_cap_hit: bool | None
    step_cap_reason: str
    empty_retrieval_context: bool | None
    degraded: bool | None
    answer_support_present: bool | None


@dataclass(frozen=True, slots=True)
class OutcomeCounts:
    """Correct and incorrect counts for one diagnostic bucket."""

    correct: int = 0
    incorrect: int = 0


@dataclass(frozen=True, slots=True)
class SignalObservationCounts:
    """Observed denominators for each diagnostic signal."""

    step_cap: OutcomeCounts = field(default_factory=OutcomeCounts)
    empty_context: OutcomeCounts = field(default_factory=OutcomeCounts)
    degraded: OutcomeCounts = field(default_factory=OutcomeCounts)
    all_three: OutcomeCounts = field(default_factory=OutcomeCounts)


@dataclass(frozen=True, slots=True)
class ExclusiveSignalCounts:
    """Exclusive partition over the three core diagnostic signals."""

    none: OutcomeCounts = field(default_factory=OutcomeCounts)
    step_cap_only: OutcomeCounts = field(default_factory=OutcomeCounts)
    empty_context_only: OutcomeCounts = field(default_factory=OutcomeCounts)
    degraded_only: OutcomeCounts = field(default_factory=OutcomeCounts)
    step_cap_empty_context: OutcomeCounts = field(default_factory=OutcomeCounts)
    step_cap_degraded: OutcomeCounts = field(default_factory=OutcomeCounts)
    empty_context_degraded: OutcomeCounts = field(default_factory=OutcomeCounts)
    all_three: OutcomeCounts = field(default_factory=OutcomeCounts)
    unknown: OutcomeCounts = field(default_factory=OutcomeCounts)


@dataclass(frozen=True, slots=True)
class OverlapCounts:
    """Arm or category counts for observed diagnostic signal overlaps."""

    n: int = 0
    correct: int = 0
    incorrect: int = 0
    observed: SignalObservationCounts = field(default_factory=SignalObservationCounts)
    exclusive: ExclusiveSignalCounts = field(default_factory=ExclusiveSignalCounts)
    missing_evidence: OutcomeCounts = field(default_factory=OutcomeCounts)
    step_cap: OutcomeCounts = field(default_factory=OutcomeCounts)
    empty_context: OutcomeCounts = field(default_factory=OutcomeCounts)
    degraded: OutcomeCounts = field(default_factory=OutcomeCounts)
    step_cap_empty_context: OutcomeCounts = field(default_factory=OutcomeCounts)
    step_cap_degraded: OutcomeCounts = field(default_factory=OutcomeCounts)
    empty_context_degraded: OutcomeCounts = field(default_factory=OutcomeCounts)
    all_three: OutcomeCounts = field(default_factory=OutcomeCounts)


class HasDiagnosticSignals(Protocol):
    @property
    def correct(self) -> bool: ...

    @property
    def signals(self) -> DiagnosticSignals: ...


def overlap_counts(questions: Sequence[HasDiagnosticSignals]) -> OverlapCounts:
    """Count inclusive overlaps plus observed and exclusive signal partitions."""

    return OverlapCounts(
        n=len(questions),
        correct=sum(1 for question in questions if question.correct),
        incorrect=sum(1 for question in questions if not question.correct),
        observed=SignalObservationCounts(
            step_cap=_count(questions, lambda signals: signals.step_cap_hit is not None),
            empty_context=_count(
                questions, lambda signals: signals.empty_retrieval_context is not None
            ),
            degraded=_count(questions, lambda signals: signals.degraded is not None),
            all_three=_count(questions, lambda signals: None not in core_signals(signals)),
        ),
        exclusive=ExclusiveSignalCounts(
            none=_count(questions, lambda signals: _signature(signals) == (False, False, False)),
            step_cap_only=_count(
                questions, lambda signals: _signature(signals) == (True, False, False)
            ),
            empty_context_only=_count(
                questions, lambda signals: _signature(signals) == (False, True, False)
            ),
            degraded_only=_count(
                questions, lambda signals: _signature(signals) == (False, False, True)
            ),
            step_cap_empty_context=_count(
                questions, lambda signals: _signature(signals) == (True, True, False)
            ),
            step_cap_degraded=_count(
                questions, lambda signals: _signature(signals) == (True, False, True)
            ),
            empty_context_degraded=_count(
                questions, lambda signals: _signature(signals) == (False, True, True)
            ),
            all_three=_count(questions, has_all_three),
            unknown=_count(questions, lambda signals: None in core_signals(signals)),
        ),
        missing_evidence=_count(questions, lambda signals: None in core_signals(signals)),
        step_cap=_count(questions, lambda signals: signals.step_cap_hit is True),
        empty_context=_count(questions, lambda signals: signals.empty_retrieval_context is True),
        degraded=_count(questions, lambda signals: signals.degraded is True),
        step_cap_empty_context=_count(
            questions,
            lambda signals: (
                signals.step_cap_hit is True and signals.empty_retrieval_context is True
            ),
        ),
        step_cap_degraded=_count(
            questions, lambda signals: signals.step_cap_hit is True and signals.degraded is True
        ),
        empty_context_degraded=_count(
            questions,
            lambda signals: signals.empty_retrieval_context is True and signals.degraded is True,
        ),
        all_three=_count(questions, has_all_three),
    )


def core_signals(signals: DiagnosticSignals) -> tuple[bool | None, bool | None, bool | None]:
    return signals.step_cap_hit, signals.empty_retrieval_context, signals.degraded


def has_all_three(signals: DiagnosticSignals) -> bool:
    return all(signal is True for signal in core_signals(signals))


def _signature(signals: DiagnosticSignals) -> tuple[bool, bool, bool] | None:
    values = core_signals(signals)
    if None in values:
        return None
    first, second, third = values
    return first is True, second is True, third is True


def _count(
    questions: Sequence[HasDiagnosticSignals], predicate: Callable[[DiagnosticSignals], bool]
) -> OutcomeCounts:
    matching = [question for question in questions if predicate(question.signals)]
    return OutcomeCounts(
        correct=sum(1 for question in matching if question.correct),
        incorrect=sum(1 for question in matching if not question.correct),
    )
