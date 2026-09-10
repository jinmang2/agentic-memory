"""Shared dataclasses for offline LongMemEval-V2 preparation tools.

These types keep parser, aggregate validation, and audit reporting independent
enough to avoid import cycles while preserving ``dataclasses.asdict`` output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import override


class AuditError(ValueError):
    """ValueError subclass whose message identifies the bad artifact path."""

    path: Path
    reason: str

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(str(self))

    @override
    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class QuestionMeta:
    """Stable row identity fields that must match across paired result arms."""

    question_id: str
    question_type: str
    category: str
    is_abstention: bool


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """Parsed per-question verdict with booleans kept as real booleans."""

    meta: QuestionMeta
    correct: bool
    unknown: bool


@dataclass(frozen=True, slots=True)
class CategorySummary:
    """Category denominator and verdict counts used by audit JSON output."""

    n: int
    correct: int
    unknown: int
    abstention_n: int
    non_abstention_n: int


@dataclass(frozen=True, slots=True)
class LoadedRun:
    """Fully validated run plus private maps used for paired comparisons."""

    name: str
    path: Path
    n: int
    correct: int
    unknown: int
    abstention_n: int
    non_abstention_n: int
    categories: dict[str, CategorySummary]
    scores: dict[str, bool]
    metadata: dict[str, QuestionMeta]
