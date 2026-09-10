"""Typed cost report schema for offline LongMemEval-V2 fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Final, override

from .budget import BudgetDecision

COST_SCHEMA_VERSION: Final = 1


@unique
class CostComponent(StrEnum):
    """Measured cost components in the offline LME-V2 study plan."""

    READER = "reader"
    RETRIEVAL = "retrieval"
    WRITE = "write"
    EMBEDDING = "embedding"
    JUDGE = "judge"


COMPONENT_ORDER: Final = (
    CostComponent.READER,
    CostComponent.RETRIEVAL,
    CostComponent.WRITE,
    CostComponent.EMBEDDING,
    CostComponent.JUDGE,
)
INTEGRATION_POINTS: Final = (
    "src/agmem/llm/client.py: LLMClient.chat records role/budget_key tokens and trace rows",
    "src/agmem/llm/structured.py: StructuredCaller distinguishes transport and reply retries",
    "src/agmem/embed/api_embedder.py: APIEmbedder exposes calls/tokens/errors/transport_recoveries",
    "src/agmem/bench/lme_v2.py: answer/judge paths already tag per-question budget keys",
)


@dataclass(frozen=True, slots=True)
class CostInputError(ValueError):
    """Raised when a cost fixture cannot produce an unambiguous report."""

    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """One deduplicated priced or unpriced producer usage event."""

    event_id: str
    component: CostComponent
    base_event_id: str | None
    is_retry: bool
    calls: int
    tokens_in: int | None
    tokens_out: int | None
    cost_micro_usd: int | None


@dataclass(frozen=True, slots=True)
class ComponentTotal:
    """JSON-safe aggregate for one cost component."""

    events: int
    calls: int
    tokens_in: int | None
    tokens_out: int | None
    cost_micro_usd: int | None


@dataclass(frozen=True, slots=True)
class CostReport:
    """Frozen report shape consumed directly by tests and future CLI output."""

    schema_version: int
    expected_components: tuple[str, ...]
    zero_cost_components: tuple[str, ...]
    components: dict[str, ComponentTotal]
    total_micro_usd: int | None
    retry_cost_micro_usd: int | None
    missing_coverage: tuple[str, ...]
    reservations: tuple[BudgetDecision, ...]
    decision: BudgetDecision
    live_cap_guarantee: bool
    integration_points: tuple[str, ...]
