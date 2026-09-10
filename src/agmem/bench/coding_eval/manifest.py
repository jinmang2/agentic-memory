from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Final

from agmem.bench.lme_v2_tools.json_io import JsonValue

SCHEMA_VERSION: Final = 1
ARM_MODES: Final = ("none", "raw", "runbook")
REQUIRED_SCENARIOS: Final = (
    "constraints_decisions",
    "recurring_failure",
    "corrected_stale_memory",
    "harmful_memory",
)
COST_COMPONENTS: Final = (
    "prompt_tokens",
    "completion_tokens",
    "retrieval_tokens",
    "memory_write_tokens",
    "judge_tokens",
    "provider_usd",
)


@dataclass(frozen=True, slots=True)
class Fingerprint:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class TaskPlan:
    id: str
    scenario: str
    fixture_files: tuple[Fingerprint, ...]
    test_command: tuple[str, ...]
    constraints: tuple[str, ...]
    decisions: tuple[str, ...]
    acceptance_checks: tuple[str, ...]
    memory_fixtures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArmPlan:
    name: str
    memory_mode: str
    memory_source: str | None
    retrieval_mode: str


@dataclass(frozen=True, slots=True)
class PlannedJob:
    job_id: str
    task_id: str
    scenario: str
    arm: str
    repeat: int
    result_path: str


@dataclass(frozen=True, slots=True)
class CostSchema:
    prompt_tokens: int | None
    completion_tokens: int | None
    retrieval_tokens: int | None
    memory_write_tokens: int | None
    judge_tokens: int | None
    provider_usd: float | None


@dataclass(frozen=True, slots=True)
class OutcomeSchema:
    success: bool | None
    re_explanation_required: bool | None
    correction_required: bool | None
    harmful_regression: bool | None
    latency_ms: int | None
    cost: CostSchema


@dataclass(frozen=True, slots=True)
class Manifest:
    schema_version: int
    recipe_path: str
    recipe_sha256: str
    study: str
    tasks: Fingerprint
    settings: tuple[Fingerprint, ...]
    source_files: tuple[Fingerprint, ...]
    arms: tuple[ArmPlan, ...]
    task_plans: tuple[TaskPlan, ...]
    jobs: tuple[PlannedJob, ...]
    scenarios: tuple[str, ...]
    required_scenarios: tuple[str, ...]
    result_schema: OutcomeSchema
    cost_components: tuple[str, ...]
    cost_estimate_status: str


@dataclass(frozen=True, slots=True)
class Verification:
    valid: bool
    errors: tuple[str, ...]


def blank_outcome_schema() -> OutcomeSchema:
    return OutcomeSchema(
        success=None,
        re_explanation_required=None,
        correction_required=None,
        harmful_regression=None,
        latency_ms=None,
        cost=CostSchema(
            prompt_tokens=None,
            completion_tokens=None,
            retrieval_tokens=None,
            memory_write_tokens=None,
            judge_tokens=None,
            provider_usd=None,
        ),
    )


def canonical_json(manifest: Manifest | Verification) -> str:
    return json.dumps(asdict(manifest), allow_nan=False, indent=2, sort_keys=True) + "\n"


def canonical_json_value(value: JsonValue) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
