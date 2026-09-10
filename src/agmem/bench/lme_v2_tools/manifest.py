"""Typed manifest records for offline LongMemEval-V2 measurement preparation.

The manifest stores only identifiers, paths and hashes. It deliberately omits
question text, gold answers, config contents and generated shell commands so a
reviewer can check drift and planned outputs without exposing run payloads.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Final

from .json_io import JsonValue

SCHEMA_VERSION: Final = 2
COST_COMPONENTS: Final = ("reader", "retrieval", "write", "embedding", "judge", "retries")


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """Path-scoped file digest; callers get full SHA256 but never file contents."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ArmPlan:
    """One write/read arm that can be expanded into one or more repeat jobs."""

    name: str
    write: str
    read: str
    store: str
    query_strategy: str
    settings: tuple[Fingerprint, ...]
    fixed_store_id: str | None


@dataclass(frozen=True, slots=True)
class FixedStorePlan:
    """Immutable saved memory snapshot shared by fixed-store arms."""

    id: str
    write: str
    path: str
    files: tuple[Fingerprint, ...]
    content_sha256: str


@dataclass(frozen=True, slots=True)
class PlannedJob:
    """A single planned output directory; preparation never creates or executes it."""

    arm: str
    repeat: int
    output_path: str
    memory_mode: str
    fixed_store_id: str | None


@dataclass(frozen=True, slots=True)
class CostCoverage:
    """Whole-study nullable cost slots; estimates are not multiplied by job count."""

    reader: float | None
    retrieval: float | None
    write: float | None
    embedding: float | None
    judge: float | None
    retries: float | None


@dataclass(frozen=True, slots=True)
class Manifest:
    """Frozen inventory of one measurement recipe and the exact files it depends on."""

    schema_version: int
    recipe_path: str
    recipe_sha256: str
    study: str
    domain: str
    tier: str
    data_root: str
    questions: Fingerprint
    haystack: Fingerprint
    trajectories: Fingerprint
    configs: tuple[Fingerprint, ...]
    source_files: tuple[Fingerprint, ...]
    reader: str
    judge: str
    arms: tuple[ArmPlan, ...]
    fixed_stores: tuple[FixedStorePlan, ...]
    jobs: tuple[PlannedJob, ...]
    selected_question_ids: tuple[str, ...]
    selected_trajectory_ids: tuple[str, ...]
    costs: CostCoverage
    missing_cost_components: tuple[str, ...]
    estimated_total_usd: float | None


@dataclass(frozen=True, slots=True)
class Verification:
    """Result of regenerating a manifest from its recipe and comparing fingerprints."""

    valid: bool
    errors: tuple[str, ...]
    missing_cost_components: tuple[str, ...]


def canonical_json(manifest: Manifest) -> str:
    """Serialize a manifest deterministically for file writes and drift checks."""
    return json.dumps(asdict(manifest), allow_nan=False, indent=2, sort_keys=True) + "\n"


def canonical_json_value(value: JsonValue) -> str:
    """Canonicalize parsed JSON so verification ignores formatting and key order."""
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
