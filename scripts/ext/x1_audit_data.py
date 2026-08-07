"""Loader for the dial481/locomo-audit error catalogue (X1 gold-replay input).

Pinned upstream: https://github.com/dial481/locomo-audit @ 9493fb4b4af4256ed17a18e8fd0b3cfdeec29539
cloned to ``~/.agmem/upstream/locomo-audit``. Its ``data/locomo10.json`` is
byte-identical to our copy at ``~/.agmem/datasets/locomo10.json`` (sha256
79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4), which is what
makes ``question_id`` a valid join key against our own LoCoMo runs.

The audit flags 156 questions. Only 99 of them can move a score: the remaining
57 are WRONG_CITATION, where the golden answer is right and only the evidence
pointer is wrong -- harmless to any metric that scores the answer string.

Validation is split deliberately: this module checks *structure* (the shape any
consumer needs), while tests/test_x1_audit_data.py pins the *counts* of this
particular upstream commit. Re-pinning upstream should fail the tests, not
break every caller of load_errors().
"""

from __future__ import annotations

import json
from pathlib import Path

# Observed error_type values in the pinned errors.json, with counts:
#   HALLUCINATION 33, TEMPORAL_ERROR 26, ATTRIBUTION_ERROR 24, AMBIGUOUS 13,
#   INCOMPLETE 3  (= 99 score-corrupting), plus WRONG_CITATION 57 (benign).
ERROR_TYPES_SCORE_CORRUPTING = frozenset(
    {"HALLUCINATION", "TEMPORAL_ERROR", "ATTRIBUTION_ERROR", "AMBIGUOUS", "INCOMPLETE"}
)

_REQUIRED_FIELDS = ("question_id", "question", "error_type")


def load_errors(path: Path) -> list[dict]:
    """Load errors.json, verifying every entry carries the join fields."""
    errors = json.loads(Path(path).read_text())
    if not isinstance(errors, list):
        raise ValueError(f"{path}: expected a list of error entries, got {type(errors).__name__}")
    for i, entry in enumerate(errors):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}[{i}]: expected an object, got {type(entry).__name__}")
        missing = [f for f in _REQUIRED_FIELDS if f not in entry]
        if missing:
            raise ValueError(f"{path}[{i}]: missing required field(s) {missing}")
    return errors


def score_corrupting(errors: list[dict]) -> list[dict]:
    """The subset whose error_type can actually change a score (WRONG_CITATION out)."""
    unknown = {e["error_type"] for e in errors} - ERROR_TYPES_SCORE_CORRUPTING - {"WRONG_CITATION"}
    if unknown:
        raise ValueError(f"unclassified error_type(s) {sorted(unknown)}: cannot judge score impact")
    return [e for e in errors if e["error_type"] in ERROR_TYPES_SCORE_CORRUPTING]
