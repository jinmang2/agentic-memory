"""Invariant checks on the pinned dial481/locomo-audit error catalogue.

These are data-pin tests, not logic tests: they assert that the clone at
``~/.agmem/upstream/locomo-audit`` still carries the exact error counts the
audit published (156 entries, 99 of them score-corrupting). If the upstream
repo is re-pinned to a newer commit and these fail, the join keys that X1's
replay depends on have moved and the downstream numbers must be re-derived --
do not relax the assertions to make them pass.
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from collections import Counter
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "ext" / "x1_audit_data.py"

AUDIT = Path.home() / ".agmem/upstream/locomo-audit"
ERRORS = AUDIT / "errors.json"

pytestmark = pytest.mark.skipif(not AUDIT.exists(), reason="locomo-audit clone not present")


def _load_module():
    """Import scripts/ext/x1_audit_data.py as a module -- scripts/ holds no
    packages, so this mirrors tests/test_locomo_eval.py's flat-import pattern."""
    if str(_MODULE_PATH.parent) not in sys.path:
        sys.path.insert(0, str(_MODULE_PATH.parent))
    spec = _ilu.spec_from_file_location("x1_audit_data", _MODULE_PATH)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_error_counts_match_published_claims():
    x1 = _load_module()
    errors = x1.load_errors(ERRORS)
    assert len(errors) == 156
    # published claim: 99 of the 156 are score-corrupting, the other 57 being WRONG_CITATION
    sc = x1.score_corrupting(errors)
    assert len(sc) == 99
    assert all(e["error_type"] in x1.ERROR_TYPES_SCORE_CORRUPTING for e in sc)
    # assert the per-type distribution too -- a bare total of 99 cannot catch a
    # mislabelled type that happens to preserve the sum.
    counts = sorted(Counter(e["error_type"] for e in sc).values(), reverse=True)
    assert counts == [33, 26, 24, 13, 3]


def test_benign_remainder_is_wrong_citation_only():
    x1 = _load_module()
    errors = x1.load_errors(ERRORS)
    benign = [e for e in errors if e["error_type"] not in x1.ERROR_TYPES_SCORE_CORRUPTING]
    assert len(benign) == 57
    assert {e["error_type"] for e in benign} == {"WRONG_CITATION"}


def test_entries_have_join_fields():
    x1 = _load_module()
    for e in x1.load_errors(ERRORS):
        assert e["question_id"].startswith("locomo_")
        assert e["question"].strip()


def test_question_ids_are_unique():
    x1 = _load_module()
    ids = [e["question_id"] for e in x1.load_errors(ERRORS)]
    assert len(set(ids)) == len(ids)


def test_load_errors_rejects_non_list(tmp_path):
    x1 = _load_module()
    bad = tmp_path / "errors.json"
    bad.write_text('{"question_id": "locomo_0_qa1"}')
    with pytest.raises(ValueError):
        x1.load_errors(bad)


def test_load_errors_rejects_entry_missing_join_field(tmp_path):
    x1 = _load_module()
    bad = tmp_path / "errors.json"
    bad.write_text('[{"question_id": "locomo_0_qa1", "error_type": "HALLUCINATION"}]')
    with pytest.raises(ValueError, match=r"\['question'\]"):
        x1.load_errors(bad)


def test_score_corrupting_refuses_to_silently_drop_a_new_error_type():
    # a type upstream might add later must not fall through as benign -- that
    # would understate the corrupted count without anything failing.
    x1 = _load_module()
    with pytest.raises(ValueError, match="NEW_TYPE"):
        x1.score_corrupting([{"question_id": "q", "question": "?", "error_type": "NEW_TYPE"}])
