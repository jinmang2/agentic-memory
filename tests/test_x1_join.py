"""Join tests for the audit-errors <-> our-records key mapping.

The synthetic tests pin the join *logic*; the two real-data tests at the bottom
pin the *resolution* of the pinned audit's 156 question_ids against our copy of
locomo10.json. That resolution is the whole premise of X1: if a question_id
stops naming the question the audit says it names, every downstream replay
number is measuring the wrong rows, so those tests must fail loudly rather than
be relaxed.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_JOIN_PATH = _ROOT / "scripts" / "ext" / "x1_join.py"
_AUDIT_DATA_PATH = _ROOT / "scripts" / "ext" / "x1_audit_data.py"

DATASET = Path.home() / ".agmem/datasets/locomo10.json"
AUDIT = Path.home() / ".agmem/upstream/locomo-audit"
ERRORS = AUDIT / "errors.json"
RECORDS = (
    _ROOT
    / "results/repro"
    / "gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA.records.jsonl"
)


def _load_module(path: Path, name: str):
    """Import a scripts/ext module by path -- scripts/ holds no packages, so
    this mirrors tests/test_x1_audit_data.py's flat-import pattern."""
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _join():
    return _load_module(_JOIN_PATH, "x1_join")


def _audit_data():
    return _load_module(_AUDIT_DATA_PATH, "x1_audit_data")


def test_normalize_q_collapses_whitespace_and_case():
    x1 = _join()
    assert x1.normalize_q("  When   did\tX?\n") == "when did x?"


def test_error_keys_looked_up_via_key_map_with_crosscheck():
    x1 = _join()
    key_map = {"locomo_3_qa17": (3, "when did x?")}
    keys = x1.error_question_keys(
        [{"question_id": "locomo_3_qa17", "question": "When  did X?"}], key_map
    )
    assert keys == {(3, "when did x?")}


def test_crosscheck_mismatch_raises():
    x1 = _join()
    key_map = {"locomo_3_qa17": (3, "a totally different question?")}
    with pytest.raises(ValueError, match="locomo_3_qa17"):
        x1.error_question_keys(
            [{"question_id": "locomo_3_qa17", "question": "When did X?"}], key_map
        )


def test_unknown_question_id_raises():
    x1 = _join()
    with pytest.raises(ValueError, match="locomo_9_qa999"):
        x1.error_question_keys([{"question_id": "locomo_9_qa999", "question": "?"}], {})


def _write_synthetic_dataset(tmp_path: Path) -> Path:
    """Minimal locomo10-shaped file: a list of samples, each with a qa list."""
    samples = [
        {
            "qa": [
                {"question": "Q  zero?", "answer": "a", "category": 1},
                {"question": "Q one?", "answer": "b", "category": 2},
            ],
            "conversation": {"speaker_a": "A", "speaker_b": "B"},
        },
        {
            "qa": [
                {"question": "Other  ZERO?", "answer": "c", "category": 4},
                {"question": "Other one?", "adversarial_answer": "no info", "category": 5},
            ],
            "conversation": {"speaker_a": "C", "speaker_b": "D"},
        },
    ]
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps(samples))
    return path


def test_question_key_map_enumerates_dataset_order(tmp_path):
    x1 = _join()
    key_map = x1.question_key_map(_write_synthetic_dataset(tmp_path))
    # sample index is the conv id; qa index is 0-based over the full qa list,
    # category-5 rows included (they occupy indices like any other row).
    assert key_map == {
        "locomo_0_qa0": (0, "q zero?"),
        "locomo_0_qa1": (0, "q one?"),
        "locomo_1_qa0": (1, "other zero?"),
        "locomo_1_qa1": (1, "other one?"),
    }


def test_match_report_lists_unmatched():
    x1 = _join()
    rep = x1.match_report({(0, "a?")}, {(0, "a?"), (1, "b?")})
    assert rep["matched"] == 1 and rep["unmatched_errors"] == [(1, "b?")]


def test_match_report_counts_duplicate_matches_and_excluded_rows():
    x1 = _join()
    # (0, "a?") is served by two records rows: matching it excludes both, so the
    # denominator drops by 2, not 1.
    counts = {(0, "a?"): 2, (0, "c?"): 1}
    rep = x1.match_report(counts, {(0, "a?"), (1, "b?")}, record_counts=counts)
    assert rep["matched"] == 1
    assert rep["duplicate_matched_keys"] == 1
    assert rep["excluded_rows"] == 2
    assert rep["unmatched_errors"] == [(1, "b?")]


def test_match_report_derives_counts_from_a_counter_records_arg():
    # handing over duplicate-bearing records must not silently report zero
    # duplicates just because record_counts was left off.
    x1 = _join()
    rep = x1.match_report(Counter({(0, "a?"): 2}), {(0, "a?")})
    assert rep["duplicate_matched_keys"] == 1
    assert rep["excluded_rows"] == 2


def test_match_report_without_counts_assumes_one_row_per_key():
    x1 = _join()
    rep = x1.match_report({(0, "a?"), (0, "c?")}, {(0, "a?"), (0, "c?")})
    assert rep["matched"] == 2
    assert rep["duplicate_matched_keys"] == 0
    assert rep["excluded_rows"] == 2


def test_judged_record_counts_skips_unjudged_rows(tmp_path):
    x1 = _join()
    path = tmp_path / "r.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"conv": 0, "q": "A  question?", "cat": "single-hop", "j": 1}),
                json.dumps({"conv": 0, "q": "a question?", "cat": "single-hop", "j": 0}),
                json.dumps({"conv": 1, "q": "Adversarial?", "cat": "adversarial"}),
            ]
        )
        + "\n"
    )
    counts = x1.judged_record_counts(path)
    assert counts == {(0, "a question?"): 2}


# --- real-data pins -------------------------------------------------------

_have_data = DATASET.exists() and ERRORS.exists()
_have_records = _have_data and RECORDS.exists()


@pytest.mark.skipif(not _have_data, reason="locomo10.json or locomo-audit clone not present")
def test_all_flagged_question_ids_resolve_against_our_dataset_copy():
    x1, ad = _join(), _audit_data()
    key_map = x1.question_key_map(DATASET)
    errors = ad.load_errors(ERRORS)
    # cross-check, not just lookup: every flagged id must name the same question
    # text the audit recorded. Raises on the first disagreement.
    assert len(x1.error_question_keys(errors, key_map)) == len(errors) == 156


@pytest.mark.skipif(not _have_records, reason="headline e3sA records not present")
def test_every_score_corrupting_error_matches_a_judged_record():
    x1, ad = _join(), _audit_data()
    keys = x1.error_question_keys(
        ad.score_corrupting(ad.load_errors(ERRORS)), x1.question_key_map(DATASET)
    )
    counts = x1.judged_record_counts(RECORDS)
    rep = x1.match_report(counts, keys, record_counts=counts)
    assert rep["unmatched_errors"] == []
    assert rep["matched"] == 99
