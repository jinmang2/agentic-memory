"""Aggregation tests for the gold-error exclusion replay.

The synthetic tests pin the *arithmetic*: what J and F1 mean, which rows carry a
verdict, and -- the point of the whole exercise -- that excluding a question
removes it from the denominator rather than merely zeroing its numerator. An
exclusion that only dropped correct answers would manufacture a decline; one
that only dropped the numerator would manufacture a rise. Both are hand-checked
below on six rows.

The real-data tests at the bottom pin the four headline runs against the
controller-measured values already published elsewhere in this repo. They exist
so that a change to the aggregation convention (a different J denominator, say)
fails here instead of silently re-basing every replay number.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_RESCORE_PATH = _ROOT / "scripts" / "ext" / "x1_rescore.py"

REPRO = _ROOT / "results/repro"


def _load_module(path: Path, name: str):
    """Import a scripts/ext module by path -- scripts/ holds no packages, so
    this mirrors tests/test_x1_join.py's flat-import pattern."""
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rescore():
    return _load_module(_RESCORE_PATH, "x1_rescore")


def _rec(conv: int, q: str, cat: str = "single-hop", f1: float = 1.0, j: bool | None = None):
    row = {"conv": conv, "q": q, "cat": cat, "f1": f1, "gold": "g", "pred": "p"}
    if j is not None:
        row["j"] = j
    return row


def _six_rows():
    """2 conversations x 3 questions. (0, "b?") is the flagged one.

    Note the two "a?" rows sit in different conversations: they share question
    text, so anything joining on text alone would fuse them.
    """
    return [
        _rec(0, "a?", "single-hop", 1.0, True),
        _rec(0, "b?", "temporal", 0.5, False),
        _rec(0, "adv?", "adversarial", 0.2),
        _rec(1, "a?", "single-hop", 0.0, False),
        _rec(1, "c?", "temporal", 0.8, True),
        _rec(1, "adv2?", "adversarial", 0.4),
    ]


# --- the exclusion is a denominator change, not just a numerator change ----


def test_exclusion_changes_denominator_not_just_numerator():
    x1 = _rescore()
    recs = [_rec(0, "a?", j=True), _rec(0, "b?", j=False), _rec(1, "c?", j=True)]
    full = x1.aggregate(recs)
    excl = x1.aggregate(recs, exclude_keys={(0, "b?")})
    assert full["n"] == 3 and excl["n"] == 2
    assert excl["J"] == pytest.approx(100.0)


def test_excluding_a_correct_row_lowers_J():
    # the mirror of the test above: if exclusion only ever removed wrong answers
    # the replay would be rigged to rise. Removing a *correct* row must lower J.
    x1 = _rescore()
    recs = [_rec(0, "a?", j=True), _rec(0, "b?", j=False), _rec(1, "c?", j=True)]
    excl = x1.aggregate(recs, exclude_keys={(0, "a?")})
    assert excl["n"] == 2
    assert excl["J"] == pytest.approx(50.0)


# --- hand-checked arithmetic on the six-row fixture ------------------------


def test_aggregate_full_matches_hand_computation():
    x1 = _rescore()
    agg = x1.aggregate(_six_rows())
    assert agg["n"] == 6
    assert agg["j_n"] == 4  # the two adversarial rows carry no verdict
    assert agg["J"] == pytest.approx(50.0)  # 2 of 4 judged rows true
    assert agg["F1"] == pytest.approx(100 * 2.9 / 6)  # 1.0+0.5+0.2+0.0+0.8+0.4


def test_aggregate_excluded_matches_hand_computation():
    x1 = _rescore()
    agg = x1.aggregate(_six_rows(), exclude_keys={(0, "b?")})
    assert agg["n"] == 5
    assert agg["j_n"] == 3
    assert agg["J"] == pytest.approx(100 * 2 / 3)
    assert agg["F1"] == pytest.approx(100 * 2.4 / 5)  # 1.0+0.2+0.0+0.8+0.4


def test_per_cat_matches_hand_computation():
    x1 = _rescore()
    per_cat = x1.aggregate(_six_rows())["per_cat"]
    assert per_cat["single-hop"] == {
        "n": 2,
        "j_n": 2,
        "J": pytest.approx(50.0),
        "F1": pytest.approx(50.0),
    }
    assert per_cat["temporal"]["J"] == pytest.approx(50.0)
    assert per_cat["temporal"]["F1"] == pytest.approx(65.0)


def test_per_cat_after_exclusion_renormalizes_within_the_category():
    x1 = _rescore()
    per_cat = x1.aggregate(_six_rows(), exclude_keys={(0, "b?")})["per_cat"]
    assert per_cat["temporal"] == {
        "n": 1,
        "j_n": 1,
        "J": pytest.approx(100.0),
        "F1": pytest.approx(80.0),
    }
    assert per_cat["single-hop"]["n"] == 2  # untouched


def test_adversarial_rows_score_F1_but_never_J():
    x1 = _rescore()
    per_cat = x1.aggregate(_six_rows())["per_cat"]
    assert per_cat["adversarial"]["n"] == 2
    assert per_cat["adversarial"]["j_n"] == 0
    assert per_cat["adversarial"]["J"] is None  # not 0.0 -- unmeasured, not zero
    assert per_cat["adversarial"]["F1"] == pytest.approx(30.0)


def test_conv_is_part_of_the_key():
    # (1, "a?") shares text with (0, "a?"); excluding conv 0 must leave conv 1.
    x1 = _rescore()
    agg = x1.aggregate(_six_rows(), exclude_keys={(0, "a?")})
    assert agg["n"] == 5
    assert agg["per_cat"]["single-hop"]["n"] == 1


def test_exclusion_normalizes_question_text_on_both_sides():
    x1 = _rescore()
    recs = [_rec(0, "  When   did\tX?", j=True), _rec(0, "other?", j=False)]
    agg = x1.aggregate(recs, exclude_keys={(0, "when did x?")})
    assert agg["n"] == 1


def test_exclusion_removes_every_row_sharing_a_key():
    # conv 7 ships exactly-duplicated questions: one flagged key retires both
    # rows, or the denominator reduction is understated.
    x1 = _rescore()
    recs = [_rec(0, "dup?", j=True), _rec(0, "dup?", j=True), _rec(0, "keep?", j=False)]
    agg = x1.aggregate(recs, exclude_keys={(0, "dup?")})
    assert agg["n"] == 1 and agg["j_n"] == 1
    assert agg["J"] == pytest.approx(0.0)


def test_a_category_emptied_by_exclusion_still_appears_with_n_zero():
    # the full and excluded tables must line up row for row; a vanished category
    # would read as "unchanged" rather than "entirely flagged".
    x1 = _rescore()
    per_cat = x1.aggregate(_six_rows(), exclude_keys={(0, "b?"), (1, "c?")})["per_cat"]
    assert per_cat["temporal"] == {"n": 0, "j_n": 0, "J": None, "F1": None}


def test_cats_restricts_the_scope():
    x1 = _rescore()
    agg = x1.aggregate(_six_rows(), cats={"single-hop", "temporal"})
    assert agg["n"] == 4 and agg["j_n"] == 4
    assert set(agg["per_cat"]) == {"single-hop", "temporal"}
    assert agg["F1"] == pytest.approx(100 * 2.3 / 4)


def test_no_rows_yields_none_rather_than_zero():
    x1 = _rescore()
    agg = x1.aggregate([])
    assert agg == {"n": 0, "j_n": 0, "J": None, "F1": None, "per_cat": {}}


# --- record loading -------------------------------------------------------


def _write_records(tmp_path: Path, rows: list[dict], stem: str = "run_ours_x") -> Path:
    """Write a records file. The stem matters: it carries the eval mode, which
    is what decides whether a verdict-free run is legitimate."""
    path = tmp_path / f"{stem}.records.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_load_records_reads_jsonl_and_skips_blank_lines(tmp_path):
    x1 = _rescore()
    path = tmp_path / "r.records.jsonl"
    path.write_text(json.dumps(_rec(0, "a?", j=True)) + "\n\n")
    assert len(x1.load_records(path)) == 1


def test_load_records_requires_the_scoring_fields(tmp_path):
    x1 = _rescore()
    bad = _rec(0, "a?", j=True)
    del bad["f1"]
    with pytest.raises(ValueError, match="f1"):
        x1.load_records(_write_records(tmp_path, [bad]))


def test_load_records_rejects_a_partially_judged_run(tmp_path):
    # a judged row silently missing "j" would shrink the J denominator without
    # anything looking wrong -- exactly the failure this replay must not have.
    x1 = _rescore()
    rows = [_rec(0, "a?", "single-hop", 1.0, True), _rec(0, "b?", "single-hop", 1.0)]
    with pytest.raises(ValueError, match="judge verdict"):
        x1.load_records(_write_records(tmp_path, rows))


def test_load_records_accepts_a_verdict_free_run_from_an_f1_only_eval_mode(tmp_path):
    # the wujiang eval mode scores F1/BLEU only and emits no verdicts anywhere.
    # That is a different measurement, not a damaged file -- J is unmeasurable
    # for it, which is what the artifact must say.
    x1 = _rescore()
    rows = [_rec(0, "a?", "single-hop", 1.0), _rec(0, "b?", "temporal", 0.5)]
    path = _write_records(tmp_path, rows, stem="gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed1")
    loaded = x1.load_records(path)
    assert len(loaded) == 2
    agg = x1.aggregate(loaded)
    assert agg["j_n"] == 0
    assert agg["J"] is None  # unmeasured, not zero
    assert agg["F1"] == pytest.approx(75.0)


def test_load_records_rejects_a_verdict_free_run_from_a_judging_eval_mode(tmp_path):
    # "the eval mode never judges" and "the judge pass died wholesale" produce
    # byte-identical files. Only the eval mode in the stem tells them apart, and
    # accepting both as null-J would report a crashed judge as a clean F1-only
    # measurement.
    x1 = _rescore()
    rows = [_rec(0, "a?", "single-hop", 1.0), _rec(0, "b?", "temporal", 0.5)]
    path = _write_records(tmp_path, rows, stem="gpt-4o-mini_nemori_upstream_all_k10_ours_run1_zz")
    with pytest.raises(ValueError, match="judge"):
        x1.load_records(path)


def test_is_f1_only_eval_matches_a_whole_stem_token_not_a_substring(tmp_path):
    x1 = _rescore()
    assert x1.is_f1_only_eval("gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed1") is True
    assert x1.is_f1_only_eval("gpt-4o-mini_all_k10_ours_expand-on_run1_seed1") is False
    # a token that merely contains the mode name is a different eval mode
    assert x1.is_f1_only_eval("gpt-4o-mini_all_k10_wujiangx_run1") is False


def test_load_records_accepts_adversarial_rows_without_a_verdict(tmp_path):
    x1 = _rescore()
    rows = x1.load_records(_write_records(tmp_path, [_rec(0, "a?", "adversarial", 0.2)]))
    assert len(rows) == 1 and "j" not in rows[0]


# --- per-file rescore and its join disclosure -----------------------------


def test_rescore_file_reports_both_join_bases_for_a_judged_run(tmp_path):
    x1 = _rescore()
    path = _write_records(tmp_path, _six_rows())
    r = x1.rescore_file(path, {(0, "b?")})
    assert r["judged"] is True
    assert r["n_full"] == 6 and r["n_excluded"] == 5
    assert r["join"]["matched_any_row"] == 1 and r["join"]["excluded_rows_all"] == 1
    assert r["join"]["matched_judged"] == 1 and r["join"]["excluded_judged_rows"] == 1
    assert r["join"]["judged_basis_applicable"] is True
    assert r["J_full"] == pytest.approx(50.0) and r["J_excluded"] == pytest.approx(100 * 2 / 3)
    assert r["delta_J"] == pytest.approx(100 * 2 / 3 - 50.0, abs=1e-4)


WUJIANG_STEM = "gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed1"


def test_rescore_file_on_an_unjudged_run_still_reports_the_F1_exclusion(tmp_path):
    # the wujiang shape: no verdicts anywhere. The judged basis is unmeasurable,
    # but the flagged question must still leave the F1 denominator -- reporting
    # only the judged basis here would read as a join failure.
    x1 = _rescore()
    rows = [_rec(0, "a?", "single-hop", 1.0), _rec(0, "b?", "temporal", 0.0)]
    r = x1.rescore_file(_write_records(tmp_path, rows, stem=WUJIANG_STEM), {(0, "b?")})
    assert r["judged"] is False
    assert r["J_full"] is None and r["J_excluded"] is None and r["delta_J"] is None
    assert r["n_full"] == 2 and r["n_excluded"] == 1
    assert r["join"]["judged_basis_applicable"] is False
    assert r["join"]["matched_any_row"] == 1 and r["join"]["excluded_rows_all"] == 1
    assert r["join"]["matched_judged"] is None
    assert r["F1_full"] == pytest.approx(50.0) and r["F1_excluded"] == pytest.approx(100.0)


def test_unjudged_run_still_reports_its_duplicate_key_count(tmp_path):
    # sourcing duplicates from the judged basis alone silently discards a real,
    # measured all-rows duplicate count for F1-only runs -- the one number that
    # explains why their F1 denominator drops by more than the key count.
    x1 = _rescore()
    rows = [
        _rec(0, "dup?", "single-hop", 1.0),
        _rec(0, "dup?", "single-hop", 0.0),
        _rec(0, "keep?", "temporal", 0.5),
    ]
    r = x1.rescore_file(_write_records(tmp_path, rows, stem=WUJIANG_STEM), {(0, "dup?")})
    assert r["judged"] is False
    assert r["join"]["duplicate_matched_keys_any_row"] == 1
    assert r["join"]["excluded_rows_all"] == 2  # one key, two rows
    assert r["join"]["duplicate_matched_keys_judged"] is None
    assert r["n_full"] == 3 and r["n_excluded"] == 1


def test_judged_run_reports_duplicate_key_counts_on_both_bases(tmp_path):
    x1 = _rescore()
    rows = [
        _rec(0, "dup?", "single-hop", 1.0, True),
        _rec(0, "dup?", "single-hop", 0.0, False),
        _rec(0, "keep?", "temporal", 0.5, True),
    ]
    r = x1.rescore_file(_write_records(tmp_path, rows), {(0, "dup?")})
    assert r["join"]["duplicate_matched_keys_any_row"] == 1
    assert r["join"]["duplicate_matched_keys_judged"] == 1


def test_rescore_file_refuses_an_anchored_run_that_carries_no_verdicts(tmp_path):
    # defense in depth: an anchor on an F1-only stem passes the loader (verdict
    # -free is legitimate there) but still has no J to compare. That must say so,
    # not fail obscurely on a None.
    x1 = _rescore()
    path = _write_records(tmp_path, [_rec(0, "a?", "single-hop", 1.0)], stem=WUJIANG_STEM)
    x1.HEADLINE_ANCHORS[WUJIANG_STEM] = 50.0
    try:
        with pytest.raises(ValueError, match="no judge verdicts"):
            x1.rescore_file(path, set())
    finally:
        del x1.HEADLINE_ANCHORS[WUJIANG_STEM]


def test_anchor_check_uses_the_same_rounding_the_report_prints(tmp_path):
    # the PASS/FAIL and the displayed number must come from one value, or a
    # report can show a figure that disagrees with its own verdict.
    x1 = _rescore()
    r = x1.rescore_file(_write_records(tmp_path, _six_rows()), set())
    assert r["J_full_2dp"] == round(r["J_full"], 2)


def test_rescore_file_flags_a_stem_with_no_anchor_as_not_headline(tmp_path):
    x1 = _rescore()
    r = x1.rescore_file(_write_records(tmp_path, _six_rows()), set())
    assert r["headline"] is False
    assert r["anchor_J"] is None and r["anchor_ok"] is None


# --- main(): the CLI and its one hard STOP --------------------------------


def _synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """A dataset and an errors catalogue small enough to reason about.

    main() resolves error ids against the dataset, so the two must agree; the
    records files are joined separately and need not contain these questions.
    """
    dataset = tmp_path / "locomo10.json"
    dataset.write_text(
        json.dumps([{"qa": [{"question": "Q zero?", "answer": "a", "category": 1}]}])
    )
    errors = tmp_path / "errors.json"
    errors.write_text(
        json.dumps(
            [{"question_id": "locomo_0_qa0", "question": "Q zero?", "error_type": "HALLUCINATION"}]
        )
    )
    return dataset, errors


def _run_main(x1, tmp_path: Path, records_dir: Path, out_dir: Path):
    dataset, errors = _synthetic_inputs(tmp_path)
    return x1.main(
        [
            "--records-glob",
            str(records_dir / "*.records.jsonl"),
            "--out",
            str(out_dir),
            "--dataset",
            str(dataset),
            "--errors",
            str(errors),
        ]
    )


def test_main_stops_without_writing_when_a_headline_anchor_disagrees(tmp_path):
    # THE hard requirement of this task. The gate must be fail-closed: nothing
    # may reach disk when a headline J does not reproduce. A refactor that moves
    # the write above the gate has to break this test, not ship quietly.
    x1 = _rescore()
    records_dir = tmp_path / "recs"
    records_dir.mkdir()
    stem = "gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA"  # anchor 67.60
    _write_records(records_dir, [_rec(0, "a?", "single-hop", 0.0, False)], stem=stem)

    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit):
        _run_main(x1, tmp_path, records_dir, out_dir)
    assert not out_dir.exists()  # fail-closed: no artifact, not even an empty dir


def test_main_writes_both_artifacts_when_no_anchor_is_violated(tmp_path):
    x1 = _rescore()
    records_dir = tmp_path / "recs"
    records_dir.mkdir()
    _write_records(records_dir, _six_rows(), stem="some_ablation_ours_run1")

    out_dir = tmp_path / "out"
    assert _run_main(x1, tmp_path, records_dir, out_dir) == 0
    payload = json.loads((out_dir / "rescore.json").read_text())
    assert len(payload["runs"]) == 1
    assert payload["runs"][0]["headline"] is False
    assert payload["meta"]["errors_score_corrupting"] == 1
    assert (out_dir / "rescore.md").read_text().startswith("# X1 gold-error replay")


def test_main_refuses_a_glob_that_matches_nothing(tmp_path):
    x1 = _rescore()
    empty = tmp_path / "recs"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no records files"):
        _run_main(x1, tmp_path, empty, tmp_path / "out")


def test_markdown_omits_the_f1_only_note_when_every_run_is_judged(tmp_path):
    # the wujiang paragraph explains an `n/a` that is not on the page otherwise.
    x1 = _rescore()
    judged = x1.rescore_file(_write_records(tmp_path, _six_rows()), set())
    meta = {
        "errors_path": "e",
        "errors_total": 1,
        "errors_score_corrupting": 1,
        "dataset_path": "d",
        "records_glob": "g",
    }
    assert "wujiang" not in x1.render_markdown([judged], meta)
    unjudged = x1.rescore_file(
        _write_records(tmp_path, [_rec(0, "z?", "single-hop", 1.0)], stem=WUJIANG_STEM), set()
    )
    assert "wujiang" in x1.render_markdown([judged, unjudged], meta)


# --- the anchors ----------------------------------------------------------


def test_headline_anchors_are_the_four_published_runs():
    x1 = _rescore()
    assert x1.HEADLINE_ANCHORS == {
        "gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA": 67.60,
        "gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB": 65.78,
        "gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH": 61.23,
        "gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM": 31.82,
    }


# --- real-data pins -------------------------------------------------------


@pytest.mark.parametrize(
    "stem,anchor",
    [
        ("gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA", 67.60),
        ("gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB", 65.78),
        ("gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH", 61.23),
        ("gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM", 31.82),
    ],
)
def test_headline_full_J_reproduces_the_controller_measured_value(stem, anchor):
    path = REPRO / f"{stem}.records.jsonl"
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    x1 = _rescore()
    agg = x1.aggregate(x1.load_records(path))
    assert agg["j_n"] == 1540
    assert round(agg["J"], 2) == anchor
