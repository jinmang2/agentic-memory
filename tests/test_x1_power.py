"""Discriminative-power tests: what the four-arm ranking can and cannot claim.

Three separate questions live in this module and the tests keep them apart.

``paired_delta_ci`` asks how much of the gap survives *question sampling* -- we
graded 1,540 questions, and another 1,540 drawn the same way would not give the
same J. Pairing is the whole point: the arms answered the *same* questions, so
the per-question difference vector carries far less noise than two independent
runs would, and a test below pins that two arms with identical marginals but
different agreement patterns get different intervals.

``rank_flip_prob`` asks a narrower question: the 99 audited questions have gold
we do not trust, so what if they had been graded fairly? Only those 99 vary;
the other 1,441 verdicts are held fixed. It is deliberately *not* a total
uncertainty estimate, and the test that an all-False mask makes the simulation
degenerate is what keeps that honest.

``seeds_needed`` asks a third thing entirely -- how many seed replicates a gap
of a given size needs before it clears run-to-run jitter -- and is closed-form,
so it is pinned against hand arithmetic rather than against a simulation.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_POWER_PATH = _ROOT / "scripts" / "ext" / "x1_power.py"

REPRO = _ROOT / "results/repro"


def _load_module(path: Path, name: str):
    """Import a scripts/ext module by path -- scripts/ holds no packages, so
    this mirrors tests/test_x1_rescore.py's flat-import pattern."""
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _power():
    return _load_module(_POWER_PATH, "x1_power")


# --- seeds_needed: closed form, pinned against hand arithmetic -------------


def test_seeds_needed_closed_form():
    # d=0.35pp (= the seed sd) needs n = (1.96+0.842)^2 * 2 ~= 15.7 -> 16
    x1 = _power()
    assert x1.seeds_needed(0.35) == 16
    assert x1.seeds_needed(3.5) == 1  # a large effect needs a single seed


def test_seeds_needed_is_never_below_one():
    # even an enormous gap still has to be measured once.
    x1 = _power()
    assert x1.seeds_needed(1000.0) == 1
    assert x1.seeds_needed(1.0, seed_sd_pp=0.0) == 1


def test_seeds_needed_grows_with_jitter_and_shrinks_with_effect():
    x1 = _power()
    assert x1.seeds_needed(1.0, seed_sd_pp=0.7) > x1.seeds_needed(1.0, seed_sd_pp=0.35)
    assert x1.seeds_needed(0.5) > x1.seeds_needed(1.0)


def test_seeds_needed_grows_with_stricter_alpha_and_higher_power():
    x1 = _power()
    base = x1.seeds_needed(0.35)
    assert x1.seeds_needed(0.35, alpha=0.01) > base
    assert x1.seeds_needed(0.35, power=0.95) > base


def test_seeds_needed_rejects_a_nonpositive_gap():
    # a gap of zero needs infinitely many seeds; returning some integer for it
    # would put a finite, false answer in the table.
    x1 = _power()
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="delta_pp"):
            x1.seeds_needed(bad)


def test_seeds_needed_rejects_nonsense_alpha_power_and_sd():
    x1 = _power()
    with pytest.raises(ValueError, match="alpha"):
        x1.seeds_needed(1.0, alpha=0.0)
    with pytest.raises(ValueError, match="power"):
        x1.seeds_needed(1.0, power=1.0)
    with pytest.raises(ValueError, match="seed_sd_pp"):
        x1.seeds_needed(1.0, seed_sd_pp=-0.1)


def test_min_detectable_delta_is_the_inverse_of_seeds_needed():
    # the band statement in the report is read off this function, so it must
    # agree with the seed count it claims to invert.
    x1 = _power()
    for n in (1, 2, 4, 16):
        d = x1.min_detectable_delta_pp(n)
        assert x1.seeds_needed(d * 1.0001) <= n
        assert x1.seeds_needed(d * 0.9999) > n - 1
    assert x1.min_detectable_delta_pp(16) == pytest.approx(0.35, rel=2e-2)


def test_questions_needed_is_the_seed_formula_on_the_question_axis():
    # same alpha, power and normal approximation as seeds_needed, so the two
    # tables in the report are read on one scale. Hand check: a gap exactly at
    # (z_a/2 + z_b) * SE is already powered at the size it was measured on.
    x1 = _power()
    se, n = 1.0, 1000
    at_threshold = (1.959963984540054 + 0.8416212335729143) * se
    assert x1.questions_needed(at_threshold, se, n) == n
    assert x1.questions_needed(at_threshold / 2, se, n) == 4 * n  # halve d -> 4x n
    assert x1.questions_needed(at_threshold, se * 2, n) == 4 * n  # double SE -> 4x n


def test_questions_needed_rejects_inputs_that_have_no_answer():
    x1 = _power()
    with pytest.raises(ValueError, match="delta_pp"):
        x1.questions_needed(0.0, 1.0, 100)
    with pytest.raises(ValueError, match="se_pp"):
        x1.questions_needed(1.0, 0.0, 100)
    with pytest.raises(ValueError, match="n must be"):
        x1.questions_needed(1.0, 1.0, 0)


# --- paired_delta_ci ------------------------------------------------------


def test_paired_ci_is_deterministic_and_covers_zero_for_identical_arms():
    x1 = _power()
    a = [True, False] * 100
    ci = x1.paired_delta_ci(a, a)
    assert ci["lo"] == ci["hi"] == 0.0
    assert x1.paired_delta_ci(a, list(reversed(a)))["lo"] is not None  # smoke


def test_paired_delta_ci_recovers_a_constant_difference_exactly():
    # every question differs the same way, so no resample can disagree.
    x1 = _power()
    ci = x1.paired_delta_ci([True] * 200, [False] * 200)
    assert ci["delta_pp"] == pytest.approx(100.0)
    assert ci["lo"] == ci["hi"] == 100.0


def test_paired_delta_ci_is_paired_not_unpaired():
    # THE property of this function. Both calls below have identical marginals
    # (100 true, 100 false on each side) and identical point estimates of 0.
    # Only the agreement pattern differs: an unpaired interval would be the same
    # width for both, a paired one is zero-width when the arms never disagree.
    x1 = _power()
    same = [True] * 100 + [False] * 100
    agree = x1.paired_delta_ci(same, same)
    disagree = x1.paired_delta_ci(same, [False] * 100 + [True] * 100)
    assert agree["hi"] - agree["lo"] == 0.0
    assert disagree["hi"] - disagree["lo"] > 10.0
    assert agree["delta_pp"] == disagree["delta_pp"] == 0.0


def test_paired_delta_ci_brackets_its_own_point_estimate():
    x1 = _power()
    a = [True] * 140 + [False] * 60
    b = [True] * 100 + [False] * 100
    ci = x1.paired_delta_ci(a, b)
    assert ci["lo"] <= ci["delta_pp"] <= ci["hi"]
    assert ci["delta_pp"] == pytest.approx(20.0)


def test_paired_delta_ci_same_seed_reproduces_and_a_new_seed_does_not():
    x1 = _power()
    a = [True] * 140 + [False] * 60
    b = [True] * 100 + [False] * 100
    assert x1.paired_delta_ci(a, b, seed=7) == x1.paired_delta_ci(a, b, seed=7)
    assert x1.paired_delta_ci(a, b, seed=7) != x1.paired_delta_ci(a, b, seed=8)


def test_paired_delta_ci_excludes_zero_for_a_decisive_gap():
    x1 = _power()
    ci = x1.paired_delta_ci([True] * 200, [True] * 100 + [False] * 100)
    assert ci["excludes_zero"] is True
    assert ci["lo"] > 0.0
    assert ci["p_boot"] < 0.05


def test_paired_delta_ci_covers_zero_when_the_arms_merely_disagree():
    x1 = _power()
    a = [True, False] * 100
    ci = x1.paired_delta_ci(a, list(reversed(a)))
    assert ci["lo"] < 0.0 < ci["hi"]
    assert ci["excludes_zero"] is False


def test_paired_delta_ci_se_matches_the_closed_form_paired_se():
    # an independent check on the resampling itself: for a difference of means
    # the bootstrap SE has a closed form, sd(d)/sqrt(n) over the per-question
    # difference vector (the McNemar-style paired SE). If the resampling were
    # unpaired, or the block-chunked index draw were wrong, this would drift.
    x1 = _power()
    a = [True] * 150 + [False] * 50 + [True] * 100 + [False] * 100
    b = [True] * 150 + [False] * 50 + [False] * 100 + [True] * 100
    ci = x1.paired_delta_ci(a, b, n_boot=20_000, seed=0)
    d = [int(x) - int(y) for x, y in zip(a, b)]
    mean = sum(d) / len(d)
    var = sum((v - mean) ** 2 for v in d) / len(d)
    analytic_se_pp = 100.0 * (var / len(d)) ** 0.5
    assert ci["se_pp"] == pytest.approx(analytic_se_pp, rel=0.03)


def test_paired_delta_ci_rejects_misaligned_or_empty_input():
    x1 = _power()
    with pytest.raises(ValueError, match="same length"):
        x1.paired_delta_ci([True, False], [True])
    with pytest.raises(ValueError, match="empty"):
        x1.paired_delta_ci([], [])


def test_paired_delta_ci_reports_the_inputs_it_was_run_with():
    # a CI in an artifact is unreadable without its resample count and seed.
    x1 = _power()
    ci = x1.paired_delta_ci([True] * 10, [False] * 10, n_boot=500, seed=3)
    assert ci["n"] == 10 and ci["n_boot"] == 500 and ci["seed"] == 3


# --- rank_flip_prob -------------------------------------------------------


def _arms_for_flip():
    """Four arms over 100 questions, 20 of them flagged, clearly ranked."""

    def arm(k: int) -> list[bool]:
        return [True] * k + [False] * (100 - k)

    return {"a": arm(90), "b": arm(70), "c": arm(50), "d": arm(20)}


def _mask(n_flagged: int, n: int = 100) -> list[bool]:
    return [True] * n_flagged + [False] * (n - n_flagged)


def test_rank_flip_prob_is_deterministic_under_a_fixed_seed():
    x1 = _power()
    arms = _arms_for_flip()
    assert x1.rank_flip_prob(arms, _mask(20), n_sim=500, seed=1) == x1.rank_flip_prob(
        arms, _mask(20), n_sim=500, seed=1
    )
    assert x1.rank_flip_prob(arms, _mask(20), n_sim=500, seed=1) != x1.rank_flip_prob(
        arms, _mask(20), n_sim=500, seed=2
    )


def test_rank_flip_prob_is_degenerate_when_nothing_is_flagged():
    # the model only perturbs flagged questions. With none flagged there is
    # nothing to resample, and the simulation must say so rather than
    # manufacture uncertainty out of the 1,441 verdicts it holds fixed.
    x1 = _power()
    out = x1.rank_flip_prob(_arms_for_flip(), _mask(0), n_sim=300, seed=0)
    assert out["p_observed_order"] == 1.0
    assert all(p["p_flip"] == 0.0 for p in out["pairs"])
    assert out["n_flagged"] == 0


def test_rank_flip_prob_keeps_a_wide_ranking_intact():
    x1 = _power()
    out = x1.rank_flip_prob(_arms_for_flip(), _mask(20), n_sim=2000, seed=0)
    assert out["observed_order"] == ["a", "b", "c", "d"]
    assert out["p_observed_order"] > 0.95


def test_rank_flip_prob_flips_a_tied_pair_about_half_the_time():
    # two arms that differ only on the flagged block, and only barely: once the
    # flagged verdicts are redrawn from the same rate they are a coin flip.
    x1 = _power()
    arms = {
        "x": [True] * 30 + [False] * 30 + [True] * 20 + [False] * 20,
        "y": [False] * 30 + [True] * 30 + [True] * 20 + [False] * 20,
    }
    out = x1.rank_flip_prob(arms, [True] * 60 + [False] * 40, n_sim=4000, seed=0)
    pair = out["pairs"][0]
    assert 0.25 < pair["p_flip"] + pair["p_tie"] / 2 < 0.75
    assert pair["p_tie"] > 0.0  # equal J is possible and must be counted, not hidden


def test_rank_flip_prob_probabilities_form_a_distribution():
    x1 = _power()
    out = x1.rank_flip_prob(_arms_for_flip(), _mask(40), n_sim=1000, seed=0)
    assert sum(p["prob"] for p in out["permutations"]) == pytest.approx(1.0)
    assert all(p["prob"] > 0 for p in out["permutations"])
    assert out["p_observed_order"] == pytest.approx(
        next(p["prob"] for p in out["permutations"] if p["order"] == out["observed_order"])
    )


def test_rank_flip_prob_rate_basis_changes_which_rate_is_drawn():
    # "unflagged" treats the arm's score on trusted gold as its ability; the
    # "flagged" basis redraws at the rate actually observed on the bad gold.
    # They are different models and must not silently be the same number.
    x1 = _power()
    arms = {"a": [False] * 20 + [True] * 80, "b": [False] * 20 + [True] * 40 + [False] * 40}
    mask = _mask(20)
    unflagged = x1.rank_flip_prob(arms, mask, n_sim=500, seed=0, rate_basis="unflagged")
    flagged = x1.rank_flip_prob(arms, mask, n_sim=500, seed=0, rate_basis="flagged")
    assert unflagged["arms"]["a"]["p_resample"] == pytest.approx(1.0)
    assert flagged["arms"]["a"]["p_resample"] == pytest.approx(0.0)
    assert unflagged["rate_basis"] == "unflagged" and flagged["rate_basis"] == "flagged"


def test_rank_flip_prob_rejects_an_unknown_rate_basis():
    x1 = _power()
    with pytest.raises(ValueError, match="rate_basis"):
        x1.rank_flip_prob(_arms_for_flip(), _mask(20), n_sim=10, rate_basis="vibes")


def test_rank_flip_prob_rejects_a_mask_that_does_not_fit_the_arms():
    x1 = _power()
    with pytest.raises(ValueError, match="length"):
        x1.rank_flip_prob(_arms_for_flip(), _mask(5, n=50), n_sim=10)


def test_rank_flip_prob_needs_at_least_two_arms():
    x1 = _power()
    with pytest.raises(ValueError, match="two arms"):
        x1.rank_flip_prob({"only": [True] * 10}, _mask(2, n=10), n_sim=10)


def test_rank_flip_prob_reports_the_observed_J_it_started_from():
    x1 = _power()
    out = x1.rank_flip_prob(_arms_for_flip(), _mask(20), n_sim=200, seed=0)
    assert out["arms"]["a"]["J_full"] == pytest.approx(90.0)
    assert out["arms"]["d"]["J_full"] == pytest.approx(20.0)


# --- alignment: the arms must be the same questions in the same order -----


def _rec(conv: int, q: str, cat: str = "single-hop", f1: float = 1.0, j: bool | None = None):
    row = {"conv": conv, "q": q, "cat": cat, "f1": f1}
    if j is not None:
        row["j"] = j
    return row


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_aligned_arms_returns_one_key_sequence_and_a_vector_per_arm(tmp_path):
    x1 = _power()
    rows_a = [_rec(0, "a?", j=True), _rec(0, "adv?", "adversarial"), _rec(0, "b?", j=False)]
    rows_b = [_rec(0, "a?", j=False), _rec(0, "adv?", "adversarial"), _rec(0, "b?", j=False)]
    paths = {
        "A": _write(tmp_path / "A.records.jsonl", rows_a),
        "B": _write(tmp_path / "B.records.jsonl", rows_b),
    }
    keys, arms = x1.aligned_arms(paths)
    assert keys == [(0, "a?"), (0, "b?")]  # adversarial rows carry no verdict
    assert arms == {"A": [True, False], "B": [False, False]}


def test_aligned_arms_rejects_arms_whose_question_order_differs(tmp_path):
    # positional pairing is the whole basis of the paired CI; two files in
    # different orders would silently compare question i to question j.
    x1 = _power()
    paths = {
        "A": _write(tmp_path / "A.records.jsonl", [_rec(0, "a?", j=True), _rec(0, "b?", j=False)]),
        "B": _write(tmp_path / "B.records.jsonl", [_rec(0, "b?", j=True), _rec(0, "a?", j=False)]),
    }
    with pytest.raises(ValueError, match="order"):
        x1.aligned_arms(paths)


def test_aligned_arms_pairs_duplicated_questions_by_file_order(tmp_path):
    # conv 7 ships 11 questions twice. Both copies must survive as two aligned
    # positions -- deduplicating would drop rows out of the J denominator.
    x1 = _power()
    rows = [_rec(7, "dup?", j=True), _rec(7, "dup?", j=False)]
    paths = {"A": _write(tmp_path / "A.records.jsonl", rows)}
    keys, arms = x1.aligned_arms(paths)
    assert keys == [(7, "dup?"), (7, "dup?")]
    assert arms["A"] == [True, False]


def test_error_mask_marks_every_row_sharing_a_flagged_key():
    x1 = _power()
    keys = [(7, "dup?"), (7, "dup?"), (0, "keep?")]
    assert x1.error_mask(keys, {(7, "dup?")}) == [True, True, False]


def test_check_canonical_order_accepts_the_dataset_enumeration(tmp_path):
    x1 = _power()
    dataset = tmp_path / "d.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "qa": [
                        {"question": "Q one?", "category": 1},
                        {"question": "Q adv?", "category": 5},
                        {"question": "Q two?", "category": 2},
                    ]
                }
            ]
        )
    )
    assert x1.check_canonical_order([(0, "q one?"), (0, "q two?")], dataset) is True
    with pytest.raises(ValueError, match="canonical"):
        x1.check_canonical_order([(0, "q two?"), (0, "q one?")], dataset)


# --- main(): artifacts, determinism, and the anchor gate ------------------


HEADLINE_STEMS = [
    "gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA",
    "gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB",
    "gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH",
    "gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM",
]


def _synthetic_world(x1, tmp_path: Path) -> dict:
    """Four arms over 4 judged questions plus an adversarial one, one flagged.

    Anchors are rebound to the J these rows actually produce: the gate under
    test is "does the recomputed J match its anchor", not the anchor's value.
    """
    recs = tmp_path / "recs"
    recs.mkdir()
    correct = {"e3sA": 4, "e3sB": 3, "e3sPH": 2, "e3sM": 1}
    for stem in HEADLINE_STEMS:
        arm = stem.rsplit("_", 1)[1]
        k = correct[arm]
        rows = [_rec(0, f"q{i}?", "single-hop", 1.0, i < k) for i in range(4)]
        rows.append(_rec(0, "adv?", "adversarial", 0.5))
        _write(recs / f"{stem}.records.jsonl", rows)
        x1.HEADLINE_ANCHORS[stem] = round(100.0 * k / 4, 2)

    dataset = tmp_path / "locomo10.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "qa": [{"question": f"q{i}?", "category": 1} for i in range(4)]
                    + [{"question": "adv?", "category": 5}]
                }
            ]
        )
    )
    errors = tmp_path / "errors.json"
    errors.write_text(
        json.dumps(
            [{"question_id": "locomo_0_qa3", "question": "q3?", "error_type": "HALLUCINATION"}]
        )
    )
    return {"recs": recs, "dataset": dataset, "errors": errors}


@pytest.fixture
def world(tmp_path):
    """A synthetic run world, with HEADLINE_ANCHORS restored afterwards."""
    x1 = _power()
    saved = dict(x1.HEADLINE_ANCHORS)
    try:
        yield x1, _synthetic_world(x1, tmp_path)
    finally:
        x1.HEADLINE_ANCHORS.clear()
        x1.HEADLINE_ANCHORS.update(saved)


def _argv(w: dict, out: Path, extra: list[str] | None = None) -> list[str]:
    return [
        "--records-dir",
        str(w["recs"]),
        "--dataset",
        str(w["dataset"]),
        "--errors",
        str(w["errors"]),
        "--out",
        str(out),
        "--n-boot",
        "200",
        "--n-sim",
        "200",
    ] + (extra or [])


def test_main_writes_both_artifacts(world, tmp_path):
    x1, w = world
    out = tmp_path / "out"
    assert x1.main(_argv(w, out)) == 0
    payload = json.loads((out / "power.json").read_text())
    assert set(payload["arms"]) == {"e3sA", "e3sB", "e3sPH", "e3sM"}
    assert payload["meta"]["n_flagged"] == 1
    assert (out / "power.md").read_text().startswith("# X1 discriminative power")


def test_main_output_is_byte_identical_across_runs(world, tmp_path):
    # the determinism claim in the report is checked here, not by eye.
    x1, w = world
    first, second = tmp_path / "o1", tmp_path / "o2"
    assert x1.main(_argv(w, first)) == 0
    assert x1.main(_argv(w, second)) == 0
    assert (first / "power.json").read_bytes() == (second / "power.json").read_bytes()
    assert (first / "power.md").read_bytes() == (second / "power.md").read_bytes()


def test_main_stops_without_writing_when_a_headline_anchor_disagrees(world, tmp_path):
    x1, w = world
    x1.HEADLINE_ANCHORS[HEADLINE_STEMS[0]] = 1.0  # no longer what the rows score
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="anchor"):
        x1.main(_argv(w, out))
    assert not out.exists()  # fail-closed: nothing on disk, not even an empty dir


def test_main_reports_the_four_things_the_task_asks_for(world, tmp_path):
    x1, w = world
    out = tmp_path / "out"
    assert x1.main(_argv(w, out)) == 0
    payload = json.loads((out / "power.json").read_text())
    assert [p["pair"] for p in payload["paired_delta_ci"]["full_gold"]] == [
        ["e3sA", "e3sB"],
        ["e3sB", "e3sPH"],
        ["e3sPH", "e3sM"],
    ]
    assert payload["rank_stability"]["unflagged"]["permutations"]
    assert all("seeds_full" in s for s in payload["seeds_needed"])
    assert payload["conclusion"].strip().endswith(".")


def test_main_refuses_a_records_dir_missing_an_arm(world, tmp_path):
    x1, w = world
    (w["recs"] / f"{HEADLINE_STEMS[3]}.records.jsonl").unlink()
    with pytest.raises(SystemExit, match="missing"):
        x1.main(_argv(w, tmp_path / "out"))


# --- the conclusion must follow the evidence, not the template ------------


def _mixed_report(x1):
    """Arms where the top pair is a near-tie and the lower two gaps are decisive."""
    arms = {
        "e3sA": [True] * 300 + [False] * 100,
        "e3sB": [True] * 299 + [False] * 101,  # differs from e3sA on one question
        "e3sPH": [True] * 200 + [False] * 200,
        "e3sM": [True] * 40 + [False] * 360,
    }
    mask = [True] * 40 + [False] * 360
    return x1.build_report(arms, mask, n_boot=2000, n_sim=2000, seed=0)


def test_a_pair_the_bootstrap_cannot_separate_is_not_claimed_as_a_win():
    # THE honesty property of the report. One flipped question out of 400 is a
    # gap, and it clears both the seed band and the gold-noise simulation -- so
    # a conclusion assembled around those two would call it a win. It must not:
    # the paired CI covers zero and that is what decides.
    x1 = _power()
    report = _mixed_report(x1)
    rows = {tuple(s["pair"]): s for s in report["seeds_needed"]}
    assert rows[("e3sA", "e3sB")]["separable_full"] is False
    assert rows[("e3sB", "e3sPH")]["separable_full"] is True
    assert rows[("e3sPH", "e3sM")]["separable_full"] is True

    conclusion = report["conclusion"]
    assert "Only part of the ranking" in conclusion
    assert "e3sA over e3sB" in conclusion
    # and it must point at the lever that would actually help
    assert "questions" in conclusion
    assert f"{rows[('e3sA', 'e3sB')]['questions_needed_full']:,}" in conclusion


def test_the_conclusion_claims_the_whole_ranking_when_every_gap_separates():
    # the mirror: the wording must be able to say "claimable" too, or the test
    # above would only be pinning a constant string.
    x1 = _power()
    arms = {
        "e3sA": [True] * 380 + [False] * 20,
        "e3sB": [True] * 300 + [False] * 100,
        "e3sPH": [True] * 200 + [False] * 200,
        "e3sM": [True] * 40 + [False] * 360,
    }
    report = x1.build_report(arms, [True] * 40 + [False] * 360, n_boot=2000, n_sim=2000, seed=0)
    assert "The full ranking" in report["conclusion"]
    assert "Only part" not in report["conclusion"]


def test_the_summary_table_marks_the_unseparated_pair(tmp_path):
    x1 = _power()
    report = _mixed_report(x1)
    report["meta"] = {"n_judged": 400, "n_flagged": 40, "seed": 0, "n_boot": 2000, "n_sim": 2000}
    md = x1.render_markdown(report, report["meta"])
    assert "**NOT separated**" in md
    assert "**separated**" in md


# --- real-data pins -------------------------------------------------------


def _real_paths() -> dict[str, Path] | None:
    paths = {s.rsplit("_", 1)[1]: REPRO / f"{s}.records.jsonl" for s in HEADLINE_STEMS}
    return None if any(not p.exists() for p in paths.values()) else paths


def test_real_arms_are_the_same_1540_questions_in_the_same_order():
    paths = _real_paths()
    if paths is None:
        pytest.skip("headline records files not present")
    x1 = _power()
    keys, arms = x1.aligned_arms(paths)
    assert len(keys) == 1540
    assert all(len(v) == 1540 for v in arms.values())
    assert x1.check_canonical_order(keys, x1.DEFAULT_DATASET) is True


def test_real_error_mask_lands_on_exactly_99_rows():
    paths = _real_paths()
    if paths is None or not x1_inputs_exist():
        pytest.skip("headline records or audit inputs not present")
    x1 = _power()
    keys, _ = x1.aligned_arms(paths)
    assert sum(x1.error_mask(keys, x1.flagged_keys(x1.DEFAULT_ERRORS, x1.DEFAULT_DATASET))) == 99


def x1_inputs_exist() -> bool:
    x1 = _power()
    return x1.DEFAULT_ERRORS.exists() and x1.DEFAULT_DATASET.exists()


def test_real_headline_J_reproduces_the_published_values():
    paths = _real_paths()
    if paths is None:
        pytest.skip("headline records files not present")
    x1 = _power()
    _, arms = x1.aligned_arms(paths)
    expected = {"e3sA": 67.60, "e3sB": 65.78, "e3sPH": 61.23, "e3sM": 31.82}
    for name, verdicts in arms.items():
        assert round(100.0 * sum(verdicts) / len(verdicts), 2) == expected[name]
