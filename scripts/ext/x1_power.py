"""How much of the four-arm LoCoMo ranking survives the noise under it.

Task 2 established what the arms score with and without the 99 audited-bad gold
answers. It did not establish whether the gaps between them are large enough to
claim at all, which is what this module measures. Three different noise sources,
three deliberately separate estimates -- conflating them is how a 1.8pp gap gets
reported as a result.

**Question sampling** (`paired_delta_ci`). We graded 1,540 questions; another
1,540 drawn the same way would not give the same J. A paired percentile
bootstrap over the per-question difference vector answers how much of a gap that
moves. Paired, because the arms answered the *same* questions in the *same*
order: an unpaired interval would throw away the agreement structure and roughly
double the width for no reason. The resampling unit is the question, so the
1,540 differences are drawn with replacement `n_boot` times and the 2.5/97.5
percentiles of the resulting delta-J distribution are reported. No BCa
correction: the statistic is a difference of means over 1,540 paired points, the
bootstrap distribution is symmetric to well inside the reported precision, and
an acceleration term would be an unverifiable flourish here.

**Gold noise** (`rank_flip_prob`). The audit says 99 of those questions have gold
we cannot trust, which means we do not know what a fair grader would have said
about them. The model: hold the 1,441 undisputed verdicts fixed and redraw each
flagged verdict as Bernoulli(p_arm). Because J depends on the flagged block only
through its total, the 99 draws are generated as one Binomial(99, p_arm) -- the
same distribution, not a different model. Arms are drawn independently, which is
the conservative choice: real per-question difficulty is positively correlated
across arms, and positive correlation *shrinks* the variance of the gap, so
independent draws give the ranking more chances to flip than reality would.

Two readings of "the arm's rate" are defensible and the artifact reports both.
`rate_basis="unflagged"` draws at the arm's accuracy on trustworthy gold, i.e.
treats the flagged questions as if they had been graded as fairly as the rest;
`rate_basis="flagged"` draws at the rate actually observed on the bad gold,
i.e. asks only about resampling jitter around the status quo. Note what this
model does *not* cover: the 1,441 fixed verdicts. It is a gold-noise estimate,
never a total-uncertainty one -- that is what the bootstrap above is for.

**Run-to-run jitter** (`seeds_needed`). A Track-1 seed replication put the
replicate SD at about 0.35pp. The standard two-sample normal approximation
`n = ((z_{alpha/2} + z_beta) * sd * sqrt(2) / delta)^2`, rounded up, says how
many seeds a gap of a given size needs before it clears that jitter. Closed
form, no simulation, so it is pinned against hand arithmetic in the tests.

Alignment is not assumed. All four arms come from one harness, so their judged
rows should be the same questions in the same order -- `aligned_arms` raises if
they are not, and `check_canonical_order` further pins that sequence to the
dataset enumeration Task 1 established as authoritative (0-based over the full
qa list, non-adversarial rows only). Positional pairing is the basis of every
paired number here; conv 7's eleven duplicated questions stay as two positions
each, matched by order of appearance, because deduplicating them would drop rows
out of the J denominator.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np

# Flat modules, no package: a sibling import has to work both under `python
# scripts/ext/x1_power.py` (which already puts this dir first) and under the
# tests' importlib loading (which does not).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from x1_audit_data import load_errors, score_corrupting
from x1_join import error_question_keys, normalize_q, question_key_map
from x1_rescore import HEADLINE_ANCHORS, load_records, run_stem

DEFAULT_DATASET = Path.home() / ".agmem/datasets/locomo10.json"
DEFAULT_ERRORS = Path.home() / ".agmem/upstream/locomo-audit/errors.json"
DEFAULT_RECORDS_DIR = Path("results/repro")

# The four headline arms, best first. The short names are the run-stem suffixes.
ARM_STEMS = {
    "e3sA": "gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA",
    "e3sB": "gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB",
    "e3sPH": "gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH",
    "e3sM": "gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM",
}
ADJACENT_PAIRS = [("e3sA", "e3sB"), ("e3sB", "e3sPH"), ("e3sPH", "e3sM")]

# Replicate SD from the Track-1 seed-2 replication. One number, one place.
SEED_SD_PP = 0.35

# LoCoMo category 5 is adversarial; those rows are scored by F1 and never judged.
ADVERSARIAL_CATEGORY = 5

RATE_BASES = ("unflagged", "flagged")

# Bootstrap resample indices are drawn in fixed-size blocks so a 10k x 1540
# index matrix never has to exist at once. The size is a constant, not a
# parameter, because changing it changes the numbers a given seed produces.
_BOOT_BLOCK = 1000


# --- closed-form seed arithmetic ------------------------------------------


def _z(p: float) -> float:
    """Standard-normal quantile. stdlib, so numpy stays the only dependency."""
    return statistics.NormalDist().inv_cdf(p)


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _check_alpha_power(alpha: float, power: float) -> None:
    _check_alpha(alpha)
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")


def seeds_needed(
    delta_pp: float, seed_sd_pp: float = SEED_SD_PP, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Seed replicates per arm needed to call a gap of ``delta_pp`` percentage points.

    Two-sample normal approximation, ``n = ((z_{alpha/2} + z_beta) * sd * sqrt(2) / delta)^2``
    rounded up: both arms are replicated, hence the ``sqrt(2)``. A gap of zero
    needs infinitely many seeds, so it raises rather than returning a finite
    number that would read as an answer.
    """
    if not delta_pp > 0:
        raise ValueError(f"delta_pp must be positive, got {delta_pp}")
    if seed_sd_pp < 0:
        raise ValueError(f"seed_sd_pp must be non-negative, got {seed_sd_pp}")
    _check_alpha_power(alpha, power)
    n = ((_z(1 - alpha / 2) + _z(power)) * seed_sd_pp * math.sqrt(2) / delta_pp) ** 2
    return max(1, math.ceil(n))


def min_detectable_delta_pp(
    n_seeds: int = 1, seed_sd_pp: float = SEED_SD_PP, alpha: float = 0.05, power: float = 0.8
) -> float:
    """The smallest gap ``n_seeds`` replicates can resolve -- the inverse of ``seeds_needed``."""
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be at least 1, got {n_seeds}")
    if seed_sd_pp < 0:
        raise ValueError(f"seed_sd_pp must be non-negative, got {seed_sd_pp}")
    _check_alpha_power(alpha, power)
    return (_z(1 - alpha / 2) + _z(power)) * seed_sd_pp * math.sqrt(2.0 / n_seeds)


def questions_needed(
    delta_pp: float, se_pp: float, n: int, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Benchmark size at which a gap of ``delta_pp`` would be powered, given its measured SE.

    Seeds are the wrong lever for a gap that question sampling cannot resolve --
    rerunning the same 1,540 questions with a new seed does not add questions.
    This says how many questions would. The paired SE of a difference of means
    scales as ``1/sqrt(n)``, so the size that puts ``delta_pp`` at the same
    ``(z_{alpha/2} + z_beta) * SE`` threshold ``seeds_needed`` uses is
    ``n * ((z_{alpha/2} + z_beta) * se / delta)^2``. Same alpha, same power, same
    normal approximation, so the two tables are read on one scale.

    It assumes the extra questions look like the ones already measured. A harder
    or easier extension moves the SE and therefore this number.
    """
    if not delta_pp > 0:
        raise ValueError(f"delta_pp must be positive, got {delta_pp}")
    if se_pp <= 0:
        raise ValueError(f"se_pp must be positive, got {se_pp}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    _check_alpha_power(alpha, power)
    return math.ceil(n * ((_z(1 - alpha / 2) + _z(power)) * se_pp / delta_pp) ** 2)


# --- paired bootstrap -----------------------------------------------------


def paired_delta_ci(
    a: Sequence[bool],
    b: Sequence[bool],
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Paired percentile-bootstrap CI for J(a) - J(b), in percentage points.

    ``a`` and ``b`` are the two arms' verdicts over the *same* questions in the
    same order (see ``aligned_arms``). Resampling is over questions: one index
    draw per replicate is applied to both arms, which is what makes the interval
    paired. Equivalently and as implemented, the per-question difference vector
    is resampled directly -- identical for a difference of means, and it makes
    the zero-width interval for two identical arms exact rather than approximate.
    """
    arr_a = np.asarray(a, dtype=bool)
    arr_b = np.asarray(b, dtype=bool)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"arms must be the same length, got {arr_a.size} and {arr_b.size}")
    if arr_a.size == 0:
        raise ValueError("cannot bootstrap an empty pair of arms")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    _check_alpha(alpha)

    diff = arr_a.astype(np.int8) - arr_b.astype(np.int8)
    n = diff.size
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, _BOOT_BLOCK):
        block = min(_BOOT_BLOCK, n_boot - start)
        idx = rng.integers(0, n, size=(block, n))
        deltas[start : start + block] = diff[idx].mean(axis=1)
    deltas *= 100.0

    lo, hi = (float(x) for x in np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    # achieved significance level: how much of the bootstrap distribution sits on
    # the wrong side of zero, doubled. Its resolution floor is 1/n_boot, so 0.0
    # here means "below what n_boot can measure", never "impossible".
    p_boot = min(1.0, 2.0 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0))))
    return {
        "delta_pp": round(100.0 * float(diff.mean()), 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "se_pp": round(float(deltas.std(ddof=1)) if n_boot > 1 else 0.0, 4),
        "p_boot": round(p_boot, 6),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "n": int(n),
        "n_disagree": int(np.count_nonzero(diff)),
        "n_boot": int(n_boot),
        "alpha": alpha,
        "seed": int(seed),
    }


# --- rank stability under gold noise --------------------------------------


def rank_flip_prob(
    arms: Mapping[str, Sequence[bool]],
    error_mask: Sequence[bool],
    n_sim: int = 10_000,
    seed: int = 0,
    rate_basis: str = "unflagged",
) -> dict:
    """Probability of each ranking once the flagged verdicts are redrawn.

    Undisputed verdicts are held fixed; each flagged verdict is redrawn as
    Bernoulli(p_arm), generated as one Binomial per arm per simulation since J
    sees the flagged block only through its total. See the module docstring for
    the two ``rate_basis`` readings and why arms are drawn independently.

    Ranking is on integer correct counts, not on J, so two arms tie only when
    they truly score the same -- no float comparison decides a rank. Ties are
    broken toward the observed order, which makes ``p_observed_order`` an upper
    bound whenever ties are common; ``p_any_tie`` and each pair's ``p_tie`` are
    reported alongside so that can never be read off as a strict win.
    """
    if rate_basis not in RATE_BASES:
        raise ValueError(f"rate_basis must be one of {RATE_BASES}, got {rate_basis!r}")
    if len(arms) < 2:
        raise ValueError(f"ranking needs at least two arms, got {len(arms)}")
    if n_sim < 1:
        raise ValueError(f"n_sim must be at least 1, got {n_sim}")

    mask = np.asarray(error_mask, dtype=bool)
    vectors = {name: np.asarray(v, dtype=bool) for name, v in arms.items()}
    for name, vec in vectors.items():
        if vec.shape != mask.shape:
            raise ValueError(
                f"arm {name!r} has length {vec.size} but the error mask has length {mask.size}"
            )
    n_total = int(mask.size)
    n_flagged = int(mask.sum())

    # observed order, best first; ties keep the caller's key order
    names = sorted(vectors, key=lambda k: -int(vectors[k].sum()))
    fixed = {n: int(vectors[n][~mask].sum()) for n in names}
    rates = {}
    for name in names:
        pool = vectors[name][mask] if rate_basis == "flagged" else vectors[name][~mask]
        rates[name] = float(pool.mean()) if pool.size else 0.0

    rng = np.random.default_rng(seed)
    counts = np.empty((n_sim, len(names)), dtype=np.int64)
    for i, name in enumerate(names):
        counts[:, i] = fixed[name] + rng.binomial(n_flagged, rates[name], size=n_sim)

    order_idx = np.argsort(-counts, axis=1, kind="stable")
    ranked = np.take_along_axis(counts, order_idx, axis=1)
    any_tie = np.zeros(n_sim, dtype=bool)
    if len(names) > 1:
        any_tie = (np.diff(ranked, axis=1) == 0).any(axis=1)

    orders, freq = np.unique(order_idx, axis=0, return_counts=True)
    permutations = sorted(
        (
            {
                "order": [names[i] for i in row],
                "count": int(c),
                "prob": round(float(c) / n_sim, 6),
            }
            for row, c in zip(orders, freq)
        ),
        key=lambda p: (-p["count"], p["order"]),
    )
    observed_order = list(names)
    p_observed = next(
        (p["prob"] for p in permutations if p["order"] == observed_order),
        0.0,
    )

    pairs = []
    # names are in observed-rank order and counts' columns follow them, so the
    # i-th pair is columns i and i+1 -- no lookup back through names.
    for i, (high, low) in enumerate(pairwise(names)):
        hi_c = counts[:, i]
        lo_c = counts[:, i + 1]
        gap = 100.0 * (hi_c - lo_c) / n_total
        pairs.append(
            {
                "high": high,
                "low": low,
                "p_flip": round(float(np.mean(lo_c > hi_c)), 6),
                "p_tie": round(float(np.mean(lo_c == hi_c)), 6),
                "mean_gap_pp": round(float(gap.mean()), 4),
                "sd_gap_pp": round(float(gap.std(ddof=1)) if n_sim > 1 else 0.0, 4),
            }
        )

    return {
        "rate_basis": rate_basis,
        "n_sim": int(n_sim),
        "seed": int(seed),
        "n_total": n_total,
        "n_flagged": n_flagged,
        "arms": {
            name: {
                "J_full": round(100.0 * float(vectors[name].mean()), 4),
                "fixed_correct": fixed[name],
                "p_resample": round(rates[name], 6),
                "mean_sim_J": round(float(100.0 * counts[:, i].mean() / n_total), 4),
                "sd_sim_J": round(
                    float(100.0 * counts[:, i].std(ddof=1) / n_total) if n_sim > 1 else 0.0, 4
                ),
            }
            for i, name in enumerate(names)
        },
        "observed_order": observed_order,
        "p_observed_order": p_observed,
        "p_any_tie": round(float(any_tie.mean()), 6),
        "permutations": permutations,
        "pairs": pairs,
    }


# --- alignment ------------------------------------------------------------


def aligned_arms(paths: Mapping[str, Path]) -> tuple[list[tuple[int, str]], dict[str, list[bool]]]:
    """Load each arm's judged verdicts, verifying every arm asks the same questions.

    Returns the shared join-key sequence in file order and one verdict vector
    per arm, aligned to it positionally. A disagreement in the sequence raises:
    every paired number downstream assumes position i is the same question in
    every arm, and comparing question i to question j would look like a result.
    """
    keys: list[tuple[int, str]] | None = None
    reference = ""
    arms: dict[str, list[bool]] = {}
    for name, path in paths.items():
        judged = [r for r in load_records(Path(path)) if "j" in r]
        seq = [(r["conv"], normalize_q(r["q"])) for r in judged]
        if keys is None:
            keys, reference = seq, name
        elif seq != keys:
            first = next(
                (i for i, (x, y) in enumerate(zip(seq, keys)) if x != y), min(len(seq), len(keys))
            )
            raise ValueError(
                f"arm {name!r} does not ask the same questions in the same order as "
                f"{reference!r}: {len(seq)} vs {len(keys)} judged rows, first difference at "
                f"position {first}"
            )
        arms[name] = [bool(r["j"]) for r in judged]
    if keys is None:
        raise ValueError("no arms given")
    return keys, arms


def error_mask(keys: Sequence[tuple[int, str]], flagged: set[tuple[int, str]]) -> list[bool]:
    """One flag per aligned position. Every row sharing a flagged key is marked."""
    return [k in flagged for k in keys]


def flagged_keys(errors_path: Path, dataset_path: Path) -> set[tuple[int, str]]:
    """The score-corrupting audit entries, resolved to join keys via the dataset."""
    corrupting = score_corrupting(load_errors(Path(errors_path)))
    return error_question_keys(corrupting, question_key_map(Path(dataset_path)))


def judged_question_order(dataset_path: Path) -> list[tuple[int, str]]:
    """The dataset enumeration Task 1 pinned, restricted to judged (non-adversarial) rows."""
    samples = json.loads(Path(dataset_path).read_text())
    return [
        (conv, normalize_q(qa["question"]))
        for conv, sample in enumerate(samples)
        for qa in sample["qa"]
        if qa.get("category") != ADVERSARIAL_CATEGORY
    ]


def check_canonical_order(keys: Sequence[tuple[int, str]], dataset_path: Path) -> bool:
    """Assert the arms' row order *is* the dataset enumeration order, not merely the same set.

    Task 1 made the qa enumeration authoritative for question identity, so the
    harness agreeing with it is checkable rather than assumed. Multiset equality
    would not be enough: conv 7 ships duplicated questions, and a reordering that
    preserved the multiset would still misalign the pairs.
    """
    expected = judged_question_order(dataset_path)
    if list(keys) != expected:
        first = next(
            (i for i, (x, y) in enumerate(zip(keys, expected)) if x != y),
            min(len(keys), len(expected)),
        )
        raise ValueError(
            f"records order is not the canonical dataset order: {len(keys)} vs "
            f"{len(expected)} judged questions, first difference at position {first}"
        )
    return True


# --- report ---------------------------------------------------------------


def _subset(values: Sequence[bool], keep: Sequence[bool]) -> list[bool]:
    return [v for v, k in zip(values, keep) if k]


def _j(values: Sequence[bool]) -> float:
    return 100.0 * sum(values) / len(values)


def _seeds_or_none(delta_pp: float) -> int | None:
    """Seeds for a gap of either sign; ``None`` when the arms are exactly level.

    A zero gap is not resolvable at any seed count, and that is a real outcome
    here -- dropping the flagged questions can leave two arms tied. Reporting
    some integer for it would put a finite, false answer in the table.
    """
    return None if delta_pp == 0 else seeds_needed(abs(delta_pp))


def _pair_block(arms: dict[str, list[bool]], n_boot: int, seed: int) -> list[dict]:
    """Paired CI for each adjacent rank pair, all sharing one resample stream.

    Same seed across pairs on purpose: the three intervals are then read off the
    same question resamples, so they are mutually consistent rather than three
    unrelated draws.

    ``ADJACENT_PAIRS`` is fixed to the full-gold rank order, so under a gold that
    reorders the arms these pairs would no longer be the adjacent ones. That is
    why ``rank_order_excluded`` is reported and the conclusion states outright
    whether the order held -- a reordering must be visible, not absorbed here.
    """
    out = []
    for high, low in ADJACENT_PAIRS:
        ci = paired_delta_ci(arms[high], arms[low], n_boot=n_boot, seed=seed)
        ci["pair"] = [high, low]
        out.append(ci)
    return out


def build_report(
    arms: dict[str, list[bool]],
    mask: list[bool],
    n_boot: int,
    n_sim: int,
    seed: int,
) -> dict:
    """Every number the artifact prints, computed once."""
    keep = [not m for m in mask]
    excluded = {name: _subset(v, keep) for name, v in arms.items()}

    arm_rows = {
        name: {
            "stem": ARM_STEMS[name],
            "J_full": round(_j(v), 4),
            "J_full_2dp": round(_j(v), 2),
            "J_excluded": round(_j(excluded[name]), 4),
            "n_judged": len(v),
            "n_judged_excluded": len(excluded[name]),
            "correct_full": int(sum(v)),
            "correct_on_flagged": int(sum(_subset(v, mask))),
        }
        for name, v in arms.items()
    }

    ci_full = _pair_block(arms, n_boot, seed)
    ci_excluded = _pair_block(excluded, n_boot, seed)
    stability = {
        basis: rank_flip_prob(arms, mask, n_sim=n_sim, seed=seed, rate_basis=basis)
        for basis in RATE_BASES
    }

    seeds = []
    for (high, low), cf, ce in zip(ADJACENT_PAIRS, ci_full, ci_excluded):
        seeds.append(
            {
                "pair": [high, low],
                "delta_pp_full": cf["delta_pp"],
                "seeds_full": _seeds_or_none(cf["delta_pp"]),
                "delta_pp_excluded": ce["delta_pp"],
                "seeds_excluded": _seeds_or_none(ce["delta_pp"]),
                # seeds do not add questions: for a gap the bootstrap cannot
                # separate, this is the lever that would.
                "questions_needed_full": (
                    None
                    if cf["delta_pp"] == 0
                    else questions_needed(abs(cf["delta_pp"]), cf["se_pp"], cf["n"])
                ),
                "separable_full": cf["excludes_zero"],
                "separable_excluded": ce["excludes_zero"],
            }
        )

    bands = {
        "seed_sd_pp": SEED_SD_PP,
        "alpha": 0.05,
        "power": 0.8,
        "min_detectable_delta_pp": {
            str(n): round(min_detectable_delta_pp(n), 4) for n in (1, 2, 4, 8, 16)
        },
    }
    order_excluded = sorted(excluded, key=lambda k: -_j(excluded[k]))
    return {
        "arms": arm_rows,
        "rank_order_full": sorted(arms, key=lambda k: -_j(arms[k])),
        "rank_order_excluded": order_excluded,
        "paired_delta_ci": {"full_gold": ci_full, "audit_excluded_gold": ci_excluded},
        "rank_stability": stability,
        "seeds_needed": seeds,
        "detection_bands": bands,
        "conclusion": _conclusion(
            seeds, ci_full, ci_excluded, stability, bands, order_excluded, sum(mask)
        ),
    }


def _conclusion(
    seeds: list[dict],
    ci_full: list[dict],
    ci_excluded: list[dict],
    stability: dict,
    bands: dict,
    order_excluded: list[str],
    n_flagged: int,
) -> str:
    """The band statement, built from the table so it cannot drift from it.

    A pair counts as separated only when the paired CI excludes zero under
    *both* golds -- the strictest of the three noise sources, and the only one
    that puts every question at risk rather than the 99 flagged ones. The
    sentence is assembled from that test rather than around it, so a pair the
    bootstrap cannot separate cannot be phrased as a win by the template.
    """
    mdd1 = bands["min_detectable_delta_pp"]["1"]
    order = " > ".join(s["pair"][0] for s in seeds) + " > " + seeds[-1]["pair"][1]
    p_hold = min(s["p_observed_order"] for s in stability.values())
    n_sim = stability[RATE_BASES[0]]["n_sim"]
    gold_same = order_excluded == [s["pair"][0] for s in seeds] + [seeds[-1]["pair"][1]]

    def phrase(s: dict, c: dict) -> str:
        # a bootstrap p cannot resolve below 1/n_boot, so print the bound rather
        # than a "0.000" that would read as an exact zero.
        floor = 1.0 / c["n_boot"]
        p = f"p<{floor:g}" if c["p_boot"] < floor else f"p={c['p_boot']:.3f}"
        return (
            f"{s['pair'][0]} over {s['pair'][1]} ({c['delta_pp']:+.2f}pp, "
            f"[{c['lo']:+.2f}, {c['hi']:+.2f}], {p})"
        )

    firm, weak = [], []
    for s, cf, ce in zip(seeds, ci_full, ci_excluded):
        (firm if s["separable_full"] and s["separable_excluded"] else weak).append((s, cf, ce))

    if not weak:
        head = (
            f"The full ranking {order} is claimable. Every adjacent gap is separated by its "
            f"paired 95% bootstrap CI under both golds, the tightest being "
            f"{phrase(*firm[0][:2])}."
        )
    else:
        kept = "; ".join(phrase(s, cf) for s, cf, _ in firm) or "no adjacent pair"
        lost = "; ".join(phrase(s, cf) for s, cf, _ in weak)
        on_excluded = ", ".join(
            "{:+.2f}pp [{:+.2f}, {:+.2f}]".format(ce["delta_pp"], ce["lo"], ce["hi"])
            for _, _, ce in weak
        )
        head = (
            f"Only part of the ranking {order} is claimable. The paired 95% bootstrap CIs "
            f"separate {kept}, but not {lost} -- that interval covers zero, so at "
            f"{weak[0][1]['n']:,} questions the gap is inside question-sampling noise. "
            f"Dropping the {n_flagged} audited questions does not rescue it ({on_excluded})."
        )

    lever = ""
    if weak:
        need = weak[0][0]["questions_needed_full"]
        lever = (
            f" More seeds would not fix it, because rerunning the same questions does not add "
            f"questions: at the measured paired SE, {weak[0][0]['pair'][0]} over "
            f"{weak[0][0]['pair'][1]} would need roughly {need:,} questions "
            f"(vs {weak[0][1]['n']:,}) to be powered at alpha=0.05 / power=0.80."
        )

    return (
        f"{head}{lever} The other two noise sources are not the binding constraint and must "
        f"not be read as agreement: every gap clears the {mdd1:.2f}pp a single seed resolves "
        f"at alpha=0.05 / power=0.80 given the +/-{bands['seed_sd_pp']}pp replicate SD, and "
        f"redrawing the 99 disputed verdicts leaves the order intact in at least "
        f"{100 * p_hold:.2f}% of {n_sim:,} simulations under both resampling rates, with the "
        f"order {'unchanged' if gold_same else 'CHANGED'} when those questions are dropped "
        f"instead. Both of those checks are narrower than the bootstrap: the gold-noise "
        f"simulation perturbs only 99 of {ci_full[0]['n']:,} verdicts and holds the rest "
        f"fixed, which is why it reports a stability the bootstrap over all "
        f"{ci_full[0]['n']:,} does not support. So: the ranking is claimable at one seed and "
        f"under either gold for the pairs listed as separated above, and for those pairs only; "
        f"any future arm landing within {mdd1:.2f}pp of another is a tie at one seed before "
        f"question sampling is even considered, and one at the {bands['seed_sd_pp']}pp "
        f"replicate SD would need {seeds_needed(bands['seed_sd_pp'])} seeds."
    )


def _fmt_ci(c: dict) -> str:
    return f"[{c['lo']:+.2f}, {c['hi']:+.2f}]"


def _seeds_cell(n: int | None) -> str:
    """A tied pair is unresolvable at any budget -- say that, do not print `None`."""
    return "no number of seeds (gap is 0)" if n is None else str(n)


def render_markdown(report: dict, meta: dict) -> str:
    """A page the reader can check against the JSON without rerunning anything."""
    arms = report["arms"]
    lines = [
        "# X1 discriminative power: what the four-arm ranking can be claimed to show",
        "",
        (
            f"Arms: {meta['n_judged']} judged questions each, identical question order "
            f"(verified positionally, and against the canonical dataset enumeration). "
            f"{meta['n_flagged']} of them are flagged score-corrupting by the audit."
        ),
        "",
        (
            f"Deterministic: seed={meta['seed']}, n_boot={meta['n_boot']}, "
            f"n_sim={meta['n_sim']}. The same command reproduces these numbers byte for byte."
        ),
        "",
        "## Arms",
        "",
        "| arm | run | J (full gold) | J (audit-excluded) | correct on the 99 flagged |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name in report["rank_order_full"]:
        a = arms[name]
        lines.append(
            f"| `{name}` | `{a['stem']}` | {a['J_full']:.2f} | {a['J_excluded']:.2f} "
            f"| {a['correct_on_flagged']} / {meta['n_flagged']} |"
        )
    lines += [
        "",
        (
            f"Rank order is `{' > '.join(report['rank_order_full'])}` under the full gold and "
            f"`{' > '.join(report['rank_order_excluded'])}` with the flagged questions dropped."
        ),
        "",
        "**Which adjacent gaps the evidence actually separates:**",
        "",
        "| pair | paired bootstrap (both golds) | seed jitter | gold noise |",
        "| --- | --- | --- | --- |",
    ]
    flip = {
        (p["high"], p["low"]): p["p_flip"] for p in report["rank_stability"]["unflagged"]["pairs"]
    }
    for s in report["seeds_needed"]:
        pair = (s["pair"][0], s["pair"][1])
        sep = s["separable_full"] and s["separable_excluded"]
        lines.append(
            f"| {pair[0]} over {pair[1]} | {'**separated**' if sep else '**NOT separated**'} "
            f"| clears ({_seeds_cell(s['seeds_full'])} seed) "
            f"| holds (P(flip) = {flip.get(pair, float('nan')):.4f}) |"
        )
    lines += [
        "",
        (
            "The three columns are three different noise sources, not three votes on one "
            "question. Only the first puts every question at risk; sections 2 and 3 explain "
            "why the other two are narrower and must not be read as corroboration."
        ),
        "",
        "## 1. Paired bootstrap CIs for the adjacent gaps",
        "",
        (
            "Percentile bootstrap over the per-question difference vector, resampling "
            "questions with replacement. Paired because the arms answered the same questions "
            "in the same order; an unpaired interval would discard that and come out roughly "
            "twice as wide. `p` is the achieved significance level from the bootstrap "
            "distribution and cannot resolve below 1/n_boot, so `0.000000` means "
            '"under the resolution of this many resamples", not "impossible".'
        ),
        "",
        "| gold | pair | dJ (pp) | 95% CI (pp) | SE | p | disagreeing questions | excludes 0 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, key in (("full", "full_gold"), ("audit-excluded", "audit_excluded_gold")):
        for c in report["paired_delta_ci"][key]:
            lines.append(
                f"| {label} | {c['pair'][0]} - {c['pair'][1]} | {c['delta_pp']:+.2f} "
                f"| {_fmt_ci(c)} | {c['se_pp']:.2f} | {c['p_boot']:.6f} "
                f"| {c['n_disagree']} / {c['n']} | {'yes' if c['excludes_zero'] else 'NO'} |"
            )

    lines += [
        "",
        "## 2. Rank stability when the flagged verdicts are redrawn",
        "",
        (
            f"The {meta['n_flagged']} audited questions have gold we cannot trust, so their "
            "verdicts are redrawn as Bernoulli(p) per arm while the other "
            f"{meta['n_judged'] - meta['n_flagged']} verdicts are held fixed. Arms are drawn "
            "independently, which is conservative: real difficulty is positively correlated "
            "across arms, and that correlation would shrink the variance of each gap. "
            "This is a gold-noise estimate only -- the fixed verdicts carry sampling "
            "uncertainty of their own, which is what section 1 measures."
        ),
        "",
        (
            'Two readings of "the arm\'s rate", both reported: **unflagged** draws at the '
            "arm's accuracy on trustworthy gold (as if the flagged questions had been graded "
            "as fairly as the rest), **flagged** draws at the rate actually observed on the "
            "bad gold (jitter around the status quo)."
        ),
        "",
    ]
    for basis in RATE_BASES:
        s = report["rank_stability"][basis]
        lines += [
            f"### rate basis: `{basis}`",
            "",
            (
                f"P(observed order `{' > '.join(s['observed_order'])}`) = "
                f"**{s['p_observed_order']:.4f}** over {s['n_sim']:,} simulations; "
                f"P(any tie) = {s['p_any_tie']:.4f}. Ties break toward the observed order, "
                "so the figure above is an upper bound whenever ties occur."
            ),
            "",
            "| arm | draw rate p | mean simulated J | sd | observed J |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name in s["observed_order"]:
            a = s["arms"][name]
            lines.append(
                f"| `{name}` | {a['p_resample']:.4f} | {a['mean_sim_J']:.2f} "
                f"| {a['sd_sim_J']:.3f} | {a['J_full']:.2f} |"
            )
        lines += [
            "",
            "| adjacent pair | P(flip) | P(tie) | mean gap (pp) | sd (pp) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for p in s["pairs"]:
            lines.append(
                f"| {p['high']} over {p['low']} | {p['p_flip']:.6f} | {p['p_tie']:.6f} "
                f"| {p['mean_gap_pp']:+.2f} | {p['sd_gap_pp']:.3f} |"
            )
        lines += [
            "",
            "| ranking | probability |",
            "| --- | ---: |",
        ]
        for perm in s["permutations"]:
            lines.append(f"| {' > '.join(perm['order'])} | {perm['prob']:.6f} |")
        if all(p["p_flip"] == 0.0 for p in s["pairs"]) and s["n_flagged"]:
            lines += [
                "",
                (
                    f"No flip in {s['n_sim']:,} simulations bounds each flip probability at "
                    f"roughly {3.0 / s['n_sim']:.2e} (rule of three, one-sided 95%). It is not "
                    "evidence that a flip is impossible."
                ),
            ]
        lines.append("")

    b = report["detection_bands"]
    lines += [
        "## 3. Seeds required per adjacent gap",
        "",
        (
            f"Two-sample normal approximation `n = ((z_a/2 + z_b) * sd * sqrt(2) / d)^2`, "
            f"rounded up, with sd = {b['seed_sd_pp']}pp (the Track-1 seed-2 replicate SD), "
            f"alpha = {b['alpha']}, power = {b['power']}. This is jitter between reruns of the "
            "same configuration, a different noise source from both sections above; the three "
            "do not compose into a single interval and are not summed here."
        ),
        "",
        (
            "| pair | dJ full (pp) | seeds | dJ audit-excluded (pp) | seeds "
            "| questions for the same power |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in report["seeds_needed"]:
        need = s["questions_needed_full"]
        qn = "--" if need is None else f"{need:,}"
        lines.append(
            f"| {s['pair'][0]} - {s['pair'][1]} | {s['delta_pp_full']:+.2f} "
            f"| {_seeds_cell(s['seeds_full'])} | {s['delta_pp_excluded']:+.2f} "
            f"| {_seeds_cell(s['seeds_excluded'])} | {qn} |"
        )
    lines += [
        "",
        (
            "The last column is the lever seeds do not provide. A seed replicate reruns the "
            "*same* questions, so it cannot help a gap that question sampling fails to "
            "separate; that column is the benchmark size at which the measured paired SE "
            "would put the gap at the same alpha/power threshold, assuming the added "
            "questions resemble the ones already graded."
        ),
    ]
    lines += [
        "",
        "Smallest gap each seed budget can resolve:",
        "",
        "| seeds | min detectable dJ (pp) |",
        "| ---: | ---: |",
    ]
    for n, d in b["min_detectable_delta_pp"].items():
        lines.append(f"| {n} | {d:.2f} |")

    lines += ["", "## 4. Conclusion", "", report["conclusion"], ""]
    return "\n".join(lines) + "\n"


# --- CLI ------------------------------------------------------------------


def _resolve_arms(records_dir: Path) -> dict[str, Path]:
    paths = {name: Path(records_dir) / f"{stem}.records.jsonl" for name, stem in ARM_STEMS.items()}
    missing = sorted(str(p) for p in paths.values() if not p.exists())
    if missing:
        raise SystemExit(f"missing headline records file(s) under {records_dir}: {missing}")
    return paths


def _check_anchors(paths: Mapping[str, Path], arms: Mapping[str, Sequence[bool]]) -> dict:
    """Fail-closed: an arm whose J does not reproduce is not the run we published.

    Nothing may reach disk before this passes -- an artifact built on a run that
    scores differently from its published figure puts that discrepancy into the
    record as if it were a finding.
    """
    checked = {}
    bad = []
    for name, path in paths.items():
        stem = run_stem(Path(path))
        anchor = HEADLINE_ANCHORS.get(stem)
        got = round(_j(arms[name]), 2)
        checked[name] = {"stem": stem, "anchor_J": anchor, "recomputed_J_2dp": got}
        if anchor is None or got != anchor:
            bad.append(f"{stem}: anchor {anchor}, recomputed {got}")
    if bad:
        for line in bad:
            print(f"ANCHOR MISMATCH {line}", file=sys.stderr)
        raise SystemExit("headline anchor check failed -- these are not the published runs, STOP")
    return checked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--n-sim", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    paths = _resolve_arms(args.records_dir)
    keys, arms = aligned_arms(paths)
    check_canonical_order(keys, args.dataset)
    anchors = _check_anchors(paths, arms)

    flagged = flagged_keys(args.errors, args.dataset)
    mask = error_mask(keys, flagged)
    if not any(mask):
        raise SystemExit("no flagged question landed on a judged row -- the join is broken")

    report = build_report(arms, mask, n_boot=args.n_boot, n_sim=args.n_sim, seed=args.seed)
    report["meta"] = {
        "records_dir": str(args.records_dir),
        "dataset_path": str(args.dataset),
        "errors_path": str(args.errors),
        "n_judged": len(keys),
        "n_flagged": sum(mask),
        "flagged_keys": len(flagged),
        "n_boot": args.n_boot,
        "n_sim": args.n_sim,
        "seed": args.seed,
        "seed_sd_pp": SEED_SD_PP,
        "canonical_order_verified": True,
        "anchors": anchors,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = {"meta": report.pop("meta"), **report}
    (out_dir / "power.json").write_text(json.dumps(ordered, indent=2, sort_keys=False) + "\n")
    (out_dir / "power.md").write_text(render_markdown(ordered, ordered["meta"]))
    print(f"wrote {out_dir / 'power.json'} and {out_dir / 'power.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
