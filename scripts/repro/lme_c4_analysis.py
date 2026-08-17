"""C4's decision rule, applied to the four arms' rows. Costs $0.

The rule was fixed in `docs/20-lme-reading.md` before the first call was paid
for; this script only executes it, and it executes BOTH official accuracies
because they are different numbers on this benchmark's unequal type counts
(LME-A13): `task_averaged` weights the six types equally, `overall` weights
questions equally, and the paper's own headline is ambiguous between them.

Three things are computed and kept apart, because conflating them is how a small
gap gets reported as a result:

  spread        max - min across the arms, per accuracy. The primary measure.
  paired CI     bootstrap over the SAME 500 questions. Paired because every arm
                answered them in the same order; an unpaired interval would
                discard that and roughly double the width for nothing. The
                task-averaged interval resamples WITHIN each question type, so
                the statistic being bootstrapped is the one being reported.
  McNemar       exact binomial over the discordant pairs. It asks a narrower
                question than the CI — did these two arms disagree asymmetrically
                — and it is the test the pre-registration named.

Arms are read from their `.records.jsonl` files and aligned by question_id. A
run that did not answer every question is refused rather than compared: a
missing row shifts a mean by more than the differences being measured here.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "repro"
# print_qa_metrics.py:17's six, in its report order.
TYPES = (
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)


def load_arm(tag: str) -> dict[str, dict]:
    """`{question_id: row}` for one arm, refusing anything unjudged."""
    path = OUT / f"{tag}.records.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    unjudged = [r["question_id"] for r in rows if r.get("label") is None]
    if unjudged:
        raise SystemExit(f"{tag}: {len(unjudged)} rows carry no judge verdict ({unjudged[:5]})")
    return {str(r["question_id"]): r for r in rows}


def accuracies(labels: np.ndarray, types: np.ndarray) -> tuple[float, float]:
    """(task_averaged, overall) in percent — `aggregate`'s two headlines."""
    per_type = [labels[types == t].mean() for t in TYPES if (types == t).any()]
    return 100 * float(np.mean(per_type)), 100 * float(labels.mean())


def paired_ci(
    diff_fn, a: np.ndarray, b: np.ndarray, types: np.ndarray, n_boot: int, seed: int
) -> dict:
    """Percentile CI for `diff_fn(a) - diff_fn(b)`, resampling questions WITHIN
    type.

    Stratified rather than plain, because `task_averaged` is a mean of per-type
    means: an unstratified resample would let a type's count wander between
    replicates and would put that wobble into the interval as if it were
    measurement noise. The same draw is applied to both arms, which is what
    makes the interval paired."""
    rng = np.random.default_rng(seed)
    idx_by_type = [np.flatnonzero(types == t) for t in TYPES if (types == t).any()]
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        take = np.concatenate([rng.choice(idx, size=idx.size, replace=True) for idx in idx_by_type])
        deltas[i] = diff_fn(a[take], types[take]) - diff_fn(b[take], types[take])
    lo, hi = (float(x) for x in np.percentile(deltas, [2.5, 97.5]))
    point = diff_fn(a, types) - diff_fn(b, types)
    p_boot = min(1.0, 2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0))))
    return {
        "delta_pp": round(float(point), 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "p_boot": round(p_boot, 6),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar over the discordant pairs — the pre-registered test."""
    a_only = int(np.count_nonzero(a & ~b))
    b_only = int(np.count_nonzero(b & ~a))
    n = a_only + b_only
    p = float(binomtest(a_only, n, 0.5).pvalue) if n else 1.0
    return {"a_only": a_only, "b_only": b_only, "discordant": n, "p_exact": round(p, 6)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arms", nargs="+", required=True, help="record tags, e.g. <reader>_lme_oracle_con"
    )
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT / "lme_c4_paired.json")
    args = ap.parse_args()

    arms = {tag: load_arm(tag) for tag in args.arms}
    ids = set.intersection(*(set(a) for a in arms.values()))
    for tag, rows in arms.items():
        if len(rows) != len(ids):
            raise SystemExit(
                f"{tag} has {len(rows)} rows against {len(ids)} shared ids — arms must be "
                "the same population before they can be compared"
            )
    # Canonical order once, so every arm's vector is indexed the same way.
    order = sorted(ids)
    types = np.array([arms[args.arms[0]][q]["question_type"] for q in order])
    labels = {tag: np.array([bool(rows[q]["label"]) for q in order]) for tag, rows in arms.items()}

    def task_avg(lab: np.ndarray, typ: np.ndarray) -> float:
        return 100 * float(np.mean([lab[typ == t].mean() for t in TYPES if (typ == t).any()]))

    def overall(lab: np.ndarray, typ: np.ndarray) -> float:
        # `typ` is unused and stays in the signature anyway: both statistics are
        # passed to the same bootstrap, and giving them one shape is what keeps
        # the resampler from having to know which metric it is resampling.
        del typ
        return 100 * float(lab.mean())

    per_arm = {}
    for tag, lab in labels.items():
        ta, ov = accuracies(lab, types)
        per_arm[tag] = {
            "task_averaged": round(ta, 2),
            "overall": round(ov, 2),
            "by_type": {
                t: round(100 * float(lab[types == t].mean()), 2)
                for t in TYPES
                if (types == t).any()
            },
            "abstention": round(
                100 * float(np.mean([lab[i] for i, q in enumerate(order) if "_abs" in q])),
                2,
            ),
            "n": int(lab.size),
        }

    spreads = {
        metric: {
            "max": max(per_arm[t][metric] for t in args.arms),
            "min": min(per_arm[t][metric] for t in args.arms),
            "spread_pp": round(
                max(per_arm[t][metric] for t in args.arms)
                - min(per_arm[t][metric] for t in args.arms),
                2,
            ),
            "argmax": max(args.arms, key=lambda t: per_arm[t][metric]),
            "argmin": min(args.arms, key=lambda t: per_arm[t][metric]),
        }
        for metric in ("task_averaged", "overall")
    }
    # The pre-registered rule, applied to the primary measure rather than to
    # whichever number reads better (docs/20).
    for block in spreads.values():
        s = block["spread_pp"]
        block["verdict"] = "holds" if s >= 5 else ("holds_weakly" if s >= 2 else "fails")

    pairs = {}
    for x, y in combinations(args.arms, 2):
        pairs[f"{x} - {y}"] = {
            "task_averaged": paired_ci(
                task_avg, labels[x], labels[y], types, args.n_boot, args.seed
            ),
            "overall": paired_ci(overall, labels[x], labels[y], types, args.n_boot, args.seed),
            "mcnemar": mcnemar(labels[x], labels[y]),
        }

    report = {
        "arms": args.arms,
        "n_questions": len(order),
        "per_arm": per_arm,
        "spreads": spreads,
        "pairs": pairs,
        "n_boot": args.n_boot,
        "seed": args.seed,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"spreads": spreads, "per_arm": per_arm}, indent=2))
    for name, block in pairs.items():
        ta, ov, mc = block["task_averaged"], block["overall"], block["mcnemar"]
        print(
            f"{name:>60}  task_avg {ta['delta_pp']:+6.2f} [{ta['lo']:+.2f},{ta['hi']:+.2f}]  "
            f"overall {ov['delta_pp']:+6.2f} [{ov['lo']:+.2f},{ov['hi']:+.2f}]  "
            f"McNemar {mc['a_only']}/{mc['b_only']} p={mc['p_exact']:.4g}"
        )
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
