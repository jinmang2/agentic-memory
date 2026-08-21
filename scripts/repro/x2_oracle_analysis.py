#!/usr/bin/env python
"""X2 subtractive oracle, step 3: what deleting the non-contributing half cost.

`x2_oracle_prep.py` chose the partition, `x2_oracle_surgery.py` performed the
cut, the runner answered the same 1,986 questions over the cut store, and this
script compares the two runs. It buys nothing and spends nothing: both arms'
per-question verdicts are already on disk, so the contrast is a join.

Paired throughout, and that is the whole point of the design. The two runs share
the question order, the reader, the judge, the prompts and the read path — the
only difference is which items the store could serve — so a per-question paired
statistic is available where cross-arm comparisons in this campaign only ever had
n=5 correlations. A delta is reported with McNemar (how many verdicts actually
flipped, in which direction) and a paired bootstrap CI over questions, because
"within noise" is a claim about an interval and not about a point.

Rows are paired **by index after asserting the keys match**, not by `(conv, q)`:
LoCoMo ships twelve questions whose text repeats inside a conversation, so the
key is not unique and a dict join would silently drop them.

    uv run python scripts/repro/x2_oracle_analysis.py
    uv run python scripts/repro/x2_oracle_analysis.py --arm mem0 --bootstrap 20000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "results" / "repro"
OUT = ROOT / "results" / "ext" / "x2" / "oracle_results.json"

# arm -> (headline tag, cut-store tag). The cut-store runs carry `--tag-suffix
# _x2oracle`, and their stamps now record `data_dir`, so each pair is checkable
# against the store the surgery actually wrote.
PAIRS = {
    "nemori_a": (
        "gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA",
        "gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_x2oracle",
    ),
    "nemori_b": (
        "gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB",
        "gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_x2oracle",
    ),
    "amem": (
        "gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH",
        "gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_x2oracle",
    ),
    "zep_cross_encoder": (
        "gpt-4o-mini_zep_cross_encoder_all_k10_ours_expand-off_run1_e3sZ",
        "gpt-4o-mini_zep_cross_encoder_all_k10_ours_expand-off_run1_x2oracle",
    ),
    "mem0": (
        "gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM",
        "gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_x2oracle",
    ),
}


def load_pair(before_tag: str, after_tag: str) -> tuple[list[dict], list[dict]]:
    before = [json.loads(line) for line in (REPRO / f"{before_tag}.records.jsonl").open()]
    after = [json.loads(line) for line in (REPRO / f"{after_tag}.records.jsonl").open()]
    if len(before) != len(after):
        raise SystemExit(f"{before_tag}: {len(before)} rows vs {len(after)} — not a pair")
    for i, (x, y) in enumerate(zip(before, after)):
        if (x["conv"], x["q"], x["cat"]) != (y["conv"], y["q"], y["cat"]):
            raise SystemExit(
                f"row {i} does not align: {x['conv']}/{x['cat']} vs {y['conv']}/{y['cat']}. "
                "The runs did not consume the dataset in the same order; pairing by index "
                "is invalid and this contrast must not be reported."
            )
    return before, after


def mcnemar(before: list[bool], after: list[bool]) -> dict:
    """Discordant pairs, and the exact two-sided binomial p over them.

    An exact test rather than the chi-square approximation because the counts
    here are small by design: if the cut were harmless in the strong sense,
    b + c would be near zero and chi-square would be the wrong instrument."""
    gained = sum(1 for x, y in zip(before, after) if not x and y)
    lost = sum(1 for x, y in zip(before, after) if x and not y)
    n = gained + lost
    if n == 0:
        return {"gained": 0, "lost": 0, "discordant": 0, "p": 1.0}
    from math import comb

    k = min(gained, lost)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return {"gained": gained, "lost": lost, "discordant": n, "p": min(1.0, 2 * tail)}


def paired_bootstrap(before: list[float], after: list[float], iters: int, seed: int) -> dict:
    """Resample QUESTIONS, keeping each question's before/after pair together —
    the resampling unit has to be the pair or the interval describes two
    independent runs, which these are not."""
    rng = random.Random(seed)
    n = len(before)
    idx = range(n)
    deltas = []
    for _ in range(iters):
        pick = [rng.randrange(n) for _ in idx]
        deltas.append(sum(after[i] - before[i] for i in pick) / n)
    deltas.sort()
    lo = deltas[int(0.025 * iters)]
    hi = deltas[int(0.975 * iters) - 1]
    point = sum(a - b for a, b in zip(after, before)) / n
    return {"delta": point, "ci_low": lo, "ci_high": hi, "iters": iters}


def analyse(arm: str, iters: int, seed: int) -> dict:
    before_tag, after_tag = PAIRS[arm]
    before, after = load_pair(before_tag, after_tag)

    # The judged subset is the one the J-score is defined on. `adversarial`
    # (cat 5) is answered but not judged, so it can neither define contribution
    # (prep.json) nor score this contrast — the same exclusion, kept identical.
    judged = [
        (b, a) for b, a in zip(before, after) if b.get("j") is not None and a.get("j") is not None
    ]
    jb = [bool(b["j"]) for b, _ in judged]
    ja = [bool(a["j"]) for _, a in judged]

    summary_b = json.loads((REPRO / f"{before_tag}.json").read_text())
    summary_a = json.loads((REPRO / f"{after_tag}.json").read_text())
    prep = json.loads((ROOT / "results" / "ext" / "x2" / "prep.json").read_text())["arms"][arm]

    out = {
        "arm": arm,
        "before_tag": before_tag,
        "after_tag": after_tag,
        "cut_store": summary_a["stamp"].get("data_dir"),
        "deletion_fraction": prep["proxy_oracle"]["deletion_fraction"],
        "delete_items": prep["proxy_oracle"]["delete_items"],
        "keep_items": prep["proxy_oracle"]["keep_items"],
        "n_judged": len(judged),
        "j_before": round(100 * sum(jb) / len(jb), 2),
        "j_after": round(100 * sum(ja) / len(ja), 2),
        "j_mcnemar": mcnemar(jb, ja),
        "j_paired": {
            k: (round(100 * v, 3) if isinstance(v, float) else v)
            for k, v in paired_bootstrap(
                [float(x) for x in jb], [float(x) for x in ja], iters, seed
            ).items()
        },
        "f1_before": summary_b["overall"]["f1"],
        "f1_after": summary_a["overall"]["f1"],
        "cost_after_usd": summary_a.get("cost_usd"),
    }
    # Per-category J, on the same judged subset. Reported because the deletion
    # rate is performance-conditional (prep.json): an arm with fewer correct
    # answers has fewer contributing slots, so the categories where it was
    # already weak are where a cut has the least to preserve.
    cats: dict[str, list[tuple[bool, bool]]] = {}
    for (b, _), x, y in zip(judged, jb, ja):
        cats.setdefault(b["cat"], []).append((x, y))
    out["by_category"] = {
        cat: {
            "n": len(rows),
            "before": round(100 * sum(x for x, _ in rows) / len(rows), 2),
            "after": round(100 * sum(y for _, y in rows) / len(rows), 2),
        }
        for cat, rows in sorted(cats.items())
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", choices=sorted(PAIRS), help="default: all available")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260821)
    args = ap.parse_args()

    arms = args.arm or sorted(PAIRS)
    results = []
    for arm in arms:
        after_tag = PAIRS[arm][1]
        if not (REPRO / f"{after_tag}.records.jsonl").exists():
            print(f"[skip] {arm}: no cut-store run on disk yet", flush=True)
            continue
        results.append(analyse(arm, args.bootstrap, args.seed))

    print(
        f"\n{'arm':18} {'cut':>7} {'J before':>9} {'J after':>8} {'ΔJ':>7} "
        f"{'95% CI':>18} {'flips ±':>10} {'p':>7}"
    )
    for r in results:
        p = r["j_paired"]
        m = r["j_mcnemar"]
        print(
            f"{r['arm']:18} {r['deletion_fraction']:6.1%} {r['j_before']:9.2f} {r['j_after']:8.2f} "
            f"{p['delta']:+7.2f} {f'[{p["ci_low"]:+.2f}, {p["ci_high"]:+.2f}]':>18} "
            f"{f'+{m["gained"]}/-{m["lost"]}':>10} {m['p']:7.3f}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"generated_by": "scripts/repro/x2_oracle_analysis.py", "arms": results},
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n[done] wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
