"""Which gaps in the Zep read-recipe sweep are large enough to claim.

The sweep varies ONE thing per pair — the read recipe — against a single ingest,
so the pairs below are the closest this campaign gets to a controlled comparison.
That makes it more important, not less, to say which differences survive
question-sampling noise: the sweep's headline gap is +1.11 J, and X1 already
established that a gap that size is not automatically claimable at n=1,540.

Statistics are IMPORTED from `scripts/ext/x1_power.py` rather than restated. That
module is the expansion lane's and is not edited here; only its primitives are
called, so the bootstrap that judges these pairs is bit-for-bit the one that
judged the four-arm ranking. A second implementation of a confidence interval is
how two numbers in one repository come to disagree about what "95%" means.

Pairs, and what each isolates:

  cross_encoder vs rrf   the §4.1 operating point against upstream's own default
                         family. Differs in reranker AND BFS channel, because
                         upstream ties them — the RRF recipes have no BFS
                         anywhere. Not a clean reranker ablation, and the docs
                         must not read it as one.
  rrf vs mmr             the reranker alone: same three subgraphs, neither with
                         BFS. Upstream ships mmr_lambda=1, so this measures MMR
                         with its diversity term OFF.
  edge_rrf vs
  edge_episode_mentions  the episode-mentions reranker alone, both facts-only.
                         Our implementation sorts mention count DESCENDING,
                         following the paper sentence; upstream's sorts ascending
                         and lands the most-mentioned item last (ledger B-4).
                         This pair therefore measures the paper's mechanism, not
                         the shipped one.

The two facts-only rows are never compared against the three-subgraph rows: they
serve a different memory and the difference would be the subgraph count, not the
recipe.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RECORDS_DIR = ROOT / "results" / "repro"

ARM_STEMS = {
    "cross_encoder": "gpt-4o-mini_zep_cross_encoder_all_k10_ours_expand-off_run1_e3sZ",
    "rrf": "gpt-4o-mini_zep_rrf_all_k10_ours_expand-off_run1_e3sZrrf",
    "mmr": "gpt-4o-mini_zep_mmr_all_k10_ours_expand-off_run1_e3sZmmr",
    "edge_rrf": "gpt-4o-mini_zep_edge_rrf_all_k10_ours_expand-off_run1_e3sZerrf",
    "edge_episode_mentions": (
        "gpt-4o-mini_zep_edge_episode_mentions_all_k10_ours_expand-off_run1_e3sZmentions"
    ),
}

# Each arm's published J, from its own summary. `mmr` has no summary JSON: the
# run died writing artifacts and its records were rebuilt from its trace
# (reconstruct_from_trace.py), so its anchor is the J the run printed before
# dying — the same number that reconstruction had to reproduce to be written.
HEADLINE_ANCHORS = {
    "cross_encoder": 42.73,
    "rrf": 41.62,
    "mmr": 40.78,
    "edge_rrf": 34.87,
    "edge_episode_mentions": 33.05,
}

PAIRS = [
    ("cross_encoder", "rrf", "§4.1 operating point vs upstream default family (reranker AND BFS)"),
    ("rrf", "mmr", "reranker alone, both BFS-less; MMR at upstream's lambda=1"),
    ("edge_rrf", "edge_episode_mentions", "episode-mentions reranker alone, both facts-only"),
]


def _load_x1():
    """Import the expansion lane's power module without importing its __main__."""
    path = ROOT / "scripts" / "ext" / "x1_power.py"
    if not path.exists():
        raise SystemExit(f"missing {path} — the statistics live there and are not restated here")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("x1_power", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=RECORDS_DIR / "zep_recipe_power.json")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    x1 = _load_x1()
    paths = {name: RECORDS_DIR / f"{stem}.records.jsonl" for name, stem in ARM_STEMS.items()}
    missing = sorted(str(p) for p in paths.values() if not p.exists())
    if missing:
        raise SystemExit(f"missing records: {missing}")

    keys, arms = x1.aligned_arms(paths)

    # Fail closed, exactly as x1 does: an arm whose J does not recompute from its
    # own records is not the run it claims to be, and nothing may reach disk.
    anchors = {}
    bad = []
    for name, verdicts in arms.items():
        got = round(100 * sum(verdicts) / len(verdicts), 2)
        want = HEADLINE_ANCHORS[name]
        anchors[name] = {"anchor_J": want, "recomputed_J": got, "j_n": len(verdicts)}
        if got != want:
            bad.append(f"{name}: anchor {want}, recomputed {got}")
    if bad:
        for line in bad:
            print(f"ANCHOR MISMATCH {line}", file=sys.stderr)
        raise SystemExit("anchor check failed — these are not the measured runs, STOP")

    results = []
    for hi, lo, what in PAIRS:
        # Every field below is x1's own — including the separation verdict
        # (`excludes_zero`) and the disagreement count. Recomputing either here
        # would be a second opinion about the same numbers.
        ci = x1.paired_delta_ci(arms[hi], arms[lo], n_boot=args.n_boot, seed=args.seed)
        results.append({"pair": f"{hi} - {lo}", "isolates": what, **ci})
        verdict = "SEPARATED" if ci["excludes_zero"] else "NOT separated"
        print(
            f"{hi:>22s} - {lo:<22s} dJ={ci['delta_pp']:+6.2f}pp  "
            f"95% CI [{ci['lo']:+.2f}, {ci['hi']:+.2f}]  p={ci['p_boot']:.4f}  "
            f"disagree={ci['n_disagree']:4d}/{ci['n']}  {verdict}"
        )

    args.out.write_text(
        json.dumps(
            {
                "n_judged": len(keys),
                "n_boot": args.n_boot,
                "seed": args.seed,
                "anchors": anchors,
                "pairs": results,
                "note": (
                    "Statistics imported from scripts/ext/x1_power.py; not restated here. "
                    "Facts-only arms are never compared against three-subgraph arms."
                ),
            },
            indent=2,
        )
    )
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
