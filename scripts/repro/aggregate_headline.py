"""Aggregate K independent full-run summaries into one mean±std headline.

The write-once/read-sweep design freezes the note graph at a SINGLE
non-deterministic ingest (write temperature 0.7), so a lone rung-1b/2 number
carries no estimate of the dominant variance source. This aggregator combines K
INDEPENDENT full runs (each its own fresh ingest into a distinct seed store, so
each is a fresh note-graph draw) into per-category and overall F1 mean±std — the
headline number we actually report for rungs 1b/2.

It is pure post-processing over the durable ``results/repro/<tag>.json`` summaries
(no LLM, no API, zero cost) and can be re-run any time on the already-paid seed
runs. Usage:

    uv run python scripts/repro/aggregate_headline.py \\
        --out results/repro/gpt-4o-mini_all_wujiang_headline.json \\
        results/repro/gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed*.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _mean_std(values: list[float]) -> dict[str, float]:
    """mean/std/min/max/n for a list of per-seed scalars. std is the sample
    stdev (0.0 for a single seed, where it is undefined)."""
    return {
        "mean": round(statistics.mean(values), 2),
        "std": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "n_seeds": len(values),
    }


def _collect(summaries: list[dict], block: str, metric: str) -> dict[str, list[float]]:
    """Gather ``metric`` from ``summary[block]`` across seeds. For ``block ==
    'overall'`` the single value is keyed under ``'overall'``; for ``block ==
    'by_category'`` each category name is its own key. Missing values are simply
    absent (a seed that lacks a category does not fabricate a 0)."""
    out: dict[str, list[float]] = {}
    for s in summaries:
        node = s.get(block, {})
        if block == "overall":
            if metric in node:
                out.setdefault("overall", []).append(node[metric])
        else:
            for cat, agg in node.items():
                if metric in agg:
                    out.setdefault(cat, []).append(agg[metric])
    return out


def _assert_consistent(stamps: list[dict]) -> None:
    """Refuse to average across seeds that were NOT the same experiment. The
    headline is only meaningful if every seed shares model/k/eval_mode/expand
    (a stray glob mixing k=10 and k=20, or wujiang and ours, would otherwise be
    silently averaged into one nonsense number)."""
    keys = ("model", "eval_mode", "k", "expand_links")
    ref = {k: stamps[0].get(k) for k in keys}
    for i, st in enumerate(stamps[1:], 1):
        got = {k: st.get(k) for k in keys}
        if got != ref:
            raise SystemExit(
                f"seed {i} config {got} != seed 0 config {ref} — refusing to "
                f"average across mismatched runs (check the summary glob)"
            )


def aggregate(eval_paths: list[Path], ingest_paths: list[Path] | None = None) -> dict:
    """Build the mean±std headline from K eval-summary JSONs. Aggregates F1
    always, and BLEU-1 / J-score wherever the seeds carry them.

    Cost is reported HONESTLY split, never as one optimistic 'total':
      - ``eval_cost_usd``   = summed cost of the eval passes given as positional
                              args (answer + keyword-rewrite + judge).
      - ``ingest_cost_usd`` = summed cost of the K ingest passes, ONLY when the
                              matching ``--ingest-summaries`` are supplied (the
                              paid write path lives in separate --ingest-only
                              summaries the eval glob never sees).
      - ``campaign_cost_usd`` = ingest + eval, only when ingest costs are known.
    Omitting ingest and labeling the eval sum 'total' was the prior bug; keeping
    them separate means no credit is ever silently dropped OR double-counted
    (phase2_headline reuses phase1b_headline's ingest — the ingest cost belongs
    to whichever headline is told about it, and summing two headlines' campaign
    costs would double-count the shared ingest; see ``ingest_note``)."""
    summaries = [json.loads(p.read_text()) for p in eval_paths]
    if not summaries:
        raise SystemExit("no summary files given")
    stamps = [s.get("stamp", {}) for s in summaries]
    _assert_consistent(stamps)

    metrics = {}
    for metric in ("f1", "bleu1", "j_score"):
        overall = _collect(summaries, "overall", metric)
        by_cat = _collect(summaries, "by_category", metric)
        block: dict[str, dict[str, float]] = {}
        if "overall" in overall:
            block["overall"] = _mean_std(overall["overall"])
        for cat, vals in by_cat.items():
            block[cat] = _mean_std(vals)
        if block:
            metrics[metric] = block

    provenance = [
        {
            "file": p.name,
            "git_sha": st.get("git_sha"),
            "cost_usd": s.get("cost_usd"),
            "overall_f1": s.get("overall", {}).get("f1"),
        }
        for p, s, st in zip(eval_paths, summaries, stamps)
    ]
    eval_cost = round(sum((s.get("cost_usd") or 0.0) for s in summaries), 6)

    out = {
        "kind": "headline_mean_std",
        "n_seeds": len(summaries),
        "config": {
            k: stamps[0].get(k)
            for k in ("model", "embedder", "k", "eval_mode", "expand_links", "temps", "conv")
        },
        "metrics": metrics,
        "eval_cost_usd": eval_cost,
        "sources": provenance,
    }

    if ingest_paths:
        ing = [json.loads(p.read_text()) for p in ingest_paths]
        ingest_cost = round(sum((s.get("cost_usd") or 0.0) for s in ing), 6)
        out["ingest_cost_usd"] = ingest_cost
        out["campaign_cost_usd"] = round(eval_cost + ingest_cost, 6)
        out["ingest_sources"] = [
            {"file": p.name, "cost_usd": s.get("cost_usd")} for p, s in zip(ingest_paths, ing)
        ]
        out["ingest_note"] = (
            "campaign_cost_usd includes these seed stores' ingest. If a sibling "
            "headline (e.g. phase2) reuses the SAME seed stores, its ingest is "
            "this same spend — do not sum both headlines' campaign_cost_usd."
        )
    else:
        out["ingest_cost_usd"] = None
        out["campaign_cost_usd"] = None
        out["ingest_note"] = (
            "ingest cost not supplied (pass --ingest-summaries); eval_cost_usd is "
            "the re-score/answer spend only, NOT the full campaign cost."
        )
    return out


def _print_table(agg: dict) -> None:
    """Human-readable F1 mean±std per category (+overall) to stdout."""
    f1 = agg.get("metrics", {}).get("f1", {})
    print(f"\n=== headline F1 (mean±std over {agg['n_seeds']} ingests) ===")
    order = ["multi-hop", "temporal", "open-domain", "single-hop", "adversarial", "overall"]
    keys = [c for c in order if c in f1] + [c for c in f1 if c not in order]
    for cat in keys:
        cell = f1[cat]
        print(
            f"  {cat:12s}  {cell['mean']:6.2f} ± {cell['std']:5.2f}  "
            f"(min {cell['min']:.2f}, max {cell['max']:.2f})"
        )
    print(f"  eval_cost_usd:     ${agg['eval_cost_usd']}")
    if agg.get("ingest_cost_usd") is not None:
        print(f"  ingest_cost_usd:   ${agg['ingest_cost_usd']}")
        print(f"  campaign_cost_usd: ${agg['campaign_cost_usd']}")
    else:
        print("  ingest_cost_usd:   (not supplied — eval cost only; see ingest_note)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate K full-run summaries into mean±std")
    ap.add_argument(
        "summaries", nargs="+", help="paths to results/repro/<tag>.json EVAL seed summaries"
    )
    ap.add_argument(
        "--ingest-summaries",
        nargs="*",
        default=None,
        help="paths to the matching <model>_all_ingest_seed*.json so the headline "
        "can report ingest_cost_usd + campaign_cost_usd (not just eval cost)",
    )
    ap.add_argument("--out", required=True, help="path to write the aggregate JSON")
    args = ap.parse_args()

    paths = [Path(p) for p in args.summaries]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"summary files not found: {missing}")
    ingest_paths = None
    if args.ingest_summaries:
        ingest_paths = [Path(p) for p in args.ingest_summaries]
        ing_missing = [str(p) for p in ingest_paths if not p.exists()]
        if ing_missing:
            raise SystemExit(f"ingest summary files not found: {ing_missing}")

    agg = aggregate(paths, ingest_paths)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=2, ensure_ascii=False))
    _print_table(agg)
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
