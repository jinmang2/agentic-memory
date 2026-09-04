"""Collate LongMemEval-V2 runs of the agmem arms: the per-arm table, paired
sign tests between arms, and the leaderboard's LAFS gain against the paper's
reference frontier.

    uv run python scripts/repro/lme_v2_summary.py results/lme_v2/full \
        --upstream ~/.agmem/upstream/longmemeval-v2 --tier small [--json out.json]

Every run directory the harness wrote (`aggregated_metrics.json` +
`per_question.jsonl`) under the root is one arm, named by its directory. The
paired tests use the exact two-sided sign test on discordant questions, which
is the right test for "same questions, two arms" and needs no distribution.
LAFS reuses upstream's `leaderboard/compute_lafs.py` verbatim; note that the
reference frontier is web+enterprise-combined and a single-domain run is
compared against it only as an indication."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from math import comb
from pathlib import Path

IN_USD_PER_M = 0.10  # qwen/qwen3.5-9b on OpenRouter, 2026-09-02
OUT_USD_PER_M = 0.15


def load_arm(run_dir: Path) -> dict | None:
    agg = run_dir / "aggregated_metrics.json"
    pq = run_dir / "per_question.jsonl"
    if not (agg.exists() and pq.exists()):
        return None
    a = json.loads(agg.read_text())
    rows = [json.loads(line) for line in pq.open()]
    u = a["tokens"]
    read_llm = 0.0
    trace = run_dir / "query_traces" / "agmem-llm-trace.jsonl"
    if trace.exists():
        for line in trace.open():
            r = json.loads(line)
            read_llm += r["tokens_in"] * IN_USD_PER_M / 1e6 + r["tokens_out"] * OUT_USD_PER_M / 1e6
    degraded = sum(1 for r in rows if (r.get("memory_post_query_metadata") or {}).get("degraded"))
    return {
        "name": run_dir.name,
        "n": len(rows),
        "correct": sum(r["score_bool"] for r in rows),
        "accuracy": a["overall"]["overall_full_set"] * 100,
        "non_abs": a["overall"]["overall_non_abstention_only"] * 100,
        "abs": a["overall"]["overall_abstention_only"] * 100,
        "unknown": sum(r["is_unknown"] for r in rows),
        "latency_avg": a["memory_query"]["avg_seconds"],
        "latency_p50": a["memory_query"]["p50_seconds"],
        "latency_p95": a["memory_query"]["p95_seconds"],
        "ctx_tokens": a["memory_context"]["avg_final_tokens"],
        "reader_usd": u["prompt_tokens"] * IN_USD_PER_M / 1e6
        + u["completion_tokens"] * OUT_USD_PER_M / 1e6,
        "read_llm_usd": read_llm,
        "degraded": degraded,
        "by_category": {
            k: v["pct_correct"] * 100 for k, v in a["non_abstention_by_category"].items()
        },
        "abs_by_category": {
            k: v["pct_correct"] * 100 for k, v in a["abstention_by_category"].items()
        },
        "_scores": {r["question_id"]: bool(r["score_bool"]) for r in rows},
    }


def sign_test(a: dict, b: dict) -> dict:
    ids = sorted(set(a["_scores"]) & set(b["_scores"]))
    only_a = sum(1 for q in ids if a["_scores"][q] and not b["_scores"][q])
    only_b = sum(1 for q in ids if b["_scores"][q] and not a["_scores"][q])
    both = sum(1 for q in ids if a["_scores"][q] and b["_scores"][q])
    n = only_a + only_b
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"a": a["name"], "b": b["name"], "both": both, "only_a": only_a, "only_b": only_b,
            "neither": len(ids) - both - n, "discordant": n, "p_two_sided": p}  # fmt: skip


def lafs(arms: list[dict], upstream: Path, tier: str) -> dict:
    sys.path.insert(0, str(upstream / "leaderboard"))
    import compute_lafs as cl

    points = [
        cl.Point(a["name"], acc=a["accuracy"], latency=max(a["latency_avg"], 1e-3)) for a in arms
    ]
    summary = cl.lafs_summary_for_submission(tier, points)
    per_arm = {}
    for a in arms:
        one = cl.lafs_summary_for_submission(
            tier, [cl.Point(a["name"], acc=a["accuracy"], latency=max(a["latency_avg"], 1e-3))]
        )
        per_arm[a["name"]] = one["lafs_gain"]
    return {"all_arms": summary, "gain_per_arm_alone": per_arm}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--upstream", default="~/.agmem/upstream/longmemeval-v2")
    ap.add_argument("--tier", default="small")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root)
    arms = [a for a in (load_arm(d) for d in sorted(root.iterdir()) if d.is_dir()) if a]
    if not arms:
        print("no finished runs under", root, file=sys.stderr)
        return 1
    print(
        f"{'arm':40s} {'acc':>6s} {'n':>4s} {'unk':>4s} {'deg':>4s} {'lat_avg':>8s} {'p95':>7s} {'ctx_tok':>8s} {'reader$':>8s} {'read$':>7s}"
    )
    for a in arms:
        print(
            f"{a['name']:40s} {a['accuracy']:6.1f} {a['n']:4d} {a['unknown']:4d} {a['degraded']:4d} {a['latency_avg']:8.2f} {a['latency_p95']:7.1f} {a['ctx_tokens']:8.0f} {a['reader_usd']:8.2f} {a['read_llm_usd']:7.2f}"
        )
        print(
            f"{'':40s} cats: "
            + ", ".join(f"{k} {v:.0f}%" for k, v in a["by_category"].items())
            + " | abs: "
            + ", ".join(f"{k} {v:.0f}%" for k, v in a["abs_by_category"].items())
        )
    tests = [sign_test(a, b) for a, b in combinations(arms, 2)]
    print("\npaired sign tests (same questions):")
    for t in tests:
        print(
            f"  {t['a']} vs {t['b']}: both {t['both']}, a-only {t['only_a']}, b-only {t['only_b']}, neither {t['neither']} | p={t['p_two_sided']:.3f}"
        )
    upstream = Path(args.upstream).expanduser()
    result = {
        "arms": [{k: v for k, v in a.items() if not k.startswith("_")} for a in arms],
        "sign_tests": tests,
    }
    if (upstream / "leaderboard" / "compute_lafs.py").exists():
        result["lafs"] = lafs(arms, upstream, args.tier)
        s = result["lafs"]["all_arms"]
        print(
            f"\nLAFS ({args.tier}, reference frontier is web+enterprise combined): reference {s['reference_lafs']:.2f}, with our arms {s['submission_lafs']:.2f}, gain {s['lafs_gain']:+.2f}"
        )
        for name, g in result["lafs"]["gain_per_arm_alone"].items():
            print(f"  {name}: gain alone {g:+.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
