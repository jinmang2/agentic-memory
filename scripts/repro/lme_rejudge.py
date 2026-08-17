"""Re-score answered arms with a DIFFERENT judge. Buys only judge calls.

P3 says the judge is a free variable that nobody pins: the official code asserts
`gpt-4o-2024-08-06`, and the four systems reporting LongMemEval numbers use
gpt-4o-mini, gpt-4.1-mini, gpt-5 and gpt-4o-mini respectively — some with
rewritten prompts. That is a criticism we made of other people's numbers, so it
is one we owe our own.

The generation side is already paid for. Every arm's `.records.jsonl` carries the
question, the gold, the question type and the full hypothesis, which is exactly
what `get_anscheck_prompt` needs — so re-scoring 500 questions costs 500 judge
calls and nothing else. At gpt-4o-mini's rates that is about a cent an arm
against the ~$1 the arm cost to produce.

What comes out is not "the right score". It is the size of the disagreement:
how far the headline moves, which types it moves in, and whether the ORDERING of
the arms survives — the last being the only thing a leaderboard reading of this
benchmark actually depends on.

`enforce_pin=False` is passed deliberately and the output says so on every row:
these numbers are, by the official aggregator's own assert, not comparable with
published ones. That is the point of computing them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from agmem._env import load_env_local
from agmem.bench import longmemeval as lme
from agmem.bench.registry import get_model, registry_cost_usd_split
from agmem.llm import BudgetTracker, LLMClient
from agmem.llm.client import RoleConfig

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "repro"


def rejudge_arm(tag: str, mem, judge_model: str, log_every: int = 100) -> list[dict]:
    """One row per question: the stored verdict, the new verdict, and the delta."""
    path = OUT / f"{tag}.records.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    out = []
    for i, row in enumerate(rows, 1):
        if row.get("label") is None or row.get("hypothesis") is None:
            raise SystemExit(
                f"{tag}: row {row.get('question_id')} was never judged — re-run the arm"
            )
        # The record carries everything the judge prompt needs, which is why this
        # costs one call rather than a whole arm.
        verdict = lme.judge_answer(
            mem,
            {
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "question": row["question"],
                "answer": row["answer"],
            },
            row["hypothesis"],
            enforce_pin=False,
            budget_key=f"judge|{row['question_id']}",
        )
        out.append(
            {
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "label": verdict,
                "label_pinned_judge": row["label"],
                "agrees": verdict == row["label"],
                "judge_model": judge_model,
                "judge_pinned": False,
            }
        )
        if i % log_every == 0:
            print(f"  {tag}: {i}/{len(rows)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--max-spend-usd", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    load_env_local()
    spec = get_model(args.judge_model)
    import os

    key = os.environ.get(spec.api_key_env)
    if not key:
        raise SystemExit(f"{spec.api_key_env} is not set")
    budget = BudgetTracker()
    roles = {
        "judge": RoleConfig(
            endpoint=spec.endpoint,
            model=spec.name,
            api_key=key,
            temperature=None if spec.fixed_sampling else lme.JUDGE_TEMPERATURE,
            max_tokens=lme.JUDGE_MAX_TOKENS,
            max_tokens_key=spec.max_tokens_key,
        )
    }
    trace = OUT / f"rejudge_{args.judge_model}.llm-trace.jsonl"
    client = LLMClient(roles, budget=budget, trace_path=trace)
    mem = SimpleNamespace(llm=client)

    def spent() -> float:
        folded = {"judge": {"tokens_in": 0, "tokens_out": 0}}
        for stats in budget.summary().values():
            folded["judge"]["tokens_in"] += stats["tokens_in"]
            folded["judge"]["tokens_out"] += stats["tokens_out"]
        return registry_cost_usd_split(folded, args.judge_model)

    report: dict = {"judge_model": args.judge_model, "judge_pinned": False, "arms": {}}
    t0 = time.perf_counter()
    for tag in args.arms:
        if args.max_spend_usd is not None and spent() >= args.max_spend_usd:
            raise SystemExit(f"spend cap ${args.max_spend_usd} reached before {tag}")
        rows = rejudge_arm(tag, mem, args.judge_model)
        (OUT / f"{tag}.rejudge-{args.judge_model}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        pinned = json.loads((OUT / f"{tag}.json").read_text())["aggregate"]
        fresh = lme.aggregate(rows)
        agree = sum(r["agrees"] for r in rows)
        report["arms"][tag] = {
            "pinned_judge": {
                "task_averaged": pinned["task_averaged"],
                "overall": pinned["overall"],
                "abstention": (pinned["abstention"] or {}).get("acc"),
            },
            "new_judge": {
                "task_averaged": fresh["task_averaged"],
                "overall": fresh["overall"],
                "abstention": (fresh["abstention"] or {}).get("acc"),
            },
            "delta_pp": {
                "task_averaged": round(fresh["task_averaged"] - pinned["task_averaged"], 2),
                "overall": round(fresh["overall"] - pinned["overall"], 2),
            },
            "agreement": round(100 * agree / len(rows), 2),
            "disagreements": len(rows) - agree,
            "by_type_new": {t: v["acc"] for t, v in fresh["by_type"].items()},
            "n": len(rows),
        }
        a = report["arms"][tag]
        print(
            f"{tag:<40} pinned {a['pinned_judge']['overall']:>6.2f} -> "
            f"{a['new_judge']['overall']:>6.2f} ({a['delta_pp']['overall']:+.2f})  "
            f"agreement {a['agreement']:.1f}%",
            flush=True,
        )

    # The only thing a leaderboard reading depends on: does the ORDER survive?
    order_pinned = sorted(
        report["arms"], key=lambda t: -report["arms"][t]["pinned_judge"]["overall"]
    )
    order_new = sorted(report["arms"], key=lambda t: -report["arms"][t]["new_judge"]["overall"])
    report["ranking_pinned"] = order_pinned
    report["ranking_new_judge"] = order_new
    report["ranking_preserved"] = order_pinned == order_new
    report["cost_usd"] = round(spent(), 6)
    report["timing_s"] = round(time.perf_counter() - t0, 1)
    report["llm_budget"] = {"judge": {"calls": budget.total_calls()}}
    out_path = args.out or OUT / f"lme_rejudge_{args.judge_model}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nranking preserved: {report['ranking_preserved']}  cost ${report['cost_usd']:.4f}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
