"""A-Mem x LoCoMo REPRODUCTION harness (docs/14).

Drives the reproduction ladder for our A-Mem re-implementation against the
LoCoMo dataset on an OpenAI-compatible endpoint (default gpt-4o-mini). Mirrors
``scripts/exp_locomo_conv0.py``'s ``run()`` structure but:
  - loads OPENAI_API_KEY from repo-root ``.env.local`` (stdlib, no dotenv dep),
  - defaults to the WujiangXu/A-Mem faithful eval (``--eval-mode wujiang``),
  - uses the upstream A-Mem write/answer temperatures (0.7 write, 0.7 generate,
    0.5 cat5) and the all-MiniLM-L6-v2 embedder,
  - supports ``--conv all`` (micro-average over all 10 conversations),
  - supports write-once/read-sweep via ``--data-dir`` + ``--ingest-only`` /
    ``--eval-only`` for cheap k-sweeps.

This file is separate from exp_locomo_conv0.py, which is UNTOUCHED (it answers
at t=0.0 on a local Qwen model). Do NOT merge the two.

Examples (see scripts/repro/*.sh):
    uv run python scripts/exp_amem_repro.py --conv 0 --eval-mode wujiang
    uv run python scripts/exp_amem_repro.py --conv all --eval-mode ours --judge \\
        --expand-links on
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from agmem import AgenticMemory
from agmem._env import load_env_local
from agmem.bench import locomo
from agmem.config import AgmemConfig
from agmem.embed.st_embedder import SentenceTransformerEmbedder
from agmem.llm.client import RoleConfig
from agmem.organizers.amem import AMemOrganizer

DATA = Path.home() / ".agmem/datasets/locomo10.json"
OUT = Path(__file__).resolve().parent.parent / "results" / "repro"

# A-Mem answer channel: notes-only dense retrieval with LLM keyword queries
# (WujiangXu test_advanced.py generate_query_llm), matching exp_locomo_conv0's
# `amem` config. cat5 answers at 0.5 (upstream temperature_c5).
MEMORY_TYPES = ("notes",)
CAT5_TEMPERATURE = 0.5


def make_roles(
    endpoint: str,
    model: str,
    api_key: str,
    write_temp: float = 0.7,
    generate_temp: float = 0.7,
    max_tokens: int = 1000,
) -> dict[str, RoleConfig]:
    """A-Mem upstream temperatures (test_advanced.py / memory_layer get_completion
    default 0.7): write-path roles (extract=Ps1 note, distill=Ps3 evolution) at
    ``write_temp`` 0.7; ``generate`` (cat1-4 answers) at ``generate_temp`` 0.7;
    cat5 answers override to 0.5 per-call in ``locomo.answer``. ``judge`` stays
    0.0 (deterministic grading, only used in ``ours`` mode)."""
    return {
        "extract": RoleConfig(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            temperature=write_temp,
            max_tokens=max_tokens,
        ),
        "distill": RoleConfig(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            temperature=write_temp,
            max_tokens=max_tokens,
        ),
        "judge": RoleConfig(
            endpoint=endpoint, model=model, api_key=api_key, temperature=0.0, max_tokens=max_tokens
        ),
        "generate": RoleConfig(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            temperature=generate_temp,
            max_tokens=max_tokens,
        ),
    }


def build_memory(args, embedder, conv_idx: int, roles) -> AgenticMemory:
    """Fresh A-Mem-organized memory for one conversation. When ``--data-dir`` is
    set it is threaded into ``AgmemConfig.data_dir`` so the stores persist under
    ``<data_dir>/<namespace>/`` (config.py:65-123) — enabling ``--ingest-only``
    then ``--eval-only`` reload. Link expansion is toggled on the pipeline
    (``link_expansion_cap`` 5=on / 0=off)."""
    cfg = AgmemConfig(
        profile="lite",
        data_dir=Path(args.data_dir).expanduser() if args.data_dir else None,
        llm_roles=roles,
        use_guided_json=False,
        lexical_types=("episodic",),
    )
    mem = AgenticMemory(
        namespace=f"repro-conv{conv_idx}",
        organizers=[AMemOrganizer()],
        embedder=embedder,
        config=cfg,
    )
    mem.pipeline.link_expansion_cap = 5 if args.expand_links == "on" else 0
    return mem


def combine_aggs(aggs: list[dict]) -> dict:
    """Exact micro-average of per-conversation agg dicts (each carries mean
    f1/bleu1 as 0-100 percentages plus row count ``n``). Weighting each conv's
    mean by its ``n`` recovers the total F1 sum, so this equals scoring every
    QA in one pool. Also folds ``j_score``/``j_n`` when present."""
    n = sum(a["n"] for a in aggs)
    if n == 0:
        return {}
    out = {
        "f1": round(sum(a["f1"] * a["n"] for a in aggs) / n, 2),
        "bleu1": round(sum(a["bleu1"] * a["n"] for a in aggs) / n, 2),
        "n": n,
    }
    jn = sum(a.get("j_n", 0) for a in aggs)
    if jn:
        out["j_score"] = round(sum(a.get("j_score", 0) * a.get("j_n", 0) for a in aggs) / jn, 2)
        out["j_n"] = jn
    return out


def micro_average(results: list[dict]) -> dict:
    """Combine per-conversation evaluate() outputs into one micro-averaged
    overall + by_category block."""
    overall = combine_aggs([r["overall"] for r in results if r["overall"]])
    cats: dict[str, list[dict]] = {}
    for r in results:
        for cat, agg in r["by_category"].items():
            cats.setdefault(cat, []).append(agg)
    by_category = {cat: combine_aggs(aggs) for cat, aggs in sorted(cats.items())}
    return {"overall": overall, "by_category": by_category}


def write_records_sidecar(path: Path, runs_out: list[dict]) -> int:
    """Persist the per-question audit trail to a JSONL sidecar — one JSON object
    per line, each tagged with ``run`` (run number) and ``conv`` (conversation
    index) plus the ``locomo.evaluate()`` record fields (q/gold/pred/cat/f1, and
    ``j`` when judged). JSONL keeps it appendable/streamable/greppable. Every
    question of every conversation and every run is written. Returns the number
    of lines written."""
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in runs_out:
            run_no = r.get("run")
            for rec in r.get("records", []):
                f.write(json.dumps({"run": run_no, **rec}, ensure_ascii=False) + "\n")
                n += 1
    return n


def eval_conversations(args, embedder, roles, conv_indices: list[int]) -> dict:
    """Ingest (unless --eval-only) + QA over the selected conversations, then
    micro-average. Returns the combined result plus a merged LLM budget."""
    samples = locomo.load_locomo(DATA)
    per_conv = []
    all_records: list[dict] = []
    merged_budget: dict = {}
    total_questions = 0
    for idx in conv_indices:
        sample = samples[idx]
        mem = build_memory(args, embedder, idx, roles)
        try:
            if not args.eval_only:
                n_turns = locomo.ingest(mem, sample)
                mem.consolidate()
                if args.ingest_only:
                    print(
                        f"[conv{idx}] ingested {n_turns} turns, "
                        f"notes persisted under {args.data_dir}",
                        flush=True,
                    )
                    per_conv.append({"conv": idx, "n_turns": n_turns})
                    continue
            questions = locomo.select_questions(sample)
            total_questions += len(questions)
            res = locomo.evaluate(
                mem,
                questions,
                k=args.k,
                memory_types=MEMORY_TYPES,
                keyword_queries=True,
                judge=args.judge and args.eval_mode == "ours",
                eval_mode=args.eval_mode,
                cat5_temperature=CAT5_TEMPERATURE,
                progress=lambda i, nq, _i=idx: (
                    print(f"[conv{_i}] {i}/{nq}", flush=True) if i % 40 == 0 else None
                ),
            )
            res["conv"] = idx
            per_conv.append(res)
            # keep the per-question audit trail (records) for the durable sidecar,
            # tagged with the conversation index; the summary JSON stays lean.
            for rec in res.get("records", []):
                all_records.append({"conv": idx, **rec})
            # merge budget across convs (sum calls/tokens per role)
            for role, stats in mem.budget.summary().items():
                agg = merged_budget.setdefault(
                    role, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "errors": 0}
                )
                agg["calls"] += stats.get("calls", 0)
                agg["tokens_in"] += stats.get("tokens_in", 0)
                agg["tokens_out"] += stats.get("tokens_out", 0)
                agg["errors"] += stats.get("errors", 0)
        finally:
            mem.close()

    if args.ingest_only:
        return {"ingest_only": True, "per_conv": per_conv}

    combined = micro_average([r for r in per_conv if "overall" in r])
    combined["per_conv"] = [
        {"conv": r["conv"], "overall": r["overall"], "by_category": r["by_category"]}
        for r in per_conv
        if "overall" in r
    ]
    combined["llm_budget"] = merged_budget
    combined["n_questions"] = total_questions
    # carried internally to the sidecar writer in main(); NOT inlined into the
    # summary JSON (which selects a lean set of keys).
    combined["records"] = all_records
    return combined


def main() -> None:
    ap = argparse.ArgumentParser(description="A-Mem x LoCoMo reproduction harness")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--endpoint", default="https://api.openai.com/v1")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--conv", default="0", help="conversation index (0-9) or 'all'")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--eval-mode", choices=["wujiang", "ours"], default="wujiang")
    ap.add_argument("--expand-links", choices=["off", "on"], default="off")
    ap.add_argument("--judge", action="store_true", help="J-score (ours mode only)")
    ap.add_argument("--runs", type=int, default=1, help="repeat QA for mean±std")
    ap.add_argument("--data-dir", default=None, help="persist stores here")
    ap.add_argument("--ingest-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    if args.ingest_only and not args.data_dir:
        ap.error("--ingest-only requires --data-dir (ephemeral stores would be lost)")
    if args.eval_only and not args.data_dir:
        ap.error("--eval-only requires --data-dir (nothing to reload otherwise)")
    if args.ingest_only and args.eval_only:
        ap.error("--ingest-only and --eval-only are mutually exclusive")

    load_env_local()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to repo-root .env.local "
            "(KEY=VALUE, gitignored) or export it before running."
        )

    conv_indices = list(range(10)) if args.conv == "all" else [int(args.conv)]
    roles = make_roles(args.endpoint, args.model, api_key)
    embedder = SentenceTransformerEmbedder(args.embedder)

    OUT.mkdir(parents=True, exist_ok=True)
    model_safe = args.model.replace("/", "-").replace(":", "-")
    conv_tag = "all" if args.conv == "all" else f"conv{args.conv}"

    if args.ingest_only:
        eval_conversations(args, embedder, roles, conv_indices)
        print(f"[ingest-only] done ({conv_tag}); reload with --eval-only", flush=True)
        return

    runs_out = []
    for run in range(1, args.runs + 1):
        t0 = time.perf_counter()
        combined = eval_conversations(args, embedder, roles, conv_indices)
        combined["run_seconds"] = round(time.perf_counter() - t0, 1)
        combined["run"] = run
        runs_out.append(combined)
        print(
            f"[run {run}/{args.runs}] {conv_tag} overall={combined['overall']}",
            flush=True,
        )

    # mean±std across runs (overall F1/BLEU-1)
    run_summary = {}
    if args.runs > 1:
        f1s = [r["overall"].get("f1", 0.0) for r in runs_out]
        b1s = [r["overall"].get("bleu1", 0.0) for r in runs_out]
        run_summary = {
            "f1_mean": round(statistics.mean(f1s), 2),
            "f1_std": round(statistics.stdev(f1s), 2) if len(f1s) > 1 else 0.0,
            "bleu1_mean": round(statistics.mean(b1s), 2),
            "bleu1_std": round(statistics.stdev(b1s), 2) if len(b1s) > 1 else 0.0,
        }

    stamp = {
        "model": args.model,
        "endpoint": args.endpoint,
        "embedder": args.embedder,
        "k": args.k,
        "eval_mode": args.eval_mode,
        "temps": {"write": 0.7, "generate": 0.7, "cat5": CAT5_TEMPERATURE},
        "expand_links": args.expand_links,
        "conv": args.conv,
        "runs": args.runs,
        "n_questions": runs_out[0].get("n_questions"),
        "memory_types": list(MEMORY_TYPES),
        "keyword_queries": True,
        "eval_only": args.eval_only,
        "dataset": "locomo10",
    }
    result = {
        "stamp": stamp,
        "overall": runs_out[0]["overall"],
        "by_category": runs_out[0]["by_category"],
        "per_conv": runs_out[0].get("per_conv"),
        "llm_budget": runs_out[0].get("llm_budget"),
        "run_summary": run_summary,
        "runs": [
            {"run": r["run"], "overall": r["overall"], "run_seconds": r["run_seconds"]}
            for r in runs_out
        ],
    }

    tag = (
        f"{model_safe}_{conv_tag}_k{args.k}_{args.eval_mode}"
        f"_expand-{args.expand_links}_run{args.runs}"
    )
    # durable per-question audit trail (every question, every conv, every run),
    # written next to the summary as an appendable/greppable JSONL sidecar. The
    # summary only POINTS to it via records_file — records are not inlined here.
    records_name = f"{tag}.records.jsonl"
    n_records = write_records_sidecar(OUT / records_name, runs_out)
    result["records_file"] = records_name

    out_path = OUT / f"{tag}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[done] wrote {out_path}", flush=True)
    print(f"[done] wrote {OUT / records_name} ({n_records} question records)", flush=True)
    if run_summary:
        print(f"[mean±std] {run_summary}", flush=True)


if __name__ == "__main__":
    main()
