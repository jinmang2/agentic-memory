"""Resume a LongMemEval-V2 run from its `prompt_rows.jsonl`: reader generation
and scoring only, the exploration already paid for.

    uv run python scripts/repro/lme_v2_resume.py results/lme_v2/full/<run> \
        --upstream ~/.agmem/upstream/longmemeval-v2 [--concurrency 4] [--request-timeout 600]

The upstream harness has no resume: a stall in its reader phase (2026-09-04,
experience+explorer: 97/240 generated, then 2.5 h with no progress on
hanging requests, after a 9 h gap of the same kind before generation began)
loses the prompt-building phase that took 5 h 51 min and $1.41 of exploration.
`prompt_rows.jsonl` holds everything the reader needs — the messages, the
memory context, the query timings — so this re-runs only what comes after.

Two things differ from the harness, both about resilience and neither about
the model: every reader output is appended to `reader_outputs.jsonl` the
moment it arrives, so a second stall costs nothing already generated; and the
OpenAI client is built with a per-request timeout and two retries instead of
the SDK defaults (600 s, ten retries) that let one hung request block a
concurrency slot for hours. Request parameters (model, temperature, top_p,
top_k, max tokens, thinking) come from the run's own `run_args.json`, through
the harness's `build_reader_request`, unchanged. The scoring tail is the
harness's, line for line, so `per_question.jsonl` and
`aggregated_metrics.json` come out in the same shape the leaderboard
packager reads."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--upstream", default="~/.agmem/upstream/longmemeval-v2")
    ap.add_argument("--concurrency", type=int, default=None, help="default: the run's setting")
    ap.add_argument(
        "--request-timeout", type=float, default=600.0, help="seconds per reader request"
    )
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="overwrite an existing per_question.jsonl")
    args = ap.parse_args(argv)

    root = Path(args.upstream).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from evaluation import harness as H
    from openai import AsyncOpenAI, BadRequestError
    from tqdm import tqdm

    run_dir = Path(args.run_dir).expanduser().resolve()
    per_question_path = run_dir / "per_question.jsonl"
    if per_question_path.exists() and not args.force:
        print(f"refusing: {per_question_path} exists (pass --force to overwrite)", file=sys.stderr)
        return 2
    run_args = argparse.Namespace(**json.loads((run_dir / "run_args.json").read_text()))
    if args.concurrency is not None:
        run_args.reader_max_concurrent_requests = args.concurrency
    prompt_rows = [
        json.loads(line) for line in (run_dir / "prompt_rows.jsonl").open(encoding="utf-8")
    ]
    prompt_rows.sort(key=lambda row: row["stream_index"])

    # -- generation, checkpointed per question ---------------------------------
    outputs_path = run_dir / "reader_outputs.jsonl"
    outputs: dict[str, dict[str, Any]] = {}
    if outputs_path.exists():
        for line in outputs_path.open(encoding="utf-8"):
            rec = json.loads(line)
            outputs[rec["question_id"]] = rec["output"]
    todo = [row for row in prompt_rows if row["question_id"] not in outputs]
    print(f"{len(prompt_rows)} prompt rows, {len(outputs)} already generated, {len(todo)} to go")

    async def generate() -> None:
        import os

        api_key = os.environ.get(run_args.api_key_env)
        if not api_key:
            raise RuntimeError(f"{run_args.api_key_env} is not set")
        client = AsyncOpenAI(
            base_url=run_args.base_url,
            api_key=api_key,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
        semaphore = asyncio.Semaphore(run_args.reader_max_concurrent_requests)
        lock = asyncio.Lock()

        async def run_one(row: dict[str, Any]) -> None:
            async with semaphore:
                try:
                    response_raw, usage = await H.call_reader_model_async(
                        client, run_args, row["messages"]
                    )
                except BadRequestError as exc:
                    print(
                        f"Reader request failed for question_id={row['question_id']}: {exc}. Using empty response.",
                        file=sys.stderr,
                        flush=True,
                    )
                    response_raw, usage = (
                        "",
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    )
                except Exception as exc:  # noqa: BLE001 — a request that gave up is an empty answer, not a dead run
                    print(
                        f"Reader request gave up for question_id={row['question_id']}: {exc!r}. Using empty response.",
                        file=sys.stderr,
                        flush=True,
                    )
                    response_raw, usage = (
                        "",
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    )
            parsed = H.extract_boxed_answer(response_raw)
            output = {
                "response_raw": response_raw,
                "response_parsed_boxed": parsed,
                "is_unknown": H.is_unknown(parsed),
                "usage": usage,
            }
            async with lock:
                outputs[row["question_id"]] = output
                with outputs_path.open("a", encoding="utf-8") as fp:
                    fp.write(
                        json.dumps(
                            {"question_id": row["question_id"], "output": output}, ensure_ascii=True
                        )
                        + "\n"
                    )

        tasks = [asyncio.create_task(run_one(row)) for row in todo]
        with tqdm(total=len(tasks), desc="Generating", unit="q") as progress:
            for task in asyncio.as_completed(tasks):
                await task
                progress.update(1)
        await client.close()

    if todo:
        asyncio.run(generate())

    # -- scoring: the harness's tail, verbatim in effect ------------------------
    records: list[dict[str, Any]] = []
    totals = {
        "prompt": 0,
        "completion": 0,
        "ctx": 0,
        "ctx_orig": 0,
        "q": 0.0,
        "pq": 0.0,
        "trunc": 0,
    }
    q_durs: list[float] = []
    pq_durs: list[float] = []
    eval_config = H.make_eval_config(run_args)
    with per_question_path.open("w", encoding="utf-8") as fp:
        for row in tqdm(prompt_rows, desc="Scoring", unit="q"):
            qid = row["question_id"]
            output = outputs[qid]
            row = {
                **row,
                **{
                    k: output[k]
                    for k in ("response_raw", "response_parsed_boxed", "is_unknown", "usage")
                },
            }
            score_bool, _, _ = H.score_prediction(row, eval_config)
            record = {
                key: row[key]
                for key in (
                    "index", "stream_index", "question_id", "question_type", "category",
                    "is_abstention_problem", "eval_function", "question_text", "question_image",
                    "haystack_ids", "memory_context", "memory_query_duration_seconds",
                    "memory_post_query_duration_seconds", "memory_post_query_metadata",
                    "memory_context_original_token_count", "memory_context_token_count",
                    "memory_context_was_truncated", "prompt_messages", "answer_gold",
                    "response_raw", "response_parsed_boxed", "is_unknown",
                )
            }  # fmt: skip
            record.update(
                {
                    "score": 1.0 if score_bool else 0.0,
                    "score_bool": score_bool,
                    "usage": row["usage"],
                    "timestamp_utc": H.utc_now_iso(),
                }
            )
            fp.write(json.dumps(record, ensure_ascii=True) + "\n")
            fp.flush()
            records.append(record)
            totals["prompt"] += row["usage"]["prompt_tokens"]
            totals["completion"] += row["usage"]["completion_tokens"]
            totals["ctx"] += row["memory_context_token_count"]
            totals["ctx_orig"] += row["memory_context_original_token_count"]
            totals["q"] += row["memory_query_duration_seconds"]
            totals["pq"] += row["memory_post_query_duration_seconds"]
            totals["trunc"] += int(row["memory_context_was_truncated"])
            q_durs.append(float(row["memory_query_duration_seconds"]))
            pq_durs.append(float(row["memory_post_query_duration_seconds"]))

    n = len(records)
    aggregated = H.aggregate_metrics(records)
    aggregated["tokens"] = {
        "prompt_tokens": totals["prompt"],
        "completion_tokens": totals["completion"],
        "total_tokens": totals["prompt"] + totals["completion"],
        "avg_prompt_tokens": totals["prompt"] / n if n else None,
        "avg_completion_tokens": totals["completion"] / n if n else None,
        "avg_total_tokens": (totals["prompt"] + totals["completion"]) / n if n else None,
    }
    aggregated["memory_context"] = {
        "avg_original_tokens": totals["ctx_orig"] / n if n else None,
        "avg_final_tokens": totals["ctx"] / n if n else None,
        "num_truncated_sequences": totals["trunc"],
    }

    def stats(durs: list[float], total: float) -> dict[str, Any]:
        s = sorted(durs)
        return {
            "avg_seconds": total / len(s) if s else None,
            "p50_seconds": s[len(s) // 2] if s else None,
            "p95_seconds": s[min(len(s) - 1, int(0.95 * len(s)))] if s else None,
            "max_seconds": s[-1] if s else None,
            "total_seconds": total,
        }

    aggregated["memory_query"] = stats(q_durs, totals["q"])
    aggregated["memory_post_query"] = stats(pq_durs, totals["pq"])
    aggregated["completed_at_utc"] = H.utc_now_iso()
    summary_path = run_dir / "prompt_build_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    haystacks = {tuple(row["haystack_ids"]) for row in prompt_rows}
    aggregated["shared_haystack"] = bool(summary.get("shared_haystack", len(haystacks) == 1))
    if aggregated["shared_haystack"]:
        aggregated["shared_haystack_ids"] = summary.get("shared_haystack_ids") or list(
            next(iter(haystacks))
        )
    aggregated["resumed_from_prompt_rows"] = True
    H.save_json(run_dir / "aggregated_metrics.json", aggregated)
    ov = aggregated["overall"]
    print(
        f"done: overall {ov['overall_full_set'] * 100:.1f}% over {ov['count_all_questions']} questions → {run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
