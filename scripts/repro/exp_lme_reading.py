"""LongMemEval C4: hold the memory constant, move only the reading.

`longmemeval_oracle.json` contains the evidence sessions and nothing else. It is
the benchmark's own definition of perfect retrieval, so a run over it has NO
memory variable left: whatever moves the score moves for a reason that is not
the memory system. That is the one unmeasured link in claim C
(docs/research/longmemeval.md §9.2) and the only thing this script exists to
buy — four arms over the same 500 questions:

    oracle x {gpt-4o-mini, gpt-5.6-luna} x {con, direct}

The arms differ in the reader model and in the prompt template. They do not
differ in the context: every arm sends the same bytes, and those bytes are
byte-identical to what upstream's `prepare_prompt` builds
(`scripts/repro/lme_audit/prompt_rediff.py`, config D: 500/500).

WHAT THIS SCRIPT REFUSES TO DO (§8.5, hardcoded rather than defaulted, because
each of the six is a way to get a plausible number that is not the benchmark's):

  1. render an oracle instance in the order it ships. `sort_haystack_by_date`
     runs on every instance; 34/500 render differently without it, and upstream
     sorts unconditionally (run_generation.py:225).
  2. cap sessions. There is no `--max-sessions`. A cap of 50 — run_generation.sh's
     own default — already loses evidence on 4/500 `_s` instances (§3.3), and
     the saving is zero here anyway: oracle instances are 2 sessions at p50.
  3. aggregate a partial run. The score is computed only after `len(rows)` equals
     the population, and the missing ids are printed otherwise (LME-A18: upstream
     swallows API failures with `continue` and scores the survivors).
  4. share one memory across questions. One instance is one haystack; a shared
     store merges unrelated ones and inflates every score.
  5. spend without a ceiling or a quote. `--dry-run` renders every real prompt
     against a fake client and reports the call ledger; `--max-spend-usd` is
     checked between rows against the SHARED budget.
  6. pass `max_history_tokens`. Upstream truncates with the model's tokenizer and
     the cut is a no-op on this data (0/500 over the limit); our estimate is
     chars/4 against a measured 4.61, so passing the upstream number would cut
     the median instance that upstream leaves whole (D3). Not capping is what
     makes the prompt faithful.

Cost attribution is per row, not per role: 500 questions answered by a thread
pool over one client cannot be priced by diffing a shared budget, so each call
carries `generate|<qid>` / `judge|<qid>` and the keys are folded back into role
totals for pricing (the judge is a different, much pricier model than the
reader). The full I/O trace holds every prompt and every judge reply verbatim —
the re-score-without-re-spending insurance this campaign runs on.

The memory is `passthrough` with a fake embedder, and both are deliberate: the
full-context path reads no memory at all, so a paid embedder here would bill for
vectors nothing ever queries. Ingest still runs — it is the write path the other
LongMemEval arms would pay for, and its turn count is recorded per row.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import queue
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agmem import AgenticMemory
from agmem._env import load_env_local
from agmem.bench import longmemeval as lme
from agmem.bench.registry import get_model
from agmem.capabilities import detect
from agmem.config import AgmemConfig
from agmem.embed.fake import FakeEmbedder
from agmem.llm import BudgetTracker, LLMClient

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "repro"
DATASETS = {
    "oracle": Path.home() / ".agmem/datasets/longmemeval_oracle.json",
    "s": Path.home() / ".agmem/datasets/longmemeval_s_cleaned.json",
}
# §3.1, measured on this data with the tokenizer upstream actually uses
# (o200k_base). Only the dry run needs it — a paid row reports the endpoint's own
# usage numbers — and it is stated as a constant rather than pulled through
# tiktoken so a quote costs no dependency.
CHARS_PER_TOKEN = 4.610


def _load_repro_helpers():
    """Import `scripts/exp_amem_repro.py` for `cost_usd`/`make_roles`/`git_sha`.

    A path import because `scripts/` is not a package; copying the pricing
    helpers instead is the duplication ledger C-8 is about."""
    path = ROOT / "scripts" / "exp_amem_repro.py"
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("exp_amem_repro", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DryRunLLM:
    """Answers every role without a network call, and counts what a real run
    would have sent.

    Unlike the FiNER runner's dry client this one is exact about the input side:
    the prompts it weighs are the REAL rendered prompts, because everything up to
    the API boundary (sort, strip, render, template) has already run. What it
    cannot know is the output side, so the quote prices completions at the
    benchmark's own ceiling (`max_tokens`), which makes the generate estimate an
    upper bound rather than a guess. The judge prompt carries a canned hypothesis
    instead of a real one, so its input side is slightly under-counted — at 10
    output tokens and one short prompt per row that is cents, and it is stated
    rather than hidden."""

    def __init__(self, roles: dict):
        self.roles = roles
        self.calls: dict[str, int] = {}
        self.prompt_chars: dict[str, int] = {}
        self.max_tokens: dict[str, int] = {}
        self._lock = threading.Lock()

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def chat(self, role, messages, budget_key=None, **overrides):
        prompt = " ".join(m.get("content", "") for m in messages)
        cap = overrides.get("max_tokens") or overrides.get("max_completion_tokens") or 0
        with self._lock:
            self.calls[role] = self.calls.get(role, 0) + 1
            self.prompt_chars[role] = self.prompt_chars.get(role, 0) + len(prompt)
            self.max_tokens[role] = cap
        return "yes" if role == "judge" else "dry-run hypothesis"


def fold_row_keys(budget: dict[str, dict]) -> dict[str, dict]:
    """`{"generate|q1": ..., "judge|q1": ...}` -> `{"generate": ..., "judge": ...}`.

    Pricing needs role totals: `cost_usd` prices the `judge` key at the judge
    model's rates and everything else at the reader's, so per-row keys would
    silently bill 500 judge calls at the reader's rate — off by 16.7x for a
    gpt-4o-mini reader against the pinned gpt-4o judge."""
    out: dict[str, dict] = {}
    for key, stats in budget.items():
        role = key.split("|", 1)[0]
        row = out.setdefault(
            role, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "latency_ms_avg": 0.0, "errors": 0}
        )
        row["calls"] += stats.get("calls", 0)
        row["tokens_in"] += stats.get("tokens_in", 0)
        row["tokens_out"] += stats.get("tokens_out", 0)
        row["errors"] += stats.get("errors", 0)
        row["latency_ms_avg"] += stats.get("latency_ms_avg", 0.0) * stats.get("calls", 0)
    for row in out.values():
        if row["calls"]:
            row["latency_ms_avg"] = round(row["latency_ms_avg"] / row["calls"], 1)
    return out


def resume_rows(path: Path, log) -> tuple[list[dict], set[str]]:
    """Rows already paid for, and their question ids.

    The atomic unit here is the ROW, not a window: instances are independent (one
    fresh memory each), a row is written only after its judge verdict is in, and
    a truncated final line — the shape a kill leaves — is dropped. So a resumed
    run re-buys at most one question."""
    if not path.exists():
        return [], set()
    rows = []
    for lineno, line in enumerate(path.open(encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("records line %d is truncated — resuming from before it", lineno)
            break
    return rows, {str(r.get("question_id")) for r in rows}


def run_pool(
    instances: list[dict],
    *,
    build_mem,
    reading: str,
    workers: int,
    sink,
    sink_lock: threading.Lock,
    budget: BudgetTracker,
    over_budget,
    log,
) -> list[dict]:
    """Answer every instance concurrently; return the rows this process produced.

    A queue of instances rather than `executor.map`, because the spend cap has to
    be able to STOP the run: with every future submitted up front the cap could
    only be observed, never enforced. Each worker checks it before taking the
    next question, so the overshoot is bounded by the questions already in
    flight (`workers`), not by the split."""
    work: queue.Queue = queue.Queue()
    for inst in instances:
        work.put(inst)
    rows: list[dict] = []
    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set():
            try:
                inst = work.get_nowait()
            except queue.Empty:
                return
            if over_budget is not None and over_budget():
                if not stop.is_set():
                    log.warning("spend cap reached — stopping cleanly")
                stop.set()
                return
            qid = str(inst["question_id"])
            mem = build_mem(qid)
            t0 = time.perf_counter()
            try:
                row = lme.run_instance(
                    mem,
                    inst,
                    reading_method=reading,
                    # The four guards that live at the call site (§8.5 1,2,4,6):
                    # the instance arrives sorted, no session cap exists, `mem` is
                    # this question's own, and `max_history_tokens` is not passed.
                    max_sessions=None,
                    full_context=True,
                    judge=True,
                    enforce_pin=True,
                    capture_retrieval=True,
                    budget_key=qid,
                )
            except Exception as exc:  # noqa: BLE001 - a failure must be recorded, not vanish
                # Upstream's `continue` (run_generation.py:376-378) is exactly how
                # a partial run comes to look complete. The row is written with a
                # null label so the completeness check below can see it and the
                # aggregator excludes it rather than scoring it wrong.
                log.warning("question %s failed: %s: %s", qid, type(exc).__name__, exc)
                row = {
                    "question_id": qid,
                    "question_type": str(inst.get("question_type", "")),
                    "hypothesis": None,
                    "label": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                mem.close()
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            capture = row.pop("retrieval", None) or {}
            prompt = capture.get("prompt") or ""
            usage = budget.summary()
            row.update(
                {
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()
                    if prompt
                    else None,
                    "prompt_chars": len(prompt) or None,
                    "n_sessions": len(inst.get("haystack_sessions", [])),
                    "elapsed_ms": elapsed_ms,
                    # Exact per-call usage, readable because only this thread
                    # ever writes these two keys.
                    "usage": {role: usage.get(f"{role}|{qid}") for role in ("generate", "judge")},
                }
            )
            with sink_lock:
                rows.append(row)
                sink.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                sink.flush()
                done = len(rows)
            if done % 25 == 0:
                log.info("%d/%d answered", done, len(instances))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="oracle")
    ap.add_argument("--data", type=Path, default=None, help="override the dataset path")
    ap.add_argument("--reader", default="gpt-4o-mini", help="the model under test")
    ap.add_argument("--reading", choices=["con", "direct"], default="con")
    ap.add_argument(
        "--judge-model",
        default=lme.JUDGE_MODEL_PIN,
        help="pinned by the official aggregator's assert; changing it makes the "
        "number incomparable with every published one",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N questions (smoke runs)")
    ap.add_argument("--tag", default=None, help="artifact tag; defaults to <reader>_lme_<ds>_<rm>")
    ap.add_argument(
        "--max-spend-usd",
        type=float,
        default=None,
        help="hard ceiling for this measurement, counting what earlier processes "
        "of the same run already spent (recovered from the trace)",
    )
    ap.add_argument("--dry-run", action="store_true", help="fake client, no network, $0")
    ap.add_argument("--resume", action="store_true", help="keep rows already in the records file")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="discard an existing records file instead of resuming it",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("agmem.repro.lme")
    load_env_local()
    helpers = _load_repro_helpers()

    data_path = args.data or DATASETS[args.dataset]
    args.tag = args.tag or f"{args.reader}_lme_{args.dataset}_{args.reading}"
    # A dry run writes the same filenames with none of the meaning, so it writes
    # them somewhere else: `results/repro/*.records.jsonl` is what `run_status.py`
    # reads a run's state from, and a $0 quote sitting at a paid arm's tag would
    # both block the real run's resume check and report itself as a finished
    # measurement. The subdirectory is outside that glob.
    out_dir = OUT / "dryrun" if args.dry_run else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / f"{args.tag}.records.jsonl"
    trace_path = out_dir / f"{args.tag}.llm-trace.jsonl"

    raw = lme.load_longmemeval(data_path)
    population = len(raw) if args.limit is None else min(args.limit, len(raw))
    # Sorted here, once, for every instance: the guard has to be unconditional
    # (upstream's is) and it must happen before anything reads the haystack.
    instances = [lme.sort_haystack_by_date(inst) for inst in raw[:population]]
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    log.info(
        "dataset=%s n=%d sha256=%s reader=%s reading=%s judge=%s",
        data_path.name,
        population,
        data_sha[:12],
        args.reader,
        args.reading,
        args.judge_model,
    )

    kept, done_ids = ([], set())
    if not args.dry_run:
        kept, done_ids = resume_rows(records_path, log)
        if kept and not (args.resume or args.overwrite):
            raise SystemExit(
                f"{records_path.name} already holds {len(kept)} answered rows. Pass --resume to "
                "continue it, or --overwrite to throw them away and pay for them again."
            )
        if kept and args.overwrite:
            log.warning("--overwrite: discarding %d rows already paid for", len(kept))
            kept, done_ids = [], set()
        elif kept:
            log.info("resuming: %d/%d questions already answered", len(kept), population)
    todo = [inst for inst in instances if str(inst["question_id"]) not in done_ids]

    reader_spec = get_model(args.reader)
    judge_spec = get_model(args.judge_model)
    api_key = "dry-run" if args.dry_run else helpers.resolve_api_key(reader_spec)
    # make_roles applies ONE max_tokens_key to every role, so a judge whose
    # dialect differs from the reader's is fixed up through role_temps — the
    # recipe exp_amem_repro.py uses for a split --judge-model.
    roles = helpers.make_roles(
        reader_spec.endpoint,
        args.reader,
        api_key,
        judge_endpoint=judge_spec.endpoint,
        judge_model=judge_spec.name,
        judge_api_key=api_key if args.dry_run else helpers.resolve_api_key(judge_spec),
        role_temps={"judge": {"max_tokens_key": judge_spec.max_tokens_key}},
        max_tokens_key=reader_spec.max_tokens_key,
        fixed_sampling=reader_spec.fixed_sampling,
        judge_fixed_sampling=judge_spec.fixed_sampling,
    )

    budget = BudgetTracker()
    if args.dry_run:
        client: Any = DryRunLLM(roles)
    else:
        client = LLMClient(roles, budget=budget, trace_path=trace_path)
    caps = detect()  # once: 500 memories, one capability probe

    def build_mem(qid: str) -> AgenticMemory:
        # data_dir=None -> every store in memory, so 500 fresh stores cost no
        # disk and leave nothing to clean up. sync_write keeps the organizer
        # hooks on the calling thread (passthrough has none, but a background
        # worker per instance would be 500 threads).
        mem = AgenticMemory(
            namespace=f"lme-{qid}",
            organizers=["passthrough"],
            embedder=FakeEmbedder(dim=64),
            config=AgmemConfig(profile="lite", data_dir=None, sync_write=True),
            caps=caps,
        )
        mem.llm = client
        return mem

    prior_spend = 0.0
    over_budget = None
    if args.max_spend_usd is not None and not args.dry_run:
        prior_spend = _spend_from_trace(trace_path, args.reader, args.judge_model, helpers)
        if prior_spend:
            log.info("earlier processes of this run spent $%.4f (from the trace)", prior_spend)

        def over_budget() -> bool:
            spent = prior_spend + helpers.cost_usd(
                fold_row_keys(budget.summary()), args.reader, judge_model=args.judge_model
            )
            return spent >= args.max_spend_usd

        if over_budget():
            raise SystemExit(
                f"already at ${prior_spend:.4f} against a cap of ${args.max_spend_usd:.2f} — "
                "nothing to do; raise the cap deliberately or stop here"
            )
    elif not args.dry_run:
        log.warning("no --max-spend-usd: this run has no ceiling. Quote it with --dry-run first.")

    utc_started = datetime.now(UTC).isoformat()
    t0 = time.perf_counter()
    sink = records_path.open("a" if kept else "w", encoding="utf-8")
    try:
        new_rows = run_pool(
            todo,
            build_mem=build_mem,
            reading=args.reading,
            workers=args.workers,
            sink=sink,
            sink_lock=threading.Lock(),
            budget=budget,
            over_budget=over_budget,
            log=log,
        )
    finally:
        sink.close()
    elapsed = round(time.perf_counter() - t0, 1)
    rows = kept + new_rows

    # LME-A18: the score is not computed on a partial run. Upstream has no such
    # check anywhere, which is how a run that lost instances to API errors scores
    # as if it had answered them.
    complete = len(rows) == population
    missing = sorted(
        {str(i["question_id"]) for i in instances} - {str(r["question_id"]) for r in rows}
    )
    if not complete:
        log.error(
            "INCOMPLETE: %d/%d answered — no score is reported. Missing: %s",
            len(rows),
            population,
            ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""),
        )

    raw_budget = budget.summary()
    folded = fold_row_keys(raw_budget)
    summary: dict[str, Any] = {
        "stamp": {
            "dataset": args.dataset,
            "data_file": str(data_path),
            "data_sha256": data_sha,
            "release": "longmemeval-cleaned" if "cleaned" in data_path.name else "as-shipped",
            "reader": args.reader,
            "reading_method": args.reading,
            "judge_model": args.judge_model,
            "judge_pinned": args.judge_model == lme.JUDGE_MODEL_PIN,
            "n_population": population,
            "n_answered": len(rows),
            "complete": complete,
            "missing_question_ids": missing,
            "workers": args.workers,
            "full_context": True,
            "haystack_sorted_by_date": True,
            "max_sessions": None,
            "max_history_tokens": None,
            "organizers": ["passthrough"],
            "embedder": "fake (retrieval is not exercised on the full-context path)",
            "gen_max_tokens": (
                lme.GEN_MAX_TOKENS_CON if args.reading == "con" else lme.GEN_MAX_TOKENS_DIRECT
            ),
            "gen_temperature": (None if reader_spec.fixed_sampling else lme.GEN_TEMPERATURE),
            "deviations": [
                "D2_ingest_per_turn",
                *(
                    ["D6_reader_is_fixed_sampling_temperature_omitted"]
                    if reader_spec.fixed_sampling
                    else []
                ),
            ],
            "git_sha": helpers.git_sha(),
            "utc_started": utc_started,
            "utc_finished": datetime.now(UTC).isoformat(),
            "dry_run": bool(args.dry_run),
            "resumed_from_records": len(kept),
            "max_spend_usd": args.max_spend_usd,
            "prior_spend_usd": round(prior_spend, 6),
        },
        # All three official numbers, always, with their labels (P1): a headline
        # that does not say which accuracy it is cannot be compared against.
        "aggregate": lme.aggregate(rows) if complete else None,
        "aggregate_partial": None if complete else lme.aggregate(rows),
        "llm_budget": folded,
        "llm_budget_by_row": raw_budget,
        "timing": {"eval_s": elapsed},
        "records_file": f"{args.tag}.records.jsonl",
        "trace_file": f"{args.tag}.llm-trace.jsonl",
    }
    if args.dry_run:
        summary["dry_run_quote"] = _quote(client, len(todo), args, reader_spec, judge_spec)
    else:
        this_process = helpers.cost_usd(folded, args.reader, judge_model=args.judge_model)
        summary["cost_usd"] = round(this_process + prior_spend, 6)
        summary["cost_usd_this_process"] = round(this_process, 6)

    (out_dir / f"{args.tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=helpers.json_safe)
    )
    agg = summary["aggregate"] or summary["aggregate_partial"] or {}
    log.info(
        "[done] task_averaged=%s overall=%s abstention=%s n=%s -> %s",
        agg.get("task_averaged"),
        agg.get("overall"),
        (agg.get("abstention") or {}).get("acc"),
        agg.get("n"),
        out_dir / f"{args.tag}.json",
    )


def _spend_from_trace(trace_path: Path, reader: str, judge: str, helpers) -> float:
    """USD spent by earlier processes of this run, read off the trace.

    A resumed run's tracker starts at zero, so a cap enforced on it alone would
    be a cap per process rather than per measurement. Priced per LINE by the
    model that line names, because this run has two models and they differ by
    16.7x on output."""
    if not trace_path.exists():
        return 0.0
    per_model: dict[str, dict] = {}
    for line in trace_path.open(encoding="utf-8"):
        try:
            call = json.loads(line)
        except json.JSONDecodeError:
            break
        row = per_model.setdefault(str(call.get("model")), {"tokens_in": 0, "tokens_out": 0})
        row["tokens_in"] += call.get("tokens_in") or 0
        row["tokens_out"] += call.get("tokens_out") or 0
    total = 0.0
    for model, row in per_model.items():
        if model not in (reader, judge):
            continue
        total += helpers.cost_usd({"generate": row}, model)
    return total


def _quote(client: DryRunLLM, n_rows: int, args, reader_spec, judge_spec) -> dict:
    """Price the dry run's exact call ledger at registry rates.

    Input tokens are the rendered prompts' characters over the measured 4.610
    chars/token (§3.1) — the prompts are real, so only the ratio is an estimate.
    Output is priced at the benchmark's `max_tokens` ceiling, which makes the
    total an UPPER bound; a real answer is shorter."""
    quote: dict[str, Any] = {"chars_per_token": CHARS_PER_TOKEN, "per_role": {}, "usd": 0.0}
    for role, calls in client.calls.items():
        spec = judge_spec if role == "judge" else reader_spec
        tokens_in = client.prompt_chars[role] / CHARS_PER_TOKEN
        tokens_out = calls * client.max_tokens.get(role, 0)
        usd = tokens_in / 1e6 * spec.usd_per_1m_in + tokens_out / 1e6 * spec.usd_per_1m_out
        quote["per_role"][role] = {
            "model": spec.name,
            "calls": calls,
            "prompt_chars": client.prompt_chars[role],
            "tokens_in_est": round(tokens_in),
            "tokens_out_ceiling": tokens_out,
            "usd": round(usd, 4),
        }
        quote["usd"] += usd
    quote["usd"] = round(quote["usd"], 4)
    quote["rows"] = n_rows
    quote["note"] = (
        "upper bound: completions priced at max_tokens. The judge's input is "
        "slightly under-counted (canned hypothesis)."
    )
    return quote


if __name__ == "__main__":
    main()
