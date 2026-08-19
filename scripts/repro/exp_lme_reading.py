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
import gzip
import hashlib
import importlib.util
import json
import logging
import queue
import sys
import threading
import time
from collections.abc import Iterable
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
    # `_m` has no full-context arm to offer: ~1.1M tokens per instance against a
    # 128k window, so `--retrieval` is not one option here but the only one. That
    # is exactly why it is the regime worth measuring — see §9.4 weakness 0.
    "m": Path.home() / ".agmem/datasets/longmemeval_m_cleaned.json",
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
    """Rows already paid for AND judged, and their question ids.

    The atomic unit here is the ROW, not a window: instances are independent (one
    fresh memory each), a row is written only after its judge verdict is in, and
    a truncated final line — the shape a kill leaves — is dropped. So a resumed
    run re-buys at most one question.

    A row whose `label` is null is NOT resumable, and this is the subtle half: an
    API failure writes such a row so the failure cannot vanish (LME-A18), but
    treating it as done would leave a permanent hole that a row COUNT cannot see
    — 500 rows of which one is unjudged, aggregated over 499 while the stamp says
    complete. Those rows are dropped here and re-answered, and the caller rewrites
    the file from what this returns so the hole cannot survive as a duplicate id
    either."""
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
    judged = [r for r in rows if r.get("label") is not None]
    if len(judged) != len(rows):
        log.info(
            "%d earlier rows carry no verdict — they will be re-answered", len(rows) - len(judged)
        )
    return judged, {str(r.get("question_id")) for r in judged}


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """The stamped dataset fingerprint, read in chunks.

    `path.read_bytes()` is 2.74 GB in one allocation on `_m` — the same class of
    mistake as loading the instances all at once, in the one line that exists to
    prove which bytes were measured."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def run_pool(
    instances: Iterable[dict],
    total: int,
    *,
    build_mem,
    reading: str,
    history_format: str,
    retrieval_k: int | None,
    budget_tokens: int,
    embed_batch: int | None,
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
    flight (`workers`), not by the split.

    The queue is BOUNDED and fed by a producer thread, so `instances` can be a
    generator reading the file as it goes. Draining it into the queue up front is
    what the oracle/`_s` version did, and it is the thing that made `_m`
    unrunnable: 500 `_m` instances resident at once is ~4.6 GB. `total` is passed
    separately because a generator has no length.

    Threads, not processes, and no deep copy of an instance anywhere — the two
    other rules `_s`-scale data imposes (docs/research/longmemeval.md §10.2)."""
    work: queue.Queue = queue.Queue(maxsize=max(workers * 2, 4))
    rows: list[dict] = []
    stop = threading.Event()
    DONE = object()  # a bounded queue cannot say "empty means finished"

    def producer() -> None:
        try:
            for inst in instances:
                while not stop.is_set():
                    try:
                        work.put(inst, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        finally:
            # One sentinel per worker, whatever happened above — a producer that
            # dies without these leaves every worker blocked on `get` forever.
            for _ in range(workers):
                while True:
                    try:
                        work.put(DONE, timeout=0.5)
                        break
                    except queue.Full:
                        if stop.is_set():
                            return

    def worker() -> None:
        try:
            _worker_loop()
        except BaseException:
            # A worker that dies takes its question with it and leaves a run that
            # looks merely short. `threading` would print this to stderr and
            # nothing else would ever mention it, so it goes through the run's own
            # logger — the one whose output the log file captures.
            log.exception("worker thread died; its question will show up as missing")
            raise

    def _worker_loop() -> None:
        while not stop.is_set():
            try:
                inst = work.get(timeout=0.5)
            except queue.Empty:
                continue
            if inst is DONE:
                return
            if over_budget is not None and over_budget():
                if not stop.is_set():
                    log.warning("spend cap reached — stopping cleanly")
                stop.set()
                return
            qid = str(inst["question_id"])
            t0 = time.perf_counter()
            mem = None
            try:
                # INSIDE the try, and this is not a stylistic choice. It used to
                # sit outside: a `build_mem` that raised killed the worker thread
                # while it still held a question taken off the queue, so the
                # question vanished with no row and no log line — the completeness
                # check would report it 14 hours later as `missing`. On `_m` this
                # was not hypothetical: `kuzu.Database(":memory:")` reserves 8 TiB
                # of virtual address space per instance, so at workers=20 five of
                # the twenty threads died at startup with "Mmap for size
                # 8796093022208 failed" and took five questions with them. The VA
                # ceiling caps this arm near 15 workers regardless of RAM.
                mem = build_mem(qid)
                row = lme.run_instance(
                    mem,
                    inst,
                    reading_method=reading,
                    # The four guards that live at the call site (§8.5 1,2,4,6):
                    # the instance arrives sorted, no session cap exists, `mem` is
                    # this question's own, and `max_history_tokens` is not passed.
                    max_sessions=None,
                    embed_batch=embed_batch,
                    # `retrieval_k` is what makes this a memory arm rather than a
                    # reading arm: the haystack goes through the store and only
                    # what comes back out reaches the prompt.
                    full_context=retrieval_k is None,
                    history_format=history_format,
                    k=retrieval_k or 10,
                    budget_tokens=budget_tokens,
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
                # `mem` is None when build_mem itself raised — closing it would
                # replace the recorded failure with an AttributeError and lose the
                # row all over again.
                if mem is not None:
                    mem.close()
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            capture = row.pop("retrieval", None) or {}
            prompt = capture.get("prompt") or ""
            if retrieval_k is not None:
                # The retrieved chunk list is the arm's whole mechanism, so it is
                # recorded per row: ids and scores only, because the text they
                # rendered to is already in the prompt the trace holds verbatim.
                row["retrieved"] = [
                    {
                        "id": c.get("id"),
                        "memory_type": c.get("memory_type"),
                        "score": c.get("score"),
                        # The haystack coordinates, not just the memory id. Without
                        # these a drop in accuracy cannot be split into "the
                        # evidence never arrived" and "it arrived and the reader
                        # did not use it" — which is exactly what the `_m` arm
                        # could not do with its 8.60 pp (docs/20, C7). They are
                        # free to record and unrecoverable without paying again.
                        "session_ids": c.get("session_ids"),
                    }
                    for c in capture.get("retrieved", [])
                ]
                # Gold is a set of SESSION ids, so recall is scored per session,
                # not per item. `None` means this arm's memories carry no session
                # provenance at all (an organizer summarising many sessions may
                # legitimately have none) — a different statement from 0.0, and
                # kept different.
                gold = set(lme.evidence_session_ids(inst))
                got = {s for c in row["retrieved"] for s in (c.get("session_ids") or [])}
                row["evidence_recall"] = (
                    None
                    if not gold or not any(c.get("session_ids") for c in row["retrieved"])
                    else round(len(gold & got) / len(gold), 4)
                )
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
                log.info("%d/%d answered", done, total)

    feeder = threading.Thread(target=producer, daemon=True)
    feeder.start()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()  # releases the producer if the cap ended the run early
    feeder.join(timeout=5)
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
    ap.add_argument(
        "--history-format",
        choices=["json", "nl"],
        default="json",
        help="upstream's flag: json is run_generation.sh's default; §5.5 reports the "
        "two interact with the reading method by up to 10pp",
    )
    ap.add_argument(
        "--retrieval",
        type=int,
        default=None,
        metavar="K",
        help="answer from the MEMORY's top-K instead of the full haystack. This turns a "
        "reading arm into a memory arm: with the evidence-only oracle it measures what a "
        "retrieval layer LOSES when the whole haystack already fits, which is a tax, not a "
        "benefit. Needs a real embedder.",
    )
    ap.add_argument("--budget-tokens", type=int, default=6000, help="render budget for --retrieval")
    ap.add_argument(
        "--embedder",
        default="text-embedding-3-small",
        help="only used with --retrieval; the full-context path never queries a vector",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--embed-batch",
        type=int,
        default=None,
        help="batch the ingest embedding calls N at a time (25x faster on _m). NOT "
        "comparable with a per-turn arm: batched vectors are not bit-identical "
        "(cosine 0.999999546), which reorders near-ties. Recorded in the stamp.",
    )
    ap.add_argument("--limit", type=int, default=None, help="first N questions (smoke runs)")
    ap.add_argument("--tag", default=None, help="artifact tag; defaults to <reader>_lme_<ds>_<rm>")
    ap.add_argument(
        "--max-spend-usd",
        type=float,
        default=None,
        help="hard ceiling for this measurement, counting what earlier processes "
        "of the same run already spent (recovered from the trace)",
    )
    ap.add_argument(
        "--gzip-trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="gzip the full-I/O trace (default: on for --dataset s, off for oracle)",
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
    suffix = "" if args.retrieval is None else f"_k{args.retrieval}"
    suffix += "" if args.history_format == "json" else f"_{args.history_format}"
    args.tag = args.tag or f"{args.reader}_lme_{args.dataset}_{args.reading}{suffix}"
    # A dry run writes the same filenames with none of the meaning, so it writes
    # them somewhere else: `results/repro/*.records.jsonl` is what `run_status.py`
    # reads a run's state from, and a $0 quote sitting at a paid arm's tag would
    # both block the real run's resume check and report itself as a finished
    # measurement. The subdirectory is outside that glob.
    out_dir = OUT / "dryrun" if args.dry_run else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / f"{args.tag}.records.jsonl"
    # `_s` sends 500 prompts of ~525 KB, so its trace is ~260 MB uncompressed and
    # every tool that prices a running arm would have to stream all of it. The
    # oracle arms stay uncompressed: 15 MB reads instantly and `zcat` is one more
    # thing to remember.
    gz = args.gzip_trace if args.gzip_trace is not None else args.dataset in ("s", "m")
    trace_path = out_dir / f"{args.tag}.llm-trace.jsonl{'.gz' if gz else ''}"

    # Pass 1 of 2, and the reason both exist: `_m` is 2.74 GB, its instance list
    # is ~4.6 GB against ~4.5 GB free, and this driver used to hold all of it at
    # once. Nothing but the question identity is kept here; the haystacks are
    # streamed again in pass 2, one instance at a time, straight into the pool.
    # On oracle/`_s` the second scan costs 1.4s/21s, which is not worth a second
    # code path for.
    # `--limit` stops the scan, not just the slice: a 3-question smoke on `_m`
    # should not read 2.74 GB to find out there are 500 instances in it.
    catalogue = []
    for inst in lme.iter_longmemeval(data_path):
        catalogue.append((str(inst["question_id"]), str(inst.get("question_type", ""))))
        if args.limit is not None and len(catalogue) >= args.limit:
            break
    population = len(catalogue)
    all_qids = {qid for qid, _ in catalogue}
    data_sha = _sha256_file(data_path)
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
    todo_ids = [qid for qid, _ in catalogue if qid not in done_ids]

    def todo_instances():
        """Pass 2: the haystacks, one at a time, sorted before anything reads them.

        Sorting is unconditional here because upstream's is, and it has to happen
        before the haystack reaches ingest or `render_sessions`. `limit` is applied
        by position, matching pass 1's slice, so the two passes agree on which
        instances the run covers."""
        wanted = set(todo_ids)
        for i, inst in enumerate(lme.iter_longmemeval(data_path)):
            if i >= population:
                return
            if str(inst["question_id"]) in wanted:
                yield lme.sort_haystack_by_date(inst)

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

    # A fake embedder is correct for the full-context path (nothing is ever
    # queried) and WRONG for a retrieval arm, where the vector is the mechanism.
    # Built once and shared: it is stateless, and 500 clients would be 500 pools.
    embedder = (
        FakeEmbedder(dim=64)
        if args.retrieval is None or args.dry_run
        else helpers.build_embedder(args.embedder)
    )

    def build_mem(qid: str) -> AgenticMemory:
        # data_dir=None -> every store in memory, so 500 fresh stores cost no
        # disk and leave nothing to clean up. sync_write keeps the organizer
        # hooks on the calling thread (passthrough has none, but a background
        # worker per instance would be 500 threads).
        mem = AgenticMemory(
            namespace=f"lme-{qid}",
            organizers=["passthrough"],
            embedder=embedder,
            config=AgmemConfig(profile="lite", data_dir=None, sync_write=True),
            caps=caps,
        )
        mem.llm = client
        return mem

    prior_spend = 0.0
    over_budget = None
    if args.max_spend_usd is not None and not args.dry_run:
        prior_spend = _prior_spend(
            trace_path, out_dir / f"{args.tag}.json", args.reader, args.judge_model, helpers
        )
        if prior_spend:
            log.info("earlier processes of this run spent $%.4f (from the trace)", prior_spend)

        def over_budget() -> bool:
            # Folds the embedder in too, so the ceiling binds the whole bill and
            # not just the chat half of it — a retrieval arm pays for every turn
            # it indexes, and the cap that ignores that is not the cap that was
            # quoted.
            live = fold_row_keys(budget.summary())
            helpers.fold_embed_budget(live, embedder)
            spent = prior_spend + helpers.cost_usd(
                live,
                args.reader,
                judge_model=args.judge_model,
                embed_model=helpers.embed_model_name(embedder),
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
    if kept:
        # Rewrite from the judged rows before appending, so an unjudged row from
        # an earlier attempt cannot survive alongside the answer that replaces it
        # and give one question two rows in the file that gets aggregated.
        with records_path.open("w", encoding="utf-8") as fh:
            for row in kept:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    sink = records_path.open("a" if kept else "w", encoding="utf-8")
    try:
        new_rows = run_pool(
            todo_instances(),
            len(todo_ids),
            build_mem=build_mem,
            reading=args.reading,
            history_format=args.history_format,
            retrieval_k=args.retrieval,
            budget_tokens=args.budget_tokens,
            embed_batch=args.embed_batch,
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
    #
    # The population is checked against JUDGED rows, not against row count. A
    # failed question still writes a row (so the failure is on the record), and
    # `aggregate` correctly excludes an unjudged row from every bucket — so a
    # count-only check would call a run complete and then quietly score it over
    # 499. The two ways of ending up short are the same defect and get the same
    # answer: no score.
    judged_ids = {str(r["question_id"]) for r in rows if r.get("label") is not None}
    missing = sorted(all_qids - judged_ids)
    complete = not missing
    if not complete:
        log.error(
            "INCOMPLETE: %d/%d judged (%d rows written) — no score is reported. Missing: %s",
            len(judged_ids),
            population,
            len(rows),
            ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""),
        )

    raw_budget = budget.summary()
    folded = fold_row_keys(raw_budget)
    # A retrieval arm embeds every turn it ingests, and those calls are NOT in the
    # chat budget. Left out, the arm would look cheaper than it is and could not
    # explain its own total.
    helpers.fold_embed_budget(folded, embedder)
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
            "n_judged": len(judged_ids),
            "complete": complete,
            "missing_question_ids": missing,
            "workers": args.workers,
            "full_context": args.retrieval is None,
            "haystack_sorted_by_date": True,
            "max_sessions": None,
            "max_history_tokens": None,
            "organizers": ["passthrough"],
            "read_path": "full_context"
            if args.retrieval is None
            else f"retrieval_top{args.retrieval}",
            "retrieval_k": args.retrieval,
            "budget_tokens": None if args.retrieval is None else args.budget_tokens,
            # Which embedding regime produced the index. Batched and per-turn arms
            # are not comparable to each other (see `lme.ingest`), so this is not a
            # tuning knob but a fact the paired analysis has to check.
            "embed_batch": args.embed_batch,
            "history_format": args.history_format,
            "embedder": (
                "fake (retrieval is not exercised on the full-context path)"
                if args.retrieval is None
                else args.embedder
            ),
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
        "trace_file": trace_path.name,
    }
    if args.dry_run:
        summary["dry_run_quote"] = _quote(client, len(todo_ids), args, reader_spec, judge_spec)
    else:
        this_process = helpers.cost_usd(
            folded,
            args.reader,
            judge_model=args.judge_model,
            embed_model=helpers.embed_model_name(embedder),
        )
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


def _prior_spend(trace_path: Path, summary_path: Path, reader: str, judge: str, helpers) -> float:
    """USD spent by earlier processes of this run — the larger of two records.

    The trace is the one that survives a kill, but it only holds CHAT calls: the
    embedder does not route through `LLMClient`, so a retrieval arm's embedding
    spend is invisible there. On the `_s` top-50 arm that was $1.11 of a $2.59
    total, and a resumed process would have reported $1.51 for a measurement that
    had already cost $2.59 — an under-count in the direction that matters, since
    this number is also what the spend cap is enforced against.

    So: the trace when the earlier process died without writing a summary, and
    the earlier summary's own total when it wrote one. Neither can overstate, so
    the max is the honest estimate."""
    from_summary = 0.0
    if summary_path.exists():
        try:
            from_summary = float(json.loads(summary_path.read_text()).get("cost_usd") or 0.0)
        except (json.JSONDecodeError, TypeError, ValueError):
            from_summary = 0.0
    if not trace_path.exists():
        return from_summary
    per_model: dict[str, dict] = {}
    with (
        gzip.open(trace_path, "rt", encoding="utf-8")
        if trace_path.suffix == ".gz"
        else trace_path.open(encoding="utf-8")
    ) as fh:
        for line in fh:
            try:
                call = json.loads(line)
            except (json.JSONDecodeError, EOFError, OSError):
                break
            row = per_model.setdefault(str(call.get("model")), {"tokens_in": 0, "tokens_out": 0})
            row["tokens_in"] += call.get("tokens_in") or 0
            row["tokens_out"] += call.get("tokens_out") or 0
    total = 0.0
    for model, row in per_model.items():
        if model not in (reader, judge):
            continue
        total += helpers.cost_usd({"generate": row}, model)
    return max(total, from_summary)


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
