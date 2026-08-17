"""Track 5 runner: ACE's online adaptation on FiNER, with the artifacts.

The shape is upstream's `--mode online` (ace.py:877-1141), which is the setting
its README states the FiNER claims in: the split is walked in windows, each
window is answered with the playbook as it stands and only then learned from, so
the number is a measure of adaptation rather than of a finished artifact.

Two arms, one dataset:

  base     441 test samples, empty playbook, no learning. The honest "before" —
           online window 1 is only 15 samples and cannot carry a headline.
  online   441 test samples in windows of 15, each window answered then trained
           on. The "after" is the last window; the curve between is the result.

Artifact capture is the campaign's, not this script's: trace, records sidecar,
memory snapshot, op log, cost, timing and a self-describing stamp, written
through the same helpers `exp_amem_repro.py` uses so a FiNER run can be read by
the same eyes as a LoCoMo one. Nothing here re-implements a writer — that is
how the snapshot roster came to be missing a whole memory type (ledger C-8).

Deviations from upstream, all disclosed in the stamp:

- **D3 (one attempt per sample)**: upstream's online loop answers a window for
  scoring and then calls the generator AGAIN inside its training step
  (ace.py:1009+ -> `_train_single_sample`, :470). We reflect on the attempt
  that was already scored. Our boundary is `on_task_end` — a task that has
  finished — and the scored attempt is that task; upstream regenerates only
  because its train step is shared with offline mode, where no scored attempt
  exists. The consequence is not just cost: upstream's regeneration can flip a
  sample to correct *within* the training step, and ours cannot, so its
  reflection sees outcomes ours never produces.
- **D4 (no reflection retries)**: upstream loops reflect+regenerate up to
  `max_num_rounds`=3 on an incorrect answer (ace.py:498-543). We reflect once by
  default, so per-sample adaptation cost is fixed at 3 calls where upstream's is
  5-10 and rises with the error rate — a *worse* model costs upstream strictly
  more to adapt, which is the asymmetry any latency or token-cost comparison
  between the two has to state. **`--max-rounds 3` turns the loop on**, which is
  the arm that tests whether those retries are what makes the mechanism pay; the
  two calls upstream spends re-answering a question it has already answered
  (ace.py:465-472) and re-answering it once more after curating (:608-625) stay
  unreproduced, because neither changes what is learned.
- **D5 (dedup always on)**: ledger B-6. Upstream's analyzer is opt-in, off in
  `ace.py`, off in this harness's flag, and off again by silent fallback when
  its optional dependencies are missing; ours dedups at 0.90 always and drops
  the incoming bullet rather than LLM-merging a group.

`--dry-run` executes the entire loop against a deterministic fake client and
spends nothing. It is not a smoke test of the wiring only: it reports the exact
call count and prompt-character total per role, which is the input to the
spend quote.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agmem import AgenticMemory
from agmem._env import load_env_local
from agmem.bench import finer
from agmem.config import AgmemConfig
from agmem.organizers.ace import ACEOrganizer
from agmem.organizers.reasoning_bank.organizer import (
    DEFAULT_MAX_ITEMS as RB_DEFAULT_MAX_ITEMS,
)
from agmem.organizers.reasoning_bank.organizer import ReasoningBankOrganizer

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "repro"
UPSTREAM_DATA = Path.home() / ".agmem/upstream/ace/eval/finance/data"

# ACE's own temperatures are not in the paper; the release passes whatever the
# provider defaults to and pins nothing (ace.py:33-93 takes model names only).
# We therefore pin the campaign's, and say so, rather than inheriting a runner
# default silently — the mistake `keyword_queries` taught this campaign twice.
ACE_ROLE_TEMPS = {
    "generate": {"temperature": 0.0},
    "distill": {"temperature": 0.0, "max_tokens": 4096},
}

# The organizer this runner drives, paired with the read path that belongs to it.
# Track 4 added the second entry; the first is the one four paid arms were bought
# through, and `tests/test_finer_runner.py` pins that selecting it changes nothing.
#
# The two are paired here rather than exposed as separate flags on purpose. ACE's
# read contract is whole-playbook injection; ReasoningBank's pinned operating
# point is top-1 (`RB_READ_RECIPE`). A CLI that let them be mixed would let an arm
# claim a lineage it did not actually read in.
ORGANIZER_READS = {"ace": "playbook", "rb": "experiences"}

# Items the dry-run fake emits per RB extraction. Imported from the organizer rather
# than restated so a quote cannot be built against a budget the prompt no longer
# advertises — upstream's SUCCESSFUL_SI/FAILED_SI say "at most 3" and the schema's
# maxItems is rendered from the same knob (round-12 #15).
RB_DRY_RUN_ITEMS = RB_DEFAULT_MAX_ITEMS


def _load_repro_helpers():
    """Import `scripts/exp_amem_repro.py` for its artifact writers.

    A path import rather than a package one: `scripts/` is not a package, and
    the alternative — copying `cost_usd`, `dump_memory_snapshot`, `dump_op_log`
    and `json_safe` into this file — is the duplication that ledger C-8 is
    about."""
    path = ROOT / "scripts" / "exp_amem_repro.py"
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("exp_amem_repro", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DryRunLLM:
    """Deterministic stand-in that answers every role without a network call.

    It counts calls and prompt characters per role, which is what turns a dry
    run into a quote. The accuracy a dry run reports is meaningless — the fake
    never sees a gold answer, so every sample scores wrong — but the CALL
    LEDGER is exact, because our adaptation shape spends the same three calls
    per sample whether the answer was right or not (D4). That is the property
    the quote rests on, and it is the difference from upstream, whose per-sample
    cost rises with its error rate."""

    def __init__(self, growth: str = "none"):
        # The one thing a dry run genuinely cannot predict is how many curated
        # bullets survive dedup, and the playbook is injected into all three
        # calls, so it dominates the bill. `growth` brackets it instead of
        # guessing: "none" emits one bullet the deduper always rejects (the
        # floor, a playbook that never grows), "max" emits lexically disjoint
        # bullets that always survive (the ceiling, one per sample). The real
        # run lands between, and a paid smoke is what closes the interval.
        self.growth = growth
        self.calls: dict[str, int] = {}
        self.prompt_chars: dict[str, int] = {}
        self.reply_chars: dict[str, int] = {}
        self.drops: dict[str, int] = {}
        self._n = 0
        self._curated = 0

    def call(self, role, prompt, schema, required_keys=(), **kwargs):
        self.calls[role] = self.calls.get(role, 0) + 1
        self.prompt_chars[role] = self.prompt_chars.get(role, 0) + len(prompt)
        if role == "generate":
            self._n += 1
            reply = {
                "reasoning": "dry-run reasoning " * 20,
                "bullet_ids": [],
                "final_answer": "dryrun,dryrun,dryrun,dryrun",
            }
        elif "operations" in str(schema):
            self._curated += 1
            if self.growth == "max":
                # Lexically disjoint from every other bullet, so the bag-of-words
                # deduper cannot collapse it. Length is held near a realistic
                # bullet so the ceiling prices a plausible playbook, not a toy.
                body = " ".join(f"w{self._curated}x{j}" for j in range(18))
            else:
                body = "Dry-run insight about tag selection, repeated verbatim."
            reply = {"operations": [{"type": "ADD", "section": "gaap_tagging", "content": body}]}
        elif isinstance(schema, dict) and schema.get("required") == ["items"]:
            # Matched on the schema's own `required`, NOT on a substring of it:
            # `items` is a JSON Schema keyword that every array property carries,
            # so `"items" in str(schema)` would also catch ACE's reflector (its
            # `bullet_tags` is an array) and answer it with the wrong shape.
            #
            # ReasoningBank's extraction (track 4). Answered explicitly rather than
            # falling through to the reflector shape below: `on_task_end` demands
            # `items`, so a reply without it is a DROP, and a dropped call is
            # subtracted from the very ledger the quote is read off — the dry run
            # would have priced an arm that learned nothing.
            #
            # `DEFAULT_MAX_ITEMS` items every time, because RB has no dedup gate:
            # what the curator proposes is what the store keeps, so the item count
            # is deterministic where ACE's needed bracketing. `growth` still varies
            # their text — disjoint under "max" so the vector store holds distinct
            # candidates for the top-1 read to choose between — but the interval it
            # brackets is NARROW here, and that is a finding rather than a
            # convenience: the read serves ONE item, so store size does not ride
            # into every prompt the way ACE's whole playbook does.
            self._curated += 1
            if self.growth == "max":
                body = " ".join(f"rb{self._curated}y{j}" for j in range(18))
            else:
                body = "Dry-run strategy about tag selection, repeated verbatim."
            reply = {
                "items": [
                    {
                        "title": f"Dry-run item {self._curated}-{i}",
                        "description": "A transferable step distilled from the trajectory.",
                        "content": body,
                    }
                    for i in range(RB_DRY_RUN_ITEMS)
                ]
            }
        else:
            reply = {
                "reasoning": "r" * 200,
                "error_identification": "e" * 100,
                "root_cause_analysis": "c" * 100,
                "correct_approach": "a" * 100,
                "key_insight": f"insight {self._n}",
                "bullet_tags": [],
            }
        self.reply_chars[role] = self.reply_chars.get(role, 0) + len(json.dumps(reply))
        return reply


def build_memory(args, embedder, trace_path: Path | None, roles) -> AgenticMemory:
    cfg = AgmemConfig(
        profile="lite",
        data_dir=Path(args.data_dir).expanduser() if args.data_dir else None,
        llm_roles=roles,
        use_guided_json=False,
        sync_write=True,
    )
    # The ACE branch is spelled exactly as it was before Track 4 threaded a second
    # organizer through here, because four bought arms are that call.
    if args.organizer == "rb":
        organizers = [ReasoningBankOrganizer()]
    else:
        organizers = [
            ACEOrganizer(dedup_threshold=args.dedup_threshold, max_rounds=args.max_rounds)
        ]
    mem = AgenticMemory(
        namespace=args.namespace,
        organizers=organizers,
        embedder=embedder,
        config=cfg,
    )
    if trace_path is not None and mem.llm is not None:
        mem.llm.trace_path = trace_path
    return mem


def run_arm(
    mem: AgenticMemory,
    samples: list[dict[str, Any]],
    *,
    adapt: bool,
    window: int,
    log: logging.Logger,
    sink=None,
    skip_before: int = 0,
    over_budget=None,
    retries: bool = False,
    memory_source=None,
) -> tuple[list[dict], list[dict]]:
    """Walk the split. Returns (rows, per_window).

    With `adapt=False` this is upstream's `eval_only` over the whole split; the
    window boundaries still exist so the two arms produce comparable curves,
    but nothing is learned between them.

    `memory_source` selects which read path fills the generator's memory slot;
    None means `finer.read_playbook`, which is ACE's contract and what the four
    docs/19 arms were bought through."""
    read = memory_source or finer.read_playbook
    rows: list[dict] = []
    per_window: list[dict] = []
    for start, chunk in finer.windows(samples, window):
        if start < skip_before:
            continue
        # Checked BETWEEN windows, never inside one: a window that is answered
        # but not adapted on cannot be written (see the sink ordering below), so
        # stopping mid-window would throw away calls already paid for.
        if over_budget is not None and over_budget():
            log.warning(
                "spend cap reached before window %d — stopping cleanly with %d samples done",
                start // window,
                start,
            )
            break
        w_rows: list[dict] = []
        for sample in chunk:
            capture: dict[str, Any] = {}
            try:
                pred, _bullets = finer.answer(mem, sample, capture=capture, memory_source=read)
                failed = False
            except Exception as exc:  # noqa: BLE001 - a failure must score, not vanish
                # Upstream drops a raising sample out of the denominator
                # (utils.py:248-250). Ours scores it wrong and records why, so
                # the population stays the split it claims to be.
                log.warning(
                    "sample %d failed: %s: %s", start + len(w_rows), type(exc).__name__, exc
                )
                pred, failed = finer.NO_ANSWER, True
            row = finer.score_sample(sample, pred, failed=failed)
            row["window"] = start // window
            row["index"] = start + len(w_rows)
            row["playbook_chars"] = capture.get("playbook_chars", 0)
            row["bullet_ids"] = capture.get("bullet_ids", [])
            # Carried on the row so `finer.adapt` can hand it to the reflector:
            # the reasoning IS what a reflection diagnoses, and without it the
            # curated bullets come back as generic process advice.
            row["reasoning"] = capture.get("reasoning", "")
            w_rows.append(row)
        rows.extend(w_rows)
        agg = finer.aggregate(w_rows)
        agg["window"] = start // window
        agg["start"] = start
        agg["playbook_chars_at_test"] = w_rows[0]["playbook_chars"] if w_rows else 0
        per_window.append(agg)
        log.info(
            "window %d [%d:%d] tag=%.2f sample=%.2f playbook_chars=%d",
            agg["window"],
            start,
            start + len(chunk),
            agg["tag_accuracy"],
            agg["sample_accuracy"],
            agg["playbook_chars_at_test"],
        )
        if adapt:
            for sample, row in zip(chunk, w_rows):
                # Retries never touch the SCORED answer — `row` is already
                # graded and appended. They only change what the organizer sees
                # while learning, which is the whole point of the arm, and each
                # extra attempt is recorded so the artifact says what they did.
                attempts: list[dict] = []
                finer.adapt(
                    mem,
                    sample,
                    row,
                    retries=retries,
                    attempts=attempts,
                    memory_source=read,
                )
                if attempts:
                    row["retry_attempts"] = [
                        {"pred": a["pred"], "is_correct": a["is_correct"]} for a in attempts
                    ]
                    row["corrected_in_training"] = any(a["is_correct"] for a in attempts)
            mem.flush()
        # Persist AFTER adapt, never before: a window present in the records file
        # must mean the store has also learned from it, or a resume would answer
        # with a playbook that is behind the transcript. This ordering is what
        # makes the window the atomic unit of progress.
        if sink is not None:
            for row in w_rows:
                sink.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            sink.flush()
    return rows, per_window


def spend_from_trace(trace_path: Path, model: str, helpers) -> float:
    """USD already spent by earlier processes of this run, read off the trace.

    A resumed run's `mem.budget` starts at zero, so a spend cap enforced on it
    alone would be a cap per process rather than per measurement — and the whole
    point of the cap is that it binds the thing being bought. The trace is
    appended per call and survives whatever killed the process, so it is the one
    record of what the earlier attempts cost. A truncated final line is dropped.
    """
    if not trace_path.exists():
        return 0.0
    budget: dict[str, dict] = {}
    for line in trace_path.open(encoding="utf-8"):
        try:
            call = json.loads(line)
        except json.JSONDecodeError:
            break
        row = budget.setdefault(
            call.get("role", "?"), {"calls": 0, "tokens_in": 0, "tokens_out": 0}
        )
        row["calls"] += 1
        row["tokens_in"] += call.get("tokens_in") or 0
        row["tokens_out"] += call.get("tokens_out") or 0
    return helpers.cost_usd(budget, model) if budget else 0.0


def resume_state(records_path: Path, samples: list[dict], window: int, log) -> list[dict]:
    """Rows from a previous attempt that may be kept, truncated to a window boundary.

    This exists because the host has now taken three long runs down mid-flight —
    the process leaves no traceback, the machine simply reboots — and re-buying a
    measurement that was already paid for is the one cost this campaign refuses.

    The unit of progress is the WINDOW, not the sample. `run_arm` appends a
    window's rows only after adapting on them, so "window w is in the file"
    implies "the store has learned from window w". A partial window is therefore
    discarded rather than trusted: its samples were answered with the right
    playbook but never trained on, and keeping them would leave the transcript
    ahead of the store it claims to describe. Those few samples are re-answered
    and re-paid for; that is the price of the invariant.

    Kept rows are checked against the dataset prefix, not assumed: a records file
    from a different split or a different order would otherwise be silently
    stitched onto this run.
    """
    if not records_path.exists():
        return []
    rows = []
    for lineno, line in enumerate(records_path.open(encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # The crash can land mid-write. Everything after the first bad line
            # is unusable, and stopping here is what makes that safe.
            log.warning("records line %d is truncated — resuming from before it", lineno)
            break
    if not rows:
        return []
    keep = (len(rows) // window) * window
    if keep != len(rows):
        log.info("discarding %d rows of an incomplete window", len(rows) - keep)
    rows = rows[:keep]
    for i, row in enumerate(rows):
        if row.get("target") != samples[i]["target"]:
            raise SystemExit(
                f"resume refused: row {i} of {records_path.name} is not sample {i} of this split"
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["base", "online"], required=True)
    ap.add_argument(
        "--organizer",
        choices=sorted(ORGANIZER_READS),
        default="ace",
        help=(
            "which self-evolving memory to grow alongside (default ace, the four docs/19 arms). "
            "'rb' is ReasoningBank on the same 441 questions — NOT a reproduction of its published "
            "agentic claims, which need WebArena or SWE-Bench; see the track 4 re-selection note"
        ),
    )
    ap.add_argument("--data", type=Path, default=UPSTREAM_DATA)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="first N samples (smoke runs)")
    ap.add_argument("--window", type=int, default=15, help="online_eval_frequency (ace.py default)")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--endpoint", default="https://api.openai.com/v1")
    ap.add_argument("--embedder", default="text-embedding-3-small")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--tag", default=None, help="artifact tag; defaults to model_ace_finer_<arm>")
    ap.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.90,
        help=(
            "curator dedup cosine gate. 0.90 is OUR always-on default (D5); any value above 1.0 "
            "is unreachable by a cosine and therefore switches dedup OFF, which is upstream's "
            "shipped default (ledger B-6)."
        ),
    )
    ap.add_argument(
        "--max-rounds",
        type=int,
        default=1,
        help=(
            "upstream's `max_num_rounds` (3 in every shipped eval config): how many "
            "reflect-and-re-answer rounds an incorrect sample may take during training. "
            "1 is our measured arms' behaviour — reflect once, never re-answer (D4). "
            "The SCORED answer is always the window test pass and is never a retry."
        ),
    )
    ap.add_argument(
        "--max-spend-usd",
        type=float,
        default=None,
        help=(
            "hard ceiling for this measurement, counting what earlier processes of the same "
            "run already spent (recovered from the trace). Checked between windows; the run "
            "stops cleanly and writes its artifacts rather than overshooting a quote."
        ),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "continue a run the host killed: keep whole windows already in the records file "
            "and reuse the store under --data-dir. Requires the same --data-dir and --tag."
        ),
    )
    ap.add_argument("--dry-run", action="store_true", help="fake client, no network, $0")
    ap.add_argument(
        "--dry-run-growth",
        choices=["none", "max"],
        default="none",
        help="playbook-growth bound for the dry run: none=floor, max=ceiling",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("agmem.repro.finer")
    load_env_local()

    helpers = _load_repro_helpers()
    args.tag = args.tag or f"{args.model}_ace_finer_{args.arm}"
    args.namespace = args.namespace or f"finer-{args.arm}"
    OUT.mkdir(parents=True, exist_ok=True)
    trace_path = OUT / f"{args.tag}.llm-trace.jsonl"

    raw = finer.load_finer(args.data, args.split)
    samples, fell_through = finer.process_task_data(raw)
    if fell_through:
        # Not a warning about our port: upstream's parser is written for the
        # finlora_sentiment format and FiNER carries neither marker, so the
        # `context` slot of the generator prompt is empty for every sample and
        # the whole blob arrives as the question. Recorded because a reader of
        # the prompt would otherwise conclude we dropped the context.
        log.info(
            "%d/%d samples have no Instruction:/Input: markers — upstream's parser "
            "falls through and the generator's Context section is empty (see bench/finer)",
            fell_through,
            len(samples),
        )
    if args.limit:
        samples = samples[: args.limit]
    log.info("arm=%s split=%s n=%d window=%d", args.arm, args.split, len(samples), args.window)

    if args.dry_run:
        embedder, fake = None, DryRunLLM(growth=args.dry_run_growth)
        from agmem.embed.fake import FakeEmbedder

        embedder = FakeEmbedder(dim=128)
        args.api_key = "dry-run"
        mem = build_memory(args, embedder, None, None)
        mem.structured = fake
        mem._ctx.llm = fake
    else:
        from agmem.bench.registry import get_model

        spec = get_model(args.model)
        args.endpoint = spec.endpoint
        args.api_key = helpers.resolve_api_key(spec)
        embedder = helpers.build_embedder(args.embedder)
        roles = helpers.make_roles(
            args.endpoint,
            args.model,
            args.api_key,
            role_temps=ACE_ROLE_TEMPS,
            max_tokens=4096,
        )
        mem = build_memory(args, embedder, trace_path, roles)
        fake = None

    records_path = OUT / f"{args.tag}.records.jsonl"
    kept = resume_state(records_path, samples, args.window, log) if args.resume else []
    if kept:
        log.info(
            "resuming: %d/%d samples already recorded (%d complete windows); "
            "the store under %s carries their playbook",
            len(kept),
            len(samples),
            len(kept) // args.window,
            args.data_dir,
        )
    elif args.resume:
        log.info("--resume given but nothing usable on disk; starting from the beginning")

    prior_spend = 0.0
    over_budget = None
    if args.max_spend_usd is not None and not args.dry_run:
        prior_spend = spend_from_trace(trace_path, args.model, helpers)
        if prior_spend:
            log.info("earlier processes of this run spent $%.4f (from the trace)", prior_spend)

        def over_budget() -> bool:
            spent = prior_spend + helpers.cost_usd(
                mem.budget.summary() if mem.budget else {}, args.model
            )
            return spent >= args.max_spend_usd

        if over_budget():
            raise SystemExit(
                f"already at ${prior_spend:.4f} against a cap of ${args.max_spend_usd:.2f} — "
                "nothing to do; raise the cap deliberately or stop here"
            )

    utc_started = datetime.now(UTC).isoformat()
    t0 = time.perf_counter()
    sink = records_path.open("a" if kept else "w", encoding="utf-8")
    if kept:
        # Drop anything past the last complete window, including a half-written
        # line, so the file we append to is exactly the prefix we validated.
        sink.close()
        with records_path.open("w", encoding="utf-8") as fh:
            for row in kept:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        sink = records_path.open("a", encoding="utf-8")
    try:
        new_rows, per_window = run_arm(
            mem,
            samples,
            adapt=(args.arm == "online"),
            window=args.window,
            log=log,
            sink=sink,
            skip_before=len(kept),
            over_budget=over_budget,
            retries=args.organizer == "ace" and args.max_rounds > 1,
            memory_source=finer.MEMORY_SOURCES[ORGANIZER_READS[args.organizer]],
        )
        rows = kept + new_rows
        with (OUT / f"{args.tag}.memory.jsonl").open("w", encoding="utf-8") as fh:
            capacity = helpers.dump_memory_snapshot(mem, 0, fh)
        with (OUT / f"{args.tag}.memory.ops.jsonl").open("w", encoding="utf-8") as fh:
            op_counts = helpers.dump_op_log(mem, 0, fh)
        budget = mem.budget.summary() if mem.budget else {}
        # The embedder is a paid one on the real path and its calls are NOT in
        # `mem.budget` — ACE embeds every curated bullet for dedup and every
        # task episode on ingest, so leaving them out would under-report the
        # bill and leave the artifact unable to explain its own total. Priced
        # at the embedder's own registry rates, not the chat model's: those
        # differ by 7.5x and pricing embeddings as chat tokens is the mistake
        # `cost_usd`'s `embed_model` argument exists to prevent.
        helpers.fold_embed_budget(budget, embedder)
    finally:
        sink.close()
        mem.close()
    elapsed = round(time.perf_counter() - t0, 1)

    if kept:
        # The windows answered before the crash are re-aggregated from their
        # rows rather than lost: the summary must describe the whole split, not
        # only the part this process happened to run.
        prefix = []
        for w in range(len(kept) // args.window):
            w_rows = kept[w * args.window : (w + 1) * args.window]
            agg = finer.aggregate(w_rows)
            agg["window"] = w
            agg["start"] = w * args.window
            agg["playbook_chars_at_test"] = w_rows[0].get("playbook_chars", 0)
            agg["resumed"] = True
            prefix.append(agg)
        per_window = prefix + per_window

    overall = finer.aggregate(rows)
    summary = {
        "arm": args.arm,
        "stamp": {
            "model": args.model,
            "embedder": args.embedder,
            "endpoint": args.endpoint,
            "split": args.split,
            "n_samples": len(samples),
            "window": args.window,
            "role_temps": ACE_ROLE_TEMPS,
            "organizers": [args.organizer],
            "read_path": ORGANIZER_READS[args.organizer],
            # ACE-only knobs. Emitted unconditionally for `--organizer ace` so the
            # four bought arms' stamps are unchanged; omitted for any other
            # organizer, because a `dedup_threshold` on an arm with no curator
            # would read as a setting that was in force.
            **(
                {
                    "dedup_threshold": args.dedup_threshold,
                    "dedup_enabled": args.dedup_threshold <= 1.0,
                    "max_rounds": args.max_rounds,
                }
                if args.organizer == "ace"
                else {}
            ),
            "deviations": (
                [
                    "D1_structured_output",
                    # D3/D4 are one knob seen twice: with `max_rounds` above 1 the
                    # organizer re-answers a failed sample from its own reflection,
                    # which is both the extra attempt and the retry loop. What stays
                    # deviant either way is the two upstream calls that buy no
                    # learning (ace.py:465-472, :608-625) — named so a reader cannot
                    # take "retries on" for "call structure identical".
                    *(
                        ["D3_one_attempt", "D4_no_retries"]
                        if args.max_rounds <= 1
                        else ["D3D4_retries_on_no_duplicate_or_post_curate_generation"]
                    ),
                    "D5_dedup_on"
                    if args.dedup_threshold <= 1.0
                    else "D5_dedup_off_upstream_default",
                ]
                if args.organizer == "ace"
                else [
                    "D1_structured_output",
                    # The one a reader must not miss: this is not RB's benchmark.
                    # Its published claims are agentic (WebArena, SWE-Bench), both
                    # unreachable here — a five-site Docker environment and a
                    # hosted grader against a floored gpt-4o-mini respectively.
                    # FiNER tests the MECHANISM on a single-turn task, against the
                    # same base arm docs/19 already bought.
                    "RB_D1_not_the_published_benchmark_finer_not_webarena_or_swebench",
                    # The generator prompt is ACE's, unchanged, so the arms differ
                    # only in what memory puts in the slot. Deliberate: the
                    # question is whether docs/19's null is ACE's or the task's,
                    # and that only reads with the generator held fixed.
                    "RB_D2_shared_ace_generator_prompt_memory_content_is_the_only_variable",
                    # Upstream embeds stored and incoming text alike as
                    # RETRIEVAL_DOCUMENT and prefixes an authored domain
                    # instruction (memory_management.py:49,83,128). We pin types
                    # and k only — see configs.rb_upstream.
                    "RB_D3_retrieval_geometry_not_reproduced",
                ]
            ),
            "context_markers_absent": fell_through,
            "git_sha": helpers.git_sha(),
            "utc_started": utc_started,
            "utc_finished": datetime.now(UTC).isoformat(),
            "dry_run": bool(args.dry_run),
            "resumed_from_records": len(kept),
            "max_spend_usd": args.max_spend_usd,
            "prior_spend_usd": round(prior_spend, 6),
            "complete": len(rows) == len(samples),
        },
        "overall": overall,
        "per_window": per_window,
        "llm_budget": budget,
        "memory_capacity": capacity,
        "op_counts": op_counts,
        "timing": {"eval_s": elapsed},
        "records_file": f"{args.tag}.records.jsonl",
    }
    if fake is not None:
        summary["dry_run_ledger"] = {
            "growth_bound": args.dry_run_growth,
            "calls": fake.calls,
            "prompt_chars": fake.prompt_chars,
            "reply_chars": fake.reply_chars,
        }
    else:
        this_process = helpers.cost_usd(
            budget, args.model, embed_model=helpers.embed_model_name(embedder)
        )
        # What the MEASUREMENT cost, not what this process cost. A resumed run
        # that reported only its own share would understate every arm the host
        # interrupted, and the interrupted ones are exactly the expensive ones.
        summary["cost_usd"] = round(this_process + prior_spend, 6)
        summary["cost_usd_this_process"] = this_process
        summary["stamp"]["embed_model_priced_as"] = helpers.embed_model_name(embedder)

    (OUT / f"{args.tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=helpers.json_safe)
    )
    log.info(
        "[done] tag=%.2f sample=%.2f n=%d -> %s",
        overall["tag_accuracy"],
        overall["sample_accuracy"],
        overall["n"],
        OUT / f"{args.tag}.json",
    )


if __name__ == "__main__":
    main()
