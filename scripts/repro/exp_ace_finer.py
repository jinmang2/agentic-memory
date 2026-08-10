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
  `max_num_rounds`=3 on an incorrect answer (ace.py:498-543). We reflect once.
  Per-sample adaptation cost is therefore fixed at 3 calls where upstream's is
  3-8 and rises with the error rate — a *worse* model costs upstream strictly
  more to adapt, which is the asymmetry any latency or token-cost comparison
  between the two has to state.
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
    mem = AgenticMemory(
        namespace=args.namespace,
        organizers=[ACEOrganizer()],
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
) -> tuple[list[dict], list[dict]]:
    """Walk the split. Returns (rows, per_window).

    With `adapt=False` this is upstream's `eval_only` over the whole split; the
    window boundaries still exist so the two arms produce comparable curves,
    but nothing is learned between them."""
    rows: list[dict] = []
    per_window: list[dict] = []
    for start, chunk in finer.windows(samples, window):
        w_rows: list[dict] = []
        for sample in chunk:
            capture: dict[str, Any] = {}
            try:
                pred, _bullets = finer.answer(mem, sample, capture=capture)
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
                finer.adapt(mem, sample, row)
            mem.flush()
    return rows, per_window


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["base", "online"], required=True)
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

    utc_started = datetime.now(UTC).isoformat()
    t0 = time.perf_counter()
    try:
        rows, per_window = run_arm(
            mem, samples, adapt=(args.arm == "online"), window=args.window, log=log
        )
        with (OUT / f"{args.tag}.memory.jsonl").open("w", encoding="utf-8") as fh:
            capacity = helpers.dump_memory_snapshot(mem, 0, fh)
        with (OUT / f"{args.tag}.memory.ops.jsonl").open("w", encoding="utf-8") as fh:
            op_counts = helpers.dump_op_log(mem, 0, fh)
        budget = mem.budget.summary() if mem.budget else {}
    finally:
        mem.close()
    elapsed = round(time.perf_counter() - t0, 1)

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
            "organizers": ["ace"],
            "dedup_threshold": 0.90,
            "deviations": [
                "D1_structured_output",
                "D3_one_attempt",
                "D4_no_retries",
                "D5_dedup_on",
            ],
            "context_markers_absent": fell_through,
            "git_sha": helpers.git_sha(),
            "utc_started": utc_started,
            "utc_finished": datetime.now(UTC).isoformat(),
            "dry_run": bool(args.dry_run),
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
        summary["cost_usd"] = helpers.cost_usd(budget, args.model)

    records_path = OUT / f"{args.tag}.records.jsonl"
    with records_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=helpers.json_safe) + "\n")
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
