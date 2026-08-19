"""What is every run in `results/repro` doing right now? Read off the artifacts.

WHY THIS IS NOT A PROCESS CHECK
    A session watching `pgrep -f "exp_ace_finer.py --arm online"` matched its own
    command line and waited forever on a run that had finished nineteen hours
    earlier. The process table is also the wrong source on principle: this host
    has rebooted three long runs out of existence, and a run that died still
    leaves the only thing worth knowing behind it on disk.

    So progress is read from the artifacts a run writes as it goes:

      <tag>.records.jsonl     one line per answered sample, appended per window
      <tag>.json              written at the end; `stamp.complete` says whether
                              the split was finished or a cap stopped it
                              (`<tag>.reconstructed.json` counts too: the zep
                              MMR arm's summary was rebuilt from its trace and
                              is the git-tracked record of a finished run)
      <tag>.llm-trace.jsonl   one line per model call, appended across EVERY
                              process of the run, which makes it the only
                              record of what the CHAT calls actually cost
      <tag>.embed-trace.jsonl the embedding sidecar the LME driver writes for a
                              paid embedder — the chat trace's blind spot. The
                              `_s` top-50 arm's embedding was $1.11 of $2.59;
                              priced from the llm-trace alone it showed $1.51.

    The traces are why spend here is not `summary["cost_usd"]` for a running
    arm: the summary does not exist until the run ends, and for a resumed arm
    its `llm_budget` counts only the last process (the completed nodedup arm
    files 164 calls against ~1,325 bought). Cost is therefore recomputed from
    the traces, which span every attempt.

USAGE
    uv run python scripts/repro/run_status.py            # every run with artifacts
    uv run python scripts/repro/run_status.py --active   # only unfinished ones
    uv run python scripts/repro/run_status.py --tag gpt-4o-mini_ace_finer_retry

    Costs nothing and calls nothing. Safe to put on a timer:
    `/loop 10m uv run python scripts/repro/run_status.py --active`
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RECORDS = ROOT / "results" / "repro"

# Prices stated here rather than imported, because this script must keep working
# when nothing else does. It is no longer one model: the LongMemEval arms read
# with one model and judge with another, and the judge's output costs 16.7x the
# reader's, so a single-rate total would understate a finished arm by more than
# the arm cost.
RATES = {
    "gpt-4o-mini": (0.15 / 1e6, 0.60 / 1e6),
    "gpt-4o-2024-08-06": (2.50 / 1e6, 10.00 / 1e6),
    "gpt-5.6-luna": (0.20 / 1e6, 1.20 / 1e6),
    # docs/research/longmemeval.md §8.2 (lines 721-722) — same numbers the
    # registry carries, restated here for the same keep-working-alone reason.
    "gpt-5.6-terra": (2.00 / 1e6, 12.00 / 1e6),
    "gpt-5.6-sol": (5.00 / 1e6, 30.00 / 1e6),
    "text-embedding-3-small": (0.02 / 1e6, 0.0),
}
DEFAULT_MODEL = "gpt-4o-mini"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def open_trace(path: Path):
    """The trace, gzipped or not. A `_s` arm writes ~260 MB of prompts, so its
    trace is written as `.jsonl.gz`; everything that reads a trace has to accept
    both or it will report a running arm as having spent nothing."""
    return (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open(encoding="utf-8")
    )


def trace_path_for(tag: str) -> Path:
    plain = RECORDS / f"{tag}.llm-trace.jsonl"
    gz = RECORDS / f"{tag}.llm-trace.jsonl.gz"
    return gz if gz.exists() and not plain.exists() else plain


def embed_trace_path_for(tag: str) -> Path:
    """The LME driver's embedding sidecar (`TracedEmbedder`). Chat calls go
    through `LLMClient` and land in the llm-trace; embedding calls never do, so
    a paid retrieval arm priced from the llm-trace alone under-reports by its
    whole embedding share."""
    return RECORDS / f"{tag}.embed-trace.jsonl"


def summary_path_for(tag: str) -> Path:
    """`<tag>.json`, or the reconstructed one when only that exists.

    The zep MMR arm's live summary was lost with its process; the git-tracked
    `<tag>.reconstructed.json` (rebuilt from the trace, verified against the
    records) is that run's summary of record. Looking only for `<tag>.json`
    reported the arm as STALE forever — a finished measurement showing as a
    dead run is exactly the misread this script exists to prevent."""
    plain = RECORDS / f"{tag}.json"
    reconstructed = RECORDS / f"{tag}.reconstructed.json"
    return reconstructed if reconstructed.exists() and not plain.exists() else plain


def trace_mtimes(tag: str) -> list[float]:
    """mtimes of every trace this run appends to — chat and embedding. An arm
    deep in a long ingest can go many minutes between chat calls while the
    embed sidecar ticks; freshness must read both or that arm looks stalled."""
    return [
        p.stat().st_mtime for p in (trace_path_for(tag), embed_trace_path_for(tag)) if p.exists()
    ]


def spend_from_trace(path: Path, model: str = DEFAULT_MODEL) -> tuple[float, int]:
    """(usd, calls) across every process of this run. A truncated final line —
    the shape a kill leaves — ends the count rather than raising.

    Each line is priced at the rate of the model IT names, not at one rate for
    the file: a run whose reader and judge differ has two rates in one trace, and
    `model` is only the fallback for a line that names something unregistered.

    A line may carry a `calls` count above 1: the embed sidecar records deltas
    of a shared embedder's counters, so one line can cover several concurrent
    calls. Counting lines instead would understate exactly the arm whose calls
    the sidecar exists to count."""
    if not path.exists():
        return 0.0, 0
    fallback = RATES.get(model, RATES[DEFAULT_MODEL])
    usd = 0.0
    calls = 0
    with open_trace(path) as fh:
        for line in fh:
            try:
                call = json.loads(line)
            except (json.JSONDecodeError, EOFError, OSError):
                break
            calls += int(call.get("calls") or 1)
            rate_in, rate_out = RATES.get(str(call.get("model")), fallback)
            usd += (call.get("tokens_in") or 0) * rate_in + (call.get("tokens_out") or 0) * rate_out
    return usd, calls


def describe(tag: str) -> dict:
    records = RECORDS / f"{tag}.records.jsonl"
    summary_path = summary_path_for(tag)
    trace = trace_path_for(tag)

    rows = count_lines(records)
    usd, calls = spend_from_trace(trace)
    # The embedding sidecar is the llm-trace's blind spot (see
    # `embed_trace_path_for`): both files price with the same loop, and the
    # arm's spend is their sum.
    usd_embed, calls_embed = spend_from_trace(embed_trace_path_for(tag))
    usd += usd_embed
    calls += calls_embed
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary = {}
    stamp = summary.get("stamp", {})
    overall = summary.get("overall", {})
    # An arm measured before the sidecar existed has an embed-blind trace side
    # forever (the `_s` top-50 batched arm reads ~$1.51 from its llm-trace
    # against a summary total of $4.61). Its summary folded the embedder in and
    # counted prior processes, and neither record can overstate — so a finished
    # run shows the larger of the two, the same rule the driver's own
    # `_prior_spend` applies.
    try:
        usd = max(usd, float(summary.get("cost_usd") or 0.0))
    except (TypeError, ValueError):
        pass

    # "Moving" is decided by the traces' mtime, not by a process: an appended
    # call — chat or embedding — is the only evidence something is spending.
    mtimes = trace_mtimes(tag)
    idle_s = time.time() - max(mtimes) if mtimes else None

    return {
        "tag": tag,
        "rows": rows,
        "n_samples": stamp.get("n_samples"),
        "complete": stamp.get("complete"),
        "has_summary": bool(summary),
        "cap": stamp.get("max_spend_usd"),
        "usd": usd,
        "calls": calls,
        "idle_s": idle_s,
        "tag_accuracy": overall.get("tag_accuracy"),
        "errors": sum(v.get("errors", 0) for v in summary.get("llm_budget", {}).values()),
    }


def is_finished(row: dict) -> bool:
    """Done, by either of the two things that can say so.

    `stamp.complete` postdates the base and online arms, and the LoCoMo runner
    stamps no row target at all, so a summary counts as finished unless a
    stamped target says the run stopped short — which is exactly the shape a
    spend cap leaves (441 asked for, 435 answered). One function, because the
    display and the `--active` filter asking the same question in two places is
    how they come to disagree."""
    if row["complete"] is True:
        return True
    if not row["has_summary"]:
        return False
    return not row["n_samples"] or row["rows"] >= row["n_samples"]


def is_active(row: dict) -> bool:
    """Unfinished, and touched within the last ten minutes."""
    return not is_finished(row) and row["idle_s"] is not None and row["idle_s"] < 600


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", action="append", help="restrict to these tags (repeatable)")
    ap.add_argument("--active", action="store_true", help="only runs still being written to")
    args = ap.parse_args(argv)

    tags = args.tag or sorted(
        p.name[: -len(".records.jsonl")] for p in RECORDS.glob("*.records.jsonl")
    )

    # `describe` reads a whole trace to price a run, and this campaign's traces
    # reach 193 MB — so `--active` narrows by mtime FIRST, using a stat rather
    # than a read. Without this the session-start hook that calls it took 5.85
    # seconds to answer "is anything running", which is how the product's own
    # recall hook once died on a timeout and looked like a missing feature.
    if args.active:
        fresh = []
        for tag in tags:
            mtimes = trace_mtimes(tag)
            if mtimes and time.time() - max(mtimes) < 600:
                fresh.append(tag)
        tags = fresh

    rows = [describe(t) for t in tags]
    if args.active:
        rows = [r for r in rows if is_active(r)]
    if not rows:
        print("no runs match" + (" (nothing active)" if args.active else ""))
        return

    for r in rows:
        if is_finished(r):
            state = "done"
        elif r["has_summary"]:
            state = "STOPPED at cap" if r["cap"] else "stopped"
        elif r["idle_s"] is not None and r["idle_s"] < 600:
            state = f"running ({r['idle_s']:.0f}s since last call)"
        else:
            state = f"STALE ({r['idle_s'] / 60:.0f}m silent, no summary)" if r["idle_s"] else "?"

        progress = f"{r['rows']}" + (f"/{r['n_samples']}" if r["n_samples"] else "")
        cap = f" / cap ${r['cap']:.2f}" if r["cap"] else ""
        acc = f" · tag {r['tag_accuracy']}" if r["tag_accuracy"] is not None else ""
        err = f" · ERRORS {r['errors']}" if r["errors"] else ""
        print(
            f"{r['tag']:<42} {progress:>9} rows · {state:<34} "
            f"${r['usd']:.4f}{cap} · {r['calls']} calls{acc}{err}"
        )


if __name__ == "__main__":
    main()
