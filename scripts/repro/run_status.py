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
      <tag>.llm-trace.jsonl   one line per model call, appended across EVERY
                              process of the run, which makes it the only
                              record of what the measurement actually cost

    The trace is why spend here is not `summary["cost_usd"]` for a running arm:
    the summary does not exist until the run ends, and for a resumed arm its
    `llm_budget` counts only the last process (the completed nodedup arm files
    164 calls against ~1,325 bought). Cost is therefore recomputed from the
    trace, which spans every attempt.

USAGE
    uv run python scripts/repro/run_status.py            # every run with artifacts
    uv run python scripts/repro/run_status.py --active   # only unfinished ones
    uv run python scripts/repro/run_status.py --tag gpt-4o-mini_ace_finer_retry

    Costs nothing and calls nothing. Safe to put on a timer:
    `/loop 10m uv run python scripts/repro/run_status.py --active`
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RECORDS = ROOT / "results" / "repro"

# Prices are the campaign's single model, stated where they are used rather than
# imported, because this script must keep working when nothing else does.
RATES = {"gpt-4o-mini": (0.15 / 1e6, 0.60 / 1e6)}


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def spend_from_trace(path: Path, model: str = "gpt-4o-mini") -> tuple[float, int]:
    """(usd, calls) across every process of this run. A truncated final line —
    the shape a kill leaves — ends the count rather than raising."""
    if not path.exists():
        return 0.0, 0
    rate_in, rate_out = RATES.get(model, RATES["gpt-4o-mini"])
    usd = 0.0
    calls = 0
    for line in path.open(encoding="utf-8"):
        try:
            call = json.loads(line)
        except json.JSONDecodeError:
            break
        calls += 1
        usd += (call.get("tokens_in") or 0) * rate_in + (call.get("tokens_out") or 0) * rate_out
    return usd, calls


def describe(tag: str) -> dict:
    records = RECORDS / f"{tag}.records.jsonl"
    summary_path = RECORDS / f"{tag}.json"
    trace = RECORDS / f"{tag}.llm-trace.jsonl"

    rows = count_lines(records)
    usd, calls = spend_from_trace(trace)
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary = {}
    stamp = summary.get("stamp", {})
    overall = summary.get("overall", {})

    # "Moving" is decided by the trace's mtime, not by a process: an appended
    # call is the only evidence that something is still spending.
    idle_s = time.time() - trace.stat().st_mtime if trace.exists() else None

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
            trace = RECORDS / f"{tag}.llm-trace.jsonl"
            if trace.exists() and time.time() - trace.stat().st_mtime < 600:
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
