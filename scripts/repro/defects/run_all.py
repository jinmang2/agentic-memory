"""Demo D — run every Tier-0 defect reproduction and say plainly which claims held.

The defect ledger (docs/17) makes assertions about other people's published code. This runner is
the offer that goes with them: clone the repository, run one command, and watch each claim be
re-proved on your machine — deterministically, with **no model call and no API key**, against
upstream snapshots pinned to exact commits.

**What a line means.**

  PASS   the script re-derived its claim. The `PROVEN:` lines under it are the claim, stated by the
         script itself; this runner never paraphrases them.
  SKIP   the script's evidence was not present — most often an upstream snapshot that was never
         fetched, sometimes a heavy dependency the core install omits. A skip proves NOTHING, which
         is why it is printed in full rather than folded into a count.
  FAIL   the script ran and its claim did not hold, or it crashed. Either way the ledger is wrong
         about something and the exit code says so.

**Two exit rules, and the second one is the interesting one.** A non-zero exit follows from any
FAIL, as expected. It also follows from a run in which *nothing was proved at all* — every script
skipping is exactly what a broken setup looks like, and a runner that exits 0 on it would be
advertising, not evidence. `--allow-nothing-proved` is there for the one case where that is
legitimately fine (a probe of which snapshots are present).

**Relationship to CI.** `.github/workflows/ci.yml` runs the same `repro_*.py` files on every push,
and the pinned snapshots both it and this runner use come from `fetch_upstream.py`, which reads the
SHAs out of `_common.PINS`. There is no second copy of the pins and no second fetch procedure; this
runner adds reporting, not a parallel path to the same proofs.

Run:  uv run --no-default-groups --group dev python scripts/repro/defects/run_all.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import fetch_upstream
from _common import UPSTREAM_ROOT

DEFECTS_DIR = Path(__file__).resolve().parent

PASS, SKIP, FAIL = "PASS", "SKIP", "FAIL"
STATUS_MARK = {PASS: "✓", SKIP: "–", FAIL: "✗"}


@dataclass(frozen=True)
class Outcome:
    """One script's run: what it printed, how it was classified, and how long it took."""

    script: str
    status: str
    proven: list[str]
    skipped: list[str]
    seconds: float
    output: str
    returncode: int


def classify(returncode: int, output: str) -> tuple[str, list[str], list[str]]:
    """Turn a script's exit code and stdout into a status plus the claims it actually made.

    A script may both prove and skip — several have a static half that runs anywhere and a dynamic
    half needing the clone's heavy dependencies. Any proof at all makes the run a PASS, and the
    skipped halves are still listed, because "partly proved" is the honest description and the
    alternative is a green line over a claim nobody checked.
    """
    proven = [
        line[len("PROVEN:") :].strip() for line in output.splitlines() if line.startswith("PROVEN:")
    ]
    skipped = [
        line[len("SKIP:") :].strip() for line in output.splitlines() if line.startswith("SKIP:")
    ]
    if returncode != 0:
        return FAIL, proven, skipped
    return (PASS if proven else SKIP), proven, skipped


def run_script(path: Path) -> Outcome:
    """Run one reproduction to completion and capture everything it said.

    Invoked with this interpreter rather than a nested `uv run`: the runner is already inside the
    resolved environment, and CI's `--no-default-groups --group dev` flags exist to stop a bare
    `uv run` from re-syncing the heavy groups, which a child of an already-resolved interpreter
    cannot do. Same code, same environment, one process tree.
    """
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, path.name],
        cwd=DEFECTS_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    status, proven, skipped = classify(completed.returncode, output)
    return Outcome(
        script=path.name,
        status=status,
        proven=proven,
        skipped=skipped,
        seconds=time.monotonic() - started,
        output=output.rstrip(),
        returncode=completed.returncode,
    )


def report(outcome: Outcome, verbose: bool) -> None:
    """Print one script's result: the headline line, then the claims it made in its own words."""
    print(
        f"{STATUS_MARK[outcome.status]} {outcome.status}  {outcome.script}  ({outcome.seconds:.1f}s)"
    )
    for claim in outcome.proven:
        print(f"      proved: {claim}")
    for reason in outcome.skipped:
        print(f"      skipped: {reason}")
    if outcome.status == FAIL:
        print(f"      exit code {outcome.returncode}")
        for line in outcome.output.splitlines():
            print(f"      | {line}")
    elif verbose:
        for line in outcome.output.splitlines():
            print(f"      | {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="fetch any missing pinned upstream snapshot before running (network, ~$0)",
    )
    parser.add_argument("--verbose", action="store_true", help="echo each script's full output")
    parser.add_argument(
        "--allow-nothing-proved",
        action="store_true",
        help="exit 0 even when every script skipped (a setup probe, not a proof)",
    )
    args = parser.parse_args()

    scripts = sorted(DEFECTS_DIR.glob("repro_*.py"))
    if not scripts:
        raise SystemExit(f"no repro_*.py under {DEFECTS_DIR}")

    absent = fetch_upstream.missing()
    if absent and args.fetch:
        print(f"fetching {len(absent)} pinned snapshot(s) into {UPSTREAM_ROOT}")
        for name in absent:
            fetch_upstream.fetch(name, fetch_upstream.PINS[name])
        absent = fetch_upstream.missing()
        print()
    if absent:
        print(
            f"note: {len(absent)} pinned snapshot(s) absent from {UPSTREAM_ROOT} "
            f"({', '.join(absent)}); the scripts that need them will skip."
        )
        print("      re-run with --fetch to download them (depth-1, one commit each).\n")

    print(f"running {len(scripts)} Tier-0 defect reproductions — no model call, no API key, $0\n")
    outcomes = [run_script(path) for path in scripts]
    for outcome in outcomes:
        report(outcome, args.verbose)

    passed = [outcome for outcome in outcomes if outcome.status == PASS]
    skipped = [outcome for outcome in outcomes if outcome.status == SKIP]
    failed = [outcome for outcome in outcomes if outcome.status == FAIL]
    claims = sum(len(outcome.proven) for outcome in outcomes)
    elapsed = sum(outcome.seconds for outcome in outcomes)

    print(
        f"\n{len(passed)} passed, {len(skipped)} skipped, {len(failed)} failed — "
        f"{claims} claims re-proved in {elapsed:.1f}s, $0 spent."
    )
    print(
        "The claims are the ledger's, restated by the scripts that prove them: docs/17-defect-ledger.md"
    )

    # A count of passes is not a verdict when some scripts proved nothing. Said plainly, because a
    # run that skipped most of its evidence looks identical to a clean one in a tally of exit codes.
    if failed:
        print(f"\nVERDICT: {len(failed)} reproduction(s) FAILED — a ledger claim did not hold.")
    elif skipped:
        print(
            f"\nVERDICT: partial. {len(skipped)} of {len(outcomes)} scripts established nothing on "
            f"this machine because their evidence was absent; only the {len(passed)} above were "
            f"actually checked."
        )
    else:
        print(f"\nVERDICT: all {len(outcomes)} reproductions held.")

    if failed:
        raise SystemExit(1)
    if not claims and not args.allow_nothing_proved:
        raise SystemExit(
            "nothing was proved: every script skipped. That is a setup problem, not a result — "
            "run with --fetch to get the pinned snapshots, or --allow-nothing-proved if you meant "
            "to probe what is present."
        )


if __name__ == "__main__":
    main()
