"""Fetch the pinned upstream snapshots the Tier-0 defect reproductions are proved against.

The ledger (docs/17) cites an exact SHA per project, because a proof taken against a different
commit proves something about a different program. Those SHAs live in one place — `_common.PINS` —
and this script is the one thing that turns them into checkouts, so CI and a reader's laptop fetch
the same commits by construction rather than by two lists agreeing.

Each snapshot is a depth-1 fetch of the exact SHA into `$AGMEM_UPSTREAM` (default
`~/.agmem/upstream`). GitHub serves any reachable commit, so no branch or tag is needed and nothing
but the one commit is downloaded. Re-running is cheap: a directory already at the pinned SHA is
left alone.

Nothing here reaches the network for a project that is already pinned correctly, and nothing is
ever deleted — a clone sitting at some other commit is re-fetched and checked out, never removed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _common import PINS, UPSTREAM_ROOT

# Where each pinned project is fetched from. Kept beside PINS rather than derived from it because a
# repository can be renamed or moved without the pinned commit changing.
ORIGINS = {
    "AgenticMemory": "WujiangXu/AgenticMemory",
    "GMemory": "bingreeky/GMemory",
    "MemMachine": "MemMachine/MemMachine",
    "nemori": "nemori-ai/nemori",
    "reasoning-bank": "google-research/reasoning-bank",
}


def head_sha(directory: Path) -> str | None:
    """The checked-out commit, or None when the directory is not a git repository yet."""
    if not (directory / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def fetch(name: str, sha: str) -> bool:
    """Put `name` at `sha` under the upstream root. Returns True when it ended up there.

    Failure is reported rather than raised: one unreachable project must not stop the other four
    from being fetched, and the caller decides what a partial set means for its own exit code.
    """
    directory = UPSTREAM_ROOT / name
    if head_sha(directory) == sha:
        print(f"  {name}: already at {sha[:7]}")
        return True

    directory.mkdir(parents=True, exist_ok=True)
    steps = [
        ["git", "init", "-q", str(directory)],
        [
            "git",
            "-C",
            str(directory),
            "fetch",
            "-q",
            "--depth",
            "1",
            f"https://github.com/{ORIGINS[name]}",
            sha,
        ],
        ["git", "-C", str(directory), "checkout", "-q", "FETCH_HEAD"],
    ]
    for step in steps:
        result = subprocess.run(step, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"  {name}: FAILED — {' '.join(step[:3])}: {result.stderr.strip()}")
            return False
    print(f"  {name}: fetched {sha[:7]} from {ORIGINS[name]}")
    return True


def missing() -> list[str]:
    """Pinned projects that are absent or sitting at some other commit, in PINS order."""
    return [name for name, sha in PINS.items() if head_sha(UPSTREAM_ROOT / name) != sha]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what is missing and exit non-zero, without touching the network",
    )
    args = parser.parse_args()

    print(f"upstream root: {UPSTREAM_ROOT}")
    if args.check:
        absent = missing()
        for name in PINS:
            state = "MISSING" if name in absent else "ok"
            print(f"  {name}: {state}")
        sys.exit(1 if absent else 0)

    failures = [name for name, sha in PINS.items() if not fetch(name, sha)]
    if failures:
        print(f"could not fetch: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
