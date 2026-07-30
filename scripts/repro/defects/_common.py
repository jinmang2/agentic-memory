"""Shared plumbing for the Tier-0 defect reproductions.

Each repro is a standalone script: it proves one upstream defect deterministically
(no LLM, no network, $0) or exits 0 with a SKIP line when its evidence is absent —
the same capability-gating convention the test suite uses (docs/01). Upstream
snapshots resolve under $AGMEM_UPSTREAM (default ~/.agmem/upstream); CI fetches
them at the pinned SHAs below, so "the CI proves it, not the prose".
"""

import os
import sys
from pathlib import Path
from typing import NoReturn

REPO = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = Path(os.environ.get("AGMEM_UPSTREAM", str(Path.home() / ".agmem" / "upstream")))

# The ledger (docs/17) cites these snapshots; a proof against a different SHA proves
# something else, so a mismatched local clone gets a loud warning (not a failure).
PINS = {
    "AgenticMemory": "0c8039f28fdcc08189a23c07a3437d9d2482f9c2",
    "GMemory": "7b581c51d993bd600df14691d101d7e601040cc6",
    "MemMachine": "18f1211290c50ae30e9960b90bbe57d89bf68600",
    "nemori": "d2a6dff6e5481214a0be6a2d10147feccfc16244",
}


def skip(reason: str) -> NoReturn:
    print(f"SKIP: {reason}")
    sys.exit(0)


def proven(claim: str) -> None:
    print(f"PROVEN: {claim}")


def upstream(name: str) -> Path:
    path = UPSTREAM_ROOT / name
    if not path.is_dir():
        skip(f"upstream clone '{name}' not found under {UPSTREAM_ROOT}")
    head = path / ".git"
    if head.exists():
        import subprocess

        sha = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if name in PINS and sha and sha != PINS[name]:
            print(f"WARNING: {name} is at {sha[:7]}, ledger pins {PINS[name][:7]}")
    return path
