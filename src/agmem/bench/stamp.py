"""The one reproducibility stamp every result file carries.

docs/05 §3 states the discipline in one line — *모든 결과에
``{profile, commit, model, judge, dataset_version, runs}`` 스탬프* — and names its
motivation, the Zep-LoCoMo controversy, where results could not be compared
because the exact condition behind each number was not recorded.

That discipline was restated three times instead of implemented once, and two of
the three had drifted by the time anyone checked:

- ``bench/harness.py`` stamped ``commit``/``profile``, but neither experiment
  script uses the harness;
- ``scripts/exp_amem_repro.py`` stamped rich provenance under ``git_sha`` — but
  no ``profile``;
- ``scripts/exp_locomo_conv0.py`` stamped an inline dict with **four of the six
  documented fields missing**: no ``profile``, ``commit``, ``judge`` or ``runs``.

So the headline LoCoMo results on disk cannot say which profile produced them or
which commit they came from. That was survivable only because both scripts
happen to pin ``profile="lite"`` — checked, not assumed — which is luck, not a
guarantee, and every future run would have inherited it.

``run_stamp`` is therefore the single producer: it fills the documented fields
from the live memory and the repo, and callers merge their own condition on top.
``REQUIRED_FIELDS`` is asserted by ``tests/test_repro_artifacts.py`` against the
list in docs/05 §3, so the doc and the code cannot drift apart silently again.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from agmem import __version__

# The fields docs/05 §3 promises on every result. Pinned by a test against the
# doc, so adding one there without adding it here fails the suite.
REQUIRED_FIELDS = ("profile", "commit", "model", "judge", "dataset_version", "runs")


def git_commit() -> str:
    """Full HEAD sha, or ``"unknown"`` outside a git tree / without git.

    Full rather than ``--short``: the stamp exists to identify code exactly, and
    short shas collide. Never raises — a stamp that fails the run it is
    describing would be worse than an unknown field."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def dataset_fingerprint(path: str | Path | None) -> str:
    """``"sha256:<12 hex>"`` of the dataset file, or ``"unknown"``.

    Content-addressed rather than a version label a caller types in: the
    Zep-LoCoMo dispute turned partly on *which* copy of the data a number came
    from, and a hand-written ``"locomo10"`` cannot answer that."""
    if path is None:
        return "unknown"
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()[:12]}"
    except OSError:
        return "unknown"


def run_stamp(
    memory: Any = None,
    *,
    model: str | None = None,
    judge: Any = None,
    runs: int | None = None,
    dataset: str | None = None,
    dataset_path: str | Path | None = None,
    dataset_version: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a result stamp carrying at least ``REQUIRED_FIELDS``.

    ``memory`` supplies the fields only it knows (profile, embedder, vector
    store, active organizers); pass ``None`` when stamping an aggregate that has
    no single memory behind it, and the fields resolve to ``None``/``"unknown"``.
    ``dataset_version`` defaults to the fingerprint of ``dataset_path``.
    ``extra`` is merged last, so a caller can record its own condition — but it
    cannot silently drop a required field, because those are written first and a
    caller overriding one is stating a value, not omitting it."""
    stamp: dict[str, Any] = {
        "profile": getattr(getattr(memory, "config", None), "profile", None),
        "commit": git_commit(),
        "model": model,
        "judge": judge,
        "dataset_version": dataset_version or dataset_fingerprint(dataset_path),
        "runs": runs,
        # provenance beyond the six, same for every result
        "agmem_version": __version__,
        "python": platform.python_version(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if dataset is not None:
        stamp["dataset"] = dataset
    if dataset_path is not None:
        stamp["dataset_path"] = str(dataset_path)
    if memory is not None:
        stamp["embedder"] = memory.embedder.name
        stamp["vector_store"] = type(memory.vector_store).__name__
        stamp["organizers"] = [o.name for o in memory.organizers]
    stamp.update(extra)
    return stamp
