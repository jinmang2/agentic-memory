"""The Claude Code hooks, driven the way the harness drives them.

A hook fails silently by construction — it must never break the session, so
every error path exits 0 with no output. That safety property is also what makes
a broken hook invisible, so these drive the real entry points as subprocesses
with real stdin payloads and assert on the contract, not on internals.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


def _run(module: str, payload: dict, tmp_path, extra_env: dict | None = None):
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
        "AGMEM_DATA_DIR": str(tmp_path / "data"),
        "AGMEM_NAMESPACE": "hooktest",
        **(extra_env or {}),
    }
    return subprocess.run(
        [sys.executable, "-m", module],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    """One capture, shared. Capture costs ~15 s because it loads the embedder
    (see agmem.hooks.capture), so a per-test capture would make this file the
    slowest thing in the suite for no extra coverage."""
    root = tmp_path_factory.mktemp("agmem-hooks")
    proc = _run("agmem.hooks.capture", {"session_id": "s1", "prompt": "I moved to Berlin"}, root)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return root


def test_capture_then_recall_round_trips_through_the_store(seeded):
    """The whole point, end to end: a prompt captured by one hook comes back as
    context from the other."""
    got = _run("agmem.hooks.recall", {}, seeded)
    assert got.returncode == 0, got.stderr[-2000:]
    payload = json.loads(got.stdout)
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "Berlin" in payload["hookSpecificOutput"]["additionalContext"]


def test_recall_on_an_empty_store_emits_nothing_rather_than_an_empty_block(tmp_path):
    """No memory yet must mean no injected context. Emitting a header with no
    lines under it would spend context to tell the model nothing, on every
    session, forever."""
    got = _run("agmem.hooks.recall", {}, tmp_path)
    assert got.returncode == 0
    assert got.stdout.strip() == ""


def test_hooks_survive_malformed_stdin(tmp_path):
    """The harness must never be broken by us. Garbage in, exit 0, no output —
    this is the invariant the whole package is built around."""
    for module in ("agmem.hooks.recall", "agmem.hooks.capture"):
        proc = subprocess.run(
            [sys.executable, "-m", module],
            input="not json at all {{{",
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "HOME": str(tmp_path),
                "AGMEM_DATA_DIR": str(tmp_path / "data"),
                "AGMEM_NAMESPACE": "hooktest",
            },
            timeout=120,
        )
        assert proc.returncode == 0, f"{module}: {proc.stderr[-1500:]}"


def test_capture_ignores_a_prompt_with_no_text(tmp_path):
    """An event carrying no prompt must not write an empty episode — those
    accumulate silently and dilute every later recall."""
    proc = _run("agmem.hooks.capture", {"session_id": "s1", "prompt": "   "}, tmp_path)
    assert proc.returncode == 0
    got = _run("agmem.hooks.recall", {}, tmp_path)
    assert got.stdout.strip() == ""


def test_recall_stays_fast_enough_for_a_blocking_hook(seeded):
    """Recall blocks session start, so its budget is a real constraint rather
    than a preference.

    It briefly did not meet it: opening a full `AgenticMemory` to list episodes
    cost 17.7 s, essentially all of it `SentenceTransformerEmbedder` loading
    weights it never used, which exceeds any sane `timeout` and would be killed
    — presenting as no memory rather than as an error. Reading the doc store
    directly took it to 0.21 s. The bound here is loose enough for a slow CI box
    and still an order of magnitude under the 10 s the wiring documentation
    suggests.
    """
    import time

    start = time.perf_counter()
    got = _run("agmem.hooks.recall", {}, seeded)
    elapsed = time.perf_counter() - start
    assert got.returncode == 0
    assert elapsed < 5.0, f"recall took {elapsed:.1f}s — the embedder is back on this path"
