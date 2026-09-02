"""The daemon path: hooks against a long-lived `agmem-mcp --transport http`.

Issue #2 §1 said the capture hook's ~11 s was process startup and that the
HTTP server already was the long-lived process the fix needed. These tests
drive that design the way the harness will: real hook subprocesses, a real
daemon subprocess on a loopback port, and the absent-daemon path in between.

Hermetic like `test_hooks.py`: `AGMEM_CONFIG` forces `FakeEmbedder` on both the
daemon and the hooks, so nothing downloads a model, and `AGMEM_NO_DAEMON=1`
keeps a hook from spawning a daemon of its own except in the one test that is
about spawning.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

FAKE_EMBEDDER_TOML = '[profile]\nname = "lite"\n\n[override]\nembedder = "FakeEmbedder"\n'
STARTUP_S = 40.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _health(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=0.5) as resp:
            return json.loads(resp.read())
    except Exception:  # noqa: BLE001 — "down" is a normal answer here
        return None


def _wait_health(url: str, timeout: float = STARTUP_S) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = _health(url)
        if payload and payload.get("ok"):
            return payload
        time.sleep(0.2)
    raise AssertionError(f"daemon at {url} did not come up within {timeout}s")


def _post(url: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{url}{path}",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _base_env(root, url: str) -> dict:
    cfg = root / "agmem.toml"
    if not cfg.exists():
        cfg.write_text(FAKE_EMBEDDER_TOML)
    return {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(root),
        "AGMEM_DATA_DIR": str(root / "data"),
        "AGMEM_NAMESPACE": "daemontest",
        "AGMEM_CONFIG": str(cfg),
        "AGMEM_DAEMON_URL": url,
        "AGMEM_NO_DAEMON": "1",
    }


def _run_hook(module: str, payload: dict, env: dict):
    return subprocess.run(
        [sys.executable, "-m", module],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def _spawn_daemon(env: dict, port: int, *extra: str) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agmem.mcp.server",
            "--transport",
            "http",
            "--port",
            str(port),
            "--organizers",
            "",
            "--backfill-period",
            "1",
            *extra,
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.fixture(scope="module")
def daemon(tmp_path_factory):
    root = tmp_path_factory.mktemp("agmem-daemon")
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = _base_env(root, url)
    proc = _spawn_daemon(env, port)
    try:
        _wait_health(url)
        yield {"root": root, "url": url, "env": env, "proc": proc}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_health_reports_the_store_it_resolved(daemon):
    """The daemon resolves namespace and data dir through the same environment
    the hooks use (`agmem.env`), and says which one it landed on."""
    payload = _health(daemon["url"])
    assert payload["ok"] is True
    assert payload["default_namespace"] == "daemontest"
    assert payload["open_namespaces"] == ["daemontest"]
    assert payload["pending_embed"] == {"daemontest": 0}
    assert payload["pid"] == daemon["proc"].pid


def test_capture_hook_takes_the_daemon_path_and_the_write_is_searchable(daemon):
    """With the daemon up, capture is one JSON round trip and the episode is
    embedded immediately: nothing pending, and the vector search finds it."""
    proc = _run_hook(
        "agmem.hooks.capture", {"session_id": "s1", "prompt": "I moved to Berlin"}, daemon["env"]
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert _health(daemon["url"])["pending_embed"] == {"daemontest": 0}
    reply = _post(daemon["url"], "/hooks/recall", {"query": "Berlin", "k": 3})
    assert reply["namespace"] == "daemontest"
    assert any("Berlin" in it["text"] for it in reply["items"]), reply


def test_recall_prompt_hook_injects_the_relevant_episode(daemon):
    """The query-driven counterpart to the recency hook: the prompt is the
    query, the daemon searches, the hook injects — under UserPromptSubmit."""
    got = _run_hook(
        "agmem.hooks.recall_prompt",
        {"session_id": "s1", "prompt": "Which city did I move to? Berlin?"},
        daemon["env"],
    )
    assert got.returncode == 0, got.stderr[-1500:]
    payload = json.loads(got.stdout)
    assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "Berlin" in payload["hookSpecificOutput"]["additionalContext"]
    assert "semantic search" in payload["hookSpecificOutput"]["additionalContext"]


def test_recall_prompt_emits_nothing_when_the_daemon_is_down(daemon):
    """No daemon, no injection, exit 0 — and no embedder loaded to compensate
    (the hook finishing in well under model-load time is the evidence)."""
    env = dict(daemon["env"], AGMEM_DAEMON_URL=f"http://127.0.0.1:{_free_port()}")
    started = time.perf_counter()
    got = _run_hook("agmem.hooks.recall_prompt", {"prompt": "Berlin"}, env)
    elapsed = time.perf_counter() - started
    assert got.returncode == 0
    assert got.stdout.strip() == ""
    assert elapsed < 5.0, f"recall_prompt took {elapsed:.1f}s without a daemon"


def test_capture_without_daemon_persists_and_the_daemon_backfills_the_vector(daemon):
    """The absent-daemon contract from the Phase 2 spec: the episode is written
    without a vector (fast, no model), shows up as pending on the next daemon,
    and the backfill makes it searchable."""
    env = dict(daemon["env"], AGMEM_DAEMON_URL=f"http://127.0.0.1:{_free_port()}")
    started = time.perf_counter()
    proc = _run_hook(
        "agmem.hooks.capture",
        {"session_id": "s2", "prompt": "The Lima office opens on Monday"},
        env,
    )
    elapsed = time.perf_counter() - started
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert elapsed < 5.0, f"absent-daemon capture took {elapsed:.1f}s — did it load a model?"

    # The running daemon shares the data dir, so it sees the vectorless episode
    # and, with --backfill-period 1, repairs it within a few seconds.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if _health(daemon["url"])["pending_embed"]["daemontest"] == 0:
            reply = _post(daemon["url"], "/hooks/recall", {"query": "Lima office", "k": 3})
            if any("Lima" in it["text"] for it in reply["items"]):
                break
        time.sleep(0.5)
    else:
        raise AssertionError("backfill did not make the episode searchable")


def test_ensure_running_spawns_a_daemon_that_comes_up(tmp_path):
    """`ensure_running` is how a hook asks for the daemon: a detached spawn,
    returning immediately, of the same installation with the same environment."""
    from agmem.hooks import daemon as daemon_client

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = _base_env(tmp_path, url)
    env.pop("AGMEM_NO_DAEMON")
    old = dict(os.environ)
    os.environ.update(env)
    try:
        assert daemon_client.health(url) is None
        assert daemon_client.ensure_running(url, log_path=tmp_path / "daemon.log") is False
        payload = _wait_health(url)
        assert payload["default_namespace"] == "daemontest"
        assert daemon_client.ensure_running(url) is True  # already up: no second spawn
    finally:
        os.environ.clear()
        os.environ.update(old)
        pid = payload.get("pid") if "payload" in locals() else None
        if pid:
            os.kill(pid, signal.SIGTERM)


def test_idle_timeout_stops_a_daemon_nobody_uses(tmp_path):
    """A hook-spawned daemon must not live forever: with --idle-timeout it
    exits on its own once nothing has asked for a memory."""
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = _base_env(tmp_path, url)
    proc = _spawn_daemon(env, port, "--idle-timeout", "2")
    try:
        _wait_health(url)
        proc.wait(timeout=20)
        assert proc.returncode == 0
        assert _health(url) is None
    finally:
        if proc.poll() is None:
            proc.kill()
