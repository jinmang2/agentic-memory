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

from agmem import __version__

FAKE_EMBEDDER_TOML = '[profile]\nname = "lite"\n\n[override]\nembedder = "FakeEmbedder"\n'
STARTUP_S = 40.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _health(url: str, pending: bool = False, queues: bool = False) -> dict | None:
    try:
        params = []
        if pending:
            params.append("pending=1")
        if queues:
            params.append("queues=1")
        query = "?" + "&".join(params) if params else ""
        with urllib.request.urlopen(f"{url}/health{query}", timeout=0.5) as resp:
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
    assert payload is not None
    assert payload["ok"] is True
    assert payload["default_namespace"] == "daemontest"
    assert payload["open_namespaces"] == ["daemontest"]
    assert payload["pid"] == daemon["proc"].pid
    assert payload["package_version"] == __version__
    assert payload["data_dir"] == str(daemon["root"] / "data")
    assert payload["config_path"] == str(daemon["root"] / "agmem.toml")
    assert payload["interpreter"]
    assert payload["module_path"].endswith("src/agmem/mcp/server.py")
    fingerprint = payload["runtime_fingerprint"]
    assert fingerprint["config_sha256"]
    assert fingerprint["source_sha256"]["mcp/server.py"]
    assert fingerprint["source_sha256"]["memory.py"]
    # Liveness only by default: the scan behind `pending_embed` is over the
    # hooks' 0.3 s budget on a real store, so it is asked for explicitly.
    assert "pending_embed" not in payload
    assert "queues" not in payload
    pending_payload = _health(daemon["url"], pending=True)
    assert pending_payload is not None
    assert pending_payload["pending_embed"] == {"daemontest": 0}
    queue_payload = _health(daemon["url"], queues=True)
    assert queue_payload is not None
    queues = queue_payload["queues"]["daemontest"]
    assert queues == {
        "preserve": {"queued": 0, "processing": 0, "bad": 0},
        "distill": {"queued": 0, "processing": 0, "bad": 0},
    }


def test_capture_hook_takes_the_daemon_path_and_the_write_is_searchable(daemon):
    """With the daemon up, capture is one JSON round trip and the episode is
    embedded immediately: nothing pending, and the vector search finds it."""
    proc = _run_hook(
        "agmem.hooks.capture", {"session_id": "s1", "prompt": "I moved to Berlin"}, daemon["env"]
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert _health(daemon["url"], pending=True)["pending_embed"] == {"daemontest": 0}
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


def test_recall_prompt_answers_by_keyword_when_the_daemon_is_down(daemon):
    """No daemon: the hook answers from the doc store by BM25 (docs/23 §8 — the
    ~20 s startup window used to be a silent gap), exit 0, and no embedder is
    loaded to compensate (finishing in well under model-load time is the
    evidence). The header names the path that answered."""
    env = dict(daemon["env"], AGMEM_DAEMON_URL=f"http://127.0.0.1:{_free_port()}")
    started = time.perf_counter()
    got = _run_hook("agmem.hooks.recall_prompt", {"prompt": "Berlin"}, env)
    elapsed = time.perf_counter() - started
    assert got.returncode == 0, got.stderr[-1500:]
    assert "keyword match" in got.stdout and "Berlin" in got.stdout
    assert elapsed < 5.0, f"recall_prompt took {elapsed:.1f}s without a daemon"


def test_codex_hooks_round_trip_through_the_http_daemon_with_fake_llm(tmp_path):
    from helpers import openai_stub

    runbook = json.dumps(
        {
            "summary": "kept Codex hook memory",
            "tasks": [
                {
                    "name": "Keep Codex hook memory",
                    "outcome": "success",
                    "procedure": ["capture the prompt", "distill the rollout"],
                    "keywords": ["codex-hook-e2e"],
                    "stage": "verify",
                    "steps": [0, 1],
                }
            ],
        }
    )
    with openai_stub([runbook]) as (llm_url, requests):
        cfg = tmp_path / "agmem.toml"
        cfg.write_text(
            '[profile]\nname = "lite"\n\n[override]\nembedder = "FakeEmbedder"\n'
            f'\n[llm.distill]\nendpoint = "{llm_url}"\nmodel = "stub"\napi_key = "stub"\n'
        )
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        env = {
            **_base_env(tmp_path, url),
            "AGMEM_CONFIG": str(cfg),
            "AGMEM_HOOK_SOURCE": "codex",
        }
        proc = _spawn_daemon(env, port, "--organizers", "experience")
        try:
            _wait_health(url)
            capture = _run_hook(
                "agmem.hooks.capture",
                {
                    "session_id": "codex-capture",
                    "prompt": "remember codex-hook-e2e prompt capture",
                    "cwd": str(tmp_path),
                },
                env,
            )
            assert capture.returncode == 0, capture.stderr[-1500:]
            recalled = _run_hook(
                "agmem.hooks.recall_prompt",
                {
                    "session_id": "codex-capture",
                    "prompt": "what did codex-hook-e2e capture?",
                    "cwd": str(tmp_path),
                },
                env,
            )
            assert recalled.returncode == 0, recalled.stderr[-1500:]
            assert "codex-hook-e2e prompt capture" in recalled.stdout

            ts = "2026-09-07T10:00:00.000Z"
            compact_rollout = tmp_path / "rollout-2026-09-07T10-00-00-codex-compact.jsonl"
            compact_rollout.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [
                        {
                            "timestamp": ts,
                            "type": "session_meta",
                            "payload": {
                                "id": "codex-compact",
                                "session_id": "codex-compact",
                                "timestamp": ts,
                                "cwd": str(tmp_path),
                                "thread_source": "user",
                                "source": "cli",
                            },
                        },
                        {
                            "timestamp": ts,
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": "preserve codex compact rollout",
                                    }
                                ],
                            },
                        },
                        {
                            "timestamp": ts,
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "preserved"}],
                            },
                        },
                    ]
                )
                + "\n"
            )
            preserve = _run_hook(
                "agmem.hooks.preserve",
                {"session_id": "codex-compact", "transcript_path": str(compact_rollout)},
                env,
            )
            assert preserve.returncode == 0, preserve.stderr[-1500:]

            end_rollout = tmp_path / "rollout-2026-09-07T10-01-00-codex-end.jsonl"
            end_rollout.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [
                        {
                            "timestamp": ts,
                            "type": "session_meta",
                            "payload": {
                                "id": "codex-end",
                                "session_id": "codex-end",
                                "timestamp": ts,
                                "cwd": str(tmp_path),
                                "thread_source": "user",
                                "source": "cli",
                            },
                        },
                        {
                            "timestamp": ts,
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "distill codex-hook-e2e rollout"}
                                ],
                            },
                        },
                        {
                            "timestamp": ts,
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "done"}],
                            },
                        },
                    ]
                )
                + "\n"
            )
            preserved_end = _run_hook(
                "agmem.hooks.preserve",
                {"session_id": "codex-end", "transcript_path": str(end_rollout)},
                env,
            )
            assert preserved_end.returncode == 0, preserved_end.stderr[-1500:]
            distill = _run_hook(
                "agmem.hooks.distill",
                {"session_id": "codex-end", "transcript_path": str(end_rollout)},
                {**env, "AGMEM_HOOK_TIMEOUT_SEC": "1.5"},
            )
            assert distill.returncode == 0, distill.stderr[-1500:]

            from agmem.stores.sqlite_doc import SqliteDocStore

            store = SqliteDocStore(tmp_path / "data" / "daemontest" / "memory.db")
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    episodes = store.list_episodes(namespace="daemontest")
                    runbooks = store.list_items("runbooks", namespace="daemontest")
                    if requests and runbooks:
                        break
                    time.sleep(0.2)
                else:
                    raise AssertionError("Codex distill hook did not produce a fake-LLM runbook")
                assert any("preserve codex compact rollout" in ep.content for ep in episodes)
                assert runbooks[0]["name"] == "Keep Codex hook memory"
                assert len(requests) == 1
            finally:
                store.close()
            for module in ("recall", "recall_prompt"):
                recalled = _run_hook(
                    f"agmem.hooks.{module}",
                    {
                        "session_id": "next-session",
                        "cwd": str(tmp_path),
                        "prompt": "codex-hook-e2e",
                    },
                    env,
                )
                assert recalled.returncode == 0, recalled.stderr
                context = json.loads(recalled.stdout)["hookSpecificOutput"]["additionalContext"]
                assert "Keep Codex hook memory" in context
            assert len(requests) == 1
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


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
        if _health(daemon["url"], pending=True)["pending_embed"]["daemontest"] == 0:
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
    # A dogfooding machine exports AGMEM_NO_DAEMON=1 from the harness settings
    # (the systemd daemon owns the port there), and `ensure_running` reads the
    # process environment, not `env`: dropping the key from the dict alone left
    # this test failing everywhere the product is actually installed.
    os.environ.pop("AGMEM_NO_DAEMON", None)
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


def test_a_hook_that_fires_while_the_daemon_is_still_starting_does_not_spawn_a_second(
    tmp_path, monkeypatch
):
    """The startup window is ~20 s and /health fails throughout it; two hooks
    in that window used to start two daemons, and the second died on the kuzu
    file lock (dogfood log, 2026-09-05). A spawn is now noted in a marker file
    and trusted for SPAWN_GRACE_S; a stale marker does not block."""
    from agmem.hooks import daemon as daemon_client

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    monkeypatch.delenv("AGMEM_NO_DAEMON", raising=False)
    monkeypatch.setattr(
        daemon_client, "spawn_command", lambda *a, **k: [sys.executable, "-c", "pass"]
    )
    spawned: list[list[str]] = []
    real_popen = daemon_client.subprocess.Popen

    def counting_popen(argv, **kwargs):
        spawned.append(argv)
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(daemon_client.subprocess, "Popen", counting_popen)
    marker = daemon_client.spawn_marker(url)
    marker.unlink(missing_ok=True)

    assert daemon_client.ensure_running(url) is False
    assert daemon_client.ensure_running(url) is False  # inside the grace window
    assert len(spawned) == 1 and marker.exists()
    os.utime(marker, (time.time() - 3600, time.time() - 3600))  # a stale marker
    assert daemon_client.ensure_running(url) is False
    assert len(spawned) == 2
    marker.unlink(missing_ok=True)


def test_prompt_capture_and_recall_use_hook_namespace_when_daemon_default_differs(daemon):
    env = {**daemon["env"], "AGMEM_NAMESPACE": "prompt-isolation"}
    query = "namespaceparityquartz"
    _post(daemon["url"], "/hooks/capture", {"content": query + " wrongdefault"})
    captured = _run_hook(
        "agmem.hooks.capture",
        {"prompt": query + " correctnamespace", "cwd": "/work/parity"},
        env,
    )
    assert captured.returncode == 0, captured.stderr
    recalled = _run_hook("agmem.hooks.recall_prompt", {"prompt": query, "cwd": "/work/parity"}, env)
    assert recalled.returncode == 0, recalled.stderr
    context = json.loads(recalled.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "semantic search" in context
    assert "correctnamespace" in context
    assert "wrongdefault" not in context
    fallback = _run_hook(
        "agmem.hooks.recall_prompt",
        {"prompt": query, "cwd": "/work/parity"},
        {**env, "AGMEM_DAEMON_URL": "http://127.0.0.1:1"},
    )
    fallback_context = json.loads(fallback.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "keyword match" in fallback_context
    assert "correctnamespace" in fallback_context
    assert "wrongdefault" not in fallback_context


def test_lifecycle_hooks_use_hook_namespace_when_daemon_default_differs(daemon):
    env = {**daemon["env"], "AGMEM_NAMESPACE": "lifecycle-isolation"}
    for hook_name in ("preserve", "distill"):
        query = f"lifecyclequartz{hook_name}"
        transcript = daemon["root"] / f"{hook_name}-namespace.jsonl"
        transcript.write_text(
            "\n".join(
                json.dumps(
                    {
                        "type": role,
                        "uuid": f"{hook_name}-{role}",
                        "sessionId": f"lifecycle-{hook_name}",
                        "timestamp": "2026-09-09T00:00:00.000Z",
                        "cwd": "/work/lifecycle",
                        "message": {
                            "role": role,
                            "content": content
                            if role == "user"
                            else [{"type": "text", "text": content}],
                        },
                    }
                )
                for role, content in (
                    ("user", query + " remember this decision"),
                    ("assistant", "done"),
                )
            )
            + "\n"
        )
        result = _run_hook(
            f"agmem.hooks.{hook_name}",
            {"session_id": f"lifecycle-{hook_name}", "transcript_path": str(transcript)},
            env,
        )
        assert result.returncode == 0, result.stderr
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            reply = _post(
                daemon["url"],
                "/hooks/recall",
                {"query": query, "namespace": "lifecycle-isolation", "k": 10},
            )
            if any(query in item["text"] for item in reply["items"]):
                break
            time.sleep(0.1)
        else:
            raise AssertionError(f"{hook_name} did not write to hook namespace")
        default = _post(daemon["url"], "/hooks/recall", {"query": query, "k": 10})
        assert not any(query in item["text"] for item in default["items"])
