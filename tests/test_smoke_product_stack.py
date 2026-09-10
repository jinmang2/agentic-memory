from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.smoke_product_stack_core as core
import scripts.smoke_product_stack_fixture as fixture
from agmem.env import DEFAULT_NAMESPACE


def test_wait_health_fails_clearly_when_daemon_never_answers():
    port = core.free_port()
    with pytest.raises(TimeoutError, match="did not come up"):
        core.wait_health(f"http://127.0.0.1:{port}", timeout_s=0.1)


def test_terminate_process_kills_owned_process_that_ignores_sigterm(tmp_path):
    marker = tmp_path / "ready"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, signal, sys, time; "
                "signal.signal(signal.SIGTERM, lambda *_args: None); "
                "pathlib.Path(sys.argv[1]).write_text('ready'); "
                "time.sleep(60)"
            ),
            str(marker),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert marker.exists()
    core.terminate_process(proc)
    assert proc.poll() is not None
    assert proc.returncode == -signal.SIGKILL


def test_write_config_can_force_fake_embedder_and_local_distill_stub(tmp_path):
    cfg = tmp_path / "agmem.toml"
    fixture.write_config(cfg, "http://127.0.0.1:12345")
    text = cfg.read_text()
    assert 'embedder = "FakeEmbedder"' in text
    assert 'endpoint = "http://127.0.0.1:12345"' in text


def test_hermetic_env_does_not_repoint_home_or_namespace_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("AGMEM_NAMESPACE", "user-live")
    monkeypatch.setenv("HOME", "/home/user-live")
    env = fixture.configure_env(None, True, tmp_path, "http://127.0.0.1:12345")
    assert env["HOME"] == "/home/user-live"
    assert "AGMEM_NAMESPACE" not in env
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("AGMEM_DATA_DIR", env["AGMEM_DATA_DIR"])
        patch.setenv("AGMEM_CONFIG", env["AGMEM_CONFIG"])
        patch.delenv("AGMEM_NAMESPACE", raising=False)
        from agmem.env import resolve_namespace

        assert resolve_namespace() == DEFAULT_NAMESPACE
