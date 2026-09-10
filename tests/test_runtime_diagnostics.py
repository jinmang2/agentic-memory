from pathlib import Path

import pytest

from agmem.runtime_diagnostics import compare_runtime, runtime_fingerprint


def test_unavailable_daemon_is_not_healthy(tmp_path: Path) -> None:
    result = compare_runtime(None, namespace="main", data_dir=tmp_path, config_path=None)
    assert result.status == "unavailable"


def test_runtime_detects_stale_source_and_wrong_store(tmp_path: Path) -> None:
    result = compare_runtime(
        {"data_dir": str(tmp_path / "other"), "runtime_fingerprint": {}},
        namespace="main",
        data_dir=tmp_path,
        config_path=None,
    )
    assert result.status == "mismatch"
    assert {"data_dir", "runtime_fingerprint"} <= set(result.mismatches)


def test_config_fingerprint_changes_without_exposing_contents(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("setting = 1")
    before = runtime_fingerprint(config)
    config.write_text("setting = 2")
    after = runtime_fingerprint(config)
    assert before["config_sha256"] != after["config_sha256"]
    assert before["source_sha256"] == after["source_sha256"]
    assert "setting" not in str(after)


def test_doctor_command_does_not_create_store_when_daemon_is_down(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    root = tmp_path / "absent-store"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agmem.manage",
            "--data-dir",
            str(root),
            "doctor",
            "--daemon-url",
            "http://127.0.0.1:1",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "unavailable"
    assert not root.exists()


def test_doctor_compares_running_daemon_startup_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import os
    import subprocess
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from scripts.smoke_product_stack_core import (
        free_port,
        spawn_owned_daemon,
        terminate_process,
        wait_health,
    )
    from scripts.smoke_product_stack_fixture import write_config

    config = tmp_path / "config.toml"
    write_config(config)
    root = tmp_path / "data"
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = {k: v for k, v in os.environ.items() if not k.startswith("AGMEM_")}
    env.update(AGMEM_CONFIG=str(config), AGMEM_DATA_DIR=str(root), AGMEM_NAMESPACE="doctor-test")
    proc = spawn_owned_daemon(env, port)
    command = [
        sys.executable,
        "-m",
        "agmem.manage",
        "--config",
        str(config),
        "--data-dir",
        str(root),
        "--namespace",
        "doctor-test",
        "doctor",
        "--daemon-url",
        url,
    ]
    try:
        wait_health(url, timeout_s=15)
        healthy = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=10, check=False
        )
        assert healthy.returncode == 0, healthy.stdout + healthy.stderr
        assert json.loads(healthy.stdout)["status"] == "ok"
        config.write_text(config.read_text() + "\n# changed after daemon startup\n")
        changed = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=10, check=False
        )
        assert changed.returncode == 2
        assert "runtime_fingerprint" in json.loads(changed.stdout)["mismatches"]
    finally:
        terminate_process(proc)
