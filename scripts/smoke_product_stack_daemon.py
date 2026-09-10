from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from scripts.smoke_product_stack_core import (
    NEEDLE,
    PROMPT,
    QUERY,
    RUNBOOK_KEYWORD,
    RUNBOOK_NAME,
    free_port,
    run_hook,
    spawn_owned_daemon,
    terminate_process,
    wait_backfill,
    wait_health,
)
from scripts.smoke_product_stack_fixture import write_rollout


@contextmanager
def environ(env: dict[str, str]) -> Iterator[None]:
    old = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def daemon_path(env: dict[str, str], data_dir: Path) -> bool:
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    daemon_env = {
        **env,
        "AGMEM_DAEMON_URL": url,
        "AGMEM_DAEMON_LOG": str(data_dir / "daemon.log"),
        "AGMEM_HOOK_SOURCE": "codex",
    }
    daemon_env.pop("AGMEM_NO_DAEMON", None)
    proc: subprocess.Popen[bytes] | None = None
    print(f"\n--- daemon path ({url}) ---")
    try:
        if not capture_without_daemon(env, data_dir):
            return False
        proc = spawn_owned_daemon(daemon_env, port)
        started = time.perf_counter()
        health = wait_health(url)
        print(f"daemon up          {time.perf_counter() - started:6.2f}s  pid={health['pid']}")
        started = time.perf_counter()
        health = wait_backfill(url)
        print(
            f"backfill done      {time.perf_counter() - started:6.2f}s  "
            f"pending={health['pending_embed']}"
        )
        doctor_ok = doctor_reports_owned_daemon(daemon_env, data_dir, url)
        return verify_daemon_hooks(daemon_env, data_dir) and doctor_ok
    except (OSError, TimeoutError, subprocess.SubprocessError) as exc:
        print(f"daemon path failed: {exc}")
        return False
    finally:
        if proc is not None:
            terminate_process(proc)


def capture_without_daemon(env: dict[str, str], data_dir: Path) -> bool:
    elapsed, proc = run_hook(
        "agmem.hooks.capture",
        {
            "session_id": "smoke-cold",
            "prompt": PROMPT,
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(data_dir),
        },
        env,
    )
    print(f"capture, no daemon {elapsed:6.2f}s  exit={proc.returncode}  (doc store only)")
    if proc.returncode == 0:
        return True
    print(f"  stderr: {proc.stderr[-500:]}")
    return False


def verify_daemon_hooks(env: dict[str, str], data_dir: Path) -> bool:
    injected = prompt_recall_finds_needled_capture(env, data_dir)
    runbook_written = preserve_distill_writes_runbook(env, data_dir)
    start_found = next_session_recall_finds_runbook(env, data_dir)
    prompt_found = next_prompt_recall_finds_runbook(env, data_dir)
    return injected and runbook_written and start_found and prompt_found


def prompt_recall_finds_needled_capture(env: dict[str, str], data_dir: Path) -> bool:
    elapsed, proc = run_hook(
        "agmem.hooks.recall_prompt",
        {
            "session_id": "smoke-query",
            "prompt": QUERY,
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(data_dir),
        },
        env,
    )
    found = NEEDLE in proc.stdout and proc.returncode == 0
    print(f"recall_prompt hook {elapsed:6.2f}s  exit={proc.returncode}  found={found}")
    if not found:
        print(f"  recall_prompt stdout: {proc.stdout[:400]}  stderr: {proc.stderr[-300:]}")
    return found


def preserve_distill_writes_runbook(env: dict[str, str], data_dir: Path) -> bool:
    rollout = data_dir / "rollout-product-smoke.jsonl"
    write_rollout(rollout, data_dir)
    for module in ("preserve", "distill"):
        elapsed, proc = run_hook(
            f"agmem.hooks.{module}",
            {"session_id": "product-smoke", "transcript_path": str(rollout), "cwd": str(data_dir)},
            env,
        )
        print(f"{module} hook       {elapsed:6.2f}s  exit={proc.returncode}")
        if proc.returncode != 0:
            print(f"  {module} stderr: {proc.stderr[-300:]}")
            return False
    return runbook_exists(env, data_dir)


def runbook_exists(env: dict[str, str], data_dir: Path) -> bool:
    from agmem.env import resolve_namespace
    from agmem.stores.sqlite_doc import SqliteDocStore

    with environ(env):
        namespace = resolve_namespace()
    store = SqliteDocStore(data_dir / namespace / "memory.db")
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if any(
                rb.get("name") == RUNBOOK_NAME for rb in store.list_items("runbooks", namespace)
            ):
                print("runbook written            found=True")
                return True
            time.sleep(0.2)
    finally:
        store.close()
    print("runbook written            found=False")
    return False


def doctor_reports_owned_daemon(env: dict[str, str], data_dir: Path, url: str) -> bool:
    args = [
        sys.executable,
        "-m",
        "agmem.manage",
        "--data-dir",
        str(data_dir),
        "--config",
        env["AGMEM_CONFIG"],
    ]
    if "AGMEM_NAMESPACE" in env:
        args.extend(["--namespace", env["AGMEM_NAMESPACE"]])
    args.extend(["doctor", "--daemon-url", url])
    started = time.perf_counter()
    proc = subprocess.run(args, capture_output=True, text=True, env=env, timeout=20, check=False)
    ok = proc.returncode == 0 and '"status": "ok"' in proc.stdout
    print(
        f"doctor cli        {time.perf_counter() - started:6.2f}s  exit={proc.returncode}  ok={ok}"
    )
    if not ok:
        print(f"  doctor stdout: {proc.stdout[:400]}  stderr: {proc.stderr[-300:]}")
    return ok


def next_session_recall_finds_runbook(env: dict[str, str], data_dir: Path) -> bool:
    elapsed, proc = run_hook(
        "agmem.hooks.recall",
        {
            "session_id": "next-product-smoke",
            "hook_event_name": "SessionStart",
            "cwd": str(data_dir),
        },
        env,
    )
    found = RUNBOOK_NAME in proc.stdout and proc.returncode == 0
    print(f"next recall hook  {elapsed:6.2f}s  exit={proc.returncode}  found={found}")
    if not found:
        print(f"  next recall stdout: {proc.stdout[:400]}  stderr: {proc.stderr[-300:]}")
    return found


def next_prompt_recall_finds_runbook(env: dict[str, str], data_dir: Path) -> bool:
    elapsed, proc = run_hook(
        "agmem.hooks.recall_prompt",
        {
            "session_id": "next-product-smoke",
            "prompt": RUNBOOK_KEYWORD,
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(data_dir),
        },
        env,
    )
    found = RUNBOOK_NAME in proc.stdout and proc.returncode == 0
    print(f"next prompt recall{elapsed:6.2f}s  exit={proc.returncode}  found={found}")
    if not found:
        print(f"  next prompt stdout: {proc.stdout[:400]}  stderr: {proc.stderr[-300:]}")
    return found
