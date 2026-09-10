from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypedDict

import anyio

NEEDLE: Final = "Reykjavik"
PROMPT: Final = f"Remember that the {NEEDLE} deployment window is the second Tuesday of each month."
QUERY: Final = "When can I deploy?"
RUNBOOK_NAME: Final = "Keep product smoke memory"
RUNBOOK_KEYWORD: Final = "product-smoke-runbook"
HOOK_TIMEOUT_S: Final = 60.0
MCP_TIMEOUT_S: Final = 60.0
STARTUP_TIMEOUT_S: Final = 40.0
BACKFILL_TIMEOUT_S: Final = 20.0


class HookPayload(TypedDict, total=False):
    session_id: str
    prompt: str
    hook_event_name: str
    transcript_path: str
    cwd: str
    source: str


class HttpReply(TypedDict):
    ok: bool
    pid: int | None
    pending_embed: dict[str, int]


def run_hook(
    module: str, payload: HookPayload, env: dict[str, str], timeout_s: float = HOOK_TIMEOUT_S
) -> tuple[float, subprocess.CompletedProcess[str]]:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", module],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_s,
        check=False,
    )
    return time.perf_counter() - started, proc


def _text(result) -> str:
    return " ".join(getattr(c, "text", "") for c in (getattr(result, "content", None) or []))


async def search_over_mcp(env: dict[str, str]) -> tuple[float, str, set[str], str]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    command = str(Path(sys.executable).parent / "agmem-mcp")
    params = StdioServerParameters(command=command, args=["--organizers", ""], env=env)
    started = time.perf_counter()
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        handshake = time.perf_counter() - started
        tools = {t.name for t in (await session.list_tools()).tools}
        rendered = _text(await session.call_tool("search_memory", {"query": QUERY}))
        stats = json.loads(_text(await session.call_tool("memory_stats", {})))
    return handshake, rendered, tools, str(stats["stats"]["namespace"])


def mcp_search(
    env: dict[str, str], timeout_s: float = MCP_TIMEOUT_S
) -> tuple[float, str, set[str], str]:
    return anyio.run(timed_search_over_mcp, env, timeout_s)


async def timed_search_over_mcp(
    env: dict[str, str], timeout_s: float
) -> tuple[float, str, set[str], str]:
    with anyio.fail_after(timeout_s):
        return await search_over_mcp(env)


def http_json(
    url: str, path: str, payload: dict[str, str] | None = None, timeout: float = 5.0
) -> Mapping[str, str | int | bool | dict[str, int]]:
    req = urllib.request.Request(
        f"{url}{path}",
        data=None if payload is None else json.dumps(payload).encode(),
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def parse_health(raw: Mapping[str, str | int | bool | dict[str, int]]) -> HttpReply | None:
    if raw.get("ok") is not True:
        return None
    pid_raw = raw.get("pid")
    pending_raw = raw.get("pending_embed")
    return {
        "ok": True,
        "pid": pid_raw if isinstance(pid_raw, int) else None,
        "pending_embed": pending_raw if isinstance(pending_raw, dict) else {},
    }


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_health(url: str, timeout_s: float = STARTUP_TIMEOUT_S) -> HttpReply:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            health = parse_health(http_json(url, "/health", timeout=0.5))
            if health is not None:
                return health
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(0.2)
    raise TimeoutError(f"daemon at {url} did not come up within {timeout_s:.1f}s: {last_error}")


def wait_backfill(url: str, timeout_s: float = BACKFILL_TIMEOUT_S) -> HttpReply:
    deadline = time.monotonic() + timeout_s
    last: HttpReply | None = None
    while time.monotonic() < deadline:
        health = parse_health(http_json(url, "/health?pending=1", timeout=2))
        if health is not None:
            last = health
            pending = health["pending_embed"]
            if pending and all(count == 0 for count in pending.values()):
                return health
        time.sleep(0.2)
    raise TimeoutError(f"backfill still pending after {timeout_s:.1f}s: {last}")


def terminate_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def spawn_owned_daemon(env: dict[str, str], port: int) -> subprocess.Popen[bytes]:
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
            "experience",
            "--backfill-period",
            "1",
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
