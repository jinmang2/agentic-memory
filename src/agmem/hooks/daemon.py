"""The hooks' side of the daemon: a stdlib HTTP client and a spawner.

WHY A DAEMON. A hook is a fresh process per event, and the capture hook needed
the embedder — ~11 s of import and model construction per prompt, 56 s after a
cold page cache, for ~50 ms of actual work (issue #2 §1). The fix is to load
the model once in a process that stays up, and that process already existed:
`agmem-mcp --transport http`. The hooks now talk to it over loopback HTTP and
never import torch.

WHY STDLIB ONLY. Importing the MCP SDK (or anything that pulls in pydantic and
httpx) in a hook would put seconds back onto the same path this file exists to
clear. `urllib` is enough for three JSON endpoints; the daemon exposes plain
HTTP routes for the hooks precisely so they do not need to speak MCP.

WHEN THE DAEMON IS ABSENT. Decided in the Phase 2 spec: a hook never falls back
to loading the embedder in-process (that is the problem, not a fallback). The
capture hook writes the episode to the doc store without a vector and lets the
daemon backfill it on its next start; the recall hooks emit nothing. Whoever
notices the daemon is down asks for it to be started, which is `ensure_running`:
a detached `Popen` that returns immediately. Two hooks racing to start it is
harmless — the second process fails to bind the port and exits.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agmem.env import resolve_daemon_url

# A hook must never wait on the daemon for long: recall blocks session start
# and recall_prompt blocks the turn. Health is a local socket round trip.
HEALTH_TIMEOUT_S = 0.3
REQUEST_TIMEOUT_S = 5.0
DEFAULT_IDLE_TIMEOUT_S = 1800


class DaemonUnavailable(Exception):
    """The daemon did not answer; the caller takes its documented absent-path."""


def _url(path: str, url: str | None = None) -> str:
    return f"{resolve_daemon_url(url)}{path}"


def health(url: str | None = None, timeout: float = HEALTH_TIMEOUT_S) -> dict[str, Any] | None:
    """The daemon's `/health` payload, or None when it is not answering.

    None rather than an exception because "not running" is the ordinary case
    for the first hook of the day, not an error to report."""
    try:
        with urllib.request.urlopen(_url("/health", url), timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) and payload.get("ok") else None


def post(
    path: str, payload: dict[str, Any], url: str | None = None, timeout: float = REQUEST_TIMEOUT_S
) -> dict[str, Any]:
    """POST JSON to the daemon and return its JSON reply; `DaemonUnavailable` on any failure."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        _url(path, url),
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            reply = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise DaemonUnavailable(str(exc)) from exc
    if not isinstance(reply, dict):
        raise DaemonUnavailable(f"non-object reply from {path}")
    return reply


def spawn_command(
    url: str | None = None, idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S
) -> list[str]:
    """The argv that starts the daemon this hook process would talk to.

    Uses the interpreter running the hook so the daemon is the same
    installation — the confusion `docs/05 §2.3` warns about for PATH-resolved
    console scripts applies here too. Namespace, data dir and config are NOT
    passed: the daemon reads the same environment variables the hook did
    (`agmem.env`), which is the whole point of having one resolution path."""
    parsed = urlparse(resolve_daemon_url(url))
    return [
        sys.executable,
        "-m",
        "agmem.mcp.server",
        "--transport",
        "http",
        "--host",
        parsed.hostname or "127.0.0.1",
        "--port",
        str(parsed.port or 8765),
        "--idle-timeout",
        str(idle_timeout_s),
        # v1: the daemon the hooks run is the one that distils sessions into
        # runbooks (`agmem.hooks.distill`), so it carries the experience
        # organizer. The server's own `--organizers` default is unchanged for
        # stdio use.
        "--organizers",
        "experience",
    ]


def ensure_running(
    url: str | None = None, log_path: str | Path | None = None, idle_timeout_s: int | None = None
) -> bool:
    """Start the daemon if `/health` does not answer. Returns True if it was already up.

    Detached (`start_new_session`) so the hook can exit while the daemon keeps
    running, with stdout/stderr appended to `log_path` (or discarded) — a hook
    must not inherit a pipe that would hold the harness open. The caller does
    not wait for readiness: the daemon takes ~10 s to load the embedder, and a
    hook has no business blocking on that. The next hook finds it up."""
    if health(url) is not None:
        return True
    if os.environ.get("AGMEM_NO_DAEMON") == "1":
        # Tests, and machines that run the daemon some other way: report
        # "not up" honestly and do not start anything.
        return False
    env = dict(os.environ)
    env.setdefault("AGMEM_DAEMON_SPAWNED_BY", "hook")
    log_target = open(log_path, "ab") if log_path else subprocess.DEVNULL  # noqa: SIM115
    try:
        subprocess.Popen(
            spawn_command(
                url, DEFAULT_IDLE_TIMEOUT_S if idle_timeout_s is None else idle_timeout_s
            ),
            stdin=subprocess.DEVNULL,
            stdout=log_target,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    finally:
        if log_target is not subprocess.DEVNULL:
            log_target.close()
    return False
