"""PreCompact hook: keep the session's raw transcript before the harness compacts it.

Wire as::

    {"hooks": {"PreCompact": [{"hooks": [
      {"type": "command", "command": "python -m agmem.hooks.preserve", "timeout": 5}
    ]}]}}

WHY THIS HOOK EXISTS. Working memory is the axis where the research found no
controlled evidence for compression techniques and strong demand for
*control* over compaction (docs/research/agent-memory-axes-v1.md §3.1, §5.2,
§6 #11: "not a compression method — control placement"). The harness will
compact regardless; what this hook does is make sure the original survives
somewhere the next turn can reach: every step of the transcript goes into
the store as episodes, under this session's id, with its origin bound
(`AgenticMemory.add_session(distill=False)`). After compaction, the
SessionStart hook (`agmem.hooks.recall`, `source == "compact"`) lists what
this session said before the cut, from those episodes.

WHAT IT NEVER DOES. Load the embedder or call a model: the transcript is
handed to the daemon (`POST /hooks/preserve`), which ingests it on a worker
thread and answers at once. Without a daemon the path is appended to a spool
file the daemon drains on its next start (`_Registry.drain_preserve_queue`),
and a daemon start is requested — the same absent-daemon shape `capture` has.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from agmem.hooks import _resolve, fail_open, read_event
from agmem.hooks import daemon as daemon_client

SPOOL_NAME = "preserve-queue.jsonl"


def spool_path(namespace: str | None = None, data_dir: str | None = None) -> Path:
    ns, root, _config = _resolve(namespace, data_dir)
    return root / ns / SPOOL_NAME


def request_body(event: dict) -> dict | None:
    """What the daemon needs, or None when the event names no transcript."""
    path = str(event.get("transcript_path") or "")
    if not path:
        return None
    body = {"transcript_path": path, "session_id": str(event.get("session_id") or "")}
    cwd = str(event.get("cwd") or "")
    if cwd:
        body["cwd"] = cwd
    return body


def spool(body: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(body, ensure_ascii=False) + "\n")


def main() -> None:
    try:
        body = request_body(read_event())
        if body is None:
            sys.exit(0)
        if daemon_client.health() is not None:
            daemon_client.post("/hooks/preserve", body)
        else:
            spool(body, spool_path())
            daemon_client.ensure_running(log_path=os.environ.get("AGMEM_DAEMON_LOG"))
    except BaseException as exc:  # every failure path exits 0 — see fail_open
        if isinstance(exc, SystemExit):
            raise
        fail_open(exc)
    sys.exit(0)


if __name__ == "__main__":
    main()
