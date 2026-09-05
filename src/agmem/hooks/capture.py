"""UserPromptSubmit hook: write the turn to memory as it happens.

Wire as::

    {"hooks": {"UserPromptSubmit": [{"hooks": [
      {"type": "command", "command": "python -m agmem.hooks.capture", "async": true}
    ]}]}}

`async` is the point. Capture produces nothing the model needs this turn, so it
must not sit between the user pressing enter and the model starting — the write
happens in the background and the turn is never held for it.

It emits NO context. A capture hook that also injected something would make
every prompt echo itself back into the transcript, and the recall hook already
owns the memory-to-context direction. One direction per hook.

DETERMINISM IS THE FEATURE. The portfolio note behind this file says capture has
to be deterministic before any of it is worth feeling, and that is a claim about
this hook specifically: the model deciding when to remember is the failure mode
that MCP tools already have. Here the harness decides, on every prompt, whether
or not anyone thought about it.

THE COST THAT WAS HERE, AND WHERE IT WENT. Until 2026-09-02 this hook loaded
the embedder itself: ~11 s of `import torch`, `import sentence_transformers`
and model construction per prompt (56 s after a cold page cache), for under a
second of actual work. It was documented as an architecture problem, and the
architecture is now the daemon: the same `agmem-mcp --transport http` process
the MCP layer runs, kept warm, with a plain `POST /hooks/capture` route
(`agmem.hooks.daemon`). Against it this hook is a JSON round trip.

WHEN THERE IS NO DAEMON, the hook does NOT load the model — that would be the
old cost coming back through a side door. It writes the episode to the doc
store alone (0.2 s, the recall hook's path), asks for the daemon to be started,
and exits. The episode is visible to the recency hook at once and to semantic
search after the daemon's backfill gives it a vector, which happens on the
daemon's next start and every minute after. What issue #2 warned against was
an episode with no vector *forever*; a delayed one is the trade this design
makes, and `/health` reports how many are waiting.
"""

from __future__ import annotations

import os
import sys

from agmem.hooks import daemon as daemon_client
from agmem.hooks import fail_open, open_doc_store, read_event

MAX_CHARS = 8000


def prompt_of(event: dict) -> str:
    """The user's text, across the field names the harness has used for it.

    Checked in order rather than assuming one: a payload whose shape moved would
    otherwise capture empty strings forever, and an empty capture is invisible —
    the hook keeps exiting 0 and the store keeps not growing.
    """
    for key in ("prompt", "user_prompt", "message", "content"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def write_without_daemon(text: str, meta: dict) -> None:
    """The absent-daemon path: persist the episode without a vector and request a daemon.

    `pending_embed` in `meta` is informational — the daemon derives what to
    backfill by comparing the doc store with the vector store, so a flag that
    went stale could not hide an episode from repair."""
    from agmem.core.types import Episode

    namespace, store = open_doc_store()
    try:
        store.add_episode(
            Episode(
                content=text, role="user", namespace=namespace, meta={**meta, "pending_embed": True}
            )
        )
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()
    daemon_client.ensure_running(log_path=os.environ.get("AGMEM_DAEMON_LOG"))


def main() -> None:
    try:
        event = read_event()
        text = prompt_of(event)
        if not text.strip():
            sys.exit(0)  # nothing said, nothing to store
        meta = {
            "source": str(
                event.get("source") or os.environ.get("AGMEM_HOOK_SOURCE") or "claude-code"
            ),
            # Kept so a later reader can group a session's turns without
            # inferring it from timestamps.
            "session_id": str(event.get("session_id") or ""),
            # The project this turn belongs to (origin binding, research §6
            # #8): what the recall hooks gate on, so a turn typed in one
            # repository is not served into another.
            "cwd": str(event.get("cwd") or "") or None,
        }
        if daemon_client.health() is not None:
            daemon_client.post(
                "/hooks/capture", {"content": text[:MAX_CHARS], "role": "user", "meta": meta}
            )
        else:
            write_without_daemon(text[:MAX_CHARS], meta)
    except BaseException as exc:  # every failure path exits 0 — see fail_open
        if isinstance(exc, SystemExit):
            raise
        fail_open(exc)
    sys.exit(0)


if __name__ == "__main__":
    main()
