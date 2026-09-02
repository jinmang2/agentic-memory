"""SessionStart hook: put what the store already knows in front of the model.

Wire as::

    {"hooks": {"SessionStart": [{"hooks": [
      {"type": "command", "command": "python -m agmem.hooks.recall", "timeout": 10}
    ]}]}}

WHY THIS IS THE SYNCHRONOUS ONE. Capture can run async — nobody is waiting on a
write. Recall cannot: its output only matters if it lands before the model's
first token, so it blocks session start and must therefore be fast and bounded.
The `timeout` above is the real limit; everything here is sized to stay well
under it, and anything that cannot exits 0 with no context rather than making
the user wait.

Retrieval is unconditional, not query-driven, because SessionStart has no query
to drive it — there is no prompt yet. What it can offer is recency, so it serves
the most recent episodes of this namespace. That is a deliberately weak signal
and it is why the injected block says what it is: a model told "here is what you
remember" will trust it more than a recency dump deserves. The query-driven
counterpart is `agmem.hooks.recall_prompt`, on `UserPromptSubmit`.

This hook is also where the daemon gets started (`agmem.hooks.daemon`): it
runs once per session, before any capture, and spawning is a non-blocking
`Popen`. Set `AGMEM_NO_DAEMON=1` to keep it from doing that (tests, or a
machine that runs the daemon some other way).
"""

from __future__ import annotations

import os
import sys

from agmem.hooks import daemon as daemon_client
from agmem.hooks import emit_context, fail_open, open_doc_store, read_event

MAX_EPISODES = 12
MAX_CHARS = 2000
HEADER = (
    "Recent memory from previous sessions (agmem, most recent first). "
    "This is a recency listing, not a relevance search — no query existed yet at "
    "session start. Treat it as a reminder of what was going on, and search "
    "memory explicitly when you need an answer."
)


def render(episodes: list) -> str:
    lines = []
    used = 0
    for ep in episodes:
        content = " ".join(str(getattr(ep, "content", "") or "").split())
        if not content:
            continue
        stamp = getattr(ep, "timestamp", None)
        when = stamp.date().isoformat() if stamp is not None else "?"
        line = f"- ({when}) {content}"
        if len(line) > 300:
            line = line[:297] + "..."
        if used + len(line) > MAX_CHARS:
            break
        lines.append(line)
        used += len(line)
    if not lines:
        return ""
    return HEADER + "\n" + "\n".join(lines)


def main() -> None:
    try:
        read_event()  # drained so the harness never blocks writing to our stdin
        namespace, store = open_doc_store()
        try:
            episodes = store.list_episodes(namespace=namespace)
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                close()
        # Only what a user said. The capture hook writes `role="user"` alone, so
        # this used to be the whole store; `add_session` now files every step of a
        # session log — tool-call JSON and tool output included — and twelve of
        # those would fill the listing with an agent's working notes instead of
        # the user's recent requests. A full scan is cheap enough here (0.006 s
        # per 500 episodes measured), and the ordering contract of
        # `list_episodes` still holds after the filter.
        episodes = [ep for ep in episodes if getattr(ep, "role", "user") == "user"]
        # `list_episodes` documents oldest-first, so the tail is the newest slice
        # and reversing it puts the most recent line first.
        episodes = list(episodes)[-MAX_EPISODES:][::-1]
        emit_context(render(episodes), "SessionStart")
        # Session start is where the daemon gets started if it is not up, so
        # the first capture of the session finds it warm. Non-blocking: this
        # returns as soon as the process is spawned, not when it is ready.
        if os.environ.get("AGMEM_NO_DAEMON") != "1":
            daemon_client.ensure_running(log_path=os.environ.get("AGMEM_DAEMON_LOG"))
    except BaseException as exc:  # every failure path exits 0 — see fail_open
        if isinstance(exc, SystemExit):
            raise
        fail_open(exc)
    sys.exit(0)


if __name__ == "__main__":
    main()
