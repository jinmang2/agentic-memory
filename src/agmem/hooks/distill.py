"""SessionEnd hook: hand the finished session to the daemon to be distilled.

Wire as::

    {"hooks": {"SessionEnd": [{"hooks": [
      {"type": "command", "command": "python -m agmem.hooks.distill", "timeout": 5}
    ]}]}}

THE MISSING LINK, until 2026-09-05. Everything a runbook needs existed —
`add_session` preserved a session's steps and the `experience` organizer
distilled them in one model call — but only the `ingest` CLI ever called it,
so a session ended and nothing was learned from it until someone backfilled
by hand. This hook closes that: when the harness ends a session it names the
transcript, and the daemon runs `add_session(distill=True)` for it in the
background (`POST /hooks/distill`, answered at once — a distillation takes
tens of seconds, a hook has a few).

WHAT IT NEVER DOES. Load a model or pay for one itself. Whether a model is
called at all is the daemon's configuration (`[llm.distill]` in the agmem
TOML); without it the organizer skips explicitly and the raw steps are still
kept. Without a daemon the path is spooled and the daemon drains the spool
on its next start, exactly as `preserve` does.
"""

from __future__ import annotations

import sys

from agmem.hooks import daemon as daemon_client
from agmem.hooks import fail_open, read_event
from agmem.hooks.preserve import request_body, spool, spool_path

SPOOL_NAME = "distill-queue.jsonl"


def main() -> None:
    try:
        body = request_body(read_event())
        if body is None:
            sys.exit(0)
        if daemon_client.health() is not None:
            daemon_client.post("/hooks/distill", body)
        else:
            spool(body, spool_path().with_name(SPOOL_NAME))
            daemon_client.ensure_running()
    except BaseException as exc:  # every failure path exits 0 — see fail_open
        if isinstance(exc, SystemExit):
            raise
        fail_open(exc)
    sys.exit(0)


if __name__ == "__main__":
    main()
