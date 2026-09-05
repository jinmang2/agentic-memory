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

from agmem.core.origin import item_cwd, same_project
from agmem.hooks import daemon as daemon_client
from agmem.hooks import emit_context, fail_open, open_doc_store, read_event

MAX_EPISODES = 12
MAX_CHARS = 2000
MAX_RUNBOOKS = 5
HEADER = (
    "Recent memory from previous sessions (agmem, most recent first). "
    "This is a recency listing, not a relevance search — no query existed yet at "
    "session start. Treat it as a reminder of what was going on, and search "
    "memory explicitly when you need an answer."
)


COMPACT_HEADER = (
    "What this session said before the context was compacted (agmem, preserved by the "
    "PreCompact hook, most recent first). The transcript's raw steps are in memory under "
    "this session id; search memory for anything the summary lost."
)


RUNBOOK_HEADER = (
    "Runbooks distilled from previous sessions in this project (agmem, newest first): "
    "the task, how it ended, and the stage it reached. The steps behind each are in "
    "memory under that session; search_memory / research_memory find them."
)


def recent_runbooks(store, namespace: str, project: str | None, limit: int = MAX_RUNBOOKS) -> list:
    """The newest live runbooks written from `project` (research §6 #9: another
    repository's experience is not this session's), newest session first.

    This is the block the dogfood record (docs/23 §8) found missing: the
    SessionStart listing showed the user's recent turns and nothing of what the
    previous sessions had been distilled into, so a runbook reached the model
    only when a prompt happened to retrieve it."""
    rows = [d for d in store.list_items("runbooks", namespace=namespace) if not d.get("deleted")]
    if project:
        rows = [d for d in rows if same_project(item_cwd(d), project)]
    # Newest session first; inside a session, the latest work first — a
    # segmented distillation (docs/23 §8) leaves thirty-odd runbooks with one
    # ended_at, and the five shown should be where the session left off.
    rows.sort(
        key=lambda d: (
            str((d.get("origin") or {}).get("ended_at") or ""),
            (d.get("step_range") or [-1])[0],
        ),
        reverse=True,
    )
    return rows[:limit]


def render_runbooks(rows: list) -> str:
    """One line per runbook. A session's runbooks share one session summary
    (the organizer writes it once per session), so the summary follows the
    first runbook of each session only; the keywords are per runbook."""
    lines = []
    seen_summaries: set[str] = set()
    for d in rows:
        when = str((d.get("origin") or {}).get("ended_at") or "")[:10] or "?"
        name = " ".join(str(d.get("name") or "").split())
        if not name:
            continue
        marks = " ".join(f"{k}:{d[k]}" for k in ("outcome", "stage") if d.get(k))
        keywords = d.get("keywords") or []
        if isinstance(keywords, list):
            keywords = ", ".join(str(k) for k in keywords)
        summary = " ".join(str(d.get("summary") or "").split())
        if summary in seen_summaries:
            summary = ""
        elif summary:
            seen_summaries.add(summary)
        line = f"- ({when}) {name}" + (f" [{marks}]" if marks else "")
        if keywords:
            line += f" — {keywords}"
        if summary:
            line += f". {summary}"
        if len(line) > 300:
            line = line[:297] + "..."
        lines.append(line)
    if not lines:
        return ""
    return RUNBOOK_HEADER + "\n" + "\n".join(lines)


def render(episodes: list, header: str = HEADER) -> str:
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
    return header + "\n" + "\n".join(lines)


def main() -> None:
    try:
        event = read_event()  # also drained so the harness never blocks on our stdin
        project = str(event.get("cwd") or "") or None
        namespace, store = open_doc_store()
        try:
            episodes = store.list_episodes(namespace=namespace)
            runbooks = recent_runbooks(store, namespace, project)
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
        # After a compaction (research §6 #11), the loss to repair is THIS
        # session's own turns, which the PreCompact hook preserved under its
        # id; a global recency listing would be the wrong memory. Falls
        # through to recency when nothing of this session is in the store.
        session_id = str(event.get("session_id") or "")
        if event.get("source") == "compact" and session_id:
            own = [
                ep
                for ep in episodes
                if (getattr(ep, "meta", None) or {}).get("session_id") == session_id
            ]
            if own:
                emit_context(render(own[-MAX_EPISODES:][::-1], COMPACT_HEADER), "SessionStart")
                daemon_client.ensure_running()
                sys.exit(0)
        # Project gating (research §6 #9): a turn typed in another repository
        # is not this session's recent memory. Turns with no recorded cwd pass.
        if project:
            episodes = [ep for ep in episodes if same_project(item_cwd(ep), project)]
        # `list_episodes` documents oldest-first, so the tail is the newest slice
        # and reversing it puts the most recent line first.
        episodes = list(episodes)[-MAX_EPISODES:][::-1]
        # Runbooks first: they are what the previous sessions concluded; the
        # turns below them are what was said. Either block may be empty.
        blocks = [b for b in (render_runbooks(runbooks), render(episodes)) if b]
        emit_context("\n\n".join(blocks), "SessionStart")
        # Session start is where the daemon gets started if it is not up, so
        # the first capture of the session finds it warm. Non-blocking: this
        # returns as soon as the process is spawned, not when it is ready.
        if os.environ.get("AGMEM_NO_DAEMON") != "1":
            daemon_client.ensure_running(log_path=os.environ.get("AGMEM_DAEMON_LOG"))
    except BaseException as exc:  # every failure path exits 0 — see fail_open
        if isinstance(exc, SystemExit):
            raise
        fail_open(
            exc, notice="agmem: session-start recall failed, this session runs without memory"
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
