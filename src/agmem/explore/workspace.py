"""The memory, materialized as files a shell tool can walk.

WHY A DIRECTORY AND NOT AN INDEX. docs/research/agent-memory-axes-v1.md §6
sets v1's control arm: not "no memory", but the raw session logs with a
grep-capable agent on them. LongMemEval-V2's AgentRunbook-C is exactly that —
no vector index, `rg` and `find` over a tree — and it beat the runbook-only
variant by 16.3pp on the small tier. That arm cannot exist while the only way
into this repo's store is `search()`, so this module writes the store out in
the shape those tools already know how to read.

WHAT IT IS NOT. It is a projection, never the source of truth. The doc store
stays authoritative; nothing here reads a file back into memory. A file that
no longer has a row behind it is deleted rather than kept, so a runbook
retired by a re-distillation (``AgenticMemory._retire_session_items``) stops
being greppable at the same moment it stops being searchable — otherwise the
explorer would cite a memory the memory itself has dropped.

THREE INVARIANTS, all of them for the measurement rather than for tidiness:

- **Deterministic.** The same store yields the same bytes. Latency is a
  first-class number on this path (§7.3), and a re-export that rewrites every
  file would put a full disk write inside the timing of every query.
- **Incremental.** A file is opened for writing only when its bytes differ, so
  ``refresh=True`` on ``AgenticMemory.research`` costs a scan and not a
  rewrite.
- **Bounded.** Only ``sessions/``, ``messages/`` and ``runbooks/`` (plus
  ``INDEX.md``) are managed. Anything else under the root — a user's own
  notes, a git checkout, an editor's dotfiles — is never read, written or
  removed. The workspace is a directory somebody may point at their own tools.

THE STEP LABELS ARE THE TRANSCRIPT'S. A session block is written exactly as
``SessionTrajectory.render`` writes it (``[i] KIND(tool)`` then the text), and
``i`` is the stored ``step_index`` rather than a position in this file. That
is what makes a runbook's ``source: ... steps 3-7`` footer resolvable by
reading: the distiller counted the same steps the explorer is now looking at.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("agmem.explore")

MANAGED_DIRS = ("sessions", "messages", "runbooks")
RUNBOOK_TYPE = "runbooks"
INDEX_NAME = "INDEX.md"

# The first user turn shown in the index — enough to recognise a session by,
# short enough that the index stays one screen per host.
INDEX_SNIPPET_CHARS = 120

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


@dataclass
class WorkspaceStats:
    """What one export did. ``written`` counts files whose bytes changed (new
    ones included), so a steady state reports ``written=0`` and a full
    ``unchanged``; that is the assertion a caller makes to show the refresh is
    cheap rather than merely fast on this run."""

    written: int = 0
    unchanged: int = 0
    removed: int = 0
    sessions: int = 0
    runbooks: int = 0
    messages: int = 0


def safe_name(raw: str) -> str:
    """A file name for an id that came from a store.

    Session ids and item ids are already ``[A-Za-z0-9-]`` in every writer we
    have, so this is defence rather than translation: an id is a value some
    host or model supplied, and it becomes a path segment here. Anything else
    collapses to ``_``, and a name that would resolve to a directory entry
    other than itself (``.``, ``..``, empty) becomes ``_``."""
    name = _UNSAFE.sub("_", raw.strip())
    return name if name and name.strip(".") else "_"


def _collapse(text: str) -> str:
    """One line, whitespace normalized — the shape a message index line takes."""
    return " ".join(str(text).split())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _step_label(episode: Any) -> str:
    """``KIND`` or ``KIND(tool)``, the label ``SessionTrajectory.render`` uses."""
    meta = episode.meta or {}
    kind = str(meta.get("kind") or episode.role or "step").upper()
    tool = meta.get("tool_name")
    return f"{kind}({tool})" if tool else kind


def _session_files(episodes: list[Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Group session steps into one file each, and describe them for the index.

    Grouping is on ``meta["session_id"]``, which ``add_session`` writes and the
    hook capture path does not — episodes without one are conversation, not a
    trajectory, and go to ``messages/`` instead."""
    grouped: dict[tuple[str, str], list[Any]] = {}
    for episode in episodes:
        meta = episode.meta or {}
        session_id = meta.get("session_id")
        if not session_id:
            continue
        host = str(meta.get("source") or "unknown")
        grouped.setdefault((host, str(session_id)), []).append(episode)

    files: dict[str, str] = {}
    index: list[dict[str, Any]] = []
    for (host, session_id), items in sorted(grouped.items()):
        # `step_index` is the trajectory's own numbering; the timestamp is the
        # tiebreak for a store that predates it. Sorting by the pair keeps the
        # order stable whichever is present.
        items.sort(key=lambda e: ((e.meta or {}).get("step_index", 0), e.timestamp))
        cwd = next((e.meta.get("cwd") for e in items if (e.meta or {}).get("cwd")), None)
        started = min(e.timestamp for e in items)
        header = [f"session: {session_id}", f"host: {host}"]
        if cwd:
            header.append(f"cwd: {cwd}")
        header.append(f"started: {started.isoformat()}")
        blocks = ["\n".join(header)]
        for episode in items:
            step_index = (episode.meta or {}).get("step_index", 0)
            blocks.append(f"[{step_index}] {_step_label(episode)}\n{episode.content}")
        relpath = f"sessions/{safe_name(host)}/{safe_name(session_id)}.md"
        files[relpath] = "\n\n".join(blocks) + "\n"
        first_user = next(
            (e.content for e in items if ((e.meta or {}).get("kind") or e.role) == "user"), ""
        )
        index.append(
            {
                "host": host,
                "id": session_id,
                "date": started.date().isoformat(),
                "cwd": cwd or "",
                "steps": len(items),
                "first_user": _clip(_collapse(first_user), INDEX_SNIPPET_CHARS),
                "path": relpath,
            }
        )
    return files, index


def _message_files(episodes: list[Any]) -> tuple[dict[str, str], int]:
    """Everything that is not a session step, one file per month, oldest first.

    Monthly rather than one file per message because these are hook captures
    and single turns: a directory of thousands of two-line files is slower to
    grep and unreadable to list, and the month is the coarsest bucket that
    still lets a search narrow by date."""
    by_month: dict[str, list[str]] = {}
    count = 0
    for episode in episodes:
        if (episode.meta or {}).get("session_id"):
            continue
        stamp = episode.timestamp
        line = f"- ({stamp.date().isoformat()}) [{episode.role}] {_collapse(episode.content)}"
        by_month.setdefault(f"{stamp.year:04d}-{stamp.month:02d}", []).append(line)
        count += 1
    files = {
        f"messages/{month}.md": "\n".join(lines) + "\n" for month, lines in sorted(by_month.items())
    }
    return files, count


def _runbook_files(items: list[dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, str]]]:
    """One file per runbook, its ``content`` verbatim.

    Verbatim on purpose: the rendered markdown already carries the ``source:``
    footer the organizer writes for exactly this reader, so re-rendering here
    would give a grep two different texts for one memory."""
    files: dict[str, str] = {}
    index: list[dict[str, str]] = []
    for item in sorted(items, key=lambda d: str(d.get("id") or "")):
        item_id = str(item.get("id") or "")
        content = str(item.get("content") or "")
        if not item_id or not content:
            continue
        relpath = f"runbooks/{safe_name(item_id)}.md"
        files[relpath] = content if content.endswith("\n") else content + "\n"
        index.append({"id": item_id, "title": content.splitlines()[0], "path": relpath})
    return files, index


def _index_file(sessions: list[dict[str, Any]], runbooks: list[dict[str, str]]) -> str:
    """The map an explorer reads first: every session on one line, grouped by
    host, then the runbooks. Its job is to make ``list`` unnecessary for
    orientation and to let a search land on a session id."""
    lines = ["# workspace index", ""]
    lines.append("## sessions")
    if not sessions:
        lines += ["", "(none)"]
    for host in sorted({s["host"] for s in sessions}):
        lines += ["", f"### {host}"]
        for entry in sorted(
            (s for s in sessions if s["host"] == host), key=lambda s: (s["date"], s["id"])
        ):
            cwd = f" cwd={entry['cwd']}" if entry["cwd"] else ""
            lines.append(
                f"- {entry['id']}  {entry['date']}  steps={entry['steps']}{cwd}"
                f"  {entry['path']}  {entry['first_user']}"
            )
    lines += ["", "## runbooks"]
    if not runbooks:
        lines += ["", "(none)"]
    else:
        lines.append("")
        for entry in runbooks:
            lines.append(f"- {entry['id']}  {entry['path']}  {entry['title']}")
    return "\n".join(lines) + "\n"


def _sync(root: Path, files: dict[str, str]) -> WorkspaceStats:
    """Write ``files`` under ``root`` and delete managed files that are not in it.

    The delete half is what keeps the projection honest, and it is scoped to
    ``MANAGED_DIRS`` plus ``INDEX.md`` so that a workspace holding a user's own
    files loses none of them."""
    stats = WorkspaceStats()
    for relpath, text in sorted(files.items()):
        path = root / relpath
        data = text.encode()
        if path.exists() and path.read_bytes() == data:
            stats.unchanged += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        stats.written += 1
    for managed in MANAGED_DIRS:
        base = root / managed
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.relative_to(root).as_posix() not in files:
                path.unlink()
                stats.removed += 1
    return stats


def export_workspace(mem: Any, root: Path | str) -> WorkspaceStats:
    """Materialize ``mem``'s namespace under ``root`` and return what changed.

    ``mem`` is any object with ``doc_store`` and ``namespace`` — the facade in
    practice, but the narrow requirement keeps this usable from a harness that
    holds a store and no memory.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    episodes = mem.doc_store.list_episodes(mem.namespace)
    session_files, session_index = _session_files(episodes)
    message_files, message_count = _message_files(episodes)
    runbook_items = mem.doc_store.list_items(RUNBOOK_TYPE, namespace=mem.namespace)
    runbook_files, runbook_index = _runbook_files(runbook_items)

    files = {**session_files, **message_files, **runbook_files}
    files[INDEX_NAME] = _index_file(session_index, runbook_index)
    stats = _sync(root, files)
    stats.sessions = len(session_files)
    stats.runbooks = len(runbook_files)
    stats.messages = message_count
    logger.info(
        "explore: exported %s (sessions=%d runbooks=%d messages=%d written=%d removed=%d)",
        root,
        stats.sessions,
        stats.runbooks,
        stats.messages,
        stats.written,
        stats.removed,
    )
    return stats


__all__ = ["INDEX_NAME", "MANAGED_DIRS", "WorkspaceStats", "export_workspace", "safe_name"]
