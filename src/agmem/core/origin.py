"""Origin binding and project gating (docs/research/agent-memory-axes-v1.md §6 #8, #9).

Small on purpose: the SessionStart hook imports this to filter a recency
listing and must stay a fraction of a second, so nothing heavy may live here.
"""

from __future__ import annotations

from typing import Any


def item_cwd(item: Any) -> str | None:
    """The project an item was written from: episodes carry it in ``meta``,
    derived items in their data (``cwd``, or ``origin.cwd``). None when the
    item predates origin binding or came from a bench-built trajectory."""
    meta = getattr(item, "meta", None)
    if isinstance(meta, dict) and meta.get("cwd"):
        return str(meta["cwd"])
    data = getattr(item, "data", None)
    if isinstance(data, dict):
        if data.get("cwd"):
            return str(data["cwd"])
        origin = data.get("origin")
        if isinstance(origin, dict) and origin.get("cwd"):
            return str(origin["cwd"])
    return None


def same_project(cwd: str | None, project: str) -> bool:
    """Same tree: equal paths, or one inside the other. Unknown passes — the
    gate refuses what it knows is foreign, not what it cannot place."""
    if not cwd:
        return True
    a = cwd.rstrip("/") or "/"
    b = project.rstrip("/") or "/"
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")
