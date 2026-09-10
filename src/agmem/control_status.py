from __future__ import annotations

from typing import Any

from agmem.core.types import MEMORY_TYPES
from agmem.stores.base import DocStore

from .control_data import eligibility_reasons


def status(store: DocStore, *, namespace: str, store_path: str) -> dict[str, Any]:
    items: dict[str, dict[str, int]] = {}
    state_counts: dict[str, int] = {}
    distill_rows: list[dict[str, Any]] = []
    for memory_type in MEMORY_TYPES:
        if memory_type == "episodic":
            continue
        rows = store.list_items(memory_type, namespace=namespace)
        live_rows = [row for row in rows if not row.get("invalid_at") and not row.get("deleted")]
        corrected = sum(1 for row in rows if row.get("correction"))
        disabled = sum(1 for row in rows if "disabled_by_user" in eligibility_reasons(row))
        blocked = sum(1 for row in rows if "harmful_feedback" in eligibility_reasons(row))
        items[memory_type] = {
            "live": len(live_rows),
            "stored": len(rows),
            "disabled": disabled,
            "feedback_blocked": blocked,
            "corrected": corrected,
        }
        if memory_type == "state":
            for row in rows:
                if row.get("kind") == "session_distill":
                    key = str(row.get("status") or "unknown")
                    state_counts[key] = state_counts.get(key, 0) + 1
                    distill_rows.append(
                        {
                            "id": str(row.get("id") or ""),
                            "status": key,
                            "session_id": str(row.get("session_id") or ""),
                            "source_host": str(row.get("source_host") or ""),
                            "reason": str(row.get("reason") or ""),
                            "updated_at": str(row.get("updated_at") or ""),
                            "first_episode_id": str(row.get("first_episode_id") or ""),
                            "last_episode_id": str(row.get("last_episode_id") or ""),
                        }
                    )
    episodes = store.list_episodes(namespace)
    return {
        "namespace": namespace,
        "store": store_path,
        "episodes": store.count_episodes(namespace),
        "episode_controls": {
            "disabled": sum(
                1 for episode in episodes if "disabled_by_user" in eligibility_reasons(episode)
            ),
            "corrected": sum(1 for episode in episodes if episode.meta.get("correction")),
        },
        "items": items,
        "session_distill": state_counts,
        "session_distill_recent": distill_rows[-20:],
    }
