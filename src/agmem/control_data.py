from __future__ import annotations

from dataclasses import asdict
from typing import Any

from agmem.core.types import utcnow
from agmem.stores.base import DocStore

from .control_types import (
    AUTO_DISABLED,
    AUTO_ENABLED,
    AUTO_INJECTION,
    CONTROL_KEY,
)


def now_iso() -> str:
    return utcnow().isoformat()


def episode_dict(item: Any) -> dict[str, Any]:
    data = asdict(item)
    timestamp = getattr(item, "timestamp", None)
    data["timestamp"] = timestamp.isoformat() if timestamp is not None else None
    return data


def data_of(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    data = getattr(item, "data", None)
    if isinstance(data, dict):
        return data
    meta = getattr(item, "meta", None)
    if isinstance(meta, dict):
        return meta
    return {}


def control_owner(item: Any) -> dict[str, Any]:
    data = data_of(item)
    if CONTROL_KEY in data:
        return data
    meta = data.get("meta")
    if isinstance(meta, dict):
        return meta
    return data


def control_of(data: dict[str, Any]) -> dict[str, Any]:
    owner = control_owner(data)
    raw = owner.get(CONTROL_KEY)
    if isinstance(raw, dict):
        return raw
    control: dict[str, Any] = {}
    owner[CONTROL_KEY] = control
    return control


def append_audit(data: dict[str, Any], action: str, reason: str | None) -> None:
    control = control_of(data)
    raw = control.get("audit")
    audit = raw if isinstance(raw, list) else []
    audit.append({"at": now_iso(), "action": action, "reason": reason or ""})
    control["audit"] = audit


def load_memory(store: DocStore, memory_type: str, memory_id: str) -> dict[str, Any] | None:
    if memory_type == "episodic":
        episodes = store.get_episodes([memory_id])
        return episode_dict(episodes[0]) if episodes else None
    items = store.get_items([memory_id], memory_type)
    return items[0] if items else None


def eligibility_reasons(item: Any) -> list[str]:
    data = data_of(item)
    owner = control_owner(item)
    reasons: list[str] = []
    if data.get("deleted"):
        reasons.append("deleted")
    if data.get("invalid_at"):
        reasons.append("invalidated")
    control = owner.get(CONTROL_KEY)
    if isinstance(control, dict) and control.get(AUTO_INJECTION) == AUTO_DISABLED:
        reasons.append("disabled_by_user")
    harmful = int(data.get("harmful") or 0)
    if isinstance(control, dict) and control.get(AUTO_INJECTION) == AUTO_ENABLED:
        restored_harmful = int(control.get("restored_harmful") or 0)
        if harmful <= restored_harmful:
            return reasons
    if harmful > 0:
        reasons.append("harmful_feedback")
    return reasons


def memory_display_text(item: Any) -> str:
    data = data_of(item)
    owner = control_owner(item)
    correction = data.get("correction") or owner.get("correction")
    if isinstance(correction, dict) and isinstance(correction.get("content"), str):
        return correction["content"]
    content = getattr(item, "content", None)
    if isinstance(content, str):
        return content
    return str(data.get("content") or data.get("summary") or data.get("name") or "")
