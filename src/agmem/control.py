from __future__ import annotations

from typing import Any

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import MEMORY_TYPES
from agmem.stores.base import DocStore

from .control_data import (
    append_audit,
    control_of,
    control_owner,
    eligibility_reasons,
    load_memory,
    memory_display_text,
    now_iso,
)
from .control_replay import apply_user_control_op
from .control_status import status
from .control_types import (
    AUTO_DISABLED,
    AUTO_ENABLED,
    AUTO_INJECTION,
    CONTROL_KEY,
    ControlResult,
    MemoryControlKeyError,
    MemoryInspection,
    UnsupportedMemoryTypeError,
)

__all__ = [
    "AUTO_DISABLED",
    "AUTO_ENABLED",
    "AUTO_INJECTION",
    "CONTROL_KEY",
    "ControlResult",
    "MemoryInspection",
    "apply_user_control_op",
    "correct_memory",
    "disable_memory",
    "eligibility_reasons",
    "inspect_memory",
    "is_auto_injection_eligible",
    "memory_display_text",
    "restore_memory",
    "status",
]


def is_auto_injection_eligible(item: Any) -> bool:
    return not eligibility_reasons(item)


def inspect_memory(
    store: DocStore, *, namespace: str, memory_type: str, memory_id: str
) -> MemoryInspection:
    item = load_memory(store, memory_type, memory_id)
    if item is None or str(item.get("namespace", namespace)) != namespace:
        return MemoryInspection(False, namespace, memory_type, memory_id, {}, False, ["missing"])
    reasons = eligibility_reasons(item)
    return MemoryInspection(True, namespace, memory_type, memory_id, item, not reasons, reasons)


def disable_memory(
    store: DocStore,
    *,
    namespace: str,
    memory_type: str,
    memory_id: str,
    reason: str | None = None,
) -> ControlResult:
    item = _require_item(store, namespace, memory_type, memory_id)
    control = control_of(item)
    control[AUTO_INJECTION] = AUTO_DISABLED
    control["disabled_at"] = now_iso()
    control["disabled_reason"] = reason or ""
    append_audit(item, "disable", reason)
    _save(
        store,
        namespace,
        memory_type,
        memory_id,
        item,
        {"control": control_owner(item)[CONTROL_KEY]},
    )
    return ControlResult(True, namespace, memory_type, memory_id, item)


def restore_memory(
    store: DocStore,
    *,
    namespace: str,
    memory_type: str,
    memory_id: str,
    reason: str | None = None,
) -> ControlResult:
    item = _require_item(store, namespace, memory_type, memory_id)
    control = control_of(item)
    control[AUTO_INJECTION] = AUTO_ENABLED
    control["restored_at"] = now_iso()
    control["restore_reason"] = reason or ""
    control["restored_harmful"] = int(item.get("harmful") or 0)
    append_audit(item, "restore", reason)
    _save(
        store,
        namespace,
        memory_type,
        memory_id,
        item,
        {"control": control_owner(item)[CONTROL_KEY]},
    )
    return ControlResult(True, namespace, memory_type, memory_id, item)


def correct_memory(
    store: DocStore,
    *,
    namespace: str,
    memory_type: str,
    memory_id: str,
    content: str,
    reason: str | None = None,
) -> ControlResult:
    item = _require_item(store, namespace, memory_type, memory_id)
    current = str(item.get("content") or item.get("summary") or item.get("name") or "")
    item.setdefault("original_content", current)
    item["content"] = content
    if "summary" in item:
        item["summary"] = ""
    if "keywords" in item:
        item["keywords"] = []
    if "lexical_text" in item:
        item["lexical_text"] = content
    if item.get("embedding_text") is not None:
        item["embedding_text"] = content
    control = control_of(item)
    control["index_refresh"] = "pending"
    item["correction"] = {"at": now_iso(), "reason": reason or "", "content": content}
    append_audit(item, "correct", reason)
    delta: dict[str, Any] = {
        "content": item["content"],
        "control": control_owner(item)[CONTROL_KEY],
        "correction": item["correction"],
        "original_content": item["original_content"],
    }
    for key in ("summary", "keywords", "lexical_text", "embedding_text"):
        if key in item:
            delta[key] = item[key]
    _save(store, namespace, memory_type, memory_id, item, delta)
    return ControlResult(True, namespace, memory_type, memory_id, item)


def _require_item(
    store: DocStore, namespace: str, memory_type: str, memory_id: str
) -> dict[str, Any]:
    if memory_type not in MEMORY_TYPES:
        raise UnsupportedMemoryTypeError(memory_type)
    item = load_memory(store, memory_type, memory_id)
    if item is None or str(item.get("namespace", namespace)) != namespace:
        raise MemoryControlKeyError(memory_type, memory_id, namespace)
    return item


def _save(
    store: DocStore,
    namespace: str,
    memory_type: str,
    memory_id: str,
    item: dict[str, Any],
    delta: dict[str, Any],
) -> None:
    op = MemoryOp(
        op=OpType.UPDATE,
        target_type=memory_type,
        target_id=memory_id,
        actor="user-control",
        payload=delta,
    )
    store.append([op])
    if memory_type == "episodic":
        apply_user_control_op(store, op)
        return
    store.put_item(memory_id, memory_type, namespace, item)
