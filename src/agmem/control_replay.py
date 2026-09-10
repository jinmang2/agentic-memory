from __future__ import annotations

from dataclasses import replace

from agmem.core.ops import MemoryOp, OpType
from agmem.stores.base import DocStore

from .control_types import MemoryControlKeyError, UserControlOpError


def apply_user_control_op(store: DocStore, op: MemoryOp) -> None:
    if op.actor != "user-control" or op.op is not OpType.UPDATE:
        raise UserControlOpError(op.actor, op.op.value, op.target_type)
    if op.target_type == "episodic":
        episodes = store.get_episodes([op.target_id])
        if not episodes:
            raise MemoryControlKeyError("episodic", op.target_id)
        episode = episodes[0]
        meta = dict(episode.meta)
        content = episode.content
        for key, value in op.payload.items():
            if key == "content":
                content = str(value)
            else:
                meta[key] = value
        store.add_episode(replace(episode, content=content, meta=meta))
        return
    items = store.get_items([op.target_id], op.target_type)
    if not items:
        raise MemoryControlKeyError(op.target_type, op.target_id)
    data = dict(items[0])
    data.update(op.payload)
    store.put_item(op.target_id, op.target_type, data.get("namespace", "main"), data)
