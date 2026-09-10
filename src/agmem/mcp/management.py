from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from typing import Literal, assert_never

from mcp.server.fastmcp import FastMCP

from agmem.control import (
    correct_memory,
    disable_memory,
    restore_memory,
)
from agmem.control import (
    inspect_memory as inspect_item,
)
from agmem.memory import AgenticMemory

ControlAction = Literal["disable", "restore", "correct"]


def register_management_tools(mcp: FastMCP, get_mem: Callable[[str | None], AgenticMemory]) -> None:
    @mcp.tool()
    def inspect_memory(memory_type: str, memory_id: str, namespace: str | None = None) -> str:
        """Inspect stored content, provenance and automatic-injection exclusion reasons."""
        mem = get_mem(namespace)
        result = inspect_item(
            mem.doc_store, namespace=mem.namespace, memory_type=memory_type, memory_id=memory_id
        )
        return json.dumps(asdict(result), ensure_ascii=False, default=str)

    @mcp.tool()
    def update_memory(
        memory_type: str,
        memory_id: str,
        action: ControlAction,
        content: str | None = None,
        reason: str | None = None,
        namespace: str | None = None,
    ) -> str:
        """Explicit user correction or reversible automatic-injection disable/restore.

        Correct requires content. Original content and an audit trail are retained.
        """
        mem = get_mem(namespace)
        try:
            match action:
                case "disable":
                    result = disable_memory(
                        mem.doc_store,
                        namespace=mem.namespace,
                        memory_type=memory_type,
                        memory_id=memory_id,
                        reason=reason,
                    )
                case "restore":
                    result = restore_memory(
                        mem.doc_store,
                        namespace=mem.namespace,
                        memory_type=memory_type,
                        memory_id=memory_id,
                        reason=reason,
                    )
                case "correct":
                    if content is None or not content.strip():
                        return json.dumps({"error": "correct requires non-empty content"})
                    result = correct_memory(
                        mem.doc_store,
                        namespace=mem.namespace,
                        memory_type=memory_type,
                        memory_id=memory_id,
                        content=content,
                        reason=reason,
                    )
                case unreachable:
                    assert_never(unreachable)
        except (KeyError, ValueError) as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps(asdict(result), ensure_ascii=False, default=str)
