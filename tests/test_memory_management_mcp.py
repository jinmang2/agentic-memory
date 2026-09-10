from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent


def _payload(result: CallToolResult):
    assert not result.isError
    return json.loads("".join(c.text for c in result.content if isinstance(c, TextContent)))


async def _scenario(root: Path, config: Path) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "agmem.mcp.server",
            "--namespace",
            "manage-test",
            "--data-dir",
            str(root),
            "--config",
            str(config),
            "--organizers",
            "",
        ],
        env={k: v for k, v in os.environ.items() if not k.startswith("AGMEM_")},
    )
    with anyio.fail_after(30):
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = {tool.name for tool in (await session.list_tools()).tools}
            assert {"inspect_memory", "update_memory"} <= tools
            added = _payload(await session.call_tool("add_memory", {"content": "old constraint"}))
            reference = {"memory_type": "episodic", "memory_id": added["episode_id"]}
            disabled = _payload(
                await session.call_tool("update_memory", {**reference, "action": "disable"})
            )
            assert disabled["changed"] is True
            inspected = _payload(await session.call_tool("inspect_memory", reference))
            assert inspected["eligible"] is False
            await session.call_tool(
                "update_memory", {**reference, "action": "correct", "content": "new constraint"}
            )
            await session.call_tool("update_memory", {**reference, "action": "restore"})
            inspected = _payload(await session.call_tool("inspect_memory", reference))
            assert inspected["eligible"] is True
            assert inspected["item"]["content"] == "new constraint"


def test_mcp_memory_controls_work_over_stdio(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[profile]\nname = "lite"\n[override]\nembedder = "FakeEmbedder"\n')
    anyio.run(_scenario, tmp_path / "data", config)
