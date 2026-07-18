"""agmem MCP server (docs/05 §2).

Tool split follows the Graphiti pattern (add vs search vs admin) plus our
`report_feedback` loop (ACE counters / G-Memory reward shaping). Admin
tools are opt-in via --enable-admin-tools.

Run:
    uv run agmem-mcp --namespace main --profile lite            # stdio
    uv run agmem-mcp --transport http --port 8765               # streamable HTTP
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from agmem.config import AgmemConfig, load_config
from agmem.memory import AgenticMemory

logger = logging.getLogger("agmem.mcp")

_mem: AgenticMemory | None = None


def get_mem() -> AgenticMemory:
    assert _mem is not None, "server not initialized"
    return _mem


mcp = FastMCP("agmem")


@mcp.tool()
def add_memory(content: str, role: str = "user", timestamp: str | None = None) -> str:
    """Store a conversational message into memory. Organization (notes,
    entities, semantic facts) happens asynchronously; the raw content is
    searchable immediately."""
    from datetime import datetime

    ts = datetime.fromisoformat(timestamp) if timestamp else None
    episode = get_mem().add_message(content, role=role, timestamp=ts)
    return json.dumps({"stored": True, "episode_id": episode.id})


@mcp.tool()
def add_task_result(
    task: str, outcome: str, trajectory_json: str = "[]", agent_id: str = "agent"
) -> str:
    """Record a completed task trajectory (outcome: success|failure|unknown)
    so strategy memories can be distilled from it (ReasoningBank/ACE/G-Memory)."""
    trajectory = json.loads(trajectory_json)
    get_mem().add_task_result(trajectory=trajectory, outcome=outcome, task=task, agent_id=agent_id)
    return json.dumps({"recorded": True})


@mcp.tool()
def search_memory(
    query: str, memory_types: str = "episodic", k: int = 10, budget_tokens: int = 1600
) -> str:
    """Search memory. memory_types: comma-separated subset of episodic,
    episodes, notes, pages, semantic, entities, facts, strategies, playbook.
    Returns rendered context plus item provenance."""
    types = tuple(t.strip() for t in memory_types.split(",") if t.strip())
    bundle = get_mem().search(query, memory_types=types, k=k)
    return json.dumps(
        {
            "context": bundle.render(budget_tokens=budget_tokens),
            "items": [
                {
                    "memory_type": s.memory_type,
                    "score": round(s.score, 4),
                    "provenance": s.provenance,
                }
                for s in bundle.items
            ],
        },
        ensure_ascii=False,
    )


@mcp.tool()
def get_playbook(section: str | None = None) -> str:
    """Render the ACE playbook (strategy bullets with helpful/harmful counters)."""
    return get_mem().get_playbook(section=section) or "(playbook empty)"


@mcp.tool()
def report_feedback(memory_ids: str, helpful: bool) -> str:
    """Close the loop after using memories: comma-separated ids, helpful or not.
    Adjusts ACE bullet counters and strategy reward scores."""
    ids = [i.strip() for i in memory_ids.split(",") if i.strip()]
    updated = get_mem().report_feedback(ids, helpful=helpful)
    return json.dumps({"updated": updated})


@mcp.tool()
def memory_stats() -> str:
    """Memory counts, LLM cost accounting, active adapters, degradations."""
    m = get_mem()
    return json.dumps(
        {"stats": m.stats(), "capabilities": m.capabilities()},
        ensure_ascii=False,
        default=str,
    )


def register_admin_tools() -> None:
    @mcp.tool()
    def admin_snapshot_log(n: int = 50) -> str:
        """(admin) Tail the append-only evolution log."""
        return json.dumps(
            [json.loads(op.to_json()) for op in get_mem().log.tail(n)],
            ensure_ascii=False,
        )

    @mcp.tool()
    def admin_flush() -> str:
        """(admin) Block until queued organizer work is applied."""
        get_mem().flush()
        return json.dumps({"flushed": True})


def main() -> None:
    ap = argparse.ArgumentParser(description="agmem MCP server")
    ap.add_argument("--namespace", default="main")
    ap.add_argument("--profile", default="lite")
    ap.add_argument(
        "--organizers",
        default="nemori,reasoning_bank",
        help="comma-separated organizer names",
    )
    ap.add_argument("--config", default=None, help="path to agmem.toml")
    ap.add_argument("--data-dir", default=str(Path.home() / ".agmem/data"))
    ap.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--enable-admin-tools", action="store_true")
    args = ap.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = AgmemConfig(profile=args.profile, sync_write=False)
    if config.data_dir is None:
        config.data_dir = Path(args.data_dir)

    global _mem
    _mem = AgenticMemory(
        namespace=args.namespace,
        organizers=[o.strip() for o in args.organizers.split(",") if o.strip()],
        config=config,
    )
    logger.info(
        "agmem MCP: namespace=%s organizers=%s profile=%s",
        args.namespace,
        args.organizers,
        config.profile,
    )

    if args.enable_admin_tools:
        register_admin_tools()

    if args.transport == "http":
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
