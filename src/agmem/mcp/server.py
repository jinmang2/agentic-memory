"""agmem MCP server (docs/05 §2).

Tool split follows the Graphiti pattern (add vs search vs admin) plus our
`report_feedback` loop (ACE counters / G-Memory reward shaping). Admin
tools are opt-in via --enable-admin-tools.

Run:
    uv run agmem-mcp --profile lite                              # stdio, namespace from
                                                                 # AGMEM_NAMESPACE or "main"
    uv run agmem-mcp --transport http --port 8765               # streamable HTTP

WHICH STORE. `--namespace`, `--data-dir` and `--config` default to the
environment variables `AGMEM_NAMESPACE`, `AGMEM_DATA_DIR` and `AGMEM_CONFIG`,
which are the same three the Claude Code hooks read (`agmem.hooks`), resolved
by the same code (`agmem.env`). Before 2026-09-02 the server read no
environment at all and defaulted its namespace to `main` while the hooks
defaulted to `claude-code`, so the two layers opened different stores unless
somebody noticed and said the same thing twice (github issue #2).

ONE SERVER, MANY NAMESPACES. Every tool takes an optional `namespace`; leaving
it out means the one the server started with. Namespaces are opened lazily
and kept for the process lifetime, and they share the first one's embedder —
the model is the expensive part of opening a memory (~4 s of the ~10 s
handshake), the stores are not. Without this a daemon was one memory, and
keeping projects apart meant one daemon per project.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import replace

from mcp.server.fastmcp import FastMCP

from agmem.config import AgmemConfig, load_config
from agmem.env import (
    ENV_CONFIG,
    ENV_DATA_DIR,
    ENV_NAMESPACE,
    resolve_config_path,
    resolve_data_dir,
    resolve_namespace,
    validate_namespace,
)
from agmem.memory import AgenticMemory
from agmem.retrieval.planned import searcher_for

logger = logging.getLogger("agmem.mcp")


class _Registry:
    """The memories this process has opened, one per namespace.

    `default` is the namespace `main()` started with and the one every tool
    falls back to. Opening is serialized under a lock because FastMCP may run
    tool bodies on worker threads and two concurrent first calls for the same
    namespace would otherwise build two `AgenticMemory` instances over one set
    of files."""

    def __init__(self) -> None:
        self.default: str | None = None
        self.config: AgmemConfig | None = None
        self.organizers: list[str] = []
        self._mems: dict[str, AgenticMemory] = {}
        self._lock = threading.Lock()

    def start(self, namespace: str, organizers: list[str], config: AgmemConfig) -> AgenticMemory:
        self.default, self.organizers, self.config = namespace, organizers, config
        return self.get(namespace)

    def get(self, namespace: str | None) -> AgenticMemory:
        assert self.default is not None and self.config is not None, "server not initialized"
        ns = validate_namespace(namespace) if namespace else self.default
        with self._lock:
            mem = self._mems.get(ns)
            if mem is None:
                shared = next(iter(self._mems.values()), None)
                mem = AgenticMemory(
                    namespace=ns,
                    organizers=list(self.organizers),
                    config=self.config,
                    embedder=shared.embedder if shared is not None else None,
                )
                self._mems[ns] = mem
                if shared is not None:
                    logger.info("agmem MCP: opened namespace=%s (embedder shared)", ns)
            return mem

    def open_namespaces(self) -> list[str]:
        with self._lock:
            return sorted(self._mems)

    def close_all(self) -> None:
        with self._lock:
            mems, self._mems = list(self._mems.values()), {}
        for mem in mems:
            mem.close()


_registry = _Registry()


def get_mem(namespace: str | None = None) -> AgenticMemory:
    """The `AgenticMemory` for `namespace`, or for the server's default when
    None. Opened on first use; see `_Registry`.

    Raises `AssertionError` if called before `main()` has run — tool handlers only
    execute after the server is up, so this should never fire in practice."""
    return _registry.get(namespace)


def get_searcher(namespace: str | None = None):
    """What ``search_memory`` calls: the memory, or the memory wrapped in the
    read-side control policy ``AgmemConfig.query_strategy`` names.

    Resolved per call rather than cached at startup so the wrapper never
    outlives a rebuilt memory; ``searcher_for`` is a couple of attribute reads
    when no policy is configured, which is the default."""
    return searcher_for(get_mem(namespace))


mcp = FastMCP("agmem")


@mcp.tool()
def add_memory(
    content: str,
    role: str = "user",
    timestamp: str | None = None,
    namespace: str | None = None,
) -> str:
    """Store a conversational message into memory. Organization (notes,
    entities, semantic facts) happens asynchronously; the raw content is
    searchable immediately. namespace: which memory to write to; omit for the
    server's default (memory_stats reports it)."""
    from datetime import datetime

    ts = datetime.fromisoformat(timestamp) if timestamp else None
    mem = get_mem(namespace)
    episode = mem.add_message(content, role=role, timestamp=ts)
    return json.dumps({"stored": True, "episode_id": episode.id, "namespace": mem.namespace})


@mcp.tool()
def add_task_result(
    task: str,
    outcome: str,
    trajectory_json: str = "[]",
    agent_id: str = "agent",
    namespace: str | None = None,
) -> str:
    """Record a completed task trajectory (outcome: success|failure|unknown)
    so strategy memories can be distilled from it (ReasoningBank/ACE/G-Memory).
    namespace: omit for the server's default."""
    trajectory = json.loads(trajectory_json)
    mem = get_mem(namespace)
    mem.add_task_result(trajectory=trajectory, outcome=outcome, task=task, agent_id=agent_id)
    return json.dumps({"recorded": True, "namespace": mem.namespace})


@mcp.tool()
def search_memory(
    query: str,
    memory_types: str = "",
    k: int = 10,
    budget_tokens: int = 1600,
    namespace: str | None = None,
) -> str:
    """Search memory. memory_types: comma-separated subset of episodic,
    episodes, notes, pages, semantic, entities, facts, strategies, playbook.
    Leave it empty (the default) to search raw episodes plus whatever the
    configured organizers actually produce — the old "episodic" default silently
    returned raw messages instead of, say, A-Mem notes.
    namespace: which memory to search; omit for the server's default.
    Returns rendered context plus item provenance."""
    types = tuple(t.strip() for t in memory_types.split(",") if t.strip()) or None
    metrics: dict = {}
    bundle = get_searcher(namespace).search(query, memory_types=types, k=k, metrics=metrics)
    return json.dumps(
        {
            "namespace": get_mem(namespace).namespace,
            "context": bundle.render(budget_tokens=budget_tokens),
            # How the bundle was obtained: one plain search by default, or the
            # planned searches a configured read policy ran. Reported because a
            # policy spends LLM calls per query and a caller should be able to
            # see that it did.
            "retrieval": metrics,
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
def get_playbook(section: str | None = None, namespace: str | None = None) -> str:
    """Render the ACE playbook (strategy bullets with helpful/harmful counters).
    namespace: omit for the server's default."""
    return get_mem(namespace).get_playbook(section=section) or "(playbook empty)"


@mcp.tool()
def report_feedback(memory_ids: str, helpful: bool, namespace: str | None = None) -> str:
    """Close the loop after using memories: comma-separated ids, helpful or not.
    Adjusts ACE bullet counters and strategy reward scores. namespace: the one
    the ids came from; omit for the server's default."""
    ids = [i.strip() for i in memory_ids.split(",") if i.strip()]
    updated = get_mem(namespace).report_feedback(ids, helpful=helpful)
    return json.dumps({"updated": updated})


@mcp.tool()
def memory_stats(namespace: str | None = None) -> str:
    """Memory counts, LLM cost accounting, active adapters, degradations, plus
    the server's default namespace and every namespace it has opened.
    namespace: omit for the server's default."""
    m = get_mem(namespace)
    return json.dumps(
        {
            "stats": m.stats(),
            "capabilities": m.capabilities(),
            "default_namespace": _registry.default,
            "open_namespaces": _registry.open_namespaces(),
        },
        ensure_ascii=False,
        default=str,
    )


def register_admin_tools() -> None:
    """Register the admin-only tools (log tail, flush) onto the shared `mcp` server.

    Call only when `--enable-admin-tools` is passed — these expose internal state
    (evolution log contents) that isn't meant for untrusted MCP clients."""

    @mcp.tool()
    def admin_snapshot_log(n: int = 50, namespace: str | None = None) -> str:
        """(admin) Tail the append-only evolution log."""
        return json.dumps(
            [json.loads(op.to_json()) for op in get_mem(namespace).log.tail(n)],
            ensure_ascii=False,
        )

    @mcp.tool()
    def admin_flush(namespace: str | None = None) -> str:
        """(admin) Block until queued organizer work is applied."""
        get_mem(namespace).flush()
        return json.dumps({"flushed": True})


def main() -> None:
    """CLI entrypoint: parse args, open the default namespace, run the server.

    Blocks for the process lifetime (stdio or streamable-HTTP transport per `--transport`);
    admin tools are registered before `mcp.run()` only if `--enable-admin-tools` is set."""
    ap = argparse.ArgumentParser(description="agmem MCP server")
    ap.add_argument(
        "--namespace",
        default=None,
        help=f"default namespace for every tool (default ${ENV_NAMESPACE}, else 'main')",
    )
    ap.add_argument(
        "--profile",
        default=None,
        help="lite|standard|full (default lite); overrides [profile].name in --config",
    )
    ap.add_argument(
        "--organizers",
        default="nemori,reasoning_bank",
        help="comma-separated organizer names",
    )
    ap.add_argument(
        "--config", default=None, help=f"path to agmem.toml (default ${ENV_CONFIG}, else none)"
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help=f"store root (default ${ENV_DATA_DIR}, else [storage].data_dir, else ~/.agmem/data)",
    )
    ap.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--enable-admin-tools", action="store_true")
    args = ap.parse_args()

    # docs/05 §2.2 precedence: CLI arg > environment > agmem.toml > default. The
    # environment tier is `agmem.env`, shared with the hooks, so one exported
    # AGMEM_NAMESPACE / AGMEM_DATA_DIR / AGMEM_CONFIG configures both layers.
    #
    # `--profile` used to be dropped entirely whenever `--config` was given, which
    # inverted the rule silently — the server ran the TOML's profile while
    # reporting the flag back in its log line. Resolved here rather than in the
    # facade: the precedence rule is a property of this CLI, and AgenticMemory now
    # rejects a profile/config disagreement outright.
    config_path = resolve_config_path(args.config)
    if config_path is not None:
        config = load_config(config_path)
        if args.profile is not None and args.profile != config.profile:
            logger.info(
                "profile: --profile=%s overrides [profile].name=%s from %s",
                args.profile,
                config.profile,
                config_path,
            )
            config = replace(config, profile=args.profile)
    else:
        config = AgmemConfig(profile=args.profile or "lite", sync_write=False)
    config.data_dir = resolve_data_dir(args.data_dir, from_config=config.data_dir)
    namespace = resolve_namespace(args.namespace)

    _registry.start(namespace, [o.strip() for o in args.organizers.split(",") if o.strip()], config)
    logger.info(
        "agmem MCP: namespace=%s data_dir=%s config=%s organizers=%s profile=%s",
        namespace,
        config.data_dir,
        config_path,
        args.organizers,
        config.profile,
    )

    if args.enable_admin_tools:
        register_admin_tools()

    # close() on the way out: it drains queued organizer work (this server runs
    # sync_write=False, so a shutdown mid-queue would otherwise discard whatever
    # the worker had not applied yet) and then stops the worker and the stores.
    try:
        if args.transport == "http":
            mcp.settings.port = args.port
            mcp.run(transport="streamable-http")
        else:
            mcp.run()
    finally:
        _registry.close_all()


if __name__ == "__main__":
    main()
