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

THE HOOKS' DAEMON. Over `--transport http` this process is also what the Claude
Code / Codex hooks talk to (`agmem.hooks.daemon`), through three plain HTTP
routes that need no MCP client: `GET /health`, `POST /hooks/capture`,
`POST /hooks/recall`. That is issue #2 §1: a hook that loaded the embedder
itself cost ~11 s per prompt; against this process the same write is ~50 ms.
Two things follow. `--idle-timeout` lets a hook-spawned daemon go away when
nobody has used it for a while. And `backfill` gives vectors to episodes a
hook wrote while no daemon was running — the capture hook writes to the doc
store alone in that case rather than loading a model, so "episode exists but
has no vector" is a state this server has to expect and repair.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from dataclasses import replace

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

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

    def pending_embed(self, namespace: str | None = None) -> int:
        """Episodes in `namespace`'s doc store that have no vector yet.

        Derived by comparing the two stores rather than kept as a flag, so an
        interrupted backfill or a hook that wrote while the daemon was down
        both leave the same, recomputable answer."""
        return len(self._missing_vector_ids(self.get(namespace)))

    @staticmethod
    def _missing_vector_ids(mem: AgenticMemory) -> list[str]:
        ids = [ep.id for ep in mem.doc_store.list_episodes(namespace=mem.namespace)]
        if not ids:
            return []
        have = mem.vector_store.get(ids)
        return [i for i in ids if i not in have]

    def backfill(self, namespace: str | None = None, batch_size: int = 64) -> int:
        """Embed and index every episode in `namespace` that has no vector; returns how many.

        The hook-side absent path (`agmem.hooks.capture`) writes episodes to
        the doc store only. Without this they would be visible to the recency
        hook and invisible to every semantic search forever — the exact
        outcome issue #2 warned the fast path must not produce."""
        mem = self.get(namespace)
        missing = self._missing_vector_ids(mem)
        if not missing:
            return 0
        done = 0
        for start in range(0, len(missing), batch_size):
            chunk = mem.doc_store.get_episodes(missing[start : start + batch_size])
            vectors = mem.embedder.embed([ep.embedding_text() for ep in chunk])
            for ep, vec in zip(chunk, vectors, strict=True):
                mem.vector_store.add(ep.id, vec, memory_type="episodic", namespace=mem.namespace)
                done += 1
        persist = getattr(mem.vector_store, "persist", None)
        if persist is not None:
            persist()
        logger.info(
            "agmem daemon: backfilled %d episode vector(s) in namespace=%s", done, mem.namespace
        )
        return done

    def backfill_all(self) -> int:
        return sum(self.backfill(ns) for ns in self.open_namespaces())

    def close_all(self) -> None:
        with self._lock:
            mems, self._mems = list(self._mems.values()), {}
        for mem in mems:
            mem.close()


_registry = _Registry()


_started_at = time.monotonic()
_last_activity = time.monotonic()


def _touch() -> None:
    global _last_activity
    _last_activity = time.monotonic()


def get_mem(namespace: str | None = None) -> AgenticMemory:
    """The `AgenticMemory` for `namespace`, or for the server's default when
    None. Opened on first use; see `_Registry`.

    Also the idle clock: every tool and hook route passes through here, so
    "last activity" is simply the last time a memory was asked for.

    Raises `AssertionError` if called before `main()` has run — tool handlers only
    execute after the server is up, so this should never fire in practice."""
    _touch()
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
def research_memory(
    query: str,
    max_steps: int = 8,
    budget_tokens: int = 4000,
    namespace: str | None = None,
) -> str:
    """Explore memory the way a coding agent would: the raw session transcripts
    and runbooks are written out as files and a model searches and reads them,
    returning context with file:line citations and the wall-clock latency.
    Slower than search_memory by design (several model calls); use it when the
    question is about what happened in past sessions and exact strings matter.
    Requires an [llm.explore] role in the server's config; without one this
    returns an error object rather than a vector search."""
    mem = get_mem(namespace)
    try:
        result = mem.research(query, max_steps=max_steps, budget_tokens=budget_tokens)
    except (RuntimeError, ValueError) as exc:
        return json.dumps({"error": str(exc), "namespace": mem.namespace})
    return json.dumps(
        {
            "namespace": mem.namespace,
            "context": result.context,
            "citations": result.citations,
            "latency_s": result.latency_s,
            "llm_calls": result.llm_calls,
            "steps": len(result.steps),
            "search_tool": result.search_tool,
            "degraded": result.degraded,
        }
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


def _item_text(item) -> str:
    text = getattr(item, "content", None) or getattr(item, "text", None)
    return text if isinstance(text, str) else str(item)


def _item_timestamp(item) -> str | None:
    stamp = getattr(item, "timestamp", None)
    return stamp.isoformat() if hasattr(stamp, "isoformat") else None


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    """What a hook checks before deciding which path to take (`agmem.hooks.daemon.health`).

    `pending_embed` is reported per open namespace so a reader can tell that
    memories written while the daemon was down are not searchable yet."""
    now = time.monotonic()
    open_ns = _registry.open_namespaces()
    return JSONResponse(
        {
            "ok": True,
            "pid": os.getpid(),
            "default_namespace": _registry.default,
            "open_namespaces": open_ns,
            "pending_embed": {ns: _registry.pending_embed(ns) for ns in open_ns},
            "uptime_s": round(now - _started_at, 1),
            "idle_s": round(now - _last_activity, 1),
        }
    )


@mcp.custom_route("/hooks/capture", methods=["POST"])
async def hooks_capture(request: Request) -> JSONResponse:
    """The capture hook's fast path: the same write `add_memory` does, over plain JSON.

    Body: `{content, role?, meta?, namespace?}`. Runs on a worker thread because
    the write embeds synchronously and must not block the event loop for the
    other hook that is waiting on `/hooks/recall`."""
    import anyio

    body = await request.json()
    content = body.get("content")
    if not isinstance(content, str) or not content.strip():
        return JSONResponse({"error": "content required"}, status_code=400)
    role = body.get("role") or "user"
    meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
    namespace = body.get("namespace") or None

    def _write():
        mem = get_mem(namespace)
        episode = mem.add_message(content, role=role, meta=meta)
        return {"stored": True, "episode_id": episode.id, "namespace": mem.namespace}

    try:
        return JSONResponse(await anyio.to_thread.run_sync(_write))
    except Exception as exc:
        logger.exception("hooks/capture failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


@mcp.custom_route("/hooks/recall", methods=["POST"])
async def hooks_recall(request: Request) -> JSONResponse:
    """The recall-on-prompt hook's search: `{query, k?, namespace?}` -> ranked items with text.

    Returns the items rather than a rendered block so the hook, which owns the
    words the model sees, decides the header and the truncation."""
    import anyio

    body = await request.json()
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return JSONResponse({"error": "query required"}, status_code=400)
    k = int(body.get("k") or 5)
    namespace = body.get("namespace") or None

    def _search():
        mem = get_mem(namespace)
        bundle = get_searcher(namespace).search(query, k=k)
        return {
            "namespace": mem.namespace,
            "items": [
                {
                    "id": getattr(s.item, "id", None),
                    "memory_type": s.memory_type,
                    "score": round(float(s.score), 4),
                    "timestamp": _item_timestamp(s.item),
                    "text": _item_text(s.item),
                }
                for s in bundle.items
            ],
        }

    try:
        return JSONResponse(await anyio.to_thread.run_sync(_search))
    except Exception as exc:
        logger.exception("hooks/recall failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


def _idle_watchdog(idle_timeout_s: int, poll_s: float | None = None) -> None:
    """Exit the process once nothing has asked for a memory in `idle_timeout_s`.

    A hook-spawned daemon has no owner to stop it; this is how it goes away.
    Stores are closed first so a queued organizer write is not lost. `os._exit`
    rather than `sys.exit` because uvicorn's serving thread would otherwise
    keep the process alive."""
    poll = poll_s if poll_s is not None else min(5.0, max(0.5, idle_timeout_s / 4))
    while True:
        time.sleep(poll)
        if time.monotonic() - _last_activity >= idle_timeout_s:
            logger.info("agmem daemon: idle for %ss, exiting", idle_timeout_s)
            try:
                _registry.close_all()
            finally:
                os._exit(0)


def _backfill_loop(period_s: float) -> None:
    """Startup and periodic repair of episodes without vectors (see `_Registry.backfill`)."""
    while True:
        try:
            _registry.backfill_all()
        except Exception:
            logger.exception("agmem daemon: backfill failed")
        time.sleep(period_s)


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
    ap.add_argument("--host", default="127.0.0.1", help="http only; keep it loopback (no auth)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--idle-timeout",
        type=int,
        default=0,
        help="http only: exit after this many seconds without a request (0 = never). "
        "The hooks spawn the daemon with this set so it does not outlive its use.",
    )
    ap.add_argument(
        "--backfill-period",
        type=int,
        default=60,
        help="http only: seconds between passes that embed episodes written while no daemon ran",
    )
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
    # Whichever transport this is, repair first: a hook may have written
    # episodes while no daemon ran, and a stdio server started by Claude Code
    # is often the next process to open that store. Without this pass those
    # episodes would answer to the recency hook and to nothing else.
    pending = _registry.backfill_all()
    if pending:
        logger.info("agmem MCP: startup backfill embedded %d episode(s)", pending)

    try:
        if args.transport == "http":
            mcp.settings.host = args.host
            mcp.settings.port = args.port
            threading.Thread(
                target=_backfill_loop, args=(max(args.backfill_period, 1),), daemon=True
            ).start()
            if args.idle_timeout > 0:
                threading.Thread(
                    target=_idle_watchdog, args=(args.idle_timeout,), daemon=True
                ).start()
            mcp.run(transport="streamable-http")
        else:
            mcp.run()
    finally:
        _registry.close_all()


if __name__ == "__main__":
    main()
