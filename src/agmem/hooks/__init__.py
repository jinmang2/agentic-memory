"""Claude Code hooks: the layer that makes this memory automatic rather than asked-for.

The MCP server exposes memory as TOOLS, which means the model has to decide to
call them. A hook fires whether or not anyone decided anything, which is the
whole point — capture that depends on the model remembering to capture is the
thing it was supposed to replace.

Two entry points, one per direction:

    python -m agmem.hooks.recall     SessionStart    memory -> context
    python -m agmem.hooks.capture    UserPromptSubmit  turn -> memory

Both speak the Claude Code hook contract: a JSON object on stdin, a JSON object
on stdout whose ``hookSpecificOutput.additionalContext`` (recall) reaches the
model, and an exit code the harness reads.

THE INVARIANT BOTH OBEY: a hook must never break the session it is attached to.
A memory system that makes Claude Code fail to start is worse than no memory
system, so every failure path here exits 0 with no output. `fail_open` is the
only error handling in this package that is deliberately silent, and it is
silent because the alternative is a broken editor.

COST: capture is free by default. It writes an episode and returns; organizers
that need an LLM see no configured endpoint and skip explicitly (verified
against the MCP server, whose no-LLM path this shares). The hooks never run
organizers at all — a hook runs on somebody's keystroke, so distillation is
the MCP server's business, where a model decided to write.

WHICH STORE: namespace, data directory and config file come from the same three
environment variables the MCP server reads (`AGMEM_NAMESPACE`, `AGMEM_DATA_DIR`,
`AGMEM_CONFIG`), with the same defaults, resolved by `agmem.env`. Setting them
once therefore configures both layers onto one store. The hooks used to default
to a namespace of their own (`claude-code`, against the server's `main`), so
wiring both as documented produced two stores that could not see each other
(github issue #2). There is no per-hook namespace flag on purpose: the harness
gives hooks no arguments, and a second way to say it is a second way to
disagree.

`AGMEM_CONFIG` exists so a deployment can pick the embedder and stores for the
hooks the way it can for the server (`--config`), and so the hook tests can run
hermetically on `FakeEmbedder` — before 2026-09-02 they could not, and each run
downloaded the 471 MB default model into the test's throwaway HOME.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from agmem.env import (
    DEFAULT_DATA_DIR,
    DEFAULT_NAMESPACE,
    resolve_config_path,
    resolve_data_dir,
    resolve_namespace,
)

__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_NAMESPACE",
    "emit_context",
    "fail_open",
    "open_doc_store",
    "open_memory",
    "read_event",
]


def read_event() -> dict[str, Any]:
    """The hook payload on stdin, or `{}` when there is none.

    Never raises: a hook invoked by hand, or by a harness version that sends
    something unexpected, still has to reach `fail_open` rather than dying with
    a traceback the user sees at session start.
    """
    try:
        raw = sys.stdin.read()
    except Exception:
        return {}
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def emit_context(text: str, event_name: str) -> None:
    """Hand `text` to the model as additional context for `event_name`.

    `hookEventName` is required alongside `additionalContext` — omitting it
    makes the harness drop the payload, which fails as silence rather than as
    an error, so it is set from the caller's own event rather than defaulted.
    """
    if not text.strip():
        return
    json.dump(
        {"hookSpecificOutput": {"hookEventName": event_name, "additionalContext": text}},
        sys.stdout,
    )
    sys.stdout.write("\n")


def fail_open(exc: BaseException) -> None:
    """Exit 0, silently, after routing the reason somewhere a human can find it.

    The session must survive anything this package does to it. Diagnostics go to
    AGMEM_HOOK_LOG when set — stderr would surface in the transcript and train
    the user to ignore hook output, which is exactly when a real failure hides.
    """
    log = os.environ.get("AGMEM_HOOK_LOG")
    if log:
        try:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"{type(exc).__name__}: {exc}\n")
        except OSError:
            pass
    raise SystemExit(0)


def _load_config(config_path: str | Path | None):
    """The `AgmemConfig` named by the argument or `AGMEM_CONFIG`, or None.

    Imported lazily: `agmem.config` pulls in the LLM client, which the recall
    hook must not pay for when no config is named (the common case).
    """
    path = resolve_config_path(config_path)
    if path is None:
        return None
    from agmem.config import load_config

    return load_config(path)


def _resolve(
    namespace: str | None, data_dir: str | None, config_path: str | Path | None = None
) -> tuple[str, Path, Any]:
    config = _load_config(config_path)
    ns = resolve_namespace(namespace)
    root = resolve_data_dir(data_dir, from_config=config.data_dir if config else None)
    return ns, root, config


def open_doc_store(namespace: str | None = None, data_dir: str | None = None):
    """The doc store alone — no embedder, no vector store, no organizers.

    This exists because of a measurement, not a preference. Opening a full
    `AgenticMemory` costs 9.1 s on this machine, essentially all of it
    `SentenceTransformerEmbedder` reaching its weights; the doc store alone is
    0.18 s. A hook that only READS episodes by recency never touches a vector,
    so paying for the embedder makes it 50x slower than its job requires — and
    at that speed the recall hook exceeds any sane `timeout` and is killed,
    which presents as no memory rather than as an error.

    Those figures are from 2026-08-08, after `SentenceTransformerEmbedder`
    began loading cache-first; the same measurement read 15.1 s and 0.21 s
    (70x) before that. The ratio moved and the conclusion did not, which is the
    point — 0.18 s against 9.1 s is not a margin any tuning closes.

    Assumes the `SqliteDocStore` layout (`<data_dir>/<namespace>/memory.db`),
    which every profile except `full` uses and which `open_memory` produces
    unless a config file overrides the doc store slot. That case is refused
    rather than guessed: opening a fresh SQLite file next to somebody's
    Postgres-backed store would present as an empty memory, not as an error.
    """
    from agmem.stores.sqlite_doc import SqliteDocStore

    ns, root, config = _resolve(namespace, data_dir)
    if config is not None:
        doc_cls = config.overrides.get("doc_store") or config.slot_default("doc_store")
        if doc_cls != "SqliteDocStore":
            raise RuntimeError(
                f"recall reads the doc store directly and only knows SqliteDocStore; "
                f"the config resolves doc_store={doc_cls!r}"
            )
    return ns, SqliteDocStore(root / ns / "memory.db")


def open_memory(namespace: str | None = None, data_dir: str | None = None):
    """A memory handle for hook use: persistent, no LLM, no organizers.

    Deliberately organizer-free, whatever the config says. A hook runs on
    somebody's keystroke, so the path has to be the cheap one — an episode
    written to the store and nothing else. The MCP server is where organizers
    run, on writes a model chose to make.

    Without `AGMEM_CONFIG` this is the lite profile, which is also the server's
    default, so the two layers resolve the same stores. With it, the config's
    profile and overrides apply here exactly as `--config` applies them to the
    server; `sync_write` is forced on because a hook process exits as soon as
    `main` returns and a background worker would be killed mid-write.
    """
    from dataclasses import replace

    from agmem.config import AgmemConfig
    from agmem.memory import AgenticMemory

    ns, root, config = _resolve(namespace, data_dir)
    if config is None:
        config = AgmemConfig(profile="lite", data_dir=root, sync_write=True)
    else:
        config = replace(config, data_dir=root, sync_write=True)
    return AgenticMemory(namespace=ns, organizers=[], config=config)
