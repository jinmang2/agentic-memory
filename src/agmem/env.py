"""The three runtime knobs every deployment surface resolves the same way.

Namespace, data directory and config path decide WHICH store a process opens.
Two surfaces open stores — the MCP server (`agmem.mcp.server`) and the Claude
Code hooks (`agmem.hooks`) — and until 2026-09-02 each resolved these on its
own: the server from CLI flags with its own defaults, the hooks from environment
variables with different ones. Their namespace defaults disagreed (`main`
against `claude-code`), which meant the two layers of the product opened two
different stores by default and neither could see what the other wrote — no
error, just nothing retrieved (github issue #2).

This module is the single place those defaults live, imported by both. It is
deliberately dependency-free so the recall hook, which has a 10 s budget and
spends 0.18 s, can import it without dragging in the config loader (which
imports the LLM client).

Precedence, per docs/05 §2.2: explicit argument (CLI flag or function
parameter) > environment variable > `agmem.toml` (data_dir only) > default.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_NAMESPACE = "AGMEM_NAMESPACE"
ENV_DATA_DIR = "AGMEM_DATA_DIR"
ENV_CONFIG = "AGMEM_CONFIG"

DEFAULT_NAMESPACE = "main"
DEFAULT_DATA_DIR = Path.home() / ".agmem/data"


class InvalidNamespace(ValueError):
    """A namespace that could not be a single directory name."""


def validate_namespace(namespace: str) -> str:
    """Return `namespace` if it is a plain directory name, else raise.

    A namespace is used verbatim as one path segment under the data directory
    (`<data_dir>/<namespace>/memory.db`), and since the MCP tools now accept it
    from the model as an argument, it has to be checked where it enters rather
    than trusted to stay inside `data_dir`.
    """
    if not namespace or namespace in (".", "..") or "/" in namespace or "\\" in namespace:
        raise InvalidNamespace(f"namespace must be a single directory name, got {namespace!r}")
    if namespace.startswith(".") or any(ch.isspace() or ord(ch) < 32 for ch in namespace):
        raise InvalidNamespace(f"namespace must be a single directory name, got {namespace!r}")
    return namespace


def resolve_namespace(explicit: str | None = None) -> str:
    """Explicit value > `AGMEM_NAMESPACE` > `DEFAULT_NAMESPACE`, validated."""
    return validate_namespace(explicit or os.environ.get(ENV_NAMESPACE) or DEFAULT_NAMESPACE)


def resolve_data_dir(explicit: str | Path | None = None, from_config: Path | None = None) -> Path:
    """Explicit value > `AGMEM_DATA_DIR` > the config file's `[storage].data_dir` > default."""
    chosen = explicit or os.environ.get(ENV_DATA_DIR) or from_config or DEFAULT_DATA_DIR
    return Path(chosen).expanduser()


def resolve_config_path(explicit: str | Path | None = None) -> Path | None:
    """Explicit value > `AGMEM_CONFIG` > none (profile defaults apply)."""
    chosen = explicit or os.environ.get(ENV_CONFIG)
    return Path(chosen).expanduser() if chosen else None
