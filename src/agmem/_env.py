"""Dependency-free ``.env.local`` loader (stdlib only, no python-dotenv).

Reproduction scripts need ``OPENAI_API_KEY`` from the repo-root ``.env.local``
without adding a pip dependency. ``load_env_local`` parses ``KEY=VALUE`` lines
into ``os.environ`` and NEVER overwrites an already-set variable, so an
explicit ``export OPENAI_API_KEY=...`` in the shell always wins over the file.
"""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    """Repo root = two parents up from this file (src/agmem/_env.py)."""
    return Path(__file__).resolve().parents[2]


def load_env_local(path: str | Path | None = None) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines from ``.env.local`` into ``os.environ``.

    - ``path=None`` -> repo-root ``.env.local``.
    - Blank lines and lines whose first non-space char is ``#`` are skipped.
    - A leading ``export `` prefix is tolerated (``export KEY=VALUE``).
    - Surrounding single/double quotes on the value are stripped.
    - Already-set environment variables are NOT overwritten (shell wins).
    - A missing file is a no-op (returns an empty dict), not an error.

    Returns the mapping of keys actually SET by this call (i.e. those that were
    previously absent from ``os.environ``), so callers can log what loaded
    without ever touching the values.
    """
    env_path = Path(path) if path is not None else _repo_root() / ".env.local"
    if not env_path.is_file():
        return {}

    applied: dict[str, str] = {}
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied
