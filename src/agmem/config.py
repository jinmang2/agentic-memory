"""Configuration: profile presets + TOML overrides.

Priority (docs/01): explicit config value > profile default > capability
matching. Every experiment result must be stamped with the resolved
profile so runs are comparable.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agmem.llm.client import RoleConfig

# slot -> adapter class name per profile (docs/01 §4)
PROFILES: dict[str, dict[str, str]] = {
    "lite": {
        "vector_store": "SqliteVecStore",
        "doc_store": "SqliteDocStore",
        "graph_store": "KuzuGraphStore",
        "embedder": "SentenceTransformerEmbedder",
        "reranker": "NoopReranker",
    },
    "standard": {
        "vector_store": "LanceDBVectorStore",
        "doc_store": "SqliteDocStore",
        "graph_store": "KuzuGraphStore",
        "embedder": "SentenceTransformerEmbedder",
        "reranker": "LLMReranker",
    },
    "full": {
        "vector_store": "QdrantVectorStore",
        "doc_store": "PostgresDocStore",
        "graph_store": "Neo4jGraphStore",
        "embedder": "APIEmbedder",
        "reranker": "CrossEncoderReranker",
    },
}

DEFAULT_EMBED_MODEL = {
    "lite": "intfloat/multilingual-e5-small",
    "standard": "BAAI/bge-m3",
    "full": "text-embedding-3-small",
}


@dataclass
class AgmemConfig:
    """Resolved run configuration. `profile` selects a `PROFILES` entry that
    `overrides` can shadow slot-by-slot; `data_dir=None` forces every store to
    its in-memory mode (used by tests). `sync_write=False` routes writes
    through memory.py's background worker instead of the calling thread.
    """

    profile: str = "lite"
    data_dir: Path | None = None  # None -> in-memory (tests)
    embed_model: str | None = None  # None -> profile default
    overrides: dict[str, str] = field(default_factory=dict)  # slot -> class name
    llm_roles: dict[str, RoleConfig] = field(default_factory=dict)
    strict: bool = False
    sync_write: bool = True  # False -> background write worker (memory.py)
    use_guided_json: bool = True
    # memory types that get a BM25/FTS lexical channel fused with dense
    # (Zep hybrid search adds "facts"/"entities"; A-Mem/Nemori stay
    # dense-only as their upstream evals do)
    lexical_types: tuple[str, ...] = ("episodic",)

    def slot_default(self, slot: str) -> str | None:
        """`overrides[slot]` if set, else the profile's default class name for
        `slot`, else `None` if `slot`/`profile` is unknown (caller then falls
        back to the resolver's own preference order)."""
        if slot in self.overrides:
            return self.overrides[slot]
        return PROFILES.get(self.profile, {}).get(slot)

    @property
    def resolved_embed_model(self) -> str:
        """`embed_model` if explicitly set, else the profile's default model
        (falling back to the `lite` default for an unknown profile)."""
        return self.embed_model or DEFAULT_EMBED_MODEL.get(
            self.profile, DEFAULT_EMBED_MODEL["lite"]
        )


def load_config(path: str | Path) -> AgmemConfig:
    """Parse a TOML config file into an `AgmemConfig`. Missing `path` or
    malformed TOML raises (`FileNotFoundError`/`tomllib.TOMLDecodeError`) —
    there is no silent fallback to defaults. Unrecognized top-level tables
    are ignored; only `[profile]`, `[storage]`, `[embed]`, `[override]`, and
    `[llm.<role>]` are read."""
    raw: dict[str, Any] = tomllib.loads(Path(path).read_text())

    profile = raw.get("profile", {}).get("name", "lite")
    storage = raw.get("storage", {})
    data_dir = Path(storage["data_dir"]).expanduser() if "data_dir" in storage else None

    llm_roles: dict[str, RoleConfig] = {}
    for role, cfg in raw.get("llm", {}).items():
        llm_roles[role] = RoleConfig(
            endpoint=cfg["endpoint"],
            model=cfg["model"],
            api_key=cfg.get("api_key", "not-needed"),
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 1024),
        )

    return AgmemConfig(
        profile=profile,
        data_dir=data_dir,
        embed_model=raw.get("embed", {}).get("model"),
        overrides=dict(raw.get("override", {})),
        llm_roles=llm_roles,
        strict=raw.get("profile", {}).get("strict", False),
        sync_write=raw.get("write", {}).get("sync", True),
    )
