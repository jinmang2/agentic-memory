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
    # Memory types that additionally get Zep's graph BFS channel (φ_bfs), and
    # how deep it walks. Upstream's MAX_SEARCH_DEPTH is 3 and applies only where
    # a recipe enables the channel, which is why the depth has a default but the
    # type set is empty: no methodology except Zep searches a graph.
    bfs_types: tuple[str, ...] = ()
    bfs_max_depth: int = 3
    # Which reranker (slot resolution) gets which constructor arguments —
    # e.g. the Zep paper's BGE-m3 cross-encoder is
    # {"model_name": "BAAI/bge-reranker-v2-m3"}, and MMR at mmr_lambda=1 is
    # {"lambda_": 1.0}. Kept as a dict rather than one field per reranker
    # because the recipes vary exactly one argument each.
    reranker_params: dict[str, Any] = field(default_factory=dict)
    # Read-path post-step knobs (retrieval/steps.py). Defaults reproduce the
    # methodology-faithful behavior; 0 disables the step. The first two are
    # documented deviations from upstream (A-Mem caps per hit, not globally),
    # so they must be reachable from config to be ablatable at all.
    link_expansion_cap: int = 5  # A-Mem 1-hop note-link expansion, global cap
    attach_sources_top_r: int = 2  # Nemori r: top-r episodes carry source messages
    # GraphRecall (our own entity-hit -> incident-edge expansion) is OFF by
    # default now that `bfs_types` expresses Zep's actual φ_bfs channel; it was
    # only ever a stand-in for it. Non-zero revives it — see `GraphRecall`.
    graph_expansion_cap: int = 0
    graph_expansion_hops: int = 1
    # MemoryOS second-stage page recall: upstream `retrieval_queue_capacity`
    # and `page_similarity_threshold`. 0 disables it, which serves segment
    # summaries — a channel upstream has no counterpart for.
    #
    # The cap is LINEAGE-SPLIT and this default is the pypi one, because
    # `MEMORYOS_PRESETS` defaults to pypi: the maintained library passes 7
    # (`memoryos.py` `retrieval_queue_capacity=7`, `Retriever` default 7) while
    # the harness that produced the paper's LoCoMo numbers passes 10
    # (`eval/main_loco_parse.py`: `RetrievalAndAnswer(..., queue_capacity=10)`).
    # It sat at 10 for both, so the default preset was reading with the eval
    # lineage's queue — the mixing `MEMORYOS_PRESETS` exists to prevent. An
    # eval-lineage run has to set 10 here explicitly (the `memoryos_eval`
    # config in `scripts/exp_locomo_conv0.py` does).
    page_recall_cap: int = 7
    page_recall_threshold: float = 0.1
    # First-stage gate, `segment_similarity_threshold` — 0.1 in both lineages'
    # drivers (the eval class default of 0.8 is overridden by its own caller).
    page_recall_segment_threshold: float = 0.1
    # Which copy's keyword-overlap formula the segment score uses. MUST match
    # the organizer's `keyword_similarity`: `eval` uses the containment mean on
    # both sides, `memoryos-chromadb` uses Jaccard on both, and `memoryos-pypi`
    # has no read-side keyword term at all (so the value is inert there).
    page_recall_keyword_similarity: str = "containment_mean"

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
    are ignored; only `[profile]`, `[storage]`, `[embed]`, `[override]`,
    `[write]`, `[retrieval]`, `[llm_options]`, and `[llm.<role>]` are read.

    `[llm_options].guided_json` was read here but appeared in no docs list, no
    docstring and not in `agmem.example.toml` — an undiscoverable switch, which
    matters because both experiment scripts set `use_guided_json=False` in Python
    and a TOML-driven run had no documented way to match them.

    `[retrieval]` exists because `retrieval/steps.py` claims its read-path
    deviations (A-Mem's global link cap, Nemori's source-attachment `r`) are
    "configured from AgmemConfig, so those deviations are finally ablatable" —
    which was true through the Python API and false through TOML, the path the
    repro runbook uses. Omitted keys keep the `AgmemConfig` defaults, so an
    existing config file resolves exactly as before."""
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

    defaults = AgmemConfig()
    retrieval = raw.get("retrieval", {})
    return AgmemConfig(
        profile=profile,
        data_dir=data_dir,
        embed_model=raw.get("embed", {}).get("model"),
        overrides=dict(raw.get("override", {})),
        llm_roles=llm_roles,
        strict=raw.get("profile", {}).get("strict", False),
        sync_write=raw.get("write", {}).get("sync", defaults.sync_write),
        use_guided_json=raw.get("llm_options", {}).get("guided_json", defaults.use_guided_json),
        lexical_types=tuple(retrieval.get("lexical_types", defaults.lexical_types)),
        bfs_types=tuple(retrieval.get("bfs_types", defaults.bfs_types)),
        bfs_max_depth=retrieval.get("bfs_max_depth", defaults.bfs_max_depth),
        reranker_params=dict(retrieval.get("reranker_params", defaults.reranker_params)),
        link_expansion_cap=retrieval.get("link_expansion_cap", defaults.link_expansion_cap),
        attach_sources_top_r=retrieval.get("attach_sources_top_r", defaults.attach_sources_top_r),
        graph_expansion_cap=retrieval.get("graph_expansion_cap", defaults.graph_expansion_cap),
        graph_expansion_hops=retrieval.get("graph_expansion_hops", defaults.graph_expansion_hops),
        page_recall_cap=retrieval.get("page_recall_cap", defaults.page_recall_cap),
        page_recall_threshold=retrieval.get(
            "page_recall_threshold", defaults.page_recall_threshold
        ),
        page_recall_segment_threshold=retrieval.get(
            "page_recall_segment_threshold", defaults.page_recall_segment_threshold
        ),
        page_recall_keyword_similarity=retrieval.get(
            "page_recall_keyword_similarity", defaults.page_recall_keyword_similarity
        ),
    )
