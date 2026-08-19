"""Configuration: profile presets + TOML overrides.

Priority (docs/01): explicit config value > profile default > capability
matching. Every experiment result must be stamped with the resolved
profile so runs are comparable.
"""

from __future__ import annotations

import os
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
    # Correction turns allowed for a malformed structured reply before the
    # call is dropped. 1 is what every arm measured before 2026-08-07 used;
    # raise it on arms whose per-conversation call count makes a zero-drop
    # ingest improbable (see StructuredCaller.__init__).
    structured_reply_retries: int = 1
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
    # RRF constant for channel fusion (`rrf_fuse`): score = 1/(rrf_k + rank),
    # rank 1-based. 60 is the textbook default and stays the framework's;
    # upstream Zep's rrf uses rank_const=1 (search_utils.py:1780-1786), so the
    # Zep recipe table emits 1 — see `rrf_fuse` for the exact (off-by-one)
    # relationship between the two parametrizations.
    rrf_k: int = 60
    # Cosine cutoff on the DENSE channel before fusion. 0 = off (framework
    # default — no other methodology cuts its dense channel); upstream Zep
    # applies DEFAULT_MIN_SCORE = 0.6 to its cosine channels
    # (search_utils.py:65), so the Zep recipes emit 0.6.
    dense_min_score: float = 0.0
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
    # False = one budget of `link_expansion_cap` shared by all hits (our default);
    # True = each hit gets its own, which is upstream's shape (memory_layer.py:895
    # breaks after appending, so a hit contributes up to k+1 neighbours at search
    # k). Exposed because the cap SHAPE is the one A-Mem read-path deviation we
    # can still price — the no-dedup half cannot survive the pipeline's own
    # (memory_type, id) dedup. See LinkExpansion's docstring.
    link_expansion_per_hit: bool = False
    attach_sources_top_r: int = 2  # Nemori r: top-r episodes carry source messages
    # GraphRecall (our own entity-hit -> incident-edge expansion) is OFF by
    # default now that `bfs_types` expresses Zep's actual φ_bfs channel; it was
    # only ever a stand-in for it. Non-zero revives it — see `GraphRecall`.
    graph_expansion_cap: int = 0
    graph_expansion_hops: int = 1
    # G-Memory query-graph read: 1-hop task expansion over `task_edges`
    # (paper Eq.(5)) plus task-association insight recall (Eq.(6)). 0 disables
    # the step; inert for stores whose `strategies` items lack the fields
    # (ReasoningBank).
    task_graph_expansion_cap: int = 5
    # Eq.(6) insight serving cap. 10 is `retrieve_memory`'s SIGNATURE default
    # (GMemory.py:194); the shipped harness ran insights_topk=3
    # (tasks/run.py:130) — the published operating point, recorded as
    # `GMEMORY_READ_RECIPE` in the organizer module.
    task_graph_insight_cap: int = 10
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
    # MemMachine's contextualized retrieval (`MemMachineContextualize`). These
    # are upstream's `query_memory` DEFAULTS, not its LoCoMo run: that run is
    # `limit=30, expand_context=3` (`evaluation/episodic_memory/
    # locomo_search.py`), an operating point of the harness rather than of the
    # library. Same split as MemoryOS's `page_recall_cap`, and for the same
    # reason — a default that quietly carries the eval lineage's number makes
    # every "default config" run a mislabeled eval-lineage run.
    memmachine_expand_context: int = 0
    memmachine_context_limit: int = 20
    # Read-side control policy (policies/retrieval.py), applied by
    # `retrieval/planned.py::searcher_for`. One of STRATEGIES
    # (direct/split_query/chain_of_query/tool_select), or None for a plain
    # single search. It lives here rather than at each call site so the switch
    # reaches the benchmarks, the MCP server and library callers alike — the
    # thing the first wiring got wrong by threading it through one harness
    # function. Non-None spends LLM calls PER QUESTION, so the default is off.
    query_strategy: str | None = None
    # The policy's rerank/truncate cap — upstream's `QueryParam.limit` where it
    # is actually read (`agent_utils.process_question`'s `search_limit=20`), not
    # the `k` of the inner searches.
    query_strategy_limit: int = 20

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


def _resolve_api_key(value: str) -> str:
    """Resolve the ``env:NAME`` indirection `agmem.example.toml` documents.

    The convention existed in the example file long before any code honored it:
    a literal ``"env:OPENAI_API_KEY"`` was passed straight through to the
    OpenAI client and failed as a 401 at the first call, far from the config
    that caused it. Resolution happens here, at load time, so an unset variable
    fails loudly where it is named. Any other value — including the
    ``"not-needed"`` default local servers ignore — passes through untouched."""
    if value.startswith("env:"):
        name = value[len("env:") :]
        resolved = os.environ.get(name)
        if not resolved:
            raise ValueError(
                f"api_key = {value!r}: environment variable {name!r} is not set (or empty)"
            )
        return resolved
    return value


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
            api_key=_resolve_api_key(cfg.get("api_key", "not-needed")),
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 1024),
            # Dropped here until 2026-08-19, which made part of the model
            # registry unreachable from TOML: gpt-5.6-luna rejects `max_tokens`
            # (needs `max_completion_tokens`), and endpoints needing request
            # extras (vLLM guided_json, reasoning_effort) need `extra_body` —
            # the Python API could set both, so only the runbook's TOML path
            # silently had the smaller reachable set.
            max_tokens_key=cfg.get("max_tokens_key", "max_tokens"),
            extra_body=dict(cfg.get("extra_body", {})),
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
        structured_reply_retries=raw.get("llm_options", {}).get(
            "structured_reply_retries", defaults.structured_reply_retries
        ),
        lexical_types=tuple(retrieval.get("lexical_types", defaults.lexical_types)),
        bfs_types=tuple(retrieval.get("bfs_types", defaults.bfs_types)),
        bfs_max_depth=retrieval.get("bfs_max_depth", defaults.bfs_max_depth),
        rrf_k=retrieval.get("rrf_k", defaults.rrf_k),
        dense_min_score=retrieval.get("dense_min_score", defaults.dense_min_score),
        reranker_params=dict(retrieval.get("reranker_params", defaults.reranker_params)),
        link_expansion_cap=retrieval.get("link_expansion_cap", defaults.link_expansion_cap),
        link_expansion_per_hit=retrieval.get(
            "link_expansion_per_hit", defaults.link_expansion_per_hit
        ),
        attach_sources_top_r=retrieval.get("attach_sources_top_r", defaults.attach_sources_top_r),
        graph_expansion_cap=retrieval.get("graph_expansion_cap", defaults.graph_expansion_cap),
        graph_expansion_hops=retrieval.get("graph_expansion_hops", defaults.graph_expansion_hops),
        task_graph_expansion_cap=retrieval.get(
            "task_graph_expansion_cap", defaults.task_graph_expansion_cap
        ),
        task_graph_insight_cap=retrieval.get(
            "task_graph_insight_cap", defaults.task_graph_insight_cap
        ),
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
        memmachine_expand_context=retrieval.get(
            "memmachine_expand_context", defaults.memmachine_expand_context
        ),
        memmachine_context_limit=retrieval.get(
            "memmachine_context_limit", defaults.memmachine_context_limit
        ),
        query_strategy=retrieval.get("query_strategy", defaults.query_strategy),
        query_strategy_limit=retrieval.get("query_strategy_limit", defaults.query_strategy_limit),
    )
