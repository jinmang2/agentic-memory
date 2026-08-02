"""Named organizer configs for the repro harness — the runner-facing policy table.
Each entry is (factory, memory_types): the factory builds FRESH organizer instances
per conversation (never share organizer state across convs), and memory_types is
what the eval read path retrieves. Arm variants (e.g. nemori_merge085) live here so
an experiment arm is a --config name, reproducible from the CLI line alone."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agmem.organizers.amem import AMemOrganizer
from agmem.organizers.nemori import NemoriOrganizer


@dataclass(frozen=True)
class RunnerConfig:
    name: str
    factory: Callable[[], list]
    memory_types: tuple[str, ...]
    # run_ready=False = constructible for counting/tests but NOT wired for a real run yet.
    # Kept (not removed) for future arms that land before their threading does — Track 1
    # only flips the two nemori entries below, which now carry role_temps/per_type_k/store.
    run_ready: bool = True
    # role -> RoleConfig kwarg overrides, applied over exp_amem_repro.py's make_roles
    # defaults (None = keep those defaults, e.g. amem's byte-identical path).
    role_temps: dict[str, dict] | None = None
    # memory_type -> k, replacing the scalar --k for eval retrieval when set.
    per_type_k: dict[str, int] | None = None
    # kwargs merged into AgmemConfig(**...) construction (build_memory) — e.g. slot
    # overrides for vector_store/doc_store.
    store: dict | None = None
    # exp_locomo_conv0.py known-table's 4th tuple field: whether the read path
    # rewrites the raw question into LLM-generated keywords before retrieval
    # (A-Mem's own read-path mechanism, True there) or searches the raw
    # question (Nemori's published read path — 0 extra LLM calls, False).
    # Default True keeps amem's existing behavior; both nemori entries set it
    # False below so a nemori eval does not silently inherit A-Mem's query
    # rewrite (an extra LLM call and a protocol deviation from the claim under
    # verification).
    keyword_queries: bool = True


# Nemori "upstream" preset threading — precheck §7 (docs/_internal/plans/
# 2026-07-31-track1-nemori-fidelity-precheck.md), verified there against
# exp_locomo_conv0.py's NEMORI_TEMPS (~:350) and NEMORI_STORE (~:360).
# extract: temperature 0.2 (upstream segmenter.py:63) + max_tokens 4096
# (upstream segmenter.py:64 — F1, free to fix while threading). distill:
# temperature 0.7 / max_tokens 2000 (upstream client.py:31-32,
# orchestrator.py:38-39). generate: temperature 0.0 (upstream search.py:169)
# — exp_locomo_conv0.py's own make_roles already defaults generate to 0.0, so
# its NEMORI_TEMPS constant doesn't need this key; exp_amem_repro.py's
# make_roles defaults generate to A-Mem's 0.7 and must be told explicitly.
NEMORI_ROLE_TEMPS = {
    "extract": {"temperature": 0.2, "max_tokens": 4096},
    "distill": {"temperature": 0.7, "max_tokens": 2000},
    "generate": {"temperature": 0.0},
}
# search_top_k_semantic=20 for the predict-stage retrieval; episodes k=10
# (upstream search.py:218-219; evaluation/locomo/config.json).
NEMORI_PER_TYPE_K = {"episodes": 10, "semantic": 20}
# Lineage-faithful engines (exp_locomo_conv0.py:360-363): Nemori upstream ran
# on PostgreSQL(tsvector) + Qdrant — both real via embedded builds
# (pgserver / qdrant local mode). Shaped as {"overrides": {...}} so
# `AgmemConfig(..., **(cfg_entry.store or {}))` lands on AgmemConfig's actual
# `overrides: dict[str, str]` slot-override field (config.py), not a bare
# "vector_store"/"doc_store" kwarg AgmemConfig doesn't have.
NEMORI_STORE = {"overrides": {"vector_store": "QdrantVectorStore", "doc_store": "PostgresDocStore"}}

CONFIGS: dict[str, RunnerConfig] = {
    c.name: c
    for c in (
        RunnerConfig("amem", lambda: [AMemOrganizer()], ("notes",)),
        # factory + memory_types verbatim from exp_locomo_conv0.py:386-393:
        RunnerConfig(
            "nemori_upstream",
            lambda: [NemoriOrganizer(fidelity="upstream")],
            ("episodes", "semantic"),
            run_ready=True,
            role_temps=NEMORI_ROLE_TEMPS,
            per_type_k=NEMORI_PER_TYPE_K,
            store=NEMORI_STORE,
            keyword_queries=False,
        ),
        RunnerConfig(
            "nemori_merge085",
            lambda: [NemoriOrganizer(fidelity="upstream", merge_similarity=0.85)],
            ("episodes", "semantic"),
            run_ready=True,
            role_temps=NEMORI_ROLE_TEMPS,
            per_type_k=NEMORI_PER_TYPE_K,
            store=NEMORI_STORE,
            keyword_queries=False,
        ),
    )
}


def get_config(name: str) -> RunnerConfig:
    try:
        return CONFIGS[name]
    except KeyError:
        raise KeyError(f"unknown runner config {name!r} (known: {sorted(CONFIGS)})") from None
