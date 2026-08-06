"""Named organizer configs for the repro harness — the runner-facing policy table.
Each entry is (factory, memory_types): the factory builds FRESH organizer instances
per conversation (never share organizer state across convs), and memory_types is
what the eval read path retrieves. Arm variants (e.g. nemori_merge085) live here so
an experiment arm is a --config name, reproducible from the CLI line alone."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agmem.organizers.amem import AMemOrganizer
from agmem.organizers.mem0 import Mem0Organizer
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

# Mem0 v0.1.94 threading. Every value below was read off the pinned clone
# (~/.agmem/upstream/mem0), not off the study document.
# temperature 0.1 / max_tokens 2000 are BaseLlmConfig's defaults
# (mem0/configs/llms/base.py:17,19 @ v0.1.94) and both phases inherit them. Our
# RoleConfig already defaults temperature to 0.1 but max_tokens far lower, and a
# truncated decision response is a silent fidelity break: the response lists
# EVERY candidate back, including the NONE rows, so it is the longest output in
# the write path. generate 0.0 is the harness answer call
# (evaluation/src/memzero/search.py:114 @ evaluation-archive); exp_amem_repro's
# make_roles defaults generate to A-Mem's 0.7, so it must be set explicitly —
# the same trap Nemori's entry documents.
MEM0_ROLE_TEMPS = {
    "extract": {"temperature": 0.1, "max_tokens": 2000},
    "distill": {"temperature": 0.1, "max_tokens": 2000},
    "generate": {"temperature": 0.0},
}
# The published operating point: `make run-mem0-search` passes --top_k 30 and
# run_experiments.py:29 defaults to 30. The trap worth naming (study M0-C11):
# MemorySearch.__init__ defaults to **10**, so anyone calling the class directly
# instead of through the Makefile silently halves k. We pin 30 and footnote it,
# because each arm of the 4-way table runs its own lineage-faithful read k
# (Nemori 10/20, A-Mem 10) — symmetry across arms is not the goal, fidelity is.
MEM0_PER_TYPE_K = {"semantic": 30}
# Qdrant is the OSS default vector provider (mem0/vector_stores/configs.py
# `provider` default "qdrant" @ v0.1.94); the history db is plain SQLite
# (configs/base.py:42 history_db_path), which is already our doc-store default,
# so only the vector slot is overridden. Same {"overrides": {...}} shape as
# NEMORI_STORE for the same reason.
MEM0_STORE = {"overrides": {"vector_store": "QdrantVectorStore"}}

CONFIGS: dict[str, RunnerConfig] = {
    c.name: c
    for c in (
        RunnerConfig("amem", lambda: [AMemOrganizer()], ("notes",)),
        # Read-protocol ablation of the entry above, and ONLY of that: identical
        # organizer, memory_types, temperatures and store, so it evaluates the
        # SAME ingested store and differs at retrieval alone. It exists because
        # the LLM keyword rewrite is one of the two deviations docs/13 §6-2
        # records against A-Mem's own read path ("read는 순수 dense top-k, LLM
        # 0회") and instructs us to state whenever we compare — and the four-way
        # table is that comparison. A-Mem is the only arm paying a read-side LLM
        # call there (1,986 of them; Nemori and Mem0 pay zero), so the asymmetry
        # was uncontrolled and its DIRECTION unmeasured: keyword replacement can
        # help (drops stopwords, concentrates entities) or hurt (loses sentence
        # semantics, notably on temporal questions). This arm measures it.
        # Not a claim that raw-question is more faithful in every respect: the
        # second deviation (global-5 link-expansion cap vs upstream's per-hit)
        # is untouched here and needs its own change to move.
        RunnerConfig("amem_rawq", lambda: [AMemOrganizer()], ("notes",), keyword_queries=False),
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
        # Track 2. batch_size=2 is the PAPER-HARNESS shape, not a library
        # default: upstream's add() batches nothing and its harness passes two
        # messages per call (evaluation/src/memzero/add.py:46 @
        # evaluation-archive). Mechanism lives in the organizer, policy here.
        # keyword_queries=False because the harness searches with the RAW
        # question — answer_question hands `question` straight to search_memory
        # (search.py:90-96) and Mem0's read path has no query rewrite anywhere.
        # Inheriting A-Mem's LLM keyword rewrite would add a call upstream never
        # makes and change the retrieval protocol under test; that is exactly the
        # defect Track 1 fixed at 7d9b64e.
        RunnerConfig(
            "mem0_v0194",
            lambda: [Mem0Organizer(batch_size=2)],
            ("semantic",),
            run_ready=True,
            role_temps=MEM0_ROLE_TEMPS,
            per_type_k=MEM0_PER_TYPE_K,
            store=MEM0_STORE,
            keyword_queries=False,
        ),
    )
}


def get_config(name: str) -> RunnerConfig:
    try:
        return CONFIGS[name]
    except KeyError:
        raise KeyError(f"unknown runner config {name!r} (known: {sorted(CONFIGS)})") from None
