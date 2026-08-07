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
from agmem.organizers.zep_graph import ZepGraphOrganizer, zep_search_recipe


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
    # A-Mem link expansion, when this arm wants something other than the runner's
    # own `--expand-links` default. None = leave the runner's expression alone
    # (5 when on, 0 when off), so every existing arm is byte-identical. An int
    # sets the cap; `link_expansion_per_hit` makes that cap PER HIT, which is
    # upstream's shape (memory_layer.py:889-897). They travel together because a
    # per-hit budget of 5 is neither our design nor upstream's.
    link_expansion_cap: int | None = None
    link_expansion_per_hit: bool = False


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

# Zep's read path is a RECIPE TABLE, not a single answer: the paper presents
# three search functions and five rerankers as system components, and upstream
# ships their combinations as named SearchConfigs. §4.1 fixes which one produced
# the paper's numbers — "BGE-m3 models from BAAI for both reranking and
# embedding" — so `cross_encoder` is the operating point and the only family
# carrying a BFS channel. The recipe supplies memory_types, the lexical/BFS
# channels, rrf_k=1, dense_min_score=0.6 and the reranker slot override in one
# object, which is exactly what keeps a run from becoming a hybrid no upstream
# recipe has (this project has made that mistake: RRF fusion + a BFS-ish
# GraphRecall, a combination upstream never ships).
#
# EMBEDDER: the campaign standard `text-embedding-3-small`, NOT the paper's
# BGE-m3 (controller decision 2026-08-07). The four-way table's whole purpose is
# internal comparability, and an embedder swap moved A-Mem 9.87 J — a Zep row on
# a different embedder could not be read against the other three. A BGE-m3
# lineage arm stays available as a later addition; the recipe's RERANKER is
# unaffected either way and stays BGE.
ZEP_RECIPE_NAME = "cross_encoder"
_ZEP_RECIPE = zep_search_recipe(ZEP_RECIPE_NAME)
# Upstream applies one `limit` across every subgraph rather than a per-type
# table, so the scalar is expanded here to the shape RunnerConfig speaks.
ZEP_PER_TYPE_K = {t: _ZEP_RECIPE.limit for t in _ZEP_RECIPE.memory_types}

CONFIGS: dict[str, RunnerConfig] = {
    c.name: c
    for c in (
        RunnerConfig("amem", lambda: [AMemOrganizer()], ("notes",)),
        # Read-protocol ablation of the entry above, and ONLY of that: identical
        # organizer, memory_types, temperatures and store, so it evaluates the
        # SAME ingested store and differs at retrieval alone.
        #
        # The `amem` entry's keyword rewrite is NOT a deviation of ours — it is
        # what upstream's evaluation harness does: `answer_question` opens with
        # `keywords = self.generate_query_llm(question)` and searches with those
        # keywords instead of the question (test_advanced.py:129,134, and the
        # same in test_advanced_robust.py:111-112 @ the pinned SHA). A-Mem is
        # therefore the only arm here paying a read-side LLM call, and that
        # asymmetry belongs to A-Mem. `amem` stays the faithful headline arm.
        #
        # This arm exists to PRICE that step, which upstream never did: measured
        # at +5.26 J and -1,986 calls for dropping it (ledger B-8). Both stay
        # wired — a knob whose effect is measured is worth keeping addressable
        # even if one setting is later retired. Not a fidelity claim in either
        # direction: our one real read-path deviation is LinkExpansion's global
        # cap of 5 where upstream caps per hit, and it is untouched by both arms.
        RunnerConfig("amem_rawq", lambda: [AMemOrganizer()], ("notes",), keyword_queries=False),
        # The other read-path ablation, and the one that IS our deviation: our
        # LinkExpansion spends a single budget of 5 neighbours across all hits,
        # where upstream gives each hit its own and breaks only after appending
        # (`memory_layer.py:895`), so at the eval's k=10 each hit may contribute
        # up to k+1 = 11. cap=11 transcribes that arithmetic.
        #
        # In practice ANY per-hit cap >= 5 is the same arm, because a note cannot
        # hold more than 5 links: the write path only ever retrieves 5 neighbour
        # candidates (`AMemOrganizer(top_k=5)`, upstream `memory_layer.py:755`
        # `find_related_memories(note.content, k=5)`). Measured on this store —
        # 5,882 notes, 18,886 links, mean 3.21, and the distribution stops dead
        # at 5 with nothing above it. So upstream's per-hit cap never binds and
        # means "serve every link of every hit", while OUR global 5 does bind
        # (measured: exactly 15 items served per question, 10 hits + 5, saturated
        # every time). What this arm changes is therefore ~32 candidates versus
        # 5, not 110 versus 5.
        #
        # Everything else matches `amem`, keyword rewrite included, so this arm
        # is `amem` plus the one remaining fidelity gap closed — not a second
        # ablation stacked on the first.
        RunnerConfig(
            "amem_perhit",
            lambda: [AMemOrganizer()],
            ("notes",),
            link_expansion_cap=11,
            link_expansion_per_hit=True,
        ),
        # Both read-path changes at once. Not a fidelity arm — it is deliberately
        # the LEAST faithful of the four, dropping upstream's query rewrite while
        # keeping its link budget. It exists because the docs had to say "+5.26
        # and +1.36 were measured against different baselines, do not add them",
        # and a stated non-additivity is worth one run to resolve. It also
        # answers the deployment question the fidelity arms cannot: given this
        # store, what is the best read configuration available?
        RunnerConfig(
            "amem_rawq_perhit",
            lambda: [AMemOrganizer()],
            ("notes",),
            keyword_queries=False,
            link_expansion_cap=11,
            link_expansion_per_hit=True,
        ),
        # Track 3. run_ready=False until the counting pass and the quote gate
        # clear: the study reads upstream as "at least 4-6 LLM calls per episode"
        # (docs/research/zep-graphiti.md), which at 5,882 turns is 23.5k-35.3k
        # write calls — two to three times A-Mem's 11,754, and the most
        # expensive arm in this campaign if it holds. That range is a READ of
        # prose, not a measurement, and Track 2 established that borrowed
        # calibration is worth ~2x error; the number that gates spending has to
        # come from CountingLLM against this port.
        #
        # Also unresolved: role temperatures. Every other arm pins its lineage's
        # temps; the Zep study records none, so this entry inherits the runner's
        # A-Mem-shaped defaults, which is a stated gap rather than a decision.
        # It does not affect call COUNTS, so the counting pass can proceed, but
        # it must be settled before any paid run.
        RunnerConfig(
            "zep_cross_encoder",
            lambda: [ZepGraphOrganizer()],
            _ZEP_RECIPE.memory_types,
            run_ready=False,
            per_type_k=ZEP_PER_TYPE_K,
            store=_ZEP_RECIPE.config_kwargs(),
            keyword_queries=False,
        ),
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
