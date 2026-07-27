"""LoCoMo conv0 비교 실험 (7개 config: passthrough / amem / nemori /
*_mixed / memoryos / zep_graph), 로컬 Qwen3-0.6B.

write 경로 온도는 방법론별 업스트림 값을 따른다(round-4 결정: upstream 충실):
A-Mem은 get_completion 기본 0.7, Nemori는 segmentation 0.2 + episode/semantic
0.7(클라이언트 기본, max_tokens 2000). 답변(generate)은 공통 프레임 t=0.0.

실행:
    uv run python scripts/exp_locomo_conv0.py [--max-sessions N] [--limit N]
결과: results/locomo-conv0-<config>.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agmem import AgenticMemory
from agmem.bench import locomo
from agmem.bench import stamp as bench_stamp
from agmem.config import AgmemConfig
from agmem.embed.st_embedder import SentenceTransformerEmbedder
from agmem.llm.client import RoleConfig
from agmem.organizers.amem import AMemOrganizer
from agmem.organizers.experimental import ChainedConsumer
from agmem.organizers.gated import AdmissionGated
from agmem.organizers.memmachine import MemMachineOrganizer, MemMachineProfileOrganizer
from agmem.organizers.memoryos import MemoryOSOrganizer
from agmem.organizers.nemori import NemoriOrganizer
from agmem.organizers.zep_graph import SearchRecipe, zep_search_recipe
from agmem.policies.admission import AdmissionGate

DATA = Path.home() / ".agmem/datasets/locomo10.json"
OUT = Path(__file__).resolve().parent.parent / "results"

NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}

# Derived item types the snapshot enumerates, mirroring exp_amem_repro.py so the
# two experiments' memory_capacity blocks stay comparable. Empty types are
# skipped, so listing extras is harmless.
SNAPSHOT_ITEM_TYPES = (
    "notes",
    "semantic",
    "facts",
    "entities",
    "episodes",
    "pages",
    "playbook",
    "strategies",
    "experiences",
    "state",
)


def capture_memory(mem, snapshot_path: Path) -> dict:
    """Per-type counts + stored bytes after ingest, and a full item snapshot.

    Storage is the verdict criterion for Nemori v4 Table 7 (does a granularity
    land in the paper's 45-64% reduction band?), and neither F1 nor the LLM
    budget can answer it — so counts and bytes are captured here rather than
    re-derived from a re-run. `bytes` is the utf-8 length of each item's JSON
    form: `data_dir=None` makes every store in-memory, so an on-disk footprint
    would be 0 and meaningless. Excludes `state` from the totals — consolidate
    cursors are bookkeeping, not stored memory."""
    counts: dict[str, int] = {}
    nbytes: dict[str, int] = {}
    with snapshot_path.open("w") as out:
        episodes = mem.doc_store.list_episodes(mem.namespace)
        if episodes:
            counts["episodic"] = len(episodes)
            nbytes["episodic"] = 0
            for ep in episodes:
                line = json.dumps(
                    {
                        "memory_type": "episodic",
                        "id": ep.id,
                        "role": ep.role,
                        "content": ep.content,
                        "timestamp": str(ep.timestamp),
                        "meta": ep.meta,
                    },
                    ensure_ascii=False,
                    default=str,
                )
                nbytes["episodic"] += len(line.encode())
                out.write(line + "\n")
        for mtype in SNAPSHOT_ITEM_TYPES:
            items = mem.doc_store.list_items(mtype, namespace=mem.namespace)
            if not items:
                continue
            counts[mtype] = len(items)
            nbytes[mtype] = 0
            for item in items:
                line = json.dumps({"memory_type": mtype, **item}, ensure_ascii=False, default=str)
                nbytes[mtype] += len(line.encode())
                out.write(line + "\n")
    derived = {t: n for t, n in counts.items() if t not in ("episodic", "state")}
    return {
        "counts": counts,
        "bytes": nbytes,
        "derived_item_count": sum(derived.values()),
        "derived_bytes": sum(v for t, v in nbytes.items() if t not in ("episodic", "state")),
        "snapshot": snapshot_path.name,
    }


def make_roles(overrides: dict[str, dict] | None = None) -> dict[str, RoleConfig]:
    """Build the 4-role (extract/distill/judge/generate) local llama.cpp
    `RoleConfig` set. `overrides` maps role -> partial kwargs that are
    merged over that role's default (e.g. a methodology-specific temperature)
    before construction; unmentioned roles keep the defaults below."""
    # Defaults: write-path roles 0.1; judge/generate 0.0 (Nemori answers at
    # t=0.0, ReasoningBank judges at t=0.0). max_tokens 1000 per audit A6:
    # 300 could truncate multi-neighbor evolution JSON -> parse failure ->
    # drop. Per-methodology upstream temps come in via ``overrides``.
    base = {
        "extract": {"temperature": 0.1},
        "distill": {"temperature": 0.1},
        "judge": {"temperature": 0.0},
        "generate": {"temperature": 0.0},
    }
    for role, kwargs in (overrides or {}).items():
        base[role] = {**base[role], **kwargs}
    return {
        r: RoleConfig(
            endpoint="http://localhost:8080/v1",
            model="qwen3-0.6b",
            max_tokens=kwargs.pop("max_tokens", 1000),
            extra_body=NOTHINK,
            **kwargs,
        )
        for r, kwargs in base.items()
    }


def run(
    config_name: str,
    organizers: list[str],
    memory_types: tuple[str, ...],
    sample,
    max_sessions,
    limit,
    embedder,
    k: int | dict = 10,
    keyword_queries: bool = False,
    role_overrides: dict[str, dict] | None = None,
    slot_overrides: dict[str, str] | None = None,
    lexical_types: tuple[str, ...] = ("episodic",),
    judge: bool = False,
    recipe: SearchRecipe | None = None,
    page_recall_cap: int | None = None,
    memoryos_lineage: str = "pypi",
    memmachine_expand_context: int | None = None,
    memmachine_context_limit: int | None = None,
    query_strategy: str | None = None,
    agent_limit: int = 20,
) -> dict:
    """Run one config end-to-end: build a fresh `AgenticMemory`, ingest
    `sample` (flushing tail buffers and calling `consolidate()`), answer the
    selected questions, then write `results/locomo-conv0-<config_name>.json`
    and return the same result dict. Always closes the memory instance, even
    on failure.

    A `recipe` (Zep only, `organizers/zep_graph/search.py`) supplies the whole
    read path at once — memory types, which of them get BM25 and BFS channels,
    the reranker and its parameters, and `k`. It overrides `memory_types`,
    `k` and `lexical_types`, because upstream ships those four as ONE named
    `SearchConfig` and splitting them across call sites is how a run ends up
    being a combination no upstream recipe has. The recipe name is stamped into
    the result so a saved run says which read path produced it.

    `page_recall_cap` and `memoryos_lineage` are MemoryOS's read-path
    LINEAGE knobs, split out of `AgmemConfig`/`locomo.answer` defaults for the
    reason `MEMORYOS_PRESETS` exists: the library and the harness that produced
    the paper's numbers disagree on both, and both were pinned at the harness's
    value for every config, so the pypi-lineage config was reading with the eval
    lineage's queue size and its full assistant-knowledge dump."""
    # 0-arg factory callables (lambdas in ``known``) build a fresh organizer
    # instance per run() call — reusing one instance across configs/runs
    # would leak Nemori's message buffer and MemoryOS/A-Mem's episode-id
    # reverse-index state between them.
    organizers = [o() if callable(o) and not isinstance(o, str) else o for o in organizers]
    read_path: dict = {"lexical_types": lexical_types}
    overrides = dict(slot_overrides or {})
    if recipe is not None:
        read_path = recipe.config_kwargs()
        overrides.update(read_path.pop("overrides", {}))
        memory_types, k = recipe.memory_types, recipe.limit
    # After the recipe branch, which replaces `read_path` wholesale: an explicit
    # per-config value must not be silently dropped by a recipe that has no
    # opinion about it.
    if page_recall_cap is not None:
        read_path["page_recall_cap"] = page_recall_cap
    # MemMachine's search recipe, split out for the same reason: `AgmemConfig`
    # holds upstream's LIBRARY defaults (expand 0 / limit 20) while its LoCoMo
    # harness runs 3 / 30, and a default carrying the harness's numbers would
    # make every other config a mislabeled eval-lineage run.
    if memmachine_expand_context is not None:
        read_path["memmachine_expand_context"] = memmachine_expand_context
    if memmachine_context_limit is not None:
        read_path["memmachine_context_limit"] = memmachine_context_limit
    # The read-side control policy is config, not a call argument: that is what
    # makes it reachable from the MCP server and LongMemEval too, instead of
    # only from this script's path through `locomo.answer`.
    if query_strategy is not None:
        read_path["query_strategy"] = query_strategy
        read_path["query_strategy_limit"] = agent_limit
    mem = AgenticMemory(
        namespace=f"locomo-c0-{config_name}",
        organizers=organizers,
        embedder=embedder,
        config=AgmemConfig(
            llm_roles=make_roles(role_overrides),
            use_guided_json=False,
            overrides=overrides,
            **read_path,
        ),
    )
    try:
        t0 = time.perf_counter()
        n_turns = locomo.ingest(mem, sample, max_sessions=max_sessions)  # ingest() flushes
        # Deferred management pass (spec §1.4): call the Organizer.consolidate
        # contract unconditionally right after ingest()'s flush settles the
        # tail buffer. Organizers without a consolidate hook are a no-op
        # returning 0 (base default), so this is contract-based rather than
        # gated on Nemori's private _consolidator attribute (review M4) — it
        # also covers future consolidate users (ACE refine, Zep refresh).
        mem.consolidate()
        ingest_s = time.perf_counter() - t0
        # Capture before eval: on_retrieval feedback hooks can mutate memory
        # during evaluate(), and the storage question is about what ingest built.
        OUT.mkdir(exist_ok=True)
        capacity = capture_memory(mem, OUT / f"locomo-conv0-{config_name}.memory.jsonl")

        questions = locomo.select_questions(sample, max_sessions=max_sessions, limit=limit)
        t0 = time.perf_counter()
        res = locomo.evaluate(
            mem,
            questions,
            k=k,
            memory_types=memory_types,
            keyword_queries=keyword_queries,
            judge=judge,
            memoryos_lineage=memoryos_lineage,
            progress=lambda i, n: (
                print(f"[{config_name}] {i}/{n}", flush=True) if i % 20 == 0 else None
            ),
        )
        eval_s = time.perf_counter() - t0

        result = {
            "config": config_name,
            # organizers is now resolved to instances (factory callables are
            # consumed above) — record names for JSON-safety, not objects.
            "organizers": [
                o if isinstance(o, str) else getattr(o, "name", type(o).__name__)
                for o in organizers
            ],
            "memory_types": list(memory_types),
            # Which read path produced these numbers. Zep ships a menu of them
            # (search_config_recipes.py) and the paper measures one, so a result
            # without this field cannot be compared to a published figure.
            "search_recipe": recipe.name if recipe is not None else None,
            # MemoryOS's read path splits by lineage the same way its write path
            # does, so a stored run has to say which one it used — otherwise
            # `memoryos` and `memoryos_eval` results are indistinguishable on the
            # axis that actually differs between them.
            "page_recall_cap": read_path.get("page_recall_cap", AgmemConfig().page_recall_cap),
            "memoryos_lineage": memoryos_lineage,
            "n_turns": n_turns,
            "ingest_seconds": round(ingest_s, 1),
            "eval_seconds": round(eval_s, 1),
            "overall": res["overall"],
            "by_category": res["by_category"],
            "llm_budget": mem.budget.summary(),
            "memory_capacity": capacity,
            "structured_drops": dict(mem.structured.drops) if mem.structured else {},
            # docs/05 §3's six fields come from run_stamp; this call adds only the
            # condition specific to this script. The inline dict this replaced was
            # missing four of the six (profile, commit, judge, runs), so no result
            # in results/ can say which profile or commit produced it.
            "stamp": bench_stamp.run_stamp(
                mem,
                model="qwen3-0.6b",
                judge=judge,
                runs=1,
                dataset="locomo10 conv0",
                dataset_path=DATA,
                k=k,
                budget_tokens=6000,
                keyword_queries=keyword_queries,
                role_overrides=role_overrides,
                max_sessions=max_sessions,
                n_questions=len(questions),
                organizer_detail=[
                    {
                        "name": getattr(o, "name", type(o).__name__),
                        "fidelity": getattr(o, "fidelity", None),
                        "params": getattr(o, "params", None),
                    }
                    for o in mem.organizers
                ],
            ),
            "records": res["records"],
        }
        OUT.mkdir(exist_ok=True)
        (OUT / f"locomo-conv0-{config_name}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False)
        )
        ov = res["overall"]
        jtxt = f" j={ov['j_score']}" if "j_score" in ov else ""
        print(
            f"[{config_name}] overall={ov}{jtxt} ingest={ingest_s:.0f}s eval={eval_s:.0f}s"
            f" items={capacity['counts']} derived={capacity['derived_item_count']}"
            f" derived_bytes={capacity['derived_bytes']}",
            flush=True,
        )
        return result
    finally:
        mem.close()


def main() -> None:
    """CLI entrypoint (see module docstring for the `--max-sessions`/`--limit`
    flags). Loads the first LoCoMo sample once, then runs every config named
    in `--configs` (default passthrough+amem) against it via `run()`,
    writing one results JSON per config; unknown config names raise
    `KeyError` from the `known` lookup."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--configs", nargs="*", default=["passthrough", "amem"])
    ap.add_argument("--judge", action="store_true", default=False)
    args = ap.parse_args()

    sample = locomo.load_locomo(DATA)[0]
    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small", device="cuda")
    # config = (organizers, memory_types, k, keyword_queries, role_overrides,
    #           slot_overrides).
    # amem/nemori are methodology-pure per the 2nd fidelity re-audit: upstream
    # evals retrieve only the organizer's own memory types (A-Mem notes-only
    # with LLM keyword queries; Nemori episodes k=10 / semantic m=2k=20). The
    # *_mixed variants keep the previous raw-episodic RAG channel for
    # ablation-style comparison — their numbers are NOT paper reproductions.
    # role_overrides = upstream write-path temps (round-4): A-Mem 0.7/0.7,
    # Nemori segmentation 0.2 + distill 0.7 (max_tokens 2000, upstream default).
    AMEM_TEMPS = {"extract": {"temperature": 0.7}, "distill": {"temperature": 0.7}}
    NEMORI_TEMPS = {
        "extract": {"temperature": 0.2},
        "distill": {"temperature": 0.7, "max_tokens": 2000},
    }
    # Lineage-faithful engines (docs/03 §5): A-Mem ran on ChromaDB -> our
    # cosine-fixed ChromaVectorStore; Nemori ran on Qdrant -> local-mode
    # QdrantVectorStore. Others use the profile default (sqlite-vec).
    AMEM_STORE = {"vector_store": "ChromaVectorStore"}
    # Nemori upstream = PostgreSQL(tsvector) + Qdrant dual — both real via
    # embedded builds (pgserver / qdrant local mode)
    NEMORI_STORE = {
        "vector_store": "QdrantVectorStore",
        "doc_store": "PostgresDocStore",
    }
    known = {
        "passthrough": (["passthrough"], ("episodic",), 10, False, None, None),
        "amem": (["amem"], ("notes",), 10, True, AMEM_TEMPS, AMEM_STORE),
        "nemori": (
            ["nemori"],
            ("episodes", "semantic"),
            {"episodes": 10, "semantic": 20},
            False,
            NEMORI_TEMPS,
            NEMORI_STORE,
        ),
        # Lifecycle-redesign fidelity/chained configs (spec §5 validation
        # table) — organizers are 0-arg factory callables so run() gets a
        # fresh instance per invocation (buffer/reverse-index isolation).
        "nemori_v4": (
            [lambda: NemoriOrganizer(fidelity="v4")],
            ("episodes", "semantic"),
            {"episodes": 10, "semantic": 20},
            False,
            NEMORI_TEMPS,
            NEMORI_STORE,
        ),
        "nemori_upstream": (
            [lambda: NemoriOrganizer(fidelity="upstream")],
            ("episodes", "semantic"),
            {"episodes": 10, "semantic": 20},
            False,
            NEMORI_TEMPS,
            NEMORI_STORE,
        ),
        # batch+merge use v4, integration stays inline-append but deferred
        # semantic_offline consolidation runs after ingest — inline vs
        # deferred integration ablation axis (spec §2.3 note).
        "nemori_mix": (
            [
                lambda: NemoriOrganizer(
                    fidelity="v4",
                    semantic_integration="append",
                    consolidation="semantic_offline",
                )
            ],
            ("episodes", "semantic"),
            {"episodes": 10, "semantic": 20},
            False,
            NEMORI_TEMPS,
            NEMORI_STORE,
        ),
        # --- experimental: cross-organizer chained compositions (not paper
        # reproductions — an upstream organizer's episodes feed a second
        # organizer via ChainedConsumer; docs/13 §5, experimental-split spec).
        "nemori_memoryos": (
            [
                lambda: NemoriOrganizer(fidelity="v1"),
                lambda: ChainedConsumer(MemoryOSOrganizer(), "episodes"),
            ],
            ("episodes", "semantic", "pages"),
            {"episodes": 10, "semantic": 20, "pages": 10},
            False,
            NEMORI_TEMPS,
            None,
        ),
        "nemori_amem": (
            [
                lambda: NemoriOrganizer(fidelity="v1"),
                lambda: ChainedConsumer(AMemOrganizer(), "episodes"),
            ],
            ("episodes", "semantic", "notes"),
            {"episodes": 10, "semantic": 20, "notes": 10},
            False,
            NEMORI_TEMPS,
            None,
        ),
        # --- Nemori v4 Table 7: A-MEM fed Nemori's distilled knowledge K
        # instead of raw messages (45-64% less storage, core +1.9%~+6.1%).
        # A-MEM is the system under test, so the read path is A-Mem's own
        # (notes-only + LLM keyword queries), identical to the `amem` config —
        # the ONLY difference is what the write path ingests. The two granularity
        # variants exist because the paper never says at what unit K arrives;
        # only one of them can land in the paper's storage band (chained.py).
        "nemori_amem_k": (
            [
                lambda: NemoriOrganizer(fidelity="v1"),
                lambda: ChainedConsumer(AMemOrganizer(), "semantic"),
            ],
            ("notes",),
            10,
            True,
            NEMORI_TEMPS,
            None,
        ),
        "nemori_amem_k_batched": (
            [
                lambda: NemoriOrganizer(fidelity="v1"),
                lambda: ChainedConsumer(AMemOrganizer(), "semantic", batch_key="episode_id"),
            ],
            ("notes",),
            10,
            True,
            NEMORI_TEMPS,
            None,
        ),
        # --- A-MAC admission gate in front of A-Mem (arXiv:2603.04549; audit in
        # docs/research/amac-admission-gate.md). The read path is byte-identical
        # to `amem` (notes-only + LLM keyword queries), so `amem` vs these two is
        # a clean write-path-only contrast. The gate is LLM-free by default, so a
        # rejected turn costs 0 calls where `amem` spends 2.
        # NOT a paper reproduction of A-MAC's Table 1: that measures admission
        # decision F1 against an oracle `is_referenced` label, not answer quality.
        # Per the audit §7 the published weights/threshold need re-tuning before
        # these numbers mean anything — run the pair, not just the first one.
        "amem_amac": (
            [lambda: AdmissionGated(AMemOrganizer(), AdmissionGate())],
            ("notes",),
            10,
            True,
            AMEM_TEMPS,
            AMEM_STORE,
        ),
        # Same gate with the release's substring keyword matching restored, which
        # is what its published recall 0.972 was produced under. Pairing it with
        # `amem_amac` tests on our own data whether that recall is an artifact of
        # the matching defect (audit §2 defect 2).
        "amem_amac_upstream": (
            [lambda: AdmissionGated(AMemOrganizer(), AdmissionGate(type_matching="substring"))],
            ("notes",),
            10,
            True,
            AMEM_TEMPS,
            AMEM_STORE,
        ),
        "amem_mixed": (
            ["amem"],
            ("episodic", "notes"),
            10,
            False,
            AMEM_TEMPS,
            AMEM_STORE,
        ),
        "nemori_mixed": (
            ["nemori"],
            ("episodic", "episodes", "semantic"),
            {"episodic": 10, "episodes": 10, "semantic": 20},
            False,
            NEMORI_TEMPS,
            NEMORI_STORE,
        ),
        # Two MemoryOS lineages, because the paper's LoCoMo numbers came from
        # the repo's `eval/` harness and not from the maintained `memoryos-pypi`
        # library — different heat weights, recency handling, keyword-overlap
        # formula, STM capacity and eviction policy (MEMORYOS_PRESETS). Running
        # only one of them cannot say whether a gap is the method or the
        # lineage, so both are configs rather than a constant.
        # Methodology-PURE, on the same rule as `amem`/`nemori`: upstream's QA
        # prompt is STM history + the retrieved MTM pages + profile/knowledge,
        # and it has NO search channel over raw messages. `episodic` was one,
        # and round-8 made it worse rather than better — once `MemoryOSPageRecall`
        # started serving verbatim pages and `recent_context()` started injecting
        # the resident STM, raw text reached the prompt through three routes
        # where upstream has two. The old wiring lives on as `memoryos_mixed`.
        # `lexical_types=()`: upstream's MTM search is dense-only (FAISS IP over
        # summary embeddings; its keyword term is dead code, `query_keywords =
        # set()`), so there is nothing for BM25 to mirror.
        "memoryos": (
            ["memoryos"],  # = fidelity="pypi"
            ("pages", "semantic"),
            10,
            False,
            None,
            None,
            (),
        ),
        "memoryos_eval": (
            [lambda: MemoryOSOrganizer(fidelity="eval")],
            ("pages", "semantic"),
            10,
            False,
            None,
            None,
            (),
            None,
            # The eval lineage's read path, which is NOT the library's: its
            # driver builds the retrieval queue at capacity 10 (pypi passes 7)
            # and dumps the whole assistant-knowledge store into every prompt
            # (pypi retrieves top-20). Both were on for every config before, so
            # `memoryos` was reading with this lineage's settings.
            {"page_recall_cap": 10, "memoryos_lineage": "eval"},
        ),
        # The docs/09 MemoryOS run, reconstructed: raw-episodic channel included,
        # STM force-drained at flush, no dialogue chain. Not a paper reproduction
        # — it exists so those stored numbers stay re-derivable after round-6 and
        # round-8 changed the organizer out from under them.
        "memoryos_mixed": (
            [lambda: MemoryOSOrganizer(dialogue_chain=False, flush_stm_on_drain=True)],
            ("episodic", "pages", "semantic"),
            10,
            False,
            None,
            None,
            ("episodic",),
            None,
            # docs/09 predates the two-stage page recall, so its `pages` hits
            # were segment SUMMARIES; 0 is what restores that.
            {"page_recall_cap": 0},
        ),
        # --- MemMachine (arXiv:2604.04853), the DEPLOYED-CODE lineage its own
        # LoCoMo harness runs: declarative backend, no short-term memory, so the
        # write path spends ZERO LLM calls — the first real point between
        # `passthrough` and `amem` on the extraction axis.
        # `k=150` is not a tuning choice: upstream over-fetches
        # `min(5 * limit, 200)` DERIVATIVES and dedups them down to `limit`
        # EPISODES (`declarative_memory.py::search_scored`), so the derivative-side
        # k and the episode-side limit are different numbers and both have to be
        # set. `lexical_types=()` because that vector search is dense-only.
        # NOT comparable to the paper's 0.9169 as it stands: that run used
        # text-embedding-3-small and a Cohere `rerank-v3-5` cross-encoder scoring
        # whole episode CONTEXTS, and with a profile whose reranker cannot score
        # text the contexts keep their fused order (MemMachineContextualize).
        "memmachine": (
            ["memmachine"],
            ("derivatives",),
            150,
            False,
            None,
            None,
            (),
            None,
            {"memmachine_expand_context": 3, "memmachine_context_limit": 30},
        ),
        # Same organizer, upstream's LIBRARY read defaults instead of its
        # harness's — the pair says how much of the read path is the method and
        # how much is the operating point, which is the question the paper's own
        # abstract raises ("retrieval-stage optimizations outperformed
        # ingestion-stage improvements").
        "memmachine_library": (
            ["memmachine"],
            ("derivatives",),
            100,
            False,
            None,
            None,
            (),
        ),
        # MemMachine's Retrieval Agent on the same store — the read-side control
        # policy (policies/retrieval.py) its own evaluation runs by default
        # (`agent_name="ToolSelectAgent"`). Same ingest as `memmachine`, so the
        # pair isolates the READ path: identical derivatives, identical
        # contextualization, different number of searches and query rewrites.
        # This is the one config here whose read path spends LLM calls per
        # question (1 routing + 1-3 inside the chosen strategy), which is also
        # the paper's own claim under test ("retrieval-stage optimizations
        # outperformed ingestion-stage improvements").
        "memmachine_agent": (
            ["memmachine"],
            ("derivatives",),
            150,
            False,
            None,
            None,
            (),
            None,
            {
                "memmachine_expand_context": 3,
                "memmachine_context_limit": 30,
                "query_strategy": "tool_select",
                # `agent_utils.process_question`'s `search_limit` default, which
                # is what its LoCoMo/LongMemEval drivers pass as QueryParam.limit.
                "agent_limit": 20,
            },
        ),
        # MemMachine's OTHER tier: semantic memory (= the paper's "profile").
        # This is where its LLM budget goes — one call per message per category
        # — so `memmachine` vs this pair is the cost contrast that the "zero
        # LLM calls" headline is only true of on the left. Upstream's own LoCoMo
        # config disables it (`semantic_memory.enabled: false`), so this config
        # is OUR extension of the comparison, not a reproduction of a published
        # number; `memmachine_full` runs both tiers as a deployed server would.
        "memmachine_profile": (
            [lambda: MemMachineProfileOrganizer()],
            ("semantic",),
            10,
            False,
            None,
            None,
            (),
        ),
        "memmachine_full": (
            [lambda: MemMachineOrganizer(), lambda: MemMachineProfileOrganizer()],
            ("derivatives", "semantic"),
            {"derivatives": 150, "semantic": 10},
            False,
            None,
            None,
            (),
            None,
            {"memmachine_expand_context": 3, "memmachine_context_limit": 30},
        ),
        # Zep read paths come from the recipe table, not from this tuple: the
        # paper describes three search functions and five rerankers and upstream
        # ships the combinations as named SearchConfigs, so the read path is a
        # named preset (organizers/zep_graph/search.py) exactly as Nemori's and
        # MemoryOS's write-path lineages are. `zep_graph` is the paper's own
        # operating point (§4.1: BGE-m3 reranking = cross-encoder, the only
        # family with a BFS channel); the others are ablations over it.
        # memory_types/k/lexical_types in these tuples are ignored — the recipe
        # supplies them (see run()).
        "zep_graph": (["zep_graph"], (), 10, False, None, None, (), "cross_encoder"),
        "zep_graph_rrf": (["zep_graph"], (), 10, False, None, None, (), "rrf"),
        "zep_graph_mmr": (["zep_graph"], (), 10, False, None, None, (), "mmr"),
        "zep_graph_mentions": (
            ["zep_graph"],
            (),
            10,
            False,
            None,
            None,
            (),
            "edge_episode_mentions",
        ),
    }
    for cfg in args.configs:
        entry = known[cfg]
        (
            organizers,
            memory_types,
            k,
            keyword_queries,
            role_overrides,
            slot_overrides,
        ) = entry[:6]
        lexical_types = entry[6] if len(entry) > 6 else ("episodic",)
        recipe = zep_search_recipe(entry[7]) if len(entry) > 7 and entry[7] else None
        # Trailing slot for the knobs only a couple of configs set, so the other
        # entries do not all grow a column of Nones.
        extras = entry[8] if len(entry) > 8 else {}
        run(
            cfg,
            organizers,
            memory_types,
            sample,
            args.max_sessions,
            args.limit,
            embedder,
            k=k,
            keyword_queries=keyword_queries,
            role_overrides=role_overrides,
            slot_overrides=slot_overrides,
            lexical_types=lexical_types,
            judge=args.judge,
            recipe=recipe,
            **extras,
        )


if __name__ == "__main__":
    main()
