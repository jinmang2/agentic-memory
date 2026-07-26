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
from agmem.config import AgmemConfig
from agmem.embed.st_embedder import SentenceTransformerEmbedder
from agmem.llm.client import RoleConfig
from agmem.organizers.admission import AdmissionGate
from agmem.organizers.amem import AMemOrganizer
from agmem.organizers.experimental import ChainedConsumer
from agmem.organizers.memoryos import MemoryOSOrganizer
from agmem.organizers.nemori import NemoriOrganizer

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
) -> dict:
    """Run one config end-to-end: build a fresh `AgenticMemory`, ingest
    `sample` (flushing tail buffers and calling `consolidate()`), answer the
    selected questions, then write `results/locomo-conv0-<config_name>.json`
    and return the same result dict. Always closes the memory instance, even
    on failure."""
    # 0-arg factory callables (lambdas in ``known``) build a fresh organizer
    # instance per run() call — reusing one instance across configs/runs
    # would leak Nemori's message buffer and MemoryOS/A-Mem's episode-id
    # reverse-index state between them.
    organizers = [o() if callable(o) and not isinstance(o, str) else o for o in organizers]
    mem = AgenticMemory(
        namespace=f"locomo-c0-{config_name}",
        organizers=organizers,
        embedder=embedder,
        config=AgmemConfig(
            llm_roles=make_roles(role_overrides),
            use_guided_json=False,
            overrides=slot_overrides or {},
            lexical_types=lexical_types,
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
            "n_turns": n_turns,
            "ingest_seconds": round(ingest_s, 1),
            "eval_seconds": round(eval_s, 1),
            "overall": res["overall"],
            "by_category": res["by_category"],
            "llm_budget": mem.budget.summary(),
            "memory_capacity": capacity,
            "structured_drops": dict(mem.structured.drops) if mem.structured else {},
            "stamp": {
                "embedder": mem.embedder.name,
                "model": "qwen3-0.6b",
                "k": k,
                "budget_tokens": 6000,
                "dataset": "locomo10 conv0",
                "keyword_queries": keyword_queries,
                "role_overrides": role_overrides,
                "vector_store": type(mem.vector_store).__name__,
                "max_sessions": max_sessions,
                "n_questions": len(questions),
                "organizer_detail": [
                    {
                        "name": getattr(o, "name", type(o).__name__),
                        "fidelity": getattr(o, "fidelity", None),
                        "params": getattr(o, "params", None),
                    }
                    for o in mem.organizers
                ],
            },
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
            [lambda: AMemOrganizer(admission=AdmissionGate())],
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
            [lambda: AMemOrganizer(admission=AdmissionGate(type_matching="substring"))],
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
        "memoryos": (
            ["memoryos"],
            ("episodic", "pages", "semantic"),
            10,
            False,
            None,
            None,
        ),
        # Zep hybrid read-path (round-5 ④): facts/entities get BM25+dense
        # fusion, plus GraphRecall edge expansion wired in the pipeline.
        "zep_graph": (
            ["zep_graph"],
            ("episodic", "facts", "entities"),
            10,
            False,
            None,
            None,
            ("episodic", "facts", "entities"),
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
        )


if __name__ == "__main__":
    main()
