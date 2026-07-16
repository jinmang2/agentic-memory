"""LoCoMo conv0 비교 실험: passthrough vs A-Mem(수정판), 로컬 Qwen3-0.6B.

발표용 1차 재현 (docs/06 Phase 2). 실행:
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

DATA = Path.home() / ".agmem/datasets/locomo10.json"
OUT = Path(__file__).resolve().parent.parent / "results"

NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def make_roles() -> dict[str, RoleConfig]:
    return {r: RoleConfig(endpoint="http://localhost:8080/v1", model="qwen3-0.6b",
                          temperature=0.1, max_tokens=300, extra_body=NOTHINK)
            for r in ("extract", "distill", "judge", "generate")}


def run(config_name: str, organizers: list[str], memory_types: tuple[str, ...],
        sample, max_sessions, limit, embedder, k: int | dict = 10,
        keyword_queries: bool = False) -> dict:
    mem = AgenticMemory(
        namespace=f"locomo-c0-{config_name}", organizers=organizers,
        embedder=embedder,
        config=AgmemConfig(llm_roles=make_roles(), use_guided_json=False),
    )
    try:
        t0 = time.perf_counter()
        n_turns = locomo.ingest(mem, sample, max_sessions=max_sessions)
        ingest_s = time.perf_counter() - t0

        questions = locomo.select_questions(sample, max_sessions=max_sessions,
                                            limit=limit)
        t0 = time.perf_counter()
        res = locomo.evaluate(
            mem, questions, k=k, memory_types=memory_types,
            keyword_queries=keyword_queries,
            progress=lambda i, n: print(f"[{config_name}] {i}/{n}", flush=True)
            if i % 20 == 0 else None,
        )
        eval_s = time.perf_counter() - t0

        result = {
            "config": config_name,
            "organizers": organizers,
            "memory_types": list(memory_types),
            "n_turns": n_turns,
            "ingest_seconds": round(ingest_s, 1),
            "eval_seconds": round(eval_s, 1),
            "overall": res["overall"],
            "by_category": res["by_category"],
            "llm_budget": mem.budget.summary(),
            "structured_drops": dict(mem.structured.drops) if mem.structured else {},
            "stamp": {"embedder": mem.embedder.name, "model": "qwen3-0.6b",
                      "k": k, "budget_tokens": 6000, "dataset": "locomo10 conv0",
                      "keyword_queries": keyword_queries,
                      "max_sessions": max_sessions, "n_questions": len(questions)},
            "records": res["records"],
        }
        OUT.mkdir(exist_ok=True)
        (OUT / f"locomo-conv0-{config_name}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[{config_name}] overall={res['overall']} "
              f"ingest={ingest_s:.0f}s eval={eval_s:.0f}s", flush=True)
        return result
    finally:
        mem.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--configs", nargs="*", default=["passthrough", "amem"])
    args = ap.parse_args()

    sample = locomo.load_locomo(DATA)[0]
    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small",
                                           device="cuda")
    # config = (organizers, memory_types, k, keyword_queries).
    # amem/nemori are methodology-pure per the 2nd fidelity re-audit: upstream
    # evals retrieve only the organizer's own memory types (A-Mem notes-only
    # with LLM keyword queries; Nemori episodes k=10 / semantic m=2k=20). The
    # *_mixed variants keep the previous raw-episodic RAG channel for
    # ablation-style comparison — their numbers are NOT paper reproductions.
    known = {
        "passthrough": (["passthrough"], ("episodic",), 10, False),
        "amem": (["amem"], ("notes",), 10, True),
        "nemori": (["nemori"], ("episodes", "semantic"),
                   {"episodes": 10, "semantic": 20}, False),
        "amem_mixed": (["amem"], ("episodic", "notes"), 10, False),
        "nemori_mixed": (["nemori"], ("episodic", "episodes", "semantic"),
                         {"episodic": 10, "episodes": 10, "semantic": 20}, False),
        "memoryos": (["memoryos"], ("episodic", "pages", "semantic"), 10, False),
        "zep_graph": (["zep_graph"], ("episodic", "facts", "entities"), 10, False),
    }
    for cfg in args.configs:
        organizers, memory_types, k, keyword_queries = known[cfg]
        run(cfg, organizers, memory_types,
            sample, args.max_sessions, args.limit, embedder, k=k,
            keyword_queries=keyword_queries)


if __name__ == "__main__":
    main()
