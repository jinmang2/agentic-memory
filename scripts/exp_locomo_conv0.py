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

DATA = Path.home() / ".agmem/datasets/locomo10.json"
OUT = Path(__file__).resolve().parent.parent / "results"

NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def make_roles(overrides: dict[str, dict] | None = None) -> dict[str, RoleConfig]:
    # Defaults: write-path roles 0.1; judge/generate 0.0 (Nemori answers at
    # t=0.0, ReasoningBank judges at t=0.0). max_tokens 1000 per audit A6:
    # 300 could truncate multi-neighbor evolution JSON -> parse failure ->
    # drop. Per-methodology upstream temps come in via ``overrides``.
    base = {"extract": {"temperature": 0.1}, "distill": {"temperature": 0.1},
            "judge": {"temperature": 0.0}, "generate": {"temperature": 0.0}}
    for role, kw in (overrides or {}).items():
        base[role] = {**base[role], **kw}
    return {r: RoleConfig(endpoint="http://localhost:8080/v1", model="qwen3-0.6b",
                          max_tokens=kw.pop("max_tokens", 1000),
                          extra_body=NOTHINK, **kw)
            for r, kw in base.items()}


def run(config_name: str, organizers: list[str], memory_types: tuple[str, ...],
        sample, max_sessions, limit, embedder, k: int | dict = 10,
        keyword_queries: bool = False,
        role_overrides: dict[str, dict] | None = None) -> dict:
    mem = AgenticMemory(
        namespace=f"locomo-c0-{config_name}", organizers=organizers,
        embedder=embedder,
        config=AgmemConfig(llm_roles=make_roles(role_overrides),
                           use_guided_json=False),
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
                      "role_overrides": role_overrides,
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
    # config = (organizers, memory_types, k, keyword_queries, role_overrides).
    # amem/nemori are methodology-pure per the 2nd fidelity re-audit: upstream
    # evals retrieve only the organizer's own memory types (A-Mem notes-only
    # with LLM keyword queries; Nemori episodes k=10 / semantic m=2k=20). The
    # *_mixed variants keep the previous raw-episodic RAG channel for
    # ablation-style comparison — their numbers are NOT paper reproductions.
    # role_overrides = upstream write-path temps (round-4): A-Mem 0.7/0.7,
    # Nemori segmentation 0.2 + distill 0.7 (max_tokens 2000, upstream default).
    AMEM_TEMPS = {"extract": {"temperature": 0.7}, "distill": {"temperature": 0.7}}
    NEMORI_TEMPS = {"extract": {"temperature": 0.2},
                    "distill": {"temperature": 0.7, "max_tokens": 2000}}
    known = {
        "passthrough": (["passthrough"], ("episodic",), 10, False, None),
        "amem": (["amem"], ("notes",), 10, True, AMEM_TEMPS),
        "nemori": (["nemori"], ("episodes", "semantic"),
                   {"episodes": 10, "semantic": 20}, False, NEMORI_TEMPS),
        "amem_mixed": (["amem"], ("episodic", "notes"), 10, False, AMEM_TEMPS),
        "nemori_mixed": (["nemori"], ("episodic", "episodes", "semantic"),
                         {"episodic": 10, "episodes": 10, "semantic": 20},
                         False, NEMORI_TEMPS),
        "memoryos": (["memoryos"], ("episodic", "pages", "semantic"), 10,
                     False, None),
        "zep_graph": (["zep_graph"], ("episodic", "facts", "entities"), 10,
                      False, None),
    }
    for cfg in args.configs:
        organizers, memory_types, k, keyword_queries, role_overrides = known[cfg]
        run(cfg, organizers, memory_types,
            sample, args.max_sessions, args.limit, embedder, k=k,
            keyword_queries=keyword_queries, role_overrides=role_overrides)


if __name__ == "__main__":
    main()
