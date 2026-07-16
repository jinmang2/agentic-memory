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
        sample, max_sessions, limit, embedder) -> dict:
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
            mem, questions, k=10, memory_types=memory_types,
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
                      "k": 10, "dataset": "locomo10 conv0",
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
    for cfg in args.configs:
        if cfg == "passthrough":
            run("passthrough", ["passthrough"], ("episodic",),
                sample, args.max_sessions, args.limit, embedder)
        elif cfg == "amem":
            run("amem", ["amem"], ("episodic", "notes"),
                sample, args.max_sessions, args.limit, embedder)


if __name__ == "__main__":
    main()
