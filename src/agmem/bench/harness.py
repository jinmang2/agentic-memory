"""Benchmark harness: multi-run execution with reproducibility stamping.

Reproducibility discipline (docs/02 §4, the Zep-LoCoMo lesson): every run
is stamped with the full experiment condition, cost is recorded next to
accuracy, and multi-run mean±std is the reporting unit. Loaders: LoCoMo
(bench/locomo.py) and LongMemEval (bench/longmemeval.py).
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agmem.bench.stamp import run_stamp
from agmem.memory import AgenticMemory

# `_git_commit` moved to bench/stamp.py as `git_commit`: the result stamp is
# built in one place now (see that module for why three copies was the problem).


@dataclass
class BenchRun:
    """One benchmark configuration, executed ``runs`` times."""

    name: str
    make_memory: Callable[[], AgenticMemory]  # fresh memory per run
    runs: int = 3
    meta: dict[str, Any] = field(default_factory=dict)

    def execute(
        self,
        run_fn: Callable[[AgenticMemory], dict[str, float]],
        out_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """``run_fn`` ingests + evaluates on a fresh memory, returns metrics."""
        per_run: list[dict[str, float]] = []
        stamps: dict[str, Any] = {}
        for i in range(self.runs):
            mem = self.make_memory()
            try:
                t0 = time.perf_counter()
                metrics = run_fn(mem)
                metrics["wall_seconds"] = round(time.perf_counter() - t0, 2)
                metrics["llm_calls"] = mem.budget.total_calls()
                for role, s in mem.budget.summary().items():
                    metrics[f"tokens_{role}"] = s["tokens_in"] + s["tokens_out"]
                per_run.append(metrics)
                if i == 0:
                    stamps = run_stamp(
                        mem,
                        runs=self.runs,
                        structured_drops=(dict(mem.structured.drops) if mem.structured else {}),
                    )
            finally:
                mem.close()

        keys = sorted({k for m in per_run for k in m})
        aggregated = {
            k: {
                "mean": round(statistics.mean(vals), 4),
                "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            }
            for k in keys
            if (vals := [m[k] for m in per_run if k in m])
        }
        result = {
            "name": self.name,
            "runs": self.runs,
            "metrics": aggregated,
            "per_run": per_run,
            "stamp": {**stamps, **self.meta},
        }
        if out_dir:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{self.name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result
