"""Benchmark harness and dataset loaders (LoCoMo, LongMemEval) for cross-methodology runs.

Only ``BenchRun`` is re-exported. The per-benchmark modules are imported by name
(``from agmem.bench import locomo``) because they are alternative entry points,
not one API: LoCoMo scores with string metrics over a shared per-conversation
memory, LongMemEval scores with a pinned LLM judge over one memory per question.
"""

from agmem.bench.harness import BenchRun

__all__ = ["BenchRun"]
