"""LLM cost accounting — calls/tokens/latency are first-class metrics.

Every benchmark table we reproduce (MemoryOS Table 3, Nemori Tables 3-4,
ACE Table 4) reports cost next to accuracy, so the tracker sits inside
the client and is always on.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class RoleStats:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms_total: float = 0.0
    errors: int = 0

    @property
    def latency_ms_avg(self) -> float:
        return self.latency_ms_total / self.calls if self.calls else 0.0


@dataclass
class BudgetTracker:
    _stats: dict[str, RoleStats] = field(default_factory=lambda: defaultdict(RoleStats))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, role: str, tokens_in: int, tokens_out: int,
               latency_ms: float, error: bool = False) -> None:
        with self._lock:
            s = self._stats[role]
            s.calls += 1
            s.tokens_in += tokens_in
            s.tokens_out += tokens_out
            s.latency_ms_total += latency_ms
            if error:
                s.errors += 1

    def summary(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {
                role: {
                    "calls": s.calls,
                    "tokens_in": s.tokens_in,
                    "tokens_out": s.tokens_out,
                    "latency_ms_avg": round(s.latency_ms_avg, 1),
                    "errors": s.errors,
                }
                for role, s in self._stats.items()
            }

    def total_calls(self) -> int:
        with self._lock:
            return sum(s.calls for s in self._stats.values())
