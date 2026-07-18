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
    """Running totals for one role; `errors` counts failed calls but they
    still count toward `calls` (a failed call still costs latency/attempts)."""

    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms_total: float = 0.0
    errors: int = 0

    @property
    def latency_ms_avg(self) -> float:
        """0.0 before any calls, never raises on the empty case."""
        return self.latency_ms_total / self.calls if self.calls else 0.0


@dataclass
class BudgetTracker:
    """Thread-safe per-role accumulator; one instance is normally shared by
    an `LLMClient` and updated on every `chat()` call, success or failure."""

    _stats: dict[str, RoleStats] = field(default_factory=lambda: defaultdict(RoleStats))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(
        self,
        role: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """Add one call's cost to `role`'s running totals; creates the
        `RoleStats` entry on first use. Call this even for failed requests
        (with `error=True`) so latency/attempt counts stay complete."""
        with self._lock:
            s = self._stats[role]
            s.calls += 1
            s.tokens_in += tokens_in
            s.tokens_out += tokens_out
            s.latency_ms_total += latency_ms
            if error:
                s.errors += 1

    def summary(self) -> dict[str, dict[str, float]]:
        """Point-in-time snapshot keyed by role, JSON-serializable (latency
        rounded to 1 decimal); roles with zero calls are simply absent."""
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
        """Sum of `calls` across every role tracked so far."""
        with self._lock:
            return sum(s.calls for s in self._stats.values())
