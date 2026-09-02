"""Apply a read-side control policy to a memory's search — the read-path adapter.

The mirror of ``organizers/gated.py`` on the other half of the lifecycle, and it
exists for the same reason that one does. ``policies/`` claims its members are
cross-cutting; a policy reachable only by hand-assembling its context inside one
caller is not. The first wiring of ``policies/retrieval`` did exactly that —
``bench/locomo.py`` built the ``QueryContext`` inline — so of the three public
read entry points (LoCoMo QA, LongMemEval QA, the MCP ``search_memory`` tool)
the policy was reachable from **one**, and it was the benchmark. This module is
the missing seam:

    PlannedSearch(memory, ToolSelect())      ~     AdmissionGated(organizer, gate)

``PlannedSearch`` deliberately has the same ``search(query, ...) -> MemoryBundle``
shape as ``AgenticMemory``, so anything holding "something searchable" keeps
working when a policy is attached — no call site branches on whether one is
active. ``searcher_for`` is the config-driven form of that: it returns the
memory itself when no strategy is configured, so every entry point can call it
unconditionally and a run turns the agent on by config alone.

Layering, which the two ``retrieval`` names otherwise blur:

    caller
      └── PlannedSearch          how MANY searches, with what query text  (policy)
            └── AgenticMemory.search      what ONE search means           (facade)
                  └── RetrievalPipeline   channels -> RRF -> rerank -> steps
                        └── stores

Only this module and ``organizers/gated.py`` import ``policies``; everything
else on either side stays unaware the package exists.

Note what the layering implies for the read->write feedback hooks: each inner
search is a real ``AgenticMemory.search`` call, so ``on_retrieval`` fires once
per inner search rather than once per question. That is the honest accounting —
MemoryOS's visit heat should count three searches as three — but it does mean a
question answered through ``ChainOfQuery`` bumps those counters more than the
same question answered directly, and any comparison across strategies has to say
so.
"""

from __future__ import annotations

from typing import Any

from agmem.core.types import MemoryBundle
from agmem.policies.retrieval import STRATEGIES, QueryContext, QueryResult, QueryStrategy


class PlannedSearch:
    """A searchable facade whose searches are planned by a read policy."""

    def __init__(
        self,
        memory: Any,
        strategy: QueryStrategy,
        limit: int = 20,
        role: str = "judge",
        **search_defaults: Any,
    ) -> None:
        """``limit`` is the policy's rerank/truncate cap (upstream's
        ``QueryParam.limit`` at the point it is actually read), NOT the ``k`` of
        the inner searches — those keep whatever ``k`` the caller passes, which
        for MemMachine is the derivative over-fetch rather than the episode
        limit. ``search_defaults`` are forwarded to every inner search, so a
        caller can bind ``memory_types``/``k`` once instead of per sub-query."""
        self.memory = memory
        self.strategy = strategy
        self.limit = limit
        self.role = role
        self.search_defaults = search_defaults

    def run(self, query: str, **overrides: Any) -> QueryResult:
        """Plan and execute, returning the bundle AND what it cost.

        Per-call rather than stored on the instance: the LoCoMo harness runs
        questions on a thread pool over one shared memory, so any "last result"
        attribute would be a race with a plausible-looking value."""
        from time import perf_counter

        kwargs = {**self.search_defaults, **overrides}
        started = perf_counter()
        result = self.strategy.run(
            query,
            QueryContext(
                search=lambda sub_query: self.memory.search(sub_query, **kwargs),
                llm=getattr(self.memory, "structured", None),
                reranker=getattr(self.memory, "reranker", None),
                limit=self.limit,
                role=self.role,
            ),
        )
        # The planned read's own wall clock, LLM planning calls included — the
        # inner searches each stamp theirs, but a caller comparing read paths
        # needs the whole thing.
        result.metrics["latency_s"] = perf_counter() - started
        return result

    def search(
        self, query: str, *, metrics: dict[str, Any] | None = None, **overrides: Any
    ) -> MemoryBundle:
        """``AgenticMemory.search``'s shape, so callers need no branch.

        ``metrics`` is the same optional out-parameter the facade's ``search``
        takes; here it is filled with the strategy's own accounting (which tool
        ran, how many searches, how many LLM calls)."""
        result = self.run(query, **overrides)
        if metrics is not None:
            metrics.update(result.metrics)
        return result.bundle


def searcher_for(memory: Any) -> Any:
    """The thing to call ``.search`` on for this memory.

    Returns the memory unchanged when no read policy is configured, so an entry
    point calls this unconditionally and never learns whether one is active.
    ``AgmemConfig.query_strategy`` is therefore the single switch that reaches
    the benchmarks, the MCP server and library callers alike — the property the
    first wiring lacked."""
    name = getattr(getattr(memory, "config", None), "query_strategy", None)
    if not name:
        return memory
    if name not in STRATEGIES:
        raise ValueError(f"unknown query_strategy {name!r} (known: {sorted(STRATEGIES)})")
    limit = getattr(memory.config, "query_strategy_limit", 20)
    return PlannedSearch(memory, STRATEGIES[name](), limit=limit)
