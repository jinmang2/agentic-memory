"""MemMachine's Retrieval Agent as a read-side control policy (policies/retrieval.py).

The strategies are driven through a fake ``search`` callable so the assertions
are about the POLICY — how many searches, with what text, what survives between
rounds — and not about any mechanism's recall.
"""

import re

import pytest
from helpers import StubLLM

from agmem.core.types import MemoryBundle, ScoredItem
from agmem.policies import (
    ChainOfQuery,
    DirectRetrieval,
    QueryContext,
    SplitQuery,
    ToolSelect,
)
from agmem.retrieval.steps import _DictItem


def item(item_id: str, content: str = "") -> ScoredItem:
    return ScoredItem(
        item=_DictItem({"id": item_id, "content": content or item_id}),
        memory_type="episodic",
        score=1.0,
        provenance=[item_id],
    )


class FakeSearch:
    """Records every query it is asked and serves a canned bundle per query."""

    def __init__(self, results: dict[str, list[ScoredItem]], default=None):
        self.results = results
        self.default = default if default is not None else []
        self.queries: list[str] = []

    def __call__(self, query: str) -> MemoryBundle:
        self.queries.append(query)
        return MemoryBundle(query=query, items=list(self.results.get(query, self.default)))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class TextReranker:
    """Cross-encoder stand-in: scores by how many query tokens the text holds.

    Word tokens rather than whitespace splits, because the query it is handed is
    upstream's unseparated concatenation ("beta?q1?\\nq2?") and a whitespace
    tokenizer would see one nonsense token."""

    needs_text = True

    def rerank(self, query_emb, candidates, vectors, k, texts=None, query="", **kwargs):
        wanted = _tokens(query)
        scored = [(key, len(wanted & _tokens((texts or {}).get(key, "")))) for key, _ in candidates]
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:k]


# ---- the leaf --------------------------------------------------------------


def test_direct_retrieval_searches_once_and_does_not_rerank():
    """``MemMachineAgent`` deliberately leaves reranking to its parent, so a
    query routed straight here comes back in raw search order."""
    search = FakeSearch({"who is luna": [item("b", "beagle"), item("a", "luna")]})
    ctx = QueryContext(search=search, reranker=TextReranker(), limit=1)
    result = DirectRetrieval().run("who is luna", ctx)
    assert search.queries == ["who is luna"]
    assert [s.item.data["id"] for s in result.bundle.items] == ["b", "a"]


# ---- split query -----------------------------------------------------------


def test_split_query_runs_one_search_per_sub_query():
    llm = StubLLM({"judge": [{"queries": ["what is A?", "what is B?"]}]})
    search = FakeSearch({"what is A?": [item("a")], "what is B?": [item("b")]})
    result = SplitQuery().run("what are A and B?", QueryContext(search=search, llm=llm))
    assert search.queries == ["what is A?", "what is B?"]
    assert [s.item.data["id"] for s in result.bundle.items] == ["a", "b"]
    assert result.metrics["memory_search_called"] == 2


def test_split_query_serves_a_shared_hit_twice_because_upstream_does():
    """``SplitQueryAgent`` concatenates sub-query results with no dedup, so one
    episode matched by two sub-queries reaches the prompt twice. Reproduced by
    default; ``dedupe=True`` is the debugged variant."""
    llm = StubLLM({"judge": [{"queries": ["q1?", "q2?"]}, {"queries": ["q1?", "q2?"]}]})
    search = FakeSearch({"q1?": [item("shared")], "q2?": [item("shared")]})
    served = SplitQuery().run("q?", QueryContext(search=search, llm=llm)).bundle.items
    assert [s.item.data["id"] for s in served] == ["shared", "shared"]

    deduped = SplitQuery(dedupe=True).run("q?", QueryContext(search=search, llm=llm)).bundle.items
    assert [s.item.data["id"] for s in deduped] == ["shared"]


def test_split_query_reranks_against_the_unseparated_concatenation():
    """Upstream's ``param.query += "\\n".join(sub_queries)`` glues the original
    query onto the first sub-query with no separator. That string is what the
    cross-encoder actually scored, so it is what we score with."""
    seen: dict[str, str] = {}

    class Recorder(TextReranker):
        def rerank(self, query_emb, candidates, vectors, k, texts=None, query="", **kwargs):
            seen["query"] = query
            return super().rerank(query_emb, candidates, vectors, k, texts=texts, query=query)

    llm = StubLLM({"judge": [{"queries": ["what is A?", "what is B?"]}]})
    search = FakeSearch({"what is A?": [item("a"), item("a2")], "what is B?": [item("b")]})
    SplitQuery().run("A and B?", QueryContext(search=search, llm=llm, reranker=Recorder(), limit=2))
    assert seen["query"] == "A and B?what is A?\nwhat is B?"


def test_a_single_sub_query_reranks_against_the_original_only():
    """`if len(sub_queries) > 1` — no concatenation when it did not split."""
    seen: dict[str, str] = {}

    class Recorder(TextReranker):
        def rerank(self, query_emb, candidates, vectors, k, texts=None, query="", **kwargs):
            seen["query"] = query
            return super().rerank(query_emb, candidates, vectors, k, texts=texts, query=query)

    llm = StubLLM({"judge": [{"queries": ["only?"]}]})
    search = FakeSearch({"only?": [item("a"), item("b"), item("c")]})
    SplitQuery().run("orig?", QueryContext(search=search, llm=llm, reranker=Recorder(), limit=2))
    assert seen["query"] == "orig?"


def test_a_dropped_split_call_falls_back_to_the_original_query():
    """`if len(sub_queries) == 0: sub_queries = [query]` — upstream's ONLY
    fallback (an LLM exception there propagates uncaught); our structured
    caller reports a failed call as a drop, which lands in that same
    empty-output branch."""
    llm = StubLLM({})  # no queued response -> drop
    search = FakeSearch({"orig?": [item("a")]})
    result = SplitQuery().run("orig?", QueryContext(search=search, llm=llm))
    assert search.queries == ["orig?"]
    assert result.metrics["queries"] == ["orig?"]


def test_the_split_has_no_code_cap_on_sub_queries():
    """Round-12 finding 5: the 1-6 range is the PROMPT's contract only —
    upstream takes every non-blank line of the response
    (`split_query_agent.py` L176-182) and caps nothing, so eight sub-queries
    mean eight searches."""
    queries = [f"q{i}?" for i in range(8)]
    llm = StubLLM({"judge": [{"queries": queries}]})
    search = FakeSearch({}, default=[item("a")])
    result = SplitQuery().run("many?", QueryContext(search=search, llm=llm))
    assert search.queries == queries
    assert result.metrics["memory_search_called"] == 8


# ---- chain of query --------------------------------------------------------


def test_chain_of_query_stops_when_sufficient_and_confident():
    llm = StubLLM(
        {
            "judge": [
                {
                    "is_sufficient": True,
                    "evidence_indices": [0],
                    "new_query": "orig?",
                    "confidence_score": 0.9,
                }
            ]
        }
    )
    search = FakeSearch({"orig?": [item("a")]})
    result = ChainOfQuery().run("orig?", QueryContext(search=search, llm=llm))
    assert search.queries == ["orig?"]
    assert result.metrics["is_sufficient"] == [True]
    assert result.metrics["llm_calls"] == 1


def test_low_confidence_keeps_going_even_when_the_model_says_sufficient():
    """The floor is upstream's class default 0.8, and it is ANDed with the
    verdict — a confident-sounding "sufficient" at 0.5 does not stop the loop."""
    llm = StubLLM(
        {
            "judge": [
                {"is_sufficient": True, "new_query": "second?", "confidence_score": 0.5},
                {"is_sufficient": True, "new_query": "third?", "confidence_score": 0.95},
            ]
        }
    )
    search = FakeSearch({}, default=[item("a")])
    result = ChainOfQuery().run("orig?", QueryContext(search=search, llm=llm))
    assert search.queries == ["orig?", "second?"]
    assert result.metrics["confidence_scores"] == [0.5, 0.95]


def test_a_repeated_rewrite_ends_the_loop():
    """`if curr_query.query in used_query ... break` — the model re-proposing a
    query it already tried is upstream's "it has nothing left" signal."""
    llm = StubLLM(
        {
            "judge": [
                {"is_sufficient": False, "new_query": "orig?", "confidence_score": 0.1},
                {"is_sufficient": False, "new_query": "never used", "confidence_score": 0.1},
            ]
        }
    )
    search = FakeSearch({}, default=[item("a")])
    ChainOfQuery().run("orig?", QueryContext(search=search, llm=llm))
    assert search.queries == ["orig?"]  # round 2 would repeat "orig?" -> stop


def test_only_promoted_evidence_survives_an_earlier_round():
    """The finding that makes ``evidence_indices`` load-bearing: upstream's
    final set is ``evidence | LAST round's hits``, so a first-round hit the
    model did not cite is gone by the time the answer prompt is built."""
    llm = StubLLM(
        {
            "judge": [
                {
                    "is_sufficient": False,
                    "evidence_indices": [0],  # keeps "keep", drops "forgotten"
                    "new_query": "second?",
                    "confidence_score": 0.1,
                },
                {"is_sufficient": True, "new_query": "second?", "confidence_score": 0.9},
            ]
        }
    )
    search = FakeSearch(
        {
            "orig?": [item("keep"), item("forgotten")],
            "second?": [item("fresh")],
        }
    )
    result = ChainOfQuery().run("orig?", QueryContext(search=search, llm=llm))
    ids = {s.item.data["id"] for s in result.bundle.items}
    assert ids == {"keep", "fresh"}


def test_the_sufficiency_pool_is_presented_chronologically():
    """Round-12 finding 6: upstream sorts union(retrieved, evidence) by
    `(created_at is None, created_at)` before assigning `[idx]` labels
    (`coq_agent.py` L206-211) — the sufficiency model reads documents in time
    order, unstamped ones last. Ties keep our deterministic merge order
    (upstream's set iteration leaves them unpinned)."""

    def stamped(item_id, content, created_at=None):
        data = {"id": item_id, "content": content}
        if created_at:
            data["created_at"] = created_at
        return ScoredItem(
            item=_DictItem(data), memory_type="episodic", score=1.0, provenance=[item_id]
        )

    llm = StubLLM(
        {
            "judge": [
                {"is_sufficient": True, "new_query": "orig?", "confidence_score": 0.9},
            ]
        }
    )
    search = FakeSearch(
        {
            "orig?": [
                stamped("late", "late-doc", "2023-06-01T10:00:00"),
                stamped("early", "early-doc", "2022-01-01T10:00:00"),
                stamped("unstamped", "unstamped-doc"),
            ]
        }
    )
    ChainOfQuery().run("orig?", QueryContext(search=search, llm=llm))
    prompt = llm.calls[0][1]
    assert (
        prompt.index("[0] early-doc")
        < prompt.index("[1] late-doc")
        < prompt.index("[2] unstamped-doc")
    )


def test_max_attempts_bounds_the_loop():
    llm = StubLLM(
        {
            "judge": [
                {"is_sufficient": False, "new_query": f"q{i}?", "confidence_score": 0.1}
                for i in range(9)
            ]
        }
    )
    search = FakeSearch({}, default=[item("a")])
    result = ChainOfQuery(max_attempts=3).run("orig?", QueryContext(search=search, llm=llm))
    assert len(search.queries) == 3
    assert result.metrics["llm_calls"] == 3


# ---- tool select -----------------------------------------------------------


def test_tool_select_routes_by_substring_and_defaults_to_chain_of_query():
    """Upstream matches the model's answer by substring over the children in
    order, and an unmatched answer falls through to the configured default."""
    search = FakeSearch({}, default=[item("a")])
    llm = StubLLM({"judge": [{"tool": "MemMachineAgent"}]})
    result = ToolSelect().run("simple?", QueryContext(search=search, llm=llm))
    assert result.metrics["selected_tool"] == "MemMachineAgent"

    llm = StubLLM({"judge": [{"tool": "NONE"}, {"is_sufficient": True, "new_query": "simple?"}]})
    result = ToolSelect().run("simple?", QueryContext(search=search, llm=llm))
    assert result.metrics["selected_tool"] == "ChainOfQueryAgent"


def test_tool_select_requires_all_three_children():
    with pytest.raises(ValueError, match="tool select needs"):
        ToolSelect(children=[DirectRetrieval()])


# ---- the seam --------------------------------------------------------------


def test_every_strategy_degrades_explicitly_without_an_llm():
    """No LLM must mean "one plain search, and say so", never a silent no-op."""
    search = FakeSearch({"q?": [item("a")]})
    for strategy in (SplitQuery(), ChainOfQuery(), ToolSelect()):
        search.queries.clear()
        result = strategy.run("q?", QueryContext(search=search))
        assert search.queries == ["q?"]
        assert result.metrics["degraded"] == "no LLM configured"


def test_strategy_order_survives_bundle_rendering():
    """Upstream returns a list and renders it in order; our bundle sorts by
    score, so a strategy has to encode its order in the scores it stamps."""
    llm = StubLLM({"judge": [{"queries": ["q1?", "q2?"]}]})
    search = FakeSearch(
        {
            "q1?": [item("a1", "alpha one"), item("a2", "alpha two")],
            "q2?": [item("b", "beta")],
        }
    )
    # 3 candidates against limit 2, so the reranker really runs (the branch the
    # test above pins is the other side of that condition).
    ctx = QueryContext(search=search, llm=llm, reranker=TextReranker(), limit=2)
    rendered = SplitQuery().run("beta?", ctx).bundle.render()
    # the reranker prefers "beta", and that order has to survive the score sort
    assert rendered.index("beta") < rendered.index("alpha one")


def test_the_reranker_is_skipped_when_the_candidates_already_fit():
    """`if len(episodes) <= query.limit or self._reranker is None` — a strategy
    that under-fetches never pays the cross-encoder, which is why upstream's
    limit is also the leaf's search limit."""

    class ExplodingReranker:
        needs_text = True

        def rerank(self, *args, **kwargs):
            raise AssertionError("reranker must not run when the candidates fit")

    llm = StubLLM({"judge": [{"queries": ["q1?", "q2?"]}]})
    search = FakeSearch({"q1?": [item("a")], "q2?": [item("b")]})
    SplitQuery().run(
        "q?", QueryContext(search=search, llm=llm, reranker=ExplodingReranker(), limit=5)
    )


def test_every_read_entry_point_goes_through_one_seam():
    """The defect this adapter exists for: the first wiring assembled the
    ``QueryContext`` inside ``locomo.answer``, so of the three public read
    entry points only the benchmark could reach a policy. All three now call
    ``searcher_for``, and ``PlannedSearch`` has ``AgenticMemory.search``'s shape
    so none of them branches on whether one is attached."""
    import inspect

    from agmem.bench import locomo, longmemeval
    from agmem.mcp import server
    from agmem.memory import AgenticMemory
    from agmem.retrieval.planned import PlannedSearch

    for module in (locomo, longmemeval, server):
        assert "searcher_for" in inspect.getsource(module), module.__name__

    facade = inspect.signature(AgenticMemory.search).parameters
    planned = inspect.signature(PlannedSearch.search).parameters
    # `metrics` is the parity point: both sides report how the bundle was
    # obtained, so a caller needs no isinstance check to record it.
    assert "metrics" in facade and "metrics" in planned


def test_the_adapter_is_the_only_read_path_module_importing_policies():
    """Mirror of ``test_no_mechanism_imports_the_policies_package``: on the write
    side exactly one adapter (`organizers/gated.py`) knows policies exist, and
    the read side gets exactly one too."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "agmem" / "retrieval"
    offenders = [
        path.name
        for path in root.glob("*.py")
        if path.name != "planned.py" and "agmem.policies" in path.read_text()
    ]
    assert offenders == [], f"read-path modules importing policies: {offenders}"


def test_a_configured_strategy_reaches_search_without_a_call_site_change():
    """``AgmemConfig.query_strategy`` is the single switch. A caller that only
    ever calls ``.search(...)`` picks the policy up through ``searcher_for``."""
    from agmem import AgenticMemory
    from agmem.config import AgmemConfig
    from agmem.embed.fake import FakeEmbedder
    from agmem.retrieval.planned import PlannedSearch, searcher_for

    plain = AgenticMemory(namespace="t", organizers=["passthrough"], embedder=FakeEmbedder(dim=128))
    try:
        assert searcher_for(plain) is plain  # no policy -> the memory itself
    finally:
        plain.close()

    planned = AgenticMemory(
        namespace="t",
        organizers=["passthrough"],
        embedder=FakeEmbedder(dim=128),
        config=AgmemConfig(query_strategy="split_query", query_strategy_limit=7),
    )
    try:
        searcher = searcher_for(planned)
        assert isinstance(searcher, PlannedSearch)
        assert searcher.strategy.name == "SplitQueryAgent"
        assert searcher.limit == 7

        planned.add_message("Luna is a beagle.", "user")
        planned.structured = StubLLM({"judge": [{"queries": ["what breed?", "whose dog?"]}]})
        seen: list[str] = []
        original = planned.search

        def spy(query, **kwargs):
            seen.append(query)
            return original(query, **kwargs)

        planned.search = spy
        metrics: dict = {}
        searcher.search("what dog?", memory_types=("episodic",), metrics=metrics)
        assert seen == ["what breed?", "whose dog?"]
        assert metrics["agent"] == "SplitQueryAgent"
    finally:
        planned.close()


def test_an_unknown_strategy_name_fails_loudly():
    from agmem.retrieval.planned import searcher_for

    class FakeMemory:
        config = type("C", (), {"query_strategy": "nope", "query_strategy_limit": 20})()

    with pytest.raises(ValueError, match="unknown query_strategy"):
        searcher_for(FakeMemory())


def test_the_bench_records_which_strategy_answered_each_question():
    """``capture["agent"]`` is filled either way — one plain search reports
    itself as ``search``, a policy reports its own accounting. Artifact capture
    must not go blind depending on which path ran."""
    from agmem import AgenticMemory
    from agmem.bench.locomo import answer
    from agmem.config import AgmemConfig
    from agmem.embed.fake import FakeEmbedder
    from agmem.organizers.memmachine import MemMachineOrganizer

    class GenerateLLM:
        def chat(self, role, messages, **kwargs):
            return "stub answer"

    mem = AgenticMemory(
        namespace="t",
        organizers=[MemMachineOrganizer()],
        embedder=FakeEmbedder(dim=128),
        config=AgmemConfig(query_strategy="tool_select"),
    )
    try:
        mem.add_message("Luna is a beagle.", "user", meta={"speaker": "Caroline"})
        mem.llm = GenerateLLM()
        mem.structured = StubLLM(
            {"judge": [{"tool": "MemMachineAgent"}, {"tool": "MemMachineAgent"}]}
        )
        capture: dict = {}
        answer(mem, "what dog?", memory_types=("derivatives",), capture=capture)
        assert capture["agent"]["selected_tool"] == "MemMachineAgent"

        plain = capture_plain = {}
        answer(
            mem,
            "what dog?",
            memory_types=("derivatives",),
            capture=capture_plain,
            searcher=mem,  # explicit override wins over the configured policy
        )
        # The wall clock is stamped on every read (docs/05 `research`); it is a
        # measurement, not accounting, so it is checked apart from the rest.
        assert plain["agent"].pop("latency_s") >= 0.0
        assert plain["agent"] == {
            "agent": "search",
            "memory_search_called": 1,
            "queries": ["what dog?"],
        }
    finally:
        mem.close()


def test_policies_still_emit_no_ops_and_declare_no_memory_type():
    """The operational test for living in ``policies/`` at all."""
    for strategy in (DirectRetrieval(), SplitQuery(), ChainOfQuery(), ToolSelect()):
        assert not hasattr(strategy, "produces")
        assert not hasattr(strategy, "on_message")
