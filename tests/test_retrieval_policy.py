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
    """`if len(sub_queries) == 0: sub_queries = [query]` — including when the
    call fails, which for us is a structured-output drop."""
    llm = StubLLM({})  # no queued response -> drop
    search = FakeSearch({"orig?": [item("a")]})
    result = SplitQuery().run("orig?", QueryContext(search=search, llm=llm))
    assert search.queries == ["orig?"]
    assert result.metrics["queries"] == ["orig?"]


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


def test_the_bench_seam_is_off_by_default_and_records_the_agent_when_on():
    """``locomo.answer(query_strategy=...)`` is the only place the policy meets a
    real memory. Default None must leave the existing one-search path exactly as
    it was — every stored result depends on that."""
    from agmem import AgenticMemory
    from agmem.bench.locomo import answer
    from agmem.embed.fake import FakeEmbedder
    from agmem.organizers.memmachine import MemMachineOrganizer

    class GenerateLLM:
        def chat(self, role, messages, **kwargs):
            return "stub answer"

    mem = AgenticMemory(
        namespace="t", organizers=[MemMachineOrganizer()], embedder=FakeEmbedder(dim=128)
    )
    try:
        mem.add_message("Luna is a beagle.", "user", meta={"speaker": "Caroline"})
        mem.llm = GenerateLLM()
        searches: list[str] = []
        original_search = mem.search

        def spy(query, **kwargs):
            searches.append(query)
            return original_search(query, **kwargs)

        mem.search = spy

        capture: dict = {}
        answer(mem, "what dog?", memory_types=("derivatives",), capture=capture)
        assert searches == ["what dog?"]
        assert "agent" not in capture

        searches.clear()
        mem.structured = StubLLM(
            {
                "judge": [
                    {"tool": "SplitQueryAgent"},
                    {"queries": ["what breed?", "whose dog?"]},
                ]
            }
        )
        capture = {}
        answer(
            mem,
            "what dog?",
            memory_types=("derivatives",),
            capture=capture,
            query_strategy=ToolSelect(),
        )
        assert searches == ["what breed?", "whose dog?"]
        assert capture["agent"]["selected_tool"] == "SplitQueryAgent"
    finally:
        mem.close()


def test_policies_still_emit_no_ops_and_declare_no_memory_type():
    """The operational test for living in ``policies/`` at all."""
    for strategy in (DirectRetrieval(), SplitQuery(), ChainOfQuery(), ToolSelect()):
        assert not hasattr(strategy, "produces")
        assert not hasattr(strategy, "on_message")
