"""MemMachine's Retrieval Agent — read-side control policies (arXiv:2604.04853).

The first members of ``policies/`` that govern *retrieve* rather than *store*.
They decide **how many searches to run and with what query text**, and they own
no memory type and emit no ``MemoryOp`` — the operational test this package is
keyed on. The host mechanism is whatever ``search`` callable they are handed;
nothing here knows a MemMachine derivative from an A-Mem note, which is what
makes "cross-cutting" true rather than aspirational (the same claim
``AdmissionGated`` makes on the write side).

Attachment is ``retrieval/planned.py::PlannedSearch``, the read-side mirror of
that wrapper — a strategy is never wired at a call site. See that module for
why: doing it inline once left the policy reachable from the benchmark only.

Upstream layout (``packages/server/src/memmachine_server/retrieval_agent/``,
read at commit ``18f1211``): an ``AgentToolBase`` tree, built by
``service_locator.create_retrieval_agent``:

    ToolSelectAgent ─┬─ SplitQueryAgent ── MemMachineAgent
                     ├─ ChainOfQueryAgent ── MemMachineAgent
                     └─ MemMachineAgent

- ``MemMachineAgent`` — one search, no LLM, no rerank (its parent reranks).
- ``SplitQueryAgent`` — 1 LLM call splits into 1-6 single-hop sub-queries, each
  searched, results CONCATENATED, then reranked against the concatenation.
- ``ChainOfQueryAgent`` — up to ``max_attempts`` rounds of
  search -> "is this sufficient? if not, what is the next query?" in ONE call,
  stopping when the model says sufficient AND its confidence clears a floor.
- ``ToolSelectAgent`` — 1 LLM call routes the query to exactly one of the three.

Three findings from reading the release, all reproduced or documented rather
than silently smoothed:

1. **``QueryPolicy`` is entirely dead.** Its six fields (``token_cost``,
   ``time_cost``, ``accuracy_score``, ``confidence_score``, ``max_attempts``,
   ``max_return_len``) are threaded through every ``do_query`` signature in the
   tree, and ``grep -rn 'policy\\.' retrieval_agent/`` returns **nothing** — no
   agent reads a single field. ``MemMachineAgent`` even opens with ``_ = policy``.
   The values that do matter (``max_attempts``, ``confidence_score``) come from
   ``extra_params``, which none of the three construction sites populates, so
   the effective settings are always the class defaults 3 and 0.8 — NOT the
   ``max_attempts=3, confidence_score=10`` the eval harness passes in the
   policy object. Same family as A-MAC's dead ``X_train`` and MemoryOS's dead
   keyword term: a knob that reads as configured and is not. We therefore take
   the two live values as constructor arguments and have no policy object.
2. **Sub-query results are concatenated without dedup** (``SplitQueryAgent``:
   ``result.extend(res)``), so one episode matched by two sub-queries is printed
   twice in the answer prompt when the total fits under the limit. That is the
   same duplicate-serving defect our own pipeline was fixed for (80bcb37), but
   here it is upstream behaviour: ``dedupe=False`` is the default and ``True``
   is the debugged variant, exactly as ``AdmissionGate(type_matching=...)``
   keeps A-MAC's substring defect reachable.
3. **The rerank query is a concatenation with no separator** —
   ``param.query += "\\n".join(sub_queries)`` and
   ``q.query = query.query + "\\n".join(used_query)`` glue the original query
   directly onto the first rewritten one ("...original?What was X?"). Reproduced
   verbatim; it is what the cross-encoder actually scored against.

Our two adaptations, both forced by the seam rather than chosen:

- **Order is carried in scores.** Upstream returns a plain ``list[Episode]`` and
  its harness renders it in order; our ``MemoryBundle.render`` sorts by score.
  A strategy therefore re-scores its final selection with descending synthetic
  scores so the order it decided survives rendering. Upstream's final
  chronological sort is the one thing that cannot survive that, and it is the
  same deviation ``MemMachineContextualize`` already documents.
- **The LLM calls go through the structured caller**, so upstream's free-text
  contracts (a tool name on one line; one sub-query per line) become small JSON
  schemas. The task prompts are verbatim; only the envelope changes. The
  chain-of-query prompt already demanded strict JSON, so that one is unchanged
  in substance.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agmem.core.types import MemoryBundle, ScoredItem

logger = logging.getLogger("agmem.policies.retrieval")

# `tool_select_agent.py::TOOL_SELECT_PROMPT`, abridged to its mechanism: the
# release's copy carries five calibration examples and a restatement of the
# output format, both of which are for a free-text single-line answer we do not
# use. The classification criteria (A/B/C and the hard mapping) are verbatim.
TOOL_SELECT_PROMPT = """You are a tool router. Your task is to select exactly ONE tool name from the provided list that best fits the user query. Do not call any tools. Use only the text in the query; do not assume external context or missing metadata.

GOAL
- Choose exactly one of: {coq}, {split_query}, {memory_retrieval}
- Output NONE only when the query type cannot be determined (e.g., empty/invalid/malformed).

Classify the query type using ONLY these criteria:

A) MULTI-HOP (dependency chain; later step depends on earlier result): the query
requires two or more dependent steps where you must first determine X, then use X
to determine Y. Signals: "then", "after", "using that", "based on that result",
relationship chains ("X's spouse's company"). Comparisons count as MULTI-HOP only
if they need derived attributes found first. If any explicit dependency chain
exists, classify as MULTI-HOP.

B) SINGLE-HOP WITH MULTIPLE ENTITIES: multiple entities answerable by separate
independent lookups and then combined, with no earlier result needed to form the
later lookup. Signals: "A and B", "for each of", "separately". Directly
look-up-able comparisons belong here.

C) SINGLE-HOP / DIRECT: one main subject, no splitting needed.

Deterministic tool selection:
- MULTI-HOP -> {coq}
- SINGLE-HOP WITH MULTIPLE ENTITIES -> {split_query}
- SINGLE-HOP / DIRECT -> {memory_retrieval}
- otherwise -> NONE

AVAILABLE TOOLS
{coq}: Chain of Query agent that can decompose complex multi-hop queries into multiple simple single-hop queries and search them step by step.
{split_query}: Split Query agent that can split a single-hop query with multiple entities/keywords into multiple single-hop queries and search them separately.
{memory_retrieval}: Memory Retrieval agent that searches the query without modifying it.

Query:
{query}

Return JSON: {{"tool": "<one tool name or NONE>"}}"""

# `split_query_agent.py::SPLIT_QUERY_PROMPT`, mechanism verbatim (the decision
# rule, the ban on derived wording, the constraint-attachment rule and the
# duplicate guardrail); the release's worked examples are dropped with the
# free-text output format they illustrate.
SPLIT_QUERY_PROMPT = """You are a search expert. Transform the input query into either multiple single-hop sub-queries (2-6), or the original query unchanged (1), following the rules below.

Query
{query}

1) Decide whether to split (default: do NOT split)
- Do NOT split if the answer can be retrieved from a single page/infobox/database
  record/field set for the same entity and timeframe.
- Split ONLY if you must retrieve >=2 distinct facts that are not co-located OR
  are for different entities and/or different timepoints/locations/contexts.
- Tie-breaker: when unsure, prefer NOT splitting.

2) Special cases
- Multi-constraint single-entity queries: keep as ONE query if the attributes are
  typically co-located in the same reference entry.
- List-style questions ("all/each/every"): keep as-is unless the query explicitly
  names 2-6 specific entities.

3) If splitting: produce single-hop fact-lookups only
- Each sub-query must be directly answerable by one fact lookup, no derived
  operations. Do NOT use derived wording ("compare", "difference", "between",
  "rate", "top", "average", "change", "percent", "rank", "versus",
  "more/less than"); rewrite those into pure fact retrievals.

4) Preserve intent: keep the same entities/aliases and the same constraints
(timeframe, location, context, units), attach paired constraints to every
relevant sub-query, and add no assumptions.

5) Pronouns: if a pronoun's referent is not stated in the query, first add exactly
one sub-query resolving it, then the needed fact lookups.

6) No two sub-queries may ask for the same attribute of the same entity under the
same timeframe/location/context.

Each sub-query must be a full question ending with "?".

Return JSON: {{"queries": ["...", "..."]}}"""

# `coq_agent.py::COMBINED_SUFFICIENCY_AND_REWRITE_PROMPT`. This one already
# specified a strict JSON object with exactly these four keys, so the schema
# below is upstream's contract rather than our envelope.
SUFFICIENCY_PROMPT = """You are a meticulous expert in retrieval-augmented question answering evaluation and query rewriting.

Task: Given (1) an original user query, (2) rewritten queries already tried, and (3) retrieved documents, decide whether the documents are sufficient to answer the original query directly, completely, and explicitly. If insufficient, generate the NEXT BEST rewritten subquery to retrieve the missing evidence. If sufficient, set the rewritten query to the original query.

Hard constraints:
- Use ONLY the provided retrieved documents for sufficiency judgment. Do NOT use
  external knowledge, browsing, assumptions, or plausibility.
- Do NOT invent new entities. Only use entity names/terms present in the retrieved
  documents and/or original query.
- evidence_indices are 0-based indices into the retrieved-document list.

Decision procedure:
1) Decompose the original query into every required component: key entities,
   required attributes, required relationships / multi-hop chains, constraints.
2) A document is relevant ONLY if it explicitly contains a required fact OR
   explicitly establishes an intermediate link in a required chain.
3) Set is_sufficient = true ONLY if the documents explicitly contain all facts
   needed for every component, every link of any required chain is explicitly
   supported, exact details are present, and any "how many / list all / compare"
   requirement is satisfiable from documents covering the complete scope.
   If uncertain at any point, choose false.
4) When is_sufficient = false, new_query must target the EARLIEST blocking hop,
   be as specific as the retrieved documents allow, ask for exactly the missing
   fact, and differ from every previously tried query. When true, new_query must
   equal the original query exactly.

confidence_score reflects certainty in the is_sufficient decision, not answer
correctness: 0.90-1.00 very clear, 0.60-0.89 moderate, 0.30-0.59 low (still err
insufficient), 0.00-0.29 empty/unreadable documents. If you chose false out of
uncertainty, keep confidence_score below 0.70.

**Original Query**
{original_query}

**Rewritten Queries Tried**
{used_query}

**Retrieved Documents**
{retrieved_episodes}

Return JSON: {{"is_sufficient": true/false, "evidence_indices": [0, 1],
"new_query": "...", "confidence_score": 0.0}}"""

TOOL_SCHEMA = {
    "type": "object",
    "properties": {"tool": {"type": "string"}},
    "required": ["tool"],
}

SPLIT_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}

SUFFICIENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "is_sufficient": {"type": "boolean"},
        "evidence_indices": {"type": "array", "items": {"type": "integer"}},
        "new_query": {"type": "string"},
        "confidence_score": {"type": "number"},
    },
    "required": ["is_sufficient", "new_query"],
}


@dataclass
class QueryContext:
    """What a read policy is allowed to touch.

    ``search`` is the host mechanism, already bound to its own ``k`` and memory
    types by the caller — a policy never chooses what a search means, only how
    many it runs and with which text. Keeping it a bare callable is what stops
    this package from importing the facade (``test_no_mechanism_imports_the_
    policies_package`` guards the other direction).

    ``limit`` is upstream's ``QueryParam.limit`` at the point where it is
    actually read: the rerank/truncate cap. Upstream passes the same number to
    the leaf search, which here lives inside the bound ``search`` callable.
    ``role`` is one role for the whole strategy tree, because upstream has one
    ``retrieval_agent.llm_model`` for all three agents."""

    search: Callable[[str], MemoryBundle]
    llm: Any | None = None  # StructuredCaller
    reranker: Any | None = None
    limit: int = 0
    role: str = "judge"


@dataclass
class QueryResult:
    """The selected items plus what the strategy spent getting them.

    ``metrics`` mirrors upstream's ``perf_metrics`` where the key means the same
    thing (``queries``, ``memory_search_called``, ``is_sufficient``,
    ``confidence_scores``, ``selected_tool``) — those fields are how its
    evaluation reports agent behaviour, so keeping the names makes the two
    directly comparable."""

    bundle: MemoryBundle
    metrics: dict[str, Any] = field(default_factory=dict)


def _item_text(scored: ScoredItem) -> str:
    """The text a reranker scores — what ``MemoryBundle.render`` would print for
    this item, which is upstream's ``episodes_to_string([episode])``."""
    item = scored.item
    return item.render() if hasattr(item, "render") else getattr(item, "content", str(item))


def _rescore(items: list[ScoredItem]) -> list[ScoredItem]:
    """Stamp descending scores so the strategy's chosen order survives
    ``MemoryBundle.render``'s score sort (module docstring, adaptation 1)."""
    total = len(items)
    return [
        ScoredItem(
            item=scored.item,
            memory_type=scored.memory_type,
            score=(total - rank) / total,
            provenance=scored.provenance,
        )
        for rank, scored in enumerate(items)
    ]


class QueryStrategy:
    """Base: decide how the host's ``search`` is driven for one query."""

    name = "base"

    def run(self, query: str, ctx: QueryContext) -> QueryResult:
        raise NotImplementedError

    # ---- shared machinery ---------------------------------------------------

    def _rerank(self, query: str, items: list[ScoredItem], ctx: QueryContext) -> list[ScoredItem]:
        """``AgentToolBase._do_rerank``.

        Its three branches are load-bearing and easy to lose: ``limit <= 0``
        reranks nothing, a candidate set already within the limit reranks
        nothing (so a strategy that under-fetches never pays the cross-encoder),
        and only the third branch scores. Upstream's final chronological sort is
        dropped here — see the module docstring."""
        if ctx.limit <= 0:
            return items
        if len(items) <= ctx.limit or not getattr(ctx.reranker, "needs_text", False):
            return items[: ctx.limit]
        # Positional keys, not item ids: the candidate list may legitimately
        # contain the same item twice (finding 2), and id keys would collapse
        # those into one entry and silently change the truncation.
        texts = {str(index): _item_text(scored) for index, scored in enumerate(items)}
        candidates = [(str(index), scored.score) for index, scored in enumerate(items)]
        ranked = ctx.reranker.rerank([], candidates, {}, ctx.limit, texts=texts, query=query)
        return [items[int(key)] for key, _ in ranked]

    def _fallback(self, query: str, ctx: QueryContext, reason: str) -> QueryResult:
        """Explicit degradation to a plain search, never a silent one."""
        logger.warning("%s: %s — falling back to direct retrieval", self.name, reason)
        result = DirectRetrieval().run(query, ctx)
        result.metrics["degraded"] = reason
        return result

    def _usable_llm(self, ctx: QueryContext) -> bool:
        return ctx.llm is not None and (
            not hasattr(ctx.llm, "client") or ctx.llm.client.has_role(ctx.role)
        )


class DirectRetrieval(QueryStrategy):
    """``MemMachineAgent``: one search with the query as written.

    No LLM and no rerank — upstream's leaf deliberately leaves reranking to
    whichever parent called it, so routing a query here through
    ``ToolSelect`` really does return the raw search order."""

    name = "MemMachineAgent"

    def run(self, query: str, ctx: QueryContext) -> QueryResult:
        bundle = ctx.search(query)
        return QueryResult(
            bundle=bundle,
            metrics={"agent": self.name, "memory_search_called": 1, "queries": [query]},
        )


class SplitQuery(QueryStrategy):
    """``SplitQueryAgent``: 1 LLM call -> up to 6 independent sub-queries.

    ``dedupe=False`` is upstream (finding 2): an episode matched by two
    sub-queries is served twice. ``True`` is the debugged variant, kept
    reachable so the pair can be compared rather than argued about."""

    name = "SplitQueryAgent"

    def __init__(self, dedupe: bool = False, max_queries: int = 6) -> None:
        self.dedupe = dedupe
        self.max_queries = max_queries

    def run(self, query: str, ctx: QueryContext) -> QueryResult:
        if not self._usable_llm(ctx):
            return self._fallback(query, ctx, "no LLM configured")
        verdict = ctx.llm.call(
            ctx.role,
            SPLIT_QUERY_PROMPT.format(query=query),
            SPLIT_SCHEMA,
            required_keys=("queries",),
        )
        sub_queries = [str(q).strip() for q in (verdict or {}).get("queries", []) if str(q).strip()]
        # Upstream falls back to the original query when the split yields
        # nothing (`if len(sub_queries) == 0`), including when the call fails.
        sub_queries = sub_queries[: self.max_queries] or [query]

        items: list[ScoredItem] = []
        seen: set[tuple[str, str | None]] = set()
        for sub_query in sub_queries:
            for scored in ctx.search(sub_query).items:
                if self.dedupe:
                    key = (scored.memory_type, _item_id(scored))
                    if key in seen:
                        continue
                    seen.add(key)
                items.append(scored)

        # Upstream concatenates with no separator between the original query and
        # the first sub-query (finding 3), and only when it really split.
        rerank_query = query + "\n".join(sub_queries) if len(sub_queries) > 1 else query
        selected = self._rerank(rerank_query, items, ctx)
        return QueryResult(
            bundle=MemoryBundle(query=query, items=_rescore(selected)),
            metrics={
                "agent": self.name,
                "queries": sub_queries,
                "memory_search_called": len(sub_queries),
                "llm_calls": 1,
            },
        )


class ChainOfQuery(QueryStrategy):
    """``ChainOfQueryAgent``: search, ask whether that was enough, repeat.

    One call per round does both jobs (sufficiency verdict + next query), which
    is why upstream's prompt is named "combined". The loop stops on any of:
    the model says sufficient AND clears ``confidence_floor``; the rewritten
    query repeats one already tried or is empty; ``max_attempts`` rounds.

    Both live constants are upstream's class defaults rather than anything its
    callers pass (finding 1): ``max_attempts=3`` and ``confidence_score=0.8``."""

    name = "ChainOfQueryAgent"

    def __init__(self, max_attempts: int = 3, confidence_floor: float = 0.8) -> None:
        self.max_attempts = max_attempts
        self.confidence_floor = confidence_floor

    def run(self, query: str, ctx: QueryContext) -> QueryResult:
        if not self._usable_llm(ctx):
            return self._fallback(query, ctx, "no LLM configured")

        metrics: dict[str, Any] = {
            "agent": self.name,
            "queries": [],
            "is_sufficient": [],
            "confidence_scores": [],
            "memory_search_called": 0,
            "llm_calls": 0,
        }
        used: list[str] = []
        evidence: dict[tuple[str, str | None], ScoredItem] = {}
        round_hits: dict[tuple[str, str | None], ScoredItem] = {}
        next_query = query

        for _ in range(self.max_attempts):
            if not next_query or next_query in used:
                break
            used.append(next_query)
            metrics["queries"].append(next_query)
            metrics["memory_search_called"] += 1
            # THIS round's hits only. Upstream passes `result` — the last
            # search's output — into `combined_check_and_rewrite`, and its
            # `final_episodes` is `evidence | retrieved_episodes` over exactly
            # that. So an earlier round's hit survives ONLY by being promoted to
            # evidence; the rest are dropped when the next round runs. That is
            # what makes `evidence_indices` load-bearing rather than telemetry,
            # and accumulating every round instead (the obvious "improvement")
            # would quietly turn the agent into a union-of-all-searches.
            round_hits = {}
            for scored in ctx.search(next_query).items:
                round_hits.setdefault((scored.memory_type, _item_id(scored)), scored)

            pool = list({**round_hits, **evidence}.values())
            context = "".join(f"[{index}] {_item_text(s)}\n" for index, s in enumerate(pool))
            verdict = ctx.llm.call(
                ctx.role,
                SUFFICIENCY_PROMPT.format(
                    original_query=query,
                    used_query="\n".join(used),
                    retrieved_episodes=context,
                ),
                SUFFICIENCY_SCHEMA,
                required_keys=("is_sufficient", "new_query"),
            )
            metrics["llm_calls"] += 1
            if verdict is None:
                # Upstream logs the parse failure, retries once, then proceeds
                # with an empty response — which means is_sufficient False and
                # new_query defaulting to the ORIGINAL query, so the next round
                # repeats a used query and the loop breaks.
                verdict = {}
            for index in verdict.get("evidence_indices") or []:
                if isinstance(index, int) and 0 <= index < len(pool):
                    scored = pool[index]
                    evidence[(scored.memory_type, _item_id(scored))] = scored

            metrics["is_sufficient"].append(bool(verdict.get("is_sufficient")))
            confidence = float(verdict.get("confidence_score") or 0.0)
            metrics["confidence_scores"].append(confidence)
            next_query = str(verdict.get("new_query") or query)
            if verdict.get("is_sufficient") and confidence >= self.confidence_floor:
                break

        # `evidence | last round's hits`, per the note above.
        items = list({**round_hits, **evidence}.values())
        selected = self._rerank(query + "\n".join(used), items, ctx)
        return QueryResult(
            bundle=MemoryBundle(query=query, items=_rescore(selected)), metrics=metrics
        )


class ToolSelect(QueryStrategy):
    """``ToolSelectAgent``: 1 LLM call routes to exactly one strategy.

    Matching is upstream's: the first child whose name appears as a SUBSTRING of
    the model's answer wins, in child order, and an unmatched answer falls back
    to ``default`` (its callers all pass ChainOfQuery). Substring matching over
    ordered children is not neutral — ``MemMachineAgent`` is a substring of
    nothing else, but a model that answers with a sentence naming two tools
    picks whichever comes first in the child list."""

    name = "ToolSelectAgent"

    def __init__(
        self,
        children: list[QueryStrategy] | None = None,
        default: str = "ChainOfQueryAgent",
    ) -> None:
        # Upstream's child order: split, coq, memory.
        self.children = children or [SplitQuery(), ChainOfQuery(), DirectRetrieval()]
        self.default = default
        names = {child.name for child in self.children}
        required = {"SplitQueryAgent", "ChainOfQueryAgent", "MemMachineAgent"}
        if not required <= names:
            raise ValueError(f"tool select needs {sorted(required)}, got {sorted(names)}")

    def run(self, query: str, ctx: QueryContext) -> QueryResult:
        if not self._usable_llm(ctx):
            return self._fallback(query, ctx, "no LLM configured")
        names = {child.name: child for child in self.children}
        verdict = ctx.llm.call(
            ctx.role,
            TOOL_SELECT_PROMPT.format(
                query=query,
                coq="ChainOfQueryAgent",
                split_query="SplitQueryAgent",
                memory_retrieval="MemMachineAgent",
            ),
            TOOL_SCHEMA,
            required_keys=("tool",),
        )
        answer = str((verdict or {}).get("tool") or "")
        chosen = next((child for child in self.children if child.name in answer), None)
        if chosen is None:
            logger.warning("tool select: %r matched no tool — using %s", answer, self.default)
            chosen = names[self.default]
        result = chosen.run(query, ctx)
        result.metrics["selected_tool"] = chosen.name
        result.metrics["tool_select_llm_calls"] = 1
        return result


def _item_id(scored: ScoredItem) -> str | None:
    """Id of a scored item, whichever shape it has (same accessor the retrieval
    pipeline uses for its own ``(memory_type, id)`` dedup)."""
    return getattr(scored.item, "id", None) or getattr(scored.item, "data", {}).get("id")


STRATEGIES: dict[str, type[QueryStrategy]] = {
    "direct": DirectRetrieval,
    "split_query": SplitQuery,
    "chain_of_query": ChainOfQuery,
    "tool_select": ToolSelect,
}
