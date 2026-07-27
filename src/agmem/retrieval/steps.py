"""Read-path post-hydration steps, one plugin per methodology.

These restore upstream read-path semantics the deep fidelity audit found
missing (docs/research/fidelity-deep-audit.md). They used to be an
``if memory_type == ...`` chain inside ``RetrievalPipeline.search``, which meant
their knobs — the A-Mem link cap and Nemori's source-attachment ``r``, both
flagged as deliberate upstream deviations — were constructor defaults no caller
could reach. As plugins they are registered per memory type and configured from
``AgmemConfig``, so those deviations are finally ablatable.

Keyed on the MEMORY TYPE, never on which organizer is active: items written
straight to a store must get the same treatment as organizer-written ones
(tests/test_pipeline_p0.py relies on this, and so does any offline replay).

The uniform contract is ``run(hits, ctx) -> list[ScoredItem]`` returning the
final list for that type, which covers all three shapes the original branches
had: append (link expansion, graph recall), replace (experiences), and
mutate-then-return (source attachment).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agmem.core.types import BITEMPORAL_TYPES, ScoredItem
from agmem.stores.base import DocStore


def is_servable(data: dict, memory_type: str) -> bool:
    """Whether a stored item may still be returned by retrieval.

    Two exclusions, and the reason this is one function rather than a check at
    each site: DELETE leaves a tombstone (``{"id": ..., "deleted": True}``) and
    INVALIDATE leaves the item intact but out of service unless its type is
    bi-temporal. Every read path that pulls items BY ID has to apply both, and
    the checks were previously copied per step — ``_hydrate`` and
    ``ExpandExperiences`` and ``GraphRecall`` each tested ``deleted`` on their
    own, and ``LinkExpansion`` tested neither. So an A-Mem note retired by
    ``ChainedConsumer`` (INVALIDATE on ``notes``) came straight back through its
    inbound link, and a deleted one came back as an empty ghost hit — the
    round-5 X1 failure the other three sites had already been fixed for.

    ``facts`` stay servable after invalidation on purpose: Zep renders them with
    their validity range so they read as historical rather than current
    (``_DictItem.render``). That is why this cannot be a bare ``deleted`` test."""
    if data.get("deleted"):
        return False
    return not (data.get("invalid_at") and memory_type not in BITEMPORAL_TYPES)


@dataclass
class ReadContext:
    """Read-only handles a step needs. ``bundle_ids`` are the ids already
    selected for earlier memory types in the same search, so an expansion step
    can avoid re-serving them."""

    doc_store: DocStore
    namespace: str | None
    graph_store: Any | None = None
    bundle_ids: set[str] = field(default_factory=set)
    # For steps that score candidates the recall channels never ranked —
    # MemoryOS's second stage scores the PAGES inside a matched segment, and
    # those pages are not items, so there is no ranking to reuse.
    query_embedding: list[float] | None = None
    vector_store: Any | None = None
    # Query keywords, when the CALLER extracted them. Not derived here: the only
    # methodology that uses them gets them from an LLM call
    # (MemoryOS eval's `llm_extract_keywords`), and the retrieval pipeline has no
    # LLM — the same reason A-Mem's keyword query rewrite lives in the bench.
    # Empty means "no keyword channel", which is also the pypi lineage's state.
    query_keywords: frozenset[str] = frozenset()
    # The query text and the pipeline's reranker, for the one step that scores
    # text the ranking channels never produced: MemMachine reranks assembled
    # episode CONTEXTS, not items, so their scores cannot come from fusion and
    # the reranker has to run a second time, inside the step.
    query: str = ""
    reranker: Any | None = None


class ReadStep:
    """Base: subclasses transform one memory type's hydrated hits."""

    def run(self, hits: list[ScoredItem], ctx: ReadContext) -> list[ScoredItem]:
        return hits


class LinkExpansion(ReadStep):
    """A-Mem 1-hop: pull linked neighbor notes of retrieved notes.

    Links are unidirectional as upstream. Cap semantics deviate: upstream caps
    PER HIT (agiresearch k per hit; WujiangXu k+1 via an off-by-one), so eval
    k=10 can pull ~100 link neighbors — WujiangXu #16/#21 show even upstream
    considers this ambiguous. We use one global cap (default 5); neighbors score
    just below their parent. Keep this deviation in result caveats when
    comparing multi-hop."""

    def __init__(self, cap: int = 5) -> None:
        self.cap = cap

    def run(self, hits: list[ScoredItem], ctx: ReadContext) -> list[ScoredItem]:
        seen = {s.item.data["id"] for s in hits}
        wanted: list[tuple[str, float]] = []
        for s in sorted(hits, key=lambda s: s.score, reverse=True):
            for linked_id in s.item.data.get("links", []):
                if linked_id not in seen and len(wanted) < self.cap:
                    seen.add(linked_id)
                    wanted.append((linked_id, s.score * 0.9))
        if not wanted:
            return hits
        by_id = dict(wanted)
        out = list(hits)
        for data in ctx.doc_store.get_items(list(by_id), "notes"):
            if not is_servable(data, "notes"):
                continue
            out.append(
                ScoredItem(
                    item=_DictItem(data),
                    memory_type="notes",
                    score=by_id.get(data.get("id"), 0.0),
                    provenance=data.get("source_episode_ids", []),
                )
            )
        return out


class AttachSources(ReadStep):
    """Nemori r=2: top-r episodes carry their raw source messages, rendered as
    ``role: content`` lines as upstream search.py does."""

    def __init__(self, top_r: int = 2) -> None:
        self.top_r = top_r

    def run(self, hits: list[ScoredItem], ctx: ReadContext) -> list[ScoredItem]:
        for s in sorted(hits, key=lambda s: s.score, reverse=True)[: self.top_r]:
            source_ids = s.item.data.get("source_episode_ids", [])
            if source_ids:
                episodes = ctx.doc_store.get_episodes(source_ids)
                s.item.data["_source_messages"] = [
                    f"{episode.role}: {episode.content}" for episode in episodes
                ]
        return hits


class ExpandExperiences(ReadStep):
    """ReasoningBank experience mode: an experience hit is REPLACED by its
    member strategy items (upstream injects the top-1 experience's items, never
    the record itself). An experience with no surviving items yields nothing —
    upstream's miss -> no-injection semantics."""

    def run(self, hits: list[ScoredItem], ctx: ReadContext) -> list[ScoredItem]:
        out: list[ScoredItem] = []
        for s in hits:
            ids = s.item.data.get("item_ids", [])
            for data in ctx.doc_store.get_items(ids, "strategies"):
                if not is_servable(data, "strategies"):
                    continue
                out.append(
                    ScoredItem(
                        item=_DictItem(data),
                        memory_type="strategies",
                        score=s.score,
                        provenance=data.get("source_episode_ids", []),
                    )
                )
        return out


class GraphRecall(ReadStep):
    """Zep GraphRecall (round-5 ④): retrieved entity nodes pull the ACTIVE edges
    within ``hops`` of them; those edges' fact items join the bundle (deduped
    against already-selected facts), scored just below their entity.

    Seeding matches upstream: when ``search()`` is given no explicit BFS origins
    it derives them from what the other channels already returned
    (``search.py``: ``origin_node_uuids = [node.uuid for result in
    search_results ...]``), which is exactly these hits.

    Two things do NOT match, and both are open decisions rather than oversights
    (2026-07-27 round-7). Upstream's BFS is a ranked CHANNEL fused with the
    others, where this appends at a flat score below the top hit; and BFS
    appears only in upstream's cross-encoder recipes — ``COMBINED_HYBRID_
    SEARCH_RRF``, which the zep config otherwise mirrors, has no BFS channel at
    all. So a config with both RRF fusion and this step is a blend of two
    recipes. ``hops`` exists so the alternative is reachable: upstream's
    ``MAX_SEARCH_DEPTH`` is 3, and ``graph_expansion_cap=0`` drops the step
    entirely for the pure-RRF reading.

    No-op when no graph store is wired.

    Every pulled fact gets the SAME ``base`` score, so the served order used to
    follow ``edges_for_nodes``'s row order, which the graph backends do not
    guarantee. The expanded facts are therefore emitted in a content-derived
    order (valid_at, content). Safe to change: docs/09 excludes zep_graph from
    measurement (skeleton grade), so there was no stable ordering to preserve.
    The A-Mem link expansion deliberately keeps its store-order emission — that
    one does back measured numbers."""

    def __init__(self, cap: int = 10, hops: int = 1) -> None:
        self.cap = cap
        self.hops = hops

    def run(self, hits: list[ScoredItem], ctx: ReadContext) -> list[ScoredItem]:
        if ctx.graph_store is None:
            return hits
        namespace = ctx.namespace or "main"
        seen = set(ctx.bundle_ids) | {s.item.data.get("id") for s in hits}
        node_ids = [n for n in (s.item.data.get("id") for s in hits) if n]
        if self.hops > 1:
            # Edges within `hops` of a seed are the edges incident to everything
            # reachable in hops - 1, so the walk stops one short of the cap.
            walked = list(node_ids)
            for node_id in node_ids:
                walked.extend(
                    str(n["id"])
                    for n in ctx.graph_store.neighbors(node_id, namespace, self.hops - 1)
                )
            node_ids = list(dict.fromkeys(walked))
        edges = ctx.graph_store.edges_for_nodes(node_ids, namespace)
        wanted: dict[str, float] = {}
        base = max((s.score for s in hits), default=0.0) * 0.9
        for e in edges:
            edge_id = e.get("id")
            if edge_id and edge_id not in seen and len(wanted) < self.cap:
                seen.add(edge_id)
                wanted[edge_id] = base
        out = list(hits)
        # Stable key must be content-derived, NOT the id: fact ids are random
        # uuids, so ordering by them would just move the nondeterminism instead
        # of removing it.
        pulled = sorted(
            ctx.doc_store.get_items(list(wanted), "facts"),
            key=lambda d: (str(d.get("valid_at") or ""), str(d.get("content") or "")),
        )
        for data in pulled:
            # `facts` is bi-temporal, so is_servable keeps invalidated edges here
            # on purpose — GraphRecall already asks the graph for ACTIVE edges,
            # and this only drops tombstones.
            if not is_servable(data, "facts"):
                continue
            out.append(
                ScoredItem(
                    item=_DictItem(data),
                    memory_type="facts",
                    score=wanted.get(data.get("id"), 0.0),
                    provenance=data.get("source_episode_ids", []),
                )
            )
        return out


class MemoryOSPageRecall(ReadStep):
    """MemoryOS's second retrieval stage: matched SEGMENTS expand into the
    verbatim PAGES they hold, and the pages are what reaches the prompt.

    Upstream retrieval is two stages (``mid_term.search_sessions`` +
    ``Retriever._retrieve_mid_term_context``): a session is matched on its
    summary embedding, then every page inside it is scored against the query,
    and a global heap keeps the best ``retrieval_queue_capacity`` PAGES across
    all matched sessions. The session summary itself is never injected — it is
    a matching key, not context. Serving segment summaries instead, which this
    read path did before, is a channel upstream does not have.

    Two deviations, both forced by where the data lives:

    - Upstream embeds the page as one string (``f"User: {u} Assiant: {a}"``,
      typo theirs) and dots it with the query. Pages are not items here, so
      there is no page vector; the score is the best of the page's member
      messages' own vectors, which is the same text scored in halves. That also
      keeps the step free of an embedder call per page per query.
    - The heat feedback (``N_visit``/LFU, which upstream bumps inside
      ``search_sessions`` for every session with a matched page) still works
      because the emitted page keeps its first message's id, and MemoryOS's
      ``on_retrieval`` maps that back to the segment through ``_unit_pages``.

    ``cap`` is upstream's ``retrieval_queue_capacity``; ``threshold`` its
    ``page_similarity_threshold``. A segment with no page over the threshold
    contributes nothing — as upstream's ``if matched_pages_in_session`` does.

    ``segment_threshold`` is the FIRST stage's gate, which both lineages apply
    and this step used to skip: a segment whose relevance misses it is not
    expanded at all (``if session_relevance_score >= segment_similarity_
    threshold``). Relevance is scored the way upstream scores it, not by reusing
    the fused rank — ``semantic_sim + keyword_alpha * s_topic_keywords``.

    That keyword term is where the copies part, and there are THREE of them, not
    two — reading only one is how it got called dead:

    - ``memoryos-pypi`` sets ``query_keywords = set()`` right above the loop
      ("Keywords extraction removed"), so its term is always 0, the score is
      plain cosine, and no LLM call happens at read time.
    - ``memoryos-chromadb`` derives query keywords by running the WRITE-side
      multi-topic summary prompt over the query (``extract_keywords_from_multi_
      summary`` -> ``gpt_generate_multi_summary``, no count cap) and overlaps
      them with **Jaccard**.
    - ``eval/`` — the harness that produced the paper's LoCoMo numbers — calls a
      dedicated extractor (``llm_extract_keywords``, at most three keywords) and
      overlaps them with the **containment mean**, the same formula its merge
      step uses.

    So ``keyword_similarity`` is a knob rather than a constant, and it has to
    agree with ``MemoryOSOrganizer._keyword_overlap``: one copy uses one formula
    on both sides, and pairing eval's read formula with pypi's merge formula is a
    combination no upstream has. ``test_the_read_and_merge_keyword_formulas_are
    _the_same_function`` pins that.

    Passing ``ctx.query_keywords`` selects the live version; empty keywords
    reproduce pypi exactly, so one step covers all three. (Upstream's third term,
    a recency factor, is dead in every copy: the eval file assigns
    ``lambda_t = 1`` with the decay line commented out, and the other two never
    apply one in search.)
    """

    def __init__(
        self,
        cap: int = 10,
        threshold: float = 0.1,
        segment_threshold: float = 0.1,
        keyword_alpha: float = 1.0,
        keyword_similarity: str = "containment_mean",
    ) -> None:
        self.cap = cap
        self.threshold = threshold
        self.segment_threshold = segment_threshold
        self.keyword_alpha = keyword_alpha
        self.keyword_similarity = keyword_similarity

    def _relevance(self, data: dict, cosine: float, query_keywords: frozenset[str]) -> float:
        """Upstream's ``session_relevance_score``: cosine plus the keyword
        overlap, which is 0 whenever either side has no keywords — upstream
        guards with ``if query_keywords and session_keywords``."""
        segment_keywords = {str(k).lower() for k in data.get("keywords") or []}
        if not query_keywords or not segment_keywords:
            return cosine
        overlap = len(query_keywords & segment_keywords)
        if not overlap:
            return cosine
        if self.keyword_similarity == "jaccard":
            s_topic = overlap / len(query_keywords | segment_keywords)
        else:
            s_topic = 0.5 * (overlap / len(query_keywords) + overlap / len(segment_keywords))
        return cosine + self.keyword_alpha * s_topic

    def run(self, hits: list[ScoredItem], ctx: ReadContext) -> list[ScoredItem]:
        if ctx.query_embedding is None or ctx.vector_store is None or not hits:
            return hits
        query = np.asarray(ctx.query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query)) or 1.0

        scored: list[tuple[float, str, dict, ScoredItem]] = []
        # An item with no page structure is not a MemoryOS segment — a `pages`
        # row written straight to the store, or one from before segments
        # recorded `page_units`. It passes through untouched, keeping this step
        # type-keyed rather than organizer-keyed (module docstring). "Has pages
        # but none matched" is the different case, and that one drops.
        passthrough = [s for s in hits if not s.item.data.get("page_units")]
        segment_vectors = ctx.vector_store.get(
            [sid for s in hits if (sid := s.item.data.get("id")) and s.item.data.get("page_units")]
        )
        for segment in hits:
            data = segment.item.data
            page_units = [
                [str(unit) for unit in page] for page in data.get("page_units", []) if page
            ]
            if not page_units:
                continue
            # Stage one: does this segment clear the relevance gate at all?
            # Scored from the segment's own summary vector, because that is what
            # upstream matches on and what the fused rank is NOT.
            if (
                self._relevance(
                    data,
                    _cosine(segment_vectors.get(data.get("id")), query, query_norm),
                    ctx.query_keywords,
                )
                < self.segment_threshold
            ):
                continue
            vectors = ctx.vector_store.get([unit for page in page_units for unit in page])
            for page in page_units:
                best = max(
                    (_cosine(vectors.get(unit_id), query, query_norm) for unit_id in page),
                    default=0.0,
                )
                if best >= self.threshold:
                    scored.append((best, page[0], {"units": page, "segment": data}, segment))

        if not scored:
            # Nothing cleared the page threshold. Upstream would return an empty
            # retrieval queue; returning the segments instead would reintroduce
            # the summary channel it does not have.
            return passthrough
        scored.sort(key=lambda row: (-row[0], row[1]))
        out: list[ScoredItem] = list(passthrough)
        for score, page_id, payload, segment in scored[: self.cap]:
            episodes = ctx.doc_store.get_episodes(payload["units"])
            if not episodes:
                continue
            data = {
                "id": page_id,
                "content": "\n".join(f"{_speaker(e)}: {e.content}" for e in episodes),
                "timestamp": episodes[0].timestamp.isoformat(),
                "source_episode_ids": [e.id for e in episodes],
            }
            # The chain summary rides with the page, which is where upstream puts
            # it ("Conversation chain overview:" per retrieved page); it is stored
            # on the segment only because the segment is our MTM item.
            if payload["segment"].get("meta_info"):
                data["meta_info"] = payload["segment"]["meta_info"]
            out.append(
                ScoredItem(
                    item=_DictItem(data),
                    memory_type=segment.memory_type,
                    score=score,
                    provenance=data["source_episode_ids"],
                )
            )
        return out


class MemMachineContextualize(ReadStep):
    """MemMachine's read path: derivative hits become CONTEXTS of raw episodes.

    Upstream ``declarative_memory.py::search_scored`` in four moves, all
    reproduced here:

    1. the matched derivatives are mapped to their source episodes, which
       become "nuclear episodes" (deduped, search order preserved);
    2. each nucleus is widened into a context by walking the episode store in
       ``(timestamp, uid)`` order — ``expand_context // 3`` episodes backward
       and the REST forward. The 1:2 split is a prior about conversations, not
       a symmetric window: the answer to a question usually follows it. Our
       taxonomy notes called this "±1~2 turns" until the code said otherwise;
    3. every context is rendered as one string and the RERANKER scores that
       string against the query — the ordering signal is the context, never the
       derivative that seeded it;
    4. contexts are merged best-first into at most ``limit`` distinct episodes.
       A context that would overflow the limit contributes its episodes nearest
       the nucleus first, where "nearest" is again asymmetric
       (``_weighted_index_proximity``: a forward neighbor at distance d counts
       as ``(d - 0.5) / 2``, a backward one as ``d``, so forward wins ties).

    Deviations, all forced by where our seams are:

    - **The ordered episode index comes from the derivatives**, since
      ``DocStore`` has no time-ordered episode listing (upstream's
      ``search_directional_nodes``) and adding one would touch every backend.
      Equivalent while every episode has at least one derivative, which holds
      for both presets; an episode ingested past this organizer is invisible to
      the expansion. Cost is one ``list_items`` per query.
    - **A reranker that cannot score text leaves the order alone.** Upstream's
      declarative backend REQUIRES a reranker and its eval config wires
      Cohere ``rerank-v3-5``; with ``NoopReranker`` (profile ``lite``) there is
      nothing to score contexts with, so the nuclei keep their fused order.
      That is a weaker read path than upstream's, not a different one, and it
      is visible in the config rather than hidden here.
    - **Served items are ``episodic``**, carrying the upstream line format
      (``[date at time] speaker: "content"``, ``episodes_to_string``) as their
      content, because that prefix is part of the QA prompt upstream builds and
      a bare episode loses it. Consequence worth knowing: under
      ``search(memory_types=None)`` the plain ``episodic`` pass runs first and
      wins the ``(memory_type, id)`` dedup, so the faithful call is
      ``search(memory_types=("derivatives",))``.
    - Upstream returns one globally chronological list; ``MemoryBundle.render``
      sorts by score. Episodes inside one context share their context's score,
      so each context stays chronological internally (stable sort) but contexts
      no longer interleave by time.
    """

    def __init__(self, expand_context: int = 0, limit: int = 20) -> None:
        """``expand_context``/``limit`` are upstream's ``query_memory``
        arguments. The defaults are ITS defaults (0 and 20); the published
        LoCoMo run uses 3 and 30 (``locomo_search.py``), which is a search
        recipe and therefore config, not a constant here — the same separation
        `MEMORYOS_PRESETS` exists to enforce."""
        self.expand_context = expand_context
        self.limit = limit

    def run(self, hits: list[ScoredItem], ctx: ReadContext) -> list[ScoredItem]:
        nuclei: dict[str, float] = {}
        for scored in hits:
            for episode_id in scored.item.data.get("source_episode_ids", []):
                nuclei.setdefault(episode_id, scored.score)
        if not nuclei:
            return []

        order = self._episode_order(ctx)
        position = {episode_id: index for index, episode_id in enumerate(order)}
        expand = min(max(0, self.expand_context), max(0, self.limit - 1))
        backward = expand // 3
        forward = expand - backward

        contexts: dict[str, list[str]] = {}
        for nucleus in nuclei:
            index = position.get(nucleus)
            if index is None:  # derivative whose episode is gone
                continue
            start = max(0, index - backward)
            contexts[nucleus] = order[start:index] + order[index : index + forward + 1]

        episodes = {
            episode.id: episode
            for episode in ctx.doc_store.get_episodes(
                list(dict.fromkeys(eid for context in contexts.values() for eid in context))
            )
        }
        rendered = {
            nucleus: "".join(_memmachine_line(episodes[eid]) for eid in context if eid in episodes)
            for nucleus, context in contexts.items()
        }
        ranked = self._rank(list(contexts), nuclei, rendered, ctx)

        # Upstream `_unify_scored_anchored_episode_contexts`, including its
        # `break` (not `continue`) once the limit is reached.
        selected: dict[str, float] = {}
        for nucleus, score in ranked:
            context = [eid for eid in contexts[nucleus] if eid in episodes]
            if len(selected) >= self.limit:
                break
            if len(selected) + len(context) <= self.limit:
                for episode_id in context:
                    selected.setdefault(episode_id, score)
                continue
            nuclear_index = context.index(nucleus) if nucleus in context else 0
            for episode_id in sorted(
                context,
                key=lambda eid: _weighted_index_proximity(context.index(eid), nuclear_index),
            ):
                if len(selected) >= self.limit:
                    break
                selected.setdefault(episode_id, score)

        return [
            ScoredItem(
                item=_DictItem(
                    {
                        "id": episode_id,
                        "content": _memmachine_line(episodes[episode_id]).rstrip("\n"),
                        "source_episode_ids": [episode_id],
                    }
                ),
                memory_type="episodic",
                score=score,
                provenance=[episode_id],
            )
            for episode_id, score in sorted(
                selected.items(), key=lambda pair: (position.get(pair[0], 0), pair[0])
            )
        ]

    def _episode_order(self, ctx: ReadContext) -> list[str]:
        """Episode ids in upstream's ``(timestamp, uid)`` order, derived from
        the derivative index (see the class docstring for why)."""
        stamps: dict[str, str] = {}
        for data in ctx.doc_store.list_items("derivatives", ctx.namespace):
            if not is_servable(data, "derivatives"):
                continue
            for episode_id in data.get("source_episode_ids", []):
                stamps.setdefault(episode_id, str(data.get("timestamp") or ""))
        return sorted(stamps, key=lambda eid: (stamps[eid], eid))

    def _rank(
        self,
        nuclei: list[str],
        scores: dict[str, float],
        rendered: dict[str, str],
        ctx: ReadContext,
    ) -> list[tuple[str, float]]:
        """Score whole contexts with the pipeline's reranker
        (``_score_episode_contexts`` -> ``reranker.score(query, contexts)``).

        The reranker runs a second time here, on text no ranking channel
        produced, so its usual (id, score) contract is fed pseudo-candidates
        keyed by nucleus. ``k`` is the full length: this call orders, it does
        not truncate — the limit is applied by the unification below, over
        episodes rather than contexts."""
        candidates = [(nucleus, scores[nucleus]) for nucleus in nuclei]
        if ctx.reranker is None or not getattr(ctx.reranker, "needs_text", False):
            return sorted(candidates, key=lambda pair: pair[1], reverse=True)
        return list(
            ctx.reranker.rerank(
                ctx.query_embedding or [],
                candidates,
                {},
                len(candidates),
                texts=rendered,
                query=ctx.query,
            )
        )


def _weighted_index_proximity(index: int, nuclear_index: int) -> float:
    """``declarative_memory.py::_weighted_index_proximity`` — forward recall is
    worth more than backward recall, so a forward neighbor at distance d sorts
    at ``(d - 0.5) / 2`` and a backward one at ``d``. The nucleus itself lands
    at -0.25 and therefore always first."""
    proximity = index - nuclear_index
    if proximity >= 0:
        return (proximity - 0.5) / 2
    return float(-proximity)


def _memmachine_line(episode: Any) -> str:
    """One rendered episode, upstream's ``episodes_to_string`` /
    ``string_from_episode_context`` format (they agree, down to the
    ``json.dumps`` around the content and the zero-padded strftime day)."""
    timestamp = episode.timestamp
    return (
        f"[{timestamp:%A, %B %d, %Y} at {timestamp:%I:%M %p}] "
        f"{_speaker(episode)}: {json.dumps(episode.content, ensure_ascii=False)}\n"
    )


def _cosine(vector: Any, query: Any, query_norm: float) -> float:
    """Cosine of a stored vector against the pre-normalised query, 0.0 when the
    vector is missing (an id the vector store has no row for)."""
    if not vector:
        return 0.0
    candidate = np.asarray(vector, dtype=np.float32)
    denominator = (float(np.linalg.norm(candidate)) or 1.0) * query_norm
    return float(candidate @ query) / denominator


def _speaker(episode: Any) -> str:
    """Whose turn this is — the ingest's ``meta["speaker"]`` (LoCoMo's two named
    speakers) when present, else the role."""
    return str((getattr(episode, "meta", None) or {}).get("speaker") or episode.role)


def default_read_steps(
    link_expansion_cap: int = 5,
    attach_sources_top_r: int = 2,
    graph_expansion_cap: int = 10,
    graph_expansion_hops: int = 1,
    page_recall_cap: int = 10,
    page_recall_threshold: float = 0.1,
    page_recall_segment_threshold: float = 0.1,
    page_recall_keyword_similarity: str = "containment_mean",
    memmachine_expand_context: int = 0,
    memmachine_context_limit: int = 20,
) -> dict[str, ReadStep]:
    """The methodology-faithful default registry, memory type -> step.

    A cap of 0 drops that step entirely, preserving the falsy-cap disable the
    original ``if memory_type == "notes" and self.link_expansion_cap`` guards
    gave."""
    # MemMachine's step has no disabling cap: mapping derivatives back to their
    # episodes IS the read path, and `expand_context=0` (upstream's own default)
    # only means "no context widening", not "serve the anchors".
    steps: dict[str, ReadStep] = {
        "experiences": ExpandExperiences(),
        "derivatives": MemMachineContextualize(memmachine_expand_context, memmachine_context_limit),
    }
    if link_expansion_cap:
        steps["notes"] = LinkExpansion(link_expansion_cap)
    if attach_sources_top_r:
        steps["episodes"] = AttachSources(attach_sources_top_r)
    if graph_expansion_cap:
        steps["entities"] = GraphRecall(graph_expansion_cap, graph_expansion_hops)
    if page_recall_cap:
        steps["pages"] = MemoryOSPageRecall(
            page_recall_cap,
            page_recall_threshold,
            page_recall_segment_threshold,
            keyword_similarity=page_recall_keyword_similarity,
        )
    return steps


class _DictItem:
    """Lightweight wrapper so derived items render uniformly in a bundle.

    Render exposes methodology metadata (audit P0-4): note context/tags, item
    timestamps, and attached source messages."""

    def __init__(self, data: dict) -> None:
        """`data` is kept by reference, not copied — callers like `AttachSources`
        mutate it in place (e.g. to inject `_source_messages`)."""
        self.data = data
        self.content = data.get("content", "")

    def render(self) -> str:
        """Multi-line text injected verbatim into the LLM context — order and
        section labels here are part of the read-path prompt contract."""
        parts: list[str] = []
        title = self.data.get("title")
        head = f"{title}: " if title else ""
        # Bi-temporal facts render their validity range (Zep's context
        # template: "FACT (Date range: from - to)") so invalidated facts
        # are visibly historical instead of passing as current (round-5 X2).
        if self.data.get("valid_at") or self.data.get("invalid_at"):
            stamp = (
                f" (Date range: {self.data.get('valid_at') or 'unknown'}"
                f" - {self.data.get('invalid_at') or 'present'})"
            )
        else:
            ts = self.data.get("timestamp")
            stamp = f" ({ts})" if ts else ""
        parts.append(f"{head}{self.content}{stamp}")
        # ReasoningBank items carry when-to-apply guidance in description;
        # upstream injects the full item markdown (round-5 X3).
        if self.data.get("description"):
            parts.append(f"description: {self.data['description']}")
        if self.data.get("context"):
            parts.append(f"context: {self.data['context']}")
        # MemoryOS's conversation-chain summary, injected beside the memory it
        # belongs to exactly as upstream's QA prompt does ("Conversation chain
        # overview: {meta_info}", memoryos.py get_response / the LoCoMo driver).
        if self.data.get("meta_info"):
            parts.append(f"Conversation chain overview: {self.data['meta_info']}")
        if self.data.get("tags"):
            parts.append(f"tags: {', '.join(map(str, self.data['tags']))}")
        if self.data.get("_source_messages"):
            src = "\n".join(f"  - {m}" for m in self.data["_source_messages"])
            parts.append(f"Source Messages:\n{src}")
        return "\n".join(parts)
