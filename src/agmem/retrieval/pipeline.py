"""Retrieval pipeline: recall (dense + lexical, per memory type) -> RRF ->
rerank -> hydrate -> methodology-faithful post-steps.

The post-steps live in ``retrieval/steps.py`` as one plugin per methodology,
registered by memory type; this module only sequences them. See that module for
why they stay type-keyed rather than organizer-keyed.
"""

from __future__ import annotations

from agmem.core.types import MemoryBundle, ScoredItem
from agmem.embed.base import Embedder
from agmem.retrieval.bfs import MAX_SEARCH_DEPTH, bfs_entity_ranking, bfs_fact_ranking
from agmem.retrieval.fusion import rrf_fuse
from agmem.retrieval.steps import (
    ReadContext,
    ReadStep,
    _DictItem,
    default_read_steps,
    is_servable,
)
from agmem.stores.base import DocStore, VectorStore


def _item_id(scored: ScoredItem) -> str | None:
    """Id of a scored item, whichever shape it has — an ``Episode`` carries ``id``
    as an attribute, derived items are dict-backed (``_DictItem``)."""
    return getattr(scored.item, "id", None) or scored.item.data.get("id")


class RetrievalPipeline:
    """Facade over one namespace's stores for `search()`; stateless beyond
    its store/embedder/reranker references, so one instance is safe to
    reuse across calls with different memory types."""

    def __init__(
        self,
        doc_store: DocStore,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker=None,
        link_expansion_cap: int = 5,
        attach_sources_top_r: int = 2,
        graph_store=None,
        lexical_types: tuple[str, ...] = ("episodic",),
        bfs_types: tuple[str, ...] = (),
        bfs_max_depth: int = MAX_SEARCH_DEPTH,
        graph_expansion_cap: int = 0,
        graph_expansion_hops: int = 1,
        page_recall_cap: int = 10,
        page_recall_threshold: float = 0.1,
        page_recall_segment_threshold: float = 0.1,
        page_recall_keyword_similarity: str = "containment_mean",
        memmachine_expand_context: int = 0,
        memmachine_context_limit: int = 20,
        task_graph_expansion_cap: int = 5,
        read_steps: dict[str, ReadStep] | None = None,
    ) -> None:
        """``reranker=None`` keeps RRF fusion order as-is. ``lexical_types``
        selects which memory types get a BM25 channel fused via RRF alongside
        the dense one (Zep hybrid adds facts/entities), and ``bfs_types`` which
        additionally get Zep's graph BFS channel (φ_bfs, ``retrieval/bfs.py``)
        at depth ``bfs_max_depth``. The three axes are separate because
        upstream's recipes vary them independently: its RRF recipes run
        cosine+BM25, its cross-encoder recipes add BFS, and communities never
        get BFS in either.

        The three cap arguments configure the default post-step registry:
        ``link_expansion_cap`` bounds A-Mem's 1-hop note-link expansion,
        ``attach_sources_top_r`` how many top episode hits get their source
        messages attached, and ``graph_expansion_cap``/``graph_expansion_hops``
        how many edges an entity hit pulls and how far the walk goes first; a
        0 cap disables that step. ``graph_expansion_cap`` defaults to 0 because
        ``GraphRecall`` was our stand-in for φ_bfs and ``bfs_types`` now
        expresses that faithfully — see ``GraphRecall`` for what it still is.
        ``page_recall_cap``/``page_recall_threshold`` are MemoryOS's second
        retrieval stage (matched segment -> its pages, upstream
        ``retrieval_queue_capacity``/``page_similarity_threshold``); 0 disables
        it and serves segment summaries instead, which upstream never does.
        Pass ``read_steps`` to replace the registry outright (custom
        methodology read behavior)."""
        self.doc_store = doc_store
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker  # None -> keep fusion order
        self.graph_store = graph_store
        self.lexical_types = set(lexical_types)
        self.bfs_types = set(bfs_types)
        self.bfs_max_depth = bfs_max_depth
        self.read_steps = (
            read_steps
            if read_steps is not None
            else default_read_steps(
                link_expansion_cap=link_expansion_cap,
                attach_sources_top_r=attach_sources_top_r,
                graph_expansion_cap=graph_expansion_cap,
                graph_expansion_hops=graph_expansion_hops,
                page_recall_cap=page_recall_cap,
                page_recall_threshold=page_recall_threshold,
                page_recall_segment_threshold=page_recall_segment_threshold,
                page_recall_keyword_similarity=page_recall_keyword_similarity,
                memmachine_expand_context=memmachine_expand_context,
                memmachine_context_limit=memmachine_context_limit,
                task_graph_expansion_cap=task_graph_expansion_cap,
            )
        )

    def search(
        self,
        query: str,
        k: int | dict[str, int] = 10,
        memory_types: tuple[str, ...] = ("episodic",),
        namespace: str | None = None,
        center_node_id: str | None = None,
        bfs_origin_ids: list[str] | None = None,
        query_keywords: set[str] | frozenset[str] | None = None,
    ) -> MemoryBundle:
        """``k`` may be a dict per memory type (e.g. Nemori's official
        episodic k=10 / semantic m=2k=20).

        ``query_keywords`` reaches the read steps as-is; it is not derived here
        because the methodology that uses it derives it with an LLM and this
        layer has none (see ``ReadContext.query_keywords``).

        Fusion runs once PER memory type, and the resulting scores are then
        compared across types — ``MemoryBundle.render`` sorts the whole bundle
        and cuts it against one shared budget. ``rrf_fuse`` divides by its
        channel count for exactly that reason, so a dual-channel type
        (``episodic``: dense + BM25) no longer scores twice a dense-only
        derived type purely for having two channels (2026-07-27 audit B2; see
        ``rrf_fuse`` for the measured 0.0328-vs-0.0164 asymmetry it removes).

        Methodology-pure configs are bit-identical either way — they pass one
        derived type (``notes``) or several dense-only ones
        (``episodes``/``semantic``), so every score already came from the same
        channel count, and a common divisor cannot reorder them. What changes
        is the mixed configs (raw ``episodic`` alongside derived types) and
        ``default_memory_types``, where ``episodic`` always leads."""
        query_embedding = self.embedder.embed([query], kind="query")[0]

        bundle = MemoryBundle(query=query)
        served: set[tuple[str, str | None]] = set()
        for memory_type in memory_types:
            type_k = k.get(memory_type, 10) if isinstance(k, dict) else k
            candidate_k = type_k * 3  # over-fetch per source, fuse down

            rankings = [
                self.vector_store.search(
                    query_embedding, k=candidate_k, memory_type=memory_type, namespace=namespace
                )
            ]
            if memory_type == "episodic":
                rankings.append(
                    self.doc_store.search_lexical(query, k=candidate_k, namespace=namespace)
                )
            elif memory_type in self.lexical_types:
                rankings.append(
                    self.doc_store.search_lexical_items(
                        query, memory_type, k=candidate_k, namespace=namespace
                    )
                )
            if memory_type in self.bfs_types:
                bfs = self._bfs_ranking(
                    memory_type, rankings, candidate_k, namespace, bfs_origin_ids
                )
                if bfs:
                    rankings.append(bfs)
            fused = rrf_fuse(rankings)
            # `len(fused) > 1`, not `> type_k`. The old gate skipped the reranker
            # whenever the candidate pool did not exceed k, on the assumption
            # that a reranker only matters when it has to DROP something. It also
            # decides ORDER, and order survives truncation: `MemoryBundle.render`
            # sorts the whole bundle by score, so an unranked type keeps RRF
            # scores while a ranked one carries relevance scores, and the two get
            # compared. Upstream reranks unconditionally and truncates after.
            # No stored number moves — every measured run resolved to
            # NoopReranker (profile `lite`), whose rerank is truncation.
            if self.reranker is not None and len(fused) > 1:
                vectors = self.vector_store.get([item_id for item_id, _ in fused])
                texts, meta = None, None
                needs_text = getattr(self.reranker, "needs_text", False)
                needs_meta = getattr(self.reranker, "needs_meta", False)
                if needs_text or needs_meta:
                    hydrated_for_rerank = self._hydrate(fused, memory_type)
                    if needs_text:
                        texts = {_item_id(s): (s.item.content or "") for s in hydrated_for_rerank}
                    if needs_meta:
                        # The stored dict, not the ScoredItem: episode-mentions
                        # reads `source_episode_ids`, which lives in the item.
                        meta = {
                            _item_id(s): getattr(s.item, "data", {}) or {}
                            for s in hydrated_for_rerank
                        }
                fused = self.reranker.rerank(
                    query_embedding,
                    fused,
                    vectors,
                    type_k,
                    texts=texts,
                    query=query,
                    meta=meta,
                    center_node_id=center_node_id,
                )
            else:
                fused = fused[:type_k]
            hydrated = self._hydrate(fused, memory_type)

            step = self.read_steps.get(memory_type)
            if step is not None:
                hydrated = step.run(
                    hydrated,
                    ReadContext(
                        doc_store=self.doc_store,
                        namespace=namespace,
                        graph_store=self.graph_store,
                        bundle_ids={_item_id(s) for s in bundle.items},
                        query_embedding=query_embedding,
                        vector_store=self.vector_store,
                        query_keywords=frozenset(query_keywords or ()),
                        query=query,
                        reranker=self.reranker,
                    ),
                )

            # A step may emit a type that another pass also serves, so the same
            # item can reach the bundle twice and be rendered twice into the QA
            # prompt: ExpandExperiences replaces an experience with its strategy
            # items while reasoning_bank declares both types, and two retrieved
            # experiences can share one strategy. Dedup on (memory_type, id) —
            # never the bare id, since the items table is keyed (id, memory_type)
            # and the same id under two types is two distinct items. First
            # occurrence wins, so an expansion's score/provenance is the one kept
            # and `produces` order still decides which copy survives.
            for scored in hydrated:
                key = (scored.memory_type, _item_id(scored))
                if key in served:
                    continue
                served.add(key)
                bundle.items.append(scored)
        return bundle

    def _bfs_ranking(
        self,
        memory_type: str,
        rankings: list[list[tuple[str, float]]],
        k: int,
        namespace: str | None,
        origin_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Zep's φ_bfs as a third channel.

        ``origin_ids`` is upstream's ``bfs_origin_node_uuids``, and like upstream
        it REPLACES the derived origins rather than adding to them
        (``search.py`` only derives them ``if bfs_origin_node_uuids is None``).
        The paper's motivating case for supplying them is recency — "particularly
        valuable when using recent episodes as seeds … allowing the system to
        incorporate recently mentioned entities and relationships into the
        retrieved context" (§3.1) — which ``AgenticMemory.recent_episode_entity_ids``
        computes.

        Derived origins, when none are given: for ``entities`` the found nodes
        themselves; for ``facts`` the found edges' SUBJECT nodes only, matching
        upstream's ``source_node_uuids = [edge.source_node_uuid ...]`` — using
        both endpoints would widen the frontier beyond what it searches.
        Anything else (``communities`` included, which upstream never gives a
        BFS channel) returns nothing."""
        if self.graph_store is None:
            return []
        found = list(dict.fromkeys(item_id for ranking in rankings for item_id, _ in ranking))
        resolved_ns = namespace or "main"
        if origin_ids:
            # Explicit origins are node ids for both types: an edge channel
            # seeded from recent episodes still walks from NODES.
            exclude = set(found) if memory_type == "facts" else None
            if memory_type == "facts":
                return bfs_fact_ranking(
                    self.graph_store,
                    list(dict.fromkeys(origin_ids)),
                    resolved_ns,
                    k,
                    self.bfs_max_depth,
                    exclude=exclude,
                )
            if memory_type == "entities":
                return bfs_entity_ranking(
                    self.graph_store,
                    list(dict.fromkeys(origin_ids)),
                    resolved_ns,
                    k,
                    self.bfs_max_depth,
                )
            return []
        if not found:
            return []
        if memory_type == "entities":
            return bfs_entity_ranking(self.graph_store, found, resolved_ns, k, self.bfs_max_depth)
        if memory_type == "facts":
            origins = list(
                dict.fromkeys(
                    str(data["subject_id"])
                    for data in self.doc_store.get_items(found, "facts")
                    if data.get("subject_id")
                )
            )
            return bfs_fact_ranking(
                self.graph_store,
                origins,
                resolved_ns,
                k,
                self.bfs_max_depth,
                exclude=set(found),
            )
        return []

    def _hydrate(self, fused: list[tuple[str, float]], memory_type: str) -> list[ScoredItem]:
        ids = [item_id for item_id, _ in fused]
        scores = dict(fused)
        out: list[ScoredItem] = []
        if memory_type == "episodic":
            for episode in self.doc_store.get_episodes(ids):
                out.append(
                    ScoredItem(
                        item=episode,
                        memory_type=memory_type,
                        score=scores[episode.id],
                        provenance=[episode.id],
                    )
                )
        else:
            for data in self.doc_store.get_items(ids, memory_type):
                # Tombstones (round-5 X1) and invalidated non-bi-temporal items.
                # The vector is dropped on INVALIDATE, so dense recall already
                # misses these; the lexical channel does not, and neither did
                # this hydrate — an invalidated `semantic` item came back the
                # moment its type was added to `lexical_types`.
                if not is_servable(data, memory_type):
                    continue
                item_id = data.get("id", "?")
                out.append(
                    ScoredItem(
                        item=_DictItem(data),
                        memory_type=memory_type,
                        score=scores.get(item_id, 0.0),
                        provenance=data.get("source_episode_ids", []),
                    )
                )
        return out
