"""Retrieval pipeline: recall (dense + lexical, per memory type) -> RRF ->
rerank -> hydrate -> methodology-faithful post-steps.

The post-steps live in ``retrieval/steps.py`` as one plugin per methodology,
registered by memory type; this module only sequences them. See that module for
why they stay type-keyed rather than organizer-keyed.
"""

from __future__ import annotations

from agmem.core.types import MemoryBundle, ScoredItem
from agmem.embed.base import Embedder
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
        graph_expansion_cap: int = 10,
        read_steps: dict[str, ReadStep] | None = None,
    ) -> None:
        """``reranker=None`` keeps RRF fusion order as-is. ``lexical_types``
        selects which memory types get a BM25 channel fused via RRF alongside
        the dense one (Zep hybrid adds facts/entities).

        The three cap arguments configure the default post-step registry:
        ``link_expansion_cap`` bounds A-Mem's 1-hop note-link expansion,
        ``attach_sources_top_r`` how many top episode hits get their source
        messages attached, and ``graph_expansion_cap`` how many incident edges
        an entity hit pulls; 0 disables that step. Pass ``read_steps`` to
        replace the registry outright (custom methodology read behavior)."""
        self.doc_store = doc_store
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker  # None -> keep fusion order
        self.graph_store = graph_store
        self.lexical_types = set(lexical_types)
        self.read_steps = (
            read_steps
            if read_steps is not None
            else default_read_steps(
                link_expansion_cap=link_expansion_cap,
                attach_sources_top_r=attach_sources_top_r,
                graph_expansion_cap=graph_expansion_cap,
            )
        )

    def search(
        self,
        query: str,
        k: int | dict[str, int] = 10,
        memory_types: tuple[str, ...] = ("episodic",),
        namespace: str | None = None,
    ) -> MemoryBundle:
        """``k`` may be a dict per memory type (e.g. Nemori's official
        episodic k=10 / semantic m=2k=20)."""
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
            fused = rrf_fuse(rankings)
            if self.reranker is not None and len(fused) > type_k:
                vectors = self.vector_store.get([item_id for item_id, _ in fused])
                texts = None
                if getattr(self.reranker, "needs_text", False):
                    texts = {
                        s.item.id if hasattr(s.item, "id") else s.item.data["id"]: (
                            s.item.content or ""
                        )
                        for s in self._hydrate(fused, memory_type)
                    }
                fused = self.reranker.rerank(
                    query_embedding, fused, vectors, type_k, texts=texts, query=query
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
