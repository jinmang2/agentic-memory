"""Retrieval pipeline: recall (dense + lexical, per memory type) -> RRF -> hydrate.

Phase 0 scope: DenseRecall over every requested memory type plus
LexicalRecall (FTS5/BM25) over episodic. QueryExpansion and rerankers
land in later phases behind the same pipeline entry point.
"""

from __future__ import annotations

from agmem.core.types import Episode, MemoryBundle, ScoredItem
from agmem.embed.base import Embedder
from agmem.retrieval.fusion import rrf_fuse
from agmem.stores.base import DocStore, VectorStore


class RetrievalPipeline:
    def __init__(self, doc: DocStore, vec: VectorStore, embedder: Embedder,
                 reranker=None) -> None:
        self.doc = doc
        self.vec = vec
        self.embedder = embedder
        self.reranker = reranker  # None -> keep fusion order

    def search(
        self,
        query: str,
        k: int = 10,
        memory_types: tuple[str, ...] = ("episodic",),
        namespace: str | None = None,
    ) -> MemoryBundle:
        query_emb = self.embedder.embed([query], kind="query")[0]
        candidate_k = k * 3  # over-fetch per recall source, fuse down to k

        bundle = MemoryBundle(query=query)
        for memory_type in memory_types:
            rankings = [
                self.vec.search(query_emb, k=candidate_k,
                                memory_type=memory_type, namespace=namespace)
            ]
            if memory_type == "episodic":
                rankings.append(
                    self.doc.search_lexical(query, k=candidate_k, namespace=namespace)
                )
            fused = rrf_fuse(rankings)
            if self.reranker is not None and len(fused) > k:
                vectors = self.vec.get([cid for cid, _ in fused])
                fused = self.reranker.rerank(query_emb, fused, vectors, k)
            else:
                fused = fused[:k]
            bundle.items.extend(self._hydrate(fused, memory_type))
        return bundle

    def _hydrate(self, fused: list[tuple[str, float]], memory_type: str) -> list[ScoredItem]:
        ids = [item_id for item_id, _ in fused]
        scores = dict(fused)
        out: list[ScoredItem] = []
        if memory_type == "episodic":
            for ep in self.doc.get_episodes(ids):
                out.append(ScoredItem(item=ep, memory_type=memory_type,
                                      score=scores[ep.id], provenance=[ep.id]))
        else:
            for data in self.doc.get_items(ids, memory_type):
                item_id = data.get("id", "?")
                out.append(ScoredItem(
                    item=_DictItem(data), memory_type=memory_type,
                    score=scores.get(item_id, 0.0),
                    provenance=data.get("source_episode_ids", []),
                ))
        return out


class _DictItem:
    """Lightweight wrapper so derived items render uniformly in a bundle."""

    def __init__(self, data: dict) -> None:
        self.data = data
        self.content = data.get("content", "")

    def render(self) -> str:
        title = self.data.get("title")
        return f"{title}: {self.content}" if title else self.content
