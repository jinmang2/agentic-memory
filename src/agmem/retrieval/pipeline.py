"""Retrieval pipeline: recall (dense + lexical, per memory type) -> RRF ->
rerank -> hydrate -> methodology-faithful post-steps.

Post-hydration steps restore upstream read-path semantics that the deep
fidelity audit (docs/research/fidelity-deep-audit.md) found missing:
- notes: 1-hop link expansion (A-Mem official eval's
  find_related_memories_raw — retrieved notes pull in their linked
  neighbors, capped)
- episodes: top-r narrative episodes attach their original source messages
  (Nemori r=2 — offsets narrative-abstraction information loss)
"""

from __future__ import annotations

from agmem.core.types import MemoryBundle, ScoredItem
from agmem.embed.base import Embedder
from agmem.retrieval.fusion import rrf_fuse
from agmem.stores.base import DocStore, VectorStore


class RetrievalPipeline:
    def __init__(self, doc: DocStore, vec: VectorStore, embedder: Embedder,
                 reranker=None, link_expansion_cap: int = 5,
                 attach_sources_top_r: int = 2) -> None:
        self.doc = doc
        self.vec = vec
        self.embedder = embedder
        self.reranker = reranker  # None -> keep fusion order
        self.link_expansion_cap = link_expansion_cap
        self.attach_sources_top_r = attach_sources_top_r

    def search(
        self,
        query: str,
        k: int | dict[str, int] = 10,
        memory_types: tuple[str, ...] = ("episodic",),
        namespace: str | None = None,
    ) -> MemoryBundle:
        """``k`` may be a dict per memory type (e.g. Nemori's official
        episodic k=10 / semantic m=2k=20)."""
        query_emb = self.embedder.embed([query], kind="query")[0]

        bundle = MemoryBundle(query=query)
        for memory_type in memory_types:
            type_k = k.get(memory_type, 10) if isinstance(k, dict) else k
            candidate_k = type_k * 3  # over-fetch per source, fuse down

            rankings = [
                self.vec.search(query_emb, k=candidate_k,
                                memory_type=memory_type, namespace=namespace)
            ]
            if memory_type == "episodic":
                rankings.append(
                    self.doc.search_lexical(query, k=candidate_k, namespace=namespace)
                )
            fused = rrf_fuse(rankings)
            if self.reranker is not None and len(fused) > type_k:
                vectors = self.vec.get([cid for cid, _ in fused])
                texts = None
                if getattr(self.reranker, "needs_text", False):
                    texts = {
                        s.item.id if hasattr(s.item, "id") else s.item.data["id"]:
                        (s.item.content or "")
                        for s in self._hydrate(fused, memory_type)
                    }
                fused = self.reranker.rerank(query_emb, fused, vectors, type_k,
                                             texts=texts, query=query)
            else:
                fused = fused[:type_k]
            hydrated = self._hydrate(fused, memory_type)

            if memory_type == "notes" and self.link_expansion_cap:
                hydrated += self._expand_links(hydrated)
            if memory_type == "episodes" and self.attach_sources_top_r:
                self._attach_sources(hydrated)

            bundle.items.extend(hydrated)
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
                if data.get("deleted"):
                    continue  # tombstone (round-5 X1: legacy ghost guard)
                item_id = data.get("id", "?")
                out.append(ScoredItem(
                    item=_DictItem(data), memory_type=memory_type,
                    score=scores.get(item_id, 0.0),
                    provenance=data.get("source_episode_ids", []),
                ))
        return out

    def _expand_links(self, hits: list[ScoredItem]) -> list[ScoredItem]:
        """A-Mem 1-hop: pull linked neighbor notes of retrieved notes.

        Links are unidirectional as upstream. Cap semantics deviate:
        upstream caps PER HIT (agiresearch k per hit; WujiangXu k+1 via an
        off-by-one), so eval k=10 can pull ~100 link neighbors — WujiangXu
        #16/#21 show even upstream considers this ambiguous. We use one
        global cap (default 5); neighbors score just below their parent.
        Keep this deviation in result caveats when comparing multi-hop."""
        seen = {s.item.data["id"] for s in hits}
        wanted: list[tuple[str, float]] = []
        for s in sorted(hits, key=lambda s: s.score, reverse=True):
            for linked_id in s.item.data.get("links", []):
                if linked_id not in seen and len(wanted) < self.link_expansion_cap:
                    seen.add(linked_id)
                    wanted.append((linked_id, s.score * 0.9))
        if not wanted:
            return []
        by_id = dict(wanted)
        out: list[ScoredItem] = []
        for data in self.doc.get_items(list(by_id), "notes"):
            out.append(ScoredItem(
                item=_DictItem(data), memory_type="notes",
                score=by_id.get(data.get("id"), 0.0),
                provenance=data.get("source_episode_ids", []),
            ))
        return out

    def _attach_sources(self, hits: list[ScoredItem]) -> None:
        """Nemori r=2: top-r episodes carry their raw source messages,
        rendered as ``role: content`` lines as upstream search.py does."""
        for s in sorted(hits, key=lambda s: s.score,
                        reverse=True)[: self.attach_sources_top_r]:
            src_ids = s.item.data.get("source_episode_ids", [])
            if src_ids:
                episodes = self.doc.get_episodes(src_ids)
                s.item.data["_source_messages"] = [
                    f"{ep.role}: {ep.content}" for ep in episodes
                ]


class _DictItem:
    """Lightweight wrapper so derived items render uniformly in a bundle.

    Render exposes methodology metadata (audit P0-4): note context/tags,
    item timestamps, and attached source messages."""

    def __init__(self, data: dict) -> None:
        self.data = data
        self.content = data.get("content", "")

    def render(self) -> str:
        parts: list[str] = []
        title = self.data.get("title")
        head = f"{title}: " if title else ""
        # Bi-temporal facts render their validity range (Zep's context
        # template: "FACT (Date range: from - to)") so invalidated facts
        # are visibly historical instead of passing as current (round-5 X2).
        if self.data.get("valid_at") or self.data.get("invalid_at"):
            stamp = (f" (Date range: {self.data.get('valid_at') or 'unknown'}"
                     f" - {self.data.get('invalid_at') or 'present'})")
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
        if self.data.get("tags"):
            parts.append(f"tags: {', '.join(map(str, self.data['tags']))}")
        if self.data.get("_source_messages"):
            src = "\n".join(f"  - {m}" for m in self.data["_source_messages"])
            parts.append(f"Source Messages:\n{src}")
        return "\n".join(parts)
