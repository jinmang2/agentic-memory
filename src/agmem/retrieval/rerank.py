"""Rerankers: Noop, MMR, LLM (listwise), cross-encoder, episode-mentions,
node-distance.

Interface: ``rerank(query_emb, candidates, vectors, k, texts, query, meta,
center_node_id) -> reordered candidates``. ``candidates`` are (id, score) from
fusion and ``vectors`` maps id -> embedding; everything after ``k`` is optional
context that some rerankers need and others ignore, gated by a ``needs_*`` flag
so the pipeline only pays for what will be read:

- ``texts`` (``needs_text``): id -> item text, for the two that score text.
- ``meta`` (``needs_meta``): id -> the stored item dict, for episode-mentions.
- ``center_node_id``: the centroid node for node-distance.

The last five are the Zep paper's own list (§3.2: RRF, MMR, episode-mentions,
node-distance, cross-encoder), with RRF living in ``fusion.py`` since it is
also how the channels are combined before any reranker runs.
"""

from __future__ import annotations

import numpy as np

from agmem.capabilities.requires import Requires


class NoopReranker:
    """Keeps fusion (RRF) order; the baseline every other reranker is judged
    against."""

    requires = Requires()
    name = "noop"
    needs_text = False

    def rerank(
        self,
        query_emb: list[float],
        candidates: list[tuple[str, float]],
        vectors: dict[str, list[float]],
        k: int,
        texts: dict[str, str] | None = None,
        query: str = "",
        meta: dict[str, dict] | None = None,
        center_node_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Truncates to ``k`` without reordering."""
        return candidates[:k]


class MMRReranker:
    """Maximal Marginal Relevance — relevance/diversity trade-off.

    lambda=1.0 is pure relevance (Graphiti's default mmr_lambda=1);
    lower values penalize redundancy among selected items.

    Algorithm lineage: this is CLASSIC greedy-iterative MMR (redundancy
    measured against the already-SELECTED set, one pick per round). Upstream's
    ``maximal_marginal_relevance`` (search_utils.py:1901-1938) is a one-shot
    batch formula instead — ``λ·(q·c) + (λ−1)·max_sim(c, all OTHER
    candidates)`` scored once, then sorted — with no iterative selection. At
    the shipped recipe's λ=1 the two are equivalent (both reduce to a pure
    relevance sort, so the recipe table's operating point is unaffected); for
    any λ<1 they rank differently. Deliberate: the greedy form is the paper
    literature's MMR, and the divergence stays dormant at the only λ upstream
    ships (round-12 finding 13).
    """

    requires = Requires()
    name = "mmr"
    needs_text = False

    def __init__(self, lambda_: float = 0.5) -> None:
        """``lambda_`` trades relevance (1.0) for diversity (lower)."""
        self.lambda_ = lambda_

    def rerank(
        self,
        query_emb: list[float],
        candidates: list[tuple[str, float]],
        vectors: dict[str, list[float]],
        k: int,
        texts: dict[str, str] | None = None,
        query: str = "",
        meta: dict[str, dict] | None = None,
        center_node_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Greedily selects up to ``k`` items maximizing relevance minus
        redundancy with what's already selected; candidates missing a
        stored vector are appended after the MMR-selected ones, in their
        original fusion order, so a reranker with partial vectors can never
        drop a candidate outright."""
        pool = [(cid, score) for cid, score in candidates if cid in vectors]
        missing = [(cid, score) for cid, score in candidates if cid not in vectors]
        if not pool:
            return candidates[:k]

        q = np.asarray(query_emb, dtype=np.float32)
        qn = np.linalg.norm(q) or 1.0
        mat = {cid: np.asarray(vectors[cid], dtype=np.float32) for cid, _ in pool}
        norms = {cid: (np.linalg.norm(v) or 1.0) for cid, v in mat.items()}
        rel = {cid: float(mat[cid] @ q) / (norms[cid] * qn) for cid, _ in pool}

        selected: list[tuple[str, float]] = []
        remaining = [cid for cid, _ in pool]
        while remaining and len(selected) < k:
            best_id, best_val = None, -np.inf
            for cid in remaining:
                redundancy = max(
                    (float(mat[cid] @ mat[sid]) / (norms[cid] * norms[sid]) for sid, _ in selected),
                    default=0.0,
                )
                val = self.lambda_ * rel[cid] - (1 - self.lambda_) * redundancy
                if val > best_val:
                    best_id, best_val = cid, val
            selected.append((best_id, best_val))
            remaining.remove(best_id)

        # candidates without stored vectors keep their fusion order at the tail
        return (selected + missing)[:k]


class LLMReranker:
    """Listwise rerank via one small-LLM call ('rerank' role).

    Falls back to the incoming order on any parse failure (drop counted
    by the StructuredCaller), so it can never make results worse than
    fusion order — only reorder them.
    """

    requires = Requires(llm_endpoint=True)
    name = "llm"
    needs_text = True

    SCHEMA = {
        "type": "object",
        "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
        "required": ["ranking"],
    }
    PROMPT = """Rank these memory snippets by relevance to the query, best first.

Query: {query}

Snippets:
{snippets}

Return JSON: {{"ranking": [most relevant index numbers, e.g. 2, 0, 1, ...]}}"""

    def __init__(self, structured_caller=None) -> None:
        """``structured_caller`` is optional at construction (injected later
        by the facade); `rerank` degrades to fusion order until it's set."""
        self.llm = structured_caller  # StructuredCaller; injected by the facade

    def rerank(
        self,
        query_emb,
        candidates,
        vectors,
        k,
        texts: dict[str, str] | None = None,
        query: str = "",
        meta: dict[str, dict] | None = None,
        center_node_id: str | None = None,
    ):
        """Sends up to ``max(2k, 10)`` candidates to the 'rerank' LLM role in
        one listwise call. Falls back to incoming order (no reorder) if the
        caller has no LLM, no texts, or the LLM call fails to parse. Indices
        the model omits keep their relative fusion order at the tail."""
        if self.llm is None or not texts:
            return candidates[:k]
        pool = candidates[: max(k * 2, 10)]
        snippets = "\n".join(f"[{i}] {texts.get(cid, '')[:200]}" for i, (cid, _) in enumerate(pool))
        result = self.llm.call(
            "rerank",
            self.PROMPT.format(query=query, snippets=snippets),
            self.SCHEMA,
            required_keys=("ranking",),
        )
        if result is None:
            return candidates[:k]
        seen: list[int] = []
        for idx in result["ranking"]:
            if isinstance(idx, int) and 0 <= idx < len(pool) and idx not in seen:
                seen.append(idx)
        seen += [i for i in range(len(pool)) if i not in seen]  # unranked keep order
        return [pool[i] for i in seen][:k]


class CrossEncoderReranker:
    """Cross-encoder rerank (GPU recommended; heaviest, most precise)."""

    requires = Requires(python_pkgs=("sentence_transformers",), vram_gb=1.0)
    name = "cross-encoder"
    needs_text = True

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
        """Loads ``model_name`` eagerly (network/disk fetch on first use);
        ``device=None`` lets sentence-transformers pick (GPU if available)."""
        from sentence_transformers import CrossEncoder  # gated by requires

        self._model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query_emb,
        candidates,
        vectors,
        k,
        texts: dict[str, str] | None = None,
        query: str = "",
        meta: dict[str, dict] | None = None,
        center_node_id: str | None = None,
    ):
        """Scores each candidate's text against the query with the loaded
        cross-encoder and sorts descending. Candidates with no text fall
        back to fusion order, appended after the scored ones."""
        if not texts:
            return candidates[:k]
        pool = [(cid, s) for cid, s in candidates if cid in texts]
        missing = [(cid, s) for cid, s in candidates if cid not in texts]
        if not pool:
            return candidates[:k]
        scores = self._model.predict([(query, texts[cid][:512]) for cid, _ in pool])
        ranked = sorted(
            zip((cid for cid, _ in pool), map(float, scores)),
            key=lambda x: x[1],
            reverse=True,
        )
        return (ranked + missing)[:k]


class EpisodeMentionsReranker:
    """Zep's graph-based episode-mentions reranker (paper §3.2): "prioritizes
    results based on the frequency of entity or fact mentions within a
    conversation, enabling a system where frequently referenced information
    becomes more readily accessible."

    Upstream counts ``MENTIONS`` edges from episodes to the node/edge
    (``search_utils.episode_mentions_reranker``). Our equivalent count is
    ``len(source_episode_ids)``, the provenance every organizer already records
    — same quantity, since that list is exactly which episodes produced the
    item. Ties keep fusion order (``sorted`` is stable), so this only ever
    promotes; it never invents an order among equally-mentioned items.

    Sort DIRECTION is a named divergence from upstream: its
    ``episode_mentions_reranker`` (search_utils.py:1860-1896) sorts the counts
    ASCENDING with missing nodes at ``inf`` — the MOST-mentioned item lands
    LAST, contradicting the paper sentence quoted above. That is almost
    certainly an upstream bug (nothing there flags it as intentional), and this
    class deliberately follows the PAPER instead: mention count descending
    (round-12 finding 10)."""

    requires = Requires()
    name = "episode-mentions"
    needs_text = False
    needs_meta = True

    def rerank(
        self,
        query_emb: list[float],
        candidates: list[tuple[str, float]],
        vectors: dict[str, list[float]],
        k: int,
        texts: dict[str, str] | None = None,
        query: str = "",
        meta: dict[str, dict] | None = None,
        center_node_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Sorts by mention count descending. With no ``meta`` (nothing
        hydrated) this is fusion order truncated — the same degradation the
        text-based rerankers take when ``texts`` is empty."""
        if not meta:
            return candidates[:k]
        counts = {
            cid: len(meta.get(cid, {}).get("source_episode_ids", []) or []) for cid, _ in candidates
        }
        return sorted(candidates, key=lambda c: counts.get(c[0], 0), reverse=True)[:k]


class NodeDistanceReranker:
    """Zep's node-distance reranker (paper §3.2): "reorders results based on
    their graph distance from a designated centroid node, providing context
    localized to specific areas of the knowledge graph."

    Needs both a graph and a centroid. The graph comes at construction (a
    framework handle, like the doc store elsewhere); the centroid is per-query,
    so it arrives as ``center_node_id`` — upstream's ``center_node_uuid``
    argument to ``search()``. With no centroid this is a no-op truncation,
    which is also what upstream does: ``node_distance_reranker`` is only
    reachable from a config that supplies one.

    Distance is computed by widening BFS rings from the centroid
    (``graph_store.neighbors`` at increasing depth), so an item at hop 1 beats
    one at hop 2; unreachable items sort last, keeping fusion order among
    themselves. This is the PAPER's shape ("graph distance from a designated
    centroid node"), and a named divergence from upstream:
    ``node_distance_reranker`` (search_utils.py:1798-1856) is actually a
    1-HOP ADJACENCY TEST — direct neighbours score 1 and everything else
    ``inf`` (the "shortest path" comment in that function is stale) — so
    upstream ties hop-2 with hop-10 where this class orders them
    (round-12 finding 9).
    """

    requires = Requires()
    name = "node-distance"
    needs_text = False
    needs_meta = False

    def __init__(self, graph_store=None, namespace: str = "main", max_hops: int = 3) -> None:
        """``graph_store=None`` makes this a no-op truncation (no graph to
        measure distance in). ``max_hops`` bounds the ring expansion; items
        beyond it are treated as unreachable."""
        self.graph_store = graph_store
        self.namespace = namespace
        self.max_hops = max_hops

    def rerank(
        self,
        query_emb: list[float],
        candidates: list[tuple[str, float]],
        vectors: dict[str, list[float]],
        k: int,
        texts: dict[str, str] | None = None,
        query: str = "",
        meta: dict[str, dict] | None = None,
        center_node_id: str | None = None,
    ) -> list[tuple[str, float]]:
        if self.graph_store is None or not center_node_id:
            return candidates[:k]
        distance = {center_node_id: 0}
        for hops in range(1, self.max_hops + 1):
            for node in self.graph_store.neighbors(center_node_id, self.namespace, hops):
                distance.setdefault(str(node["id"]), hops)
        far = self.max_hops + 1
        return sorted(candidates, key=lambda c: distance.get(c[0], far))[:k]


RERANKER_CANDIDATES: list[type] = [
    CrossEncoderReranker,
    LLMReranker,
    MMRReranker,
    # Order matters: these two are capability-free, so they would be picked by
    # a bare `resolve("reranker")` if they preceded MMR. They are meaningful
    # only for the Zep recipes that name them explicitly — node-distance is a
    # no-op without a centroid — so they sit BELOW MMR, which stays the
    # capability-free fallback when a profile default is unavailable.
    EpisodeMentionsReranker,
    NodeDistanceReranker,
    NoopReranker,
]
