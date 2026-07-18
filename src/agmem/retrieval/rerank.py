"""Rerankers: Noop, MMR, LLM (listwise), cross-encoder.

Interface: rerank(query_emb, candidates, vectors, k) -> reordered candidates.
``candidates`` are (id, score) from fusion; ``vectors`` maps id -> embedding.
"""

from __future__ import annotations

import numpy as np

from agmem.capabilities.requires import Requires


class NoopReranker:
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
    ) -> list[tuple[str, float]]:
        return candidates[:k]


class MMRReranker:
    """Maximal Marginal Relevance — relevance/diversity trade-off.

    lambda=1.0 is pure relevance (Graphiti's default mmr_lambda=1);
    lower values penalize redundancy among selected items.
    """

    requires = Requires()
    name = "mmr"
    needs_text = False

    def __init__(self, lambda_: float = 0.5) -> None:
        self.lambda_ = lambda_

    def rerank(
        self,
        query_emb: list[float],
        candidates: list[tuple[str, float]],
        vectors: dict[str, list[float]],
        k: int,
        texts: dict[str, str] | None = None,
        query: str = "",
    ) -> list[tuple[str, float]]:
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
        self.llm = structured_caller  # StructuredCaller; injected by the facade

    def rerank(
        self,
        query_emb,
        candidates,
        vectors,
        k,
        texts: dict[str, str] | None = None,
        query: str = "",
    ):
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
    ):
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


RERANKER_CANDIDATES: list[type] = [
    CrossEncoderReranker,
    LLMReranker,
    MMRReranker,
    NoopReranker,
]
