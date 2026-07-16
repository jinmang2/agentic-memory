"""Rerankers. Phase 1: Noop + MMR. LLM/cross-encoder land in Phase 3.

Interface: rerank(query_emb, candidates, vectors, k) -> reordered candidates.
``candidates`` are (id, score) from fusion; ``vectors`` maps id -> embedding.
"""

from __future__ import annotations

import numpy as np

from agmem.capabilities.requires import Requires


class NoopReranker:
    requires = Requires()
    name = "noop"

    def rerank(self, query_emb: list[float], candidates: list[tuple[str, float]],
               vectors: dict[str, list[float]], k: int) -> list[tuple[str, float]]:
        return candidates[:k]


class MMRReranker:
    """Maximal Marginal Relevance — relevance/diversity trade-off.

    lambda=1.0 is pure relevance (Graphiti's default mmr_lambda=1);
    lower values penalize redundancy among selected items.
    """

    requires = Requires()
    name = "mmr"

    def __init__(self, lambda_: float = 0.5) -> None:
        self.lambda_ = lambda_

    def rerank(self, query_emb: list[float], candidates: list[tuple[str, float]],
               vectors: dict[str, list[float]], k: int) -> list[tuple[str, float]]:
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
                    (float(mat[cid] @ mat[sid]) / (norms[cid] * norms[sid])
                     for sid, _ in selected),
                    default=0.0,
                )
                val = self.lambda_ * rel[cid] - (1 - self.lambda_) * redundancy
                if val > best_val:
                    best_id, best_val = cid, val
            selected.append((best_id, best_val))
            remaining.remove(best_id)

        # candidates without stored vectors keep their fusion order at the tail
        return (selected + missing)[:k]


RERANKER_CANDIDATES: list[type] = [MMRReranker, NoopReranker]
