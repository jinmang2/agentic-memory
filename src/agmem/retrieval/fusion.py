"""Reciprocal Rank Fusion — rank-based, so no score normalization needed."""

from __future__ import annotations


def rrf_fuse(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists of (id, score). Returns (id, fused_score) sorted desc."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (item_id, _score) in enumerate(ranking):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
