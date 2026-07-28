"""Reciprocal Rank Fusion — rank-based, so no score normalization needed."""

from __future__ import annotations


def rrf_fuse(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists of (id, score). Returns (id, fused_score) sorted desc.

    The fused score is the RRF sum DIVIDED BY the number of input rankings, so
    a rank-1 hit scores ``1/(k+1)`` whether it won one channel or every channel.

    Dividing by a per-call constant cannot change the order within one call —
    it is a monotone rescale — so this is invisible to any single fusion. It
    exists for the caller that compares fused scores ACROSS calls:
    ``RetrievalPipeline.search`` fuses once per memory type and then
    ``MemoryBundle.render`` sorts the whole bundle by score under one budget.
    Without the divisor a type with two channels (``episodic``: dense + BM25)
    tops out at twice the score of a dense-only derived type, so every raw
    episode that placed in both channels outranked every note/episode/fact
    before it — an artifact of channel count, not of relevance. Measured:
    episodic 0.0328 vs notes 0.0164 at rank 1, with the whole episodic block
    above the whole notes block.

    Un-normalized RRF is the textbook form, and it is the right one when every
    candidate passes through the same set of rankings. That premise is exactly
    what per-type fusion breaks, which is why the divisor lives here rather
    than in the caller: any future second caller inherits comparable scores
    instead of rediscovering this (2026-07-27 audit B2).

    ``k`` is the textbook RRF constant — score ``1/(k + rank)`` with rank
    1-based, so at the default 60 a rank-1 hit scores 1/61 (Cormack et al.'s
    form, and this framework's default; reachable from config as ``rrf_k``).
    Upstream Zep/Graphiti's ``rrf`` is NOT this function at its own default:
    it computes ``1/(i + rank_const)`` with a 0-BASED index and
    ``rank_const=1`` (search_utils.py:1780-1786), so its rank-1 hit scores
    1/1, not 1/61 — steep front-loading that is not a monotone rescale of
    k=60 across multiple channels: items ranked (1,100) vs (3,3) in two
    channels order differently under the two constants (round-12 finding 8).
    The Zep recipe table therefore emits ``rrf_k: 1``. Note the residual
    off-by-one: rrf_k=1 here scores rank-1 at 1/2 (1/(1+rank)), one position
    shifted from upstream's exact 1/rank series — kept so the textbook
    parametrization (and the measured default) stays untouched; the shift
    reproduces upstream's front-loading and the (1,100)/(3,3) order flip, but
    not upstream's scores bit-for-bit."""
    if not rankings:
        return []
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (item_id, _score) in enumerate(ranking):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    channels = len(rankings)
    return sorted(
        ((item_id, score / channels) for item_id, score in fused.items()),
        key=lambda x: x[1],
        reverse=True,
    )
