"""Zep's third search function, breadth-first search over the graph (φ_bfs).

The paper puts BFS beside cosine and BM25 as a *search* function, not a
post-processing step (§3.1): "the breadth-first search enhances initial search
results by identifying additional nodes and edges within n-hops … breadth-first
search reveals contextual similarities — where nodes and edges closer in the
graph appear in more similar conversational contexts." All three feed the
reranker ρ together, so BFS has to produce a RANKED CANDIDATE LIST like the
other two, which is why this is a channel here rather than the flat append
``retrieval.steps.GraphRecall`` used to do.

Origins follow upstream (``search.py``): when ``search()`` is given no explicit
``bfs_origin_node_uuids`` it derives them from what the other channels of the
SAME object type already returned — the source nodes of the edges found, or the
entity nodes found. The paper also describes seeding from recent episodes
("particularly valuable when using recent episodes as seeds"); that variant is
upstream's explicit-origins path and is not what its default search does, so it
is not reproduced here.

Ordering inside the channel is ours to fix. Upstream's Cypher returns rows in
driver order, which no backend guarantees; RRF and every other reranker consume
RANK, so a nondeterministic order would make the same query answer differently
across runs. Candidates are therefore ordered by hop distance first (the BFS
ring they were found in — the closer, the more "contextually similar", which is
the paper's own justification for the channel) and then by a content-derived
key, never by id (ids are random uuids, which would move the nondeterminism
rather than remove it).

Direction: upstream's BFS queries walk ``-[:RELATES_TO|MENTIONS*1..n]->`` —
OUTGOING only (``node_bfs_search``/``edge_bfs_search``, search_utils.py:448+/
790+) — so this channel walks the graph directed (``direction="out"`` on
``neighbors``), and in that mode an edge counts as within-n-hops only when its
SOURCE node is reachable: a fact ``B -> A`` does not make B reachable from A
here, exactly as it does not upstream (round-12 finding 12). The generic store
default stays undirected ``"both"`` for the non-Zep callers (GraphRecall,
node-distance).
"""

from __future__ import annotations

from typing import Any

# Upstream search_utils.MAX_SEARCH_DEPTH, the default bfs_max_depth of every
# SearchConfig that enables the channel.
MAX_SEARCH_DEPTH = 3


def _rings(
    graph_store: Any,
    origins: list[str],
    namespace: str,
    max_depth: int,
    direction: str = "out",
) -> dict[str, int]:
    """node id -> the smallest hop count at which it was reached from any
    origin (origins themselves are 0). Built by widening ``neighbors`` calls
    rather than one deep call, because the store contract returns nodes without
    their depth and the depth is what orders this channel. ``direction="out"``
    is this module's default — see the module docstring."""
    distance: dict[str, int] = {origin: 0 for origin in origins}
    for depth in range(1, max_depth + 1):
        for origin in origins:
            for node in graph_store.neighbors(origin, namespace, depth, direction=direction):
                distance.setdefault(str(node["id"]), depth)
    return distance


def bfs_entity_ranking(
    graph_store: Any,
    origins: list[str],
    namespace: str,
    k: int,
    max_depth: int = MAX_SEARCH_DEPTH,
) -> list[tuple[str, float]]:
    """Entity nodes within ``max_depth`` hops of ``origins`` (upstream
    ``node_bfs_search``), ranked nearest-first, excluding the origins
    themselves — they are already in the other channels' results, and
    re-listing them would just double their RRF weight.

    Scores descend with distance so the list reads like any other channel's;
    only the ORDER is consumed downstream."""
    if graph_store is None or not origins or max_depth < 1:
        return []
    distance = _rings(graph_store, origins, namespace, max_depth)
    seeded = set(origins)
    ranked = sorted(
        ((node_id, hops) for node_id, hops in distance.items() if node_id not in seeded),
        key=lambda item: (item[1], item[0]),
    )
    return [(node_id, 1.0 / (1 + hops)) for node_id, hops in ranked[:k]]


def bfs_fact_ranking(
    graph_store: Any,
    origins: list[str],
    namespace: str,
    k: int,
    max_depth: int = MAX_SEARCH_DEPTH,
    exclude: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Active edges within ``max_depth`` hops of ``origins`` (upstream
    ``edge_bfs_search``), ranked by the closer of their two endpoints.

    ``exclude`` drops edges the other channels already returned, for the same
    reason the entity ranking drops its origins. Ties break on
    ``(valid_at, content)`` — a fact's own text, so the order is stable across
    stores and across runs."""
    if graph_store is None or not origins or max_depth < 1:
        return []
    # Edges within max_depth OUTGOING hops are those whose SOURCE node is
    # reachable in max_depth - 1, so the ring walk stops one short of the edge
    # horizon and edges are keyed on their src (the directed walk traverses an
    # edge src->dst only — an edge merely POINTING AT a reachable node is not
    # on any outgoing path, so it is skipped).
    distance = _rings(graph_store, origins, namespace, max(max_depth - 1, 0))
    edges = graph_store.edges_for_nodes(list(distance), namespace)
    skip = exclude or set()
    scored = []
    for edge in edges:
        edge_id = str(edge.get("id"))
        if edge_id in skip:
            continue
        hops = distance.get(str(edge.get("src")))
        if hops is None:
            continue  # incident via dst only — not reachable on a directed walk
        scored.append(
            (hops, str(edge.get("valid_at") or ""), str(edge.get("content") or ""), edge_id)
        )
    scored.sort()
    return [(edge_id, 1.0 / (1 + hops)) for hops, _, _, edge_id in scored[:k]]
