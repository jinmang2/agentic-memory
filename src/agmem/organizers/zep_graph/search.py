"""Zep search recipes: the read-path side of the methodology, as data.

The paper describes ONE write path but a *menu* of read paths — three search
functions (φ_cos, φ_bm25, φ_bfs; §3.1) and five rerankers (RRF, MMR,
episode-mentions, node-distance, cross-encoder; §3.2) — and upstream ships the
combinations as named ``SearchConfig`` recipes
(``graphiti_core/search/search_config_recipes.py``). Picking one silently is how
a reproduction becomes unfalsifiable, so they are a preset table here, the same
shape ``NEMORI_PRESETS``/``MEMORYOS_PRESETS`` use for their write-path lineages.

Which one is "the paper"? §4.1 settles it: *"Our experimental implementation
employs the BGE-m3 models from BAAI for both reranking and embedding tasks."*
BGE-m3's reranker is a cross-encoder (upstream wires it as
``BGERerankerClient(CrossEncoder('BAAI/bge-reranker-v2-m3'))``), so the DMR and
LongMemEval numbers come from the cross-encoder recipe — which is also the only
family whose ``search_methods`` include BFS. Hence ``DEFAULT_RECIPE`` below.
The RRF recipes are not a lesser variant: ``EDGE_HYBRID_SEARCH_RRF`` is what
upstream's plain ``search()`` uses, so it is what a Graphiti user gets by
default, and the paper lists RRF first among supported rerankers. Both are worth
measuring; that is the point of a table.

Note what §4 says about scope: *"While these experiments demonstrate key
retrieval capabilities of Graphiti, they represent a subset of the system's full
search functionality."* The paper contains no ablation over search functions or
rerankers, so no recipe here can claim "the paper's ablation" — only "the
paper's operating point" (cross-encoder) versus "upstream's default" (RRF)
versus mechanisms the paper describes but never measures (MMR,
episode-mentions, node-distance).

A recipe carries only read-path settings. It does not touch the organizer, which
is why one ingest can be measured under several recipes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Upstream search_utils.DEFAULT_SEARCH_LIMIT / MAX_SEARCH_DEPTH.
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_DEPTH = 3

# The paper's reranking model (§4.1), as upstream's BGERerankerClient loads it.
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# The three subgraphs φ returns (paper §3.1: "a 3-tuple containing lists of
# semantic edges, entity nodes, and community nodes"), in the order the bundle
# should serve them. `episodic` is deliberately absent: Zep's context template
# has no raw-message section, and including it turns the config into the mixed
# ablation docs/10 bars from paper-reproduction claims.
SUBGRAPH_TYPES = ("facts", "entities", "communities")


@dataclass(frozen=True)
class SearchRecipe:
    """One upstream ``SearchConfig``, expressed as this project's read-path knobs.

    ``lexical_types``/``bfs_types`` are per-type because upstream's recipes are:
    the cross-encoder recipe gives edges and nodes a BFS channel but never gives
    communities one (``CommunitySearchMethod`` has no ``bfs`` member at all).
    """

    name: str  # upstream constant name, so a stamped run is traceable
    memory_types: tuple[str, ...]
    lexical_types: tuple[str, ...]
    bfs_types: tuple[str, ...] = ()
    bfs_max_depth: int = MAX_SEARCH_DEPTH
    reranker: str | None = None  # None -> keep RRF fusion order (NoopReranker)
    reranker_params: dict[str, Any] = field(default_factory=dict)
    limit: int = DEFAULT_SEARCH_LIMIT
    note: str = ""

    def config_kwargs(self) -> dict[str, Any]:
        """The ``AgmemConfig`` fields this recipe fixes.

        ``graph_expansion_cap=0`` is explicit rather than inherited: GraphRecall
        was this project's stand-in for φ_bfs, and leaving it on beside the real
        channel would double-serve the same edges through two mechanisms, only
        one of which upstream has."""
        kwargs: dict[str, Any] = {
            "lexical_types": self.lexical_types,
            "bfs_types": self.bfs_types,
            "bfs_max_depth": self.bfs_max_depth,
            "reranker_params": dict(self.reranker_params),
            "graph_expansion_cap": 0,
        }
        if self.reranker is not None:
            kwargs["overrides"] = {"reranker": self.reranker}
        return kwargs


# Every combination upstream names, restricted to the "combined" ones (all three
# subgraphs) plus the single-subgraph recipes that are the only place two of the
# rerankers appear. Transcribed field by field from search_config_recipes.py.
ZEP_SEARCH_RECIPES: dict[str, SearchRecipe] = {
    # --- all three subgraphs -------------------------------------------------
    "cross_encoder": SearchRecipe(
        name="COMBINED_HYBRID_SEARCH_CROSS_ENCODER",
        memory_types=SUBGRAPH_TYPES,
        lexical_types=SUBGRAPH_TYPES,
        # edges and nodes get BFS; communities do not (upstream has no
        # CommunitySearchMethod.bfs)
        bfs_types=("facts", "entities"),
        reranker="CrossEncoderReranker",
        reranker_params={"model_name": BGE_RERANKER_MODEL},
        note="paper §4 operating point (§4.1: BGE-m3 for reranking and embedding)",
    ),
    "rrf": SearchRecipe(
        name="COMBINED_HYBRID_SEARCH_RRF",
        memory_types=SUBGRAPH_TYPES,
        lexical_types=SUBGRAPH_TYPES,
        # no BFS anywhere in the RRF family — this is the difference that makes
        # it a distinct recipe rather than a cheaper reranker
        reranker=None,
        note="upstream's own default family; RRF is the first reranker the paper lists",
    ),
    "mmr": SearchRecipe(
        name="COMBINED_HYBRID_SEARCH_MMR",
        memory_types=SUBGRAPH_TYPES,
        lexical_types=SUBGRAPH_TYPES,
        reranker="MMRReranker",
        # upstream sets mmr_lambda=1 in this recipe, i.e. pure relevance: the
        # diversity term is switched OFF in the shipped configuration
        reranker_params={"lambda_": 1.0},
        note="paper §3.2 mechanism, never measured there",
    ),
    # --- single subgraph: the only recipes carrying these two rerankers ------
    "edge_episode_mentions": SearchRecipe(
        name="EDGE_HYBRID_SEARCH_EPISODE_MENTIONS",
        memory_types=("facts",),
        lexical_types=("facts",),
        reranker="EpisodeMentionsReranker",
        note="paper §3.2: frequently-referenced information becomes more accessible",
    ),
    "node_distance": SearchRecipe(
        name="NODE_HYBRID_SEARCH_NODE_DISTANCE",
        memory_types=("entities",),
        lexical_types=("entities",),
        reranker="NodeDistanceReranker",
        note="paper §3.2; needs a per-query centroid (search(center_node_id=...))",
    ),
    "edge_rrf": SearchRecipe(
        name="EDGE_HYBRID_SEARCH_RRF",
        memory_types=("facts",),
        lexical_types=("facts",),
        reranker=None,
        note="what upstream's plain search() does",
    ),
}

# The recipe a bare "reproduce Zep" means, per §4.1.
DEFAULT_RECIPE = "cross_encoder"


def zep_search_recipe(name: str = DEFAULT_RECIPE) -> SearchRecipe:
    """Look up a recipe, raising on an unknown name.

    No silent fallback to the default: a typo'd recipe name would otherwise
    produce a run stamped with settings nobody asked for, which is the failure
    mode this table exists to prevent."""
    try:
        return ZEP_SEARCH_RECIPES[name]
    except KeyError:
        raise KeyError(
            f"unknown Zep search recipe {name!r}; known: {sorted(ZEP_SEARCH_RECIPES)}"
        ) from None
