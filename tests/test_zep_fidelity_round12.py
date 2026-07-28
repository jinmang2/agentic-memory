"""Round-12 pinning tests: Zep/Graphiti temporal truth table, dedup fast paths,
batched resolution shape, RRF constant, and directed BFS.

Each test pins one of the round-12 fidelity fixes
(docs/research/fidelity-round12-fresh-eyes-reaudit.md, # [zep]) so a later
refactor cannot silently revert to the pre-audit behavior."""

import pytest
from helpers import StubLLM

from agmem import AgenticMemory
from agmem.core.ops import MemoryOp, OpType
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.zep_graph import ZEP_SEARCH_RECIPES, ZepGraphOrganizer
from agmem.organizers.zep_graph.dedup import build_candidate_indexes, deterministic_resolve
from agmem.organizers.zep_graph.organizer import (
    NODE_DEDUP_CANDIDATE_LIMIT,
    expire_new_edge,
    resolve_edge_contradictions,
)
from agmem.retrieval.bfs import bfs_fact_ranking
from agmem.retrieval.fusion import rrf_fuse
from agmem.stores.sqlite_graph import SqliteGraphStore


def make_mem(organizer, llm):
    mem = AgenticMemory(namespace="t", organizers=[organizer], embedder=FakeEmbedder(dim=128))
    mem.structured = llm
    mem._ctx.llm = llm
    return mem


def ops_of(mem, ttype):
    return [o for o in mem.log.tail(100) if o.target_type == ttype]


# ---------------- temporal truth table (findings 5/6) ----------------


def test_zep_strictly_older_truth_table():
    """Upstream resolve_edge_contradictions (edge_operations.py:538-573):
    after the two skip conditions, an edge is invalidated ONLY when its
    valid_at is STRICTLY older than the new fact's — equal, newer, and None
    all survive."""
    older = {"id": "older", "valid_at": "2020-01-01", "invalid_at": None}
    equal = {"id": "equal", "valid_at": "2024-01-01", "invalid_at": None}
    newer = {"id": "newer", "valid_at": "2025-01-01", "invalid_at": None}
    dateless = {"id": "dateless", "valid_at": None, "invalid_at": None}
    new_edge = {"valid_at": "2024-01-01", "invalid_at": None}

    out = resolve_edge_contradictions(new_edge, [older, equal, newer, dateless])
    assert [e["id"] for e in out] == ["older"]
    # t_invalid = the invalidating fact's valid_at, stamped on the local view
    assert older["invalid_at"] == "2024-01-01"

    # skip condition (a): the candidate had already ended before the new fact began
    ended = {"id": "ended", "valid_at": "2019-01-01", "invalid_at": "2023-01-01"}
    assert (
        resolve_edge_contradictions({"valid_at": "2024-01-01", "invalid_at": None}, [ended]) == []
    )
    # skip condition (b): the new fact ended before the candidate began
    later_start = {"id": "later", "valid_at": "2020-06-01", "invalid_at": None}
    assert (
        resolve_edge_contradictions(
            {"valid_at": "2020-01-01", "invalid_at": "2020-03-01"}, [later_start]
        )
        == []
    )


def test_zep_none_valid_at_is_inert_both_ways():
    """A None valid_at can neither invalidate nor be invalidated — every
    upstream condition requires non-None (finding 6, no `or ref_time` default)."""
    candidate = {"id": "c", "valid_at": "2020-01-01", "invalid_at": None}
    assert resolve_edge_contradictions({"valid_at": None, "invalid_at": None}, [candidate]) == []
    assert candidate["invalid_at"] is None

    dateless = {"id": "d", "valid_at": None, "invalid_at": None}
    assert (
        resolve_edge_contradictions({"valid_at": "2024-01-01", "invalid_at": None}, [dateless])
        == []
    )

    # ...and a None-valid_at new edge is never self-expired either
    new_edge = {"valid_at": None, "invalid_at": None}
    expire_new_edge(new_edge, [candidate])
    assert new_edge["invalid_at"] is None


def test_zep_new_edge_self_expiry_picks_the_earliest_later_candidate():
    """Upstream edge_operations.py:826-841: a candidate with a strictly LATER
    valid_at expires the NEW edge at write time — invalid_at = the EARLIEST
    such candidate's valid_at (upstream sorts by valid_at, None last)."""
    new_edge = {"valid_at": "2024-01-01", "invalid_at": None}
    expire_new_edge(
        new_edge,
        [
            {"valid_at": "2026-01-01"},
            {"valid_at": None},
            {"valid_at": "2025-01-01"},
            {"valid_at": "2020-01-01"},  # older: cannot expire the new edge
        ],
    )
    assert new_edge["invalid_at"] == "2025-01-01"

    # a model-supplied invalid_at wins: upstream only self-expires when the
    # resolved edge is not already expired
    already = {"valid_at": "2024-01-01", "invalid_at": "2024-06-01"}
    expire_new_edge(already, [{"valid_at": "2025-01-01"}])
    assert already["invalid_at"] == "2024-06-01"


def test_zep_self_expired_new_edge_enters_the_graph_inactive():
    """Integration: a new fact contradicted by already-known NEWER information
    is written already expired (invalid_at = the newer fact's valid_at), and
    the newer fact is NOT invalidated (finding 5b)."""
    llm = StubLLM(
        {
            "extract": [
                {
                    "entities": [
                        {"name": "Alice", "summary": "s"},
                        {"name": "Paris", "summary": "s"},
                    ]
                },
                {
                    "facts": [
                        {
                            "subject": "Alice",
                            "predicate": "lives_in",
                            "object": "Paris",
                            "statement": "Alice lives in Paris.",
                            "valid_at": "2025-01-01T00:00:00",
                        }
                    ]
                },
                {
                    "entities": [
                        {"name": "Alice", "summary": "s"},
                        {"name": "London", "summary": "s"},
                    ]
                },
                {
                    "facts": [
                        {
                            "subject": "Alice",
                            "predicate": "moved_to",
                            "object": "London",
                            "statement": "Alice moved to London.",
                            "valid_at": "2024-01-01T00:00:00",
                        }
                    ]
                },
            ],
            "distill": [{"duplicate_of": None, "contradicts": ["__EDGE__"]}],
        }
    )
    org = ZepGraphOrganizer(community_refresh=False)
    mem = make_mem(org, llm)
    try:
        mem.add_message("Alice lives in Paris.")
        paris_edge = ops_of(mem, "facts")[0].target_id
        llm.responses["distill"][0]["contradicts"] = [paris_edge]
        mem.add_message("Alice moved to London.")

        adds = [o for o in ops_of(mem, "facts") if o.op is OpType.ADD]
        assert len(adds) == 2
        london = adds[1].payload
        assert london["invalid_at"] == "2025-01-01T00:00:00"  # expired at write time
        # the NEWER Paris fact survives untouched
        assert [o for o in ops_of(mem, "facts") if o.op is OpType.INVALIDATE] == []
        alice = org.graph.find_node_by_name("Alice", "t")["id"]
        london_node = org.graph.find_node_by_name("London", "t")["id"]
        assert org.graph.edges_between(alice, london_node, "t") == []  # never active
    finally:
        mem.close()


def test_zep_cross_pair_contradiction_reaches_the_invalidation_pool():
    """Finding 4's flagship case: `Alice LIVES_IN Paris` IS invalidated when
    `Alice MOVED_TO London` arrives as an edge to a DIFFERENT node — the
    invalidation candidates come from the graph-wide dense search, not from
    the same entity pair."""
    llm = StubLLM(
        {
            "extract": [
                {
                    "entities": [
                        {"name": "Alice", "summary": "s"},
                        {"name": "Paris", "summary": "s"},
                    ]
                },
                {
                    "facts": [
                        {
                            "subject": "Alice",
                            "predicate": "lives_in",
                            "object": "Paris",
                            "statement": "Alice lives in Paris.",
                            "valid_at": "2020-01-01T00:00:00",
                        }
                    ]
                },
                {
                    "entities": [
                        {"name": "Alice", "summary": "s"},
                        {"name": "London", "summary": "s"},
                    ]
                },
                {
                    "facts": [
                        {
                            "subject": "Alice",
                            "predicate": "moved_to",
                            "object": "London",
                            "statement": "Alice moved to London.",
                            "valid_at": "2024-01-01T00:00:00",
                        }
                    ]
                },
            ],
            "distill": [{"duplicate_of": None, "contradicts": ["__EDGE__"]}],
        }
    )
    org = ZepGraphOrganizer(community_refresh=False)
    mem = make_mem(org, llm)
    try:
        mem.add_message("Alice lives in Paris.")
        paris_edge = ops_of(mem, "facts")[0].target_id
        llm.responses["distill"][0]["contradicts"] = [paris_edge]
        mem.add_message("Alice moved to London.")

        # the resolve prompt carried the cross-pair candidate
        resolve_prompts = [p for role, p in llm.calls if role == "distill"]
        assert len(resolve_prompts) == 1 and paris_edge in resolve_prompts[0]

        invalidations = [o for o in ops_of(mem, "facts") if o.op is OpType.INVALIDATE]
        assert [o.target_id for o in invalidations] == [paris_edge]
        assert invalidations[0].payload["t_invalid"] == "2024-01-01T00:00:00"
        alice = org.graph.find_node_by_name("Alice", "t")["id"]
        paris = org.graph.find_node_by_name("Paris", "t")["id"]
        assert org.graph.edges_between(alice, paris, "t") == []  # no longer active
    finally:
        mem.close()


# ---------------- dedup fast paths (finding 7) ----------------


def test_zep_verbatim_duplicate_skips_the_llm():
    """Normalized fact text + directional endpoints matching an existing edge
    exactly reuse it with a provenance append and NO resolve call (upstream
    edge_operations.py:687-700)."""
    llm = StubLLM(
        {
            "extract": [
                {"entities": [{"name": "Ann", "summary": "s"}, {"name": "Bob", "summary": "s"}]},
                {
                    "facts": [
                        {
                            "subject": "Ann",
                            "predicate": "works_with",
                            "object": "Bob",
                            "statement": "Ann works with Bob.",
                        }
                    ]
                },
                {"entities": [{"name": "Ann", "summary": "s"}, {"name": "Bob", "summary": "s"}]},
                {
                    "facts": [
                        {
                            "subject": "Ann",
                            "predicate": "works_with",
                            "object": "Bob",
                            # case/whitespace variant of the same normalized text
                            "statement": "ANN  works with  Bob.",
                        }
                    ]
                },
            ],
            "distill": [],
        }
    )
    org = ZepGraphOrganizer(community_refresh=False)
    mem = make_mem(org, llm)
    try:
        mem.add_message("Ann works with Bob.")
        first_edge = ops_of(mem, "facts")[0].target_id
        mem.add_message("Ann works with Bob again.")

        assert [role for role, _ in llm.calls if role == "distill"] == []  # no LLM resolve
        adds = [o for o in ops_of(mem, "facts") if o.op is OpType.ADD]
        assert len(adds) == 1  # reused, not re-added
        update = next(o for o in ops_of(mem, "facts") if o.op is OpType.UPDATE)
        assert update.target_id == first_edge
        assert len(update.payload["source_episode_ids"]) == 2  # provenance appended
    finally:
        mem.close()


def test_zep_in_batch_identical_extractions_collapse():
    """Identical (endpoints, normalized fact) extractions within ONE message
    are pre-deduped before any resolution work (upstream
    edge_operations.py:344-358)."""
    fact = {
        "subject": "Ann",
        "predicate": "works_with",
        "object": "Bob",
        "statement": "Ann works with Bob.",
    }
    llm = StubLLM(
        {
            "extract": [
                {"entities": [{"name": "Ann", "summary": "s"}, {"name": "Bob", "summary": "s"}]},
                {"facts": [fact, dict(fact, statement="ann  works with bob.")]},
            ],
            "distill": [],
        }
    )
    org = ZepGraphOrganizer(community_refresh=False)
    mem = make_mem(org, llm)
    try:
        mem.add_message("Ann works with Bob.")
        assert len([o for o in ops_of(mem, "facts") if o.op is OpType.ADD]) == 1
        assert [role for role, _ in llm.calls if role == "distill"] == []
    finally:
        mem.close()


# ---------------- entity resolution (findings 1/3/15) ----------------


def test_zep_minhash_folds_near_duplicates_without_llm():
    """The deterministic stage's fuzzy sub-stage (dedup_helpers.py:220-280):
    a punctuation variant folds at Jaccard 1.0 with no LLM; a sub-threshold
    near-duplicate (Jaccard < 0.9) escalates; an AMBIGUOUS exact match
    escalates instead of first-wins; whitespace-collapsed exact match holds."""
    idx = build_candidate_indexes([{"id": "c1", "name": "John A. Smith"}])
    assert deterministic_resolve("John A Smith", idx) == "c1"  # fuzzy, deterministic
    assert deterministic_resolve("Jane Doe", idx) is None

    # "Katherine"/"Katharine": 3-gram Jaccard ~0.65 < 0.9 — the LLM decides
    idx2 = build_candidate_indexes([{"id": "k", "name": "Katherine Johnson"}])
    assert deterministic_resolve("Katharine Johnson", idx2) is None

    # >1 candidates sharing the normalized name -> escalate (dedup_helpers.py:245-249)
    idx3 = build_candidate_indexes(
        [{"id": "a", "name": "John Smith"}, {"id": "b", "name": "john  smith"}]
    )
    assert deterministic_resolve("John Smith", idx3) is None

    # exact normalization = lowercase + whitespace-collapse (dedup_helpers.py:39-42)
    idx4 = build_candidate_indexes([{"id": "a", "name": "John  Smith"}])
    assert deterministic_resolve("john smith", idx4) == "a"


class _SpyVectorStore:
    """Wraps the real vector store, recording (memory_type, k) per search."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[tuple[str, int]] = []

    def search(self, embedding, k, memory_type=None, namespace=None):
        self.calls.append((memory_type, k))
        return self._inner.search(embedding, k=k, memory_type=memory_type, namespace=namespace)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_zep_dedupe_is_one_batched_call_with_k15_candidates():
    """Two unresolved entities in one message cost ONE dedupe LLM call
    (node_operations.py:552-556) against candidates fetched at
    NODE_DEDUP_CANDIDATE_LIMIT = 15 (node_operations.py:64) — call-count
    parity with upstream is load-bearing for any cost comparison."""

    class BatchStub:
        def __init__(self):
            self.calls: list[tuple[str, str]] = []

        def call(self, role, prompt, schema, required_keys=(), **kwargs):
            self.calls.append((role, prompt))
            if "Extract the distinct real-world entities" in prompt:
                return {
                    "entities": [
                        {"name": "Alpha One Inc", "summary": "the alpha firm"},
                        {"name": "Beta Two Corp", "summary": "the beta firm"},
                    ]
                }
            if "Decide for each NEW entity" in prompt:
                return {
                    "resolutions": [
                        {"id": 0, "duplicate_candidate_id": 0, "name": "Alpha One", "summary": "m"},
                        {"id": 1, "duplicate_candidate_id": -1},
                    ]
                }
            if "Extract relationship facts" in prompt:
                return {"facts": []}
            raise AssertionError(f"unexpected prompt: {prompt[:60]!r}")

    llm = BatchStub()
    org = ZepGraphOrganizer(community_refresh=False)
    mem = make_mem(org, llm)
    try:
        # existing near-duplicates: cosine >= 0.6 to the new names under
        # FakeEmbedder (2 of 3 tokens shared), but neither exact nor
        # fuzzy-matchable, so both new entities reach the LLM stage
        for node_id, name in (("E1", "Alpha One Corp"), ("E2", "Beta Two Inc")):
            mem._apply_one(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="entities",
                    target_id=node_id,
                    payload={
                        "id": node_id,
                        "name": name,
                        "summary": "seeded",
                        "content": name,
                        "embedding_text": name,
                        "source_episode_ids": [],
                    },
                )
            )
        spy = _SpyVectorStore(mem.vector_store)
        mem._ctx.vector_store = spy
        mem.add_message("Alpha One Inc partners with Beta Two Corp.")

        resolve_calls = [p for role, p in llm.calls if "Decide for each NEW entity" in p]
        assert len(resolve_calls) == 1  # ONE batched call for BOTH entities
        assert 'id=0 name="Alpha One Inc"' in resolve_calls[0]
        assert 'id=1 name="Beta Two Corp"' in resolve_calls[0]
        # candidate search ran at upstream's limit
        entity_searches = [k for mt, k in spy.calls if mt == "entities"]
        assert entity_searches and all(k == NODE_DEDUP_CANDIDATE_LIMIT for k in entity_searches)
        assert NODE_DEDUP_CANDIDATE_LIMIT == 15

        # id 0 merged into E1 (refresh op), id 1 became a new node
        updates = [o for o in ops_of(mem, "entities") if o.op is OpType.UPDATE]
        assert [o.target_id for o in updates] == ["E1"]
        assert updates[0].payload["name"] == "Alpha One"
        new_names = {
            o.payload["name"]
            for o in ops_of(mem, "entities")
            if o.op is OpType.ADD and o.target_id not in ("E1", "E2")
        }
        assert new_names == {"Beta Two Corp"}
    finally:
        mem.close()


# ---------------- RRF constant (finding 8) ----------------


def test_rrf_k_1_flips_the_1_100_vs_3_3_ordering():
    """The constant is not a monotone rescale across channels: an item ranked
    (1,100) and one ranked (3,3) order differently under the textbook k=60
    and upstream's steep constant (rank_const=1, search_utils.py:1780-1786)."""

    def ranking(ids):
        return [(item_id, 1.0) for item_id in ids]

    ch1 = ranking(["X", "a1", "Y"])  # X rank 1, Y rank 3
    ch2 = ranking(["b1", "b2", "Y"] + [f"b{i}" for i in range(3, 99)] + ["X"])  # X rank 100

    order60 = [item_id for item_id, _ in rrf_fuse([ch1, ch2], k=60)]
    order1 = [item_id for item_id, _ in rrf_fuse([ch1, ch2], k=1)]
    assert order60.index("Y") < order60.index("X")  # consistency wins at 60
    assert order1.index("X") < order1.index("Y")  # the rank-1 hit wins at 1


def test_zep_recipes_emit_upstream_rrf_and_min_score_constants():
    from agmem.config import AgmemConfig

    for recipe in ZEP_SEARCH_RECIPES.values():
        kwargs = recipe.config_kwargs()
        assert kwargs["rrf_k"] == 1  # upstream rank_const
        assert kwargs["dense_min_score"] == 0.6  # upstream DEFAULT_MIN_SCORE
    # ...while the framework defaults stay textbook / off
    assert AgmemConfig().rrf_k == 60
    assert AgmemConfig().dense_min_score == 0.0


# ---------------- directed BFS (finding 12) ----------------


def _stores():
    yield SqliteGraphStore()
    try:
        from agmem.stores.kuzu_graph import KuzuGraphStore

        yield KuzuGraphStore()
    except ImportError:
        pass


@pytest.mark.parametrize("store", list(_stores()), ids=lambda s: type(s).__name__)
def test_zep_bfs_direction_out_walks_source_to_target_only(store):
    """Upstream's BFS is `-[:RELATES_TO|MENTIONS*1..n]->` — outgoing only. In
    direction="out", an edge B -> A does not make B reachable from A; the
    generic default "both" keeps the undirected walk for non-Zep callers."""
    for node in ("A", "B", "C"):
        store.upsert_node(node, "t", node)
    store.upsert_edge("e1", "t", "A", "B", "REL", "A rel B")
    store.upsert_edge("e2", "t", "C", "B", "REL", "C rel B")  # points INTO B

    assert {n["id"] for n in store.neighbors("A", "t", 2)} == {"B", "C"}  # undirected
    assert {n["id"] for n in store.neighbors("A", "t", 2, direction="out")} == {"B"}
    assert store.neighbors("B", "t", 2, direction="out") == []  # B has no outgoing edges


def test_zep_bfs_fact_ranking_is_directed():
    """The φ_bfs edge channel counts an edge only when its SOURCE node is
    reachable on the outgoing walk — e2 (C -> B) is incident to reachable B
    but lies on no outgoing path from A, so it is not served."""
    store = SqliteGraphStore()
    for node in ("A", "B", "C"):
        store.upsert_node(node, "t", node)
    store.upsert_edge("e1", "t", "A", "B", "REL", "A rel B")
    store.upsert_edge("e2", "t", "C", "B", "REL", "C rel B")

    ranked = bfs_fact_ranking(store, ["A"], "t", k=10, max_depth=2)
    assert [edge_id for edge_id, _ in ranked] == ["e1"]
