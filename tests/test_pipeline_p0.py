"""Fidelity-audit P0 read-path behaviors (docs/research/fidelity-deep-audit.md §6)."""

from agmem import AgenticMemory
from agmem.core.types import Episode
from agmem.embed.fake import FakeEmbedder


def make_mem():
    return AgenticMemory(namespace="t", organizers=["passthrough"], embedder=FakeEmbedder(dim=128))


def put_indexed(mem, item_id, memory_type, data, text=None):
    mem.doc_store.put_item(item_id, memory_type, "t", {"id": item_id, **data})
    mem.vector_store.add(
        item_id,
        mem.embedder.embed([text or data.get("content", "")])[0],
        memory_type=memory_type,
        namespace="t",
    )


def test_amem_one_hop_link_expansion():
    mem = make_mem()
    try:
        put_indexed(mem, "n1", "notes", {"content": "paris travel museums", "links": ["n2"]})
        # n2 is lexically unrelated to the query — reachable only via the link
        put_indexed(mem, "n2", "notes", {"content": "budget three million won", "links": ["n1"]})
        bundle = mem.search("paris museums", memory_types=["notes"], k=1)
        ids = [s.item.data["id"] for s in bundle.items]
        assert "n1" in ids and "n2" in ids  # neighbor pulled in via 1-hop
        n1 = next(s for s in bundle.items if s.item.data["id"] == "n1")
        n2 = next(s for s in bundle.items if s.item.data["id"] == "n2")
        assert n2.score < n1.score  # neighbor ranks below its parent
    finally:
        mem.close()


def test_link_expansion_per_hit_budget_is_not_the_global_one():
    """The cap SHAPE is A-Mem's remaining read-path deviation, so both are pinned.

    Ours spends one budget across every hit; upstream gives each hit its own and
    breaks only after appending (`memory_layer.py:895` at the pinned SHA), so a
    hit contributes up to k+1 neighbours. With two hits each carrying two
    exclusive neighbours and a cap of 2, the two modes are distinguishable:
    global serves the top hit's two and starves the second, per-hit serves both
    hits' two. If this ever passes with equal sets, the mode is not wired.
    """
    from agmem.retrieval.steps import LinkExpansion, ReadContext

    mem = make_mem()
    try:
        put_indexed(mem, "h1", "notes", {"content": "paris museums", "links": ["a1", "a2"]})
        put_indexed(mem, "h2", "notes", {"content": "paris museums tour", "links": ["b1", "b2"]})
        for nid in ("a1", "a2", "b1", "b2"):
            put_indexed(mem, nid, "notes", {"content": f"neighbour {nid}", "links": []})
        ctx = ReadContext(doc_store=mem.doc_store, namespace="t", vector_store=mem.vector_store)

        def served(step):
            hits = [
                s
                for s in mem.search("paris museums", memory_types=["notes"], k=2).items
                if s.item.data["id"] in ("h1", "h2")
            ]
            assert len(hits) == 2, "both anchors must be retrieved for this to test anything"
            return {s.item.data["id"] for s in step.run(hits, ctx)}

        global_two = served(LinkExpansion(cap=2, per_hit=False))
        per_hit_two = served(LinkExpansion(cap=2, per_hit=True))
        assert len(global_two - {"h1", "h2"}) == 2  # one shared budget of 2
        assert len(per_hit_two - {"h1", "h2"}) == 4  # 2 for each of the 2 hits
        assert global_two < per_hit_two
    finally:
        mem.close()


def test_link_expansion_does_not_resurrect_retired_notes():
    """``LinkExpansion`` was the one read step with neither guard: it pulled
    neighbours by id and appended them unfiltered, so an invalidated note came
    back with full content and a deleted one came back as an empty ghost hit —
    the round-5 X1 failure ``_hydrate``/``ExpandExperiences``/``GraphRecall``
    were already fixed for. Reachable in the default config: ``ChainedConsumer``
    retires a wrapped A-Mem's notes with INVALIDATE (docs/04 §3.3)."""
    from agmem.core.ops import MemoryOp, OpType

    mem = make_mem()
    try:
        put_indexed(mem, "n1", "notes", {"content": "paris travel museums", "links": ["n2", "n3"]})
        put_indexed(mem, "n2", "notes", {"content": "budget three million won"})
        put_indexed(mem, "n3", "notes", {"content": "flight departs at nine"})
        assert {
            s.item.data["id"]
            for s in mem.search("paris museums", memory_types=["notes"], k=1).items
        } == {
            "n1",
            "n2",
            "n3",
        }

        mem._apply_ops(
            [MemoryOp(op=OpType.INVALIDATE, target_type="notes", target_id="n2", payload={})],
            actor="chained",
        )
        mem._apply_ops(
            [MemoryOp(op=OpType.DELETE, target_type="notes", target_id="n3", payload={})],
            actor="chained",
        )
        served = mem.search("paris museums", memory_types=["notes"], k=1).items
        assert {s.item.data["id"] for s in served} == {"n1"}
        assert all(s.item.content for s in served)  # no empty ghost
    finally:
        mem.close()


def test_lexical_channel_drops_invalidated_items_but_keeps_facts():
    """The vector is dropped on INVALIDATE, so dense recall misses these — but
    the lexical channel re-fetches by id, and ``_hydrate`` only filtered
    tombstones. Adding a derived type to ``lexical_types`` (the documented Zep
    hybrid knob) therefore resurrected invalidated items. ``facts`` must NOT be
    dropped: Zep renders them with their validity range instead."""
    from agmem.config import AgmemConfig
    from agmem.core.ops import MemoryOp, OpType

    mem = AgenticMemory(
        namespace="t",
        organizers=["passthrough"],
        embedder=FakeEmbedder(dim=128),
        config=AgmemConfig(lexical_types=("episodic", "semantic", "facts")),
    )
    try:
        put_indexed(mem, "s1", "semantic", {"content": "user lives in Paris"})
        put_indexed(
            mem, "f1", "facts", {"content": "user lives in Paris", "valid_at": "2024-01-01"}
        )
        for target_type, target_id in (("semantic", "s1"), ("facts", "f1")):
            mem._apply_ops(
                [
                    MemoryOp(
                        op=OpType.INVALIDATE,
                        target_type=target_type,
                        target_id=target_id,
                        payload={"t_invalid": "2025-06-01"},
                    )
                ],
                actor="nemori",
            )
        bundle = mem.search("Paris", memory_types=["semantic", "facts"])
        assert [s.item.data["id"] for s in bundle.items] == ["f1"]
        assert "Date range: 2024-01-01 - 2025-06-01" in bundle.render(200)
    finally:
        mem.close()


def test_link_expansion_capped():
    mem = make_mem()
    try:
        links = [f"x{i}" for i in range(10)]
        put_indexed(mem, "hub", "notes", {"content": "hub note", "links": links})
        for lid in links:
            put_indexed(mem, lid, "notes", {"content": f"leaf {lid}", "links": []})
        bundle = mem.search("hub note", memory_types=["notes"], k=1)
        # the cap now lives on the registered read step (retrieval/steps.py)
        assert len(bundle.items) <= 1 + mem.pipeline.read_steps["notes"].cap
    finally:
        mem.close()


def test_link_application_preserves_insertion_order_and_duplicates():
    """Round-12 finding 3: upstream is ``note.links.extend`` in both editions
    (memory_layer.py:835) — insertion order, duplicates kept — and the read path
    consumes links in stored order under the cap, so overflow selection is
    first-linked-wins. The old ``sorted(set())`` LINK merge silently made it
    lowest-id-wins. Deliberate residual deviation, asserted here: upstream's
    read (memory_layer.py:889-897) has no dedup and would serve a duplicate
    twice while burning a cap slot per occurrence; our seen-set serves it once
    and the duplicate burns no cap budget."""
    from agmem.config import AgmemConfig
    from agmem.core.ops import MemoryOp, OpType

    mem = AgenticMemory(
        namespace="t",
        organizers=["passthrough"],
        embedder=FakeEmbedder(dim=128),
        config=AgmemConfig(link_expansion_cap=2),
    )
    try:
        put_indexed(mem, "hub", "notes", {"content": "hub note", "links": []})
        for lid in ("z9", "a1", "m5"):
            put_indexed(mem, lid, "notes", {"content": f"leaf {lid}", "links": []})
        # two evolutions, as two LINK ops: z9 first, then a duplicate z9 + a1 + m5
        for payload in (["z9"], ["z9", "a1", "m5"]):
            mem._apply_ops(
                [
                    MemoryOp(
                        op=OpType.LINK,
                        target_type="notes",
                        target_id="hub",
                        payload={"links": payload},
                    )
                ],
                actor="amem",
            )
        hub = mem.doc_store.get_items(["hub"], "notes")[0]
        assert hub["links"] == ["z9", "z9", "a1", "m5"]  # order + duplicate survive storage

        bundle = mem.search("hub note", memory_types=["notes"], k=1)
        served = [s.item.data["id"] for s in bundle.items]
        # cap=2 overflow is first-linked-wins: z9 then a1; the duplicate z9 is
        # served exactly once and burns no cap slot; m5 overflows. The old
        # sorted-set merge would have served {a1, m5} here.
        assert sorted(served) == ["a1", "hub", "z9"]
        assert served.count("z9") == 1
    finally:
        mem.close()


def test_read_step_caps_are_configurable():
    """The A-Mem link cap and Nemori's r are documented upstream deviations, so
    they must be ablatable from config — they used to be pipeline constructor
    defaults no caller could reach."""
    from agmem.config import AgmemConfig

    mem = AgenticMemory(
        namespace="t",
        organizers=["passthrough"],
        embedder=FakeEmbedder(dim=128),
        config=AgmemConfig(link_expansion_cap=2),
    )
    try:
        links = [f"x{i}" for i in range(10)]
        put_indexed(mem, "hub", "notes", {"content": "hub note", "links": links})
        for lid in links:
            put_indexed(mem, lid, "notes", {"content": f"leaf {lid}", "links": []})
        bundle = mem.search("hub note", memory_types=["notes"], k=1)
        assert len(bundle.items) == 3  # 1 hit + cap 2, not the default 5
    finally:
        mem.close()


def test_zero_cap_disables_the_step():
    """0 keeps the falsy-cap disable the old `and self.link_expansion_cap`
    guard provided: no expansion step is registered at all."""
    from agmem.config import AgmemConfig

    mem = AgenticMemory(
        namespace="t",
        organizers=["passthrough"],
        embedder=FakeEmbedder(dim=128),
        config=AgmemConfig(link_expansion_cap=0),
    )
    try:
        put_indexed(mem, "n1", "notes", {"content": "paris travel museums", "links": ["n2"]})
        put_indexed(mem, "n2", "notes", {"content": "budget three million won", "links": ["n1"]})
        assert "notes" not in mem.pipeline.read_steps
        bundle = mem.search("paris museums", memory_types=["notes"], k=1)
        assert [s.item.data["id"] for s in bundle.items] == ["n1"]  # no neighbor pulled
    finally:
        mem.close()


def test_expanded_items_are_not_served_twice():
    """A step may emit a type a later pass also searches: ExpandExperiences
    replaces an experience with its strategy items, and reasoning_bank declares
    both types, so `strategies` got rendered twice into the QA prompt. Reachable
    via the memory_types=None default (MCP), never by the explicit-type bench
    calls."""
    mem = AgenticMemory(
        namespace="t", organizers=["reasoning_bank"], embedder=FakeEmbedder(dim=128)
    )
    try:
        assert mem.default_memory_types == ("episodic", "experiences", "strategies")
        put_indexed(mem, "s1", "strategies", {"content": "prefer the search box"})
        put_indexed(mem, "s2", "strategies", {"content": "verify the cart total"})
        put_indexed(
            mem, "x1", "experiences", {"content": "checkout flow", "item_ids": ["s1", "s2"]}
        )
        bundle = mem.search("checkout search box cart total")
        served = [(s.memory_type, s.item.data["id"]) for s in bundle.items]
        assert sorted(served) == [("strategies", "s1"), ("strategies", "s2")]
        # the render is the actual contract — it goes verbatim into the prompt
        assert bundle.render().count("prefer the search box") == 1
    finally:
        mem.close()


def test_two_experiences_sharing_a_strategy_serve_it_once():
    """The same duplication within a single pass: one strategy reachable from two
    retrieved experiences."""
    mem = AgenticMemory(
        namespace="t", organizers=["reasoning_bank"], embedder=FakeEmbedder(dim=128)
    )
    try:
        put_indexed(mem, "s1", "strategies", {"content": "prefer the search box"})
        put_indexed(mem, "x1", "experiences", {"content": "checkout flow", "item_ids": ["s1"]})
        put_indexed(mem, "x2", "experiences", {"content": "checkout retry", "item_ids": ["s1"]})
        bundle = mem.search("checkout flow retry", memory_types=["experiences"])
        assert [s.item.data["id"] for s in bundle.items] == ["s1"]
    finally:
        mem.close()


def test_cross_type_id_collision_is_not_deduped():
    """Dedup keys on (memory_type, id), never the bare id: the items table is
    keyed (id, memory_type), so one id under two types is two distinct items.

    The `shared` note is reached by link expansion (a doc-store fetch) rather than
    the vector index, because the vector stores upsert by bare `item_id` and would
    otherwise let the `pages` row overwrite the `notes` one."""
    mem = make_mem()
    try:
        put_indexed(mem, "n1", "notes", {"content": "a note about paris", "links": ["shared"]})
        mem.doc_store.put_item("shared", "notes", "t", {"id": "shared", "content": "linked note"})
        put_indexed(mem, "shared", "pages", {"content": "a page about paris"})
        bundle = mem.search("paris", memory_types=["notes", "pages"])
        served = [(s.memory_type, s.item.data["id"]) for s in bundle.items]
        assert ("notes", "shared") in served and ("pages", "shared") in served
    finally:
        mem.close()


def test_nemori_top_r_source_attachment():
    mem = make_mem()
    try:
        raw = Episode(content="I paid exactly 2,340,000 won for flights", namespace="t")
        mem.doc_store.add_episode(raw)
        put_indexed(
            mem,
            "ep1",
            "episodes",
            {
                "title": "Flight booking",
                "content": "The user booked flights to Paris.",
                "source_episode_ids": [raw.id],
            },
        )
        bundle = mem.search("flight booking", memory_types=["episodes"], k=1)
        rendered = bundle.render(budget_tokens=6000)
        assert "Source Messages:" in rendered
        assert "2,340,000" in rendered  # verbatim detail restored from raw
    finally:
        mem.close()


def test_render_exposes_metadata():
    mem = make_mem()
    try:
        put_indexed(
            mem,
            "n1",
            "notes",
            {
                "content": "likes museums",
                "context": "user preference",
                "tags": ["travel", "art"],
                "links": [],
            },
        )
        put_indexed(
            mem,
            "s1",
            "semantic",
            {"content": "The user lives in Seoul.", "timestamp": "2023-05-07T13:00:00"},
        )
        rendered = mem.search("museums seoul", memory_types=["notes", "semantic"], k=3).render(
            budget_tokens=6000
        )
        assert "context: user preference" in rendered
        assert "tags: travel, art" in rendered
        assert "(2023-05-07T13:00:00)" in rendered
    finally:
        mem.close()


def test_per_type_k_dict():
    mem = make_mem()
    try:
        for i in range(6):
            put_indexed(mem, f"e{i}", "episodes", {"content": f"episode about cats {i}"})
            put_indexed(mem, f"s{i}", "semantic", {"content": f"fact about cats {i}"})
        bundle = mem.search(
            "cats", memory_types=["episodes", "semantic"], k={"episodes": 2, "semantic": 5}
        )
        by_type = {}
        for s in bundle.items:
            by_type[s.memory_type] = by_type.get(s.memory_type, 0) + 1
        assert by_type["episodes"] == 2 and by_type["semantic"] == 5
    finally:
        mem.close()


def test_bfs_channel_depth_reaches_the_second_ring():
    """Zep's third search function as a channel (`retrieval/bfs.py`, φ_bfs).

    Depth is upstream's ``bfs_max_depth`` (``MAX_SEARCH_DEPTH = 3``). Origins are
    the subject nodes of the facts the other channels found, so on a chain
    A-B-C a query matching the A-B fact seeds at A: at depth 1 the ring is A
    alone and BFS adds nothing beyond what dense already returned, while at
    depth 2 it reaches B and the B-C fact joins as a ranked candidate.

    ``GraphRecall`` is deliberately NOT involved — ``graph_expansion_cap``
    defaults to 0 now that this channel exists."""
    from agmem.config import AgmemConfig
    from agmem.core.ops import MemoryOp, OpType

    def build(depth):
        mem = AgenticMemory(
            namespace="t",
            organizers=["passthrough"],
            embedder=FakeEmbedder(dim=128),
            config=AgmemConfig(bfs_types=("facts",), bfs_max_depth=depth),
        )
        for node_id, name in (("A", "Alpha"), ("B", "Beta"), ("C", "Gamma")):
            mem._apply_one(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="entities",
                    target_id=node_id,
                    payload={
                        "id": node_id,
                        "name": name,
                        "content": f"{name}: node",
                        "embedding_text": name,
                    },
                )
            )
        for edge_id, src, dst in (("e1", "A", "B"), ("e2", "B", "C")):
            mem._apply_one(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="facts",
                    target_id=edge_id,
                    payload={
                        "id": edge_id,
                        "content": f"{src} relates to {dst}",
                        "subject_id": src,
                        "object_id": dst,
                        "predicate": "rel",
                        "embedding_text": f"{src} relates to {dst}",
                    },
                )
            )
        return mem

    def pulled(depth):
        mem = build(depth)
        try:
            # Drop e2's vector so the dense channel CANNOT return it: with only
            # two facts stored, a top-k vector search returns both regardless of
            # similarity, and then the test would pass without BFS running at
            # all. Reachable only via the graph is exactly the condition this
            # channel exists for.
            mem.vector_store.delete(["e2"])
            bundle = mem.search("A relates to B", memory_types=["facts"], k={"facts": 5})
            return sorted(s.item.data["id"] for s in bundle.items if s.memory_type == "facts")
        finally:
            mem.close()

    assert pulled(1) == ["e1"]  # ring is the origin alone -> nothing new
    assert pulled(2) == ["e1", "e2"]  # ring reaches B -> the B-C fact joins


def test_graph_recall_is_off_unless_asked_for():
    """``GraphRecall`` was our stand-in for φ_bfs before the channel existed, and
    it is not a mechanism any upstream recipe has (it appends at a flat score
    instead of ranking). It stays reachable for ablations, but only explicitly."""
    from agmem.config import AgmemConfig
    from agmem.retrieval.steps import GraphRecall, default_read_steps

    assert AgmemConfig().graph_expansion_cap == 0
    assert "entities" not in default_read_steps(graph_expansion_cap=0)
    assert isinstance(default_read_steps(graph_expansion_cap=10)["entities"], GraphRecall)


def test_reranker_runs_even_when_candidates_fit_in_k():
    """The gate used to be ``len(fused) > k``, i.e. "only rerank if something has
    to be dropped". A reranker also decides ORDER, and order outlives truncation:
    ``MemoryBundle.render`` sorts the whole bundle by score, so a type that
    skipped reranking keeps RRF scores while a reranked type carries relevance
    scores, and the two are then compared against one shared budget. Upstream
    reranks first and truncates second.

    Episode-mentions makes this observable without a model: three facts, k=3, so
    nothing is dropped and only the order can differ."""
    from agmem.config import AgmemConfig
    from agmem.core.ops import MemoryOp, OpType

    mem = AgenticMemory(
        namespace="t",
        organizers=["passthrough"],
        embedder=FakeEmbedder(dim=128),
        config=AgmemConfig(
            lexical_types=("facts",), overrides={"reranker": "EpisodeMentionsReranker"}
        ),
    )
    try:
        for i, provenance in enumerate((["a"], ["a", "b", "c"], ["a", "b"])):
            mem._apply_one(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="facts",
                    target_id=f"f{i}",
                    payload={
                        "id": f"f{i}",
                        "content": f"cats fact {i}",
                        "subject_id": "A",
                        "object_id": "B",
                        "predicate": "p",
                        "embedding_text": f"cats fact {i}",
                        "source_episode_ids": provenance,
                    },
                )
            )
        bundle = mem.search("cats fact", memory_types=["facts"], k={"facts": 3})
        served = [s.item.data["id"] for s in bundle.items]
        assert served == ["f1", "f2", "f0"]  # 3 mentions, then 2, then 1
    finally:
        mem.close()


def test_node_distance_reranker_orders_by_hops_from_the_centroid():
    """Zep's node-distance reranker (paper §3.2). The centroid is per-query, so it
    arrives as ``search(center_node_id=...)`` — upstream's ``center_node_uuid``.
    Without one the reranker is a no-op, which is also upstream's behavior."""
    from agmem.config import AgmemConfig
    from agmem.core.ops import MemoryOp, OpType
    from agmem.organizers.zep_graph import zep_search_recipe

    recipe = zep_search_recipe("node_distance")
    kwargs = recipe.config_kwargs()
    overrides = kwargs.pop("overrides")
    mem = AgenticMemory(
        namespace="t",
        organizers=["passthrough"],
        embedder=FakeEmbedder(dim=128),
        config=AgmemConfig(overrides=overrides, **kwargs),
    )
    try:
        for node_id, name in (("A", "Alpha"), ("B", "Beta"), ("C", "Gamma")):
            mem._apply_one(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="entities",
                    target_id=node_id,
                    payload={
                        "id": node_id,
                        "name": name,
                        "content": f"{name}: person",
                        "embedding_text": name,
                    },
                )
            )
        for edge_id, src, dst in (("e1", "A", "B"), ("e2", "B", "C")):
            mem._apply_one(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="facts",
                    target_id=edge_id,
                    payload={
                        "id": edge_id,
                        "content": f"{src} knows {dst}",
                        "subject_id": src,
                        "object_id": dst,
                        "predicate": "knows",
                        "embedding_text": f"{src} knows {dst}",
                    },
                )
            )
        chain = mem.search(
            "person", memory_types=["entities"], k={"entities": 3}, center_node_id="A"
        )
        assert [s.item.data["id"] for s in chain.items] == ["A", "B", "C"]
    finally:
        mem.close()
