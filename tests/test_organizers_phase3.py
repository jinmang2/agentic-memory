"""Phase 3: Nemori, MemoryOS, Zep-graph, G-Memory through the same MemoryOp contract."""

import pytest
from helpers import StubLLM

from agmem import AgenticMemory
from agmem.core.ops import OpType
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.gmemory import GMemoryOrganizer
from agmem.organizers.memoryos.organizer import PROFILE_ITEM_ID, MemoryOSOrganizer
from agmem.organizers.nemori import NemoriOrganizer
from agmem.organizers.zep_graph import ZepGraphOrganizer


def make_mem(organizer, llm):
    mem = AgenticMemory(namespace="t", organizers=[organizer], embedder=FakeEmbedder(dim=128))
    mem.structured = llm
    mem._ctx.llm = llm
    return mem


def ops_of(mem, ttype):
    return [o for o in mem.log.tail(50) if o.target_type == ttype]


# ---------------- Nemori ----------------


def test_nemori_boundary_flush_and_calibrate():
    llm = StubLLM(
        {
            "extract": [  # boundary checks (from 2nd message on)
                {"boundary": False, "confidence": 0.9},
                {"boundary": True, "confidence": 0.95},
            ],
            "distill": [
                {
                    "title": "Paris trip planning",
                    "narrative": "On 1 May 2023, the user planned a trip.",
                    "timestamp": "2023-05-01",
                },
                # cold start (no prior semantic memory) -> direct extraction, one call
                {"facts": ["The user's trip budget is 3,000,000 KRW."]},
            ],
        }
    )
    mem = make_mem(NemoriOrganizer(buffer_min=2), llm)
    try:
        mem.add_message("파리 여행 계획을 세우자")
        mem.add_message("예산은 300만원이면 될까?")
        mem.add_message("그건 그렇고 고양이 사료 얘기인데")  # boundary -> flush first two
        episodes = ops_of(mem, "episodes")
        assert len(episodes) == 1
        assert episodes[0].payload["title"] == "Paris trip planning"
        assert episodes[0].payload["timestamp"] == "2023-05-01"
        assert len(episodes[0].payload["source_episode_ids"]) == 2  # newest stays buffered
        facts = ops_of(mem, "semantic")
        assert len(facts) == 1 and "3,000,000" in facts[0].payload["content"]
    finally:
        mem.close()


def test_nemori_mechanical_fallback_on_generation_failure():
    llm = StubLLM(
        {
            "extract": [{"boundary": True, "confidence": 0.9}],
            "distill": [],  # episode generation returns None -> fallback
        }
    )
    mem = make_mem(NemoriOrganizer(buffer_min=2), llm)
    try:
        mem.add_message("first message about topic A here")
        mem.add_message("완전 다른 주제")
        episodes = ops_of(mem, "episodes")
        assert len(episodes) == 1
        assert episodes[0].payload["title"].startswith("first message")  # mechanical
    finally:
        mem.close()


def test_nemori_no_llm_degrades_quietly():
    mem = AgenticMemory(
        namespace="t", organizers=[NemoriOrganizer()], embedder=FakeEmbedder(dim=128)
    )
    try:
        mem.add_message("hello")
        assert not ops_of(mem, "episodes")
    finally:
        mem.close()


# ---------------- MemoryOS ----------------


def test_memoryos_eviction_creates_segment_and_promotes():
    llm = StubLLM(
        {
            "distill": [
                {
                    "groups": [
                        {
                            "topic": "travel",
                            "summary": "User plans a Paris trip.",
                            "page_indexes": [0, 1, 2],
                        }
                    ]
                },
                # LPM update = profile document + the two knowledge FIFOs
                {"profile": "The user is planning a Paris trip and prefers art museums."},
                {
                    "private": ["Trip budget: 3,000,000 KRW (Paris, 2026)"],
                    "assistant_knowledge": ["Assistant suggested museums at planning time"],
                },
            ],
        }
    )
    # dialogue_chain off: this test drives the MTM/LPM path, and the chain's two
    # per-page calls share the `distill` queue with the topic/profile ones.
    org = MemoryOSOrganizer(stm_capacity=3, heat_threshold=1.0, dialogue_chain=False)
    mem = make_mem(org, llm)
    try:
        for text in ("파리 가자", "예산 300", "미술관 위주"):
            mem.add_message(text)
        pages = ops_of(mem, "pages")
        assert len(pages) == 1 and pages[0].payload["topic"] == "travel"
        by_kind = {o.payload.get("kind"): o for o in ops_of(mem, "semantic")}
        assert set(by_kind) == {"profile", "user_knowledge", "assistant_knowledge"}
        # the profile is ONE document under a stable id, replaced rather than appended
        assert by_kind["profile"].target_id == PROFILE_ITEM_ID
        assert "Paris" in by_kind["profile"].payload["content"]
        # heat reset after promotion
        seg_id = pages[0].target_id
        assert org._heat[seg_id]["n_visit"] == 0 and org._heat[seg_id]["length"] == 0
    finally:
        mem.close()


def test_memoryos_profile_is_replaced_and_knowledge_fifo_evicts():
    """LPM is a rewritten document plus bounded deques, not an append log (docs/10 M1).

    Upstream feeds the previous profile back into the analysis prompt and stores
    the result with ``update_user_profile(merge=False)`` — a full replace — while
    knowledge goes into ``deque(maxlen=100)``. Appending facts instead could only
    accumulate and contradict, and never evicted."""
    llm = StubLLM(
        {
            "distill": [
                {"groups": [{"topic": "a", "summary": "s1", "page_indexes": [0]}]},
                {"profile": "First profile: the user is planning a trip to Paris."},
                {"private": ["k1: ctx", "None", "k2: ctx"], "assistant_knowledge": []},
                {"groups": [{"topic": "b", "summary": "s2", "page_indexes": [0]}]},
                {"profile": "Second profile: the user also enjoys art museums a lot."},
                {"private": ["k3: ctx"], "assistant_knowledge": []},
            ]
        }
    )
    # knowledge_capacity=2 so the third entry must evict the first
    org = MemoryOSOrganizer(
        stm_capacity=1, heat_threshold=1.0, knowledge_capacity=2, dialogue_chain=False
    )
    mem = make_mem(org, llm)
    try:
        mem.add_message("first")
        mem.add_message("second")
        semantic = ops_of(mem, "semantic")
        profiles = [o for o in semantic if o.target_id == PROFILE_ITEM_ID]
        # two writes, same id -> the store holds one document, the newest
        assert len(profiles) == 2 and all(o.op is OpType.ADD for o in profiles)
        stored = mem.doc_store.get_items([PROFILE_ITEM_ID], "semantic")[0]
        assert stored["content"].startswith("Second profile")

        knowledge = [o for o in semantic if o.payload.get("kind") == "user_knowledge"]
        assert [o.payload["content"] for o in knowledge] == ["k1: ctx", "k2: ctx", "k3: ctx"]
        # "None" filtered out, and k3 pushed k1 out of the capacity-2 FIFO
        evictions = [
            o for o in semantic if o.op is OpType.DELETE and o.target_id == knowledge[0].target_id
        ]
        assert len(evictions) == 1
        assert not mem.doc_store.get_items([knowledge[0].target_id], "semantic")[0].get("content")
    finally:
        mem.close()


def test_memoryos_fidelity_presets_separate_the_two_upstream_lineages():
    """`memoryos-pypi/` and `eval/` are different code with different constants,
    and the paper's LoCoMo numbers came from `eval/` — so a reproduction has to
    be able to reach it (MEMORYOS_PRESETS).

    Each assertion below is a divergence verified against upstream, not a
    preference: heat weights (1/1/1 vs 0.8/0.8/1e-4), live vs stored recency,
    Jaccard vs the mean of containment ratios, and STM capacity in pages.
    Eviction is deliberately NOT in that list: both lineages call `evict_lfu`,
    so both presets say "lfu" — the earlier pypi="lowest_heat" label encoded
    the paper's sentence, not pypi code (docs/16 session 3)."""
    pypi, ev = MemoryOSOrganizer(), MemoryOSOrganizer(fidelity="eval")
    assert (pypi.heat_weights, pypi.recency, pypi.eviction) == (
        (1.0, 1.0, 1.0),
        "live",
        "lfu",
    )
    assert (ev.heat_weights, ev.recency, ev.eviction) == ((0.8, 0.8, 0.0001), "stored", "lfu")
    assert (pypi.stm_capacity, ev.stm_capacity) == (10, 1)
    # shared across lineages — theta, tau, MTM capacity
    assert (pypi.similarity_threshold, pypi.heat_threshold, pypi.mtm_capacity) == (0.6, 5.0, 2000)
    assert (ev.similarity_threshold, ev.heat_threshold, ev.mtm_capacity) == (0.6, 5.0, 2000)

    # keyword term: containment-mean is strictly larger than Jaccard whenever
    # the sets differ, so the same theta merges more readily under `eval`
    a, b = {"trip", "paris"}, {"trip", "paris", "museum", "budget"}
    assert pypi._keyword_overlap(a, b) == 0.5  # 2/4 union
    assert ev._keyword_overlap(a, b) == 0.75  # 0.5*(2/2 + 2/4)
    assert ev._keyword_overlap(a, b) > pypi._keyword_overlap(a, b)

    # an explicit kwarg still overrides the preset it came from
    assert MemoryOSOrganizer(fidelity="eval", stm_capacity=4).stm_capacity == 4


def test_memoryos_eval_lineage_recency_cannot_change_a_comparison():
    """MEMORYOS_PRESETS E2: the eval core weights a STORED R_recency at 1e-4.

    Reproduced rather than corrected — it is the published operating point —
    but pinned here so nobody later reads the recency term as load-bearing: two
    segments differing only in recency must compare equal to well under one
    interaction's worth of heat."""
    org = MemoryOSOrganizer(fidelity="eval")
    org._heat = {
        "fresh": {"n_visit": 0, "length": 3, "last_access": None, "recency": 1.0},
        "stale": {"n_visit": 0, "length": 3, "last_access": None, "recency": 0.0},
    }
    gap = org._segment_heat("fresh") - org._segment_heat("stale")
    assert gap == pytest.approx(1e-4)  # vs beta=0.8 for a single page
    assert gap < 0.8 / 1000


def test_memoryos_counts_pages_not_messages():
    """MemoryOS's unit is the PAGE — one ``add_qa_pair`` exchange — and both
    ``stm_capacity`` (upstream ``ShortTermMemory`` is a deque of pairs) and
    ``L_interaction`` (``len(processed_details)``) are page counts.

    Counting messages instead put both in the wrong unit: the STM flushed at
    half the upstream batch size and heat ran ~2x high, so the
    ``heat_threshold`` promotion to LPM fired at about half the content
    (2026-07-27 audit B1). The ``i == 1`` assertion is the discriminating one —
    under message counting, ``stm_capacity=2`` would already have flushed
    there."""
    llm = StubLLM(
        {"distill": [{"groups": [{"topic": "t", "summary": "s", "page_indexes": [0, 1]}]}]}
    )
    org = MemoryOSOrganizer(stm_capacity=2, heat_threshold=99.0)  # promotion out of scope
    mem = make_mem(org, llm)
    try:
        for i, speaker in enumerate(["A", "B", "A", "B"]):
            mem.add_message(f"m{i}", meta={"speaker": speaker})
            # A then B is ONE page, so no flush; the second A opens page 2 and
            # trips `>= stm_capacity`. The trailing B lands in the next buffer —
            # the documented split-exchange deviation from receiving the two
            # halves as separate calls.
            assert len(ops_of(mem, "pages")) == (1 if i >= 2 else 0), i
        (heat,) = org._heat.values()
        # ONE page reaches MTM (upstream's rolling eviction pops a single page),
        # and that page holds TWO messages — so `length == 1` is the
        # discriminator against message counting, which would say 2.
        assert heat["length"] == 1, heat
    finally:
        mem.close()


def test_memoryos_no_llm_mechanical_segment():
    org = MemoryOSOrganizer(stm_capacity=2, dialogue_chain=False)
    mem = AgenticMemory(namespace="t", organizers=[org], embedder=FakeEmbedder(dim=128))
    try:
        mem.add_message("a")
        mem.add_message("b")
        pages = ops_of(mem, "pages")
        assert len(pages) == 1 and "a" in pages[0].payload["content"]
    finally:
        mem.close()


# ---------------- Zep-graph ----------------


def test_zep_graph_entities_facts_and_invalidation():
    llm = StubLLM(
        {
            "extract": [
                {
                    "entities": [
                        {"name": "Caroline", "type": "Person", "summary": "a user"},
                        {"name": "Seoul", "type": "Place", "summary": "a city"},
                    ]
                },
                {
                    "facts": [
                        {
                            "subject": "Caroline",
                            "predicate": "lives_in",
                            "object": "Seoul",
                            "statement": "Caroline lives in Seoul.",
                        }
                    ]
                },
                {
                    "entities": [
                        {"name": "Caroline", "type": "Person", "summary": "a user"},
                        {"name": "Busan", "type": "Place", "summary": "a city"},
                    ]
                },
                {
                    "facts": [
                        {
                            "subject": "Caroline",
                            "predicate": "lives_in",
                            "object": "Busan",
                            "statement": "Caroline lives in Busan.",
                            "valid_at": "2024-01-01T00:00:00",
                        }
                    ]
                },
            ],
            # No edges between Caroline-Busan, but the GRAPH-WIDE invalidation
            # pool (round-12 finding 4) surfaces the Seoul fact for the second
            # message, so an edge-resolve call happens; it flags nothing.
            "distill": [{"duplicate_of": None, "contradicts": []}],
        }
    )
    org = ZepGraphOrganizer()
    mem = make_mem(org, llm)
    try:
        mem.add_message("Caroline lives in Seoul.")
        mem.add_message("Caroline moved to Busan.")
        # entity resolution: Caroline (identical name+summary embedding) reused
        assert org.graph.counts()["nodes"] == 3  # Caroline, Seoul, Busan
        facts = [o for o in ops_of(mem, "facts") if o.op is OpType.ADD]
        assert len(facts) == 2
        assert all(o.payload["valid_at"] for o in facts)
    finally:
        mem.close()


def test_zep_graph_contradiction_invalidates():
    llm = StubLLM(
        {
            "extract": [
                {"entities": [{"name": "A", "summary": "s"}, {"name": "B", "summary": "s"}]},
                {
                    "facts": [
                        {
                            "subject": "A",
                            "predicate": "likes",
                            "object": "B",
                            "statement": "A likes B.",
                            # STRICTLY OLDER than the contradicting fact — the
                            # upstream truth table invalidates only that case
                            # (round-12 finding 5); a dateless fact would be
                            # inert now that null valid_at stays None.
                            "valid_at": "2020-01-01T00:00:00",
                        }
                    ]
                },
                {"entities": [{"name": "A", "summary": "s"}, {"name": "B", "summary": "s"}]},
                {
                    "facts": [
                        {
                            "subject": "A",
                            "predicate": "dislikes",
                            "object": "B",
                            "statement": "A dislikes B.",
                            "valid_at": "2024-01-01T00:00:00",
                        }
                    ]
                },
            ],
            "distill": [{"contradicts": ["__EDGE__"]}],
        }
    )
    org = ZepGraphOrganizer()
    mem = make_mem(org, llm)
    try:
        mem.add_message("A likes B")
        first_edge = ops_of(mem, "facts")[0].target_id
        llm.responses["distill"][0]["contradicts"] = [first_edge]
        mem.add_message("A now dislikes B")
        invalidations = [o for o in ops_of(mem, "facts") if o.op is OpType.INVALIDATE]
        assert len(invalidations) == 1 and invalidations[0].target_id == first_edge
        # bi-temporal: doc item got invalid_at; graph edge no longer active
        item = mem.doc_store.get_items([first_edge], "facts")[0]
        assert item.get("invalid_at")
        a_node = org.graph.find_node_by_name("A", "t")
        b_node = org.graph.find_node_by_name("B", "t")
        active = org.graph.edges_between(a_node["id"], b_node["id"], "t")
        assert len(active) == 1 and active[0]["content"] == "A dislikes B."
    finally:
        mem.close()


def test_zep_graph_is_rebuildable_from_the_evolution_log():
    """The property the direct-write path could not provide (2026-07-27 audit B3).

    Graph mutations used to happen inside the organizer, so the log described
    only the doc/vector side: replaying it produced an EMPTY graph, and
    ``GraphRecall`` — whose only no-op guard is "no graph store at all" — then
    returned nothing and silently degraded Zep's read path to plain vector RAG.
    Replaying the same ops into a fresh memory must now reconstruct the graph
    exactly, with no Zep organizer involved at all, because the mutation lives
    in ``AgenticMemory._apply_graph`` rather than in the methodology."""
    llm = StubLLM(
        {
            "extract": [
                {
                    "entities": [
                        {"name": "Caroline", "type": "Person", "summary": "a user"},
                        {"name": "Seoul", "type": "Place", "summary": "a city"},
                    ]
                },
                {
                    "facts": [
                        {
                            "subject": "Caroline",
                            "predicate": "lives_in",
                            "object": "Seoul",
                            "statement": "Caroline lives in Seoul.",
                            "valid_at": "2023-01-01T00:00:00",
                        }
                    ]
                },
            ],
            "distill": [],
        }
    )
    org = ZepGraphOrganizer()
    mem = make_mem(org, llm)
    try:
        mem.add_message("Caroline lives in Seoul.")
        original_counts = org.graph.counts()
        caroline = org.graph.find_node_by_name("Caroline", "t")["id"]
        replayable = [o for o in mem.log.tail(100) if o.target_type in ("entities", "facts")]
    finally:
        mem.close()

    assert original_counts["nodes"] == 2 and original_counts["edges"] == 1

    # A plain passthrough memory — nothing here knows what Zep is.
    replay = AgenticMemory(
        namespace="t", organizers=["passthrough"], embedder=FakeEmbedder(dim=128)
    )
    try:
        for op in replayable:
            replay._apply_one(op)
        assert replay.graph_store.counts() == original_counts
        edges = replay.graph_store.edges_for_nodes([caroline], "t")
        assert len(edges) == 1 and edges[0]["content"] == "Caroline lives in Seoul."
    finally:
        replay.close()


class ZepStub:
    """StubLLM keyed on the prompt instead of a per-role queue.

    Zep's write path makes a variable number of calls — entity resolution only
    asks the LLM when the embedding search returned candidates, and edge
    resolution only when the pair already has edges — so a positional queue
    silently hands a fact response to a resolution call as soon as the
    (embedder-dependent) candidate set changes. Dispatching on the prompt keeps
    these tests about the community pipeline rather than about FakeEmbedder."""

    def __init__(self, entities: list[list[str]], facts: list[list[tuple[str, str]]]):
        self.entities = list(entities)
        self.facts = list(facts)
        self.calls: list[str] = []
        self.pair_calls = 0

    def call(self, role, prompt, schema, required_keys=(), **kwargs):
        self.calls.append(role)
        if "Extract the distinct real-world entities" in prompt:
            names = self.entities.pop(0) if self.entities else []
            return {
                "entities": [
                    {"name": n, "type": "Person", "summary": f"{n} is known"} for n in names
                ]
            }
        if "Extract relationship facts" in prompt:
            pairs = self.facts.pop(0) if self.facts else []
            return {
                "facts": [
                    {
                        "subject": s,
                        "predicate": "knows",
                        "object": o,
                        "statement": f"{s} knows {o}.",
                    }
                    for s, o in pairs
                ]
            }
        if "Decide for each NEW entity" in prompt:
            # empty resolutions -> every unresolved entity becomes a new node
            # (the batched-call guardrail path)
            return {"resolutions": []}
        if "A new fact arrived" in prompt:
            return {"contradicts": []}
        if "Synthesize the information" in prompt:
            self.pair_calls += 1
            return {"summary": "merged summary."}
        if "Create a short one sentence description" in prompt:
            return {"description": "A cluster of people who know each other."}
        raise AssertionError(f"unexpected prompt: {prompt[:60]!r}")


def test_zep_entity_items_carry_content_for_render_and_bm25():
    """Entity items used to store only name/summary (2026-07-27 round-7).

    Two channels read ``content`` and both came back empty: ``_DictItem`` renders
    it, so a retrieved entity became a bare ``- `` bullet in the QA prompt, and
    the doc store's FTS index is fed from it, so the BM25 half of the hybrid
    search the zep config enables for ``entities`` matched nothing at all. The
    dense channel still embeds the NAME alone, as upstream's semantic candidate
    search does — the two are deliberately different texts."""
    org = ZepGraphOrganizer(community_refresh=False)
    llm = ZepStub([["Caroline", "Seoul"]], [[("Caroline", "Seoul")]])
    mem = make_mem(org, llm)
    try:
        mem.add_message("Caroline lives in Seoul.")
        item = next(o for o in ops_of(mem, "entities") if o.op is OpType.ADD).payload
        assert item["content"] == "Caroline: Caroline is known"
        assert item["embedding_text"] == "Caroline"
        # the lexical channel now has something to match on
        assert mem.doc_store.search_lexical_items("Caroline", "entities", namespace="t")
        rendered = mem.search("Caroline", memory_types=("entities",)).render()
        assert "Caroline: Caroline is known" in rendered
    finally:
        mem.close()


def test_label_propagation_reproduces_upstream_weight_gate_and_isolates():
    """Upstream's rules, including the two its own docstring gets wrong.

    - ``a``/``b``, one edge: a weight-1 plurality does NOT win
      (``candidate_rank > 1``), so both fall through to
      ``max(candidate, current)`` and land on the higher label — together, but
      not by the "plurality of neighbours" the docstring describes.
    - ``lonely``: a node with no relations keeps its own label and forms a
      singleton, rather than dropping out of the partition.
    - ``c``/``d``, two edges: each clears the weight gate, so they SWAP labels
      every round and never converge. Upstream's ``while True`` loops forever
      on this shape; we stop on the detected 2-cycle, and since neither state
      of the cycle puts them together they stay singletons."""
    from agmem.organizers.zep_graph.community import label_propagation

    projection = {
        "a": {"b": 1},
        "b": {"a": 1},
        "c": {"d": 2},
        "d": {"c": 2},
        "lonely": {},
    }
    clusters = sorted(sorted(c) for c in label_propagation(projection))
    assert clusters == [["a", "b"], ["c"], ["d"], ["lonely"]]


def test_zep_communities_are_built_at_flush_and_replay_from_the_log():
    """The third subgraph (paper §2.2.4), under the same op-log contract as the
    other two: a passthrough memory replaying only the ops must reconstruct the
    community nodes and their membership, with no Zep organizer involved."""
    org = ZepGraphOrganizer(community_refresh=False)  # refresh enabled below
    org.community_refresh = True
    llm = ZepStub(
        [["Ann", "Bob"], ["Cal", "Dee"]],
        [[("Ann", "Bob")], [("Cal", "Dee")]],
    )
    mem = make_mem(org, llm)
    try:
        mem.add_message("Ann knows Bob.")
        mem.add_message("Cal knows Dee.")
        mem.flush()

        communities = org.graph.communities("t")
        assert len(communities) == 2  # two disconnected pairs -> two clusters
        assert all(len(org.graph.community_members(c["id"])) == 2 for c in communities)
        # map-reduce cost: one pair call per cluster of two, plus one naming call
        assert llm.pair_calls == 2

        item = next(o for o in ops_of(mem, "communities") if o.op is OpType.ADD).payload
        assert item["embedding_text"] == item["name"] == item["lexical_text"]
        assert item["content"].startswith(item["name"])
        assert len(item["member_ids"]) == 2

        calls_before = len(llm.calls)
        mem.flush()
        # an unchanged partition costs nothing — this is the one place the
        # implementation departs from upstream's unconditional wipe-and-rebuild
        assert len(llm.calls) == calls_before
        assert not [o for o in mem.log.tail(200) if o.target_type == "communities"][2:]

        replayable = [
            o for o in mem.log.tail(200) if o.target_type in ("entities", "facts", "communities")
        ]
        expected = org.graph.counts()
    finally:
        mem.close()

    replay = AgenticMemory(
        namespace="t", organizers=["passthrough"], embedder=FakeEmbedder(dim=128)
    )
    try:
        for op in replayable:
            replay._apply_one(op)
        assert replay.graph_store.counts() == expected
        assert sorted(
            len(replay.graph_store.community_members(c["id"]))
            for c in replay.graph_store.communities("t")
        ) == [2, 2]
    finally:
        replay.close()


def test_zep_incremental_extension_joins_the_neighbour_community():
    """``update_communities=True`` — the single-step extension the paper gives as
    label propagation's motivation. A new entity related to an entity that is
    already in a community joins that community and the community is
    re-summarized, without a full rebuild."""
    org = ZepGraphOrganizer(community_refresh=True, update_communities=True)
    llm = ZepStub(
        [["Ann", "Bob"], ["Bob", "Eve"]],
        [[("Ann", "Bob")], [("Bob", "Eve")]],
    )
    mem = make_mem(org, llm)
    try:
        mem.add_message("Ann knows Bob.")
        mem.flush()  # builds the {Ann, Bob} community
        before = org.graph.communities("t")
        assert len(before) == 1

        mem.add_message("Bob knows Eve.")  # Eve is new, Bob is in the community
        eve = org.graph.find_node_by_name("Eve", "t")["id"]
        assert org.graph.community_of_node(eve, "t") is not None
        assert len(org.graph.communities("t")) == 1  # joined, not created
        assert eve in org.graph.community_members(before[0]["id"])
    finally:
        mem.close()


# ---------------- G-Memory ----------------


def test_gmemory_trajectory_and_insight_finetune():
    """Write-path call and payload shape (round-12 findings 2, 6, 7, 8, 9, 14b).

    Call count is load-bearing: a success task costs 1 sparsify call; a failed
    task costs 2 (key steps + mistake detection, upstream U:265-290); one
    finetune point costs 1 compare-pair call + 1 success-chunk call
    (U:719-748) — 5 total here. Trajectories embed the TASK ONLY (upstream
    ``page_content = task_main``, U:96-98), carry the full task text as the
    correlation/repeat key, and insights carry ``embedding_text: None`` so
    they NEVER enter the vector store (upstream serves rules by correlation
    counting alone, U:490-506) — which also keeps the top-10 edge-candidate
    pool all-trajectory (finding 3)."""
    llm = StubLLM(
        {
            "distill": [
                {"key_steps": ["searched", "clicked"]},  # task one sparsify
                {"key_steps": ["opened settings"]},  # task two sparsify
                {"mistakes": ["wrong tab first"]},  # task two failed -> 2nd call
                {  # finetune point: compare-pair call
                    "operations": [
                        {"op": "ADD", "rule": "Always open settings from the sidebar."},
                        {"op": "EDIT", "id": "fake-id", "rule": "ignored"},  # hallucinated
                    ]
                },
                {"operations": []},  # finetune point: success-chunk call
            ],
        }
    )
    org = GMemoryOrganizer(finetune_every=2, finetune_points=1, finetune_seed=0)
    mem = make_mem(org, llm)
    try:
        mem.add_task_result([{"s": 1}], "success", "task one")
        mem.add_task_result([{"s": 2}], "failure", "task two")
        assert len(llm.calls) == 5  # 1 + 2 + (1 compare + 1 success-chunk)
        strategies = ops_of(mem, "strategies")
        trajs = [o for o in strategies if o.payload.get("kind") == "trajectory"]
        insights = [o for o in strategies if o.payload.get("kind") == "insight"]
        assert len(trajs) == 2
        assert trajs[0].payload["score"] == 1.0 and trajs[1].payload["score"] == -2.0
        # task-only embedding + full task text stored beside the display title
        assert trajs[0].payload["embedding_text"] == "task one"
        assert trajs[0].payload["task"] == "task one"
        assert "Mistakes: wrong tab first" in trajs[1].payload["content"]
        assert len(insights) == 1  # hallucinated EDIT ignored
        assert not [o for o in strategies if o.op is OpType.UPDATE]
        # correlation keys are the FULL task strings shown in the compare
        # prompt (upstream `relative_tasks`, U:725-728), not 80-char titles
        assert insights[0].payload["positive_correlation_tasks"] == ["task one", "task two"]
        # insights never enter the vector store; trajectories do
        insight_id = insights[0].payload["id"]
        traj_ids = [t.payload["id"] for t in trajs]
        assert insights[0].payload["embedding_text"] is None
        assert insight_id not in mem.vector_store.get([insight_id])
        assert set(mem.vector_store.get(traj_ids)) == set(traj_ids)
        # ...so the k=10 strategies candidate pool is all-trajectory
        pool = mem.vector_store.search(
            mem.embedder.embed(["task"])[0], k=10, memory_type="strategies", namespace="t"
        )
        assert {i for i, _ in pool} == set(traj_ids)
    finally:
        mem.close()


def test_gmemory_edge_gate_is_effective_cosine_085():
    """U:390-392 thresholds ``1 - distance`` at 0.7 where distance is Chroma's
    default squared L2 over normalized MiniLM vectors: 1-(2-2cos) >= 0.7 is
    cos >= 0.85. A pair at true cosine 0.75 — over upstream's literal 0.7 —
    must NOT link; a pair at 0.875 must."""
    from agmem.embed.fake import FakeEmbedder

    near, far = "alpha beta gamma delta", "alpha beta gamma epsilon"
    va, vb = FakeEmbedder(dim=128).embed([near, far])
    cos = sum(x * y for x, y in zip(va, vb))
    assert 0.7 <= cos < 0.85  # the pair that separates the two gates

    org = GMemoryOrganizer(finetune_every=100)
    mem = make_mem(org, StubLLM({}))  # sparsify falls back mechanically
    try:
        mem.add_task_result([{"s": 1}], "success", near)
        mem.add_task_result([{"s": 2}], "success", far)  # 0.75: below the gate
        mem.add_task_result([{"s": 3}], "success", "go buy fresh red apples at the market")
        mem.add_task_result([{"s": 4}], "success", "go buy fresh red apples at the store")
        adds = [o for o in ops_of(mem, "strategies") if o.op is OpType.ADD]
        updates = [o for o in ops_of(mem, "strategies") if o.op is OpType.UPDATE]
        assert adds[1].payload["task_edges"] == []  # 0.75 < 0.85: no edge
        assert adds[3].payload["task_edges"] == [adds[2].payload["id"]]  # 0.875: edge
        assert [u.target_id for u in updates] == [adds[2].payload["id"]]
    finally:
        mem.close()


def test_gmemory_repeat_task_skips_store_and_edges_globally():
    """Repeat detection is global and exact on the full task text (upstream
    ``if task_main in self.graph: return``, U:380-381): the second occurrence
    stores nothing and links nothing — repeats unify into the one node. The
    sparsify call still happens (upstream extracts BEFORE the repeat check,
    U:90-93), so call parity holds."""
    llm = StubLLM({})
    org = GMemoryOrganizer(finetune_every=100)
    mem = make_mem(org, llm)
    try:
        mem.add_task_result([{"s": 1}], "success", "assemble the bookshelf")
        mem.add_task_result([{"s": 2}], "success", "assemble the bookshelf")
        strategies = ops_of(mem, "strategies")
        assert len([o for o in strategies if o.op is OpType.ADD]) == 1
        assert not [o for o in strategies if o.op is OpType.UPDATE]
        assert len(llm.calls) == 2  # sparsify ran both times
    finally:
        mem.close()


def test_gmemory_feedback_touches_served_insights_only():
    """Upstream ``backward`` walks ``insights_cache`` — served RULES
    exclusively (U:239, 292-297). Trajectory scores are write-time labels and
    never move; an empty cache updates nothing (no bypass, round-12 #6)."""
    org = GMemoryOrganizer()
    mem = make_mem(org, StubLLM({}))
    try:
        mem.doc_store.put_item(
            "traj1",
            "strategies",
            "t",
            {"id": "traj1", "kind": "trajectory", "task": "task x", "content": "c", "score": 1.0},
        )
        mem.doc_store.put_item(
            "ins1",
            "strategies",
            "t",
            {"id": "ins1", "kind": "insight", "content": "rule", "score": 2.0},
        )
        # nothing served yet: nothing updates
        assert mem.report_feedback(["traj1", "ins1"], helpful=True) == 0

        org.on_retrieval([("ins1", "strategies", 1.0), ("traj1", "strategies", 0.9)], mem._ctx)
        assert org._served == {"ins1"}  # the cache holds insight ids only
        # -2 drops the insight to 0 -> UPDATE + prune DELETE; trajectory untouched
        assert mem.report_feedback(["traj1", "ins1"], helpful=False) == 2
        assert mem.doc_store.get_items(["traj1"], "strategies")[0]["score"] == 1.0
        assert mem.doc_store.get_items(["ins1"], "strategies")[0].get("deleted")
        assert org._served == set()
    finally:
        mem.close()


def test_gmemory_read_recipe_constant():
    """The published operating point is the shipped harness's argparse
    defaults (tasks/run.py:128-131), not ``retrieve_memory``'s signature
    defaults (2/1/10/0.3) — and all three MAS workflows discard the failed
    list at read time. The Eq.(6) insight cap defaults to the signature's 10
    and is config-reachable; a harness-faithful run sets 3."""
    from agmem.config import AgmemConfig
    from agmem.organizers.gmemory.organizer import GMEMORY_READ_RECIPE
    from agmem.retrieval.steps import TaskGraphExpansion, default_read_steps

    assert GMEMORY_READ_RECIPE == {
        "successful_topk": 1,
        "failed_topk": 0,
        "insights_topk": 3,
        "threshold": 0.0,
    }
    assert AgmemConfig().task_graph_insight_cap == 10
    step = default_read_steps(task_graph_insight_cap=3)["strategies"]
    assert isinstance(step, TaskGraphExpansion)
    assert step.insight_cap == 3
    assert step.threshold == 0.0  # the recipe's threshold, not the 0.3 signature default


def test_gmemory_task_graph_edges_and_hop_expansion():
    """Paper Eq.(9)/(5)/(6) — the query graph and its read side.

    Two similar tasks (shared-token cosine >= 0.85 under FakeEmbedder — the
    effective gate, see the organizer) get linked at on_task_end: the new
    trajectory's ADD carries ``task_edges`` and the earlier one gains the
    back-edge via UPDATE (upstream ``TaskLayer.add_task_node``). At read time
    ``TaskGraphExpansion`` serves the 1-hop neighbour of a trajectory hit
    (Eq.(5)) — re-scored by true cosine to the query when the context carries
    the query embedding and vector store (upstream
    ``sort_and_filter_by_similarity``, U:122-169), parent*0.9 stand-in when
    it does not — and any insight whose ``positive_correlation_tasks``
    intersect the served FULL task texts (Eq.(6), upstream
    ``_find_related_insights``)."""
    from agmem.core.types import ScoredItem
    from agmem.retrieval.steps import ReadContext, TaskGraphExpansion, _DictItem

    llm = StubLLM(
        {
            "distill": [
                {"key_steps": ["x"]},
                {"key_steps": ["x"]},
            ],
        }
    )
    org = GMemoryOrganizer(finetune_every=100)  # keep finetune out of this test
    mem = make_mem(org, llm)
    task_a = "go buy fresh red apples at the market"
    task_b = "go buy fresh red apples at the store"  # 7/8 shared tokens: cos 0.875
    try:
        mem.add_task_result([{"s": 1}], "success", task_a)
        mem.add_task_result([{"s": 2}], "success", task_b)
        strategies = ops_of(mem, "strategies")
        adds = [o for o in strategies if o.op is OpType.ADD]
        updates = [o for o in strategies if o.op is OpType.UPDATE]
        first_id, second_id = adds[0].payload["id"], adds[1].payload["id"]
        assert adds[1].payload["task_edges"] == [first_id]
        back_edge = next(u for u in updates if u.target_id == first_id)
        assert back_edge.payload["task_edges"] == [second_id]

        # read side, bare context: the 1-hop neighbour joins at parent*0.9
        first_item = mem.doc_store.get_items([first_id], "strategies")[0]
        hits = [ScoredItem(item=_DictItem(first_item), memory_type="strategies", score=1.0)]
        ctx = ReadContext(doc_store=mem.doc_store, namespace="t")
        by_id = {s.item.data["id"]: s.score for s in TaskGraphExpansion().run(hits, ctx)}
        assert by_id[second_id] == pytest.approx(0.9)

        # Eq.(5) re-scoring: with query embedding + vector store the neighbour
        # scores its true cosine to the query (task-only vectors make this
        # task-vs-task), and the threshold knob cuts it
        query_vec = mem.embedder.embed([task_b])[0]
        rich_ctx = ReadContext(
            doc_store=mem.doc_store,
            namespace="t",
            query_embedding=query_vec,
            vector_store=mem.vector_store,
        )
        by_id = {s.item.data["id"]: s.score for s in TaskGraphExpansion().run(hits, rich_ctx)}
        assert by_id[second_id] == pytest.approx(1.0)  # the query IS the neighbour's task
        served = {
            s.item.data["id"]
            for s in TaskGraphExpansion(threshold=0.99).run(
                hits,
                ReadContext(
                    doc_store=mem.doc_store,
                    namespace="t",
                    query_embedding=mem.embedder.embed(["completely unrelated words"])[0],
                    vector_store=mem.vector_store,
                ),
            )
        }
        assert second_id not in served

        # an insight supported by a served FULL task text joins too (Eq.(6));
        # one correlated with a different task does not
        mem.doc_store.put_item(
            "ins1",
            "strategies",
            "t",
            {
                "id": "ins1",
                "kind": "insight",
                "content": "prefer the sidebar",
                "score": 2.0,
                "positive_correlation_tasks": [task_a],
            },
        )
        mem.doc_store.put_item(
            "ins2",
            "strategies",
            "t",
            {
                "id": "ins2",
                "kind": "insight",
                "content": "unrelated",
                "score": 2.0,
                "positive_correlation_tasks": ["file taxes"],
            },
        )
        served = {s.item.data["id"] for s in TaskGraphExpansion().run(hits, ctx)}
        assert "ins1" in served and "ins2" not in served
    finally:
        mem.close()


def test_zep_search_recipes_match_upstream_and_name_the_papers_operating_point():
    """The read path is a named preset, not a call-site decision.

    Zep describes three search functions and five rerankers and upstream ships
    the combinations as ``SearchConfig`` recipes, so a run has to say which one
    it used. Each assertion below is a field of
    ``search_config_recipes.py`` or a sentence of the paper:

    - BFS appears ONLY in the cross-encoder family; the RRF and MMR recipes list
      ``[bm25, cosine_similarity]`` and nothing else.
    - Communities never get a BFS channel (there is no
      ``CommunitySearchMethod.bfs``), even in the recipe that gives edges and
      nodes one.
    - §4.1 — "BGE-m3 models from BAAI for both reranking and embedding tasks" —
      makes the cross-encoder recipe the paper's operating point, and BGE-m3's
      reranker is the model upstream's ``BGERerankerClient`` loads.
    - Upstream's MMR recipe sets ``mmr_lambda=1``, which switches the diversity
      term off entirely.
    """
    from agmem.organizers.zep_graph import (
        DEFAULT_RECIPE,
        ZEP_SEARCH_RECIPES,
        zep_search_recipe,
    )

    assert DEFAULT_RECIPE == "cross_encoder"
    paper = zep_search_recipe()
    assert paper.name == "COMBINED_HYBRID_SEARCH_CROSS_ENCODER"
    assert paper.reranker == "CrossEncoderReranker"
    assert paper.reranker_params == {"model_name": "BAAI/bge-reranker-v2-m3"}
    assert paper.bfs_types == ("facts", "entities")  # not communities
    assert "communities" in paper.memory_types and "communities" in paper.lexical_types

    assert ZEP_SEARCH_RECIPES["rrf"].bfs_types == ()
    assert ZEP_SEARCH_RECIPES["mmr"].bfs_types == ()
    assert ZEP_SEARCH_RECIPES["mmr"].reranker_params == {"lambda_": 1.0}
    # raw episodes are not one of Zep's three subgraphs; a config carrying them
    # is the mixed ablation docs/10 bars from reproduction claims
    assert all("episodic" not in r.memory_types for r in ZEP_SEARCH_RECIPES.values())
    # GraphRecall must be off in every recipe: it is our stand-in for the BFS
    # channel, and running both double-serves the same edges
    assert all(r.config_kwargs()["graph_expansion_cap"] == 0 for r in ZEP_SEARCH_RECIPES.values())

    with pytest.raises(KeyError):
        zep_search_recipe("no_such_recipe")  # no silent fallback to the default


def test_memoryos_stm_rolls_one_page_and_capacity_pages_stay_resident():
    """Upstream pypi's STM is a FIFO rolling window that flushes at the START
    of the overflowing add (``memoryos.py:242-246``: ``if is_full():
    process(...)`` runs BEFORE ``add_qa_pair``): resident STM sits at
    ``capacity`` pages — not ``capacity - 1``, the pre-round-12 claim
    (finding 13) — and one page rolls to MTM when the (capacity+1)-th
    arrives. The resident window feeds the QA-time recent-context channel
    (paper §3.3, "the oldest dialogue page is transferred ... according to
    the FIFO principle")."""
    llm = StubLLM(
        {"distill": [{"groups": [{"topic": f"t{i}", "summary": f"s{i}"}]} for i in range(4)]}
    )
    org = MemoryOSOrganizer(stm_capacity=2, dialogue_chain=False, heat_threshold=99.0)
    mem = make_mem(org, llm)
    try:
        # two full exchanges: X0 = m0/r0, X1 = m1/r1
        for i in range(2):
            mem.add_message(f"m{i}", meta={"speaker": "A"})
            mem.add_message(f"r{i}", meta={"speaker": "B"})
        # STM RESIDENT count equals capacity — nothing evicted yet
        assert ops_of(mem, "pages") == []
        assert [[e.content for e in p] for p in org._stm_pages] == [["m0", "r0"], ["m1", "r1"]]

        # the third exchange overflows: the OLDEST page rolls out, one page only
        mem.add_message("m2", meta={"speaker": "A"})
        mem.add_message("r2", meta={"speaker": "B"})
        assert len(ops_of(mem, "pages")) == 1
        assert [[e.content for e in p] for p in org._stm_pages] == [["m1", "r1"], ["m2", "r2"]]

        # ...and the resident window is what the recent-context channel serves
        assert "m1" in org.recent_context() and "m2" in org.recent_context()
        assert "m0" not in org.recent_context()
        # a drain does NOT empty it: upstream never drains STM
        assert org.flush_buffer(mem._ctx) == []
        assert len(org._stm_pages) == 2
    finally:
        mem.close()


def test_memoryos_dialogue_chain_summarizes_and_renders():
    """``meta_info``: upstream checks page-to-page continuity and keeps a running
    chain summary (2 LLM calls per evicted page), then injects it beside the
    retrieved memory as "Conversation chain overview:" (``get_response``)."""
    llm = StubLLM(
        {
            "judge": [{"continuous": True}],
            "distill": [
                {"meta_info": "The user is planning a Paris trip."},
                # The summary has to carry the query terms: retrieval is TWO
                # gates now, and the first one scores the SEGMENT's summary
                # (upstream matches sessions on `summary_embedding`), so a
                # summary sharing no terms with the query means the pages inside
                # are never scored at all.
                {"groups": [{"topic": "travel", "summary": "파리 여행 요약", "page_indexes": [0]}]},
                {"meta_info": "The user is planning a Paris trip on a 3M KRW budget."},
                {"groups": [{"topic": "travel", "summary": "예산 요약", "page_indexes": [0]}]},
            ],
        }
    )
    org = MemoryOSOrganizer(stm_capacity=1, heat_threshold=99.0)
    mem = make_mem(org, llm)
    try:
        mem.add_message("파리 여행")
        # first page has no predecessor -> no continuity call yet
        assert "judge" not in [role for role, _ in llm.calls]
        mem.add_message("예산은 300만원")
        assert [role for role, _ in llm.calls].count("judge") == 1

        adds = [o for o in ops_of(mem, "pages") if o.op is OpType.ADD]
        metas = [o.payload.get("meta_info") for o in adds]
        assert metas[0] == "The user is planning a Paris trip."
        assert "3M KRW" in metas[-1]  # chain carried forward, not restarted
        # The query has to clear BOTH upstream gates: the segment's summary must
        # pass `segment_similarity_threshold`, and then a page inside it must
        # pass `page_similarity_threshold`. Missing either returns nothing rather
        # than falling back to the segment summary.
        rendered = mem.search("파리 여행", memory_types=("pages",)).render()
        assert "Conversation chain overview:" in rendered
        assert "user: 파리 여행" in rendered  # verbatim page text, not the summary
    finally:
        mem.close()


def test_memoryos_eval_lineage_merges_the_profile_in_a_second_call():
    """The eval driver analyses WITHOUT the old profile
    (``gpt_personality_analysis``) and merges in a separate ``gpt_update_profile``
    call; pypi folds the old profile into the analysis and does one call. So the
    first promotion costs one call in both lineages (nothing to merge against)
    and the second costs two only under ``fidelity="eval"``."""

    def make_llm():
        return StubLLM(
            {
                "distill": [
                    {"groups": [{"topic": "a", "summary": "s1", "page_indexes": [0]}]},
                    {"profile": "Analysis one: the user is planning a trip to Paris."},
                    {"private": [], "assistant_knowledge": []},
                    {"groups": [{"topic": "b", "summary": "s2", "page_indexes": [0]}]},
                    {"profile": "Analysis two: the user also enjoys art museums."},
                    {"profile": "MERGED: trip to Paris and a taste for art museums."},
                    {"private": [], "assistant_knowledge": []},
                ]
            }
        )

    llm = make_llm()
    # the eval lineage weights length at 0.8, so one page is heat ~0.8
    org = MemoryOSOrganizer(fidelity="eval", heat_threshold=0.5, dialogue_chain=False)
    mem = make_mem(org, llm)
    try:
        mem.add_message("first")
        mem.add_message("second")
        stored = mem.doc_store.get_items([PROFILE_ITEM_ID], "semantic")[0]
        assert stored["content"].startswith("MERGED:")
        # the merge prompt must have seen the FIRST profile as the old one
        merge_prompts = [p for role, p in llm.calls if "Old User Profile:" in p]
        assert len(merge_prompts) == 1 and "Analysis one" in merge_prompts[0]
    finally:
        mem.close()

    # pypi lineage: no merge call at all, the analysis prompt carries the old profile
    llm = make_llm()
    org = MemoryOSOrganizer(
        fidelity="pypi", stm_capacity=1, heat_threshold=1.0, dialogue_chain=False
    )
    mem = make_mem(org, llm)
    try:
        mem.add_message("first")
        mem.add_message("second")
        assert not [p for role, p in llm.calls if "Old User Profile:" in p]
        stored = mem.doc_store.get_items([PROFILE_ITEM_ID], "semantic")[0]
        assert stored["content"].startswith("Analysis two")
    finally:
        mem.close()


def test_zep_predicate_is_screaming_snake_case_and_self_loops_drop():
    """Two paper-§6.1.3 requirements the write path did not enforce.

    The relation type is "a concise, all-caps description of the fact (e.g.,
    LOVES, IS_FRIENDS_WITH, WORKS_FOR)" — upstream's ``relation_type`` field says
    SCREAMING_SNAKE_CASE — and it is part of the edge's identity, so ``lives_in``
    from one message and ``LIVES_IN`` from the next would read as two relation
    types. Normalizing after the model, not just asking in the prompt, is what
    makes it hold with a small model.

    And "each fact should represent a clear relationship between two DISTINCT
    nodes": a self-loop is dropped, which also spares the graph an edge that
    ``edges_between(x, x)`` matches twice through its either-direction clause."""
    from agmem.organizers.zep_graph.organizer import _relation_type

    assert _relation_type("lives in") == "LIVES_IN"
    assert _relation_type("is-friends-with") == "IS_FRIENDS_WITH"
    assert _relation_type("") == "RELATED_TO"

    llm = ZepStub([["Ann", "Bob"]], [[("Ann", "Bob"), ("Ann", "Ann")]])
    org = ZepGraphOrganizer(community_refresh=False)
    mem = make_mem(org, llm)
    try:
        mem.add_message("Ann knows Bob.")
        facts = [o for o in ops_of(mem, "facts") if o.op is OpType.ADD]
        assert len(facts) == 1  # the Ann->Ann fact was dropped
        assert facts[0].payload["predicate"] == "KNOWS"
    finally:
        mem.close()


def test_zep_recent_episode_entities_seed_the_bfs_channel():
    """The paper's motivation for explicit BFS origins is recency: "particularly
    valuable when using recent episodes as seeds ... allowing the system to
    incorporate recently mentioned entities and relationships into the retrieved
    context" (§3.1). Upstream takes them as ``bfs_origin_node_uuids`` and, when
    given, uses them INSTEAD of the origins derived from the other channels."""
    from agmem.config import AgmemConfig
    from agmem.core.ops import MemoryOp

    mem = AgenticMemory(
        namespace="t",
        organizers=["passthrough"],
        embedder=FakeEmbedder(dim=128),
        config=AgmemConfig(bfs_types=("facts",), bfs_max_depth=2),
    )
    try:
        for text in ("oldest turn", "newest turn"):
            mem.add_message(text)
        episode_ids = [e.id for e in mem.doc_store.list_episodes(namespace="t")]
        for node_id, name, provenance in (
            ("A", "Alpha", episode_ids[:1]),
            ("B", "Beta", episode_ids[-1:]),
        ):
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
                        "source_episode_ids": provenance,
                    },
                )
            )
        # newest episode first, so the most recently mentioned entity leads
        assert mem.recent_episode_entity_ids(1) == ["B"]
        assert mem.recent_episode_entity_ids(2) == ["B", "A"]
    finally:
        mem.close()


def test_the_segment_keyword_term_is_dead_in_pypi_and_live_in_the_eval_lineage():
    """`search_sessions` scores a segment as `semantic_sim + alpha * keyword
    overlap`, and only one lineage still computes the second term.

    `memoryos-pypi` sets `query_keywords = set()` right above the loop
    ("Keywords extraction removed"), so its keyword term is structurally 0 — the
    dead-code family this project reproduces rather than repairs. The eval
    harness that produced the paper's LoCoMo numbers instead spends an LLM call
    per query (`llm_extract_keywords`, at most three) and adds the containment
    mean at alpha=1.0. Passing keywords is therefore the whole difference, and it
    is enough to pull in a segment whose summary alone misses the gate."""
    llm = StubLLM(
        {
            "distill": [
                {
                    "groups": [
                        {
                            # summary deliberately shares nothing with the query,
                            # so cosine alone cannot clear segment_threshold
                            "topic": "budget",
                            "summary": "예산 요약",
                            "keywords": ["파리"],
                            "page_indexes": [0],
                        }
                    ]
                }
            ]
        }
    )
    org = MemoryOSOrganizer(stm_capacity=1, heat_threshold=99.0, dialogue_chain=False)
    mem = make_mem(org, llm)
    try:
        mem.add_message("파리 여행")
        assert mem.search("파리 여행", memory_types=("pages",)).items == []  # pypi
        rendered = mem.search(
            "파리 여행", memory_types=("pages",), query_keywords={"파리"}
        ).render()
        assert "user: 파리 여행" in rendered  # eval lineage
    finally:
        mem.close()


def test_the_read_and_merge_keyword_formulas_are_the_same_function():
    """One upstream copy uses ONE overlap formula on both sides, so the read
    step's `keyword_similarity` and the organizer's must agree: `eval` is the
    containment mean, `memoryos-chromadb` is Jaccard (`intersection/union`, both
    in its `insert_pages_into_session` and its `search_sessions`). Pairing one
    copy's read formula with another's merge formula is a combination no
    upstream has, and the two implementations live in different modules — this
    is what stops them drifting."""
    from agmem.retrieval.steps import MemoryOSPageRecall

    a, b = {"x", "y", "z"}, {"x", "w"}  # overlap 1, |A|=3, |B|=2, union 4
    for mode in ("containment_mean", "jaccard"):
        org = MemoryOSOrganizer(keyword_similarity=mode)
        step = MemoryOSPageRecall(keyword_similarity=mode)
        # the step adds the term to a cosine of 0, so the term is the whole score
        from_step = step._relevance({"keywords": sorted(b)}, 0.0, frozenset(a))
        assert from_step == pytest.approx(org._keyword_overlap(a, b)), mode
    # and they are genuinely different formulas, so the pairing matters
    assert MemoryOSOrganizer(keyword_similarity="jaccard")._keyword_overlap(a, b) != pytest.approx(
        MemoryOSOrganizer(keyword_similarity="containment_mean")._keyword_overlap(a, b)
    )


def test_memoryos_page_recall_serves_pages_not_segment_summaries():
    """MemoryOS retrieval is two stages, and the session SUMMARY is a matching
    key rather than context.

    Upstream matches sessions on their summary embedding, then scores every page
    inside a matched session and keeps a global top-`retrieval_queue_capacity`
    of PAGES (`search_sessions` -> `Retriever._retrieve_mid_term_context`). It
    never injects the summary. Serving segment summaries — what this read path
    did before — is a channel upstream has no counterpart for.

    The heat feedback has to survive the substitution: upstream bumps the
    SESSION's N_visit for every session with a matched page, so a served page id
    is resolved back through the unit -> segment index."""
    llm = StubLLM(
        {
            "distill": [
                {
                    "groups": [
                        {"topic": "travel", "summary": "SUMMARY-ONLY-TEXT", "page_indexes": [0]}
                    ]
                }
            ]
        }
    )
    org = MemoryOSOrganizer(stm_capacity=1, heat_threshold=99.0, dialogue_chain=False)
    mem = make_mem(org, llm)
    try:
        mem.add_message("파리 여행 계획")
        (segment_id,) = org._heat
        assert org._heat[segment_id]["n_visit"] == 0

        bundle = mem.search("파리 여행 계획", memory_types=("pages",))
        rendered = bundle.render()
        assert "파리 여행 계획" in rendered  # the verbatim page
        assert "SUMMARY-ONLY-TEXT" not in rendered  # never the summary
        # heat bumped on the SEGMENT even though a page id was served
        assert org._heat[segment_id]["n_visit"] == 1
        assert org._access[segment_id] == 1
    finally:
        mem.close()
