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
