"""Fidelity-audit P0 read-path behaviors (docs/research/fidelity-deep-audit.md §6)."""

from agmem import AgenticMemory
from agmem.core.types import Episode
from agmem.embed.fake import FakeEmbedder


def make_mem():
    return AgenticMemory(namespace="t", organizers=["passthrough"],
                         embedder=FakeEmbedder(dim=128))


def put_indexed(mem, item_id, memory_type, data, text=None):
    mem.doc_store.put_item(item_id, memory_type, "t", {"id": item_id, **data})
    mem.vector_store.add(item_id, mem.embedder.embed([text or data.get("content", "")])[0],
                memory_type=memory_type, namespace="t")


def test_amem_one_hop_link_expansion():
    mem = make_mem()
    try:
        put_indexed(mem, "n1", "notes",
                    {"content": "paris travel museums", "links": ["n2"]})
        # n2 is lexically unrelated to the query — reachable only via the link
        put_indexed(mem, "n2", "notes",
                    {"content": "budget three million won", "links": ["n1"]})
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
        assert len(bundle.items) <= 1 + mem.pipeline.link_expansion_cap
    finally:
        mem.close()


def test_nemori_top_r_source_attachment():
    mem = make_mem()
    try:
        raw = Episode(content="I paid exactly 2,340,000 won for flights",
                      namespace="t")
        mem.doc_store.add_episode(raw)
        put_indexed(mem, "ep1", "episodes",
                    {"title": "Flight booking",
                     "content": "The user booked flights to Paris.",
                     "source_episode_ids": [raw.id]})
        bundle = mem.search("flight booking", memory_types=["episodes"], k=1)
        rendered = bundle.render(budget_tokens=6000)
        assert "Source Messages:" in rendered
        assert "2,340,000" in rendered  # verbatim detail restored from raw
    finally:
        mem.close()


def test_render_exposes_metadata():
    mem = make_mem()
    try:
        put_indexed(mem, "n1", "notes",
                    {"content": "likes museums", "context": "user preference",
                     "tags": ["travel", "art"], "links": []})
        put_indexed(mem, "s1", "semantic",
                    {"content": "The user lives in Seoul.",
                     "timestamp": "2023-05-07T13:00:00"})
        rendered = mem.search("museums seoul", memory_types=["notes", "semantic"],
                              k=3).render(budget_tokens=6000)
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
        bundle = mem.search("cats", memory_types=["episodes", "semantic"],
                            k={"episodes": 2, "semantic": 5})
        by_type = {}
        for s in bundle.items:
            by_type[s.memory_type] = by_type.get(s.memory_type, 0) + 1
        assert by_type["episodes"] == 2 and by_type["semantic"] == 5
    finally:
        mem.close()
