"""Phase 3: Nemori, MemoryOS, Zep-graph, G-Memory through the same MemoryOp contract."""

from agmem import AgenticMemory
from agmem.core.ops import OpType
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.gmemory import GMemoryOrganizer
from agmem.organizers.memoryos import MemoryOSOrganizer
from agmem.organizers.nemori import NemoriOrganizer
from agmem.organizers.zep_graph import ZepGraphOrganizer

from helpers import StubLLM


def make_mem(organizer, llm):
    mem = AgenticMemory(namespace="t", organizers=[organizer],
                        embedder=FakeEmbedder(dim=128))
    mem.structured = llm
    mem._ctx.llm = llm
    return mem


def ops_of(mem, ttype):
    return [o for o in mem.log.tail(50) if o.target_type == ttype]


# ---------------- Nemori ----------------

def test_nemori_boundary_flush_and_calibrate():
    llm = StubLLM({
        "extract": [  # boundary checks (from 2nd message on)
            {"boundary": False, "confidence": 0.9},
            {"boundary": True, "confidence": 0.95},
        ],
        "distill": [
            {"title": "Paris trip planning", "narrative": "On 1 May 2023, the user planned a trip.",
             "timestamp": "2023-05-01"},
            # cold start (no prior semantic memory) -> direct extraction, one call
            {"facts": ["The user's trip budget is 3,000,000 KRW."]},
        ],
    })
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
    llm = StubLLM({
        "extract": [{"boundary": True, "confidence": 0.9}],
        "distill": [],  # episode generation returns None -> fallback
    })
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
    mem = AgenticMemory(namespace="t", organizers=[NemoriOrganizer()],
                        embedder=FakeEmbedder(dim=128))
    try:
        mem.add_message("hello")
        assert not ops_of(mem, "episodes")
    finally:
        mem.close()


# ---------------- MemoryOS ----------------

def test_memoryos_eviction_creates_segment_and_promotes():
    llm = StubLLM({
        "distill": [
            {"groups": [{"topic": "travel", "summary": "User plans a Paris trip.",
                         "message_indexes": [0, 1, 2]}]},
            {"profile_facts": ["User is planning a Paris trip."]},  # heat trigger
        ],
    })
    org = MemoryOSOrganizer(stm_capacity=3, heat_threshold=1.0)  # trigger promotion
    mem = make_mem(org, llm)
    try:
        for text in ("파리 가자", "예산 300", "미술관 위주"):
            mem.add_message(text)
        pages = ops_of(mem, "pages")
        assert len(pages) == 1 and pages[0].payload["topic"] == "travel"
        profile = ops_of(mem, "semantic")
        assert len(profile) == 1 and profile[0].payload["kind"] == "profile"
        # heat reset after promotion
        seg_id = pages[0].target_id
        assert org._heat[seg_id]["n_visit"] == 0 and org._heat[seg_id]["length"] == 0
    finally:
        mem.close()


def test_memoryos_no_llm_mechanical_segment():
    org = MemoryOSOrganizer(stm_capacity=2)
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
    llm = StubLLM({
        "extract": [
            {"entities": [{"name": "Caroline", "type": "Person", "summary": "a user"},
                          {"name": "Seoul", "type": "Place", "summary": "a city"}]},
            {"facts": [{"subject": "Caroline", "predicate": "lives_in",
                        "object": "Seoul", "statement": "Caroline lives in Seoul."}]},
            {"entities": [{"name": "Caroline", "type": "Person", "summary": "a user"},
                          {"name": "Busan", "type": "Place", "summary": "a city"}]},
            {"facts": [{"subject": "Caroline", "predicate": "lives_in",
                        "object": "Busan", "statement": "Caroline lives in Busan."}]},
        ],
        "distill": [],  # no edges between Caroline-Busan yet -> no contradiction call
    })
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
    llm = StubLLM({
        "extract": [
            {"entities": [{"name": "A", "summary": "s"}, {"name": "B", "summary": "s"}]},
            {"facts": [{"subject": "A", "predicate": "likes", "object": "B",
                        "statement": "A likes B."}]},
            {"entities": [{"name": "A", "summary": "s"}, {"name": "B", "summary": "s"}]},
            {"facts": [{"subject": "A", "predicate": "dislikes", "object": "B",
                        "statement": "A dislikes B."}]},
        ],
        "distill": [{"contradicts": ["__EDGE__"]}],
    })
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
        item = mem.doc.get_items([first_edge], "facts")[0]
        assert item.get("invalid_at")
        a_node = org.graph.find_node_by_name("A", "t")
        b_node = org.graph.find_node_by_name("B", "t")
        active = org.graph.edges_between(a_node["id"], b_node["id"], "t")
        assert len(active) == 1 and active[0]["content"] == "A dislikes B."
    finally:
        mem.close()


# ---------------- G-Memory ----------------

def test_gmemory_trajectory_and_insight_finetune():
    llm = StubLLM({
        "distill": [
            {"key_steps": ["searched", "clicked"], "mistakes": []},
            {"key_steps": ["opened settings"], "mistakes": ["wrong tab first"]},
            {"operations": [
                {"op": "ADD", "rule": "Always open settings from the sidebar."},
                {"op": "EDIT", "id": "fake-id", "rule": "ignored"},  # hallucinated
            ]},
        ],
    })
    org = GMemoryOrganizer(finetune_every=2)
    mem = make_mem(org, llm)
    try:
        mem.add_task_result([{"s": 1}], "success", "task one")
        mem.add_task_result([{"s": 2}], "failure", "task two")
        strategies = ops_of(mem, "strategies")
        trajs = [o for o in strategies if o.payload.get("kind") == "trajectory"]
        insights = [o for o in strategies if o.payload.get("kind") == "insight"]
        assert len(trajs) == 2
        assert trajs[0].payload["score"] == 1.0 and trajs[1].payload["score"] == -2.0
        assert len(insights) == 1  # hallucinated EDIT ignored
        assert not [o for o in strategies if o.op is OpType.UPDATE]
    finally:
        mem.close()
