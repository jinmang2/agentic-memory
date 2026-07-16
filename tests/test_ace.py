from agmem import AgenticMemory
from agmem.core.ops import OpType
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.ace import ACEOrganizer

from helpers import StubLLM


def make_mem(llm):
    mem = AgenticMemory(namespace="t", organizers=[ACEOrganizer()],
                        embedder=FakeEmbedder(dim=128))
    mem.structured = llm
    mem._ctx.llm = llm
    return mem


def ace_llm():
    return StubLLM({
        "distill": [
            {"key_insight": "Always validate filter state",
             "lessons": ["Check filters before submitting"],
             "bullet_tags": []},
            {"operations": [
                {"type": "ADD", "section": "web_forms",
                 "content": "Verify every filter control state before submit."},
            ]},
        ],
    })


def test_ace_adds_playbook_bullet_and_renders():
    mem = make_mem(ace_llm())
    try:
        mem.add_task_result(trajectory=[{"a": 1}], outcome="success",
                            task="filter products")
        adds = [o for o in mem.log.tail(10)
                if o.target_type == "playbook" and o.op is OpType.ADD]
        assert len(adds) == 1 and adds[0].actor == "ace"
        rendered = mem.get_playbook()
        assert "web_forms" in rendered and "helpful=0" in rendered
    finally:
        mem.close()


def test_ace_dedup_skips_near_duplicate():
    llm = ace_llm()
    # second task curates an identical bullet -> must be deduped
    llm.responses["distill"] += [
        {"key_insight": "same", "lessons": ["same"], "bullet_tags": []},
        {"operations": [{"type": "ADD", "section": "web_forms",
                         "content": "Verify every filter control state before submit."}]},
    ]
    mem = make_mem(llm)
    try:
        mem.add_task_result(trajectory=[], outcome="success", task="task one")
        mem.add_task_result(trajectory=[], outcome="success", task="task two")
        adds = [o for o in mem.log.tail(20)
                if o.target_type == "playbook" and o.op is OpType.ADD]
        assert len(adds) == 1  # exact duplicate embedding -> sim 1.0 >= 0.90
    finally:
        mem.close()


def test_ace_feedback_increments_counters():
    mem = make_mem(ace_llm())
    try:
        mem.add_task_result(trajectory=[], outcome="success", task="filter products")
        bullet_id = [o for o in mem.log.tail(10) if o.target_type == "playbook"][0].target_id
        assert mem.report_feedback([bullet_id], helpful=True) == 1
        assert mem.report_feedback([bullet_id], helpful=False) == 1
        data = mem.doc.get_items([bullet_id], "playbook")[0]
        assert data["helpful"] == 1 and data["harmful"] == 1
    finally:
        mem.close()


def test_llm_reranker_reorders_and_survives_failure():
    from agmem.retrieval.rerank import LLMReranker

    candidates = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    texts = {"a": "cat food", "b": "paris travel budget", "c": "gym schedule"}

    good = StubLLM({"rerank": [{"ranking": [1, 99, 0]}]})  # 99 invalid -> ignored
    out = LLMReranker(good).rerank(None, candidates, {}, 2, texts=texts, query="trip cost")
    assert [c for c, _ in out] == ["b", "a"]

    broken = StubLLM({})  # returns None -> drop -> fusion order preserved
    out = LLMReranker(broken).rerank(None, candidates, {}, 2, texts=texts, query="q")
    assert [c for c, _ in out] == ["a", "b"]
