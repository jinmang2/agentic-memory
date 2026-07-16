"""Phase 1 exit criterion: MemoryOp abstraction holds for RB + A-Mem."""

import pytest

from agmem import AgenticMemory
from agmem.config import AgmemConfig
from agmem.core.ops import OpType
from agmem.embed.fake import FakeEmbedder


class StubLLM:
    """StructuredCaller stand-in: returns queued responses per role."""

    def __init__(self, responses: dict[str, list]):
        self.responses = {role: list(items) for role, items in responses.items()}
        self.calls: list[tuple[str, str]] = []
        self.drops: dict[str, int] = {}

    def call(self, role, prompt, schema, required_keys=(), **kw):
        self.calls.append((role, prompt))
        items = self.responses.get(role)
        if not items:
            self.drops[role] = self.drops.get(role, 0) + 1
            return None
        return items.pop(0)


def make_mem(organizer, llm) -> AgenticMemory:
    mem = AgenticMemory(namespace="t", organizers=[organizer],
                        embedder=FakeEmbedder(dim=128))
    mem.structured = llm
    mem._ctx.llm = llm
    return mem


# ---------------- ReasoningBank ----------------

def rb_llm(success=True):
    return StubLLM({
        "judge": [{"success": success, "reason": "checked final state"}],
        "distill": [{"items": [
            {"title": "Verify filters before submit",
             "description": "Use when a form has filter controls",
             "content": "Check each filter state, then submit."},
            {"title": "", "description": "broken item", "content": "x"},  # dropped
            {"title": "Re-read error banners",
             "description": "Use after any failed action",
             "content": "Error text usually names the missing field."},
        ]}],
    })


def test_reasoning_bank_distills_and_indexes():
    from agmem.organizers.reasoning_bank import ReasoningBankOrganizer

    llm = rb_llm()
    mem = make_mem(ReasoningBankOrganizer(), llm)
    try:
        mem.add_task_result(trajectory=[{"a": 1}], outcome="unknown",
                            task="filter products by price")
        # judge was consulted for unknown outcome, then distill
        assert [r for r, _ in llm.calls] == ["judge", "distill"]
        ops = mem.log.tail(10)
        strategy_ops = [o for o in ops if o.target_type == "strategies"]
        assert len(strategy_ops) == 2  # broken item skipped (field fallback)
        assert all(o.actor == "reasoning_bank" for o in strategy_ops)

        bundle = mem.search("form filters", memory_types=["strategies"], k=2)
        titles = {s.item.data["title"] for s in bundle.items}
        assert "Verify filters before submit" in titles
    finally:
        mem.close()


def test_reasoning_bank_known_outcome_skips_judge():
    from agmem.organizers.reasoning_bank import ReasoningBankOrganizer

    llm = rb_llm()
    mem = make_mem(ReasoningBankOrganizer(), llm)
    try:
        mem.add_task_result(trajectory=[], outcome="failure", task="t")
        assert [r for r, _ in llm.calls] == ["distill"]
        assert "FAILED" in llm.calls[0][1]  # failure prompt variant chosen
    finally:
        mem.close()


def test_reasoning_bank_no_llm_explicit_skip():
    from agmem.organizers.reasoning_bank import ReasoningBankOrganizer

    mem = AgenticMemory(namespace="t", organizers=[ReasoningBankOrganizer()],
                        embedder=FakeEmbedder(dim=128))
    try:
        mem.add_task_result(trajectory=[], outcome="success", task="t")
        # raw episode logged, but no strategy ops
        assert all(o.target_type != "strategies" for o in mem.log.tail(10))
    finally:
        mem.close()


# ---------------- A-Mem ----------------

def test_amem_note_link_and_evolution():
    from agmem.organizers.amem import AMemOrganizer

    llm = StubLLM({
        "extract": [
            {"keywords": ["파리", "여행"], "context": "파리 여행 계획", "tags": ["travel"]},
            {"keywords": ["파리", "예산"], "context": "파리 여행 예산", "tags": ["travel", "budget"]},
        ],
        "distill": [
            # second note's evolution: link to first + update its context
            {"should_evolve": True, "connections": ["__FIRST__"],
             "neighbor_updates": [{"id": "__FIRST__",
                                   "new_context": "파리 여행 계획 (예산 300만원 확정)",
                                   "new_tags": ["travel", "budget"]}]},
        ],
    })
    mem = make_mem(AMemOrganizer(top_k=5), llm)
    try:
        mem.add_message("다음 달에 파리로 여행 가려고 해")
        first_ops = [o for o in mem.log.tail(10) if o.target_type == "notes"]
        first_id = first_ops[0].target_id

        # patch the stub's placeholder with the real neighbor id
        resp = llm.responses["distill"][0]
        resp["connections"] = [first_id]
        resp["neighbor_updates"][0]["id"] = first_id

        mem.add_message("파리 여행 예산은 300만원이야")

        ops = mem.log.tail(20)
        kinds = [(o.op, o.target_type) for o in ops]
        assert (OpType.LINK, "notes") in kinds
        assert (OpType.UPDATE, "notes") in kinds

        # bidirectional link: first note gained a link to the second
        first = mem.doc.get_items([first_id], "notes")[0]
        assert first["links"], "neighbor should be linked back"
        # UPDATE merged, not clobbered: content survived the context rewrite
        assert first["content"] == "다음 달에 파리로 여행 가려고 해"
        assert "300만원" in first["context"]
    finally:
        mem.close()


def test_amem_hallucinated_neighbor_ids_ignored():
    from agmem.organizers.amem import AMemOrganizer

    llm = StubLLM({
        "extract": [
            {"keywords": ["a"], "context": "c1", "tags": ["t"]},
            {"keywords": ["a"], "context": "c2", "tags": ["t"]},
        ],
        "distill": [
            {"should_evolve": True, "connections": ["not-a-real-id"],
             "neighbor_updates": [{"id": "also-fake", "new_context": "x"}]},
        ],
    })
    mem = make_mem(AMemOrganizer(), llm)
    try:
        mem.add_message("first note about topic alpha")
        mem.add_message("second note about topic alpha")
        ops = mem.log.tail(20)
        # bug-fix #32 behavior: fake ids produce no LINK/UPDATE ops
        assert all(o.op not in (OpType.LINK, OpType.UPDATE) for o in ops)
    finally:
        mem.close()


def test_amem_degrades_without_llm():
    from agmem.organizers.amem import AMemOrganizer

    mem = AgenticMemory(namespace="t", organizers=[AMemOrganizer()],
                        embedder=FakeEmbedder(dim=128))
    try:
        mem.add_message("bare note without llm")
        notes = [o for o in mem.log.tail(10) if o.target_type == "notes"]
        assert len(notes) == 1  # bare note stored, no crash
    finally:
        mem.close()


# ---------------- async worker ----------------

def test_async_write_flush():
    from agmem.organizers.reasoning_bank import ReasoningBankOrganizer

    llm = rb_llm()
    mem = AgenticMemory(namespace="t", organizers=[ReasoningBankOrganizer()],
                        embedder=FakeEmbedder(dim=128),
                        config=AgmemConfig(sync_write=False))
    mem.structured = llm
    mem._ctx.llm = llm
    try:
        mem.add_task_result(trajectory=[], outcome="success", task="do a thing")
        mem.flush()  # block until worker applied the ops
        strategy_ops = [o for o in mem.log.tail(10) if o.target_type == "strategies"]
        assert len(strategy_ops) == 2
    finally:
        mem.close()


# ---------------- MMR ----------------

def test_mmr_prefers_diversity():
    from agmem.retrieval.rerank import MMRReranker

    # two near-duplicates + one distinct; query must not coincide with the
    # duplicates (when query == dup, relevance == redundancy and MMR ties)
    vectors = {
        "dup1": [0.95, 0.31, 0.0],
        "dup2": [0.95, 0.312, 0.0],
        "other": [0.5, -0.866, 0.0],
    }
    candidates = [("dup1", 0.9), ("dup2", 0.89), ("other", 0.7)]
    picked = MMRReranker(lambda_=0.5).rerank([1.0, 0.0, 0.0], candidates, vectors, k=2)
    ids = [c for c, _ in picked]
    assert ids[0] == "dup1"
    assert ids[1] == "other"  # diversity beats the near-duplicate
