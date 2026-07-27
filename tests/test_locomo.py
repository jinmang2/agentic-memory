from agmem import AgenticMemory
from agmem.bench.locomo import (
    answer,
    bleu1,
    evidence_sessions,
    iter_turns,
    select_questions,
    token_f1,
)
from agmem.embed.fake import FakeEmbedder


def test_token_f1_squad_normalization():
    assert token_f1("The 7 May, 2023", "7 May 2023") == 1.0
    assert token_f1("May 2023", "7 May 2023") > 0.5
    assert token_f1("no idea", "7 May 2023") == 0.0


def test_bleu1_brevity():
    assert bleu1("7 May 2023", "7 May 2023") == 1.0
    assert 0 < bleu1("May", "7 May 2023") < 1.0  # short pred penalized


SAMPLE = {
    "conversation": {
        "session_1": [{"speaker": "A", "text": "hi"}, {"speaker": "B", "text": "hey"}],
        "session_1_date_time": "1 pm on 1 May, 2023",
        "session_2": [{"speaker": "A", "text": "news!"}],
        "session_2_date_time": "2 pm on 8 May, 2023",
    },
    "qa": [
        {"question": "q1", "answer": "a1", "evidence": ["D1:1"], "category": 4},
        {"question": "q2", "answer": "a2", "evidence": ["D2:1"], "category": 2},
        {"question": "q5", "adversarial_answer": "x", "evidence": [], "category": 5},
    ],
}


def test_iter_turns_ordered_with_dates():
    turns = list(iter_turns(SAMPLE))
    assert len(turns) == 3
    assert turns[0] == (1, "1 pm on 1 May, 2023", "A", "hi")
    assert turns[-1][0] == 2


def test_select_questions_respects_session_prefix():
    qs = select_questions(SAMPLE, max_sessions=1)
    ids = [q["question"] for q in qs]
    assert "q1" in ids and "q2" not in ids  # q2 evidence in session 2
    assert "q5" in ids  # no evidence -> always eligible


def test_evidence_sessions_parsing():
    assert evidence_sessions({"evidence": ["D3:12", "D10:4"]}) == {3, 10}


class _StubLLM:
    def chat(self, role, messages):
        return "stub answer"


class _StubStructured:
    def __init__(self, keywords):
        self.keywords = keywords

    def call(self, role, prompt, schema, required_keys=()):
        return {"keywords": self.keywords}


def test_keyword_queries_rewrite_the_search_query():
    """A-Mem official eval: the LLM keyword string replaces the raw question."""
    mem = AgenticMemory(namespace="t", organizers=["passthrough"], embedder=FakeEmbedder(dim=128))
    try:
        mem.llm = _StubLLM()
        mem.structured = _StubStructured("paris, museums, travel")
        seen = {}
        original_search = mem.search

        def spy(query, **kwargs):
            seen["query"] = query
            return original_search(query, **kwargs)

        mem.search = spy
        answer(mem, "Where did A travel?", keyword_queries=True)
        assert seen["query"] == "paris, museums, travel"

        answer(mem, "Where did A travel?", keyword_queries=False)
        assert seen["query"] == "Where did A travel?"
    finally:
        mem.close()


class _RecordingLLM:
    def __init__(self):
        self.prompts = []

    def chat(self, role, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        return "stub answer"


def _mem_with_assistant_knowledge():
    mem = AgenticMemory(namespace="t", organizers=["passthrough"], embedder=FakeEmbedder(dim=128))
    mem.llm = _RecordingLLM()
    mem.doc_store.put_item(
        "ak1",
        "semantic",
        "t",
        {"id": "ak1", "kind": "assistant_knowledge", "content": "Assistant booked a flight"},
    )
    return mem


def test_full_assistant_knowledge_is_the_eval_lineage_only():
    """MemoryOS's two lineages serve assistant knowledge differently: the pypi
    library retrieves top-20 through the same `semantic` channel our search
    already covers, while the eval harness that produced the paper's LoCoMo
    numbers dumps the whole store into every prompt
    (`main_loco_parse`: `long_mem.get_assistant_knowledge()`). Injecting it
    unconditionally gave a pypi-lineage run BOTH, which is the lineage mixing
    `MEMORYOS_PRESETS` exists to prevent."""
    mem = _mem_with_assistant_knowledge()
    try:
        answer(mem, "what happened?", memory_types=("semantic",))
        assert "Assistant Knowledge:" not in mem.llm.prompts[-1]

        answer(
            mem,
            "what happened?",
            memory_types=("semantic",),
            memoryos_lineage="eval",
        )
        prompt = mem.llm.prompts[-1]
        assert "Assistant Knowledge:" in prompt
        assert "Assistant booked a flight" in prompt
    finally:
        mem.close()


def test_keyword_queries_fall_back_to_question_on_llm_failure():
    mem = AgenticMemory(namespace="t", organizers=["passthrough"], embedder=FakeEmbedder(dim=128))
    try:
        mem.llm = _StubLLM()
        failing = _StubStructured("")
        failing.call = lambda *a, **k: None
        mem.structured = failing
        seen = {}
        original_search = mem.search

        def spy(query, **kwargs):
            seen["query"] = query
            return original_search(query, **kwargs)

        mem.search = spy
        answer(mem, "Where did A travel?", keyword_queries=True)
        assert seen["query"] == "Where did A travel?"
    finally:
        mem.close()
