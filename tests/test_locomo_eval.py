"""LoCoMo eval fidelity fixes (issue #1): normalize+stemming, cat3 gold
truncation, cat5 answer prompt selection, and the optional Mem0-style J-score
judge. Unit-level only — no LLM/server (fakes throughout)."""

from agmem import AgenticMemory
from agmem.bench._porter import PorterStemmer
from agmem.bench.locomo import (
    ANSWER_PROMPT,
    ANSWER_PROMPT_NO_ABSTAIN,
    answer,
    gold_for,
    judge_answer,
    normalize,
)
from agmem.embed.fake import FakeEmbedder

_NO_INFO = "No information available"


# ---------------- A) normalize(): stopword "and" + Porter stemming ----------------


def test_normalize_removes_and():
    # "and" joins a/an/the as a removed stopword now.
    assert normalize("cats and dogs") == ["cat", "dog"]
    assert "and" not in normalize("bread and butter")


def test_normalize_porter_stems_tokens():
    assert normalize("running") == ["run"]
    assert normalize("cars") == ["car"]
    assert normalize("studies") == ["studi"]


def test_porter_stemmer_canonical_cases():
    s = PorterStemmer()
    assert s.stem("running") == "run"
    assert s.stem("cars") == "car"
    assert s.stem("studies") == "studi"
    assert s.stem("ponies") == "poni"
    assert s.stem("caresses") == "caress"
    # short words (<=2 chars) are left untouched
    assert s.stem("an") == "an"


# ---------------- B) gold_for(): cat3 semicolon truncation ----------------


def test_gold_for_cat3_semicolon_truncation():
    q = {"answer": "A; B; C", "category": 3}
    assert gold_for(q) == "A"


def test_gold_for_non_cat3_unchanged():
    q = {"answer": "A; B; C", "category": 1}
    assert gold_for(q) == "A; B; C"


def test_gold_for_cat3_without_semicolon_unchanged():
    q = {"answer": "single", "category": 3}
    assert gold_for(q) == "single"


def test_gold_for_adversarial_fallback():
    q = {"adversarial_answer": "no way", "category": 5}
    assert gold_for(q) == "no way"


def test_gold_for_missing_all_is_empty_string():
    assert gold_for({"category": 4}) == ""


# ---------------- C) answer(): cat5 drops the abstention line ----------------


class _CapturingLLM:
    """Records the last generate prompt so we can inspect prompt selection."""

    def __init__(self):
        self.last_prompt = None

    def chat(self, role, messages):
        self.last_prompt = messages[0]["content"]
        return "captured"


def _mem_with_capturing_llm():
    mem = AgenticMemory(namespace="t", organizers=["passthrough"], embedder=FakeEmbedder(dim=128))
    mem.llm = _CapturingLLM()
    return mem


def test_answer_cat5_prompt_has_no_abstention_line():
    # the constant itself must differ only by the abstention sentence
    assert _NO_INFO in ANSWER_PROMPT
    assert _NO_INFO not in ANSWER_PROMPT_NO_ABSTAIN

    mem = _mem_with_capturing_llm()
    try:
        answer(mem, "q?", category=5)
        assert _NO_INFO not in mem.llm.last_prompt
    finally:
        mem.close()


def test_answer_cat1_prompt_keeps_abstention_line():
    mem = _mem_with_capturing_llm()
    try:
        answer(mem, "q?", category=1)
        assert _NO_INFO in mem.llm.last_prompt
    finally:
        mem.close()


# ---------------- D) judge_answer(): Mem0-style binary verdict ----------------


class _StubJudge:
    def __init__(self, label):
        self.label = label

    def call(self, role, prompt, schema, required_keys=()):
        if self.label is None:
            return None
        return {"label": self.label}


class _FakeMem:
    def __init__(self, structured):
        self.structured = structured


def test_judge_answer_correct():
    mem = _FakeMem(_StubJudge("CORRECT"))
    assert judge_answer(mem, "q", "gold", "pred") is True


def test_judge_answer_wrong():
    mem = _FakeMem(_StubJudge("WRONG"))
    assert judge_answer(mem, "q", "gold", "pred") is False


def test_judge_answer_none_client():
    mem = _FakeMem(None)
    assert judge_answer(mem, "q", "gold", "pred") is None


def test_judge_answer_empty_result():
    mem = _FakeMem(_StubJudge(None))
    assert judge_answer(mem, "q", "gold", "pred") is None
