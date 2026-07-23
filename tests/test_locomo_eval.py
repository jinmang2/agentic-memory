"""LoCoMo eval fidelity fixes (issue #1): normalize+stemming, cat3 gold
truncation, cat5 answer prompt selection, and the optional Mem0-style J-score
judge. Unit-level only — no LLM/server (fakes throughout)."""

from agmem import AgenticMemory
from agmem.bench._porter import PorterStemmer
from agmem.bench.locomo import (
    ANSWER_PROMPT,
    ANSWER_PROMPT_NO_ABSTAIN,
    CAT5_NOT_MENTIONED,
    answer,
    cat5_options,
    gold_for,
    gold_for_wujiang,
    judge_answer,
    normalize,
    resolve_cat5_reply,
    token_f1_wujiang,
    _tok_wujiang,
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


# ---------------- E) WujiangXu/A-Mem faithful eval (eval_mode="wujiang") -------


def test_wujiang_tokenizer_replaces_punctuation_and_lowercases():
    # utils.py:34-38 — lowercase, replace . , ! ? with space, split.
    assert set(_tok_wujiang("A, b! c.")) == {"a", "b", "c"}
    assert _tok_wujiang("What? Yes!") == ["what", "yes"]
    # NO stemming (unlike ours normalize): "running" stays "running".
    assert _tok_wujiang("running cars") == ["running", "cars"]


def test_wujiang_f1_is_set_based_dedup():
    # Repeated tokens must NOT inflate the score (set semantics, utils.py:136-137).
    assert token_f1_wujiang("cat dog", "cat dog") == 1.0
    assert token_f1_wujiang("cat cat cat dog", "cat dog") == 1.0
    # partial overlap: pred set {a,b}, gold set {a} -> P=1/2, R=1, F1=2/3.
    assert abs(token_f1_wujiang("a b", "a") - (2 / 3)) < 1e-9
    # empty pred or gold -> 0.0
    assert token_f1_wujiang("", "gold") == 0.0
    assert token_f1_wujiang("pred", "") == 0.0
    # no overlap -> 0.0
    assert token_f1_wujiang("x y", "a b") == 0.0


def test_wujiang_gold_no_cat3_truncation():
    # unlike gold_for (ours), wujiang keeps the whole cat3 gold.
    q = {"answer": "A; B; C", "category": 3}
    assert gold_for_wujiang(q) == "A; B; C"
    assert gold_for(q) == "A"  # ours mode still truncates
    # cat5 pulls adversarial_answer.
    assert gold_for_wujiang({"adversarial_answer": "no way", "category": 5}) == "no way"


def test_cat5_options_reproducible_and_contains_both():
    # same question -> same option order across calls (md5-seeded, not random).
    o1 = cat5_options("Did the trip happen?", "yes it did")
    o2 = cat5_options("Did the trip happen?", "yes it did")
    assert o1 == o2
    assert set(o1) == {"yes it did", CAT5_NOT_MENTIONED}
    # a different question can legitimately flip the order — but each is stable.
    assert cat5_options("Another q?", "gold") == cat5_options("Another q?", "gold")


def test_resolve_cat5_reply_letter_and_text():
    opts = ("Not mentioned in the conversation", "he moved to Berlin")
    assert resolve_cat5_reply("(a)", opts) == opts[0]
    assert resolve_cat5_reply("b", opts) == opts[1]
    # a raw text reply is returned as-is (upstream scores raw reply vs gold).
    assert resolve_cat5_reply("he moved to Berlin", opts) == "he moved to Berlin"


def test_answer_wujiang_cat5_uses_mcq_prompt():
    # cat5 in wujiang mode presents the 2-option MCQ; both options appear and
    # the abstention line from ANSWER_PROMPT does NOT.
    mem = _mem_with_capturing_llm()
    try:
        answer(mem, "Did X happen?", category=5, eval_mode="wujiang", gold="yes X happened")
        prompt = mem.llm.last_prompt
        assert "Select the correct answer:" in prompt
        assert CAT5_NOT_MENTIONED in prompt
        assert "yes X happened" in prompt
        assert _NO_INFO not in prompt
    finally:
        mem.close()


def test_answer_ours_cat5_still_uses_no_abstain_prompt():
    # eval_mode defaults to "ours"; cat5 there keeps the no-abstain span prompt,
    # NOT the MCQ — proves the two paths are cleanly selected.
    mem = _mem_with_capturing_llm()
    try:
        answer(mem, "Did X happen?", category=5)
        assert "Select the correct answer:" not in mem.llm.last_prompt
        assert _NO_INFO not in mem.llm.last_prompt  # cat5 drops abstention
    finally:
        mem.close()
