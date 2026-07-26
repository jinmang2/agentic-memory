"""LongMemEval pipeline: the parts where getting it subtly wrong is silent.

Every assertion here is anchored to the official repo (xiaowu0162/LongMemEval)
rather than to our own convenience — see src/agmem/bench/longmemeval.py for the
file:line citations.
"""

import pytest

from agmem import AgenticMemory
from agmem.bench.longmemeval import (
    ANSWER_PROMPT_CON,
    JUDGE_MODEL_PIN,
    aggregate,
    answer,
    check_judge_model,
    get_anscheck_prompt,
    ingest,
    is_abstention,
    iter_turns,
    render_sessions,
    run_instance,
    upstream_max_history_tokens,
)
from agmem.embed.fake import FakeEmbedder
from agmem.llm.client import RoleConfig

INSTANCE = {
    "question_id": "q1",
    "question_type": "single-session-user",
    "question": "What did I say about the trip?",
    "answer": "Paris in May",
    "question_date": "2023/06/01 (Thu) 10:00",
    "haystack_session_ids": ["s1", "s2"],
    "haystack_dates": ["2023/05/01 (Mon) 09:00", "2023/05/08 (Mon) 09:00"],
    "haystack_sessions": [
        [
            {"role": "user", "content": "I am going to Paris in May", "has_answer": True},
            {"role": "assistant", "content": "Sounds great"},
        ],
        [{"role": "user", "content": "unrelated filler"}],
    ],
    "answer_session_ids": ["s1"],
}


def _mem(organizers=("passthrough",)):
    return AgenticMemory(namespace="t", organizers=list(organizers), embedder=FakeEmbedder(dim=64))


class _StubLLM:
    """Records every chat call so prompt/temperature/max_tokens are assertable."""

    def __init__(self, replies=None):
        self.replies = dict(replies or {})
        self.calls = []

    def has_role(self, role):
        return role in ("generate", "judge")

    def chat(self, role, messages, **kwargs):
        self.calls.append({"role": role, "prompt": messages[0]["content"], **kwargs})
        return self.replies.get(role, "stub reply")


# ---------------- loading / ingest ----------------


def test_iter_turns_zips_the_parallel_haystack_arrays():
    turns = list(iter_turns(INSTANCE))
    assert [(t[0], t[2]) for t in turns] == [
        ("s1", "user"),
        ("s1", "assistant"),
        ("s2", "user"),
    ]
    assert turns[0][1] == "2023/05/01 (Mon) 09:00"  # session date, not turn date
    assert turns[0][4] is True and turns[1][4] is False  # has_answer evidence label


def test_ingest_never_writes_the_has_answer_gold_label():
    """`has_answer` marks which turns contain the answer. Leaking it into the
    memory would let retrieval cheat, and the leak would look like a good score.

    Checked against what retrieval actually serves, not against
    `list_items("episodic")` — raw episodes live in their own table, so that
    call returns [] and would make this assertion vacuously true."""
    mem = _mem()
    try:
        assert ingest(mem, INSTANCE) == 3
        served = mem.search("Paris trip", k=10)
        assert len(served.items) == 3, "all three turns are retrievable"
        blob = served.render(budget_tokens=4000) + "".join(
            str(getattr(s.item, "meta", "")) for s in served.items
        )
        assert "has_answer" not in blob
        assert "I am going to Paris in May" in blob  # content itself did land
    finally:
        mem.close()


def test_max_sessions_truncates_the_haystack():
    mem = _mem()
    try:
        assert ingest(mem, INSTANCE, max_sessions=1) == 2
    finally:
        mem.close()


def test_max_sessions_zero_means_zero_not_all():
    """`sessions[:max_sessions or len(sessions)]` reads 0 as "no cap" and
    silently runs the FULL haystack — the smallest ablation point turning into
    the largest run is the kind of error that only shows up in the bill."""
    assert list(iter_turns(INSTANCE, max_sessions=0)) == []
    assert render_sessions(INSTANCE, max_sessions=0) == ""


# ---------------- abstention ----------------


def test_abstention_is_a_substring_test_as_upstream():
    """evaluate_qa.py:101 uses `'_abs' in question_id`, NOT endswith. Tightening
    it here would silently regrade any id carrying `_abs` mid-string."""
    assert is_abstention("q5_abs")
    assert is_abstention("q5_abs_variant2")  # endswith() would say False
    assert not is_abstention("q5")


# ---------------- judge prompts ----------------


def test_knowledge_update_drops_the_subset_clause():
    """The five branches are not stylistic variants. knowledge-update omits the
    "only contains a subset -> answer no" rule the default branch carries,
    because a KU answer legitimately restates old information alongside the new."""
    default = get_anscheck_prompt("single-session-user", "q", "a", "r", abstention=False)
    ku = get_anscheck_prompt("knowledge-update", "q", "a", "r", abstention=False)
    assert "only contains a subset" in default
    assert "only contains a subset" not in ku
    assert "along with an updated answer" in ku


def test_preference_grades_against_a_rubric_not_an_answer():
    p = get_anscheck_prompt("single-session-preference", "q", "a", "r", abstention=False)
    assert "Rubric: a" in p and "Correct Answer:" not in p


def test_temporal_reasoning_forgives_off_by_one():
    p = get_anscheck_prompt("temporal-reasoning", "q", "a", "r", abstention=False)
    assert "off-by-one" in p and "only contains a subset" in p


def test_abstention_overrides_the_question_type():
    for qtype in ("single-session-user", "temporal-reasoning", "knowledge-update"):
        p = get_anscheck_prompt(qtype, "q", "a", "r", abstention=True)
        assert "unanswerable" in p and "Explanation: a" in p


def test_unknown_question_type_raises_like_upstream():
    """evaluate_qa.py raises NotImplementedError rather than defaulting — a
    mis-branched prompt grades under the wrong rules and still returns a number."""
    with pytest.raises(NotImplementedError):
        get_anscheck_prompt("made-up-type", "q", "a", "r", abstention=False)


def test_judge_model_pin_is_enforced():
    """print_qa_metrics.py:20 asserts the judge model; results judged by
    anything else are not comparable with published numbers."""
    check_judge_model(JUDGE_MODEL_PIN)  # no raise
    with pytest.raises(ValueError):
        check_judge_model("gpt-4o-mini-2024-07-18")


def _stub_with_judge_model(model):
    stub = _StubLLM({"generate": "x", "judge": "yes"})
    stub.roles = {"judge": RoleConfig(endpoint="http://localhost:8001/v1", model=model)}
    return stub


def test_an_offpin_judge_fails_before_it_is_ever_called():
    """The pin is worth nothing as a constant nobody reads: a 500-question run
    judged by gpt-4o-mini produces a file the official aggregator asserts
    against. It has to fail on call zero, not after the spend."""
    mem = _mem()
    try:
        mem.llm = _stub_with_judge_model("gpt-4o-mini-2024-07-18")
        with pytest.raises(ValueError, match="gpt-4o-2024-08-06"):
            run_instance(mem, INSTANCE)
        assert [c["role"] for c in mem.llm.calls] == ["generate"]  # no judge call spent
    finally:
        mem.close()


def test_the_pinned_judge_runs_and_the_pin_can_be_waived():
    mem = _mem()
    try:
        mem.llm = _stub_with_judge_model(JUDGE_MODEL_PIN)
        assert run_instance(mem, INSTANCE)["label"] is True
        mem.llm = _stub_with_judge_model("gpt-4o-mini-2024-07-18")
        assert run_instance(mem, INSTANCE, enforce_pin=False)["label"] is True
    finally:
        mem.close()


def test_a_client_that_does_not_expose_its_model_is_not_treated_as_pinned():
    """`None` from `configured_judge_model` means "cannot be checked". Blocking
    on it would make every stub-driven and non-role-based client unjudgeable."""
    mem = _mem()
    try:
        mem.llm = _StubLLM({"generate": "x", "judge": "yes"})  # no `roles` attribute
        assert run_instance(mem, INSTANCE)["label"] is True
    finally:
        mem.close()


# ---------------- metrics ----------------


def test_task_averaged_and_overall_accuracy_differ_on_unequal_counts():
    """Both numbers are printed by print_qa_metrics.py and they are NOT the
    same statistic — one weights types equally, the other weights questions.
    A published figure that does not say which cannot be compared against."""
    records = [
        {"question_id": "a1", "question_type": "single-session-user", "label": True},
        {"question_id": "a2", "question_type": "single-session-user", "label": True},
        {"question_id": "a3", "question_type": "single-session-user", "label": True},
        {"question_id": "b1", "question_type": "multi-session", "label": False},
    ]
    out = aggregate(records)
    assert out["by_type"]["single-session-user"] == {"acc": 100.0, "n": 3}
    assert out["by_type"]["multi-session"] == {"acc": 0.0, "n": 1}
    assert out["task_averaged"] == 50.0  # mean of the two type means
    assert out["overall"] == 75.0  # mean over the four questions
    assert out["task_averaged"] != out["overall"]


def test_abstention_is_a_crosscut_not_a_seventh_type():
    """print_qa_metrics.py appends every row to its question_type bucket and
    ADDITIONALLY appends `_abs` rows to the abstention bucket."""
    records = [
        {"question_id": "q1", "question_type": "multi-session", "label": True},
        {"question_id": "q2_abs", "question_type": "multi-session", "label": False},
    ]
    out = aggregate(records)
    assert out["by_type"]["multi-session"]["n"] == 2  # the _abs row counted here too
    assert out["abstention"] == {"acc": 0.0, "n": 1}
    assert out["overall"] == 50.0  # and in the overall
    assert out["n"] == 2


def test_task_average_is_taken_over_unrounded_type_means():
    """Upstream averages the raw per-type means and rounds once (:31). Averaging
    the *reported* percentages instead drifts the headline by up to 0.01pp — in
    the one module whose premise is that a LongMemEval number is only readable
    if you know exactly which statistic it is."""
    counts = {  # (hits, n) chosen so per-type means do not land on 2 decimals
        "single-session-user": (30, 43),
        "single-session-preference": (20, 30),
        "single-session-assistant": (40, 56),
        "multi-session": (50, 77),
        "temporal-reasoning": (60, 88),
        "knowledge-update": (50, 78),
    }
    records = [
        {"question_id": f"q{qtype}{i}", "question_type": qtype, "label": i < hits}
        for qtype, (hits, n) in counts.items()
        for i in range(n)
    ]
    exact = sum(100 * hits / n for hits, n in counts.values()) / len(counts)
    assert aggregate(records)["task_averaged"] == round(exact, 2) == 67.51


def test_unjudged_rows_are_excluded_not_counted_wrong():
    out = aggregate(
        [
            {"question_id": "q1", "question_type": "multi-session", "label": True},
            {"question_id": "q2", "question_type": "multi-session", "label": None},
        ]
    )
    assert out["overall"] == 100.0 and out["n"] == 1


def test_type_order_follows_the_official_report_order():
    records = [
        {"question_id": "a", "question_type": "knowledge-update", "label": True},
        {"question_id": "b", "question_type": "single-session-user", "label": True},
    ]
    assert list(aggregate(records)["by_type"]) == ["single-session-user", "knowledge-update"]


# ---------------- generation ----------------


def test_con_reading_method_uses_the_official_prompt_and_token_budget():
    mem = _mem()
    try:
        mem.llm = _StubLLM()
        ingest(mem, INSTANCE)
        answer(mem, INSTANCE, reading_method="con")
        call = mem.llm.calls[-1]
        assert call["role"] == "generate"
        assert "Answer (step by step):" in call["prompt"]
        assert "Current Date: 2023/06/01 (Thu) 10:00" in call["prompt"]
        assert call["max_tokens"] == 800 and call["temperature"] == 0.0
    finally:
        mem.close()


def test_direct_reading_method_switches_prompt_and_budget():
    mem = _mem()
    try:
        mem.llm = _StubLLM()
        answer(mem, INSTANCE, reading_method="direct")
        call = mem.llm.calls[-1]
        assert call["prompt"].rstrip().endswith("Answer:")
        assert "step by step" not in call["prompt"]
        assert call["max_tokens"] == 500
    finally:
        mem.close()


def test_con_separate_is_rejected_rather_than_silently_downgraded():
    """Upstream's third reading method runs a per-session note-extraction LLM
    pass first; accepting the alias while ignoring that would report a `con`
    number under a `con-separate` label."""
    mem = _mem()
    try:
        mem.llm = _StubLLM()
        with pytest.raises(ValueError, match="con-separate"):
            answer(mem, INSTANCE, reading_method="con-separate")
    finally:
        mem.close()


def test_full_context_history_bypasses_retrieval():
    mem = _mem()
    try:
        mem.llm = _StubLLM()
        called = []
        mem.search = lambda *a, **k: called.append(1)  # must never fire
        answer(mem, INSTANCE, history=render_sessions(INSTANCE))
        assert called == []
        prompt = mem.llm.calls[-1]["prompt"]
        assert "### Session 1:" in prompt and "### Session 2:" in prompt
        assert "Session Date: 2023/05/01 (Mon) 09:00" in prompt
        assert '"role": "user"' in prompt  # history_format=json
    finally:
        mem.close()


def test_full_context_never_shows_the_model_which_turns_hold_the_answer():
    """Upstream pops `has_answer` off every turn before formatting
    (run_generation.py:177-191). Rendering the raw session instead marks the
    evidence turns in the prompt, which does not crash — it inflates the very
    baseline this path reproduces (GPT-4o 60.6% on _s)."""
    rendered = render_sessions(INSTANCE)
    assert "has_answer" not in rendered
    assert "I am going to Paris in May" in rendered  # the content itself stays


def test_stripping_the_evidence_label_does_not_destroy_the_recall_gold():
    """Upstream can pop in place because it never scores turn-level recall in
    the same process. We yield `has_answer` from `iter_turns` for exactly that,
    so the strip has to be on a copy."""
    render_sessions(INSTANCE)
    assert INSTANCE["haystack_sessions"][0][0]["has_answer"] is True
    assert [t[4] for t in iter_turns(INSTANCE)] == [True, False, False]


def test_history_can_be_capped_the_way_upstream_caps_it():
    """`budget_tokens` governs the retrieved bundle only, so an explicit
    `history` reaches the API unbounded — on _m that is ~1.5M tokens."""
    assert upstream_max_history_tokens(128000, "con") == 126200  # 128000 - 800 - 1000
    assert upstream_max_history_tokens(128000, "direct") == 126500
    mem = _mem()
    try:
        mem.llm = _StubLLM()
        long_history = "x" * 4000
        answer(mem, INSTANCE, history=long_history, max_history_tokens=100)
        sent = mem.llm.calls[-1]["prompt"]
        assert "x" * 400 in sent and "x" * 401 not in sent  # 100 tokens * 4 chars
    finally:
        mem.close()


def test_answer_prompt_template_is_verbatim_upstream():
    """Pinned literally: the template is transcribed from run_generation.py:55
    and a well-meaning rewording is exactly the kind of drift that moves scores
    by several points without any code looking wrong."""
    assert ANSWER_PROMPT_CON.startswith(
        "I will give you several history chats between you and a user. "
        "Please answer the question based on the relevant chat history. "
        "Answer the question step by step: first extract all the relevant information, "
        "and then reason over the information to get the answer."
    )
    assert ANSWER_PROMPT_CON.endswith(
        "\n\nHistory Chats:\n\n{history}\n\nCurrent Date: {question_date}\n"
        "Question: {question}\nAnswer (step by step):"
    )


def test_multiline_chain_of_note_answers_survive_intact():
    """LoCoMo's `answer` keeps only the first line (short-span answers). A
    chain-of-note reply is multi-line by construction and the judge reads all
    of it, so truncating here would throw away the answer itself."""
    mem = _mem()
    try:
        mem.llm = _StubLLM({"generate": "Step 1: user mentioned Paris.\nAnswer: Paris in May"})
        out = answer(mem, INSTANCE)
        assert out == "Step 1: user mentioned Paris.\nAnswer: Paris in May"
    finally:
        mem.close()


# ---------------- driver ----------------


def test_run_instance_produces_an_aggregatable_judged_record():
    mem = _mem()
    try:
        mem.llm = _StubLLM({"generate": "Paris in May", "judge": "yes"})
        row = run_instance(mem, INSTANCE)
        assert row["question_id"] == "q1" and row["turns"] == 3
        assert row["hypothesis"] == "Paris in May" and row["label"] is True
        assert aggregate([row])["overall"] == 100.0
        # the judge call is free-text with the official kwargs, not guided JSON
        judge_call = [c for c in mem.llm.calls if c["role"] == "judge"][0]
        assert judge_call["max_tokens"] == 10 and judge_call["temperature"] == 0.0
        assert "Answer yes or no only." in judge_call["prompt"]
    finally:
        mem.close()


def test_judge_verdict_is_upstreams_substring_test():
    """`'yes' in response.lower()` (evaluate_qa.py:113) — looser than equality,
    and kept so a judge-disagreement rate means what upstream's means."""
    mem = _mem()
    try:
        mem.llm = _StubLLM({"generate": "x", "judge": "No, the response is wrong"})
        assert run_instance(mem, INSTANCE)["label"] is False
        mem.llm = _StubLLM({"generate": "x", "judge": "Yes"})
        assert run_instance(mem, INSTANCE)["label"] is True
    finally:
        mem.close()


def test_judging_is_skipped_without_a_judge_role():
    mem = _mem()
    try:
        stub = _StubLLM({"generate": "x"})
        stub.has_role = lambda role: role == "generate"
        mem.llm = stub
        assert run_instance(mem, INSTANCE)["label"] is None
    finally:
        mem.close()
