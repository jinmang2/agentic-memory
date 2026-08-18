"""LongMemEval pipeline: the parts where getting it subtly wrong is silent.

Every assertion here is anchored to the official repo (xiaowu0162/LongMemEval)
rather than to our own convenience — see src/agmem/bench/longmemeval.py for the
file:line citations.
"""

import json

import pytest

from agmem import AgenticMemory
from agmem.bench.longmemeval import (
    ANSWER_PROMPT_CON,
    CHARS_PER_TOKEN,
    JUDGE_MODEL_PIN,
    aggregate,
    answer,
    check_judge_model,
    get_anscheck_prompt,
    ingest,
    is_abstention,
    iter_longmemeval,
    iter_turns,
    load_longmemeval,
    render_sessions,
    run_instance,
    sort_haystack_by_date,
    truncate_history,
    upstream_max_history_tokens,
)
from agmem.core.types import MemoryBundle
from agmem.embed.fake import FakeEmbedder
from agmem.llm.client import RoleConfig
from agmem.organizers.base import Organizer

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


UNSORTED = {
    **INSTANCE,
    "haystack_session_ids": ["late", "early", "mid"],
    "haystack_dates": [
        "2023/05/20 (Sat) 09:00",
        "2023/05/01 (Mon) 09:00",
        "2023/05/10 (Wed) 09:00",
    ],
    "haystack_sessions": [
        [{"role": "user", "content": "third thing"}],
        [{"role": "user", "content": "first thing"}],
        [{"role": "user", "content": "second thing"}],
    ],
}


def test_oracle_haystack_is_sorted_as_pairs_not_as_arrays():
    """`longmemeval_oracle.json` ships out of date order and upstream sorts right
    before formatting (run_generation.py:225). The three haystack arrays are
    positional, so sorting dates alone would re-label every session and mis-key
    the retrieval gold — the failure this permutes-together shape prevents."""
    out = sort_haystack_by_date(UNSORTED)
    assert out["haystack_dates"] == [
        "2023/05/01 (Mon) 09:00",
        "2023/05/10 (Wed) 09:00",
        "2023/05/20 (Sat) 09:00",
    ]
    assert out["haystack_session_ids"] == ["early", "mid", "late"]
    assert [s[0]["content"] for s in out["haystack_sessions"]] == [
        "first thing",
        "second thing",
        "third thing",
    ]
    assert [t[0] for t in iter_turns(out)] == ["early", "mid", "late"]


def test_sorting_leaves_the_loaded_instance_untouched():
    """A driver sorts 500 instances; sorting in place would mutate the dataset
    under every later reader of it, including the scorer."""
    before = list(UNSORTED["haystack_dates"])
    sort_haystack_by_date(UNSORTED)
    assert UNSORTED["haystack_dates"] == before


def test_sorting_is_stable_so_equal_dates_keep_haystack_order():
    tied = {
        **INSTANCE,
        "haystack_session_ids": ["a", "b"],
        "haystack_dates": ["2023/05/01 (Mon) 09:00", "2023/05/01 (Mon) 09:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ],
    }
    assert sort_haystack_by_date(tied)["haystack_session_ids"] == ["a", "b"]


def test_sorting_is_what_renumbers_the_rendered_sessions():
    """Session numbering is applied AFTER the permutation, as upstream's
    enumerate over sorted chunks is — so the guard changes the prompt bytes, not
    just an ordering nobody sees."""
    unsorted_prompt = render_sessions(UNSORTED)
    sorted_prompt = render_sessions(sort_haystack_by_date(UNSORTED))
    assert unsorted_prompt != sorted_prompt
    assert sorted_prompt.index("first thing") < sorted_prompt.index("third thing")
    assert "### Session 1:\nSession Date: 2023/05/01 (Mon) 09:00" in sorted_prompt


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


def test_sampling_params_are_spoken_in_the_roles_own_dialect():
    """Upstream pins `temperature=0` and `max_tokens=800`, and a literal override
    of both is a hard 400 on the newer Chat Completions models: gpt-5.6-luna
    requires `max_completion_tokens` and refuses any non-default temperature
    (both recorded on its ModelSpec). The arm definition must not have to change
    with the reader, so the key comes off the RoleConfig and the temperature is
    dropped when the role was built without one."""
    mem = _mem()
    try:
        stub = _StubLLM()
        stub.roles = {
            "generate": RoleConfig(
                endpoint="x",
                model="gpt-5.6-luna",
                temperature=None,
                max_tokens_key="max_completion_tokens",
            )
        }
        mem.llm = stub
        answer(mem, INSTANCE, reading_method="con", history="h")
        call = stub.calls[-1]
        assert call["max_completion_tokens"] == 800
        assert "max_tokens" not in call and "temperature" not in call
    finally:
        mem.close()


def test_a_conventional_role_still_gets_upstreams_literal_kwargs():
    mem = _mem()
    try:
        stub = _StubLLM()
        stub.roles = {"generate": RoleConfig(endpoint="x", model="gpt-4o-mini", temperature=0.0)}
        mem.llm = stub
        answer(mem, INSTANCE, reading_method="direct", history="h")
        call = stub.calls[-1]
        assert call["max_tokens"] == 500 and call["temperature"] == 0.0
    finally:
        mem.close()


def test_budget_key_labels_the_row_so_concurrent_arms_can_be_priced():
    """500 questions answered by a thread pool over ONE client: a per-row cost
    cannot be recovered by diffing a shared budget, because the diff belongs to
    whichever rows finished in between."""
    mem = _mem()
    try:
        mem.llm = _StubLLM({"judge": "yes"})
        mem.llm.roles = {"judge": RoleConfig(endpoint="x", model=JUDGE_MODEL_PIN)}
        run_instance(mem, INSTANCE, full_context=True, enforce_pin=False, budget_key="q1")
        keys = [c.get("budget_key") for c in mem.llm.calls]
        assert keys == ["generate|q1", "judge|q1"]
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


def test_the_verbatim_recency_window_reaches_this_benchmark_too():
    """`Organizer.recent_context()` is a channel the methodology injects OUTSIDE
    retrieval on every question (MemoryOS's resident STM). LoCoMo's `answer` has
    always honoured it; this path did not, so measuring such an organizer here
    dropped the channel with no degradation note. Not on the full-context
    baseline, which reads no memory at all."""

    class _Recent(Organizer):
        name = "recent"

        def recent_context(self):
            return "A: the keycode is 4417"

    mem = _mem(organizers=[_Recent()])
    try:
        mem.llm = _StubLLM()
        answer(mem, INSTANCE)
        prompt = mem.llm.calls[-1]["prompt"]
        assert "Recent conversation:" in prompt and "the keycode is 4417" in prompt

        answer(mem, INSTANCE, history=render_sessions(INSTANCE))
        assert "the keycode is 4417" not in mem.llm.calls[-1]["prompt"]
    finally:
        mem.close()


def test_nl_history_format_is_upstreams_other_branch():
    """`history_format` is upstream's flag (run_generation.py:234-247) and §5.5
    reports it interacts with the reading method by up to 10pp — JSON does not
    consistently beat NL without chain-of-note and always beats it with. Both
    branches are transcribed, including the asymmetry that `nl` strips each
    turn's content and `json` does not."""
    nl = render_sessions(INSTANCE, history_format="nl")
    assert "\n\nuser: I am going to Paris in May" in nl
    assert "\n\nassistant: Sounds great" in nl
    assert '{"role"' not in nl  # not the json branch
    assert "### Session 1:" in nl and "Session Date: 2023/05/01 (Mon) 09:00" in nl
    assert "has_answer" not in nl  # the evidence label is stripped in BOTH formats


def test_an_unknown_history_format_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="history_format"):
        render_sessions(INSTANCE, history_format="markdown")


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
    `history` reaches the API unbounded — on _m that is ~1.1M tokens.

    This assertion used to read `100 tokens * 4 chars`, which pinned D3 in place:
    4 is the framework's generic chars-per-token, and this corpus measures 4.610,
    so the cap fired 13% early. The old number was harmless while nothing passed
    a cap, and wrong the moment `_m` needed one."""
    assert upstream_max_history_tokens(128000, "con") == 126200  # 128000 - 800 - 1000
    assert upstream_max_history_tokens(128000, "direct") == 126500
    mem = _mem()
    try:
        mem.llm = _StubLLM()
        long_history = "x" * 4000
        answer(mem, INSTANCE, history=long_history, max_history_tokens=100)
        sent = mem.llm.calls[-1]["prompt"]
        assert "x" * 461 in sent and "x" * 462 not in sent  # 100 tokens * 4.610 chars
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


# ---------------- streaming loader (`_m` does not fit any other way) ----------------


def _write_array(tmp_path, objs, sep=", ", pretty=False):
    text = (
        "["
        + sep.join(json.dumps(o, ensure_ascii=False, indent=2 if pretty else None) for o in objs)
        + "]"
    )
    p = tmp_path / "arr.json"
    p.write_text(text, encoding="utf-8")
    return p


# Content chosen to break a naive brace counter: unbalanced braces inside strings,
# an escaped quote, a trailing backslash, a non-BMP character (which is what makes
# the decoded `_s` string 4 bytes/char and its full load 2.42 GB), and a nested
# object so depth has to be counted rather than assumed to be 1.
NASTY = [
    {"question_id": "a", "t": "brace } alone and { alone", "nested": {"deep": {"deeper": 1}}},
    {"question_id": "b", "t": 'escaped quote \\" then }', "u": "café \U0001f600"},
    {"question_id": "c", "t": "trailing backslash \\\\", "haystack_sessions": [[{"role": "user"}]]},
]


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 64, 1 << 22])
def test_streaming_loader_matches_json_loads_at_every_chunk_size(tmp_path, chunk):
    """The scan position has to persist across reads. Restarting it at 0 after
    appending a chunk replays the braces already counted, and the corruption is
    silent — it read 175 of oracle's 500 instances, not an error. Tiny chunk
    sizes put a boundary inside strings, escape pairs and objects on purpose."""
    p = _write_array(tmp_path, NASTY)
    assert list(iter_longmemeval(p, chunk_bytes=chunk)) == json.loads(p.read_text())


def test_streaming_loader_survives_whitespace_and_indentation(tmp_path):
    p = _write_array(tmp_path, NASTY, sep=",\n\n  ", pretty=True)
    assert [i["question_id"] for i in iter_longmemeval(p, chunk_bytes=5)] == ["a", "b", "c"]


def test_an_empty_release_yields_nothing_rather_than_hanging(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("[]", encoding="utf-8")
    assert list(iter_longmemeval(p)) == []


def test_a_truncated_file_raises_rather_than_returning_what_it_got(tmp_path):
    """A 2.7 GB download that stopped early must not read as a short dataset."""
    p = tmp_path / "cut.json"
    p.write_text('[{"question_id": "a"}, {"question_id": "b", "hays', encoding="utf-8")
    with pytest.raises(ValueError, match="unterminated"):
        list(iter_longmemeval(p))


def test_load_longmemeval_still_returns_the_whole_list(tmp_path):
    p = _write_array(tmp_path, NASTY)
    assert load_longmemeval(p) == json.loads(p.read_text())


# ---------------- D3: the chars-per-token estimate was 13% short ----------------


class _Encoder:
    """tiktoken's surface, minus tiktoken: one token per character."""

    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


def test_the_cap_uses_the_corpus_measured_ratio_not_the_generic_four():
    """D3. `MemoryBundle.CHARS_PER_TOKEN` is 4; o200k_base over `_s` measures
    4.610 (docs/research/longmemeval.md §3.1). Converting with 4 cut 13% early.
    Latent on `_s`/oracle, where fidelity is passing no cap at all — `_m` is the
    one variant where the cap binds."""
    assert CHARS_PER_TOKEN == 4.610
    history = "x" * 10_000
    assert len(truncate_history(history, 1000)) == 4610
    assert len(truncate_history(history, 1000)) > 1000 * MemoryBundle.CHARS_PER_TOKEN


def test_a_supplied_encoder_makes_the_cap_exact_instead_of_estimated():
    """Upstream truncates with the reader's own tokenizer (run_generation.py:266-279).
    With one to hand we do the same thing rather than estimating."""
    assert truncate_history("abcdefghij", 4, encoder=_Encoder()) == "abcd"


def test_a_history_under_the_cap_is_returned_untouched():
    for encoder in (None, _Encoder()):
        assert truncate_history("short", 1000, encoder=encoder) == "short"


def test_the_encoder_reaches_the_cap_through_answer():
    """The full-context path is the only one the cap can bind on, so the encoder
    has to survive the trip from the driver through `answer`."""
    mem = _mem()
    try:
        mem.llm = _StubLLM({"generate": "ok"})
        answer(
            mem, INSTANCE, history="abcdefghij", max_history_tokens=4, history_encoder=_Encoder()
        )
        prompt = mem.llm.calls[0]["prompt"]
        assert "abcd" in prompt and "abcde" not in prompt
    finally:
        mem.close()
