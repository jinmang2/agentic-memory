"""FiNER pipeline: the places where matching upstream and being right diverge.

Every assertion is anchored to the official ACE clone rather than to our
convenience — see src/agmem/bench/finer.py for the file:line citations. The
recurring shape is that upstream's harness reports one number while computing
several, so most of these tests are about what a caller can and cannot conclude
from a FiNER accuracy.
"""

import json

import pytest
from helpers import StubLLM, make_mem_multi

from agmem.bench.finer import (
    NO_ANSWER,
    SPLITS,
    TAGS_PER_SAMPLE,
    adapt,
    aggregate,
    answer,
    answer_is_correct,
    extract_answer,
    load_finer,
    parse_instruction_and_input,
    process_task_data,
    score_sample,
    split_tags,
    tag_counts,
    windows,
)
from agmem.organizers.ace import ACEOrganizer


def _row(pred, target, failed=False):
    return score_sample({"question": "q", "context": "c", "target": target}, pred, failed=failed)


# ---------------- the two accuracies ----------------


def test_the_two_accuracies_are_different_numbers_and_both_are_reported():
    """Upstream's evaluate_test_set prints the TAG-level rate next to the
    SAMPLE-level fraction as one fact (utils.py:286), and the online result
    dict pairs them the same way (ace.py:1097-1098). They do not agree, so a
    reader who recomputes correct/total from the artifact gets a different
    number than the artifact's headline. `aggregate` returns both, named."""
    rows = [
        _row("a,b,c,d", "a,b,c,d"),  # 4/4 tags, sample correct
        _row("a,b,c,x", "a,b,c,d"),  # 3/4 tags, sample WRONG
        _row("a,x,y,z", "a,b,c,d"),  # 1/4 tags, sample wrong
    ]
    agg = aggregate(rows)
    assert agg["tag_accuracy"] == pytest.approx(100 * 8 / 12, abs=0.01)
    assert agg["sample_accuracy"] == pytest.approx(100 * 1 / 3, abs=0.01)
    # The whole point: the two must not be confusable for one another.
    assert agg["tag_accuracy"] != agg["sample_accuracy"]
    assert agg["n"] == 3 and agg["n_tags"] == 12


def test_the_training_signal_is_the_stricter_of_the_two():
    """The reflection loop branches on all-or-nothing correctness (ace.py:477)
    while the headline is tag-level, so a 3-of-4 sample is simultaneously a
    failure to learn from and a 0.75 to report (trap 5)."""
    pred, gold = "a,b,c,x", "a,b,c,d"
    correct, total, _ = tag_counts(pred, gold)
    assert (correct, total) == (3, 4)
    assert answer_is_correct(pred, gold) is False


# ---------------- the denominator ----------------


def test_a_failed_sample_costs_accuracy_instead_of_leaving_the_split():
    """Upstream drops a raising sample before it reaches total/answers/targets
    (utils.py:198-199, 248-250), so failures shrink the population rather than
    scoring zero. Ours counts it wrong and says how many there were."""
    rows = [_row("a,b,c,d", "a,b,c,d"), _row(NO_ANSWER, "a,b,c,d", failed=True)]
    agg = aggregate(rows)
    assert agg["n"] == 2, "the failed row must stay in the denominator"
    assert agg["sample_accuracy"] == pytest.approx(50.0)
    assert agg["tag_accuracy"] == pytest.approx(50.0)
    assert agg["n_failed"] == 1


def test_a_no_answer_reply_is_wrong_and_counted():
    rows = [_row(NO_ANSWER, "a,b,c,d")]
    agg = aggregate(rows)
    assert agg["sample_accuracy"] == 0.0
    assert agg["n_no_answer"] == 1


# ---------------- the metric's own asymmetry ----------------


def test_over_prediction_is_free_in_upstreams_metric_and_we_surface_it():
    """A longer prediction is truncated to the gold's length and the extras are
    never scored (data_processor.py:134-139); a shorter one is padded with ""
    and each pad counts wrong. We keep the arithmetic — it is the metric ACE's
    number is stated in — and report the discarded count so the leniency is
    visible rather than silent (trap 4)."""
    over = tag_counts("a,b,c,d,zzz,yyy", "a,b,c,d")
    assert over == (4, 4, 2), "the two extra tags cost nothing"
    assert answer_is_correct("a,b,c,d,zzz,yyy", "a,b,c,d") is True

    under = tag_counts("a,b", "a,b,c,d")
    assert under == (2, 4, 0), "missing tags are padded and count wrong"
    assert aggregate([_row("a,b,c,d,zzz", "a,b,c,d")])["n_over_predicted"] == 1


def test_numeric_equivalence_survives_without_running_eval_on_model_output():
    """Upstream reaches numeric equality by calling eval() on the model's own
    string (data_processor.py:142-146) inside a bare except. We parse a literal
    instead: the equalities it bought still hold, and a tag name is compared as
    a string rather than executed (trap 3)."""
    assert tag_counts("5.0", "5")[0] == 1
    assert tag_counts("$1200.00", "1200")[0] == 1
    # A GAAP tag is an identifier: upstream's eval raises NameError here and the
    # bare except restores the string compare. Ours never evaluates it at all.
    assert tag_counts("Revenues", "Revenues")[0] == 1
    assert tag_counts("Revenues", "Goodwill")[0] == 0


def test_a_comma_formatted_number_is_torn_apart_before_any_coercion_can_help():
    """The comma strip inside upstream's eval (`prediction.replace(",", "")`,
    data_processor.py:144) can never fire: the string was split on commas one
    line earlier (:128), so "1,000" has already become two tags. Ours carries
    the same dead strip for the same reason — reproduce a dead branch as dead —
    and this test pins that it stays unreachable rather than being "fixed" into
    a divergence from the metric ACE publishes."""
    correct, total, n_over = tag_counts("1,000", "1000")
    assert (correct, total) == (0, 1), "'1,000' scores as the tag '1', not as 1000"
    assert n_over == 1


def test_tags_are_compared_case_and_whitespace_insensitively():
    """data_processor.py:128-131 lowercases and strips both sides."""
    assert split_tags(" A , b ") == ["a", "b"]
    assert answer_is_correct("Revenues, Goodwill", "revenues,goodwill") is True


# ---------------- answer extraction ----------------


def test_extract_answer_walks_upstreams_ladder_in_order():
    """utils.py:100-130: JSON, then Finish[...], then the double- and
    single-quoted regexes, then a sentinel — not an exception."""
    assert extract_answer(json.dumps({"final_answer": "a,b"})) == "a,b"
    assert extract_answer("blah Finish[x,y] blah") == "x,y"
    assert extract_answer('... "final_answer": "p,q" ...') == "p,q"
    assert extract_answer("... 'final_answer': 'm,n' ...") == "m,n"
    assert extract_answer("I think the tags are Revenues and Goodwill.") == NO_ANSWER


def test_the_last_match_wins_not_the_first():
    """findall()[-1] in every rung — a response that restates the format
    example before answering must score on its answer."""
    assert extract_answer("Finish[wrong] then Finish[right]") == "right"


# ---------------- loading and windowing ----------------


def _write_split(tmp_path, name, rows):
    (tmp_path / name).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_load_finer_refuses_a_file_that_is_not_the_split_it_claims(tmp_path):
    """The row count and the tags-per-sample are both load-bearing for the
    metric, and a quietly different file would move a published-looking number
    without moving anything a stamp records."""
    name, expected = SPLITS["test"]
    _write_split(tmp_path, name, [{"context": "c", "target": "a,b,c,d"}] * (expected - 1))
    with pytest.raises(ValueError, match="expected 441 rows"):
        load_finer(tmp_path, "test")


def test_load_finer_refuses_rows_that_break_the_online_loops_arithmetic(tmp_path):
    """ace.py:971 multiplies a TAG-level rate by a SAMPLE count. That is only
    defensible while every sample carries the same number of tags, so the
    loader checks it rather than trusting it."""
    name, expected = SPLITS["test"]
    rows = [{"context": "c", "target": ",".join("abcd")}] * (expected - 1)
    rows.append({"context": "c", "target": "a,b"})
    _write_split(tmp_path, name, rows)
    with pytest.raises(ValueError, match=f"do not carry {TAGS_PER_SAMPLE} tags"):
        load_finer(tmp_path, "test")


def test_windows_reproduce_the_online_loops_slicing():
    """ace.py:934-942: consecutive slices, last one short, and each window is
    tested BEFORE it is trained on."""
    got = list(windows(list(range(7)), 3))
    assert [start for start, _ in got] == [0, 3, 6]
    assert [chunk for _, chunk in got] == [[0, 1, 2], [3, 4, 5], [6]]


def test_a_context_without_upstreams_markers_falls_through_to_the_question():
    """parse_instruction_and_input returns ("", all_context) when either marker
    is missing (data_processor.py:46), which reshapes the prompt instead of
    raising. Reproduced, and counted, so a run can tell it happened."""
    assert parse_instruction_and_input("no markers here") == ("", "no markers here")
    samples, fell_through = process_task_data(
        [
            {"context": "Instruction: do it.\nInput: the text\nAnswer: ", "target": "a"},
            {"context": "no markers here", "target": "b"},
        ]
    )
    assert samples[0] == {"context": "the text", "question": "do it.", "target": "a"}
    assert samples[1]["context"] == "" and samples[1]["question"] == "no markers here"
    assert fell_through == 1


# ---------------- the read contract ----------------


def test_answer_injects_the_whole_playbook_and_never_a_top_k_slice():
    """ACE's read contract is full-playbook injection (organizers/ace, round-5
    §2). The prompt the generator sees must therefore contain every bullet in
    the store, not a retrieved subset — swapping this for top-k would be a
    different methodology wearing ACE's name."""
    reflection = {
        "reasoning": "r",
        "error_identification": "e",
        "root_cause_analysis": "c",
        "correct_approach": "a",
        "key_insight": "k",
        "bullet_tags": [],
    }
    n_bullets = 5
    llm = StubLLM(
        {
            # Two distill calls per task (reflect, then curate) -- the organizer
            # rides one role for both (round-12 #7).
            "distill": [
                item
                for i in range(n_bullets)
                for item in (
                    reflection,
                    {
                        "operations": [
                            {"type": "ADD", "section": "gaap", "content": f"bullet number {i}"}
                        ]
                    },
                )
            ],
            "generate": [{"final_answer": "a,b,c,d", "bullet_ids": ["fin-00001"]}],
        }
    )
    mem = make_mem_multi([ACEOrganizer()], llm)
    try:
        for i in range(n_bullets):
            mem.add_task_result([{"step": i}], outcome="failure", task=f"tag sample {i}")
        assert mem.get_playbook().count("bullet number") == n_bullets
        pred, bullet_ids = answer(mem, {"question": "q", "context": "ctx", "target": "a,b,c,d"})
        assert pred == "a,b,c,d"
        assert bullet_ids == ["fin-00001"]
        prompt = next(p for role, p in llm.calls if role == "generate")
        for i in range(n_bullets):
            assert f"bullet number {i}" in prompt, "the full playbook must reach the generator"
        assert "ctx" in prompt and "q" in prompt
    finally:
        mem.close()


def test_answer_refuses_to_score_a_run_that_never_called_a_model():
    mem = make_mem_multi([ACEOrganizer()], None)
    try:
        mem.structured = None
        with pytest.raises(RuntimeError, match="structured LLM client"):
            answer(mem, {"question": "q", "context": "c", "target": "a"})
    finally:
        mem.close()


def test_the_generators_reasoning_reaches_the_reflector():
    """Upstream reflects on `reasoning_trace=gen_response` (ace.py:509-517).
    A first pass here passed only the prediction and the gold, and the curated
    bullets came back as process-improvement filler — with no reasoning to
    fault, a reflector has nothing to be specific about. Pinned so the trace
    cannot quietly fall out of the trajectory again."""
    captured = {}

    class _Mem:
        def add_task_result(self, trajectory, outcome, task):
            captured["trajectory"] = trajectory
            captured["outcome"] = outcome

    sample = {"question": "q", "context": "c", "target": "a,b,c,d"}
    row = score_sample(sample, "a,b,c,x")
    row["reasoning"] = "I matched 231,312 to CommonStockSharesAuthorized because..."
    row["bullet_ids"] = ["gaap-00007"]
    adapt(_Mem(), sample, row)

    step = captured["trajectory"][0]
    assert step["reasoning"].startswith("I matched 231,312")
    assert step["bullets_cited"] == ["gaap-00007"]
    assert captured["outcome"] == "failure", "3-of-4 is a failure to learn from (trap 5)"
