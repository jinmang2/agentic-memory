from agmem.bench.locomo import (bleu1, evidence_sessions, iter_turns,
                                select_questions, token_f1)


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
    assert "q5" in ids                       # no evidence -> always eligible


def test_evidence_sessions_parsing():
    assert evidence_sessions({"evidence": ["D3:12", "D10:4"]}) == {3, 10}
