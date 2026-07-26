"""LongMemEval benchmark pipeline (xiaowu0162/LongMemEval, ICLR'25, MIT).

Unlike LoCoMo there is no string metric here: the benchmark is scored by an LLM
judge with a per-question-type prompt, and the official code pins the judge model
by assertion. Everything in the JUDGE and METRICS sections below is transcribed
from the official repo (cloned and read for this port); every file:line citation
refers to that clone:

    src/evaluation/evaluate_qa.py     -- get_anscheck_prompt, judge kwargs
    src/evaluation/print_qa_metrics.py -- the two accuracy definitions
    src/generation/run_generation.py  -- answer prompt templates, history format
    src/generation/run_generation.sh  -- reading_method / history_format aliases

**Structural difference from LoCoMo, which drives the whole API shape**: a LoCoMo
sample is one conversation with many questions attached, so one memory serves all
of them. A LongMemEval instance is ONE question plus its own haystack of ~40
sessions, so a faithful run builds one memory per question (500 of them for the
_s variant). ``ingest`` therefore takes an instance, not a corpus, and the caller
owns the per-instance memory lifecycle -- see ``run_instance``.

Three traps this module exists to keep us out of, all found by reading the
official code rather than the paper:

1. **There are two "accuracy" numbers and they differ.** print_qa_metrics.py
   prints ``Task-averaged Accuracy`` (mean of the six per-type means) *and*
   ``Overall Accuracy`` (mean over all questions). The question types have
   unequal counts, so these are not the same number, and a published figure that
   does not say which one it is cannot be compared against. ``aggregate``
   returns both, always.
2. **Abstention questions are double-counted.** print_qa_metrics.py appends every
   entry to its ``question_type`` bucket and *additionally* appends ``_abs`` ones
   to the abstention bucket, so abstention accuracy is a cross-cut, not a
   seventh type, and both accuracies above include abstention questions.
3. **The judge model is pinned by an assert**, not by convention:
   ``assert entry['autoeval_label']['model'] == 'gpt-4o-2024-08-06'``. Judging
   with anything else produces numbers the official aggregator refuses to read.
   ``JUDGE_MODEL_PIN`` and ``check_judge_model`` carry that forward instead of
   letting it be a comment.

Deviations from upstream, deliberate and listed so results can be caveated:

- **D1 (context ordering)**: upstream sorts retrieved chunks chronologically
  before formatting (``retrieved_chunks.sort(key=lambda x: x[0])``,
  run_generation.py:225). Our context comes from ``MemoryBundle.render``, which
  emits in fused-score order and owns the token budget. Organizer-produced
  memories are not sessions and often carry no comparable date, so there is no
  faithful chronological key for them; the full-context baseline
  (``render_sessions``) IS chronological, matching upstream exactly.
- **D2 (ingest unit)**: upstream never ingests -- it retrieves over the haystack.
  We feed one ``add_message`` per turn, formatted as LoCoMo's ingest is
  (``"(date) role: content"``), so the two benchmarks exercise the same write
  path. ``history_format=json`` is upstream's *reading* format for its retrieval
  baselines, not an ingest format, and is reproduced in ``render_sessions``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from agmem.memory import AgenticMemory

# print_qa_metrics.py:15 initialises type2acc with exactly these six, in this
# order -- it is also the report order, and a type absent from the data would
# make the official script emit nan rather than skip it.
QUESTION_TYPES = (
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)

# evaluate_qa.py:14 model_zoo['gpt-4o'], and asserted in print_qa_metrics.py:20.
JUDGE_MODEL_PIN = "gpt-4o-2024-08-06"

# evaluate_qa.py:108-109 / run_generation.py:363-365.
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 10
# run_generation.py:341 -- 800 with chain-of-note reading, 500 without.
GEN_MAX_TOKENS_CON = 800
GEN_MAX_TOKENS_DIRECT = 500
GEN_TEMPERATURE = 0.0


def is_abstention(question_id: str) -> bool:
    """Upstream test is ``'_abs' in entry['question_id']`` (evaluate_qa.py:101,
    print_qa_metrics.py:22) -- a SUBSTRING test, not ``endswith``.

    Kept as-is rather than "corrected" to ``endswith``: the abstention branch of
    the judge prompt is selected by this exact test upstream, so tightening it
    here would silently regrade any id with ``_abs`` in the middle."""
    return "_abs" in question_id


# ---------------- data loading ----------------


def load_longmemeval(path: str | Path) -> list[dict[str, Any]]:
    """Raw instances from ``longmemeval_{s,m,oracle}.json``; raises on a missing
    file or invalid JSON -- no fallback to an empty list.

    Which file, and whether it is the 2025/09 ``longmemeval-cleaned`` release,
    must be recorded with any result: the cleaned version removed answer-
    interfering sessions, so the two score differently."""
    return json.loads(Path(path).read_text())


def iter_turns(
    instance: dict[str, Any], max_sessions: int | None = None
) -> Iterator[tuple[str, str, str, str, bool]]:
    """Yield ``(session_id, date, role, content, has_answer)`` in haystack order.

    ``haystack_sessions``/``_dates``/``_session_ids`` are parallel arrays
    (run_generation.py:75-81). ``has_answer`` is the turn-level evidence label
    the benchmark ships for retrieval recall; it is metadata about the gold, so
    it must never reach the memory -- it is yielded for recall scoring only."""
    sessions = instance.get("haystack_sessions", [])
    dates = instance.get("haystack_dates", [])
    ids = instance.get("haystack_session_ids", [])
    for idx, session in enumerate(sessions[: max_sessions or len(sessions)]):
        date = dates[idx] if idx < len(dates) else ""
        session_id = ids[idx] if idx < len(ids) else f"session_{idx}"
        for turn in session:
            yield (
                session_id,
                str(date),
                str(turn.get("role", "user")),
                str(turn.get("content", "")),
                bool(turn.get("has_answer", False)),
            )


def evidence_session_ids(instance: dict[str, Any]) -> set[str]:
    """``answer_session_ids`` -- the session-level recall gold. Empty for
    abstention questions, which have no answer location by construction (the
    official retrieval evaluation drops them for exactly that reason)."""
    return {str(s) for s in instance.get("answer_session_ids", [])}


# ---------------- ingest (D2) ----------------


def ingest(mem: AgenticMemory, instance: dict[str, Any], max_sessions: int | None = None) -> int:
    """Feed every haystack turn into ``mem.add_message`` in order, then flush;
    returns the turn count. Mutates ``mem`` -- not idempotent across calls.

    One instance is one haystack, so ``mem`` must be scoped to this question (a
    fresh namespace or a fresh memory). Reusing one memory across instances
    silently merges unrelated haystacks and inflates every score."""
    n = 0
    for session_id, date, role, content, _has_answer in iter_turns(instance, max_sessions):
        mem.add_message(
            f"({date}) {role}: {content}",
            role=role,
            meta={"session_id": session_id, "date": date},
        )
        n += 1
    mem.flush()
    return n


# ---------------- generation (run_generation.py) ----------------

# run_generation.py:55 (retrieval + cot) -- the paper's recommended reading
# method, `con` in run_generation.sh's aliases, which maps to `--cot true`.
ANSWER_PROMPT_CON = """I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{history}\n\nCurrent Date: {question_date}\nQuestion: {question}\nAnswer (step by step):"""

# run_generation.py:57 (retrieval, no cot) -- `direct`.
ANSWER_PROMPT_DIRECT = """I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.\n\n\nHistory Chats:\n\n{history}\n\nCurrent Date: {question_date}\nQuestion: {question}\nAnswer:"""


def render_sessions(instance: dict[str, Any], max_sessions: int | None = None) -> str:
    """Full-history context in upstream's ``history_format=json`` shape.

    Block layout is run_generation.py:252 and the per-session value is
    ``'\\n' + json.dumps(chunk_entry)`` (:238) where ``chunk_entry`` is the raw
    session (a list of ``{role, content}``). Sessions are emitted in haystack
    order, which is already chronological for the ``_s``/``_m`` variants --
    ``longmemeval_oracle.json`` is NOT sorted, a documented upstream quirk, so
    an oracle run must sort before calling this.

    This reproduces the paper's full-context baseline (GPT-4o 60.6% on _s), the
    one configuration that needs no memory system at all."""
    sessions = instance.get("haystack_sessions", [])
    dates = instance.get("haystack_dates", [])
    out = []
    for i, session in enumerate(sessions[: max_sessions or len(sessions)]):
        date = dates[i] if i < len(dates) else ""
        out.append(
            "\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n".format(
                i + 1, date, "\n" + json.dumps(session)
            )
        )
    return "".join(out)


def answer(
    mem: AgenticMemory,
    instance: dict[str, Any],
    k: int | dict = 10,
    memory_types: tuple[str, ...] | None = None,
    budget_tokens: int = 6000,
    reading_method: str = "con",
    history: str | None = None,
    capture: dict[str, Any] | None = None,
) -> str:
    """One QA turn: retrieve, then generate with the official answer prompt.

    ``reading_method`` is upstream's alias set minus ``con-separate``:
    ``"con"`` (chain-of-note, the paper's recommendation, max_tokens 800) or
    ``"direct"`` (max_tokens 500). ``con-separate`` runs a per-session note-
    extraction LLM pass *before* answering (run_generation.py:195-207); it is a
    read-path method of the benchmark harness rather than of a memory system,
    and is deliberately not ported -- passing it raises rather than silently
    degrading to ``con``.

    ``history`` overrides the retrieved context, which is how the full-context
    baseline runs (``history=render_sessions(instance)``, no memory read at all).
    Otherwise the context is ``mem.search(...).render(budget_tokens)`` -- see D1
    for the ordering deviation.

    ``memory_types=None`` defers to the memory's ``default_memory_types``, so a
    caller does not have to know what the configured organizers produce.

    Raises ``RuntimeError`` if no ``generate`` LLM is configured. Returns the
    reply stripped; unlike LoCoMo's ``answer`` it is NOT truncated to the first
    line, because chain-of-note answers are multi-line by construction and the
    judge reads the whole response."""
    if reading_method not in ("con", "direct"):
        raise ValueError(
            f"unsupported reading_method {reading_method!r} "
            "(upstream aliases: con | direct | con-separate; con-separate is not ported)"
        )
    question = str(instance["question"])
    if history is None:
        bundle = mem.search(question, memory_types=memory_types, k=k)
        history = bundle.render(budget_tokens=budget_tokens) or "(no memories found)"
        if capture is not None:
            capture["query"] = question
            capture["k"] = k
            capture["memory_types"] = list(memory_types) if memory_types else None
            capture["retrieved"] = [
                {
                    "id": getattr(s.item, "id", None)
                    or (s.item.data.get("id") if hasattr(s.item, "data") else None),
                    "memory_type": s.memory_type,
                    "score": s.score,
                    "text": (
                        s.item.render()
                        if hasattr(s.item, "render")
                        else getattr(s.item, "content", str(s.item))
                    ),
                }
                for s in bundle.items
            ]
    if capture is not None:
        capture["history"] = history
    if mem.llm is None:
        raise RuntimeError("generate role LLM required for LongMemEval QA")
    template = ANSWER_PROMPT_CON if reading_method == "con" else ANSWER_PROMPT_DIRECT
    prompt = template.format(
        history=history,
        question_date=str(instance.get("question_date", "")),
        question=question,
    )
    reply = mem.llm.chat(
        "generate",
        [{"role": "user", "content": prompt}],
        temperature=GEN_TEMPERATURE,
        max_tokens=(GEN_MAX_TOKENS_CON if reading_method == "con" else GEN_MAX_TOKENS_DIRECT),
    )
    return reply.strip()


# ---------------- judge (evaluate_qa.py) ----------------

# get_anscheck_prompt (evaluate_qa.py:24-43), transcribed verbatim. The five
# branches differ in more than tone:
#   - knowledge-update DROPS the "only contains a subset -> no" sentence the
#     other non-temporal branches carry.
#   - single-session-preference labels the gold "Rubric", not "Correct Answer",
#     and grades against a rubric rather than an answer string.
#   - the abstention branch labels the gold "Explanation" and overrides the
#     question type entirely.
_BASE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
)
_TAIL = (
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

ANSCHECK_TEMPLATES = {
    "default": _BASE + _TAIL,
    "temporal-reasoning": _BASE
    + (
        "In addition, do not penalize off-by-one errors for the number of days. "
        "If the question asks for the number of days/weeks/months, etc., and the model "
        "makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the "
        "model's response is still correct. "
    )
    + _TAIL,
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response contains some previous information along with an updated answer, "
        "the response should be considered as correct as long as the updated answer is the "
        "required answer." + _TAIL
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a "
        "response from a model. Please answer yes if the response satisfies the desired "
        "response. Otherwise, answer no. The model does not need to reflect all the points "
        "in the rubric. The response is correct as long as it recalls and utilizes the "
        "user's personal information correctly."
        "\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a "
        "model. Please answer yes if the model correctly identifies the question as "
        "unanswerable. The model could say that the information is incomplete, or some "
        "other information is given but the asked information is not."
        "\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    ),
}

_DEFAULT_TASKS = ("single-session-user", "single-session-assistant", "multi-session")


def get_anscheck_prompt(task: str, question: str, answer_: str, response: str, abstention: bool):
    """Port of evaluate_qa.py:24-43, including its refusal to guess.

    Upstream ``raise NotImplementedError`` on an unrecognised task rather than
    falling back to the default template, and that strictness is the point: a
    silently mis-branched prompt grades knowledge-update questions under rules
    that penalise exactly the behaviour those questions are testing."""
    if abstention:
        return ANSCHECK_TEMPLATES["abstention"].format(question, answer_, response)
    if task in _DEFAULT_TASKS:
        template = ANSCHECK_TEMPLATES["default"]
    elif task in ("temporal-reasoning", "knowledge-update", "single-session-preference"):
        template = ANSCHECK_TEMPLATES[task]
    else:
        raise NotImplementedError(f"unknown question_type {task!r} (known: {QUESTION_TYPES})")
    return template.format(question, answer_, response)


def check_judge_model(model: str) -> None:
    """Mirror of print_qa_metrics.py:20's assert on the judge model.

    Raising here rather than at aggregation time means an unpinned judge fails
    before spending 500 calls, not after."""
    if model != JUDGE_MODEL_PIN:
        raise ValueError(
            f"judge model {model!r} is not the pinned {JUDGE_MODEL_PIN!r}; "
            "the official aggregator asserts this, so results judged otherwise are "
            "not comparable with published numbers. Pass the pin explicitly to override."
        )


def judge_answer(mem: AgenticMemory, instance: dict[str, Any], hypothesis: str) -> bool | None:
    """Binary judge verdict for one answered instance, or ``None`` when no
    ``judge`` role is configured.

    Deliberately NOT routed through ``mem.structured``: the official judge is a
    free-text call with ``max_tokens=10`` scored by ``'yes' in
    response.lower()`` (evaluate_qa.py:112-113). Wrapping it in guided JSON
    would change both the model's output distribution and the decision rule.
    The substring test is upstream's and is kept -- it is looser than equality
    (any reply containing "yes" passes), which matters when reading a judge's
    disagreement rate."""
    if mem.llm is None or not mem.llm.has_role("judge"):
        return None
    prompt = get_anscheck_prompt(
        str(instance["question_type"]),
        str(instance["question"]),
        str(instance["answer"]),
        hypothesis,
        abstention=is_abstention(str(instance["question_id"])),
    )
    reply = mem.llm.chat(
        "judge",
        [{"role": "user", "content": prompt}],
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    return "yes" in reply.strip().lower()


# ---------------- metrics (print_qa_metrics.py) ----------------


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Official aggregation: BOTH accuracies, plus the abstention cross-cut.

    ``records`` rows need ``question_type``, ``question_id`` and a bool
    ``label``; rows with ``label is None`` (unjudged) are excluded from every
    bucket rather than counted as wrong.

    Mirrors print_qa_metrics.py exactly, including two things worth restating:
    abstention rows are counted BOTH in their question type and in the
    abstention bucket (:19-23), and ``task_averaged`` weights the six types
    equally while ``overall`` weights questions equally -- so on the unequal
    official type counts they are different numbers. A type with no judged rows
    is omitted from ``by_type`` and from the task average (upstream would emit
    nan for it)."""
    by_type: dict[str, list[int]] = defaultdict(list)
    abstention: list[int] = []
    everything: list[int] = []
    for row in records:
        label = row.get("label")
        if label is None:
            continue
        hit = 1 if label else 0
        by_type[str(row["question_type"])].append(hit)
        everything.append(hit)
        if is_abstention(str(row["question_id"])):
            abstention.append(hit)

    def pct(values: list[int]) -> float:
        return round(100 * sum(values) / len(values), 2)

    ordered = [t for t in QUESTION_TYPES if by_type.get(t)]
    extra = sorted(t for t in by_type if t not in QUESTION_TYPES)
    return {
        "by_type": {t: {"acc": pct(by_type[t]), "n": len(by_type[t])} for t in [*ordered, *extra]},
        "task_averaged": (
            round(sum(pct(by_type[t]) for t in ordered) / len(ordered), 2) if ordered else None
        ),
        "overall": pct(everything) if everything else None,
        "abstention": {"acc": pct(abstention), "n": len(abstention)} if abstention else None,
        "n": len(everything),
    }


# ---------------- one-instance driver ----------------


def run_instance(
    mem: AgenticMemory,
    instance: dict[str, Any],
    k: int | dict = 10,
    memory_types: tuple[str, ...] | None = None,
    budget_tokens: int = 6000,
    reading_method: str = "con",
    max_sessions: int | None = None,
    full_context: bool = False,
    judge: bool = True,
    capture_retrieval: bool = False,
) -> dict[str, Any]:
    """Ingest one instance's haystack into ``mem``, answer its question, judge it.

    ``mem`` must be scoped to this instance (see ``ingest``). Returns one record
    in ``aggregate``'s input shape, plus the fields needed to rebuild the
    official ``{question_id, hypothesis}`` hypothesis file.

    ``full_context=True`` skips retrieval and feeds the whole haystack via
    ``render_sessions`` -- the paper's no-memory baseline. Ingest still runs, so
    write-path cost stays measurable and comparable against the retrieval runs;
    pass a ``passthrough`` memory to keep that cost at zero.

    Judging is on by default because there is no string metric to fall back to:
    an unjudged LongMemEval run has no score at all, only hypotheses."""
    turns = ingest(mem, instance, max_sessions)
    capture: dict[str, Any] | None = {} if capture_retrieval else None
    hypothesis = answer(
        mem,
        instance,
        k=k,
        memory_types=memory_types,
        budget_tokens=budget_tokens,
        reading_method=reading_method,
        history=render_sessions(instance, max_sessions) if full_context else None,
        capture=capture,
    )
    row = {
        "question_id": str(instance["question_id"]),
        "question_type": str(instance["question_type"]),
        "question": str(instance["question"]),
        "answer": str(instance["answer"]),
        "hypothesis": hypothesis,
        "label": judge_answer(mem, instance, hypothesis) if judge else None,
        "turns": turns,
    }
    if capture is not None:
        row["retrieval"] = capture
    return row
