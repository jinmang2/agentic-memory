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
   ``judge_answer`` enforces the pin on its first call so a mis-configured judge
   costs zero calls, not 500.
4. **The context is scrubbed of ``has_answer`` before formatting.**
   run_generation.py pops the turn-level evidence label off every chunk
   (:177-191) before it reaches the prompt. Skipping that hands the model the
   gold location, and it inflates the full-context baseline rather than
   crashing -- see ``render_sessions``.

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
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from agmem.core.types import MemoryBundle
from agmem.memory import AgenticMemory
from agmem.retrieval.planned import searcher_for

logger = logging.getLogger("agmem.bench.longmemeval")

# print_qa_metrics.py:17 initialises type2acc with exactly these six, in this
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
# run_generation.py:343 -- `max_retrieval_length = model_max_length - gen_length - 1000`,
# the headroom upstream leaves for the answer plus the non-history prompt text.
UPSTREAM_CONTEXT_RESERVE = 1000

# The turn-level evidence label the benchmark ships. Upstream strips it from
# every chunk before formatting (run_generation.py:177-191,
# `turn_entry.pop('has_answer')`) -- it marks WHICH turns hold the answer, so
# leaving it in the prompt hands the model the gold location.
EVIDENCE_LABEL_KEY = "has_answer"


def is_abstention(question_id: str) -> bool:
    """Upstream test is ``'_abs' in entry['question_id']`` (evaluate_qa.py:101,
    print_qa_metrics.py:22) -- a SUBSTRING test, not ``endswith``.

    Kept as-is rather than "corrected" to ``endswith``: the abstention branch of
    the judge prompt is selected by this exact test upstream, so tightening it
    here would silently regrade any id with ``_abs`` in the middle."""
    return "_abs" in question_id


# ---------------- data loading ----------------


def _capped(items: list[Any], max_items: int | None) -> list[Any]:
    """``items`` truncated to ``max_items``, where ``None`` (not ``0``) is the
    "no cap" sentinel."""
    return items if max_items is None else items[:max_items]


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
    it must never reach the memory -- it is yielded for recall scoring only.

    ``max_sessions=None`` means the whole haystack; ``0`` means zero sessions
    and is NOT a synonym for "all" (a truthiness fallback here would turn the
    smallest ablation point into a silent full run)."""
    sessions = instance.get("haystack_sessions", [])
    dates = instance.get("haystack_dates", [])
    ids = instance.get("haystack_session_ids", [])
    for idx, session in enumerate(_capped(sessions, max_sessions)):
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


def sort_haystack_by_date(instance: dict[str, Any]) -> dict[str, Any]:
    """``instance`` with its haystack re-ordered chronologically, as a copy.

    Upstream sorts the assembled chunks right before formatting
    (``retrieved_chunks.sort(key=lambda x: x[0])``, run_generation.py:225) —
    a plain string sort on the date, which is well-defined because every date in
    this benchmark is ``YYYY/MM/DD (HH:MM)``. It is stable, so equal dates keep
    haystack order.

    It is a no-op on ``_s``/``_m``, whose haystacks ship in date order, and it is
    NOT a no-op on ``longmemeval_oracle.json``: measured on the released file,
    34/500 instances render a different prompt sorted than unsorted (same bytes,
    different session order), which is exactly the byte-diff
    ``scripts/repro/lme_audit/prompt_rediff.py`` reports. An oracle run that
    skips this is comparing against the paper's .870/.924 with a prompt the
    paper never sent.

    The three parallel arrays are permuted TOGETHER — ``haystack_sessions``,
    ``haystack_dates`` and ``haystack_session_ids`` are indexed by position
    (run_generation.py:75-81), so sorting one alone would silently re-label
    every session and mis-key the retrieval gold. Upstream carries no ids
    through ``prepare_prompt`` and so cannot make that mistake there; we can.

    Returns a shallow copy and leaves the argument untouched, for the same
    reason ``render_sessions`` strips ``has_answer`` onto a copy: the caller
    still needs the original instance for scoring, and a driver that sorts
    in place would mutate the loaded dataset under every later reader of it.
    Session numbering (``### Session {i+1}``) is applied by ``render_sessions``
    AFTER the permutation, matching upstream's enumerate over sorted chunks.
    """
    sessions = instance.get("haystack_sessions", [])
    dates = instance.get("haystack_dates", [])
    ids = instance.get("haystack_session_ids", [])

    def date_at(i: int) -> str:
        return str(dates[i]) if i < len(dates) else ""

    order = sorted(range(len(sessions)), key=date_at)
    out = dict(instance)
    out["haystack_sessions"] = [sessions[i] for i in order]
    # A ragged array is padded rather than filtered: dropping the tail would
    # shorten the array and re-pair every session with the wrong date, which is
    # the failure this function exists to prevent. "" and ``session_{i}`` are
    # the fallbacks ``iter_turns``/``render_sessions`` already use.
    if dates:
        out["haystack_dates"] = [date_at(i) for i in order]
    if ids:
        out["haystack_session_ids"] = [ids[i] if i < len(ids) else f"session_{i}" for i in order]
    return out


def evidence_session_ids(instance: dict[str, Any]) -> set[str]:
    """``answer_session_ids`` -- the session-level recall gold.

    It is NOT empty for abstention questions, contrary to what this docstring
    claimed until 2026-08-17: measured on ``longmemeval_s_cleaned``, all 30
    abstention instances carry exactly one ``answer_session_id`` (the session
    holding the false premise's subject) and **zero** ``has_answer`` turns. So
    the session-level gold exists while the turn-level gold does not, and the
    official retrieval evaluation drops abstention instances for the second
    reason, not the first (run_retrieval.py:396)."""
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


def render_sessions(
    instance: dict[str, Any], max_sessions: int | None = None, history_format: str = "json"
) -> str:
    """Full-history context in upstream's ``history_format`` shape.

    ``history_format`` is upstream's flag of the same name (run_generation.py:234-247),
    both branches transcribed from the clone rather than reconstructed:

    - ``json`` — the session dumped as ``json.dumps([{role, content}, ...])``,
      preceded by a newline (:238). The default here because it is
      run_generation.sh's.
    - ``nl`` — ``"\\n\\n{role}: {content.strip()}"`` per turn (:245). Note the
      ``.strip()``, which ``json`` does not apply: the two formats therefore do
      not carry byte-identical content, and that is upstream's behaviour, not a
      simplification of it.

    The choice is not cosmetic. §5.5 reports that JSON does not consistently beat
    NL *without* chain-of-note and always beats it *with* — an interaction of up
    to 10 pp between two flags that are usually reported as neither.

    Block layout is run_generation.py:252 and the per-session value is
    ``'\\n' + json.dumps(chunk_entry)`` (:238) where ``chunk_entry`` is the
    session as a list of ``{role, content}``. Sessions are emitted in haystack
    order, which is already chronological for the ``_s``/``_m`` variants --
    ``longmemeval_oracle.json`` is NOT sorted, a documented upstream quirk, so
    an oracle run must call ``sort_haystack_by_date`` before this (34/500
    instances render differently otherwise).

    ``has_answer`` is stripped from every turn first, as upstream's clean-up
    loop does before formatting (:177-191). That is not cosmetic: the label
    marks which turns hold the answer, so rendering the raw session tells the
    model where to look and inflates the very baseline this function exists to
    reproduce (the paper's GPT-4o 60.6% on ``_s``, the one configuration that
    needs no memory system at all). Upstream ``pop``s in place; we strip onto a
    copy so the instance keeps its turn-level recall gold for scoring."""
    if history_format not in ("json", "nl"):
        raise ValueError(f"unknown history_format {history_format!r} (upstream: json | nl)")
    sessions = instance.get("haystack_sessions", [])
    dates = instance.get("haystack_dates", [])
    out = []
    for i, session in enumerate(_capped(sessions, max_sessions)):
        date = dates[i] if i < len(dates) else ""
        cleaned = [
            {k: v for k, v in turn.items() if k != EVIDENCE_LABEL_KEY}
            if isinstance(turn, dict)
            else turn
            for turn in session
        ]
        if history_format == "json":
            sess_string = "\n" + json.dumps(cleaned)
        else:
            sess_string = "".join(
                "\n\n{}: {}".format(turn.get("role", ""), str(turn.get("content", "")).strip())
                if isinstance(turn, dict)
                else f"\n\n{turn}"
                for turn in cleaned
            )
        out.append(
            "\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n".format(
                i + 1, date, sess_string
            )
        )
    return "".join(out)


def upstream_max_history_tokens(model_max_length: int, reading_method: str = "con") -> int:
    """Upstream's context arithmetic: ``model_max_length - gen_length - 1000``
    (run_generation.py:343), where ``gen_length`` is 800 for ``con`` and 500 for
    ``direct``. Exposed rather than inlined because the full-context path has no
    other bound -- ``budget_tokens`` governs the retrieved bundle only -- and a
    caller reinventing the reserve would silently reproduce a different prompt
    length. For gpt-4o (128k) with ``con`` that is 126,200."""
    gen_length = GEN_MAX_TOKENS_CON if reading_method == "con" else GEN_MAX_TOKENS_DIRECT
    return model_max_length - gen_length - UPSTREAM_CONTEXT_RESERVE


def _sampling_kwargs(
    mem: AgenticMemory, role: str, max_tokens: int, temperature: float
) -> dict[str, Any]:
    """Upstream's ``temperature``/``max_tokens`` for one call, expressed the way
    THIS role's model accepts them.

    Both benchmark constants are per-reading-method, not per-run, so they have to
    be passed as call overrides rather than baked into the role config — and a
    literal ``max_tokens=800, temperature=0.0`` override is what
    ``LLMClient.chat`` puts on the wire, ahead of everything the role knows.
    That is fine for gpt-4o/gpt-4o-mini and a hard 400 for the newer Chat
    Completions models: gpt-5.6-luna rejects ``max_tokens`` (it requires
    ``max_completion_tokens``) and rejects any non-default ``temperature``,
    both documented on its ``ModelSpec`` (bench/registry.py). Reading the key
    off ``RoleConfig.max_tokens_key`` and dropping ``temperature`` when the role
    was built with ``temperature=None`` (``make_roles(fixed_sampling=True)``,
    the flag that says the model has no temperature to set) is what lets the
    same arm definition run on both.

    **Disclose this as a deviation when the reader is fixed-sampling**: upstream
    pins ``temperature=0`` for every model it evaluates (run_generation.py:363),
    and a model that only samples at 1.0 is not being read deterministically.
    It is not a choice we can make differently — the request fails otherwise —
    but it belongs in the run's stamp, not in a footnote.

    Falls back to upstream's literal kwargs when the role config is unreadable
    (a stub client in tests), which keeps every existing caller byte-identical.
    """
    roles = getattr(mem.llm, "roles", None)
    cfg = roles.get(role) if isinstance(roles, dict) else None
    if cfg is None:
        return {"temperature": temperature, "max_tokens": max_tokens}
    kwargs: dict[str, Any] = {getattr(cfg, "max_tokens_key", "max_tokens"): max_tokens}
    if getattr(cfg, "temperature", temperature) is not None:
        kwargs["temperature"] = temperature
    return kwargs


def answer(
    mem: AgenticMemory,
    instance: dict[str, Any],
    k: int | dict = 10,
    memory_types: tuple[str, ...] | None = None,
    budget_tokens: int = 6000,
    reading_method: str = "con",
    history: str | None = None,
    max_history_tokens: int | None = None,
    capture: dict[str, Any] | None = None,
    searcher: Any | None = None,
    budget_key: str | None = None,
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

    ``max_history_tokens`` caps whichever history is used, mirroring upstream's
    truncation of the assembled history string (run_generation.py:266-279) --
    which it does unconditionally, having the model's tokenizer to hand. We do
    not, so the cap is opt-in and measured with ``MemoryBundle``'s chars-per-
    token estimate; ``upstream_max_history_tokens`` computes the value upstream
    would use. It matters on the full-context path: ``budget_tokens`` never
    reaches an explicit ``history``, so an uncapped ``_m`` haystack (~1.5M
    tokens) goes to the API whole.

    ``memory_types=None`` defers to the memory's ``default_memory_types``, so a
    caller does not have to know what the configured organizers produce.

    ``budget_key`` labels this call's row in the client's ``BudgetTracker`` and
    in its I/O trace (``LLMClient.chat``'s own parameter, which never reaches
    the API payload). A LongMemEval run is one question per instance answered by
    a THREAD POOL sharing one client, so a driver cannot get per-question tokens
    by diffing a shared budget — the diff belongs to whichever rows happened to
    finish in between. Passing e.g. ``f"generate|{question_id}"`` gives each row
    its own exact usage, and the driver folds the keys back into role totals
    before pricing (the judge model is not the reader model, and a per-row key
    would otherwise be priced at the reader's rates). ``None`` keeps the
    tracker's old shape: one bucket per role.

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
        # Same uniform read entry as LoCoMo's `answer`: the memory itself, or
        # the memory wrapped in the read policy its config names. This path used
        # to be the one that silently could NOT reach a policy.
        agent_metrics: dict[str, Any] = {}
        bundle = (searcher if searcher is not None else searcher_for(mem)).search(
            question, memory_types=memory_types, k=k, metrics=agent_metrics
        )
        if capture is not None:
            capture["agent"] = agent_metrics
        history = bundle.render(budget_tokens=budget_tokens) or "(no memories found)"
        # The verbatim recency window a methodology keeps OUTSIDE retrieval and
        # injects on every question (``Organizer.recent_context()`` — MemoryOS's
        # resident STM, which upstream's ``get_response`` puts at the front of
        # the prompt). LoCoMo's ``answer`` has always done this; this path did
        # not, so measuring such a methodology here silently dropped the channel
        # without so much as a degradation note. It leads the context for the
        # same reason it does there, and because ``max_history_tokens`` truncates
        # from the tail.
        #
        # Retrieval path only: an explicit ``history`` is the full-context
        # baseline, which reads no memory at all.
        recent = "\n".join(
            text for text in (org.recent_context() for org in mem.organizers) if text
        )
        if recent:
            history = "Recent conversation:\n" + recent + "\n\n" + history
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
    if max_history_tokens is not None:
        limit = max_history_tokens * MemoryBundle.CHARS_PER_TOKEN
        if len(history) > limit:
            logger.warning(
                "longmemeval: truncating history from %d to %d chars (~%d tokens)",
                len(history),
                limit,
                max_history_tokens,
            )
            history = history[:limit]
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
    if capture is not None:
        # The assembled prompt, not just the history: a driver's row-level
        # fingerprint (sha256 + length) has to cover the template and the
        # question date too, or two arms that differ only in reading method
        # fingerprint identically. Recorded here rather than rebuilt by the
        # caller so the hash can never describe a prompt we did not send.
        capture["prompt"] = prompt
    reply = mem.llm.chat(
        "generate",
        [{"role": "user", "content": prompt}],
        budget_key=budget_key,
        **_sampling_kwargs(
            mem,
            "generate",
            GEN_MAX_TOKENS_CON if reading_method == "con" else GEN_MAX_TOKENS_DIRECT,
            GEN_TEMPERATURE,
        ),
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
            "not comparable with published numbers. Pass enforce_pin=False to judge "
            "anyway (and label the result as not officially comparable)."
        )


def configured_judge_model(mem: AgenticMemory) -> str | None:
    """The model name configured for the ``judge`` role, or ``None`` when the
    client does not expose one (a stub, or a client without ``roles``).

    ``None`` means "cannot be checked", never "checked and fine" -- the pin is
    only enforceable against a name we can actually read."""
    roles = getattr(mem.llm, "roles", None)
    role_config = roles.get("judge") if isinstance(roles, dict) else None
    model = getattr(role_config, "model", None)
    return str(model) if model else None


def judge_answer(
    mem: AgenticMemory,
    instance: dict[str, Any],
    hypothesis: str,
    enforce_pin: bool = True,
    budget_key: str | None = None,
) -> bool | None:
    """Binary judge verdict for one answered instance, or ``None`` when no
    ``judge`` role is configured.

    The pin is enforced here, on the first call, so a mis-configured judge costs
    zero calls instead of 500 (``check_judge_model``). It can only fire when the
    client exposes its role config -- see ``configured_judge_model`` -- so an
    unreadable client still runs, and ``enforce_pin=False`` runs a deliberately
    off-pin judge (e.g. the release's ``gpt-4o-mini`` or local-vLLM entries) for
    a cheap disagreement study.

    Deliberately NOT routed through ``mem.structured``: the official judge is a
    free-text call with ``max_tokens=10`` scored by ``'yes' in
    response.lower()`` (evaluate_qa.py:112-113). Wrapping it in guided JSON
    would change both the model's output distribution and the decision rule.
    The substring test is upstream's and is kept -- it is looser than equality
    (any reply containing "yes" passes), which matters when reading a judge's
    disagreement rate."""
    if mem.llm is None or not mem.llm.has_role("judge"):
        return None
    if enforce_pin:
        model = configured_judge_model(mem)
        if model is not None:
            check_judge_model(model)
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
        budget_key=budget_key,
        **_sampling_kwargs(mem, "judge", JUDGE_MAX_TOKENS, JUDGE_TEMPERATURE),
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
    nan for it).

    ``task_averaged`` averages the UNROUNDED type means and rounds once, as
    upstream does (``round(np.mean(task_acc), 4)`` over raw means, :31).
    Averaging the reported per-type percentages instead shifts the headline by
    up to 0.01pp -- small, but self-inflicted drift in the one number this
    module exists to report unambiguously.

    ``overall`` is the concatenation of the SIX known type buckets, not of every
    row: upstream builds ``all_acc`` inside its ``for k, v in type2acc.items()``
    report loop (:28-30), so a row whose ``question_type`` is outside
    ``QUESTION_TYPES`` cannot reach it -- upstream would in fact ``KeyError`` on
    one. Such rows still appear in ``by_type`` (as ``extra``) so they are not
    silently dropped, but they stay out of both headline numbers. On official
    data the two definitions coincide; they diverge only on augmented or
    relabelled sets, which is exactly when a headline must not shift for a
    reason the paper never had."""
    by_type: dict[str, list[int]] = defaultdict(list)
    abstention: list[int] = []
    for row in records:
        label = row.get("label")
        if label is None:
            continue
        hit = 1 if label else 0
        by_type[str(row["question_type"])].append(hit)
        if is_abstention(str(row["question_id"])):
            abstention.append(hit)

    def mean(values: list[int]) -> float:
        return 100 * sum(values) / len(values)

    def pct(values: list[int]) -> float:
        return round(mean(values), 2)

    ordered = [t for t in QUESTION_TYPES if by_type.get(t)]
    extra = sorted(t for t in by_type if t not in QUESTION_TYPES)
    everything = [hit for t in ordered for hit in by_type[t]]
    return {
        "by_type": {t: {"acc": pct(by_type[t]), "n": len(by_type[t])} for t in [*ordered, *extra]},
        "task_averaged": (
            round(sum(mean(by_type[t]) for t in ordered) / len(ordered), 2) if ordered else None
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
    max_history_tokens: int | None = None,
    full_context: bool = False,
    history_format: str = "json",
    judge: bool = True,
    enforce_pin: bool = True,
    capture_retrieval: bool = False,
    budget_key: str | None = None,
) -> dict[str, Any]:
    """Ingest one instance's haystack into ``mem``, answer its question, judge it.

    ``mem`` must be scoped to this instance (see ``ingest``). Returns one record
    in ``aggregate``'s input shape, plus the fields needed to rebuild the
    official ``{question_id, hypothesis}`` hypothesis file.

    ``full_context=True`` skips retrieval and feeds the whole haystack via
    ``render_sessions`` in ``history_format`` (``json``|``nl``, upstream's flag;
    it has no effect on the retrieval path, whose context comes from
    ``MemoryBundle.render``) -- the paper's no-memory baseline. Ingest still runs, so
    write-path cost stays measurable and comparable against the retrieval runs;
    pass a ``passthrough`` memory to keep that cost at zero. That path is the
    one that needs ``max_history_tokens`` (see ``answer``): nothing else bounds
    it, and on ``_m`` the whole haystack would go to the API uncapped.

    Judging is on by default because there is no string metric to fall back to:
    an unjudged LongMemEval run has no score at all, only hypotheses.
    ``enforce_pin`` is forwarded to ``judge_answer``.

    ``budget_key`` is forwarded to both calls with a ``generate|``/``judge|``
    prefix, so a concurrent driver can attribute tokens to this question rather
    than to a shared bucket — see ``answer``. An oracle instance is NOT sorted
    here: sorting is the caller's decision because it depends on the dataset
    variant (``sort_haystack_by_date`` is a no-op on ``_s``, load-bearing on
    ``longmemeval_oracle.json``), and doing it silently would hide from the
    record which prompt was actually sent."""
    turns = ingest(mem, instance, max_sessions)
    capture: dict[str, Any] | None = {} if capture_retrieval else None
    hypothesis = answer(
        mem,
        instance,
        k=k,
        memory_types=memory_types,
        budget_tokens=budget_tokens,
        reading_method=reading_method,
        history=(
            render_sessions(instance, max_sessions, history_format) if full_context else None
        ),
        max_history_tokens=max_history_tokens,
        capture=capture,
        budget_key=f"generate|{budget_key}" if budget_key else None,
    )
    row = {
        "question_id": str(instance["question_id"]),
        "question_type": str(instance["question_type"]),
        "question": str(instance["question"]),
        "answer": str(instance["answer"]),
        "hypothesis": hypothesis,
        "label": (
            judge_answer(
                mem,
                instance,
                hypothesis,
                enforce_pin,
                budget_key=f"judge|{budget_key}" if budget_key else None,
            )
            if judge
            else None
        ),
        "turns": turns,
    }
    if capture is not None:
        row["retrieval"] = capture
    return row
