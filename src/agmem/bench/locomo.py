"""LoCoMo benchmark pipeline (snap-research/locomo, 10 conversations).

The primary metric is string-based token F1 with SQuAD-style normalization and
Porter stemming, mirroring snap-research/locomo's ``normalize_answer`` +
``f1_score`` — the cheapest reproduction entry point (docs/06 Phase 2). BLEU-1
accompanies it but has two distinct provenances, which must not be conflated:
``bleu1`` is ours (upstream locomo computes no BLEU at all — its third metric is
a rouge-1 F under a ``rougel_score`` name), while ``bleu1_wujiang`` mirrors
WujiangXu/A-Mem's nltk BLEU-1 and is what ``eval_mode="wujiang"`` scores. A Mem0-style binary J-score LLM judge is additionally
available (opt-in via ``evaluate(judge=True)``) for cat 1-4. Categories:
1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial (gold in
``adversarial_answer``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import string
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from agmem._porter import PorterStemmer
from agmem.memory import AgenticMemory
from agmem.organizers.base import overrides as base_overrides
from agmem.retrieval.planned import searcher_for

logger = logging.getLogger("agmem.bench.locomo")

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

# A-Mem official eval (WujiangXu test_advanced.py generate_query_llm): the
# question is first rewritten into LLM keywords and the keyword string becomes
# the retrieval query. Prompt kept verbatim, including the 'cosmos' quirk the
# comma-separated example overrides in practice.
KEYWORD_QUERY_SCHEMA = {
    "type": "object",
    "properties": {"keywords": {"type": "string"}},
    "required": ["keywords"],
}

KEYWORD_QUERY_PROMPT = """Given the following question, generate several keywords, using 'cosmos' as the separator.

Question: {question}

Format your response as a JSON object with a "keywords" field containing the selected text.

Example response format:
{{"keywords": "keyword1, keyword2, keyword3"}}"""

# MemoryOS EVAL lineage only: `llm_extract_keywords` (eval/utils.py:361), one LLM
# call per question whose result is added to the segment score as a containment
# overlap. `memoryos-pypi` deleted the same term (`query_keywords = set()`
# "Keywords extraction removed"), so this is a real lineage difference and a
# read-time cost the pypi shape does not pay. Prompt transcribed; the object
# wrapper is this project's structured-output adaptation of a bare comma string.
MEMORYOS_KEYWORD_SCHEMA = {
    "type": "object",
    "properties": {"keywords": {"type": "string"}},
    "required": ["keywords"],
}

MEMORYOS_KEYWORD_PROMPT = """You are a keyword extraction expert. Please extract the keywords \
of the conversation topic from the following dialogue, separated by commas, and do
not exceed three:
{question}

Return JSON: {{"keywords": "keyword1, keyword2, keyword3"}}"""

# Common across all methodologies (fair relative comparison). The temporal
# instructions are shared by both upstream evals (Nemori search.py computes
# absolute dates from timestamps; A-Mem cat2 instructs date use).
ANSWER_PROMPT = """Answer the question using ONLY the retrieved memories below.
If the question asks when something happened, compute the absolute date from
the timestamps shown with the memories (e.g. "last year" in a memory dated
4 May 2022 means 2021). If memories conflict, prefer the most recent one.
Reply with the shortest span that answers the question — a name, phrase, or \
date — with no explanation. If the memories do not contain the answer, reply \
exactly: No information available.

Memories:
{context}

Question: {question}
Short answer:"""

# cat5 (adversarial) upstream drops the abstention option: the answer is
# expected to be produced, not refused, so ANSWER_PROMPT minus the
# "No information available" sentence (everything else identical).
ANSWER_PROMPT_NO_ABSTAIN = """Answer the question using ONLY the retrieved memories below.
If the question asks when something happened, compute the absolute date from
the timestamps shown with the memories (e.g. "last year" in a memory dated
4 May 2022 means 2021). If memories conflict, prefer the most recent one.
Reply with the shortest span that answers the question — a name, phrase, or \
date — with no explanation.

Memories:
{context}

Question: {question}
Short answer:"""

# Mem0-standard binary LLM judge (cat 1-4 only). Grades the generated answer
# CORRECT/WRONG relative to gold with the same generous rubric Mem0 uses.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["CORRECT", "WRONG"]}},
    "required": ["label"],
}

JUDGE_PROMPT = """Label the generated answer as CORRECT or WRONG relative to the gold answer. Grade generously:
- Paraphrases and differently-worded answers that convey the same meaning are CORRECT.
- If the gold answer has multiple items, getting at least one right is CORRECT.
- Extra detail is fine as long as it does not contradict the gold answer.
- Dates within ~14 days of the gold, or durations within ~50%, are CORRECT; relative dates that resolve into the gold's time window are CORRECT.
Otherwise label WRONG.

Question: {question}
Gold answer: {gold}
Generated answer: {pred}

Respond with a JSON object: {{"label": "CORRECT"}} or {{"label": "WRONG"}}."""


# ---------------- data loading ----------------


def load_locomo(path: str | Path) -> list[dict[str, Any]]:
    """Raw snap-research/locomo samples (typically 10); raises on a missing
    file or invalid JSON — no fallback to an empty list."""
    return json.loads(Path(path).read_text())


def iter_turns(sample: dict[str, Any], max_sessions: int | None = None):
    """Yield (session_no, date_str, speaker, text) in conversation order."""
    conv = sample["conversation"]
    numbers = sorted(
        int(m.group(1))
        for k in conv
        if (m := re.fullmatch(r"session_(\d+)", k)) and isinstance(conv[k], list)
    )
    for n in numbers[: max_sessions or len(numbers)]:
        date = conv.get(f"session_{n}_date_time", "")
        for turn in conv[f"session_{n}"]:
            yield n, date, turn.get("speaker", "?"), turn.get("text", "")


def evidence_sessions(q: dict[str, Any]) -> set[int]:
    """Session numbers referenced by `q["evidence"]` entries (e.g. `"D3:2"` ->
    `3`); empty set if there's no evidence field or no `D<n>:` prefixed refs."""
    out = set()
    for ev in q.get("evidence", []):
        if m := re.match(r"D(\d+):", str(ev)):
            out.add(int(m.group(1)))
    return out


def select_questions(
    sample: dict[str, Any],
    max_sessions: int | None = None,
    categories: tuple[int, ...] = (1, 2, 3, 4, 5),
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Questions answerable within the ingested session prefix."""
    out = []
    for q in sample["qa"]:
        if q.get("category") not in categories:
            continue
        ev = evidence_sessions(q)
        if max_sessions and ev and max(ev) > max_sessions:
            continue
        out.append(q)
    return out[:limit] if limit else out


# ---------------- metrics (SQuAD-style) ----------------

_ARTICLES = re.compile(r"\b(a|an|the|and)\b")
# mode="original" is the pre-existing behaviour every stored run under
# results/repro/ was scored with, NOT the faithful one: snap-research/locomo's
# normalize_answer stems with nltk's default (NLTK_EXTENSIONS). Switching to
# mode="nltk" moves F1/BLEU-1 and so is an explicit, deferred decision — see
# agmem._porter's docstring and docs/research/amac-admission-gate.md §4.
_STEMMER = PorterStemmer(mode="original")


def normalize(text: str) -> list[str]:
    """Tokens, not text: lowercased, punctuation stripped, stopwords
    (a/an/the/and) removed, whitespace-split, then Porter-stemmed — matching
    the official snap-research/locomo ``normalize_answer`` used for F1/BLEU-1."""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _ARTICLES.sub(" ", text)
    return [_STEMMER.stem(tok) for tok in text.split()]


def token_f1(pred: str, gold: str) -> float:
    """1.0 if both normalize to no tokens, 0.0 if only one does, else the
    harmonic mean of precision/recall over the token multiset overlap."""
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def bleu1(pred: str, gold: str) -> float:
    """Same empty-token edge case as `token_f1`; otherwise unigram precision
    scaled by a brevity penalty when `pred` is shorter than `gold`.

    Ours, not anyone's: snap-research/locomo computes NO BLEU at all (its
    `evaluation.py` scores exact match, stemmed token F1, and a `rougel_score`
    that actually returns rouge-1 F). This shares `normalize` with `token_f1`
    so the two `ours`-mode numbers are at least mutually consistent, but it is
    not a port of an official scorer and must not be cited as one — see
    `bleu1_wujiang` for the one BLEU that does mirror an upstream."""
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    precision = sum(common.values()) / len(p)
    brevity = 1.0 if len(p) >= len(g) else pow(2.718281828, 1 - len(g) / len(p))
    return brevity * precision


def gold_for(q: dict[str, Any]) -> str:
    """Gold answer for a question: ``answer`` (falling back to
    ``adversarial_answer`` for cat5). For cat3 (open-domain) the upstream
    gold packs multiple aliases as ``"A; B; C"`` — take the first, matching
    snap-research/locomo scoring which keys on the primary answer."""
    raw = q.get("answer")
    if raw is None:
        raw = q.get("adversarial_answer", "")
    gold = str(raw)
    if q.get("category") == 3 and ";" in gold:
        gold = gold.split(";")[0].strip()
    return gold


# ------------- WujiangXu/A-Mem faithful eval (eval_mode="wujiang") -------------
# Mirrors the official A-Mem reproduction repo (WujiangXu/AgenticMemory, cloned
# to scratchpad). All file:line citations below reference that clone. This path
# is DISTINCT from the snap-research-style `ours` metrics above (which stem +
# strip articles); nothing here alters the `ours` functions.

CAT5_NOT_MENTIONED = "Not mentioned in the conversation"

# cat5 adversarial MCQ prompt — verbatim shape of test_advanced.py
# advancedMemAgent.answer_question (:155-159): a 2-option choice between the
# gold answer and the "Not mentioned" distractor, scored by set-based F1
# against the gold. Kept as one f-string with the same wording/spacing.
# MemoryOS's role-play system prompt, transcribed from its LoCoMo driver
# (`main_loco_parse.generate_system_response_with_meta`). OPT-IN, and off by
# default, for a reason that is about this harness rather than about fidelity:
# every config here answers through the same `ANSWER_PROMPT` so that a score
# difference is attributable to the memory system, and a per-methodology answer
# prompt breaks that. Upstream's own harness does use this, so a run that wants
# to reproduce MemoryOS's published operating point end-to-end should pass
# `persona=` — and then its numbers are comparable to upstream's, not to the
# other configs in docs/09.
PERSONA_SYSTEM_PROMPT = """You are role-playing as {speaker_b} in a conversation with the user \
who is playing {speaker_a}. Here are some of your character traits and knowledge:
{assistant_knowledge}
Any content referring to 'User' in the prompt refers to {speaker_a}'s content, \
and any content referring to 'AI' or 'assistant' refers to {speaker_b}'s content.
Your task is to answer questions about {speaker_a} or {speaker_b} in an \
extremely concise manner.
When the question is: "What did the charity race raise awareness for?", you \
should not answer in the form of: "The charity race raised awareness for mental \
health." Instead, it should be: "mental health", as this is more concise."""

CAT5_MCQ_PROMPT = """Based on the context: {context}, answer the following \
question. {question}

Select the correct answer: {opt_a} or {opt_b}  Short answer:"""


def _tok_wujiang(text: str) -> list[str]:
    """Upstream ``simple_tokenize`` (utils.py:34-38): to-str, lowercase, replace
    each of ``. , ! ?`` with a space, whitespace-split. NO Porter stemming, NO
    article/stopword removal — this is the *only* normalization upstream F1
    applies."""
    text = str(text)
    return (
        text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()
    )


def token_f1_wujiang(pred: str, gold: str) -> float:
    """Set-based token F1 mirroring ``utils.calculate_metrics`` (utils.py:129-145):
    both sides ``.strip()``-ed, tokenized, then de-duplicated into SETS;
    ``precision = |inter| / |pred_set|``, ``recall = |inter| / |gold_set|``,
    F1 = harmonic mean. An empty prediction or reference yields 0.0 (utils.py's
    top-level empty guard :112 and the ``not pred_tokens or not ref_tokens``
    guard :140). Repeated tokens do NOT inflate the score (set semantics)."""
    pred = str(pred).strip()
    gold = str(gold).strip()
    if not pred or not gold:
        return 0.0
    pred_set = set(_tok_wujiang(pred))
    gold_set = set(_tok_wujiang(gold))
    if not pred_set or not gold_set:
        return 0.0
    common = pred_set & gold_set
    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


_WUJIANG_BLEU_WARNED = False


def bleu1_wujiang(pred: str, gold: str) -> float | None:
    """Upstream BLEU-1 (``utils.calculate_metrics``), or ``None`` when nltk is absent.

    Upstream computes BLEU with a DIFFERENT tokenizer than its F1:
    ``nltk.word_tokenize(text.lower())`` (not ``simple_tokenize``), fed to
    ``nltk.translate.bleu_score.sentence_bleu`` with ``weights=(1, 0, 0, 0)``
    and ``SmoothingFunction().method1``. Reference is a single-element list.

    This calls the real nltk rather than reimplementing it, on the opposite
    terms from ``agmem._porter``: Porter is a fully specified algorithm that
    transcribes cleanly, whereas ``word_tokenize`` is punkt sentence splitting
    plus the Treebank tokenizer, and a hand-rolled approximation of it would be
    exactly the "naive in-python fallback" this project bans (docs/03 §5).
    So nltk is an optional dependency (``[eval]``) and its absence degrades
    EXPLICITLY to ``None`` — never to ``bleu1``, whose article-stripping,
    Porter-stemming ``normalize`` would silently report an ``ours`` number
    under a ``wujiang`` label. That substitution is what
    ``eval_mode="wujiang"`` used to do, so every stored wujiang ``bleu1``
    predating this function is an ``ours``/``wujiang`` hybrid (docs/14 §4)."""
    global _WUJIANG_BLEU_WARNED
    try:
        from nltk.tokenize import word_tokenize
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        if not _WUJIANG_BLEU_WARNED:
            logger.warning(
                "locomo: nltk not installed — upstream BLEU-1 unavailable in wujiang mode, "
                "omitting the metric (explicit degradation; install the [eval] extra). "
                "F1 is unaffected."
            )
            _WUJIANG_BLEU_WARNED = True
        return None
    try:
        pred_tokens = word_tokenize(str(pred).lower())
        ref_tokens = [word_tokenize(str(gold).lower())]
    except LookupError:  # nltk present, punkt corpus not downloaded
        if not _WUJIANG_BLEU_WARNED:
            logger.warning(
                "locomo: nltk punkt data missing — upstream BLEU-1 unavailable in wujiang "
                "mode, omitting the metric (run nltk.download('punkt_tab'))."
            )
            _WUJIANG_BLEU_WARNED = True
        return None
    if not pred_tokens:
        return 0.0  # sentence_bleu warns and returns 0 on an empty hypothesis
    return float(
        sentence_bleu(
            ref_tokens,
            pred_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=SmoothingFunction().method1,
        )
    )


def gold_for_wujiang(q: dict[str, Any]) -> str:
    """Upstream gold = ``QA.final_answer`` (load_dataset.py:17-21):
    ``adversarial_answer`` for cat5, else ``answer``. NO cat3 semicolon
    truncation — neither load_dataset.py nor utils.py splits gold on ``;``
    (verified against the clone), so unlike ``gold_for`` we keep cat3 gold
    whole here."""
    if q.get("category") == 5:
        raw = q.get("adversarial_answer")
        if raw is None:
            raw = q.get("answer", "")
    else:
        raw = q.get("answer")
        if raw is None:
            raw = q.get("adversarial_answer", "")
    return str(raw if raw is not None else "")


def cat5_options(question: str, gold: str) -> tuple[str, str]:
    """Deterministic 2-option order for the cat5 MCQ. Upstream randomizes the
    order with an UNSEEDED ``random.random() < 0.5`` (test_advanced.py:149-154),
    making re-runs non-reproducible. We derive the coin from an md5 of the
    question so the option order is byte-stable across runs (the intentional,
    documented deviation): even coin -> (distractor, gold); odd -> (gold,
    distractor). Options are the gold answer and the literal "Not mentioned in
    the conversation" distractor, exactly as upstream."""
    coin = int(hashlib.md5(question.encode("utf-8")).hexdigest(), 16) & 1
    if coin == 0:
        return (CAT5_NOT_MENTIONED, gold)
    return (gold, CAT5_NOT_MENTIONED)


def resolve_cat5_reply(reply: str, options: tuple[str, str]) -> str:
    """Map a cat5 MCQ reply back to one option's text. Upstream scores the raw
    reply text directly against gold (test_advanced.py parses the JSON ``answer``
    field then calls ``calculate_metrics``), so the raw reply IS the resolved
    answer. We additionally resolve a bare ``(a)``/``(b)``/``a``/``b`` letter to
    the corresponding option so a model that replies by letter is scored on the
    option it actually chose rather than on the literal letter."""
    r = reply.strip()
    low = r.lower().strip("() .")
    if low in ("a", "opt_a", "option a", "first"):
        return options[0]
    if low in ("b", "opt_b", "option b", "second"):
        return options[1]
    return r


def aggregate_cells(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean F1/BLEU-1 as 0-100 percentages, plus the row count. When any
    row carries a judge verdict, also emits `j_score` (mean of the judged
    rows as a 0-100 percentage) and `j_n` (count of judged rows).

    `bleu1` is OMITTED entirely when no row scored one — `bleu1_wujiang`
    returns None without nltk, and reporting 0.0 (or the `ours` BLEU) for
    an unavailable metric is exactly the mislabeling this path exists to
    stop. A wujiang result JSON with no `bleu1` key means "not measured",
    not "measured as zero".

    Module-level rather than a closure inside `evaluate` because a second
    caller exists: `scripts/repro/reconstruct_from_trace.py` rebuilds a run's
    records from its LLM trace when the run finished measuring but died before
    writing its artifacts. A reconstruction that re-implemented this arithmetic
    would be a second copy of the scoring rule, and "several copies, one of them
    read" is the defect class this project catalogs in others. One function, two
    callers, no chance of drift.
    """
    out = {
        "f1": round(100 * sum(r["f1"] for r in rows) / len(rows), 2),
        "n": len(rows),
    }
    scored_bleu = [r["b1"] for r in rows if r["b1"] is not None]
    if scored_bleu:
        out["bleu1"] = round(100 * sum(scored_bleu) / len(scored_bleu), 2)
        if len(scored_bleu) != len(rows):
            out["bleu1_n"] = len(scored_bleu)
    judged = [r["j"] for r in rows if r["j"] is not None]
    if judged:
        out["j_score"] = round(100 * sum(judged) / len(judged), 2)
        out["j_n"] = len(judged)
    return out


def judge_answer(mem: AgenticMemory, question: str, gold: str, pred: str) -> bool | None:
    """Mem0-style binary LLM judge. Returns True/False, or None if no judge
    client (``mem.structured``) is configured or the call yields nothing."""
    if mem.structured is None:
        return None
    res = mem.structured.call(
        "judge",
        JUDGE_PROMPT.format(question=question, gold=gold, pred=pred),
        JUDGE_SCHEMA,
        required_keys=("label",),
    )
    if not res:
        return None
    return str(res.get("label", "")).strip().upper() == "CORRECT"


# ---------------- pipeline ----------------


def ingest(mem: AgenticMemory, sample: dict[str, Any], max_sessions: int | None = None) -> int:
    """Feeds every turn (up to `max_sessions`) into `mem.add_message` in
    conversation order, then flushes; returns the number of turns ingested.
    Mutates `mem`'s namespace state — not idempotent across repeated calls."""
    n = 0
    for _sess, date, speaker, text in iter_turns(sample, max_sessions):
        mem.add_message(
            f"({date}) {speaker}: {text}",
            role="user",
            meta={"speaker": speaker, "date": date},
        )
        n += 1
    mem.flush()
    return n


def answer(
    mem: AgenticMemory,
    question: str,
    k: int | dict = 10,
    memory_types: tuple[str, ...] = ("episodic",),
    budget_tokens: int = 6000,
    keyword_queries: bool = False,
    category: int | None = None,
    eval_mode: str = "ours",
    gold: str | None = None,
    cat5_temperature: float | None = None,
    capture: dict[str, Any] | None = None,
    persona: dict[str, str] | None = None,
    memoryos_lineage: str = "pypi",
    searcher: Any | None = None,
) -> str:
    """One QA turn: optionally rewrite `question` into keywords (A-Mem's
    upstream eval query style) before `mem.search`, inject the MemoryOS
    profile section unconditionally when `"semantic"` is in `memory_types`,
    then generate. Raises `RuntimeError` if `mem.llm` is unset. Returns `""`
    if the LLM reply is empty after stripping.

    `memoryos_lineage` selects the READ path of one of MemoryOS's two upstream
    lineages, and it is ONE argument rather than several because that is the
    lesson of the defect it replaces: the assistant-knowledge dump and the
    retrieval-queue size were independent knobs pinned at the eval harness's
    values, so the default (pypi) config read with settings no single upstream
    has. Two things move together here:

    - assistant knowledge — `"pypi"` retrieves top-20 through the `semantic`
      channel our search already covers; `"eval"` additionally dumps the whole
      store into every prompt (`main_loco_parse`: `get_assistant_knowledge()`).
    - query keywords — `"eval"` spends an LLM call per question
      (`llm_extract_keywords`) and the result is added to the segment score;
      `"pypi"` deleted that term, so no call and no keyword channel.

    The third setting, `page_recall_cap` (7 vs 10), is construction-time config
    and so lives on `AgmemConfig`; a lineage-faithful run has to set it too. It
    is an argument rather than a constant for the same reason `persona` is: a
    run that mixes lineages cannot be compared to either.

    When `capture` (a mutable dict) is passed, it is filled with the exact
    retrieval detail for this question — the raw `query`, the keyword-rewritten
    `rewritten_query`, `k`, `memory_types`, and `retrieved` (one
    `{id, memory_type, score, text}` per bundle hit, where `text` is the chunk
    actually rendered into context). Purely additive: the return value and the
    non-capturing path are unchanged."""
    # budget default raised 1600->6000 per fidelity audit P0-3: the tight
    # budget structurally penalized long-item methodologies (Nemori).
    query = question
    if keyword_queries and mem.structured is not None:
        keyword_result = mem.structured.call(
            "extract",
            KEYWORD_QUERY_PROMPT.format(question=question),
            KEYWORD_QUERY_SCHEMA,
            required_keys=("keywords",),
        )
        if keyword_result and str(keyword_result.get("keywords", "")).strip():
            query = str(keyword_result["keywords"]).strip()
    # MemoryOS eval lineage: keywords ADD a term to the segment score rather than
    # replacing the query, so this is a separate extraction from A-Mem's above
    # and the two can be on at once without interfering. A dropped call degrades
    # to no keyword term, which is exactly the pypi behaviour.
    query_keywords: set[str] = set()
    if memoryos_lineage == "eval" and mem.structured is not None:
        extracted = mem.structured.call(
            "extract",
            MEMORYOS_KEYWORD_PROMPT.format(question=question),
            MEMORYOS_KEYWORD_SCHEMA,
            required_keys=("keywords",),
        )
        if extracted:
            query_keywords = {
                word.strip().lower()
                for word in str(extracted.get("keywords", "")).split(",")
                if word.strip()
            }
    # `searcher` is either the memory itself or the same memory wrapped in the
    # read-side control policy its config names (`retrieval/planned.py`). Both
    # expose the same `search(...)` shape, so this call site never branches on
    # whether a policy is active — which is the property the first wiring of
    # `policies/retrieval` lacked, when this function assembled the context by
    # hand and was therefore the only entry point that could reach one.
    agent_metrics: dict[str, Any] = {}
    bundle = (searcher if searcher is not None else searcher_for(mem)).search(
        query,
        memory_types=memory_types,
        k=k,
        query_keywords=query_keywords,
        metrics=agent_metrics,
    )
    if capture is not None:
        capture["agent"] = agent_metrics
    if capture is not None:
        capture["query"] = question
        capture["rewritten_query"] = query if query != question else None
        capture["k"] = k
        capture["memory_types"] = list(memory_types)
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
    context = bundle.render(budget_tokens=budget_tokens) or "(no memories found)"
    # MemoryOS injects the user profile UNCONDITIONALLY (upstream eval puts
    # the whole profile doc in every QA prompt — round-5 memoryos §3). Only
    # organizers that emit kind="profile" items produce this section.
    #
    # That is now ONE document, not a bulleted fact list: MemoryOS's LPM keeps a
    # single evolving profile replaced by `update_user_profile(merge=False)`, so
    # it is injected verbatim. The old `- {fact}` bullets and the `[-100:]` cap
    # belonged to the append-only profile this replaced; a cap on a single
    # document would just have been a truncation with no upstream counterpart.
    # Several documents can only appear if two profile-producing organizers are
    # active at once, which no config does — they are joined rather than dropped.
    #
    # This unconditional injection is the profile's ONLY channel now (round-12
    # finding 14): the organizer writes the document with an empty
    # `embedding_text`, keeping it out of the vector index, so it can no longer
    # also win slots in the k of the `semantic` retrieval above and appear
    # twice in one prompt — upstream has the single channel too.
    if "semantic" in memory_types:
        semantic_items = mem.doc_store.list_items("semantic", namespace=mem.namespace)
        profile = [d.get("content", "") for d in semantic_items if d.get("kind") == "profile"]
        if profile:
            context = "User Profile:\n" + "\n\n".join(profile) + "\n\n" + context
        # Assistant knowledge, in FULL — the EVAL lineage only, hence the gate.
        # Upstream's two lineages differ here: the harness that produced the
        # paper's LoCoMo numbers injects all of it
        # (`main_loco_parse.generate_system_response_with_meta`:
        # `long_mem.get_assistant_knowledge()`), while the pypi library searches
        # top-20 (`Retriever._retrieve_assistant_knowledge`) — which the
        # `semantic` retrieval channel already approximates, since these items
        # live in that type and are indexed. Running the section unconditionally,
        # as this did, gave a pypi-lineage run BOTH channels: top-20 retrieval
        # plus the eval lineage's full dump. Same class of mixing as
        # `page_recall_cap`, and `MEMORYOS_PRESETS` exists to prevent it.
        assistant_knowledge = [
            d.get("content", "")
            for d in semantic_items
            if d.get("kind") == "assistant_knowledge" and d.get("content")
        ]
        if assistant_knowledge and memoryos_lineage == "eval":
            context = (
                "Assistant Knowledge:\n"
                + "\n".join(f"- {line}" for line in assistant_knowledge)
                + "\n\n"
                + context
            )
    # The recent-turn window some methodologies keep OUTSIDE retrieval and inject
    # verbatim on every question (MemoryOS's STM — upstream `get_response` builds
    # `history_text` from `short_term_memory.get_all()`). It leads the context
    # because it is the most recent thing said, and upstream puts it first too.
    recent = "\n".join(text for text in (org.recent_context() for org in mem.organizers) if text)
    if recent:
        context = "Recent conversation:\n" + recent + "\n\n" + context
    if mem.llm is None:
        raise RuntimeError("generate role LLM required for LoCoMo QA")
    # wujiang cat5: 2-option MCQ (gold vs "Not mentioned") in a deterministic
    # order at the upstream cat5 temperature (0.5). Every other case keeps the
    # `ours` prompts unchanged.
    wujiang_cat5 = eval_mode == "wujiang" and category == 5
    if wujiang_cat5:
        options = cat5_options(question, str(gold if gold is not None else ""))
        prompt_text = CAT5_MCQ_PROMPT.format(
            context=context, question=question, opt_a=options[0], opt_b=options[1]
        )
    else:
        prompt = ANSWER_PROMPT_NO_ABSTAIN if category == 5 else ANSWER_PROMPT
        prompt_text = prompt.format(context=context, question=question)
    chat_overrides: dict[str, Any] = {}
    if wujiang_cat5 and cat5_temperature is not None:
        chat_overrides["temperature"] = cat5_temperature
    messages = [{"role": "user", "content": prompt_text}]
    if persona:
        # Upstream puts the assistant's knowledge in the SYSTEM turn as the
        # role-played character's traits, not in the retrieved context.
        messages.insert(
            0,
            {
                "role": "system",
                "content": PERSONA_SYSTEM_PROMPT.format(
                    speaker_a=persona.get("speaker_a", "the user"),
                    speaker_b=persona.get("speaker_b", "the assistant"),
                    assistant_knowledge=persona.get("assistant_knowledge")
                    or "- No assistant knowledge recorded yet.",
                ),
            },
        )
    reply = mem.llm.chat("generate", messages, **chat_overrides)
    reply = reply.strip().splitlines()[0] if reply.strip() else ""
    if wujiang_cat5:
        return resolve_cat5_reply(reply, options)
    return reply


def evaluate(
    mem: AgenticMemory,
    questions: list[dict[str, Any]],
    k: int | dict = 10,
    memory_types: tuple[str, ...] = ("episodic",),
    budget_tokens: int = 6000,
    keyword_queries: bool = False,
    judge: bool = False,
    eval_mode: str = "ours",
    cat5_temperature: float | None = None,
    capture_retrieval: bool = False,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
    persona: dict[str, str] | None = None,
    memoryos_lineage: str = "pypi",
    searcher: Any | None = None,
) -> dict[str, Any]:
    """Runs `answer` over every question and aggregates F1/BLEU-1 overall and
    per category; `progress(i, total)` (1-indexed `i`) fires after each
    question if given. `records` in the result carries one row per question
    for error inspection, not just the aggregates. When `judge=True`, a
    Mem0-style binary LLM judge (`judge_answer`) also scores cat 1-4 (cat5
    adversarial is excluded, matching Mem0), adding `j_score`/`j_n` to each
    aggregate bucket that has judged rows and a `j` bool to those `records`.

    ``persona`` turns on MemoryOS's role-play system turn
    (``PERSONA_SYSTEM_PROMPT``) with ``{"speaker_a", "speaker_b"}`` and, when the
    caller wants upstream's exact framing, ``"assistant_knowledge"``. It is off
    by default because it replaces the shared answer framing for one
    methodology — see that constant. ``memoryos_lineage`` selects which of
    MemoryOS's two upstream read paths runs and defaults to the pypi one; see
    ``answer``.

    ``eval_mode="wujiang"`` swaps in the WujiangXu/A-Mem faithful path: gold via
    ``gold_for_wujiang`` (no cat3 truncation), F1 via ``token_f1_wujiang``
    (set-based, no stemming), BLEU-1 via ``bleu1_wujiang`` (nltk word_tokenize +
    sentence_bleu, omitted when nltk is absent), and the cat5 2-option MCQ
    generation. Upstream ``utils.py`` has NO LLM judge, so ``judge`` is forced
    off in this mode.

    When ``capture_retrieval`` is set, each record additionally carries a
    ``retrieval`` field (the raw+rewritten query and the retrieved chunks with
    their text) so the durable records sidecar preserves exactly what went into
    context for every question. Off by default — the non-capturing path and the
    record schema are otherwise unchanged.

    ``workers`` > 1 answers/scores questions concurrently: each question is
    independent, results are reassembled in the original question order, and the
    aggregates are IDENTICAL to the sequential path (``workers=1``, the default) —
    only wall-clock and the interleaving of trace lines differ. This is a
    throughput knob for the write-once/read-sweep eval passes, never a fidelity
    change.

    That identity holds because every store read path is lock-guarded AND the
    active organizers make the read path side-effect-free. The second half is a
    property of the METHODOLOGY, not of this function: A-Mem's ``on_retrieval``
    is the base no-op, but MemoryOS bumps ``n_visit``/``last_access`` in
    ``self._heat`` and G-Memory accumulates ``self._served`` — plain dict/set
    mutations from every pool thread, so their read-path state (and any heat
    that later depends on it) becomes worker-count dependent. The claim was
    previously stated unconditionally with "A-Mem's on_retrieval is a no-op" as
    the reason, which reads as a guarantee for organizers it was never checked
    against; a run that trips this now says so instead."""
    wujiang = eval_mode == "wujiang"
    if wujiang:
        judge = False  # upstream utils.calculate_metrics has no J-score judge
    if workers > 1:
        feedback_organizers = [
            org.name for org in mem.organizers if base_overrides(org, "on_retrieval")
        ]
        if feedback_organizers:
            logger.warning(
                "locomo: workers=%d with read->write organizers %s — their on_retrieval "
                "mutates in-memory state per hit, so results are no longer bit-identical "
                "to workers=1. Use workers=1 for measured runs of these methodologies.",
                workers,
                feedback_organizers,
            )

    def _answer_and_score(q: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Answer + score ONE question. Returns (cat_name, per_cat_cell, record).
        Pure w.r.t. shared state — the memory read path and LLM client are all
        thread-safe — so it is safe to run concurrently for `workers>1` and gives
        the same cells/records as the sequential call."""
        gold = gold_for_wujiang(q) if wujiang else gold_for(q)
        cat_num = q.get("category")
        capture: dict[str, Any] | None = {} if capture_retrieval else None
        pred = answer(
            mem,
            q["question"],
            k=k,
            memory_types=memory_types,
            budget_tokens=budget_tokens,
            keyword_queries=keyword_queries,
            category=cat_num,
            eval_mode=eval_mode,
            gold=gold,
            cat5_temperature=cat5_temperature,
            capture=capture,
            persona=persona,
            memoryos_lineage=memoryos_lineage,
            searcher=searcher,
        )
        if wujiang:
            # BLEU must switch with F1: upstream's BLEU tokenizer is nltk
            # word_tokenize, not `normalize` (bleu1_wujiang). Scoring F1 the
            # upstream way while scoring BLEU the `ours` way produced a hybrid
            # number under a `wujiang` label — the defect this pairing fixes.
            f1, b1 = token_f1_wujiang(pred, gold), bleu1_wujiang(pred, gold)
        else:
            f1, b1 = token_f1(pred, gold), bleu1(pred, gold)
        j = (
            judge_answer(mem, q["question"], gold, pred)
            if judge and cat_num in (1, 2, 3, 4)
            else None
        )
        cat = CATEGORY_NAMES.get(cat_num, "?")
        cell = {"f1": f1, "b1": b1, "j": j}
        row = {
            "q": q["question"],
            "gold": gold,
            "pred": pred,
            "cat": cat,
            "f1": round(f1, 3),
        }
        if j is not None:
            row["j"] = j
        if capture is not None:
            row["retrieval"] = capture
        return cat, cell, row

    # Compute one (cat, cell, row) triple per question, preserving question order.
    # workers>1 dispatches the independent QA turns to a thread pool but writes
    # results back by index, so downstream aggregation is order- and
    # worker-count-invariant (asserted by test_repro_artifacts).
    triples: list[tuple[str, dict[str, Any], dict[str, Any]] | None] = [None] * len(questions)
    if workers > 1 and len(questions) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_answer_and_score, q): i for i, q in enumerate(questions)}
            done = 0
            for fut in as_completed(futs):
                triples[futs[fut]] = fut.result()
                done += 1
                if progress:
                    progress(done, len(questions))
    else:
        for i, q in enumerate(questions):
            triples[i] = _answer_and_score(q)
            if progress:
                progress(i + 1, len(questions))

    per_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records = []
    for cat, cell, row in triples:  # type: ignore[misc]
        per_cat[cat].append(cell)
        records.append(row)

    all_rows = [r for rows in per_cat.values() for r in rows]
    return {
        "overall": aggregate_cells(all_rows) if all_rows else {},
        "by_category": {cat: aggregate_cells(rows) for cat, rows in sorted(per_cat.items())},
        "records": records,
    }
