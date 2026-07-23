"""LoCoMo benchmark pipeline (snap-research/locomo, 10 conversations).

The primary metrics are string-based (token F1 + BLEU-1, SQuAD-style
normalization with Porter stemming) — the cheapest reproduction entry point
(docs/06 Phase 2). A Mem0-style binary J-score LLM judge is additionally
available (opt-in via ``evaluate(judge=True)``) for cat 1-4. Categories:
1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial (gold in
``adversarial_answer``).
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from agmem.bench._porter import PorterStemmer
from agmem.memory import AgenticMemory

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
_STEMMER = PorterStemmer()


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
    scaled by a brevity penalty when `pred` is shorter than `gold`."""
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
) -> str:
    """One QA turn: optionally rewrite `question` into keywords (A-Mem's
    upstream eval query style) before `mem.search`, inject the MemoryOS
    profile section unconditionally when `"semantic"` is in `memory_types`,
    then generate. Raises `RuntimeError` if `mem.llm` is unset. Returns `""`
    if the LLM reply is empty after stripping."""
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
    bundle = mem.search(query, memory_types=memory_types, k=k)
    context = bundle.render(budget_tokens=budget_tokens) or "(no memories found)"
    # MemoryOS injects the user profile UNCONDITIONALLY (upstream eval puts
    # the whole profile doc in every QA prompt — round-5 memoryos §3). Only
    # organizers that emit kind="profile" facts produce this section.
    if "semantic" in memory_types:
        profile = [
            d.get("content", "")
            for d in mem.doc_store.list_items("semantic", namespace=mem.namespace)
            if d.get("kind") == "profile"
        ][-100:]  # upstream KB cap=100
        if profile:
            context = "User Profile:\n" + "\n".join(f"- {p}" for p in profile) + "\n\n" + context
    if mem.llm is None:
        raise RuntimeError("generate role LLM required for LoCoMo QA")
    prompt = ANSWER_PROMPT_NO_ABSTAIN if category == 5 else ANSWER_PROMPT
    reply = mem.llm.chat(
        "generate",
        [
            {
                "role": "user",
                "content": prompt.format(context=context, question=question),
            },
        ],
    )
    return reply.strip().splitlines()[0] if reply.strip() else ""


def evaluate(
    mem: AgenticMemory,
    questions: list[dict[str, Any]],
    k: int | dict = 10,
    memory_types: tuple[str, ...] = ("episodic",),
    budget_tokens: int = 6000,
    keyword_queries: bool = False,
    judge: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Runs `answer` over every question and aggregates F1/BLEU-1 overall and
    per category; `progress(i, total)` (1-indexed `i`) fires after each
    question if given. `records` in the result carries one row per question
    for error inspection, not just the aggregates. When `judge=True`, a
    Mem0-style binary LLM judge (`judge_answer`) also scores cat 1-4 (cat5
    adversarial is excluded, matching Mem0), adding `j_score`/`j_n` to each
    aggregate bucket that has judged rows and a `j` bool to those `records`."""
    per_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records = []
    for i, q in enumerate(questions):
        gold = gold_for(q)
        cat_num = q.get("category")
        pred = answer(
            mem,
            q["question"],
            k=k,
            memory_types=memory_types,
            budget_tokens=budget_tokens,
            keyword_queries=keyword_queries,
            category=cat_num,
        )
        f1, b1 = token_f1(pred, gold), bleu1(pred, gold)
        j = (
            judge_answer(mem, q["question"], gold, pred)
            if judge and cat_num in (1, 2, 3, 4)
            else None
        )
        cat = CATEGORY_NAMES.get(cat_num, "?")
        per_cat[cat].append({"f1": f1, "b1": b1, "j": j})
        row = {
            "q": q["question"],
            "gold": gold,
            "pred": pred,
            "cat": cat,
            "f1": round(f1, 3),
        }
        if j is not None:
            row["j"] = j
        records.append(row)
        if progress:
            progress(i + 1, len(questions))

    def agg(rows: list[dict[str, Any]]) -> dict[str, float]:
        """Mean F1/BLEU-1 as 0-100 percentages, plus the row count. When any
        row carries a judge verdict, also emits `j_score` (mean of the judged
        rows as a 0-100 percentage) and `j_n` (count of judged rows)."""
        out = {
            "f1": round(100 * sum(r["f1"] for r in rows) / len(rows), 2),
            "bleu1": round(100 * sum(r["b1"] for r in rows) / len(rows), 2),
            "n": len(rows),
        }
        judged = [r["j"] for r in rows if r["j"] is not None]
        if judged:
            out["j_score"] = round(100 * sum(judged) / len(judged), 2)
            out["j_n"] = len(judged)
        return out

    all_rows = [r for rows in per_cat.values() for r in rows]
    return {
        "overall": agg(all_rows) if all_rows else {},
        "by_category": {cat: agg(rows) for cat, rows in sorted(per_cat.items())},
        "records": records,
    }
