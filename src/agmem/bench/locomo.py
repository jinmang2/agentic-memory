"""LoCoMo benchmark pipeline (snap-research/locomo, 10 conversations).

Metrics are string-based (token F1 + BLEU-1, SQuAD-style normalization) so
no judge LLM is needed — the cheapest reproduction entry point (docs/06
Phase 2). Categories: 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop,
5=adversarial (gold in ``adversarial_answer``).
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from agmem.memory import AgenticMemory

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain",
                  4: "single-hop", 5: "adversarial"}

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


# ---------------- data loading ----------------

def load_locomo(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())


def iter_turns(sample: dict[str, Any], max_sessions: int | None = None):
    """Yield (session_no, date_str, speaker, text) in conversation order."""
    conv = sample["conversation"]
    numbers = sorted(
        int(m.group(1)) for k in conv
        if (m := re.fullmatch(r"session_(\d+)", k)) and isinstance(conv[k], list)
    )
    for n in numbers[: max_sessions or len(numbers)]:
        date = conv.get(f"session_{n}_date_time", "")
        for turn in conv[f"session_{n}"]:
            yield n, date, turn.get("speaker", "?"), turn.get("text", "")


def evidence_sessions(q: dict[str, Any]) -> set[int]:
    out = set()
    for ev in q.get("evidence", []):
        if m := re.match(r"D(\d+):", str(ev)):
            out.add(int(m.group(1)))
    return out


def select_questions(sample: dict[str, Any], max_sessions: int | None = None,
                     categories: tuple[int, ...] = (1, 2, 3, 4, 5),
                     limit: int | None = None) -> list[dict[str, Any]]:
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

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(text: str) -> list[str]:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _ARTICLES.sub(" ", text)
    return text.split()


def token_f1(pred: str, gold: str) -> float:
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
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    precision = sum(common.values()) / len(p)
    brevity = 1.0 if len(p) >= len(g) else pow(2.718281828, 1 - len(g) / len(p))
    return brevity * precision


# ---------------- pipeline ----------------

def ingest(mem: AgenticMemory, sample: dict[str, Any],
           max_sessions: int | None = None) -> int:
    n = 0
    for _sess, date, speaker, text in iter_turns(sample, max_sessions):
        mem.add_message(f"({date}) {speaker}: {text}", role="user",
                        meta={"speaker": speaker, "date": date})
        n += 1
    mem.flush()
    return n


def answer(mem: AgenticMemory, question: str, k: int | dict = 10,
           memory_types: tuple[str, ...] = ("episodic",),
           budget_tokens: int = 6000, keyword_queries: bool = False) -> str:
    # budget default raised 1600->6000 per fidelity audit P0-3: the tight
    # budget structurally penalized long-item methodologies (Nemori).
    query = question
    if keyword_queries and mem.structured is not None:
        kw = mem.structured.call(
            "extract", KEYWORD_QUERY_PROMPT.format(question=question),
            KEYWORD_QUERY_SCHEMA, required_keys=("keywords",))
        if kw and str(kw.get("keywords", "")).strip():
            query = str(kw["keywords"]).strip()
    bundle = mem.search(query, memory_types=memory_types, k=k)
    context = bundle.render(budget_tokens=budget_tokens) or "(no memories found)"
    # MemoryOS injects the user profile UNCONDITIONALLY (upstream eval puts
    # the whole profile doc in every QA prompt — round-5 memoryos §3). Only
    # organizers that emit kind="profile" facts produce this section.
    if "semantic" in memory_types:
        profile = [d.get("content", "") for d in
                   mem.doc.list_items("semantic", namespace=mem.namespace)
                   if d.get("kind") == "profile"][-100:]  # upstream KB cap=100
        if profile:
            context = ("User Profile:\n" + "\n".join(f"- {p}" for p in profile)
                       + "\n\n" + context)
    if mem.llm is None:
        raise RuntimeError("generate role LLM required for LoCoMo QA")
    reply = mem.llm.chat("generate", [
        {"role": "user", "content": ANSWER_PROMPT.format(context=context,
                                                         question=question)},
    ])
    return reply.strip().splitlines()[0] if reply.strip() else ""


def evaluate(mem: AgenticMemory, questions: list[dict[str, Any]],
             k: int | dict = 10,
             memory_types: tuple[str, ...] = ("episodic",),
             budget_tokens: int = 6000, keyword_queries: bool = False,
             progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    per_cat: dict[str, list[tuple[float, float]]] = defaultdict(list)
    records = []
    for i, q in enumerate(questions):
        gold = q.get("answer") or q.get("adversarial_answer") or ""
        pred = answer(mem, q["question"], k=k, memory_types=memory_types,
                      budget_tokens=budget_tokens, keyword_queries=keyword_queries)
        f1, b1 = token_f1(pred, str(gold)), bleu1(pred, str(gold))
        cat = CATEGORY_NAMES.get(q.get("category"), "?")
        per_cat[cat].append((f1, b1))
        records.append({"q": q["question"], "gold": str(gold), "pred": pred,
                        "cat": cat, "f1": round(f1, 3)})
        if progress:
            progress(i + 1, len(questions))

    def agg(pairs: list[tuple[float, float]]) -> dict[str, float]:
        return {
            "f1": round(100 * sum(p[0] for p in pairs) / len(pairs), 2),
            "bleu1": round(100 * sum(p[1] for p in pairs) / len(pairs), 2),
            "n": len(pairs),
        }

    all_pairs = [p for pairs in per_cat.values() for p in pairs]
    return {
        "overall": agg(all_pairs) if all_pairs else {},
        "by_category": {cat: agg(pairs) for cat, pairs in sorted(per_cat.items())},
        "records": records,
    }
