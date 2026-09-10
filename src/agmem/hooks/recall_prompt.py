"""UserPromptSubmit hook: put what memory has to say about THIS prompt in front of the model.

Wire as::

    {"hooks": {"UserPromptSubmit": [{"hooks": [
      {"type": "command", "command": "python -m agmem.hooks.recall_prompt", "timeout": 5},
      {"type": "command", "command": "python -m agmem.hooks.capture", "async": true}
    ]}]}}

Order matters: recall before capture, so a prompt is not served back to itself.

WHY THIS HOOK EXISTS. `recall` (SessionStart) can only offer recency, because
it has no query. This one has the best query there is — the user's prompt —
and asks the daemon for the top-k episodes by the vector path, the same search
the MCP `search_memory` tool runs. This repo's LongMemEval result is the
reason to bother: on the `_s` set, retrieval through the read path moved the
score by +21.2pp with zero write-side LLM calls (docs/21), and the survey in
docs/research/product-memory-landscape.md found the shipping products that
inject at all inject per prompt, not per session.

WHAT IT NEVER DOES. Load the embedder. If the daemon is not answering, this
hook answers from the doc store alone — BM25 over runbooks and past user
turns, the keyword path a `grep` would give — and exits 0. `capture` is the
hook responsible for asking the daemon to start. Blocking a turn for ~10 s of
model loading would be worse than a turn without memory.

WHY THE FALLBACK. Dogfooding (docs/23 §8) found the gap: a hook-spawned daemon
takes ~20 s to come up (measured 2026-09-05: model load + backfill), and the
SessionStart hook is what spawns it after an idle exit, so every prompt typed
in the first 20 s of a session got no memory at all — silently, since the
hook exits 0 either way. The doc store opens in 0.18 s and its FTS index is
already there; a keyword match is a worse ranking than the vector path, and
far better than nothing.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from typing import TypedDict

from agmem.control import is_auto_injection_eligible, memory_display_text
from agmem.core.origin import item_cwd, same_project
from agmem.env import resolve_namespace
from agmem.hooks import daemon as daemon_client
from agmem.hooks import emit_context, fail_open, open_doc_store, read_event
from agmem.hooks.capture import prompt_of


class RecallItem(TypedDict):
    id: str | None
    memory_type: str
    score: float
    timestamp: str
    text: str


RequestPayload = dict[str, str | int]

K = 5
MIN_SCORE = 0.0
MAX_CHARS = 2000
# Bound daemon-less BM25 widening to keep the blocking hook within its latency budget.
MAX_FALLBACK_LIMIT = 4096
HEADER = (
    "Memory relevant to this prompt (agmem, semantic search over past turns, "
    "best match first). These are retrieved, not verified: treat them as leads "
    "and check the code or the user when it matters."
)
FALLBACK_HEADER = (
    "Memory relevant to this prompt (agmem, keyword match over the local store: the "
    "memory daemon did not answer this search, so this is BM25 over runbooks and past turns "
    "rather than the vector search; runbooks first, then best match first). These are "
    "retrieved, not verified: treat them as leads and check the code or the user when "
    "it matters."
)


def fallback_items(
    store, namespace: str, query: str, k: int, project: str | None
) -> list[RecallItem]:
    """The daemon-less answer: BM25 over runbooks, then over past user turns,
    each gated by project the way the daemon path is. Rendered by `render`, so
    the shape matches what `/hooks/recall` returns."""
    if k <= 0:
        return []

    limit = min(k, MAX_FALLBACK_LIMIT)
    items: list[RecallItem] = []
    while True:
        rb_scores: dict[str, float] = dict(
            store.search_lexical_items(query, "runbooks", k=limit, namespace=namespace)
        )
        items = []
        for d in store.get_items(list(rb_scores), "runbooks"):
            if not is_auto_injection_eligible(d):
                continue
            if project and not same_project(item_cwd(d), project):
                continue
            items.append(
                {
                    "id": str(d.get("id")) if d.get("id") is not None else None,
                    "memory_type": "runbooks",
                    "score": rb_scores.get(str(d.get("id")), 0.0),
                    "timestamp": str((d.get("origin") or {}).get("ended_at") or ""),
                    "text": memory_display_text(d),
                }
            )
        pruned_items = _prune(query, items)
        if len(pruned_items) >= k:
            return pruned_items[:k]
        if len(rb_scores) < limit or limit >= MAX_FALLBACK_LIMIT:
            break
        limit = min(limit * 2, MAX_FALLBACK_LIMIT)

    turn_k = k - len(pruned_items)
    limit = min(turn_k, MAX_FALLBACK_LIMIT)
    turns: list[RecallItem] = []
    while True:
        ep_scores: dict[str, float] = dict(
            store.search_lexical(query, k=limit, namespace=namespace)
        )
        turns = []
        for ep in store.get_episodes(list(ep_scores)):
            if not is_auto_injection_eligible(ep):
                continue
            if getattr(ep, "role", "user") != "user":
                continue
            if project and not same_project(item_cwd(ep), project):
                continue
            turns.append(
                {
                    "id": ep.id,
                    "memory_type": "episodic",
                    "score": ep_scores.get(ep.id, 0.0),
                    "timestamp": ep.timestamp.isoformat(),
                    "text": ep.content,
                }
            )
        pruned_turns = _prune(query, turns)
        if len(pruned_turns) >= turn_k or len(ep_scores) < limit or limit >= MAX_FALLBACK_LIMIT:
            break
        limit = min(limit * 2, MAX_FALLBACK_LIMIT)

    return (pruned_items + pruned_turns)[:k]


MIN_TOKEN_CHARS = 4
STEM_CHARS = 5
_TOKEN = re.compile(r"\w+", re.UNICODE)


def _stems(text: str) -> set[str]:
    """The words of `text` that count for overlap, cut to a crude stem.

    The coding evaluation (docs/28, 2026-09-10) showed why a whole-word match
    is not enough: the prompt said "corrected" and "reject", the memory said
    "Correction" and "rejected", and the one memory that mattered — the fix
    for a stale field name — was pruned while the stale note survived on the
    word "stale". English inflection mostly lives in the suffix, so the first
    `STEM_CHARS` letters stand in for a stemmer: "corre" covers corrected and
    correction, "rejec" covers reject and rejected. Shorter tokens are kept
    whole, and tokens under `MIN_TOKEN_CHARS` stay out as before."""
    return {t.lower()[:STEM_CHARS] for t in _TOKEN.findall(text) if len(t) >= MIN_TOKEN_CHARS}


def _prune(query: str, group: list[RecallItem]) -> list[RecallItem]:
    """Best first, and only hits that share a real word with the prompt. The
    store's FTS query is an OR over every token, so "how do I use the pnpm
    filter" also matches a turn that merely contains "the"; in a small store
    BM25 barely separates the two (measured 2.9e-6 against 1.0e-6 with two
    turns), so the score is not the filter — token overlap is. Tokens shorter
    than `MIN_TOKEN_CHARS` are the stopwords of this rule; a prompt made only
    of short tokens keeps every hit rather than none. Overlap is on stems
    (`_stems`), not whole words."""
    group = sorted(group, key=lambda it: -float(it["score"]))
    words = _stems(query)
    if not words:
        return group
    return [it for it in group if words & _stems(str(it.get("text") or ""))]


def request_body(event: Mapping[str, str], query: str, k: int) -> RequestPayload:
    """What the daemon is asked: the prompt as the query, and the session's
    cwd so the daemon gates the answer by project (research §6 #9) — memory
    from another repository is not a lead for this one."""
    body: RequestPayload = {"query": query, "k": k}
    body["namespace"] = resolve_namespace()
    cwd = str(event.get("cwd") or "")
    if cwd:
        body["cwd"] = cwd
    return body


def render(items: list[RecallItem], header: str = HEADER) -> str:
    lines = []
    used = 0
    for it in items:
        if float(it.get("score", 0.0)) < MIN_SCORE:
            continue
        text = " ".join(str(it.get("text") or "").split())
        if not text:
            continue
        stamp = str(it.get("timestamp") or "")[:10] or "?"
        line = f"- ({stamp}) {text}"
        if len(line) > 300:
            line = line[:297] + "..."
        if used + len(line) > MAX_CHARS:
            break
        lines.append(line)
        used += len(line)
    if not lines:
        return ""
    return header + "\n" + "\n".join(lines)


def main() -> None:
    try:
        event = read_event()
        query = prompt_of(event)
        if not query.strip():
            sys.exit(0)
        k = int(os.environ.get("AGMEM_RECALL_K") or K)
        if daemon_client.health() is not None:
            # A daemon that answers /health can still fail this one search, and
            # the branch used to have no way back: the raise reached `fail_open`
            # and the turn got no memory and no notice. Measured in the
            # 2026-09-06 dogfood at 3 prompts in 10, all of them the
            # `MAX_KNN_K` refusal. The fallback below is already the answer for
            # a daemon that is not there; a daemon that cannot answer is not a
            # better case.
            reply = None
            try:
                reply = daemon_client.post("/hooks/recall", request_body(event, query, k))
            except daemon_client.DaemonUnavailable:
                reply = None
            if reply is not None:
                emit_context(render(list(reply.get("items") or [])), "UserPromptSubmit")
                sys.exit(0)
        project = str(event.get("cwd") or "") or None
        namespace, store = open_doc_store()
        try:
            items = fallback_items(store, namespace, query, k, project)
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                close()
        emit_context(render(items, FALLBACK_HEADER), "UserPromptSubmit")
    except BaseException as exc:  # every failure path exits 0 — see fail_open
        if isinstance(exc, SystemExit):
            raise
        fail_open(exc)
    sys.exit(0)


if __name__ == "__main__":
    main()
