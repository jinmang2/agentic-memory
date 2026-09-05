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
hook emits nothing and exits 0 — the session already has the recency block
from `recall`, and `capture` is the hook responsible for asking the daemon to
start. Blocking a turn for ~10 s of model loading would be worse than a turn
without memory.
"""

from __future__ import annotations

import os
import sys

from agmem.hooks import daemon as daemon_client
from agmem.hooks import emit_context, fail_open, read_event
from agmem.hooks.capture import prompt_of

K = 5
MIN_SCORE = 0.0
MAX_CHARS = 2000
HEADER = (
    "Memory relevant to this prompt (agmem, semantic search over past turns, "
    "best match first). These are retrieved, not verified: treat them as leads "
    "and check the code or the user when it matters."
)


def request_body(event: dict, query: str, k: int) -> dict:
    """What the daemon is asked: the prompt as the query, and the session's
    cwd so the daemon gates the answer by project (research §6 #9) — memory
    from another repository is not a lead for this one."""
    body: dict = {"query": query, "k": k}
    cwd = str(event.get("cwd") or "")
    if cwd:
        body["cwd"] = cwd
    return body


def render(items: list[dict]) -> str:
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
    return HEADER + "\n" + "\n".join(lines)


def main() -> None:
    try:
        event = read_event()
        query = prompt_of(event)
        if not query.strip() or daemon_client.health() is None:
            sys.exit(0)
        k = int(os.environ.get("AGMEM_RECALL_K") or K)
        reply = daemon_client.post("/hooks/recall", request_body(event, query, k))
        emit_context(render(list(reply.get("items") or [])), "UserPromptSubmit")
    except BaseException as exc:  # every failure path exits 0 — see fail_open
        if isinstance(exc, SystemExit):
            raise
        fail_open(exc)
    sys.exit(0)


if __name__ == "__main__":
    main()
