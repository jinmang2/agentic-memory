"""UserPromptSubmit hook: write the turn to memory as it happens.

Wire as::

    {"hooks": {"UserPromptSubmit": [{"hooks": [
      {"type": "command", "command": "python -m agmem.hooks.capture", "async": true}
    ]}]}}

`async` is the point. Capture produces nothing the model needs this turn, so it
must not sit between the user pressing enter and the model starting — the write
happens in the background and the turn is never held for it.

It emits NO context. A capture hook that also injected something would make
every prompt echo itself back into the transcript, and the recall hook already
owns the memory-to-context direction. One direction per hook.

DETERMINISM IS THE FEATURE. The portfolio note behind this file says capture has
to be deterministic before any of it is worth feeling, and that is a claim about
this hook specifically: the model deciding when to remember is the failure mode
that MCP tools already have. Here the harness decides, on every prompt, whether
or not anyone thought about it.

KNOWN COST, MEASURED 2026-08-08, AND STILL NOT FIXED. Unlike `recall`, this hook
needs the embedder — an episode written without a vector is invisible to every
semantic search, including the MCP server's `search_memory` (verified end to
end: a prompt captured here comes back from that tool). A fresh process costs
10.8 s wall, decomposing into `import torch` 2.1 s, `import
sentence_transformers` 4.4 s, constructing the model 3.7 s. `async: true` keeps
that off the user's turn, but it is still ~11 s of CPU per prompt, and prompts
arriving closer together than that will overlap.

That is down from ~15 s: loading the model cache-first removed a hub revision
check worth ~5 s of the construction (`SentenceTransformerEmbedder._load`).
Worth having, and it does not change the paragraph below — halving a cost that
should not be paid per prompt at all leaves it still paid per prompt.

The honest fix is architectural, not a tweak: either capture through a
long-lived process that loads the model once (the MCP server already is one), or
write the episode immediately and let a batch pass attach vectors later. Both
are real work and neither is done here. What is deliberately NOT done is the
cheap-looking version — writing episodes with no vector and calling it fast —
because that ships a capture hook whose memories never surface in a search,
which would present as working.
"""

from __future__ import annotations

import sys

from agmem.hooks import fail_open, open_memory, read_event

MAX_CHARS = 8000


def prompt_of(event: dict) -> str:
    """The user's text, across the field names the harness has used for it.

    Checked in order rather than assuming one: a payload whose shape moved would
    otherwise capture empty strings forever, and an empty capture is invisible —
    the hook keeps exiting 0 and the store keeps not growing.
    """
    for key in ("prompt", "user_prompt", "message", "content"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def main() -> None:
    try:
        event = read_event()
        text = prompt_of(event)
        if not text.strip():
            sys.exit(0)  # nothing said, nothing to store
        mem = open_memory()
        try:
            mem.add_message(
                text[:MAX_CHARS],
                role="user",
                meta={
                    "source": "claude-code",
                    # Kept so a later reader can group a session's turns without
                    # inferring it from timestamps.
                    "session_id": str(event.get("session_id") or ""),
                },
            )
            mem.flush()
        finally:
            mem.close()
    except BaseException as exc:  # every failure path exits 0 — see fail_open
        if isinstance(exc, SystemExit):
            raise
        fail_open(exc)
    sys.exit(0)


if __name__ == "__main__":
    main()
