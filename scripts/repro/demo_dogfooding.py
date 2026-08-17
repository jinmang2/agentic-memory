"""Demo B — run this repository's own memory layer the way Claude Code runs it, and time it.

The problem is the one every session of this project opened with: a fresh session knows nothing, so
its first minutes go to rediscovering where the work was. That is what the hooks in
`src/agmem/hooks/` exist to remove, and the honest way to show it is not a screenshot of a session —
it is to drive the hook binaries under the same contract the harness uses (a JSON event on stdin, a
JSON object on stdout), then start the MCP server the same way a client starts it, and print what
comes back with the clock on it.

**What is real here and what is staged.** The hooks, the store, the embedder and the MCP server are
the shipped code paths, invoked as subprocesses exactly as Claude Code invokes them — nothing is
stubbed and nothing is mocked. What is staged is the transcript: the prompts fed to the capture hook
are real statements about this project's campaign rather than a captured session, because a real
transcript cannot be published and a synthetic one that pretends otherwise would be the kind of
claim this repository exists to argue against. The demo is about the mechanism and its latency, both
of which are measured.

**Cost: $0, no API key.** Capture is deliberately organizer-free (`hooks.open_memory` passes
`organizers=[]`), so no LLM is called on anyone's keystroke; the only model involved is a local
sentence-transformer for embeddings. Nothing here reaches a paid endpoint.

**The store is a throwaway.** Everything is written under a temp directory, never
`~/.agmem/data` — a demo must not write into the reader's real memory, and it must not read from
ours either, or the output would depend on a machine nobody else has.

**Plumbing is imported, not rewritten.** `scripts/smoke_product_stack.py` already drives this seam
and is the script `docs/05` §2.4 cites for its timings; this demo reuses its hook runner and its MCP
client rather than growing a second copy that could drift from it.

Run:  uv run python scripts/repro/demo_dogfooding.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from demo_svg import ACCENT, BACKGROUND, GRID, INK, MUTED, reflow, text_element, wrap

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SVG = REPO_ROOT / "docs" / "demos" / "assets" / "dogfooding.svg"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "demos" / "dogfooding.md"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from smoke_product_stack import run_hook, search_over_mcp

NAMESPACE = "demo-dogfooding"

# Statements from this project's own campaign, in the order a working session would have produced
# them. They are facts with sources — docs/19 for the ACE arms, docs/17 for the ledger, the plan for
# the demo order — not invented session chatter, because a demo of a memory system is worth nothing
# if what it remembers is fiction.
SESSION_PROMPTS = [
    (
        "The ACE nodedup arm on FiNER cost $8.63 against the online arm's $1.46, and the LLM call "
        "count moved by two: 1,323 to 1,325."
    ),
    (
        "That means the cost of a self-evolving playbook is carried in tokens, not in requests — "
        "the playbook is injected whole into every generator and curator call."
    ),
    (
        "None of the three ACE learning arms separates from the base arm that never learned. The "
        "null survived turning our dedup gate off, which was the last explanation for it."
    ),
    "Track 6 is the demo track. Agreed order is C, D, then B, then A; C and D spend nothing.",
    (
        "The LongMemEval claim-C lane is finished at $3.77. The only remaining spend decision "
        "there is the _s extension at about $21, which is not approved."
    ),
    (
        "Rule for this repo: never restate a measured number without its condition — the arm, the "
        "model, the read path. Every headline in the campaign has at least one."
    ),
]

# Asked after the restart, and deliberately not a keyword match for any single prompt. Note what the
# run then showed: at six episodes the server returns the whole store on a nearly flat ranking, so
# this question demonstrates the ASK path existing, not retrieval working. The page says so.
RESTART_QUESTION = "What did we conclude about where the money goes in a self-evolving playbook?"


@dataclass
class Measurement:
    """One timed step of the demo, kept with enough context to print it without re-deriving it."""

    label: str
    seconds: float
    detail: str = ""
    outputs: list[str] = field(default_factory=list)


def capture_session(env: dict, prompts: list[str]) -> tuple[list[Measurement], float]:
    """Feed each prompt through the real UserPromptSubmit hook. Returns per-prompt timings.

    Every call is kept rather than averaged away, because the shape is the finding: each hook is a
    fresh process, so each one reloads the embedder and none of them is ever "warm". There is no
    cold-start-then-fast curve here — the whole run sits at tens of seconds per keystroke, which is
    precisely why `docs/05` §2.4 wires capture as `async: true` and why the page reports the range
    rather than a single number that would imply one call is representative.
    """
    measurements = []
    for index, prompt in enumerate(prompts):
        seconds, completed = run_hook(
            "agmem.hooks.capture",
            {
                "session_id": "demo-dogfooding",
                "prompt": prompt,
                "hook_event_name": "UserPromptSubmit",
            },
            env,
        )
        if completed.returncode != 0:
            raise SystemExit(f"capture hook failed on prompt {index}: {completed.stderr[-500:]}")
        measurements.append(Measurement(f"capture #{index + 1}", seconds, prompt))
    return measurements, sum(m.seconds for m in measurements)


def restart_session(env: dict) -> Measurement:
    """Run the SessionStart hook exactly as the harness does, and keep what it injects verbatim.

    The hook's contract is that its `hookSpecificOutput.additionalContext` is what reaches the
    model. That string is the demo's actual product, so it is parsed out of the hook's stdout rather
    than reconstructed from the store — reconstructing it would demo a function nobody calls.
    """
    seconds, completed = run_hook(
        "agmem.hooks.recall",
        {"session_id": "demo-dogfooding-restarted", "hook_event_name": "SessionStart"},
        env,
    )
    if completed.returncode != 0:
        raise SystemExit(f"recall hook failed: {completed.stderr[-500:]}")
    injected = ""
    if completed.stdout.strip():
        payload = json.loads(completed.stdout)
        injected = payload.get("hookSpecificOutput", {}).get("additionalContext", "")
    return Measurement("SessionStart recall", seconds, "", injected.splitlines())


def ask_over_mcp(
    data_dir: Path, env: dict
) -> tuple[Measurement, Measurement, set[str], list[dict]]:
    """Start the MCP server as a client would and ask one question through `search_memory`.

    The server's payload is returned parsed rather than as a blob. The scores are the interesting
    part and they are what decides whether the page may claim the vector path did anything — a demo
    that pastes the raw JSON is asking the reader to grade it, which is how a flat ranking gets
    published as retrieval.
    """
    started = time.perf_counter()
    handshake_s, rendered, tools = asyncio.run(
        search_over_mcp_with_query(NAMESPACE, data_dir, env, RESTART_QUESTION)
    )
    total = time.perf_counter() - started
    handshake = Measurement("MCP handshake", handshake_s, f"{len(tools)} tools")

    payload = json.loads(rendered)
    items = payload.get("items", [])
    context_lines = [
        line.lstrip("- ").strip()
        for line in payload.get("context", "").splitlines()
        if line.startswith("- ")
    ]
    # The server returns items and context as parallel lists in served order, so the text of ranked
    # item i is context line i. Zipped rather than assumed equal length: a shorter list must
    # truncate the pairing, not raise on a payload shape this demo does not control.
    ranked = [
        {"score": item.get("score"), "memory_type": item.get("memory_type"), "text": text}
        for item, text in zip(items, context_lines, strict=False)
    ]
    answer = Measurement(
        "MCP search_memory", total - handshake_s, RESTART_QUESTION, rendered.splitlines()
    )
    return handshake, answer, tools, ranked


async def search_over_mcp_with_query(namespace: str, data_dir: Path, env: dict, query: str):
    """`smoke_product_stack.search_over_mcp` with the query swapped, without copying its client.

    The smoke pins its own needle query as a module constant because its job is a pass/fail seam
    check. Rebinding that constant for the duration of the call is uglier than a parameter and
    strictly better than a second stdio client implementation that could drift from the one
    `docs/05` §2.4 cites.
    """
    import smoke_product_stack

    original = smoke_product_stack.QUERY
    smoke_product_stack.QUERY = query
    try:
        return await search_over_mcp(namespace, data_dir, env)
    finally:
        smoke_product_stack.QUERY = original


# ------------------------------------------------------------------------------------------- SVG

WIDTH, HEIGHT = 1000, 380
CHART_LEFT, CHART_RIGHT = 250, 820
CHART_TOP = 118
NOTE_COLUMN = CHART_RIGHT + 14
ROW_HEIGHT = 44

# Measured on this machine, 2026-08-08, and quoted from `hooks/__init__.py`'s own docstring rather
# than re-derived here: opening a full AgenticMemory handle costs 9.1 s, essentially all of it the
# embedder reaching its weights, against 0.18 s for the doc store alone. Before that loader became
# cache-first the same measurement read 15.1 s. Both are drawn because the design decision is only
# legible next to the number that forced it.
FULL_HANDLE_SECONDS = 9.1
FULL_HANDLE_BEFORE_CACHE_SECONDS = 15.1

# The `timeout` in the wiring snippet `docs/05` §2.4 ships. SessionStart blocks on it.
SESSION_START_TIMEOUT_SECONDS = 10.0


def render_svg(recall: Measurement) -> str:
    """A latency chart whose whole argument is which bars cross the SessionStart timeout line."""
    bars = [
        ("SessionStart recall (doc store only)", recall.seconds, "#059669", "what ships"),
        (
            "…if it opened a full memory handle",
            FULL_HANDLE_SECONDS,
            ACCENT,
            "measured 2026-08-08",
        ),
        (
            "…before the loader went cache-first",
            FULL_HANDLE_BEFORE_CACHE_SECONDS,
            ACCENT,
            "killed by the timeout",
        ),
    ]
    # Capture is deliberately NOT on this axis. It is wired async and has no deadline, so at tens of
    # seconds it would set the scale and push the timeout line — the only thing this chart is about —
    # into the left margin. Its numbers are in the page's table, where the comparison is honest.
    span = FULL_HANDLE_BEFORE_CACHE_SECONDS * 1.2

    def x_of(seconds: float) -> float:
        return CHART_LEFT + (CHART_RIGHT - CHART_LEFT) * seconds / span

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'width="{WIDTH}" height="{HEIGHT}" role="img" '
            f'aria-label="Hook latencies against the SessionStart timeout">'
        ),
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>',
        text_element(
            48,
            34,
            "Why the recall hook opens the doc store and nothing else",
            size=17,
            weight="700",
        ),
        text_element(
            48,
            56,
            "SessionStart blocks on this hook. A memory layer that misses its timeout presents as "
            "no memory, not as an error.",
            size=11.5,
            fill=MUTED,
        ),
    ]

    for index, (label, seconds, color, note) in enumerate(bars):
        y = CHART_TOP + index * ROW_HEIGHT
        parts.append(text_element(CHART_LEFT - 12, y + 14, label, size=11.5, anchor="end"))
        width = max(x_of(seconds) - CHART_LEFT, 1.5)
        parts.append(
            f'<rect x="{CHART_LEFT}" y="{y}" width="{width:.1f}" height="20" fill="{color}" rx="2"/>'
        )
        # A long bar carries its value INSIDE. Outside, the 9.1 s label lands on the dashed timeout
        # line, which is the one mark on this chart that has to stay unambiguous.
        inside = width > 90
        parts.append(
            text_element(
                CHART_LEFT + width + (-10 if inside else 8),
                y + 14,
                f"{seconds:.2f}s",
                size=11.5,
                fill=BACKGROUND if inside else color,
                anchor="end" if inside else "start",
                weight="600",
            )
        )
        # Notes sit in a fixed column past the plot area rather than trailing their bar: trailing,
        # the 9.1 s note runs straight through the dashed timeout line.
        if note:
            parts.append(text_element(NOTE_COLUMN, y + 14, note, size=10.5, fill=MUTED))

    timeout_x = x_of(SESSION_START_TIMEOUT_SECONDS)
    bottom = CHART_TOP + len(bars) * ROW_HEIGHT
    parts.append(
        f'<line x1="{timeout_x:.1f}" y1="{CHART_TOP - 14}" x2="{timeout_x:.1f}" y2="{bottom + 4}" '
        f'stroke="{INK}" stroke-width="1.4" stroke-dasharray="5 4"/>'
    )
    parts.append(
        text_element(
            timeout_x,
            CHART_TOP - 22,
            f"SessionStart timeout — {SESSION_START_TIMEOUT_SECONDS:.0f}s",
            size=11,
            weight="600",
            anchor="middle",
        )
    )
    for tick in range(0, int(span) + 1, 2):
        x = x_of(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{bottom + 4}" x2="{x:.1f}" y2="{bottom + 9}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            text_element(x, bottom + 23, f"{tick}s", size=10.5, fill=MUTED, anchor="middle")
        )

    caption = (
        "The green bar is measured by scripts/repro/demo_dogfooding.py, driving the shipped hook "
        "binary under the Claude Code hook contract on a throwaway store; absolute seconds are "
        "load-dependent and the ratio is not. The two red bars are quoted from "
        "src/agmem/hooks/__init__.py, which records why the hook was narrowed to the doc store. The "
        "capture hook is deliberately off this axis: it is wired async and has no deadline, so its "
        "tens of seconds would set the scale and hide the line this chart is about. "
        "No API key, no model call, $0."
    )
    for index, line in enumerate(wrap(caption, int((CHART_RIGHT - 48) / 5.4))):
        parts.append(text_element(48, HEIGHT - 76 + index * 15, line, size=11, fill=MUTED))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# -------------------------------------------------------------------------------------- markdown


def render_markdown(
    captures: list[Measurement],
    recall: Measurement,
    handshake: Measurement,
    answer: Measurement,
    tools: set[str],
    ranked: list[dict],
    svg_path: Path,
    markdown_path: Path,
) -> str:
    """The page a reviewer reads: the mechanism, the latency, and the defect that shaped it."""
    relative_svg = Path(svg_path).relative_to(markdown_path.parent)
    capture_times = [m.seconds for m in captures]
    capture_mean = sum(capture_times) / len(capture_times)
    # os.getloadavg is POSIX-only; a page generated on a host without it says so rather than
    # printing a zero that would read as an idle machine.
    load_average = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
    timeout_margin = SESSION_START_TIMEOUT_SECONDS / recall.seconds
    capture_ratio = capture_mean / recall.seconds
    injected = "\n".join(recall.outputs)

    scores = [item["score"] for item in ranked if isinstance(item.get("score"), (int, float))]
    score_spread = (max(scores) - min(scores)) if scores else 0.0
    ranked_rows = "\n".join(
        f"| {index + 1} | {item['score']:.4f} | {item['text'][:110]} |"
        for index, item in enumerate(ranked)
    )

    prompt_rows = "\n".join(f"{index + 1}. {m.detail}" for index, m in enumerate(captures))
    timing_rows = "\n".join(
        f"| {row} |"
        for row in (
            (
                f"`UserPromptSubmit` capture | {capture_mean:.2f}s mean "
                f"({min(capture_times):.2f}–{max(capture_times):.2f}s over {len(captures)}) | "
                f"every call is a fresh process, so every call reloads the embedder and none is "
                f"ever warm — which is why it is wired `async: true` and nobody waits on it"
            ),
            (
                f"`SessionStart` recall | **{recall.seconds:.2f}s** | blocking — this is the one "
                f"with a deadline"
            ),
            f"MCP server handshake | {handshake.seconds:.2f}s | {len(tools)} tools registered",
            f"MCP `search_memory` | {answer.seconds:.2f}s | one query over the same store",
        )
    )

    return f"""# Demo B — the memory layer, running inside the tool that needed it

Every session of this project used to open the same way: the model knew nothing, so its first
minutes went to rediscovering where the work had stopped. That is the problem this repository is
built around, and this page runs the fix against itself.

**No API key, no model call, $0.** Capture is deliberately organizer-free — `hooks.open_memory`
passes `organizers=[]`, so nothing reaches a paid endpoint on somebody's keystroke. The only model
involved is a local sentence-transformer producing embeddings.

```
uv run python scripts/repro/demo_dogfooding.py
```

![Hook latencies against the SessionStart timeout]({relative_svg})

## What was run

The hooks are subprocesses invoked under the Claude Code hook contract — a JSON event on stdin, a
JSON object on stdout — which is exactly how the harness invokes them. Nothing is stubbed. The
store is a throwaway temp directory, never `~/.agmem/data`: a demo must not write into the reader's
real memory, and reading from ours would make the output depend on a machine nobody else has.

**Act one — a session that learns things.** {len(captures)} prompts through the real
`UserPromptSubmit` hook:

{prompt_rows}

These are statements about this project's own campaign, with sources — `docs/19-ace-finer.md` for
the ACE arms, the demo plan for the track order. They are not a captured transcript, because a real
transcript cannot be published and a synthetic one presented as real would be exactly the kind of
claim this repository exists to argue against. What is measured is the mechanism and its latency;
what is staged is the content.

**Act two — the session ends. A new one starts.** The `SessionStart` hook fires and this is the
block it hands the model, verbatim from `hookSpecificOutput.additionalContext`:

```
{injected}
```

**Act three — the other layer, and what it does not prove here.** The recall block is a *recency
listing*; it says so itself, in its own header, because a model told "here is what you remember"
will trust it more than a recency dump deserves. Anything better has to be *asked for*, which is
what the MCP server is: the same store, exposed as {len(tools)} tools the model can call. Asking it

> {answer.detail}

returns, in served order:

| rank | score | text |
|---|---|---|
{ranked_rows}

**Read that ranking honestly: it is flat, and it is the whole store.** {len(ranked)} episodes went
in and {len(ranked)} came back, spread over {score_spread:.4f} of score. At this size the vector
path cannot be shown to beat the recency dump, because there is nothing for it to leave out. Nobody
should read this section as evidence that retrieval works; the LoCoMo and LongMemEval measurements
are where that question is answered, with intervals — [`docs/18-locomo-4way.md`](../18-locomo-4way.md)
and [`docs/20-lme-reading.md`](../20-lme-reading.md).

What this section *does* establish is the seam, and it is the one thing neither layer's own tests
cover: **an episode a hook wrote on a keystroke came back out of a separately-launched server
process, over the vector path, in a different session.** `tests/test_hooks.py` drives the hooks and
`tests/test_mcp_server.py` drives the server; neither crosses between them, which is exactly why
[`scripts/smoke_product_stack.py`](../../scripts/smoke_product_stack.py) exists and why this demo
runs through it.

## The clock

| step | measured | note |
|---|---|---|
{timing_rows}

**These absolutes are load-dependent and the ratio is not.** This run was taken at a one-minute load average of {load_average:.2f} on a shared workstation; repeated runs move every number here by a factor of two or three together. What does not move is the shape the design was built around — recall lands {timeout_margin:.0f}x inside its deadline while capture, which has none, runs {capture_ratio:.0f}x slower than recall. Quote the ratio; the seconds belong to this box on this afternoon.

## The defect this shape came from, which is the point of showing it

The recall hook is fast because an earlier version of it was not, and the reason is recorded at the
code site rather than in a changelog nobody reads:

> Opening a full `AgenticMemory` costs 9.1 s on this machine, essentially all of it
> `SentenceTransformerEmbedder` reaching its weights; the doc store alone is 0.18 s. […] Those
> figures are from 2026-08-08, after `SentenceTransformerEmbedder` began loading cache-first; the
> same measurement read 15.1 s and 0.21 s (70×) before that.
> — [`src/agmem/hooks/__init__.py`](../../src/agmem/hooks/__init__.py)

At 15 s a blocking `SessionStart` hook exceeds any sane `timeout` and is killed. **And because every
failure path in this package exits 0 silently — a memory system that makes the editor fail to start
is worse than no memory system — the failure did not present as an error.** It presented as a
feature that did nothing. That is the worst failure mode a hook can have, and it is why the recall
path was narrowed to the doc store, why `AGMEM_HOOK_LOG` exists, and why this page draws the
timeout line at all.

A demo that hid its own defect history would be advertising. This repository's whole argument is
that the interesting part of a system is what it does when it is wrong.

## What this does not show

- **One machine, one run.** The timings are this box (the embedder lands on `cuda:0` here); they
  are a shape, not a benchmark.
- **The recall block is recency, not relevance.** There is no query at session start, so there is
  nothing to be relevant to. Anything better requires the model to *ask*, which is what the MCP
  tools are for — and a tool the model must decide to call is exactly the thing hooks exist to stop
  depending on. Both layers ship because neither is sufficient.
- **The transcript is staged**, as stated above. The mechanism, the store, the embedder, the server
  and the latencies are not.

Wiring for a real session — the registration snippets, the namespace and data-dir environment
variables, and why capture must be `async` — is [`docs/05-api-design.md`](../05-api-design.md) §2.4.
The seam these two layers share is checked end to end by
[`scripts/smoke_product_stack.py`](../../scripts/smoke_product_stack.py), whose hook runner and MCP
client this demo reuses rather than reimplements.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-svg", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--keep-store",
        action="store_true",
        help="leave the throwaway store on disk for inspection instead of deleting it",
    )
    args = parser.parse_args()

    data_dir = Path(tempfile.mkdtemp(prefix="agmem-demo-"))
    env = dict(os.environ, AGMEM_DATA_DIR=str(data_dir), AGMEM_NAMESPACE=NAMESPACE)
    print(f"throwaway store: {data_dir}")
    try:
        captures, total = capture_session(env, SESSION_PROMPTS)
        print(f"captured {len(captures)} prompts in {total:.2f}s")
        recall = restart_session(env)
        print(f"recall {recall.seconds:.2f}s, {len(recall.outputs)} lines injected")
        handshake, answer, tools, ranked = ask_over_mcp(data_dir, env)
        print(
            f"mcp handshake {handshake.seconds:.2f}s ({len(tools)} tools), "
            f"search {answer.seconds:.2f}s"
        )

        if not recall.outputs:
            raise SystemExit("recall injected nothing — the demo has no product to show")
        if not ranked:
            raise SystemExit("search_memory returned nothing — the capture/search seam is broken")

        args.out_svg.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_svg.write_text(render_svg(recall))
        args.out_md.write_text(
            reflow(
                render_markdown(
                    captures,
                    recall,
                    handshake,
                    answer,
                    tools,
                    ranked,
                    args.out_svg,
                    args.out_md,
                )
            )
        )
        print(f"wrote {args.out_svg.relative_to(REPO_ROOT)}")
        print(f"wrote {args.out_md.relative_to(REPO_ROOT)}")
    finally:
        if args.keep_store:
            print(f"store kept at {data_dir}")
        else:
            shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
