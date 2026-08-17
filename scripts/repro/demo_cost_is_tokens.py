"""Demo C — draw "the cost of a self-evolving playbook is tokens, not calls" from the artifacts.

Two findings from the ACE/FiNER campaign (docs/19-ace-finer.md) that a reader will not believe
from prose, because both are shaped like a mistake:

  1. Turning the curator's dedup gate off multiplied cost by 5.9 (online $1.461 -> nodedup $8.633)
     while the LLM call count moved by TWO calls (1,323 -> 1,325). A cost model that counts
     requests is wrong here by more than an order of magnitude.
  2. The playbook is injected whole into every generator and curator call, so by the last window a
     single generator prompt carried 639,054 characters of playbook — 98.6% of the prompt — against
     a task whose own prompt averages 1,951 tokens in the arm that never learned.

Nothing here is re-measured or re-run. Every number is read out of an artifact that already exists
and is named in the provenance table this script emits beside the figure, so a reader can check any
of them without spending a cent. That is the point of the demo, and it is why the figure is
generated rather than drawn: a hand-drawn chart cannot be re-derived from the run that produced it.

**Why the SVG is hand-written rather than plotted.** The repository has no plotting dependency and
this figure needs none — two panels of straight lines. A committed binary-ish plot output would
also defeat the demo's own claim, because a reviewer could not diff it against the numbers.

**Two traps in the inputs, both load-bearing.**

  Traces append across processes. `--max-spend-usd` stops an arm between windows and a resume opens
  the same trace file, so `gpt-4o-mini_ace_finer_nodedup.llm-trace.jsonl` holds 447 generate rows
  for a 441-question arm — the extra 6 are calls from an attempt the host killed. They were paid
  for, so they belong in the cost/call totals (which is why those come from `finer_paired.json`,
  which counts off the trace deliberately), but they must NOT be used to reconstruct a per-window
  curve. The curve therefore comes from each summary's `per_window[].playbook_chars_at_test`.

  The summaries' `llm_budget` is per-process and undercounts any resumed arm (nodedup: 164 of
  1,325), and it also counts embedder calls that the trace does not. It is used here for exactly
  one thing — the base arm's task-prompt size, 860,537 / 441 = 1,951 tokens — where the arm ran in
  a single process and made no embedder call.

Run:  uv run python scripts/repro/demo_cost_is_tokens.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "repro"
DEFAULT_SVG = REPO_ROOT / "docs" / "demos" / "assets" / "cost-is-tokens.svg"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "demos" / "cost-is-tokens.md"
TRACE_CACHE = RESULTS_DIR / "demo_cost_is_tokens.trace.json"

# The cache is COMMITTED, unlike the traces it summarises: `.llm-trace.jsonl` is gitignored for
# repo size (306 MB for the nodedup arm alone), so without this file a reader who clones the
# repository could not regenerate the figure at all, and "regenerable from the artifacts" would be
# true only on the machine that ran the campaign. The note travels inside the file so nobody
# mistakes a derived index for a measurement sitting beside the real ones.
TRACE_CACHE_NOTE = (
    "Derived index, not a measurement. Token counts read off the gitignored "
    "results/repro/gpt-4o-mini_ace_finer_*.llm-trace.jsonl files by "
    "scripts/repro/demo_cost_is_tokens.py, cached so the demo regenerates without them. "
    "Invalidated by trace mtime; rebuild with --rescan where the traces exist."
)

ARMS = ("base", "online", "nodedup", "retry")

# gpt-4o-mini's context window, the denominator for "what share of the window did one call eat".
# https://platform.openai.com/docs/models/gpt-4o-mini — 128K, and the run's model per docs/19 §Protocol.
CONTEXT_WINDOW_TOKENS = 128_000

# Colours are fixed per arm so the two panels can be read against each other without a shared legend.
ARM_COLOR = {
    "base": "#6b7280",
    "online": "#2563eb",
    "nodedup": "#dc2626",
    "retry": "#d97706",
}
ARM_LABEL = {
    "base": "base — no playbook",
    "online": "online — dedup 0.90",
    "nodedup": "nodedup — dedup off",
    "retry": "retry — 3 rounds",
}

# (text-anchor, dx, dy) per arm on the cost panel. Placed by hand rather than by rule because the
# four points are deliberately crowded — online and nodedup share an x, and base sits on the axis —
# so a uniform offset puts two labels on top of each other and one under the tick row.
LABEL_PLACEMENT = {
    "base": ("end", -12, -20),
    "online": ("start", 12, -4),
    "nodedup": ("start", 12, -4),
    "retry": ("end", -12, -4),
}

# The trace rows are one JSON object per line and the message payload dominates each line, so the
# role and token count are pulled with a regex rather than parsed: json.loads over the nodedup
# trace builds 307 MB of dicts to read two integers per row.
TRACE_ROLE = re.compile(rb'"role":\s*"([a-z_]+)"')
TRACE_TOKENS_IN = re.compile(rb'"tokens_in":\s*(\d+)')


@dataclass(frozen=True)
class ArmFacts:
    """Everything one arm contributes to the figure, already cross-checked against its sources.

    `playbook_chars` is per window at test time (30 entries) and never reconstructed from the
    trace; `llm_calls` and `cost_usd` come from `finer_paired.json`, which counts off the trace and
    so includes calls from killed attempts — they were billed.
    """

    arm: str
    tag_accuracy: float
    playbook_chars: list[int]
    llm_calls: int
    cost_usd: float
    generate_tokens_in_last_window: float
    generate_tokens_in_max: int
    generate_tokens_in_mean: float
    final_prompt_chars: int
    final_playbook_chars: int
    final_prompt_tokens_in: int
    generate_calls_in_trace: int


def load_summary(arm: str) -> dict:
    """Read one arm's run summary. Raises if absent — a missing arm is not a figure with a gap."""
    path = RESULTS_DIR / f"gpt-4o-mini_ace_finer_{arm}.json"
    if not path.exists():
        raise SystemExit(f"missing artifact: {path.relative_to(REPO_ROOT)}")
    return json.loads(path.read_text())


def scan_trace(arm: str) -> tuple[dict[str, list[int]], bytes | None]:
    """Token counts per role off one arm's trace, in file order, plus the final generate row.

    The first element is `{"generate": [tokens_in, ...], "distill": [...]}`. Those lists include
    calls from attempts the host killed (see module docstring) — correct for cost, wrong for a
    per-window curve, so callers must not index them by window. The second element is the raw last
    `generate` line, kept so the playbook's share of a real prompt can be measured rather than
    assumed; None if the arm made no generate call.
    """
    path = RESULTS_DIR / f"gpt-4o-mini_ace_finer_{arm}.llm-trace.jsonl"
    if not path.exists():
        raise SystemExit(f"missing artifact: {path.relative_to(REPO_ROOT)}")
    tokens_by_role: dict[str, list[int]] = {}
    last_generate: bytes | None = None
    with path.open("rb") as handle:
        for line in handle:
            role_match = TRACE_ROLE.search(line)
            tokens_match = TRACE_TOKENS_IN.search(line)
            if role_match is None or tokens_match is None:
                continue
            # The top-level "role" precedes the messages array, so the first match is the call role.
            role = role_match.group(1).decode()
            tokens_by_role.setdefault(role, []).append(int(tokens_match.group(1)))
            if role == "generate":
                last_generate = line
    return tokens_by_role, last_generate


# Section headers of ACE's generator prompt. The playbook is everything between them, and measuring
# it beats estimating: at the end of the nodedup run the rest of the prompt is a rounding error.
PLAYBOOK_OPEN = "**Playbook:**"
PLAYBOOK_CLOSE = "**Reflection:**"


def measure_playbook_share(generate_row: bytes | None) -> dict:
    """Characters of playbook against characters of whole prompt, for one real generator call.

    Returns zeros when the row is absent or the prompt carries no playbook section — the base arm
    never grows one, and a demo that silently invents a share for it would be the kind of claim
    this repository exists to argue against.
    """
    if generate_row is None:
        return {"prompt_chars": 0, "playbook_chars": 0, "tokens_in": 0}
    row = json.loads(generate_row)
    prompt = "".join(message["content"] for message in row["messages"])
    open_at = prompt.find(PLAYBOOK_OPEN)
    close_at = prompt.find(PLAYBOOK_CLOSE)
    playbook_chars = 0
    if open_at != -1 and close_at > open_at:
        playbook_chars = len(prompt[open_at + len(PLAYBOOK_OPEN) : close_at].strip())
    return {
        "prompt_chars": len(prompt),
        "playbook_chars": playbook_chars,
        "tokens_in": row["tokens_in"],
    }


def load_trace_stats(rescan: bool) -> dict[str, dict]:
    """Trace-derived token counts for every arm, cached because the traces total ~500 MB.

    The cache is invalidated by trace mtime, so an arm that gets re-run is re-scanned and a reader
    who clones the repo without the traces still gets the figure from the cache alone.
    """
    traces = {arm: RESULTS_DIR / f"gpt-4o-mini_ace_finer_{arm}.llm-trace.jsonl" for arm in ARMS}
    stamps = {
        arm: (path.stat().st_mtime_ns if path.exists() else None) for arm, path in traces.items()
    }
    if not rescan and TRACE_CACHE.exists():
        cached = json.loads(TRACE_CACHE.read_text())
        fresh = all(
            cached.get("mtime_ns", {}).get(arm) == stamps[arm] or stamps[arm] is None
            for arm in ARMS
        )
        if fresh and all(arm in cached.get("stats", {}) for arm in ARMS):
            return cached["stats"]

    stats: dict[str, dict] = {}
    for arm in ARMS:
        tokens_by_role, last_generate = scan_trace(arm)
        generate = tokens_by_role.get("generate", [])
        # A window is 15 questions, but the tail of a resumed arm's trace spans the seam between
        # the process the host killed and the one that finished — nodedup's last 15 rows are 9
        # from the killed process and 6 from the resume, both carrying a fully grown playbook.
        # So the tail mean is kept as a range and the headline is taken from the FINAL row, which
        # belongs to exactly one process and needs no averaging to defend.
        tail = generate[-15:]
        stats[arm] = {
            "generate_calls_in_trace": len(generate),
            "generate_tokens_in_total": sum(generate),
            "generate_tokens_in_max": max(generate) if generate else 0,
            "generate_tokens_in_mean": sum(generate) / len(generate) if generate else 0.0,
            "generate_tokens_in_last_window": sum(tail) / len(tail) if tail else 0.0,
            "distill_calls_in_trace": len(tokens_by_role.get("distill", [])),
            "distill_tokens_in_total": sum(tokens_by_role.get("distill", [])),
            "final_prompt": measure_playbook_share(last_generate),
        }
    TRACE_CACHE.write_text(
        json.dumps(
            {"_note": TRACE_CACHE_NOTE, "mtime_ns": stamps, "stats": stats},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return stats


def collect() -> tuple[dict[str, ArmFacts], dict]:
    """Assemble the four arms plus the paired-comparison artifact the cost/call panel is drawn from."""
    paired_path = RESULTS_DIR / "finer_paired.json"
    if not paired_path.exists():
        raise SystemExit(f"missing artifact: {paired_path.relative_to(REPO_ROOT)}")
    paired = json.loads(paired_path.read_text())
    trace_stats = load_trace_stats(rescan=False)

    facts: dict[str, ArmFacts] = {}
    for arm in ARMS:
        summary = load_summary(arm)
        playbook_chars = [window["playbook_chars_at_test"] for window in summary["per_window"]]
        stats = trace_stats[arm]
        facts[arm] = ArmFacts(
            arm=arm,
            tag_accuracy=paired["anchors"][arm]["tag_accuracy"],
            playbook_chars=playbook_chars,
            llm_calls=paired["llm_calls"][arm],
            cost_usd=paired["cost_usd"][arm],
            generate_tokens_in_last_window=stats["generate_tokens_in_last_window"],
            generate_tokens_in_max=stats["generate_tokens_in_max"],
            generate_tokens_in_mean=stats["generate_tokens_in_mean"],
            final_prompt_chars=stats["final_prompt"]["prompt_chars"],
            final_playbook_chars=stats["final_prompt"]["playbook_chars"],
            final_prompt_tokens_in=stats["final_prompt"]["tokens_in"],
            generate_calls_in_trace=stats["generate_calls_in_trace"],
        )
    return facts, paired


# --------------------------------------------------------------------------- SVG

# A committed SVG is rendered by GitHub as an <img>, where the page's dark theme does not reach the
# document's own CSS. So the panel paints its own light ground explicitly and never relies on the
# host background — a transparent figure with dark text vanishes for half the readers.
BACKGROUND = "#ffffff"
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"

WIDTH, HEIGHT = 1000, 470
PANEL_TOP, PANEL_BOTTOM = 86, 366
PANEL_A_LEFT, PANEL_A_RIGHT = 72, 468
PANEL_B_LEFT, PANEL_B_RIGHT = 596, 962


def escape(text: str) -> str:
    """XML-escape a label. Arm names and numbers only, but the figure is committed, so be exact."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text_element(
    x: float,
    y: float,
    content: str,
    *,
    size: float = 12,
    fill: str = INK,
    anchor: str = "start",
    weight: str = "normal",
) -> str:
    """One <text> node, with the family fixed so the committed file renders identically everywhere."""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" '
        f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"'
        f">{escape(content)}</text>"
    )


def thousands(value: float) -> str:
    """Group digits so a six-figure character count reads as one at a glance."""
    return f"{value:,.0f}"


def decade_label(exponent: int) -> str:
    """Axis label for a power of ten, as a reader says it rather than as a formula."""
    if exponent < 3:
        return thousands(10**exponent)
    if exponent < 6:
        return f"{10 ** (exponent - 3)}K"
    return f"{10 ** (exponent - 6)}M"


def wrap(text: str, width: int) -> list[str]:
    """Greedy wrap by word count. SVG has no flow layout, so the caption wraps itself or overruns."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def panel_playbook(facts: dict[str, ArmFacts]) -> list[str]:
    """Left panel: playbook characters carried into every call, per window, log y.

    Log scale because the four arms span 7 to 639,054 characters; on a linear axis the base arm and
    the online arm are both the x-axis. The label on each line is the arm's final value, because
    that endpoint is the number quoted in docs/19 and the reader should be able to match them.
    """
    parts: list[str] = []
    n_windows = len(next(iter(facts.values())).playbook_chars)
    log_min, log_max = 0.0, 6.0  # 1 char to 1,000,000 chars

    def x_of(window: int) -> float:
        return PANEL_A_LEFT + (PANEL_A_RIGHT - PANEL_A_LEFT) * window / (n_windows - 1)

    def y_of(chars: int) -> float:
        position = (math.log10(max(chars, 1)) - log_min) / (log_max - log_min)
        return PANEL_BOTTOM - (PANEL_BOTTOM - PANEL_TOP) * position

    parts.append(
        text_element(PANEL_A_LEFT, 44, "Playbook injected into every call", size=15, weight="600")
    )
    parts.append(
        text_element(
            PANEL_A_LEFT,
            63,
            "characters, at each window's test point (log scale)",
            size=11.5,
            fill=MUTED,
        )
    )

    for decade in range(int(log_min), int(log_max) + 1):
        y = y_of(10**decade)
        parts.append(
            f'<line x1="{PANEL_A_LEFT}" y1="{y:.1f}" x2="{PANEL_A_RIGHT}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            text_element(
                PANEL_A_LEFT - 8, y + 4, decade_label(decade), size=10.5, fill=MUTED, anchor="end"
            )
        )

    parts.append(
        f'<line x1="{PANEL_A_LEFT}" y1="{PANEL_BOTTOM}" x2="{PANEL_A_RIGHT}" y2="{PANEL_BOTTOM}" '
        f'stroke="{INK}" stroke-width="1.2"/>'
    )
    for window in (0, 9, 19, 29):
        x = x_of(window)
        parts.append(
            f'<line x1="{x:.1f}" y1="{PANEL_BOTTOM}" x2="{x:.1f}" y2="{PANEL_BOTTOM + 5}" '
            f'stroke="{INK}" stroke-width="1.2"/>'
        )
        parts.append(
            text_element(x, PANEL_BOTTOM + 19, str(window), size=10.5, fill=MUTED, anchor="middle")
        )
    parts.append(
        text_element(
            (PANEL_A_LEFT + PANEL_A_RIGHT) / 2,
            PANEL_BOTTOM + 38,
            "window (30 x 15 questions)",
            size=11,
            fill=MUTED,
            anchor="middle",
        )
    )

    for arm in ARMS:
        chars = facts[arm].playbook_chars
        points = " ".join(f"{x_of(i):.1f},{y_of(value):.1f}" for i, value in enumerate(chars))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{ARM_COLOR[arm]}" '
            f'stroke-width="2" stroke-linejoin="round"/>'
        )
        end_y = y_of(chars[-1])
        parts.append(
            text_element(
                PANEL_A_RIGHT + 6,
                end_y + 4,
                thousands(chars[-1]),
                size=10.5,
                fill=ARM_COLOR[arm],
                weight="600",
            )
        )
    return parts


def panel_cost(facts: dict[str, ArmFacts]) -> list[str]:
    """Right panel: dollars against LLM calls. This panel IS the argument.

    `online` and `nodedup` sit two calls apart on x and 5.9x apart on y; `retry` sits at more than
    twice the calls of either and below nodedup. If cost tracked requests these four points would
    lie on a line through the origin, and the panel exists to show that they do not.
    """
    parts: list[str] = []
    max_calls, max_cost = 3200.0, 9.5

    def x_of(calls: float) -> float:
        return PANEL_B_LEFT + (PANEL_B_RIGHT - PANEL_B_LEFT) * calls / max_calls

    def y_of(cost: float) -> float:
        return PANEL_BOTTOM - (PANEL_BOTTOM - PANEL_TOP) * cost / max_cost

    parts.append(text_element(PANEL_B_LEFT, 44, "Cost against LLM calls", size=15, weight="600"))
    parts.append(
        text_element(
            PANEL_B_LEFT,
            63,
            "same benchmark, same model, same 441 questions",
            size=11.5,
            fill=MUTED,
        )
    )

    for dollars in range(0, 10, 2):
        y = y_of(dollars)
        parts.append(
            f'<line x1="{PANEL_B_LEFT}" y1="{y:.1f}" x2="{PANEL_B_RIGHT}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            text_element(
                PANEL_B_LEFT - 8, y + 4, f"${dollars}", size=10.5, fill=MUTED, anchor="end"
            )
        )
    parts.append(
        f'<line x1="{PANEL_B_LEFT}" y1="{PANEL_BOTTOM}" x2="{PANEL_B_RIGHT}" y2="{PANEL_BOTTOM}" '
        f'stroke="{INK}" stroke-width="1.2"/>'
    )
    for calls in (0, 1000, 2000, 3000):
        x = x_of(calls)
        parts.append(
            f'<line x1="{x:.1f}" y1="{PANEL_BOTTOM}" x2="{x:.1f}" y2="{PANEL_BOTTOM + 5}" '
            f'stroke="{INK}" stroke-width="1.2"/>'
        )
        parts.append(
            text_element(
                x, PANEL_BOTTOM + 19, thousands(calls), size=10.5, fill=MUTED, anchor="middle"
            )
        )
    parts.append(
        text_element(
            (PANEL_B_LEFT + PANEL_B_RIGHT) / 2,
            PANEL_BOTTOM + 38,
            "generator + curator calls (off each arm's trace)",
            size=11,
            fill=MUTED,
            anchor="middle",
        )
    )

    # The connector between online and nodedup is the finding: near-vertical, because x barely moves.
    online, nodedup = facts["online"], facts["nodedup"]
    x_online, x_nodedup = x_of(online.llm_calls), x_of(nodedup.llm_calls)
    parts.append(
        f'<line x1="{x_online:.1f}" y1="{y_of(online.cost_usd):.1f}" '
        f'x2="{x_nodedup:.1f}" y2="{y_of(nodedup.cost_usd):.1f}" '
        f'stroke="{INK}" stroke-width="1.2" stroke-dasharray="4 3"/>'
    )
    mid_y = (y_of(online.cost_usd) + y_of(nodedup.cost_usd)) / 2
    delta_calls = nodedup.llm_calls - online.llm_calls
    ratio = nodedup.cost_usd / online.cost_usd
    # Annotated on the LEFT of the connector: the right side belongs to the retry arm's label.
    parts.append(
        text_element(
            x_nodedup - 14, mid_y - 6, f"+{delta_calls} calls", size=12, weight="600", anchor="end"
        )
    )
    parts.append(
        text_element(
            x_nodedup - 14,
            mid_y + 11,
            f"x{ratio:.1f} cost",
            size=12,
            fill="#dc2626",
            weight="600",
            anchor="end",
        )
    )

    # Where the four arms would sit if a call cost what a base-arm call cost.
    base = facts["base"]
    per_call = base.cost_usd / base.llm_calls
    parts.append(
        f'<line x1="{x_of(0):.1f}" y1="{y_of(0):.1f}" '
        f'x2="{x_of(max_calls):.1f}" y2="{y_of(per_call * max_calls):.1f}" '
        f'stroke="{MUTED}" stroke-width="1" stroke-dasharray="2 4"/>'
    )
    parts.append(
        text_element(
            x_of(max_calls) - 4,
            y_of(per_call * max_calls) + 16,
            "if cost tracked calls",
            size=10.5,
            fill=MUTED,
            anchor="end",
        )
    )

    for arm in ARMS:
        fact = facts[arm]
        x, y = x_of(fact.llm_calls), y_of(fact.cost_usd)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{ARM_COLOR[arm]}"/>')
        anchor, offset_x, offset_y = LABEL_PLACEMENT[arm]
        parts.append(
            text_element(
                x + offset_x,
                y + offset_y,
                ARM_LABEL[arm],
                size=11,
                fill=ARM_COLOR[arm],
                anchor=anchor,
                weight="600",
            )
        )
        parts.append(
            text_element(
                x + offset_x,
                y + offset_y + 15,
                f"{thousands(fact.llm_calls)} calls  ${fact.cost_usd:.2f}",
                size=10.5,
                fill=MUTED,
                anchor=anchor,
            )
        )
    return parts


def render_svg(facts: dict[str, ArmFacts]) -> str:
    """The whole figure as one self-contained SVG string, deterministic given the artifacts."""
    nodedup = facts["nodedup"]
    share = nodedup.final_prompt_tokens_in / CONTEXT_WINDOW_TOKENS
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'width="{WIDTH}" height="{HEIGHT}" role="img" '
            f'aria-label="ACE on FiNER: playbook growth per window, and cost against LLM calls">'
        ),
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>',
        text_element(
            PANEL_A_LEFT,
            26,
            "ACE on FiNER — the cost of a growing playbook is tokens, not calls",
            size=17,
            weight="700",
        ),
    ]
    parts += panel_playbook(facts)
    parts += panel_cost(facts)
    parts.append(
        f'<line x1="{(PANEL_A_RIGHT + PANEL_B_LEFT) / 2 - 24:.1f}" y1="70" '
        f'x2="{(PANEL_A_RIGHT + PANEL_B_LEFT) / 2 - 24:.1f}" y2="{PANEL_BOTTOM + 40}" '
        f'stroke="{GRID}" stroke-width="1"/>'
    )
    caption = (
        f"gpt-4o-mini, temp 0, FiNER 441 questions x 4 tags, one seed. The nodedup arm's final "
        f"generator call carried {thousands(nodedup.final_prompt_tokens_in)} input tokens — "
        f"{share * 100:.0f}% of the 128K context window, and "
        f"{nodedup.final_playbook_chars / nodedup.final_prompt_chars * 100:.1f}% of it was playbook. "
        f"Accuracy is not separable across these four arms (docs/19-ace-finer.md). Generated by "
        f"scripts/repro/demo_cost_is_tokens.py from results/repro/ — no model call, no re-run."
    )
    # ~7.4 px per character at 11 px in this family, so the caption is wrapped to the drawing width
    # rather than trusted to fit: an overrunning <text> is silently clipped by the viewBox.
    caption_width = int((PANEL_B_RIGHT - PANEL_A_LEFT) / 5.6)
    for index, line in enumerate(wrap(caption, caption_width)):
        parts.append(
            text_element(PANEL_A_LEFT, HEIGHT - 42 + index * 16, line, size=11, fill=MUTED)
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------- markdown

MARKDOWN_WIDTH = 100
LIST_MARKER = re.compile(r"^([-*+]\s|\d+\.\s)")


def reflow(markdown: str) -> str:
    """Re-wrap prose paragraphs to the repo's 100 columns, leaving structure alone.

    The document is assembled from f-strings whose interpolated numbers have no fixed width, so
    without this the committed file is ragged in a way that reads as carelessness and makes every
    regeneration a noisy diff. Tables, headings, fences, quotes and list items are structural and
    pass through untouched.
    """
    out: list[str] = []
    paragraph: list[str] = []
    in_fence = False

    def flush() -> None:
        if paragraph:
            out.extend(wrap(" ".join(paragraph), MARKDOWN_WIDTH))
            paragraph.clear()

    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        # `**bold` opens a prose paragraph here and `* ` opens a list, so the list test requires
        # the space — without it every emphasised lead sentence escapes the reflow and stays ragged.
        structural = (
            in_fence
            or not stripped
            or stripped.startswith(("|", "#", ">", "![", "```"))
            or LIST_MARKER.match(stripped) is not None
        )
        if structural:
            flush()
            out.append(line)
        else:
            paragraph.append(stripped)
    flush()
    return "\n".join(out)


def render_markdown(
    facts: dict[str, ArmFacts], paired: dict, svg_path: Path, markdown_path: Path
) -> str:
    """The figure's numbers in text, each next to the artifact field it was read from.

    Every headline in this campaign carries at least one condition, so none of them is restated
    here without its arm, its model and its benchmark — a number lifted out of its arm is how a
    null result gets quoted as a win.
    """
    base, online, nodedup, retry = (facts[arm] for arm in ARMS)
    relative_svg = Path(svg_path).relative_to(markdown_path.parent)
    base_prompt_tokens = load_summary("base")["llm_budget"]["generate"]
    average_task_tokens = base_prompt_tokens["tokens_in"] / base_prompt_tokens["calls"]
    share = nodedup.final_prompt_tokens_in / CONTEXT_WINDOW_TOKENS
    growth = nodedup.final_prompt_tokens_in / average_task_tokens
    # `llm_budget` counts one process, which for a resumed arm is exactly the process that finished
    # it — the one field where being per-process is the useful property rather than the trap. Its
    # total also folds in the embedder, which is why it can never be compared with a trace count.
    nodedup_budget = load_summary("nodedup")["llm_budget"]
    final_process_generate_calls = nodedup_budget["generate"]["calls"]
    final_process_calls = sum(role["calls"] for role in nodedup_budget.values())
    # The interval is READ, never restated: a second copy of a confidence interval is how two
    # numbers in one repository come to disagree about what 95% means (finer_paired.py's own rule).
    paired_delta = SimpleNamespace(**paired["comparisons"]["nodedup"]["sample_accuracy_paired"])

    rows = "\n".join(
        f"| `{fact.arm}` | {fact.tag_accuracy:.2f} | {thousands(fact.llm_calls)} | "
        f"${fact.cost_usd:.3f} | {thousands(fact.playbook_chars[-1])} | "
        f"{thousands(fact.final_prompt_tokens_in)} |"
        for fact in (base, online, nodedup, retry)
    )

    return f"""# Demo C — cost is tokens, not calls

> Runs in under a second, spends **$0**, makes **no model call**. Everything below is read out of
> artifacts already in `results/repro/`.
>
> ```
> uv run python scripts/repro/demo_cost_is_tokens.py
> ```

![ACE on FiNER: playbook growth per window, and cost against LLM calls]({relative_svg})

## The two numbers a reader will assume are typos

**Turning one boolean off multiplied the bill by {nodedup.cost_usd / online.cost_usd:.1f}× and moved
the call count by {nodedup.llm_calls - online.llm_calls}.** The `online` arm ran ACE's curator with
our 0.90-cosine dedup gate; `nodedup` ran it at upstream's shipped default, which is no gate at all.
{thousands(online.llm_calls)} calls → {thousands(nodedup.llm_calls)} calls.
${online.cost_usd:.3f} → ${nodedup.cost_usd:.3f}.

**The nodedup arm's final generator call carried
{thousands(nodedup.final_prompt_tokens_in)} input tokens — {share * 100:.0f}% of
`gpt-4o-mini`'s 128 K context window — around a task whose own prompt averages
{average_task_tokens:,.0f} tokens.** That is {growth:.0f}× the task, and it is the same task: the
playbook is injected whole into every generator and curator call, so it rides along
{thousands(nodedup.playbook_chars[-1])} characters at a time — measured on that arm's final
generator prompt, **{nodedup.final_playbook_chars / nodedup.final_prompt_chars * 100:.1f}% of
everything the model was handed** ({thousands(nodedup.final_playbook_chars)} of
{thousands(nodedup.final_prompt_chars)} characters).

**Any cost model for a self-evolving playbook that counts requests is wrong here by more than an
order of magnitude.**

## The four arms

Condition for every row: FiNER shipped test split, 441 questions × 4 US-GAAP tags, `gpt-4o-mini` at
temperature 0 as both generator and curator, `text-embedding-3-small`, ACE's online mode over 30
windows of 15, **one seed**.

| arm | tag accuracy | LLM calls | cost | playbook chars, final window | final generator call, tokens in |
|---|---|---|---|---|---|
{rows}

The last column is one measured call, not an average, because `nodedup` and `retry` were resumed
after the host killed them and the tail of a resumed trace straddles two processes. For the record,
`nodedup`'s last fifteen generator calls average {thousands(nodedup.generate_tokens_in_last_window)}
tokens — {15 - final_process_generate_calls} from the killed process and
{final_process_generate_calls} from the one that finished, all carrying a fully grown playbook.

**The accuracy column is the reason this is a cost demo and not a win.** None of the three learning
arms separates from `base` — the arm that carries no playbook at all — in a paired bootstrap over
the same questions. The most expensive arm against the reference: `nodedup − base` =
{paired_delta.delta_pp:+.2f} pp on sample accuracy, 95% CI [{paired_delta.lo:+.2f},
{paired_delta.hi:+.2f}], p = {paired_delta.p_boot:.3f}, over {paired_delta.n_boot:,} resamples at
seed {paired_delta.seed}. The spending in the cost column bought nothing measurable. Full treatment, including what the playbook did contain
and why the retry arm makes things worse: [docs/19-ace-finer.md](../19-ace-finer.md).

## Where each number comes from

| number | artifact | field |
|---|---|---|
| playbook chars per window | `results/repro/gpt-4o-mini_ace_finer_{{arm}}.json` | `per_window[].playbook_chars_at_test` |
| cost and calls per arm | `results/repro/finer_paired.json` | `cost_usd`, `llm_calls` |
| tag accuracy per arm | `results/repro/finer_paired.json` | `anchors[arm].tag_accuracy` |
| generator tokens per call | `results/repro/gpt-4o-mini_ace_finer_{{arm}}.llm-trace.jsonl` | `tokens_in` on `generate` rows |
| base arm's task-prompt size | `results/repro/gpt-4o-mini_ace_finer_base.json` | `llm_budget.generate.tokens_in / calls` |

**Two things the artifacts will not let you do.** The traces append across processes, so
`nodedup.llm-trace.jsonl` holds {thousands(nodedup.generate_calls_in_trace)} generate rows for a
441-question arm — the {nodedup.generate_calls_in_trace - 441} extras are calls from an attempt the
host killed. They were billed, so they count toward cost, and they must not be used to rebuild a
per-window curve; that curve comes from the summaries. And the summaries' `llm_budget` counts only
the last process of a resumed arm ({final_process_calls} calls of
{thousands(nodedup.llm_calls)} for `nodedup`) while also counting the embedder calls the trace does
not — it is used here for the base arm alone, which ran in one process and embedded nothing.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-svg", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="re-read the ~500 MB of traces instead of using the cached token counts",
    )
    args = parser.parse_args()

    load_trace_stats(rescan=args.rescan)
    facts, paired = collect()

    args.out_svg.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_svg.write_text(render_svg(facts))
    args.out_md.write_text(reflow(render_markdown(facts, paired, args.out_svg, args.out_md)))

    nodedup, online = facts["nodedup"], facts["online"]
    print(f"wrote {args.out_svg.relative_to(REPO_ROOT)}")
    print(f"wrote {args.out_md.relative_to(REPO_ROOT)}")
    print(
        f"headline: {online.llm_calls} -> {nodedup.llm_calls} calls "
        f"(+{nodedup.llm_calls - online.llm_calls}), "
        f"${online.cost_usd:.3f} -> ${nodedup.cost_usd:.3f} "
        f"(x{nodedup.cost_usd / online.cost_usd:.1f}); "
        f"final generator prompt {nodedup.final_prompt_tokens_in:,} tokens "
        f"({nodedup.final_prompt_tokens_in / CONTEXT_WINDOW_TOKENS * 100:.0f}% of 128K)"
    )


if __name__ == "__main__":
    main()
