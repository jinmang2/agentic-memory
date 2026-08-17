"""Demo A — one conversation, five methodologies, one interface, and what each of them stored.

Nine agentic-memory methodologies live in this repository behind a single API, selected by name.
The claim that is hard to believe from prose is how *differently* they behave on identical input, so
this page takes one LoCoMo conversation that all five write paths have already been run over — same
419 turns, same `gpt-4o-mini`, same `text-embedding-3-small`, one harness — and puts side by side
what each one decided to keep, how many model calls it spent deciding, and what it wrote about the
same single event.

**THIS IS AN ILLUSTRATION, NOT A MEASUREMENT.** One conversation, one seed, no accuracy number
anywhere on the page. The measurements are `docs/18-locomo-4way.md` and `docs/19-ace-finer.md`, and
they carry intervals. A count of stored items says nothing about whether the right things were
stored — a system can write 1,388 items and answer worse than one that wrote 193, which is exactly
why the counts are not ranked here and the word "best" does not appear.

**Cost: $0.** Nothing is ingested and no model is called. Every number is read out of an ingest
summary that the LoCoMo campaign already paid for and committed, and every quoted item is read out
of that run's memory snapshot.

**Why ACE is absent.** ACE's arm in this campaign runs on FiNER, a tagging benchmark, not on a
conversation. Putting it in this table would mean comparing what it stored about a different input,
which is the class of error the whole repository is about. It has its own page: `cost-is-tokens.md`.

**The snapshots are gitignored, so a derived index is committed.** `*.memory.jsonl` is excluded from
git for size (docs/14 §Artifacts); without the cache this script writes, a reader who clones the
repository could not regenerate the page. Same arrangement, and same reasoning, as
`demo_cost_is_tokens.py`.

Run:  uv run python scripts/repro/demo_methodology_switch.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from demo_svg import BACKGROUND, GRID, MUTED, reflow, text_element, thousands, wrap

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "repro"
DEFAULT_SVG = REPO_ROOT / "docs" / "demos" / "assets" / "methodology-switch.svg"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "demos" / "methodology-switch.md"
SNAPSHOT_CACHE = RESULTS_DIR / "demo_methodology_switch.snapshot.json"

SNAPSHOT_CACHE_NOTE = (
    "Derived index, not a measurement. Item-type counts and quoted items read off the gitignored "
    "results/repro/*.memory.jsonl snapshots by scripts/repro/demo_methodology_switch.py, cached so "
    "the demo regenerates without them. Rebuild with --rescan where the snapshots exist."
)

# The topic every arm is quoted on, chosen by hand because all five wrote about it. It is a TOPIC
# filter, not an event selector: the first match per kind is taken in file order, and those matches
# are demonstrably not all the same event (one arm's first matching episode is from a different day).
# The page says so, because the claim the quotes actually support is about the SHAPE of what each
# methodology writes, and shape does not need the sentences to be about one moment.
ANCHOR = "support group"

# One conversation, one embedder, one model. The five arms of the docs/18 campaign's write paths.
# `config` is the literal --config value: the point of the page is that this column is the ONLY
# thing that differs between these runs.
ARMS = (
    ("A-Mem", "amem", "gpt-4o-mini_conv0_ingest_e3s_c0"),
    ("Mem0", "mem0_v0194", "gpt-4o-mini_mem0_v0194_conv0_ingest_e3sM_c0"),
    ("Nemori (upstream)", "nemori_upstream", "gpt-4o-mini_nemori_upstream_conv0_ingest_e3sA_c0"),
    ("Nemori (merge 0.85)", "nemori_merge085", "gpt-4o-mini_nemori_merge085_conv0_ingest_e3sB_c0"),
    ("Zep/Graphiti", "zep_cross_encoder", "gpt-4o-mini_zep_cross_encoder_conv0_ingest_e3sZ_c0"),
)

# `episodic` is the raw transcript — the INPUT, not a decision — so it is excluded from "what this
# methodology chose to keep", following the harness's own `derived` convention
# (exp_locomo_conv0.capture_memory). It is NOT uniformly present across these snapshots, which the
# generated page reports rather than smooths over.
RAW_TRANSCRIPT_TYPE = "episodic"

TYPE_COLOR = {
    "notes": "#2563eb",
    "semantic": "#059669",
    "episodes": "#7c3aed",
    "facts": "#dc2626",
    "entities": "#d97706",
}


@dataclass(frozen=True)
class Arm:
    """One methodology's run over the shared conversation, summary and snapshot already joined."""

    label: str
    config: str
    stem: str
    cost_usd: float
    ingest_seconds: float
    commit: str
    op_counts: dict[str, int]
    write_calls: dict[str, int]
    embed_calls: int
    stored_by_type: dict[str, int]
    raw_turns: int
    quoted: list[dict]

    @property
    def total_write_calls(self) -> int:
        return sum(self.write_calls.values())

    @property
    def total_stored(self) -> int:
        return sum(self.stored_by_type.values())


def turn_count_of(arms: list[Arm]) -> int:
    """The conversation's turn count, from the op log every arm agrees on.

    Read from `ADD:episodic` rather than from a snapshot because two of these five snapshots hold no
    episodic rows at all; a snapshot-derived count would quietly be one arm's answer printed for
    five. A disagreement here means the arms did not see the same input, which is this page's whole
    premise, so it raises instead of picking a winner.
    """
    counts = {arm.op_counts.get("ADD:episodic", 0) for arm in arms}
    if len(counts) != 1:
        raise SystemExit(
            f"arms disagree on the conversation's turn count ({sorted(counts)}); they are not over "
            f"the same input and this page's premise does not hold"
        )
    return counts.pop()


def load_summary(stem: str) -> dict:
    path = RESULTS_DIR / f"{stem}.json"
    if not path.exists():
        raise SystemExit(f"missing artifact: {path.relative_to(REPO_ROOT)}")
    return json.loads(path.read_text())


def snapshot_path(stem: str, summary: dict) -> Path:
    """Where this run's memory snapshot actually is.

    The summary carries a `memory_file` pointer, and for one arm in this set that pointer is stale —
    `nemori_upstream`'s says `..._e3sA.memory.jsonl` where the file on disk is `..._e3sA_c0`,
    because the run label gained its `_c0` suffix after the field was written. The pointer is tried
    first and the run's own stem second, so the demo neither trusts a wrong pointer nor silently
    papers over one that is right.
    """
    pointed = RESULTS_DIR / summary.get("memory_file", "")
    if pointed.is_file():
        return pointed
    return RESULTS_DIR / f"{stem}.memory.jsonl"


def scan_snapshot(stem: str, summary: dict) -> dict:
    """Item counts by type plus the items that mention the anchor, off one run's memory snapshot."""
    path = snapshot_path(stem, summary)
    if not path.exists():
        raise SystemExit(
            f"missing snapshot for {stem}: {path.name}. It is gitignored; run with the campaign's "
            f"artifacts present, or use the committed cache without --rescan."
        )
    counts: Counter[str] = Counter()
    quoted: list[dict] = []
    for line in path.open():
        record = json.loads(line)
        memory_type = record.get("memory_type", "?")
        counts[memory_type] += 1
        content = str(record.get("content", ""))
        if (
            memory_type != RAW_TRANSCRIPT_TYPE
            and ANCHOR.lower() in content.lower()
            and not any(item["memory_type"] == memory_type for item in quoted)
        ):
            quoted.append({"memory_type": memory_type, "content": content})
    return {"counts": dict(counts), "quoted": quoted}


def load_snapshots(stems_and_summaries: list[tuple[str, dict]], rescan: bool) -> dict[str, dict]:
    """Snapshot-derived facts for every arm, cached because the snapshots are not in git."""
    paths = {stem: snapshot_path(stem, summary) for stem, summary in stems_and_summaries}
    stamps = {
        stem: (path.stat().st_mtime_ns if path.exists() else None) for stem, path in paths.items()
    }
    if not rescan and SNAPSHOT_CACHE.exists():
        cached = json.loads(SNAPSHOT_CACHE.read_text())
        fresh = all(
            cached.get("mtime_ns", {}).get(stem) == stamps[stem] or stamps[stem] is None
            for stem in stamps
        )
        if fresh and all(stem in cached.get("stats", {}) for stem in stamps):
            return cached["stats"]

    stats = {stem: scan_snapshot(stem, summary) for stem, summary in stems_and_summaries}
    SNAPSHOT_CACHE.write_text(
        json.dumps(
            {"_note": SNAPSHOT_CACHE_NOTE, "anchor": ANCHOR, "mtime_ns": stamps, "stats": stats},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return stats


def collect(rescan: bool) -> list[Arm]:
    """The five arms, each with its summary numbers and its snapshot contents."""
    summaries = {stem: load_summary(stem) for _, _, stem in ARMS}
    snapshots = load_snapshots([(stem, summaries[stem]) for _, _, stem in ARMS], rescan)

    arms = []
    for label, config, stem in ARMS:
        summary = summaries[stem]
        budget = summary["llm_budget"]
        # Embedding calls are separated from generative ones throughout this repository: they are
        # priced differently by three orders of magnitude, and a "call count" that mixes them is the
        # ledger's own C-entry about MemoryOS repeating itself.
        write_calls = {role: spend["calls"] for role, spend in budget.items() if role != "embed"}
        counts = dict(snapshots[stem]["counts"])
        arms.append(
            Arm(
                label=label,
                config=config,
                stem=stem,
                cost_usd=summary["cost_usd"],
                ingest_seconds=summary["timing"]["ingest_s"],
                commit=summary["stamp"]["commit"][:7],
                op_counts=summary["op_counts"],
                write_calls=write_calls,
                embed_calls=budget.get("embed", {}).get("calls", 0),
                stored_by_type={
                    key: value for key, value in counts.items() if key != RAW_TRANSCRIPT_TYPE
                },
                raw_turns=counts.get(RAW_TRANSCRIPT_TYPE, 0),
                quoted=snapshots[stem]["quoted"],
            )
        )
    return arms


# ------------------------------------------------------------------------------------------- SVG

WIDTH, HEIGHT = 1000, 470
CHART_LEFT, CHART_RIGHT = 210, 830
CHART_TOP = 104
ROW_HEIGHT = 58
BAR_HEIGHT = 26


def render_svg(arms: list[Arm], turn_count: int) -> str:
    """Stacked bars: what each methodology kept, by kind, with its bill in the margin.

    Stacked rather than a single total because the kinds are the finding — 750 facts plus 219
    entities is a different object from 419 notes, and a bar chart of totals would hide exactly the
    difference the page is about.
    """
    widest = max(arm.total_stored for arm in arms)
    span = widest * 1.06

    def x_of(items: float) -> float:
        return CHART_LEFT + (CHART_RIGHT - CHART_LEFT) * items / span

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'width="{WIDTH}" height="{HEIGHT}" role="img" '
            f'aria-label="What five memory methodologies stored from one identical conversation">'
        ),
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>',
        text_element(
            40,
            34,
            "One conversation, five methodologies, five different memories",
            size=17,
            weight="700",
        ),
        text_element(
            40,
            56,
            f"LoCoMo conversation 0 — {thousands(turn_count)} turns, gpt-4o-mini, "
            f"text-embedding-3-small, one harness. The --config name is what differs; the runs also sit "
            f"at different repo commits, which the page states.",
            size=11.5,
            fill=MUTED,
        ),
    ]

    for index, arm in enumerate(arms):
        y = CHART_TOP + index * ROW_HEIGHT
        parts.append(text_element(CHART_LEFT - 12, y + 17, arm.label, size=11.5, anchor="end"))
        parts.append(
            text_element(
                CHART_LEFT - 12,
                y + 31,
                f"--config {arm.config}",
                size=9.5,
                fill=MUTED,
                anchor="end",
            )
        )
        cursor = float(CHART_LEFT)
        for memory_type, count in sorted(arm.stored_by_type.items(), key=lambda kv: -kv[1]):
            width = x_of(count) - CHART_LEFT
            color = TYPE_COLOR.get(memory_type, MUTED)
            parts.append(
                f'<rect x="{cursor:.1f}" y="{y}" width="{max(width, 1.0):.1f}" '
                f'height="{BAR_HEIGHT}" fill="{color}"/>'
            )
            if width > 62:
                parts.append(
                    text_element(
                        cursor + width / 2,
                        y + 12,
                        memory_type,
                        size=10,
                        fill=BACKGROUND,
                        anchor="middle",
                        weight="600",
                    )
                )
                parts.append(
                    text_element(
                        cursor + width / 2,
                        y + 24,
                        thousands(count),
                        size=10,
                        fill=BACKGROUND,
                        anchor="middle",
                    )
                )
            cursor += width
        parts.append(
            text_element(
                cursor + 10,
                y + 12,
                f"{thousands(arm.total_stored)} items",
                size=11,
                weight="600",
            )
        )
        parts.append(
            text_element(
                cursor + 10,
                y + 26,
                f"${arm.cost_usd:.3f} · {thousands(arm.total_write_calls)} calls",
                size=10.5,
                fill=MUTED,
            )
        )

    cheapest = min(arms, key=lambda arm: arm.cost_usd)
    dearest = max(arms, key=lambda arm: arm.cost_usd)
    smallest = min(arms, key=lambda arm: arm.total_stored)
    largest = max(arms, key=lambda arm: arm.total_stored)
    footer = (
        f"Identical input, {largest.total_stored / smallest.total_stored:.0f}x apart in what was "
        f"kept ({thousands(smallest.total_stored)} to {thousands(largest.total_stored)} items) and "
        f"{dearest.cost_usd / cheapest.cost_usd:.0f}x apart in what it cost to decide "
        f"(${cheapest.cost_usd:.3f} to ${dearest.cost_usd:.3f}). The raw transcript "
        f"({thousands(turn_count)} episodic turns) is excluded — it is the input, "
        f"not a decision. THIS IS AN ILLUSTRATION, NOT A MEASUREMENT: one conversation, one seed, "
        f"no accuracy anywhere on this chart. More stored is not better. "
        f"Generated by scripts/repro/demo_methodology_switch.py, $0, no model call."
    )
    for index, line in enumerate(wrap(footer, int((CHART_RIGHT - 40) / 4.9))):
        parts.append(text_element(40, HEIGHT - 76 + index * 15, line, size=11, fill=MUTED))
    parts.append(f'<rect x="0" y="{HEIGHT - 96}" width="{WIDTH}" height="1" fill="{GRID}"/>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# -------------------------------------------------------------------------------------- markdown


def render_markdown(arms: list[Arm], svg_path: Path, markdown_path: Path) -> str:
    """The page: one input, five configs, and what came out of each."""
    relative_svg = Path(svg_path).relative_to(markdown_path.parent)
    cheapest = min(arms, key=lambda arm: arm.cost_usd)
    dearest = max(arms, key=lambda arm: arm.cost_usd)
    smallest = min(arms, key=lambda arm: arm.total_stored)
    largest = max(arms, key=lambda arm: arm.total_stored)

    stored_rows = "\n".join(
        "| `{config}` | {label} | {kinds} | {total} | {calls} | {embed} | ${cost:.3f} |".format(
            config=arm.config,
            label=arm.label,
            kinds=", ".join(
                f"{thousands(count)} {kind}"
                for kind, count in sorted(arm.stored_by_type.items(), key=lambda kv: -kv[1])
            ),
            total=thousands(arm.total_stored),
            calls=thousands(arm.total_write_calls),
            embed=thousands(arm.embed_calls),
            cost=arm.cost_usd,
        )
        for arm in arms
    )

    op_rows = "\n".join(
        f"| `{arm.config}` | "
        + ", ".join(f"`{op}` {thousands(count)}" for op, count in arm.op_counts.items())
        + " |"
        for arm in arms
    )

    quoted_blocks = "\n\n".join(
        f"**{arm.label}** — `--config {arm.config}`\n\n"
        + "\n>\n".join(
            f"> *{item['memory_type']}* — {' '.join(item['content'].split())[:400]}"
            for item in arm.quoted
        )
        for arm in arms
        if arm.quoted
    )

    turn_count = thousands(turn_count_of(arms))
    arm_count = len(arms)
    arms_with_transcript = sum(1 for arm in arms if arm.raw_turns)
    transcript_rows = "\n".join(
        f"| `{arm.config}` | {thousands(arm.op_counts.get('ADD:episodic', 0))} | "
        f"{thousands(arm.raw_turns) if arm.raw_turns else '**0**'} |"
        for arm in arms
    )

    provenance_rows = "\n".join(
        f"| `{arm.config}` | `results/repro/{arm.stem}.json` | `{arm.commit}` | "
        f"{arm.ingest_seconds / 60:.0f} min |"
        for arm in arms
    )

    return f"""# Demo A — one conversation, five methodologies, one interface

> **This page is an illustration, not a measurement.** One conversation, one seed, and no accuracy
> number anywhere on it. The measurements live in [`docs/18-locomo-4way.md`](../18-locomo-4way.md)
> and [`docs/19-ace-finer.md`](../19-ace-finer.md), and they carry confidence intervals. **More
> stored is not better** — a system can keep {thousands(largest.total_stored)} items and answer
> worse than one that kept {thousands(smallest.total_stored)}. Nothing here is ranked.

Nine agentic-memory methodologies are implemented in this repository behind one API and selected by
name. What that buys is the ability to feed all of them the same input and *look*, instead of
arguing from papers. Below, five write paths over the same LoCoMo conversation — **{turn_count}
turns, `gpt-4o-mini`, `text-embedding-3-small`, one harness** — where the `--config` name is what
differs between the runs.

**Cost to produce this page: $0.** Nothing was ingested and no model was called. Every number is
read out of an ingest summary the campaign already paid for and committed.

```
uv run python scripts/repro/demo_methodology_switch.py
```

![What five memory methodologies stored from one identical conversation]({relative_svg})

## What each one kept

These counts are the **derived** items — what the methodology decided to write. The raw transcript
(`episodic`) is excluded, following the harness's own convention, because it is the input rather than
a decision. Every arm's op log records {turn_count} `ADD:episodic` operations for it.

| config | methodology | what it stored | items | write calls | embedding calls | cost |
|---|---|---|---|---|---|---|
{stored_rows}

**{largest.total_stored / smallest.total_stored:.0f}× apart in what was kept and
{dearest.cost_usd / cheapest.cost_usd:.0f}× apart in what it cost to decide, on identical input.**
`{smallest.config}` kept {thousands(smallest.total_stored)} items for ${smallest.cost_usd:.3f};
`{largest.config}` kept {thousands(largest.total_stored)} for ${largest.cost_usd:.3f}. They are not
doing better or worse jobs of the same task — they are doing different tasks, and that is the point
of keeping them separable rather than blending them into one "memory layer".

Embedding calls are listed apart from generative ones on purpose. They are priced three orders of
magnitude differently, and a single "LLM calls" column that silently merges them is how one of this
repository's own cost claims went wrong ([ledger C-entry on MemoryOS](../17-defect-ledger.md)).

### An inconsistency this page found and will not paper over

Every arm's op log records {turn_count} `ADD:episodic` operations, but the raw transcript is present
in only {arms_with_transcript} of the {arm_count} memory snapshots:

| config | `ADD:episodic` in the op log | `episodic` rows in the snapshot |
|---|---|---|
{transcript_rows}

The two arms missing it are exactly the two that override the doc store to `PostgresDocStore`
(`configs.NEMORI_STORE`), while the others use the default — so the likeliest reading is that the
snapshot writer's `list_episodes` came back empty against that backend, not that the episodes were
never written. **That is a guess, and it is left as one.** Nothing on this page depends on it: the
counts above are derived items only, and the op log is the durable record either way. It is recorded
here because a demo that noticed an artifact disagreeing with itself and said nothing would be worth
less than no demo.

## The operations behind those counts

The op log is the durable record — every mutation is appended before it is applied — so the *verbs*
each methodology uses are recoverable, not inferred:

| config | ops |
|---|---|
{op_rows}

Read the verbs, not just the totals. `{arms[1].config}` is the only arm here that **deletes**;
Zep is the only one that builds `entities` and `communities` alongside facts; the two Nemori arms
differ *only* in a merge threshold and that shows up as
{arms[2].op_counts.get("MERGE:episodes", 0)} merges against
{arms[3].op_counts.get("MERGE:episodes", 0)}.

## Five shapes of the same subject matter

All five arms wrote about the support group Caroline attends in this conversation. Below is the first
item of each kind that mentions `{ANCHOR}`, in snapshot order, verbatim:

{quoted_blocks}

**These are not five records of one event, and the page will not claim they are.** The filter is
topical, the pick is first-match-in-file-order, and the dates give it away — one arm's first matching
episode is from a different day than another's. Selecting harder until five sentences lined up would
mean choosing what looked good, which is worse than saying this plainly.

What the quotes do support is the claim they were assembled for: **these are different kinds of
object.** A turn kept whole; a natural-language sentence; a dict with a `date` field and, depending on
the arm, `description` or `impact`; a narrated multi-turn episode; a subject-predicate-object fact;
an entity with a gloss. That is the argument for pinning lineage instead of shipping one blended
"memory layer" — a benchmark number attached to "memory" without saying which of these was built is
not attached to anything.

## Provenance

| config | ingest summary | repo commit at run | ingest wall-clock |
|---|---|---|---|
{provenance_rows}

**The commits differ, and that is a real limitation of this page.** These arms were ingested on
different days as the campaign progressed, so this is not a controlled five-way comparison of one
codebase — it is five runs that shared a conversation, a model and an embedder. The controlled
comparison is [`docs/18-locomo-4way.md`](../18-locomo-4way.md), which is also where the accuracy
numbers and their intervals are.

**ACE is missing on purpose.** Its arm in this campaign runs on FiNER, a tagging benchmark, not on a
conversation; putting it in this table would compare what it stored about a different input. It has
its own page: [cost-is-tokens.md](cost-is-tokens.md).

Item counts and quoted text are read from each run's `*.memory.jsonl` snapshot, which is gitignored
for size (docs/14 §Artifacts). The derived index is committed as
`results/repro/demo_methodology_switch.snapshot.json` so this page regenerates from a fresh clone;
`--rescan` rebuilds it where the snapshots exist.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-svg", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="re-read the memory snapshots instead of using the committed index",
    )
    args = parser.parse_args()

    arms = collect(rescan=args.rescan)
    args.out_svg.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    turn_count = turn_count_of(arms)
    args.out_svg.write_text(render_svg(arms, turn_count))
    args.out_md.write_text(reflow(render_markdown(arms, args.out_svg, args.out_md)))

    print(f"wrote {args.out_svg.relative_to(REPO_ROOT)}")
    print(f"wrote {args.out_md.relative_to(REPO_ROOT)}")
    for arm in arms:
        print(
            f"  {arm.config:20} {arm.total_stored:5} items  "
            f"{arm.total_write_calls:5} write calls  ${arm.cost_usd:.3f}"
        )


if __name__ == "__main__":
    main()
