"""Rebuild a killed FiNER run's per-window records from its own LLM trace.

The second time this campaign has needed it, and the reason it is a script now
rather than a one-off: the host reboots, the process leaves no traceback, and
everything held in memory is gone — while the trace, appended per call, is
intact and already paid for.

It issues no model calls. Answers are read back out of the trace exactly as the
model returned them; only the tag scoring is recomputed, through the same
`bench.finer` functions the live path calls.

**Two invariants decide where the rebuild is allowed to stop, and they are not
the same place.**

- A sample is *answered* when its generate call is in the trace.
- A sample is *adapted* when both of its distill calls (reflect, curate) are.

`run_arm` answers a whole window, then adapts on it, so a crash lands between
the two more often than not. Records may only cover ANSWERED samples up to a
window boundary — those are the rows a resume will trust — and any adapt
deficit inside the last kept window is a real, disclosable difference: those
samples were scored but never contributed to the playbook.

What is NOT at risk is leakage. Every sample is answered before any adapt
derived from it runs, so a rebuilt row can never have been produced by a
playbook that had already seen its gold answer. That ordering is why resuming
past the deficit is sound and resuming *before* it would not be — re-answering
a sample the store has already learned from is the one thing this must not do.

Usage:
    uv run python scripts/repro/finer_records_from_trace.py \\
        --tag gpt-4o-mini_ace_finer_nodedup --window 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "repro"
UPSTREAM_DATA = Path.home() / ".agmem/upstream/ace/eval/finance/data"
sys.path.insert(0, str(ROOT / "src"))

from agmem.bench import finer

_Q_OPEN = "**Question:**\n"
_Q_CLOSE = "\n\n**Context:**"
_PB_OPEN = "**Playbook:**\n"
_PB_CLOSE = "\n\n**Reflection:**"


def parse_trace(path: Path) -> tuple[list[dict], int]:
    """Answered attempts in order, and the count of distill calls beside them."""
    answers: list[dict] = []
    distills = 0
    for lineno, line in enumerate(path.open(encoding="utf-8"), 1):
        try:
            call = json.loads(line)
        except json.JSONDecodeError:
            # A crash can land mid-write; everything after is unusable.
            print(f"[warn] {path.name}:{lineno} truncated — stopping there", file=sys.stderr)
            break
        role = call.get("role")
        if role == "distill":
            distills += 1
            continue
        if role != "generate":
            continue
        prompt = call["messages"][-1]["content"]
        if _Q_OPEN not in prompt or _Q_CLOSE not in prompt:
            raise SystemExit(f"{path.name}:{lineno}: generate prompt is not the FiNER template")
        question = prompt.split(_Q_OPEN, 1)[1].split(_Q_CLOSE, 1)[0]
        playbook = ""
        if _PB_OPEN in prompt and _PB_CLOSE in prompt:
            playbook = prompt.split(_PB_OPEN, 1)[1].split(_PB_CLOSE, 1)[0]
        reply = call.get("response_text") or ""
        try:
            parsed = json.loads(reply)
        except json.JSONDecodeError:
            parsed = {}
        pred = str(parsed.get("final_answer", "")).strip() or finer.extract_answer(reply)
        answers.append(
            {
                "question": question,
                "pred": pred,
                "reasoning": str(parsed.get("reasoning", "")),
                "bullet_ids": [str(b) for b in (parsed.get("bullet_ids") or [])],
                "playbook_chars": len(playbook),
            }
        )
    return answers, distills


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--data", type=Path, default=UPSTREAM_DATA)
    ap.add_argument("--split", default="test")
    ap.add_argument("--dry", action="store_true", help="report and write nothing")
    args = ap.parse_args()

    samples, _ = finer.process_task_data(finer.load_finer(args.data, args.split))
    answers, distills = parse_trace(OUT / f"{args.tag}.llm-trace.jsonl")

    keep = (len(answers) // args.window) * args.window
    adapted = distills // 2
    print(f"trace: {len(answers)} answered, {adapted} adapted ({distills} distill calls)")
    print(f"keeping {keep} rows ({keep // args.window} whole windows of {args.window})")
    if adapted < keep:
        print(
            f"[disclose] {keep - adapted} of the kept samples were scored but never adapted on — "
            "the playbook is that much thinner than an uninterrupted run's"
        )
    if adapted > keep:
        # The store would be ahead of the records, so resuming at `keep` would
        # re-answer samples the playbook has already learned from. Refuse.
        raise SystemExit(
            f"REFUSED: the store adapted on {adapted} samples but only {keep} rows are keepable. "
            "Resuming would re-answer samples the playbook has already seen. Start clean instead."
        )

    rows = []
    for i in range(keep):
        got, want = answers[i]["question"], samples[i]["question"]
        if got.strip() != want.strip():
            raise SystemExit(f"row {i}: trace question does not match the dataset's")
        row = finer.score_sample(samples[i], answers[i]["pred"])
        row["window"] = i // args.window
        row["index"] = i
        row["playbook_chars"] = answers[i]["playbook_chars"]
        row["bullet_ids"] = answers[i]["bullet_ids"]
        row["reasoning"] = answers[i]["reasoning"]
        row["rebuilt_from_trace"] = True
        rows.append(row)

    agg = finer.aggregate(rows)
    print(f"rebuilt: tag={agg['tag_accuracy']} sample={agg['sample_accuracy']} n={agg['n']}")
    if args.dry:
        print("[dry] nothing written")
        return
    path = OUT / f"{args.tag}.records.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"[done] wrote {path} ({len(rows)} rows) — `--resume` will continue from sample {keep}")


if __name__ == "__main__":
    main()
