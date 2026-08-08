"""Rebuild a run's per-question records from its LLM trace.

For the case where a run FINISHED measuring and then died before writing its
artifacts. That happened to `zep_mmr`: 1,986 questions answered, 1,540 judged,
J 40.78 / F1 27.08 / BLEU-1 22.84 printed — and then `json.dumps` hit an
`np.float32` the encoder had no rule for and took the records sidecar and the
summary down with it. Every model call was already paid for.

This is a RECONSTRUCTION, not a re-measurement. It issues no model calls. Every
answer and every verdict is read back out of the trace exactly as the model
returned them; only the lexical scores are recomputed, by calling the same
`bench.locomo` scorers the live path calls.

What makes it checkable rather than plausible: the run printed its overall
metrics before dying, so `--expect` re-derives them from the reconstruction and
refuses to write anything unless they match to the last decimal. A
reconstruction that cannot reproduce the number the run announced is a bug in
this script, and it should not be allowed to leave an artifact behind.

What it CANNOT recover, and what every consumer of the output must know: the
structured `retrieval` capture (item ids, scores and types per question). The
trace records prompts and responses, so the served memory TEXT survives inside
the generate prompt, but the id/score triples were never in it. Reconstructed
records therefore carry no `retrieval` key, and the summary stamps
`reconstructed_from_trace` so no reader mistakes this row for one whose capture
was merely empty.

Usage:
    uv run python scripts/repro/reconstruct_from_trace.py \
        --trace results/repro/<tag>.llm-trace.jsonl.rewriteon-<ts> \
        --out-tag <tag> \
        --expect-j 40.78 --expect-f1 27.08 --expect-bleu1 22.84
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from agmem.bench import locomo

DATA = Path.home() / ".agmem/datasets/locomo10.json"
OUT = Path(__file__).resolve().parent.parent.parent / "results" / "repro"

# The generate prompt ends with the question and the answer cue. Anchored on
# both sides: a question containing the word "Question:" would otherwise let a
# greedy match swallow the wrong span. Verified to extract 1,986 of 1,986 on the
# run this script was written for, including the 445 category-5 turns whose
# prompt is built from a different template.
_QUESTION_RE = re.compile(r"\nQuestion: (.+?)\nShort answer:", re.S)
# The judge prompt restates the gold answer; used only to CHECK the dataset join,
# never as the source of gold (the dataset is the authority both paths read).
_JUDGE_GOLD_RE = re.compile(r"\nGold answer: (.+?)\nGenerated answer:", re.S)


def parse_trace(path: Path) -> list[dict[str, Any]]:
    """Walk the trace in order, pairing each generate call with the judge call
    that follows it.

    Relies on the run being sequential (`--workers 1`, the default), which is
    what makes trace order equal question order. The pairing is checked rather
    than assumed: a judge call that does not follow a generate call, or a second
    judge for one question, raises instead of silently shifting every subsequent
    verdict onto the wrong question — an off-by-one here would corrupt the whole
    reconstruction while still producing plausible-looking output.
    """
    turns: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.open(encoding="utf-8"), 1):
        call = json.loads(line)
        role = call.get("role")
        if role == "generate":
            match = _QUESTION_RE.search(call["messages"][-1]["content"])
            if not match:
                raise SystemExit(f"{path}:{lineno}: generate prompt has no question")
            turns.append(
                {
                    "q_from_prompt": match.group(1).strip(),
                    "pred": call.get("response_text") or "",
                    "judge_raw": None,
                    "judge_gold": None,
                }
            )
        elif role == "judge":
            if not turns:
                raise SystemExit(f"{path}:{lineno}: judge call with no preceding generate")
            if turns[-1]["judge_raw"] is not None:
                raise SystemExit(f"{path}:{lineno}: second judge for one question")
            turns[-1]["judge_raw"] = call.get("response_text") or ""
            gold = _JUDGE_GOLD_RE.search(call["messages"][-1]["content"])
            turns[-1]["judge_gold"] = gold.group(1).strip() if gold else None
    return turns


def verdict(raw: str | None) -> bool | None:
    """Parse a judge reply the way `locomo.judge_answer` does — CORRECT, and
    nothing else, is True. Kept identical on purpose: a reconstruction that
    graded 'correct.' or 'Yes' as True would silently score higher than the run
    it claims to restore."""
    if raw is None:
        return None
    try:
        label = json.loads(raw).get("label", "")
    except (json.JSONDecodeError, AttributeError):
        return None
    return str(label).strip().upper() == "CORRECT"


def rebuild(turns: list[dict[str, Any]], convs: list[int]) -> tuple[list[dict], list[dict]]:
    """Join the parsed turns onto the dataset's questions, in the same order and
    through the same helpers the live path uses (`select_questions`, `gold_for`,
    `token_f1`, `bleu1`). Returns (records, per_conv_aggregates)."""
    samples = locomo.load_locomo(DATA)
    records: list[dict[str, Any]] = []
    per_conv: list[dict[str, Any]] = []
    cursor = 0
    for conv in convs:
        questions = locomo.select_questions(samples[conv])
        per_cat: dict[str, list[dict[str, Any]]] = {}
        for question in questions:
            if cursor >= len(turns):
                raise SystemExit(
                    f"trace ran out at conv{conv}: {len(turns)} generate calls for "
                    f"{cursor + 1}+ questions — the trace is from a partial run"
                )
            turn = turns[cursor]
            cursor += 1
            # The alignment assertion. Everything downstream is ordered by this
            # join, so if the Nth generate call is not the Nth question the run
            # answered, the reconstruction is worthless and must not be written.
            if turn["q_from_prompt"] != question["question"].strip():
                raise SystemExit(
                    f"conv{conv} question {cursor}: trace/dataset mismatch\n"
                    f"  trace:   {turn['q_from_prompt'][:120]!r}\n"
                    f"  dataset: {question['question'][:120]!r}"
                )
            gold = locomo.gold_for(question)
            cat_num = question.get("category")
            pred = turn["pred"]
            f1, b1 = locomo.token_f1(pred, gold), locomo.bleu1(pred, gold)
            j = verdict(turn["judge_raw"]) if cat_num in (1, 2, 3, 4) else None
            cat = locomo.CATEGORY_NAMES.get(cat_num, "?")
            row = {
                "q": question["question"],
                "gold": gold,
                "pred": pred,
                "cat": cat,
                "f1": round(f1, 3),
            }
            if j is not None:
                row["j"] = j
            records.append({"run": 1, "conv": conv, **row})
            per_cat.setdefault(cat, []).append({"f1": f1, "b1": b1, "j": j})
        rows = [cell for cells in per_cat.values() for cell in cells]
        per_conv.append(
            {
                "conv": conv,
                "overall": locomo.aggregate_cells(rows),
                "by_category": {
                    cat: locomo.aggregate_cells(cells) for cat, cells in sorted(per_cat.items())
                },
            }
        )
    if cursor != len(turns):
        raise SystemExit(f"trace has {len(turns)} generate calls but only {cursor} were consumed")
    return records, per_conv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out-tag", required=True, help="artifact tag to write under results/repro")
    ap.add_argument("--convs", default="all")
    ap.add_argument("--expect-j", type=float, default=None)
    ap.add_argument("--expect-f1", type=float, default=None)
    ap.add_argument("--expect-bleu1", type=float, default=None)
    args = ap.parse_args()

    convs = list(range(10)) if args.convs == "all" else [int(c) for c in args.convs.split(",")]
    turns = parse_trace(Path(args.trace))
    records, per_conv = rebuild(turns, convs)

    # combine_aggs lives in the runner; import it there rather than restating the
    # micro-average, for the same reason aggregate_cells was promoted.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from exp_amem_repro import combine_aggs  # noqa: PLC0415

    overall = combine_aggs([c["overall"] for c in per_conv])

    checks = [
        ("j_score", args.expect_j, overall.get("j_score")),
        ("f1", args.expect_f1, overall.get("f1")),
        ("bleu1", args.expect_bleu1, overall.get("bleu1")),
    ]
    failures = [(name, want, got) for name, want, got in checks if want is not None and want != got]
    for name, want, got in checks:
        if want is not None:
            print(
                f"[check] {name}: expected {want} got {got} {'OK' if want == got else 'MISMATCH'}"
            )
    if failures:
        raise SystemExit(
            "reconstruction does not reproduce the run's announced metrics: "
            + ", ".join(f"{n} want {w} got {g}" for n, w, g in failures)
            + "\nNothing was written."
        )

    records_path = OUT / f"{args.out_tag}.records.jsonl"
    with records_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[done] wrote {records_path} ({len(records)} question records)")

    summary_path = OUT / f"{args.out_tag}.reconstructed.json"
    summary_path.write_text(
        json.dumps(
            {
                "reconstructed_from_trace": str(args.trace),
                "provenance": (
                    "records rebuilt from the run's LLM trace after the run measured "
                    "successfully and died writing artifacts; no model calls were made. "
                    "The structured per-question `retrieval` capture is NOT recoverable "
                    "from a trace and is absent from these records."
                ),
                "verified_against": {n: w for n, w, _ in checks if w is not None},
                "overall": overall,
                "per_conv": per_conv,
                "n_records": len(records),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
