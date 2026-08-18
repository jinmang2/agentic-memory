"""Did ACE's playbook buy anything on FiNER? Paired, against the arm that never learned.

The online arm's window curve cannot answer this on its own, and that is the
whole reason the base arm exists. Each online window is scored on a DIFFERENT
15 questions, so a window-to-window change confounds adaptation with sample
difficulty — and the base arm, which never learns anything, swings 35.0 to 65.0
across its own windows. Upstream's online mode reads its curve exactly that way
(ace.py:939-997), and its 69.1 -> 81.9 claim inherits the confound.

Here every arm answers the SAME questions in the SAME order with the SAME model,
differing only in what was being grown alongside. So each comparison is
per-question paired against `base`, and the difference vector is what gets
resampled.

Three arms, and the third one is the point of the third:

  base      no organizer at all. The reference every comparison is taken against.
  online    our always-on curator dedup at cosine 0.90 (deviation D5).
  nodedup   dedup off, which is upstream's shipped default (ledger B-6). It
            exists because a null result under OUR dedup cannot distinguish
            "adaptation does not transfer" from "our dedup threw the adaptation
            away" — 276 of 441 curated bullets were discarded in the online arm.

An arm may be SHORTER than base: `--max-spend-usd` stops a run cleanly between
windows, so a capped arm ends on a whole-window boundary. Comparisons then run
on the common prefix, which is a valid pairing because records are written in
split order and resumes keep only whole windows — the prefix is the same
questions in the same order for both sides. The n actually compared is printed
and stamped, never silently the full 441.

Statistics are IMPORTED from `scripts/ext/x1_power.py`, never restated — a
second confidence-interval implementation is how two numbers in one repository
come to disagree about what "95%" means. That import decides what can be tested
here, and the limit is stated rather than worked around:

  sample_accuracy  every question is one boolean (all four tags right, upstream's
                   `answer_is_correct`), which is exactly what `paired_delta_ci`
                   takes. Tested.
  tag_accuracy     upstream's PUBLISHED metric, and a per-question RATE (k of 4),
                   not a boolean. The correct resampling unit is still the
                   question — the four tags of one sample share a filing excerpt
                   and are not independent — so the right test is the same
                   cluster bootstrap over a float statistic. `paired_delta_ci`
                   casts its inputs to bool, so it cannot run it, and writing a
                   float twin here would be the duplicate this module refuses.
                   Reported as a point estimate with its per-window sign record,
                   and NOT as an interval.

Two supporting counts, because a null result invites the question of whether
anything happened at all: how often the two arms disagreed per question, and how
much playbook the learning arm was carrying when it did.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RECORDS = ROOT / "results" / "repro"

REFERENCE = "base"
ARMS = {
    "base": "gpt-4o-mini_ace_finer_base",
    "online": "gpt-4o-mini_ace_finer_online",
    "nodedup": "gpt-4o-mini_ace_finer_nodedup",
    "retry": "gpt-4o-mini_ace_finer_retry",
    # Track 4. A SECOND methodology, not a fifth ACE setting: ReasoningBank over
    # the same 441 questions in the same order at the same model and embedder,
    # differing in what was grown alongside and how much of it is read back
    # (whole playbook vs top-1). It is here because "is the null ACE's or the
    # task's?" is the same paired test against `base` that every row above is.
    #
    # NOT a reproduction of ReasoningBank. Its published claims are agentic —
    # WebArena and SWE-Bench, both unreachable on this machine — so this arm
    # measures the mechanism on a single-turn task and its artifact carries
    # `RB_D1_not_the_published_benchmark_...` saying so.
    "rb": "gpt-4o-mini_rb_finer_online",
}


def _load_x1():
    path = ROOT / "scripts" / "ext" / "x1_power.py"
    if not path.exists():
        raise SystemExit(f"missing {path} — the statistics live there and are not restated here")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("x1_power", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_rows(stem: str) -> list[dict]:
    path = RECORDS / f"{stem}.records.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_summary(stem: str) -> dict:
    return json.loads((RECORDS / f"{stem}.json").read_text())


def llm_calls(stem: str) -> int:
    """Generator + reflector/curator calls, counted off the trace.

    NOT `sum(llm_budget[*].calls)` from the summary, and the difference is a
    trap: a resumed run's `mem.budget` starts at zero, so its summary records
    only what the LAST process spent — the completed nodedup arm files 164 calls
    against roughly 1,325 actually bought. `cost_usd` in the same file IS the
    measurement total (the runner folds in prior spend recovered from the
    trace), so the two fields of one artifact disagree about scope. The trace is
    appended per call across every process of a run, which makes it the only
    per-arm record with one meaning.

    Two properties of this count travel with it. Embedding calls are absent —
    the trace carries chat completions, so the summary's larger totals include
    an embedder this does not; and calls made by attempts the host killed are
    included, because they were paid for even though their windows were
    re-answered.
    """
    path = RECORDS / f"{stem}.llm-trace.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def anchor(name: str, arm: list[dict], summary: dict) -> dict:
    """Recompute an arm's headline from its own records, or stop.

    Fail closed on the headlines, as the Zep sweep does: an arm whose accuracy
    does not recompute from its own records is not the run it claims to be.
    """
    overall = summary["overall"]
    tag = round(100 * sum(r["correct_tags"] for r in arm) / sum(r["total_tags"] for r in arm), 2)
    sample = round(100 * sum(1 for r in arm if r["is_correct"]) / len(arm), 2)
    if (tag, sample) != (overall["tag_accuracy"], overall["sample_accuracy"]):
        raise SystemExit(
            f"{name}: records recompute to tag {tag} / sample {sample}, "
            f"summary says {overall['tag_accuracy']} / {overall['sample_accuracy']} — STOP"
        )
    return {
        "tag_accuracy": tag,
        "sample_accuracy": sample,
        "n": len(arm),
        "complete": summary["stamp"].get("complete", True),
    }


def compare(name: str, arm: list[dict], base: list[dict], x1, n_boot: int, seed: int) -> dict:
    """One arm against the reference, on the questions they both answered."""
    n = min(len(arm), len(base))

    # The pairing assertion. Everything below is ordered by it: if row i of one
    # arm is not the same question as row i of the other, the difference vector
    # is meaningless while still producing a plausible-looking interval.
    #
    # `index` is checked and not only `target`, because FiNER's prompt template
    # is constant across the split — every record carries the SAME `question`
    # string, with the filing excerpt varying inside it. An assertion written on
    # question text therefore proves nothing, and one written on the gold tags
    # alone would accept two different samples that happen to share a target.
    for i, (b, o) in enumerate(zip(base[:n], arm[:n])):
        if b["index"] != i or o["index"] != i or b["target"] != o["target"]:
            raise SystemExit(f"{name}: row {i} is not the sample base answered at that position")

    ci = x1.paired_delta_ci(
        [r["is_correct"] for r in arm[:n]],
        [r["is_correct"] for r in base[:n]],
        n_boot=n_boot,
        seed=seed,
    )

    # Both headlines are recomputed ON THE PREFIX. Reusing the full-run anchors
    # here would compare an arm's 441 questions against base's first 375.
    def tag_of(rows: list[dict]) -> float:
        return round(
            100 * sum(r["correct_tags"] for r in rows) / sum(r["total_tags"] for r in rows), 2
        )

    tag_delta = round(tag_of(arm[:n]) - tag_of(base[:n]), 2)
    per_q = [
        (o["correct_tags"] - b["correct_tags"]) / b["total_tags"] for b, o in zip(base[:n], arm[:n])
    ]
    moved = sum(1 for d in per_q if d != 0)

    return {
        "n_compared": n,
        "prefix_of_base": n < len(base),
        "base_tag_accuracy_on_prefix": tag_of(base[:n]),
        "arm_tag_accuracy_on_prefix": tag_of(arm[:n]),
        "sample_accuracy_paired": ci,
        "tag_accuracy_point_delta_pp": tag_delta,
        "tag_accuracy_interval": None,
        "n_questions_with_tag_movement": moved,
    }


def window_table(name: str, summary: dict, base_summary: dict) -> dict:
    windows = [
        {
            "window": wb["window"],
            "base_tag": wb["tag_accuracy"],
            "arm_tag": wo["tag_accuracy"],
            "delta_tag": round(wo["tag_accuracy"] - wb["tag_accuracy"], 2),
            "playbook_chars": wo["playbook_chars_at_test"],
        }
        for wb, wo in zip(base_summary["per_window"], summary["per_window"])
    ]
    wins = sum(1 for w in windows if w["delta_tag"] > 0)
    losses = sum(1 for w in windows if w["delta_tag"] < 0)

    # Second half vs first: if adaptation needs a playbook to accumulate, the
    # effect should be larger once one has. Descriptive, for the same reason as
    # the tag metric above — a dozen windows a side is not an interval.
    half = len(windows) // 2
    first = sum(w["delta_tag"] for w in windows[:half]) / half if half else 0.0
    second = sum(w["delta_tag"] for w in windows[half:]) / max(1, len(windows) - half)

    return {
        "per_window": windows,
        "window_sign": {"arm_ahead": wins, "behind": losses, "tied": len(windows) - wins - losses},
        "mean_delta_tag_first_half": round(first, 2),
        "mean_delta_tag_second_half": round(second, 2),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=RECORDS / "finer_paired.json")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--arms",
        nargs="*",
        default=None,
        help=f"arms to compare against the reference (default: all but {REFERENCE})",
    )
    # The question "did learning help?" is asked against base, but "did our
    # dedup cost anything?" is asked between the two learning arms, and that is
    # the same paired test with a different reference — not a different module.
    ap.add_argument("--reference", default=REFERENCE, choices=sorted(ARMS))
    args = ap.parse_args(argv)

    x1 = _load_x1()
    reference = args.reference

    wanted = args.arms or [n for n in ARMS if n != reference]
    unknown = [n for n in wanted if n not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s): {', '.join(unknown)}")

    rows = {reference: load_rows(ARMS[reference])}
    summaries = {reference: load_summary(ARMS[reference])}
    for name in wanted:
        rows[name] = load_rows(ARMS[name])
        summaries[name] = load_summary(ARMS[name])

    anchors = {name: anchor(name, rows[name], summaries[name]) for name in rows}
    for name, a in anchors.items():
        state = "" if a["complete"] else "  (capped, incomplete)"
        print(
            f"{name:8s} n={a['n']:4d}  tag {a['tag_accuracy']:5.2f}  sample {a['sample_accuracy']:5.2f}{state}"
        )

    comparisons = {}
    for name in wanted:
        cmp = compare(name, rows[name], rows[reference], x1, args.n_boot, args.seed)
        cmp.update(window_table(name, summaries[name], summaries[reference]))
        comparisons[name] = cmp

        ci = cmp["sample_accuracy_paired"]
        verdict = "SEPARATED" if ci["excludes_zero"] else "NOT separated"
        scope = f"n={cmp['n_compared']}" + (" prefix" if cmp["prefix_of_base"] else "")
        print(
            f"\n{name} - {reference}  [{scope}]\n"
            f"  sample_accuracy  d={ci['delta_pp']:+6.2f}pp  "
            f"95% CI [{ci['lo']:+.2f}, {ci['hi']:+.2f}]  p={ci['p_boot']:.4f}  "
            f"disagree={ci['n_disagree']}/{ci['n']}  {verdict}"
        )
        print(
            f"  tag_accuracy     d={cmp['tag_accuracy_point_delta_pp']:+6.2f}pp  "
            f"({cmp['base_tag_accuracy_on_prefix']:.2f} -> "
            f"{cmp['arm_tag_accuracy_on_prefix']:.2f}, point estimate, no interval: "
            f"clustered rate, see module docstring)  "
            f"moved={cmp['n_questions_with_tag_movement']}/{cmp['n_compared']}"
        )
        sign = cmp["window_sign"]
        print(
            f"  per-window sign: ahead in {sign['arm_ahead']}, behind in {sign['behind']}, "
            f"tied in {sign['tied']} of {len(cmp['per_window'])}   "
            f"mean delta_tag first half {cmp['mean_delta_tag_first_half']:+.2f}pp / "
            f"second half {cmp['mean_delta_tag_second_half']:+.2f}pp"
        )

    cost = {name: summaries[name].get("cost_usd") for name in rows}
    calls = {name: llm_calls(ARMS[name]) for name in rows}
    print(
        "\ncost  "
        + "  ".join(f"{name} ${cost[name]:.3f}" for name in rows)
        + "   LLM calls  "
        + "  ".join(f"{name} {calls[name]}" for name in rows)
    )

    args.out.write_text(
        json.dumps(
            {
                "reference": reference,
                "anchors": anchors,
                "n_boot": args.n_boot,
                "seed": args.seed,
                "tag_accuracy_note": (
                    "Upstream's published metric is a per-question RATE over four "
                    "clustered tags. The correct test is a cluster bootstrap over a "
                    "float statistic; x1_power.paired_delta_ci takes booleans only, and "
                    "this module does not write a second confidence interval. Point "
                    "estimate and per-window signs only."
                ),
                "comparisons": comparisons,
                "cost_usd": cost,
                "llm_calls": calls,
                "llm_calls_note": (
                    "Generator + reflector/curator calls counted off each arm's trace, "
                    "which is the only per-arm record that spans every process of a run. "
                    "The summaries' `llm_budget` is per-process and undercounts a resumed "
                    "arm; it also includes embedder calls, which the trace does not. Calls "
                    "from attempts the host killed are included — they were paid for."
                ),
            },
            indent=2,
        )
    )
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
