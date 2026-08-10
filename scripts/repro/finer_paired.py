"""Did ACE's playbook buy anything on FiNER? Paired, on the same 441 samples.

The online arm's window curve cannot answer this on its own, and that is the
whole reason the base arm exists. Each online window is scored on a DIFFERENT
15 questions, so a window-to-window change confounds adaptation with sample
difficulty — and the base arm, which never learns anything, swings 35.0 to 65.0
across its own windows. Upstream's online mode reads its curve exactly that way
(ace.py:939-997), and its 69.1 -> 81.9 claim inherits the confound.

Here the two arms answer the SAME questions in the SAME order with the SAME
model, differing only in whether a playbook was being grown alongside. So the
comparison is per-question paired, and the difference vector is what gets
resampled.

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
much playbook the online arm was carrying when it did.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RECORDS = ROOT / "results" / "repro"

ARMS = {
    "base": "gpt-4o-mini_ace_finer_base",
    "online": "gpt-4o-mini_ace_finer_online",
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=RECORDS / "finer_paired.json")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    x1 = _load_x1()
    rows = {name: load_rows(stem) for name, stem in ARMS.items()}

    # The pairing assertion. Everything below is ordered by it: if row i of one
    # arm is not the same question as row i of the other, the difference vector
    # is meaningless while still producing a plausible-looking interval.
    base, online = rows["base"], rows["online"]
    if len(base) != len(online):
        raise SystemExit(f"arms differ in length: base {len(base)}, online {len(online)}")
    for i, (b, o) in enumerate(zip(base, online)):
        if b["target"] != o["target"] or b["question"] != o["question"]:
            raise SystemExit(f"row {i}: the two arms did not answer the same question")

    # Fail closed on the headlines, as the Zep sweep does: an arm whose accuracy
    # does not recompute from its own records is not the run it claims to be.
    anchors = {}
    for name, arm in rows.items():
        summary = json.loads((RECORDS / f"{ARMS[name]}.json").read_text())["overall"]
        tag = round(
            100 * sum(r["correct_tags"] for r in arm) / sum(r["total_tags"] for r in arm), 2
        )
        sample = round(100 * sum(1 for r in arm if r["is_correct"]) / len(arm), 2)
        if (tag, sample) != (summary["tag_accuracy"], summary["sample_accuracy"]):
            raise SystemExit(
                f"{name}: records recompute to tag {tag} / sample {sample}, "
                f"summary says {summary['tag_accuracy']} / {summary['sample_accuracy']} — STOP"
            )
        anchors[name] = {"tag_accuracy": tag, "sample_accuracy": sample, "n": len(arm)}

    # --- the tested claim: sample-level, boolean, x1's own bootstrap ---------
    ci = x1.paired_delta_ci(
        [r["is_correct"] for r in online],
        [r["is_correct"] for r in base],
        n_boot=args.n_boot,
        seed=args.seed,
    )
    verdict = "SEPARATED" if ci["excludes_zero"] else "NOT separated"
    print(
        f"sample_accuracy  online - base  d={ci['delta_pp']:+6.2f}pp  "
        f"95% CI [{ci['lo']:+.2f}, {ci['hi']:+.2f}]  p={ci['p_boot']:.4f}  "
        f"disagree={ci['n_disagree']}/{ci['n']}  {verdict}"
    )

    # --- the published metric: point estimate only, and labelled as such ------
    tag_delta = anchors["online"]["tag_accuracy"] - anchors["base"]["tag_accuracy"]
    per_q = [
        (o["correct_tags"] - b["correct_tags"]) / b["total_tags"] for b, o in zip(base, online)
    ]
    moved = sum(1 for d in per_q if d != 0)
    print(
        f"tag_accuracy     online - base  d={tag_delta:+6.2f}pp  (point estimate, no interval: "
        f"clustered rate, see module docstring)  moved={moved}/{len(per_q)}"
    )

    summaries = {name: json.loads((RECORDS / f"{ARMS[name]}.json").read_text()) for name in ARMS}
    windows = [
        {
            "window": wb["window"],
            "base_tag": wb["tag_accuracy"],
            "online_tag": wo["tag_accuracy"],
            "delta_tag": round(wo["tag_accuracy"] - wb["tag_accuracy"], 2),
            "playbook_chars": wo["playbook_chars_at_test"],
        }
        for wb, wo in zip(summaries["base"]["per_window"], summaries["online"]["per_window"])
    ]
    wins = sum(1 for w in windows if w["delta_tag"] > 0)
    losses = sum(1 for w in windows if w["delta_tag"] < 0)
    print(
        f"per-window sign: online ahead in {wins}, behind in {losses}, tied in "
        f"{len(windows) - wins - losses} of {len(windows)}"
    )

    # Second half vs first: if adaptation needs a playbook to accumulate, the
    # effect should be larger once one has. Descriptive, for the same reason as
    # above — 15 windows a side is not an interval.
    half = len(windows) // 2
    first = sum(w["delta_tag"] for w in windows[:half]) / half
    second = sum(w["delta_tag"] for w in windows[half:]) / (len(windows) - half)
    print(f"mean delta_tag  first half {first:+.2f}pp   second half {second:+.2f}pp")

    cost = {name: summaries[name].get("cost_usd") for name in ARMS}
    calls = {name: sum(v["calls"] for v in summaries[name]["llm_budget"].values()) for name in ARMS}
    print(
        f"cost  base ${cost['base']:.3f}  online ${cost['online']:.3f}  "
        f"({cost['online'] / cost['base']:.1f}x)   calls {calls['base']} -> {calls['online']}"
    )

    args.out.write_text(
        json.dumps(
            {
                "anchors": anchors,
                "n_boot": args.n_boot,
                "seed": args.seed,
                "sample_accuracy_paired": ci,
                "tag_accuracy_point_delta_pp": round(tag_delta, 2),
                "tag_accuracy_interval": None,
                "tag_accuracy_note": (
                    "Upstream's published metric is a per-question RATE over four "
                    "clustered tags. The correct test is a cluster bootstrap over a "
                    "float statistic; x1_power.paired_delta_ci takes booleans only, and "
                    "this module does not write a second confidence interval. Point "
                    "estimate and per-window signs only."
                ),
                "n_questions_with_tag_movement": moved,
                "per_window": windows,
                "window_sign": {"online_ahead": wins, "behind": losses},
                "mean_delta_tag_first_half": round(first, 2),
                "mean_delta_tag_second_half": round(second, 2),
                "cost_usd": cost,
                "llm_calls": calls,
            },
            indent=2,
        )
    )
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
