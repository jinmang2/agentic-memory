"""Was FiNER learnable, and did the playbook learn it? Structure, not accuracy.

A null result on an adaptive method has two very different explanations, and the
headline number cannot tell them apart:

  (a) the task has nothing transferable in it — every question fails its own way,
      so no accumulated text could help; or
  (b) the task is repetitive, the knowledge is exactly the kind a playbook holds,
      and the method still failed to capture it.

This script decides which, using only committed records and no model calls. It
answers three questions:

1. **How repetitive are the errors?** Confusions are counted at the tag slot —
   `(gold_tag, predicted_tag)` — and concentration is reported as the share of all
   error slots covered by the top-k pairs. A long tail means (a); a fat head
   means the knowledge exists to be written down.
2. **Did the playbook cover the head?** For each of the top confusions, whether
   BOTH of its tags appear anywhere in the 140 surviving bullets. Naming both is
   the weakest possible test of coverage — a bullet that mentions the pair may
   still say nothing useful about it — which is the point: a coverage number
   measured this generously and still low is a real finding.
3. **Where it did cover, did errors move?** Per-confusion base-vs-online counts.
   Descriptive only; single seed, and a confusion occurring 30 times is not an
   interval.

The generous-test choice matters for how the output may be cited: `covered`
here means "both tags appear somewhere in the playbook", NOT "the playbook
contains a correct rule distinguishing them". It is an upper bound on coverage,
so a low number is safe to quote and a high one would not have been.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RECORDS = ROOT / "results" / "repro"
sys.path.insert(0, str(ROOT / "src"))

from agmem.bench.finer import split_tags

ARMS = {"base": "gpt-4o-mini_ace_finer_base", "online": "gpt-4o-mini_ace_finer_online"}


def confusions(stem: str) -> tuple[collections.Counter, collections.Counter]:
    """Per-slot confusions and the gold-tag frequency, from one arm's records."""
    conf: collections.Counter = collections.Counter()
    gold: collections.Counter = collections.Counter()
    path = RECORDS / f"{stem}.records.jsonl"
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        golds, preds = split_tags(row["target"]), split_tags(row["pred"])
        for tag in golds:
            gold[tag] += 1
        # zip stops at the shorter side, which is upstream's truncation applied
        # to the same slots the scorer scored — this must agree with tag_counts
        # or the two views of one run would disagree.
        for g, p in zip(golds, preds):
            if g != p:
                conf[(g, p)] += 1
    return conf, gold


def playbook_text(stem: str) -> tuple[str, int]:
    """The surviving bullets' content, lowercased, plus their count."""
    path = RECORDS / f"{stem}.memory.jsonl"
    bullets = []
    for line in path.open(encoding="utf-8"):
        item = json.loads(line)
        if item.get("memory_type") == "playbook":
            bullets.append(str(item.get("content", "")))
    return " ".join(bullets).lower(), len(bullets)


def coverage(conf: collections.Counter, text: str, top: int) -> dict:
    head = conf.most_common(top)
    both = [(g, p) for (g, p), _ in head if g in text and p in text]
    return {
        "top_k": top,
        "pairs_with_both_tags_named": len(both),
        "share_of_all_errors": round(100 * sum(c for _, c in head) / sum(conf.values()), 1),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=RECORDS / "finer_error_structure.json")
    ap.add_argument("--top", type=int, default=8, help="confusions to print")
    args = ap.parse_args(argv)

    base_conf, gold = confusions(ARMS["base"])
    online_conf, _ = confusions(ARMS["online"])
    text, n_bullets = playbook_text(ARMS["online"])

    total_slots = sum(gold.values())
    distinct = len(gold)
    singletons = sum(1 for c in gold.values() if c == 1)
    ranked = gold.most_common()

    def cum_share(k: int) -> float:
        return round(100 * sum(c for _, c in ranked[:k]) / total_slots, 1)

    print(f"gold: {total_slots} slots, {distinct} distinct tags, {singletons} seen once")
    print(f"  top-25 tags cover {cum_share(25)}% of slots, top-50 {cum_share(50)}%")
    err = sum(base_conf.values())
    once = sum(1 for c in base_conf.values() if c == 1)
    print(
        f"errors (base): {err} slots wrong across {len(base_conf)} distinct pairs, {once} seen once"
    )
    for k in (5, 10, 20, 50):
        share = round(100 * sum(c for _, c in base_conf.most_common(k)) / err, 1)
        print(f"  top-{k:2d} confusions cover {share}% of error slots")

    print(f"\nplaybook: {n_bullets} bullets, {len(text)} chars")
    cov = [coverage(base_conf, text, k) for k in (10, 25, 50)]
    for c in cov:
        print(
            f"  of the top-{c['top_k']} confusions ({c['share_of_all_errors']}% of errors), "
            f"{c['pairs_with_both_tags_named']} have BOTH tags named in it"
        )
    named_gold = sum(1 for t in {g for g, _ in base_conf} if t in text)
    print(
        f"  gold tags ever missed: {len({g for g, _ in base_conf})}, named in playbook: {named_gold}"
    )

    print(f"\n{'confusion (gold -> predicted)':<74}{'base':>6}{'online':>8}  covered")
    rows = []
    for (g, p), c in base_conf.most_common(args.top):
        covered = g in text and p in text
        rows.append(
            {
                "gold": g,
                "pred": p,
                "base": c,
                "online": online_conf.get((g, p), 0),
                "covered": covered,
            }
        )
        print(f"{g[:35]:<36}-> {p[:33]:<34}{c:>6}{online_conf.get((g, p), 0):>8}  {covered}")

    args.out.write_text(
        json.dumps(
            {
                "gold": {
                    "slots": total_slots,
                    "distinct_tags": distinct,
                    "tags_seen_once": singletons,
                    "top25_share_pct": cum_share(25),
                    "top50_share_pct": cum_share(50),
                },
                "errors_base": {
                    "slots_wrong": err,
                    "distinct_pairs": len(base_conf),
                    "pairs_seen_once": once,
                    "concentration_pct": {
                        str(k): round(100 * sum(c for _, c in base_conf.most_common(k)) / err, 1)
                        for k in (5, 10, 20, 50)
                    },
                },
                "errors_online_slots_wrong": sum(online_conf.values()),
                "playbook": {"bullets": n_bullets, "chars": len(text)},
                "coverage": cov,
                "gold_tags_ever_missed": len({g for g, _ in base_conf}),
                "gold_tags_named_in_playbook": named_gold,
                "top_confusions": rows,
                "coverage_note": (
                    "`covered` means both tags of the pair appear SOMEWHERE in the playbook "
                    "text, not that a correct distinguishing rule is present. It is an upper "
                    "bound on coverage; a low value is safe to cite, a high one would not be."
                ),
            },
            indent=2,
        )
    )
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
