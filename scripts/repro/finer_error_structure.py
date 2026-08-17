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
   BOTH of its tags appear anywhere in the bullets that survived into the store.
   Naming both is the weakest possible test of coverage — a bullet that mentions
   the pair may still say nothing useful about it — which is the point: a
   coverage number measured this generously and still low is a real finding.
3. **Where it did cover, did errors move?** Per-confusion counts, the learning
   arm against the base arm. Descriptive only; single seed, and a confusion
   occurring 30 times is not an interval.

The generous-test choice matters for how the output may be cited: `covered`
here means "both tags appear somewhere in the playbook", NOT "the playbook
contains a correct rule distinguishing them". It is an upper bound on coverage,
so a low number is safe to quote and a high one would not have been.

Every learning arm is measured against the same base errors, because the second
question only has one interesting answer per playbook: `online` carries the 140
bullets that survived our 0.90 dedup, `nodedup` the 2,165 upstream's default
keeps. Coverage is what separates "the gate threw the knowledge away" from "the
knowledge was never written down" — the two readings the accuracy null cannot
tell apart. A fourth question follows from what that comparison turned up, and
it is why the trace is read as well as the store: whether the curator wrote
different things, or the same things were kept differently. Movement columns are computed on the questions both arms answered,
since a capped arm is shorter than base and comparing its errors against base's
full 441 would credit it with the errors it never had the chance to make.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

# Three humps or more, so `AmortizationOfFinancingCosts` counts and an ordinary
# capitalised sentence does not.
IDENTIFIER = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+){2,}\b")

ROOT = Path(__file__).resolve().parent.parent.parent
RECORDS = ROOT / "results" / "repro"
sys.path.insert(0, str(ROOT / "src"))

from agmem.bench.finer import split_tags

REFERENCE = "base"
ARMS = {
    "base": "gpt-4o-mini_ace_finer_base",
    "online": "gpt-4o-mini_ace_finer_online",
    "nodedup": "gpt-4o-mini_ace_finer_nodedup",
    "retry": "gpt-4o-mini_ace_finer_retry",
}


def n_rows(stem: str) -> int:
    with (RECORDS / f"{stem}.records.jsonl").open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def confusions(
    stem: str, limit: int | None = None
) -> tuple[collections.Counter, collections.Counter]:
    """Per-slot confusions and the gold-tag frequency, from one arm's records."""
    conf: collections.Counter = collections.Counter()
    gold: collections.Counter = collections.Counter()
    path = RECORDS / f"{stem}.records.jsonl"
    for i, line in enumerate(path.open(encoding="utf-8")):
        if limit is not None and i >= limit:
            break
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


def proposed_bullets(stem: str) -> list[str]:
    """Every bullet the curator PROPOSED, from the arm's trace.

    The store holds what survived; the trace holds what was written. Those are
    different populations whenever a gate sits between them, and the online
    arm's gate dropped 276 of 416 proposals — so a specificity measured on the
    store alone cannot say whether the curator wrote differently or the gate
    kept different things.
    """
    out: list[str] = []
    path = RECORDS / f"{stem}.llm-trace.jsonl"
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        try:
            call = json.loads(line)
        except json.JSONDecodeError:
            continue
        if call.get("role") != "distill":
            continue
        try:
            # reflector responses are not curator JSON and drop out here
            body = json.loads(call.get("response_text") or "")
        except json.JSONDecodeError:
            continue
        for op in body.get("operations") or []:
            if isinstance(op, dict) and op.get("content"):
                out.append(str(op["content"]))
    return out


def specificity(bullets: list[str], gold_tags: set[str], bands: int = 5) -> dict:
    """How often a bullet names a US-GAAP element rather than describing one.

    Two measures, because neither is clean alone. `names_gold_tag` searches for
    an actual tag from the split's gold vocabulary, which is what matters but
    over-counts: a handful of tags (`Revenues`) are ordinary English words that
    appear in prose about revenue. `has_identifier` looks for any CamelCase
    identifier at all, which cannot over-count that way but also counts a tag
    outside the gold set. They agree closely on the online arm and diverge on
    the arm whose bullets are prose, which is itself the finding.

    The band curve is ordered by the run, so it reads as "what the curator was
    writing early versus late" — the axis that matters when the playbook in its
    context grows by a factor of fifteen.
    """
    n = len(bullets)
    if not n:
        return {"n_proposed": 0}
    low = [b.lower() for b in bullets]
    named = sum(1 for b in low if any(g in b for g in gold_tags))
    ident = sum(1 for b in bullets if IDENTIFIER.search(b))
    width = n // bands
    curve = (
        [
            round(
                100
                * sum(1 for b in bullets[i * width : (i + 1) * width] if IDENTIFIER.search(b))
                / width,
                1,
            )
            for i in range(bands)
        ]
        if width
        else []
    )
    return {
        "n": n,
        "names_gold_tag_pct": round(100 * named / n, 1),
        "has_identifier_pct": round(100 * ident / n, 1),
        "has_identifier_pct_by_band": curve,
        "distinct_gold_tags_named": len({g for g in gold_tags for b in low if g in b}),
    }


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
    ap.add_argument(
        "--arms",
        nargs="*",
        default=None,
        help=f"learning arms to measure against {REFERENCE} (default: all present on disk)",
    )
    args = ap.parse_args(argv)

    wanted = args.arms or [
        n for n in ARMS if n != REFERENCE and (RECORDS / f"{ARMS[n]}.records.jsonl").exists()
    ]
    unknown = [n for n in wanted if n not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s): {', '.join(unknown)}")

    base_conf, gold = confusions(ARMS[REFERENCE])
    n_base = n_rows(ARMS[REFERENCE])

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

    per_arm = {}
    for name in wanted:
        text, n_bullets = playbook_text(ARMS[name])
        n = min(n_base, n_rows(ARMS[name]))
        arm_conf, _ = confusions(ARMS[name], limit=n)
        # The reference errors are re-counted on the same prefix, so the two
        # columns of the movement table describe the same questions.
        ref_conf_prefix, _ = confusions(ARMS[REFERENCE], limit=n)

        print(f"\nplaybook ({name}): {n_bullets} bullets, {len(text)} chars")
        cov = [coverage(base_conf, text, k) for k in (10, 25, 50)]
        for c in cov:
            print(
                f"  of the top-{c['top_k']} confusions ({c['share_of_all_errors']}% of errors), "
                f"{c['pairs_with_both_tags_named']} have BOTH tags named in it"
            )
        named_gold = sum(1 for t in {g for g, _ in base_conf} if t in text)
        print(
            f"  gold tags ever missed: {len({g for g, _ in base_conf})}, "
            f"named in playbook: {named_gold}"
        )

        scope = f"first {n} questions" if n < n_base else f"all {n} questions"
        print(
            f"\n{'confusion (gold -> predicted)':<74}{REFERENCE:>6}{name:>9}  covered   [{scope}]"
        )
        rows = []
        for (g, p), c in base_conf.most_common(args.top):
            covered = g in text and p in text
            rows.append(
                {
                    "gold": g,
                    "pred": p,
                    "base": ref_conf_prefix.get((g, p), 0),
                    "arm": arm_conf.get((g, p), 0),
                    "base_all": c,
                    "covered": covered,
                }
            )
            print(
                f"{g[:35]:<36}-> {p[:33]:<34}"
                f"{ref_conf_prefix.get((g, p), 0):>6}{arm_conf.get((g, p), 0):>9}  {covered}"
            )

        survivors = [
            str(json.loads(line)["content"])
            for line in (RECORDS / f"{ARMS[name]}.memory.jsonl").open(encoding="utf-8")
            if json.loads(line).get("memory_type") == "playbook"
        ]
        spec_proposed = specificity(proposed_bullets(ARMS[name]), set(gold))
        spec_survived = specificity(survivors, set(gold))
        print(
            f"  specificity  proposed {spec_proposed.get('has_identifier_pct')}% "
            f"({spec_proposed.get('n')} bullets)  ->  survived "
            f"{spec_survived.get('has_identifier_pct')}% ({spec_survived.get('n')})   "
            f"by band {spec_proposed.get('has_identifier_pct_by_band')}"
        )

        per_arm[name] = {
            "playbook": {"bullets": n_bullets, "chars": len(text)},
            "specificity_proposed": spec_proposed,
            "specificity_survived": spec_survived,
            "coverage": cov,
            "gold_tags_named_in_playbook": named_gold,
            "n_questions_compared": n,
            "prefix_of_base": n < n_base,
            "errors_slots_wrong_on_prefix": sum(arm_conf.values()),
            "errors_base_slots_wrong_on_prefix": sum(ref_conf_prefix.values()),
            "top_confusions": rows,
        }

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
                "n_questions_base": n_base,
                "gold_tags_ever_missed": len({g for g, _ in base_conf}),
                "arms": per_arm,
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
