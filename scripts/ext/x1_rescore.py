"""Re-aggregate our LoCoMo runs with the audited-bad gold answers removed.

The locomo-audit catalogue flags 99 questions whose golden answer can move a
score (Task 0), and Task 1 resolved every one of them to a judged row in our
records (99/99, no unmatched, no duplicate-key collisions). What is left is the
arithmetic: what does each arm score once those questions are dropped?

Dropped, not re-graded. We have no corrected gold to score against, so the only
defensible move is to remove the question from both numerator and denominator --
the reported J becomes "accuracy over the questions the audit did not dispute".
Zeroing the flagged rows instead would punish arms for failing an ungradeable
question, and leaving them in is what the audit says is wrong. Because the
denominator moves, ``delta_J`` is not simply "how many flagged rows this arm got
right"; an arm above its own average on the flagged set falls, one below rises.

The J convention is not a choice this module makes. ``j`` exists only on
non-adversarial rows (1,540 of 1,986; adversarial questions are scored by F1
alone), so mean(j) over rows that carry a verdict *is* the published J. The four
headline stems in ``HEADLINE_ANCHORS`` pin that: if a future edit changes the
denominator, the full-J recomputation stops matching the controller's own
published number and the run aborts rather than quietly re-basing the table.

Not every run in the glob has a J at all: the wujiang eval mode scores F1 and
BLEU only and emits no verdicts, so its J is reported as unmeasurable while its
F1 is replayed like any other. That is why the join is disclosed on two bases --
see ``rescore_file``.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from collections.abc import Collection, Iterable
from pathlib import Path

# Flat modules, no package: a sibling import has to work both under `python
# scripts/ext/x1_rescore.py` (which already puts this dir first) and under the
# tests' importlib loading (which does not).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from x1_audit_data import load_errors, score_corrupting
from x1_join import (
    error_question_keys,
    judged_record_counts,
    match_report,
    normalize_q,
    question_key_map,
)

DEFAULT_DATASET = Path.home() / ".agmem/datasets/locomo10.json"
DEFAULT_ERRORS = Path.home() / ".agmem/upstream/locomo-audit/errors.json"

# Anchors apply to the four headline runs only. The glob deliberately also
# catches seed replicates, the older-embedder armA/armB pair, the luna-model
# runs and the rawq/perhit ablations: they are worth reporting beside the
# headline numbers, but none of them is a published figure, so none is pinned.
HEADLINE_ANCHORS = {
    "gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA": 67.60,
    "gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB": 65.78,
    "gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH": 61.23,
    "gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM": 31.82,
}

ADVERSARIAL = "adversarial"
RECORDS_SUFFIX = ".records.jsonl"

# Eval modes that score F1/BLEU only and legitimately emit no judge verdicts.
# Matched as a whole underscore-delimited token in the run stem, never as a
# substring, so a future "wujiang2" mode is a stranger until it is added here.
F1_ONLY_EVAL_MODES = frozenset({"wujiang"})

_REQUIRED_FIELDS = ("conv", "q", "cat", "f1")


def run_stem(path: Path) -> str:
    """The run identifier: the file name with the records suffix removed."""
    name = Path(path).name
    return name[: -len(RECORDS_SUFFIX)] if name.endswith(RECORDS_SUFFIX) else Path(path).stem


def is_f1_only_eval(stem: str) -> bool:
    """Does this run's eval mode score without a judge at all?"""
    return bool(F1_ONLY_EVAL_MODES & set(stem.split("_")))


def load_records(path: Path) -> list[dict]:
    """Load a run's records.jsonl, verifying every row can be scored.

    Verdicts are all-or-nothing across a run's non-adversarial rows: judged runs
    carry ``j`` on every one of them (1,540 of 1,986; adversarial questions are
    never judged). Two ways that can fail, and they need different answers.

    A run judged in *part* is always an error: a row that quietly lost its
    verdict shrinks the J denominator with nothing looking wrong, which is the
    precise failure this replay exists to rule out.

    A run with *no* verdicts at all is ambiguous on the bytes alone -- an
    F1-only eval mode and a judge pass that died wholesale produce identical
    files. Only the eval mode in the stem separates them, so that is what is
    consulted: wujiang is F1/BLEU by design and reports an unmeasurable J, while
    the same file from a judging mode is a broken run and raises. Accepting both
    would let a total judge failure be published as a clean F1-only measurement.
    """
    rows: list[dict] = []
    with Path(path).open() as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [f for f in _REQUIRED_FIELDS if f not in row]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing required field(s) {missing}")
            rows.append(row)

    judgeable = [r for r in rows if r["cat"] != ADVERSARIAL]
    unjudged = [r for r in judgeable if "j" not in r]
    if unjudged and len(unjudged) != len(judgeable):
        raise ValueError(
            f"{path}: {len(unjudged)} of {len(judgeable)} non-adversarial rows carry no "
            f"judge verdict 'j' (first: cat={unjudged[0]['cat']!r} q={unjudged[0]['q']!r}) "
            "-- a partially judged run cannot be aggregated; the J denominator is unsound"
        )
    if judgeable and len(unjudged) == len(judgeable) and not is_f1_only_eval(run_stem(path)):
        raise ValueError(
            f"{path}: not one of {len(judgeable)} non-adversarial rows carries a judge "
            f"verdict 'j', and the eval mode is not one of the known F1-only modes "
            f"{sorted(F1_ONLY_EVAL_MODES)} -- this looks like a judge pass that failed "
            "wholesale, which must not be reported as an unmeasurable J"
        )
    return rows


def _stats(rows: list[dict]) -> dict:
    """J over the rows carrying a verdict, F1 over all of them.

    Both are None when there is nothing to average: an unmeasured category (no
    judged rows at all, as with adversarial) must not report 0.0, which reads as
    "got everything wrong".
    """
    judged = [r for r in rows if "j" in r]
    return {
        "n": len(rows),
        "j_n": len(judged),
        "J": 100.0 * sum(bool(r["j"]) for r in judged) / len(judged) if judged else None,
        "F1": 100.0 * sum(r["f1"] for r in rows) / len(rows) if rows else None,
    }


def aggregate(
    records: Iterable[dict],
    exclude_keys: Collection[tuple[int, str]] = frozenset(),
    cats: Collection[str] | None = None,
) -> dict:
    """Score a run, optionally dropping the questions named by ``exclude_keys``.

    Excluded rows leave the denominator, they are not zeroed -- see the module
    docstring. Every row sharing an excluded key goes, which matters because
    conv 7 ships eleven questions twice: retiring one key there retires two rows.

    ``per_cat`` is keyed off the categories present *before* exclusion, so the
    full and excluded tables line up row for row. A category emptied entirely by
    the audit therefore reports n=0 rather than disappearing, which would read
    as "unchanged".
    """
    scoped = [r for r in records if cats is None or r["cat"] in cats]
    keys = frozenset(exclude_keys)
    kept = [r for r in scoped if (r["conv"], normalize_q(r["q"])) not in keys]
    out = _stats(kept)
    out["per_cat"] = {
        cat: _stats([r for r in kept if r["cat"] == cat])
        for cat in sorted({r["cat"] for r in scoped})
    }
    return out


def _round(x: float | None, places: int = 4) -> float | None:
    return None if x is None else round(x, places)


def _delta(after: float | None, before: float | None) -> float | None:
    return None if after is None or before is None else round(after - before, 4)


def _row_counts(rows: Iterable[dict]) -> Counter[tuple[int, str]]:
    """Join-key counts over whatever rows are handed in, judged or not."""
    counts: Counter[tuple[int, str]] = Counter()
    for row in rows:
        counts[(row["conv"], normalize_q(row["q"]))] += 1
    return counts


def rescore_file(path: Path, error_keys: set[tuple[int, str]]) -> dict:
    """Full and errors-excluded scores for one run, plus the join disclosure.

    The join is reported through Task 1's ``match_report`` rather than inferred
    from the aggregation, so a file where the audit lands on fewer rows than
    expected says so in the artifact instead of showing a small delta.

    Two bases, because the two metrics have different denominators. J is scored
    over judged rows, so its reduction is measured against ``judged_record_counts``;
    F1 is scored over every row, so its reduction is measured over all of them.
    For a judged run the two agree at 99 (no flagged question turns out to sit on
    an adversarial row). For an F1-only run like wujiang the judged basis has
    nothing to match and reporting only that would read as a broken join, when in
    fact the flagged questions do leave the F1 denominator.
    """
    stem = run_stem(path)
    records = load_records(path)
    run_is_judged = any("j" in r for r in records)

    all_report = match_report(_row_counts(records), error_keys)
    judged_report = None
    if run_is_judged:
        judged_counts = judged_record_counts(path)
        # our loader and Task 1's independent reader must agree on the judged
        # subset, or the two are normalizing question text differently.
        if judged_counts != _row_counts(r for r in records if "j" in r):
            raise ValueError(f"{path}: judged-row key counts disagree with judged_record_counts()")
        judged_report = match_report(judged_counts, error_keys)

    full = aggregate(records)
    excluded = aggregate(records, exclude_keys=error_keys)

    # the aggregation and the join are separate code paths over the same keys;
    # if they disagree about how many rows left, every delta below is suspect.
    if full["n"] - excluded["n"] != all_report["excluded_rows"]:
        raise ValueError(
            f"{path}: row exclusion disagrees -- aggregate dropped "
            f"{full['n'] - excluded['n']}, match_report expected {all_report['excluded_rows']}"
        )
    if (
        judged_report is not None
        and full["j_n"] - excluded["j_n"] != judged_report["excluded_rows"]
    ):
        raise ValueError(
            f"{path}: judged-row exclusion disagrees -- aggregate dropped "
            f"{full['j_n'] - excluded['j_n']}, match_report expected "
            f"{judged_report['excluded_rows']}"
        )

    per_cat = {
        cat: {
            "n_full": full["per_cat"][cat]["n"],
            "n_excluded": excluded["per_cat"][cat]["n"],
            "J_full": _round(full["per_cat"][cat]["J"]),
            "J_excluded": _round(excluded["per_cat"][cat]["J"]),
            "delta_J": _delta(excluded["per_cat"][cat]["J"], full["per_cat"][cat]["J"]),
            "F1_full": _round(full["per_cat"][cat]["F1"]),
            "F1_excluded": _round(excluded["per_cat"][cat]["F1"]),
            "delta_F1": _delta(excluded["per_cat"][cat]["F1"], full["per_cat"][cat]["F1"]),
        }
        for cat in full["per_cat"]
    }

    anchor = HEADLINE_ANCHORS.get(stem)
    if anchor is not None and full["J"] is None:
        raise ValueError(
            f"{path}: {stem} is a headline run with an anchor of {anchor}, but it carries no "
            "judge verdicts at all -- there is no J here to compare against"
        )
    # the value the anchor is judged on, kept so the report cannot display a
    # differently-rounded number beside its own PASS/FAIL.
    j_full_2dp = None if full["J"] is None else round(full["J"], 2)
    return {
        "stem": stem,
        "records_file": str(path),
        "headline": anchor is not None,
        "judged": run_is_judged,
        "anchor_J": anchor,
        "anchor_ok": None if anchor is None else j_full_2dp == anchor,
        "n_full": full["n"],
        "n_excluded": excluded["n"],
        "j_n_full": full["j_n"],
        "j_n_excluded": excluded["j_n"],
        "J_full": _round(full["J"]),
        "J_full_2dp": j_full_2dp,
        "J_excluded": _round(excluded["J"]),
        "delta_J": _delta(excluded["J"], full["J"]),
        "F1_full": _round(full["F1"]),
        "F1_excluded": _round(excluded["F1"]),
        "delta_F1": _delta(excluded["F1"], full["F1"]),
        "join": {
            "error_keys": len(error_keys),
            # all-rows basis: the F1 denominator reduction, defined for every run
            "matched_any_row": all_report["matched"],
            "excluded_rows_all": all_report["excluded_rows"],
            "unmatched_any_row": [list(k) for k in all_report["unmatched_errors"]],
            "duplicate_matched_keys_any_row": all_report["duplicate_matched_keys"],
            # judged basis: the J denominator reduction, None where J is unmeasurable
            "judged_basis_applicable": run_is_judged,
            "matched_judged": None if judged_report is None else judged_report["matched"],
            "excluded_judged_rows": (
                None if judged_report is None else judged_report["excluded_rows"]
            ),
            "unmatched_judged": (
                None
                if judged_report is None
                else [list(k) for k in judged_report["unmatched_errors"]]
            ),
            "duplicate_matched_keys_judged": (
                None if judged_report is None else judged_report["duplicate_matched_keys"]
            ),
            "row_counts_known": all_report["row_counts_known"],
        },
        "per_cat": per_cat,
    }


def _fmt(x: float | None, places: int = 2) -> str:
    return "--" if x is None else f"{x:.{places}f}"


def _signed(x: float | None) -> str:
    return "--" if x is None else f"{x:+.2f}"


def render_markdown(results: list[dict], meta: dict) -> str:
    """A table the reader can check against the published numbers by eye."""
    lines = [
        "# X1 gold-error replay: J with and without the audited questions",
        "",
        (
            f"Audit: `{meta['errors_path']}` -- {meta['errors_total']} flagged questions, "
            f"{meta['errors_score_corrupting']} of them score-corrupting "
            "(WRONG_CITATION is benign: the gold answer is right, only its evidence "
            "pointer is wrong)."
        ),
        "",
        f"Dataset (join bridge): `{meta['dataset_path']}`.",
        f"Records glob: `{meta['records_glob']}` -- {len(results)} run(s).",
        "",
        (
            "Flagged questions are **dropped, not zeroed**: with no corrected gold to "
            "score against, the honest reading of an excluded J is *accuracy over the "
            "questions the audit did not dispute*. Because the denominator moves too, "
            "an arm scoring above its own average on the flagged set goes down and one "
            "scoring below goes up -- the sign of delta-J is informative, not automatic."
        ),
        "",
        (
            "Headline runs are marked; every other row is an ablation, seed replicate "
            "or alternate-model run carried along for context and pinned to nothing."
        ),
        "",
        "## Overall",
        "",
        (
            "| run | headline | n (full -> excl) | judged n (full -> excl) | J full | J excl "
            "| dJ | F1 full | F1 excl | dF1 |"
        ),
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(results, key=lambda r: (not r["headline"], r["stem"])):
        mark = "**yes**" if r["headline"] else ""
        lines.append(
            f"| `{r['stem']}` | {mark} | {r['n_full']} -> {r['n_excluded']} "
            f"| {r['j_n_full']} -> {r['j_n_excluded']} | {_fmt(r['J_full'])} "
            f"| {_fmt(r['J_excluded'])} | {_signed(r['delta_J'])} | {_fmt(r['F1_full'])} "
            f"| {_fmt(r['F1_excluded'])} | {_signed(r['delta_F1'])} |"
        )

    lines += ["", "## Anchor check (headline runs only)", ""]
    lines += ["| run | anchor J | recomputed J (2dp) | full precision | ok |"]
    lines += ["| --- | ---: | ---: | ---: | --- |"]
    for r in sorted(results, key=lambda r: r["stem"]):
        if not r["headline"]:
            continue
        lines.append(
            f"| `{r['stem']}` | {r['anchor_J']:.2f} | {r['J_full_2dp']:.2f} | {r['J_full']} "
            f"| {'PASS' if r['anchor_ok'] else 'FAIL'} |"
        )

    lines += ["", "## Join disclosure", ""]
    lines += [
        (
            "Excluded-row counts are the true denominator reductions, and they exceed "
            "`matched` wherever one question serves more than one records row. Two "
            "bases are reported because the two metrics have different denominators: "
            "J is scored over judged rows, F1 over all of them."
        ),
        "",
    ]
    if any(not r["judged"] for r in results):
        lines += [
            (
                "The F1-only runs (`wujiang`, which emits no judge verdicts) have no judged "
                "basis at all -- `n/a` there means the question is unmeasurable for that "
                "run, not that the join failed. Their flagged questions do leave the F1 "
                "denominator, as the all-rows columns show."
            ),
            "",
        ]
    lines += [
        (
            "| run | error keys | matched (all rows) | excluded rows (F1 denom) "
            "| dup keys (all rows) | matched (judged) | excluded judged rows (J denom) "
            "| dup keys (judged) | unmatched |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(results, key=lambda r: (not r["headline"], r["stem"])):
        j = r["join"]
        na = "n/a"
        # unmatched is reported on whichever basis actually applies to the run
        unmatched = len(
            j["unmatched_judged"] if j["judged_basis_applicable"] else j["unmatched_any_row"]
        )
        lines.append(
            f"| `{r['stem']}` | {j['error_keys']} | {j['matched_any_row']} "
            f"| {j['excluded_rows_all']} | {j['duplicate_matched_keys_any_row']} "
            f"| {na if j['matched_judged'] is None else j['matched_judged']} "
            f"| {na if j['excluded_judged_rows'] is None else j['excluded_judged_rows']} "
            f"| {na if j['duplicate_matched_keys_judged'] is None else j['duplicate_matched_keys_judged']} "
            f"| {unmatched} |"
        )

    lines += ["", "## Per category", ""]
    for r in sorted(results, key=lambda r: (not r["headline"], r["stem"])):
        lines += [
            f"### `{r['stem']}`" + (" (headline)" if r["headline"] else ""),
            "",
            "| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for cat, c in r["per_cat"].items():
            lines.append(
                f"| {cat} | {c['n_full']} -> {c['n_excluded']} | {_fmt(c['J_full'])} "
                f"| {_fmt(c['J_excluded'])} | {_signed(c['delta_J'])} | {_fmt(c['F1_full'])} "
                f"| {_fmt(c['F1_excluded'])} | {_signed(c['delta_F1'])} |"
            )
        lines.append("")
        if any(c["J_full"] is None for c in r["per_cat"].values()):
            lines += [
                (
                    "`--` in a J column means the category carries no judge verdicts "
                    "(adversarial questions are scored by F1 only), not a score of zero."
                ),
                "",
            ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records-glob", required=True, help="glob for *.records.jsonl runs")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    args = ap.parse_args(argv)

    paths = [Path(p) for p in sorted(glob.glob(args.records_glob))]
    if not paths:
        raise SystemExit(f"no records files matched {args.records_glob!r}")

    all_errors = load_errors(args.errors)
    corrupting = score_corrupting(all_errors)
    error_keys = error_question_keys(corrupting, question_key_map(args.dataset))

    results = [rescore_file(p, error_keys) for p in paths]

    # STOP before writing anything: an artifact whose headline J does not
    # reproduce the published figure is measuring a different aggregation, and
    # committing it would put that discrepancy into the record as a finding.
    failed = [r for r in results if r["anchor_ok"] is False]
    if failed:
        for r in failed:
            print(
                f"ANCHOR MISMATCH {r['stem']}: recomputed J={r['J_full']} "
                f"(rounds to {round(r['J_full'], 2)}), anchor {r['anchor_J']}",
                file=sys.stderr,
            )
        raise SystemExit("aggregation convention disagrees with the published headline J -- STOP")

    missing_anchors = sorted(set(HEADLINE_ANCHORS) - {r["stem"] for r in results})
    meta = {
        "records_glob": args.records_glob,
        "dataset_path": str(args.dataset),
        "errors_path": str(args.errors),
        "errors_total": len(all_errors),
        "errors_score_corrupting": len(corrupting),
        "error_keys": len(error_keys),
        "headline_anchors": HEADLINE_ANCHORS,
        "headline_files_not_matched": missing_anchors,
    }
    if missing_anchors:
        # stderr: a warning must not land in stdout beside the success line,
        # where a caller capturing output would read it as part of the result.
        print(
            f"warning: headline run(s) not matched by the glob: {missing_anchors}",
            file=sys.stderr,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rescore.json").write_text(
        json.dumps({"meta": meta, "runs": results}, indent=2, sort_keys=False) + "\n"
    )
    (out_dir / "rescore.md").write_text(render_markdown(results, meta))
    print(f"wrote {out_dir / 'rescore.json'} and {out_dir / 'rescore.md'} ({len(results)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
