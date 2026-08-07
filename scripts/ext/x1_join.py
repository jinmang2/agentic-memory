"""Join the locomo-audit error catalogue to our own LoCoMo run records.

The audit names questions by ``question_id`` ("locomo_{conv}_qa{n}"); our
records name them by conversation index plus the question text. The bridge is
``locomo10.json``, which Task 0 verified is byte-identical between the audit's
copy and ours -- so enumerating it in order reconstructs the audit's ids exactly.

Index convention, measured against the pinned audit (all 156 flagged ids):
``n`` is 0-based over the sample's *full* qa list. 1-based fails wholesale
(155 text mismatches + 1 id that does not exist). Excluding the category-5
adversarial rows from the enumeration would also cross-check clean, but only
because every flagged index falls strictly before its conversation's first
category-5 row -- the two conventions are indistinguishable on this catalogue,
and the full list is the one that keeps ids stable if the audit ever flags an
adversarial question.

Matching is on (conv, normalized question), never on the id alone: the id is
how we look a question up, and the text is how we prove the lookup landed on
the question the audit meant. A silent id-only join would keep producing
numbers after an upstream re-pin shifted the ordering.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path


def normalize_q(q: str) -> str:
    """The one place question text is normalized -- both sides of every join."""
    return " ".join(q.split()).lower()


def question_key_map(dataset_path: Path) -> dict[str, tuple[int, str]]:
    """Reconstruct the audit's question_id -> (conv, normalized question) map.

    Sample index is the conversation id; qa index is 0-based over the full qa
    list (see module docstring for how that was established).
    """
    samples = json.loads(Path(dataset_path).read_text())
    if not isinstance(samples, list):
        raise ValueError(f"{dataset_path}: expected a list of samples")
    key_map: dict[str, tuple[int, str]] = {}
    for conv, sample in enumerate(samples):
        for n, qa in enumerate(sample["qa"]):
            key_map[f"locomo_{conv}_qa{n}"] = (conv, normalize_q(qa["question"]))
    return key_map


def error_question_keys(
    errors: list[dict], key_map: dict[str, tuple[int, str]]
) -> set[tuple[int, str]]:
    """Resolve each error's question_id to a join key, cross-checking the text.

    Raises on an unknown id or on any text disagreement: either means the
    ordering assumption behind ``question_key_map`` has broken, and a partial
    join would understate the audit's reach without anything looking wrong.
    """
    keys: set[tuple[int, str]] = set()
    for e in errors:
        qid = e["question_id"]
        if qid not in key_map:
            raise ValueError(f"{qid}: no such question in the dataset -- id scheme has moved")
        conv, mapped = key_map[qid]
        asked = normalize_q(e["question"])
        if mapped != asked:
            raise ValueError(
                f"{qid}: audit asks {asked!r} but the dataset holds {mapped!r} "
                "-- qa ordering assumption is broken"
            )
        keys.add((conv, mapped))
    return keys


def judged_record_counts(records_path: Path) -> Counter[tuple[int, str]]:
    """Count judged records rows per join key.

    Only rows carrying a judge verdict ("j") join: adversarial rows are never
    judged and the audit catalogue is scoped to the judged set. The count (not
    a bare set) is what lets a matched key retire every row it serves -- conv 7
    ships 11 exactly-duplicated questions, each present twice.
    """
    counts: Counter[tuple[int, str]] = Counter()
    with Path(records_path).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if "j" not in row:
                continue
            counts[(row["conv"], normalize_q(row["q"]))] += 1
    return counts


def match_report(
    records_qs,
    error_keys: set[tuple[int, str]],
    record_counts: dict[tuple[int, str], int] | None = None,
) -> dict:
    """Report how the error keys land on our records.

    ``records_qs`` is any membership-testable collection of join keys: a set, or
    the Counter from ``judged_record_counts``, in which case it doubles as the
    row counts unless ``record_counts`` overrides it. Counts drive
    ``excluded_rows``, the true denominator reduction, which exceeds ``matched``
    wherever one question serves two rows -- deriving them from a passed Counter
    is deliberate, so that handing over duplicate-bearing records cannot quietly
    report zero duplicates. ``unmatched_errors`` is always listed in full: a
    quiet partial join is the failure mode this whole module exists to prevent.
    """
    matched = sorted(k for k in error_keys if k in records_qs)
    if record_counts is None:
        record_counts = records_qs if isinstance(records_qs, Mapping) else None
    counts = record_counts or {}
    return {
        "matched": len(matched),
        "unmatched_errors": sorted(k for k in error_keys if k not in records_qs),
        "duplicate_matched_keys": sum(1 for k in matched if counts.get(k, 1) > 1),
        "excluded_rows": sum(counts.get(k, 1) for k in matched),
    }
