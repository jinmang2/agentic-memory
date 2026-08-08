"""Re-scoring replay over stored eval artifacts — no new tokens spent.

1. Self-consistency: each records file is replayed with the scorer lineage that
   produced it (files with `*_wujiang_*` in the name use token_f1_wujiang, others
   use token_f1). Recomputing with the correct scorer reproduces stored f1 to
   rounding drift (< 1e-3; measured max 5.00e-04 across all 8 files). Uniform
   replay with a single scorer would NOT reproduce stored f1 — uniform token_f1
   on wujiang files yields max drift 1.0. This asymmetry is itself scorer-lineage
   evidence. Any auditor can re-derive the stored headline numbers offline if they
   match the scorer to the eval_mode that produced it (docs/research/
   upstream-defect-catalog.md §10b/§11 documents the ours-mode stemmer audit
   measured 4.8e-04 over its 4-file subset).

2. Scorer-lineage divergence: the upstream (WujiangXu) scorer disagrees with the
   uniform scorer on a stable nonzero subset of questions (stopword/article
   partial credit), which is why any cross-edition F1 comparison must name its
   scorer. The divergence count is pinned; a changed count means a scorer edit.

Evidence: docs/research/amac-admission-gate.md §4 (scorer selection); harness
eval_mode (agmem/bench/locomo.py:704-781).
"""

import json
import subprocess
import sys
from pathlib import Path

from _common import REPO, proven, skip

sys.path.insert(0, str(REPO / "src"))

from agmem.bench.locomo import token_f1, token_f1_wujiang  # noqa: E402

# Pinned on first run over the 8 committed record files (12,314 questions);
# see Step 2 of the implementing task for the pinning procedure.
EXPECTED_QUESTIONS = 12314  # pin: total records scored
EXPECTED_DIVERGING = 2804  # pin: |{q : |f1_wujiang - f1_ours| > 1e-9}|


def tracked_records() -> list[Path]:
    """The COMMITTED record files, which is what the pins above are pins on.

    A bare glob was wrong: `results/repro/` is also where in-flight campaigns
    write, those artifacts are gitignored rather than absent, and the counts
    below then trip on a local working tree while CI (which only ever has the
    committed set) passed. Both directions of that were bad — a false local
    failure, and a pin whose meaning would have shifted silently the day a new
    artifact was committed. Falling back to the glob keeps the script runnable
    outside a checkout; the assertions then say what they always said."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "results/repro/*.records.jsonl"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return sorted((REPO / "results" / "repro").glob("*.records.jsonl"))
    return sorted(REPO / p for p in out)


def main() -> None:
    records = tracked_records()
    if not records:
        skip("no committed results/repro/*.records.jsonl artifacts present")

    total = diverging = 0
    drift_max = 0.0
    example = None
    for path in records:
        # Determine which scorer was used based on filename
        is_wujiang_run = "wujiang" in path.name
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            # Use the scorer that matches how the record was originally scored
            if is_wujiang_run:
                computed = token_f1_wujiang(rec["pred"], rec["gold"])
            else:
                computed = token_f1(rec["pred"], rec["gold"])
            drift_max = max(drift_max, abs(computed - float(rec["f1"])))
            # Always compare across both scorers to measure lineage divergence
            ours = token_f1(rec["pred"], rec["gold"])
            wujiang = token_f1_wujiang(rec["pred"], rec["gold"])
            if abs(wujiang - ours) > 1e-9:
                diverging += 1
                if example is None:
                    example = (rec["q"], rec["gold"], rec["pred"], ours, wujiang)
            total += 1

    assert drift_max < 1e-3, f"stored f1 no longer re-derivable: max drift {drift_max}"
    proven(f"self-consistency: {total} questions re-scored, max drift {drift_max:.2e}")

    assert diverging > 0, "scorer lineages agree everywhere — the partial-credit claim is dead"
    if example:
        q, gold, pred, ours, wujiang = example
        print(f"  e.g. {q!r}: gold={gold!r} pred={pred!r} ours={ours:.3f} wujiang={wujiang:.3f}")
    if EXPECTED_QUESTIONS is not None:
        assert total == EXPECTED_QUESTIONS, f"question count moved: {total}"
    if EXPECTED_DIVERGING is not None:
        assert diverging == EXPECTED_DIVERGING, f"divergence count moved: {diverging}"
    proven(f"scorer lineage matters: {diverging}/{total} questions score differently")


if __name__ == "__main__":
    main()
