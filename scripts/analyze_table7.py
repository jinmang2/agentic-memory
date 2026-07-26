"""Nemori v4 Table 7 verdict: does feeding A-Mem distilled knowledge K instead of
raw messages land in the paper's 45-64% storage-reduction band?

The paper reports 45-64% less storage with +1.9%~+6.1% on core categories, but
never says at what unit K arrives, so we wired two granularities and only one can
be the paper's. This computes the reduction against the A-Mem-alone baseline from
the runs' `memory_capacity` blocks, on two measures -- item count and stored bytes
-- because "storage" is ambiguous in the paper and the two can disagree: K is
fewer but individually longer units than raw turns.

Usage: uv run python scripts/analyze_table7.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
BAND = (45.0, 64.0)  # Nemori v4 Table 7 storage-reduction band, percent
BASELINE = "amem"
VARIANTS = ("nemori_amem_k", "nemori_amem_k_batched")
# LoCoMo cat1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial
# (bench/locomo.py:26). Nemori v4 reports "+1.9%~+6.1% on core categories" but
# never says WHICH are core (docs/research/nemori-reasoningbank.md:170), so we
# print all five and make no core-match claim rather than encode a guess.
CATEGORY_NAMES = {
    "1": "multi-hop",
    "2": "temporal",
    "3": "open-domain",
    "4": "single-hop",
    "5": "adversarial",
}


def load(tag: str) -> dict | None:
    path = RESULTS / f"locomo-conv0-{tag}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def pct_drop(baseline: float, value: float) -> float | None:
    """Percent reduction from `baseline` to `value` (None if no baseline)."""
    if not baseline:
        return None
    return round((baseline - value) / baseline * 100, 1)


def main() -> None:
    base = load(BASELINE)
    if base is None:
        raise SystemExit(
            f"missing baseline results/locomo-conv0-{BASELINE}.json — "
            "run `--configs amem` first (the storage denominator)"
        )
    if "memory_capacity" not in base:
        raise SystemExit(
            f"results/locomo-conv0-{BASELINE}.json predates memory_capacity capture; "
            "re-run `--configs amem` so the storage denominator exists"
        )

    bcap = base["memory_capacity"]
    b_notes = bcap["counts"].get("notes", 0)
    b_bytes = bcap["bytes"].get("notes", 0)
    b_write = sum(
        s.get("tokens_in", 0) + s.get("tokens_out", 0)
        for role, s in base["llm_budget"].items()
        if role != "generate"  # generate is read-path answering, not write cost
    )
    print(f"baseline {BASELINE}: notes={b_notes} note_bytes={b_bytes} write_tokens={b_write}")
    print(f"  overall={base['overall']} turns={base.get('n_turns')}")
    print()

    for tag in VARIANTS:
        run = load(tag)
        if run is None:
            print(f"{tag}: MISSING (not run yet)")
            continue
        cap = run["memory_capacity"]
        notes = cap["counts"].get("notes", 0)
        nbytes = cap["bytes"].get("notes", 0)
        # A-Mem is the system under test; the Nemori units that produced K are
        # upstream scaffolding, so report both the notes-only view (comparable to
        # the baseline's stored memory) and the all-derived view (honest total).
        write = sum(
            s.get("tokens_in", 0) + s.get("tokens_out", 0)
            for role, s in run["llm_budget"].items()
            if role != "generate"
        )
        d_count = pct_drop(b_notes, notes)
        d_bytes = pct_drop(b_bytes, nbytes)
        in_band_count = d_count is not None and BAND[0] <= d_count <= BAND[1]
        in_band_bytes = d_bytes is not None and BAND[0] <= d_bytes <= BAND[1]
        print(f"{tag}:")
        print(
            f"  notes {notes} vs {b_notes}  -> {d_count}% reduction"
            f"  {'IN BAND' if in_band_count else 'outside 45-64%'}"
        )
        print(
            f"  note bytes {nbytes} vs {b_bytes}  -> {d_bytes}% reduction"
            f"  {'IN BAND' if in_band_bytes else 'outside 45-64%'}"
        )
        print(
            f"  all derived items {cap['derived_item_count']}"
            f" ({cap['derived_bytes']} bytes) incl. Nemori scaffolding"
        )
        print(f"  write tokens {write} vs {b_write} -> {pct_drop(b_write, write)}% reduction")
        print(f"  overall={run['overall']}")
        for cat, cname in CATEGORY_NAMES.items():
            bc = base["by_category"].get(cat)
            rc = run["by_category"].get(cat)
            if bc and rc:
                delta = round(rc["f1"] - bc["f1"], 2)
                note = "  (K has no timestamps — not a K result)" if cat == "2" else ""
                print(f"  cat{cat} {cname:12s} f1 {rc['f1']:6} vs {bc['f1']:6} -> {delta:+}{note}")
        print()

    print(
        "Caveat: local Qwen3-0.6B is below the paper's >=1B backbone, so F1 deltas are\n"
        "directional only (docs/13 §6.1). The storage verdict is structural -- it counts\n"
        "units and bytes, not answer quality -- so it survives the small backbone.\n"
        "K carries no timestamps, so temporal (cat2) must not be read as a K result.\n"
        "The paper does not say which categories are 'core', so no core-match is claimed."
    )


if __name__ == "__main__":
    main()
