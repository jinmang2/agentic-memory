"""G-Memory's 0.7 edge gate is an effective-cosine 0.85 gate.

Upstream thresholds `similarity = 1 - distance` over a Chroma collection created
without `collection_metadata`, so the space is Chroma's default `l2` — *squared*
L2. The harness embedder (all-MiniLM-L6-v2) normalizes its outputs, so
d = 2 - 2*cos and the gate `1 - d >= 0.7` is exactly `cos >= 0.85`. Our port
gates true cosine at 0.85 and says so (fix 6fad7bb).

Evidence: docs/research/upstream-defect-catalog.md §6 (G-MEM entries);
round-12 verification (`# [gmemory]` — cos 0.85 -> 1 - d = 0.700 exactly).
"""

import re

from _common import REPO, proven, upstream


def main() -> None:
    src = (upstream("GMemory") / "mas/memory/mas_memory/GMemory.py").read_text()

    # 1. The gate really is `1 - distance` ...
    assert re.search(r"=\s*1\s*-\s*\w*distance", src), "gate is no longer 1 - distance"
    # ... over a Chroma store built with no explicit space (=> default l2, squared).
    ctors = re.findall(r"Chroma\s*\(([^)]*)\)", src, re.DOTALL)
    assert ctors, "no Chroma constructor found"
    assert all("collection_metadata" not in c for c in ctors), (
        "Chroma ctor now pins a space; the effective-gate derivation must be redone"
    )

    # 2. For unit vectors the two predicates are identical, boundary at exactly 0.700.
    for cos_x1000 in range(-1000, 1001):
        cos = cos_x1000 / 1000
        upstream_gate = (1.0 - (2.0 - 2.0 * cos)) >= 0.7
        assert upstream_gate == (cos >= 0.85), f"gates diverge at cos={cos}"
    assert abs((1.0 - (2.0 - 2.0 * 0.85)) - 0.700) < 1e-12

    # 3. Our port encodes the derived constant, not the misleading literal.
    ours = (REPO / "src/agmem/organizers/gmemory/organizer.py").read_text()
    assert "< 0.85" in ours, "our gate no longer thresholds true cosine at 0.85"

    proven("upstream 0.7 on (1 - squared-L2) == cosine >= 0.85; our gate uses 0.85 directly")


if __name__ == "__main__":
    main()
