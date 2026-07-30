"""Nemori's merge similarity_threshold=0.85 is plumbed into a field nothing reads.

config.merge_similarity_threshold flows through the factory into
EpisodeMerger._similarity_threshold, which no code loads: the top-5 qdrant hits
go to the merge-decision LLM unfiltered. AST proof: exactly one Store of the
attribute, zero Loads, and the factory really plumbs the config value in.

(Round 12 caught our own "upstream" preset resurrecting this dead knob as a live
0.85 filter — the exact defect class that caught the MemoryOS eviction mislabel;
commit 688c959 reverted it, and the dead-knobs-stay-dead rule is now standing.)

Evidence: docs/research/upstream-defect-catalog.md §3; round-12 `# [nemori]` #1
(merger.py:27,34 store; grep 0 reads; config.py:86 -> factory.py:55).
"""

import ast

from _common import proven, upstream


def main() -> None:
    root = upstream("nemori")
    tree = ast.parse((root / "nemori/llm/generators/merger.py").read_text())
    stores = loads = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "_similarity_threshold":
            if isinstance(node.ctx, ast.Store):
                stores += 1
            elif isinstance(node.ctx, ast.Load):
                loads += 1
    assert stores == 1, f"expected exactly one assignment of _similarity_threshold, found {stores}"
    assert loads == 0, f"the knob came alive: found {loads} read(s)"

    factory = (root / "nemori/factory.py").read_text()
    assert "similarity_threshold=config.merge_similarity_threshold" in factory, (
        "factory no longer plumbs merge_similarity_threshold into the merger"
    )
    proven("merger._similarity_threshold: 1 store, 0 loads — config plumbs into a dead field")


if __name__ == "__main__":
    main()
