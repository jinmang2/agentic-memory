"""A-Mem's paper-repro edition spends the Ps1 LLM call and discards its output.

memory_layer.py (plain edition — the published-numbers path) never imports `re`,
but analyze_content's parser calls re.sub after the LLM responds. The NameError
lands in a bare `except:` whose handler prints an undefined `e` (a second
NameError), so the outer handler swallows everything and returns empty metadata:
every note gets keywords=[], context="General", tags=[] after paying for the call.

Static half (runs anywhere the clone exists): AST proof that `re` is used but
never imported. Dynamic half (needs the clone's heavy deps — sentence_transformers,
nltk, litellm): calls the real analyze_content with a stub LLM returning perfectly
valid JSON, and shows the empty-metadata return plus exactly one wasted call.

Evidence: docs/research/upstream-defect-catalog.md §2; round-12 `# [amem]`
"Verified clean" item 1 (Ps1 death trace, memory_layer.py:380-393).
"""

import ast
import sys
from types import SimpleNamespace

from _common import proven, skip, upstream


def main() -> None:
    path = upstream("AgenticMemory") / "memory_layer.py"
    tree = ast.parse(path.read_text())

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    uses_re = any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "re"
        for node in ast.walk(tree)
    )
    assert uses_re, "memory_layer.py no longer calls re.* — defect shape changed"
    assert "re" not in imported, "an `import re` appeared — the defect is gone"
    proven("static: memory_layer.py uses re.* without importing re -> NameError is inevitable")

    sys.path.insert(0, str(path.parent))
    try:
        import memory_layer
    except Exception as exc:  # the clone's module-level deps are heavy and optional here
        skip(f"dynamic half needs the clone's deps: {type(exc).__name__}: {exc}")

    calls: list[str] = []

    def get_completion(prompt, response_format=None, **kwargs):
        calls.append(prompt)
        return '{"keywords": ["real"], "context": "extracted", "tags": ["fine"]}'

    stub = SimpleNamespace(llm=SimpleNamespace(get_completion=get_completion))
    result = memory_layer.MemoryNote.analyze_content("Alice moved to Paris.", stub)
    assert len(calls) == 1, f"expected exactly one (wasted) LLM call, saw {len(calls)}"
    assert result == {
        "keywords": [],
        "context": "General",
        "category": "Uncategorized",
        "tags": [],
    }, f"expected the empty-metadata fallback, got {result}"
    proven("dynamic: valid LLM JSON still yields empty metadata after 1 spent call")


if __name__ == "__main__":
    main()
