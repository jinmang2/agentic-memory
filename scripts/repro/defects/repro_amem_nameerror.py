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
import types
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

    def _stub_module(name: str) -> None:
        """Stub a missing module so the clone's import block succeeds.

        The traced execution path (analyze_content -> re.sub) never touches these
        stubs — they exist only so module-level imports don't fail. The defect
        (missing re import) and the proof (empty metadata on NameError) stand
        independently of whether sentence_transformers, nltk, etc. are available.
        """

        class _StubAttr:
            """Stub object that handles arbitrary attribute access and calls."""

            def __getattr__(self, name):
                return self

            def __call__(self, *args, **kwargs):
                return self

            def __str__(self):
                return ""

            def __repr__(self):
                return "<stubbed>"

            def __fspath__(self):
                return "<stubbed>"

            # String-like methods that might be called during initialization
            def endswith(self, suffix):
                return False

            def split(self, sep):
                return []

            def splitext(self):
                return ("", "")

            # Make it work in path operations
            def startswith(self, prefix):
                return False

        module = types.ModuleType(name)
        module.__getattr__ = lambda attr: _StubAttr()
        sys.modules[name] = module

    sys.path.insert(0, str(path.parent))
    memory_layer = None
    for _ in range(20):  # each round stubs one missing dep of the clone's import block
        try:
            import memory_layer  # noqa: F811

            break
        except ModuleNotFoundError as exc:
            if exc.name is None:
                skip(f"dynamic half: unstubbable import failure: {exc}")
            _stub_module(exc.name)
    if memory_layer is None:
        skip("dynamic half: could not stub the clone's deps after 20 rounds")

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
