# Portfolio Phase 1 — Defect Ledger + Tier-0 Repro Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `docs/17-defect-ledger.md` (EN canon) plus five deterministic $0 defect-reproduction
scripts under `scripts/repro/defects/`, running in CI against pinned upstream snapshots.

**Architecture:** Each repro is a standalone script proving one upstream defect with no LLM, no
network, no spend — a static source/AST proof that runs anywhere the pinned clone exists, plus an
optional dynamic proof that self-skips when heavy deps are absent (the repo's capability-gating
convention). CI shallow-fetches the four needed upstream repos at pinned SHAs so the proofs run
for real on every push. The ledger reorganizes `docs/research/upstream-defect-catalog.md` into
three tiers and cites the scripts as its proof column.

**Tech Stack:** Python 3.12, stdlib (`ast`, `json`, `re`) + pydantic (already a core dep);
GitHub Actions; existing `agmem.bench.locomo` scorers.

## Global Constraints

- Formatting: `uvx ruff@0.16.0 format` (100-col style); CI checks `scripts/` too.
- Never `git add -A`; add files explicitly. Push only on user instruction.
- No LLM/API/local-model calls anywhere in Tier-0 (spec §2: "deterministic reproductions — $0").
- Upstream clones live under `~/.agmem/upstream` (override: `AGMEM_UPSTREAM` env var); scripts
  must exit 0 with a `SKIP:` line when evidence is absent (spec §2 capability-gating convention).
- Pinned SHAs (full, for CI fetch): AgenticMemory `0c8039f28fdcc08189a23c07a3437d9d2482f9c2`,
  GMemory `7b581c51d993bd600df14691d101d7e601040cc6`, MemMachine
  `18f1211290c50ae30e9960b90bbe57d89bf68600`, nemori `d2a6dff6e5481214a0be6a2d10147feccfc16244`.
- Spec: `docs/superpowers/specs/2026-07-30-portfolio-defect-ledger-design.md` §1–§2.
  Evidence base: `docs/research/upstream-defect-catalog.md`.

---

### Task 1: `_common.py` helper + G-Memory threshold-equivalence repro

**Files:**
- Create: `scripts/repro/defects/_common.py`
- Create: `scripts/repro/defects/repro_gmemory_threshold.py`

**Interfaces:**
- Produces: `_common.upstream(name: str) -> Path` (resolves clone or SKIP-exits),
  `_common.skip(reason: str) -> NoReturn`, `_common.proven(claim: str) -> None`,
  `_common.REPO: Path` (repo root). All later repro scripts import exactly these.

- [ ] **Step 1: Write `_common.py`**

```python
"""Shared plumbing for the Tier-0 defect reproductions.

Each repro is a standalone script: it proves one upstream defect deterministically
(no LLM, no network, $0) or exits 0 with a SKIP line when its evidence is absent —
the same capability-gating convention the test suite uses (docs/01). Upstream
snapshots resolve under $AGMEM_UPSTREAM (default ~/.agmem/upstream); CI fetches
them at the pinned SHAs below, so "the CI proves it, not the prose".
"""

import os
import sys
from pathlib import Path
from typing import NoReturn

REPO = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = Path(os.environ.get("AGMEM_UPSTREAM", str(Path.home() / ".agmem" / "upstream")))

# The ledger (docs/17) cites these snapshots; a proof against a different SHA proves
# something else, so a mismatched local clone gets a loud warning (not a failure).
PINS = {
    "AgenticMemory": "0c8039f28fdcc08189a23c07a3437d9d2482f9c2",
    "GMemory": "7b581c51d993bd600df14691d101d7e601040cc6",
    "MemMachine": "18f1211290c50ae30e9960b90bbe57d89bf68600",
    "nemori": "d2a6dff6e5481214a0be6a2d10147feccfc16244",
}


def skip(reason: str) -> NoReturn:
    print(f"SKIP: {reason}")
    sys.exit(0)


def proven(claim: str) -> None:
    print(f"PROVEN: {claim}")


def upstream(name: str) -> Path:
    path = UPSTREAM_ROOT / name
    if not path.is_dir():
        skip(f"upstream clone '{name}' not found under {UPSTREAM_ROOT}")
    head = path / ".git"
    if head.exists():
        import subprocess

        sha = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        if name in PINS and sha and sha != PINS[name]:
            print(f"WARNING: {name} is at {sha[:7]}, ledger pins {PINS[name][:7]}")
    return path
```

- [ ] **Step 2: Write `repro_gmemory_threshold.py`**

```python
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
```

- [ ] **Step 3: Run it against the real clone**

Run: `cd scripts/repro/defects && uv run python repro_gmemory_threshold.py`
Expected: `PROVEN: upstream 0.7 on (1 - squared-L2) == cosine >= 0.85; ...`, exit 0.

- [ ] **Step 4: Run the skip path**

Run: `cd scripts/repro/defects && AGMEM_UPSTREAM=/nonexistent uv run python repro_gmemory_threshold.py`
Expected: `SKIP: upstream clone 'GMemory' not found under /nonexistent`, exit 0.

- [ ] **Step 5: Format + commit**

Run: `uvx ruff@0.16.0 format scripts/repro/defects/ && uvx ruff@0.16.0 format --check scripts/`
```bash
git add scripts/repro/defects/_common.py scripts/repro/defects/repro_gmemory_threshold.py
git commit -m "feat(repro): Tier-0 defect proofs begin — the 0.7 that was 0.85 all along"
```

---

### Task 2: A-Mem NameError repro

**Files:**
- Create: `scripts/repro/defects/repro_amem_nameerror.py`

**Interfaces:**
- Consumes: `_common.upstream`, `_common.skip`, `_common.proven` (Task 1).

- [ ] **Step 1: Confirm `analyze_content`'s binding** (it takes `(content, llm_controller)` with
  no `self`; check whether it carries `@staticmethod`):

Run: `sed -n '304,312p' ~/.agmem/upstream/AgenticMemory/memory_layer.py`
If there is NO `@staticmethod` decorator, the dynamic half below must call it as
`memory_layer.MemoryNote.analyze_content.__func__(...)` only if it is wrapped; with a bare
`def analyze_content(content, llm_controller)` inside the class, `MemoryNote.analyze_content("x", stub)`
already binds `content="x"` — keep the plain class-attribute call in that case.

- [ ] **Step 2: Write `repro_amem_nameerror.py`**

```python
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
```

- [ ] **Step 3: Run it** — `cd scripts/repro/defects && uv run python repro_amem_nameerror.py`
Expected: both `PROVEN:` lines locally (full install has the deps); if the dynamic import
fails, fix per Step 1's binding note before assuming a dep problem.

- [ ] **Step 4: Skip path** — `AGMEM_UPSTREAM=/nonexistent uv run python repro_amem_nameerror.py`
Expected: `SKIP: upstream clone 'AgenticMemory' not found ...`, exit 0.

- [ ] **Step 5: Format + commit**

```bash
uvx ruff@0.16.0 format scripts/repro/defects/repro_amem_nameerror.py
git add scripts/repro/defects/repro_amem_nameerror.py
git commit -m "feat(repro): the extraction call A-Mem pays for and throws away"
```

---

### Task 3: MemMachine TypeError repro

**Files:**
- Create: `scripts/repro/defects/repro_memmachine_typeerror.py`

**Interfaces:**
- Consumes: `_common.upstream`, `_common.proven` (Task 1).

- [ ] **Step 1: Write `repro_memmachine_typeerror.py`**

```python
"""Every MemMachine eval entry point crashes before construction at the audited SHA.

At 18f1211, `LongTermMemoryParams` is an Annotated discriminated union — a type
expression, not a class — while `evaluation/utils/agent_utils.py` still calls it
like a constructor. Python raises TypeError before pydantic ever sees the kwargs,
so no eval harness at this SHA can have produced the published numbers.

Static half: both source facts asserted in the pinned clone. Dynamic half: the
exact construct replayed with pydantic (version-independent — the call fails at
the typing layer, not in pydantic).

Evidence: docs/research/upstream-defect-catalog.md §9; round-12 `# [memmachine]` #1.
"""

import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from _common import proven, upstream

LTM_PATH = (
    "packages/server/src/memmachine_server/episodic_memory/"
    "long_term_memory/long_term_memory.py"
)


def main() -> None:
    root = upstream("MemMachine")
    definition = (root / LTM_PATH).read_text()
    assert re.search(r"LongTermMemoryParams\s*=\s*Annotated\[", definition), (
        "LongTermMemoryParams is no longer the Annotated union"
    )
    harness = (root / "evaluation/utils/agent_utils.py").read_text()
    assert re.search(r"LongTermMemoryParams\s*\(", harness), (
        "the harness no longer calls LongTermMemoryParams(...)"
    )
    proven("static: the harness constructor-calls a type alias that is not a class")

    class DeclarativeBackendParams(BaseModel):
        backend: Literal["declarative"] = "declarative"

    class EventBackendParams(BaseModel):
        backend: Literal["event"] = "event"

    long_term_memory_params = Annotated[
        DeclarativeBackendParams | EventBackendParams, Field(discriminator="backend")
    ]
    try:
        long_term_memory_params(backend="declarative")
    except TypeError as exc:
        assert "not callable" in str(exc), f"unexpected TypeError text: {exc}"
        proven(f"dynamic: calling the Annotated union raises TypeError ({exc})")
        return
    raise AssertionError("the Annotated-union call unexpectedly succeeded")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it** — expected two `PROVEN:` lines
  (dynamic one ends `...('types.UnionType' object is not callable)`).
- [ ] **Step 3: Skip path** with `AGMEM_UPSTREAM=/nonexistent` — `SKIP:`, exit 0.
- [ ] **Step 4: Format + commit**

```bash
uvx ruff@0.16.0 format scripts/repro/defects/repro_memmachine_typeerror.py
git add scripts/repro/defects/repro_memmachine_typeerror.py
git commit -m "feat(repro): the eval harness MemMachine's HEAD cannot run"
```

---

### Task 4: Nemori dead-knob repro

**Files:**
- Create: `scripts/repro/defects/repro_nemori_dead_knob.py`

**Interfaces:**
- Consumes: `_common.upstream`, `_common.proven` (Task 1).

- [ ] **Step 1: Write `repro_nemori_dead_knob.py`**

```python
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
```

- [ ] **Step 2: Run it** — `PROVEN:` line, exit 0.
- [ ] **Step 3: Skip path** with `AGMEM_UPSTREAM=/nonexistent` — `SKIP:`, exit 0.
- [ ] **Step 4: Format + commit**

```bash
uvx ruff@0.16.0 format scripts/repro/defects/repro_nemori_dead_knob.py
git add scripts/repro/defects/repro_nemori_dead_knob.py
git commit -m "feat(repro): the 0.85 Nemori configures, plumbs, and never reads"
```

---

### Task 5: LoCoMo re-scoring replay

**Files:**
- Create: `scripts/repro/defects/repro_locomo_rescoring.py`

**Interfaces:**
- Consumes: `_common.REPO`, `_common.skip`, `_common.proven` (Task 1);
  `agmem.bench.locomo.token_f1(pred, gold) -> float` and
  `token_f1_wujiang(pred, gold) -> float` (existing).
- Record schema (`results/repro/*.records.jsonl`, committed, 8 files / 11,914 questions):
  one JSON object per line with keys `run, conv, q, gold, pred, cat, f1, j, retrieval`.

- [ ] **Step 1: Write `repro_locomo_rescoring.py`**

```python
"""Re-scoring replay over stored eval artifacts — no new tokens spent.

1. Self-consistency: recomputing token_f1 from each stored (pred, gold) pair
   reproduces the stored per-question `f1` to rounding drift (< 1e-3; measured
   max 4.8e-04) — the stored headline numbers are a pure function of the
   persisted artifacts, so any auditor can re-derive them offline.
2. Scorer-lineage divergence: the upstream (WujiangXu) scorer disagrees with the
   uniform scorer on a stable nonzero subset of questions (stopword/article
   partial credit), which is why any cross-edition F1 comparison must name its
   scorer. The divergence count is pinned; a changed count means a scorer edit.

Evidence: docs/research/upstream-defect-catalog.md §10b/§11 (stemmer null result:
11,914 questions, delta-F1 = 0.000); docs/research/amac-admission-gate.md §4.
"""

import json
import sys

from _common import REPO, proven, skip

sys.path.insert(0, str(REPO / "src"))

from agmem.bench.locomo import token_f1, token_f1_wujiang  # noqa: E402

# Pinned on first run over the 8 committed record files (11,914 questions);
# see Step 2 of the implementing task for the pinning procedure.
EXPECTED_QUESTIONS = None  # pin: total records scored
EXPECTED_DIVERGING = None  # pin: |{q : |f1_wujiang - f1_ours| > 1e-9}|


def main() -> None:
    records = sorted((REPO / "results" / "repro").glob("*.records.jsonl"))
    if not records:
        skip("no results/repro/*.records.jsonl artifacts present")

    total = diverging = 0
    drift_max = 0.0
    example = None
    for path in records:
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            ours = token_f1(rec["pred"], rec["gold"])
            drift_max = max(drift_max, abs(ours - float(rec["f1"])))
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
```

- [ ] **Step 2: Run once, then pin.** Run
  `cd scripts/repro/defects && uv run python repro_locomo_rescoring.py`; it prints the observed
  `total` and `diverging`. Edit the two `EXPECTED_*` constants from `None` to exactly those
  printed integers (the repo's established pinning-test pattern), re-run, confirm both
  `PROVEN:` lines still appear and exit 0.
- [ ] **Step 3: Skip path.** `results/` is committed, so simulate absence:
  run from a temp copy or verify by inspection that the empty-glob branch hits `skip(...)`
  (e.g. `uv run python - <<'EOF'` snippet gating on a nonexistent dir is NOT needed — a
  one-line check: temporarily pass `AGMEM_UPSTREAM=/nonexistent` does not apply here; instead
  run `python -c` against a scratch REPO copy without results/, or accept the glob-guard by
  code review since the branch is two lines).
- [ ] **Step 4: Format + commit**

```bash
uvx ruff@0.16.0 format scripts/repro/defects/repro_locomo_rescoring.py
git add scripts/repro/defects/repro_locomo_rescoring.py
git commit -m "feat(repro): the stored artifacts re-score themselves, and the two F1s disagree on cue"
```

---

### Task 6: CI wiring — pinned upstream fetch + repro run

**Files:**
- Modify: `.github/workflows/ci.yml` (append steps to the existing `test` job, after "Run tests")

**Interfaces:**
- Consumes: all five scripts from Tasks 1–5 (run as `python <script>` from their directory).

- [ ] **Step 1: Append two steps to `.github/workflows/ci.yml`**

```yaml
      # Tier-0 defect reproductions (docs/17-defect-ledger.md): each script proves
      # one upstream defect deterministically, $0, against the pinned snapshots
      # fetched here. GitHub serves arbitrary reachable SHAs, so a depth-1 fetch
      # of the exact pin is enough. Scripts self-skip when evidence is absent
      # (e.g. the A-Mem dynamic half needs heavy deps the core install omits).
      - name: Fetch pinned upstream snapshots
        run: |
          set -eu
          fetch_pin() {
            dir="$HOME/.agmem/upstream/$1"
            git init -q "$dir"
            git -C "$dir" fetch -q --depth 1 "https://github.com/$2" "$3"
            git -C "$dir" checkout -q FETCH_HEAD
          }
          fetch_pin AgenticMemory WujiangXu/AgenticMemory 0c8039f28fdcc08189a23c07a3437d9d2482f9c2
          fetch_pin GMemory bingreeky/GMemory 7b581c51d993bd600df14691d101d7e601040cc6
          fetch_pin MemMachine MemMachine/MemMachine 18f1211290c50ae30e9960b90bbe57d89bf68600
          fetch_pin nemori nemori-ai/nemori d2a6dff6e5481214a0be6a2d10147feccfc16244

      - name: Run Tier-0 defect reproductions
        working-directory: scripts/repro/defects
        run: |
          set -eu
          for script in repro_*.py; do
            echo "== $script"
            uv run --no-default-groups --group dev python "$script"
          done
```

- [ ] **Step 2: Rehearse the CI loop locally**

Run: `cd scripts/repro/defects && for s in repro_*.py; do echo "== $s"; uv run python "$s"; done`
Expected: every script prints `PROVEN:` (or a legitimate `SKIP:`), overall exit 0.

- [ ] **Step 3: Rehearse the fetch function once** (proves GitHub serves the pinned SHAs):

Run the `fetch_pin` body by hand into a scratch dir for one repo, e.g.
`d=$(mktemp -d); git init -q $d; git -C $d fetch -q --depth 1 https://github.com/nemori-ai/nemori d2a6dff6e5481214a0be6a2d10147feccfc16244 && echo FETCH-OK; rm -rf $d`
Expected: `FETCH-OK`. If GitHub refuses the bare-SHA fetch for any repo, fall back for that
repo to `git clone -q --filter=blob:none <url> $dir && git -C $dir checkout -q <sha>`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: the defect proofs run on every push, against the SHAs the ledger cites"
```

- [ ] **Step 5: After the next push (user-authorized), watch the workflow once** —
  `gh run watch` — and confirm the two new steps pass. Do not push just for this.

---

### Task 7: `docs/17-defect-ledger.md` + README pointer

**Files:**
- Create: `docs/17-defect-ledger.md`
- Modify: `README.md` (add one line to the docs index/links section pointing at docs/17)

**Interfaces:**
- Consumes: `docs/research/upstream-defect-catalog.md` (entry evidence),
  `docs/superpowers/specs/2026-07-30-portfolio-defect-ledger-design.md` §1 (tier definitions,
  entry schema), the five Task 1–5 scripts (proof column), `_common.PINS` (snapshot table).

- [ ] **Step 1: Write the ledger.** Structure (spec §1, catalog as source — the writer reads
  both in full before writing):
  - **Header + thesis** (~2 paragraphs): the repo's strongest claim — defects found in source
    papers and their official code, what we did about them, and how each is proven. State the
    evidence standard: every entry cites upstream `file:line` at a pinned SHA, and the proof
    column names either a Tier-0 script in `scripts/repro/defects/` (run by CI) or the research
    doc + verification verdict that established it (96 adversarial verdicts, 94 confirmed).
  - **How to read an entry** — the schema verbatim from spec §1: paper claim → what the
    official code actually does (code-line citation) → our handling (fix, or disclosed
    deviation at the code site) → proof method → impact on published numbers.
  - **Snapshot table**: the 9 upstream repos, full pinned SHAs (from `_common.PINS` plus the
    catalog's SHAs for MemoryOS `587ed77`, ace `bcb7cea`, amac `40407ae`, graphiti `9140123`,
    reasoning-bank `ed80611`), licenses where recorded in the catalog.
  - **Tier A — published numbers are artifacts**: A-MAC (substring recall + N≡1.0/R≈0 + CV
    that never fits), A-Mem plain edition (`NameError` → empty metadata), MemMachine
    (eval entry `TypeError` at HEAD). Sources: catalog §1, §2, §9.
  - **Tier B — paper ≠ code**: MemoryOS `evict_lfu` (catalog §4), G-Memory effective-0.85
    gate (§6), Nemori dead 0.85 knob (§3), Zep/Graphiti code-ahead-of-paper + ascending
    episode-mentions (§5), A-MAC θ* degenerate fit (§1).
  - **Tier C — evaluation-harness defects**: LoCoMo cat5 MCQ construction, stopword/article
    partial credit (scorer lineage), MemoryOS batching cost distortion. Sources: catalog §10,
    §11; docs/research memories of the 1b 채점감사 (cite `docs/research/` docs only, EN prose).
  - **Cross-cutting patterns** (catalog §12, condensed to ~5 bullets with the "dead knobs stay
    dead" and "pin the lineage" rules as adopted project stances).
  - Every entry's proof cell links the concrete artifact: `scripts/repro/defects/<file>` for the
    five scripted proofs, `tests/...` where an existing pinning test is the proof (A-MAC
    substring facade run), or the research-doc anchor otherwise.
- [ ] **Step 2: Self-check the ledger** — every `file:line` and SHA citation in Tier A–C spot
  verified against the catalog (no new claims invented beyond catalog + round-12 docs); every
  Tier-0 script referenced actually exists; EN only (KO is Phase 4).
- [ ] **Step 3: Add the README line** in the existing docs list, matching its style, e.g.
  `- [17 — Defect ledger](docs/17-defect-ledger.md): what the source papers' own code does,
  proven in CI.`
- [ ] **Step 4: Format check** — `uvx ruff@0.16.0 format --check src/ tests/ scripts/` (no
  Python touched here, but the check is cheap) and a manual markdown render skim.
- [ ] **Step 5: Commit**

```bash
git add docs/17-defect-ledger.md README.md
git commit -m "docs(ledger): the defect ledger — three tiers, every claim with a proof column"
```

---

### Task 8: Full-suite gate

- [ ] **Step 1:** `uv run pytest tests/ -q` — expected 441 passed / 1 skipped (or more passed if
  any task added tests; none planned).
- [ ] **Step 2:** `uvx ruff@0.16.0 format --check src/ tests/ scripts/` — clean.
- [ ] **Step 3:** `cd scripts/repro/defects && for s in repro_*.py; do uv run python "$s"; done`
  — all PROVEN/SKIP, exit 0.
- [ ] **Step 4:** No commit here unless fixes were needed; report results.

## Self-Review Notes

- Spec §1 coverage: ledger = Task 7. Spec §2 Tier-0 list: G-Memory (T1), A-Mem (T2),
  MemMachine (T3), Nemori (T4), LoCoMo replay (T5); CI + skip convention (T6, `_common`).
- The spec's "one per defect where feasible" is satisfied for the five named; A-MAC substring
  already has a pinning test in the suite (`tests/` facade run) — ledger cites it (T7).
- Types consistent: all scripts import exactly `REPO/PINS/skip/proven/upstream` from `_common`.
- Known open point carried to Phase 2: GraphRecall fusion decision (round-12 leftover).
