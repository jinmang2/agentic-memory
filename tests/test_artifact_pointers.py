"""Every committed run summary must name its own heavy artifacts by the harness's naming rule.

A run writes `<tag>.json` plus up to four heavy files beside it and records their names in
`llm_trace_file` / `memory_file` / `op_log_file` / `records_file`. Both halves come from one `tag`
in `exp_amem_repro`, so a pointer that does not equal `<summary stem><suffix>` means the artifacts
were renamed and the summary's pointers did not follow — the reader then looks for evidence at a
path nothing ever wrote.

This has now happened twice. Track 2 hit it when a pilot launched directly rather than through the
parallel orchestrator produced summaries without the `_c{i}` segment; the repair renamed the files
and updated the pointer keys by hand, and two summaries' worth were missed. The second time, a demo
found it by failing to read a snapshot. Hence a test rather than a third round of by-hand repair.

**Why this is a test and the rest of the audit is not.** The naming rule is a string property of
committed JSON, so it holds on a fresh clone and in CI. Whether the artifacts are *present* cannot
be: they are gitignored for size (docs/14 §Artifacts), so a clone has none of them. That half lives
in `scripts/repro/audit_artifact_pointers.py` as a local census.
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_AUDIT_PATH = _REPO / "scripts" / "repro" / "audit_artifact_pointers.py"
_RESULTS = _REPO / "results" / "repro"


def _load_audit():
    """Import the auditor by path — `scripts/` is not a package, as elsewhere in this suite.

    The module is registered in `sys.modules` BEFORE it executes. That is not optional here: with
    `from __future__ import annotations`, `@dataclass` resolves its field annotations by looking the
    defining module up in `sys.modules`, so a path-loaded module that skips the registration raises
    from inside `dataclasses` rather than from anything this file wrote.
    """
    if str(_AUDIT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(_AUDIT_PATH.parent))
    spec = _ilu.spec_from_file_location("audit_artifact_pointers", _AUDIT_PATH)
    module = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_committed_summary_points_at_its_own_artifacts():
    """No summary may name an artifact that is not `<its own stem><suffix>`.

    Asserted over whatever summaries the tree carries rather than a pinned list, so a run added
    later is covered without anyone remembering to extend this test.
    """
    audit = _load_audit()
    pointers = audit.collect(_RESULTS)
    assert pointers, (
        f"no summaries with artifact pointers under {_RESULTS} — the test lost its subject"
    )

    broken = audit.misnamed(pointers)
    detail = "\n".join(
        f"  {p.summary}: {p.key} = {p.named!r}, rule says {p.expected!r}" for p in broken
    )
    assert not broken, (
        f"{len(broken)} artifact pointer(s) do not follow '<summary stem><suffix>':\n{detail}\n"
        f"Repair with: uv run python scripts/repro/audit_artifact_pointers.py --fix"
    )


def test_audit_flags_a_planted_misnamed_pointer(tmp_path):
    """The check must actually fail on the defect it exists for.

    Without this, a bug that made `collect` return nothing would leave the test above passing
    vacuously over an empty list — which is the shape of failure the repo's own ledger keeps
    finding in other people's harnesses.
    """
    audit = _load_audit()
    good = tmp_path / "model_conv0_ingest_tag.json"
    good.write_text('{"memory_file": "model_conv0_ingest_tag.memory.jsonl"}')
    assert audit.misnamed(audit.collect(tmp_path)) == []

    bad = tmp_path / "model_conv1_ingest_tag.json"
    bad.write_text('{"memory_file": "model_conv1_ingest_tag_RENAMED.memory.jsonl"}')
    broken = audit.misnamed(audit.collect(tmp_path))
    assert [(p.summary, p.key) for p in broken] == [(bad.name, "memory_file")]
    assert broken[0].expected == "model_conv1_ingest_tag.memory.jsonl"


def test_audit_ignores_json_that_is_not_a_run_summary(tmp_path):
    """A malformed or non-summary `.json` beside the artifacts must not break the sweep.

    `results/repro/` holds analysis outputs and dry-run quotes as well as run summaries, and the
    audit is not the place to police their shape — it has to skip them and keep going.
    """
    audit = _load_audit()
    (tmp_path / "not-json.json").write_text("{ this is not json")
    (tmp_path / "a-list.json").write_text("[1, 2, 3]")
    (tmp_path / "no-pointers.json").write_text('{"cost_usd": 1.23}')
    assert audit.collect(tmp_path) == []


@pytest.mark.parametrize(
    "key,suffix",
    [
        ("llm_trace_file", ".llm-trace.jsonl"),
        ("memory_file", ".memory.jsonl"),
        ("op_log_file", ".memory.ops.jsonl"),
        ("records_file", ".records.jsonl"),
    ],
)
def test_pointer_suffixes_match_what_the_harness_writes(key, suffix):
    """Pin the suffix table: it is the contract between the harness's tag and this audit's rule.

    A suffix edited on one side only would make the audit compare against names nobody writes, and
    it would fail loudly rather than silently — but it would fail on every summary at once, which
    reads like data corruption instead of like a one-line mistake.
    """
    audit = _load_audit()
    assert audit.POINTER_SUFFIXES[key] == suffix
