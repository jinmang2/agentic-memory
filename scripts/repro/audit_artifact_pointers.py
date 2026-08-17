"""Audit the filename pointers every run summary carries into its own heavy artifacts.

Each run writes a small `<tag>.json` summary plus up to four heavy files beside it — the LLM I/O
trace, the memory snapshot, the evolution log and the per-question records — and the summary names
them in `llm_trace_file` / `memory_file` / `op_log_file` / `records_file`. Those pointers are how
anything downstream finds a run's evidence, and nothing checked them until a demo tripped over one.

**Two checks, deliberately separate, because only one of them is machine-independent.**

  NAMING (portable). Every pointer must equal `<summary stem><suffix>`. That is the harness's own
  construction — `exp_amem_repro` builds both names from one `tag` — so a deviation means the
  artifacts were renamed and the summary's pointers did not follow. This is a pure string check on
  committed JSON: no artifact has to be present, so it holds on a fresh clone and in CI, and
  `tests/test_repro_artifacts.py` pins it.

  PRESENCE (local only). Whether the file a pointer names is actually on this disk. The heavy
  artifacts are gitignored for size (docs/14 §Artifacts), so a clone legitimately has none of them
  and this half can never be a test. It is a census, and its interesting cell is the one where the
  summary itself records that the file WAS written — a non-zero `memory_capacity.memory_jsonl_bytes`
  against a missing snapshot means a paid artifact is gone, which the repository's never-delete rule
  says should not happen.

**Why the naming check earns its place.** The same defect has now appeared twice. Track 2 found it
when a pilot run launched directly (rather than through the parallel orchestrator) produced
summaries without the `_c{i}` segment the orchestrator expects; the fix renamed the files and
updated the three pointer keys by hand. This audit exists because "by hand" is how two of them got
missed, and because the way it surfaced the second time was a demo reading a snapshot that was not
where its summary said it was.

Run:  uv run python scripts/repro/audit_artifact_pointers.py
      uv run python scripts/repro/audit_artifact_pointers.py --fix     # repoint misnamed only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "repro"

# Pointer key -> the suffix the harness appends to the run tag for that artifact.
POINTER_SUFFIXES = {
    "llm_trace_file": ".llm-trace.jsonl",
    "memory_file": ".memory.jsonl",
    "op_log_file": ".memory.ops.jsonl",
    "records_file": ".records.jsonl",
}


@dataclass(frozen=True)
class Pointer:
    """One `summary -> artifact` reference, with everything needed to judge it."""

    summary: str
    key: str
    named: str
    expected: str
    named_exists: bool
    expected_exists: bool
    recorded_bytes: int | None

    @property
    def is_misnamed(self) -> bool:
        """The pointer does not follow the harness's own naming construction."""
        return self.named != self.expected

    @property
    def was_written(self) -> bool:
        """The summary itself records that this artifact had content when the run ended.

        Only the memory snapshot carries a byte count (`memory_capacity.memory_jsonl_bytes`), so
        this is None-safe rather than universal: absent evidence is not evidence of absence, and a
        trace with no recorded size is reported as unknown rather than as intact or as lost.
        """
        return bool(self.recorded_bytes)


def recorded_bytes_for(key: str, summary: dict) -> int | None:
    """Bytes the summary claims for this artifact, when it claims any."""
    if key != "memory_file":
        return None
    value = (summary.get("memory_capacity") or {}).get("memory_jsonl_bytes")
    return value if isinstance(value, int) else None


def collect(results_dir: Path) -> list[Pointer]:
    """Every pointer in every summary under `results_dir`, in filename order."""
    pointers: list[Pointer] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            summary = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not every .json here is a run summary; a malformed one is not this audit's business.
            continue
        if not isinstance(summary, dict):
            continue
        for key, suffix in POINTER_SUFFIXES.items():
            named = summary.get(key)
            if not isinstance(named, str) or not named:
                continue
            expected = path.stem + suffix
            pointers.append(
                Pointer(
                    summary=path.name,
                    key=key,
                    named=named,
                    expected=expected,
                    named_exists=(results_dir / named).exists(),
                    expected_exists=(results_dir / expected).exists(),
                    recorded_bytes=recorded_bytes_for(key, summary),
                )
            )
    return pointers


def misnamed(pointers: list[Pointer]) -> list[Pointer]:
    """Pointers that break the naming construction. This is the portable, always-a-bug class."""
    return [pointer for pointer in pointers if pointer.is_misnamed]


def apply_fix(results_dir: Path, pointers: list[Pointer]) -> list[str]:
    """Repoint misnamed keys at the file the naming rule says they mean. Returns summaries touched.

    Refuses any pointer whose expected file is not on this disk: repointing at a name nobody can
    check would replace a wrong pointer with an unverifiable one. **No measured value is touched** —
    only the three-or-four filename strings — which is what makes rewriting a committed measurement
    artifact acceptable here, and it is the same repair Track 2 made by hand.
    """
    by_summary: dict[str, list[Pointer]] = {}
    for pointer in misnamed(pointers):
        if not pointer.expected_exists:
            print(
                f"  SKIP {pointer.summary} {pointer.key}: "
                f"{pointer.expected} is not on this disk either"
            )
            continue
        by_summary.setdefault(pointer.summary, []).append(pointer)

    touched = []
    for summary_name, items in sorted(by_summary.items()):
        path = results_dir / summary_name
        summary = json.loads(path.read_text())
        for pointer in items:
            summary[pointer.key] = pointer.expected
            print(f"  {summary_name}: {pointer.key} -> {pointer.expected}")
        # `indent=2, ensure_ascii=False` and NO trailing newline, matching what the harness itself
        # writes (`exp_amem_repro`'s `out_path.write_text(json.dumps(...))`), so the diff on a
        # committed measurement artifact is the pointer lines and nothing else.
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        touched.append(summary_name)
    return touched


def report_presence(pointers: list[Pointer]) -> None:
    """The local census: what is on this disk, and which absences the summaries contradict."""
    resolved = [p for p in pointers if not p.is_misnamed]
    present = [p for p in resolved if p.named_exists]
    absent = [p for p in resolved if not p.named_exists]
    contradicted = [p for p in absent if p.was_written]

    print("\npresence census (this machine only — heavy artifacts are gitignored)")
    print(f"  pointers whose artifact is present: {len(present)} / {len(resolved)}")
    print(f"  pointers whose artifact is absent:  {len(absent)}")
    print(f"    of those, the summary records the file HAD content: {len(contradicted)}")
    if contradicted:
        print(
            "\n  These are the ones worth a decision: a recorded byte count against a missing file\n"
            "  means the artifact existed and is gone, which the never-delete rule forbids.\n"
        )
        for pointer in contradicted:
            print(f"    {pointer.summary}")
            print(
                f"        {pointer.key}: {pointer.named} ({pointer.recorded_bytes:,} bytes recorded)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--fix",
        action="store_true",
        help="repoint misnamed keys at the artifact the naming rule names (filenames only)",
    )
    parser.add_argument(
        "--naming-only",
        action="store_true",
        help="skip the presence census; the naming check is the part that works on a clone",
    )
    args = parser.parse_args()

    pointers = collect(args.results_dir)
    broken = misnamed(pointers)
    print(f"summaries scanned in {args.results_dir.relative_to(REPO_ROOT)}")
    print(f"pointers checked: {len(pointers)}")
    print(f"pointers breaking the naming rule '<summary stem><suffix>': {len(broken)}")
    for pointer in broken:
        print(f"  {pointer.summary}")
        print(f"      {pointer.key}: {pointer.named} (present: {pointer.named_exists})")
        print(f"      rule says:  {pointer.expected} (present: {pointer.expected_exists})")

    if args.fix and broken:
        print("\nrepointing:")
        touched = apply_fix(args.results_dir, pointers)
        print(f"rewrote {len(touched)} summary file(s); no measured value was modified")
        broken = misnamed(collect(args.results_dir))

    if not args.naming_only:
        report_presence(collect(args.results_dir))

    if broken:
        raise SystemExit(
            f"\n{len(broken)} pointer(s) still break the naming rule. "
            f"Re-run with --fix to repoint them at the artifacts that exist."
        )
    print("\nnaming rule: OK")


if __name__ == "__main__":
    main()
