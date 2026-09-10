"""The coding-eval runner, driven by a stub agent instead of a model.

The stub reads the prompt on stdin, applies the known fix for the timezone
task when told to, and prints the JSON shape ``claude -p --output-format
json`` prints. Everything else is the real thing: the staged workspace, the
per-arm store, the product's own prompt-recall hook probed for what it
would inject, the acceptance command, and the manifest's result validation.
"""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from agmem.bench.coding_eval.prepare import prepare
from agmem.bench.coding_eval.run import NAMESPACE, RunOptions, result_row, run, seed_store
from agmem.control import is_auto_injection_eligible
from agmem.stores.sqlite_doc import SqliteDocStore

RECIPE = Path(__file__).resolve().parents[1] / "experiments" / "coding_eval" / "recipe.json"
TASKS = json.loads((RECIPE.parent / "tasks.json").read_text(encoding="utf-8"))

STUB_AGENT = """
import json, os, sys
from pathlib import Path
prompt = sys.stdin.read()
FIX = (
    "from __future__ import annotations\\n\\nfrom datetime import datetime, timedelta\\n\\n\\n"
    "def is_on_or_before_cutoff(timestamp: str, cutoff_date: str) -> bool:\\n"
    "    event_time = datetime.fromisoformat(timestamp)\\n"
    "    end = datetime.fromisoformat(f\\"{cutoff_date}T00:00:00+00:00\\") + timedelta(days=1)\\n"
    "    return event_time < end\\n"
)
target = Path("fixtures") / "timezone_cutoff" / "task.py"
if target.exists() and os.environ.get("CODING_EVAL_STUB_FIX") == "1":
    target.write_text(FIX)
print(json.dumps({
    "type": "result", "is_error": False, "result": "done", "duration_ms": 1234,
    "num_turns": 3, "total_cost_usd": 0.0123,
    "usage": {"input_tokens": 100, "cache_creation_input_tokens": 20,
              "cache_read_input_tokens": 30, "output_tokens": 40},
}))
"""


@pytest.fixture
def stub_agent(tmp_path: Path) -> str:
    script = tmp_path / "stub_agent.py"
    script.write_text(STUB_AGENT, encoding="utf-8")
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"


def _task(scenario: str) -> dict:
    return next(t for t in TASKS if t["scenario"] == scenario)


def _eligible_texts(data_dir: Path) -> tuple[list[str], list[str]]:
    store = SqliteDocStore(data_dir / NAMESPACE / "memory.db")
    try:
        episodes = [e.content for e in store.list_episodes(NAMESPACE)]
        runbooks = [
            str(d["content"])
            for d in store.list_items("runbooks", namespace=NAMESPACE)
            if is_auto_injection_eligible(d)
        ]
        return episodes, runbooks
    finally:
        store.close()


def test_seed_store_gives_each_arm_the_memory_the_scenario_is_about(tmp_path: Path):
    ws = tmp_path / "ws"
    stale = _task("corrected_stale_memory")
    harmful = _task("harmful_memory")

    assert seed_store(tmp_path / "none", "none", stale, ws) == 0
    assert _eligible_texts(tmp_path / "none") == ([], [])

    # raw: every fixture, stale and harmful included — preservation has no controls
    assert seed_store(tmp_path / "raw", "raw", stale, ws) == 2
    episodes, runbooks = _eligible_texts(tmp_path / "raw")
    assert len(episodes) == 2 and runbooks == []
    assert any("userId" in e and "Stale" in e for e in episodes)
    assert seed_store(tmp_path / "raw-h", "raw", harmful, ws) == 1

    # runbook: the control layer applies — superseded is disabled, harmful is excluded
    assert seed_store(tmp_path / "rb", "runbook", stale, ws) == 1
    episodes, runbooks = _eligible_texts(tmp_path / "rb")
    assert episodes == [] and len(runbooks) == 1 and "owner_id" in runbooks[0]
    assert seed_store(tmp_path / "rb-h", "runbook", harmful, ws) == 0
    assert _eligible_texts(tmp_path / "rb-h") == ([], [])


def test_result_row_measures_what_it_can_and_leaves_the_rest_null():
    job = {"job_id": "j", "task_id": "t", "arm": "raw_memory", "repeat": 1}
    report = {
        "duration_ms": 1500,
        "total_cost_usd": 0.02,
        "usage": {"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 7},
    }
    row = result_row(job, report, False, "CACHE_DOES_NOT_BYPASS_REVOCATION", "harmful_memory", 99)
    assert row["success"] is False
    assert row["harmful_regression"] is True
    assert row["correction_required"] is None
    assert row["re_explanation_required"] is None
    assert row["latency_ms"] == 1500
    assert row["cost"]["prompt_tokens"] == 15
    assert row["cost"]["completion_tokens"] == 7
    assert row["cost"]["provider_usd"] == 0.02
    assert row["cost"]["judge_tokens"] is None

    # no agent report at all: wall clock stands in, tokens and usd stay unknown
    bare = result_row(job, {}, True, None, "constraints_decisions", 99)
    assert bare["latency_ms"] == 99
    assert bare["cost"]["prompt_tokens"] is None
    assert bare["cost"]["provider_usd"] is None
    assert bare["harmful_regression"] is None

    # the corrected-field scenario decides correction_required, not harmful_regression
    stale = result_row(job, {}, False, "API_FIELD_USER_ID_REJECTED", "corrected_stale_memory", 1)
    assert stale["correction_required"] is True and stale["harmful_regression"] is None


def test_run_stages_probes_the_hook_runs_the_agent_and_validates_results(
    tmp_path: Path, stub_agent: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("AGMEM_NO_DAEMON", "1")
    monkeypatch.setenv("CODING_EVAL_STUB_FIX", "1")
    manifest = prepare(RECIPE)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
    tz_none = "coding-timezone-cutoff-001__none_baseline__r1"
    tz_raw = "coding-timezone-cutoff-001__raw_memory__r1"
    harm_raw = "coding-cache-validation-001__raw_memory__r1"
    out = tmp_path / "out"
    report = run(
        RunOptions(
            manifest_path=manifest_path,
            work_dir=tmp_path / "work",
            agent_command=stub_agent,
            only_jobs=frozenset({tz_none, tz_raw, harm_raw}),
            timeout_s=60,
            output_root=out,
        )
    )

    by_id = {o.job_id: o for o in report.outcomes}
    assert set(by_id) == {tz_none, tz_raw, harm_raw}
    # the stub fixes the timezone task in both arms; it leaves the harmful task alone, which
    # fails on the revocation check — exactly the regression that scenario watches for
    assert by_id[tz_none].row["success"] is True
    assert by_id[tz_raw].row["success"] is True
    assert by_id[harm_raw].row["success"] is False
    assert by_id[harm_raw].acceptance_failed_check == "CACHE_DOES_NOT_BYPASS_REVOCATION"
    assert by_id[harm_raw].row["harmful_regression"] is True
    # the product's own hook served the raw arm its memory and the none arm nothing
    assert by_id[tz_none].injected_chars == 0
    assert by_id[tz_raw].injected_chars > 0
    injected = (tmp_path / "work" / tz_raw / "injected.txt").read_text(encoding="utf-8")
    assert "inclusive UTC day" in injected
    # usage flowed through from the agent report
    assert by_id[tz_raw].row["cost"]["prompt_tokens"] == 150
    assert by_id[tz_raw].row["cost"]["provider_usd"] == 0.0123
    assert by_id[tz_raw].row["latency_ms"] == 1234
    # rows, the combined results and a summary landed in the output root; the manifest's
    # validator accepts the rows it got and reports only the jobs a partial run skipped
    assert json.loads((out / f"{tz_raw}.result.json").read_text())["job_id"] == tz_raw
    assert len(json.loads((out / "results.json").read_text())) == 3
    summary = json.loads((out / "summary.json").read_text())
    assert summary["by_arm"]["raw_memory"]["jobs"] == 2
    assert summary["by_arm"]["raw_memory"]["harmful_regressions"] == 1
    assert report.verification_valid is False
    assert report.verification_errors
    assert all(e.startswith("missing result for job") for e in report.verification_errors)
    # the modified file was kept as evidence next to the job
    after = (tmp_path / "work" / tz_raw / "after.timezone_cutoff.task.py").read_text()
    assert "timedelta(days=1)" in after
