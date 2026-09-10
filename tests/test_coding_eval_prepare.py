from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agmem.bench.coding_eval.manifest import ARM_MODES, COST_COMPONENTS, canonical_json
from agmem.bench.coding_eval.prepare import prepare, verify
from agmem.bench.lme_v2_tools.json_io import JsonValue, parse_json, parse_json_object
from agmem.bench.lme_v2_tools.recipe_values import RecipeError

MODULE = "agmem.bench.coding_eval"
pytest_plugins = ("test_lme_v2_tools_cli",)


def _write_json(path: Path, value: JsonValue) -> None:
    _ = path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[JsonValue]) -> None:
    _ = path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "fixture"
    root.mkdir()
    _ = (root / "settings.json").write_text('{"prepare_only": true}\n', encoding="utf-8")
    _ = (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    for name in ("a", "b", "c", "d"):
        fixture = root / "fixtures" / name
        fixture.mkdir(parents=True)
        _ = (fixture / "task.py").write_text(f"VALUE = {name!r}\n", encoding="utf-8")
        _ = (fixture / "acceptance.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
    _write_json(root / "tasks.json", _tasks())
    _write_json(root / "recipe.json", _recipe())
    return root


def _recipe() -> JsonValue:
    return {
        "schema_version": 1,
        "study": "coding-memory-fixture",
        "tasks_path": "tasks.json",
        "settings_paths": ["settings.json"],
        "source_files": ["source.py"],
        "arms": [
            {"name": "none_baseline", "memory_mode": "none", "retrieval_mode": "disabled"},
            {
                "name": "raw_memory",
                "memory_mode": "raw",
                "memory_source": "raw_store",
                "retrieval_mode": "prompt_recall",
            },
            {
                "name": "runbook_memory",
                "memory_mode": "runbook",
                "memory_source": "runbook_store",
                "retrieval_mode": "prompt_recall",
            },
        ],
        "repeats": 1,
        "output_root": "out",
    }


def _tasks() -> JsonValue:
    return [
        _task("t1", "constraints_decisions", "constraint", "a"),
        _task("t2", "recurring_failure", "recurring_failure", "b"),
        _task("t3", "corrected_stale_memory", "correction", "c"),
        _task("t4", "harmful_memory", "harmful_advice", "d"),
    ]


def _task(task_id: str, scenario: str, memory_kind: str, fixture: str) -> JsonValue:
    return {
        "id": task_id,
        "scenario": scenario,
        "prompt": "Implement the requested coding change.",
        "fixture_files": [f"fixtures/{fixture}/task.py", f"fixtures/{fixture}/acceptance.py"],
        "test_command": ["python", "-m", f"fixtures.{fixture}.acceptance"],
        "constraints": ["keep changes local"],
        "decisions": ["prepare before measurement"],
        "acceptance_checks": ["targeted tests pass"],
        "memory_fixtures": [
            {"kind": memory_kind, "content": f"{scenario} memory", "status": "current"}
        ],
    }


def _blank_result_row(job_id: str, task_id: str, arm: str, repeat: int) -> JsonValue:
    return {
        "job_id": job_id,
        "task_id": task_id,
        "arm": arm,
        "repeat": repeat,
        "success": None,
        "re_explanation_required": None,
        "correction_required": None,
        "harmful_regression": None,
        "latency_ms": None,
        "cost": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "retrieval_tokens": None,
            "memory_write_tokens": None,
            "judge_tokens": None,
            "provider_usd": None,
        },
    }


def test_prepare_freezes_comparable_arms_scenarios_and_blank_outcomes(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)

    manifest = prepare(root / "recipe.json")

    assert [arm.memory_mode for arm in manifest.arms] == list(ARM_MODES)
    assert manifest.scenarios == (
        "constraints_decisions",
        "corrected_stale_memory",
        "harmful_memory",
        "recurring_failure",
    )
    assert len(manifest.jobs) == 12
    assert {job.arm for job in manifest.jobs} == {
        "none_baseline",
        "raw_memory",
        "runbook_memory",
    }
    assert manifest.result_schema.success is None
    assert manifest.result_schema.cost.provider_usd is None
    assert manifest.cost_components == COST_COMPONENTS
    assert manifest.cost_estimate_status == "unknown"
    assert manifest.settings[0].sha256
    assert manifest.source_files[0].sha256
    assert manifest.task_plans[0].test_command == ("python", "-m", "fixtures.a.acceptance")
    assert [Path(item.path).name for item in manifest.task_plans[0].fixture_files] == [
        "task.py",
        "acceptance.py",
    ]
    assert all(item.sha256 for item in manifest.task_plans[0].fixture_files)


def test_verify_rejects_input_drift(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    manifest_path = root / "manifest.json"
    _ = manifest_path.write_text(canonical_json(prepare(root / "recipe.json")), encoding="utf-8")

    _ = (root / "settings.json").write_text('{"prepare_only": false}\n', encoding="utf-8")

    assert verify(manifest_path).errors == ("manifest drift: regenerated manifest differs",)


def test_prepare_rejects_missing_required_scenario(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    tasks = _tasks()
    assert isinstance(tasks, list)
    _write_json(root / "tasks.json", tasks[:-1])

    with pytest.raises(RecipeError, match="missing required coding scenario: harmful_memory"):
        _ = prepare(root / "recipe.json")


def test_prepare_rejects_arm_mismatch(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    recipe = parse_json_object((root / "recipe.json").read_text(encoding="utf-8"), "recipe")
    recipe["arms"] = [
        {"name": "none_baseline", "memory_mode": "none", "retrieval_mode": "disabled"},
        {
            "name": "raw_memory",
            "memory_mode": "raw",
            "memory_source": "raw_store",
            "retrieval_mode": "prompt_recall",
        },
    ]
    _write_json(root / "recipe.json", recipe)

    with pytest.raises(RecipeError, match="exactly one none, raw, and runbook"):
        _ = prepare(root / "recipe.json")


def test_verify_rejects_invalid_result_rows(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    manifest = prepare(root / "recipe.json")
    manifest_path = root / "manifest.json"
    _ = manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    rows: list[JsonValue] = [
        _blank_result_row(job.job_id, job.task_id, job.arm, job.repeat) for job in manifest.jobs
    ]
    assert isinstance(rows[0], dict)
    rows[0]["task_id"] = "wrong-task"
    assert isinstance(rows[0]["cost"], dict)
    rows[0]["cost"]["provider_usd"] = "unknown"
    results_path = root / "results.json"
    _write_rows(results_path, rows)

    verification = verify(manifest_path, results_path)

    assert not verification.valid
    assert "task_id mismatch for job" in " ".join(verification.errors)
    assert "provider_usd must be null or a non-negative number" in verification.errors


def test_verify_allows_completed_result_paths_when_results_are_valid(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    manifest = prepare(root / "recipe.json")
    manifest_path = root / "manifest.json"
    _ = manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    output_root = root / "out"
    output_root.mkdir()
    rows: list[JsonValue] = [
        _blank_result_row(job.job_id, job.task_id, job.arm, job.repeat) for job in manifest.jobs
    ]
    for row, job in zip(rows, manifest.jobs, strict=True):
        _ = Path(job.result_path).write_text(json.dumps(row), encoding="utf-8")
    results_path = root / "results.json"
    _write_rows(results_path, rows)

    verification = verify(manifest_path, results_path)

    assert verification.valid
    assert verification.errors == ()


def test_public_fixture_acceptance_commands_replay_known_failures() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments/coding_eval"
    tasks = parse_json((root / "tasks.json").read_text(encoding="utf-8"))
    assert isinstance(tasks, list)
    failures: dict[str, str] = {}
    for item in tasks:
        assert isinstance(item, dict)
        command = _string_list(item["test_command"], "test_command")
        checks = _string_list(item["acceptance_checks"], "acceptance_checks")
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 1
        failures[str(item["id"])] = result.stdout.strip()
        assert result.stdout.strip() in checks
    assert failures == {
        "coding-cache-validation-001": "CACHE_DOES_NOT_BYPASS_REVOCATION",
        "coding-corrected-api-field-001": "API_FIELD_OWNER_ID_USED",
        "coding-retry-idempotency-001": "RETRY_IDEMPOTENCY_SINGLE_CHARGE",
        "coding-timezone-cutoff-001": "TZ_CUTOFF_INCLUDES_UTC_DAY",
    }


def _string_list(value: JsonValue, key: str) -> list[str]:
    assert isinstance(value, list), key
    assert all(isinstance(part, str) for part in value), key
    return [part for part in value if isinstance(part, str)]


def test_cli_prepare_and_verify_are_dry_run_json_surfaces(
    tmp_path: Path, offline_env: dict[str, str]
) -> None:
    root = _fixture_root(tmp_path)
    manifest_path = root / "manifest.json"
    prepare_result = subprocess.run(
        [
            sys.executable,
            "-m",
            MODULE,
            "prepare",
            str(root / "recipe.json"),
            "--output",
            str(manifest_path),
        ],
        env=offline_env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert prepare_result.returncode == 0, prepare_result.stderr

    verify_result = subprocess.run(
        [sys.executable, "-m", MODULE, "verify", str(manifest_path)],
        env=offline_env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    payload = json.loads(verify_result.stdout)

    assert verify_result.returncode == 0, verify_result.stderr
    assert payload == {"errors": [], "valid": True}
