"""Drive the preparation CLI in a fresh process with network access denied."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

MODULE = "agmem.bench.lme_v2_tools"


@pytest.fixture
def offline_env(tmp_path: Path) -> dict[str, str]:
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import sys\n"
        "def guard(event, args):\n"
        "    if event in {'socket.connect', 'socket.getaddrinfo'}:\n"
        "        raise RuntimeError('network forbidden in preparation tests')\n"
        "sys.addaudithook(guard)\n",
        encoding="utf-8",
    )
    env = {
        name: os.environ[name] for name in ("PATH", "HOME", "LANG", "TMPDIR") if name in os.environ
    }
    env["PYTHONPATH"] = os.pathsep.join(
        [str(guard), str(Path(__file__).resolve().parents[1] / "src")]
    )
    return env


def test_help_exposes_only_preparation_commands(offline_env: dict[str, str]) -> None:
    # Given a fresh process whose network calls are denied.
    command = [sys.executable, "-m", MODULE, "--help"]
    # When invoking the actual CLI.
    result = subprocess.run(
        command, env=offline_env, capture_output=True, text=True, timeout=20, check=False
    )
    # Then it exposes preparation commands without importing the paid harness.
    assert result.returncode == 0, result.stderr
    assert "{prepare,verify,audit,diagnose,costs}" in result.stdout


@pytest.mark.parametrize("action", ["prepare", "verify", "audit", "diagnose", "costs"])
def test_missing_input_fails_without_creating_output(
    tmp_path: Path, offline_env: dict[str, str], action: str
) -> None:
    # Given a nonexistent source and a reserved output path.
    output = tmp_path / "output.json"
    command = [sys.executable, "-m", MODULE, action, str(tmp_path / "missing")]
    if action != "verify":
        command.extend(["--output", str(output)])
    # When an offline command cannot read its inputs.
    result = subprocess.run(
        command, env=offline_env, capture_output=True, text=True, timeout=20, check=False
    )
    # Then the CLI reports the error and leaves the output untouched.
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_run_is_not_a_preparation_command(offline_env: dict[str, str]) -> None:
    command = [sys.executable, "-m", MODULE, "run", "input.json"]
    result = subprocess.run(
        command, env=offline_env, capture_output=True, text=True, timeout=20, check=False
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


@pytest.fixture
def scored_root(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    arm = root / "raw_vector"
    arm.mkdir(parents=True)
    row = {
        "question_id": "q1",
        "question_type": "static-environment",
        "category": "static",
        "is_abstention_problem": False,
        "score_bool": True,
        "score": 1.0,
        "is_unknown": False,
    }
    (arm / "per_question.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    aggregate = {
        "overall": {
            "count_all_questions": 1,
            "count_non_abstention": 1,
            "count_abstention": 0,
            "overall_full_set": 1.0,
            "overall_non_abstention_only": 1.0,
            "overall_abstention_only": None,
        },
        "non_abstention_by_category": {
            category: {
                "count": 1 if category == "static" else 0,
                "pct_correct": 1.0 if category == "static" else None,
                "pct_answered_wrong": 0.0 if category == "static" else None,
                "pct_unknown": 0.0 if category == "static" else None,
            }
            for category in ("static", "dynamic", "procedure", "gotchas")
        },
        "abstention_by_category": {
            category: {
                "count": 0,
                "pct_correct": None,
                "pct_answered_wrong": None,
                "pct_unknown": None,
            }
            for category in ("static-abs", "dynamic-abs", "procedure-abs")
        },
    }
    (arm / "aggregated_metrics.json").write_text(json.dumps(aggregate), encoding="utf-8")
    return root


def test_audit_writes_json_without_touching_source(
    scored_root: Path, offline_env: dict[str, str], tmp_path: Path
) -> None:
    source = scored_root / "raw_vector" / "per_question.jsonl"
    original = source.read_bytes()
    output = tmp_path / "audit.json"
    command = [sys.executable, "-m", MODULE, "audit", str(scored_root), "--output", str(output)]
    result = subprocess.run(
        command, env=offline_env, capture_output=True, text=True, timeout=20, check=False
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["arms"]
    assert source.read_bytes() == original


def test_existing_output_is_preserved(
    scored_root: Path, offline_env: dict[str, str], tmp_path: Path
) -> None:
    output = tmp_path / "audit.json"
    output.write_text("existing result", encoding="utf-8")
    command = [sys.executable, "-m", MODULE, "audit", str(scored_root), "--output", str(output)]
    result = subprocess.run(
        command, env=offline_env, capture_output=True, text=True, timeout=20, check=False
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert output.read_text(encoding="utf-8") == "existing result"


@pytest.fixture
def prepared_recipe(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    (data / "haystacks").mkdir(parents=True)
    (data / "questions.jsonl").write_text(
        '{"id":"q1","domain":"web","answer":"payload-must-stay-local"}\n', encoding="utf-8"
    )
    (data / "trajectories.jsonl").write_text('{"id":"t1"}\n', encoding="utf-8")
    (data / "haystacks" / "lme_v2_small.json").write_text('{"q1":["t1"]}', encoding="utf-8")
    (tmp_path / "config.toml").write_text('[profile]\nname = "lite"\n', encoding="utf-8")
    (tmp_path / "source.py").write_text("baseline = 1\n", encoding="utf-8")
    recipe = {
        "schema_version": 1,
        "study": "cli-smoke",
        "domain": "web",
        "tier": "small",
        "data_root": "data",
        "config_paths": ["config.toml"],
        "source_files": ["source.py"],
        "reader": "fixture-reader",
        "judge": "fixture-judge",
        "arms": [{"name": "raw_vector", "write": "raw", "read": "vector"}],
        "repeats": 2,
        "output_root": "future-results",
        "costs": {},
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    return path


def test_prepare_then_verify_round_trip_without_network(
    prepared_recipe: Path, offline_env: dict[str, str], tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.json"
    command = [
        sys.executable,
        "-m",
        MODULE,
        "prepare",
        str(prepared_recipe),
        "--output",
        str(manifest),
    ]
    prepared = subprocess.run(
        command, env=offline_env, capture_output=True, text=True, timeout=20, check=False
    )
    assert prepared.returncode == 0, prepared.stderr
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(payload["jobs"]) == 2
    assert len(payload["selected_question_ids"]) == 1
    assert len(payload["missing_cost_components"]) == 6
    assert payload["estimated_total_usd"] is None
    assert "payload-must-stay-local" not in manifest.read_text(encoding="utf-8")
    assert not (tmp_path / "future-results").exists()
    verified = subprocess.run(
        [sys.executable, "-m", MODULE, "verify", str(manifest)],
        env=offline_env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["valid"] is True


def test_verify_exits_nonzero_when_config_changes(
    prepared_recipe: Path, offline_env: dict[str, str], tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [sys.executable, "-m", MODULE, "prepare", str(prepared_recipe), "--output", str(manifest)],
        env=offline_env,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    (tmp_path / "config.toml").write_text('[profile]\nname = "other"\n', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", MODULE, "verify", str(manifest)],
        env=offline_env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["valid"] is False
    assert report["errors"]


@pytest.mark.parametrize("payload", ["[]", "{}", '{"recipe_path":false}', "broken-json"])
def test_invalid_manifest_is_a_reported_error(
    tmp_path: Path, offline_env: dict[str, str], payload: str
) -> None:
    path = tmp_path / "broken.json"
    path.write_text(payload, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", MODULE, "verify", str(path)],
        env=offline_env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("payload", ["[]", "{}", "broken-json"])
def test_invalid_recipe_is_a_reported_error(
    tmp_path: Path, offline_env: dict[str, str], payload: str
) -> None:
    path = tmp_path / "broken.json"
    path.write_text(payload, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", MODULE, "prepare", str(path)],
        env=offline_env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
