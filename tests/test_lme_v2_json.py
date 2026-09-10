from __future__ import annotations

import json
from pathlib import Path

import pytest

from agmem.bench.lme_v2_tools.audit import audit_root
from agmem.bench.lme_v2_tools.manifest import canonical_json
from agmem.bench.lme_v2_tools.prepare import prepare, verify


@pytest.fixture
def recipe_path(tmp_path: Path) -> Path:
    (tmp_path / "haystacks").mkdir()
    (tmp_path / "questions.jsonl").write_text('{"id":"q1","domain":"web"}\n')
    (tmp_path / "trajectories.jsonl").write_text('{"id":"t1"}\n')
    (tmp_path / "haystacks/lme_v2_small.json").write_text('{"q1":["t1"]}')
    (tmp_path / "config.toml").write_text('[profile]\nname="lite"\n')
    (tmp_path / "source.py").write_text("baseline = 1\n")
    recipe = {
        "schema_version": 1,
        "study": "duplicates",
        "domain": "web",
        "tier": "small",
        "data_root": ".",
        "config_paths": ["config.toml"],
        "source_files": ["source.py"],
        "reader": "fixture",
        "judge": "fixture",
        "repeats": 1,
        "arms": [{"name": "raw", "write": "raw", "read": "vector"}],
        "output_root": "future",
        "costs": {},
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    return path


def test_verify_rejects_duplicate_nested_job_paths(recipe_path: Path, tmp_path: Path) -> None:
    stored = canonical_json(prepare(recipe_path))
    stored = stored.replace('"output_path":', '"output_path":"tampered", "output_path":', 1)
    path = tmp_path / "manifest.json"
    path.write_text(stored)
    result = verify(path)
    assert not result.valid
    assert "duplicate JSON key" in " ".join(result.errors)


@pytest.mark.parametrize(
    "target",
    ["recipe.json", "haystacks/lme_v2_small.json", "questions.jsonl", "trajectories.jsonl"],
)
def test_prepare_rejects_duplicate_keys_in_its_inputs(
    recipe_path: Path, tmp_path: Path, target: str
) -> None:
    path = tmp_path / target
    original = path.read_text()
    key = (
        "schema_version"
        if target == "recipe.json"
        else "q1"
        if target.startswith("haystacks")
        else "id"
    )
    replacement = f'"{key}":null, "{key}":'
    path.write_text(original.replace(f'"{key}":', replacement, 1))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        prepare(recipe_path)


@pytest.mark.parametrize("target", ["aggregated_metrics.json", "per_question.jsonl"])
def test_audit_rejects_duplicate_keys_before_loading_results(tmp_path: Path, target: str) -> None:
    (tmp_path / "aggregated_metrics.json").write_text("{}")
    (tmp_path / "per_question.jsonl").write_text("{}\n")
    payload = (
        '{"overall":{},"overall":{}}'
        if target.startswith("aggregated")
        else '{"score_bool":false,"score_bool":true}\n'
    )
    (tmp_path / target).write_text(payload)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        audit_root(tmp_path)
