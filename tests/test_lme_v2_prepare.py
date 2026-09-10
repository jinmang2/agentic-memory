from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from agmem.bench.lme_v2_tools.json_io import JsonValue, parse_json_object
from agmem.bench.lme_v2_tools.manifest import canonical_json
from agmem.bench.lme_v2_tools.prepare import prepare, verify
from agmem.bench.lme_v2_tools.recipe import RecipeError


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    _ = path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _fixture_recipe(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    _ = (tmp_path / "cfg.toml").write_text('[profile]\nname = "lite"\n', encoding="utf-8")
    source = tmp_path / "adapter.py"
    _ = source.write_text("MEMORY_TYPE = 'agmem'\n", encoding="utf-8")
    _write_jsonl(
        data / "questions.jsonl",
        [
            {"id": "q1", "domain": "web", "question": "hidden", "answer": "hidden"},
            {"id": "q2", "domain": "enterprise", "question": "hidden", "answer": "hidden"},
        ],
    )
    _write_jsonl(
        data / "trajectories.jsonl",
        [{"id": "traj-1", "goal": "hidden"}, {"id": "traj-2", "goal": "hidden"}],
    )
    haystack = data / "haystacks"
    haystack.mkdir()
    _ = (haystack / "lme_v2_small.json").write_text(
        json.dumps({"q1": ["traj-1", "traj-2"], "q2": ["traj-2"]}, sort_keys=True),
        encoding="utf-8",
    )
    recipe = {
        "schema_version": 1,
        "study": "role-policy-ablation",
        "domain": "web",
        "tier": "small",
        "data_root": "data",
        "config_paths": ["cfg.toml"],
        "source_files": ["adapter.py"],
        "reader": "qwen/qwen3.5-9b@openrouter",
        "judge": "gpt-5.2@openai",
        "arms": [
            {"name": "raw_vector", "write": "raw", "read": "vector"},
            {"name": "raw_explorer", "write": "raw", "read": "explorer", "store": "fresh"},
        ],
        "repeats": 2,
        "output_root": "out",
        "costs": {
            "reader": 0.10,
            "retrieval": None,
            "write": 0.02,
            "embedding": 0.03,
            "judge": None,
            "retries": None,
        },
    }
    path = tmp_path / "recipe.json"
    _ = path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _replace_recipe(path: Path, recipe: dict[str, JsonValue]) -> None:
    _ = path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_prepare_writes_a_deterministic_manifest_without_secret_payloads(tmp_path: Path) -> None:
    # Given a complete offline recipe with config, source, questions, haystack and trajectories.
    recipe = _fixture_recipe(tmp_path)

    # When preparing the manifest twice.
    first = prepare(recipe)
    second = prepare(recipe)
    output = tmp_path / "manifest.json"
    _ = output.write_text(canonical_json(first), encoding="utf-8")

    # Then the serialized manifest is stable and contains hashes, not input payload text.
    assert first == second
    assert first.estimated_total_usd is None
    assert first.missing_cost_components == ("retrieval", "judge", "retries")
    assert [
        (arm.name, arm.query_strategy, arm.settings, arm.fixed_store_id) for arm in first.arms
    ] == [
        ("raw_vector", "direct", (), None),
        ("raw_explorer", "bounded_explorer", (), None),
    ]
    assert first.fixed_stores == ()
    assert [Path(job.output_path).name for job in first.jobs] == [
        "raw_vector_r1",
        "raw_vector_r2",
        "raw_explorer_r1",
        "raw_explorer_r2",
    ]
    stored = output.read_text(encoding="utf-8")
    assert "hidden" not in stored
    assert "qwen/qwen3.5-9b@openrouter" in stored
    assert verify(output).valid


def test_verify_accepts_manifest_whitespace_and_key_order_changes(tmp_path: Path) -> None:
    # Given a manifest written through a caller's JSON serializer rather than canonical_json.
    manifest = prepare(_fixture_recipe(tmp_path))
    output = tmp_path / "manifest.json"
    _ = output.write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # When verifying the stored manifest, then semantic equality still passes.
    assert verify(output).valid


def test_verify_rejects_semantic_manifest_tampering(tmp_path: Path) -> None:
    # Given a manifest whose planned job output was changed after preparation.
    manifest = parse_json_object(canonical_json(prepare(_fixture_recipe(tmp_path))), "manifest")
    jobs = manifest["jobs"]
    assert isinstance(jobs, list)
    first_job = jobs[0]
    assert isinstance(first_job, dict)
    first_job["output_path"] = str(tmp_path / "elsewhere")
    output = tmp_path / "manifest.json"
    _ = output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # When verifying it, then the semantic drift is rejected.
    assert not verify(output).valid


def test_verify_reports_drift_separately_from_missing_costs(tmp_path: Path) -> None:
    # Given a stored manifest whose cost estimates are incomplete.
    recipe = _fixture_recipe(tmp_path)
    output = tmp_path / "manifest.json"
    _ = output.write_text(canonical_json(prepare(recipe)), encoding="utf-8")

    # When one allowlisted source file changes after preparation.
    _ = (tmp_path / "adapter.py").write_text("MEMORY_TYPE = 'changed'\n", encoding="utf-8")
    result = verify(output)

    # Then fingerprint integrity fails, while cost gaps remain separately visible.
    assert not result.valid
    assert result.missing_cost_components == ("retrieval", "judge", "retries")
    assert any("manifest drift" in error for error in result.errors)


def test_prepare_rejects_duplicate_question_ids(tmp_path: Path) -> None:
    # Given a recipe whose selected question catalogue has a duplicate ID.
    recipe = _fixture_recipe(tmp_path)
    questions = tmp_path / "data" / "questions.jsonl"
    _write_jsonl(
        questions,
        [
            {"id": "q1", "domain": "web", "question": "one"},
            {"id": "q1", "domain": "web", "question": "two"},
        ],
    )

    # When preparing it, then the manifest is refused before any run inventory is trusted.
    with pytest.raises(RecipeError, match="duplicate question id"):
        _ = prepare(recipe)


def test_prepare_rejects_haystack_references_without_trajectories(tmp_path: Path) -> None:
    # Given a selected question whose haystack references a missing trajectory.
    recipe = _fixture_recipe(tmp_path)
    haystack = tmp_path / "data" / "haystacks" / "lme_v2_small.json"
    _ = haystack.write_text(json.dumps({"q1": ["missing-traj"]}), encoding="utf-8")

    # When preparing it, then the broken data reference is rejected offline.
    with pytest.raises(RecipeError, match="missing trajectory"):
        _ = prepare(recipe)


def test_prepare_rejects_questions_without_domain(tmp_path: Path) -> None:
    # Given a selected question with no explicit domain.
    recipe = _fixture_recipe(tmp_path)
    _write_jsonl(tmp_path / "data" / "questions.jsonl", [{"id": "q1", "question": "one"}])

    # When preparing it, then the ambiguous domain selection is rejected.
    with pytest.raises(RecipeError, match="non-empty domain"):
        _ = prepare(recipe)


def test_prepare_rejects_whitespace_question_ids(tmp_path: Path) -> None:
    # Given a selected question whose ID is only whitespace.
    recipe = _fixture_recipe(tmp_path)
    _write_jsonl(
        tmp_path / "data" / "questions.jsonl",
        [{"id": "  ", "domain": "web", "question": "one"}],
    )

    # When preparing it, then the ambiguous identifier is rejected.
    with pytest.raises(RecipeError, match="non-empty id"):
        _ = prepare(recipe)


def test_prepare_rejects_non_string_haystack_ids(tmp_path: Path) -> None:
    # Given a haystack that would otherwise coerce a numeric trajectory ID to a string.
    recipe = _fixture_recipe(tmp_path)
    haystack = tmp_path / "data" / "haystacks" / "lme_v2_small.json"
    _ = haystack.write_text(json.dumps({"q1": [1]}), encoding="utf-8")

    # When preparing it, then the invalid reference type is rejected.
    with pytest.raises(RecipeError, match="trajectory strings"):
        _ = prepare(recipe)


@pytest.mark.parametrize(
    ("arm_update", "match"),
    [
        ({"name": "../escape"}, "path-safe"),
        ({"store": "fixed"}, "fixed store mode"),
    ],
)
def test_prepare_rejects_deferred_or_unsafe_arm_modes(
    tmp_path: Path, arm_update: dict[str, JsonValue], match: str
) -> None:
    # Given an arm that cannot produce a trustworthy fresh output manifest.
    recipe_path = _fixture_recipe(tmp_path)
    recipe = parse_json_object(recipe_path.read_text(encoding="utf-8"), "recipe")
    arms = recipe["arms"]
    assert isinstance(arms, list)
    first_arm = arms[0]
    assert isinstance(first_arm, dict)
    first_arm.update(arm_update)
    _replace_recipe(recipe_path, recipe)

    # When preparing it, then the ambiguous arm is rejected.
    with pytest.raises(RecipeError, match=match):
        _ = prepare(recipe_path)


@pytest.mark.parametrize("bad_cost", [True, float("nan")])
def test_prepare_rejects_invalid_cost_estimates(tmp_path: Path, bad_cost: JsonValue) -> None:
    # Given a cost slot that cannot be a finite whole-study dollar estimate.
    recipe_path = _fixture_recipe(tmp_path)
    recipe = parse_json_object(recipe_path.read_text(encoding="utf-8"), "recipe")
    costs = recipe["costs"]
    assert isinstance(costs, dict)
    costs["reader"] = bad_cost
    _replace_recipe(recipe_path, recipe)

    # When preparing it, then the cost boundary rejects the value.
    with pytest.raises(RecipeError, match="finite non-negative"):
        _ = prepare(recipe_path)


def test_prepare_rejects_existing_or_duplicate_job_outputs(tmp_path: Path) -> None:
    # Given an output path that already exists for a planned repeat.
    recipe = _fixture_recipe(tmp_path)
    existing = tmp_path / "out" / "raw_vector_r1"
    existing.mkdir(parents=True)

    # When preparing it, then output collision is rejected before any execution.
    with pytest.raises(RecipeError, match="output path already exists"):
        _ = prepare(recipe)


def test_prepare_rejects_blocked_output_roots(tmp_path: Path) -> None:
    # Given an output root that is already a file.
    recipe_path = _fixture_recipe(tmp_path)
    recipe = parse_json_object(recipe_path.read_text(encoding="utf-8"), "recipe")
    blocked_root = tmp_path / "blocked-root"
    _ = blocked_root.write_text("not a directory", encoding="utf-8")
    recipe["output_root"] = str(blocked_root)
    _replace_recipe(recipe_path, recipe)

    # When preparing it, then the root collision is rejected.
    with pytest.raises(RecipeError, match="output root"):
        _ = prepare(recipe_path)


def test_prepare_rejects_dangling_job_output_links(tmp_path: Path) -> None:
    # Given a dangling symlink at a planned output path.
    recipe_path = _fixture_recipe(tmp_path)
    planned = tmp_path / "out" / "raw_vector_r1"
    planned.parent.mkdir(parents=True)
    planned.symlink_to(tmp_path / "missing-target")

    # When preparing it, then the symlink is treated as a collision.
    with pytest.raises(RecipeError, match="output path already exists"):
        _ = prepare(recipe_path)
