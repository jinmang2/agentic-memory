from __future__ import annotations

import json
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
    _ = (tmp_path / "adapter.py").write_text("MEMORY_TYPE = 'agmem'\n", encoding="utf-8")
    _write_jsonl(data / "questions.jsonl", [{"id": "q1", "domain": "web"}])
    _write_jsonl(data / "trajectories.jsonl", [{"id": "traj-1"}])
    haystack = data / "haystacks"
    haystack.mkdir()
    _ = (haystack / "lme_v2_small.json").write_text('{"q1": ["traj-1"]}\n')
    recipe: dict[str, JsonValue] = {
        "schema_version": 1,
        "study": "fixed-store-reader-repeat",
        "domain": "web",
        "tier": "small",
        "data_root": "data",
        "config_paths": ["cfg.toml"],
        "source_files": ["adapter.py"],
        "reader": "qwen/qwen3.5-9b@openrouter",
        "judge": "gpt-5.2@openai",
        "arms": [],
        "repeats": 2,
        "output_root": "out",
        "costs": {
            "reader": None,
            "retrieval": None,
            "write": None,
            "embedding": None,
            "judge": None,
            "retries": None,
        },
    }
    path = tmp_path / "recipe.json"
    _replace_recipe(path, recipe)
    return path


def _replace_recipe(path: Path, recipe: dict[str, JsonValue]) -> None:
    _ = path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixed_store(root: Path) -> None:
    (root / "agmem" / "main").mkdir(parents=True)
    for relative in (
        "memory_config.json",
        "agmem_state.json",
        "agmem/main/memory.db",
        "agmem/main/vectors.db",
        "agmem/main/graph.kuzu",
    ):
        _ = (root / relative).write_text(f"{relative}\n", encoding="utf-8")


def _store_entry(store_id: str, write: str, path: str) -> JsonValue:
    return {"id": store_id, "write": write, "path": path}


def _vector_arm() -> JsonValue:
    return {
        "name": "raw_vector_fixed",
        "write": "raw",
        "read": "vector",
        "store": "fixed",
        "fixed_store_id": "raw-baseline",
    }


def _fixed_recipe(recipe_path: Path) -> dict[str, JsonValue]:
    recipe = parse_json_object(recipe_path.read_text(encoding="utf-8"), "recipe")
    recipe["fixed_stores"] = [_store_entry("raw-baseline", "raw", "raw-store")]
    recipe["arms"] = [_vector_arm()]
    return recipe


def test_prepare_records_fixed_store_sharing_and_settings(tmp_path: Path) -> None:
    # Given two read arms sharing one raw memory snapshot fingerprint.
    recipe_path = _fixture_recipe(tmp_path)
    _fixed_store(tmp_path / "raw-store")
    _ = (tmp_path / "vector-settings.json").write_text('{"k": 50}\n', encoding="utf-8")
    _ = (tmp_path / "explorer-settings.json").write_text('{"max_steps": 4}\n')
    recipe = parse_json_object(recipe_path.read_text(encoding="utf-8"), "recipe")
    recipe["fixed_stores"] = [_store_entry("raw-baseline", "raw", "raw-store")]
    recipe["arms"] = [
        {
            "name": "raw_vector_fixed",
            "write": "raw",
            "read": "vector",
            "store": "fixed",
            "fixed_store_id": "raw-baseline",
            "query_strategy": "direct",
            "settings_paths": ["vector-settings.json"],
        },
        {
            "name": "raw_explorer_fixed",
            "write": "raw",
            "read": "explorer",
            "store": "fixed",
            "fixed_store_id": "raw-baseline",
            "query_strategy": "bounded_explorer",
            "settings_paths": ["explorer-settings.json"],
        },
    ]
    _replace_recipe(recipe_path, recipe)

    # When preparing the manifest.
    manifest = prepare(recipe_path)

    # Then shared snapshot and per-arm execution fingerprints are explicit.
    assert manifest.schema_version == 2
    assert [(arm.name, arm.fixed_store_id, arm.query_strategy) for arm in manifest.arms] == [
        ("raw_vector_fixed", "raw-baseline", "direct"),
        ("raw_explorer_fixed", "raw-baseline", "bounded_explorer"),
    ]
    assert [Path(item.path).name for item in manifest.arms[0].settings] == ["vector-settings.json"]
    assert manifest.fixed_stores[0].id == "raw-baseline"
    assert manifest.fixed_stores[0].write == "raw"
    assert [item.path for item in manifest.fixed_stores[0].files] == [
        "memory_config.json",
        "agmem_state.json",
        "agmem/main/memory.db",
        "agmem/main/vectors.db",
        "agmem/main/graph.kuzu",
    ]
    assert {job.memory_mode for job in manifest.jobs} == {"fixed"}
    assert {job.fixed_store_id for job in manifest.jobs} == {"raw-baseline"}


def test_verify_rejects_fixed_store_content_drift(tmp_path: Path) -> None:
    # Given a manifest prepared against a fixed memory snapshot.
    recipe_path = _fixture_recipe(tmp_path)
    _fixed_store(tmp_path / "raw-store")
    _replace_recipe(recipe_path, _fixed_recipe(recipe_path))
    output = tmp_path / "manifest.json"
    _ = output.write_text(canonical_json(prepare(recipe_path)), encoding="utf-8")

    # When one snapshot file changes after preparation.
    _ = (tmp_path / "raw-store" / "agmem_state.json").write_text(
        '{"changed": true}\n', encoding="utf-8"
    )

    # Then verification reports manifest drift.
    assert not verify(output).valid


@pytest.mark.parametrize(
    ("store_update", "match"),
    [
        ({"write": "experience"}, "does not match"),
        ({"path": "missing-store"}, "fixed store root"),
    ],
)
def test_prepare_rejects_unusable_fixed_store_definitions(
    tmp_path: Path, store_update: dict[str, JsonValue], match: str
) -> None:
    # Given a fixed arm whose snapshot definition is incompatible or unavailable.
    recipe_path = _fixture_recipe(tmp_path)
    _fixed_store(tmp_path / "raw-store")
    recipe = _fixed_recipe(recipe_path)
    store: dict[str, JsonValue] = {
        "id": "raw-baseline",
        "write": "raw",
        "path": "raw-store",
    }
    store.update(store_update)
    recipe["fixed_stores"] = [store]
    _replace_recipe(recipe_path, recipe)

    # When preparing it, then the snapshot cannot enter the manifest.
    with pytest.raises(RecipeError, match=match):
        _ = prepare(recipe_path)


def test_prepare_rejects_unallowlisted_fixed_store_files(tmp_path: Path) -> None:
    # Given a fixed snapshot with an extra file outside the loading allowlist.
    recipe_path = _fixture_recipe(tmp_path)
    _fixed_store(tmp_path / "raw-store")
    _ = (tmp_path / "raw-store" / "secret.txt").write_text("secret\n", encoding="utf-8")
    _replace_recipe(recipe_path, _fixed_recipe(recipe_path))

    # When preparing it, then arbitrary snapshot payload is rejected.
    with pytest.raises(RecipeError, match="unallowlisted fixed store file"):
        _ = prepare(recipe_path)


def test_prepare_rejects_fixed_store_root_symlinks(tmp_path: Path) -> None:
    # Given a fixed snapshot path that is a symlink to another store root.
    recipe_path = _fixture_recipe(tmp_path)
    _fixed_store(tmp_path / "real-store")
    (tmp_path / "raw-store").symlink_to(tmp_path / "real-store")
    _replace_recipe(recipe_path, _fixed_recipe(recipe_path))

    # When preparing it, then the unresolved root symlink is rejected.
    with pytest.raises(RecipeError, match="root must not be a symlink"):
        _ = prepare(recipe_path)


def test_prepare_rejects_duplicate_fixed_store_paths(tmp_path: Path) -> None:
    # Given two fixed store ids that point at the same resolved snapshot path.
    recipe_path = _fixture_recipe(tmp_path)
    _fixed_store(tmp_path / "raw-store")
    recipe = _fixed_recipe(recipe_path)
    recipe["fixed_stores"] = [
        _store_entry("raw-baseline", "raw", "raw-store"),
        _store_entry("raw-copy", "raw", "raw-store"),
    ]
    _replace_recipe(recipe_path, recipe)

    # When preparing it, then the contradictory provenance is rejected.
    with pytest.raises(RecipeError, match="duplicate fixed store path"):
        _ = prepare(recipe_path)
