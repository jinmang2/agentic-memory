from __future__ import annotations

import re
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Final

from agmem.bench.coding_eval.manifest import ARM_MODES
from agmem.bench.lme_v2_tools.json_io import JsonObject, JsonValue, parse_json_object
from agmem.bench.lme_v2_tools.recipe_values import (
    RecipeError,
    optional_str,
    reject_unknown_keys,
    required_int,
    required_str,
    required_str_list,
    resolve_path,
)

RECIPE_SCHEMA_VERSION: Final = 1
NAME_RE: Final = re.compile(r"^[A-Za-z0-9_-]+$")
ROOT_KEYS: Final = frozenset(
    {
        "schema_version",
        "study",
        "tasks_path",
        "settings_paths",
        "source_files",
        "arms",
        "repeats",
        "output_root",
    }
)
ARM_KEYS: Final = frozenset({"name", "memory_mode", "memory_source", "retrieval_mode"})


@dataclass(frozen=True, slots=True)
class ArmRecipe:
    name: str
    memory_mode: str
    memory_source: str | None
    retrieval_mode: str


@dataclass(frozen=True, slots=True)
class Recipe:
    path: Path
    schema_version: int
    study: str
    tasks_path: Path
    settings_paths: tuple[Path, ...]
    source_files: tuple[Path, ...]
    arms: tuple[ArmRecipe, ...]
    repeats: int
    output_root: Path


def load_recipe(path: Path) -> Recipe:
    recipe_path = path.expanduser().resolve()
    try:
        raw = parse_json_object(recipe_path.read_text(encoding="utf-8"), "recipe")
    except FileNotFoundError as error:
        raise RecipeError(f"recipe not found: {recipe_path}") from error
    except (JSONDecodeError, TypeError, ValueError) as error:
        raise RecipeError(f"invalid recipe JSON: {error}") from error
    reject_unknown_keys(raw, ROOT_KEYS, "recipe")
    schema_version = required_int(raw, "schema_version")
    if schema_version != RECIPE_SCHEMA_VERSION:
        raise RecipeError(f"unsupported recipe schema_version: {schema_version}")
    base_dir = recipe_path.parent
    return Recipe(
        path=recipe_path,
        schema_version=schema_version,
        study=required_str(raw, "study"),
        tasks_path=resolve_path(base_dir, required_str(raw, "tasks_path")),
        settings_paths=tuple(
            resolve_path(base_dir, item) for item in required_str_list(raw, "settings_paths")
        ),
        source_files=tuple(
            resolve_path(base_dir, item) for item in required_str_list(raw, "source_files")
        ),
        arms=_arms(raw),
        repeats=_positive_repeats(raw),
        output_root=resolve_path(base_dir, required_str(raw, "output_root")),
    )


def _arms(raw: JsonObject) -> tuple[ArmRecipe, ...]:
    value = raw.get("arms")
    if not isinstance(value, list) or not value:
        raise RecipeError("arms must be a non-empty list")
    arms = tuple(_arm(item) for item in value)
    names = [arm.name for arm in arms]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RecipeError(f"duplicate arm name: {duplicates[0]}")
    modes = tuple(arm.memory_mode for arm in arms)
    if tuple(sorted(modes)) != ARM_MODES:
        raise RecipeError("arms must contain exactly one none, raw, and runbook memory mode")
    return arms


def _arm(raw: JsonValue) -> ArmRecipe:
    if not isinstance(raw, dict):
        raise RecipeError("arm must be an object")
    reject_unknown_keys(raw, ARM_KEYS, "arm")
    name = required_str(raw, "name")
    if NAME_RE.fullmatch(name) is None:
        raise RecipeError(f"arm name must be path-safe: {name}")
    memory_mode = required_str(raw, "memory_mode")
    if memory_mode not in ARM_MODES:
        raise RecipeError(f"unknown memory_mode: {memory_mode}")
    source = optional_str(raw, "memory_source")
    if memory_mode == "none" and source is not None:
        raise RecipeError("none arm must not declare memory_source")
    if memory_mode != "none" and source is None:
        raise RecipeError(f"{memory_mode} arm requires memory_source")
    retrieval_mode = required_str(raw, "retrieval_mode")
    if NAME_RE.fullmatch(retrieval_mode) is None:
        raise RecipeError(f"retrieval_mode must be path-safe: {retrieval_mode}")
    return ArmRecipe(
        name=name,
        memory_mode=memory_mode,
        memory_source=source,
        retrieval_mode=retrieval_mode,
    )


def _positive_repeats(raw: JsonObject) -> int:
    repeats = required_int(raw, "repeats")
    if repeats <= 0:
        raise RecipeError("repeats must be positive")
    return repeats
