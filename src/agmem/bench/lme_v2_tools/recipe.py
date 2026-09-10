"""Boundary parser for offline LongMemEval-V2 preparation recipes.

Recipes are the only untrusted JSON boundary for preparation. This module
turns that JSON into frozen dataclasses, rejects unknown keys, and resolves all
allowlisted paths relative to the recipe file before manifest generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Final

from agmem.bench.lme_v2_tools.manifest import COST_COMPONENTS, CostCoverage
from agmem.bench.lme_v2_tools.recipe_values import (
    RecipeError,
    choice,
    cost,
    optional_str,
    optional_str_list,
    positive_int,
    reject_unknown_keys,
    required_int,
    required_str,
    required_str_list,
    resolve_path,
    resolve_snapshot_root,
)

from .json_io import JsonObject, JsonValue, parse_json_object

RECIPE_SCHEMA_VERSION: Final = 1
WRITES: Final = ("raw", "experience")
READS: Final = ("vector", "explorer")
STORES: Final = ("fresh", "fixed")
DOMAINS: Final = ("web", "enterprise")
TIERS: Final = ("small", "medium")
ARM_NAME_RE: Final = re.compile(r"^[A-Za-z0-9_-]+$")
ROOT_KEYS: Final = frozenset(
    {
        "schema_version",
        "study",
        "domain",
        "tier",
        "data_root",
        "config_paths",
        "source_files",
        "reader",
        "judge",
        "arms",
        "fixed_stores",
        "repeats",
        "output_root",
        "costs",
    }
)
ARM_KEYS: Final = frozenset(
    {"name", "write", "read", "store", "query_strategy", "settings_paths", "fixed_store_id"}
)
FIXED_STORE_KEYS: Final = frozenset({"id", "write", "path"})
QUERY_STRATEGIES: Final = {"vector": "direct", "explorer": "bounded_explorer"}


@dataclass(frozen=True, slots=True)
class ArmRecipe:
    """Parsed arm choice; write/read/store values have already been checked."""

    name: str
    write: str
    read: str
    store: str
    query_strategy: str
    settings_paths: tuple[Path, ...]
    fixed_store_id: str | None


@dataclass(frozen=True, slots=True)
class FixedStoreRecipe:
    id: str
    write: str
    path: Path


@dataclass(frozen=True, slots=True)
class Recipe:
    """Fully parsed recipe with absolute allowlisted input and output paths."""

    path: Path
    schema_version: int
    study: str
    domain: str
    tier: str
    data_root: Path
    questions_path: Path
    haystack_path: Path
    trajectories_path: Path
    config_paths: tuple[Path, ...]
    source_files: tuple[Path, ...]
    reader: str
    judge: str
    arms: tuple[ArmRecipe, ...]
    fixed_stores: tuple[FixedStoreRecipe, ...]
    repeats: int
    output_root: Path
    costs: CostCoverage


def load_recipe(path: Path) -> Recipe:
    """Parse a JSON recipe and resolve its explicit file allowlist."""
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
    domain = choice(required_str(raw, "domain"), DOMAINS, "domain")
    tier = choice(required_str(raw, "tier"), TIERS, "tier")
    base_dir = recipe_path.parent
    data_root = resolve_path(base_dir, required_str(raw, "data_root"))
    arms = _arms(raw, base_dir)
    fixed_stores = _fixed_stores(raw, base_dir)
    _check_fixed_store_usage(arms, fixed_stores)
    return Recipe(
        path=recipe_path,
        schema_version=schema_version,
        study=required_str(raw, "study"),
        domain=domain,
        tier=tier,
        data_root=data_root,
        questions_path=data_root / "questions.jsonl",
        haystack_path=data_root / "haystacks" / f"lme_v2_{tier}.json",
        trajectories_path=data_root / "trajectories.jsonl",
        config_paths=tuple(
            resolve_path(base_dir, item) for item in required_str_list(raw, "config_paths")
        ),
        source_files=tuple(
            resolve_path(base_dir, item) for item in required_str_list(raw, "source_files")
        ),
        reader=required_str(raw, "reader"),
        judge=required_str(raw, "judge"),
        arms=arms,
        fixed_stores=fixed_stores,
        repeats=positive_int(raw, "repeats"),
        output_root=resolve_path(base_dir, required_str(raw, "output_root")),
        costs=_costs(raw),
    )


def _arms(raw: JsonObject, base_dir: Path) -> tuple[ArmRecipe, ...]:
    value = raw.get("arms")
    if not isinstance(value, list) or not value:
        raise RecipeError("arms must be a non-empty list")
    arms = tuple(_arm(item, base_dir) for item in value)
    names = [arm.name for arm in arms]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RecipeError(f"duplicate arm name: {duplicates[0]}")
    return arms


def _arm(raw: JsonValue, base_dir: Path) -> ArmRecipe:
    if not isinstance(raw, dict):
        raise RecipeError("arm must be an object")
    reject_unknown_keys(raw, ARM_KEYS, "arm")
    write = required_str(raw, "write")
    read = required_str(raw, "read")
    store = str(raw.get("store", "fresh"))
    if write not in WRITES:
        raise RecipeError(f"unknown write arm: {write}")
    if read not in READS:
        raise RecipeError(f"unknown read arm: {read}")
    if store not in STORES:
        raise RecipeError(f"unknown store mode: {store}")
    fixed_store_id = optional_str(raw, "fixed_store_id")
    if store == "fixed" and fixed_store_id is None:
        raise RecipeError("fixed store mode requires fixed_store_id")
    if store == "fresh" and fixed_store_id is not None:
        raise RecipeError("fresh store mode must not declare fixed_store_id")
    name = required_str(raw, "name")
    if ARM_NAME_RE.fullmatch(name) is None:
        raise RecipeError(f"arm name must be path-safe: {name}")
    return ArmRecipe(
        name=name,
        write=write,
        read=read,
        store=store,
        query_strategy=_query_strategy(raw, read),
        settings_paths=tuple(
            resolve_path(base_dir, item) for item in optional_str_list(raw, "settings_paths")
        ),
        fixed_store_id=fixed_store_id,
    )


def _fixed_stores(raw: JsonObject, base_dir: Path) -> tuple[FixedStoreRecipe, ...]:
    value = raw.get("fixed_stores", [])
    if not isinstance(value, list):
        raise RecipeError("fixed_stores must be a list")
    stores = tuple(_fixed_store(item, base_dir) for item in value)
    ids = [store.id for store in stores]
    duplicates = sorted({store_id for store_id in ids if ids.count(store_id) > 1})
    if duplicates:
        raise RecipeError(f"duplicate fixed store id: {duplicates[0]}")
    paths = [store.path for store in stores]
    duplicate_paths = sorted({path for path in paths if paths.count(path) > 1})
    if duplicate_paths:
        raise RecipeError(f"duplicate fixed store path: {duplicate_paths[0]}")
    return stores


def _fixed_store(raw: JsonValue, base_dir: Path) -> FixedStoreRecipe:
    if not isinstance(raw, dict):
        raise RecipeError("fixed store must be an object")
    reject_unknown_keys(raw, FIXED_STORE_KEYS, "fixed store")
    store_id = required_str(raw, "id")
    if ARM_NAME_RE.fullmatch(store_id) is None:
        raise RecipeError(f"fixed store id must be path-safe: {store_id}")
    write = required_str(raw, "write")
    if write not in WRITES:
        raise RecipeError(f"unknown fixed store write arm: {write}")
    return FixedStoreRecipe(
        id=store_id,
        write=write,
        path=resolve_snapshot_root(base_dir, required_str(raw, "path")),
    )


def _check_fixed_store_usage(
    arms: tuple[ArmRecipe, ...], fixed_stores: tuple[FixedStoreRecipe, ...]
) -> None:
    stores = {store.id: store for store in fixed_stores}
    for arm in arms:
        if arm.store == "fresh":
            continue
        if arm.fixed_store_id not in stores:
            raise RecipeError(f"unknown fixed_store_id: {arm.fixed_store_id}")
        store = stores[arm.fixed_store_id]
        if store.write != arm.write:
            raise RecipeError(
                f"fixed store {store.id} write {store.write} does not match arm {arm.name}"
            )


def _query_strategy(raw: JsonObject, read: str) -> str:
    value = raw.get("query_strategy", QUERY_STRATEGIES[read])
    if not isinstance(value, str) or not value.strip():
        raise RecipeError("query_strategy must be a non-empty string")
    if ARM_NAME_RE.fullmatch(value) is None:
        raise RecipeError(f"query_strategy must be path-safe: {value}")
    return value


def _costs(raw: JsonObject) -> CostCoverage:
    value = raw.get("costs")
    if not isinstance(value, dict):
        raise RecipeError("costs must be an object")
    reject_unknown_keys(value, frozenset(COST_COMPONENTS), "cost")
    return CostCoverage(
        reader=cost(value, "reader"),
        retrieval=cost(value, "retrieval"),
        write=cost(value, "write"),
        embedding=cost(value, "embedding"),
        judge=cost(value, "judge"),
        retries=cost(value, "retries"),
    )
