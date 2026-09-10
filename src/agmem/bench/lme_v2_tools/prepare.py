"""Prepare and verify offline LongMemEval-V2 measurement manifests.

This module reads only explicit recipe inputs and dataset indexes. It never
imports the benchmark harness, memory adapter or provider clients, so callers
can run it before approving any paid measurement.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from json import JSONDecodeError
from pathlib import Path

from agmem.bench.lme_v2_tools.manifest import (
    SCHEMA_VERSION,
    ArmPlan,
    Fingerprint,
    Manifest,
    PlannedJob,
    Verification,
    canonical_json,
    canonical_json_value,
)
from agmem.bench.lme_v2_tools.recipe import Recipe, RecipeError, load_recipe

from .json_io import JsonObject, parse_json, parse_json_object
from .store_inventory import StoreInventoryError, fixed_store_plans


def prepare(recipe_path: Path) -> Manifest:
    """Build a frozen manifest without writing files or executing planned jobs."""
    recipe = load_recipe(recipe_path)
    selected_question_ids = _selected_question_ids(recipe)
    haystack_by_question = _haystack(recipe.haystack_path)
    trajectory_ids = _selected_trajectory_ids(selected_question_ids, haystack_by_question)
    _require_trajectories(recipe.trajectories_path, trajectory_ids)
    jobs = _jobs(recipe)
    try:
        fixed_stores = fixed_store_plans(recipe.fixed_stores)
    except StoreInventoryError as error:
        raise RecipeError(str(error)) from error
    cost_values = _cost_values(recipe)
    missing_costs = tuple(component for component, value in cost_values if value is None)
    total = (
        None
        if missing_costs
        else round(sum(value for _, value in cost_values if value is not None), 6)
    )
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        recipe_path=str(recipe.path),
        recipe_sha256=_sha256(recipe.path),
        study=recipe.study,
        domain=recipe.domain,
        tier=recipe.tier,
        data_root=str(recipe.data_root),
        questions=_fingerprint(recipe.questions_path),
        haystack=_fingerprint(recipe.haystack_path),
        trajectories=_fingerprint(recipe.trajectories_path),
        configs=tuple(_fingerprint(path) for path in recipe.config_paths),
        source_files=tuple(_fingerprint(path) for path in recipe.source_files),
        reader=recipe.reader,
        judge=recipe.judge,
        arms=tuple(
            ArmPlan(
                name=arm.name,
                write=arm.write,
                read=arm.read,
                store=arm.store,
                query_strategy=arm.query_strategy,
                settings=tuple(_fingerprint(path) for path in arm.settings_paths),
                fixed_store_id=arm.fixed_store_id,
            )
            for arm in recipe.arms
        ),
        fixed_stores=fixed_stores,
        jobs=jobs,
        selected_question_ids=selected_question_ids,
        selected_trajectory_ids=trajectory_ids,
        costs=recipe.costs,
        missing_cost_components=missing_costs,
        estimated_total_usd=total,
    )
    return manifest


def verify(manifest_path: Path) -> Verification:
    """Regenerate a stored manifest from its recipe and report drift separately from cost gaps."""
    try:
        stored_json = manifest_path.expanduser().resolve().read_text(encoding="utf-8")
        raw = parse_json_object(stored_json, "manifest")
        recipe_path = raw.get("recipe_path")
        if not isinstance(recipe_path, str) or not recipe_path:
            raise RecipeError("manifest recipe_path must be a non-empty string")
        regenerated = prepare(Path(recipe_path))
    except (FileNotFoundError, TypeError, ValueError, RecipeError) as error:
        return Verification(valid=False, errors=(str(error),), missing_cost_components=())
    regenerated_json = canonical_json(regenerated)
    errors = (
        ()
        if canonical_json_value(raw) == regenerated_json
        else ("manifest drift: regenerated manifest differs",)
    )
    return Verification(
        valid=not errors,
        errors=errors,
        missing_cost_components=regenerated.missing_cost_components,
    )


def _selected_question_ids(recipe: Recipe) -> tuple[str, ...]:
    seen: set[str] = set()
    selected: list[str] = []
    for row in _jsonl_objects(recipe.questions_path):
        question_id = _identifier(row, "question")
        if question_id in seen:
            raise RecipeError(f"duplicate question id: {question_id}")
        seen.add(question_id)
        domain = row.get("domain")
        if not isinstance(domain, str) or not domain.strip():
            raise RecipeError(f"question {question_id} must have a non-empty domain")
        if domain == recipe.domain:
            selected.append(question_id)
    if not selected:
        raise RecipeError(f"no questions selected for domain: {recipe.domain}")
    return tuple(selected)


def _haystack(path: Path) -> dict[str, tuple[str, ...]]:
    try:
        raw = parse_json_object(path.read_text(encoding="utf-8"), "haystack")
    except FileNotFoundError as error:
        raise RecipeError(f"haystack not found: {path}") from error
    except (JSONDecodeError, TypeError, ValueError) as error:
        raise RecipeError(f"invalid haystack JSON: {error}") from error
    parsed: dict[str, tuple[str, ...]] = {}
    for question_id, value in raw.items():
        if not isinstance(value, list):
            raise RecipeError("haystack entries must map question ids to trajectory id lists")
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise RecipeError(
                f"haystack for {question_id} must contain non-empty trajectory strings"
            )
        trajectory_ids = tuple(item for item in value if isinstance(item, str))
        duplicates = sorted({item for item in trajectory_ids if trajectory_ids.count(item) > 1})
        if duplicates:
            raise RecipeError(
                f"duplicate trajectory id in haystack for {question_id}: {duplicates[0]}"
            )
        parsed[question_id] = trajectory_ids
    return parsed


def _selected_trajectory_ids(
    selected_question_ids: tuple[str, ...], haystack_by_question: dict[str, tuple[str, ...]]
) -> tuple[str, ...]:
    missing_questions = [
        question_id
        for question_id in selected_question_ids
        if question_id not in haystack_by_question
    ]
    if missing_questions:
        raise RecipeError(f"missing haystack for question id: {missing_questions[0]}")
    trajectory_ids = {
        trajectory_id
        for question_id in selected_question_ids
        for trajectory_id in haystack_by_question[question_id]
    }
    if not trajectory_ids:
        raise RecipeError("selected haystack has no trajectories")
    return tuple(sorted(trajectory_ids))


def _require_trajectories(path: Path, wanted: tuple[str, ...]) -> None:
    available: set[str] = set()
    for row in _jsonl_objects(path):
        trajectory_id = _identifier(row, "trajectory")
        if trajectory_id in available:
            raise RecipeError(f"duplicate trajectory id: {trajectory_id}")
        available.add(trajectory_id)
    missing = [trajectory_id for trajectory_id in wanted if trajectory_id not in available]
    if missing:
        raise RecipeError(f"missing trajectory: {missing[0]}")


def _jobs(recipe: Recipe) -> tuple[PlannedJob, ...]:
    if recipe.output_root.is_symlink() or (
        recipe.output_root.exists() and not recipe.output_root.is_dir()
    ):
        raise RecipeError(f"output root exists but is not a directory: {recipe.output_root}")
    jobs: list[PlannedJob] = []
    seen: set[Path] = set()
    for arm in recipe.arms:
        for repeat in range(1, recipe.repeats + 1):
            output_path = recipe.output_root / f"{arm.name}_r{repeat}"
            if output_path in seen:
                raise RecipeError(f"duplicate planned output path: {output_path}")
            if output_path.exists() or output_path.is_symlink():
                raise RecipeError(f"output path already exists: {output_path}")
            seen.add(output_path)
            jobs.append(
                PlannedJob(
                    arm=arm.name,
                    repeat=repeat,
                    output_path=str(output_path),
                    memory_mode=arm.store,
                    fixed_store_id=arm.fixed_store_id,
                )
            )
    return tuple(jobs)


def _jsonl_objects(path: Path) -> Iterator[JsonObject]:
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError as error:
        raise RecipeError(f"jsonl input not found: {path}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = parse_json(line)
            except (JSONDecodeError, TypeError, ValueError) as error:
                raise RecipeError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(raw, dict):
                raise RecipeError(f"{path}:{line_number} must be an object")
            yield raw


def _identifier(row: JsonObject, kind: str) -> str:
    raw = row.get("id", row.get("question_id"))
    if not isinstance(raw, str) or not raw.strip():
        raise RecipeError(f"{kind} row must have non-empty id")
    return raw


def _fingerprint(path: Path) -> Fingerprint:
    return Fingerprint(path=str(path), sha256=_sha256(path), size_bytes=path.stat().st_size)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    except FileNotFoundError as error:
        raise RecipeError(f"input file not found: {path}") from error
    return hasher.hexdigest()


def _cost_values(recipe: Recipe) -> tuple[tuple[str, float | None], ...]:
    return (
        ("reader", recipe.costs.reader),
        ("retrieval", recipe.costs.retrieval),
        ("write", recipe.costs.write),
        ("embedding", recipe.costs.embedding),
        ("judge", recipe.costs.judge),
        ("retries", recipe.costs.retries),
    )
