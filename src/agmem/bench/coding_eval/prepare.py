from __future__ import annotations

import hashlib
from pathlib import Path

from agmem.bench.coding_eval.manifest import (
    COST_COMPONENTS,
    REQUIRED_SCENARIOS,
    SCHEMA_VERSION,
    ArmPlan,
    Fingerprint,
    Manifest,
    PlannedJob,
    Verification,
    blank_outcome_schema,
    canonical_json,
    canonical_json_value,
)
from agmem.bench.coding_eval.recipe import Recipe, load_recipe
from agmem.bench.coding_eval.results import validate_results
from agmem.bench.coding_eval.tasks import CodingTask, load_tasks, task_plans
from agmem.bench.lme_v2_tools.json_io import parse_json_object
from agmem.bench.lme_v2_tools.recipe_values import RecipeError


def prepare(recipe_path: Path) -> Manifest:
    recipe = load_recipe(recipe_path)
    tasks = load_tasks(recipe.tasks_path)
    _require_output_root(recipe.output_root)
    fixture_fingerprints = _fixture_fingerprints(tasks)
    return Manifest(
        schema_version=SCHEMA_VERSION,
        recipe_path=str(recipe.path),
        recipe_sha256=_sha256(recipe.path),
        study=recipe.study,
        tasks=_fingerprint(recipe.tasks_path),
        settings=tuple(_fingerprint(path) for path in recipe.settings_paths),
        source_files=tuple(_fingerprint(path) for path in recipe.source_files),
        arms=tuple(
            ArmPlan(
                name=arm.name,
                memory_mode=arm.memory_mode,
                memory_source=arm.memory_source,
                retrieval_mode=arm.retrieval_mode,
            )
            for arm in recipe.arms
        ),
        task_plans=task_plans(tasks, fixture_fingerprints),
        jobs=_jobs(recipe, tasks, allow_existing_results=False),
        scenarios=tuple(sorted({task.scenario for task in tasks})),
        required_scenarios=REQUIRED_SCENARIOS,
        result_schema=blank_outcome_schema(),
        cost_components=COST_COMPONENTS,
        cost_estimate_status="unknown",
    )


def verify(manifest_path: Path, results_path: Path | None = None) -> Verification:
    try:
        stored = parse_json_object(
            manifest_path.expanduser().resolve().read_text(encoding="utf-8"), "manifest"
        )
        recipe_path = stored.get("recipe_path")
        if not isinstance(recipe_path, str) or not recipe_path:
            raise RecipeError("manifest recipe_path must be a non-empty string")
        regenerated = _prepare_for_verify(Path(recipe_path))
    except (OSError, TypeError, ValueError, RecipeError) as error:
        return Verification(valid=False, errors=(str(error),))
    errors: list[str] = []
    if canonical_json_value(stored) != canonical_json(regenerated):
        errors.append("manifest drift: regenerated manifest differs")
    if results_path is not None:
        errors.extend(validate_results(results_path, regenerated).errors)
    return Verification(valid=not errors, errors=tuple(errors))


def _prepare_for_verify(recipe_path: Path) -> Manifest:
    recipe = load_recipe(recipe_path)
    tasks = load_tasks(recipe.tasks_path)
    _require_output_root(recipe.output_root)
    fixture_fingerprints = _fixture_fingerprints(tasks)
    return Manifest(
        schema_version=SCHEMA_VERSION,
        recipe_path=str(recipe.path),
        recipe_sha256=_sha256(recipe.path),
        study=recipe.study,
        tasks=_fingerprint(recipe.tasks_path),
        settings=tuple(_fingerprint(path) for path in recipe.settings_paths),
        source_files=tuple(_fingerprint(path) for path in recipe.source_files),
        arms=tuple(
            ArmPlan(
                name=arm.name,
                memory_mode=arm.memory_mode,
                memory_source=arm.memory_source,
                retrieval_mode=arm.retrieval_mode,
            )
            for arm in recipe.arms
        ),
        task_plans=task_plans(tasks, fixture_fingerprints),
        jobs=_jobs(recipe, tasks, allow_existing_results=True),
        scenarios=tuple(sorted({task.scenario for task in tasks})),
        required_scenarios=REQUIRED_SCENARIOS,
        result_schema=blank_outcome_schema(),
        cost_components=COST_COMPONENTS,
        cost_estimate_status="unknown",
    )


def _jobs(
    recipe: Recipe, tasks: tuple[CodingTask, ...], *, allow_existing_results: bool
) -> tuple[PlannedJob, ...]:
    jobs: list[PlannedJob] = []
    seen: set[Path] = set()
    for task in tasks:
        for arm in recipe.arms:
            for repeat in range(1, recipe.repeats + 1):
                job_id = f"{task.id}__{arm.name}__r{repeat}"
                result_path = recipe.output_root / f"{job_id}.result.json"
                if result_path in seen:
                    raise RecipeError(f"duplicate planned result path: {result_path}")
                if not allow_existing_results and (
                    result_path.exists() or result_path.is_symlink()
                ):
                    raise RecipeError(f"result path already exists: {result_path}")
                seen.add(result_path)
                jobs.append(
                    PlannedJob(
                        job_id=job_id,
                        task_id=task.id,
                        scenario=task.scenario,
                        arm=arm.name,
                        repeat=repeat,
                        result_path=str(result_path),
                    )
                )
    return tuple(jobs)


def _fixture_fingerprints(tasks: tuple[CodingTask, ...]) -> dict[Path, Fingerprint]:
    paths = {path.resolve() for task in tasks for path in task.fixture_files}
    return {path: _fingerprint(path) for path in paths}


def _require_output_root(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise RecipeError(f"output root exists but is not a directory: {path}")


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
