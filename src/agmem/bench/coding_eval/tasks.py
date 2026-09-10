from __future__ import annotations

from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

from agmem.bench.coding_eval.manifest import REQUIRED_SCENARIOS, Fingerprint, TaskPlan
from agmem.bench.lme_v2_tools.json_io import JsonObject, JsonValue, parse_json
from agmem.bench.lme_v2_tools.recipe_values import RecipeError, reject_unknown_keys, required_str

TASK_KEYS = frozenset(
    {
        "id",
        "scenario",
        "prompt",
        "fixture_files",
        "test_command",
        "constraints",
        "decisions",
        "acceptance_checks",
        "memory_fixtures",
    }
)
MEMORY_KEYS = frozenset({"kind", "content", "status"})


@dataclass(frozen=True, slots=True)
class CodingTask:
    id: str
    scenario: str
    fixture_files: tuple[Path, ...]
    test_command: tuple[str, ...]
    constraints: tuple[str, ...]
    decisions: tuple[str, ...]
    acceptance_checks: tuple[str, ...]
    memory_fixtures: tuple[str, ...]


def load_tasks(path: Path) -> tuple[CodingTask, ...]:
    base_dir = path.expanduser().resolve().parent
    try:
        raw = parse_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RecipeError(f"tasks not found: {path}") from error
    except (JSONDecodeError, TypeError, ValueError) as error:
        raise RecipeError(f"invalid tasks JSON: {error}") from error
    if not isinstance(raw, list) or not raw:
        raise RecipeError("tasks root must be a non-empty list")
    tasks = tuple(_task(item, base_dir) for item in raw)
    ids = [task.id for task in tasks]
    duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
    if duplicates:
        raise RecipeError(f"duplicate task id: {duplicates[0]}")
    scenarios = {task.scenario for task in tasks}
    missing = [scenario for scenario in REQUIRED_SCENARIOS if scenario not in scenarios]
    if missing:
        raise RecipeError(f"missing required coding scenario: {missing[0]}")
    return tasks


def task_plans(
    tasks: tuple[CodingTask, ...],
    fixture_fingerprints: dict[Path, Fingerprint],
) -> tuple[TaskPlan, ...]:
    return tuple(
        TaskPlan(
            id=task.id,
            scenario=task.scenario,
            fixture_files=tuple(fixture_fingerprints[path] for path in task.fixture_files),
            test_command=task.test_command,
            constraints=task.constraints,
            decisions=task.decisions,
            acceptance_checks=task.acceptance_checks,
            memory_fixtures=task.memory_fixtures,
        )
        for task in tasks
    )


def _task(raw: JsonValue, base_dir: Path) -> CodingTask:
    if not isinstance(raw, dict):
        raise RecipeError("task must be an object")
    reject_unknown_keys(raw, TASK_KEYS, "task")
    scenario = required_str(raw, "scenario")
    if scenario not in REQUIRED_SCENARIOS:
        raise RecipeError(f"unknown coding scenario: {scenario}")
    return CodingTask(
        id=required_str(raw, "id"),
        scenario=scenario,
        fixture_files=tuple(
            (base_dir / item).resolve() for item in _required_strs(raw, "fixture_files")
        ),
        test_command=_required_strs(raw, "test_command"),
        constraints=_required_strs(raw, "constraints"),
        decisions=_required_strs(raw, "decisions"),
        acceptance_checks=_required_strs(raw, "acceptance_checks"),
        memory_fixtures=_memory_fixture_kinds(raw.get("memory_fixtures")),
    )


def _required_strs(raw: JsonObject, key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise RecipeError(f"{key} must be a non-empty string list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise RecipeError(f"{key} must be a non-empty string list")
    return tuple(item for item in value if isinstance(item, str))


def _memory_fixture_kinds(raw: JsonValue | None) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise RecipeError("memory_fixtures must be a non-empty list")
    kinds: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RecipeError("memory fixture must be an object")
        reject_unknown_keys(item, MEMORY_KEYS, "memory fixture")
        kinds.append(required_str(item, "kind"))
        _ = required_str(item, "content")
        _ = required_str(item, "status")
    return tuple(kinds)
