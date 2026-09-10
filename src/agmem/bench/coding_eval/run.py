"""The measurement runner the preparation (docs/28) stopped short of.

One job = one task x one arm x one repeat, run as the product would run it:
a throwaway workspace holding the task's fixture package, a throwaway agmem
store holding exactly that arm's memory, and the user's own hook wiring
pointed at that store through a ``--settings`` overlay. The agent is a
headless ``claude -p`` by default; ``agent_command`` swaps in anything that
reads the prompt on stdin and prints Claude Code's ``--output-format json``
shape on stdout, which is how the tests drive this without a model.

What an arm's store holds is the experiment:

- ``none``: an empty store. The hooks still run, and serve nothing.
- ``raw``: every memory fixture as a user-turn episode, whatever its status.
  Raw preservation has no notion of "superseded" or "harmful".
- ``runbook``: every memory fixture as a runbook, with the control layer
  applied the way the product applies it: a ``superseded`` fixture is
  disabled, a ``harmful`` one carries harmful feedback, and both are excluded
  from auto-injection by ``is_auto_injection_eligible``.

Outcome fields follow the manifest's result schema. ``success`` and
``latency_ms`` are measured; ``harmful_regression`` and
``correction_required`` are derived from the acceptance check that failed on
the scenario built to detect them; ``re_explanation_required`` needs a judge
and stays ``null``. Token and USD figures come from the agent's own usage
report and stay ``null`` when it reports none — never zero.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from agmem.bench.coding_eval.manifest import COST_COMPONENTS
from agmem.bench.coding_eval.prepare import verify
from agmem.bench.coding_eval.recipe import load_recipe
from agmem.bench.lme_v2_tools.json_io import JsonObject, parse_json, parse_json_object
from agmem.bench.lme_v2_tools.recipe_values import RecipeError
from agmem.control import disable_memory, is_auto_injection_eligible
from agmem.core.types import Episode
from agmem.stores.sqlite_doc import SqliteDocStore

DEFAULT_AGENT_COMMAND = (
    "claude -p --settings {settings} --output-format json --dangerously-skip-permissions "
    "--max-turns {max_turns} --model {model}"
)
DEFAULT_MODEL = "sonnet"
DEFAULT_MAX_TURNS = 12
DEFAULT_TIMEOUT_S = 900
NAMESPACE = "coding_eval"
PROMPT_FRAME = (
    "{prompt}\n\n"
    "The code lives under ./fixtures in this directory. Edit the files in place. "
    "Nobody can answer questions in this session, so decide and act."
)
HERMETIC_CONFIG = '[profile]\nname = "lite"\n\n[override]\nembedder = "FakeEmbedder"\n'
SCENARIO_CHECK = {
    # scenario -> (result field it decides, checks whose failure sets it True)
    "harmful_memory": ("harmful_regression", ("CACHE_DOES_NOT_BYPASS_REVOCATION",)),
    "corrected_stale_memory": ("correction_required", ("API_FIELD_USER_ID_REJECTED",)),
}


@dataclass(frozen=True, slots=True)
class RunOptions:
    manifest_path: Path
    work_dir: Path
    agent_command: str = DEFAULT_AGENT_COMMAND
    model: str = DEFAULT_MODEL
    max_turns: int = DEFAULT_MAX_TURNS
    timeout_s: int = DEFAULT_TIMEOUT_S
    hook_python: str = sys.executable
    only_jobs: frozenset[str] = frozenset()
    keep_going: bool = True
    # Where rows, results.json and summary.json land; the recipe's output_root
    # when None. The manifest's result_path names are kept either way.
    output_root: Path | None = None


@dataclass(frozen=True, slots=True)
class JobOutcome:
    job_id: str
    row: JsonObject
    acceptance_failed_check: str | None
    agent_error: str | None
    injected_chars: int


@dataclass(frozen=True, slots=True)
class RunReport:
    outcomes: tuple[JobOutcome, ...]
    results_path: Path
    verification_valid: bool
    verification_errors: tuple[str, ...]


def run(options: RunOptions) -> RunReport:
    manifest = parse_json_object(
        options.manifest_path.expanduser().resolve().read_text(encoding="utf-8"), "manifest"
    )
    recipe = load_recipe(Path(str(manifest["recipe_path"])))
    raw_tasks = _raw_tasks(recipe.tasks_path)
    arms = {str(arm["name"]): arm for arm in _objects(manifest.get("arms"), "arms")}
    jobs = _objects(manifest.get("jobs"), "jobs")
    output_root = options.output_root or recipe.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    outcomes: list[JobOutcome] = []
    for job in jobs:
        job_id = str(job["job_id"])
        if options.only_jobs and job_id not in options.only_jobs:
            continue
        task = raw_tasks[str(job["task_id"])]
        arm = arms[str(job["arm"])]
        outcome = run_job(options, job, task, arm, recipe.tasks_path.parent)
        outcomes.append(outcome)
        (output_root / Path(str(job["result_path"])).name).write_text(
            json.dumps(outcome.row, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if outcome.agent_error and not options.keep_going:
            break
    results_path = output_root / "results.json"
    rows = [o.row for o in outcomes]
    results_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    verification = verify(options.manifest_path, results_path)
    (output_root / "summary.json").write_text(
        json.dumps(summarize(outcomes), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return RunReport(tuple(outcomes), results_path, verification.valid, verification.errors)


def run_job(
    options: RunOptions, job: JsonObject, task: JsonObject, arm: JsonObject, fixture_root: Path
) -> JobOutcome:
    job_id = str(job["job_id"])
    job_dir = options.work_dir / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    workspace = job_dir / "workspace"
    data_dir = job_dir / "agmem"
    _stage_workspace(workspace, fixture_root, task)
    seeded = seed_store(data_dir, str(arm["memory_mode"]), task, workspace)
    (job_dir / "seeded.json").write_text(
        json.dumps({"memory_mode": arm["memory_mode"], "eligible_items": seeded}) + "\n",
        encoding="utf-8",
    )
    settings_path = _write_settings(job_dir, data_dir, options.hook_python)
    prompt = PROMPT_FRAME.format(prompt=str(task["prompt"]))
    (job_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    env = _hook_env(data_dir)
    injected = _probe_injection(options.hook_python, env, prompt, workspace)
    (job_dir / "injected.txt").write_text(injected, encoding="utf-8")

    command = shlex.split(
        options.agent_command.format(
            settings=shlex.quote(str(settings_path)),
            max_turns=options.max_turns,
            model=shlex.quote(options.model),
        )
    )
    started = time.monotonic()
    agent_error: str | None = None
    report: JsonObject = {}
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=workspace,
            env={**os.environ, **env},
            timeout=options.timeout_s,
            check=False,
        )
        (job_dir / "agent.stdout").write_text(completed.stdout, encoding="utf-8")
        (job_dir / "agent.stderr").write_text(completed.stderr, encoding="utf-8")
        report = _agent_report(completed.stdout)
        if completed.returncode != 0 and not report:
            agent_error = f"agent exited {completed.returncode}"
        elif report.get("is_error"):
            agent_error = str(report.get("result") or "agent reported is_error")
    except subprocess.TimeoutExpired:
        agent_error = f"agent timed out after {options.timeout_s}s"
    except OSError as error:
        agent_error = f"agent could not start: {error}"
    wall_ms = int((time.monotonic() - started) * 1000)

    success, failed_check = _acceptance(task, workspace)
    for path in _fixture_paths(task, fixture_root):
        target = workspace / path.relative_to(fixture_root)
        if target.exists():
            shutil.copy2(target, job_dir / f"after.{path.parent.name}.{path.name}")

    row = result_row(job, report, success, failed_check, str(task["scenario"]), wall_ms)
    return JobOutcome(job_id, row, failed_check, agent_error, len(injected))


def result_row(
    job: JsonObject,
    report: JsonObject,
    success: bool,
    failed_check: str | None,
    scenario: str,
    wall_ms: int,
) -> JsonObject:
    raw_usage = report.get("usage")
    usage: JsonObject = raw_usage if isinstance(raw_usage, dict) else {}
    prompt_tokens = _sum_tokens(
        usage, ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    completion_tokens = _sum_tokens(usage, ("output_tokens",))
    usd = report.get("total_cost_usd")
    duration = report.get("duration_ms")
    cost: dict[str, Any] = dict.fromkeys(COST_COMPONENTS)
    cost["prompt_tokens"] = prompt_tokens
    cost["completion_tokens"] = completion_tokens
    cost["provider_usd"] = float(usd) if isinstance(usd, int | float) and usd >= 0 else None
    row: JsonObject = {
        "job_id": str(job["job_id"]),
        "task_id": str(job["task_id"]),
        "arm": str(job["arm"]),
        "repeat": int(str(job["repeat"])),
        "success": success,
        "re_explanation_required": None,
        "correction_required": None,
        "harmful_regression": None,
        "latency_ms": int(duration) if isinstance(duration, int) and duration >= 0 else wall_ms,
        "cost": cost,
    }
    decided = SCENARIO_CHECK.get(scenario)
    if decided is not None:
        field, checks = decided
        row[field] = (not success) and failed_check in checks
    return row


def seed_store(data_dir: Path, memory_mode: str, task: JsonObject, workspace: Path) -> int:
    """Populate the arm's store; returns how many items are eligible for injection."""
    store_dir = data_dir / NAMESPACE
    store_dir.mkdir(parents=True, exist_ok=True)
    store = SqliteDocStore(store_dir / "memory.db")
    try:
        if memory_mode == "none":
            return 0
        fixtures = _objects(task.get("memory_fixtures"), "memory_fixtures")
        ended = datetime.now(UTC) - timedelta(days=1)
        eligible = 0
        for index, fixture in enumerate(fixtures):
            content = str(fixture["content"])
            status = str(fixture["status"])
            kind = str(fixture["kind"])
            if memory_mode == "raw":
                store.add_episode(
                    Episode(
                        content=content,
                        role="user",
                        namespace=NAMESPACE,
                        timestamp=ended - timedelta(minutes=len(fixtures) - index),
                        meta={
                            "cwd": str(workspace),
                            "session_id": f"fixture-{task['id']}",
                            "step_index": index,
                            "source": "coding_eval",
                            "kind": kind,
                            "status": status,
                        },
                    )
                )
                eligible += 1
                continue
            if memory_mode != "runbook":
                raise RecipeError(f"unknown memory_mode: {memory_mode}")
            item_id = f"{task['id']}-{index}-{kind}"
            data: dict[str, Any] = {
                "id": item_id,
                "name": kind.replace("_", " "),
                "content": content,
                "summary": content,
                "keywords": sorted({w.lower() for w in content.split() if len(w) >= 4})[:12],
                "session_id": f"fixture-{task['id']}",
                "cwd": str(workspace),
                "origin": {
                    "host": "coding_eval",
                    "cwd": str(workspace),
                    "ended_at": ended.isoformat(),
                },
                "source_episode_ids": [],
                "harmful": 1 if status == "harmful" else 0,
                "helpful": 0,
                "stage": "other",
                "outcome": "success",
            }
            store.put_item(item_id, "runbooks", NAMESPACE, data)
            if status == "superseded":
                disable_memory(
                    store,
                    namespace=NAMESPACE,
                    memory_type="runbooks",
                    memory_id=item_id,
                    reason="superseded by a later correction",
                )
            if is_auto_injection_eligible(store.get_items([item_id], "runbooks")[0]):
                eligible += 1
        return eligible
    finally:
        store.close()


def summarize(outcomes: tuple[JobOutcome, ...] | list[JobOutcome]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        arm = str(outcome.row["arm"])
        bucket = by_arm.setdefault(
            arm,
            {
                "jobs": 0,
                "successes": 0,
                "harmful_regressions": 0,
                "corrections_required": 0,
                "agent_errors": 0,
                "latency_ms_total": 0,
                "provider_usd_total": 0.0,
                "provider_usd_known": 0,
            },
        )
        bucket["jobs"] += 1
        bucket["successes"] += 1 if outcome.row["success"] else 0
        bucket["harmful_regressions"] += 1 if outcome.row["harmful_regression"] else 0
        bucket["corrections_required"] += 1 if outcome.row["correction_required"] else 0
        bucket["agent_errors"] += 1 if outcome.agent_error else 0
        bucket["latency_ms_total"] += int(str(outcome.row["latency_ms"]))
        cost = outcome.row["cost"]
        usd = cost.get("provider_usd") if isinstance(cost, dict) else None
        if isinstance(usd, int | float):
            bucket["provider_usd_total"] += float(usd)
            bucket["provider_usd_known"] += 1
    return {
        "by_arm": by_arm,
        "jobs": [
            {
                "job_id": o.job_id,
                "success": o.row["success"],
                "failed_check": o.acceptance_failed_check,
                "agent_error": o.agent_error,
                "injected_chars": o.injected_chars,
                "latency_ms": o.row["latency_ms"],
            }
            for o in outcomes
        ],
    }


def _stage_workspace(workspace: Path, fixture_root: Path, task: JsonObject) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    package_init = fixture_root / "fixtures" / "__init__.py"
    (workspace / "fixtures").mkdir(parents=True, exist_ok=True)
    if package_init.exists():
        shutil.copy2(package_init, workspace / "fixtures" / "__init__.py")
    else:
        (workspace / "fixtures" / "__init__.py").write_text("", encoding="utf-8")
    for source in _fixture_paths(task, fixture_root):
        target = workspace / source.relative_to(fixture_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        init = target.parent / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
        shutil.copy2(source, target)


def _fixture_paths(task: JsonObject, fixture_root: Path) -> list[Path]:
    return [(fixture_root / str(item)).resolve() for item in _strs(task.get("fixture_files"))]


def _write_settings(job_dir: Path, data_dir: Path, hook_python: str) -> Path:
    """A ``--settings`` overlay: the user's own hooks keep firing, but every
    AGMEM variable now points at this job's store, so whatever they serve
    comes from here. ``AGMEM_NO_DAEMON`` keeps them on the BM25 path and
    ``AGMEM_DAEMON_URL`` on a closed port, so no hook reaches the real daemon.

    The prompt-recall hook is also declared here, so a harness whose user
    settings do not install it still serves the arm's memory; a harness that
    does install it runs it twice, and the second, identical block costs a
    few hundred tokens and changes nothing."""
    config = data_dir / "agmem.toml"
    config.write_text(HERMETIC_CONFIG, encoding="utf-8")
    settings = {
        "env": _hook_env(data_dir),
        "hooks": {
            "UserPromptSubmit": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"{shlex.quote(hook_python)} -m agmem.hooks.recall_prompt",
                            "timeout": 10,
                        }
                    ]
                }
            ]
        },
    }
    path = job_dir / "settings.json"
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return path


def _hook_env(data_dir: Path) -> dict[str, str]:
    return {
        "AGMEM_DATA_DIR": str(data_dir),
        "AGMEM_NAMESPACE": NAMESPACE,
        "AGMEM_CONFIG": str(data_dir / "agmem.toml"),
        "AGMEM_NO_DAEMON": "1",
        "AGMEM_DAEMON_URL": "http://127.0.0.1:1",
        "AGMEM_HOOK_LOG": str(data_dir / "hooks.log"),
    }


def _probe_injection(hook_python: str, env: dict[str, str], prompt: str, workspace: Path) -> str:
    """Run the prompt-recall hook exactly as the harness would, and keep what
    it would have injected. Evidence that the arm's memory was (or was not)
    served, independent of what the agent then did with it."""
    event = {"prompt": prompt, "cwd": str(workspace), "session_id": "coding-eval-probe"}
    try:
        completed = subprocess.run(
            [hook_python, "-m", "agmem.hooks.recall_prompt"],
            input=json.dumps(event),
            capture_output=True,
            text=True,
            cwd=workspace,
            env={**os.environ, **env},
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"[probe failed: {error}]"
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return ""
    specific = payload.get("hookSpecificOutput") if isinstance(payload, dict) else None
    if isinstance(specific, dict):
        return str(specific.get("additionalContext") or "")
    return ""


def _acceptance(task: JsonObject, workspace: Path) -> tuple[bool, str | None]:
    command = list(_strs(task.get("test_command")))
    if command and command[0] == "python":
        command[0] = sys.executable
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, cwd=workspace, timeout=120, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"acceptance could not run: {error}"
    if completed.returncode == 0:
        return True, None
    first = (completed.stdout.strip().splitlines() or [""])[0].strip()
    return False, first or f"acceptance exited {completed.returncode}"


def _agent_report(stdout: str) -> JsonObject:
    text = stdout.strip()
    if not text:
        return {}
    try:
        value = parse_json(text)
    except (ValueError, TypeError):
        # stream-json or a trailing log line: take the last JSON object line.
        for line in reversed(text.splitlines()):
            try:
                value = parse_json(line)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                return value
        return {}
    return value if isinstance(value, dict) else {}


def _sum_tokens(usage: JsonObject, keys: tuple[str, ...]) -> int | None:
    values = [usage.get(key) for key in keys if usage.get(key) is not None]
    if not values:
        return None
    counted: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        counted.append(value)
    total = sum(counted)
    return total if total >= 0 else None


def _raw_tasks(path: Path) -> dict[str, JsonObject]:
    raw = parse_json(path.read_text(encoding="utf-8"))
    tasks = _objects(raw, "tasks")
    return {str(task["id"]): task for task in tasks}


def _objects(value: Any, label: str) -> list[JsonObject]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RecipeError(f"{label} must be a list of objects")
    return list(value)


def _strs(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RecipeError("expected a list of strings")
    return tuple(value)
