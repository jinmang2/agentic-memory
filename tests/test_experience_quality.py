from __future__ import annotations

from typing import Any

from agmem import AgenticMemory
from agmem.core.ops import OpType
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.experience import ExperienceOrganizer
from agmem.organizers.experience.organizer import MEMORY_TYPE
from agmem.sessions import SessionTrajectory, Step


class StubLLM:
    def __init__(self, responses: dict[str, list[dict[str, Any] | None]]) -> None:
        self.responses = {role: list(items) for role, items in responses.items()}
        self.calls: list[tuple[str, str]] = []
        self.systems: list[str] = []
        self.drops: dict[str, int] = {}

    def call(
        self,
        role: str,
        prompt: str,
        _schema: dict[str, Any],
        required_keys: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        self.calls.append((role, prompt))
        self.systems.append(str(kwargs.get("system", "")))
        items = self.responses.get(role)
        if not items:
            self.drops[role] = self.drops.get(role, 0) + 1
            return None
        return items.pop(0)


def _traj() -> SessionTrajectory:
    traj = SessionTrajectory(id="quality-1", host="codex", source_path="x", cwd="/repo")
    traj.steps = [
        Step(kind="user", text="Fix the importer"),
        Step(kind="assistant", text="I will inspect it."),
        Step(kind="tool_call", text="uv run pytest tests/test_importer.py -q", tool_name="Bash"),
        Step(kind="tool_result", text="1 passed", tool_name="Bash"),
        Step(kind="user", text="Record that this was verified by pytest."),
    ]
    return traj


def _run(reply: dict[str, Any]) -> tuple[AgenticMemory, list[dict[str, Any]]]:
    llm = StubLLM({"distill": [reply]})
    mem = AgenticMemory(
        namespace="quality", organizers=[ExperienceOrganizer()], embedder=FakeEmbedder(dim=64)
    )
    mem._ctx.llm = llm
    traj = _traj()
    mem.add_session(traj)
    mem.flush()
    return mem, mem.doc_store.list_items(MEMORY_TYPE, namespace="quality")


def _task(steps: list[int] | None) -> dict[str, Any]:
    task: dict[str, Any] = {
        "name": "Fix importer",
        "outcome": "success",
        "reusable_knowledge": ["`uv run pytest tests/test_importer.py -q` passed"],
        "procedure": ["Run the importer test before changing the loader."],
        "keywords": ["importer", "pytest"],
    }
    if steps is not None:
        task["steps"] = steps
    return task


def test_valid_citation_records_visible_source_coverage_separately_from_cited_coverage() -> None:
    mem, items = _run({"summary": "Importer test passed.", "tasks": [_task([2, 3])]})

    assert len(items) == 1
    quality = items[0]["quality"]
    assert quality["citation"]["status"] == "valid"
    assert quality["citation"]["step_range"] == [2, 3]
    assert quality["source_coverage"]["model_visible_steps"] == 5
    assert quality["source_coverage"]["model_visible_ratio"] == 1.0
    assert quality["source_coverage"]["cited_steps"] == 2
    assert quality["source_coverage"]["cited_ratio"] == 0.4
    assert quality["fact_basis"]["outcome_basis"] == "model_label_unverified"
    assert quality["fact_basis"]["includes_tool_result"] is True
    assert quality["fact_basis"]["verified_result_claim"] is False
    noop = [op for op in mem.log.tail(10) if op.op is OpType.NOOP and op.actor == "experience"]
    assert noop[-1].payload["distill_status"] == "completed"
    mem.close()


def test_missing_and_invalid_citations_keep_compatible_fallback_with_explicit_status() -> None:
    missing_mem, missing_items = _run({"summary": "", "tasks": [_task(None)]})
    invalid_mem, invalid_items = _run({"summary": "", "tasks": [_task([1, 99])]})

    missing = missing_items[0]
    invalid = invalid_items[0]
    assert missing["step_range"] is None and missing["cited_steps"] is None
    assert missing["quality"]["citation"]["status"] == "missing"
    assert missing["quality"]["source_coverage"]["scope"] == "whole_session_fallback"
    assert len(missing["source_episode_ids"]) == 5
    assert invalid["step_range"] is None and invalid["cited_steps"] is None
    assert invalid["quality"]["citation"]["status"] == "invalid"
    assert invalid["quality"]["citation"]["raw_steps"] == [1, 99]
    assert invalid["quality"]["source_coverage"]["scope"] == "whole_session_fallback"
    assert len(invalid["source_episode_ids"]) == 5
    missing_mem.close()
    invalid_mem.close()


def test_citation_into_omitted_middle_is_invalid_not_missing() -> None:
    llm = StubLLM(
        {
            "distill": [
                {
                    "summary": "",
                    "tasks": [
                        {
                            "name": "Middle omitted citation",
                            "outcome": "uncertain",
                            "reusable_knowledge": ["middle step was not actually shown"],
                            "keywords": ["middle"],
                            "steps": [1, 3],
                        }
                    ],
                }
            ]
        }
    )
    mem = AgenticMemory(
        namespace="quality",
        organizers=[ExperienceOrganizer(max_chars=120)],
        embedder=FakeEmbedder(dim=64),
    )
    mem._ctx.llm = llm
    traj = _traj()
    mem.add_task_result(
        trajectory=traj.as_task_trajectory(), outcome="unknown", task=traj.task_text
    )
    mem.flush()

    (item,) = mem.doc_store.list_items(MEMORY_TYPE, namespace="quality")
    assert item["quality"]["citation"]["status"] == "invalid"
    assert item["quality"]["citation"]["reason"] == "steps_outside_model_visible_transcript"
    assert item["quality"]["source_coverage"]["model_visible_steps"] < 5
    mem.close()


def test_explicit_skip_and_dropped_reply_are_observable_noops() -> None:
    no_llm = AgenticMemory(
        namespace="quality", organizers=[ExperienceOrganizer()], embedder=FakeEmbedder(dim=64)
    )
    traj = _traj()
    no_llm.add_task_result(
        trajectory=traj.as_task_trajectory(), outcome="unknown", task=traj.task_text
    )
    no_llm.flush()
    (skip,) = [op for op in no_llm.log.tail(10) if op.actor == "experience"]
    assert skip.op is OpType.NOOP
    assert skip.payload["distill_status"] == "failed"
    assert skip.payload["reason"] == "no_llm_configured"

    dropped = StubLLM({"distill": []})
    failed = AgenticMemory(
        namespace="quality", organizers=[ExperienceOrganizer()], embedder=FakeEmbedder(dim=64)
    )
    failed._ctx.llm = dropped
    failed.add_task_result(
        trajectory=traj.as_task_trajectory(), outcome="unknown", task=traj.task_text
    )
    failed.flush()
    (softdrop,) = [op for op in failed.log.tail(10) if op.actor == "experience"]
    assert softdrop.op is OpType.NOOP
    assert softdrop.payload["distill_status"] == "failed"
    assert softdrop.payload["retryable"] is True
    assert softdrop.payload["review_required"] is True
    no_llm.close()
    failed.close()


def test_exact_duplicate_blocks_are_not_stored_twice_and_mark_partial_review() -> None:
    task = _task([2, 3])
    mem, items = _run({"summary": "duplicate", "tasks": [task, dict(task)]})

    assert len(items) == 1
    noop = [op for op in mem.log.tail(10) if op.op is OpType.NOOP and op.actor == "experience"]
    assert noop[-1].payload["distill_status"] == "partial"
    assert noop[-1].payload["skipped_outputs"] == 1
    assert noop[-1].payload["review_required"] is True
    mem.close()


def test_a_single_visible_step_can_still_be_only_partially_read() -> None:
    llm = StubLLM(
        {
            "distill": [
                {
                    "summary": "Long step was clipped.",
                    "tasks": [
                        {
                            "name": "Long single step",
                            "outcome": "uncertain",
                            "reusable_knowledge": [
                                "The step label was visible but body was clipped."
                            ],
                            "keywords": ["long-step"],
                            "steps": [0],
                        }
                    ],
                }
            ]
        }
    )
    mem = AgenticMemory(
        namespace="quality",
        organizers=[ExperienceOrganizer(max_chars=300)],
        embedder=FakeEmbedder(dim=64),
    )
    mem._ctx.llm = llm
    traj = SessionTrajectory(id="quality-long", host="codex", source_path="x", cwd="/repo")
    traj.steps = [Step(kind="user", text="x" * 5_000)]
    mem.add_session(traj)
    mem.flush()

    (item,) = mem.doc_store.list_items(MEMORY_TYPE, namespace="quality")
    coverage = item["quality"]["source_coverage"]
    assert coverage["model_visible_steps"] == 1
    assert coverage["model_visible_ratio"] == 1.0
    assert coverage["visible_step_semantics"] == "step_presence_not_full_text"
    assert coverage["has_clipped_steps"] is True
    assert coverage["rendered_chars"] < coverage["source_chars"]
    assert coverage["rendered_char_ratio"] < 1.0
    mem.close()


def test_dropped_segment_with_empty_surviving_reply_is_failed_not_skipped() -> None:
    llm = StubLLM({"distill": [{"summary": "", "tasks": []}, None]})
    mem = AgenticMemory(
        namespace="quality",
        organizers=[ExperienceOrganizer(max_chars=50, max_calls=2)],
        embedder=FakeEmbedder(dim=64),
    )
    mem._ctx.llm = llm
    traj = _traj()
    mem.add_session(traj)
    mem.flush()

    assert len(llm.calls) == 2
    assert mem.doc_store.list_items(MEMORY_TYPE, namespace="quality") == []
    (noop,) = [op for op in mem.log.tail(10) if op.op is OpType.NOOP and op.actor == "experience"]
    assert noop.payload["distill_status"] == "failed"
    assert noop.payload["reason"] == "distill_output_rejected"
    assert noop.payload["retryable"] is True
    assert noop.payload["review_required"] is True
    assert noop.payload["skipped_outputs"] == 1
    mem.close()


def test_malformed_tasks_only_are_failed_not_no_signal() -> None:
    llm = StubLLM({"distill": [{"summary": "", "tasks": ["not-a-task"]}]})
    mem = AgenticMemory(
        namespace="quality", organizers=[ExperienceOrganizer()], embedder=FakeEmbedder(dim=64)
    )
    mem._ctx.llm = llm
    traj = _traj()
    mem.add_session(traj)
    mem.flush()

    assert mem.doc_store.list_items(MEMORY_TYPE, namespace="quality") == []
    (noop,) = [op for op in mem.log.tail(10) if op.op is OpType.NOOP and op.actor == "experience"]
    assert noop.payload["distill_status"] == "failed"
    assert noop.payload["reason"] == "distill_output_rejected"
    assert noop.payload["skipped_outputs"] == 1
    mem.close()
