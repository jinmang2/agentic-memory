from __future__ import annotations

import threading

import pytest

from agmem import AgenticMemory
from agmem.config import AgmemConfig
from agmem.core.ops import MemoryOp, OpType
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.base import Organizer
from agmem.sessions import SessionTrajectory, Step


class CountingRunbookOrganizer(Organizer):
    name = "counting-runbook"
    produces = ("runbooks",)

    def __init__(self) -> None:
        self.calls = 0

    def on_task_end(self, trajectory, outcome, task, ctx) -> list[MemoryOp]:
        self.calls += 1
        return [
            MemoryOp(
                op=OpType.ADD,
                target_type="runbooks",
                target_id=f"rb-{self.calls}",
                payload={
                    "content": f"{task}: {outcome}",
                    "session_id": trajectory[0]["session_id"],
                    "source_host": trajectory[0]["host"],
                },
            )
        ]


class DistillFailedError(Exception):
    pass


class RaisingOrganizer(Organizer):
    name = "raising"
    produces = ("runbooks",)

    def on_task_end(self, trajectory, outcome, task, ctx) -> list[MemoryOp]:
        raise DistillFailedError("distill failed")


def _session(
    session_id: str = "s1", *, extra_step: bool = False, tool_result: str = "raw persisted"
) -> SessionTrajectory:
    steps = [
        Step(kind="user", text="Fix session distill idempotency"),
        Step(kind="assistant", text="I will inspect add_session."),
        Step(kind="tool_result", text=tool_result, tool_name="Bash"),
    ]
    if extra_step:
        steps.append(Step(kind="user", text="Also handle a grown trajectory."))
    return SessionTrajectory(
        id=session_id,
        host="codex",
        source_path="session.jsonl",
        cwd="/home/u/proj",
        steps=steps,
    )


def _mem(organizer: Organizer) -> AgenticMemory:
    return AgenticMemory(
        namespace="t",
        organizers=[organizer],
        embedder=FakeEmbedder(dim=64),
        config=AgmemConfig(sync_write=True),
    )


def _async_mem(organizer: Organizer) -> AgenticMemory:
    return AgenticMemory(
        namespace="t",
        organizers=[organizer],
        embedder=FakeEmbedder(dim=64),
        config=AgmemConfig(sync_write=False),
    )


class BlockingRunbookOrganizer(Organizer):
    name = "blocking-runbook"
    produces = ("runbooks",)

    def __init__(self) -> None:
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def on_task_end(self, trajectory, outcome, task, ctx) -> list[MemoryOp]:
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=5)
        return [
            MemoryOp(
                op=OpType.ADD,
                target_type="runbooks",
                target_id=f"blocked-rb-{self.calls}",
                payload={"content": task, "session_id": trajectory[0]["session_id"]},
            )
        ]


def test_distill_runs_after_raw_only_session_was_already_persisted() -> None:
    org = CountingRunbookOrganizer()
    mem = _mem(org)
    traj = _session()
    try:
        raw = mem.add_session(traj, distill=False)
        assert raw.dispatched is False

        distilled = mem.add_session(traj, distill=True)

        assert distilled.already_ingested is True
        assert distilled.dispatched is True
        assert distilled.episode_ids == raw.episode_ids
        assert org.calls == 1
        assert len(mem.doc_store.list_items("runbooks", namespace="t")) == 1
    finally:
        mem.close()


def test_async_duplicate_distill_is_skipped_while_revision_is_in_flight() -> None:
    org = BlockingRunbookOrganizer()
    mem = _async_mem(org)
    traj = _session()
    try:
        first = mem.add_session(traj)
        assert org.started.wait(timeout=5)

        second = mem.add_session(traj)
        forced = mem.add_session(traj, force=True)
        org.release.set()
        mem.flush()
        third = mem.add_session(traj)

        assert first.dispatched is True
        assert second.already_ingested is True
        assert second.dispatched is False
        assert forced.already_ingested is True
        assert forced.dispatched is False
        assert third.already_ingested is True
        assert third.dispatched is False
        assert org.calls == 1
    finally:
        org.release.set()
        mem.close()


def test_successful_distill_marks_revision_and_is_not_called_twice() -> None:
    org = CountingRunbookOrganizer()
    mem = _mem(org)
    traj = _session()
    try:
        first = mem.add_session(traj)
        second = mem.add_session(traj)

        assert first.dispatched is True
        assert second.already_ingested is True
        assert second.dispatched is False
        assert org.calls == 1
    finally:
        mem.close()


def test_legacy_live_runbook_counts_as_completed_distill_marker() -> None:
    org = CountingRunbookOrganizer()
    mem = _mem(org)
    traj = _session()
    try:
        mem.add_session(traj, distill=False)
        mem._apply_ops(
            [
                MemoryOp(
                    op=OpType.ADD,
                    target_type="runbooks",
                    target_id="legacy-rb",
                    payload={"content": "legacy", "session_id": traj.id},
                )
            ],
            actor="legacy",
        )

        result = mem.add_session(traj)

        assert result.already_ingested is True
        assert result.dispatched is False
        assert org.calls == 0
    finally:
        mem.close()


def test_grown_session_revision_distills_again_without_force() -> None:
    org = CountingRunbookOrganizer()
    mem = _mem(org)
    try:
        mem.add_session(_session())
        grown = mem.add_session(_session(extra_step=True))

        assert grown.already_ingested is False
        assert grown.dispatched is True
        assert org.calls == 2
    finally:
        mem.close()


def test_same_length_content_change_is_a_new_distill_revision() -> None:
    org = CountingRunbookOrganizer()
    mem = _mem(org)
    try:
        mem.add_session(_session())
        changed = mem.add_session(_session(tool_result="raw persisted with a corrected detail"))

        assert changed.already_ingested is True
        assert changed.dispatched is True
        assert org.calls == 2
    finally:
        mem.close()


def test_raised_distill_does_not_mark_revision_complete() -> None:
    mem = _mem(RaisingOrganizer())
    traj = _session()
    try:
        with pytest.raises(DistillFailedError, match="distill failed"):
            mem.add_session(traj)

        counter = CountingRunbookOrganizer()
        mem.organizers = [counter]

        result = mem.add_session(traj)

        assert result.already_ingested is True
        assert result.dispatched is True
        assert counter.calls == 1
    finally:
        mem.close()


def test_async_raised_distill_clears_in_flight_claim_for_retry() -> None:
    mem = _async_mem(RaisingOrganizer())
    traj = _session()
    try:
        first = mem.add_session(traj)
        mem.flush()

        counter = CountingRunbookOrganizer()
        mem.organizers = [counter]
        retry = mem.add_session(traj)
        mem.flush()

        assert first.dispatched is True
        assert retry.already_ingested is True
        assert retry.dispatched is True
        assert counter.calls == 1
    finally:
        mem.close()
