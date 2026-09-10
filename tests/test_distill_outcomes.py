from __future__ import annotations

import pytest

from agmem import AgenticMemory
from agmem.config import AgmemConfig
from agmem.core.ops import MemoryOp, OpType
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.base import Organizer
from agmem.sessions import SessionTrajectory, Step


class OutcomeOrganizer(Organizer):
    name = "outcome-fixture"

    def __init__(self, status: str) -> None:
        self.status = status
        self.calls = 0

    def on_task_end(self, trajectory, outcome, task, ctx) -> list[MemoryOp]:
        self.calls += 1
        return [
            MemoryOp(
                op=OpType.NOOP,
                target_type="runbooks",
                target_id="outcome-session",
                payload={"distill_status": self.status, "reason": "fixture outcome"},
            )
        ]


@pytest.mark.parametrize("status, calls", [("failed", 2), ("skipped", 1), ("partial", 1)])
def test_explicit_distill_outcome_controls_retry(status: str, calls: int) -> None:
    org = OutcomeOrganizer(status)
    trajectory = SessionTrajectory(
        id="outcome-session",
        host="codex",
        source_path="fixture.jsonl",
        steps=[Step(kind="user", text="Remember the deployment constraint")],
    )
    with AgenticMemory(
        namespace="outcomes",
        organizers=[org],
        embedder=FakeEmbedder(dim=64),
        config=AgmemConfig(sync_write=True),
    ) as memory:
        memory.add_session(trajectory)
        memory.add_session(trajectory)
        assert org.calls == calls
        states = memory.doc_store.list_items("state", namespace="outcomes")
        assert len(states) == 1
        assert states[0]["status"] == status
        assert memory.session_distill_status(trajectory) == status


def test_failed_revision_can_retry_after_reopening_persistent_store(tmp_path) -> None:
    trajectory = SessionTrajectory(
        id="restart-session",
        host="codex",
        source_path="fixture.jsonl",
        steps=[Step(kind="user", text="Remember the retry boundary")],
    )
    config = AgmemConfig(data_dir=tmp_path, sync_write=True)
    with AgenticMemory(
        namespace="restart",
        organizers=[OutcomeOrganizer("failed")],
        embedder=FakeEmbedder(dim=64),
        config=config,
    ) as memory:
        memory.add_session(trajectory)
        assert memory.session_distill_status(trajectory) == "failed"
    retry = OutcomeOrganizer("skipped")
    with AgenticMemory(
        namespace="restart",
        organizers=[retry],
        embedder=FakeEmbedder(dim=64),
        config=config,
    ) as memory:
        assert memory.session_distill_status(trajectory) == "failed"
        memory.add_session(trajectory)
        assert retry.calls == 1
        assert memory.session_distill_status(trajectory) == "skipped"


class PartialReplacementOrganizer(Organizer):
    name = "partial-replacement"
    produces = ("runbooks",)

    def __init__(self) -> None:
        self.calls = 0

    def on_task_end(self, trajectory, outcome, task, ctx) -> list[MemoryOp]:
        self.calls += 1
        return [
            MemoryOp(
                op=OpType.ADD,
                target_type="runbooks",
                target_id=f"runbook-{self.calls}",
                payload={"content": "retained result", "session_id": "partial-session"},
            ),
            MemoryOp(
                op=OpType.NOOP,
                target_type="runbooks",
                target_id="partial-session",
                payload={"distill_status": "completed" if self.calls == 1 else "partial"},
            ),
        ]


def test_partial_replacement_preserves_previous_runbooks() -> None:
    trajectory = SessionTrajectory(
        id="partial-session",
        host="codex",
        source_path="fixture.jsonl",
        steps=[Step(kind="user", text="Preserve the earlier complete result")],
    )
    with AgenticMemory(
        namespace="partial",
        organizers=[PartialReplacementOrganizer()],
        embedder=FakeEmbedder(dim=64),
        config=AgmemConfig(sync_write=True),
    ) as memory:
        memory.add_session(trajectory)
        memory.add_session(trajectory, force=True)
        rows = memory.doc_store.list_items("runbooks", namespace="partial")
        assert {row["id"] for row in rows if not row.get("invalid_at")} == {
            "runbook-1",
            "runbook-2",
        }
        assert memory.session_distill_status(trajectory) == "partial"
