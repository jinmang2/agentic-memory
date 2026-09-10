from __future__ import annotations

import json
from pathlib import Path

from agmem import AgenticMemory
from agmem.config import AgmemConfig
from agmem.core.ops import MemoryOp, OpType
from agmem.embed.fake import FakeEmbedder
from agmem.mcp.server import _Registry
from agmem.organizers.base import Organizer
from agmem.sessions import load


class OutcomeOrganizer(Organizer):
    name = "durable-outcome"

    def __init__(self, status: str) -> None:
        self.status = status
        self.calls = 0

    def on_task_end(self, trajectory, outcome, task, ctx) -> list[MemoryOp]:
        self.calls += 1
        return [
            MemoryOp(
                op=OpType.NOOP,
                target_type="runbooks",
                target_id="durable-outcome",
                payload={"distill_status": self.status},
            )
        ]


def _transcript(path: Path, session_id: str, text: str) -> Path:
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "type": "user",
                    "uuid": f"{session_id}-user",
                    "isSidechain": False,
                    "timestamp": "2026-09-09T00:00:00.000Z",
                    "cwd": "/work/durable",
                    "sessionId": session_id,
                    "message": {"role": "user", "content": text},
                },
                {
                    "type": "assistant",
                    "uuid": f"{session_id}-assistant",
                    "isSidechain": False,
                    "timestamp": "2026-09-09T00:00:01.000Z",
                    "cwd": "/work/durable",
                    "sessionId": session_id,
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "recorded"}],
                    },
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _registry(tmp_path: Path) -> _Registry:
    reg = _Registry()
    reg.start(
        "default",
        [],
        AgmemConfig(
            profile="lite",
            data_dir=tmp_path / "data",
            sync_write=True,
            overrides={"embedder": "FakeEmbedder"},
        ),
    )
    return reg


def test_preserve_hook_job_is_durable_before_it_is_processed(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    transcript = _transcript(tmp_path / "preserve.jsonl", "preserve-job", "keep this raw")
    try:
        accepted = reg.accept_preserve_job(
            {"transcript_path": str(transcript), "namespace": "accepted-ns"}
        )

        assert accepted["queued"] is True
        assert reg.queue_status("accepted-ns")["preserve"]["queued"] == 1

        assert reg.drain_preserve_queue("accepted-ns") == 1
        mem = reg.get("accepted-ns")
        assert any("keep this raw" in ep.content for ep in mem.doc_store.list_episodes())
        assert reg.queue_status("accepted-ns")["preserve"]["queued"] == 0
    finally:
        reg.close_all()


def test_backfill_all_discovers_queues_for_namespaces_that_were_not_opened(
    tmp_path: Path,
) -> None:
    reg = _registry(tmp_path)
    transcript = _transcript(tmp_path / "unopened.jsonl", "unopened-job", "discover queue")
    queue = tmp_path / "data" / "unopened-ns" / "preserve-queue.jsonl"
    queue.parent.mkdir(parents=True)
    queue.write_text(
        json.dumps({"transcript_path": str(transcript), "namespace": "unopened-ns"}) + "\n",
        encoding="utf-8",
    )
    try:
        assert "unopened-ns" not in reg.open_namespaces()

        assert reg.backfill_all() >= 1

        assert "unopened-ns" in reg.open_namespaces()
        mem = reg.get("unopened-ns")
        assert any("discover queue" in ep.content for ep in mem.doc_store.list_episodes())
    finally:
        reg.close_all()


def test_accepted_job_recovers_after_registry_restart(tmp_path: Path) -> None:
    first = _registry(tmp_path)
    transcript = _transcript(tmp_path / "restart.jsonl", "restart-job", "recover after restart")
    try:
        first.accept_preserve_job({"transcript_path": str(transcript), "namespace": "restart-ns"})
        assert first.queue_status("restart-ns")["preserve"]["queued"] == 1
    finally:
        first.close_all()

    second = _registry(tmp_path)
    try:
        assert "restart-ns" not in second.open_namespaces()

        assert second.backfill_all() >= 1

        mem = second.get("restart-ns")
        assert any("recover after restart" in ep.content for ep in mem.doc_store.list_episodes())
        assert second.queue_status("restart-ns")["preserve"]["queued"] == 0
    finally:
        second.close_all()


def test_distill_hook_job_survives_until_the_queue_consumer_runs(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    transcript = _transcript(tmp_path / "distill.jsonl", "distill-job", "distill me later")
    try:
        accepted = reg.accept_distill_job(
            {"transcript_path": str(transcript), "namespace": "durable-distill"}
        )

        assert accepted == {
            "queued": True,
            "transcript_path": str(transcript),
            "namespace": "durable-distill",
        }
        assert reg.queue_status("durable-distill")["distill"]["queued"] == 1

        assert reg.drain_distill_queue("durable-distill") == 1
        assert reg.queue_status("durable-distill")["distill"]["queued"] == 0
        mem = reg.get("durable-distill")
        assert any("distill me later" in ep.content for ep in mem.doc_store.list_episodes())
    finally:
        reg.close_all()


def test_failed_distill_status_keeps_durable_job_retryable(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    transcript = _transcript(tmp_path / "failed.jsonl", "failed-job", "retry failed distill")
    failing = OutcomeOrganizer("failed")
    try:
        accepted = reg.accept_distill_job(
            {"transcript_path": str(transcript), "namespace": "failed-distill"}
        )
        mem = reg.get("failed-distill")
        mem.organizers = [failing]

        assert reg.drain_distill_queue(accepted["namespace"]) == 0

        assert failing.calls == 1
        assert reg.queue_status("failed-distill")["distill"] == {
            "queued": 0,
            "processing": 1,
            "bad": 0,
        }
        traj = load(transcript)
        with AgenticMemory(
            namespace="failed-distill",
            organizers=[],
            embedder=FakeEmbedder(),
            config=AgmemConfig(data_dir=tmp_path / "data", sync_write=True),
        ) as reopened:
            assert reopened.session_distill_status(traj) == "failed"

        retry = OutcomeOrganizer("skipped")
        mem.organizers = [retry]
        assert reg.drain_distill_queue("failed-distill") == 1
        assert retry.calls == 1
        assert reg.queue_status("failed-distill")["distill"]["processing"] == 0
    finally:
        reg.close_all()
