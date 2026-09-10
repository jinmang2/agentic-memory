from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import pytest
from starlette.requests import Request

from agmem.config import AgmemConfig
from agmem.control import (
    apply_user_control_op,
    correct_memory,
    disable_memory,
    inspect_memory,
    is_auto_injection_eligible,
    restore_memory,
)
from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode
from agmem.hooks import recall, recall_prompt
from agmem.mcp import server
from agmem.memory import AgenticMemory
from agmem.stores.sqlite_doc import SqliteDocStore


class FailingPutStore(SqliteDocStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_put = False

    def put_item(
        self, item_id: str, memory_type: str, namespace: str, data: dict[str, Any]
    ) -> None:
        if self.fail_put:
            raise OSError("forced put failure")
        super().put_item(item_id, memory_type, namespace, data)


@pytest.fixture
def store() -> Iterator[SqliteDocStore]:
    doc_store = SqliteDocStore()
    try:
        yield doc_store
    finally:
        doc_store.close()


def _runbook(item_id: str, content: str, *, harmful: int = 0) -> dict[str, Any]:
    return {
        "id": item_id,
        "name": item_id,
        "content": content,
        "origin": {"cwd": "/w/project-a", "ended_at": "2026-09-09T00:00:00+00:00"},
        "source_episode_ids": [],
        "harmful": harmful,
        "helpful": 0,
    }


def test_disable_restore_and_harmful_feedback_gate_auto_injection(
    store: SqliteDocStore,
) -> None:
    store.put_item("bad", "runbooks", "controls", _runbook("bad", "needlealpha bad advice"))

    disabled = disable_memory(
        store,
        namespace="controls",
        memory_type="runbooks",
        memory_id="bad",
        reason="wrong for this repo",
    )

    assert disabled.changed is True
    assert is_auto_injection_eligible(disabled.item) is False
    assert store.tail(1)[0].actor == "user-control"
    assert store.tail(1)[0].payload["control"]["automatic_injection"] == "disabled"
    assert "item" not in store.tail(1)[0].payload
    assert recall_prompt.fallback_items(store, "controls", "needlealpha", 1, "/w/project-a") == []
    assert recall.render_runbooks(recall.recent_runbooks(store, "controls", "/w/project-a")) == ""

    restored = restore_memory(
        store,
        namespace="controls",
        memory_type="runbooks",
        memory_id="bad",
        reason="corrected by user",
    )

    assert restored.changed is True
    assert is_auto_injection_eligible(restored.item) is True
    assert [
        item["id"]
        for item in recall_prompt.fallback_items(
            store, "controls", "needlealpha", 1, "/w/project-a"
        )
    ] == ["bad"]

    store.put_item(
        "harmful",
        "runbooks",
        "controls",
        _runbook("harmful", "needlealpha trap", harmful=1),
    )

    assert (
        recall_prompt.fallback_items(store, "controls", "needlealpha", 2, "/w/project-a")[0]["id"]
        == "bad"
    )
    assert is_auto_injection_eligible(store.get_items(["harmful"], "runbooks")[0]) is False

    restored.item["harmful"] = 1
    store.put_item("bad", "runbooks", "controls", restored.item)

    assert is_auto_injection_eligible(store.get_items(["bad"], "runbooks")[0]) is False


def test_correction_retains_original_and_audit(store: SqliteDocStore) -> None:
    store.put_item(
        "fixme",
        "runbooks",
        "controls",
        _runbook("fixme", "Use stale command for needlebeta"),
    )

    result = correct_memory(
        store,
        namespace="controls",
        memory_type="runbooks",
        memory_id="fixme",
        content="Use fresh command for needlebeta",
        reason="user correction",
    )
    inspected = inspect_memory(
        store, namespace="controls", memory_type="runbooks", memory_id="fixme"
    )

    assert result.changed is True
    assert inspected.found is True
    assert inspected.item["content"] == "Use fresh command for needlebeta"
    assert inspected.item["original_content"] == "Use stale command for needlebeta"
    assert inspected.item["correction"]["reason"] == "user correction"
    assert inspected.item["control"]["audit"][-1]["action"] == "correct"
    assert inspected.item["control"]["index_refresh"] == "pending"
    rendered = recall.render_runbooks(recall.recent_runbooks(store, "controls", "/w/project-a"))
    assert "Use fresh command" in rendered
    assert "Use stale command" not in rendered


def test_control_update_op_replays_as_flat_delta(store: SqliteDocStore) -> None:
    store.put_item(
        "fixme",
        "runbooks",
        "controls",
        _runbook("fixme", "Use stale command for needledelta"),
    )
    correct_memory(
        store,
        namespace="controls",
        memory_type="runbooks",
        memory_id="fixme",
        content="Use fresh command for needledelta",
        reason="user correction",
    )
    op = store.tail(1)[0]
    replay = AgenticMemory(
        namespace="controls",
        organizers=[],
        config=AgmemConfig(
            data_dir=None,
            overrides={
                "doc_store": "SqliteDocStore",
                "vector_store": "SqliteVecStore",
                "graph_store": "SqliteGraphStore",
                "embedder": "FakeEmbedder",
            },
        ),
    )
    try:
        replay.doc_store.put_item(
            "fixme",
            "runbooks",
            "controls",
            _runbook("fixme", "Use stale command for needledelta"),
        )

        replay._apply_one(op)
        replayed = replay.doc_store.get_items(["fixme"], "runbooks")[0]

        assert replayed["content"] == "Use fresh command for needledelta"
        assert replayed["original_content"] == "Use stale command for needledelta"
        assert "item" not in replayed
    finally:
        replay.close()


def test_control_append_happens_before_apply_failure() -> None:
    store = FailingPutStore()
    try:
        store.put_item(
            "fixme",
            "runbooks",
            "controls",
            _runbook("fixme", "Use stale command for needleepsilon"),
        )
        store.fail_put = True

        with pytest.raises(OSError):
            correct_memory(
                store,
                namespace="controls",
                memory_type="runbooks",
                memory_id="fixme",
                content="Use fresh command for needleepsilon",
                reason="user correction",
            )

        assert store.tail(1)[0].actor == "user-control"
        assert store.tail(1)[0].payload["content"] == "Use fresh command for needleepsilon"
        assert store.get_items(["fixme"], "runbooks")[0]["content"] == (
            "Use stale command for needleepsilon"
        )
    finally:
        store.close()


def test_episode_auto_eligibility_uses_meta_control() -> None:
    episode = Episode(
        id="ep1",
        content="needlealpha old user text",
        namespace="controls",
        meta={"control": {"automatic_injection": "disabled"}},
    )

    assert is_auto_injection_eligible(episode) is False


def test_episode_disable_and_correction_apply_to_prompt_fallback(store: SqliteDocStore) -> None:
    store.add_episode(
        Episode(
            id="ep1",
            content="needlegamma stale raw evidence",
            role="user",
            namespace="controls",
            meta={"cwd": "/w/project-a"},
        )
    )

    disable_memory(
        store,
        namespace="controls",
        memory_type="episodic",
        memory_id="ep1",
        reason="bad raw memory",
    )

    assert recall_prompt.fallback_items(store, "controls", "needlegamma", 1, "/w/project-a") == []

    corrected = correct_memory(
        store,
        namespace="controls",
        memory_type="episodic",
        memory_id="ep1",
        content="needlegamma corrected raw evidence",
        reason="user correction",
    )
    restore_memory(
        store,
        namespace="controls",
        memory_type="episodic",
        memory_id="ep1",
        reason="corrected",
    )
    items = recall_prompt.fallback_items(store, "controls", "needlegamma", 1, "/w/project-a")
    inspected = inspect_memory(store, namespace="controls", memory_type="episodic", memory_id="ep1")

    assert corrected.item["original_content"] == "needlegamma stale raw evidence"
    assert inspected.item["meta"]["original_content"] == "needlegamma stale raw evidence"
    assert inspected.item["meta"]["control"]["index_refresh"] == "pending"
    assert items[0]["text"] == "needlegamma corrected raw evidence"


def test_episode_user_control_op_replays_through_helper(store: SqliteDocStore) -> None:
    store.add_episode(
        Episode(
            id="ep2",
            content="needlezeta stale raw evidence",
            role="user",
            namespace="controls",
            meta={"cwd": "/w/project-a"},
        )
    )
    correct_memory(
        store,
        namespace="controls",
        memory_type="episodic",
        memory_id="ep2",
        content="needlezeta corrected raw evidence",
        reason="user correction",
    )
    op = store.tail(1)[0]
    replay = SqliteDocStore()
    try:
        replay.add_episode(
            Episode(
                id="ep2",
                content="needlezeta stale raw evidence",
                role="user",
                namespace="controls",
                meta={"cwd": "/w/project-a"},
            )
        )

        apply_user_control_op(replay, op)
        [episode] = replay.get_episodes(["ep2"])

        assert episode.content == "needlezeta corrected raw evidence"
        assert episode.meta["original_content"] == "needlezeta stale raw evidence"
    finally:
        replay.close()


def test_episode_user_control_op_replays_through_facade_branch() -> None:
    mem = AgenticMemory(
        namespace="controls",
        organizers=[],
        config=AgmemConfig(
            data_dir=None,
            overrides={
                "doc_store": "SqliteDocStore",
                "vector_store": "SqliteVecStore",
                "graph_store": "SqliteGraphStore",
                "embedder": "FakeEmbedder",
            },
        ),
    )
    try:
        mem.doc_store.add_episode(
            Episode(
                id="ep3",
                content="needleeta stale raw evidence",
                role="user",
                namespace="controls",
                meta={"cwd": "/w/project-a"},
            )
        )
        op = _episode_control_op()

        mem._apply_one(op)
        [episode] = mem.doc_store.get_episodes(["ep3"])

        assert episode.content == "needleeta corrected raw evidence"
        assert episode.meta["correction"]["reason"] == "facade replay"
    finally:
        mem.close()


def _episode_control_op() -> MemoryOp:
    return MemoryOp(
        op=OpType.UPDATE,
        target_type="episodic",
        target_id="ep3",
        actor="user-control",
        payload={
            "content": "needleeta corrected raw evidence",
            "original_content": "needleeta stale raw evidence",
            "correction": {
                "at": "2026-09-09T00:00:00+00:00",
                "reason": "facade replay",
                "content": "needleeta corrected raw evidence",
            },
            "control": {"index_refresh": "pending"},
        },
    )


def test_manage_cli_status_and_inspect_are_read_only(tmp_path) -> None:
    data_dir = tmp_path / "data"
    store = SqliteDocStore(data_dir / "controls" / "memory.db")
    try:
        store.put_item("cli-rb", "runbooks", "controls", _runbook("cli-rb", "CLI visible"))
        invalid = _runbook("old-rb", "CLI invalidated")
        invalid["invalid_at"] = "2026-09-09T00:00:00+00:00"
        store.put_item("old-rb", "runbooks", "controls", invalid)
    finally:
        store.close()

    status = subprocess.run(
        [
            sys.executable,
            "-m",
            "agmem.manage",
            "--data-dir",
            str(data_dir),
            "--namespace",
            "controls",
            "status",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    inspect = subprocess.run(
        [
            sys.executable,
            "-m",
            "agmem.manage",
            "--data-dir",
            str(data_dir),
            "--namespace",
            "controls",
            "inspect",
            "--type",
            "runbooks",
            "--id",
            "cli-rb",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert status.returncode == 0, status.stderr
    assert inspect.returncode == 0, inspect.stderr
    assert json.loads(status.stdout)["namespace"] == "controls"
    assert json.loads(status.stdout)["items"]["runbooks"]["live"] == 1
    assert json.loads(status.stdout)["items"]["runbooks"]["stored"] == 2
    assert json.loads(inspect.stdout)["item"]["content"] == "CLI visible"


def test_manage_status_reports_config_data_dir(tmp_path) -> None:
    data_dir = tmp_path / "from-config"
    cfg = tmp_path / "agmem.toml"
    cfg.write_text(
        f'[profile]\nname = "lite"\n\n[storage]\ndata_dir = "{data_dir}"\n',
        encoding="utf-8",
    )
    store = SqliteDocStore(data_dir / "controls" / "memory.db")
    try:
        store.put_item("cfg-rb", "runbooks", "controls", _runbook("cfg-rb", "config visible"))
    finally:
        store.close()

    status = subprocess.run(
        [
            sys.executable,
            "-m",
            "agmem.manage",
            "--config",
            str(cfg),
            "--namespace",
            "controls",
            "status",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    payload = json.loads(status.stdout)
    assert status.returncode == 0, status.stderr
    assert payload["store"] == str(data_dir / "controls" / "memory.db")
    assert payload["source"]["data_dir"] == str(data_dir)
    assert payload["items"]["runbooks"]["live"] == 1


def test_daemon_recall_suppresses_disabled_and_harmful_runbooks(tmp_path) -> None:
    server._registry.close_all()
    mem = server._registry.start(
        "controlsdaemon",
        ["experience"],
        AgmemConfig(
            data_dir=tmp_path / "data",
            overrides={
                "doc_store": "SqliteDocStore",
                "vector_store": "SqliteVecStore",
                "graph_store": "SqliteGraphStore",
                "embedder": "FakeEmbedder",
            },
        ),
    )
    try:
        for item_id, content, harmful in (
            ("good", "needlealpha correct runbook", 0),
            ("harmful", "needlealpha harmful runbook", 1),
            ("disabled", "needlealpha disabled runbook", 0),
        ):
            data = _runbook(item_id, content, harmful=harmful)
            mem.doc_store.put_item(item_id, "runbooks", mem.namespace, data)
            mem.vector_store.add(
                item_id,
                mem.embedder.embed([content])[0],
                memory_type="runbooks",
                namespace=mem.namespace,
            )
        disable_memory(
            mem.doc_store,
            namespace=mem.namespace,
            memory_type="runbooks",
            memory_id="disabled",
        )

        async def receive() -> dict[str, str | bytes | bool]:
            return {
                "type": "http.request",
                "body": json.dumps(
                    {"query": "needlealpha", "k": 3, "namespace": "controlsdaemon"}
                ).encode(),
                "more_body": False,
            }

        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/hooks/recall",
                "headers": [(b"content-type", b"application/json")],
            },
            receive,
        )

        response = pytest.importorskip("anyio").run(server.hooks_recall, request)
        body = json.loads(bytes(response.body).decode())

        assert response.status_code == 200
        assert [item["id"] for item in body["items"]] == ["good"]
    finally:
        server._registry.close_all()
