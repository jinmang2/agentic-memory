from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import anyio
from starlette.requests import Request

from agmem.config import AgmemConfig
from agmem.mcp import server


class RecallItem(TypedDict):
    id: str | None
    memory_type: str
    score: float
    timestamp: str | None
    text: str


class RecallBody(TypedDict):
    namespace: str
    items: list[RecallItem]


class ErrorBody(TypedDict):
    error: str


type RecallResponse = RecallBody | ErrorBody


def _config(root: Path) -> AgmemConfig:
    return AgmemConfig(
        data_dir=root / "data",
        overrides={
            "doc_store": "SqliteDocStore",
            "vector_store": "SqliteVecStore",
            "graph_store": "SqliteGraphStore",
            "embedder": "FakeEmbedder",
        },
    )


def _start(root: Path):
    server._registry.close_all()
    return server._registry.start("hookrecall", ["experience"], _config(root))


def _indexed_runbook(mem, item_id: str, content: str, cwd: str) -> None:
    data = {
        "id": item_id,
        "content": content,
        "name": item_id,
        "origin": {"cwd": cwd, "ended_at": "2026-09-09T00:00:00+00:00"},
        "source_episode_ids": [],
    }
    mem.doc_store.put_item(item_id, "runbooks", mem.namespace, data)
    mem.vector_store.add(
        item_id,
        mem.embedder.embed([content])[0],
        memory_type="runbooks",
        namespace=mem.namespace,
    )


def _recall(payload: dict[str, str | int]) -> tuple[int, RecallResponse]:
    encoded = json.dumps(payload).encode()

    async def receive() -> dict[str, str | bytes | bool]:
        return {"type": "http.request", "body": encoded, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/hooks/recall",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )
    response = anyio.run(server.hooks_recall, request)
    loaded = json.loads(bytes(response.body).decode())
    assert isinstance(loaded, dict)
    if "items" in loaded:
        raw_namespace = loaded.get("namespace")
        raw_items = loaded.get("items")
        assert isinstance(raw_namespace, str)
        assert isinstance(raw_items, list)
        items: list[RecallItem] = []
        for raw_item in raw_items:
            assert isinstance(raw_item, dict)
            item_id = raw_item.get("id")
            timestamp = raw_item.get("timestamp")
            assert item_id is None or isinstance(item_id, str)
            assert timestamp is None or isinstance(timestamp, str)
            assert isinstance(raw_item.get("memory_type"), str)
            assert isinstance(raw_item.get("score"), int | float)
            assert isinstance(raw_item.get("text"), str)
            items.append(
                {
                    "id": item_id,
                    "memory_type": raw_item["memory_type"],
                    "score": float(raw_item["score"]),
                    "timestamp": timestamp,
                    "text": raw_item["text"],
                }
            )
        return response.status_code, {"namespace": raw_namespace, "items": items}
    raw_error = loaded.get("error")
    assert isinstance(raw_error, str)
    return response.status_code, {"error": raw_error}


def _recall_items(body: RecallResponse) -> list[RecallItem]:
    assert "items" in body
    return body["items"]


def _recall_body(body: RecallResponse) -> RecallBody:
    assert "items" in body
    return body


def test_hooks_recall_filters_episodic_role_and_project_before_top_k(tmp_path: Path) -> None:
    mem = _start(tmp_path)
    try:
        query = "needle alpha beta gamma"
        for idx in range(12):
            mem.add_message(
                f"{query} ineligible assistant trace {idx}",
                role="assistant",
                meta={"cwd": "/work/project-a"},
            )
        mem.add_message(
            f"{query} ineligible tool output",
            role="tool_result",
            meta={"cwd": "/work/project-a"},
        )
        for idx in range(12):
            mem.add_message(
                f"{query} foreign user trace {idx}",
                role="user",
                meta={"cwd": "/work/project-b"},
            )
        mem.add_message(
            "needle alpha beta eligible user decision",
            role="user",
            meta={"cwd": "/work/project-a"},
        )
        _indexed_runbook(
            mem,
            "rb-eligible",
            "needle alpha beta gamma reusable runbook",
            "/work/project-a",
        )
        raw_roles = {
            getattr(item.item, "role", "")
            for item in mem.search(query, memory_types=("episodic",), k=10).items
        }
        assert "assistant" in raw_roles

        status, body = _recall({"query": query, "k": 2, "cwd": "/work/project-a"})

        assert status == 200
        ok_body = _recall_body(body)
        assert ok_body["namespace"] == "hookrecall"
        texts = [item["text"] for item in ok_body["items"]]
        assert len(texts) == 2
        assert any("eligible user decision" in text for text in texts), body
        assert any("reusable runbook" in text for text in texts), body
        assert all(
            "assistant trace" not in text
            and "tool output" not in text
            and "foreign user trace" not in text
            for text in texts
        )
    finally:
        server._registry.close_all()


def test_hooks_recall_sorts_and_truncates_globally(tmp_path: Path) -> None:
    mem = _start(tmp_path)
    try:
        query = "global sort recall"
        for idx in range(3):
            mem.add_message(
                f"{query} user memory {idx}",
                role="user",
                meta={"cwd": "/work/project-a"},
            )
        _indexed_runbook(mem, "rb-1", f"{query} runbook one", "/work/project-a")
        _indexed_runbook(mem, "rb-2", f"{query} runbook two", "/work/project-a")

        status, body = _recall({"query": query, "k": 3, "cwd": "/work/project-a"})

        assert status == 200
        items = _recall_items(body)
        scores = [item["score"] for item in items]
        assert len(items) == 3
        assert scores == sorted(scores, reverse=True)
    finally:
        server._registry.close_all()


def test_hooks_recall_handles_non_positive_k_at_the_boundary(tmp_path: Path) -> None:
    mem = _start(tmp_path)
    try:
        mem.add_message("zero k should not retrieve this", role="user")

        zero_status, zero_body = _recall({"query": "zero", "k": 0})
        negative_status, negative_body = _recall({"query": "zero", "k": -1})
        invalid_status, invalid_body = _recall({"query": "zero", "k": "nope"})

        assert zero_status == 200
        assert _recall_items(zero_body) == []
        assert negative_status == 400
        assert negative_body == {"error": "k must be non-negative"}
        assert invalid_status == 400
        assert invalid_body == {"error": "k must be an integer"}
    finally:
        server._registry.close_all()
