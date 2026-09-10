from __future__ import annotations

from collections.abc import Iterator

import pytest

from agmem.core.types import Episode
from agmem.hooks import recall_prompt
from agmem.stores.sqlite_doc import SqliteDocStore


@pytest.fixture
def store() -> Iterator[SqliteDocStore]:
    doc_store = SqliteDocStore()
    try:
        yield doc_store
    finally:
        doc_store.close()


def test_request_body_sends_the_resolved_hook_namespace(monkeypatch):
    monkeypatch.setenv("AGMEM_NAMESPACE", "hook-side")

    body = recall_prompt.request_body({"cwd": "/w/proj-a", "prompt": "q"}, "q", 5)

    assert body == {"query": "q", "k": 5, "cwd": "/w/proj-a", "namespace": "hook-side"}


def test_fallback_returns_no_items_when_k_is_not_positive(store: SqliteDocStore):
    store.add_episode(Episode(content="needlealpha local", namespace="hooktest"))

    zero = recall_prompt.fallback_items(store, "hooktest", "needlealpha", 0, "/w/proj-a")
    negative = recall_prompt.fallback_items(store, "hooktest", "needlealpha", -1, "/w/proj-a")

    assert zero == []
    assert negative == []


def test_fallback_widens_runbook_candidates_after_project_filtering(store: SqliteDocStore):
    for idx in range(3):
        store.put_item(
            f"foreign-{idx}",
            "runbooks",
            "hooktest",
            {
                "id": f"foreign-{idx}",
                "content": " ".join(["quartzfilter"] * 20),
                "origin": {"cwd": "/w/proj-b", "ended_at": "2026-09-01T00:00:00+00:00"},
            },
        )
    store.put_item(
        "local-runbook-1",
        "runbooks",
        "hooktest",
        {
            "id": "local-runbook-1",
            "content": "quartzfilter local fix",
            "origin": {"cwd": "/w/proj-a", "ended_at": "2026-09-02T00:00:00+00:00"},
        },
    )
    store.put_item(
        "local-runbook-2",
        "runbooks",
        "hooktest",
        {
            "id": "local-runbook-2",
            "content": "quartzfilter local second fix",
            "origin": {"cwd": "/w/proj-a", "ended_at": "2026-09-03T00:00:00+00:00"},
        },
    )
    store.add_episode(
        Episode(
            id="local-user",
            content=" ".join(["quartzfilter"] * 10),
            role="user",
            namespace="hooktest",
            meta={"cwd": "/w/proj-a"},
        )
    )

    items = recall_prompt.fallback_items(store, "hooktest", "quartzfilter", 2, "/w/proj-a")

    assert [item["id"] for item in items] == ["local-runbook-1", "local-runbook-2"]


def test_fallback_widens_episode_candidates_after_role_and_project_filtering(store: SqliteDocStore):
    distractors = (
        ("assistant", "/w/proj-a", " ".join(["needlealpha"] * 25)),
        ("tool_result", "/w/proj-a", " ".join(["needlealpha"] * 20)),
        ("user", "/w/proj-b", " ".join(["needlealpha"] * 15)),
    )
    for role, cwd, content in distractors:
        store.add_episode(
            Episode(content=content, role=role, namespace="hooktest", meta={"cwd": cwd})
        )
    store.add_episode(
        Episode(
            id="local-user",
            content="needlealpha local user request",
            role="user",
            namespace="hooktest",
            meta={"cwd": "/w/proj-a"},
        )
    )

    items = recall_prompt.fallback_items(store, "hooktest", "needlealpha", 1, "/w/proj-a")

    assert [item["id"] for item in items] == ["local-user"]


def test_fallback_store_error_preserves_fail_open(monkeypatch):
    import sqlite3

    monkeypatch.setattr(recall_prompt, "read_event", lambda: {"prompt": "needlealpha"})
    monkeypatch.setattr(recall_prompt.daemon_client, "health", lambda: None)

    def broken_store():
        raise sqlite3.OperationalError("test database unavailable")

    monkeypatch.setattr(recall_prompt, "open_doc_store", broken_store)
    with pytest.raises(SystemExit) as result:
        recall_prompt.main()
    assert result.value.code == 0
