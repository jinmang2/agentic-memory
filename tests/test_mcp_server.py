"""The MCP server, driven over the transport it actually ships on.

This is the only production surface this package exposes — `agmem-mcp` is its
one console script — and until 2026-08-07 nothing exercised it. The module
imported and registered six tools, which is what made the gap easy to miss: a
smoke that stopped at `import` would have passed while every tool body, the
stdio transport and the server's own memory wiring went unrun. This session
found four separate "wired but dead" defects elsewhere in the repo by checking
the path the caller actually takes, so the server gets the same treatment.

Hermetic on purpose: a temp `agmem.toml` forces `FakeEmbedder`, so the test
neither downloads a model nor depends on one being cached. The lite profile's
real default (`intfloat/multilingual-e5-small`) is exercised by hand, not here —
a CI machine without the cache would otherwise fetch ~120 MB to assert routing.
"""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

mcp_client = pytest.importorskip("mcp.client.stdio", reason="mcp SDK not installed")
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {
    "add_memory",
    "add_task_result",
    "search_memory",
    "get_playbook",
    "report_feedback",
    "memory_stats",
}


def _text(result) -> str:
    return " ".join(getattr(c, "text", "") for c in (getattr(result, "content", None) or []))


async def _drive(data_dir, config_path) -> dict:
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "agmem.mcp.server",
            "--namespace",
            "smoke",
            "--data-dir",
            str(data_dir),
            "--config",
            str(config_path),
        ],
        env=None,
    )
    out: dict = {}
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        out["tools"] = {t.name for t in (await session.list_tools()).tools}
        for msg in (
            "I moved to Berlin in March and started at Acme.",
            "My sister Diana works at the museum in Prague.",
        ):
            r = await session.call_tool("add_memory", {"content": msg, "role": "user"})
            out.setdefault("adds", []).append(_text(r))
        out["search"] = _text(
            await session.call_tool("search_memory", {"query": "Where did I move?"})
        )
        out["stats"] = _text(await session.call_tool("memory_stats", {}))
        out["playbook"] = _text(await session.call_tool("get_playbook", {}))
        out["task"] = _text(
            await session.call_tool("add_task_result", {"task": "ship it", "outcome": "success"})
        )
    return out


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    root = tmp_path_factory.mktemp("agmem-mcp")
    cfg = root / "agmem.toml"
    # FakeEmbedder keeps this hermetic; it is a real Embedder (hashed
    # bag-of-words), so retrieval still ranks rather than being stubbed out.
    cfg.write_text('[profile]\nname = "lite"\n\n[override]\nembedder = "FakeEmbedder"\n')
    return asyncio.run(_drive(root / "data", cfg))


def test_server_speaks_mcp_and_advertises_every_tool(served):
    """A handshake plus the tool list: the server starts under the real
    transport, not just as an importable module."""
    assert EXPECTED_TOOLS <= served["tools"], EXPECTED_TOOLS - served["tools"]


def test_add_memory_persists_and_reports_the_episode_id(served):
    for body in served["adds"]:
        payload = json.loads(body)
        assert payload["stored"] is True
        assert payload["episode_id"]


def test_search_returns_the_ingested_fact_not_an_empty_bundle(served):
    """The assertion that would have caught a broken read path. `search_memory`
    returning valid-but-empty JSON is the failure this repo keeps meeting —
    a tool that answers successfully while retrieving nothing."""
    payload = json.loads(served["search"])
    assert payload["items"], "search returned no items"
    assert "Berlin" in payload["context"]


def test_memory_stats_reports_the_wiring_it_actually_resolved(served):
    """Stats must name the live adapters. A run whose embedder or vector store
    silently fell back is indistinguishable from a healthy one otherwise —
    the same reason the bench stamp gained `reranker`/`degradations`."""
    payload = json.loads(served["stats"])["stats"]
    assert payload["namespace"] == "smoke"
    assert payload["episodes"] == 2
    assert payload["embedder"] == "fake-hash-256"  # the override took effect
    assert payload["vector_store"]


async def _drive_namespaces(data_dir, config_path, env: dict | None) -> dict:
    """A second server, started WITHOUT --namespace, to check the env tier and
    the per-tool namespace argument. Separate from `served` so that fixture
    keeps pinning the explicit-flag path."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agmem.mcp.server", "--data-dir", str(data_dir), "--config", str(config_path)],
        env=env,
    )
    out: dict = {}
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        out["stats_default"] = json.loads(_text(await session.call_tool("memory_stats", {})))
        await session.call_tool("add_memory", {"content": "The default store has Oslo."})
        await session.call_tool(
            "add_memory", {"content": "The side store has Lima.", "namespace": "side"}
        )
        out["search_default"] = json.loads(
            _text(await session.call_tool("search_memory", {"query": "Which city?"}))
        )
        out["search_side"] = json.loads(
            _text(
                await session.call_tool(
                    "search_memory", {"query": "Which city?", "namespace": "side"}
                )
            )
        )
        out["stats_after"] = json.loads(_text(await session.call_tool("memory_stats", {})))
        bad = await session.call_tool(
            "add_memory", {"content": "escape", "namespace": "../elsewhere"}
        )
        out["bad_is_error"] = bool(getattr(bad, "isError", False))
        out["bad_text"] = _text(bad)
    return out


@pytest.fixture(scope="module")
def namespaced(tmp_path_factory):
    root = tmp_path_factory.mktemp("agmem-mcp-ns")
    cfg = root / "agmem.toml"
    cfg.write_text('[profile]\nname = "lite"\n\n[override]\nembedder = "FakeEmbedder"\n')
    import os

    env = {k: v for k, v in os.environ.items() if k != "AGMEM_NAMESPACE"}
    env["AGMEM_NAMESPACE"] = "from-env"
    out = asyncio.run(_drive_namespaces(root / "data", cfg, env))
    out["dirs"] = sorted(p.name for p in (root / "data").iterdir() if p.is_dir())
    return out


def test_server_takes_its_default_namespace_from_the_environment(namespaced):
    """The env tier of docs/05 §2.2, which the server did not have until
    2026-09-02: AGMEM_NAMESPACE, the variable the hooks read, now names the
    server's default too, so one export configures both layers."""
    assert namespaced["stats_default"]["stats"]["namespace"] == "from-env"
    assert namespaced["stats_default"]["default_namespace"] == "from-env"


def test_tools_take_a_namespace_and_keep_stores_apart(namespaced):
    """One server, two memories: what went into `side` is not found from the
    default and vice versa, and both live under the same data dir."""
    assert "Oslo" in namespaced["search_default"]["context"]
    assert "Lima" not in namespaced["search_default"]["context"]
    assert "Lima" in namespaced["search_side"]["context"]
    assert "Oslo" not in namespaced["search_side"]["context"]
    assert namespaced["search_side"]["namespace"] == "side"
    assert namespaced["stats_after"]["open_namespaces"] == ["from-env", "side"]
    assert namespaced["dirs"] == ["from-env", "side"]


def test_a_namespace_that_is_not_a_directory_name_is_refused(namespaced):
    """The model supplies this argument, and it becomes a path segment under
    the data dir. It is validated where it enters."""
    assert namespaced["bad_is_error"]
    assert "single directory name" in namespaced["bad_text"]


def test_procedural_tools_answer_without_an_llm_configured(served):
    """No endpoint is configured here, so the organizers must skip explicitly
    rather than raise. `get_playbook` returning empty and `add_task_result`
    still recording is the documented no-LLM behaviour."""
    assert "empty" in served["playbook"].lower()
    assert json.loads(served["task"])["recorded"] is True
