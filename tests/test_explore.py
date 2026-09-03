"""The explorer read path: a file view of the raw memory, and an agent that
greps it.

docs/research/agent-memory-axes-v1.md §6 says the honest v1 baseline is not "no
memory" but "raw session logs plus a grep-capable agent", and §7.3 lists what
the read path is missing for that comparison to exist: no way for a model to
reach the transcript with a shell tool, and no per-query wall clock. These
tests cover both halves — the materialization (deterministic, incremental) and
the loop (path-safe, cited, refusing rather than degrading into a plain vector
search).

Nothing here spends: `FakeEmbedder` for the vectors, `StubLLM` for the loop,
and the searches run the real `grep` on a tmp directory.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from helpers import StubLLM

from agmem import AgenticMemory
from agmem.config import AgmemConfig
from agmem.core.ops import MemoryOp, OpType
from agmem.embed.fake import FakeEmbedder
from agmem.explore import Explorer, export_workspace
from agmem.llm.client import RoleConfig
from agmem.sessions import SessionTrajectory, Step

TS = datetime(2026, 9, 2, 1, 0, tzinfo=UTC)


def _session(session_id: str = "sess-42", host: str = "claude-code") -> SessionTrajectory:
    traj = SessionTrajectory(
        id=session_id,
        host=host,
        source_path="x",
        cwd="/home/u/proj",
        started_at=TS,
    )
    traj.steps = [
        Step(kind="user", text="Fix the flaky test in tests/test_daemon.py", timestamp=TS),
        Step(kind="assistant", text="Looking.", timestamp=TS),
        Step(
            kind="tool_call", text='{"command": "uv run pytest -q"}', tool_name="Bash", timestamp=TS
        ),
        Step(
            kind="tool_result",
            text="1 failed: test_idle_timeout TimeoutExpired",
            tool_name="Bash",
            timestamp=TS,
        ),
        Step(kind="user", text="use --idle-timeout 2 in that test, not 30", timestamp=TS),
    ]
    return traj


def _mem(**config_kwargs) -> AgenticMemory:
    return AgenticMemory(
        namespace="t",
        organizers=["experience"],
        embedder=FakeEmbedder(dim=128),
        config=AgmemConfig(sync_write=True, **config_kwargs),
    )


def _add_runbook(mem: AgenticMemory, item_id: str, content: str, session_id: str) -> None:
    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.ADD,
                target_type="runbooks",
                target_id=item_id,
                actor="experience",
                payload={
                    "id": item_id,
                    "content": content,
                    "embedding_text": content.splitlines()[0],
                    "session_id": session_id,
                    "source_host": "claude-code",
                    "step_range": [0, 4],
                    "source_episode_ids": [],
                },
            )
        ],
        actor="experience",
    )


def _populated(mem: AgenticMemory) -> AgenticMemory:
    mem.add_session(_session(), distill=False)
    mem.add_session(_session("sess-99", host="codex"), distill=False)
    mem.add_message("remember the release cadence", role="user", timestamp=TS)
    _add_runbook(
        mem,
        "rb-1",
        "# Task: fix the flaky daemon test\noutcome: success\n\n## Procedure\n- pass"
        " --idle-timeout 2\n\nsource: claude-code session sess-42 steps 0-4",
        "sess-42",
    )
    return mem


# --- workspace export -------------------------------------------------------


def test_export_writes_sessions_messages_runbooks_and_an_index(tmp_path):
    mem = _populated(_mem())
    stats = export_workspace(mem, tmp_path / "ws")
    root = tmp_path / "ws"

    assert (root / "sessions" / "claude-code" / "sess-42.md").exists()
    assert (root / "sessions" / "codex" / "sess-99.md").exists()
    assert (root / "runbooks" / "rb-1.md").read_text().startswith("# Task: fix the flaky")
    assert (root / "messages" / "2026-09.md").exists()
    assert stats.sessions == 2 and stats.runbooks == 1 and stats.messages == 1
    mem.close()


def test_a_session_file_reads_like_the_rendered_transcript(tmp_path):
    mem = _populated(_mem())
    export_workspace(mem, tmp_path / "ws")
    text = (tmp_path / "ws" / "sessions" / "claude-code" / "sess-42.md").read_text()

    assert "session: sess-42" in text
    assert "host: claude-code" in text
    assert "cwd: /home/u/proj" in text
    # The `[i] KIND(tool)` label of SessionTrajectory.render, so the distiller's
    # step citations and the explorer's line numbers name the same steps.
    assert "[0] USER" in text
    assert "[2] TOOL_CALL(Bash)" in text
    assert "[4] USER\nuse --idle-timeout 2 in that test, not 30" in text
    mem.close()


def test_a_message_with_no_session_lands_in_its_month_file(tmp_path):
    mem = _populated(_mem())
    export_workspace(mem, tmp_path / "ws")
    text = (tmp_path / "ws" / "messages" / "2026-09.md").read_text()
    assert "- (2026-09-02) [user] remember the release cadence" in text
    mem.close()


def test_the_index_names_every_session_and_runbook(tmp_path):
    mem = _populated(_mem())
    export_workspace(mem, tmp_path / "ws")
    index = (tmp_path / "ws" / "INDEX.md").read_text()

    assert "claude-code" in index and "codex" in index
    assert "sess-42" in index and "sess-99" in index
    assert "steps=5" in index
    assert "Fix the flaky test in tests/test_daemon.py" in index
    assert "rb-1" in index and "# Task: fix the flaky daemon test" in index
    mem.close()


def test_a_second_export_writes_nothing_and_is_byte_identical(tmp_path):
    mem = _populated(_mem())
    root = tmp_path / "ws"
    first = export_workspace(mem, root)
    before = {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}

    second = export_workspace(mem, root)
    after = {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}

    assert first.written > 0
    assert second.written == 0
    assert second.unchanged == first.written
    assert before == after
    mem.close()


def test_a_retired_runbook_loses_its_file(tmp_path):
    mem = _populated(_mem())
    root = tmp_path / "ws"
    export_workspace(mem, root)
    assert (root / "runbooks" / "rb-1.md").exists()

    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.DELETE,
                target_type="runbooks",
                target_id="rb-1",
                actor="ingest",
                payload={"reason": "re-distillation"},
            )
        ],
        actor="ingest",
    )
    stats = export_workspace(mem, root)

    assert not (root / "runbooks" / "rb-1.md").exists()
    assert stats.removed == 1
    mem.close()


def test_export_leaves_files_it_did_not_write_alone(tmp_path):
    mem = _populated(_mem())
    root = tmp_path / "ws"
    export_workspace(mem, root)
    (root / "NOTES.md").write_text("mine")

    export_workspace(mem, root)

    assert (root / "NOTES.md").read_text() == "mine"
    mem.close()


def test_a_hook_captured_turn_is_a_message_even_though_it_names_a_session(tmp_path):
    """The capture hook keeps the host's session id on each turn but numbers no
    steps; those turns are conversation, not a trajectory, and must not be
    filed as a session with every step labelled [0]."""
    mem = _mem()
    for text in ("what is the release cadence?", "and the daemon timeout?"):
        mem.add_message(
            text,
            role="user",
            timestamp=TS,
            meta={"source": "claude-code", "session_id": "b3f1-from-the-host"},
        )
    stats = export_workspace(mem, tmp_path / "ws")
    assert stats.sessions == 0 and stats.messages == 2
    assert not (tmp_path / "ws" / "sessions").exists()
    text = (tmp_path / "ws" / "messages" / "2026-09.md").read_text()
    assert "release cadence" in text and "daemon timeout" in text
    mem.close()


def test_mixed_naive_and_aware_timestamps_do_not_break_the_export(tmp_path):
    from datetime import datetime as _dt

    from agmem.core.types import Episode

    mem = _mem()
    naive = _dt.fromisoformat("2026-09-02T01:00:00")  # a log line without an offset
    for index, (stamp, text) in enumerate(((naive, "first"), (TS, "second"))):
        mem.doc_store.add_episode(
            Episode(
                content=text,
                role="user",
                namespace="t",
                timestamp=stamp,
                meta={
                    "source": "codex",
                    "session_id": "mixed",
                    "step_index": index,
                    "kind": "user",
                },
            )
        )
    stats = export_workspace(mem, tmp_path / "ws")
    assert stats.sessions == 1
    text = (tmp_path / "ws" / "sessions" / "codex" / "mixed.md").read_text()
    assert "[0] USER\nfirst" in text and "[1] USER\nsecond" in text
    mem.close()


# --- the explorer loop ------------------------------------------------------


def _workspace(tmp_path) -> Path:
    root = tmp_path / "ws"
    (root / "sessions" / "claude-code").mkdir(parents=True)
    (root / "sessions" / "claude-code" / "sess-42.md").write_text(
        "session: sess-42\nhost: claude-code\n\n"
        "[0] USER\nFix the flaky test in tests/test_daemon.py\n\n"
        "[1] TOOL_RESULT(Bash)\n1 failed: test_idle_timeout TimeoutExpired\n\n"
        "[2] USER\nuse --idle-timeout 2 in that test, not 30\n"
    )
    (root / "runbooks").mkdir()
    (root / "runbooks" / "rb-1.md").write_text("# Task: fix the flaky daemon test\n")
    return root


def _final(context: str, citations: list[dict]) -> dict:
    return {
        "action": "final",
        "reason": "found it",
        "context": context,
        "citations": citations,
    }


def test_the_loop_searches_reads_and_returns_a_cited_context(tmp_path):
    root = _workspace(tmp_path)
    llm = StubLLM(
        {
            "explore": [
                {"action": "search", "reason": "find the flag", "pattern": "idle-timeout"},
                {
                    "action": "read",
                    "reason": "read around the hit",
                    "path": "sessions/claude-code/sess-42.md",
                    "start": 1,
                    "end": 12,
                },
                _final(
                    "The user asked for --idle-timeout 2 in the daemon test.",
                    [{"file": "sessions/claude-code/sess-42.md", "lines": [9, 10]}],
                ),
            ]
        }
    )
    result = Explorer(root, search_tool="grep").research("what idle timeout?", llm)

    assert result.degraded is None
    assert "idle-timeout 2" in result.context
    assert result.citations == [{"file": "sessions/claude-code/sess-42.md", "lines": [9, 10]}]
    assert [s["action"] for s in result.steps] == ["search", "read", "final"]
    assert result.llm_calls == 3
    assert result.search_tool == "grep"
    assert result.latency_s >= 0.0
    assert all(s["seconds"] >= 0.0 for s in result.steps)
    # The search really ran: the observation carried the matching line.
    assert "idle-timeout 2" in result.steps[0]["observation"]


def test_the_search_observation_is_the_tools_own_output(tmp_path):
    root = _workspace(tmp_path)
    seen: list[str] = []

    class Recording(StubLLM):
        def call(self, role, prompt, schema, required_keys=(), **kwargs):
            seen.append(prompt)
            return super().call(role, prompt, schema, required_keys, **kwargs)

    llm = Recording(
        {
            "explore": [
                {"action": "search", "reason": "find it", "pattern": "TimeoutExpired"},
                _final("ok", []),
            ]
        }
    )
    Explorer(root, search_tool="grep").research("why did it fail?", llm)

    assert "sessions/claude-code/sess-42.md" in seen[-1]
    assert "TimeoutExpired" in seen[-1]


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
def test_ripgrep_is_used_when_it_is_available(tmp_path):
    root = _workspace(tmp_path)
    llm = StubLLM(
        {
            "explore": [
                {"action": "search", "reason": "find it", "pattern": "TimeoutExpired"},
                _final("ok", []),
            ]
        }
    )
    result = Explorer(root).research("why did it fail?", llm)
    assert result.search_tool == "rg"
    assert "TimeoutExpired" in result.steps[0]["observation"]


def test_listing_a_directory_is_bounded_and_relative(tmp_path):
    root = _workspace(tmp_path)
    llm = StubLLM(
        {
            "explore": [
                {"action": "list", "reason": "see the layout", "path": "sessions"},
                _final("ok", []),
            ]
        }
    )
    result = Explorer(root, search_tool="grep").research("what is here?", llm)
    assert result.steps[0]["action"] == "list"
    assert "sessions/claude-code/sess-42.md" in result.steps[0]["observation"]


@pytest.mark.parametrize("path", ["../../etc/passwd", "/etc/passwd", "sessions/../../.."])
def test_a_path_outside_the_root_is_an_observation_not_an_exception(tmp_path, path):
    root = _workspace(tmp_path)
    llm = StubLLM(
        {
            "explore": [
                {"action": "read", "reason": "peek", "path": path, "start": 1, "end": 5},
                _final("nothing", []),
            ]
        }
    )
    result = Explorer(root, search_tool="grep").research("read the host", llm)

    assert result.degraded is None
    assert result.steps[0]["action"] == "read"
    assert "outside" in result.steps[0]["observation"].lower()


def test_a_citation_that_does_not_resolve_is_dropped(tmp_path):
    root = _workspace(tmp_path)
    llm = StubLLM(
        {
            "explore": [
                _final(
                    "the answer",
                    [
                        {"file": "sessions/claude-code/sess-42.md", "lines": [1, 2]},
                        {"file": "sessions/claude-code/sess-42.md", "lines": [900, 901]},
                        {"file": "nope.md", "lines": [1, 2]},
                    ],
                )
            ]
        }
    )
    result = Explorer(root, search_tool="grep").research("q", llm)

    assert result.citations == [{"file": "sessions/claude-code/sess-42.md", "lines": [1, 2]}]
    assert result.steps[-1]["dropped_citations"] == 2


def test_the_context_is_truncated_to_the_budget(tmp_path):
    root = _workspace(tmp_path)
    llm = StubLLM({"explore": [_final("x" * 5000, [])]})
    result = Explorer(root, search_tool="grep", budget_tokens=100).research("q", llm)

    assert len(result.context) <= 100 * 4 + 64
    assert "truncated" in result.context


def test_running_out_of_steps_forces_one_final_call(tmp_path):
    root = _workspace(tmp_path)
    search = {"action": "search", "reason": "again", "pattern": "USER"}
    llm = StubLLM({"explore": [dict(search) for _ in range(2)] + [_final("late answer", [])]})
    result = Explorer(root, max_steps=2, search_tool="grep").research("q", llm)

    assert result.degraded is None
    assert result.context == "late answer"
    assert result.llm_calls == 3  # two explorations plus the forced one
    assert "answer now" in llm.calls[-1][1].lower()


def test_never_answering_degrades_instead_of_inventing_one(tmp_path):
    root = _workspace(tmp_path)
    search = {"action": "search", "reason": "again", "pattern": "USER"}
    llm = StubLLM({"explore": [dict(search) for _ in range(5)]})
    result = Explorer(root, max_steps=2, search_tool="grep").research("q", llm)

    assert result.degraded == "max_steps"
    assert result.context == ""


def test_a_dropped_call_degrades_rather_than_falling_back_to_a_vector_search(tmp_path):
    root = _workspace(tmp_path)
    llm = StubLLM({"explore": []})  # every call drops
    result = Explorer(root, search_tool="grep").research("q", llm)

    assert result.degraded == "llm_drop"
    assert result.context == ""
    assert result.citations == []


def test_the_system_prompt_says_the_transcript_wins_over_a_runbook(tmp_path):
    root = _workspace(tmp_path)
    llm = StubLLM({"explore": [_final("ok", [])]})
    Explorer(root, search_tool="grep").research("q", llm)

    system = llm.systems[0].lower()
    assert "sessions/" in system and "runbook" in system


# --- facade and latency -----------------------------------------------------


def test_research_exports_the_workspace_before_exploring(tmp_path):
    mem = _populated(_mem())
    mem.structured = StubLLM({"explore": [_final("the cadence is weekly", [])]})
    root = tmp_path / "ws"

    result = mem.research("what cadence?", root=root)

    assert (root / "INDEX.md").exists()
    assert result.context == "the cadence is weekly"
    mem.close()


def test_research_refuses_when_no_explore_role_is_configured(tmp_path):
    cfg = AgmemConfig(
        sync_write=True,
        llm_roles={"distill": RoleConfig(endpoint="http://x/v1", model="m")},
    )
    mem = AgenticMemory(
        namespace="t", organizers=["experience"], embedder=FakeEmbedder(dim=128), config=cfg
    )
    with pytest.raises(RuntimeError, match=r"\[llm\.explore\]"):
        mem.research("q", root=tmp_path / "ws")
    mem.close()


def test_research_needs_a_root_when_the_memory_has_no_data_dir():
    mem = _mem()
    mem.structured = StubLLM({"explore": [_final("ok", [])]})
    with pytest.raises(ValueError, match="root"):
        mem.research("q")
    mem.close()


def test_research_defaults_the_root_under_the_data_dir(tmp_path):
    mem = _mem(data_dir=tmp_path / "data")
    mem.structured = StubLLM({"explore": [_final("ok", [])]})
    mem.research("q")
    assert (tmp_path / "data" / "t" / "workspace" / "INDEX.md").exists()
    mem.close()


def test_research_latency_includes_the_export(tmp_path):
    mem = _populated(_mem())
    mem.structured = StubLLM({"explore": [_final("ok", [])]})
    result = mem.research("q", root=tmp_path / "ws")
    assert result.export_s > 0.0
    assert result.latency_s >= result.export_s
    mem.close()


def test_search_records_its_own_wall_clock(tmp_path):
    mem = _populated(_mem())
    metrics: dict = {}
    mem.search("idle timeout", metrics=metrics)

    assert "latency_s" in metrics and metrics["latency_s"] >= 0.0
    assert metrics["agent"] == "search"  # the existing accounting is still there
    mem.close()


def test_a_planned_search_reports_latency_too():
    from agmem.policies.retrieval import STRATEGIES
    from agmem.retrieval.planned import PlannedSearch

    mem = _mem()
    mem.add_message("the daemon idle timeout is 30 seconds", role="user")
    planned = PlannedSearch(mem, STRATEGIES["direct"]())
    metrics: dict = {}
    planned.search("idle timeout", metrics=metrics)

    assert "latency_s" in metrics and metrics["latency_s"] >= 0.0
    mem.close()


# --- CLI --------------------------------------------------------------------


def _run_cli(args, tmp_path, extra_config: str = ""):
    """`python -m agmem.explore` in its own process, against a config whose
    embedder slot is `FakeEmbedder` — the real one would fetch weights."""
    config = tmp_path / "agmem.toml"
    config.write_text(
        '[profile]\nname = "lite"\n\n'
        f'[storage]\ndata_dir = "{tmp_path / "data"}"\n\n'
        '[override]\nembedder = "FakeEmbedder"\n' + extra_config
    )
    env = dict(os.environ)
    env["AGMEM_CONFIG"] = str(config)
    env.pop("AGMEM_DATA_DIR", None)
    env.pop("AGMEM_NAMESPACE", None)
    return subprocess.run(
        [sys.executable, "-m", "agmem.explore", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        check=False,
    )


def test_cli_export_materializes_the_workspace(tmp_path):
    # The CLI resolves `FakeEmbedder` at its default dim, so the memory that
    # seeds the store must use the same one or the vector index refuses to open.
    mem = _populated(
        AgenticMemory(
            namespace="main",
            organizers=["experience"],
            embedder=FakeEmbedder(),
            config=AgmemConfig(sync_write=True, data_dir=tmp_path / "data"),
        )
    )
    mem.close()

    proc = _run_cli(["export"], tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "data" / "main" / "workspace" / "INDEX.md").exists()
    assert "sessions=2" in proc.stdout


def test_cli_ask_refuses_without_an_explore_role(tmp_path):
    proc = _run_cli(["ask", "what happened?"], tmp_path)
    assert proc.returncode == 2
    assert "[llm.explore]" in (proc.stderr + proc.stdout)


def test_cli_ask_caps_the_step_count(tmp_path):
    proc = _run_cli(["ask", "q", "--max-steps", "99"], tmp_path)
    assert proc.returncode == 2
    assert "--max-steps" in (proc.stderr + proc.stdout)


# --- MCP --------------------------------------------------------------------


def test_the_mcp_tool_returns_an_error_object_instead_of_crashing(tmp_path):
    """The server must answer a `research_memory` call on a memory with no
    explore role, because a crash there takes the whole stdio session down."""
    from agmem.mcp import server

    cfg = AgmemConfig(sync_write=True, data_dir=tmp_path / "data")
    mem = AgenticMemory(
        namespace="t", organizers=["experience"], embedder=FakeEmbedder(dim=128), config=cfg
    )
    server._registry._mems["t"] = mem  # the server's per-namespace cache
    server._registry.default = "t"
    server._registry.config = cfg
    try:
        payload = json.loads(server.research_memory("q", namespace="t"))
    finally:
        server._registry._mems.pop("t", None)
        mem.close()

    assert "[llm.explore]" in payload["error"]


def test_the_mcp_tool_clamps_the_step_count(tmp_path):
    """A model fills this surface in; a step is a paid call."""
    from agmem.explore.explorer import MAX_STEPS_CAP
    from agmem.mcp import server

    cfg = AgmemConfig(sync_write=True, data_dir=tmp_path / "data")
    mem = AgenticMemory(
        namespace="t", organizers=["experience"], embedder=FakeEmbedder(dim=128), config=cfg
    )
    search = {"action": "search", "reason": "again", "pattern": "USER"}
    mem.structured = StubLLM({"explore": [dict(search) for _ in range(500)]})
    server._registry._mems["t"] = mem
    server._registry.default = "t"
    server._registry.config = cfg
    try:
        payload = json.loads(server.research_memory("q", max_steps=200, namespace="t"))
    finally:
        server._registry._mems.pop("t", None)
        mem.close()
    assert payload["degraded"] == "max_steps"
    assert payload["llm_calls"] == MAX_STEPS_CAP + 1  # the cap, plus the forced final
    assert payload["export_s"] >= 0.0


# --- tool edge cases ----------------------------------------------------------


def _run_actions(tmp_path, actions: list[dict], **kw):
    root = _workspace(tmp_path)
    llm = StubLLM({"explore": [*actions, _final("ok", [])]})
    return Explorer(root, search_tool="grep", **kw).research("q", llm)


def test_search_refuses_an_empty_pattern_and_reports_no_matches(tmp_path):
    result = _run_actions(
        tmp_path,
        [
            {"action": "search", "reason": "r", "pattern": ""},
            {"action": "search", "reason": "r", "pattern": "definitely-not-in-there"},
        ],
    )
    assert "non-empty pattern" in result.steps[0]["observation"]
    assert result.steps[1]["observation"] == "(no matches)"


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
def test_ripgrep_ignores_ignore_files_in_the_workspace(tmp_path):
    root = _workspace(tmp_path)
    (root / ".ignore").write_text("*.md\n")
    llm = StubLLM(
        {
            "explore": [
                {"action": "search", "reason": "r", "pattern": "TimeoutExpired"},
                _final("ok", []),
            ]
        }
    )
    result = Explorer(root, search_tool="rg").research("q", llm)
    assert "TimeoutExpired" in result.steps[0]["observation"]


def test_read_bounds_are_enforced(tmp_path):
    result = _run_actions(
        tmp_path,
        [
            {
                "action": "read",
                "reason": "r",
                "path": "sessions/claude-code/sess-42.md",
                "start": 5,
                "end": 2,
            },
            {
                "action": "read",
                "reason": "r",
                "path": "sessions/claude-code/sess-42.md",
                "start": "x",
            },
            {
                "action": "read",
                "reason": "r",
                "path": "sessions/claude-code/sess-42.md",
                "start": 900,
            },
            {
                "action": "read",
                "reason": "r",
                "path": "sessions/claude-code/sess-42.md",
                "start": 1,
                "end": 999,
            },
            {"action": "read", "reason": "r", "path": "sessions", "start": 1, "end": 2},
        ],
    )
    obs = [s["observation"] for s in result.steps[:5]]
    assert "end before start" in obs[0]
    assert "must be integers" in obs[1]
    assert "beyond end of file" in obs[2]
    assert obs[3].startswith("1: session: sess-42") and len(obs[3].splitlines()) <= 200
    assert "no such file" in obs[4]


def test_list_edge_cases_and_unknown_actions(tmp_path):
    result = _run_actions(
        tmp_path,
        [
            {"action": "list", "reason": "r", "path": "sessions/claude-code/sess-42.md"},
            {"action": "list", "reason": "r", "path": "does-not-exist"},
            {"action": "teleport", "reason": "r"},
        ],
    )
    obs = [s["observation"] for s in result.steps[:3]]
    assert "not a directory" in obs[0] and "not a directory" in obs[1]
    assert result.steps[2]["action"] == "teleport" and obs[2] == "unknown action"
    assert result.degraded is None and result.context == "ok"


def test_the_prompt_numbers_steps_consecutively(tmp_path):
    root = _workspace(tmp_path)
    llm = StubLLM(
        {
            "explore": [
                {"action": "search", "reason": "r", "pattern": "USER"},
                {"action": "list", "reason": "r", "path": "."},
                _final("ok", []),
            ]
        }
    )
    Explorer(root, search_tool="grep").research("q", llm)
    last_prompt = llm.calls[-1][1]
    assert "## step 1: search" in last_prompt and "## step 2: list" in last_prompt
    assert "(2 of 8 used)" in last_prompt


def test_explorer_refuses_a_bad_tool_or_zero_steps(tmp_path):
    with pytest.raises(ValueError, match="search_tool"):
        Explorer(tmp_path, search_tool="ack")
    with pytest.raises(ValueError, match="max_steps"):
        Explorer(tmp_path, max_steps=0)


def test_an_observation_is_clipped(tmp_path):
    root = _workspace(tmp_path)
    (root / "sessions" / "claude-code" / "big.md").write_text("needle\n" * 5000)
    llm = StubLLM(
        {"explore": [{"action": "search", "reason": "r", "pattern": "needle"}, _final("ok", [])]}
    )
    result = Explorer(root, search_tool="grep", max_observation_chars=500).research("q", llm)
    assert result.steps[0]["observation"].endswith("…[observation clipped]")
    assert result.steps[0]["observation_chars"] <= 500 + 32


def test_colliding_ids_get_distinct_files_and_a_retired_host_dir_is_removed(tmp_path):
    from agmem.explore.workspace import safe_name

    assert safe_name("sess-42") == "sess-42"
    assert safe_name("a:b") != safe_name("a/b")
    assert safe_name(" a ") != safe_name("a")
    assert safe_name("..").startswith("_-")

    mem = _mem()
    mem.add_session(_session("only", host="codex"), distill=False)
    root = tmp_path / "ws"
    export_workspace(mem, root)
    assert (root / "sessions" / "codex").is_dir()
    # Retire the session's steps out from under the export.
    for ep in mem.doc_store.list_episodes("t"):
        mem.doc_store._conn.execute("DELETE FROM episodes WHERE id = ?", (ep.id,))
    mem.doc_store._conn.commit()
    export_workspace(mem, root)
    assert not (root / "sessions" / "codex").exists()
    mem.close()


def test_the_mcp_tool_reports_any_failure_as_an_error_object(tmp_path, monkeypatch):
    from agmem.mcp import server

    cfg = AgmemConfig(sync_write=True, data_dir=tmp_path / "data")
    mem = AgenticMemory(
        namespace="t", organizers=["experience"], embedder=FakeEmbedder(dim=128), config=cfg
    )
    mem.structured = StubLLM({"explore": [_final("ok", [])]})

    def boom(*a, **k):
        raise TypeError("can't compare offset-naive and offset-aware datetimes")

    monkeypatch.setattr("agmem.explore.export_workspace", boom)
    server._registry._mems["t"] = mem
    server._registry.default = "t"
    server._registry.config = cfg
    try:
        payload = json.loads(server.research_memory("q", namespace="t"))
    finally:
        server._registry._mems.pop("t", None)
        mem.close()
    assert payload["error"].startswith("TypeError:")


# --- the remaining uncovered paths ---------------------------------------------


def test_a_tool_timeout_is_an_observation(tmp_path, monkeypatch):
    import subprocess as _sp

    def slow(*a, **k):
        raise _sp.TimeoutExpired(cmd=a[0], timeout=k.get("timeout", 0))

    monkeypatch.setattr("agmem.explore.explorer.subprocess.run", slow)
    result = _run_actions(tmp_path, [{"action": "search", "reason": "r", "pattern": "x"}])
    assert "timed out" in result.steps[0]["observation"]
    assert result.degraded is None


def test_refresh_false_reads_the_workspace_as_it_is(tmp_path):
    mem = _populated(_mem())
    mem.structured = StubLLM({"explore": [_final("a", []), _final("b", [])]})
    root = tmp_path / "ws"
    mem.research("q", root=root)  # exports
    mem.add_message("a brand new message", role="user", timestamp=TS)
    result = mem.research("q", root=root, refresh=False)
    assert result.export_s == 0.0
    assert "brand new" not in (root / "messages" / "2026-09.md").read_text()
    mem.close()


def test_an_empty_store_exports_an_index_that_says_so(tmp_path):
    mem = _mem()
    stats = export_workspace(mem, tmp_path / "ws")
    assert stats.sessions == 0 and stats.runbooks == 0 and stats.messages == 0
    index = (tmp_path / "ws" / "INDEX.md").read_text()
    assert index.count("(none)") == 2
    mem.close()


def test_messages_are_split_by_month_oldest_first(tmp_path):
    from datetime import datetime as _dt

    mem = _mem()
    mem.add_message("august", role="user", timestamp=_dt(2026, 8, 3, tzinfo=UTC))
    mem.add_message("september", role="user", timestamp=TS)
    export_workspace(mem, tmp_path / "ws")
    assert (tmp_path / "ws" / "messages" / "2026-08.md").read_text().strip().endswith("august")
    assert (tmp_path / "ws" / "messages" / "2026-09.md").read_text().strip().endswith("september")
    mem.close()


def test_the_mcp_tool_success_path_serialises_citations_and_steps(tmp_path):
    from agmem.mcp import server

    cfg = AgmemConfig(sync_write=True, data_dir=tmp_path / "data")
    mem = _populated(
        AgenticMemory(
            namespace="t", organizers=["experience"], embedder=FakeEmbedder(dim=128), config=cfg
        )
    )
    mem.structured = StubLLM(
        {
            "explore": [
                {"action": "search", "reason": "r", "pattern": "idle-timeout"},
                _final("use --idle-timeout 2", [{"file": "runbooks/rb-1.md", "lines": [1, 2]}]),
            ]
        }
    )
    server._registry._mems["t"] = mem
    server._registry.default = "t"
    server._registry.config = cfg
    try:
        payload = json.loads(server.research_memory("q", namespace="t"))
    finally:
        server._registry._mems.pop("t", None)
        mem.close()
    assert payload["context"] == "use --idle-timeout 2"
    assert payload["citations"] == [{"file": "runbooks/rb-1.md", "lines": [1, 2]}]
    assert payload["steps"] == 2 and payload["llm_calls"] == 2
    assert payload["degraded"] is None and payload["latency_s"] >= payload["export_s"] >= 0.0


def test_cli_ask_runs_the_loop_against_a_stub_model_and_keeps_the_trace(tmp_path):
    """The one CLI path the bundle-C review listed as uncovered: `ask` with a
    real (stubbed) model role, in the CLI's own process. The stub answers with
    a `list` then a `final` citing a session file the export is known to
    write, and the full LLM I/O lands beside the store."""
    from helpers import openai_stub

    mem = _populated(
        AgenticMemory(
            namespace="main",
            organizers=["experience"],
            embedder=FakeEmbedder(),
            config=AgmemConfig(sync_write=True, data_dir=tmp_path / "data"),
        )
    )
    root = tmp_path / "data" / "main" / "workspace"
    export_workspace(mem, root)  # deterministic: `ask` re-exports to the same bytes
    mem.close()
    cited = min((root / "sessions").rglob("*.md"))
    rel = cited.relative_to(root).as_posix()
    n_lines = len(cited.read_text().splitlines())

    replies = [
        json.dumps({"action": "list", "reason": "see the layout", "path": "sessions"}),
        json.dumps(
            _final("The answer is in the session log.", [{"file": rel, "lines": [1, n_lines]}])
        ),
    ]
    with openai_stub(replies) as (url, requests):
        proc = _run_cli(
            ["ask", "what happened?"],
            tmp_path,
            extra_config=f'\n[llm.explore]\nendpoint = "{url}"\nmodel = "stub"\napi_key = "stub"\n',
        )
    assert proc.returncode == 0, proc.stderr
    assert len(requests) == 2 and all(r["model"] == "stub" for r in requests)
    assert "The answer is in the session log." in proc.stdout
    assert f'"file": "{rel}"' in proc.stdout
    assert "llm_calls=2 steps=2" in proc.stdout and "degraded=None" in proc.stdout

    trace_lines = [line for line in proc.stdout.splitlines() if line.startswith("trace: ")]
    assert len(trace_lines) == 1
    trace = Path(trace_lines[0].removeprefix("trace: "))
    assert trace.parent == tmp_path / "data" / "main" / "traces"
    assert trace.name.startswith("ask-")
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [r["role"] for r in records] == ["explore", "explore"]
    assert records[1]["response_text"] == replies[1]
