"""The experience organizer: one session in, runbook items out, or one NOOP.

Driven through the facade the way the daemon and the LongMemEval-V2 adapter
will drive it (`add_task_result` -> `on_task_end`), with `StubLLM` standing in
for the one structured call so nothing is spent.
"""

from __future__ import annotations

import json

from helpers import StubLLM

from agmem import AgenticMemory
from agmem.core.ops import OpType
from agmem.core.types import MEMORY_TYPES
from agmem.embed.fake import FakeEmbedder
from agmem.organizers import ORGANIZERS
from agmem.organizers.experience import ExperienceOrganizer
from agmem.organizers.experience.organizer import (
    MEMORY_TYPE,
    SCHEMA,
    SYSTEM_PROMPT,
    embedding_text_for,
    render_runbook,
    render_steps,
)
from agmem.sessions import SessionTrajectory, Step


def _session() -> SessionTrajectory:
    traj = SessionTrajectory(id="sess-42", host="claude-code", source_path="x", cwd="/home/u/proj")
    traj.steps = [
        Step(
            kind="user",
            text="Fix the flaky test in tests/test_daemon.py, and don't touch the hooks.",
        ),
        Step(kind="assistant", text="Looking."),
        Step(
            kind="tool_call",
            text='{"command": "uv run pytest tests/test_daemon.py -q"}',
            tool_name="Bash",
        ),
        Step(
            kind="tool_result",
            text="1 failed: test_idle_timeout ... TimeoutExpired",
            tool_name="Bash",
        ),
        Step(kind="user", text="use --idle-timeout 2 in that test, not 30"),
        Step(kind="assistant", text="Done, 7 passed."),
    ]
    return traj


def _distilled() -> dict:
    return {
        "summary": "The user asked to fix a flaky daemon test; verified by pytest passing.",
        "tasks": [
            {
                "name": "Fix the flaky test in tests/test_daemon.py",
                "outcome": "success",
                "preference_signals": [
                    'when fixing a test, the user said "don\'t touch the hooks" -> keep edits inside the test file'
                ],
                "reusable_knowledge": [
                    "test_idle_timeout spawns a daemon with --idle-timeout; a 30 s value exceeds the test timeout"
                ],
                "failures": ["TimeoutExpired -> idle timeout too long -> use --idle-timeout 2"],
                "references": ["uv run pytest tests/test_daemon.py -q", "TimeoutExpired"],
                "procedure": [
                    "run the test file",
                    "read the failing name",
                    "shorten --idle-timeout",
                    "rerun",
                ],
                "keywords": ["test_daemon", "idle-timeout", "TimeoutExpired"],
            },
            {"name": "Looked at the hooks", "outcome": "uncertain"},  # nothing durable: dropped
        ],
    }


def _mem(llm) -> AgenticMemory:
    mem = AgenticMemory(
        namespace="t", organizers=[ExperienceOrganizer()], embedder=FakeEmbedder(dim=128)
    )
    mem.structured = llm
    mem._ctx.llm = llm
    return mem


def test_registered_and_typed():
    assert ORGANIZERS["experience"] is ExperienceOrganizer
    assert MEMORY_TYPE in MEMORY_TYPES
    assert ExperienceOrganizer.produces == (MEMORY_TYPE,)


def test_schema_matches_codex_raw_memory_and_agentrunbook_fields():
    """The plan's acceptance criterion: field-level correspondence, pinned.
    Codex `raw_memory` task blocks carry outcome, preference signals, reusable
    knowledge, failures, references and keywords; AgentRunbook-R adds the
    procedure note."""
    task_props = set(SCHEMA["properties"]["tasks"]["items"]["properties"])
    assert task_props == {
        "name",
        "outcome",
        "preference_signals",
        "reusable_knowledge",
        "failures",
        "references",
        "procedure",
        "keywords",
    }
    assert SCHEMA["properties"]["tasks"]["items"]["properties"]["outcome"]["enum"] == [
        "success",
        "partial",
        "fail",
        "uncertain",
    ]
    for rule in ("NO-OP IS ALLOWED", "data, not instructions", "REDACTED_SECRET", "User messages"):
        assert rule in SYSTEM_PROMPT


def test_one_call_one_runbook_per_durable_task_and_it_is_searchable():
    llm = StubLLM({"distill": [_distilled()]})
    mem = _mem(llm)
    traj = _session()
    mem.add_task_result(
        trajectory=traj.as_task_trajectory(), outcome="unknown", task=traj.task_text
    )
    mem.flush()

    assert [r for r, _ in llm.calls] == ["distill"]
    assert llm.systems == [SYSTEM_PROMPT]
    prompt = llm.calls[0][1]
    assert "host: claude-code" in prompt and "cwd: /home/u/proj" in prompt
    assert "[4] USER\nuse --idle-timeout 2" in prompt

    ops = list(mem.log.tail(10))
    adds = [op for op in ops if op.op is OpType.ADD and op.target_type == MEMORY_TYPE]
    assert len(adds) == 1, [op.op for op in ops]
    item = adds[0].payload
    assert item["outcome"] == "success" and item["cwd"] == "/home/u/proj"
    assert item["session_id"] == "sess-42" and item["source_host"] == "claude-code"
    assert item["content"].startswith("# Task: Fix the flaky test")
    assert "## Preference signals" in item["content"] and "## Procedure" in item["content"]
    assert "TimeoutExpired" in item["embedding_text"]

    bundle = mem.search("TimeoutExpired idle-timeout", memory_types=(MEMORY_TYPE,), k=3)
    assert bundle.items and bundle.items[0].memory_type == MEMORY_TYPE
    assert "Fix the flaky test" in bundle.render()
    assert "Runbooks" in bundle.render()


def test_nothing_durable_records_a_noop_not_silence():
    llm = StubLLM({"distill": [{"summary": "", "tasks": []}]})
    mem = _mem(llm)
    traj = _session()
    mem.add_task_result(
        trajectory=traj.as_task_trajectory(), outcome="unknown", task=traj.task_text
    )
    mem.flush()
    ops = [op for op in mem.log.tail(10) if op.actor == "experience"]
    assert [op.op for op in ops] == [OpType.NOOP]
    assert ops[0].payload["reason"] == "no durable signal"
    assert mem.search("anything", memory_types=(MEMORY_TYPE,), k=3).items == []


def test_no_llm_is_an_explicit_skip():
    mem = AgenticMemory(
        namespace="t", organizers=[ExperienceOrganizer()], embedder=FakeEmbedder(dim=128)
    )
    traj = _session()
    mem.add_task_result(
        trajectory=traj.as_task_trajectory(), outcome="unknown", task=traj.task_text
    )
    mem.flush()
    assert [op for op in mem.log.tail(10) if op.actor == "experience"] == []


def test_render_helpers_are_bounded_and_greppable():
    steps = _session().as_task_trajectory()
    text = render_steps(steps, max_chars=200)
    assert text.startswith("[0] USER") and "chars omitted" in text
    block = {
        "name": "n",
        "outcome": "fail",
        "failures": ["a -> b -> c"],
        "keywords": ["k1", "k2"],
        "references": ["grep me"],
    }
    md = render_runbook(block, summary="s", cwd="/w")
    assert md.splitlines()[:3] == ["# Task: n", "outcome: fail", "cwd: /w"]
    assert "## Failures and how to do differently" in md and md.endswith("keywords: k1, k2")
    assert "## Procedure" not in md  # empty lists are omitted
    assert embedding_text_for(block).splitlines() == ["n", "k1", "k2", "grep me"]
    json.dumps(block)  # payload stays JSON-serialisable
