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
    render_transcript,
    validated_step_range,
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
    procedure note. `steps` is ours and has no upstream counterpart — it is the
    citation that lets a runbook point back at the persisted transcript."""
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
        "steps",
    }
    assert SCHEMA["properties"]["tasks"]["items"]["properties"]["steps"] == {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 1,
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
    assert text.startswith("[0] USER") and "…[steps 2-4 omitted]…" in text
    assert "[5] ASSISTANT" in text and "[3]" not in text  # whole steps, head and tail
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


def test_a_citation_into_omitted_steps_is_refused():
    """The model can only cite what it was shown. With the middle clipped out,
    a range that crosses the gap is an invention and must not become a pointer."""
    steps = _session().as_task_trajectory()
    text, visible = render_transcript(steps, max_chars=200)
    assert visible == frozenset({0, 1, 5})
    assert validated_step_range([0, 1], len(steps), visible) == [0, 1]
    assert validated_step_range([1, 5], len(steps), visible) is None
    assert validated_step_range([3, 3], len(steps), visible) is None
    # Without a visibility set the bounds check alone still applies.
    assert validated_step_range([0, 5], len(steps)) == [0, 5]

    # One step larger than the whole budget is shown clipped, and stays citable.
    huge = [{"kind": "user", "text": "x" * 5000}, {"kind": "assistant", "text": "ok"}]
    text, visible = render_transcript(huge, max_chars=300)
    assert text.startswith("[0] USER") and "chars omitted" in text
    assert text.endswith("[1] ASSISTANT\nok") and visible == frozenset({0, 1})


def test_the_prompt_requires_a_step_range_and_says_how_many_steps_there_are():
    """The 2026-09-04 smoke: two sessions, seven runbooks, `steps` on none of
    them — the prompt had said "omit the field if … unsure", and a 9B model
    took that every time, so every pointer fell back to the whole session. The
    field is now asked for as required, listed before the free-text lists, and
    the user turn states the label range so the model has the numbers to cite."""
    assert "steps: REQUIRED on every task" in SYSTEM_PROMPT
    assert "omit the field" not in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.index("- steps:") < SYSTEM_PROMPT.index("- preference_signals:")

    llm = StubLLM({"distill": [{"summary": "", "tasks": []}]})
    mem = _mem(llm)
    traj = _session()
    steps = traj.as_task_trajectory()
    mem.add_task_result(trajectory=steps, outcome="unknown", task=traj.task_text)
    mem.flush()
    prompt = llm.calls[0][1]
    assert f"- steps: {len(steps)}, labelled [0] to [{len(steps) - 1}]" in prompt
    mem.close()


def test_a_list_field_returned_as_a_string_is_kept_not_dropped():
    """The 2026-09-04 smoke, third call: qwen3.5-9b returned every list field
    but `procedure` as one string, and `_strings` turned "not a list" into
    "empty" — the stored runbook had a procedure and nothing else, and the
    call counted as a success. A string is one item per non-empty line."""
    reply = _distilled()
    task = reply["tasks"][0]
    task["preference_signals"] = 'when fixing a test, the user said "don\'t touch the hooks"'
    task["failures"] = "TimeoutExpired -> idle timeout too long -> use --idle-timeout 2\n\n"
    task["references"] = "uv run pytest tests/test_daemon.py -q\nTimeoutExpired"
    task["keywords"] = "test_daemon, idle-timeout, TimeoutExpired"  # one handle, not split
    llm = StubLLM({"distill": [reply]})
    mem = _mem(llm)
    traj = _session()
    mem.add_task_result(
        trajectory=traj.as_task_trajectory(), outcome="unknown", task=traj.task_text
    )
    mem.flush()
    (item,) = mem.doc_store.list_items(MEMORY_TYPE, namespace="t")
    assert item["preference_signals"] == [
        'when fixing a test, the user said "don\'t touch the hooks"'
    ]
    assert item["failures"] == ["TimeoutExpired -> idle timeout too long -> use --idle-timeout 2"]
    assert item["references"] == ["uv run pytest tests/test_daemon.py -q", "TimeoutExpired"]
    assert item["keywords"] == ["test_daemon, idle-timeout, TimeoutExpired"]
    assert "## Failures" in item["content"] and "TimeoutExpired" in item["embedding_text"]
    mem.close()


def test_an_enumeration_of_steps_is_a_citation_too():
    """2026-09-04: qwen3.5-9b cited `[11, 14, 17, …, 42]` — the steps it had
    read, all of them shown — and a `[first, last]`-only validator threw the
    citation away. An enumeration is accepted when every listed step is in
    bounds and visible; the range it spans is what the footer shows, and the
    pointers are the listed steps only."""
    from agmem.organizers.experience.organizer import cited_steps, source_episode_ids

    visible = frozenset({0, 1, 2, 3, 8, 9})
    assert cited_steps([1, 3, 8], 10, visible) == [1, 3, 8]
    assert cited_steps([8, 3, 1, 3], 10, visible) == [1, 3, 8]  # order and repeats do not matter
    assert cited_steps([1, 3, 5], 10, visible) is None  # 5 was omitted from the transcript
    assert cited_steps([1, 3, 12], 10, visible) is None  # out of bounds
    assert cited_steps([1, 3], 10, visible) == [1, 2, 3]  # a pair in order is a range
    assert cited_steps([3, 1], 10, visible) is None  # a pair out of order is a mistyped range
    assert cited_steps([2], 10, visible) == [2]
    assert cited_steps([], 10, visible) is None and cited_steps([True, 1], 10, visible) is None
    assert validated_step_range([1, 3, 8], 10, visible) == [1, 8]

    trajectory = [{"episode_id": f"e{i}"} for i in range(10)]
    assert source_episode_ids(trajectory, [1, 3, 8]) == ["e1", "e3", "e8"]
    assert source_episode_ids(trajectory, [1, 3]) == ["e1", "e2", "e3"]
    assert source_episode_ids(trajectory, None) == [f"e{i}" for i in range(10)]


def test_each_runbook_is_tagged_with_deterministic_labels():
    """`TAG` had no emitter (core/ops.py). The experience organizer now tags
    every runbook it adds with what is known without judgement: the block's
    outcome, the host and cwd, how many transcript steps it cites (0 = fell
    back to the whole session), and how many blocks the session yielded. The
    facade merges them into the item's `tags`, so they are queryable and in
    the evolution log."""
    reply = _distilled()
    reply["tasks"][0]["steps"] = [2, 4]
    llm = StubLLM({"distill": [reply]})
    mem = _mem(llm)
    traj = _session()
    mem.add_task_result(
        trajectory=traj.as_task_trajectory(), outcome="unknown", task=traj.task_text
    )
    mem.flush()
    ops = list(mem.log.tail(10))
    adds = [op for op in ops if op.op is OpType.ADD and op.target_type == MEMORY_TYPE]
    tags = [op for op in ops if op.op is OpType.TAG and op.target_type == MEMORY_TYPE]
    assert len(adds) == 1 and len(tags) == 1 and tags[0].target_id == adds[0].target_id
    assert ops.index(tags[0]) > ops.index(adds[0])  # the item exists when the tag lands
    (item,) = mem.doc_store.list_items(MEMORY_TYPE, namespace="t")
    assert item["tags"] == sorted(
        ["outcome:success", "host:claude-code", "cited:3", "tasks:1", "cwd:/home/u/proj"]
    )
    mem.close()
