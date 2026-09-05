"""`add_session` and `python -m agmem.sessions ingest`: a session's raw steps
land in the store, once, and the distillation runs against them.

The gap this closes is docs/research/agent-memory-axes-v1.md §7.1: the facade
kept only the task line, so there was no raw trajectory to read back and no
caller for `as_task_trajectory()` outside the MCP tool. Nothing here spends —
`FakeEmbedder` for the vectors, `StubLLM` for the one distill call.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import StubLLM

from agmem import AgenticMemory
from agmem.config import AgmemConfig
from agmem.core.ops import MemoryOp, OpType
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.base import Organizer
from agmem.organizers.experience.organizer import MEMORY_TYPE
from agmem.sessions import SessionTrajectory, Step

TS = "2026-09-02T01:00:00.000Z"


class RecordingOrganizer(Organizer):
    """Counts `on_task_end` calls and keeps the trajectory it was handed, so a
    test can assert that a re-ingest did NOT reach the distiller."""

    name = "recording"
    produces = ("runbooks",)

    def __init__(self) -> None:
        self.task_ends: list[tuple[list[dict], str, str]] = []

    def on_task_end(self, trajectory, outcome, task, ctx) -> list[MemoryOp]:
        self.task_ends.append((trajectory, outcome, task))
        return []


def _session(session_id: str = "sess-42") -> SessionTrajectory:
    traj = SessionTrajectory(id=session_id, host="claude-code", source_path="x", cwd="/home/u/proj")
    traj.steps = [
        Step(kind="user", text="Fix the flaky test in tests/test_daemon.py"),
        Step(kind="assistant", text="Looking."),
        Step(kind="tool_call", text='{"command": "uv run pytest -q"}', tool_name="Bash"),
        Step(
            kind="tool_result", text="1 failed: test_idle_timeout TimeoutExpired", tool_name="Bash"
        ),
        Step(kind="user", text="use --idle-timeout 2 in that test, not 30"),
    ]
    return traj


def _mem(organizer: Organizer) -> AgenticMemory:
    return AgenticMemory(
        namespace="t",
        organizers=[organizer],
        embedder=FakeEmbedder(dim=128),
        config=AgmemConfig(sync_write=True),
    )


def test_add_session_persists_every_step_and_they_are_searchable():
    org = RecordingOrganizer()
    mem = _mem(org)
    traj = _session()
    ingest = mem.add_session(traj, outcome="success")
    mem.flush()

    assert ingest.session_id == "sess-42" and ingest.host == "claude-code"
    assert ingest.already_ingested is False and ingest.dispatched is True
    assert ingest.episode_ids == [traj.episode_id(i) for i in range(len(traj.steps))]

    stored = mem.doc_store.get_episodes(list(ingest.episode_ids))
    assert len(stored) == len(traj.steps)
    by_id = {e.id: e for e in stored}
    for index, step in enumerate(traj.steps):
        episode = by_id[traj.episode_id(index)]
        assert episode.content == step.text
        assert episode.role == step.kind
        assert episode.meta["session_id"] == "sess-42"
        assert episode.meta["step_index"] == index
        assert episode.meta["source"] == "claude-code"

    bundle = mem.search("TimeoutExpired idle-timeout", memory_types=("episodic",), k=5)
    assert {s.item.id for s in bundle.items} <= set(ingest.episode_ids)
    assert "TimeoutExpired" in bundle.render()
    mem.close()


def test_add_session_dispatches_the_distiller_with_pointered_steps():
    org = RecordingOrganizer()
    mem = _mem(org)
    traj = _session()
    ingest = mem.add_session(traj, outcome="partial")
    mem.flush()

    assert len(org.task_ends) == 1
    steps, outcome, task = org.task_ends[0]
    assert outcome == "partial"
    assert task == "Fix the flaky test in tests/test_daemon.py"
    assert [s["episode_id"] for s in steps] == list(ingest.episode_ids)
    # The pointers resolve: persist ran before dispatch.
    assert len(mem.doc_store.get_episodes([s["episode_id"] for s in steps])) == len(steps)
    mem.close()


def test_add_session_does_not_fan_out_on_message():
    """Session steps are not conversation turns — see the method's docstring."""

    class MessageCounter(Organizer):
        name = "counter"

        def __init__(self) -> None:
            self.seen = 0

        def on_message(self, episode, ctx) -> list[MemoryOp]:
            self.seen += 1
            return []

    org = MessageCounter()
    mem = _mem(org)
    mem.add_session(_session())
    mem.flush()
    assert org.seen == 0
    mem.close()


def test_reingesting_the_same_session_is_a_no_op_until_forced():
    org = RecordingOrganizer()
    mem = _mem(org)
    traj = _session()
    first = mem.add_session(traj)
    mem.flush()
    before = mem.doc_store.count_episodes()

    second = mem.add_session(_session())
    mem.flush()
    assert second.already_ingested is True and second.dispatched is False
    assert mem.doc_store.count_episodes() == before
    assert len(org.task_ends) == 1  # the distiller was not paid twice

    forced = mem.add_session(_session(), force=True)
    mem.flush()
    assert forced.already_ingested is True and forced.dispatched is True
    assert forced.episode_ids == first.episode_ids
    assert mem.doc_store.count_episodes() == before  # INSERT OR REPLACE, not duplicates
    assert len(org.task_ends) == 2
    mem.close()


def test_persist_and_distill_can_each_be_turned_off():
    org = RecordingOrganizer()
    mem = _mem(org)
    ingest = mem.add_session(_session(), distill=False)
    mem.flush()
    assert ingest.dispatched is False
    assert mem.doc_store.count_episodes() == len(_session().steps)
    assert org.task_ends == []

    org2 = RecordingOrganizer()
    mem2 = _mem(org2)
    ingest2 = mem2.add_session(_session("other"), persist_steps=False)
    mem2.flush()
    assert ingest2.episode_ids == []
    assert mem2.doc_store.count_episodes() == 0
    assert len(org2.task_ends) == 1
    # Nothing was persisted, so the steps must not carry pointers to episodes
    # that do not exist — an organizer would copy them into a runbook as if
    # they resolved.
    assert all("episode_id" not in step for step in org2.task_ends[0][0])
    mem.close()
    mem2.close()


def test_add_task_result_strips_pointers_it_did_not_persist():
    org = RecordingOrganizer()
    mem = _mem(org)
    traj = _session()
    mem.add_task_result(traj.as_task_trajectory(), outcome="unknown", task=traj.task_text)
    mem.flush()
    assert len(org.task_ends) == 1
    steps = org.task_ends[0][0]
    assert all("episode_id" not in step for step in steps)
    assert steps[0]["session_id"] == "sess-42"  # the rest of the step is untouched
    assert mem.doc_store.count_episodes() == 1  # only the task line, as before
    mem.close()


# --------------------------------------------------------------------------- pointers


def _distilled(steps: list[int] | None) -> dict:
    task: dict = {
        "name": "Fix the flaky test",
        "outcome": "success",
        "reusable_knowledge": ["--idle-timeout 30 exceeds the test timeout"],
        "keywords": ["test_daemon"],
    }
    if steps is not None:
        task["steps"] = steps
    return {"summary": "The user asked to fix a flaky test.", "tasks": [task]}


def _experience_mem(llm) -> AgenticMemory:
    from agmem.organizers.experience import ExperienceOrganizer

    mem = AgenticMemory(
        namespace="t",
        organizers=[ExperienceOrganizer()],
        embedder=FakeEmbedder(dim=128),
        config=AgmemConfig(sync_write=True),
    )
    mem.structured = llm
    mem._ctx.llm = llm
    return mem


def test_runbook_carries_the_step_range_and_the_episode_ids_it_is_grounded_in():
    llm = StubLLM({"distill": [_distilled([2, 4])]})
    mem = _experience_mem(llm)
    traj = _session()
    ingest = mem.add_session(traj)
    mem.flush()

    adds = [op for op in mem.log.tail(50) if op.op is OpType.ADD and op.target_type == MEMORY_TYPE]
    assert len(adds) == 1
    payload = adds[0].payload
    assert payload["step_range"] == [2, 4]
    assert payload["source_episode_ids"] == [traj.episode_id(i) for i in (2, 3, 4)]
    assert set(payload["source_episode_ids"]) <= set(ingest.episode_ids)
    assert "source: claude-code session sess-42 steps 2-4" in payload["content"]
    json.dumps(payload)
    mem.close()


def test_a_missing_or_impossible_step_range_falls_back_to_the_whole_session():
    for steps in (None, [9, 99], [4, 2], ["a", "b"]):
        llm = StubLLM({"distill": [_distilled(steps)]})
        mem = _experience_mem(llm)
        traj = _session()
        mem.add_session(traj)
        mem.flush()
        adds = [
            op for op in mem.log.tail(50) if op.op is OpType.ADD and op.target_type == MEMORY_TYPE
        ]
        payload = adds[0].payload
        assert payload["step_range"] is None, steps
        assert payload["source_episode_ids"] == [
            traj.episode_id(i) for i in range(len(traj.steps))
        ], steps
        mem.close()


def test_an_interrupted_ingest_is_not_sealed_as_complete():
    """Only the first step present means an earlier run died mid-loop; the next
    pass must re-persist and distil, not report `already ingested`."""
    org = RecordingOrganizer()
    mem = _mem(org)
    traj = _session()
    first_only = traj.to_episodes("t")[:1]
    mem.doc_store.add_episode(first_only[0])  # the state a crash after batch 1 leaves
    ingest = mem.add_session(traj)
    mem.flush()
    assert ingest.already_ingested is False and ingest.dispatched is True
    assert mem.doc_store.count_episodes() == len(traj.steps)
    assert len(org.task_ends) == 1
    # And now it IS complete, so the next pass is the no-op.
    again = mem.add_session(traj)
    assert again.already_ingested is True and len(org.task_ends) == 1
    mem.close()


def test_force_replaces_the_earlier_runbook_instead_of_adding_a_second():
    llm = StubLLM({"distill": [_distilled([2, 4]), _distilled([2, 4])]})
    mem = _experience_mem(llm)
    traj = _session()
    mem.add_session(traj)
    mem.flush()
    first = mem.doc_store.list_items(MEMORY_TYPE, namespace="t")
    assert len(first) == 1

    mem.add_session(traj, force=True)
    mem.flush()
    after = mem.doc_store.list_items(MEMORY_TYPE, namespace="t")
    assert len(after) == 1, "a re-distillation must retire the runbook it replaces"
    assert after[0]["id"] != first[0]["id"]
    ops = list(mem.log.tail(20))
    deletes = [op for op in ops if op.op is OpType.DELETE and op.target_type == MEMORY_TYPE]
    assert [op.target_id for op in deletes] == [first[0]["id"]]
    assert deletes[0].payload["reason"] == "re-distillation"
    mem.close()


def test_steps_without_pointers_still_distil():
    """`add_task_result` callers (FiNER, the MCP tool) pass step dicts with no
    `episode_id`; the payload then carries an empty list, not a crash."""
    llm = StubLLM({"distill": [_distilled([0, 1])]})
    mem = _experience_mem(llm)
    steps = [{"kind": "user", "text": "hi", "host": "claude-code", "session_id": "s"}]
    mem.add_task_result(trajectory=steps, outcome="success", task="hi")
    mem.flush()
    adds = [op for op in mem.log.tail(50) if op.op is OpType.ADD and op.target_type == MEMORY_TYPE]
    assert adds[0].payload["source_episode_ids"] == []
    mem.close()


# --------------------------------------------------------------------------- CLI


def _claude_session_file(tmp_path):
    """One synthetic Claude Code session file, in the shape tests/test_sessions.py
    builds (field names observed on 2026-09-02)."""
    records = [
        {
            "type": "user",
            "uuid": "u1",
            "isSidechain": False,
            "timestamp": TS,
            "cwd": "/home/u/proj",
            "sessionId": "cli-1",
            "gitBranch": "main",
            "message": {"role": "user", "content": "Fix the failing test"},
        },
        {
            "type": "assistant",
            "uuid": "u2",
            "isSidechain": False,
            "timestamp": TS,
            "cwd": "/home/u/proj",
            "sessionId": "cli-1",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Fixed."}]},
        },
    ]
    path = tmp_path / "cli-1.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def _run_cli(args, tmp_path, home=None, extra_config: str = ""):
    """The CLI in its own process, pointed at a config that overrides the
    embedder slot to `FakeEmbedder` — the real one would fetch weights, and this
    test must neither download nor spend. `extra_config` is appended verbatim
    (an `[llm.distill]` section pointing at `openai_stub`, for the paid path)."""
    import os

    config = tmp_path / "agmem.toml"
    config.write_text(
        '[profile]\nname = "lite"\n\n'
        f'[storage]\ndata_dir = "{tmp_path / "data"}"\n\n'
        '[override]\nembedder = "FakeEmbedder"\n' + extra_config
    )
    env = dict(os.environ)
    env["AGMEM_CONFIG"] = str(config)
    if home is not None:
        env["HOME"] = str(home)
    env.pop("AGMEM_DATA_DIR", None)
    env.pop("AGMEM_NAMESPACE", None)
    return subprocess.run(
        [sys.executable, "-m", "agmem.sessions", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        check=False,
    )


def test_cli_dry_run_prints_the_plan_and_writes_nothing(tmp_path):
    path = _claude_session_file(tmp_path)
    proc = _run_cli(["ingest", str(path), "--dry-run"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "cli-1" in proc.stdout and "steps=    2" in proc.stdout
    assert "would ingest" in proc.stdout

    from agmem.stores.sqlite_doc import SqliteDocStore

    store = SqliteDocStore(tmp_path / "data" / "main" / "memory.db")
    assert store.count_episodes() == 0  # a dry run opens the store and writes nothing
    store.close()


def test_cli_no_distill_persists_the_raw_steps(tmp_path):
    path = _claude_session_file(tmp_path)
    proc = _run_cli(["ingest", str(path), "--no-distill", "--namespace", "cli"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "persisted=2" in proc.stdout

    from agmem.stores.sqlite_doc import SqliteDocStore

    store = SqliteDocStore(tmp_path / "data" / "cli" / "memory.db")
    assert store.count_episodes() == 2
    store.close()

    # Second run over the same file re-reads it and writes nothing new.
    again = _run_cli(["ingest", str(path), "--no-distill", "--namespace", "cli"], tmp_path)
    assert again.returncode == 0
    assert "already ingested" in again.stdout


def test_cli_refuses_to_distil_without_a_limit_or_an_llm(tmp_path):
    path = _claude_session_file(tmp_path)
    no_limit = _run_cli(["ingest", str(path)], tmp_path)
    assert no_limit.returncode == 2
    assert "--limit" in (no_limit.stderr + no_limit.stdout)

    too_many = _run_cli(["ingest", str(path), "--limit", "21"], tmp_path)
    assert too_many.returncode == 2
    assert "--limit" in (too_many.stderr + too_many.stdout)

    no_llm = _run_cli(["ingest", str(path), "--limit", "1"], tmp_path)
    assert no_llm.returncode == 2
    assert "no LLM" in (no_llm.stderr + no_llm.stdout)


def test_cli_rejects_a_negative_limit_instead_of_dropping_sessions(tmp_path):
    path = _claude_session_file(tmp_path)
    proc = _run_cli(["ingest", str(path), "--no-distill", "--limit", "-1"], tmp_path)
    assert proc.returncode == 2
    assert "non-negative" in proc.stderr


def _second_session_file(tmp_path):
    records = [
        {
            "type": "user",
            "uuid": "v1",
            "isSidechain": False,
            "timestamp": TS,
            "cwd": "/home/u/proj",
            "sessionId": "cli-2",
            "gitBranch": "main",
            "message": {"role": "user", "content": "Add a changelog entry"},
        },
        {
            "type": "assistant",
            "uuid": "v2",
            "isSidechain": False,
            "timestamp": TS,
            "cwd": "/home/u/proj",
            "sessionId": "cli-2",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Added."}]},
        },
    ]
    path = tmp_path / "cli-2.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_cli_limit_counts_sessions_taken_in_not_files_seen(tmp_path):
    """A backfill run in slices must advance: sessions already in the store do
    not consume the limit, so `--limit 1` over [ingested, new] takes the new one."""
    first = _claude_session_file(tmp_path)
    second = _second_session_file(tmp_path)
    ns = ["--namespace", "cli"]
    assert _run_cli(["ingest", str(first), "--no-distill", *ns], tmp_path).returncode == 0
    proc = _run_cli(
        ["ingest", str(first), str(second), "--no-distill", "--limit", "1", *ns], tmp_path
    )
    assert proc.returncode == 0, proc.stderr
    assert "cli-1" in proc.stdout and "already ingested" in proc.stdout
    assert "cli-2" in proc.stdout and "persisted=2" in proc.stdout
    assert "1 session(s) ingested this run" in proc.stdout


@pytest.mark.parametrize("host", ["claude-code", "codex", "all"])
def test_cli_discovers_sessions_the_way_scan_does(tmp_path, host):
    """Discovery under an empty HOME finds nothing and says so, rather than
    reaching for this machine's real sessions."""
    home = tmp_path / "home"
    home.mkdir()
    proc = _run_cli(["ingest", "--host", host, "--dry-run"], tmp_path, home=home)
    assert proc.returncode == 0, proc.stderr
    assert "0 session(s)" in proc.stdout


def _empty_session_file(tmp_path):
    """A session file the adapter reads as zero steps — the shape the 2026-09-04
    smoke met (a file holding only a summary record)."""
    from agmem.sessions import load

    path = tmp_path / "cli-empty.jsonl"
    path.write_text(json.dumps({"type": "summary", "summary": "nothing typed"}) + "\n")
    assert load(path).steps == []  # the precondition the test is about
    return path


def test_cli_skips_an_empty_session_without_counting_it_against_the_limit(tmp_path):
    """The 2026-09-04 smoke ran `--limit 2` over [big, empty, big] and took only
    the first big one: the empty session stored nothing, called nothing, and
    still consumed a slot. The dry run had (correctly) skipped it. Both paths
    now print the same line and count the same way."""
    empty = _empty_session_file(tmp_path)
    first = _claude_session_file(tmp_path)
    proc = _run_cli(
        ["ingest", str(empty), str(first), "--no-distill", "--limit", "1", "--namespace", "cli"],
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "empty, skipped" in proc.stdout
    assert "cli-1" in proc.stdout and "persisted=2" in proc.stdout
    assert "1 session(s) ingested this run" in proc.stdout


def _distill_config(url: str) -> str:
    return f'\n[llm.distill]\nendpoint = "{url}"\nmodel = "stub"\napi_key = "stub"\n'


def test_cli_distil_keeps_the_full_llm_trace_beside_the_store(tmp_path):
    """The paid path end to end, in the CLI's own process, against a local
    OpenAI-shaped stub: the model call happens, the runbook lands, and the full
    prompt/response trace is written beside the store by default (or where
    `--trace` says). The 2026-09-04 smoke had no trace, so its `calls: 2` for
    one session could not be explained."""
    from helpers import openai_stub

    from agmem.stores.sqlite_doc import SqliteDocStore

    def runbooks() -> int:
        # The store API, not a raw row count: a DELETE is a tombstone the
        # readers skip, and the row stays for replay.
        store = SqliteDocStore(tmp_path / "data" / "cli" / "memory.db")
        try:
            return len(store.list_items("runbooks", namespace="cli"))
        finally:
            store.close()

    path = _claude_session_file(tmp_path)
    reply = json.dumps(
        {
            "summary": "Fixed a failing test",
            "tasks": [
                {
                    "name": "Fix the failing test",
                    "outcome": "success",
                    "procedure": ["run pytest", "fix the assertion"],
                    "keywords": ["pytest"],
                    "steps": [0, 1],
                }
            ],
        }
    )
    with openai_stub([reply]) as (url, requests):
        proc = _run_cli(
            ["ingest", str(path), "--limit", "1", "--namespace", "cli"],
            tmp_path,
            extra_config=_distill_config(url),
        )
    assert proc.returncode == 0, proc.stderr
    assert len(requests) == 1 and requests[0]["model"] == "stub"
    assert "persisted=2" in proc.stdout and "'calls': 1" in proc.stdout

    trace_lines = [line for line in proc.stdout.splitlines() if line.startswith("trace: ")]
    assert len(trace_lines) == 1
    trace = Path(trace_lines[0].removeprefix("trace: "))
    assert trace.parent == tmp_path / "data" / "cli" / "traces"
    assert trace.name.startswith("ingest-") and trace.suffix == ".jsonl"
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["role"] == "distill" and records[0]["response_text"] == reply
    assert "Fix the failing test" in records[0]["messages"][-1]["content"]

    assert runbooks() == 1

    # An explicit `--trace` wins over the default location, and `--force` really
    # re-distils: one more model call, one more trace line, still one runbook.
    explicit = tmp_path / "elsewhere" / "run.jsonl"
    with openai_stub([reply]) as (url, requests):
        again = _run_cli(
            [
                "ingest",
                str(path),
                "--limit",
                "1",
                "--namespace",
                "cli",
                "--force",
                "--trace",
                str(explicit),
            ],
            tmp_path,
            extra_config=_distill_config(url),
        )
    assert again.returncode == 0, again.stderr
    assert len(requests) == 1
    assert f"trace: {explicit}" in again.stdout
    assert len(explicit.read_text().splitlines()) == 1
    assert runbooks() == 1


def test_force_keeps_the_earlier_runbook_when_the_redistillation_drops(caplog):
    """The 2026-09-04 smoke: `--force` retired the session's four runbooks,
    then both replies of the new call were malformed and dropped, and the
    session was left with none. The earlier items go only once the new call
    has put something in their place."""
    import logging

    llm = StubLLM({"distill": [_distilled([2, 4])]})  # the second call has no reply: a drop
    mem = _experience_mem(llm)
    traj = _session()
    mem.add_session(traj)
    mem.flush()
    (first,) = mem.doc_store.list_items(MEMORY_TYPE, namespace="t")

    with caplog.at_level(logging.WARNING, logger="agmem.memory"):
        mem.add_session(traj, force=True)
        mem.flush()
    (kept,) = mem.doc_store.list_items(MEMORY_TYPE, namespace="t")
    assert kept["id"] == first["id"]
    ops = list(mem.log.tail(20))
    assert not [op for op in ops if op.op is OpType.DELETE and op.target_type == MEMORY_TYPE]
    assert any("produced nothing" in r.getMessage() for r in caplog.records)
    mem.close()


def test_session_admission_refuses_before_anything_is_stored_or_called():
    """Session-level admission: the place for "is this session worth remembering
    at all", which the message-level gates never had (research §7.1). A
    refused session leaves no episodes, makes no call, and says why."""
    from agmem.sessions import SessionAdmission

    policy = SessionAdmission(min_user_turns=1, min_steps=2)
    traj = _session()
    assert policy(traj) is None
    tool_only = SessionTrajectory(id="s-tools", host="claude-code", source_path="x", steps=[
        Step(kind="tool_call", text="ls", tool_name="Bash", timestamp=TS),
        Step(kind="tool_result", text="a b", tool_name="Bash", timestamp=TS),
    ])  # fmt: skip
    assert policy(tool_only) == "0 user turn(s) < min_user_turns 1"
    assert SessionAdmission(min_steps=10)(traj) == f"{len(traj.steps)} step(s) < min_steps 10"

    llm = StubLLM({"distill": [_distilled([2, 4])]})
    mem = _experience_mem(llm)
    ingest = mem.add_session(tool_only, admit=policy)
    mem.flush()
    assert not ingest.admitted and ingest.reason.startswith("0 user turn(s)")
    assert ingest.episode_ids == [] and not ingest.dispatched
    assert mem.doc_store.count_episodes() == 0 and llm.calls == []
    admitted = mem.add_session(traj, admit=policy)
    assert (
        admitted.admitted
        and admitted.reason is None
        and len(admitted.episode_ids) == len(traj.steps)
    )
    mem.close()


def test_cli_admission_flags_refuse_without_consuming_the_limit(tmp_path):
    first = _claude_session_file(tmp_path)  # one user turn, two steps
    second = _second_session_file(tmp_path)
    proc = _run_cli(
        ["ingest", str(first), str(second), "--no-distill", "--limit", "1", "--namespace", "cli",
         "--min-user-turns", "2"],
        tmp_path,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("refused (1 user turn(s) < min_user_turns 2)") == 2
    assert "0 session(s) ingested this run" in proc.stdout
    ok = _run_cli(["ingest", str(first), "--no-distill", "--namespace", "cli"], tmp_path)
    assert "persisted=2" in ok.stdout  # the default floors admit it


def test_a_runbook_hit_carries_the_transcript_steps_it_cites():
    """The runbook read step (research §2.2: there was none): the top runbook
    hits get the steps they cite attached as source messages, clipped, so the
    reader sees the evidence beside the procedure. A runbook that fell back to
    the whole session attaches nothing."""
    from agmem.retrieval.steps import AttachCitedSteps

    llm = StubLLM({"distill": [_distilled([2, 4])]})
    mem = _experience_mem(llm)
    traj = _session()
    mem.add_session(traj)
    mem.flush()
    assert isinstance(mem.pipeline.read_steps.get("runbooks"), AttachCitedSteps)
    bundle = mem.search("idle timeout", memory_types=["runbooks"], k=3)
    (hit,) = bundle.items
    messages = hit.item.data["_source_messages"]
    assert [m.split(" ", 1)[0] for m in messages] == ["[2]", "[3]", "[4]"]
    assert messages[0].startswith("[2] ") and traj.steps[2].text[:20] in messages[0]
    assert "Source Messages:" in hit.item.render()
    mem.close()

    whole = StubLLM({"distill": [_distilled(None)]})
    mem = _experience_mem(whole)
    mem.add_session(_session())
    mem.flush()
    (hit,) = mem.search("idle timeout", memory_types=["runbooks"], k=3).items
    assert "_source_messages" not in hit.item.data
    mem.close()


def _session_in(
    cwd: str, session_id: str, ended: str, task: str = "Fix the flaky test"
) -> SessionTrajectory:
    from datetime import UTC, datetime

    steps = [
        Step(kind="user", text=task, timestamp=datetime.fromisoformat(ended).replace(tzinfo=UTC)),
        Step(kind="tool_result", text="1 failed: test_idle_timeout TimeoutExpired", tool_name="Bash",
             timestamp=datetime.fromisoformat(ended).replace(tzinfo=UTC)),
    ]  # fmt: skip
    return SessionTrajectory(
        id=session_id, host="claude-code", source_path="x", cwd=cwd, git_branch="main", steps=steps
    )


def test_every_item_written_from_a_session_carries_its_origin():
    """Origin binding at write time (research §6 #8): host, session, project,
    branch and the session's clock on every episode and on the runbook, in
    one shape, so gating and freshness read the same record."""
    llm = StubLLM({"distill": [_distilled([0, 1])]})
    mem = _experience_mem(llm)
    traj = _session_in("/home/u/proj", "s-origin", "2026-09-05T10:00:00")
    mem.add_session(traj)
    mem.flush()
    (episode,) = mem.doc_store.get_episodes([traj.episode_id(0)])
    assert episode.meta["cwd"] == "/home/u/proj" and episode.meta["git_branch"] == "main"
    assert episode.meta["session_ended_at"] == "2026-09-05T10:00:00+00:00"
    (runbook,) = mem.doc_store.list_items(MEMORY_TYPE, namespace="t")
    assert runbook["origin"] == {
        "host": "claude-code", "session_id": "s-origin", "cwd": "/home/u/proj", "git_branch": "main",
        "started_at": "2026-09-05T10:00:00+00:00", "ended_at": "2026-09-05T10:00:00+00:00",
    }  # fmt: skip
    mem.close()


def test_search_gated_by_project_drops_what_another_project_wrote():
    """Cross-project leakage gating at read time (research §6 #9): with
    `project=`, items whose origin cwd is a different tree are not served;
    the same tree (equal, or nested either way) and items with no cwd pass."""
    llm = StubLLM({"distill": [_distilled([0, 1]), _distilled([0, 1])]})
    mem = _experience_mem(llm)
    mem.add_session(_session_in("/home/u/proj-a", "s-a", "2026-09-05T10:00:00"))
    mem.add_session(_session_in("/home/u/proj-b", "s-b", "2026-09-05T11:00:00"))
    mem.add_task_result(
        trajectory=[{"kind": "user", "text": "flaky test elsewhere"}],
        outcome="unknown",
        task="no cwd",
    )
    mem.flush()
    metrics: dict = {}
    everything = mem.search("flaky test", k=20, memory_types=["episodic", MEMORY_TYPE])
    gated = mem.search(
        "flaky test",
        k=20,
        memory_types=["episodic", MEMORY_TYPE],
        project="/home/u/proj-a/sub",
        metrics=metrics,
    )
    cwds = {
        s.item.meta.get("cwd") if hasattr(s.item, "meta") else s.item.data.get("cwd")
        for s in gated.items
    }
    assert cwds <= {"/home/u/proj-a", None}
    assert "/home/u/proj-b" in {
        s.item.meta.get("cwd") if hasattr(s.item, "meta") else s.item.data.get("cwd")
        for s in everything.items
    }
    assert metrics["project_gated"] >= 1 and len(gated.items) < len(everything.items)
    assert (
        len(mem.search("flaky test", k=20, project="/somewhere/else").items) <= 1
    )  # only the cwd-less task line, if ranked
    mem.close()


def test_a_newer_session_supersedes_the_same_task_in_the_same_project():
    """Freshness by deterministic signal (research §6 #7): a runbook about the
    same task in the same project from a newer session INVALIDATEs the older
    one — session clock, not a model. Different projects coexist."""
    llm = StubLLM({"distill": [_distilled([0, 1]), _distilled([0, 1]), _distilled([0, 1])]})
    mem = _experience_mem(llm)
    mem.add_session(_session_in("/home/u/proj", "s-old", "2026-09-01T10:00:00"))
    mem.flush()
    (old,) = mem.doc_store.list_items(MEMORY_TYPE, namespace="t")
    mem.add_session(_session_in("/home/u/proj", "s-new", "2026-09-05T10:00:00"))
    mem.flush()
    live = mem.doc_store.list_items(MEMORY_TYPE, namespace="t")
    by_session = {i["session_id"]: i for i in live}
    assert by_session["s-old"]["invalid_at"] == "2026-09-05T10:00:00+00:00"
    assert by_session["s-old"]["superseded_by"] == by_session["s-new"]["id"]
    assert "invalid_at" not in by_session["s-new"]
    served = {
        s.item.data.get("session_id")
        for s in mem.search("flaky", k=5, memory_types=[MEMORY_TYPE]).items
    }
    assert served == {"s-new"}
    ops = [op for op in mem.log.tail(30) if op.op is OpType.INVALIDATE]
    assert (
        ops
        and ops[-1].target_id == old["id"]
        and ops[-1].payload["reason"] == "newer session, same project and task"
    )
    # another project, same task name: nothing superseded
    mem.add_session(_session_in("/home/u/other", "s-other", "2026-09-06T10:00:00"))
    mem.flush()
    assert (
        "invalid_at"
        not in {i["session_id"]: i for i in mem.doc_store.list_items(MEMORY_TYPE, namespace="t")}[
            "s-new"
        ]
    )
    mem.close()
