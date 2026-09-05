"""The Claude Code hooks, driven the way the harness drives them.

A hook fails silently by construction — it must never break the session, so
every error path exits 0 with no output. That safety property is also what makes
a broken hook invisible, so these drive the real entry points as subprocesses
with real stdin payloads and assert on the contract, not on internals.

Hermetic since 2026-09-02: `AGMEM_CONFIG` points at a temp `agmem.toml` that
forces `FakeEmbedder`, the same seam `tests/test_mcp_server.py` has used since
it was written. Before that the hooks pinned the lite profile with no way in,
and because `HOME` is a temp dir here (so the hooks cannot touch a real
`~/.agmem`), the HuggingFace cache was invisible too — every run of this file
downloaded the 471 MB default model into the throwaway HOME, and on 2026-09-02
that took longer than the 120 s subprocess timeout.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

FAKE_EMBEDDER_TOML = '[profile]\nname = "lite"\n\n[override]\nembedder = "FakeEmbedder"\n'


def _env(tmp_path, extra_env: dict | None = None) -> dict:
    cfg = tmp_path / "agmem.toml"
    if not cfg.exists():
        cfg.write_text(FAKE_EMBEDDER_TOML)
    return {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
        "AGMEM_DATA_DIR": str(tmp_path / "data"),
        "AGMEM_NAMESPACE": "hooktest",
        "AGMEM_CONFIG": str(cfg),
        # No daemon in this file: these tests pin the in-process contract, and
        # a hook that spawned one would leave a 30-minute process behind.
        "AGMEM_NO_DAEMON": "1",
        "AGMEM_DAEMON_URL": "http://127.0.0.1:1",
        **(extra_env or {}),
    }


def _run(module: str, payload: dict, tmp_path, extra_env: dict | None = None):
    return subprocess.run(
        [sys.executable, "-m", module],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=_env(tmp_path, extra_env),
        timeout=120,
    )


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    """One capture, shared: it still imports the facade and opens three stores
    (~1 s), so a per-test capture would add nothing but wall time."""
    root = tmp_path_factory.mktemp("agmem-hooks")
    proc = _run("agmem.hooks.capture", {"session_id": "s1", "prompt": "I moved to Berlin"}, root)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return root


def test_capture_without_a_daemon_writes_the_episode_and_no_vector(seeded):
    """The absent-daemon contract (Phase 2 spec): the episode is persisted, no
    embedder is loaded, and the vector is left for the daemon to backfill.
    Opening the memory here with the same config also pins that the hooks'
    `AGMEM_CONFIG` seam resolves FakeEmbedder — a silent fall back to the lite
    default would pull 471 MB per run."""
    from agmem.hooks import open_memory

    env = _env(seeded)
    import os

    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        mem = open_memory()
        try:
            assert mem.embedder.name == "fake-hash-256"
            stats = mem.stats()
            assert stats["episodes"] == 1
            assert stats["vectors"] == 0, "capture loaded an embedder without a daemon"
            episode = mem.doc_store.list_episodes(namespace=mem.namespace)[0]
            assert episode.meta.get("pending_embed") is True
        finally:
            mem.close()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_capture_then_recall_round_trips_through_the_store(seeded):
    """The whole point, end to end: a prompt captured by one hook comes back as
    context from the other."""
    got = _run("agmem.hooks.recall", {}, seeded)
    assert got.returncode == 0, got.stderr[-2000:]
    payload = json.loads(got.stdout)
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "Berlin" in payload["hookSpecificOutput"]["additionalContext"]


def test_recall_lists_what_the_user_said_not_the_agents_tool_output(seeded):
    """`add_session` files tool calls and tool output as episodes too; the
    session-start listing must still be the user's recent requests."""
    from agmem.hooks import open_doc_store

    ns, store = open_doc_store(namespace="hooktest", data_dir=str(seeded / "data"))
    from agmem.core.types import Episode

    for role, text in (
        ("tool_call", '{"command": "rg -n TODO src"}'),
        ("tool_result", "src/x.py:3: TODO remove"),
        ("assistant", "Found one TODO."),
    ):
        store.add_episode(Episode(content=text, role=role, namespace=ns))
    assert store.count_episodes(namespace=ns) == 4  # the seeded prompt + these three
    store.close()
    got = _run("agmem.hooks.recall", {}, seeded)
    assert got.returncode == 0, got.stderr[-2000:]
    context = json.loads(got.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "Berlin" in context
    assert "rg -n TODO" not in context and "Found one TODO" not in context


def test_recall_on_an_empty_store_emits_nothing_rather_than_an_empty_block(tmp_path):
    """No memory yet must mean no injected context. Emitting a header with no
    lines under it would spend context to tell the model nothing, on every
    session, forever."""
    got = _run("agmem.hooks.recall", {}, tmp_path)
    assert got.returncode == 0
    assert got.stdout.strip() == ""


def test_hooks_survive_malformed_stdin(tmp_path):
    """The harness must never be broken by us. Garbage in, exit 0, no output —
    this is the invariant the whole package is built around."""
    for module in ("agmem.hooks.recall", "agmem.hooks.capture"):
        proc = subprocess.run(
            [sys.executable, "-m", module],
            input="not json at all {{{",
            capture_output=True,
            text=True,
            env=_env(tmp_path),
            timeout=120,
        )
        assert proc.returncode == 0, f"{module}: {proc.stderr[-1500:]}"


def test_hooks_and_server_agree_on_the_default_namespace(tmp_path):
    """Issue #2: the server defaulted to `main`, the hooks to `claude-code`,
    and a registration that followed the docs for both produced two stores that
    could not see each other. Both now resolve through `agmem.env`; this pins
    that a capture with NO namespace told lands where the server's default
    would look."""
    from agmem.env import DEFAULT_NAMESPACE

    env = _env(tmp_path)
    env.pop("AGMEM_NAMESPACE")
    proc = subprocess.run(
        [sys.executable, "-m", "agmem.hooks.capture"],
        input=json.dumps({"session_id": "s1", "prompt": "namespace default probe"}),
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    written = sorted(p.name for p in (tmp_path / "data").iterdir() if p.is_dir())
    assert written == [DEFAULT_NAMESPACE]


def test_recall_refuses_a_config_whose_doc_store_it_cannot_read(tmp_path):
    """Recall opens SQLite directly. A config that puts the doc store elsewhere
    must make it exit 0 with nothing (fail_open), not open an empty SQLite file
    beside the real store and present that as no memory."""
    cfg = tmp_path / "agmem.toml"
    cfg.write_text(
        '[profile]\nname = "lite"\n\n[override]\nembedder = "FakeEmbedder"\n'
        'doc_store = "PostgresDocStore"\n'
    )
    log = tmp_path / "hook.log"
    got = _run("agmem.hooks.recall", {}, tmp_path, {"AGMEM_HOOK_LOG": str(log)})
    assert got.returncode == 0
    assert got.stdout.strip() == ""
    assert "PostgresDocStore" in log.read_text()


def test_capture_ignores_a_prompt_with_no_text(tmp_path):
    """An event carrying no prompt must not write an empty episode — those
    accumulate silently and dilute every later recall."""
    proc = _run("agmem.hooks.capture", {"session_id": "s1", "prompt": "   "}, tmp_path)
    assert proc.returncode == 0
    got = _run("agmem.hooks.recall", {}, tmp_path)
    assert got.stdout.strip() == ""


def test_recall_stays_fast_enough_for_a_blocking_hook(seeded):
    """Recall blocks session start, so its budget is a real constraint rather
    than a preference.

    It briefly did not meet it: opening a full `AgenticMemory` to list episodes
    cost 17.7 s, essentially all of it `SentenceTransformerEmbedder` loading
    weights it never used, which exceeds any sane `timeout` and would be killed
    — presenting as no memory rather than as an error. Reading the doc store
    directly took it to 0.21 s. The bound here is loose enough for a slow CI box
    and still an order of magnitude under the 10 s the wiring documentation
    suggests.
    """
    import time

    start = time.perf_counter()
    got = _run("agmem.hooks.recall", {}, seeded)
    elapsed = time.perf_counter() - start
    assert got.returncode == 0
    assert elapsed < 5.0, f"recall took {elapsed:.1f}s — the embedder is back on this path"


def test_recall_serves_only_the_current_projects_turns(tmp_path):
    """Project gating at the hooks (research §6 #9): capture records the turn's
    cwd; the SessionStart recall lists only turns from the same project tree,
    and a turn with no recorded cwd still shows."""
    a = _run(
        "agmem.hooks.capture",
        {"session_id": "a", "prompt": "In proj-a I use pnpm", "cwd": "/w/proj-a"},
        tmp_path,
    )
    b = _run(
        "agmem.hooks.capture",
        {"session_id": "b", "prompt": "In proj-b I use poetry", "cwd": "/w/proj-b"},
        tmp_path,
    )
    n = _run("agmem.hooks.capture", {"session_id": "n", "prompt": "no cwd recorded here"}, tmp_path)
    assert a.returncode == b.returncode == n.returncode == 0
    out = _run("agmem.hooks.recall", {"cwd": "/w/proj-a/packages/x"}, tmp_path).stdout
    assert "pnpm" in out and "no cwd recorded" in out and "poetry" not in out
    everything = _run("agmem.hooks.recall", {}, tmp_path).stdout
    assert "pnpm" in everything and "poetry" in everything


def test_recall_prompt_asks_the_daemon_to_gate_by_the_sessions_cwd():
    from agmem.hooks.recall_prompt import request_body

    assert request_body({"cwd": "/w/proj-a", "prompt": "q"}, "q", 5) == {
        "query": "q",
        "k": 5,
        "cwd": "/w/proj-a",
    }
    assert request_body({"prompt": "q"}, "q", 3) == {"query": "q", "k": 3}


def test_preserve_without_a_daemon_spools_the_transcript_for_the_next_daemon(tmp_path):
    """PreCompact (research §6 #11): the hook never loads a model; with no
    daemon it spools the transcript path and asks for a daemon, and exits 0."""
    transcript = tmp_path / "s-compact.jsonl"
    transcript.write_text("{}\n")
    proc = _run(
        "agmem.hooks.preserve",
        {
            "session_id": "s-compact",
            "transcript_path": str(transcript),
            "cwd": "/w/p",
            "trigger": "auto",
        },
        tmp_path,
    )
    assert proc.returncode == 0 and proc.stdout == "", proc.stderr[-1000:]
    spool = tmp_path / "data" / "hooktest" / "preserve-queue.jsonl"
    (line,) = spool.read_text().splitlines()
    assert json.loads(line) == {
        "transcript_path": str(transcript),
        "session_id": "s-compact",
        "cwd": "/w/p",
    }
    # an event without a transcript is a no-op
    assert _run("agmem.hooks.preserve", {"session_id": "x"}, tmp_path).returncode == 0
    assert len(spool.read_text().splitlines()) == 1


def test_recall_after_compaction_restores_this_sessions_own_turns(tmp_path):
    """After the harness compacts, SessionStart arrives with source=compact and
    the session id; the listing is then this session's preserved turns, not
    the global recency, and falls back to recency when none are stored."""
    for i, text in enumerate(
        ["first, set up the venv", "then fix the flaky test", "now write the docs"]
    ):
        assert (
            _run(
                "agmem.hooks.capture", {"session_id": "c1", "prompt": text, "cwd": "/w/p"}, tmp_path
            ).returncode
            == 0
        )
    assert (
        _run(
            "agmem.hooks.capture", {"session_id": "other", "prompt": "unrelated session"}, tmp_path
        ).returncode
        == 0
    )
    out = _run(
        "agmem.hooks.recall", {"session_id": "c1", "source": "compact", "cwd": "/w/p"}, tmp_path
    ).stdout
    assert "before the context was compacted" in out
    assert "flaky test" in out and "venv" in out and "unrelated session" not in out
    fallback = _run(
        "agmem.hooks.recall", {"session_id": "never-seen", "source": "compact"}, tmp_path
    ).stdout
    assert "Recent memory from previous sessions" in fallback and "unrelated session" in fallback


def test_daemon_preserve_ingests_a_transcript_and_drains_the_spool(tmp_path):
    """The daemon side, in process: `/hooks/preserve`'s work and the spool
    drain both go through `add_session(distill=False)` — raw steps under the
    session id, no model — and a second pass is idempotent."""
    from agmem.config import AgmemConfig
    from agmem.mcp.server import _Registry

    transcript = tmp_path / "t1.jsonl"
    records = [
        {"type": "user", "uuid": "u1", "isSidechain": False, "timestamp": "2026-09-05T01:00:00.000Z",
         "cwd": "/w/p", "sessionId": "t1", "message": {"role": "user", "content": "keep this before compaction"}},
        {"type": "assistant", "uuid": "u2", "isSidechain": False, "timestamp": "2026-09-05T01:00:01.000Z",
         "cwd": "/w/p", "sessionId": "t1", "message": {"role": "assistant", "content": [{"type": "text", "text": "kept."}]}},
    ]  # fmt: skip
    transcript.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    reg = _Registry()
    reg.start(
        "t",
        [],
        AgmemConfig(
            profile="lite",
            data_dir=tmp_path / "data",
            sync_write=True,
            overrides={"embedder": "FakeEmbedder"},
        ),
    )
    try:
        result = reg.preserve(str(transcript), "t")
        assert result["session_id"] == "t1" and result["episodes"] == 2 and result["admitted"]
        mem = reg.get("t")
        assert mem.doc_store.count_episodes() == 2
        spool = tmp_path / "data" / "t" / "preserve-queue.jsonl"
        spool.write_text(
            json.dumps({"transcript_path": str(transcript), "session_id": "t1"})
            + "\n"
            + json.dumps({"transcript_path": str(tmp_path / "missing.jsonl")})
            + "\n"
        )
        assert reg.drain_preserve_queue("t") == 1  # the missing file is skipped, not fatal
        assert spool.read_text() == "" and mem.doc_store.count_episodes() == 2  # idempotent
    finally:
        reg.close_all()
