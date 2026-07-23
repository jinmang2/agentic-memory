"""Full run-artifact capture for the A-Mem reproduction harness (issue #1):
the LLM-call trace sink, the post-ingest memory snapshot, per-question
retrieval capture in records, and the write-once/read-sweep guarantee that
--eval-only issues ZERO write-path LLM calls. Unit/integration level only —
fake embedder + fake LLM throughout, no API/server, no paid calls."""

from __future__ import annotations

import importlib.util as _ilu
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from agmem import AgenticMemory
from agmem.bench import locomo
from agmem.config import AgmemConfig
from agmem.embed.fake import FakeEmbedder
from agmem.llm.client import LLMClient, RoleConfig
from agmem.llm.structured import StructuredCaller
from agmem.organizers.amem import AMemOrganizer

_REPRO_PATH = Path(__file__).resolve().parent.parent / "scripts" / "exp_amem_repro.py"


def _load_repro():
    if str(_REPRO_PATH.parent) not in sys.path:
        sys.path.insert(0, str(_REPRO_PATH.parent))
    spec = _ilu.spec_from_file_location("exp_amem_repro", _REPRO_PATH)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------- 1) LLM-call trace sink: full I/O, one line per call ----------


class _FakeOpenAI:
    """Minimal OpenAI-compatible stub: records the kwargs it was called with and
    returns a canned completion with usage, so LLMClient.chat exercises the real
    success path (budget + trace) without any network."""

    def __init__(self, content="canned reply", tokens=(11, 7)):
        self._content = content
        self._tokens = tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        msg = SimpleNamespace(content=self._content)
        usage = SimpleNamespace(prompt_tokens=self._tokens[0], completion_tokens=self._tokens[1])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


def _client_with_fake(trace_path, content="canned reply"):
    roles = {"generate": RoleConfig(endpoint="http://x", model="m", temperature=0.3)}
    client = LLMClient(roles, trace_path=trace_path)
    fake = _FakeOpenAI(content=content)
    client._client_for = lambda cfg: fake  # bypass real openai construction
    return client


def test_trace_sink_writes_full_io_line(tmp_path):
    trace = tmp_path / "run.llm-trace.jsonl"
    client = _client_with_fake(trace, content="the FULL response text")
    messages = [{"role": "user", "content": "the FULL prompt sent"}]
    out = client.chat("generate", messages, budget_key="generate/answer")
    assert out == "the FULL response text"

    lines = trace.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # exactly one JSON line per call
    row = json.loads(lines[0])
    # schema: full prompt + full response, never truncated, plus token/latency
    assert row["role"] == "generate"
    assert row["budget_key"] == "generate/answer"
    assert row["model"] == "m"
    assert row["messages"] == messages  # FULL prompt preserved verbatim
    assert row["response_text"] == "the FULL response text"
    assert row["tokens_in"] == 11 and row["tokens_out"] == 7
    assert row["error"] is None
    assert "ts_iso" in row and isinstance(row["latency_ms"], (int, float))


def test_trace_sink_records_failure(tmp_path):
    trace = tmp_path / "err.llm-trace.jsonl"
    roles = {"generate": RoleConfig(endpoint="http://x", model="m")}
    client = LLMClient(roles, trace_path=trace)

    class _Boom:
        def _create(self, **kwargs):
            raise RuntimeError("boom-503")

        chat = None

    boom = _Boom()
    boom.chat = SimpleNamespace(completions=SimpleNamespace(create=boom._create))
    client._client_for = lambda cfg: boom

    try:
        client.chat("generate", [{"role": "user", "content": "q"}])
        raised = False
    except RuntimeError:
        raised = True
    assert raised  # exception is re-raised, never swallowed
    row = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert "boom-503" in row["error"]  # failure captured with the error text
    assert row["response_text"] == "" and row["tokens_in"] == 0


def test_trace_sink_off_by_default_writes_nothing(tmp_path):
    # No trace_path -> backward compatible: no file, no behavior change.
    roles = {"generate": RoleConfig(endpoint="http://x", model="m")}
    client = LLMClient(roles)
    client._client_for = lambda cfg: _FakeOpenAI()
    assert client.trace_path is None
    client.chat("generate", [{"role": "user", "content": "q"}])
    assert not any(tmp_path.iterdir())  # nothing written


# ---------------- 2) retrieval-chunk capture threads into records --------------


class _StubAnswerLLM:
    def chat(self, role, messages, **kwargs):
        return "stub answer"


def test_retrieval_captured_in_records(tmp_path):
    mem = AgenticMemory(namespace="t", organizers=["passthrough"], embedder=FakeEmbedder(dim=64))
    try:
        mem.llm = _StubAnswerLLM()
        # seed one episode so retrieval returns a hit with real chunk text
        mem.add_message("Alice moved to Berlin in 2021.", role="user")
        mem.flush()
        questions = [{"question": "Where did Alice move?", "answer": "Berlin", "category": 4}]
        res = locomo.evaluate(mem, questions, memory_types=("episodic",), capture_retrieval=True)
    finally:
        mem.close()

    rec = res["records"][0]
    assert "retrieval" in rec
    cap = rec["retrieval"]
    assert cap["query"] == "Where did Alice move?"
    assert cap["k"] == 10 and cap["memory_types"] == ["episodic"]
    assert isinstance(cap["retrieved"], list) and cap["retrieved"]
    hit = cap["retrieved"][0]
    assert {"id", "memory_type", "score", "text"} <= set(hit)
    assert "Berlin" in hit["text"]  # the ACTUAL chunk text put into context


def test_retrieval_capture_off_by_default_keeps_record_schema(tmp_path):
    mem = AgenticMemory(namespace="t", organizers=["passthrough"], embedder=FakeEmbedder(dim=64))
    try:
        mem.llm = _StubAnswerLLM()
        mem.add_message("some content", role="user")
        mem.flush()
        res = locomo.evaluate(
            mem, [{"question": "q", "answer": "a", "category": 4}], memory_types=("episodic",)
        )
    finally:
        mem.close()
    assert "retrieval" not in res["records"][0]  # non-capturing path unchanged


# ---------------- 3) post-ingest memory snapshot enumerates items --------------


def test_memory_snapshot_enumerates_episodic_and_derived(tmp_path):
    repro = _load_repro()
    mem = AgenticMemory(namespace="snap", organizers=["passthrough"], embedder=FakeEmbedder(dim=64))
    try:
        mem.add_message("episode one", role="user")
        mem.add_message("episode two", role="user")
        # a derived note item straight into the store (no LLM needed)
        mem.doc_store.put_item(
            "n1", "notes", "snap", {"id": "n1", "content": "note body", "tags": ["x"], "links": []}
        )
        mem.flush()
        out = tmp_path / "snap.memory.jsonl"
        with out.open("w", encoding="utf-8") as f:
            counts = repro.dump_memory_snapshot(mem, conv_idx=0, out=f)
    finally:
        mem.close()

    assert counts.get("episodic") == 2 and counts.get("notes") == 1
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    kinds = {r["memory_type"] for r in rows}
    assert kinds == {"episodic", "notes"}
    for r in rows:
        assert r["conv"] == 0 and "content" in r  # every line tagged + carries content
    note_row = next(r for r in rows if r["memory_type"] == "notes")
    assert note_row["tags"] == ["x"] and "links" in note_row


# ---------------- 4) write-once / read-sweep: eval-only = ZERO write calls ------


class _FakeCountingLLM:
    """Counts chat() calls per role and returns schema-valid canned JSON so the
    AMem write path (extract note + distill evolution) and the eval read path
    (extract keyword-rewrite + generate answer) both run without any API."""

    def __init__(self):
        self.calls: dict[str, int] = {}

    def chat(self, role, messages, budget_key=None, **overrides):
        self.calls[role] = self.calls.get(role, 0) + 1
        prompt = " ".join(m.get("content", "") for m in messages)
        if role == "extract":
            if "generate several keywords" in prompt:
                return '{"keywords": "alpha, beta"}'
            return '{"keywords": ["k1"], "context": "ctx", "tags": ["t1", "t2", "t3"]}'
        if role == "distill":
            return '{"should_evolve": false, "connections": []}'
        return "stub answer"


def _build_counting_mem(data_dir, namespace):
    cfg = AgmemConfig(
        profile="lite",
        data_dir=data_dir,
        llm_roles={"extract": RoleConfig(endpoint="x", model="m")},
        use_guided_json=False,
        sync_write=True,
        lexical_types=("episodic",),
    )
    mem = AgenticMemory(
        namespace=namespace, organizers=[AMemOrganizer()], embedder=FakeEmbedder(dim=64), config=cfg
    )
    fake = _FakeCountingLLM()
    mem.llm = fake
    mem.structured = StructuredCaller(fake, use_guided_json=False)
    mem._ctx.llm = mem.structured  # organizer writes through the fake too
    return mem, fake


def test_write_once_then_eval_only_issues_zero_write_calls(tmp_path):
    ns = "repro-conv0"
    sample_msgs = [
        "(2021) Alice: I moved to Berlin last year.",
        "(2021) Bob: How is the new job going?",
        "(2021) Alice: The job at the museum is great.",
    ]

    # --- INGEST phase: build + persist the store, count write-path calls -------
    mem, ing = _build_counting_mem(tmp_path, ns)
    try:
        for m in sample_msgs:
            mem.add_message(m, role="user")
        mem.flush()
        mem.consolidate()
        notes_after_ingest = len(mem.doc_store.list_items("notes", namespace=ns))
        episodes_after_ingest = mem.doc_store.count_episodes(ns)
    finally:
        mem.close()

    # write path actually fired during ingest (extract per note; distill once a
    # neighbor exists) — otherwise the "zero on eval" assertion would be vacuous.
    assert ing.calls.get("extract", 0) >= len(sample_msgs)
    assert ing.calls.get("distill", 0) >= 1
    assert notes_after_ingest >= 1
    # store dir was actually written to disk (persisted, not in-memory)
    assert (tmp_path / ns).exists()

    # --- EVAL-ONLY phase: reopen the SAME store, run QA ------------------------
    mem2, ev = _build_counting_mem(tmp_path, ns)
    try:
        # the persisted notes/episodes reloaded — no re-ingest happened
        assert len(mem2.doc_store.list_items("notes", namespace=ns)) == notes_after_ingest
        assert mem2.doc_store.count_episodes(ns) == episodes_after_ingest
        questions = [
            {"question": "Where did Alice move?", "answer": "Berlin", "category": 4},
            {"question": "Where does Alice work?", "answer": "museum", "category": 4},
        ]
        locomo.evaluate(
            mem2, questions, memory_types=("notes",), keyword_queries=True, eval_mode="wujiang"
        )
        # THE GUARANTEE: eval-only issues ZERO write-path calls (no note
        # extraction, no evolution). distill is write-path-only and never fires.
        assert ev.calls.get("distill", 0) == 0
        # notes/episodes unchanged by evaluation (no writes to the store)
        assert len(mem2.doc_store.list_items("notes", namespace=ns)) == notes_after_ingest
        assert mem2.doc_store.count_episodes(ns) == episodes_after_ingest
        # read path DID run (keyword rewrite = extract, answer = generate)
        assert ev.calls.get("extract", 0) == len(questions)
        assert ev.calls.get("generate", 0) == len(questions)
    finally:
        mem2.close()
