"""Full run-artifact capture for the A-Mem reproduction harness (issue #1):
the LLM-call trace sink, the post-ingest memory snapshot, per-question
retrieval capture in records, and the write-once/read-sweep guarantee that
--eval-only issues ZERO write-path LLM calls. Unit/integration level only —
fake embedder + fake LLM throughout, no API/server, no paid calls."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util as _ilu
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from agmem import AgenticMemory
from agmem.bench import locomo
from agmem.bench.counting import build_counting_memory
from agmem.embed.fake import FakeEmbedder
from agmem.llm.client import LLMClient, RoleConfig
from agmem.organizers.amem import AMemOrganizer

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_REPRO_PATH = _SCRIPTS / "exp_amem_repro.py"
_CONFIGS_PATH = Path(__file__).resolve().parent.parent / "scripts" / "repro" / "configs.py"


def _load_repro():
    if str(_REPRO_PATH.parent) not in sys.path:
        sys.path.insert(0, str(_REPRO_PATH.parent))
    spec = _ilu.spec_from_file_location("exp_amem_repro", _REPRO_PATH)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_configs():
    spec = _ilu.spec_from_file_location("repro_configs", _CONFIGS_PATH)
    mod = _ilu.module_from_spec(spec)
    # configs.py's RunnerConfig is a frozen dataclass under `from __future__
    # import annotations`; dataclasses resolves deferred annotations via
    # sys.modules[cls.__module__], so the module must be registered there
    # BEFORE exec_module runs the class body (unlike _load_repro, which has
    # no dataclasses of its own and never needed this).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------- 1) LLM-call trace sink: full I/O, one line per call ----------


class _FakeOpenAI:
    """Minimal OpenAI-compatible stub: records the kwargs it was called with and
    returns a canned completion with usage, so LLMClient.chat exercises the real
    success path (budget + trace) without any network."""

    def __init__(self, content="canned reply", tokens=(11, 7), finish_reason="stop"):
        self._content = content
        self._tokens = tokens
        self._finish_reason = finish_reason
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        msg = SimpleNamespace(content=self._content)
        usage = SimpleNamespace(prompt_tokens=self._tokens[0], completion_tokens=self._tokens[1])
        choice = SimpleNamespace(message=msg, finish_reason=self._finish_reason)
        return SimpleNamespace(choices=[choice], usage=usage)


def _client_with_fake(trace_path, content="canned reply", finish_reason="stop"):
    roles = {"generate": RoleConfig(endpoint="http://x", model="m", temperature=0.3, max_tokens=64)}
    client = LLMClient(roles, trace_path=trace_path)
    fake = _FakeOpenAI(content=content, finish_reason=finish_reason)
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
    assert row["finish_reason"] == "stop"
    assert "ts_iso" in row and isinstance(row["latency_ms"], (int, float))


def test_a_reply_cut_by_the_output_cap_is_marked_in_the_trace_and_warned(tmp_path, caplog):
    """`finish_reason == "length"` is the one signal that a JSON reply failed
    because the cap cut it, not because the model could not follow the schema.
    The 2026-09-04 OpenRouter smoke paid for every session twice this way and
    the budget line could not say so; the trace and the log now can."""
    import logging

    trace = tmp_path / "run.llm-trace.jsonl"
    client = _client_with_fake(trace, content='{"summary": "cut mid', finish_reason="length")
    with caplog.at_level(logging.WARNING, logger="agmem.llm.client"):
        out = client.chat("generate", [{"role": "user", "content": "p"}])
    assert out == '{"summary": "cut mid'  # returned as-is: the caller decides
    row = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert row["finish_reason"] == "length"
    assert any(
        "truncated at the output cap" in r.getMessage() and "max_tokens=64" in r.getMessage()
        for r in caplog.records
    )


def test_a_gzipped_trace_stays_readable_line_by_line(tmp_path):
    """A `_s` LongMemEval arm sends 500 prompts of ~525 KB, so its trace is ~260 MB
    uncompressed and is written as `.jsonl.gz` instead. Appending one gzip member
    per call is what keeps it streamable AFTER a kill — the property the whole
    re-score-without-re-spending plan rests on, and the one a single-member
    compressor would quietly take away."""
    trace = tmp_path / "run.llm-trace.jsonl.gz"
    client = _client_with_fake(trace, content="answer")
    for i in range(3):
        client.chat("generate", [{"role": "user", "content": f"prompt {i}"}])

    with gzip.open(trace, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    assert [r["messages"][0]["content"] for r in rows] == ["prompt 0", "prompt 1", "prompt 2"]
    # and it really is compressed, not just named .gz
    assert trace.read_bytes()[:2] == b"\x1f\x8b"


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


# _FakeCountingLLM / _build_counting_mem live at agmem.bench.counting now
# (CountingLLM / build_counting_memory) — one home for the fixture, shared with
# the dry-run quote harness.


def _build_counting_mem(data_dir, namespace):
    return build_counting_memory("amem", lambda: [AMemOrganizer()], data_dir, namespace, ("notes",))


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


# ---------------- 5) concurrent eval == sequential eval (bit-for-bit) ----------


class _DetAnswerLLM:
    """Deterministic, thread-safe fake: the reply is a hash of the full prompt,
    so a given question over a fixed store yields the SAME pred no matter which
    thread runs it. Any deterministic fake makes workers>1 and workers=1 produce
    identical aggregates/records — that is exactly the invariant under test."""

    def chat(self, role, messages, **kwargs):
        prompt = " ".join(m.get("content", "") for m in messages)
        return "resp-" + hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]


def _seed_qa_mem():
    mem = AgenticMemory(namespace="conc", organizers=["passthrough"], embedder=FakeEmbedder(dim=64))
    mem.llm = _DetAnswerLLM()
    for msg in [
        "Alice moved to Berlin in 2021.",
        "Bob started a new job at the museum.",
        "Carol adopted a dog named Rex.",
        "Dan visited Tokyo last spring.",
        "Alice learned to play the cello.",
        "Bob's museum job involves restoring paintings.",
    ]:
        mem.add_message(msg, role="user")
    mem.flush()
    return mem


def test_concurrent_eval_matches_sequential(tmp_path):
    questions = [
        {"question": "Where did Alice move?", "answer": "Berlin", "category": 4},
        {"question": "Where does Bob work?", "answer": "museum", "category": 4},
        {"question": "What pet did Carol adopt?", "answer": "dog", "category": 1},
        {"question": "Where did Dan visit?", "answer": "Tokyo", "category": 2},
        {"question": "What instrument did Alice learn?", "answer": "cello", "category": 3},
        {"question": "Is Alice a pilot?", "adversarial_answer": "No", "category": 5},
        {"question": "What does Bob restore?", "answer": "paintings", "category": 4},
    ]
    mem = _seed_qa_mem()
    try:
        seq = locomo.evaluate(
            mem, questions, memory_types=("episodic",), capture_retrieval=True, workers=1
        )
    finally:
        mem.close()
    mem2 = _seed_qa_mem()
    try:
        conc = locomo.evaluate(
            mem2, questions, memory_types=("episodic",), capture_retrieval=True, workers=8
        )
    finally:
        mem2.close()

    # aggregates identical, and records in the SAME (question) order
    assert conc["overall"] == seq["overall"]
    assert conc["by_category"] == seq["by_category"]
    assert [r["q"] for r in conc["records"]] == [r["q"] for r in seq["records"]]
    assert [r["pred"] for r in conc["records"]] == [r["pred"] for r in seq["records"]]
    assert [r["f1"] for r in conc["records"]] == [r["f1"] for r in seq["records"]]


# ---------------- 6) budget merge: latency is call-weighted, cost sums ---------


def _fake_mem(summary: dict, drops: dict | None = None, organizers=()):
    """A stand-in exposing just what _merge_budget touches: a budget with a
    summary(), an optional structured.drops, and the organizer list it folds
    per-organizer `discarded` counters out of."""
    structured = None if drops is None else SimpleNamespace(drops=drops)
    return SimpleNamespace(
        budget=SimpleNamespace(summary=lambda: summary),
        structured=structured,
        organizers=list(organizers),
    )


def test_merge_budget_latency_is_call_weighted_not_last(tmp_path):
    repro = _load_repro()
    merged: dict = {}
    drops: dict = {}
    # conv A: 2 generate calls @ 100ms avg (200ms total)
    repro._merge_budget(
        merged,
        drops,
        _fake_mem(
            {
                "generate": {
                    "calls": 2,
                    "tokens_in": 10,
                    "tokens_out": 5,
                    "latency_ms_avg": 100.0,
                    "errors": 0,
                }
            }
        ),
    )
    # conv B: 8 generate calls @ 200ms avg (1600ms total)
    repro._merge_budget(
        merged,
        drops,
        _fake_mem(
            {
                "generate": {
                    "calls": 8,
                    "tokens_in": 40,
                    "tokens_out": 20,
                    "latency_ms_avg": 200.0,
                    "errors": 0,
                }
            }
        ),
    )
    g = merged["generate"]
    assert g["calls"] == 10 and g["tokens_in"] == 50 and g["tokens_out"] == 25
    # correct call-weighted mean = 1800/10 = 180, NOT the last conv's 200
    assert g["latency_ms_avg"] == 180.0
    assert g["latency_ms_total"] == 1800.0


def test_merge_run_budgets_sums_across_runs(tmp_path):
    repro = _load_repro()
    run1: dict = {}
    repro._merge_budget(
        run1,
        {},
        _fake_mem(
            {
                "generate": {
                    "calls": 3,
                    "tokens_in": 30,
                    "tokens_out": 9,
                    "latency_ms_avg": 100.0,
                    "errors": 0,
                }
            }
        ),
    )
    run2: dict = {}
    repro._merge_budget(
        run2,
        {},
        _fake_mem(
            {
                "generate": {
                    "calls": 3,
                    "tokens_in": 30,
                    "tokens_out": 9,
                    "latency_ms_avg": 100.0,
                    "errors": 0,
                }
            }
        ),
    )
    total = repro._merge_run_budgets([run1, run2])
    # every credit across BOTH runs is counted, not just run 1
    assert total["generate"]["calls"] == 6
    assert total["generate"]["tokens_in"] == 60 and total["generate"]["tokens_out"] == 18
    # cost of the summed budget == 2x a single run's cost
    assert repro.cost_usd(total, "gpt-4o-mini") == round(2 * repro.cost_usd(run1, "gpt-4o-mini"), 6)


# ---------------- 7) ingest-completion sentinel: write + verify guard ----------


def test_sentinel_roundtrip_and_partial_guard(tmp_path):
    repro = _load_repro()
    store = tmp_path / "store"
    store.mkdir()
    repro.write_ingest_sentinel(str(store), [0, 1, 2], [{"conv": 0}], "deadbeef")
    assert (store / repro.SENTINEL_NAME).exists()
    # subset of ingested convs -> OK (no raise)
    repro.verify_ingest_sentinel(str(store), [0, 2])
    # requesting a conv that was NOT ingested -> loud refusal
    try:
        repro.verify_ingest_sentinel(str(store), [0, 3])
        raised = False
    except SystemExit:
        raised = True
    assert raised


def test_verify_sentinel_missing_refuses(tmp_path):
    repro = _load_repro()
    store = tmp_path / "empty"
    store.mkdir()  # dir exists but ingest never completed (no sentinel)
    try:
        repro.verify_ingest_sentinel(str(store), [0])
        raised = False
    except SystemExit:
        raised = True
    assert raised  # a bare dir without the sentinel is rejected, not trusted


# ---------------- 8) headline aggregator: mean±std over K summaries ------------


def _load_aggregator():
    path = Path(__file__).resolve().parent.parent / "scripts" / "repro" / "aggregate_headline.py"
    spec = _ilu.spec_from_file_location("aggregate_headline", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_eval_seed(path, f1, seed_i, k=10, eval_mode="wujiang", expand="off", cost=0.7):
    path.write_text(
        json.dumps(
            {
                "stamp": {
                    "model": "gpt-4o-mini",
                    "eval_mode": eval_mode,
                    "k": k,
                    "expand_links": expand,
                    "git_sha": f"sha{seed_i}",
                },
                "overall": {"f1": f1, "bleu1": f1 - 5},
                "by_category": {"temporal": {"f1": f1 + 10, "n": 100}},
                "cost_usd": cost,
            }
        )
    )
    return path


def test_headline_aggregator_mean_std(tmp_path):
    agg_mod = _load_aggregator()
    seeds = [
        _write_eval_seed(tmp_path / f"seed{i}.json", f1, i)
        for i, f1 in enumerate([30.0, 34.0, 32.0])  # overall F1 -> mean 32, std 2
    ]
    out = agg_mod.aggregate(seeds)
    assert out["n_seeds"] == 3
    ov = out["metrics"]["f1"]["overall"]
    assert ov["mean"] == 32.0 and ov["std"] == 2.0 and ov["min"] == 30.0 and ov["max"] == 34.0
    assert out["metrics"]["f1"]["temporal"]["mean"] == 42.0
    # cost is HONEST: eval-only when no ingest supplied, and NOT mislabeled total
    assert out["eval_cost_usd"] == round(3 * 0.7, 6)
    assert out["ingest_cost_usd"] is None and out["campaign_cost_usd"] is None
    assert "cost_usd_total" not in out  # the misleading field is gone
    # The fixture writes the OLD artifact shape (`git_sha`); the aggregator now
    # reports the canonical `commit`, resolved through the fallback — that is what
    # keeps already-spent runs on disk aggregatable.
    assert len(out["sources"]) == 3 and out["sources"][0]["commit"] == "sha0"


def test_headline_cost_includes_ingest_when_supplied(tmp_path):
    agg_mod = _load_aggregator()
    evals = [_write_eval_seed(tmp_path / f"e{i}.json", f1, i) for i, f1 in enumerate([30.0, 32.0])]
    ing = []
    for i in range(2):
        p = tmp_path / f"ing{i}.json"
        p.write_text(json.dumps({"stamp": {"git_sha": f"i{i}"}, "cost_usd": 0.9}))
        ing.append(p)
    out = agg_mod.aggregate(evals, ing)
    # eval and ingest reported separately, campaign = sum (no credit dropped)
    assert out["eval_cost_usd"] == round(2 * 0.7, 6)
    assert out["ingest_cost_usd"] == round(2 * 0.9, 6)
    assert out["campaign_cost_usd"] == round(2 * 0.7 + 2 * 0.9, 6)
    assert "do not sum" in out["ingest_note"].lower()


def test_headline_refuses_mismatched_config(tmp_path):
    agg_mod = _load_aggregator()
    a = _write_eval_seed(tmp_path / "a.json", 30.0, 0, k=10)
    b = _write_eval_seed(tmp_path / "b.json", 32.0, 1, k=20)  # different k -> incompatible
    try:
        agg_mod.aggregate([a, b])
        raised = False
    except SystemExit:
        raised = True
    assert raised  # refuses to average k=10 and k=20 seeds into one headline


# ---------------- 9) parallel-ingest orchestrator: pure helpers ----------------


def _load_parallel():
    path = Path(__file__).resolve().parent.parent / "scripts" / "repro" / "ingest_parallel.py"
    spec = _ilu.spec_from_file_location("ingest_parallel", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parallel_parse_convs():
    p = _load_parallel()
    assert p.parse_convs("all") == list(range(10))
    assert p.parse_convs("0-3") == [0, 1, 2, 3]
    assert p.parse_convs("0,2,5") == [0, 2, 5]
    assert p.parse_convs("0-2,7,9") == [0, 1, 2, 7, 9]
    assert p.parse_convs("3,3,3") == [3]  # de-duplicated
    for bad in ("10", "-1", "8-11", ""):
        try:
            p.parse_convs(bad)
            raised = False
        except SystemExit:
            raised = True
        assert raised, f"parse_convs({bad!r}) should reject out-of-range/empty"


def test_parallel_conv_done_needs_summary_and_nonempty_store(tmp_path, monkeypatch):
    p = _load_parallel()
    monkeypatch.setattr(p.H, "OUT", tmp_path)  # per-conv summaries land here
    data_dir = tmp_path / "store"
    # nothing yet -> not done
    assert not p.conv_is_done("gpt-4o-mini", str(data_dir), 0, "_seed1")
    # summary present but store missing -> still not done (orphan summary)
    p.per_conv_summary_path("gpt-4o-mini", 0, "_seed1").write_text("{}")
    assert not p.conv_is_done("gpt-4o-mini", str(data_dir), 0, "_seed1")
    # store present but EMPTY -> not done (crashed mid-ingest)
    (data_dir / "repro-conv0").mkdir(parents=True)
    assert not p.conv_is_done("gpt-4o-mini", str(data_dir), 0, "_seed1")
    # summary + non-empty store -> done
    (data_dir / "repro-conv0" / "db.sqlite").write_text("x")
    assert p.conv_is_done("gpt-4o-mini", str(data_dir), 0, "_seed1")


def _fake_conv_summary(n_turns, calls, tin, tout, cost, per_type, op_counts=None):
    """Shape one per-conv --ingest-only summary the way exp_amem_repro emits it.

    `op_counts` is part of that shape (`combined["op_counts"] = op_counts or
    None`) and defaults to a non-empty tally here on purpose: while this fixture
    omitted the field, no test could notice that the parallel merge was dropping
    it, and every orchestrated campaign wrote a combined summary claiming the
    write path had done nothing.
    """
    return {
        "op_counts": op_counts if op_counts is not None else {"ADD:notes": calls},
        "stamp": {"model": "gpt-4o-mini", "embedder": "all-MiniLM-L6-v2", "conv": "0"},
        "ingest_only": True,
        "per_conv": [{"conv": 0, "n_turns": n_turns}],
        "llm_budget": {
            "extract": {
                "calls": calls,
                "tokens_in": tin,
                "tokens_out": tout,
                "errors": 0,
                "latency_ms_total": 100.0 * calls,
                "latency_ms_avg": 100.0,
            }
        },
        "drops": {},
        "timing": {"ingest_s": 60.0, "total_s": 60.0},
        "memory_capacity": {"per_type": per_type, "total_items": sum(per_type.values())},
    }


def test_parallel_merge_sums_budget_cost_and_capacity():
    p = _load_parallel()
    a = _fake_conv_summary(400, calls=800, tin=1000, tout=500, cost=0.0, per_type={"notes": 400})
    b = _fake_conv_summary(600, calls=1200, tin=2000, tout=800, cost=0.0, per_type={"notes": 600})
    merged = p.merge_ingest_summaries([a, b], "gpt-4o-mini")
    # tokens/calls summed across convs
    assert merged["llm_budget"]["extract"]["calls"] == 2000
    assert merged["llm_budget"]["extract"]["tokens_in"] == 3000
    assert merged["llm_budget"]["extract"]["tokens_out"] == 1300
    # cost recomputed from the SUMMED tokens (not summed per-conv costs) via the
    # harness cost model -> single source of truth, matches sequential --conv all
    assert merged["cost_usd"] == p.H.cost_usd(merged["llm_budget"], "gpt-4o-mini")
    # ingest seconds = summed compute time; memory per-type counts summed
    assert merged["ingest_s"] == 120.0
    assert merged["memory_capacity"]["per_type"]["notes"] == 1000
    assert merged["memory_capacity"]["total_items"] == 1000


def test_parallel_finalize_writes_combined_summary_and_sentinel(tmp_path, monkeypatch):
    """The finalize step is a drop-in for the sequential --conv all ingest: given
    each conv's per-conv summary, it must emit the SAME <model>_all_ingest<sfx>.json
    (cost summed) + the SINGLE combined sentinel covering every conv, so --eval-only
    and the headline aggregator work unchanged. No subprocess, no paid call."""
    p = _load_parallel()
    monkeypatch.setattr(p.H, "OUT", tmp_path)
    data_dir = tmp_path / "store"
    convs = [0, 1, 2]
    # lay down each conv's per-conv summary + a non-empty store dir (as workers would)
    for c in convs:
        summ = _fake_conv_summary(
            400 + c, calls=800, tin=1000, tout=500, cost=0.0, per_type={"notes": 400 + c}
        )
        p.per_conv_summary_path("gpt-4o-mini", c, "_seed1").write_text(json.dumps(summ))
        sd = data_dir / f"repro-conv{c}"
        sd.mkdir(parents=True)
        (sd / "db.sqlite").write_text("x")

    args = SimpleNamespace(
        model="gpt-4o-mini",
        data_dir=str(data_dir),
        tag_suffix="_seed1",
        workers=3,
        config="amem",
    )
    out_path, sentinel, combined = p.finalize_combined(args, convs, wall_s=90.0)

    # drop-in NAME: exactly what the sequential --conv all --ingest-only would write
    assert out_path.name == "gpt-4o-mini_all_ingest_seed1.json"
    assert out_path.exists() and combined["ingest_only"] is True
    # cost summed across the 3 convs via the harness cost model
    assert combined["cost_usd"] == p.H.cost_usd(combined["llm_budget"], "gpt-4o-mini")
    assert combined["llm_budget"]["extract"]["calls"] == 2400
    # wall clock recorded separately from summed compute time (the speedup evidence)
    assert combined["timing"]["wall_s"] == 90.0 and combined["timing"]["ingest_s"] == 180.0
    # the SINGLE authoritative sentinel covers every conv -> --eval-only will accept
    assert sentinel.name == p.H.SENTINEL_NAME
    done = set(json.loads(sentinel.read_text())["conv_indices"])
    assert done == set(convs)
    # the write path's own record of itself survives the merge INTO the emitted
    # dict — summing it in `merge_ingest_summaries` is not enough if the key is
    # then left out of the summary that actually gets written to disk.
    assert combined["op_counts"] == {"ADD:notes": 2400}
    assert json.loads(out_path.read_text())["op_counts"] == {"ADD:notes": 2400}


def test_required_stamp_fields_match_the_documented_discipline():
    """docs/05 §3 names the fields every result must carry. That line was
    restated in three stampers and two had drifted — the LoCoMo results on disk
    are missing four of the six. Tie the constant to the doc so the next
    divergence is a test failure, not something found by reading JSON."""
    import re
    from pathlib import Path

    from agmem.bench.stamp import REQUIRED_FIELDS

    doc = Path(__file__).resolve().parents[1] / "docs" / "05-api-design.md"
    line = next(ln for ln in doc.read_text().splitlines() if "재현성 규율" in ln)
    documented = tuple(re.findall(r"\w+", re.search(r"\{([^}]*)\}", line).group(1)))
    assert documented == REQUIRED_FIELDS


def test_run_stamp_always_carries_the_required_fields():
    from agmem import AgenticMemory
    from agmem.bench.stamp import REQUIRED_FIELDS, run_stamp
    from agmem.embed.fake import FakeEmbedder

    mem = AgenticMemory(namespace="t", organizers=["passthrough"], embedder=FakeEmbedder(dim=64))
    try:
        stamp = run_stamp(mem, model="m", judge=True, runs=3, dataset="d")
        assert set(REQUIRED_FIELDS) <= set(stamp)
        assert stamp["profile"] == "lite" and stamp["runs"] == 3
        assert stamp["commit"] and stamp["commit"] != "unknown"
    finally:
        mem.close()

    # no memory (aggregate stamps): the fields still exist rather than vanishing
    bare = run_stamp(None, model=None, dataset_version="v1")
    assert set(REQUIRED_FIELDS) <= set(bare)
    assert bare["profile"] is None and bare["dataset_version"] == "v1"


def test_dataset_fingerprint_is_content_addressed(tmp_path):
    """A hand-written label cannot say WHICH copy of the data a number came
    from — the substance of the Zep-LoCoMo dispute the discipline cites."""
    from agmem.bench.stamp import dataset_fingerprint

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text('{"x": 1}')
    b.write_text('{"x": 2}')
    assert dataset_fingerprint(a) == dataset_fingerprint(a)
    assert dataset_fingerprint(a) != dataset_fingerprint(b)
    assert dataset_fingerprint(tmp_path / "missing.json") == "unknown"


def test_amem_rawq_differs_from_amem_at_retrieval_only():
    """The read-protocol ablation must change the read protocol and nothing else.

    `amem_rawq` prices a step that belongs to A-Mem, not to us: its evaluation
    harness rewrites each question into LLM-generated keywords before searching
    (`test_advanced.py:129,134` at the pinned SHA), which the paper's account of
    the read path does not mention, and which no other arm of the four-way table
    pays. `amem` therefore stays the faithful headline arm and this one measures
    what the step costs — +5.26 J and 1,986 calls to drop it (ledger B-8).

    An ablation that also moved the organizer, the retrieved types, the
    temperatures or the store would not isolate the step, and its delta would be
    uninterpretable — the exact failure mode that cost Track 1 a $2.1 eval when a
    Nemori run silently inherited this same rewrite.
    """
    cfgmod = _load_configs()
    base, abl = cfgmod.get_config("amem"), cfgmod.get_config("amem_rawq")
    assert base.keyword_queries is True and abl.keyword_queries is False
    for field in ("memory_types", "role_temps", "per_type_k", "store", "run_ready"):
        assert getattr(base, field) == getattr(abl, field), field
    assert type(base.factory()[0]).__name__ == type(abl.factory()[0]).__name__ == "AMemOrganizer"


def test_amem_perhit_pins_the_cap_shape_and_leaves_other_arms_alone():
    """`amem_perhit` closes A-Mem's last read-path gap; nothing else may move.

    cap=11 reproduces upstream's k+1 at the eval's k=10 (`memory_layer.py:895`
    breaks after appending). The two fields travel together — a per-hit budget of
    5 is neither our shape nor upstream's — and every other arm must keep
    `link_expansion_cap=None`, which is what makes the runner's own
    `--expand-links` expression byte-identical for them.
    """
    cfgmod = _load_configs()
    ph = cfgmod.get_config("amem_perhit")
    assert (ph.link_expansion_cap, ph.link_expansion_per_hit) == (11, True)
    base = cfgmod.get_config("amem")
    for field in ("memory_types", "role_temps", "per_type_k", "store", "keyword_queries"):
        assert getattr(base, field) == getattr(ph, field), field
    # Only the arms that exist to move the cap may carry it; everything else must
    # stay None so the runner's own --expand-links expression is byte-identical.
    cap_arms = {"amem_perhit", "amem_rawq_perhit"}
    for name, cfg in cfgmod.CONFIGS.items():
        if name in cap_arms:
            assert (cfg.link_expansion_cap, cfg.link_expansion_per_hit) == (11, True), name
        else:
            assert cfg.link_expansion_cap is None, name
            assert cfg.link_expansion_per_hit is False, name
    # The 2x2 is only interpretable if each axis moves alone: the combined arm
    # must differ from `amem` in BOTH read knobs and in nothing else.
    combo = cfgmod.get_config("amem_rawq_perhit")
    assert (combo.keyword_queries, combo.link_expansion_per_hit) == (False, True)
    for field in ("memory_types", "role_temps", "per_type_k", "store"):
        assert getattr(base, field) == getattr(combo, field), field


def test_zep_config_takes_its_whole_read_path_from_the_recipe():
    """Zep's read path is a recipe table, so the arm must not assemble one itself.

    The paper presents three search functions and five rerankers as components,
    and §4.1 fixes which combination produced its numbers — BGE for reranking.
    Composing those knobs at the call site is how this project once built a
    hybrid upstream never ships (RRF fusion beside a BFS-ish GraphRecall), so
    the config takes `memory_types`, the lexical and BFS channels, `rrf_k`,
    `dense_min_score` and the reranker slot from one `SearchRecipe` object.

    `lexical_types` is the assertion that matters most: Zep gives its three
    subgraphs the BM25 channel and the raw turns none, the inverse of every
    other arm, and the runner used to hardcode `("episodic",)`. Any arm that
    does NOT carry the key must still inherit that hardcoded default, or this
    refactor silently re-channelled the whole campaign.
    """
    cfgmod = _load_configs()
    zep = cfgmod.get_config("zep_cross_encoder")
    assert type(zep.factory()[0]).__name__ == "ZepGraphOrganizer"
    assert zep.memory_types == ("facts", "entities", "communities")
    assert zep.per_type_k == dict.fromkeys(zep.memory_types, 10)
    assert zep.keyword_queries is False
    # ungated 2026-08-07: the conv0 pilot ran a complete ingest through this
    # entry, which is the only thing that can verify temps/k/store threading
    assert zep.run_ready is True

    store = dict(zep.store or {})
    assert store["lexical_types"] == ("facts", "entities", "communities")
    assert store["bfs_types"] == ("facts", "entities")  # communities have no BFS upstream
    assert store["rrf_k"] == 1  # upstream's rank_const, not the textbook 60
    assert store["dense_min_score"] == 0.6  # upstream DEFAULT_MIN_SCORE, not 0
    assert store["graph_expansion_cap"] == 0  # our GraphRecall must not double-serve φ_bfs
    assert store["overrides"] == {"reranker": "CrossEncoderReranker"}
    assert "bge-reranker" in store["reranker_params"]["model_name"]

    # The property is "no NON-Zep arm was re-channelled", so the exemption is
    # every Zep arm rather than this one name — the read-recipe sweep landed
    # four more, and narrowing the loop to the flagship would have turned a
    # guard about the other lineages into a guard about the roster's size.
    for name, cfg in cfgmod.CONFIGS.items():
        if not name.startswith("zep_"):
            assert "lexical_types" not in (cfg.store or {}), name


def test_zep_recipe_power_imports_its_statistics_and_never_compares_across_subgraph_counts():
    """Two properties this analysis must not lose.

    One: the bootstrap is imported from the expansion lane's `x1_power`, not
    restated. Two implementations of a confidence interval is how two numbers in
    one repository come to disagree about what 95% means.

    Two: the facts-only arms are never paired against the three-subgraph arms.
    Such a pair would differ in how many subgraphs are served, not in the recipe,
    and reporting it as a recipe effect would be the "two things at once" error
    the sweep was designed to avoid.
    """
    import importlib.util as ilu

    root = Path(__file__).resolve().parent.parent
    path = root / "scripts" / "repro" / "zep_recipe_power.py"
    spec = ilu.spec_from_file_location("zep_recipe_power", path)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    src = path.read_text()
    assert "x1_power" in src and "def paired_delta_ci" not in src

    facts_only = {"edge_rrf", "edge_episode_mentions"}
    for hi, lo, _what in mod.PAIRS:
        assert (hi in facts_only) == (lo in facts_only), (
            f"{hi} vs {lo} compares a facts-only arm against a three-subgraph arm"
        )
    # every arm the module scores must carry an anchor, or the fail-closed check
    # silently degrades to "no check" for the arm that lacks one
    assert set(mod.ARM_STEMS) == set(mod.HEADLINE_ANCHORS)


def _load_reconstruct():
    """Load the trace-reconstruction script the same flat-module way the other
    scripts/repro helpers are loaded in this file."""
    import importlib.util
    import sys

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "reconstruct_from_trace", root / "scripts" / "repro" / "reconstruct_from_trace.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trace_reconstruction_pairs_judges_to_their_own_question():
    """Off-by-one in the generate/judge pairing would shift every verdict onto a
    neighbouring question and still produce plausible output, so the pairing
    raises rather than assumes.

    The reconstruction only exists because a run can finish measuring and die
    before writing; its whole value is that its output is the run's, so a silent
    misalignment is worse than no reconstruction at all.
    """
    R = _load_reconstruct()

    def trace(tmp_path, calls):
        p = tmp_path / "t.jsonl"
        p.write_text("\n".join(json.dumps(c) for c in calls) + "\n")
        return p

    def gen(q, pred):
        return {
            "role": "generate",
            "messages": [{"role": "user", "content": f"ctx\nQuestion: {q}\nShort answer:"}],
            "response_text": pred,
        }

    def judge(gold, label):
        return {
            "role": "judge",
            "messages": [
                {
                    "role": "user",
                    "content": f"Question: q\nGold answer: {gold}\nGenerated answer: p\n",
                }
            ],
            "response_text": json.dumps({"label": label}),
        }

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # a judged question, then an UNjudged one (category 5), then judged
        turns = R.parse_trace(
            trace(
                tmp,
                [
                    gen("q1", "a1"),
                    judge("g1", "CORRECT"),
                    gen("q2", "a2"),
                    gen("q3", "a3"),
                    judge("g3", "WRONG"),
                ],
            )
        )
        assert [t["q_from_prompt"] for t in turns] == ["q1", "q2", "q3"]
        assert [R.verdict(t["judge_raw"]) for t in turns] == [True, None, False]

        with pytest.raises(SystemExit, match="no preceding generate"):
            R.parse_trace(trace(tmp, [judge("g", "CORRECT")]))
        with pytest.raises(SystemExit, match="second judge"):
            R.parse_trace(trace(tmp, [gen("q", "a"), judge("g", "CORRECT"), judge("g", "WRONG")]))


def test_trace_reconstruction_grades_a_verdict_exactly_as_the_live_judge_does():
    """`judge_answer` treats CORRECT and nothing else as True. A reconstruction
    that also accepted 'correct.' or 'Yes' would score higher than the run it
    claims to restore — the failure would look like a better result, which is
    the direction nobody double-checks."""
    R = _load_reconstruct()

    assert R.verdict(json.dumps({"label": "CORRECT"})) is True
    assert R.verdict(json.dumps({"label": "correct"})) is True  # upper()d, as upstream does
    assert R.verdict(json.dumps({"label": "WRONG"})) is False
    assert R.verdict(json.dumps({"label": "yes"})) is False
    assert R.verdict(None) is None
    assert R.verdict("not json") is None


def test_scoring_aggregation_has_exactly_one_implementation():
    """`evaluate` and the reconstruction must not each own a copy of the
    scoring arithmetic. Several copies with one of them read is the defect class
    this repo catalogs in upstream projects; it applies here too."""
    from agmem.bench import locomo as L

    assert callable(L.aggregate_cells)
    src = (Path(__file__).resolve().parent.parent / "src/agmem/bench/locomo.py").read_text()
    assert "def aggregate_cells" in src
    assert src.count("def agg(") == 0, "the closure came back — that is the second copy"

    rows = [{"f1": 1.0, "b1": 0.5, "j": True}, {"f1": 0.0, "b1": None, "j": False}]
    out = L.aggregate_cells(rows)
    assert out == {"f1": 50.0, "n": 2, "bleu1": 50.0, "bleu1_n": 1, "j_score": 50.0, "j_n": 2}


def test_mmr_scores_are_json_serializable_python_floats():
    """MMR's `(id, score)` pairs must not carry numpy dtypes.

    `np.linalg.norm` of a float32 array returns np.float32, and a Python float
    divided BY a numpy scalar is a numpy scalar — so casting only the dot-product
    numerator left the scores np.float32. Those scores ride into the per-question
    retrieval capture the harness records, and `json.dumps` has no encoder for
    them: a finished 1,986-question run died at artifact-write time with every
    model call already paid for.

    Dormant from 518f788 until the first arm that actually resolved to this class,
    which is why the sibling rerankers are asserted alongside it — the property is
    "no reranker leaks a dtype", not "MMR was fixed once".
    """
    import json as _json

    from agmem.retrieval.rerank import EpisodeMentionsReranker, MMRReranker, NoopReranker

    vectors = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]}
    candidates = [("a", 0.9), ("b", 0.5)]
    for reranker in (
        MMRReranker(lambda_=1.0),
        MMRReranker(lambda_=0.5),
        NoopReranker(),
        EpisodeMentionsReranker(),
    ):
        out = reranker.rerank([1.0, 0.0, 0.0], candidates, vectors, 2, meta={"a": {}, "b": {}})
        for _cid, score in out:
            assert type(score) is float, (type(reranker).__name__, type(score).__name__)
        _json.dumps(out)  # the call that actually died


def test_artifact_writer_survives_a_numpy_scalar_but_not_an_arbitrary_object():
    """An artifact writer must not be the thing that loses a completed run.

    By the time it runs the money is spent and the numbers are right, so a dtype
    the encoder lacks a rule for is the worst possible place to fail. The
    coercion stays narrow on purpose: a 0-d numpy scalar unwraps exactly via
    `.item()`, and anything else still raises rather than being stringified into
    an artifact that looks fine and is not.
    """
    H = _load_repro()
    np = pytest.importorskip("numpy")

    assert H.json_safe(np.float32(0.875)) == pytest.approx(0.875)
    assert H.json_safe(np.int64(7)) == 7
    with pytest.raises(TypeError):
        H.json_safe(object())
    with pytest.raises(TypeError):
        H.json_safe(np.zeros(3))  # not 0-d: a real bug, not a dtype wrapper


def test_every_memory_type_an_arm_can_serve_is_in_the_snapshot():
    """The snapshot must cover every type some arm retrieves from, or the durable
    record contradicts the run that produced it.

    This failed silently for the whole Zep campaign: `communities` is one of the
    three subgraphs its recipe serves, the ingest wrote 1,243 of them, and
    `*.memory.jsonl` recorded zero — the artifact described a two-subgraph arm.
    Driven off the config roster rather than a hand-kept list so the next arm
    that serves a new type cannot reintroduce the gap.
    """
    H = _load_repro()
    cfgmod = _load_configs()

    served = {t for cfg in cfgmod.CONFIGS.values() for t in cfg.memory_types}
    # episodic is dumped by its own branch (list_episodes), not from this tuple
    missing = served - set(H.SNAPSHOT_ITEM_TYPES) - {"episodic"}
    assert not missing, f"arms retrieve {sorted(missing)} but the snapshot never writes them"


def test_zep_read_recipe_sweep_arms_derive_everything_from_the_recipe_table():
    """Track 3's sweep arms: same ingest, different read recipe.

    The entry these siblings copy took 27,449 write calls and $4.71 to produce a
    store; a second recipe against that store costs an eval. That asymmetry is
    the whole reason Zep was picked for a recipe TABLE rather than a single
    pinned operating point, and it went uncollected for a campaign — so these
    assertions pin the derivation, not the values: an arm may not restate what
    the recipe already says, because two copies of a number are two chances to
    disagree.
    """
    cfgmod = _load_configs()
    from agmem.organizers.zep_graph import zep_search_recipe

    for key in cfgmod.ZEP_SWEEP_RECIPES:
        cfg = cfgmod.get_config(f"zep_{key}")
        recipe = zep_search_recipe(key)
        assert type(cfg.factory()[0]).__name__ == "ZepGraphOrganizer", key
        # derived, never restated
        assert cfg.memory_types == recipe.memory_types, key
        assert cfg.per_type_k == dict.fromkeys(recipe.memory_types, recipe.limit), key
        assert (cfg.store or {})["lexical_types"] == recipe.lexical_types, key
        assert (cfg.store or {})["bfs_types"] == recipe.bfs_types, key
        # shared with the flagship verbatim: these arms re-read ITS store, so a
        # write-side difference would quietly make them a different arm's read
        assert cfg.role_temps == cfgmod.ZEP_ROLE_TEMPS, key
        assert (cfg.store or {})["structured_reply_retries"] == 4, key
        assert cfg.keyword_queries is False, key
        # exact, not cautious: no ingest has been threaded through these entries
        assert cfg.run_ready is False, key

    # what each pair isolates — a sweep whose rows differ in two things at once
    # measures neither, so the comparable pairs are pinned here
    rrf = cfgmod.get_config("zep_rrf")
    mmr = cfgmod.get_config("zep_mmr")
    assert rrf.memory_types == mmr.memory_types == ("facts", "entities", "communities")
    assert (rrf.store or {})["bfs_types"] == (mmr.store or {})["bfs_types"] == ()
    assert "overrides" not in (rrf.store or {})  # NoopReranker keeps RRF fusion order
    assert (mmr.store or {})["overrides"] == {"reranker": "MMRReranker"}
    # upstream ships the diversity term OFF; this measures MMR as configured
    assert (mmr.store or {})["reranker_params"]["lambda_"] == 1.0

    edge_rrf = cfgmod.get_config("zep_edge_rrf")
    mentions = cfgmod.get_config("zep_edge_episode_mentions")
    assert edge_rrf.memory_types == mentions.memory_types == ("facts",)
    assert "overrides" not in (edge_rrf.store or {})
    assert (mentions.store or {})["overrides"] == {"reranker": "EpisodeMentionsReranker"}


def test_node_distance_is_left_out_of_the_sweep_because_it_would_measure_nothing():
    """`node_distance` is implemented and NOT wired, deliberately.

    Its reranker reorders by graph distance from a per-query centroid and is a
    no-op truncation without one; nothing in the LoCoMo harness supplies a
    centroid. Wiring it would produce a run stamped `NodeDistanceReranker` that
    actually measured entities-only fusion order — a recipe filed under a name
    it is not, which is the failure the `degradations` field was added to make
    impossible. Pinned as a test so a future sweep does not "complete the table"
    by adding the one entry whose absence is the finding.
    """
    cfgmod = _load_configs()
    from agmem.organizers.zep_graph import ZEP_SEARCH_RECIPES

    assert "node_distance" in ZEP_SEARCH_RECIPES  # implemented upstream-side
    assert "node_distance" not in cfgmod.ZEP_SWEEP_RECIPES
    assert "zep_node_distance" not in cfgmod.CONFIGS


def test_zep_write_temperature_is_the_paper_era_value_not_the_pinned_sha_one():
    """Zep writes at temperature 0, and the pinned SHA is the wrong place to read
    that from.

    graphiti @ 9140123 has `DEFAULT_TEMPERATURE = 1`. Dating the constant across
    pypi wheels puts the 0 -> 1 change between 0.18.9 (2025-08-19) and 0.19.0
    (2025-09-02), and 0.19.0 is the release that made `gpt-5-mini` the default
    model — GPT-5 reasoning models accept only temperature 1. The constant
    therefore tracks the default MODEL, not a view about extraction. At 0.5.1,
    live when the paper was submitted and running gpt-4o-mini as this arm does,
    the value is 0 on max_tokens 16384.

    Pinned as a test because the failure mode is silent: 1.0 would look like
    fidelity (it IS in the pinned clone) while making this the only arm in the
    campaign sampling its write path at full temperature.
    """
    cfgmod = _load_configs()
    zep = cfgmod.get_config("zep_cross_encoder")
    for role in ("extract", "distill"):
        assert zep.role_temps[role]["temperature"] == 0.0, role
        assert zep.role_temps[role]["max_tokens"] == 16384, role
    # not a lineage value — graphiti ships no answer generator; this is the
    # harness-neutral setting shared with nemori and mem0
    assert zep.role_temps["generate"]["temperature"] == 0.0


def test_runner_configs_construct_and_name_arms():
    cfgmod = _load_configs()
    CONFIGS, get_config = cfgmod.CONFIGS, cfgmod.get_config
    import pytest as _pytest

    assert set(CONFIGS) >= {"amem", "nemori_upstream", "nemori_merge085"}
    amem = get_config("amem")
    assert amem.memory_types == ("notes",)
    orgs = amem.factory()
    assert type(orgs[0]).__name__ == "AMemOrganizer"
    merged = get_config("nemori_merge085").factory()[0]
    # NemoriOrganizer has no `.merge_similarity` attribute (only `.params[...]`
    # and `._merger.similarity`) — see organizer.py:267-380; assert the knob
    # via `.params` rather than the brief's literal `.merge_similarity`.
    assert merged.params["merge_similarity"] == 0.85  # the B-3 enforced arm, live knob
    # Track 1: nemori is now run-ready, carrying the exact temps/k/store the
    # fidelity precheck's §7 table verified against exp_locomo_conv0.py
    # (NEMORI_TEMPS ~:350, NEMORI_STORE ~:360, k table).
    nemori_upstream = get_config("nemori_upstream")
    assert nemori_upstream.run_ready is True
    assert nemori_upstream.role_temps == {
        "extract": {"temperature": 0.2, "max_tokens": 4096},
        "distill": {"temperature": 0.7, "max_tokens": 2000},
        "generate": {"temperature": 0.0},
    }
    assert nemori_upstream.per_type_k == {"episodes": 10, "semantic": 20}
    assert nemori_upstream.store == {
        "overrides": {"vector_store": "QdrantVectorStore", "doc_store": "PostgresDocStore"}
    }
    # Fix round 2: Nemori's published read path is 0-LLM raw-question dense
    # retrieval (exp_locomo_conv0.py known-table 4th field, False for both
    # nemori entries) — A-Mem's LLM keyword-rewrite query must not leak in.
    assert nemori_upstream.keyword_queries is False
    nemori_merge085 = get_config("nemori_merge085")
    assert nemori_merge085.run_ready is True
    assert nemori_merge085.role_temps == nemori_upstream.role_temps
    assert nemori_merge085.per_type_k == nemori_upstream.per_type_k
    assert nemori_merge085.store == nemori_upstream.store
    assert nemori_merge085.keyword_queries is False
    amem = get_config("amem")
    assert amem.run_ready is True
    # amem's path stays byte-identical: all three Track-1 fields None, and
    # keyword_queries keeps its True default (amem's LLM keyword-rewrite).
    assert amem.role_temps is None
    assert amem.per_type_k is None
    assert amem.store is None
    assert amem.keyword_queries is True
    with _pytest.raises(KeyError):
        get_config("nope")


def test_amem_noevolve_arm_differs_from_amem_in_the_switch_alone_and_stays_gated():
    """The Table 3 w/o-evolution ablation arm (docs/14 "evolution ablation 주의"
    follow-up). Interpretable only if the switch is the ONLY difference from
    `amem` — same read path, temps, k and store, so any measured gap belongs to
    the skipped evolution call. run_ready flipped True on 2026-08-20: the gate
    attests runs, and the conv0 pilot survived intact (ADD-only op log,
    spend_capped=False — see the configs.py comment for the artifact)."""
    cfgmod = _load_configs()
    base = cfgmod.get_config("amem")
    arm = cfgmod.get_config("amem_noevolve")
    base_org, arm_org = base.factory()[0], arm.factory()[0]
    assert type(arm_org).__name__ == "AMemOrganizer"
    assert (base_org.evolve, arm_org.evolve) == (True, False)
    for field in ("memory_types", "role_temps", "per_type_k", "store", "keyword_queries"):
        assert getattr(base, field) == getattr(arm, field), field
    assert arm.run_ready is True


def test_stamp_k_temps_reflect_the_selected_configs_actual_values():
    """_stamp's k/temps must carry the SELECTED config's real values — amem's
    scalar k / write-generate-cat5 temps stay byte-identical (the fallback
    path, since amem carries neither per_type_k nor role_temps), while
    nemori's stamp must carry its per-type k dict and extract/distill/generate
    role temps instead of the amem-shaped defaults. scripts/repro/
    aggregate_headline.py reads both fields by these exact shapes for
    grouping/reporting — a nemori run stamped with amem's shape would silently
    mis-group there."""
    repro = _load_repro()
    spec = repro.get_model("gpt-4o-mini")

    def _args(config):
        return SimpleNamespace(
            model="gpt-4o-mini",
            runs=1,
            endpoint=None,
            embedder="all-MiniLM-L6-v2",
            k=10,
            eval_mode="wujiang",
            expand_links="off",
            conv="0",
            workers=1,
            config=config,
            eval_only=False,
            data_dir=None,
        )

    amem_stamp = repro._stamp(_args("amem"), spec, None, "t0", "t1", 5)
    assert amem_stamp["k"] == 10  # unchanged from before this fix — no per_type_k on amem
    assert amem_stamp["temps"] == {"write": 0.7, "generate": 0.7, "cat5": repro.CAT5_TEMPERATURE}
    assert amem_stamp["keyword_queries"] is True  # unchanged — amem's LLM query rewrite
    # The link-expansion knobs join the stamp for the same reason keyword_queries
    # is there: `expand_links` records only THAT the step ran, so without these an
    # amem_perhit artifact reads exactly like an amem one and the arm that
    # produced a number is recoverable only by resolving the config name against
    # a particular commit of configs.py.
    assert amem_stamp["link_expansion_cap"] is None  # amem defers to the runner's own 5
    assert amem_stamp["link_expansion_per_hit"] is False
    perhit_stamp = repro._stamp(_args("amem_perhit"), spec, None, "t0", "t1", 5)
    assert perhit_stamp["link_expansion_cap"] == 11
    assert perhit_stamp["link_expansion_per_hit"] is True

    nemori_stamp = repro._stamp(_args("nemori_upstream"), spec, None, "t0", "t1", 5)
    assert nemori_stamp["k"] == {"episodes": 10, "semantic": 20}
    assert nemori_stamp["temps"] == {
        "extract": {"temperature": 0.2, "max_tokens": 4096},
        "distill": {"temperature": 0.7, "max_tokens": 2000},
        "generate": {"temperature": 0.0},
    }
    # Fix round 2: nemori must NOT inherit amem's LLM keyword-rewrite query —
    # its published read path is raw-question dense retrieval (0 extra calls).
    assert nemori_stamp["keyword_queries"] is False


def test_mem0_config_fields_are_all_threaded():
    """Every RunnerConfig field decided, none left to a default that would
    silently inherit A-Mem's read protocol (the defect Track 1 fixed at 7d9b64e)."""
    cfg = _load_configs().get_config("mem0_v0194")
    assert cfg.memory_types == ("semantic",)
    assert cfg.keyword_queries is False  # upstream searches the raw question
    assert cfg.per_type_k == {"semantic": 30}  # Makefile --top_k 30, not the class default 10
    assert cfg.role_temps["distill"]["max_tokens"] == 2000  # BaseLlmConfig default
    assert cfg.role_temps["generate"]["temperature"] == 0.0  # harness answer call
    assert cfg.run_ready is True
    org = cfg.factory()[0]
    assert type(org).__name__ == "Mem0Organizer"
    assert org.batch_size == 2  # paper-harness shape (add.py:46)
    assert org.top_k == 5  # upstream's hardcoded limit=5


def test_rb_upstream_preset_is_bound_to_the_read_recipe_constant():
    """round-12 #10: the preset must DERIVE the operating point, never restate it.

    `RB_READ_RECIPE` is the citable form of upstream's WebArena read loop
    (top-1 experience by task-query embedding, no item-level channel); the
    preset is how a run selects it by name. Binding them here is the whole
    point — if the recipe is ever re-read and its numbers change, a preset that
    silently kept the old literals would be the drift the constant exists to
    prevent. The `strategies` pair is asserted from BOTH sides for the same
    reason: `strategies_topk: 0` is expressed as the type's ABSENCE from
    memory_types, so a future nonzero topk would leave no trace in the preset
    unless this test forces it to be revisited.

    run_ready is False and stays False until an agent-bench harness exists —
    RB's write path is `add_task_result`-shaped, not conversation ingest, so
    this arm is a citable operating point, not a runnable row (ledger C-4).
    """
    from agmem.organizers.reasoning_bank.organizer import RB_READ_RECIPE

    cfg = _load_configs().get_config("rb_upstream")
    assert cfg.per_type_k["experiences"] == RB_READ_RECIPE["experiences_topk"]
    assert "strategies" not in cfg.memory_types
    assert RB_READ_RECIPE["strategies_topk"] == 0
    orgs = cfg.factory()
    assert len(orgs) == 1
    assert type(orgs[0]).__name__ == "ReasoningBankOrganizer"
    assert cfg.run_ready is False


def test_op_log_artifact_captures_every_op_including_noop(tmp_path):
    """The evolution log becomes a durable artifact.

    Track 2's op-structure claim (ADD/UPDATE/DELETE/NOOP proportions) is not
    measurable from the memory snapshot: the snapshot holds final items, so an
    UPDATE is invisible, a DELETEd item is absent by construction, and a NOOP
    leaves no trace at all. Without this dump the headline would be measured
    from nothing.
    """
    from agmem.core.ops import MemoryOp, OpType

    H = _load_repro()
    mem = AgenticMemory(namespace="ops", organizers=["passthrough"], embedder=FakeEmbedder(dim=64))
    try:
        mem._apply_ops(
            [
                MemoryOp(
                    op=OpType.ADD,
                    target_type="semantic",
                    target_id="a",
                    payload={"id": "a", "content": "x"},
                ),
                MemoryOp(op=OpType.NOOP, target_type="semantic", target_id="a", payload={}),
            ],
            actor="mem0",
        )
        out = tmp_path / "ops.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            counts = H.dump_op_log(mem, 0, fh)
    finally:
        mem.close()

    lines = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert [ln["op"] for ln in lines] == ["ADD", "NOOP"]
    assert lines[0]["conv"] == 0 and lines[0]["actor"] == "mem0" and "seq" in lines[0]
    assert lines[0]["target_type"] == "semantic" and lines[0]["target_id"] == "a"
    assert "t_transaction" in lines[0] and lines[0]["payload"]["content"] == "x"
    assert counts == {"ADD:semantic": 1, "NOOP:semantic": 1}


def test_op_log_pages_past_the_ops_since_limit(tmp_path):
    """`ops_since` caps at `limit` rows per call (default 10_000).

    A Mem0 conversation emits up to ~10 ops per decision call over ~200 adds, so
    a single unpaged read is within one order of magnitude of that cap — and a
    silent truncation here would understate exactly the counts the claim rests
    on. Verified against a small explicit limit rather than by generating 10k
    ops, so the test stays fast while still exercising the paging loop.
    """
    from agmem.core.ops import MemoryOp, OpType

    H = _load_repro()
    mem = AgenticMemory(namespace="ops", organizers=["passthrough"], embedder=FakeEmbedder(dim=64))
    try:
        mem._apply_ops(
            [
                MemoryOp(
                    op=OpType.ADD,
                    target_type="semantic",
                    target_id=f"i{i}",
                    payload={"id": f"i{i}", "content": "x"},
                )
                for i in range(25)
            ],
            actor="mem0",
        )
        out = tmp_path / "ops.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            counts = H.dump_op_log(mem, 0, fh, page=4)
    finally:
        mem.close()

    assert counts == {"ADD:semantic": 25}
    assert len(out.read_text(encoding="utf-8").splitlines()) == 25


def test_organizer_discards_are_folded_into_the_drops_block():
    """An organizer's `discarded` counters belong beside structured-output drops:
    both are work that was paid for and thrown away, and a quote that ignores
    them under-reports waste."""
    H = _load_repro()
    mem = AgenticMemory(namespace="d", organizers=["mem0"], embedder=FakeEmbedder(dim=64))
    try:
        mem.organizers[0].discarded = {"hallucinated_id": 2, "empty_text": 1}
        merged_budget: dict = {}
        merged_drops: dict = {}
        H._merge_budget(merged_budget, merged_drops, mem)
    finally:
        mem.close()
    assert merged_drops["mem0/hallucinated_id"] == 2
    assert merged_drops["mem0/empty_text"] == 1


def test_organizer_without_discarded_is_unaffected():
    H = _load_repro()
    mem = AgenticMemory(namespace="d", organizers=["passthrough"], embedder=FakeEmbedder(dim=64))
    try:
        merged_drops: dict = {}
        H._merge_budget({}, merged_drops, mem)
    finally:
        mem.close()
    assert merged_drops == {}


def test_conv_is_done_rejects_a_conv_that_dropped_structured_output(tmp_path, monkeypatch):
    """A finished-but-degraded ingest must not count as done.

    Discovered live on 2026-08-04: under memory pressure the box started failing
    DNS, one conversation crashed outright and another COMPLETED while losing 15
    structured-output calls (`drops={"extract":3,"distill":12}`). The second is
    the dangerous one — it wrote every artifact, so an existence check accepted
    it, and its capacity (185 semantic / 84 episodes) silently entered a
    comparison whose baseline arm has zero drops in all ten conversations. A
    partial ingest is a wrong measurement, not a cheap one.
    """
    P = _load_parallel()
    sd = tmp_path / "repro-conv3"
    sd.mkdir()
    (sd / "x").write_text("store")
    monkeypatch.setattr(P.H, "OUT", tmp_path)

    sp = tmp_path / f"{P._model_safe('gpt-4o-mini')}_conv3_ingest_t_c3.json"
    clean = {"drops": {}, "llm_budget": {"extract": {"calls": 5, "errors": 0}}}
    sp.write_text(json.dumps(clean))
    assert P.conv_is_done("gpt-4o-mini", str(tmp_path), 3, "_t") is True

    sp.write_text(json.dumps({**clean, "drops": {"extract": 3, "distill": 12}}))
    assert P.conv_is_done("gpt-4o-mini", str(tmp_path), 3, "_t") is False

    sp.write_text(json.dumps({"drops": {}, "llm_budget": {"extract": {"calls": 5, "errors": 2}}}))
    assert P.conv_is_done("gpt-4o-mini", str(tmp_path), 3, "_t") is False


def test_merge_ingest_summaries_carries_op_counts():
    """The parallel path must not drop the write path's own record of itself.

    `finalize_combined` promises the combined summary is "byte-for-byte the pair
    the sequential `--conv all --ingest-only` emits". The sequential path sets
    `op_counts` (exp_amem_repro.py: `combined["op_counts"] = op_counts or None`)
    — the per-op tally that is the ONLY artifact able to show an UPDATE, a
    DELETE, or a NOOP, since a snapshot shows just what survived. The merge
    summed every other block and silently omitted this one, so every campaign
    ingested through the orchestrator wrote a combined summary reporting no
    evolution at all, while its per-conv summaries carried the counts intact.

    Found on the Mem0 Stage C ingest (2026-08-06), where `op_counts` is the
    headline: 79.0% of 33,167 semantic decisions were NOOP. The same omission
    had already blanked the field for the A-Mem and both Nemori arms.
    """
    P = _load_parallel()
    merged = P.merge_ingest_summaries(
        [
            {
                "llm_budget": {"extract": {"calls": 2, "tokens_in": 10, "tokens_out": 1}},
                "op_counts": {"ADD:semantic": 3, "NOOP:semantic": 7},
            },
            {
                "llm_budget": {"extract": {"calls": 1, "tokens_in": 5, "tokens_out": 1}},
                "op_counts": {"NOOP:semantic": 2, "DELETE:semantic": 1},
            },
        ],
        "gpt-4o-mini",
    )
    assert merged["op_counts"] == {"ADD:semantic": 3, "NOOP:semantic": 9, "DELETE:semantic": 1}

    # an arm whose organizers log nothing collapses to None, matching the
    # sequential path's `op_counts or None` rather than an empty dict.
    bare = P.merge_ingest_summaries([{"llm_budget": {}}, {"llm_budget": {}}], "gpt-4o-mini")
    assert bare["op_counts"] is None


def test_conv_is_done_accepts_organizer_discards_but_still_rejects_transport_drops(
    tmp_path, monkeypatch
):
    """An organizer's judged non-application is not lost work, and must not gate.

    `_merge_budget` folds two different things into one `drops` block: the
    structured-output layer's parse failures, keyed by ROLE, and each
    organizer's own discards, namespaced `"{organizer}/{reason}"`. Pooling them
    is right for cost accounting — both are calls that were paid for — but they
    are opposite signals of run health. A role-keyed drop means the harness
    lost a response it had already bought: the 2026-08-04 failure this gate
    exists for. An organizer-keyed discard means the response arrived intact
    and the organizer decided not to apply it.

    For Mem0 the distinction decides whether the run is affordable at all.
    `mem0/hallucinated_id` fires on every conversation by construction —
    upstream maps memory ids onto integers precisely because the model invents
    UUIDs otherwise — so gating on it marks every conversation not-done, and
    `_run_one` then wipes the store and re-ingests up to `--retries` times
    before reporting FAILED. The conv0 pilot measured 6 such discards, which
    would have turned a $1.74 nine-conversation ingest into $5.22 with no
    usable result.
    """
    P = _load_parallel()
    sd = tmp_path / "repro-conv3"
    sd.mkdir()
    (sd / "x").write_text("store")
    monkeypatch.setattr(P.H, "OUT", tmp_path)
    sp = tmp_path / f"{P._model_safe('gpt-4o-mini')}_conv3_ingest_t_c3.json"
    clean = {"drops": {}, "llm_budget": {"extract": {"calls": 5, "errors": 0}}}

    # measured on the conv0 pilot (2026-08-05): a clean Mem0 ingest, 420 calls,
    # zero LLM errors, six ids the decision step hallucinated and we refused.
    sp.write_text(json.dumps({**clean, "drops": {"mem0/hallucinated_id": 6}}))
    assert P.conv_is_done("gpt-4o-mini", str(tmp_path), 3, "_t") is True

    sp.write_text(json.dumps({**clean, "drops": {"mem0/empty_text": 2}}))
    assert P.conv_is_done("gpt-4o-mini", str(tmp_path), 3, "_t") is True

    # a transport drop alongside organizer discards still fails the gate — the
    # namespaced keys must not mask a role-keyed one sharing the block.
    sp.write_text(json.dumps({**clean, "drops": {"mem0/hallucinated_id": 6, "distill": 1}}))
    assert P.conv_is_done("gpt-4o-mini", str(tmp_path), 3, "_t") is False


def test_worker_command_passes_the_config_through():
    """Without this the driver is A-Mem-only: every conv would silently ingest
    with the default `amem` organizer while the caller asked for nemori."""
    P = _load_parallel()
    args = SimpleNamespace(
        model="gpt-4o-mini",
        endpoint="e",
        embedder="text-embedding-3-small",
        expand_links="off",
        data_dir="/tmp/x",
        tag_suffix="_t",
        config="nemori_upstream",
    )
    cmd = P.worker_cmd(args, 3)
    assert "--config" in cmd
    assert cmd[cmd.index("--config") + 1] == "nemori_upstream"
    assert cmd[cmd.index("--embedder") + 1] == "text-embedding-3-small"


def test_artifact_names_carry_the_config_segment_like_the_harness():
    """The orchestrator must compute the SAME filenames the harness writes.

    exp_amem_repro appends a non-default --config to the model part of every
    artifact name. An orchestrator that omitted it would look for files nobody
    wrote: conv_is_done would answer False for conversations that were already
    ingested, so a resume or retry would re-ingest and RE-PAY for all of them,
    and the merge step would then fail on missing files.
    """
    P = _load_parallel()
    assert P._model_safe("gpt-4o-mini") == "gpt-4o-mini"  # amem default unperturbed
    assert P._model_safe("gpt-4o-mini", "amem") == "gpt-4o-mini"
    assert P._model_safe("gpt-4o-mini", "nemori_upstream") == "gpt-4o-mini_nemori_upstream"
    p = P.per_conv_summary_path("gpt-4o-mini", 3, "_e3sA", "nemori_upstream")
    assert p.name == "gpt-4o-mini_nemori_upstream_conv3_ingest_e3sA_c3.json"


def test_worker_log_path_is_per_conv_and_under_the_durable_logs_dir():
    P = _load_parallel()
    p = P.worker_log_path("gpt-4o-mini", 3, "_e3sB", "nemori_merge085")
    assert p.parent.name == "logs"  # docs/14 keeps results/repro/logs/ git-tracked
    assert p.name == "gpt-4o-mini_nemori_merge085_conv3_ingest_e3sB_c3.log"


def test_worker_output_is_persisted_not_discarded(tmp_path, monkeypatch):
    """The worker's stdout carries the ONLY copy of some measurements.

    Nemori logs its merge-candidate similarity scores on an INFO channel
    (`organizers/nemori/stages.py`), and that distribution is what quantifies
    whether the 0.85 threshold is embedder-relative — the design risk the Track 1
    precheck refused to ship without a mitigation. `subprocess.run(...,
    capture_output=True)` collected it and then dropped everything but the last
    line of a FAILURE, so a successful run threw the whole channel away and the
    standing instruction to persist it was silently unmet.
    """
    P = _load_parallel()
    monkeypatch.setattr(P.H, "OUT", tmp_path)
    (tmp_path / "logs").mkdir()

    captured = {}

    class _Proc:
        returncode = 0
        stdout = "nemori: merge candidate scores namespace=x threshold=0.85 hits=[0.91, 0.72]\n"
        stderr = ""

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(P.subprocess, "run", fake_run)
    monkeypatch.setattr(P, "conv_is_done", lambda *a, **k: "call" not in captured or True)
    calls = {"n": 0}

    def done(*a, **k):
        calls["n"] += 1
        return calls["n"] > 1  # not done before the run, done after

    monkeypatch.setattr(P, "conv_is_done", done)
    args = SimpleNamespace(
        model="gpt-4o-mini",
        endpoint="e",
        embedder="text-embedding-3-small",
        expand_links="off",
        data_dir=str(tmp_path / "store"),
        tag_suffix="_e3sB",
        config="nemori_merge085",
        retries=0,
    )
    (tmp_path / "store" / "repro-conv3").mkdir(parents=True)
    conv, ok, note = P._run_one(args, 3)
    assert ok
    log = P.worker_log_path("gpt-4o-mini", 3, "_e3sB", "nemori_merge085")
    assert log.exists(), "worker output must be written to a durable log"
    assert "merge candidate scores" in log.read_text()


def test_runner_configures_the_agmem_log_channel_so_info_diagnostics_survive():
    """A `logger.info` in the library reaches the driver's captured output.

    `exp_amem_repro` configured logging NOWHERE, so Python's default WARNING
    root level silently dropped every INFO record — including Nemori's
    merge-candidate scores, whose call site asserts in a comment that "a run at
    the library's default logging config captures this at INFO". It did not.
    The scores were the only evidence for whether the 0.85 threshold is
    embedder-relative, and a run could complete looking perfectly successful
    while recording none of them.
    """
    import logging as _logging

    H = _load_repro()
    agmem_logger = _logging.getLogger("agmem")
    prev_level, prev_handlers = agmem_logger.level, list(agmem_logger.handlers)
    try:
        agmem_logger.handlers.clear()
        agmem_logger.setLevel(_logging.NOTSET)
        H.configure_logging("INFO")
        assert agmem_logger.level == _logging.INFO
        assert agmem_logger.handlers, "a handler must be attached, not just a level"
        # scoped to agmem: turning the ROOT logger to INFO would flood the
        # captured log with httpx/openai per-request chatter
        assert _logging.getLogger().level != _logging.INFO or prev_level == _logging.INFO
    finally:
        agmem_logger.handlers[:] = prev_handlers
        agmem_logger.setLevel(prev_level)


def test_configure_logging_is_idempotent():
    """The runner may be imported more than once in one process (the test
    harness does exactly that); duplicate handlers would double every line."""
    import logging as _logging

    H = _load_repro()
    agmem_logger = _logging.getLogger("agmem")
    prev_level, prev_handlers = agmem_logger.level, list(agmem_logger.handlers)
    try:
        agmem_logger.handlers.clear()
        H.configure_logging("INFO")
        H.configure_logging("INFO")
        assert len(agmem_logger.handlers) == 1
    finally:
        agmem_logger.handlers[:] = prev_handlers
        agmem_logger.setLevel(prev_level)


def test_pilot_override_is_bounded_to_one_conversation():
    """`run_ready=False` blocks an arm whose temps/k/store threading has never
    survived a real run — but a pilot is HOW that threading gets verified, so the
    gate needs a deliberate way through that is not "flip the gate off".

    The override is bounded to a single conversation on purpose. Flipping
    `run_ready` to True would open the pilot and the full campaign in one move,
    and the full campaign is the spend the gate exists to stop. Bounded this way,
    the escape hatch cannot become the thing it was guarding against.

    Driven against a fabricated entry, not a real arm: the first version of this
    test used whichever config happened to be gated, and broke the moment that
    arm passed its pilot — testing the guard through today's roster made the
    roster part of the guard's contract.
    """
    H = _load_repro()
    gated = SimpleNamespace(run_ready=False)
    ready = SimpleNamespace(run_ready=True)

    with pytest.raises(SystemExit, match="not run-ready"):
        H.check_run_ready(gated, "x", allow_unverified=False, conv="0")
    with pytest.raises(SystemExit, match="single-conversation pilot only"):
        H.check_run_ready(gated, "x", allow_unverified=True, conv="all")
    # the pilot itself is allowed, and a ready arm is never touched
    assert H.check_run_ready(gated, "x", allow_unverified=True, conv="0") is None
    assert H.check_run_ready(ready, "x", allow_unverified=False, conv="all") is None


def test_eval_only_is_exempt_from_the_pilot_bound_but_ingest_still_is_not():
    """An --eval-only run reloads a store somebody already paid for and issues no
    write calls, so it cannot incur the spend the gate names.

    This is what makes a read-recipe sweep possible: Zep's sibling arms re-read
    the cross-encoder store for an eval apiece, and bounding them to one
    conversation would charge the campaign's own structure for a risk it does
    not carry. The exemption has to stay narrow, though — the same gated entry
    must still be refused when it would INGEST, because that is the threading
    `run_ready` attests and these arms have never piloted it.
    """
    H = _load_repro()
    gated = SimpleNamespace(run_ready=False)

    assert H.check_run_ready(gated, "x", allow_unverified=False, conv="all", eval_only=True) is None
    with pytest.raises(SystemExit, match="not run-ready"):
        H.check_run_ready(gated, "x", allow_unverified=False, conv="all", eval_only=False)


def test_parallel_orchestrator_refuses_a_multi_conv_pilot_before_spawning():
    """Same bound, enforced by the orchestrator too. The worker would refuse each
    conversation anyway, but only after a subprocess spawn apiece; here it costs
    nothing and the message arrives once."""
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS / "repro" / "ingest_parallel.py"),
            "--data-dir",
            "/tmp/nope",
            "--config",
            "zep_cross_encoder",
            "--convs",
            "0-3",
            "--allow-unverified-config",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "single-conversation pilot only" in (proc.stderr + proc.stdout)


def test_drop_budget_is_proportional_and_still_rejects_the_incident():
    """The clean-ingest bar has to separate two things a zero-drop gate cannot.

    It was set against a host-failure run that lost 15 structured calls out of
    ~1,180 — a corrupted measurement. The 2026-08-07 Zep pilot lost ONE out of
    2,177, which is ordinary model non-determinism, and paid for the whole
    conversation twice because the gate could not tell them apart. A
    proportional budget separates them: at 0.1% the incident is still rejected
    and the pilot passes.
    """
    P = _load_parallel()
    incident = {"llm_budget": {"extract": {"calls": 590}, "distill": {"calls": 590}}}
    pilot = {"llm_budget": {"extract": {"calls": 943}, "distill": {"calls": 1234}}}

    assert P.drop_budget(incident, 0.001) == 1  # 15 drops > 1 -> still rejected
    assert P.drop_budget(pilot, 0.001) == 2  # 1 drop <= 2 -> accepted

    # embed is not a structured call and cannot drop this way
    with_embed = {"llm_budget": {**pilot["llm_budget"], "embed": {"calls": 40000}}}
    assert P.drop_budget(with_embed, 0.001) == P.drop_budget(pilot, 0.001)

    # default is the historical bar, exactly
    assert P.drop_budget(pilot, 0.0) == 0


def test_conv_is_done_tolerance_never_covers_llm_errors(tmp_path):
    """A drop is a reply that arrived and would not parse. An error is a call
    that never completed — the host/transport signal the gate was built for — so
    no drop tolerance may excuse one."""
    P = _load_parallel()
    summary = P.per_conv_summary_path("gpt-4o-mini", 0, "_t", "zep_cross_encoder")
    summary.parent.mkdir(parents=True, exist_ok=True)
    store = P.store_dir_for(str(tmp_path), 0)
    store.mkdir(parents=True, exist_ok=True)
    (store / "x.db").write_text("x")
    try:
        summary.write_text(
            json.dumps(
                {
                    "drops": {"distill": 1},
                    "llm_budget": {
                        "extract": {"calls": 943, "errors": 0},
                        "distill": {"calls": 1234, "errors": 0},
                    },
                }
            )
        )
        args = ("gpt-4o-mini", str(tmp_path), 0, "_t", "zep_cross_encoder")
        assert P.conv_is_done(*args, 0.0) is False  # historical bar: 1 drop fails
        assert P.conv_is_done(*args, 0.001) is True  # proportional bar: passes

        summary.write_text(
            json.dumps(
                {
                    "drops": {},
                    "llm_budget": {
                        "extract": {"calls": 943, "errors": 0},
                        "distill": {"calls": 1234, "errors": 1},
                    },
                }
            )
        )
        assert P.conv_is_done(*args, 0.5) is False  # an ERROR is never tolerated
    finally:
        summary.unlink(missing_ok=True)


def test_structured_caller_reply_retries_default_is_unchanged_and_overridable():
    """Raising the correction-turn budget must not silently re-time every arm
    measured before it existed: the default stays 1, and an explicit
    `max_retries` still wins over the instance default."""
    from agmem.llm.structured import StructuredCaller

    assert StructuredCaller(client=None, use_guided_json=False).reply_retries == 1
    assert StructuredCaller(client=None, use_guided_json=False, reply_retries=4).reply_retries == 4


def test_zep_raises_reply_retries_because_its_call_volume_demands_it():
    cfgmod = _load_configs()
    zep = cfgmod.get_config("zep_cross_encoder")
    assert (zep.store or {})["structured_reply_retries"] == 4
    # and no arm measured before this budget existed has its timing touched.
    # Zep's read-recipe siblings share it on purpose (they re-read ITS store, so
    # they must not differ on the write side); the property being guarded is
    # that the OTHER lineages were left alone, not that exactly one name carries
    # the key.
    for name, cfg in cfgmod.CONFIGS.items():
        if not name.startswith("zep_"):
            assert "structured_reply_retries" not in (cfg.store or {}), name


def test_run_stamp_records_the_reranker_and_any_degradation():
    """A downgraded read path must be visible in the artifact.

    Capability resolution substitutes a working adapter when the configured one
    is unavailable, so a run that lost its reranker still finishes and still
    looks healthy — the only difference is that it measured a different read
    path than its label claims. Zep makes this concrete: its arm is named for
    upstream's cross-encoder recipe, and a silent substitution turns it into the
    RRF-order recipe, a different upstream config with its own identity. Neither
    field was stamped before 2026-08-07, so nothing downstream could tell.
    """
    from agmem.bench import stamp as bench_stamp

    mem = SimpleNamespace(
        config=SimpleNamespace(profile="lite"),
        embedder=SimpleNamespace(name="text-embedding-3-small"),
        vector_store=object(),
        organizers=[SimpleNamespace(name="zep_graph")],
        reranker=SimpleNamespace(),
        _degradations=["[reranker] configured 'CrossEncoderReranker' unavailable; falling back"],
    )
    s = bench_stamp.run_stamp(mem, model="gpt-4o-mini", judge="gpt-4o-mini", runs=1)
    assert s["reranker"] == "SimpleNamespace"
    assert s["degradations"] and "reranker" in s["degradations"][0]

    # a clean run records an empty list, not a missing key — absence of the key
    # would be indistinguishable from an older artifact that never had it
    clean = SimpleNamespace(
        config=SimpleNamespace(profile="lite"),
        embedder=SimpleNamespace(name="e"),
        vector_store=object(),
        organizers=[],
        reranker=None,
        _degradations=[],
    )
    s2 = bench_stamp.run_stamp(clean, model="m", judge="j", runs=1)
    assert s2["degradations"] == [] and s2["reranker"] is None


def test_ingest_only_does_not_build_the_configured_reranker(tmp_path):
    """An ingest never reranks, so it must not pay to construct one.

    No organizer calls the retrieval pipeline — they read the vector and doc
    stores directly — so the reranker slot is resolved, built, and never
    touched for the entire write phase. Free for six of the seven rerankers and
    not for `CrossEncoderReranker`, which loads its weights in `__init__`:
    measured 2026-08-07, two Zep ingest workers held 3.4 GB of a 6 GB GPU for
    hours without issuing one rerank call.
    """
    H = _load_repro()
    args = SimpleNamespace(
        config="zep_cross_encoder", data_dir=str(tmp_path), expand_links="off", ingest_only=True
    )
    mem = H.build_memory(args, FakeEmbedder(dim=16), 0, {})
    try:
        assert type(mem.reranker).__name__ == "NoopReranker"
    finally:
        mem.close()

    # ...and an eval run still gets the arm's real reranker slot
    eval_args = SimpleNamespace(
        config="zep_cross_encoder", data_dir=str(tmp_path), expand_links="off", ingest_only=False
    )
    cfgmod = _load_configs()
    assert (cfgmod.get_config(eval_args.config).store or {})["overrides"]["reranker"] == (
        "CrossEncoderReranker"
    )


def test_stamp_records_the_read_path_through_the_runners_actual_call(tmp_path):
    """The reranker/degradation fields must survive the path the HARNESS uses.

    This is the finding that produced them being pinned this way. `run_stamp`
    grew the fields under `if memory is not None`, and a test that called
    `run_stamp` directly with a fabricated memory passed — while the runner
    passes `memory=None` for every stamp it writes, because a run spans many
    conversations. The fields were dead in the only place they mattered, and the
    artifact from a live campaign had neither key. Drive `_stamp` here, not
    `run_stamp`.
    """
    H = _load_repro()

    class _Args(SimpleNamespace):
        # `_stamp` reads a long tail of CLI fields this test does not care
        # about; anything unset is None rather than an AttributeError, so the
        # test stays about the read-path fields instead of tracking the parser.
        def __getattr__(self, name):
            return None

    args = _Args(
        config="zep_cross_encoder",
        data_dir=str(tmp_path),
        expand_links="off",
        ingest_only=True,
        model="gpt-4o-mini",
        endpoint="https://api.openai.com/v1",
        embedder="fake",
        k=10,
        eval_mode="ours",
        runs=1,
        conv="0",
    )
    mem = H.build_memory(args, FakeEmbedder(dim=16), 0, {})
    mem.close()
    assert args._read_path["reranker"] == "NoopReranker"
    assert args._read_path["reranker_skipped_ingest_only"] is True

    stamp = H._stamp(args, H.get_model("gpt-4o-mini"), "sha", "t0", "t1", None)
    assert stamp["reranker"] == "NoopReranker"
    assert stamp["reranker_skipped_ingest_only"] is True

    # The field must be PRESENT and a list. Not empty: whether anything degraded
    # is a property of the machine, and CI has no `kuzu`, so the graph slot
    # legitimately falls back and says so. Asserting emptiness pinned this host's
    # installed packages as the contract and, worse, asserted that a field whose
    # entire job is recording degradations never records one. What this arm can
    # claim is narrower and is what matters here: nothing degraded the RERANKER,
    # because the ingest path chose NoopReranker deliberately rather than losing
    # one it wanted.
    assert isinstance(stamp["degradations"], list)
    assert not [d for d in stamp["degradations"] if "reranker" in d]


def test_finer_paired_imports_its_statistics_and_declines_to_interval_a_clustered_rate():
    """Track 5's paired comparison must not grow a second confidence interval.

    Two properties, both load-bearing. It IMPORTS `paired_delta_ci` from the
    expansion lane rather than restating it — the rule the Zep sweep follows,
    because two CI implementations in one repository is how two numbers come to
    disagree about what "95%" means. And it refuses to put an interval on
    `tag_accuracy`: upstream's published metric is a per-question rate over four
    tags that share a filing excerpt, so the resampling unit is the question and
    the statistic is a float, which the imported bool-only function cannot do.
    Declining is the honest outcome; quietly bootstrapping over TAGS would treat
    clustered observations as independent and report an interval too narrow.

    A third property once the module grew a third arm: an arm stopped by its
    spend cap is shorter than the reference, and every comparison must state the
    n it actually paired. Comparing an arm's 441 answers against base's first
    375 would be an unpaired difference wearing a paired interval.
    """
    repo = Path(__file__).resolve().parent.parent
    src = (repo / "scripts" / "repro" / "finer_paired.py").read_text(encoding="utf-8")
    assert "x1_power" in src and "paired_delta_ci" in src
    # no home-grown resampler
    assert "np.percentile" not in src, "the percentile bootstrap must be imported, not restated"
    assert "rng.integers" not in src, "resampling belongs to x1_power"
    assert "tag_accuracy_interval" in src
    out = repo / "results" / "repro" / "finer_paired.json"
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
        ref = data["reference"]
        assert data["comparisons"], "a paired file with no comparison in it is not a comparison"
        for name, cmp in data["comparisons"].items():
            assert cmp["tag_accuracy_interval"] is None
            assert "sample_accuracy_paired" in cmp
            # the pairing: n_compared is the prefix both arms answered, and the
            # bootstrap ran over exactly that many questions
            expected = min(data["anchors"][ref]["n"], data["anchors"][name]["n"])
            assert cmp["n_compared"] == expected
            assert cmp["sample_accuracy_paired"]["n"] == expected


def test_finer_call_counts_come_from_the_trace_not_the_resumed_summary(tmp_path, monkeypatch):
    """A resumed arm's summary counts one process; the trace counts the run.

    `mem.budget` is constructed per process, so the nodedup arm — bought across
    four of them — files 164 calls in its summary against roughly 1,325 actually
    made, while `cost_usd` in the same file IS the run total because the runner
    folds in prior spend recovered from the trace. Reporting the summary's
    figure beside that cost would put two scopes in one row. The trace is
    appended per call across every process, so it is what the comparison reads.
    """
    script = Path(__file__).resolve().parent.parent / "scripts" / "repro" / "finer_paired.py"
    spec = _ilu.spec_from_file_location("finer_paired_calls", script)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(mod, "RECORDS", tmp_path)
    trace = tmp_path / "arm.llm-trace.jsonl"
    trace.write_text(
        "".join(json.dumps({"role": r, "tokens_in": 1}) + "\n" for r in ("generate", "distill"))
        + "\n",  # a blank line, which a crash can leave behind
        encoding="utf-8",
    )
    assert mod.llm_calls("arm") == 2

    # the trap this replaces: a summary whose budget describes the last process
    (tmp_path / "arm.json").write_text(
        json.dumps({"llm_budget": {"generate": {"calls": 1}}, "cost_usd": 8.63}), encoding="utf-8"
    )
    summary_calls = sum(v["calls"] for v in mod.load_summary("arm")["llm_budget"].values())
    assert summary_calls != mod.llm_calls("arm")


def test_finer_specificity_reads_proposals_from_the_trace_and_survivors_from_the_store(
    tmp_path, monkeypatch
):
    """The gate sits between what was written and what was kept.

    Measuring bullet specificity on the store alone cannot separate "the curator
    wrote prose" from "the gate kept the specific ones" — which is exactly the
    question the nodedup arm raised, since its 2,165 bullets name fewer US-GAAP
    elements than the online arm's 140. So proposals are read from the trace,
    where reflector responses (not curator JSON) must drop out rather than
    parse into empty bullets.
    """
    script = (
        Path(__file__).resolve().parent.parent / "scripts" / "repro" / "finer_error_structure.py"
    )
    spec = _ilu.spec_from_file_location("finer_error_structure_spec", script)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(mod, "RECORDS", tmp_path)
    lines = [
        # a reflector turn: prose, not JSON, and must not become a bullet
        {"role": "distill", "response_text": "The model confused two similar tags."},
        {
            "role": "distill",
            "response_text": json.dumps(
                {
                    "operations": [
                        {"content": "Prefer DebtInstrumentBasisSpreadOnVariableRate1 here"}
                    ]
                }
            ),
        },
        {"role": "generate", "response_text": "AmortizationOfFinancingCosts"},
    ]
    (tmp_path / "arm.llm-trace.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8"
    )
    proposed = mod.proposed_bullets("arm")
    assert proposed == ["Prefer DebtInstrumentBasisSpreadOnVariableRate1 here"]

    gold = {"debtinstrumentbasisspreadonvariablerate1"}
    spec_specific = mod.specificity(proposed, gold, bands=1)
    assert spec_specific["has_identifier_pct"] == 100.0
    assert spec_specific["names_gold_tag_pct"] == 100.0
    assert spec_specific["distinct_gold_tags_named"] == 1

    # generic advice names nothing, which is what the late nodedup bullets look like
    generic = mod.specificity(
        ["Consider the potential impact of external economic factors."], gold, bands=1
    )
    assert generic["has_identifier_pct"] == 0.0
    assert generic["distinct_gold_tags_named"] == 0


def test_finer_resume_keeps_only_whole_windows_and_refuses_a_foreign_records_file(tmp_path):
    """The host has taken three long runs down mid-flight, so the FiNER runner
    resumes. Two properties decide whether a resumed number can be trusted.

    The window is the atomic unit, not the sample: `run_arm` appends a window's
    rows only AFTER adapting on them, so a window present in the file implies
    the store learned from it. A partial window must therefore be discarded —
    its samples were answered but never trained on, and keeping them would leave
    the transcript ahead of the store it describes. And the kept prefix is
    checked against the dataset rather than assumed, so a records file from
    another split cannot be silently stitched onto this run.
    """
    import logging

    script = Path(__file__).resolve().parent.parent / "scripts" / "repro" / "exp_ace_finer.py"
    spec = _ilu.spec_from_file_location("exp_ace_finer_resume", script)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    samples = [{"target": f"t{i},b,c,d", "question": f"q{i}", "context": ""} for i in range(45)]
    path = tmp_path / "r.records.jsonl"
    log = logging.getLogger("test.resume")

    # 22 rows written: one whole window of 15, then 7 of the next
    with path.open("w", encoding="utf-8") as fh:
        for i in range(22):
            fh.write(json.dumps({"target": samples[i]["target"], "index": i}) + "\n")
    kept = mod.resume_state(path, samples, 15, log)
    assert len(kept) == 15, "the 7 rows of the unfinished window must be dropped"

    # a truncated final line is the shape a crash actually leaves
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"target": "t22,b,c')
    assert len(mod.resume_state(path, samples, 15, log)) == 15

    # rows from a different split must stop the run, not be appended to
    other = tmp_path / "other.records.jsonl"
    with other.open("w", encoding="utf-8") as fh:
        for i in range(15):
            fh.write(json.dumps({"target": f"WRONG{i},b,c,d", "index": i}) + "\n")
    with pytest.raises(SystemExit, match="resume refused"):
        mod.resume_state(other, samples, 15, log)

    assert mod.resume_state(tmp_path / "absent.jsonl", samples, 15, log) == []


# ---------------- run_status: what the artifacts say a run is doing ----------------


_RUN_STATUS_PATH = _SCRIPTS / "repro" / "run_status.py"


def _load_run_status(records_dir):
    spec = _ilu.spec_from_file_location("run_status_mod", _RUN_STATUS_PATH)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.RECORDS = records_dir  # point the artifact root at the fixture
    return mod


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_run_status_prices_the_embedding_sidecar_with_the_chat_trace(tmp_path):
    """The llm-trace holds only chat calls, so pricing from it alone showed the
    `_s` top-50 batched arm at ~$1.51 against its summary's $4.61 — the gap
    being exactly the embedding spend. The status line must sum both traces,
    and count a sidecar line's `calls` field (a delta record can cover several
    concurrent calls), not the line."""
    rs = _load_run_status(tmp_path)
    tag = "arm"
    _jsonl(tmp_path / f"{tag}.records.jsonl", [{"question_id": "q1"}, {"question_id": "q2"}])
    _jsonl(
        tmp_path / f"{tag}.llm-trace.jsonl",
        [
            {"model": "gpt-4o-mini", "tokens_in": 1_000_000, "tokens_out": 0},
            {"model": "gpt-4o-2024-08-06", "tokens_in": 0, "tokens_out": 100_000},
        ],
    )
    _jsonl(
        tmp_path / f"{tag}.embed-trace.jsonl",
        [
            {
                "kind": "embedding",
                "model": "text-embedding-3-small",
                "calls": 128,
                "tokens_in": 50_000_000,
                "tokens_out": 0,
            }
        ],
    )
    row = rs.describe(tag)
    # $0.15 reader + $1.00 judge output + $1.00 embedding
    assert round(row["usd"], 4) == round(0.15 + 1.00 + 1.00, 4)
    assert row["calls"] == 2 + 128
    assert row["idle_s"] is not None  # freshness reads the sidecar's mtime too


def test_run_status_accepts_a_reconstructed_summary_as_finished(tmp_path):
    """The zep MMR arm's summary of record is the git-tracked
    `<tag>.reconstructed.json` (rebuilt from the trace); with only `<tag>.json`
    accepted, run_status reported that finished measurement as STALE forever."""
    rs = _load_run_status(tmp_path)
    tag = "zep_mmr_arm"
    _jsonl(tmp_path / f"{tag}.records.jsonl", [{"question_id": "q1"}])

    row = rs.describe(tag)
    assert not rs.is_finished(row)  # no summary of any kind yet

    (tmp_path / f"{tag}.reconstructed.json").write_text(
        json.dumps(
            {
                "reconstructed_from_trace": True,
                "overall": {"f1": 27.08, "n": 1986, "j_score": 40.78},
                "n_records": 1986,
            }
        )
    )
    row = rs.describe(tag)
    assert row["has_summary"] and rs.is_finished(row)
    # A live `<tag>.json` still wins when both exist — the reconstruction is a
    # fallback for a lost summary, never an override of a real one.
    (tmp_path / f"{tag}.json").write_text(
        json.dumps({"stamp": {"complete": True, "n_samples": 1}, "overall": {}})
    )
    row = rs.describe(tag)
    assert row["complete"] is True and rs.is_finished(row)


def test_run_status_shows_a_finished_arms_summary_total_when_the_trace_is_embed_blind(tmp_path):
    """Arms measured before the embed sidecar existed cannot be re-priced from
    their traces — the embedding share was never recorded anywhere but the
    summary (`cost_usd` folds the embedder and prior processes). A finished
    run therefore shows max(trace, summary), never the blind trace alone."""
    rs = _load_run_status(tmp_path)
    tag = "old_arm"
    _jsonl(tmp_path / f"{tag}.records.jsonl", [{"question_id": "q1"}])
    _jsonl(
        tmp_path / f"{tag}.llm-trace.jsonl",
        [{"model": "gpt-4o-mini", "tokens_in": 10_000_000, "tokens_out": 0}],  # $1.50
    )
    (tmp_path / f"{tag}.json").write_text(
        json.dumps({"stamp": {"complete": True, "n_samples": 1}, "cost_usd": 4.61})
    )
    assert rs.describe(tag)["usd"] == 4.61


def test_max_spend_usd_caps_between_conversations_and_prices_like_cost_usd():
    """The LoCoMo runner's cap prices the merged budget exactly as the final
    cost_usd will (registry rates, judge split, embedder fold) and only ever
    fires between conversations — no cap configured means it never fires."""
    R = _load_repro()

    class FreeEmbedder:  # no budget_row -> fold_embed_budget no-ops, like MiniLM
        pass

    emb = FreeEmbedder()
    budget = {"generate": {"calls": 10, "tokens_in": 1_000_000, "tokens_out": 100_000}}
    args = SimpleNamespace(max_spend_usd=None, model="gpt-4o-mini", judge_model=None)
    assert R._spend_cap_reached(args, budget, emb) is False
    # gpt-4o-mini list rates: $0.15/M in + $0.60/M out -> this budget is $0.21.
    args.max_spend_usd = 0.5
    assert R._spend_cap_reached(args, budget, emb) is False
    args.max_spend_usd = 0.2
    assert R._spend_cap_reached(args, budget, emb) is True
