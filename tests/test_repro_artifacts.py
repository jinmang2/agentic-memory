"""Full run-artifact capture for the A-Mem reproduction harness (issue #1):
the LLM-call trace sink, the post-ingest memory snapshot, per-question
retrieval capture in records, and the write-once/read-sweep guarantee that
--eval-only issues ZERO write-path LLM calls. Unit/integration level only —
fake embedder + fake LLM throughout, no API/server, no paid calls."""

from __future__ import annotations

import hashlib
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


def _fake_mem(summary: dict, drops: dict | None = None):
    """A stand-in exposing just what _merge_budget touches: a budget with a
    summary() and an optional structured.drops."""
    structured = None if drops is None else SimpleNamespace(drops=drops)
    return SimpleNamespace(budget=SimpleNamespace(summary=lambda: summary), structured=structured)


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
    assert repro.cost_usd(total) == round(2 * repro.cost_usd(run1), 6)


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
    assert len(out["sources"]) == 3 and out["sources"][0]["git_sha"] == "sha0"


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
