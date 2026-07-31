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
from agmem.bench.counting import build_counting_memory
from agmem.embed.fake import FakeEmbedder
from agmem.llm.client import LLMClient, RoleConfig
from agmem.organizers.amem import AMemOrganizer

_REPRO_PATH = Path(__file__).resolve().parent.parent / "scripts" / "exp_amem_repro.py"
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


def _fake_conv_summary(n_turns, calls, tin, tout, cost, per_type):
    """Shape one per-conv --ingest-only summary the way exp_amem_repro emits it."""
    return {
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
        model="gpt-4o-mini", data_dir=str(data_dir), tag_suffix="_seed1", workers=3
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
    nemori_merge085 = get_config("nemori_merge085")
    assert nemori_merge085.run_ready is True
    assert nemori_merge085.role_temps == nemori_upstream.role_temps
    assert nemori_merge085.per_type_k == nemori_upstream.per_type_k
    assert nemori_merge085.store == nemori_upstream.store
    amem = get_config("amem")
    assert amem.run_ready is True
    # amem's path stays byte-identical: all three new fields None.
    assert amem.role_temps is None
    assert amem.per_type_k is None
    assert amem.store is None
    with _pytest.raises(KeyError):
        get_config("nope")


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
        )

    amem_stamp = repro._stamp(_args("amem"), spec, None, "t0", "t1", 5)
    assert amem_stamp["k"] == 10  # unchanged from before this fix — no per_type_k on amem
    assert amem_stamp["temps"] == {"write": 0.7, "generate": 0.7, "cat5": repro.CAT5_TEMPERATURE}

    nemori_stamp = repro._stamp(_args("nemori_upstream"), spec, None, "t0", "t1", 5)
    assert nemori_stamp["k"] == {"episodes": 10, "semantic": 20}
    assert nemori_stamp["temps"] == {
        "extract": {"temperature": 0.2, "max_tokens": 4096},
        "distill": {"temperature": 0.7, "max_tokens": 2000},
        "generate": {"temperature": 0.0},
    }
