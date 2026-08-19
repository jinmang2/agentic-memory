"""The two places the LongMemEval C4 driver can lose a question quietly.

Both are about the same failure the benchmark's own code has (LME-A18): a run
that did not answer everything reporting a score anyway. The driver's guard is a
completeness check, and these tests cover the parts of it that a row COUNT
cannot see — plus the cost fold, which is what keeps a 500-call judge from being
priced at the reader's rate.

No dataset and no network: the driver's helpers are pure functions over rows.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import logging
import sys
from pathlib import Path

_DRIVER = Path(__file__).resolve().parent.parent / "scripts" / "repro" / "exp_lme_reading.py"


def _load_driver():
    if str(_DRIVER.parent) not in sys.path:
        sys.path.insert(0, str(_DRIVER.parent))
    spec = _ilu.spec_from_file_location("exp_lme_reading", _DRIVER)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_an_unjudged_row_is_not_resumable(tmp_path):
    """A question whose API call failed writes a row so the failure cannot vanish.
    Treating that row as done would leave a hole no row count can see: 500 rows,
    499 judged, aggregated over 499 while the stamp says complete."""
    driver = _load_driver()
    path = tmp_path / "arm.records.jsonl"
    _write(
        path,
        [
            {"question_id": "q1", "label": True},
            {"question_id": "q2", "label": None, "error": "APIError: 429"},
            {"question_id": "q3", "label": False},
        ],
    )
    kept, done = driver.resume_rows(path, logging.getLogger("t"))
    assert [r["question_id"] for r in kept] == ["q1", "q3"]
    assert done == {"q1", "q3"}, "q2 must be re-answered, not skipped"


def test_a_truncated_final_line_ends_the_resume_rather_than_raising(tmp_path):
    driver = _load_driver()
    path = tmp_path / "arm.records.jsonl"
    path.write_text(
        json.dumps({"question_id": "q1", "label": True})
        + "\n"
        + '{"question_id": "q2", "lab'  # the shape a kill leaves
    )
    kept, done = driver.resume_rows(path, logging.getLogger("t"))
    assert done == {"q1"}
    assert [r["question_id"] for r in kept] == ["q1"]


def test_row_keys_fold_back_into_roles_so_the_judge_prices_as_the_judge(tmp_path):
    """Per-row budget keys are what make a concurrent run's cost attributable at
    all, and they are also what would misprice it: `cost_usd` prices the key
    named `judge` at the judge model's rates, so 500 keys named `judge|<qid>`
    would bill the pinned gpt-4o judge at gpt-4o-mini's rates."""
    driver = _load_driver()
    raw = {
        "generate|q1": {
            "calls": 1,
            "tokens_in": 1_000_000,
            "tokens_out": 0,
            "latency_ms_avg": 10.0,
        },
        "generate|q2": {
            "calls": 1,
            "tokens_in": 1_000_000,
            "tokens_out": 0,
            "latency_ms_avg": 30.0,
        },
        "judge|q1": {"calls": 1, "tokens_in": 1_000_000, "tokens_out": 0, "latency_ms_avg": 5.0},
    }
    folded = driver.fold_row_keys(raw)
    assert folded["generate"]["calls"] == 2
    assert folded["generate"]["tokens_in"] == 2_000_000
    assert folded["generate"]["latency_ms_avg"] == 20.0  # call-weighted, not key-averaged
    assert folded["judge"]["calls"] == 1

    from agmem.bench.registry import registry_cost_usd_split as cost_usd_split

    priced = cost_usd_split(folded, "gpt-4o-mini", {"judge": "gpt-4o-2024-08-06"})
    # 2M reader tokens at $0.15/M + 1M judge tokens at $2.50/M
    assert round(priced, 4) == round(2 * 0.15 + 2.50, 4)
    # The same call ledger priced WITHOUT folding cannot see a judge at all.
    unfolded = cost_usd_split(raw, "gpt-4o-mini", {"judge": "gpt-4o-2024-08-06"})
    assert round(unfolded, 4) == round(3 * 0.15, 4)


# ---------------- read-budget alignment + prompt-level recall (docs §10.3 / docs/20) ----------


class _Chunk:
    def __init__(self, text: str):
        self._text = text

    def render(self) -> str:
        return self._text


class _BundleSearcher:
    """Hands `answer` a pre-built bundle, so the budget math is tested against
    the real render path without a store or an embedder."""

    def __init__(self, bundle):
        self.bundle = bundle

    def search(self, query, memory_types=None, k=10, metrics=None):
        return self.bundle


def _three_item_bundle():
    from agmem.core.types import MemoryBundle, ScoredItem

    # A fits a 200-char budget alone; B (500 chars) overflows it; C never gets
    # a turn. Texts are chosen so none is a substring of another.
    return MemoryBundle(
        query="q",
        items=[
            ScoredItem(item=_Chunk("A" * 100), memory_type="episodes", score=3.0),
            ScoredItem(item=_Chunk("B" * 500), memory_type="semantic", score=2.0),
            ScoredItem(item=_Chunk("C-unique"), memory_type="semantic", score=1.0),
        ],
    )


class _AnswerLLM:
    def chat(self, role, messages, budget_key=None, **overrides):
        return "an answer"


def _bare_mem():
    from types import SimpleNamespace

    return SimpleNamespace(llm=_AnswerLLM(), organizers=[])


def test_in_prompt_marks_exactly_the_items_that_survived_the_render_budget():
    """docs/20's capture defect: `capture["retrieved"]` is built before
    `render()` truncates, so bundle membership says nothing about what the
    reader saw. The flag must read True only for items whose text is in the
    prompt that was actually sent."""
    from agmem.bench import longmemeval as lme

    capture: dict = {}
    lme.answer(
        _bare_mem(),
        {"question": "q?", "question_date": "2024/01/01"},
        budget_tokens=50,  # * CHARS_PER_TOKEN(4) = 200 chars: A fits, B breaks
        searcher=_BundleSearcher(_three_item_bundle()),
        capture=capture,
    )
    flags = {c["text"][:1]: c["in_prompt"] for c in capture["retrieved"]}
    assert flags == {"A": True, "B": False, "C": False}
    assert "A" * 100 in capture["prompt"] and "B" not in capture["prompt"]


def test_k_total_caps_the_bundle_across_types_by_score():
    """The read-budget alignment (docs/research/longmemeval.md §10.3; docs/20
    measured the per-type wiring handing an organizer 1.14-1.54x more context).
    `k_total` must cut across ALL memory types by score — and None must keep
    the old wiring byte-for-byte."""
    from agmem.bench import longmemeval as lme

    capped: dict = {}
    lme.answer(
        _bare_mem(),
        {"question": "q?", "question_date": ""},
        k_total=2,
        budget_tokens=10_000,
        searcher=_BundleSearcher(_three_item_bundle()),
        capture=capped,
    )
    assert [c["score"] for c in capped["retrieved"]] == [3.0, 2.0]
    assert capped["k_total"] == 2

    uncapped: dict = {}
    lme.answer(
        _bare_mem(),
        {"question": "q?", "question_date": ""},
        budget_tokens=10_000,
        searcher=_BundleSearcher(_three_item_bundle()),
        capture=uncapped,
    )
    assert len(uncapped["retrieved"]) == 3 and uncapped["k_total"] is None


def test_recall_fields_split_bundle_recall_from_prompt_recall():
    """`evidence_recall_bundle` is the old `evidence_recall` (same math, named
    for what it scores); `evidence_recall_prompt` counts only items that
    survived rendering. The defect case is exactly the one that must differ:
    evidence in the bundle, cut from the prompt."""
    driver = _load_driver()
    retrieved = [
        {"session_ids": ["s1"], "in_prompt": True},
        {"session_ids": ["s2"], "in_prompt": False},  # in the bundle, cut from the prompt
    ]
    fields = driver.recall_fields(retrieved, {"s1", "s2"})
    assert fields == {"evidence_recall_bundle": 1.0, "evidence_recall_prompt": 0.5}

    # None semantics survive: no provenance at all is not 0.0 …
    none_fields = driver.recall_fields([{"session_ids": None}], {"s1"})
    assert none_fields == {"evidence_recall_bundle": None, "evidence_recall_prompt": None}
    # … and a row captured before `in_prompt` existed gets no prompt recall
    # rather than a silent 0.0.
    legacy = driver.recall_fields([{"session_ids": ["s1"], "in_prompt": None}], {"s1"})
    assert legacy["evidence_recall_bundle"] == 1.0
    assert legacy["evidence_recall_prompt"] is None


# ---------------- embedding spend is durable (the llm-trace's blind spot) ----------------


class _CountingEmbedder:
    """`APIEmbedder`'s accounting surface: cumulative `calls`/`tokens` counters
    incremented inside `embed`, which is what makes naive before/after diffs
    racy and the delta-under-lock design necessary."""

    name = "text-embedding-3-small"

    def __init__(self):
        self.calls = 0
        self.tokens = 0

    def embed(self, texts, kind="passage"):
        self.calls += 1
        self.tokens += 7 * len(texts)
        return [[0.0] for _ in texts]


def test_traced_embedder_records_totals_that_match_the_counters(tmp_path):
    driver = _load_driver()
    inner = _CountingEmbedder()
    trace = tmp_path / "arm.embed-trace.jsonl"
    wrapped = driver.TracedEmbedder(inner, trace)

    wrapped.embed(["a", "b"])
    wrapped.embed(["c"])
    lines = [json.loads(ln) for ln in trace.read_text().splitlines()]
    assert sum(ln["calls"] for ln in lines) == inner.calls == 2
    assert sum(ln["tokens_in"] for ln in lines) == inner.tokens == 21
    assert all(ln["kind"] == "embedding" and ln["tokens_out"] == 0 for ln in lines)
    assert all(ln["model"] == "text-embedding-3-small" for ln in lines)

    # Counter growth the wrapper did not see itself is flushed, not lost.
    inner.calls += 1
    inner.tokens += 5
    wrapped.flush_trace()
    lines = [json.loads(ln) for ln in trace.read_text().splitlines()]
    assert sum(ln["calls"] for ln in lines) == 3
    assert sum(ln["tokens_in"] for ln in lines) == 26
    # Delegation: pricing helpers read the same counters through the wrapper.
    assert wrapped.name == "text-embedding-3-small" and wrapped.tokens == 26


def test_prior_spend_carries_the_embedding_sidecar_across_a_resume(tmp_path):
    """The resume banner (and the cap it feeds) priced earlier processes from
    the llm-trace alone — chat calls only. With the sidecar present the
    trace-side estimate must include the embedding share, priced at the
    embedding model's own rates."""
    driver = _load_driver()
    helpers = driver._load_repro_helpers()
    trace = tmp_path / "arm.llm-trace.jsonl"
    _write(
        trace,
        [
            {"model": "gpt-4o-mini", "tokens_in": 1_000_000, "tokens_out": 0},
            {"model": "gpt-4o-2024-08-06", "tokens_in": 1_000_000, "tokens_out": 0},
        ],
    )
    embed_trace = tmp_path / "arm.embed-trace.jsonl"
    _write(
        embed_trace,
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
    got = driver._prior_spend(
        trace, embed_trace, tmp_path / "arm.json", "gpt-4o-mini", "gpt-4o-2024-08-06", helpers
    )
    # 1M reader at $0.15/M + 1M judge at $2.50/M + 50M embed at $0.02/M
    assert round(got, 4) == round(0.15 + 2.50 + 1.00, 4)

    # A summary that already priced MORE (it folded spend a dead trace lost)
    # still wins the max — neither record can overstate.
    (tmp_path / "arm.json").write_text(json.dumps({"cost_usd": 10.0}))
    got = driver._prior_spend(
        trace, embed_trace, tmp_path / "arm.json", "gpt-4o-mini", "gpt-4o-2024-08-06", helpers
    )
    assert got == 10.0


# ---------------- organizer-arm memory snapshots (full-artifact-capture rule) ----------------


def test_snapshot_is_wanted_exactly_for_write_path_arms():
    """Passthrough stores are the stamped dataset verbatim — dumping 500 of
    them re-copies the corpus and captures nothing. Any organizer arm's derived
    state is paid and unrecoverable, so it must be dumped."""
    driver = _load_driver()
    assert driver.snapshot_wanted(["passthrough"]) is False
    assert driver.snapshot_wanted(["nemori"]) is True
    assert driver.snapshot_wanted(["passthrough", "nemori"]) is True


def test_instance_dump_writes_episodic_and_derived_state_keyed_by_qid(tmp_path):
    """$0 verification of the per-instance dump: episodic turns and a derived
    item land in .memory.jsonl, ops land in .memory.ops.jsonl, and every line
    carries the question id (the helpers' `conv` unit key)."""
    from agmem import AgenticMemory
    from agmem.embed.fake import FakeEmbedder

    driver = _load_driver()
    helpers = driver._load_repro_helpers()
    mem = AgenticMemory(
        namespace="lme-q7", organizers=["passthrough"], embedder=FakeEmbedder(dim=64)
    )
    try:
        mem.add_message("(2023/05/20) user: hello", role="user")
        mem.add_message("(2023/05/20) assistant: hi", role="assistant")
        # A derived item straight into the store — the state an organizer arm
        # pays for, seeded without an LLM (same trick as the ace snapshot test).
        mem.doc_store.put_item(
            "n1", "notes", "lme-q7", {"id": "n1", "content": "derived", "tags": [], "links": []}
        )
        mem.flush()
        snap = tmp_path / "arm.memory.jsonl"
        ops = tmp_path / "arm.memory.ops.jsonl"
        with snap.open("w", encoding="utf-8") as sfh, ops.open("w", encoding="utf-8") as ofh:
            driver.dump_instance_state(helpers, mem, "q7", sfh, ofh)
    finally:
        mem.close()

    snap_rows = [json.loads(ln) for ln in snap.read_text().splitlines()]
    kinds = sorted({r["memory_type"] for r in snap_rows})
    assert kinds == ["episodic", "notes"]
    assert all(r["conv"] == "q7" for r in snap_rows)
    assert len([r for r in snap_rows if r["memory_type"] == "episodic"]) == 2
    ops_rows = [json.loads(ln) for ln in ops.read_text().splitlines()]
    assert ops_rows and all(r["conv"] == "q7" for r in ops_rows)


# ---------------- terra/sol registration (docs/research/longmemeval.md §8.2) ----------------


def test_terra_and_sol_are_registered_at_their_quoted_list_prices():
    """§10.3's open item. The numbers are §8.2's (lines 721-722), not invented:
    terra $2.00/$12.00, sol $5.00/$30.00 per 1M in/out, gpt-5.6 dialect."""
    from agmem.bench.registry import get_model

    terra = get_model("gpt-5.6-terra")
    sol = get_model("gpt-5.6-sol")
    assert (terra.usd_per_1m_in, terra.usd_per_1m_out) == (2.00, 12.00)
    assert (sol.usd_per_1m_in, sol.usd_per_1m_out) == (5.00, 30.00)
    for spec in (terra, sol):
        assert spec.max_tokens_key == "max_completion_tokens" and spec.fixed_sampling
