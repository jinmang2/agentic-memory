"""CountingLLM (src/agmem/bench/counting.py) drives a real organizer write path
with zero API calls, so per-role call counts are real counts, not per-turn
constants — see the module docstring for why that matters for branchy
organizers (eviction, merge)."""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

from agmem.bench.counting import build_counting_memory
from agmem.llm.structured import StructuredCaller

_CONFIGS_PATH = Path(__file__).resolve().parent.parent / "scripts" / "repro" / "configs.py"


def _load_configs():
    spec = _ilu.spec_from_file_location("repro_configs", _CONFIGS_PATH)
    mod = _ilu.module_from_spec(spec)
    # configs.py's RunnerConfig is a frozen dataclass under `from __future__
    # import annotations`; dataclasses resolves deferred annotations via
    # sys.modules[cls.__module__], so the module must be registered there
    # BEFORE exec_module runs the class body.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_amem_ingest_counts_extract_per_turn_distill_after_first(tmp_path):
    cfg = _load_configs().get_config("amem")
    mem, fake = build_counting_memory("amem", cfg.factory, tmp_path, "count-test", cfg.memory_types)
    msgs = ["(2021) A: I moved to Berlin.", "(2021) B: Nice!", "(2021) A: New job too."]
    try:
        for m in msgs:
            mem.add_message(m, role="user")
        mem.flush()
    finally:
        mem.close()
    # A-Mem write path is extract(Ps1) per turn always, and distill(Ps3) once a
    # prior note exists as a neighbor — so turn 1 has none.
    assert fake.calls["extract"] == len(msgs)
    assert fake.calls["distill"] == len(msgs) - 1


def test_unknown_canned_profile_fails_loud(tmp_path):
    import pytest

    cfg = _load_configs().get_config("amem")
    with pytest.raises(KeyError):
        build_counting_memory("nope", cfg.factory, tmp_path, "x", cfg.memory_types)


def test_nemori_counting_profile(tmp_path):
    """The nemori canned profile must be schema-valid against every
    structured call the upstream-preset write path can reach (segmentation,
    narration, merge-decision, predict/calibrate, cold-start direct-extract)
    — a bounded, deterministic ingest that exercises all of them with zero
    parse failures."""
    cfg = _load_configs().get_config("nemori_upstream")
    mem, fake = build_counting_memory(
        "nemori", cfg.factory, tmp_path, "count-test", cfg.memory_types
    )
    msgs = [f"(2021) {'A' if i % 2 == 0 else 'B'}: message number {i}." for i in range(24)]
    try:
        for m in msgs:
            mem.add_message(m, role="user")
        mem.flush()
    finally:
        mem.close()
    # extract = segmentation calls (BatchPartitioner's window=20 gate fires
    # once on this 24-message ingest); distill = narrate + merge-decision +
    # predict/calibrate/direct-extract, all of which route through "distill"
    # per stages.py / organizer.py.
    assert fake.calls.get("extract", 0) > 0
    assert fake.calls.get("distill", 0) > 0
    assert mem.structured.drops == {}


def test_mem0_ingest_counts_two_calls_per_add(tmp_path):
    """The structural claim under verification: exactly 2 LLM calls per add(),
    independent of how many facts extraction returned."""
    cfg = _load_configs().get_config("mem0_v0194")
    mem, fake = build_counting_memory("mem0", cfg.factory, tmp_path, "count-test", cfg.memory_types)
    msgs = [
        "(2023) A: I moved to Berlin.",
        "(2023) B: Nice!",
        "(2023) A: New job too.",
        "(2023) B: Congrats!",
    ]
    try:
        for m in msgs:
            mem.add_message(m, role="user")
        mem.flush()
    finally:
        mem.close()
    # batch_size=2 -> 4 messages = 2 adds = 2 extract + 2 distill
    assert fake.calls["extract"] == 2
    assert fake.calls["distill"] == 2
    assert mem.structured.drops == {}  # canned responses are schema-valid


def test_mem0_odd_tail_is_flushed_not_stranded(tmp_path):
    cfg = _load_configs().get_config("mem0_v0194")
    mem, fake = build_counting_memory("mem0", cfg.factory, tmp_path, "count-test", cfg.memory_types)
    try:
        for m in ["(2023) A: one.", "(2023) B: two.", "(2023) A: three."]:
            mem.add_message(m, role="user")
        mem.flush()
    finally:
        mem.close()
    # 3 messages at batch_size=2 -> 2 adds, the second issued by flush()
    assert fake.calls["extract"] == 2
    assert fake.calls["distill"] == 2


def _zep_mem(tmp_path, profile):
    cfg = _load_configs().get_config("zep_cross_encoder")
    return build_counting_memory(profile, cfg.factory, tmp_path, "count-test", cfg.memory_types)


_ZEP_MSGS = [
    "(2023-05-01) Caroline: I met Melanie at the Berlin office last week.",
    "(2023-05-01) Melanie: Caroline told me about her new job at Acme.",
    "(2023-05-02) Caroline: Melanie and I went to Prague together.",
    "(2023-05-02) Melanie: That trip to Prague was great.",
    "(2023-05-03) Caroline: My sister Diana joined Acme too.",
    "(2023-05-03) Melanie: Diana works with Caroline now.",
]


def _run_zep(tmp_path, profile, msgs=None):
    """Ingest through the zep profile, returning (calls, per-site counts, drops)."""
    mem, fake = _zep_mem(tmp_path, profile)
    sites = {}
    original = fake.chat

    def chat(role, messages, budget_key=None, **kw):
        prompt = " ".join(m.get("content", "") for m in messages)
        for key, marker in (
            ("entity_extract", "Extract the distinct real-world entities"),
            ("entity_resolve", "Decide for each NEW entity"),
            ("fact_extract", "Extract relationship facts"),
            ("edge_resolve", "A new fact arrived"),
            ("community_summarize", "Synthesize the information"),
            ("community_describe", "one sentence description"),
        ):
            if marker in prompt:
                sites[key] = sites.get(key, 0) + 1
        return original(role, messages, budget_key=budget_key, **kw)

    fake.chat = chat
    mem.llm = fake
    mem.structured = StructuredCaller(fake, use_guided_json=False)
    mem._ctx.llm = mem.structured
    try:
        for m in msgs if msgs is not None else _ZEP_MSGS:
            mem.add_message(m, role="user")
        mem.flush()
        drops = dict(mem.structured.drops)
    finally:
        mem.close()
    return fake.calls, sites, drops


def test_zep_canned_profile_reaches_every_call_site_schema_validly(tmp_path):
    """All six of the organizer's structured calls must parse. A canned response
    the schema rejects does not merely lose one call — it changes which branch
    runs next (a dropped entity extraction returns [] and suppresses resolution,
    facts and communities downstream), so the counted total would be wrong in a
    direction no assertion on the total alone would catch."""
    _calls, sites, drops = _run_zep(tmp_path, "zep")
    assert drops == {}
    assert set(sites) == {
        "entity_extract",
        "entity_resolve",
        "fact_extract",
        "edge_resolve",
        "community_summarize",
        "community_describe",
    }, sites


def test_zep_entity_extraction_is_one_call_per_message_at_every_yield(tmp_path):
    """The structural floor the quote rests on: entity extraction is
    unconditional, so it is exact regardless of the yield assumptions the other
    sites depend on. If this ever became yield-sensitive, the quote would have
    no measured component left at all."""
    for profile in ("zep_low", "zep", "zep_high"):
        _calls, sites, _drops = _run_zep(tmp_path / profile, profile)
        assert sites["entity_extract"] == len(_ZEP_MSGS), (profile, sites)


def test_zep_entity_poor_message_skips_fact_extraction(tmp_path):
    """`on_message` returns before fact extraction when fewer than two entities
    resolve. The profile must reproduce that gate rather than paper over it with
    a fixed entity list — an entity-poor corpus pays one call per turn, not two,
    and quoting the wrong one of those is a 5,882-call error at campaign scale."""
    msgs = ["(2023-05-01) Caroline: yeah, i think so too.", "(2023-05-01) Caroline: sure."]
    _calls, sites, drops = _run_zep(tmp_path, "zep", msgs)
    assert drops == {}
    assert sites["entity_extract"] == 2
    assert "fact_extract" not in sites


def test_zep_facts_about_one_pair_differ_between_messages(tmp_path):
    """Regression guard for the defect the first conv0 counting pass exposed.

    An earlier profile emitted "<subject> is related to <object>" — identical on
    every recurrence of a pair — so the organizer's verbatim fast path matched
    and skipped edge resolution, suppressing 79% of that site's calls on conv0
    (160 where the corpus-derived statement yields 752). The quote would have
    priced a fifth of the real write bill and looked entirely healthy doing it.
    Two messages about the same pair must produce different statement text, as
    real extraction does."""
    msgs = [
        "(2023-05-01) Caroline: I met Melanie at the Berlin office.",
        "(2023-05-02) Caroline: Melanie and I flew to Prague.",
    ]
    mem, _fake = _zep_mem(tmp_path, "zep")
    try:
        for m in msgs:
            mem.add_message(m, role="user")
        mem.flush()
        facts = mem.doc_store.list_items("facts", namespace="count-test")
        statements = {str(f.get("content", "")) for f in facts}
    finally:
        mem.close()
    assert len(statements) == len(facts) > 1, statements


def test_zep_band_profiles_move_the_yield_driven_sites_only(tmp_path):
    """The band exists because edge resolution scales with fact yield while
    entity extraction does not. This pins that separation: the low and high
    points must disagree on edge resolution and agree on entity extraction."""
    _l, low, _ = _run_zep(tmp_path / "low", "zep_low")
    _h, high, _ = _run_zep(tmp_path / "high", "zep_high")
    assert low["entity_extract"] == high["entity_extract"]
    assert low["edge_resolve"] < high["edge_resolve"], (low, high)


def test_mem0_canned_profile_grows_the_store_so_retrieval_keeps_branching(tmp_path):
    """Always-ADD is what keeps counting deterministic AND realistic.

    A profile that answered NONE would leave the store empty, so every later
    decision prompt would show zero candidates — the retrieval branch would
    never be exercised and the quote would price a shape the real run does not
    have. Call count is unaffected either way, which is why always-ADD is safe.
    """
    cfg = _load_configs().get_config("mem0_v0194")
    mem, _ = build_counting_memory("mem0", cfg.factory, tmp_path, "count-test", cfg.memory_types)
    try:
        for i in range(6):
            mem.add_message(f"(2023) A: message {i}.", role="user")
        mem.flush()
        n = len(mem.doc_store.list_items("semantic", namespace="count-test"))
    finally:
        mem.close()
    assert n == 3  # one ADD per add(), 6 messages at batch_size=2
