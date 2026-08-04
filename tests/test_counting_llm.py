"""CountingLLM (src/agmem/bench/counting.py) drives a real organizer write path
with zero API calls, so per-role call counts are real counts, not per-turn
constants — see the module docstring for why that matters for branchy
organizers (eviction, merge)."""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

from agmem.bench.counting import build_counting_memory

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
