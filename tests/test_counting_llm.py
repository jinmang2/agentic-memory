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


def test_amem_ingest_counts_two_calls_per_turn(tmp_path):
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
