import importlib.util as _ilu
import sys
from pathlib import Path

import pytest

from agmem.bench.registry import (  # noqa: F401
    MODEL_REGISTRY,
    ModelSpec,
    get_model,
    registry_cost_usd,
    registry_cost_usd_split,
)

_REPRO_PATH = Path(__file__).resolve().parent.parent / "scripts" / "exp_amem_repro.py"


def _load_repro():
    if str(_REPRO_PATH.parent) not in sys.path:
        sys.path.insert(0, str(_REPRO_PATH.parent))
    spec = _ilu.spec_from_file_location("exp_amem_repro", _REPRO_PATH)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_get_model_known():
    spec = get_model("gpt-4o-mini")
    assert spec.endpoint == "https://api.openai.com/v1"
    assert spec.api_key_env == "OPENAI_API_KEY"
    assert (spec.usd_per_1m_in, spec.usd_per_1m_out) == (0.15, 0.60)


def test_get_model_unknown_fails_loud():
    with pytest.raises(KeyError) as exc:
        get_model("gpt-imaginary")
    # the error must NAME the known models so the fix is obvious from the traceback
    assert "gpt-4o-mini" in str(exc.value)


def test_judge_pin_present():
    # LongMemEval-convention judge pin must be registered so --judge-model can resolve it
    assert "gpt-4o-2024-08-06" in MODEL_REGISTRY


def test_cost_usd_uses_the_requested_models_rates():
    budget = {
        "extract": {"tokens_in": 1_000_000, "tokens_out": 0},
        "generate": {"tokens_in": 0, "tokens_out": 1_000_000},
    }
    assert registry_cost_usd(budget, "gpt-4o-mini") == pytest.approx(0.15 + 0.60)


def test_cost_usd_unknown_model_fails_loud():
    with pytest.raises(KeyError):
        registry_cost_usd({}, "gpt-imaginary")


def test_specs_are_immutable():
    spec = get_model("gpt-4o-mini")
    with pytest.raises(Exception):  # noqa: B017
        spec.usd_per_1m_in = 0.0  # frozen dataclass


def test_rates_dict_shape():
    spec = get_model("gpt-4o-mini")
    assert spec.rates_dict() == {
        "model": "gpt-4o-mini",
        "usd_per_1m_in": 0.15,
        "usd_per_1m_out": 0.60,
    }


def test_repro_cost_usd_delegates_to_registry():
    H = _load_repro()  # reuse the exact loader below
    budget = {"extract": {"tokens_in": 2_000_000, "tokens_out": 1_000_000}}
    assert H.cost_usd(budget, "gpt-4o-mini") == pytest.approx(2 * 0.15 + 0.60)
    with pytest.raises(KeyError):
        H.cost_usd(budget, "gpt-imaginary")


def test_registry_cost_usd_split_prices_judge_role_at_judge_rates():
    budget = {
        "judge": {"tokens_in": 1_000_000, "tokens_out": 1_000_000},
        "generate": {"tokens_in": 1_000_000, "tokens_out": 0},
    }
    got = registry_cost_usd_split(budget, "gpt-4o-mini", {"judge": "gpt-4o-2024-08-06"})
    # generate priced at gpt-4o-mini (0.15/1M in); judge priced at gpt-4o-2024-08-06
    # (2.50/1M in, 10.00/1M out) -- NOT at gpt-4o-mini's rates.
    expected = 1.0 * 0.15 + 1.0 * 2.50 + 1.0 * 10.00
    assert got == pytest.approx(expected)


def test_registry_cost_usd_split_no_role_models_equals_single_rate_math():
    budget = {
        "judge": {"tokens_in": 1_000_000, "tokens_out": 1_000_000},
        "generate": {"tokens_in": 1_000_000, "tokens_out": 0},
    }
    assert registry_cost_usd_split(budget, "gpt-4o-mini", None) == pytest.approx(
        registry_cost_usd(budget, "gpt-4o-mini")
    )


def test_registry_cost_usd_split_unknown_judge_model_fails_loud():
    budget = {"judge": {"tokens_in": 1, "tokens_out": 1}}
    with pytest.raises(KeyError):
        registry_cost_usd_split(budget, "gpt-4o-mini", {"judge": "gpt-imaginary"})


def test_repro_cost_usd_splits_judge_rates_when_judge_model_set():
    H = _load_repro()
    budget = {
        "judge": {"tokens_in": 1_000_000, "tokens_out": 1_000_000},
        "generate": {"tokens_in": 1_000_000, "tokens_out": 0},
    }
    got = H.cost_usd(budget, "gpt-4o-mini", judge_model="gpt-4o-2024-08-06")
    expected = 1.0 * 0.15 + 1.0 * 2.50 + 1.0 * 10.00
    assert got == pytest.approx(expected)


def test_repro_cost_usd_judge_model_none_equals_old_single_rate_math():
    H = _load_repro()
    budget = {
        "judge": {"tokens_in": 1_000_000, "tokens_out": 1_000_000},
        "generate": {"tokens_in": 1_000_000, "tokens_out": 0},
    }
    assert H.cost_usd(budget, "gpt-4o-mini", judge_model=None) == pytest.approx(
        registry_cost_usd(budget, "gpt-4o-mini")
    )
    # equal-to-main judge_model behaves the same as None (no split)
    assert H.cost_usd(budget, "gpt-4o-mini", judge_model="gpt-4o-mini") == pytest.approx(
        registry_cost_usd(budget, "gpt-4o-mini")
    )


def test_repro_cost_usd_unknown_judge_model_fails_loud():
    H = _load_repro()
    budget = {"judge": {"tokens_in": 1, "tokens_out": 1}}
    with pytest.raises(KeyError):
        H.cost_usd(budget, "gpt-4o-mini", judge_model="gpt-imaginary")


def _stamp_args(**overrides):
    """Minimal argparse.Namespace covering every field H._stamp reads off `args`."""
    import argparse

    base = {
        "model": "gpt-4o-mini",
        "eval_mode": "wujiang",
        "runs": 1,
        "endpoint": "https://api.openai.com/v1",
        "embedder": "all-MiniLM-L6-v2",
        "k": 5,
        "expand_links": "off",
        "conv": 0,
        "workers": 1,
        "config": "amem",
        "eval_only": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_stamp_omits_judge_cost_rates_when_judge_model_matches_main():
    H = _load_repro()
    spec = get_model("gpt-4o-mini")
    stamp = H._stamp(_stamp_args(), spec, "deadbeef", "t0", "t1", 5, judge_model=None)
    assert "judge_cost_rates" not in stamp
    stamp = H._stamp(_stamp_args(), spec, "deadbeef", "t0", "t1", 5, judge_model="gpt-4o-mini")
    assert "judge_cost_rates" not in stamp


def test_stamp_records_judge_cost_rates_when_judge_model_splits():
    H = _load_repro()
    spec = get_model("gpt-4o-mini")
    stamp = H._stamp(
        _stamp_args(), spec, "deadbeef", "t0", "t1", 5, judge_model="gpt-4o-2024-08-06"
    )
    assert stamp["judge_cost_rates"] == get_model("gpt-4o-2024-08-06").rates_dict()
    assert stamp["cost_rates"] == spec.rates_dict()  # main rates untouched


def test_luna_uses_max_completion_tokens_key():
    spec = get_model("gpt-5.6-luna")
    assert spec.max_tokens_key == "max_completion_tokens"


def test_default_entries_use_max_tokens_key():
    for name in ("gpt-4o-mini", "gpt-4o-2024-08-06"):
        assert get_model(name).max_tokens_key == "max_tokens"


def test_make_roles_forwards_max_tokens_key():
    H = _load_repro()
    roles = H.make_roles(
        "https://api.openai.com/v1",
        "gpt-5.6-luna",
        "k",
        max_tokens_key="max_completion_tokens",
    )
    assert roles["extract"].max_tokens_key == "max_completion_tokens"
    assert roles["distill"].max_tokens_key == "max_completion_tokens"
    assert roles["generate"].max_tokens_key == "max_completion_tokens"
    assert roles["judge"].max_tokens_key == "max_completion_tokens"
    # default: byte-identical to today's behavior
    default_roles = H.make_roles("e", "m", "k")
    assert default_roles["generate"].max_tokens_key == "max_tokens"


def test_make_roles_judge_split():
    H = _load_repro()  # same spec_from_file_location loader as the cost test
    roles = H.make_roles(
        "https://api.openai.com/v1",
        "gpt-4o-mini",
        "k",
        judge_endpoint="https://api.openai.com/v1",
        judge_model="gpt-4o-2024-08-06",
        judge_api_key="jk",
    )
    assert roles["judge"].model == "gpt-4o-2024-08-06"
    assert roles["judge"].api_key == "jk"
    assert roles["generate"].model == "gpt-4o-mini"  # model under test untouched
    # default: judge inherits the model under test (behavior identical to today)
    roles = H.make_roles("e", "m", "k")
    assert roles["judge"].model == "m"
