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
        # Joined 2026-08-20 with the stamp field of the same name: an artifact has
        # to say which store it read or wrote, because `eval_only: true` alone
        # never said WHICH (docs/14 §Artifacts).
        "data_dir": None,
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


def test_luna_has_fixed_sampling():
    assert get_model("gpt-5.6-luna").fixed_sampling is True


def test_default_entries_have_fixed_sampling_false():
    for name in ("gpt-4o-mini", "gpt-4o-2024-08-06"):
        assert get_model(name).fixed_sampling is False


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


def test_make_roles_fixed_sampling_omits_temperature_but_keeps_max_tokens():
    H = _load_repro()
    roles = H.make_roles(
        "https://api.openai.com/v1",
        "gpt-5.6-luna",
        "k",
        role_temps={
            "extract": {"temperature": 0.2, "max_tokens": 4096},
            "distill": {"temperature": 0.7, "max_tokens": 2000},
            "generate": {"temperature": 0.0},
        },
        max_tokens=1000,
        max_tokens_key="max_completion_tokens",
        fixed_sampling=True,
    )
    for role in ("extract", "distill", "generate", "judge"):
        assert roles[role].temperature is None
    # max_tokens (from role_temps or the default) still applies — only
    # temperature is stripped.
    assert roles["extract"].max_tokens == 4096
    assert roles["distill"].max_tokens == 2000
    assert roles["generate"].max_tokens == 1000
    assert roles["judge"].max_tokens == 1000
    assert roles["extract"].max_tokens_key == "max_completion_tokens"


def test_make_roles_fixed_sampling_false_is_byte_identical_default():
    H = _load_repro()
    roles = H.make_roles(
        "https://api.openai.com/v1",
        "gpt-4o-mini",
        "k",
        role_temps={"extract": {"temperature": 0.2, "max_tokens": 4096}},
    )
    assert roles["extract"].temperature == 0.2
    assert roles["distill"].temperature == 0.7
    assert roles["generate"].temperature == 0.7
    assert roles["judge"].temperature == 0.0


def test_make_roles_judge_fixed_sampling_inherits_main_by_default():
    H = _load_repro()
    # no split judge model, main model is fixed-sampling -> judge inherits
    roles = H.make_roles("e", "gpt-5.6-luna", "k", fixed_sampling=True)
    assert roles["judge"].temperature is None
    # non-fixed-sampling main model -> judge stays deterministic (0.0)
    roles = H.make_roles("e", "gpt-4o-mini", "k", fixed_sampling=False)
    assert roles["judge"].temperature == 0.0


def test_make_roles_judge_fixed_sampling_explicit_overrides_main():
    H = _load_repro()
    # main model is NOT fixed-sampling but a split judge model IS
    roles = H.make_roles(
        "e",
        "gpt-4o-mini",
        "k",
        judge_model="gpt-5.6-luna",
        fixed_sampling=False,
        judge_fixed_sampling=True,
    )
    assert roles["judge"].temperature is None
    assert roles["extract"].temperature == 0.7  # main-model roles untouched
    # inverse: main model IS fixed-sampling but judge is a normal split model
    roles = H.make_roles(
        "e",
        "gpt-5.6-luna",
        "k",
        judge_model="gpt-4o-mini",
        fixed_sampling=True,
        judge_fixed_sampling=False,
    )
    assert roles["judge"].temperature == 0.0
    assert roles["extract"].temperature is None


def test_stamp_temps_records_fixed_sampling_reality_for_luna():
    H = _load_repro()
    spec = get_model("gpt-5.6-luna")
    stamp = H._stamp(
        _stamp_args(model="gpt-5.6-luna"), spec, "deadbeef", "t0", "t1", 5, judge_model=None
    )
    assert stamp["temps"] == {"fixed_sampling": True, "model": "gpt-5.6-luna"}


def test_stamp_temps_unchanged_for_default_model():
    H = _load_repro()
    spec = get_model("gpt-4o-mini")
    stamp = H._stamp(_stamp_args(), spec, "deadbeef", "t0", "t1", 5, judge_model=None)
    assert stamp["temps"] == {"write": 0.7, "generate": 0.7, "cat5": H.CAT5_TEMPERATURE}
