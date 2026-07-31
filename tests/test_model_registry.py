import importlib.util as _ilu
import sys
from pathlib import Path

import pytest

from agmem.bench.registry import (  # noqa: F401
    MODEL_REGISTRY,
    ModelSpec,
    get_model,
    registry_cost_usd,
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
