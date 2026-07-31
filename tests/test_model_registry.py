import pytest

from agmem.bench.registry import MODEL_REGISTRY, ModelSpec, get_model, registry_cost_usd


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
    budget = {"extract": {"tokens_in": 1_000_000, "tokens_out": 0},
              "generate": {"tokens_in": 0, "tokens_out": 1_000_000}}
    assert registry_cost_usd(budget, "gpt-4o-mini") == pytest.approx(0.15 + 0.60)


def test_cost_usd_unknown_model_fails_loud():
    with pytest.raises(KeyError):
        registry_cost_usd({}, "gpt-imaginary")


def test_specs_are_immutable():
    spec = get_model("gpt-4o-mini")
    with pytest.raises(Exception):
        spec.usd_per_1m_in = 0.0  # frozen dataclass
