"""The OpenRouter entry in the model registry: the v1 reader route.

Pinned because the registry is where every quote and every ledger line gets its
rates, and an entry that drifted to OpenAI's endpoint or key variable would
authenticate the wrong account and price the wrong model without an error.
"""

from __future__ import annotations

from agmem.bench.registry import get_model, registry_cost_usd


def test_qwen35_9b_is_registered_on_openrouter_with_its_own_key():
    spec = get_model("qwen/qwen3.5-9b")
    assert spec.endpoint == "https://openrouter.ai/api/v1"
    assert spec.api_key_env == "OPENROUTER_API_KEY"
    assert (spec.usd_per_1m_in, spec.usd_per_1m_out) == (0.10, 0.15)
    # Chat Completions dialect: plain max_tokens, sampling adjustable.
    assert spec.max_tokens_key == "max_tokens"
    assert spec.fixed_sampling is False


def test_a_small_tier_reader_pass_quotes_under_a_dollar():
    """451 questions at ~12.5K input / ~300 output tokens each — the plan's own
    arithmetic for the reader half of a LongMemEval-V2 small run."""
    budget = {"reader": {"tokens_in": 451 * 12_500, "tokens_out": 451 * 300}}
    usd = registry_cost_usd(budget, "qwen/qwen3.5-9b")
    assert 0.5 < usd < 0.7, usd
