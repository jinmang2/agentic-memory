"""Model/provider registry: the single source for endpoint, key env var, and list
price per model. Every script that spends or quotes goes through here so an
unknown model fails loud instead of silently pricing at 4o-mini rates (the
pre-Phase-2 bug this replaces: exp_amem_repro's fixed COST_RATES applied to any
--model). Prices are list prices at registration time; the run stamp records the
resolved spec so later price changes can't silently reprice old artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    endpoint: str
    api_key_env: str
    usd_per_1m_in: float
    usd_per_1m_out: float
    # Some newer Chat Completions models (e.g. gpt-5.6-luna) reject `max_tokens`
    # outright (400 "Unsupported parameter") and require `max_completion_tokens`
    # instead. Defaulting to "max_tokens" keeps every existing positional
    # ModelSpec(...) construction (this file is the only caller) byte-identical.
    max_tokens_key: str = "max_tokens"
    # Some newer Chat Completions models (e.g. gpt-5.6-luna) reject any
    # non-default `temperature` outright (400 "Unsupported value: 'temperature'
    # does not support 0... Only the default (1) value is supported") — the
    # `temperature` field must be omitted from the request entirely, not just
    # set to 1. Defaulting to False keeps every existing ModelSpec(...)
    # construction byte-identical.
    fixed_sampling: bool = False

    def rates_dict(self) -> dict:
        """Stamp-friendly dict (drop-in for the old COST_RATES stamp shape)."""
        d = asdict(self)
        return {
            "model": d["name"],
            "usd_per_1m_in": d["usd_per_1m_in"],
            "usd_per_1m_out": d["usd_per_1m_out"],
        }


_OPENAI = "https://api.openai.com/v1"
# OpenRouter serves open-weight models behind an OpenAI-compatible endpoint. It is
# the v1 (LongMemEval-V2) reader route the user chose on 2026-09-02: the benchmark
# fixes the reader as Qwen3.5-9B, this machine cannot host it, and the hosted
# price makes a full small-tier run a few dollars. The key lives in its own
# variable so a run cannot accidentally be priced or authenticated as OpenAI.
_OPENROUTER = "https://openrouter.ai/api/v1"

MODEL_REGISTRY: dict[str, ModelSpec] = {
    s.name: s
    for s in (
        # Anchor + judge pin. Further models (luna, flash-lite, ...) are added when a
        # track's quote actually needs them — seed list stays honest about what's wired.
        ModelSpec("gpt-4o-mini", _OPENAI, "OPENAI_API_KEY", 0.15, 0.60),
        ModelSpec("gpt-4o-2024-08-06", _OPENAI, "OPENAI_API_KEY", 2.50, 10.00),
        # Track 1 model axis (user-approved 2026-08-03): list prices verified 2026-07-31
        # via web sweep (pricepertoken/openrouter) at Luna's post-2026-07-30 cut.
        ModelSpec(
            "gpt-5.6-luna",
            _OPENAI,
            "OPENAI_API_KEY",
            0.20,
            1.20,
            max_tokens_key="max_completion_tokens",
            fixed_sampling=True,
        ),
        # LongMemEval quote confirmation — the §10.3 open item "terra/sol 레지스트리
        # 등록 (견적 확정에 필요)". Prices copied verbatim from
        # docs/research/longmemeval.md §8.2 (lines 721-722; 2026-07-30 인하 반영):
        # terra $2.00/$12.00, sol $5.00/$30.00 per 1M in/out. The dialect flags are
        # the gpt-5.6 family's (same as luna above: `max_completion_tokens`, fixed
        # sampling) — family wiring, not a priced fact from the doc.
        ModelSpec(
            "gpt-5.6-terra",
            _OPENAI,
            "OPENAI_API_KEY",
            2.00,
            12.00,
            max_tokens_key="max_completion_tokens",
            fixed_sampling=True,
        ),
        ModelSpec(
            "gpt-5.6-sol",
            _OPENAI,
            "OPENAI_API_KEY",
            5.00,
            30.00,
            max_tokens_key="max_completion_tokens",
            fixed_sampling=True,
        ),
        # Embedding models. They live in the same registry because they are
        # priced the same way and quoted in the same table — a run that folds its
        # embedder spend in under an `embed` role prices it through
        # `registry_cost_usd_split({"embed": "text-embedding-3-small"})`, exactly
        # as a split --judge-model is priced. Output rate is a hard 0.0:
        # embeddings have no completion tokens, and pricing `embed` at a chat
        # model's rates would overstate it 7.5x. List price verified 2026-08-04.
        ModelSpec("text-embedding-3-small", _OPENAI, "OPENAI_API_KEY", 0.02, 0.0),
        # LongMemEval-V2's fixed reader (docs/_internal/plans/2026-09-02-v1-experience-memory.md
        # §5). List price on OpenRouter verified 2026-09-02 by the hosting survey:
        # $0.10 / $0.15 per 1M in/out, 262,144-token context. The same rate was
        # listed by DeepInfra and SiliconFlow that day. OpenRouter's model id keeps
        # the vendor prefix; the leaderboard packager only checks that the reader
        # name contains "qwen3.5-9b", which this does.
        ModelSpec("qwen/qwen3.5-9b", _OPENROUTER, "OPENROUTER_API_KEY", 0.10, 0.15),
    )
}


def get_model(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"model {name!r} is not in MODEL_REGISTRY (known: {sorted(MODEL_REGISTRY)}). "
            f"Register it in src/agmem/bench/registry.py with endpoint/key-env/prices."
        ) from None


def registry_cost_usd(merged_budget: dict, model: str) -> float:
    """USD from summed per-role token counts at `model`'s registered rates."""
    return registry_cost_usd_split(merged_budget, model)


def registry_cost_usd_split(
    merged_budget: dict, model: str, role_models: dict[str, str] | None = None
) -> float:
    """USD from per-role token counts, pricing each role at its own model's rates.

    Roles named in `role_models` (e.g. ``{"judge": "gpt-4o-2024-08-06"}``) are
    priced at that override model's registered rates; every other role prices at
    `model`'s rates. ``role_models=None`` (or empty) reproduces
    ``registry_cost_usd`` exactly — every role at `model`'s rates — which is what
    a caller with no split judge wants."""
    role_models = role_models or {}
    spec = get_model(model)
    override_specs = {role: get_model(m) for role, m in role_models.items()}
    total = 0.0
    for role, s in merged_budget.items():
        rs = override_specs.get(role, spec)
        total += s.get("tokens_in", 0) / 1_000_000 * rs.usd_per_1m_in
        total += s.get("tokens_out", 0) / 1_000_000 * rs.usd_per_1m_out
    return round(total, 6)
