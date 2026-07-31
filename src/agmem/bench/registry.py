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

    def rates_dict(self) -> dict:
        """Stamp-friendly dict (drop-in for the old COST_RATES stamp shape)."""
        d = asdict(self)
        return {
            "model": d["name"],
            "usd_per_1m_in": d["usd_per_1m_in"],
            "usd_per_1m_out": d["usd_per_1m_out"],
        }


_OPENAI = "https://api.openai.com/v1"

MODEL_REGISTRY: dict[str, ModelSpec] = {
    s.name: s
    for s in (
        # Anchor + judge pin. Further models (luna, flash-lite, ...) are added when a
        # track's quote actually needs them — seed list stays honest about what's wired.
        ModelSpec("gpt-4o-mini", _OPENAI, "OPENAI_API_KEY", 0.15, 0.60),
        ModelSpec("gpt-4o-2024-08-06", _OPENAI, "OPENAI_API_KEY", 2.50, 10.00),
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
