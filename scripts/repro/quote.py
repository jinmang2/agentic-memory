"""Dry-run quote for the Phase 2 approval gate: run an organizer config's ingest
over a LoCoMo conv subset with the CountingLLM (zero API calls), price the counted
calls with per-role token means calibrated from a REAL prior summary artifact, and
emit the quote JSON the user approves before any spend. A role with no calibration
data fails loud — an uncalibrated guess presented as a quote is how budgets die.

IMPORTANT: this quote counts INGEST (write-path: extract/distill) calls only — it
does not count eval (generate/judge) calls, so the printed and written cost is a
FLOOR on the full experiment's cost, not the total."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from agmem.bench.registry import get_model

# exp_amem_repro.py lives in scripts/ (one level up) and configs.py lives
# alongside this file in scripts/repro/ -- neither directory is guaranteed to
# already be on sys.path (it depends on how this module was reached: direct
# script execution only auto-adds scripts/repro/, and a spec_from_file_location
# load in a test adds neither), so insert both explicitly before importing.
# exp_amem_repro's __main__ guard means importing it never runs the harness.
_REPRO_DIR = Path(__file__).resolve().parent
_SCRIPTS = _REPRO_DIR.parent
for _p in (_SCRIPTS, _REPRO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import exp_amem_repro as H
from configs import get_config

from agmem.bench import locomo
from agmem.bench.counting import build_counting_memory

# The canned-response profile (agmem.bench.counting.CANNED_RESPONSES) that
# schema-validly answers each runner config's organizer family. A config with
# no entry here has no CountingLLM coverage yet -- canned_profile_for refuses
# to guess, so quoting it is impossible rather than silently wrong.
CONFIG_TO_CANNED: dict[str, str] = {
    "amem": "amem",
    "nemori_upstream": "nemori",
    "nemori_merge085": "nemori",
}


def canned_profile_for(config_name: str) -> str:
    try:
        return CONFIG_TO_CANNED[config_name]
    except KeyError:
        raise KeyError(
            f"no canned-response profile mapped for config {config_name!r} "
            f"(known: {sorted(CONFIG_TO_CANNED)}). Wire one in "
            f"agmem.bench.counting.CANNED_RESPONSES and add it to CONFIG_TO_CANNED "
            f"before quoting this config."
        ) from None


def estimate_quote(
    calls_per_role: dict[str, int], tok_means: dict[str, tuple[float, float]], model: str
) -> dict:
    spec = get_model(model)
    tin = tout = 0.0
    for role, n in calls_per_role.items():
        if role not in tok_means:
            raise KeyError(
                f"no token calibration for role {role!r} (have: {sorted(tok_means)}). "
                f"Pass --calibrate-from a summary whose budget covers this role."
            )
        m_in, m_out = tok_means[role]
        tin += n * m_in
        tout += n * m_out
    return {
        "model": spec.name,
        "calls_total": sum(calls_per_role.values()),
        "calls_per_role": dict(sorted(calls_per_role.items())),
        "tokens_in_est": round(tin),
        "tokens_out_est": round(tout),
        "cost_usd_est": round(tin / 1e6 * spec.usd_per_1m_in + tout / 1e6 * spec.usd_per_1m_out, 4),
        "rates": spec.rates_dict(),
    }


def tok_means_from_summary(path: Path) -> dict[str, tuple[float, float]]:
    """Per-role (tokens_in/call, tokens_out/call) from a prior run summary's merged
    budget dict — the calibration source that makes counted calls priceable."""
    budget = json.loads(path.read_text()).get("llm_budget") or {}
    means = {}
    for role, s in budget.items():
        calls = s.get("calls", 0)
        if calls:
            means[role] = (s["tokens_in"] / calls, s["tokens_out"] / calls)
    return means


def quote_for_sample(sample: dict, config: str, model: str, calibrate_from: Path) -> dict:
    """Drive one LoCoMo-shaped conversation `sample` through `config`'s organizer
    arm on a CountingLLM (zero API calls, real call counts), then price the
    counted calls at `model`'s registry rates using token means calibrated from
    `calibrate_from` — a REAL prior run's summary artifact."""
    canned = canned_profile_for(config)
    cfg_entry = get_config(config)
    with tempfile.TemporaryDirectory() as tmp:
        mem, fake = build_counting_memory(
            canned, cfg_entry.factory, Path(tmp), "quote", cfg_entry.memory_types
        )
        try:
            # Mirror exp_amem_repro's real ingest path exactly (eval_conversations
            # calls the same two lines) so the counted calls match what a real
            # ingest of this conversation would issue.
            locomo.ingest(mem, sample)
            mem.consolidate()
        finally:
            mem.close()

    tok_means = tok_means_from_summary(calibrate_from)
    quote = estimate_quote(fake.calls, tok_means, model)
    quote["config"] = config
    quote["calibrated_from"] = str(calibrate_from)
    return quote


def run_quote(config: str, conv: int, model: str, calibrate_from: Path) -> dict:
    sample = locomo.load_locomo(H.DATA)[conv]
    quote = quote_for_sample(sample, config, model, calibrate_from)
    quote["conv"] = conv
    return quote


def print_quote_table(quote: dict) -> None:
    print(f"config={quote['config']}  conv={quote['conv']}  model={quote['model']}")
    print(f"calibrated_from={quote['calibrated_from']}")
    print(f"{'role':<12}{'calls':>10}")
    for role, n in quote["calls_per_role"].items():
        print(f"{role:<12}{n:>10}")
    print(f"{'total':<12}{quote['calls_total']:>10}")
    print(f"tokens_in_est={quote['tokens_in_est']:,}  tokens_out_est={quote['tokens_out_est']:,}")
    print(f"cost_usd_est=${quote['cost_usd_est']:.4f}  (rates: {quote['rates']})")
    print(
        "NOTE: this is an INGEST-ONLY estimate (write-path extract/distill calls); "
        "it does not count eval (generate/judge) calls, so it is a FLOOR on the "
        "full experiment's cost, not the total."
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dry-run quote: counted CountingLLM calls x calibrated tokens "
        "x registry rates. Zero API calls."
    )
    ap.add_argument(
        "--config", default="amem", help="organizer config from scripts/repro/configs.py"
    )
    ap.add_argument("--conv", type=int, default=0, help="LoCoMo conversation index (0-9)")
    ap.add_argument("--model", default="gpt-4o-mini", help="registry model to price at")
    ap.add_argument(
        "--calibrate-from",
        required=True,
        help="prior run summary JSON with an llm_budget block to calibrate token means from",
    )
    ap.add_argument("--out", required=True, help="path to write the quote JSON")
    args = ap.parse_args()

    quote = run_quote(args.config, args.conv, args.model, Path(args.calibrate_from))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(quote, indent=2))

    print_quote_table(quote)
    print(f"\nquote written to {out_path}")


if __name__ == "__main__":
    main()
