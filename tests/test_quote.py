"""Dry-run quote CLI (scripts/repro/quote.py): pure-math pricing tests plus a
zero-spend, small-conversation integration test through CountingLLM — never the
real (huge) LoCoMo conversation, which is exercised only by the manual smoke
command (no test may make this suite slow or pull the real dataset into CI)."""

from __future__ import annotations

import importlib.util as _ilu
import json
import sys
from pathlib import Path

import pytest

_QUOTE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "repro" / "quote.py"


def _load_quote():
    # scripts/repro/quote.py itself inserts scripts/ and scripts/repro/ into
    # sys.path on import (needed for its own `exp_amem_repro` / `configs`
    # imports) -- spec_from_file_location alone would not add either.
    spec = _ilu.spec_from_file_location("repro_quote", _QUOTE_PATH)
    mod = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


quote = _load_quote()
estimate_quote = quote.estimate_quote
tok_means_from_summary = quote.tok_means_from_summary
canned_profile_for = quote.canned_profile_for
quote_for_sample = quote.quote_for_sample


def test_estimate_quote_prices_measured_input_and_calibrated_output():
    """Input comes from the MEASURED prompt characters, output from the
    calibrated per-call mean. Mixing the two up is the Track 2 failure mode:
    borrowed per-call input means were 2.1x low across organizer families."""
    calls = {"extract": 100, "distill": 100}
    means = {"extract": (600.0, 150.0), "distill": (700.0, 200.0)}  # (in, out) tokens/call
    chars = {"extract": 404500, "distill": 809000}
    q = estimate_quote(calls, means, "gpt-4o-mini", chars, chars_per_token=4.045)
    tin = (404500 + 809000) / 4.045
    tout = 100 * 150 + 100 * 200
    assert q["tokens_in_measured"] == round(tin) and q["tokens_out_est"] == tout
    # the calibrated INPUT means (600/700) must not have been used at all
    assert q["tokens_in_measured"] != 100 * 600 + 100 * 700
    assert q["cost_usd_est"] == pytest.approx(tin / 1e6 * 0.15 + tout / 1e6 * 0.60, rel=1e-6)
    assert q["model"] == "gpt-4o-mini" and q["calls_total"] == 200


def test_estimate_quote_role_without_calibration_fails_loud():
    with pytest.raises(KeyError):
        estimate_quote({"judge": 10}, {}, "gpt-4o-mini", {"judge": 100})


def test_estimate_quote_role_without_measured_prompts_fails_loud():
    """A counted call the prompt tap never saw would silently price at zero
    input. That is a wiring bug in the tap, and it must not degrade quietly into
    a cheap-looking quote."""
    with pytest.raises(KeyError):
        estimate_quote({"extract": 10}, {"extract": (1.0, 1.0)}, "gpt-4o-mini", {})


def test_tok_means_from_summary_reads_merged_budget(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "llm_budget": {
                    "extract": {"calls": 4, "tokens_in": 400, "tokens_out": 40},
                    "distill": {"calls": 2, "tokens_in": 1000, "tokens_out": 100},
                }
            }
        )
    )
    means = tok_means_from_summary(summary)
    assert means["extract"] == (100.0, 10.0)
    assert means["distill"] == (500.0, 50.0)


def test_tok_means_from_summary_skips_zero_call_roles(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"llm_budget": {"generate": {"calls": 0, "tokens_in": 0, "tokens_out": 0}}})
    )
    assert tok_means_from_summary(summary) == {}


def test_tok_means_from_summary_missing_budget_is_empty(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"stamp": {}}))
    assert tok_means_from_summary(summary) == {}


def test_canned_profile_for_known_configs():
    assert canned_profile_for("amem") == "amem"
    assert canned_profile_for("nemori_upstream") == "nemori"
    assert canned_profile_for("nemori_merge085") == "nemori"


def test_canned_profile_for_unmapped_config_fails_loud():
    with pytest.raises(KeyError):
        canned_profile_for("mem0")


_FAKE_SAMPLE = {
    "conversation": {
        "session_1_date_time": "1:00 pm on 1 January, 2021",
        "session_1": [
            {"speaker": "Alice", "text": "I moved to Berlin last year."},
            {"speaker": "Bob", "text": "How is the new job going?"},
            {"speaker": "Alice", "text": "The job at the museum is great."},
        ],
    }
}


def _calibration_summary(tmp_path):
    p = tmp_path / "calib.json"
    p.write_text(
        json.dumps(
            {
                "llm_budget": {
                    "extract": {"calls": 419, "tokens_in": 63242, "tokens_out": 25242},
                    "distill": {"calls": 418, "tokens_in": 468374, "tokens_out": 125247},
                }
            }
        )
    )
    return p


def test_quote_for_sample_is_zero_spend_and_priced(tmp_path):
    calib = _calibration_summary(tmp_path)
    q = quote_for_sample(_FAKE_SAMPLE, "amem", "gpt-4o-mini", calib)
    # the real organizer write path ran (extract per note; distill once a
    # neighbor note exists) -- both roles are covered by the calibration.
    assert q["calls_per_role"].get("extract", 0) >= 3
    assert q["calls_per_role"].get("distill", 0) >= 1
    assert q["calls_total"] == sum(q["calls_per_role"].values())
    assert q["tokens_in_measured"] > 0 and q["tokens_out_est"] > 0
    assert q["cost_usd_est"] > 0
    assert q["model"] == "gpt-4o-mini"
    assert q["config"] == "amem"
    assert q["calibrated_from"] == str(calib)


def test_quote_for_sample_unmapped_config_fails_loud(tmp_path):
    calib = _calibration_summary(tmp_path)
    with pytest.raises(KeyError):
        quote_for_sample(_FAKE_SAMPLE, "mem0", "gpt-4o-mini", calib)


def test_quote_measures_prompts_per_call_site(tmp_path):
    """The per-site table is the auditable part of a quote: it maps counted
    calls onto the organizer's own control flow, so a site that costs more than
    its call count suggests is visible instead of averaged away inside a role."""
    calib = _calibration_summary(tmp_path)
    q = quote_for_sample(_FAKE_SAMPLE, "amem", "gpt-4o-mini", calib)
    sites = q["calls_per_site"]
    assert sites, "no call sites recorded — the prompt tap did not run"
    assert all(s.startswith(("extract:", "distill:")) for s in sites), sites
    assert sum(sites.values()) == q["calls_total"]
    # The label must come from the organizer's own prompt. `StructuredCaller`
    # opens every structured call with the same JSON-mode system turn, so a tap
    # that labels off the joined message list names every site after that
    # preamble and the whole table silently becomes one bucket.
    assert not any("json_object" in s for s in sites), sites
    # A-Mem issues two distinct prompts (Ps1 extraction, Ps3 evolution); if the
    # labeller cannot tell them apart it is not distinguishing anything.
    assert len(sites) >= 2, sites
    # every site carries measured characters, and they sum to the role totals
    for role in q["calls_per_role"]:
        per_site = sum(v for k, v in q["prompt_chars_measured"].items() if k.startswith(f"{role}:"))
        assert per_site == q["prompt_chars_measured"][role] > 0


def test_quote_for_zep_covers_its_yield_band(tmp_path):
    """`zep_cross_encoder` maps to the CENTRAL point of a yield band. The other
    two points must remain quotable, because Zep's edge-resolution and community
    calls scale with a per-message entity/fact yield that no canned response can
    measure — a single number here would read as measured when it is not."""
    from agmem.bench.counting import CANNED_RESPONSES

    assert canned_profile_for("zep_cross_encoder") == "zep"
    assert {"zep", "zep_low", "zep_high"} <= set(CANNED_RESPONSES)
