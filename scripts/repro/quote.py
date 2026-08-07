"""Dry-run quote for the Phase 2 approval gate: run an organizer config's ingest
over a LoCoMo conv subset with the CountingLLM (zero API calls), price the counted
calls, and emit the quote JSON the user approves before any spend.

INPUT tokens are MEASURED, not calibrated. The CountingLLM drives the organizer's
real prompt templates over the real corpus, so the prompt string every call would
have carried exists in memory during the dry run — it only has to be weighed. Track
2 established why this matters: pricing Mem0's ingest from A-Mem's per-call token
means was 2.1x low, because Mem0 puts a 3.2k-character system prompt on every
extraction and A-Mem has no counterpart. Borrowed per-call means cannot survive a
change of prompt family, and every new arm is a new prompt family. Measured on
conv0 this technique came within -7.3% of the real Mem0 ingest.

OUTPUT tokens still come from `--calibrate-from`, a REAL prior summary artifact: a
canned response is the one thing in the dry run that is NOT the real thing, so its
length carries no information. A role with no calibration data fails loud — an
uncalibrated guess presented as a quote is how budgets die. When the calibration
source is a different organizer family, the output side is BORROWED and the quote
says so.

IMPORTANT: this quote counts INGEST (write-path: extract/distill) calls only — it
does not count eval (generate/judge) calls, so the printed and written cost is a
FLOOR on the full experiment's cost, not the total."""

from __future__ import annotations

import argparse
import json
import re
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
from agmem.llm.structured import StructuredCaller

# The canned-response profile (agmem.bench.counting.CANNED_RESPONSES) that
# schema-validly answers each runner config's organizer family. A config with
# no entry here has no CountingLLM coverage yet -- canned_profile_for refuses
# to guess, so quoting it is impossible rather than silently wrong.
CONFIG_TO_CANNED: dict[str, str] = {
    "amem": "amem",
    "nemori_upstream": "nemori",
    "nemori_merge085": "nemori",
    "mem0_v0194": "mem0",
    # Track 3. "zep" is the CENTRAL point of a yield band, not a measurement:
    # Zep's edge-resolution and community calls scale with how many entities and
    # facts each message yields, which no canned response can measure. Quote this
    # config against "zep_low"/"zep_high" too (both are registered in
    # CANNED_RESPONSES) and report the range — see counting.py's zep section.
    "zep_cross_encoder": "zep",
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


# Measured on this corpus during Track 2: 4.045 characters per token across 1,674
# real gpt-4o-mini calls. Used to convert MEASURED prompt characters into input
# tokens without a tokenizer dependency; it is corpus-specific, not universal.
CHARS_PER_TOKEN = 4.045


def estimate_quote(
    calls_per_role: dict[str, int],
    tok_means: dict[str, tuple[float, float]],
    model: str,
    prompt_chars: dict[str, int],
    chars_per_token: float = CHARS_PER_TOKEN,
    tokens_out_per_call: dict[str, float] | None = None,
) -> dict:
    """Price counted calls: input from `prompt_chars` (MEASURED during the dry
    run), output from `tok_means` (calibrated from a real prior run) unless
    `tokens_out_per_call` overrides a role.

    The override exists because a calibrated OUTPUT mean is only transferable
    within a prompt family, and nothing enforces that the calibration source is
    in the same one. A-Mem's `distill` averages 307.7 output tokens per call
    (its note-evolution JSON); Zep's `distill` is 85% edge-resolution answering
    `{"duplicate_of": null, "contradicts": []}` — about 15 tokens. Silently
    borrowing the former to price the latter is a ~20x error on the site that
    dominates that role, and it would look exactly like a calibrated quote.
    Whichever source a role used is recorded per role, so the artifact always
    says which numbers were measured, which calibrated, and which assumed."""
    spec = get_model(model)
    overrides = tokens_out_per_call or {}
    tin = tout = 0.0
    out_per_call: dict[str, float] = {}
    out_source: dict[str, str] = {}
    for role, n in calls_per_role.items():
        if role not in overrides and role not in tok_means:
            raise KeyError(
                f"no token calibration for role {role!r} (have: {sorted(tok_means)}). "
                f"Pass --calibrate-from a summary whose budget covers this role, or "
                f"--tokens-out-per-call {role}=N to state an assumption explicitly."
            )
        if role not in prompt_chars:
            raise KeyError(
                f"no measured prompt characters for role {role!r} — the counting run "
                f"issued calls the prompt tap did not see, so the input side would be "
                f"silently short. This is a wiring bug, not a calibration gap."
            )
        per_call = overrides[role] if role in overrides else tok_means[role][1]
        out_per_call[role] = round(per_call, 1)
        out_source[role] = "assumed" if role in overrides else "calibrated"
        tin += prompt_chars[role] / chars_per_token
        tout += n * per_call
    return {
        "model": spec.name,
        "calls_total": sum(calls_per_role.values()),
        "calls_per_role": dict(sorted(calls_per_role.items())),
        "tokens_in_measured": round(tin),
        "tokens_out_est": round(tout),
        "tokens_out_per_call": dict(sorted(out_per_call.items())),
        "tokens_out_source": dict(sorted(out_source.items())),
        "chars_per_token": chars_per_token,
        "cost_usd_est": round(tin / 1e6 * spec.usd_per_1m_in + tout / 1e6 * spec.usd_per_1m_out, 4),
        "rates": spec.rates_dict(),
    }


def _parse_tokens_out(spec: str | None) -> dict[str, float]:
    """`"extract=120,distill=45"` -> `{"extract": 120.0, "distill": 45.0}`."""
    if not spec:
        return {}
    out = {}
    for item in spec.split(","):
        role, _, value = item.partition("=")
        if not value:
            raise ValueError(f"--tokens-out-per-call entry {item!r} is not role=NUMBER")
        out[role.strip()] = float(value)
    return out


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


def _tap_prompts(mem, fake, prompt_chars: dict[str, int], call_sites: dict[str, int]):
    """Weigh every prompt the counting run issues, by role and by call site.

    The per-SITE split is not decoration: an organizer's roles say nothing about
    which of its prompts is expensive, and the site table is what makes a quote
    auditable against the organizer's own control flow (Zep's edge-resolution
    site carries three quarters of its input tokens while sharing the `distill`
    role with two community prompts a tenth its size)."""
    original = fake.chat

    def chat(role, messages, budget_key=None, **overrides):
        # Weigh the WHOLE conversation (that is what the endpoint bills), but
        # label from the USER turn: `StructuredCaller` puts the same JSON-mode
        # instruction in the system turn of every structured call, so labelling
        # off the joined string collapses every site of every organizer into one
        # bucket named after the harness's own preamble.
        prompt = " ".join(m.get("content", "") for m in messages)
        user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
        prompt_chars[role] = prompt_chars.get(role, 0) + len(prompt)
        site = f"{role}:{_site_label(user_turns[-1] if user_turns else prompt)}"
        prompt_chars[site] = prompt_chars.get(site, 0) + len(prompt)
        call_sites[site] = call_sites.get(site, 0) + 1
        return original(role, messages, budget_key=budget_key, **overrides)

    fake.chat = chat
    mem.llm = fake
    mem.structured = StructuredCaller(fake, use_guided_json=False)
    mem._ctx.llm = mem.structured


def _site_label(prompt: str) -> str:
    """A stable, organizer-agnostic name for a prompt: its first meaningful
    line, normalized. Derived from the prompt itself rather than a registry so a
    newly added prompt shows up in the table instead of vanishing into a
    catch-all bucket."""
    for line in prompt.splitlines():
        stripped = line.strip()
        if len(stripped) > 12:
            return re.sub(r"[^a-z0-9]+", "_", stripped.lower())[:48].strip("_")
    return "unlabeled"


def count_for_sample(sample: dict, config: str, canned: str | None = None) -> dict:
    """Drive one LoCoMo-shaped conversation `sample` through `config`'s organizer
    arm on a CountingLLM (zero API calls, real call counts), returning the counted
    calls and the measured size of every prompt they carried.

    `canned` overrides the config's mapped profile — the band points of a config
    whose call count is not fixed by its control flow are quoted this way."""
    canned = canned or canned_profile_for(config)
    cfg_entry = get_config(config)
    prompt_chars: dict[str, int] = {}
    call_sites: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as tmp:
        mem, fake = build_counting_memory(
            canned, cfg_entry.factory, Path(tmp), "quote", cfg_entry.memory_types
        )
        _tap_prompts(mem, fake, prompt_chars, call_sites)
        try:
            # Mirror exp_amem_repro's real ingest path exactly (eval_conversations
            # calls the same two lines) so the counted calls match what a real
            # ingest of this conversation would issue.
            turns = locomo.ingest(mem, sample)
            mem.consolidate()
            drops = dict(mem.structured.drops)
        finally:
            mem.close()
    if drops:
        # A dropped canned response means a branch did not run, so the counts
        # below are of a control flow the real run does not have.
        raise RuntimeError(
            f"canned profile {canned!r} produced schema-invalid responses ({drops}); "
            f"the counts would be of a different control flow than the real run"
        )
    return {
        "turns": turns,
        "calls_per_role": dict(fake.calls),
        "calls_per_site": dict(sorted(call_sites.items())),
        "prompt_chars": prompt_chars,
    }


def quote_for_sample(
    sample: dict,
    config: str,
    model: str,
    calibrate_from: Path,
    chars_per_token: float = CHARS_PER_TOKEN,
    tokens_out_per_call: dict[str, float] | None = None,
    canned: str | None = None,
) -> dict:
    """Count and price ONE conversation `sample`. `run_quote` is the campaign
    entry point; this is the single-sample one, which is also what lets the tests
    exercise the whole path on a three-turn fixture instead of the real dataset."""
    counted = count_for_sample(sample, config, canned)
    quote = estimate_quote(
        counted["calls_per_role"],
        tok_means_from_summary(calibrate_from),
        model,
        counted["prompt_chars"],
        chars_per_token,
        tokens_out_per_call,
    )
    quote["config"] = config
    quote["canned_profile"] = canned or canned_profile_for(config)
    quote["turns"] = counted["turns"]
    quote["calls_per_site"] = counted["calls_per_site"]
    quote["prompt_chars_measured"] = counted["prompt_chars"]
    quote["calibrated_from"] = str(calibrate_from)
    return quote


def run_quote(
    config: str,
    convs: list[int],
    model: str,
    calibrate_from: Path,
    chars_per_token: float = CHARS_PER_TOKEN,
    tokens_out_per_call: dict[str, float] | None = None,
    canned: str | None = None,
) -> dict:
    """Count and price `config` over `convs`, summing the per-conversation runs.

    Summing is the faithful shape, not a shortcut: the campaign ingests each
    conversation into its OWN store, so cross-conversation state never exists
    and a single fused run would overstate every candidate pool."""
    samples = locomo.load_locomo(H.DATA)
    per_conv = {}
    calls: dict[str, int] = {}
    chars: dict[str, int] = {}
    sites: dict[str, int] = {}
    turns = 0
    for conv in convs:
        counted = count_for_sample(samples[conv], config, canned)
        per_conv[str(conv)] = counted
        turns += counted["turns"]
        for role, n in counted["calls_per_role"].items():
            calls[role] = calls.get(role, 0) + n
        for key, n in counted["prompt_chars"].items():
            chars[key] = chars.get(key, 0) + n
        for key, n in counted["calls_per_site"].items():
            sites[key] = sites.get(key, 0) + n

    tok_means = tok_means_from_summary(calibrate_from)
    quote = estimate_quote(calls, tok_means, model, chars, chars_per_token, tokens_out_per_call)
    quote["config"] = config
    quote["canned_profile"] = canned or canned_profile_for(config)
    quote["convs"] = convs
    quote["turns"] = turns
    quote["calls_per_site"] = dict(sorted(sites.items()))
    quote["prompt_chars_measured"] = dict(sorted(chars.items()))
    quote["calibrated_from"] = str(calibrate_from)
    quote["per_conv"] = per_conv
    return quote


def print_quote_table(quote: dict) -> None:
    print(
        f"config={quote['config']}  canned={quote['canned_profile']}  "
        f"convs={quote['convs']}  turns={quote['turns']}  model={quote['model']}"
    )
    print(f"calibrated_from={quote['calibrated_from']}  (OUTPUT tokens only)")
    print(f"{'call site':<52}{'calls':>8}{'tok_in':>12}")
    for site, n in quote["calls_per_site"].items():
        tin = round(quote["prompt_chars_measured"].get(site, 0) / quote["chars_per_token"])
        print(f"{site:<52}{n:>8}{tin:>12,}")
    print(f"{'TOTAL':<52}{quote['calls_total']:>8}{quote['tokens_in_measured']:>12,}")
    print(
        f"tokens_in_measured={quote['tokens_in_measured']:,}  "
        f"tokens_out_est={quote['tokens_out_est']:,}"
    )
    print(f"cost_usd_est=${quote['cost_usd_est']:.4f}  (rates: {quote['rates']})")
    print(
        "NOTE: this is an INGEST-ONLY estimate (write-path extract/distill calls); "
        "it does not count eval (generate/judge) calls, so it is a FLOOR on the "
        "full experiment's cost, not the total."
    )
    print(
        "NOTE: input tokens are MEASURED from the real prompts; output tokens are "
        "CALIBRATED from the run above and are borrowed if that run used a "
        "different organizer family."
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dry-run quote: counted CountingLLM calls x calibrated tokens "
        "x registry rates. Zero API calls."
    )
    ap.add_argument(
        "--config", default="amem", help="organizer config from scripts/repro/configs.py"
    )
    ap.add_argument(
        "--convs",
        default="0",
        help="LoCoMo conversation indices: 'all', or a comma-separated list (default 0)",
    )
    ap.add_argument("--model", default="gpt-4o-mini", help="registry model to price at")
    ap.add_argument(
        "--chars-per-token",
        type=float,
        default=CHARS_PER_TOKEN,
        help=f"characters per token for the measured input side (default {CHARS_PER_TOKEN}, "
        "measured on this corpus over 1,674 real gpt-4o-mini calls)",
    )
    ap.add_argument(
        "--canned",
        default=None,
        help="override the canned-response profile (default: the config's mapping). Use to "
        "quote the yield band of a config whose call count is not fixed by control flow, "
        "e.g. --canned zep_low / zep_high",
    )
    ap.add_argument(
        "--tokens-out-per-call",
        default=None,
        help="assume output tokens per call for a role instead of calibrating it, as "
        "'role=N,role=N'. Use when the calibration source is a DIFFERENT organizer "
        "family, where a borrowed output mean can be off by an order of magnitude. "
        "Recorded in the artifact as 'assumed'.",
    )
    ap.add_argument(
        "--calibrate-from",
        required=True,
        help="prior run summary JSON with an llm_budget block to calibrate token means from",
    )
    ap.add_argument("--out", required=True, help="path to write the quote JSON")
    args = ap.parse_args()

    convs = (
        list(range(len(locomo.load_locomo(H.DATA))))
        if args.convs == "all"
        else [int(c) for c in args.convs.split(",")]
    )
    quote = run_quote(
        args.config,
        convs,
        args.model,
        Path(args.calibrate_from),
        args.chars_per_token,
        _parse_tokens_out(args.tokens_out_per_call),
        args.canned,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(quote, indent=2))

    print_quote_table(quote)
    print(f"\nquote written to {out_path}")


if __name__ == "__main__":
    main()
