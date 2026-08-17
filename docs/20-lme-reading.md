# LongMemEval C4 — hold the memory constant, move only the reading

Claim C ([`research/longmemeval.md`](research/longmemeval.md) §9) says: *write spend buys no
measurable score; the score is decided by what the answerer was handed.* Three of its four links are
measured ([`18-locomo-4way.md`](18-locomo-4way.md) C1/C2, MemMachine's own ablation C3), and
each of them compares arms whose **memories differ**, so each invites the same rebuttal — *maybe the
memory is why*. C4 removes the rebuttal by removing the variable:

> `longmemeval_oracle.json` ships the evidence sessions and nothing else. It is the benchmark's own
> definition of perfect retrieval. Whatever moves the score there moves for a reason that is not the
> memory system, because there is no memory system left to move.

Four arms over the same 500 questions, differing only in the reader model and the reading prompt:

| arm | dataset | reader | reading | quote (upper bound) |
|---|---|---|---|---|
| A1 | oracle | gpt-4o-mini | `con` (chain-of-note) | $0.94 |
| A2 | oracle | gpt-4o-mini | `direct` | $0.85 |
| A3 | oracle | gpt-5.6-luna | `con` | $1.33 |
| A4 | oracle | gpt-5.6-luna | `direct` | $1.15 |

Quotes are `scripts/repro/exp_lme_reading.py --dry-run` **[실측]**: the prompts are the real rendered
prompts (3,045,858 estimated input tokens per `con` arm, against §3.1's measured 3.06M), completions
priced at the benchmark's `max_tokens` ceiling. They exclude the judge's real hypothesis text, which
adds roughly $0.5 per arm at the pinned gpt-4o judge's rates.

## Pre-registration (written 2026-08-17, before any call was paid for)

This section is the analysis plan, fixed before the data existed. It is here rather than in a private
note so that a reader can check the result against the rule that was set for it, not against a rule
chosen once the numbers were in.

**Primary measure.** The spread (max − min) across the four arms, reported for **both** official
accuracies — `task_averaged` and `overall` — separately, and never as a single unlabelled "accuracy"
(P1: three different numbers travel under that name, LME-A13).

**Decision rule.**

| spread | verdict |
|---|---|
| **≥ 5 pp** | C4 holds. Reading alone moves the score by as much as memory systems claim for their memory (Zep +11.0 pp abs on `_s` with gpt-4o, +8.4 pp with mini; the paper's own oracle direct→CoN is +5.4 pp, the expected floor). |
| 2–5 pp | C4 holds weakly. The claim's wording drops to "a substantial part of". |
| < 2 pp | **C4 fails and is reported as failing.** Claim C keeps only C1–C3 and stays in its negative form. |

**Paired comparisons.** All four arms answer the same 500 questions, so every pairwise contrast is
paired: McNemar plus a bootstrap 95% CI (`scripts/ext/x1_power.py`, the resampler docs/19 used). A
spread that clears 5 pp but whose interval covers zero is reported as not separated.

**Secondary, pre-registered as secondary.**

1. **The sign of CoN.** §1.3 records that CoN's benefit flips sign with context length in the paper's
   own Figure 3b (Llama-70B: oracle +10.4 pp, `_s` −4.8 pp) while the text reports only "up to 10 pp".
   A1−A2 and A3−A4 measure that sign at oracle length for two 2026 readers.
2. **The paper anchor.** `longmemeval_oracle.json` is byte-identical across the withdrawn and cleaned
   releases (§3.2, sha256 `821a2034…`), so it is the **only** LongMemEval number that may be compared
   across time. The paper's GPT-4o oracle is .870 (direct) / .924 (CoN). Our readers are not GPT-4o,
   so this is an anchor, not a reproduction — stated in advance so it cannot be presented as one later.
3. **Per-type decomposition**, 7 rows: the six question types plus the abstention cross-cut (which is
   a cross-cut, not a seventh type — LME-A14).

**What would falsify the setup rather than the claim.** If the arms differ by less than 2 pp *and*
all four sit far below the paper's .870 anchor, the reading is that our harness is weak, not that
reading does not matter. The stamp records everything needed to tell those apart: data sha256, judge
pin, reading method, session sort, and the absence of any session or token cap.

## Protocol

| | |
|---|---|
| Data | `longmemeval_oracle.json`, sha256 `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c` — the file the paper measured, byte-for-byte |
| Population | **500 questions, all of them.** A partial run is not scored (LME-A18) |
| Context | full haystack, `history_format=json`, date-sorted, `has_answer` stripped — byte-identical to upstream's `prepare_prompt` (500/500, `scripts/repro/lme_audit/prompt_rediff.py` config D) |
| Memory | `passthrough`, one fresh in-memory store per question, fake embedder (nothing is retrieved) |
| Reader | `gpt-4o-mini` at temperature 0 / `gpt-5.6-luna`, which admits no temperature at all (deviation D6, disclosed in the stamp) |
| Judge | **`gpt-4o-2024-08-06`**, temperature 0, `max_tokens` 10, per-type prompt, `'yes' in reply` — the pin the official aggregator asserts |
| Reported | `task_averaged`, `overall`, abstention, per-type — all four, always |
| Runner | `scripts/repro/exp_lme_reading.py`, workers 8 |

## Results

*Not yet run. Phase 2 (paid smoke, 10 questions) precedes the four arms; this section is filled from
the run artifacts, and the pre-registration above is not edited when it is.*
