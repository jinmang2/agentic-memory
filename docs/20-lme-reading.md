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

All four arms 500/500, judged by the pinned `gpt-4o-2024-08-06`, $3.74 total. **[실측]**

| arm | task-averaged | overall | abstention | cost |
|---|---|---|---|---|
| A2 mini × `direct` | 79.57 | 79.20 | 70.00 | $0.76 |
| A1 mini × `con` | 83.89 | 83.60 | 83.33 | $1.05 |
| A4 luna × `direct` | 89.96 | 91.40 | 70.00 | $0.91 |
| A3 luna × `con` | **92.14** | **94.60** | 86.67 | $1.02 |
| | **spread 12.57 pp** | **spread 15.40 pp** | | |

**Both spreads clear the pre-registered 5 pp. C4 holds.** The memory was a constant across all four
arms — the same 500 questions, the same evidence sessions, the same bytes in the prompt — and the
score moved 12.6 to 15.4 points. For comparison, Zep's entire memory system is reported as +11.0 pp
over a full-context baseline on `_s` with gpt-4o, and +8.4 pp with mini (§5.2). **Changing nothing but
who reads and how they are asked to read moves this benchmark further than a memory system's whole
claimed contribution.**

### Which knob moved it

Paired over the same 500 questions, bootstrap 10,000 resamples (stratified within type for the
task-averaged statistic), seed 0, plus exact McNemar:

```
reader model, reading held fixed
  luna - mini   @ direct    task_avg +10.39 [ +6.76, +14.12]   overall +12.20 [ +9.00, +15.40]   McNemar 68/7  p<1e-4
  luna - mini   @ con       task_avg  +8.25 [ +3.76, +12.72]   overall +11.00 [ +7.80, +14.40]   McNemar 68/13 p<1e-4

reading method, model held fixed
  con  - direct @ mini      task_avg  +4.32 [ +0.58,  +8.10]   overall  +4.40 [ +1.00,  +7.80]   McNemar 50/28 p=0.017
  con  - direct @ luna      task_avg  +2.18 [ -1.09,  +5.54]   overall  +3.20 [ +1.00,  +5.40]   McNemar 24/8  p=0.007

corner to corner
  luna·con - mini·direct    task_avg +12.57 [ +7.95, +17.20]   overall +15.40 [+12.00, +19.00]   McNemar 84/7  p<1e-4
```

Every interval excludes zero except one: CoN over direct on luna's task-averaged number, where the
point estimate is +2.18 and the interval covers zero — reported as not separated even though the same
pair separates on `overall` and on McNemar. That is the pre-registration being followed rather than
the reading being chosen.

**The reader is the bigger knob, by roughly 3x.** Swapping gpt-4o-mini for gpt-5.6-luna is worth
+11 to +12 pp; switching direct to chain-of-note is worth +3 to +4 pp. Both are memory-system-sized
effects, and neither is a memory system. This is the same ordering Mastra reports from the other
direction (memory fixed, reader gpt-4o → gpt-5-mini, +10.6 pp — §5.3) and the same ordering
MemMachine's own ablation reports (model swap +2.6 pp vs its entire ingest improvement +0.8 pp).

### CoN's sign, and where it comes from

§1.3 flagged that CoN's benefit flips sign with context length in the paper's own Figure 3b (Llama-70B
oracle +10.4 pp, `_s` −4.8 pp) while the text reports only "up to 10 pp". **At oracle length the sign
is positive for both 2026 readers** (+4.4 mini, +3.2 luna on overall), which is the direction the
paper's oracle column shows. We did not buy the `_s` half, so this measures the sign at one length,
not the flip.

Where CoN's gain lands is more specific than the headline: **abstention** is +13.3 pp on mini
(70.00 → 83.33) and +16.7 pp on luna (70.00 → 86.67), the single largest per-type move in the study,
while the two single-session recall types do not move at all in the direction of the headline (SSU
97.14 in both mini arms; SSA 100.0 → 98.21, i.e. one question *worse* under CoN).

One caveat we can show rather than assert. Reading the abstention flips by hand, some are presentation
rather than comprehension — both readings identify the false premise, and the judge grades the
structured one correct and the flat one wrong:

> *Q: "Which task did I complete first, fixing the fence or purchasing three cows from Peter?"*
> **con** — "1. You said you fixed the broken fence … three weeks ago. 2. The history does **not
> mention purchasing three cows from Peter** …"  → graded correct
> **direct** — "You completed fixing the fence first. … purchasing three cows from Peter isn't
> mentioned." → graded wrong

Five of the thirty abstention questions flip between the two arms on luna. This is P3 (the judge is a
free variable) showing up inside a single controlled contrast, and it is a reason to read the
abstention row as the softest of the seven, not a reason to discard it.

### Per type

| type | n | mini direct | mini con | luna direct | luna con |
|---|---|---|---|---|---|
| single-session-user | 70 | 97.14 | 97.14 | 100.0 | 100.0 |
| single-session-assistant | 56 | 100.0 | 98.21 | 100.0 | 100.0 |
| single-session-preference | 30 | 53.33 | 63.33 | 66.67 | 66.67 |
| multi-session | 133 | 72.93 | 75.19 | 87.22 | 93.23 |
| temporal-reasoning | 133 | 70.68 | 79.70 | 90.98 | 95.49 |
| knowledge-update | 78 | 83.33 | 89.74 | 94.87 | 97.44 |
| *abstention (cross-cut)* | 30 | 70.00 | 83.33 | 70.00 | 86.67 |

The spread is not spread evenly. The two single-session recall types are saturated in every arm —
they are at or near ceiling with evidence-only context, so nothing about reading can move them. All
of the movement is in **multi-session (+20.3 pp across the corners), temporal-reasoning (+24.8) and
preference (+13.3)** — the types that require combining or comparing what was retrieved rather than
locating it. That is a mechanism, not just a number: when retrieval is perfect, what is left to win is
reasoning over the retrieved text, and that is exactly the part a memory system does not own.

**Preference is the floor everywhere** (53–67), and it is the type graded against a rubric rather than
an answer — the type where the paper's own judge agrees with humans only .90 of the time (Table 6).

### The paper anchor

`longmemeval_oracle.json` is byte-identical across the withdrawn and cleaned releases (§3.2), so this
is the one LongMemEval comparison that is legitimate across time. The paper's GPT-4o oracle is
**.870 direct / .924 CoN** — though it does not say which of the two accuracies those are (P1), so
both of our columns are shown:

| | direct | CoN |
|---|---|---|
| paper, GPT-4o (2024) | .870 | .924 |
| ours, gpt-4o-mini | .792 / .796 | .836 / .839 |
| ours, gpt-5.6-luna | .914 / .900 | **.946 / .921** |

(cells are overall / task-averaged). Not a reproduction — different readers — but the two facts worth
keeping are that the paper's +5.4 pp direct→CoN gap is bracketed by our +4.4 (mini) and +3.2 (luna),
and that **Mastra's self-reported oracle of 82.4 sits between our mini arms**, below both luna arms and
9–10 points under the paper's own oracle. §5.3 read that as a sign their oracle reading setting was
weak rather than their system being strong; this measurement is consistent with that reading.

### What this does and does not license

**Does**: C4's claim — *hold the memory constant and the score still moves by a memory-system-sized
amount.* The rebuttal available to C1–C3 ("the arms had different memories") cannot be made here.

**Does not**: any claim that memory systems do not help. This measures the ceiling condition, where
retrieval is already perfect; it says nothing about what a memory system buys when retrieval is the
problem. It is also a single seed, one dataset variant, and two readers, and the judge is a free
variable we pinned but did not validate (P3).

**Cost, for the ledger**: four arms, 4,000 paid calls, $3.74 total, ~13 minutes of wall clock. The
smoke that preceded them was $0.03. This is the cheapest link in claim C's chain and it is the one
that closes it.

### Reproduce

```bash
uv run python scripts/repro/exp_lme_reading.py --dataset oracle --reading con \
    --reader gpt-4o-mini --workers 8 --max-spend-usd 2.0          # one arm, ~$1
uv run --with scipy python scripts/repro/lme_c4_analysis.py --arms \
    gpt-4o-mini_lme_oracle_direct gpt-4o-mini_lme_oracle_con \
    gpt-5.6-luna_lme_oracle_direct gpt-5.6-luna_lme_oracle_con    # $0
```

Artifacts per arm: `results/repro/<tag>.json` (committed), `<tag>.records.jsonl` (per-row verdict,
prompt sha256, usage, timing) and `<tag>.llm-trace.jsonl` (every prompt and every judge reply
verbatim, ~15 MB/arm) — both gitignored, both kept locally so any of this can be re-scored without
re-spending. `results/repro/lme_c4_paired.json` holds the paired statistics above.
