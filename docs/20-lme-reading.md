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

## `_s`: the same 500 questions, 19x the haystack

Everything above is the ceiling condition. `longmemeval_s` is the same 500
questions with the evidence buried in ~48 filler sessions — 113,840 tokens at the
median against oracle's 6,127 — and it is where a memory system would have
something to do. One arm, full context, no memory: **gpt-4o-mini × con, 500/500,
$9.07.** **[실측]**

| | oracle | `_s` | drop |
|---|---|---|---|
| task-averaged | 83.89 | **58.40** | −25.49 |
| overall | 83.60 | **60.40** | −23.20 |
| single-session-user | 97.14 | 84.29 | −12.85 |
| single-session-assistant | 98.21 | 92.86 | −5.35 |
| **single-session-preference** | 63.33 | **3.33** | **−60.00** |
| multi-session | 75.19 | 50.38 | −24.81 |
| temporal-reasoning | 79.70 | 54.14 | −25.56 |
| knowledge-update | 89.74 | 65.38 | −24.36 |
| abstention | 83.33 | 73.33 | −10.00 |

**The 23-point drop corroborates the paper with a different reader.** GPT-4o falls
.924 → .640 with CoN (−28.4 pp, −30.7% relative); our gpt-4o-mini falls
83.60 → 60.40 (−23.2 pp, −27.8% relative). Nothing about the prompt was capped —
`max_history_tokens` is never passed, so the whole 523 KB haystack goes to the
API, which is what upstream does too (its truncation is a no-op on this data).

**Reading a haystack that FITS still costs 23 points.** `_s` is inside a 128 K
window by construction (§2.7: it was packed to `enforce_json_length=115000`), so
this is not a truncation effect. It is the reason the benchmark still has
something to measure in 2026 — and the reason a memory system can be worth
paying for even when the context technically fits.

### One type does not degrade — it collapses

**single-session-preference goes 63.33 → 3.33.** One of thirty. That is not a
harness artefact, and it was checked by hand before being written down:

> *Q: "Can you recommend some resources where I can learn more about video editing?"*
> *Rubric: the user prefers resources specific to Adobe Premiere Pro, especially its advanced settings.*
> **oracle** (1 session, 27 KB) — "From the chat history, the assistant provided several resources for learning about **Adobe Premiere Pro**…" → correct
> **`_s`** (50 sessions, 523 KB) — "Video editing can be learned through various types of resources: Online courses… **Udemy**… YouTube tutorials…" → generic, correct answer to a question nobody asked

The model does not retrieve the preference *and misuse* it. It **stops
personalising at all** and answers the question as if no history existed. At
113 K tokens, the first thing to die is not recall of facts (SSU is still 84.29,
SSA 92.86) but the *use of the reader's own context* to shape an answer.

That lands directly on a published claim. Zep reports its largest relative gain
on exactly this type — **preference +77.7%** (§5.2) — and the number beside it
here says why that is so easy: the full-context baseline it improves on is at 3%.
**A memory system does not have to be good at preference to post a huge relative
gain on it; it only has to put the relevant session in front of the reader.**

### Against the published baselines — carefully

Zep's `_s` baselines are 55.4 (gpt-4o-mini) and 60.2 (gpt-4o) full-context, with
Zep itself at 63.8 / 71.2. **Our 60.4 for full-context gpt-4o-mini is 5 points
above their gpt-4o-mini baseline**, and the release explains the direction: theirs
is the withdrawn release, ours is `cleaned`, and cleaning was a pure deletion of
1,243 distractor sessions (§3.2), so cleaned scores are structurally higher. The
comparison is therefore **not** "we beat their baseline" — it is that the two are
consistent once the release is named, which nobody does (P4).

What it does let us say: the gap Zep reports between its baseline and its system
(+8.4 pp on mini) is **the same size as the reader effect we measured on oracle**
(+11 to +12 pp) and **smaller than the collapse `_s` induces on one type alone**.

### Plain retrieval, with zero write spend, recovers 91% of that loss

The same 500 questions, the same reader, the same `con` prompt — answered from
the memory's **top-50 turns** instead of the whole haystack. The write path is
`passthrough`: **no extraction, no summarisation, no LLM call of any kind at
ingest.** Only embeddings, $1.12 of them. **[실측]**

| | `_s` full context | **`_s` retrieval top-50** | oracle (ceiling) |
|---|---|---|---|
| task-averaged | 58.40 | **80.81** | 83.89 |
| overall | 60.40 | **81.60** | 83.60 |
| single-session-user | 84.29 | **97.14** | 97.14 |
| single-session-assistant | 92.86 | **98.21** | 98.21 |
| single-session-preference | 3.33 | **53.33** | 63.33 |
| multi-session | 50.38 | **76.69** | 75.19 |
| temporal-reasoning | 54.14 | **77.44** | 79.70 |
| knowledge-update | 65.38 | **82.05** | 89.74 |
| abstention | 73.33 | **83.33** | 83.33 |
| prompt, median | 517,430 chars | **55,108 chars** | ~27,000 |
| cost | $9.07 | **$2.62** | $1.05 |

```
retrieval-top50  -  full-context     task_avg +22.42 [+17.66, +27.23]
                                     overall  +21.20 [+16.60, +25.60]
                                     McNemar 132/26   p < 1e-16
```

**Reading everything loses 23.20 points against oracle. Retrieving fifty turns
gets 21.20 of them back — 91% — on one ninth of the context and one third of the
cost.** Three types come back to the oracle number *exactly* (SSU 97.14, SSA
98.21, abstention 83.33) and multi-session goes slightly past it (76.69 vs 75.19),
which is what "search was the whole problem" looks like when it is true.

Two things this does and does not say, because they are easy to run together.

**It does say that on this benchmark the read path is where the points are, and
that they are not bought with write spend.** This arm spent zero LLM calls on
ingest. Zep spends 27,449 write calls on LoCoMo and reports +8.4 pp over its own
`_s` baseline (§5.2); the retrieval-only arm here is +21.20 pp over ours. The
protocols differ and the releases differ, so those two numbers are not each
other's competitors — but they are the same *kind* of number, and they are an
order of magnitude apart in favour of the one that paid nothing to write.

**It does not say memory systems are pointless — it says the opposite of what the
oracle arm alone suggested.** On oracle, adding retrieval COST 3 points (a tax on
a haystack that already fits). On `_s` the same mechanism is worth +21. The sign
flips with whether retrieval is needed, which is exactly why a claim measured only
where the context fits should not be generalised. What stays constant across both
is narrower and survives: **what changes the score is what reaches the reader, and
we have not yet found a case where paying an LLM to write buys that.**

Two caveats we can state precisely rather than gesture at:

1. **Our index is not upstream's, and it is fairer to the benchmark.**
   run_retrieval.py indexes **user turns only** (LME-A1), which is why its own
   retrieval metrics silently drop 51 single-session-assistant questions (P7).
   We index every turn, and SSA comes back at 98.21 — the type the benchmark
   advertises as its differentiator, and the one upstream's own retrieval
   evaluation cannot score.
2. **k=50 turns is a deliberate choice, stated in advance**: upstream's baselines
   retrieve top-5 to top-10 *sessions* (≈50–100 turns), so this is inside their
   budget and far under Mem0's top-200. The oracle arm used k=10 because 10 turns
   there is already half the haystack; the two k values are not comparable and the
   arms are never compared to each other.

The remaining 2.0 points to the oracle ceiling sit almost entirely in
**knowledge-update (−7.69)** and **preference (−10.00)** — the two types where
finding *a* relevant session is not enough, because KU needs the *latest* of
several contradicting ones and preference needs the reader to actually use what it
found. That is a retrieval-ranking problem and a reading problem respectively,
and it is where a memory system that does more than retrieve would have to earn
its cost.

### The run also tested the harness

16 of 500 rows failed on the first pass — 12 `APIConnectionError`, 4
`APITimeoutError`, all on 113 K-token requests, none a rate limit. The retrieval
arm then failed 5 more, and those had a cause worth keeping: **five turns out of
246,750 are longer than the embedder's 8,192-token ceiling** (the longest is
76,591 characters), and the hosted API answers an oversized input with a 400 that
no transport retry can fix — taking the whole instance down with it. The embedder
now truncates its INPUT to the model's ceiling, counts each time it does, and
reports the count in the run artifact (`input_truncations: 5`); the stored content
is untouched, so a retrieved item still renders whole. The design held
in the way it was built to: each failure was written as a row rather than
swallowed (upstream's `continue`, LME-A18), the completeness check saw 484 judged
against a population of 500 and **refused to report a score**, and `--resume`
re-bought exactly those 16 rows for $0.27 rather than re-running the arm for $8.78.
It is worth noting one true fix is still missing: our LLM **client** has no
transport retry (the embedder gained one on 2026-08-17), so a 3.2% failure rate on
long requests is paid for by a second process rather than absorbed by the first.
And the resumed process under-reported its own measurement's cost, because
embeddings do not route through `LLMClient` and so are invisible in the trace the
resume prices itself from — $1.51 reported against $2.62 actually spent. The
driver now carries the earlier process's own summary total forward, which is the
only record that includes the embedder.

## Follow-ups on the same 500 questions

The four arms above are paid for, and their records carry every hypothesis. That
makes a set of second questions cheap enough to answer that not answering them
would be a choice. **[실측]**

### The judge is worth 2 points, and it does not change the answer

§6 P3 says the judge is a free variable nobody keeps pinned: the official code
asserts `gpt-4o-2024-08-06`, and the four systems publishing LongMemEval numbers
use gpt-4o-mini, gpt-4.1-mini, gpt-5 and gpt-4o-mini — Mem0 with a rewritten
prompt that tells the judge to *"lean toward yes"* when unsure. That criticism is
one we owe our own numbers, so all four arms were re-scored with `gpt-4o-mini` as
judge. Only judge calls were bought: 2,000 of them, **$0.078**, 22 minutes
(`scripts/repro/lme_rejudge.py`).

| arm | pinned gpt-4o | gpt-4o-mini judge | Δ | agreement |
|---|---|---|---|---|
| mini × direct | 79.20 | 77.60 | −1.60 | 97.6% |
| mini × con | 83.60 | 82.20 | −1.40 | 96.2% |
| luna × direct | 91.40 | 90.20 | −1.20 | 98.0% |
| luna × con | 94.60 | 92.60 | −2.00 | 98.0% |

(overall accuracy; task-averaged moves the same way, −0.25 to −1.89.)

Two things, and they point opposite ways.

**The headline is judge-dependent at the 1–2 pp scale.** The cheaper judge is
consistently *stricter*, and 2.00 pp is larger than the CoN effect we measured on
luna's task-averaged number (+2.18, whose interval already covered zero). **Any
LongMemEval claim of about two points is inside the judge's own swing** — which
is precisely why publishing a number without naming the judge, as every system in
§5.1 does, makes it uncomparable.

**The comparison is not.** The shift is systematic, so what the arms are being
contrasted on barely moves:

```
spread, pinned gpt-4o judge     task-averaged 12.57    overall 15.40
spread, gpt-4o-mini judge       task-averaged 12.63    overall 15.00
arm ranking                      IDENTICAL under both judges
```

**C4's finding is judge-invariant** — the spread it rests on is 6–8x the judge's
own effect. This is the general shape worth keeping from this benchmark: its
*levels* are not comparable across papers, and its *contrasts*, held under one
protocol, are.

The exception is the row we already flagged as softest. Abstention falls further
than any type under the swap — 70.00 → 66.67 and 86.67 → 80.00 — consistent with
the hand-read finding above that some abstention verdicts turn on presentation
rather than comprehension.

### How much of a gap is nothing at all

The pre-registration's own weakness list starts with *single seed*. So the
luna × con arm was simply run again, same everything, and the two runs paired:

```
luna·con  -  luna·con (replicate)   task_avg -0.47 [-3.39, +2.34]   overall +0.40 [-1.40, +2.20]
                                    McNemar 11/9  p=0.82   (20 of 500 verdicts differ)
```

**Run-to-run noise is about half a point, with an interval of ±2–3 pp.** This arm
is the honest place to measure it: gpt-5.6-luna admits no temperature, so it
answers at the provider's default rather than at 0, and its replicate variance is
real sampling rather than an API artefact. Even so, 480 of 500 verdicts are
identical.

Put beside the judge swap, there are now two independent ~2 pp noise sources
measured on this benchmark, and the same conclusion falls out of both: **a
two-point LongMemEval claim is not a finding.** C4's 12.6–15.4 pp is 6–30x either
one, which is why it survives both.

### Upstream's other history format, and its interaction with reading

§5.5 reports that JSON does not consistently beat NL *without* chain-of-note and
always beats it *with* — an interaction of up to 10 pp between two flags usually
reported as neither. Both formats are transcribed from upstream and both render
byte-identically to it (`prompt_rediff.py` configs D and E, 500/500 each), so a
gap here is the format's, not our rendering's. Measured on mini, all four cells:

| | json | nl | json − nl (overall) |
|---|---|---|---|
| **con** | 83.60 | 81.20 | **+2.40** [−0.40, +5.40], McNemar 34/22, p=0.14 |
| **direct** | 79.20 | 79.00 | **+0.20** [−2.20, +2.60], McNemar 21/20, p=1.00 |

**The direction reproduces the paper's interaction exactly** — JSON's advantage
exists with chain-of-note (+2.40) and is indistinguishable from zero without it
(+0.20) — but neither cell separates on its own at 500 questions, and we did not
buy an interval for the difference-of-differences. So: consistent with §5.5,
not a confirmation of it. What it does establish for our own arms is that the
format flag is worth about as much as the judge — another reason a bare number
from another paper cannot be compared.

### What a retrieval layer costs when the haystack already fits

The same arm answered from the memory's top-10 instead of the whole haystack
(`--retrieval 10`, real embedder, everything else identical):

| mini × con | task-averaged | overall | abstention | cost |
|---|---|---|---|---|
| full context | 83.89 | 83.60 | 83.33 | $1.05 |
| retrieval top-10 | 81.97 | 80.60 | **66.67** | $0.85 |
| difference | +1.92 [−1.88, +5.77] | **+3.00** [−0.40, +6.60] | −16.67 | |

**Retrieval costs about 3 points here, and that is the expected sign, not a
surprise**: oracle is evidence-only, the whole haystack is 6.1 K tokens at the
median, and it already fits. A retrieval layer in that condition can only drop
something the reader would otherwise have had. The interval covers zero, so the
tax is not separated at 500 questions — but the direction is consistent across
both accuracies and the discordance is asymmetric (48 questions the full-context
arm gets and retrieval misses, against 33 the other way).

The type it costs most is **abstention, −16.67 pp**. That fits the mechanism:
answering "this was never mentioned" requires having seen what *was* mentioned,
and top-10 turns of an already-short haystack is exactly the condition under
which a false premise stops being visibly false.

**This is a ceiling-condition measurement and it does not generalise downward.**
It says what a memory layer costs when retrieval is not needed. It says nothing
about what one buys when it is — that question lives on `_s`, where the paper's
own numbers show GPT-4o falling from .870 on oracle to .606 on the full haystack.

### Reproduce

```bash
# one arm (~$1 on oracle). --dry-run first prints the call ledger and a priced
# quote for $0; --max-spend-usd is enforced between rows.
uv run python scripts/repro/exp_lme_reading.py --dataset oracle --reading con \
    --reader gpt-4o-mini --workers 8 --max-spend-usd 2.0

# the axes the follow-ups turned
  ... --history-format nl        # upstream's other format (byte-checked, config E)
  ... --retrieval 10             # answer from the store's top-K, not the haystack
  ... --dataset s                # 113K-token haystacks; trace is gzipped by default

# re-score answered arms with another judge — judge calls only, ~2 cents an arm
uv run python scripts/repro/lme_rejudge.py --arms <tag> ... \
    --judge-model gpt-4o-mini --max-spend-usd 0.5

# the pre-registered rule, and every pairwise CI. $0.
uv run --with scipy python scripts/repro/lme_c4_analysis.py --arms \
    gpt-4o-mini_lme_oracle_direct gpt-4o-mini_lme_oracle_con \
    gpt-5.6-luna_lme_oracle_direct gpt-5.6-luna_lme_oracle_con
```

Artifacts per arm: `results/repro/<tag>.json` (committed), `<tag>.records.jsonl` (per-row verdict,
prompt sha256, usage, timing, and the retrieved ids on a retrieval arm) and `<tag>.llm-trace.jsonl`
(every prompt and every judge reply verbatim — ~15 MB on oracle; on `_s` it is written `.jsonl.gz`,
~78 MB compressed against ~260 MB raw) — both gitignored, both kept locally, which is what made the
judge swap cost two cents instead of another four arms. Paired statistics:
`lme_c4_paired.json`, `lme_replicate_paired.json`, `lme_format_retrieval_paired.json`,
`lme_rejudge_gpt-4o-mini.json`.

---

## Pre-registration II — `_m` (written 2026-08-18, with 12 of 500 rows on disk)

**Disclosure of timing, first.** This plan was written after the arm had started, with 12 rows
answered. All twelve are `single-session-assistant`, the type that saturates at 97–100 in every arm
we have ever run (§9.3) — the release ships in type order, so the first block carries no information
about the contrast below. Nothing else had been looked at. The honest statement is that this is a
pre-registration against the *result*, not against the first byte of data, and that distinction is
recorded here rather than smoothed over.

**The question.** `_s` measured that retrieval top-50 beats reading the whole haystack by +21.20 pp
(C6). `_m` carries the **same 500 questions** against a haystack 9.9x longer (4,894 turns per
instance, 476 sessions, 1.11M tokens — §3.1). There is no full-context arm to compare against here:
the median instance is 8.8x a 128k window. So the contrast is

> **`gpt-4o-mini_lme_s_con_k50` vs `gpt-4o-mini_lme_m_con_k50`** — same reader, same CoN prompt, same
> pinned judge, same `k=50`, same `budget_tokens=20000`, same passthrough memory, same 500 question
> ids. **The only thing that differs is how many distractors the retriever had to reject**: top-50
> out of ~494 turns, against top-50 out of ~4,894.

**Primary measure.** The paired difference `_s` − `_m`, reported for **both** official accuracies
separately (LME-A13), with the paired bootstrap 95% CI and McNemar that
`scripts/repro/lme_c4_analysis.py` already computes — the arms align by `question_id`, so that
script executes this rule unmodified.

| paired Δ (`_s` − `_m`) | verdict |
|---|---|
| **< 2 pp** (CI includes 0) | **Retrieval is invariant to haystack length over a 10x range.** This is the strongest available form of C6: the read path buys its points at O(1) in corpus size, and the ~$500–700 write arms are being asked to beat a baseline that does not decay. |
| 2–5 pp | Mild decay. C6 survives with a stated length dependence. |
| **≥ 5 pp** | **Real decay, and C6 acquires a boundary.** The +21.20 pp is then partly a property of a haystack that fits, not of retrieval as such, and every sentence quoting it must carry the length it was measured at. |

A Δ under 2 pp is not "no difference" but "not separable from this benchmark's noise floor", which
§9.4.1 measured on this exact harness: same-arm rerun +0.40 pp [−1.40, +2.20], judge swap −1.2 to
−2.0 pp. Any verdict at 2 pp or below is reported in those words.

**Secondary, pre-registered as secondary.**

1. **Per-type decomposition**, 7 rows (six types + the abstention cross-cut, LME-A14). Movement in
   `single-session-user`/`-assistant` is **not** read as signal: both saturate at 97–100 in every arm
   to date, and a saturated type can only move down. The types that carried C4's spread were
   multi-session, temporal-reasoning and preference, and those are where a dilution effect should
   land if there is one.
2. **`single-session-preference`.** It went 63.33 → 3.33 when `_s` was read in full, and returned to
   oracle levels under retrieval (§9.3a). If it collapses again on `_m`, the collapse is about
   context length reaching the reader, not about the haystack — a distinction worth one line.

**What this arm cannot answer, stated in advance.**

- **Retrieval recall is not computable from this artifact.** Rows record retrieved *episode* ids and
  scores, not the session ids they came from, so "did the evidence session make top-50" cannot be
  recovered without re-running. A drop in accuracy therefore cannot be attributed to retrieval
  failure versus reader failure from these files alone. Fixing that is a capture change, not an
  analysis one.
- **This is one arm, one seed, one reader.** §9.4.1 applies unchanged.
- **It measures retrieval, not memory systems** (§9.4.5), and on `_m` that gap is wider, not
  narrower: an organizer arm here is $500–700 against this arm's $12.47, so the regime that most
  needs a write path is the one where we can least afford to measure it.

**What would falsify the setup rather than the claim.** If `_m` lands near `_s`'s *full-context*
score (60.40) rather than near its retrieval score (81.60), check the retriever before believing the
result: at k=50 out of 4,894 turns the arm indexes 1% of the haystack, and a silent indexing failure
would look exactly like graceful degradation. The stamp and the 3 recorded embedding truncations
(inputs over 24,576 chars) are the first place to look.

**Reproduce.**

```bash
uv run python scripts/repro/exp_lme_reading.py --dataset m --reading con --retrieval 50 \
    --budget-tokens 20000 --workers 12 --tag gpt-4o-mini_lme_m_con_k50 --max-spend-usd 14
# workers is capped near 15 by kuzu's 8 TiB virtual-address reservation per in-memory
# database, not by RAM (the arm never writes to the graph store) — see 68d4ec8.

uv run --with scipy python scripts/repro/lme_c4_analysis.py \
    --arms gpt-4o-mini_lme_s_con_k50 gpt-4o-mini_lme_m_con_k50 \
    --out results/repro/lme_m_paired.json
```
