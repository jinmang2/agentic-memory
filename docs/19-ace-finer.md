# ACE on FiNER — what a self-evolving playbook bought, measured against not having one

ACE's headline for online adaptation is stated on FiNER: **69.1% → 81.9%**, plus "−91.5% latency
and −83.6% token cost vs Dynamic Cheatsheet" (release README). FiNER is also the one ACE experiment
whose data actually ships — the AppWorld code that carries the paper's flagship result is absent from
the public repository ([ledger B-6](17-defect-ledger.md)). So it is the only ACE claim this campaign
can put a number against, and this document is that number.

**What this is and is not.** It is not a reproduction of 69.1 → 81.9. That figure was produced by a
much stronger generator than the one priced here, through a training loop that spends two to three
times the calls ours does. It is a controlled measurement of one question: *on the same 441 questions,
does growing an ACE playbook alongside beat not growing one?* — asked four ways, because the answer
came back null and a null is only worth reading once the knobs that could have caused it have been
turned: under our dedup and under upstream's default, with one reflection per sample and with
upstream's reflect-and-re-answer rounds.

**A fifth arm answers a question the other four cannot.** Four settings of one methodology cannot
say whether a null belongs to ACE or to the task, so ReasoningBank — a different self-evolving
memory, with a different read shape — was run over the same 441 questions. It lands on the same
accuracy as learning nothing, to the second decimal place. That arm is **not** a reproduction of
ReasoningBank's own claims, which are agentic and unreachable here; see its section below.

## Protocol

| | |
|---|---|
| Benchmark | FiNER, shipped test split `finer_test_subset_006_seed42.jsonl` — 441 samples × exactly 4 US-GAAP tags = **1,764 tagging decisions** |
| Generator | `gpt-4o-mini`, temperature 0 |
| Reflector / Curator | `gpt-4o-mini`, temperature 0, `max_tokens` 4096 (one role — see condition 4) |
| Embedder | `text-embedding-3-small` (curator dedup + task episodes) |
| Adaptation | upstream's **online** mode: 30 windows of 15, each answered with the playbook as it stands and only then trained on (`ace.py:939-997`) |
| Curator dedup | **0.90 cosine** (ours, deviation D5), and **off** in the `nodedup` arm (upstream's shipped default) |
| Adaptation rounds | **1** reflection per sample, and **up to 3** reflect-and-re-answer rounds in the `retry` arm (upstream's `max_num_rounds`, condition 2) |
| Seeds | **one** |

## Results

| arm | tag accuracy | sample accuracy | LLM calls | cost | playbook |
|---|---|---|---|---|---|
| **base** — empty playbook, no learning | **48.24** | 16.10 | 441 | $0.192 | — |
| **online** — grown, our dedup at 0.90 | 46.71 | 15.42 | 1,323 | $1.461 | 140 bullets / 43.7 K chars |
| **nodedup** — grown, dedup off (upstream's default) | **48.98** | **17.01** | 1,325 | **$8.633** | 2,165 bullets / 639 K chars |
| **retry** — grown, dedup 0.90, upstream's reflect-and-re-answer rounds | 45.80 | 12.70 | 2,918 | $4.394 | 217 bullets / 63.6 K chars |
| **rb** — ReasoningBank, top-1 read (a different methodology, not an ACE setting) | 48.24 | 16.33 | 882 | $0.367 | 441 experiences + 1,323 strategies / **read stays under 1 K chars** |

Paired over the same questions in the same order (bootstrap imported from `scripts/ext/x1_power.py`,
10,000 resamples, seed 0):

```
sample_accuracy   online  - base     Δ = -0.68 pp   95% CI [-3.40, +2.04]   p = 0.687   NOT separated
sample_accuracy   nodedup - base     Δ = +0.91 pp   95% CI [-2.04, +3.86]   p = 0.607   NOT separated
sample_accuracy   nodedup - online   Δ = +1.59 pp   95% CI [-1.59, +4.76]   p = 0.337   NOT separated
sample_accuracy   retry   - base     Δ = -3.40 pp   95% CI [-6.12, -0.68]   p = 0.021   separated*
sample_accuracy   retry   - online   Δ = -2.72 pp   95% CI [-5.67, +0.00]   p = 0.067   NOT separated
tag_accuracy      online  - base     Δ = -1.53 pp   (point estimates — see "one interval is missing")
tag_accuracy      nodedup - base     Δ = +0.74 pp
tag_accuracy      retry   - base     Δ = -2.44 pp
sample_accuracy   rb      - base     Δ = +0.23 pp   95% CI [-2.04, +2.72]   p = 0.934   NOT separated
tag_accuracy      rb      - base     Δ = +0.00 pp   (48.24 -> 48.24)
per-window sign   online ahead in 10 of 30; nodedup 14; retry 12; rb 12
```

\* **and that asterisk is load-bearing.** Five arms means five comparisons against
the same base, and a 0.05 threshold applied five times is not a 0.05 threshold: a
Bonferroni correction puts the bar at 0.01, which p = 0.021 does not clear.
(It did not clear the 0.0125 the four-arm version of this table required either;
adding the `rb` arm tightened the bar and changed no conclusion.) The
comparison that isolates the retry loop with everything else held fixed — retry
against online, same dedup, same prompts, same model — has an interval that
touches zero. **So the defensible claim is that upstream's extra calls bought no
measurable improvement, with every point estimate on the negative side. It is not
that retries are proven harmful.**

**Growing a playbook over 441 samples produced no measurable improvement — under our dedup at 7.6×
the cost, under upstream's default at 45×, and with upstream's retry rounds at 23×.** Four of the
five intervals cover zero; the fifth (retry against base) sits below it but does not survive a
correction for having asked five times. So the claim this supports is *no measurable effect*, **not**
that the playbook helps, and not that it demonstrably hurts.

The third arm exists because the second one could not answer for itself. A null under our
always-on dedup has two readings — the adaptation does not transfer, or our gate threw the
adaptation away — and only running upstream's shipped default separates them. It does: with the
gate off, 2,165 bullets accumulate instead of 140, and the result is still not separable from not
learning at all. **Our dedup is not the reason for the null.**

**The price of the third arm is the finding beside it.** Turning the gate off multiplied cost by 5.9
against the online arm while the LLM call count moved by 2 calls — 1,323 to 1,325. Nothing about the
*number* of calls changed; the playbook is injected whole into every generator and curator call, so
the cost of adaptation is carried in tokens, not in requests. By the end of the run each generator
call was carrying **~117 K tokens of playbook** — the arm's final generator call billed **118,619
input tokens, 93% of `gpt-4o-mini`'s 128 K context window**, of which **98.6% was playbook** (639,054
of 648,064 characters) — around a task whose own prompt averages 1,951 tokens in the base arm. Any
cost model for this mechanism that counts calls will be wrong by more than an order of magnitude.

> **Correction, 2026-08-17.** This paragraph previously read "~95 K tokens of playbook — 74% of the
> context window", an estimate that was never taken off the trace. The trace says 93%: generator
> calls in the arm's last window billed 110,646–118,692 input tokens. The claim moved in the
> direction that strengthens it, which is exactly why it needed checking — an understated number
> attracts no scrutiny. Read out of `tokens_in` on the `generate` rows of
> `results/repro/gpt-4o-mini_ace_finer_nodedup.llm-trace.jsonl` by
> `scripts/repro/demo_cost_is_tokens.py`, which regenerates
> [docs/demos/cost-is-tokens.md](demos/cost-is-tokens.md) from the artifacts.

The playbook was not empty and not junk. In the online arm 441 curator calls proposed 416 bullets, of
which **140 survived** 0.90 dedup — **43,699 characters as rendered into the prompt** (35,920 of that
bullet text, the rest the `[id] helpful=N harmful=M ::` prefixes), injected in full into every
generator call thereafter.
Its content was specific: *"Differentiate between `AmortizationOfFinancingCosts` and
`AmortizationOfIntangibleAssets`"*, *"Clarify the distinction between `DeferredFinanceCostsGross` and
`DeferredFinanceCostsNet`"*. It simply did not convert into answers.

The deficit narrows as the playbook accumulates (mean Δ per window **−2.44 pp** over the first
fifteen, **−0.72 pp** over the second fifteen). That is consistent with no effect, and equally
consistent with regression to the mean; with 15 windows a side it is a record, not a result, and
nothing here rests on it.

## What there was to learn, and what got learned

A null result on an adaptive method has two very different readings, and the headline cannot
separate them: either the task holds nothing transferable, or it does and the method missed it.
Measured from the committed records (`scripts/repro/finer_error_structure.py`, no model calls), it is
the second.

**The task is repetitive.** 1,764 gold slots draw on **137 distinct tags**, of which only **3** appear
once; the top 25 tags cover half of all slots. The base arm's 913 wrong slots concentrate the same
way: the **top 50 confusions account for 51.5% of all errors**, and the single most common one recurs
**34 times**.

| recurs | gold tag | model answered instead |
|---|---|---|
| 34× | `DebtInstrumentBasisSpreadOnVariableRate1` | `DebtInstrumentInterestRateEffectivePercentage` |
| 33× | `LineOfCreditFacilityMaximumBorrowingCapacity` | `LineOfCreditFacilityCurrentBorrowingCapacity` |
| 28× | `DebtInstrumentInterestRateStatedPercentage` | `DebtInstrumentInterestRateEffectivePercentage` |
| 20× | `ConcentrationRiskPercentage1` | `Revenues` |

This is *exactly* the knowledge a playbook is shaped to hold — "when the sentence describes a spread
over a variable rate, the tag is the basis-spread one, not the effective-rate one" is one bullet.

**The playbook did not cover it.** Of the top 50 confusions — half of all errors — only **7 have both
of their tags named anywhere** in the 140 surviving bullets. Of the 115 gold tags the model ever got
wrong, **24** appear in the playbook at all. And that test is deliberately generous: naming both tags
is not the same as containing a correct rule distinguishing them, so 7/50 is an *upper bound* on
coverage.

**Where it did cover, errors moved the right way.** Both covered confusions in the table above fell
(33→29 and 18→13), while the largest uncovered one rose (28→43). Single seed, small counts, and
uncovered pairs moved in both directions — this is a record, not a result. But it does mean the
mechanism is not inert; the failure is upstream of it.

**Two structural reasons the head went uncovered**, one upstream's and one ours:

1. **Nothing in the loop can see that a confusion recurred.** The reflector diagnoses one sample at a
   time, so a mistake made 34 times and a mistake made once look identical to it. The curator is then
   instructed to add only what is *missing* and to avoid redundancy — which actively suppresses
   re-stating a rule for a confusion that keeps coming back. There is no frequency signal anywhere in
   the pipeline. This is upstream's design, reproduced faithfully.
2. **Our dedup discarded 276 of the 416 bullets the curator proposed** as near-duplicates at cosine
   0.90. The curator *was* re-proposing overlapping knowledge and our gate dropped it. **This is our
   deviation, not upstream's** — upstream ships its analyzer off by default (ledger B-6). Whether
   the repetition our gate treated as redundancy was in fact reinforcement was the second open
   confound here, and the `nodedup` arm has now closed it: see below.

The playbook's own late-run statistics point the same way: 139 bullets, **96 of them never cited by
the generator**, 18 marked high-performing, 0 problematic.

### What the gate was hiding: the curator stops naming things

With the gate off the playbook reaches **2,165 bullets and 639 K rendered characters**, fifteen times
the online arm's. Coverage of the recurring confusions does not rise with it — **it falls to zero**.
Of the top 50 confusions, the 140-bullet playbook names both tags of 7; the 2,165-bullet playbook
names both tags of **none**.

The reason is in what the bullets say. Measured two ways over the bullet text
(`scripts/repro/finer_error_structure.py`, no model calls) — a gold tag from the split's own
vocabulary appearing verbatim, and the weaker test of any CamelCase identifier at all:

| | online | nodedup |
|---|---|---|
| bullets carrying a CamelCase identifier | **17.1%** | **4.0%** |
| distinct gold tags named, of the split's 137 | 30 | 12 |
| of the 115 gold tags the model ever got wrong | 24 | 9 |

Two mechanisms could produce that, and they are separable because the trace holds what the curator
*proposed* while the store holds what *survived*:

- **The curator drifted.** Specificity of proposed bullets, in fifths of each run:
  online `4.8 / 16.9 / 7.2 / 12.0 / 4.8`, nodedup `2.3 / 15.9 / 1.8 / 0.0 / 0.0`. **In the last two
  fifths of the nodedup run — roughly 866 bullets — not one names a US-GAAP element.** As the
  playbook in its context grows, the curator moves from *"Differentiate between
  `AmortizationOfFinancingCosts` and `AmortizationOfIntangibleAssets`"* to *"Consider the potential
  impact of external economic factors on the financial entities being reported"*. The prompt asks it
  to add what is missing and avoid redundancy; against 2,000 existing bullets, every specific rule
  looks already covered, and only abstraction is left to write. That reading is a hypothesis; the
  curve itself is a measurement.
- **And the gate selected.** Online's proposals carry an identifier 9.4% of the time and its
  survivors 17.1% — **our dedup roughly doubled the specificity density** of what it kept, because
  generic advice is what clusters in embedding space. It bought specificity it could not convert
  into accuracy.

So both readings were true, and neither rescues the mechanism: the gate did throw away bullets, and
running without it produced *more text carrying less knowledge*.

**Where the head confusions went is the counterweight.** The nodedup arm's errors on the top
confusions fell sharply — 34→12, 33→23, 28→18, 20→5, 15→2 — but its total wrong slots barely moved:
**913 → 900**, while those top eight pairs alone fell **183 → 95**. Roughly 75 errors left the head
and reappeared in the tail. A method can reshape the error distribution and buy nothing on the
metric, and this is what that looks like from the inside.

### The retry arm writes the knowledge — and it still does not convert

Everything above diagnosed the null as a *capture* failure: the task was learnable, the errors
were repetitive, and the playbook did not cover them. The `retry` arm falsifies that diagnosis. Give
the reflector a REGENERATED attempt to diagnose — upstream's loop, `max_num_rounds`=3, early stop —
and the curator starts naming things:

| | online | nodedup | **retry** |
|---|---|---|---|
| bullets kept | 140 | 2,165 | 217 |
| top-50 confusions with both tags named | 7 | 0 | **30** |
| of the 115 gold tags ever missed, named | 24 | 9 | **73** |
| proposed bullets carrying an identifier | 9.4% | 4.0% | **52.8%** |
| …by fifth of the run | 4.8 / 16.9 / 7.2 / 12.0 / 4.8 | 2.3 / 15.9 / 1.8 / 0.0 / 0.0 | **11.8 / 47.3 / 67.3 / 68.2 / 69.1** |

The specificity does not merely start higher, it **climbs** — from 11.8% in the first fifth to 69.1%
in the last, the opposite of the drift the other two arms show. The mechanism is visible in the
trace: 385 of 441 samples were re-answered, **129 of them corrected inside the training step**, and a
reflection written about an attempt that a specific correction fixed is a reflection with a specific
thing to say. Upstream's extra calls buy exactly the thing this document said was missing.

**And the accuracy went down.** Not "did not improve" — down, on every metric and against every
reference, with the one separated comparison sitting on the negative side.

That is a harder result than the null it replaces, because it removes the comfortable explanation.
The playbook was not vague, and it was not empty; it covered 30 of the 50 confusions that produce
half the errors, in the arm's own words, and 441 questions later the arm answered fewer of them
correctly. Two readings remain, and this run does not separate them:

- **Injection cost.** 217 specific tag-pair rules ride in every generator call. A rule that
  disambiguates `DebtInstrumentBasisSpreadOnVariableRate1` is noise on a question about leases, and
  there are now 217 of them competing for the model's attention on every question. The per-window
  record is consistent with this: retry runs +1.00 pp against online over the first half and
  **−3.28 pp over the second**, as the rule set grows.
- **Selection.** A bullet written about a sample that a retry corrected is knowledge derived from
  an attempt the generator only produced *after* being told what it got wrong. It may not describe
  anything reachable on a first attempt.

Distinguishing them needs an arm that grows the retry playbook and then serves it at a fixed size,
or one that answers with the retry playbook but never learns — neither of which is in this budget.
**What the campaign can say is stated and no more: the knowledge-capture explanation for ACE's null
on FiNER is now excluded, at this operating point.**

## Why the base arm is half the finding

Upstream reads adaptation off its online window curve — window *w* is scored, then trained on, and
the curve across windows is the evidence. Every window is a **different 15 questions**, so a
window-to-window change contains sample difficulty as well as learning.

Our base arm never learns anything, and its own window curve looks like this:

```
58.3  35.0  50.0  46.7  46.7  40.0  61.7  51.7  40.0  41.7
48.3  63.3  41.7  60.0  43.3  41.7  51.7  45.0  53.3  56.7
43.3  55.0  65.0  38.3  43.3  45.0  36.7  40.0  50.0  62.5
```

**A 30-point spread — 35.0 to 65.0 — with no mechanism in it at all.** Any reading of adaptation
from a window curve of this shape is reading difficulty, and a 12.8-point claim taken across such a
curve is inside its noise. That is a statement about the *measurement protocol*, and it applies to
ACE's own 69.1 → 81.9 as much as to us. The paired comparisons above do not have the problem: all
three arms answer the same 441 questions in the same order, so the difference vector cancels
difficulty exactly. The nodedup arm makes the point twice over: across the last five windows it runs
45.0 / 28.33 / 43.33 / 48.33 / 54.17 while base, on those same questions, runs
45.0 / 36.67 / 40.0 / 50.0 / 62.5 — a swing of about 26 points in each, one arm carrying a 639 K-char
playbook and the other carrying nothing.

This is also the reason the early windows are not evidence of anything. The three pre-flight smokes
all showed accuracy falling from window 0 to window 1 (61.7→35.0, 61.7→36.7, 55.0→41.7), which reads
as "the playbook immediately hurts" — until the base arm scores **35.0** on the same window 1 with no
playbook in existence. It was the questions.

## The null is not ACE's — a second methodology lands on the same number

Everything above is one methodology. A null under four settings of ACE says the playbook did not
convert *here*, and leaves open whether that belongs to ACE or to FiNER: a task where 441 questions
draw on the same handful of GAAP distinctions might simply not reward distilled advice, whoever
distils it.

Separating those needs a second self-evolving memory on the identical questions.
**ReasoningBank**, run over the same 441 samples in the same order at the same model and embedder,
differing in what was grown alongside and — the sharper difference — in how much of it is read
back:

```
rb - base  [n=441]
  sample_accuracy  d= +0.23pp  95% CI [-2.04, +2.72]  p=0.9338  disagree= 29/441  NOT separated
  tag_accuracy     d= +0.00pp  (48.24 -> 48.24, point estimate)                   moved=117/441
  per-window sign: ahead in 12, behind in 14, tied in 4 of 30
```

**48.24 → 48.24.** The tag accuracy does not move at the second decimal place, on a metric with
1,764 tagging decisions under it. And the arm is the *least* disruptive of the four: 117 questions
changed answer against base, where ACE's arms moved 181, 195 and 191.

**So the null generalises past ACE.** Two mechanisms, built on different premises — grow one
document and inject all of it, versus distil discrete items and retrieve the single best — arrive at
the same accuracy as growing nothing. That is a stronger statement than either arm alone, and it
moves the open question from "is ACE's playbook worth its cost" to "does this task reward
self-evolving memory at all".

**What this arm is NOT.** ReasoningBank's published claims are *agentic* — WebArena and SWE-Bench —
and neither is reachable here: WebArena needs a five-site Docker environment, SWE-Bench needs 500
containerised agent rollouts, a hosted grader, and a model far stronger than `gpt-4o-mini`, which
sits near floor on it and would leave no room above the baseline to measure. Mind2Web, which the
paper also reports, is absent from the release entirely. **This is not a reproduction of
ReasoningBank and must not be cited as one**; the artifact's own stamp carries
`RB_D1_not_the_published_benchmark_finer_not_webarena_or_swebench` so a reader of the JSON cannot
miss it. What is measured is the mechanism on a single-turn task, against a control that was already
bought.

Two conditions travel with the number. The **generator prompt is ACE's, unchanged** — RB's items
land in the same slot the playbook does, under the same instructions — because the question only
reads if the generator is held fixed and memory content is the single variable; a prompt tuned for
RB would answer a different question and could not be compared with `base`. And the **retrieval
geometry is not upstream's**: we pin RB's memory types and k (`RB_READ_RECIPE`, `experiences_topk`
= 1) but not its embedding-side instruction prefix, disclosed at `configs.rb_upstream`.

**The cost side is the mirror of the nodedup finding.** RB spends **882 calls for $0.367** where the
online arm spent 1,323 for $1.461 and nodedup 1,325 for $8.633 — one distillation per sample instead
of two, and, decisively, a read that serves **one item**. Its store grew to 441 experiences plus
1,323 strategies and the injected text stayed between 774 and 965 characters from window 1 to window
29, where ACE's reached 639,054. Same task, same null, **23× less money**, because the read is
bounded. If a later result does show a self-evolving memory paying off on FiNER, this arm is the
evidence that the payment does not have to scale with the store.

One upstream observation the run surfaced: RB writes both `experiences` (one per task) and
`strategies` (three per task), while its own pinned recipe serves `strategies_topk` = **0**. Three
quarters of what its extractor produces is never read back at the operating point the release ships.

One process note, because it nearly became a false claim in an artifact. `configs.rb_upstream`
carries `run_ready=False`, and having run this arm looked like grounds to flip it. It is not: that
flag gates the **LoCoMo** runner's ingest, and this arm never touches that registry — it selects its
organizer through the FiNER runner's own switch. A methodology having been run *somewhere* is not
that entry having been piloted. A pinning test refused the flip, which is the flag doing exactly the
job it exists for.

## Is 48.24 a real number?

It sits 21 points under the paper's 69.1 baseline, which is exactly the situation where a harness
defect can masquerade as a result. Three checks, all `$0`:

- **The task is winnable as served.** Every one of the 1,764 gold tags appears in the candidate list
  inside its own prompt (0 missing). The ceiling on `tag_accuracy` is 100, not something lower that we
  built.
- **No degenerate mode.** Per-sample correct counts spread across the whole range (0:86, 1:92, 2:101,
  3:91, 4:71). Zero unanswerable samples, zero no-answers, zero over-predictions in either arm.
- **The errors are the task's errors.** Wrong answers are plausible alternatives —
  `SharePrice` for `ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1`,
  `StockIssuedDuringPeriodSharesNewIssues` for `SaleOfStockNumberOfSharesIssuedInTransaction` — which
  is what a weaker model does on a large-vocabulary label-selection task.

So 48.24 is a capability number for `gpt-4o-mini` on FiNER, and the gap to 69.1 is the generator, not
the harness.

## One interval is missing, deliberately

`tag_accuracy` is the metric ACE publishes, and it is reported here as a point estimate with no
confidence interval. The reason is a property of the metric, not an oversight: it is a per-question
**rate** over four tags that share one filing excerpt, so the tags are clustered and the correct
resampling unit is the question, with a float statistic. The campaign's bootstrap
(`x1_power.paired_delta_ci`) takes booleans, which fits `sample_accuracy` exactly and cannot run the
clustered rate. Writing a float twin of it here would put **two confidence-interval implementations
in one repository**, which is how two numbers come to disagree about what "95%" means; bootstrapping
over tags instead of questions would treat clustered observations as independent and report an
interval narrower than the truth. Neither is worth an interval, so `finer_paired.json` records
`tag_accuracy_interval: null` and says why.

## Conditions that travel with these numbers

1. **The generator is not the paper's.** `gpt-4o-mini` against DeepSeek-V3.1. Our base is 20.9 points
   under the paper's baseline, so this is a different operating point, and a mechanism that needs a
   strong reasoner to exploit its own playbook would not show up here.
2. **Our adaptation loop is 3 calls per sample; upstream's online mode is 5 to 10.** Earlier
   revisions of this document said "3 to 8", which counted only the retry block; reading the whole
   step at the pinned SHA gives:

   | upstream, per sample | calls | ours |
   |---|---|---|
   | window test pass — **the answer that is scored** (`ace.py:955`) | 1 | 1 (and ours is also the trajectory reflected on) |
   | `_train_step` answers the same sample **again** with the same playbook (`ace.py:465-472`) | 1 | 0 |
   | reflect; on an incorrect sample also regenerate, up to `max_num_rounds`=3 rounds with early stop when a retry lands (`ace.py:498-543`), on a correct one reflect once (`ace.py:548-570`) | 1–6 | 1 |
   | curate — `curator_frequency`=1 on FiNER (`eval/finance/run.py:49`) | 1 | 1 |
   | post-curator generate, filed as `post_train_result` and never fed back (`ace.py:608-625`) | 1 | 0 |

   Two of those extra calls buy no learning: one re-answers a question already answered under the
   same playbook, and one exists to measure post-training accuracy. **The retry block is the one that
   can change what gets learned** — it can flip a sample to correct *inside* the training step, so
   the reflector sees outcomes ours never produces, bullet counters accrue per round, and the curator
   receives the reflection written about the *final* attempt rather than the first. **This is the
   largest open question in the table**, and an `online_retry` arm is what would close it. (→ It was: the arm ran 2026-08-17 and closed the question — see the "Measured" paragraph two below, and "The retry arm writes the knowledge".)

   Reading the online loop also settles a question worth stating: **the reported accuracy is not
   contaminated by training on the sample being scored.** The window is tested in full before any
   training happens on it (`ace.py:950-996`), which is the protocol our arms follow.

   **Measured 2026-08-17 (the `retry` arm).** The rounds are reproduced and the question is
   answered: they buy no measurable accuracy (retry − online Δ −2.72 pp, CI [−5.67, +0.00]) at
   2.2x the calls, and they transform what the playbook contains — see "the retry arm writes the
   knowledge". The two calls upstream spends re-answering a question it has already answered and
   re-answering it once more after curating remain unreproduced, because neither feeds learning.
3. **One attempt is scored and learned from.** Upstream answers a window for scoring and then calls
   the generator *again* inside its training step; we reflect on the attempt already scored.
4. **Reflector and curator share one role**, so they cannot take different models or temperatures;
   upstream allows distinct ones (`ace.py:36-63`).
5. **Dedup is always on at 0.90 in the online arm and drops the incoming bullet.** Upstream's
   analyzer is opt-in, off in `ace.py`, off in this harness's flag, and off again by silent fallback
   when its optional dependencies are missing, and when on it LLM-merges groups rather than dropping
   (ledger B-6). Ours is a third behavior, not upstream's switched on. **This was the leading
   candidate explanation for the null and is now measured, not assumed**: the `nodedup` arm runs at
   upstream's shipped default over the same 441 samples and is still not separable from not learning
   (Δ +0.91 pp, CI [−2.04, +3.86]). What the gate changed was cost and specificity, not the verdict.
6. **Single seed**, and no replication of the sign.
7. **Coverage is measured by what the playbook says, not by what it means.** A confusion counts as
   covered when both of its tags appear in the bullet text — an upper bound, since no check was made
   that a correct distinguishing rule accompanies them.

## Two defects in the harness these numbers come from

Neither is ours, and both change what a FiNER accuracy can be compared against.

**"Accuracy" names two quantities, printed as one.** `evaluate_test_set` builds a dict holding
`accuracy` — the tag-level micro rate — beside `correct`/`total`, which are sample-level exact match,
and prints `Accuracy: {tag_rate} ({sample_correct}/{total})` (`utils.py:286`). The parenthetical does
not equal the number in front of it. The saved online result pairs them the same way
(`ace.py:1097-1098`), so recomputing `correct/total` from the artifact contradicts the artifact's own
headline. On our run those two numbers are **48.24 and 16.10** — a factor of three apart. Any citation
of a FiNER accuracy that does not say which one it is cannot be compared against.

**Failures leave the denominator instead of scoring zero.** A sample that raises is caught, printed
and `continue`d (`utils.py:198-199, 248-250`), so it never reaches `total`, `answers` or `targets`.
Ours counts it wrong and reports `n_failed` separately; on these runs it was zero either way.

Three more, with their reproduction, are in [`bench/finer.py`](../src/agmem/bench/finer.py): `eval()`
runs on model output inside the scorer; over-prediction is free while under-prediction is penalised;
and the training signal (all-or-nothing) is not the reported metric (per-tag).

A dataset note found while checking the above: FiNER's gold labels are **internally inconsistent in
case** — `ShareBased...` and `Sharebased...` appear in the same row for the same element. Harmless,
because the scorer lowercases both sides, but it is noise in the labels rather than in the answers.

## Artifacts

| | |
|---|---|
| base | `results/repro/gpt-4o-mini_ace_finer_base.json` |
| online | `results/repro/gpt-4o-mini_ace_finer_online.json` |
| nodedup | `results/repro/gpt-4o-mini_ace_finer_nodedup.json` |
| retry | `results/repro/gpt-4o-mini_ace_finer_retry.json` — records carry `retry_attempts` and `corrected_in_training` per sample |
| rb | `results/repro/gpt-4o-mini_rb_finer_online.json` — ReasoningBank, `--organizer rb`; its stamp carries `RB_D1_not_the_published_benchmark_...`, `read_path: experiences`, and no ACE knobs |
| rb quote | `..._rb_finer_smoke30.json` ($0.0240, 30 samples) and the $0 dry runs `DRYRUN_finer_rb_441_{none,max}.json` + `DRYRUN_finer_base_full441.json`, which together priced the arm at $0.29–0.38 before it was bought (it cost $0.367) |
| paired analysis | `results/repro/finer_paired.json` (`scripts/repro/finer_paired.py`, `--reference` selects which arm the others are paired against) |
| error structure | `results/repro/finer_error_structure.json` (`scripts/repro/finer_error_structure.py`) |
| pre-flight smokes | `..._ace_finer_smoke30{,b,c}.json` — the runs that caught the prompt defect below |
| runner | `scripts/repro/exp_ace_finer.py`; adapter `src/agmem/bench/finer.py` |

Each carries per-window summaries, the full LLM I/O trace, the memory snapshot, the evolution log and
the per-question records. The base and online arms are stamped at commit `6783c24` and the nodedup
arm at `7d29f2b`; all 882 reflector/curator calls of the online arm are verifiable in its trace as
having used the post-fix prompts.

**Three things about the nodedup artifact that a reader would otherwise get wrong.** It was bought
across four processes — the host rebooted twice and a spend cap stopped it once — and that shows up
in the files:

- **Its summary's `llm_budget` counts only the last process** (164 calls), because a resumed run's
  budget object starts at zero, while `cost_usd` in the same file **is** the measurement total
  ($8.633, prior spend recovered from the trace). Two fields of one artifact, two scopes. Call counts
  in the table above are therefore taken from the traces, which append across every process: 441 /
  1,323 / 1,325 generator-plus-curator calls. Embedding calls are not in the trace and not in that
  count. The same per-process/cumulative split applies to `seq` in the evolution log.
- **The store holds 440 episodes for 441 scored samples.** The records file is complete and its
  headline recomputes from it exactly (the paired script refuses to run otherwise), so the
  measurement is unaffected; the likeliest cause is one buffered store write lost to a host reboot.
  Disclosed rather than reconciled, because reconciling it means rewriting a paid artifact.
- **The spend cap overshot by 3.8%** — $8.31 against a $8.00 ceiling before the final resume — because
  it is checked between windows, never inside one. A window already answered but not yet adapted on
  cannot be written without leaving the transcript ahead of the store, so the cap's guarantee is "at
  most one window past the line", not "never past it".

**The fix that commit carries is worth reading before trusting any earlier ACE artifact.** Our
reflector prompt said "critique this task execution" where upstream's says the subject is *a model's*
reasoning, and our curator prompt had dropped upstream's statement that the playbook's reader is a
model answering a similar question *without* the ground truth. With both missing, the reflector
personified a human operator and the curator wrote advice for that operator: every bullet in a
30-sample run was process improvement — *"establish a periodic training programme"*, *"pair less
experienced users with seasoned professionals"*. The mechanism depends on who the playbook is for,
and two paraphrased sentences had removed the answer.
