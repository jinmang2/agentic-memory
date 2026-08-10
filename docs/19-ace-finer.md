# ACE on FiNER — what a self-evolving playbook bought, measured against not having one

ACE's headline for online adaptation is stated on FiNER: **69.1% → 81.9%**, plus "−91.5% latency
and −83.6% token cost vs Dynamic Cheatsheet" (release README). FiNER is also the one ACE experiment
whose data actually ships — the AppWorld code that carries the paper's flagship result is absent from
the public repository ([ledger B-6](17-defect-ledger.md)). So it is the only ACE claim this campaign
can put a number against, and this document is that number.

**What this is and is not.** It is not a reproduction of 69.1 → 81.9. That figure was produced by a
much stronger generator than the one priced here, through a training loop that spends two to three
times the calls ours does. It is a controlled measurement of one question: *on the same 441 questions,
does growing an ACE playbook alongside beat not growing one?*

## Protocol

| | |
|---|---|
| Benchmark | FiNER, shipped test split `finer_test_subset_006_seed42.jsonl` — 441 samples × exactly 4 US-GAAP tags = **1,764 tagging decisions** |
| Generator | `gpt-4o-mini`, temperature 0 |
| Reflector / Curator | `gpt-4o-mini`, temperature 0, `max_tokens` 4096 (one role — see condition 4) |
| Embedder | `text-embedding-3-small` (curator dedup + task episodes) |
| Adaptation | upstream's **online** mode: 30 windows of 15, each answered with the playbook as it stands and only then trained on (`ace.py:939-997`) |
| Seeds | **one** |

## Results

| arm | tag accuracy | sample accuracy | LLM calls | cost | wall clock |
|---|---|---|---|---|---|
| **base** — empty playbook, no learning | **48.24** | 16.10 | 441 | $0.192 | 27 min |
| **online** — playbook grown over all 441 | 46.71 | 15.42 | **3,775** | $1.461 | 80 min |

Paired over the same questions in the same order (bootstrap imported from `scripts/ext/x1_power.py`,
10,000 resamples, seed 0):

```
sample_accuracy   online - base   Δ = -0.68 pp   95% CI [-3.40, +2.04]   p = 0.687   NOT separated
tag_accuracy      online - base   Δ = -1.53 pp   (point estimate — see "one interval is missing")
per-window sign   online ahead in 10, behind in 16, tied in 4 of 30
```

**Growing a playbook over 441 samples produced no measurable improvement, at 7.6× the cost.** The
point estimates sit on the wrong side of zero on both metrics and in 16 of 30 windows, but the
interval covers zero, so the claim this supports is *no measurable effect* — **not** that the playbook
hurts.

The playbook was not empty and not junk. 441 curator calls produced **140 bullets** that survived
0.90 dedup, reaching **43,699 characters**, injected in full into every generator call thereafter.
Its content was specific: *"Differentiate between `AmortizationOfFinancingCosts` and
`AmortizationOfIntangibleAssets`"*, *"Clarify the distinction between `DeferredFinanceCostsGross` and
`DeferredFinanceCostsNet`"*. It simply did not convert into answers.

The deficit narrows as the playbook accumulates (mean Δ per window **−2.44 pp** over the first
fifteen, **−0.72 pp** over the second fifteen). That is consistent with no effect, and equally
consistent with regression to the mean; with 15 windows a side it is a record, not a result, and
nothing here rests on it.

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
ACE's own 69.1 → 81.9 as much as to us. The paired comparison above does not have the problem: the
two arms answer the same 441 questions, so the difference vector cancels difficulty exactly.

This is also the reason the early windows are not evidence of anything. The three pre-flight smokes
all showed accuracy falling from window 0 to window 1 (61.7→35.0, 61.7→36.7, 55.0→41.7), which reads
as "the playbook immediately hurts" — until the base arm scores **35.0** on the same window 1 with no
playbook in existence. It was the questions.

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
2. **Our adaptation loop is 3 calls per sample; upstream's is 3 to 8.** Upstream regenerates after
   reflecting, up to `max_num_rounds`=3 times, and stops early when a retry lands (`ace.py:498-543`);
   we reflect once at `on_task_end` and never re-answer. **This is the largest open question in the
   table** — upstream's extra calls can flip a sample to correct *inside* the training step, so its
   reflector sees outcomes ours never produces, and it is not excluded that those retries are what
   makes the mechanism pay. An `online_retry` arm is what would close it.
3. **One attempt is scored and learned from.** Upstream answers a window for scoring and then calls
   the generator *again* inside its training step; we reflect on the attempt already scored.
4. **Reflector and curator share one role**, so they cannot take different models or temperatures;
   upstream allows distinct ones (`ace.py:36-63`).
5. **Dedup is always on at 0.90 and drops the incoming bullet.** Upstream's analyzer is opt-in, off in
   `ace.py`, off in this harness's flag, and off again by silent fallback when its optional
   dependencies are missing, and when on it LLM-merges groups rather than dropping (ledger B-6). Ours
   is a third behavior, not upstream's switched on.
6. **Single seed**, and no replication of the sign.

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
| paired analysis | `results/repro/finer_paired.json` (`scripts/repro/finer_paired.py`) |
| pre-flight smokes | `..._ace_finer_smoke30{,b,c}.json` — the runs that caught the prompt defect below |
| runner | `scripts/repro/exp_ace_finer.py`; adapter `src/agmem/bench/finer.py` |

Each carries per-window summaries, the full LLM I/O trace, the memory snapshot, the evolution log and
the per-question records. Both arms are stamped at commit `6783c24`, and all 882 reflector/curator
calls of the online arm are verifiable in its trace as having used the post-fix prompts.

**The fix that commit carries is worth reading before trusting any earlier ACE artifact.** Our
reflector prompt said "critique this task execution" where upstream's says the subject is *a model's*
reasoning, and our curator prompt had dropped upstream's statement that the playbook's reader is a
model answering a similar question *without* the ground truth. With both missing, the reflector
personified a human operator and the curator wrote advice for that operator: every bullet in a
30-sample run was process improvement — *"establish a periodic training programme"*, *"pair less
experienced users with seasoned professionals"*. The mechanism depends on who the playbook is for,
and two paraphrased sentences had removed the answer.
