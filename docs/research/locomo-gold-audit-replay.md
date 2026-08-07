# The gold-error replay — LoCoMo scored without its audited questions, and what the four-way ranking can be claimed to show

An independent audit flags 99 of LoCoMo's 1,540 judged questions as carrying a wrong golden answer.
This document replays our four-arm campaign ([`18-locomo-4way.md`](../18-locomo-4way.md)) against
that audit twice over: once by dropping the flagged questions and re-aggregating (does the ranking
survive a cleaner gold?), and once by asking what the ranking is entitled to claim at all, given
question sampling, seed jitter and the disputed verdicts themselves.

**Every number below is transcribed from two artifacts and nothing here is hand-computed:**

| artifact | produced by | what it holds |
|---|---|---|
| [`results/ext/x1/rescore.json`](../../results/ext/x1/rescore.json) (+ `rescore.md`) | `scripts/ext/x1_rescore.py` | per-run J and F1 with and without the 99, per category, and the join disclosure |
| [`results/ext/x1/power.json`](../../results/ext/x1/power.json) (+ `power.md`) | `scripts/ext/x1_power.py` | paired bootstrap CIs, rank stability under gold noise, seed and question counts |

Both are deterministic and both fail closed: each recomputes the four published J values from the
raw records first and aborts before writing if any disagrees with `18-locomo-4way.md`. All four
anchors reproduce exactly — 67.60 / 65.78 / 61.23 / 31.82 — so the tables below are re-aggregations
of the *same* runs, not a re-measurement.

## 1. The audit, and the pin

| | |
|---|---|
| Source | [`dial481/locomo-audit`](https://github.com/dial481/locomo-audit) — *LoCoMo Benchmark Audit*, an independent audit of the LoCoMo benchmark and the EverMemOS evaluation framework |
| Pin | `9493fb4b4af4256ed17a18e8fd0b3cfdeec29539` (subject: *feat: add statistical validity analysis, expand Category 5 evaluation gap*) |
| License | **CC BY-NC 4.0** — the same license as the underlying LoCoMo dataset, which was created by Maharana, A., Lee, D. H., Tulyakov, S., & Bansal, M.[^attr] and published by SNAP Research. The audit repository contains annotations and analysis derived from that dataset; this document reuses those annotations non-commercially and attributes them here. |
| Local clone | `~/.agmem/upstream/locomo-audit` (annotations at `errors.json`) |

[^attr]: The audit repository renders the third author as "Tuber, S." in both its `README.md`
and its `THIRD-PARTY-NOTICES.md`. That is a typo for **Sergey Tulyakov**, and it is corrected
here rather than transcribed, because propagating it would misname a real researcher. It is the
only place this document departs from the audit's own wording.

**The dataset the audit annotates is byte-identical to ours.** `sha256sum` on the audit's
`data/locomo10.json` and on our `~/.agmem/datasets/locomo10.json` both give
`79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`. That is what makes the join
below legitimate at all: the audit's `question_id` values are positional (`locomo_<conv>_qa<n>`),
so a diverged dataset copy would silently point them at different questions.

`errors.json` carries **156** flagged questions, of which **99 are score-corrupting**:

| `error_type` | count | score-corrupting? |
|---|---:|---|
| `WRONG_CITATION` | 57 | **no** — the golden answer is right; only its evidence pointer is wrong |
| `HALLUCINATION` | 33 | yes |
| `TEMPORAL_ERROR` | 26 | yes |
| `ATTRIBUTION_ERROR` | 24 | yes |
| `AMBIGUOUS` | 13 | yes |
| `INCOMPLETE` | 3 | yes |

The audit's own headline framing of the same 99 is that they are 6.4% of the judged set and put a
theoretical scoring ceiling of 93.57% on the benchmark. This document does not use the ceiling
figure; it uses only the 99 question ids.

## 2. The join — 99 of 99, nothing unmatched

The audit keys on `question_id`; our records key on `(conv, question text)`. The bridge is the
dataset both sides share: enumerate each conversation's qa list, and `locomo_<conv>_qa<n>` is
position `n` in it.

**`n` is 0-based over the sample's full qa list, adversarial rows included.** This was settled
empirically rather than assumed — all four candidate conventions were run against the 156 flagged
ids and the error's own `question` text compared against the mapped text. 0-based cross-checks on
**156 / 156** with zero mismatches; 1-based produces 155 mismatches and one absent id. The
include-or-exclude-adversarial half of the question is unresolvable on this catalogue and
harmlessly so: every flagged index falls strictly below its conversation's first category-5 index
(checked for all ten conversations and all 156 flagged ids), so the prefix the two enumerations
share contains no adversarial row and they agree on every id the audit uses. The full-list reading is the one implemented, and it is pinned by a
test so a future re-pin that shifts ordering fails loudly instead of replaying quietly wrong.

Measured against the headline `e3sA` records:

| | |
|---|---:|
| score-corrupting error keys | 99 |
| matched to a records row | **99** |
| **unmatched** | **0** — the disclosure list is empty |
| duplicate-matched keys | **0** |
| rows removed from the J denominator | 99 |

The same disclosure holds for **all 19 runs** in the glob: 99 keys, 99 matched, 99 rows excluded,
0 unmatched, 0 duplicate-serving keys.

**The duplicate policy, and why it is inert here.** Conversation 7 contains 11 questions that each
appear twice in the judged rows (1,540 rows resolve to 1,529 distinct keys). A flagged question
landing on one of those would remove *two* rows per key, so the policy is to exclude every row a
flagged key serves and to report the row count separately from the key count. None of the 11 is
among the flagged 99, so `excluded_rows == matched == 99` on this catalogue: the denominator drops
by exactly 99, not more. The handling exists and is tested; it simply does not fire.

**Two denominators, because J and F1 have two.** LoCoMo's 446 category-5 (adversarial) rows are
answered but never judged — they carry no verdict at all — so J is scored over 1,540 rows while F1
is scored over all 1,986. The join is therefore disclosed on both bases. That the all-rows and
judged-row exclusions are *equal* at 99 is itself the proof that no flagged question lands on an
adversarial row.

**Three runs in the glob have no J by design.** The `wujiang` eval mode scores F1 and BLEU-1 only
and emits no judge verdicts, so `gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed{1,2,3}` report
`J: null` and `n/a` on the judged basis. That is a different measurement, not a damaged file — and
the distinction is enforced rather than assumed: a run judged *in part* still raises, and only the
allowlisted F1-only mode is accepted verdict-free. Their flagged questions do leave the F1
denominator (1,986 → 1,887), as the all-rows columns show.

## 3. Re-aggregation with the 99 dropped

**Flagged questions are dropped, not zeroed.** With no corrected gold to score against, the honest
reading of an excluded J is *accuracy over the questions the audit did not dispute*. The
denominator moves with the numerator, so an arm scoring above its own average on the flagged set
goes **down** and one scoring below goes **up** — the sign of ΔJ is informative, not automatic.

All four headline arms rise. n 1,986 → 1,887; judged n 1,540 → 1,441, identically for all four.

| arm | run | J full | J excl | ΔJ | F1 full | F1 excl | ΔF1 |
|---|---|---:|---:|---:|---:|---:|---:|
| **Nemori** arm A (upstream) | `e3sA` | 67.60 | 70.02 | **+2.42** | 46.79 | 47.91 | +1.12 |
| **Nemori** arm B (0.85 filter live) | `e3sB` | 65.78 | 68.22 | **+2.44** | 45.79 | 46.93 | +1.14 |
| **A-Mem** (per-hit) | `e3sPH` | 61.23 | 63.91 | **+2.68** | 42.92 | 43.99 | +1.07 |
| **Mem0** `v0.1.94` | `e3sM` | 31.82 | 32.89 | **+1.08** | 24.71 | 25.43 | +0.71 |

**The ranking is unchanged**, and the spread widens slightly rather than closing: Mem0 gains least,
because it was already failing the flagged questions at close to its own average while the stronger
arms were carrying more of them as losses. That asymmetry is visible directly in how many of the 99
each arm got "right" against the disputed gold — 32 / 30 / 22 / 16 out of 99 for
`e3sA` / `e3sB` / `e3sPH` / `e3sM`.

Per category, ΔJ (adversarial is unjudged, so it has no J to move, and its n stays 446 → 446):

| category | n (full → excl) | `e3sA` | `e3sB` | `e3sPH` | `e3sM` |
|---|---|---:|---:|---:|---:|
| single-hop | 841 → 805 | +2.01 | +2.05 | +1.82 | +0.82 |
| multi-hop | 282 → 254 | +2.19 | +2.47 | +3.58 | +1.59 |
| temporal | 321 → 295 | +2.94 | +2.68 | +3.82 | +0.98 |
| open-domain | 96 → 87 | +1.87 | +1.65 | +1.87 | +1.01 |
| adversarial | 446 → 446 | — | — | — | — |

The other 15 runs in the glob — seed replicates, the older-embedder `armA`/`armB` pair, the
`rawq`/`perhit` ablations, the `gpt-5.6-luna` runs and the three F1-only `wujiang` runs — are
re-aggregated in `rescore.md` for context and are pinned to nothing.

## 4. Discriminative power — which gaps the evidence separates

Three different noise sources, on the same four arms. They are **not three votes on one question**:
only the first puts every question at risk.

| adjacent pair | paired bootstrap (both golds) | seed jitter | gold noise (worst basis) |
|---|---|---|---|
| `e3sA` over `e3sB` | **NOT separated** | clears at 1 seed | holds (P(not held) = 0.0001) |
| `e3sB` over `e3sPH` | **separated** | clears at 1 seed | holds (P(not held) = 0.0000) |
| `e3sPH` over `e3sM` | **separated** | clears at 1 seed | holds (P(not held) = 0.0000) |

### 4.1 Paired bootstrap over the questions

Percentile bootstrap over the per-question difference vector, resampling questions with
replacement, `n_boot = 10,000`, `seed = 0`. Paired because the arms answered the same questions in
the same order — verified positionally against the canonical dataset enumeration, not merely as a
set. `p` cannot resolve below `1/n_boot`.

| gold | pair | ΔJ (pp) | 95% CI (pp) | SE | p | disagreeing questions | excludes 0 |
|---|---|---:|---:|---:|---:|---:|---|
| full | `e3sA` − `e3sB` | +1.82 | **[−0.32, +3.90]** | 1.07 | 0.097 | 274 / 1540 | **NO** |
| full | `e3sB` − `e3sPH` | +4.55 | [+1.88, +7.14] | 1.35 | 0.001 | 434 / 1540 | yes |
| full | `e3sPH` − `e3sM` | +29.42 | [+26.56, +32.21] | 1.44 | <0.0001 | 629 / 1540 | yes |
| audit-excluded | `e3sA` − `e3sB` | +1.80 | **[−0.42, +4.03]** | 1.13 | 0.116 | 264 / 1441 | **NO** |
| audit-excluded | `e3sB` − `e3sPH` | +4.30 | [+1.60, +7.01] | 1.40 | 0.002 | 414 / 1441 | yes |
| audit-excluded | `e3sPH` − `e3sM` | +31.02 | [+28.04, +33.93] | 1.50 | <0.0001 | 607 / 1441 | yes |

### 4.2 Rank stability when the disputed verdicts are redrawn

The 99 audited questions have gold we cannot trust, so their verdicts are redrawn as Bernoulli(p)
per arm — 10,000 simulations — while the other 1,441 are held fixed. Arms are drawn independently,
which is the conservative direction: real difficulty is positively correlated across arms, and that
correlation would shrink the variance of each gap. Both readings of "the arm's rate" are reported.

| rate basis | P(observed order `e3sA > e3sB > e3sPH > e3sM`) | P(any tie) | other permutations seen |
|---|---:|---:|---|
| `unflagged` (as if graded as fairly as the rest) | **0.9999** | 0.0000 | `e3sB > e3sA > e3sPH > e3sM` at 0.0001 |
| `flagged` (jitter around the status quo) | **1.0000** | 0.0000 | none |

Five of the six pair-and-basis cells saw no flip at all across the 10,000 simulations. The
exception is the one visible in the table: `e3sA` over `e3sB` on the `unflagged` basis flipped
once. **Where a cell's flip count is zero, that is a bound rather than an impossibility** — the
rule of three (one-sided 95%) puts the flip probability at roughly `3.00e-04`, which is as sharp
as 10,000 simulations can be. Ties break toward the observed order, so each figure is an upper
bound wherever ties occur; `P(any tie)` is 0 on both bases, so none do here.

### 4.3 Seeds, and the lever seeds do not provide

Two-sample normal approximation at α = 0.05, power = 0.80, with sd = **0.35pp** (the Track-1 seed-2
replicate SD — one replication, so every seed number inherits that single measurement's
uncertainty).

| pair | ΔJ full (pp) | seeds | ΔJ audit-excluded (pp) | seeds | questions for the same power |
|---|---:|---:|---:|---:|---:|
| `e3sA` − `e3sB` | +1.82 | 1 | +1.80 | 1 | **4,186** |
| `e3sB` − `e3sPH` | +4.55 | 1 | +4.30 | 1 | 1,073 |
| `e3sPH` − `e3sM` | +29.42 | 1 | +31.02 | 1 | 29 |

Against 1,540 questions measured. A seed replicate reruns the *same* questions, so it cannot help a
gap that question sampling fails to separate — which is why the A/B row needs a benchmark nearly
three times the size rather than more seeds.

**Both counts condition on the observed gap being the true effect.** They are point estimates
standing in for an unknown, so each reads as "what it would take to detect a gap this big, if it is
really this big" — not "what it would take to settle this comparison". Both scale as 1/ΔJ², so a
true gap half the observed one costs four times as much, and a true gap of zero is undetectable at
any budget rather than merely expensive. That caveat bites hardest exactly where the number is most
tempting: the A/B row's own CI covers zero.

Smallest gap each seed budget can resolve at the same α and power:

| seeds | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| min detectable ΔJ (pp) | 1.39 | 0.98 | 0.69 | 0.49 | 0.35 |

## 5. Conclusion — what may be claimed

Reproduced verbatim from `power.json`, which generates it from the tables above rather than
asserting it:

> Only part of the ranking e3sA > e3sB > e3sPH > e3sM is claimable. The paired 95% bootstrap CIs
> separate e3sB over e3sPH (+4.55pp, [+1.88, +7.14], p=0.001); e3sPH over e3sM (+29.42pp,
> [+26.56, +32.21], p<0.0001), but not e3sA over e3sB (+1.82pp, [-0.32, +3.90], p=0.097) -- that
> interval covers zero, so at 1,540 questions the gap is inside question-sampling noise. Dropping
> the 99 audited questions does not rescue it (+1.80pp [-0.42, +4.03]). More seeds would not fix
> it, because rerunning the same questions does not add questions. At the measured paired SE, e3sA
> over e3sB would need roughly 4,186 questions (vs 1,540) to be powered at alpha=0.05 / power=0.80
> -- a figure that conditions on the observed gap being the true effect, which is exactly what its
> own CI declines to establish. Read it as the size needed if the gap is real and this big; a
> smaller true gap needs more questions without bound, and a true gap of zero makes the number
> undefined rather than large. The other two noise sources are not the binding constraint and must
> not be read as agreement: every gap clears the 1.39pp a single seed resolves at alpha=0.05 /
> power=0.80 given the +/-0.35pp replicate SD, and redrawing the 99 disputed verdicts leaves the
> order intact in at least 99.99% of 10,000 simulations under both resampling rates, with the order
> unchanged when those questions are dropped instead. Both of those checks are narrower than the
> bootstrap: the gold-noise simulation perturbs only 99 of 1,540 verdicts and holds the rest fixed,
> which is why it reports a stability the bootstrap over all 1,540 does not support. So: the
> ranking is claimable at one seed and under either gold for the pairs listed as separated above,
> and for those pairs only; any future arm landing within 1.39pp of another is a tie at one seed
> before question sampling is even considered, and one at the 0.35pp replicate SD would need 16
> seeds.

Two consequences follow directly and are worth stating in the form a citation would need:

1. **"Nemori arm A beats Nemori arm B" is not an established ranking** in this campaign. It is a
   +1.82pp point estimate whose 95% interval includes zero under both golds. Anything downstream
   treating it as settled needs revisiting.
2. **The audit does not change any ranking verdict.** Every arm rises when the 99 are dropped, the
   order is identical under both golds, and the two separated pairs stay separated while the
   unseparated one stays unseparated. The bad gold was never what the ranking rested on.

## 6. Reproducing this

Two commands regenerate both artifacts, deterministically and byte-identically:

```bash
uv run python scripts/ext/x1_rescore.py \
  --records-glob 'results/repro/*_all_*.records.jsonl' \
  --out results/ext/x1

uv run python scripts/ext/x1_power.py \
  --out results/ext/x1 \
  --records-dir results/repro
```

Both default `--dataset` to `~/.agmem/datasets/locomo10.json` and `--errors` to
`~/.agmem/upstream/locomo-audit/errors.json`; pass them explicitly if your clones live elsewhere.
Neither makes an LLM or network call, and neither costs anything to run.

**Extending to five arms when Zep lands.** The rescore command extends by itself: a new
`*_all_*.records.jsonl` under `results/repro/` is picked up by the glob and appears in every table
with no code change. The power command does **not** — its four arms, their adjacent pairs and their
anchors are named explicitly (`ARM_STEMS` and `ADJACENT_PAIRS` in `scripts/ext/x1_power.py`,
`HEADLINE_ANCHORS` in `scripts/ext/x1_rescore.py`), by design: the pair chain is fixed to the
full-gold rank order so that a reordering has to be noticed rather than absorbed. Adding Zep means
adding its stem and anchor at its rank position in those three places; the command line itself is
unchanged, and the anchor check will refuse the run if the new arm's recomputed J does not match its
published one.

**The input records are durable on disk but not in git.** All four headline
`*.records.jsonl` files — the actual inputs to both commands — are ignored by
`results/repro/*.records.jsonl` in `.gitignore`, a rule added on 2026-08-04 when "small" stopped
being true (A-Mem's are ~19MB each, Track-1 Nemori's ~46MB; committing one track's six would have
taken the repo from 141MB to over 400MB). They follow the repo's standing rule for heavy artifacts:
never deleted, so nothing is lost, but not committed. Eight earlier A-Mem-campaign records files
predate the rule and remain tracked, since a new ignore rule does not affect already-tracked files.

The practical consequence for anyone outside this machine: **a third party cannot re-run §3 from
this repository alone.** What the repository does commit is the output of that computation —
`results/ext/x1/{rescore,power}.{json,md}` — plus every run's summary `.json` under
`results/repro/`. That substitutes at the aggregate level (the per-run J, F1, per-category and join
figures are all there and all checkable against the anchors) but not at the per-question level: the
paired difference vectors, and therefore an independent bootstrap, need the records files
themselves.
