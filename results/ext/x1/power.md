# X1 discriminative power: what the four-arm ranking can be claimed to show

Arms: 1540 judged questions each, identical question order (verified positionally, and against the canonical dataset enumeration). 99 of them are flagged score-corrupting by the audit.

Deterministic: seed=0, n_boot=10000, n_sim=10000. The same command reproduces these numbers byte for byte.

## Arms

| arm | run | J (full gold) | J (audit-excluded) | correct on the 99 flagged |
| --- | --- | ---: | ---: | ---: |
| `e3sA` | `gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA` | 67.60 | 70.02 | 32 / 99 |
| `e3sB` | `gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB` | 65.78 | 68.22 | 30 / 99 |
| `e3sPH` | `gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH` | 61.23 | 63.91 | 22 / 99 |
| `e3sM` | `gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM` | 31.82 | 32.89 | 16 / 99 |

Rank order is `e3sA > e3sB > e3sPH > e3sM` under the full gold and `e3sA > e3sB > e3sPH > e3sM` with the flagged questions dropped.

**Which adjacent gaps the evidence actually separates:**

| pair | paired bootstrap (both golds) | seed jitter | gold noise (worst basis) |
| --- | --- | --- | --- |
| e3sA over e3sB | **NOT separated** | clears at 1 seed | holds (P(not held) = 0.0001) |
| e3sB over e3sPH | **separated** | clears at 1 seed | holds (P(not held) = 0.0000) |
| e3sPH over e3sM | **separated** | clears at 1 seed | holds (P(not held) = 0.0000) |

The three columns are three different noise sources, not three votes on one question. Only the first puts every question at risk; sections 2 and 3 explain why the other two are narrower and must not be read as corroboration.

## 1. Paired bootstrap CIs for the adjacent gaps

Percentile bootstrap over the per-question difference vector, resampling questions with replacement. Paired because the arms answered the same questions in the same order; an unpaired interval would discard that and come out roughly twice as wide. `p` is the achieved significance level from the bootstrap distribution and cannot resolve below 1/n_boot, so `0.000000` means "under the resolution of this many resamples", not "impossible".

| gold | pair | dJ (pp) | 95% CI (pp) | SE | p | disagreeing questions | excludes 0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| full | e3sA - e3sB | +1.82 | [-0.32, +3.90] | 1.07 | 0.097200 | 274 / 1540 | NO |
| full | e3sB - e3sPH | +4.55 | [+1.88, +7.14] | 1.35 | 0.001000 | 434 / 1540 | yes |
| full | e3sPH - e3sM | +29.42 | [+26.56, +32.21] | 1.44 | 0.000000 | 629 / 1540 | yes |
| audit-excluded | e3sA - e3sB | +1.80 | [-0.42, +4.03] | 1.13 | 0.116200 | 264 / 1441 | NO |
| audit-excluded | e3sB - e3sPH | +4.30 | [+1.60, +7.01] | 1.40 | 0.002400 | 414 / 1441 | yes |
| audit-excluded | e3sPH - e3sM | +31.02 | [+28.04, +33.93] | 1.50 | 0.000000 | 607 / 1441 | yes |

## 2. Rank stability when the flagged verdicts are redrawn

The 99 audited questions have gold we cannot trust, so their verdicts are redrawn as Bernoulli(p) per arm while the other 1441 verdicts are held fixed. Arms are drawn independently, which is conservative: real difficulty is positively correlated across arms, and that correlation would shrink the variance of each gap. This is a gold-noise estimate only -- the fixed verdicts carry sampling uncertainty of their own, which is what section 1 measures.

Two readings of "the arm's rate", both reported: **unflagged** draws at the arm's accuracy on trustworthy gold (as if the flagged questions had been graded as fairly as the rest), **flagged** draws at the rate actually observed on the bad gold (jitter around the status quo).

### rate basis: `unflagged`

P(observed order `e3sA > e3sB > e3sPH > e3sM`) = **0.9999** over 10,000 simulations; P(any tie) = 0.0000. Ties break toward the observed order, so the figure above is an upper bound whenever ties occur.

| arm | draw rate p | mean simulated J | sd | observed J |
| --- | ---: | ---: | ---: | ---: |
| `e3sA` | 0.7002 | 70.02 | 0.297 | 67.60 |
| `e3sB` | 0.6822 | 68.22 | 0.301 | 65.78 |
| `e3sPH` | 0.6391 | 63.91 | 0.309 | 61.23 |
| `e3sM` | 0.3289 | 32.89 | 0.303 | 31.82 |

| adjacent pair | P(flip) | P(tie) | mean gap (pp) | sd (pp) |
| --- | ---: | ---: | ---: | ---: |
| e3sA over e3sB | 0.000100 | 0.000000 | +1.80 | 0.424 |
| e3sB over e3sPH | 0.000000 | 0.000000 | +4.31 | 0.427 |
| e3sPH over e3sM | 0.000000 | 0.000000 | +31.02 | 0.436 |

| ranking | probability |
| --- | ---: |
| e3sA > e3sB > e3sPH > e3sM | 0.999900 |
| e3sB > e3sA > e3sPH > e3sM | 0.000100 |

### rate basis: `flagged`

P(observed order `e3sA > e3sB > e3sPH > e3sM`) = **1.0000** over 10,000 simulations; P(any tie) = 0.0000. Ties break toward the observed order, so the figure above is an upper bound whenever ties occur.

| arm | draw rate p | mean simulated J | sd | observed J |
| --- | ---: | ---: | ---: | ---: |
| `e3sA` | 0.3232 | 67.59 | 0.305 | 67.60 |
| `e3sB` | 0.3030 | 65.78 | 0.297 | 65.78 |
| `e3sPH` | 0.2222 | 61.23 | 0.270 | 61.23 |
| `e3sM` | 0.1616 | 31.82 | 0.239 | 31.82 |

| adjacent pair | P(flip) | P(tie) | mean gap (pp) | sd (pp) |
| --- | ---: | ---: | ---: | ---: |
| e3sA over e3sB | 0.000000 | 0.000000 | +1.81 | 0.426 |
| e3sB over e3sPH | 0.000000 | 0.000000 | +4.55 | 0.400 |
| e3sPH over e3sM | 0.000000 | 0.000000 | +29.42 | 0.362 |

| ranking | probability |
| --- | ---: |
| e3sA > e3sB > e3sPH > e3sM | 1.000000 |

No flip in 10,000 simulations bounds each flip probability at roughly 3.00e-04 (rule of three, one-sided 95%). It is not evidence that a flip is impossible.

## 3. Seeds required per adjacent gap

Two-sample normal approximation `n = ((z_a/2 + z_b) * sd * sqrt(2) / d)^2`, rounded up, with sd = 0.35pp (the Track-1 seed-2 replicate SD), alpha = 0.05, power = 0.8. This is jitter between reruns of the same configuration, a different noise source from both sections above; the three do not compose into a single interval and are not summed here.

| pair | dJ full (pp) | seeds | dJ audit-excluded (pp) | seeds | questions for the same power |
| --- | ---: | ---: | ---: | ---: | ---: |
| e3sA - e3sB | +1.82 | 1 | +1.80 | 1 | 4,186 |
| e3sB - e3sPH | +4.55 | 1 | +4.30 | 1 | 1,073 |
| e3sPH - e3sM | +29.42 | 1 | +31.02 | 1 | 29 |

The last column is the lever seeds do not provide. A seed replicate reruns the *same* questions, so it cannot help a gap that question sampling fails to separate; that column is the benchmark size at which the measured paired SE would put the gap at the same alpha/power threshold, assuming the added questions resemble the ones already graded.

**Both seed and question counts condition on the observed gap being the true effect.** They are point estimates standing in for an unknown, so every count here reads as "what it would take to detect a gap this big, if it is really this big" -- not "what it would take to settle this comparison". Both scale as 1/dJ^2, so a true gap half the observed one costs four times as much, and a true gap of zero is undetectable at any budget rather than merely expensive. This matters most where the number is most tempting: a pair marked **NOT separated** above has a CI covering zero, so its row is a conditional answer to a question the data has not settled, not a plan.

Smallest gap each seed budget can resolve:

| seeds | min detectable dJ (pp) |
| ---: | ---: |
| 1 | 1.39 |
| 2 | 0.98 |
| 4 | 0.69 |
| 8 | 0.49 |
| 16 | 0.35 |

## 4. Conclusion

Only part of the ranking e3sA > e3sB > e3sPH > e3sM is claimable. The paired 95% bootstrap CIs separate e3sB over e3sPH (+4.55pp, [+1.88, +7.14], p=0.001); e3sPH over e3sM (+29.42pp, [+26.56, +32.21], p<0.0001), but not e3sA over e3sB (+1.82pp, [-0.32, +3.90], p=0.097) -- that interval covers zero, so at 1,540 questions the gap is inside question-sampling noise. Dropping the 99 audited questions does not rescue it (+1.80pp [-0.42, +4.03]). More seeds would not fix it, because rerunning the same questions does not add questions. At the measured paired SE, e3sA over e3sB would need roughly 4,186 questions (vs 1,540) to be powered at alpha=0.05 / power=0.80 -- a figure that conditions on the observed gap being the true effect, which is exactly what its own CI declines to establish. Read it as the size needed if the gap is real and this big; a smaller true gap needs more questions without bound, and a true gap of zero makes the number undefined rather than large. The other two noise sources are not the binding constraint and must not be read as agreement: every gap clears the 1.39pp a single seed resolves at alpha=0.05 / power=0.80 given the +/-0.35pp replicate SD, and redrawing the 99 disputed verdicts leaves the order intact in at least 99.99% of 10,000 simulations under both resampling rates, with the order unchanged when those questions are dropped instead. Both of those checks are narrower than the bootstrap: the gold-noise simulation perturbs only 99 of 1,540 verdicts and holds the rest fixed, which is why it reports a stability the bootstrap over all 1,540 does not support. So: the ranking is claimable at one seed and under either gold for the pairs listed as separated above, and for those pairs only; any future arm landing within 1.39pp of another is a tie at one seed before question sampling is even considered, and one at the 0.35pp replicate SD would need 16 seeds.

