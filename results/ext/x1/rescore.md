# X1 gold-error replay: J with and without the audited questions

Audit: `/home/jinmang2/.agmem/upstream/locomo-audit/errors.json` -- 156 flagged questions, 99 of them score-corrupting (WRONG_CITATION is benign: the gold answer is right, only its evidence pointer is wrong).

Dataset (join bridge): `/home/jinmang2/.agmem/datasets/locomo10.json`.
Records glob: `results/repro/*_all_*.records.jsonl` -- 19 run(s).

Flagged questions are **dropped, not zeroed**: with no corrected gold to score against, the honest reading of an excluded J is *accuracy over the questions the audit did not dispute*. Because the denominator moves too, an arm scoring above its own average on the flagged set goes down and one scoring below goes up -- the sign of delta-J is informative, not automatic.

Headline runs are marked; every other row is an ablation, seed replicate or alternate-model run carried along for context and pinned to nothing.

## Overall

| run | headline | n (full -> excl) | judged n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH` | **yes** | 1986 -> 1887 | 1540 -> 1441 | 61.23 | 63.91 | +2.68 | 42.92 | 43.99 | +1.07 |
| `gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM` | **yes** | 1986 -> 1887 | 1540 -> 1441 | 31.82 | 32.89 | +1.08 | 24.71 | 25.43 | +0.71 |
| `gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB` | **yes** | 1986 -> 1887 | 1540 -> 1441 | 65.78 | 68.22 | +2.44 | 45.79 | 46.93 | +1.14 |
| `gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA` | **yes** | 1986 -> 1887 | 1540 -> 1441 | 67.60 | 70.02 | +2.42 | 46.79 | 47.91 | +1.12 |
| `gpt-4o-mini_all_k10_ours_expand-on_run1_e3s` |  | 1986 -> 1887 | 1540 -> 1441 | 59.87 | 62.32 | +2.45 | 41.42 | 42.43 | +1.01 |
| `gpt-4o-mini_all_k10_ours_expand-on_run1_seed1` |  | 1986 -> 1887 | 1540 -> 1441 | 50.00 | 52.12 | +2.12 | 34.15 | 35.17 | +1.03 |
| `gpt-4o-mini_all_k10_ours_expand-on_run1_seed2` |  | 1986 -> 1887 | 1540 -> 1441 | 50.91 | 53.23 | +2.32 | 35.03 | 36.01 | +0.98 |
| `gpt-4o-mini_all_k10_ours_expand-on_run1_seed3` |  | 1986 -> 1887 | 1540 -> 1441 | 51.10 | 52.88 | +1.78 | 35.21 | 36.10 | +0.90 |
| `gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed1` |  | 1986 -> 1887 | 0 -> 0 | -- | -- | -- | 34.29 | 35.41 | +1.12 |
| `gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed2` |  | 1986 -> 1887 | 0 -> 0 | -- | -- | -- | 35.05 | 36.19 | +1.13 |
| `gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed3` |  | 1986 -> 1887 | 0 -> 0 | -- | -- | -- | 35.19 | 36.26 | +1.08 |
| `gpt-4o-mini_amem_rawq_all_k10_ours_expand-on_run1_e3sRAWQ` |  | 1986 -> 1887 | 1540 -> 1441 | 65.13 | 68.01 | +2.88 | 46.21 | 47.60 | +1.39 |
| `gpt-4o-mini_amem_rawq_perhit_all_k10_ours_expand-on_run1_e3sRQPH` |  | 1986 -> 1887 | 1540 -> 1441 | 65.58 | 68.29 | +2.70 | 47.08 | 48.34 | +1.26 |
| `gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_armB` |  | 1986 -> 1887 | 1540 -> 1441 | 65.52 | 68.36 | +2.84 | 44.91 | 46.22 | +1.30 |
| `gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_armB_s2` |  | 1986 -> 1887 | 1540 -> 1441 | 65.19 | 67.87 | +2.67 | 44.48 | 45.75 | +1.28 |
| `gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_armA` |  | 1986 -> 1887 | 1540 -> 1441 | 63.38 | 65.72 | +2.34 | 42.66 | 43.73 | +1.07 |
| `gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_armA_s2` |  | 1986 -> 1887 | 1540 -> 1441 | 63.57 | 66.00 | +2.42 | 43.84 | 44.96 | +1.12 |
| `gpt-5.6-luna_nemori_merge085_all_k10_ours_expand-off_run1_lunaB` |  | 1986 -> 1887 | 1540 -> 1441 | 76.56 | 79.32 | +2.76 | 49.54 | 50.57 | +1.03 |
| `gpt-5.6-luna_nemori_upstream_all_k10_ours_expand-off_run1_lunaA` |  | 1986 -> 1887 | 1540 -> 1441 | 76.88 | 79.94 | +3.06 | 48.76 | 49.86 | +1.10 |

## Anchor check (headline runs only)

| run | anchor J | recomputed J (2dp) | full precision | ok |
| --- | ---: | ---: | ---: | --- |
| `gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH` | 61.23 | 61.23 | 61.2338 | PASS |
| `gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM` | 31.82 | 31.82 | 31.8182 | PASS |
| `gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB` | 65.78 | 65.78 | 65.7792 | PASS |
| `gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA` | 67.60 | 67.60 | 67.5974 | PASS |

## Join disclosure

Excluded-row counts are the true denominator reductions, and they exceed `matched` wherever one question serves more than one records row. Two bases are reported because the two metrics have different denominators: J is scored over judged rows, F1 over all of them.

The F1-only runs (`wujiang`, which emits no judge verdicts) have no judged basis at all -- `n/a` there means the question is unmeasurable for that run, not that the join failed. Their flagged questions do leave the F1 denominator, as the all-rows columns show.

| run | error keys | matched (all rows) | excluded rows (F1 denom) | dup keys (all rows) | matched (judged) | excluded judged rows (J denom) | dup keys (judged) | unmatched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |
| `gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |
| `gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |
| `gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |
| `gpt-4o-mini_all_k10_ours_expand-on_run1_e3s` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |
| `gpt-4o-mini_all_k10_ours_expand-on_run1_seed1` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |
| `gpt-4o-mini_all_k10_ours_expand-on_run1_seed2` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |
| `gpt-4o-mini_all_k10_ours_expand-on_run1_seed3` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |
| `gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed1` | 99 | 99 | 99 | 0 | n/a | n/a | n/a | 0 |
| `gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed2` | 99 | 99 | 99 | 0 | n/a | n/a | n/a | 0 |
| `gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed3` | 99 | 99 | 99 | 0 | n/a | n/a | n/a | 0 |
| `gpt-4o-mini_amem_rawq_all_k10_ours_expand-on_run1_e3sRAWQ` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |
| `gpt-4o-mini_amem_rawq_perhit_all_k10_ours_expand-on_run1_e3sRQPH` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |
| `gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_armB` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |
| `gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_armB_s2` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |
| `gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_armA` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |
| `gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_armA_s2` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |
| `gpt-5.6-luna_nemori_merge085_all_k10_ours_expand-off_run1_lunaB` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |
| `gpt-5.6-luna_nemori_upstream_all_k10_ours_expand-off_run1_lunaA` | 99 | 99 | 99 | 0 | 99 | 99 | 0 | 0 |

## Per category

### `gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH` (headline)

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 31.90 | 31.90 | +0.00 |
| multi-hop | 282 -> 254 | 57.45 | 61.02 | +3.58 | 35.95 | 37.65 | +1.70 |
| open-domain | 96 -> 87 | 29.17 | 31.03 | +1.87 | 21.67 | 23.06 | +1.40 |
| single-hop | 841 -> 805 | 65.64 | 67.45 | +1.82 | 51.01 | 52.33 | +1.32 |
| temporal | 321 -> 295 | 62.62 | 66.44 | +3.82 | 49.54 | 51.17 | +1.64 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM` (headline)

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 25.78 | 25.78 | +0.00 |
| multi-hop | 282 -> 254 | 32.27 | 33.86 | +1.59 | 18.83 | 19.95 | +1.12 |
| open-domain | 96 -> 87 | 20.83 | 21.84 | +1.01 | 14.76 | 15.44 | +0.68 |
| single-hop | 841 -> 805 | 34.96 | 35.78 | +0.82 | 25.56 | 26.20 | +0.64 |
| temporal | 321 -> 295 | 26.48 | 27.46 | +0.98 | 29.16 | 30.45 | +1.29 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB` (headline)

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 38.69 | 38.69 | +0.00 |
| multi-hop | 282 -> 254 | 65.25 | 67.72 | +2.47 | 40.56 | 41.82 | +1.26 |
| open-domain | 96 -> 87 | 27.08 | 28.74 | +1.65 | 18.09 | 19.12 | +1.03 |
| single-hop | 841 -> 805 | 73.60 | 75.65 | +2.05 | 53.98 | 55.51 | +1.53 |
| temporal | 321 -> 295 | 57.32 | 60.00 | +2.68 | 47.07 | 48.58 | +1.51 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA` (headline)

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 37.65 | 37.65 | +0.00 |
| multi-hop | 282 -> 254 | 69.86 | 72.05 | +2.19 | 43.47 | 45.03 | +1.56 |
| open-domain | 96 -> 87 | 29.17 | 31.03 | +1.87 | 20.40 | 21.72 | +1.32 |
| single-hop | 841 -> 805 | 75.51 | 77.52 | +2.01 | 55.57 | 57.02 | +1.45 |
| temporal | 321 -> 295 | 56.39 | 59.32 | +2.94 | 47.28 | 48.75 | +1.47 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_all_k10_ours_expand-on_run1_e3s`

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 27.92 | 27.92 | +0.00 |
| multi-hop | 282 -> 254 | 56.03 | 59.06 | +3.03 | 35.97 | 38.06 | +2.09 |
| open-domain | 96 -> 87 | 23.96 | 25.29 | +1.33 | 17.00 | 17.92 | +0.92 |
| single-hop | 841 -> 805 | 64.21 | 65.96 | +1.75 | 49.49 | 50.75 | +1.26 |
| temporal | 321 -> 295 | 62.62 | 66.10 | +3.48 | 51.09 | 52.66 | +1.57 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_all_k10_ours_expand-on_run1_seed1`

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 23.34 | 23.34 | +0.00 |
| multi-hop | 282 -> 254 | 50.71 | 53.54 | +2.83 | 32.03 | 34.33 | +2.30 |
| open-domain | 96 -> 87 | 21.88 | 22.99 | +1.11 | 15.23 | 15.96 | +0.73 |
| single-hop | 841 -> 805 | 51.84 | 53.79 | +1.95 | 39.25 | 40.73 | +1.48 |
| temporal | 321 -> 295 | 52.96 | 54.92 | +1.96 | 43.31 | 44.29 | +0.98 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_all_k10_ours_expand-on_run1_seed2`

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 23.58 | 23.58 | +0.00 |
| multi-hop | 282 -> 254 | 51.42 | 54.33 | +2.91 | 31.93 | 33.69 | +1.76 |
| open-domain | 96 -> 87 | 20.83 | 21.84 | +1.01 | 17.53 | 18.55 | +1.02 |
| single-hop | 841 -> 805 | 52.68 | 54.53 | +1.86 | 40.27 | 41.60 | +1.33 |
| temporal | 321 -> 295 | 54.83 | 57.97 | +3.14 | 45.17 | 46.67 | +1.50 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_all_k10_ours_expand-on_run1_seed3`

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 23.94 | 23.94 | +0.00 |
| multi-hop | 282 -> 254 | 52.84 | 54.72 | +1.89 | 32.18 | 33.86 | +1.69 |
| open-domain | 96 -> 87 | 20.83 | 21.84 | +1.01 | 17.11 | 18.04 | +0.93 |
| single-hop | 841 -> 805 | 52.20 | 53.79 | +1.59 | 40.01 | 41.34 | +1.33 |
| temporal | 321 -> 295 | 55.76 | 57.97 | +2.20 | 46.33 | 47.44 | +1.11 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed1`

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 30.32 | 30.32 | +0.00 |
| multi-hop | 282 -> 254 | -- | -- | -- | 29.07 | 31.06 | +1.99 |
| open-domain | 96 -> 87 | -- | -- | -- | 9.59 | 10.25 | +0.66 |
| single-hop | 841 -> 805 | -- | -- | -- | 37.30 | 38.58 | +1.28 |
| temporal | 321 -> 295 | -- | -- | -- | 43.88 | 45.60 | +1.72 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed2`

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 31.73 | 31.73 | +0.00 |
| multi-hop | 282 -> 254 | -- | -- | -- | 30.03 | 31.40 | +1.37 |
| open-domain | 96 -> 87 | -- | -- | -- | 11.30 | 12.09 | +0.79 |
| single-hop | 841 -> 805 | -- | -- | -- | 38.00 | 39.46 | +1.46 |
| temporal | 321 -> 295 | -- | -- | -- | 43.48 | 45.23 | +1.75 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed3`

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 32.05 | 32.05 | +0.00 |
| multi-hop | 282 -> 254 | -- | -- | -- | 28.36 | 30.07 | +1.71 |
| open-domain | 96 -> 87 | -- | -- | -- | 11.41 | 12.27 | +0.85 |
| single-hop | 841 -> 805 | -- | -- | -- | 37.97 | 39.06 | +1.09 |
| temporal | 321 -> 295 | -- | -- | -- | 45.35 | 47.40 | +2.05 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_amem_rawq_all_k10_ours_expand-on_run1_e3sRAWQ`

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 35.20 | 35.20 | +0.00 |
| multi-hop | 282 -> 254 | 62.06 | 66.14 | +4.08 | 38.84 | 41.21 | +2.36 |
| open-domain | 96 -> 87 | 27.08 | 28.74 | +1.65 | 21.06 | 22.39 | +1.34 |
| single-hop | 841 -> 805 | 70.51 | 72.67 | +2.16 | 54.91 | 56.62 | +1.71 |
| temporal | 321 -> 295 | 65.11 | 68.47 | +3.37 | 52.73 | 54.69 | +1.96 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_amem_rawq_perhit_all_k10_ours_expand-on_run1_e3sRQPH`

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 35.96 | 35.96 | +0.00 |
| multi-hop | 282 -> 254 | 63.12 | 67.32 | +4.20 | 40.56 | 42.91 | +2.35 |
| open-domain | 96 -> 87 | 28.12 | 29.89 | +1.76 | 21.87 | 23.29 | +1.42 |
| single-hop | 841 -> 805 | 71.94 | 73.91 | +1.97 | 55.82 | 57.47 | +1.65 |
| temporal | 321 -> 295 | 62.31 | 65.08 | +2.78 | 52.92 | 54.22 | +1.30 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_armB`

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 35.77 | 35.77 | +0.00 |
| multi-hop | 282 -> 254 | 65.25 | 69.29 | +4.04 | 41.79 | 43.99 | +2.20 |
| open-domain | 96 -> 87 | 32.29 | 34.48 | +2.19 | 21.76 | 23.22 | +1.46 |
| single-hop | 841 -> 805 | 72.18 | 74.04 | +1.86 | 52.71 | 54.15 | +1.44 |
| temporal | 321 -> 295 | 58.26 | 62.03 | +3.78 | 46.87 | 49.06 | +2.20 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_armB_s2`

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 33.49 | 33.49 | +0.00 |
| multi-hop | 282 -> 254 | 66.31 | 69.69 | +3.37 | 42.65 | 44.89 | +2.24 |
| open-domain | 96 -> 87 | 28.12 | 29.89 | +1.76 | 18.80 | 19.95 | +1.16 |
| single-hop | 841 -> 805 | 72.53 | 74.66 | +2.13 | 53.79 | 55.41 | +1.62 |
| temporal | 321 -> 295 | 56.07 | 58.98 | +2.91 | 44.64 | 46.32 | +1.67 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_armA`

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 32.05 | 32.05 | +0.00 |
| multi-hop | 282 -> 254 | 64.18 | 67.32 | +3.14 | 41.53 | 43.29 | +1.77 |
| open-domain | 96 -> 87 | 28.12 | 29.89 | +1.76 | 18.52 | 19.64 | +1.13 |
| single-hop | 841 -> 805 | 71.22 | 73.04 | +1.82 | 52.01 | 53.54 | +1.53 |
| temporal | 321 -> 295 | 52.65 | 54.92 | +2.27 | 41.14 | 42.10 | +0.96 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_armA_s2`

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 34.97 | 34.97 | +0.00 |
| multi-hop | 282 -> 254 | 66.67 | 69.29 | +2.62 | 41.75 | 43.41 | +1.66 |
| open-domain | 96 -> 87 | 28.12 | 29.89 | +1.76 | 18.63 | 19.72 | +1.08 |
| single-hop | 841 -> 805 | 70.63 | 72.55 | +1.92 | 52.81 | 54.30 | +1.48 |
| temporal | 321 -> 295 | 52.96 | 55.93 | +2.97 | 42.04 | 43.35 | +1.32 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-5.6-luna_nemori_merge085_all_k10_ours_expand-off_run1_lunaB`

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 34.68 | 34.68 | +0.00 |
| multi-hop | 282 -> 254 | 71.99 | 75.59 | +3.60 | 43.48 | 45.35 | +1.87 |
| open-domain | 96 -> 87 | 51.04 | 54.02 | +2.98 | 33.51 | 36.19 | +2.68 |
| single-hop | 841 -> 805 | 81.93 | 83.85 | +1.92 | 58.29 | 59.65 | +1.37 |
| temporal | 321 -> 295 | 74.14 | 77.63 | +3.48 | 57.39 | 58.52 | +1.14 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

### `gpt-5.6-luna_nemori_upstream_all_k10_ours_expand-off_run1_lunaA`

| category | n (full -> excl) | J full | J excl | dJ | F1 full | F1 excl | dF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adversarial | 446 -> 446 | -- | -- | -- | 34.59 | 34.59 | +0.00 |
| multi-hop | 282 -> 254 | 73.05 | 75.98 | +2.93 | 43.39 | 45.23 | +1.84 |
| open-domain | 96 -> 87 | 43.75 | 45.98 | +2.23 | 26.10 | 26.86 | +0.76 |
| single-hop | 841 -> 805 | 83.71 | 86.09 | +2.38 | 58.32 | 59.67 | +1.35 |
| temporal | 321 -> 295 | 72.27 | 76.61 | +4.34 | 54.90 | 56.97 | +2.07 |

`--` in a J column means the category carries no judge verdicts (adversarial questions are scored by F1 only), not a score of zero.

