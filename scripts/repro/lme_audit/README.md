# LongMemEval audit scripts — every measurement in `docs/research/longmemeval.md`

All eleven scripts spend **$0**: no model call, no network except where noted. They exist so the
numbers in the research doc can be re-derived rather than trusted, which is the same standard
`scripts/repro/defects/` holds upstream claims to.

Unlike `scripts/repro/defects/`, these are **not** run by CI, because they need two local inputs CI
does not have:

- `~/.agmem/datasets/longmemeval_s_cleaned.json` and `longmemeval_oracle.json`
  (`m_stats.py` additionally needs `longmemeval_m_cleaned.json`, 2.55 GiB)
- `~/.agmem/upstream/longmemeval` @ `9e0b455f4ef0e2ab8f2e582289761153549043fc`

Run them from the repo root (`uv run python scripts/repro/lme_audit/<name>.py`).

| script | answers | needs |
|---|---|---|
| `lme_stats.py` | port invariants (500/6/30), scale, evidence position, head-vs-tail truncation loss | dataset |
| `lme_tokens.py` | real chars/token and whether upstream's 126,200 cap binds | dataset + `tiktoken` |
| `oracle_stats.py` | oracle scale, and how many instances ship out of date order | dataset + `tiktoken` |
| `cleaning_diff.py` | what the 2025-09-19 cleaning changed in `_s` | dataset + the **deprecated** release (download noted in the file) |
| `prompt_rediff.py` | our full-context prompt vs a literal transcription of upstream's `prepare_prompt` | dataset + `tiktoken` |
| `rediff.py` | our `aggregate` vs the official `print_qa_metrics.py`, run as a subprocess | upstream clone + `numpy` |
| `retrieval_gold_audit.py` | what user-only indexing drops from the official retrieval metrics | dataset |
| `abs_evidence.py` | whether abstention questions carry evidence sessions | dataset |
| `dup_sessions.py` | independent check of upstream issue #54 — duplicate `haystack_session_id` | dataset |
| `gold_issue_check.py` | independent check of upstream issues #41 and #22 — gold defects | dataset |
| `m_stats.py` | `_m` scale, whether the streamed instances are the array's real elements, whether upstream's cap binds | `_m` + `_s` (+ `tiktoken` for `LME_EXACT_TOKENS=1`) |

Three of them pull an ephemeral dependency rather than adding one to the project: run
`lme_tokens.py`, `oracle_stats.py` and `prompt_rediff.py` with `uv run --with tiktoken python ...`,
and `rediff.py` with `uv run --with numpy python ...`.

`cleaning_diff.py` additionally needs the withdrawn release, which is not kept in `~/.agmem/datasets`
because nothing should ever run on it by accident:

```bash
curl -sL -o /tmp/longmemeval_s_orig.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s
LME_S_ORIG=/tmp/longmemeval_s_orig.json uv run python scripts/repro/lme_audit/cleaning_diff.py
```
