#!/usr/bin/env bash
# phase3_ablation.sh — Rung 3 (evolution ablation): A-Mem Full vs w/o-evolution.
# Isolates: the contribution of Memory Evolution (Ps3) — the paper's Table 3
# ablation shows evolution's biggest lift is on Temporal (+14.6 F1). Full LoCoMo,
# WujiangXu-faithful eval.
#
# 2026-08-19: the evolution-off switch EXISTS — `AMemOrganizer(evolve=False)`
# (src/agmem/organizers/amem/organizer.py), reachable as `--config amem_noevolve`.
# The switch is ADD-only, not the "ADD + LINK" this banner once sketched: upstream
# decides links INSIDE the evolve call (the "strengthen" action — there is no
# link-only prompt to keep), so skipping evolution skips links with it, which is
# exactly what Table 3's w/o-evolution ablation removes.
#
# 2026-08-20: both arms below now run. The conv0 pilot survived and lifted
# run_ready (configs.py), and the 10-conv arm landed at F1 33.80 / BLEU-1 28.58
# for $0.7043 against the evolution-on seeds' 34.84 / 31.86 for $2.752 — half the
# write calls, a quarter of the cost, an F1 delta inside the seed spread and a
# BLEU-1 delta outside it (docs/14 §rung-3 for the full table and its caveat).
# Cost: Full arm ~$2.75 (ingest $2.32 + eval $0.43), ablation arm ~$0.70.
# Prereq: repo-root .env.local with OPENAI_API_KEY; embedder downloaded.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Durable, in-repo run log: tee all output to results/repro/logs/ (git-tracked,
# see .gitignore un-ignore) so nothing is lost to an ephemeral scratchpad. The
# exec redirect keeps set -euo pipefail intact (tee runs async, never masking a
# command's exit status).
LOG_DIR="results/repro/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$(basename "$0" .sh)_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

WORKERS="${WORKERS:-8}"   # concurrent QA workers (results identical to 1)

# Full A-Mem (link generation + evolution) — this arm works today.
uv run python scripts/exp_amem_repro.py \
    --conv all \
    --k 10 \
    --eval-mode wujiang \
    --expand-links off \
    --workers "$WORKERS"

# w/o-evolution arm — switch implemented 2026-08-19, gate lifted by the conv0
# pilot 2026-08-20. `--max-spend-usd` caps BETWEEN conversations (the runner has
# no mid-conversation cut point that leaves a resumable store).
uv run python scripts/exp_amem_repro.py \
    --conv all \
    --k 10 \
    --eval-mode wujiang \
    --expand-links off \
    --workers "$WORKERS" \
    --config amem_noevolve \
    --max-spend-usd 2.5
