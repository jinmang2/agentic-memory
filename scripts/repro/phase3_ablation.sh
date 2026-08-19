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
# exactly what Table 3's w/o-evolution ablation removes. The arm is gated
# run_ready=False until a one-conv pilot survives (`--allow-unverified-config`).
# Cost (Full arm only): ~$1.6 on gpt-4o-mini. With the ablation arm added: ~$3.2.
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

# w/o-evolution arm — switch implemented 2026-08-19; still spend-gated (approval +
# a one-conv pilot to lift run_ready=False). When approved:
# uv run python scripts/exp_amem_repro.py --conv all --k 10 \
#     --eval-mode wujiang --expand-links off \
#     --config amem_noevolve --allow-unverified-config
