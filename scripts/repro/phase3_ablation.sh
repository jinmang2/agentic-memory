#!/usr/bin/env bash
# phase3_ablation.sh — Rung 3 (evolution ablation): A-Mem Full vs w/o-evolution.
# Isolates: the contribution of Memory Evolution (Ps3) — the paper's Table 3
# ablation shows evolution's biggest lift is on Temporal (+14.6 F1). Full LoCoMo,
# WujiangXu-faithful eval.
#
# !! FOLLOW-UP REQUIRED — no evolution-off switch exists yet. !!
# Our AMemOrganizer (src/agmem/organizers/amem.py) gates evolution on the LLM's
# per-note `should_evolve` verdict (amem.py:209), NOT on a constructor flag.
# There is currently NO `AMemOrganizer(evolve=False)` (or equivalent) to force
# link-generation-only. To run this ablation faithfully, first add such a switch
# (e.g. skip the EVOLVE_PROMPT call and emit only ADD + LINK ops when
# evolution is disabled), then wire a `--no-evolution` flag into
# scripts/exp_amem_repro.py. Until then this script only runs the Full arm and
# documents the gap.
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

# w/o-evolution arm — BLOCKED until an evolution-off switch is implemented.
# echo "TODO: add AMemOrganizer(evolve=False) + --no-evolution flag, then:"
# uv run python scripts/exp_amem_repro.py --conv all --k 10 \
#     --eval-mode wujiang --expand-links off --no-evolution
