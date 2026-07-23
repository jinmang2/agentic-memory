#!/usr/bin/env bash
# phase3_ksweep.sh — Rung 3 (k sensitivity): sweep retrieval k over
# {10,20,30,40,50} on full LoCoMo, WujiangXu-faithful eval.
# Isolates: how much QA k alone moves F1 (the A-Mem paper Table 8 tunes k per
# category 10..50 on the eval set; we report a flat sweep instead of tuning).
# Write-once/read-sweep: INGEST the A-Mem notes exactly ONCE into a persistent
# store (--ingest-only --data-dir), then reload that store for each k with
# --eval-only. This avoids re-paying the (expensive) write-path evolution calls
# five times — only the answer calls repeat per k.
# Cost: 1x write (~$0.9) + 5x answer (~$0.9 each) ≈ $5.4 on gpt-4o-mini.
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

DATA_DIR="results/repro/ksweep_store"

# 1) Ingest once — builds + persists the A-Mem notes/links for all 10 convs.
uv run python scripts/exp_amem_repro.py \
    --conv all \
    --eval-mode wujiang \
    --data-dir "$DATA_DIR" \
    --ingest-only

# 2) Reload the persisted store and answer at each k (no re-ingest).
for K in 10 20 30 40 50; do
    echo "=== k=$K ==="
    uv run python scripts/exp_amem_repro.py \
        --conv all \
        --k "$K" \
        --eval-mode wujiang \
        --expand-links off \
        --data-dir "$DATA_DIR" \
        --eval-only
done
