#!/usr/bin/env bash
# smoke.sh — cheapest end-to-end check of the reproduction harness.
# Isolates: does the full ingest->QA->score loop run against the real OpenAI
# endpoint on ONE conversation (conv0), for BOTH eval modes?
# Config: our A-Mem organizer @ upstream-aligned (all-MiniLM-L6-v2, notes-only
# flat cosine, k=10, write/gen 0.7, cat5 0.5).
#
# WRITE-ONCE / READ-SWEEP (the re-spend fix): INGEST conv0 exactly ONCE into a
# persistent store (--ingest-only --data-dir), then run TWO eval passes over
# that SAME store with --eval-only — wujiang, then ours+judge. The (paid) write
# path (note extraction + evolution) is spent once; only the cheap answer calls
# repeat per eval mode.
# Cost: ~$0.15 ingest + 2× cheap eval ≈ $0.35 on gpt-4o-mini.
# Prereq: repo-root .env.local with OPENAI_API_KEY (gitignored).
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

STORE="results/repro/stores/smoke_conv0"

# 1) Ingest conv0 ONCE — builds + persists the A-Mem notes/links.
uv run python scripts/exp_amem_repro.py \
    --conv 0 \
    --eval-mode wujiang \
    --data-dir "$STORE" \
    --ingest-only

# 2a) Reload the persisted store — WujiangXu-faithful metric (no re-ingest).
uv run python scripts/exp_amem_repro.py \
    --conv 0 --k 10 --eval-mode wujiang \
    --data-dir "$STORE" --eval-only

# 2b) Reload the SAME store — our-production metric + J-score judge.
uv run python scripts/exp_amem_repro.py \
    --conv 0 --k 10 --eval-mode ours --judge \
    --data-dir "$STORE" --eval-only
