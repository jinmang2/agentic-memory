#!/usr/bin/env bash
# phase2.sh — Rung 2: our PRODUCTION config, full LoCoMo (all 10 convs).
# Isolates: the gap between an upstream-faithful reproduction (rung 1b) and how
# we would actually ship A-Mem — our `ours` metric (SQuAD-style normalize +
# Porter stemming + cat3 gold truncation), the Mem0-style binary J-score judge,
# and 1-hop note-link expansion ON (the A-Mem "linked memories are automatically
# accessed" read behavior). Numbers here are NOT a paper reproduction; they are
# our organizer at full strength.
#
# WRITE-ONCE / READ-SWEEP: reuses the SAME shared store as phase1b.sh
# (results/repro/stores/full_all) — the notes/links are identical regardless of
# eval-mode, expand-links, or k (all are retrieval-/scoring-time), so the write
# path is never re-paid. Ingest is guarded (skipped if the store exists); run
# phase1b.sh first, or this script ingests it once. --expand-links on and the
# ours metric + J-judge apply purely at --eval-only time.
# Cost: ~$0.9 ingest (once, shared) + answers + judge on cat1-4 ≈ $3.2 total, or
# just ~$2.3 eval when phase1b.sh already ingested the shared store.
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

STORE="results/repro/stores/full_all"

# 1) Ingest ONCE (shared with phase1b.sh) — skip if already persisted.
if [ ! -d "$STORE" ]; then
    uv run python scripts/exp_amem_repro.py \
        --conv all --eval-mode wujiang \
        --data-dir "$STORE" --ingest-only
fi

# 2) Score the persisted store — our-production metric + J-judge + link expansion.
uv run python scripts/exp_amem_repro.py \
    --conv all \
    --k 10 \
    --eval-mode ours \
    --judge \
    --expand-links on \
    --data-dir "$STORE" \
    --eval-only
