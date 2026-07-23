#!/usr/bin/env bash
# phase2.sh — Rung 2: our PRODUCTION config, full LoCoMo (all 10 convs).
# Isolates: the gap between an upstream-faithful reproduction (rung 1b) and how
# we would actually ship A-Mem — our `ours` metric (SQuAD-style normalize +
# Porter stemming + cat3 gold truncation), the Mem0-style binary J-score judge,
# and 1-hop note-link expansion ON (the A-Mem "linked memories are automatically
# accessed" read behavior). Numbers here are NOT a paper reproduction; they are
# our organizer at full strength.
# Cost: ~$1.6 (answers) + judge calls on cat1-4 ≈ $1.6 more ≈ $3.2 on gpt-4o-mini.
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

uv run python scripts/exp_amem_repro.py \
    --conv all \
    --k 10 \
    --eval-mode ours \
    --judge \
    --expand-links on
