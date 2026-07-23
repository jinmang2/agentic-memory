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

uv run python scripts/exp_amem_repro.py \
    --conv all \
    --k 10 \
    --eval-mode ours \
    --judge \
    --expand-links on
