#!/usr/bin/env bash
# smoke.sh — cheapest end-to-end check of the reproduction harness.
# Isolates: does the full ingest->QA->score loop run against the real OpenAI
# endpoint on ONE conversation (conv0), for BOTH eval modes?
# Config: our A-Mem organizer @ upstream-aligned (all-MiniLM-L6-v2, notes-only
# flat cosine, k=10, write/gen 0.7, cat5 0.5).
# Cost: ~$0.15 (wujiang) + ~$0.20 (ours+judge) ≈ $0.35 on gpt-4o-mini.
# Prereq: repo-root .env.local with OPENAI_API_KEY (gitignored).
set -euo pipefail
cd "$(dirname "$0")/../.."

# Rung 1b flavor on a single conversation: WujiangXu-faithful metric.
uv run python scripts/exp_amem_repro.py --conv 0 --k 10 --eval-mode wujiang

# Our-production flavor on a single conversation: ours metric + J-score judge.
uv run python scripts/exp_amem_repro.py --conv 0 --k 10 --eval-mode ours --judge
