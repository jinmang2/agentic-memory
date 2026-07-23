#!/usr/bin/env bash
# phase1b.sh — Rung 1b: our A-Mem re-implementation @ upstream-aligned config,
# full LoCoMo (all 10 conversations, 1,986 QA), WujiangXu-faithful eval.
# Isolates: the re-implementation gap — how close OUR organizer gets to the
# upstream A-Mem numbers when the config is matched (MiniLM embedder, notes-only
# flat cosine retrieval, k=10, set-based F1, cat5 MCQ). Compare against rung 1a
# (scripts/repro/phase1a_upstream.sh) which runs upstream's OWN code, and rung 0
# (the paper's published A-Mem Table 1). Link expansion OFF here to match the
# plain (non-robust) upstream read path.
# Cost: ~$1.6 on gpt-4o-mini (1,986 QA, write + answer calls).
# Prereq: repo-root .env.local with OPENAI_API_KEY; embedder downloaded.
set -euo pipefail
cd "$(dirname "$0")/../.."

uv run python scripts/exp_amem_repro.py \
    --conv all \
    --k 10 \
    --eval-mode wujiang \
    --expand-links off
