#!/usr/bin/env bash
# phase1b.sh — Rung 1b: our A-Mem re-implementation @ upstream-aligned config,
# full LoCoMo (all 10 conversations, 1,986 QA), WujiangXu-faithful eval.
# Isolates: the re-implementation gap — how close OUR organizer gets to the
# upstream A-Mem numbers when the config is matched (MiniLM embedder, notes-only
# flat cosine retrieval, k=10, set-based F1, cat5 MCQ). Compare against rung 1a
# (scripts/repro/phase1a_upstream.sh) which runs upstream's OWN code, and rung 0
# (the paper's published A-Mem Table 1). Link expansion OFF here to match the
# plain (non-robust) upstream read path.
#
# WRITE-ONCE / READ-SWEEP: ingest all 10 convs ONCE into a shared persistent
# store (guarded — skipped if it already exists, e.g. from a prior run or from
# phase2.sh which reuses the SAME store), then score with --eval-only. The notes
# are identical regardless of eval-mode/expand/k, so the (paid) write path is
# spent once and re-scoring is free. Delete the store dir to force a fresh ingest.
# Cost: ~$0.9 ingest (once) + ~$0.7 answers ≈ $1.6 on gpt-4o-mini (skips ingest
# if the shared store already exists).
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
WORKERS="${WORKERS:-8}"   # concurrent QA workers (results identical to 1)

# 1) Ingest ONCE (shared with phase2.sh) — skip only if a COMPLETE ingest is
# proven by the sentinel. A bare/partial store dir (e.g. a crashed 10-conv
# ingest) is wiped and re-ingested clean, so eval never micro-averages over a
# truncated store.
if [ ! -f "$STORE/.ingest_complete.json" ]; then
    rm -rf "$STORE"
    uv run python scripts/exp_amem_repro.py \
        --conv all --eval-mode wujiang \
        --data-dir "$STORE" --ingest-only
fi

# 2) Score the persisted store — WujiangXu-faithful metric, no re-ingest.
uv run python scripts/exp_amem_repro.py \
    --conv all \
    --k 10 \
    --eval-mode wujiang \
    --expand-links off \
    --workers "$WORKERS" \
    --data-dir "$STORE" \
    --eval-only
