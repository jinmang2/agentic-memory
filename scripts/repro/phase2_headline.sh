#!/usr/bin/env bash
# phase2_headline.sh — Rung 2 HEADLINE: K independent full ingests → mean±std.
# Same production config as phase2.sh (our `ours` metric + Mem0-style J-judge +
# 1-hop link expansion ON), but reported as a mean±std over K=3 INDEPENDENT
# note-graph draws — the same write-path-variance rationale as phase1b_headline.sh.
#
# Reuses the SAME per-seed stores as phase1b_headline.sh
# (results/repro/stores/full_all_seed<SEED>): the notes/links are identical
# regardless of eval-mode/expand/k (all retrieval-/scoring-time), so if
# phase1b_headline.sh already ingested them, this only pays the eval passes. Each
# seed keeps its own five artifacts via --tag-suffix.
# Cost: K × eval (answers + judge on cat1-4) ≈ $2.3 for K=3 when the seed stores
# already exist from phase1b_headline.sh; + K × ~$0.9 ingest if they do not.
# Prereq: repo-root .env.local with OPENAI_API_KEY; embedder downloaded.
set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="results/repro/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$(basename "$0" .sh)_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

K="${K:-3}"                       # number of independent ingests (note-graph draws)
WORKERS="${WORKERS:-8}"           # concurrent QA workers (results identical to 1)
INGEST_WORKERS="${INGEST_WORKERS:-4}"  # concurrent CONVERSATION ingests ≈ in-flight
                                       # API calls (rate-limit knob; identical to
                                       # sequential). RAM ≈ INGEST_WORKERS × ~1GB.

for SEED in $(seq 1 "$K"); do
    echo "=== seed ${SEED}/${K} ==="
    STORE="results/repro/stores/full_all_seed${SEED}"

    # Shared with phase1b_headline.sh — ingest only if this seed store is not
    # already complete. Conversation-parallel (byte-identical to sequential),
    # resumable (skips completed convs, wipes only partial ones), and writes ONE
    # combined sentinel when every conv is done.
    if [ ! -f "$STORE/.ingest_complete.json" ]; then
        uv run python scripts/repro/ingest_parallel.py \
            --convs all --workers "$INGEST_WORKERS" \
            --tag-suffix "_seed${SEED}" \
            --data-dir "$STORE"
    fi

    # Score this seed's store — our-production metric + J-judge + link expansion.
    uv run python scripts/exp_amem_repro.py \
        --conv all \
        --k 10 \
        --eval-mode ours \
        --judge \
        --expand-links on \
        --workers "$WORKERS" \
        --tag-suffix "_seed${SEED}" \
        --data-dir "$STORE" \
        --eval-only
done

# Aggregate the K per-seed summaries into one mean±std headline. The ingest
# summaries are the SAME seed stores as phase1b_headline (shared): campaign_cost
# here counts that shared ingest, so do NOT sum this headline's campaign_cost
# with phase1b's (see ingest_note in the output).
uv run python scripts/repro/aggregate_headline.py \
    --out results/repro/gpt-4o-mini_all_ours_headline.json \
    --ingest-summaries results/repro/gpt-4o-mini_all_ingest_seed*.json \
    -- \
    results/repro/gpt-4o-mini_all_k10_ours_expand-on_run1_seed*.json
