#!/usr/bin/env bash
# phase1b_headline.sh — Rung 1b HEADLINE: K independent full ingests → mean±std.
# Isolates the SAME gap as phase1b.sh (our re-implementation vs upstream A-Mem,
# WujiangXu-faithful eval), but reports a mean±std over K=3 INDEPENDENT note-graph
# draws instead of a single one. Rationale: the write path runs at temperature
# 0.7, so the note graph — the dominant variance source (~3-6 F1 run-to-run) — is
# a random draw. A lone number hides that; --runs only repeats the ANSWER path and
# cannot see write-path variance. So each seed gets its OWN fresh ingest.
#
# Each seed: fresh store + full ingest + full WujiangXu eval (--workers). Then
# aggregate_headline.py combines the K per-seed summaries into per-category +
# overall F1 mean±std. All five artifacts are kept PER SEED (distinct --tag-suffix),
# so nothing is overwritten and any seed can be re-scored offline for free.
# Cost: K × (~$0.9 ingest + ~$0.7 eval) ≈ $4.8 for K=3 on gpt-4o-mini.
# Prereq: repo-root .env.local with OPENAI_API_KEY; embedder downloaded.
set -euo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="results/repro/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$(basename "$0" .sh)_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

K="${K:-3}"                 # number of independent ingests (note-graph draws)
WORKERS="${WORKERS:-8}"     # concurrent QA workers (results identical to 1)

for SEED in $(seq 1 "$K"); do
    echo "=== seed ${SEED}/${K} ==="
    STORE="results/repro/stores/full_all_seed${SEED}"

    # Fresh, independent ingest per seed (guarded on the completion sentinel;
    # a partial/crashed store is wiped so the re-ingest is clean).
    if [ ! -f "$STORE/.ingest_complete.json" ]; then
        rm -rf "$STORE"
        uv run python scripts/exp_amem_repro.py \
            --conv all --eval-mode wujiang \
            --tag-suffix "_seed${SEED}" \
            --data-dir "$STORE" --ingest-only
    fi

    # Score this seed's store — WujiangXu-faithful metric, no re-ingest.
    uv run python scripts/exp_amem_repro.py \
        --conv all \
        --k 10 \
        --eval-mode wujiang \
        --expand-links off \
        --workers "$WORKERS" \
        --tag-suffix "_seed${SEED}" \
        --data-dir "$STORE" \
        --eval-only
done

# Aggregate the K per-seed summaries into one mean±std headline. Pass the ingest
# summaries too so the headline reports ingest_cost_usd + campaign_cost_usd, not
# just the eval/re-score spend.
uv run python scripts/repro/aggregate_headline.py \
    --out results/repro/gpt-4o-mini_all_wujiang_headline.json \
    --ingest-summaries results/repro/gpt-4o-mini_all_ingest_seed*.json \
    -- \
    results/repro/gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed*.json
