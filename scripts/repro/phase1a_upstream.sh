#!/usr/bin/env bash
# phase1a_upstream.sh — Rung 1a: run the UPSTREAM A-Mem code itself.
# Isolates: the "can upstream's own code reproduce the paper on gpt-4o-mini"
# baseline — the anchor rung 1b (our re-implementation) is measured against.
#
# !! THIS RUNS EXTERNAL CODE (WujiangXu/AgenticMemory), NOT ours. It is a
#    DOCUMENTED WRAPPER — it does NOT execute here. Read and run it yourself in a
#    separate virtualenv with the upstream repo's own dependencies. !!
#
# Prerequisites (do these manually):
#   1. Clone the upstream repro repo:
#        git clone https://github.com/WujiangXu/AgenticMemory
#        cd AgenticMemory
#   2. Create an isolated env and install THEIR deps (heavy: torch,
#      sentence-transformers, rouge_score, nltk, bert_score, openai, tqdm):
#        python -m venv .venv && . .venv/bin/activate
#        pip install -r requirements.txt
#   3. Provide the OpenAI key THEIR way (their code reads OPENAI_API_KEY from the
#      environment / their own config — NOT our .env.local):
#        export OPENAI_API_KEY=sk-...        # never commit this
#   4. Ensure the dataset is at data/locomo10.json (ships in their repo).
#
# Exact command (openai backend, k=10, gpt-4o-mini, cat5 temp 0.5 default):
#
#     python test_advanced.py \
#         --dataset data/locomo10.json \
#         --model gpt-4o-mini \
#         --backend openai \
#         --retrieve_k 10 \
#         --temperature_c5 0.5 \
#         --ratio 1.0 \
#         --output results_upstream_gpt4omini.json
#
# Notes on faithfulness:
#   - test_advanced.py uses memory_layer.py (plain). The PAPER numbers come from
#     the *_robust.py path (memory_layer_robust.py + test_advanced_robust.py);
#     see docs/13 §3 and docs/14. For the paper-faithful rung, run:
#         python test_advanced_robust.py --model gpt-4o-mini --backend openai \
#             --retrieve_k 10 --temperature_c5 0.5 --output results_upstream_robust.json
#   - Scoring is utils.calculate_metrics (set-based token F1 + ROUGE/BLEU/BERT).
#     Our --eval-mode wujiang mirrors ONLY its set-based token F1.
#   - Cost on gpt-4o-mini for the full 1,986-QA run: ~$1.5–2.0.
echo "This is a documentation-only wrapper. Read the comments and run upstream" \
     "test_advanced.py in its own environment. Nothing is executed here."
exit 0
