#!/usr/bin/env bash
# 로컬 LLM 데몬 (llama.cpp, OpenAI-compatible :8080/v1)
# 사용: scripts/serve-llm.sh [모델경로] [포트]
set -euo pipefail

MODEL="${1:-$HOME/.agmem/models/Qwen3-0.6B-Q8_0.gguf}"
PORT="${2:-8080}"

# CUDA 빌드(직접 컴파일) 우선, 없으면 프리빌트 CPU 바이너리로 fallback
CUDA_BIN="$HOME/.agmem/src/llama.cpp/build/bin/llama-server"
CPU_BIN="$(find "$HOME/.agmem/bin" -name llama-server -type f 2>/dev/null | head -1)"
if [[ -x "$CUDA_BIN" ]]; then
  BIN="$CUDA_BIN"
  GPU_ARGS=(--n-gpu-layers 99)   # 0.6B는 전 레이어 GPU 상주 (~1GB VRAM)
else
  BIN="$CPU_BIN"
  GPU_ARGS=()
fi

if [[ -z "${BIN:-}" ]]; then
  echo "llama-server binary not found" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL" >&2
  exit 1
fi

# 4 vCPU 호스트 기준: 3 threads (1개는 시스템/워커 여유분)
exec "$BIN" \
  --model "$MODEL" \
  --port "$PORT" \
  --threads 3 \
  --ctx-size 8192 \
  --parallel 2 \
  --alias qwen3-0.6b \
  --jinja \
  "${GPU_ARGS[@]}"
