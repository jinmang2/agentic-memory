#!/usr/bin/env bash
# 로컬 LLM 스택 셋업 (멱등 — 이미 있으면 스킵)
#
# 하는 일:
#   1. llama.cpp 프리빌트 CPU 바이너리 다운로드  → ~/.agmem/bin/
#   2. Qwen3-0.6B-Q8_0 GGUF 다운로드            → ~/.agmem/models/
#   3. (GPU + nvcc 있으면) llama.cpp CUDA 빌드   → ~/.agmem/src/llama.cpp/build/
#
# 이후 서버 기동: scripts/serve-llm.sh  (CUDA 빌드가 있으면 자동 선택, :8080)
# 상세 설명: docs/07-local-llm-setup.md
set -euo pipefail

AGMEM="$HOME/.agmem"
LLAMA_TAG="${LLAMA_TAG:-b10037}"   # 검증된 릴리스 태그 (2026-07-16)
MODEL_URL="https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
mkdir -p "$AGMEM/bin" "$AGMEM/models" "$AGMEM/src"

# --- 1. 프리빌트 CPU 바이너리 (fallback용, ~40MB) ---------------------------
if ! find "$AGMEM/bin" -name llama-server -type f | grep -q .; then
  echo "[1/3] downloading llama.cpp prebuilt ($LLAMA_TAG)..."
  curl -sL -o "$AGMEM/bin/llama.tar.gz" \
    "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/llama-$LLAMA_TAG-bin-ubuntu-x64.tar.gz"
  tar xzf "$AGMEM/bin/llama.tar.gz" -C "$AGMEM/bin"
  rm "$AGMEM/bin/llama.tar.gz"
else
  echo "[1/3] prebuilt binary exists — skip"
fi

# --- 2. 모델 (Qwen3-0.6B Q8_0, ~610MB) --------------------------------------
if [[ ! -f "$AGMEM/models/Qwen3-0.6B-Q8_0.gguf" ]]; then
  echo "[2/3] downloading Qwen3-0.6B-Q8_0.gguf (~610MB)..."
  curl -L -o "$AGMEM/models/Qwen3-0.6B-Q8_0.gguf" "$MODEL_URL"
else
  echo "[2/3] model exists — skip"
fi

# --- 3. CUDA 빌드 (선택; RTX 2060 기준 20~60분) ------------------------------
# CPU 추론은 RAM 압박 시 0.3 tok/s까지 떨어지므로 GPU 빌드를 강력 권장 (90x).
if command -v nvcc >/dev/null && command -v nvidia-smi >/dev/null; then
  if [[ ! -x "$AGMEM/src/llama.cpp/build/bin/llama-server" ]]; then
    echo "[3/3] building llama.cpp with CUDA (compute 7.5)..."
    [[ -d "$AGMEM/src/llama.cpp" ]] || \
      git clone --depth 1 https://github.com/ggml-org/llama.cpp "$AGMEM/src/llama.cpp"
    cd "$AGMEM/src/llama.cpp"
    # RTX 2060 = compute 7.5. 다른 GPU면 CMAKE_CUDA_ARCHITECTURES 수정
    # (예: 3090=86, 4090=89). -j2는 RAM 8GB 호스트 기준 — 여유 있으면 올릴 것.
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-75}" -DLLAMA_CURL=OFF
    cmake --build build --target llama-server -j"${BUILD_JOBS:-2}"
  else
    echo "[3/3] CUDA build exists — skip"
  fi
else
  echo "[3/3] nvcc/nvidia-smi not found — CPU binary only (느림 주의)"
fi

echo
echo "done. 서버 기동:  scripts/serve-llm.sh   (health: curl localhost:8080/health)"
