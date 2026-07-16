# 로컬 LLM 스택 셋업 기록 (2026-07-16 실측)

Phase 0에서 수행한 로컬 0.6B 서빙 셋업의 전체 기록. **재수행은 스크립트 두 개면 끝:**

```bash
scripts/setup-local-llm.sh   # 다운로드 + (GPU면) CUDA 빌드 — 멱등
scripts/serve-llm.sh         # 데몬 기동 (:8080, OpenAI-compatible)
```

## 구성 요소와 위치

| 항목 | 위치 | 비고 |
|---|---|---|
| llama.cpp 프리빌트(CPU) | `~/.agmem/bin/llama-b10037/` | 릴리스 b10037, fallback용 |
| llama.cpp CUDA 빌드 | `~/.agmem/src/llama.cpp/build/bin/llama-server` | compute 7.5 전용 컴파일 |
| 모델 | `~/.agmem/models/Qwen3-0.6B-Q8_0.gguf` | 610MB, HF `Qwen/Qwen3-0.6B-GGUF` |
| 서버 로그 | `~/.agmem/llm-server.log` | serve-llm.sh 기동 시 |

## 왜 CUDA 직접 빌드인가

- 릴리스 b10037에는 **Ubuntu CUDA 프리빌트가 없음** (cpu/vulkan/rocm/sycl만).
- WSL2 Vulkan(Dozen)은 신뢰성이 낮아 제외.
- WSL에 CUDA 12.0 toolkit(`/usr/bin/nvcc`)과 `/usr/lib/wsl/lib/libcuda.so`가 이미 있어
  소스 빌드가 가능했음. `-DCMAKE_CUDA_ARCHITECTURES=75`(RTX 2060 전용)로 한정해
  컴파일 시간을 단축 (전체 아키텍처 빌드 대비 수 배 빠름). 4코어 -j2로 약 55분.

## 실측 성능 (Qwen3-0.6B-Q8_0, structured JSON 추출 태스크)

| 백엔드 | 조건 | 속도 |
|---|---|---|
| CPU (3 threads) | RAM 압박(가용 1.4GB, 스왑 스래싱) | **0.3~0.5 tok/s, 143s/call** — 사용 불가 |
| CUDA (전 레이어 오프로드) | VRAM ~1.5GB 사용 | **1.6s/call** (3/3 유효 JSON, drop 0) |

교훈:
1. **이 호스트에서 CPU 추론은 비상용으로도 위험** — WSL RAM 7.8GB에 pylance/노드류가
   4~5GB를 상시 점유해 mmap된 모델이 스왑으로 밀림. GPU 상주가 사실상 필수.
2. Qwen3는 기본 thinking 모드가 켜져 있어 추출 태스크에서 토큰 낭비 —
   요청에 `extra_body={"chat_template_kwargs": {"enable_thinking": false}}` 필수
   (agmem `RoleConfig.extra_body`로 지정, `agmem.example.toml` 참고).
3. guided_json(vLLM 전용)은 llama.cpp가 무시하므로 agmem의 파싱 방어층(재시도→drop)이
   실질 방어선. 0.6B에서 3/3 통과 확인.

## serve-llm.sh 동작

- CUDA 빌드가 있으면 자동 선택(`--n-gpu-layers 99`), 없으면 프리빌트 CPU.
- `--threads 3`(4 vCPU 중 1개 여유), `--ctx-size 8192`, `--parallel 2`,
  `--alias qwen3-0.6b`, `--jinja`(Qwen3 chat template).
- 백그라운드 기동 예: `setsid nohup scripts/serve-llm.sh > ~/.agmem/llm-server.log 2>&1 &`

## 다른 GPU/환경에서 재수행 시

- `CUDA_ARCH=86 scripts/setup-local-llm.sh` (3090=86, 4090=89, A100=80)
- RAM 여유가 있으면 `BUILD_JOBS=8`로 빌드 가속.
- nvcc가 없으면: `sudo apt install nvidia-cuda-toolkit` (WSL은 Windows 드라이버만
  최신이면 toolkit은 apt 버전으로 충분).
- 더 큰 모델: Qwen3-4B-AWQ급은 vLLM 전환 검토 (guided_json 네이티브 지원 이점).
