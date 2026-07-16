# 로컬 환경 제약 (2026-07-16 측정)

이 프로젝트의 모든 설계 결정은 아래 하드웨어 제약을 전제로 한다.

## 하드웨어

| 항목 | 값 | 시사점 |
|---|---|---|
| CPU | Intel i7-9750H (WSL2에 4 vCPU 노출) | 병렬 ingestion은 4 worker 이하로 제한 |
| RAM | 7.8 GiB (WSL 할당, 가용 ~2.4 GiB) | **가장 강한 제약.** 서버형 DB(Neo4j JVM 등) 상주 부적합 → embedded 스토리지 필수 |
| GPU | NVIDIA RTX 2060 6 GiB (driver 581.57, WSL CUDA) | 0.5B~1.7B FP16 inference 가능, 0.5B QLoRA 학습 가능, 7B는 4bit inference까지만 |
| Disk | 603 GB 여유 | 데이터셋/체크포인트 충분 |
| OS | WSL2 (Linux 6.6.87.2) | Windows 쪽 메모리와 공유. `.wslconfig`로 RAM 상향 검토 가능 |
| Python | 3.12.3 + uv, miniconda | uv 기반 프로젝트 관리 권장 |

## 파생 설계 원칙

> **핵심 원칙: 방법론은 버리지 않는다.** 환경에 부적합한 구성요소(예: Neo4j, cross-encoder
> rerank, 대형 judge 모델)라도 구현에서 제외하지 않고, **capability detection으로 코드
> 레벨에서 걸러서 선택**되게 한다. 모든 무거운 구성요소는 어댑터 인터페이스 뒤에 두고,
> 현재 머신이 감당 가능한 구현이 자동/수동으로 선택된다.

1. **Capability-gated 아키텍처**: 시작 시 RAM/VRAM/CPU/외부 서비스 가용성을 감지하는
   `capabilities` 모듈을 두고, 각 backend 어댑터는 자신의 요구 스펙을 선언
   (`requires: {ram_gb, vram_gb, service: neo4j, ...}`). 프로파일 리졸버가
   요구 스펙 ↔ 감지된 capability를 매칭해 사용 가능한 구현만 활성화.
   부적합 시 에러가 아니라 **명시적 fallback 로그와 함께 대체 구현으로 강등**.
2. **프로파일 시스템**: `lite`(현재 PC 기본) / `standard` / `full`(서버·클라우드) 프리셋
   + config 오버라이드. 동일 코드베이스가 프로파일만 바꿔 모든 방법론을 실행 가능해야 함.
3. **스토리지 어댑터 이원화**: embedded(SQLite+sqlite-vec / LanceDB / Kuzu)와
   서버형(Neo4j / Qdrant / FalkorDB)을 동일 인터페이스로 모두 구현.
   현재 PC에서는 embedded가 기본 선택되지만, 서버형 코드도 1급 시민으로 유지.
4. **LLM 호출 추상화**: OpenAI-compatible endpoint 하나로 통일
   (로컬 vLLM/llama.cpp/Ollama ↔ 클라우드 API 스위칭 가능하게). 모델 티어
   (extraction/judge/rerank용)를 역할별로 분리 설정.
5. **Embedding/Rerank 티어링**: bge-m3, gte-small, Qwen3-Embedding-0.6B 등 소형 모델을
   기본으로 하되, cross-encoder rerank·대형 embedder도 어댑터로 구현하고 capability에
   따라 활성화.
6. **비동기 write path**: memory 구성(LLM 추출·링크·요약)은 background queue로,
   read path(retrieval)는 저지연 동기로 분리.
7. **재현 실험 기본 티어는 0.5B급**: Qwen3-0.6B / Qwen2.5-0.5B-Instruct 중심,
   judge는 API 모델 또는 로컬 4B(4bit) 병용. 단, 실험 하네스는 모델 크기에 독립적으로
   설계해 서버 환경에서 동일 코드로 대형 모델 재현 가능하게.
