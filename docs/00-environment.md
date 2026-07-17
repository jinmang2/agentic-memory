# 로컬 환경 정보 (2026-07-16 측정; 2026-07-17 지위 격하)

> **2026-07-17 정책 교체 (사용자 지시)**: 하드웨어 제약은 **설계의 전제가 아니라
> 런타임 참고 정보**다. 어떤 구성요소도 "이 PC에서 무겁다"는 이유로 설계·구현에서
> 제외하거나 naive 구현으로 대체하지 않는다 (naive in-python 런타임 기본값 금지 —
> docs/03 §5.2). 제약 대응은 전적으로 capability detection의 **런타임 해석**
> (감지→강등 로그)에 맡기고, 코드베이스에는 실 스택을 전부 배선한다. 원 시스템의
> 엔진이 과중하면 같은 계열의 **경량 실물**(임베디드/local 모드)로 대체한다 —
> 예: Qdrant 서버→qdrant-client local, PostgreSQL 서버→pgserver(임베디드),
> Neo4j→Kuzu. 0.6B 로컬 측정은 하드웨어로 인해 중단됐고, 측정은 최저가 API로
> 전환한다(콜 수·토큰 역산 후) — 즉 **측정 병목도 더 이상 설계 제약이 아니다**.

## 하드웨어 (참고용 스냅샷)

| 항목 | 값 | 런타임 시사점 (설계 제약 아님) |
|---|---|---|
| CPU | Intel i7-9750H (WSL2에 4 vCPU 노출) | 병렬 ingestion worker 수 기본값 산정에만 사용 |
| RAM | 7.8 GiB (WSL 할당, 가용 ~2.4 GiB) | JVM 상주형(Neo4j 등)은 capability 감지로 미검출 시 강등될 수 있음 — 코드에서는 1급 유지 |
| GPU | NVIDIA RTX 2060 6 GiB (driver 581.57, WSL CUDA) | 로컬 inference 티어 참고치 (측정은 API 전환 예정) |
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
3. **스토리지 어댑터 이원화**: embedded(SQLite+sqlite-vec / LanceDB / Qdrant local /
   Chroma persistent / Kuzu / pgserver)와 서버형(Neo4j / Qdrant / FalkorDB /
   PostgreSQL)을 동일 인터페이스로 **모두 실물로 구현**. naive in-python 대체는
   런타임 기본값 금지(docs/03 §5.2). embedded/서버형 선택은 런타임 capability가
   결정하며, 어느 쪽도 코드에서 2급이 아니다.
4. **LLM 호출 추상화**: OpenAI-compatible endpoint 하나로 통일
   (로컬 vLLM/llama.cpp/Ollama ↔ 클라우드 API 스위칭 가능하게). 모델 티어
   (extraction/judge/rerank용)를 역할별로 분리 설정.
5. **Embedding/Rerank 티어링**: bge-m3, gte-small, Qwen3-Embedding-0.6B 등 소형 모델을
   기본으로 하되, cross-encoder rerank·대형 embedder도 어댑터로 구현하고 capability에
   따라 활성화.
6. **비동기 write path**: memory 구성(LLM 추출·링크·요약)은 background queue로,
   read path(retrieval)는 저지연 동기로 분리.
7. **재현 실험 티어 (2026-07-17 개정)**: 0.5B급 로컬 측정은 품질·속도 문제로 중단.
   측정은 **최저가 API 모델**로 전환하되, 전환 전에 방법론별 LLM 콜 수·토큰량을
   역산해 비용 상한을 추정하고 지표(F1/BLEU vs LLM judge) 검증을 선행한다.
   실험 하네스는 모델 크기·엔드포인트에 독립적(역할별 RoleConfig)이므로 코드 변경
   없이 전환 가능. 로컬 0.6B 서빙(scripts/serve-llm.sh)은 개발용 스모크로만 유지.
