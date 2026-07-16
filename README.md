# Agentic Memory Study

MemoryOS · Zep · Nemori · A-Mem · ACE · ReasoningBank · G-Memory 7개 방법론을 직접 구현하고,
LongMemEval/LoCoMo로 재현 평가하며, MCP로 배포하고, 0.5B급 소형 모델로 PC 재현 한계를 탐구하는 프로젝트.

## 문서 인덱스

| 문서 | 내용 |
|---|---|
| [docs/00-environment.md](docs/00-environment.md) | 로컬 하드웨어 제약 + 파생 설계 원칙 |
| [docs/01-capability-system.md](docs/01-capability-system.md) | capability detection / profile(lite·standard·full) 스펙 — "방법론은 버리지 않고 코드로 거른다" |
| [docs/02-survey-comparison.md](docs/02-survey-comparison.md) | 8개 시스템 서베이·메커니즘·벤치마크·재현성 비교 |
| [docs/03-spec-considerations.md](docs/03-spec-considerations.md) | retrieval/rerank·graph·latency·cold-start·storage·소형모델 대응 상세 스펙 |
| [docs/04-architecture.md](docs/04-architecture.md) | 모듈 구조 (Organizer 플러그인 + MemoryOp/EvolutionLog 추상화) |
| [docs/05-api-design.md](docs/05-api-design.md) | Python API + MCP 도구 설계 + 벤치 CLI |
| [docs/06-roadmap.md](docs/06-roadmap.md) | Phase 0–5 개발 계획, 리스크, 즉시 다음 액션 |
| docs/research/ | 논문·공식 코드 원자료 리서치 노트 (시스템별) |

## 설계 한 줄 요약

- **Write-path가 방법론의 전부**: retrieval은 공통 인프라(hybrid recall→RRF→rerank), 방법론 차이는 `Organizer` 플러그인으로 캡슐화.
- **모든 메모리 변경은 append-only `EvolutionLog` 연산**(ADD/UPDATE/MERGE/DELETE/INVALIDATE/LINK) — ACE delta, G-Memory rule ops, Zep invalidation을 하나의 추상화로.
- **Capability-gated**: Neo4j·cross-encoder 등 무거운 구성요소도 전부 구현하되 감지된 하드웨어에 따라 자동 선택/강등.
- **원문 episode 불변 보존** — 그래프/요약 추상화의 verbatim 손실(Zep 실증)에 대한 방어.
