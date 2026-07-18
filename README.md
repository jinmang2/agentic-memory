# Agentic Memory Study

MemoryOS · Zep · Nemori · A-Mem · ACE · ReasoningBank · G-Memory 7개 방법론을 직접 구현하고,
LongMemEval/LoCoMo로 재현 평가하며, MCP로 배포

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
| [docs/12-code-conventions.md](docs/12-code-conventions.md) | 코드 컨벤션 (이름/docstring/구조) — 리뷰 게이트 기준 |
| docs/research/ | 논문·공식 코드 원자료 리서치 노트 (시스템별) |

## Quickstart

```bash
uv sync                          # embed 그룹 포함 기본 설치
scripts/setup-local-llm.sh       # llama.cpp + Qwen3-0.6B (~/.agmem, 멱등)
scripts/serve-llm.sh &           # 로컬 LLM 데몬 :8080
uv run pytest tests/ -q          # 126 tests

# 라이브러리
uv run python -c "
from agmem import AgenticMemory
mem = AgenticMemory(organizers=['nemori','reasoning_bank'])
mem.add_message('파리 여행 예산은 300만원')
print(mem.search('여행 예산').render(400))"

# MCP 서버 (Claude Code 등록은 docs/05 §2.3)
uv run agmem-mcp --namespace main --organizers nemori,reasoning_bank

# LoCoMo 재현 실험
uv run python scripts/exp_locomo_conv0.py --configs passthrough amem nemori memoryos
```

## 설계 한 줄 요약

- **Write-path가 방법론의 전부**: retrieval은 공통 인프라(hybrid recall→RRF→rerank), 방법론 차이는 `Organizer` 플러그인으로 캡슐화.
- **모든 메모리 변경은 append-only `EvolutionLog` 연산**(ADD/UPDATE/MERGE/DELETE/INVALIDATE/LINK) — ACE delta, G-Memory rule ops, Zep invalidation을 하나의 추상화로.
- **Capability-gated**: Neo4j·cross-encoder 등 무거운 구성요소도 전부 구현하되 감지된 하드웨어에 따라 자동 선택/강등.
- **원문 episode 불변 보존** — 그래프/요약 추상화의 verbatim 손실(Zep 실증)에 대한 방어.
- **Organizer 간 chaining**: 한 organizer의 산출물을 다른 organizer가 `consumes` 구독으로 받는 이벤트 훅(`on_memory_event`)과, 명시적 `mem.consolidate()`로만 도는 유예 배치 훅(`consolidate`)을 계약으로 분리 — Nemori(방출)→MemoryOS/A-Mem(`input="episodes"`) 조합으로 검증 (docs/04 §2–3).
- **Nemori fidelity 스위치**: `NemoriOrganizer(fidelity="v1"|"v4"|"upstream")`으로 논문 v1/v4/upstream 코드/우리 mixing을 같은 구현에서 재현 (docs/11).
