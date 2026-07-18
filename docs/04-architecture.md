# 모듈 구조 설계

> 핵심 아이디어: **retrieval은 공통 인프라, 방법론 차이는 write-path의 `Organizer`로 캡슐화,
> 모든 메모리 변경은 append-only `evolution_log` 연산으로 표현.**

## 1. 패키지 레이아웃

```
agentic_memory/                      # 패키지명: agmem (pip install agmem)
├── pyproject.toml                   # uv 관리, dependency-groups: dev/embed/backends (+train extra)
├── src/agmem/
│   ├── capabilities/                # §01 문서의 capability detection
│   │   ├── detect.py                # HostCapabilities 감지 (+캐시 ~/.agmem)
│   │   ├── requires.py              # Requires 선언, CapabilityWarning
│   │   └── resolver.py              # override > profile > capability 매칭 → 구현 선택
│   │
│   ├── core/                        # 방법론 독립적 도메인 모델
│   │   ├── types.py                 # Episode, Note, Entity, Fact(bi-temporal),
│   │   │                            #   StrategyItem, Bullet, MemoryBundle
│   │   └── ops.py                   # MemoryOp = ADD|UPDATE|MERGE|DELETE|INVALIDATE|LINK|TAG
│   │                                #   + EvolutionLog Protocol (append-only)
│   │
│   ├── stores/                      # 저장 어댑터 (전부 동일 인터페이스)
│   │   ├── base.py                  # DocStore / VectorStore Protocol
│   │   ├── sqlite_doc.py            # episodes/notes/... + FTS5 + evolution_log (lite/standard)
│   │   ├── postgres_doc.py          # embedded PostgreSQL(pgserver) + tsvector (full)
│   │   ├── sqlite_vec.py            # lite 기본
│   │   ├── lance_vec.py             # standard
│   │   ├── qdrant_vec.py            # full
│   │   ├── chroma_vec.py            # 원논문 재현용 (A-Mem/G-Memory fidelity)
│   │   ├── numpy_vec.py             # 테스트 전용 (런타임 후보 제외)
│   │   ├── sqlite_graph.py          # 재귀 CTE k-hop (최후 수단)
│   │   ├── kuzu_graph.py            # embedded 실물 그래프 (lite/standard 기본)
│   │   └── neo4j_graph.py           # full (Zep/Graphiti 자체 엔진)
│   │
│   ├── llm/                         # 모델 접근 (전부 OpenAI-compatible)
│   │   ├── client.py                # 역할별 라우팅: extract/distill/judge/rerank/generate
│   │   ├── structured.py            # guided_json + 재시도 + drop 카운터
│   │   └── budget.py                # calls/tokens/latency 계측 (1급 메트릭)
│   │
│   ├── embed/
│   │   ├── base.py                  # Embedder Protocol + Embeddable(embedding_text())
│   │   ├── st_embedder.py           # sentence-transformers (e5-small/bge-m3 등)
│   │   └── fake.py                  # 결정적 해시 임베더 (테스트 전용)
│   │
│   ├── retrieval/                   # §03 문서의 파이프라인
│   │   ├── pipeline.py              # Recall(dense/lexical/graph) → Fusion → Rerank
│   │   │                            #   + 링크/그래프/experience 확장, bi-temporal 렌더
│   │   ├── fusion.py                # RRF
│   │   └── rerank.py                # Noop / MMR / LLMReranker / CrossEncoder
│   │
│   ├── organizers/                  # ★ 방법론 = Organizer 플러그인
│   │   ├── base.py                  # Organizer Protocol:
│   │   │                            #   on_message / on_task_end → list[MemoryOp]
│   │   │                            #   (+ warm_start, on_retrieval, flush_buffer,
│   │   │                            #   on_memory_event(consumes 구독)/consolidate — §2)
│   │   ├── passthrough.py           # no-op baseline (raw episode만)
│   │   ├── amem.py                  # 노트 구성→링크→이웃 진화 (버그 수정판 명시)
│   │   ├── memoryos.py              # STM/MTM/LPM + heat 승격 + LFU eviction
│   │   ├── nemori.py                # boundary 분절→서사(시간 절대화)→predict-calibrate 증류
│   │   ├── nemori_stages.py         # Segmenter/EpisodeMerger/Integrator/Consolidator
│   │   │                            #   스테이지 (fidelity 스위치, docs/11 §4)
│   │   ├── zep_graph.py             # entity 추출→resolution→fact→invalidation
│   │   ├── ace.py                   # Generator/Reflector/Curator + playbook delta
│   │   ├── reasoning_bank.py        # self-judge→성공/실패 증류→append (+MaTTS 훅)
│   │   └── gmemory.py               # MAS 궤적 sparsify→insight, reward 기반 프루닝
│   │
│   ├── memory.py                    # AgenticMemory 퍼사드 (05 문서의 공개 API)
│   │                                #   + 비동기 write 워커 (내장 스레드+큐, sync_write=False 시)
│   ├── config.py                    # TOML 로딩 + profile 프리셋
│   │
│   ├── mcp/                         # MCP 서버 (05 문서)
│   │   └── server.py                # FastMCP, stdio + streamable HTTP
│   │
│   ├── bench/                       # 평가 하네스
│   │   ├── harness.py               # multi-run + mean/std 집계, 재현성 스탬프
│   │   └── locomo.py                # LoCoMo 로더, F1/BLEU-1 (judge 불필요 경로)
│   │
│   └── train/                       # 0.5B 보조모델 학습 (roadmap Phase 4)
│       ├── distill_data.py          # 대형 모델로 추출/분절/증류 SFT 데이터 생성
│       └── sft_lora.py              # peft QLoRA (RTX 2060 6GB 타깃)
│
├── scripts/                         # 실험 진입점 (exp_locomo_conv0.py 등)
├── tests/                           # pytest
└── docs/                            # 본 문서들
    └── 12-code-conventions.md       # 이름/docstring/구조 컨벤션 (리뷰 게이트 기준)
```

로드맵(미구현 모듈): `core/namespace.py`, `retrieval/expand.py`(time-range 추출), `embed/api_embedder.py`, `bench/longmemeval.py·judges.py·report.py`, `train/eval_extract.py`, graph store용 `QueueStore`/`GraphStore` 공통 Protocol.

## 2. 데이터 흐름

### Write (모든 방법론 공통 골격)

```
add_message(msg) / add_task_result(traj)
  │  sync: DocStore에 raw episode 기록 (+즉시 검색 노출), queue enqueue, <10ms 반환
  ▼
worker: Organizer.on_message/on_task_end 실행 (LLM 호출 발생 지점)
  │  산출물 = list[MemoryOp]
  ▼
EvolutionLog.append(ops) → 각 store에 반영 (vector upsert, graph upsert/invalidate, ...)
  │  applied ADD/UPDATE/MERGE op → MemoryEvent(source, op, target_type, target_id,
  │  payload, supersedes) 로 변환
  ▼
_propagate_events: target_type ∈ consumes 인 다른 organizer에 순서대로 전달
  │  (자기 자신 제외, depth=1 — 응답 op는 적용되지만 재전파 안 됨;
  │  DELETE/INVALIDATE는 전파하지 않음 — supersedes는 MERGE에 실려 원자적으로 전달)
  ▼
(명시적) AgenticMemory.consolidate() → 등록 순서대로 Organizer.consolidate(ctx) 호출
  │  각자 evolution log seq 커서(target_type="state", id="consolidate:{name}")를 읽어
  │  이후분만 배치 처리(dedup/merge/재조직) → INVALIDATE+ADD/UPDATE 반환, 마지막에 커서 전진
```

- Organizer는 store를 직접 만지지 않고 **MemoryOp만 반환** → 방법론 코드와 스토리지가 완전 분리, 로그 재생으로 상태 복원/디버깅 가능.
- 동기 모드(`sync_write=True`)도 지원 — 원논문 재현 실험은 동기로 돌려 원 구현과 조건을 맞춘다.
- **인라인 vs 유예 위상**: `on_message`/`on_task_end`/`on_retrieval`/`on_memory_event`/
  `flush_buffer`는 인라인(ingest 경로에서 즉시 실행, 방법론 원형 재현의 자리)이고
  `consolidate`는 유예(명시적 API 호출로만 실행, 배치 dedup/merge/재조직의 자리) —
  두 위상은 서로를 강제하지 않는다 (스펙 §1.1).
- INVALIDATE는 기존 `invalid_at`을 보존(최초 시각 유지)하며, bi-temporal 렌더 타입
  (`memory.py::BITEMPORAL_TYPES` = `facts`)이 아니면 벡터도 함께 제거한다 — semantic/
  episodes 등은 무효화 즉시 검색에서 빠지고, facts는 validity 구간과 함께 계속 렌더된다.

### Read

```
search(query, memory_types=[...], k=...)
  → retrieval.pipeline (타입별: dense+lexical recall → RRF → rerank → hydrate → 확장)
  → MemoryBundle { episodes, facts, semantic, strategies, playbook, provenance }
```

- QueryExpansion(time-range 추출 등, docs/03 §1.2)은 로드맵 — 현재 파이프라인은 recall부터 시작.
- recall은 memory_type별로 순차 실행(스레드/async 병렬화 없음); dense는 항상, lexical은
  `episodic`과 `lexical_types`에 포함된 타입만 BM25/FTS 채널을 추가로 RRF 융합한다.

- `MemoryBundle.render(budget_tokens)` — 타입별 우선순위/토큰 예산으로 프롬프트 주입용 텍스트 생성 (Zep의 context block, ReasoningBank의 system prompt 주입 형식 지원).

## 3. 방법론 → 공통 추상화 매핑 검증

| 방법론 | on_message | on_task_end | consumes / on_memory_event | consolidate | 사용하는 MemoryOp | 필요 store |
|---|---|---|---|---|---|---|
| A-Mem | note 생성+링크+진화 (`input="messages"`, 기본) | — | `input="episodes"`: `consumes=("episodes",)`, `on_message` no-op, episode가 note 원문이 됨; MERGE의 `supersedes` 수신 시 흡수된 episode의 note를 INVALIDATE | — (base no-op) | ADD(note), LINK, UPDATE(이웃) | doc+vec |
| MemoryOS | STM append; 방출 시 배치 조직화 (`input="messages"`, 기본) | — | `input="episodes"`: `consumes=("episodes",)`, `on_message` no-op, `on_memory_event`가 STM append 대체(episode→page 역매핑 유지); `supersedes` 수신 시 해당 page의 source가 전부 superseded일 때만 INVALIDATE | — (base no-op) | ADD(page/segment), MERGE, UPDATE(profile), DELETE(LFU) | doc+vec |
| Nemori | buffer; boundary 시 episode+증류 (distiller 역할 — `consumes=()`, 구독 없음) | — | 자신은 구독하지 않고 **방출**만 함: episode merge 시 MERGE(신규 병합 episode)+INVALIDATE(흡수된 구 episode)를 같은 배치로 반환, MERGE op의 `payload["supersedes"]`에 흡수 id들을 명시 | `consolidation="semantic_offline"`: semantic 전수/증분 클러스터링→LLM merge/conflict 판정→INVALIDATE+ADD, 커서(`consolidate:nemori`) 전진 | ADD(episode), ADD(semantic), MERGE(episode/semantic)+INVALIDATE(supersedes) | doc+vec |
| Zep-graph | episode→entity/fact 파이프라인 | — | — (base no-op) | — (base no-op) | ADD(entity/fact), MERGE(dedup), INVALIDATE(모순) | doc+vec+graph |
| ACE | — | reflect→curate | — (base no-op) | — (base no-op) | ADD(bullet), UPDATE(카운터), MERGE(dedup) | doc+vec(dedup용) |
| ReasoningBank | — | judge→증류 | — (base no-op) | — (base no-op) | ADD(strategy) | doc+vec |
| G-Memory | — (MAS 메시지 수집) | sparsify→insight | — (base no-op) | — (base no-op) | ADD(traj/insight), UPDATE(reward), MERGE(FINCH) | doc+vec+graph |

→ 7개 방법론 모두 `(2개 훅 × MemoryOp 7종 × store 3종)` 안에 들어감. 추상화 누수 없음을 Phase 1에서 A-Mem/ReasoningBank로 먼저 검증.
→ `on_memory_event`/`consolidate`는 이번 라이프사이클 재설계(스펙:
  `docs/superpowers/specs/2026-07-18-nemori-lifecycle-redesign-design.md`)에서 Nemori(distiller)
  +MemoryOS·A-Mem(`input="episodes"` consumer) 2개 조합만 마이그레이션했고, 나머지 5개
  organizer는 base 디폴트(no-op)로 무변경 — management-agnostic 계약이 참여 organizer 수와
  무관하게 성립함을 최소 조합으로 검증하는 단계.

## 4. 프로세스 토폴로지

```
[LLM 데몬]  vLLM/llama.cpp (Qwen3-0.6B 등, OpenAI-compatible :8000)  ← 상주
[MCP 서버]  agmem.mcp (stdio 또는 :8765)      ─┐
[Python API] import agmem                      ├─ 같은 memory.db/vectors 공유
[벤치 하네스] scripts/exp_locomo_conv0.py 등   ─┘
[워커]      memory.py 내장 백그라운드 스레드 (sync_write=False 시)
```

- lite에서는 전부 단일 머신·단일 DB 파일. full에서는 store들이 서버형으로 바뀔 뿐 토폴로지 동일.
