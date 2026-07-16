# 모듈 구조 설계

> 핵심 아이디어: **retrieval은 공통 인프라, 방법론 차이는 write-path의 `Organizer`로 캡슐화,
> 모든 메모리 변경은 append-only `evolution_log` 연산으로 표현.**

## 1. 패키지 레이아웃

```
agentic_memory/                      # 패키지명: agmem (pip install agmem)
├── pyproject.toml                   # uv 관리, extras: [neo4j,qdrant,rerank,train,eval]
├── src/agmem/
│   ├── capabilities/                # §01 문서의 capability detection
│   │   ├── detect.py                # HostCapabilities 감지 (+캐시)
│   │   ├── requires.py              # Requires 선언, CapabilityWarning
│   │   └── resolver.py              # profile+config+capability → 구현 선택
│   │
│   ├── core/                        # 방법론 독립적 도메인 모델
│   │   ├── types.py                 # Episode, Note, Entity, Fact(bi-temporal),
│   │   │                            #   StrategyItem, Bullet, Community, MemoryBundle
│   │   ├── ops.py                   # MemoryOp = ADD|UPDATE|MERGE|DELETE|INVALIDATE|LINK|TAG
│   │   │                            #   + EvolutionLog (append-only, replay)
│   │   └── namespace.py             # {user_id}/{agent_id}/{memory_type}
│   │
│   ├── stores/                      # 저장 어댑터 (전부 동일 인터페이스)
│   │   ├── base.py                  # DocStore / VectorStore / GraphStore / QueueStore Protocol
│   │   ├── sqlite_doc.py            # episodes/notes/... + FTS5 + evolution_log + queue
│   │   ├── sqlite_vec.py            # lite 기본
│   │   ├── lancedb_vec.py           # standard
│   │   ├── qdrant_vec.py            # full (Nemori 호환)
│   │   ├── chroma_vec.py            # 원논문 재현용 (A-Mem/G-Memory fidelity)
│   │   ├── sqlite_graph.py          # 재귀 CTE k-hop, lite 기본
│   │   ├── kuzu_graph.py            # embedded 옵션 (deprecated 리스크 명시)
│   │   ├── neo4j_graph.py           # full (Graphiti 호환 스키마 지향)
│   │   └── falkordb_graph.py        # full 옵션
│   │
│   ├── llm/                         # 모델 접근 (전부 OpenAI-compatible)
│   │   ├── client.py                # 역할별 라우팅: extract/distill/judge/rerank/generate
│   │   ├── structured.py            # guided_json/outlines + 재시도 + drop 카운터
│   │   └── budget.py                # calls/tokens/latency 계측 (1급 메트릭)
│   │
│   ├── embed/
│   │   ├── base.py                  # Embedder Protocol + Embeddable(embedding_text())
│   │   ├── st_embedder.py           # sentence-transformers (bge-small/bge-m3/MiniLM)
│   │   └── api_embedder.py          # OpenAI-compatible
│   │
│   ├── retrieval/                   # §03 문서의 4단계 파이프라인
│   │   ├── pipeline.py              # QueryExpansion → Recall(병렬) → Fusion → Rerank
│   │   ├── expand.py                # time-range 추출 (LongMemEval temporal pruning)
│   │   ├── recall.py                # DenseRecall / LexicalRecall / GraphRecall
│   │   ├── fusion.py                # RRF
│   │   └── rerank.py                # Noop / MMR / LLMReranker / CrossEncoder
│   │
│   ├── organizers/                  # ★ 방법론 = Organizer 플러그인
│   │   ├── base.py                  # Organizer Protocol:
│   │   │                            #   on_message(msg) / on_task_end(traj, outcome)
│   │   │                            #   → list[MemoryOp]  (+ warm_start(corpus))
│   │   ├── passthrough.py           # no-op baseline (raw episode만)
│   │   ├── amem.py                  # 노트 구성→링크→이웃 진화 (버그 수정판 명시)
│   │   ├── memoryos.py              # STM/MTM/LPM + heat 승격 + LFU eviction
│   │   ├── nemori.py                # boundary 분절→서사(시간 절대화)→predict-calibrate 증류
│   │   ├── zep_graph.py             # entity 추출→resolution→fact→invalidation→community
│   │   ├── ace.py                   # Generator/Reflector/Curator + playbook delta
│   │   ├── reasoning_bank.py        # self-judge→성공/실패 증류→append (+MaTTS 훅)
│   │   └── gmemory.py               # MAS 궤적 sparsify→insight finetune/merge (FINCH)
│   │
│   ├── memory.py                    # AgenticMemory 퍼사드 (05 문서의 공개 API)
│   ├── worker.py                    # 비동기 write 워커 (OrganizerQueue 소비)
│   ├── config.py                    # TOML 로딩 + profile 프리셋
│   │
│   ├── mcp/                         # MCP 서버 (05 문서)
│   │   └── server.py                # FastMCP, stdio + streamable HTTP
│   │
│   ├── bench/                       # 평가 하네스
│   │   ├── harness.py               # ingest 캐시, multi-run, calls/tokens/latency 기록
│   │   ├── longmemeval.py           # S/M/Oracle, judge pin, reading method 'con'+json 고정
│   │   ├── locomo.py                # F1/BLEU-1 (judge 불필요 경로)
│   │   ├── judges.py                # API judge + 로컬 judge 합치율 리포트
│   │   └── report.py                # 결과 테이블 (md/csv) + profile/버전 스탬프
│   │
│   └── train/                       # 0.5B 보조모델 학습 (roadmap Phase 4)
│       ├── distill_data.py          # 대형 모델로 추출/분절/증류 SFT 데이터 생성
│       ├── sft_lora.py              # peft QLoRA (RTX 2060 6GB 타깃)
│       └── eval_extract.py          # 구조화 출력 준수율/추출 품질 평가
│
├── tests/                           # pytest; stores/organizers는 property-based 테스트
├── docs/                            # 본 문서들
└── examples/
```

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
```

- Organizer는 store를 직접 만지지 않고 **MemoryOp만 반환** → 방법론 코드와 스토리지가 완전 분리, 로그 재생으로 상태 복원/디버깅 가능.
- 동기 모드(`sync_write=True`)도 지원 — 원논문 재현 실험은 동기로 돌려 원 구현과 조건을 맞춘다.

### Read

```
search(query, memory_types=[...], k=...)
  → retrieval.pipeline (expansion → recall 병렬 → RRF → rerank)
  → MemoryBundle { episodes, facts, semantic, strategies, playbook, provenance }
```

- `MemoryBundle.render(budget_tokens)` — 타입별 우선순위/토큰 예산으로 프롬프트 주입용 텍스트 생성 (Zep의 context block, ReasoningBank의 system prompt 주입 형식 지원).

## 3. 방법론 → 공통 추상화 매핑 검증

| 방법론 | on_message | on_task_end | 사용하는 MemoryOp | 필요 store |
|---|---|---|---|---|
| A-Mem | note 생성+링크+진화 | — | ADD(note), LINK, UPDATE(이웃) | doc+vec |
| MemoryOS | STM append; 방출 시 배치 조직화 | — | ADD(page/segment), MERGE, UPDATE(profile), DELETE(LFU) | doc+vec |
| Nemori | buffer; boundary 시 episode+증류 | — | ADD(episode), ADD(semantic), MERGE(episode) | doc+vec |
| Zep-graph | episode→entity/fact 파이프라인 | — | ADD(entity/fact), MERGE(dedup), INVALIDATE(모순) | doc+vec+graph |
| ACE | — | reflect→curate | ADD(bullet), UPDATE(카운터), MERGE(dedup) | doc+vec(dedup용) |
| ReasoningBank | — | judge→증류 | ADD(strategy) | doc+vec |
| G-Memory | — (MAS 메시지 수집) | sparsify→insight | ADD(traj/insight), UPDATE(reward), MERGE(FINCH) | doc+vec+graph |

→ 7개 방법론 모두 `(2개 훅 × MemoryOp 7종 × store 3종)` 안에 들어감. 추상화 누수 없음을 Phase 1에서 A-Mem/ReasoningBank로 먼저 검증.

## 4. 프로세스 토폴로지

```
[LLM 데몬]  vLLM/llama.cpp (Qwen3-0.6B 등, OpenAI-compatible :8000)  ← 상주
[MCP 서버]  agmem.mcp (stdio 또는 :8765)  ─┐
[Python API] import agmem                  ├─ 같은 memory.db/vectors 공유
[벤치 하네스] agmem-bench CLI              ─┘
[워커]      agmem.worker (MCP/API 프로세스 내 asyncio task 또는 별도 프로세스)
```

- lite에서는 전부 단일 머신·단일 DB 파일. full에서는 store들이 서버형으로 바뀔 뿐 토폴로지 동일.
