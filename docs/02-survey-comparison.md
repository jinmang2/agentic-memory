# Agentic Memory 시스템 서베이 & 비교 분석

> 근거 자료: `docs/research/*.md` (2026-07-16 조사, 8개 시스템 전체 완료).

## 1. 한눈에 보는 8개 시스템

| 시스템 | 한 줄 요약 | 메모리 표현 | 채택 학회 | 공식 코드 | 라이선스 | MCP |
|---|---|---|---|---|---|---|
| **MemoryOS** | OS 메모리 계층(STM/MTM/LPM) + heat 승격 | 대화 page + segment 요약 + profile | EMNLP'25 Oral | BAI-LAB/MemoryOS ★1.5k | Apache-2.0 | **공식 있음** (3 tools) |
| **A-Mem** | Zettelkasten 노트 + LLM 링크/진화 | 평면 노트(keywords/tags/context/links) | NeurIPS'25 | agiresearch/A-mem ★1.1k | MIT | 없음 |
| **Zep/Graphiti** | Bi-temporal knowledge graph | episode/entity/community 3-subgraph | arXiv | getzep/graphiti | Apache-2.0 | **공식 있음** (7 tools) |
| **Nemori** | 자기조직화 episodic memory (topic 분절 + 예측오차 증류) | episode(제목+3인칭 서사) + semantic fact | **ACL'26** (v4: "What Deserves Memory") | nemori-ai/nemori ★207 | MIT | 없음 |
| **A C E** | 컨텍스트=진화하는 playbook (delta 연산) | itemized bullet + helpful/harmful 카운터 | ICLR'26 | ace-agent/ace ★1.2k | Apache-2.0 | 없음 |
| **ReasoningBank** | 성공+실패 궤적에서 전략 증류 + MaTTS | title/description/content 전략 아이템 | **ICLR'26** (Google) | google-research/reasoning-bank ★438 ("not for production") | Apache-2.0 | 없음 |
| **G-Memory** | Multi-agent용 3-tier graph (insight/query/interaction) | 규칙 리스트 + 쿼리 그래프 + 궤적 체인 | NeurIPS'25 | bingreeky/GMemory ★260 | **없음(주의)** | 없음 |
| **LongMemEval** | 벤치마크 (5능력, 500문항, S/M/Oracle) | — | ICLR'25 | xiaowu0162/LongMemEval | MIT | — |

## 2. 설계 공간에서의 위치

여덟 개는 사실상 **세 그룹**으로 나뉜다.

### 그룹 1 — 대화형 장기기억 (personalization): MemoryOS, A-Mem, Zep, Nemori
"사용자와의 긴 대화 히스토리를 어떻게 저장/검색하나". LongMemEval/LoCoMo로 평가.

- **조직화 스펙트럼**: 평면 노트(A-Mem) → 계층 버퍼(MemoryOS) → episodic 분절(Nemori) → 완전한 temporal KG(Zep).
- 공통점: **retrieval은 전부 dense cosine top-k가 기본**이고, 차별화는 write-path의 LLM 조직화에 있음.
- Zep만 read-path에 hybrid(BM25+cosine+BFS)와 pluggable reranker(RRF/MMR/cross-encoder)를 갖춤.

### 그룹 2 — 절차적/전략 기억 (self-improvement): ACE, ReasoningBank, G-Memory
"태스크 수행 경험에서 전략을 추출해 다음 태스크를 개선". AppWorld/WebArena/ALFWorld류로 평가.

- 셋 다 **연산 기반 진화(operation-based evolution)** 를 씀:
  - ACE: Curator가 ADD/UPDATE/MERGE/DELETE (실코드는 ADD만 완전 지원 + embedding dedup 후처리)
  - G-Memory: LLM이 REMOVE/EDIT/ADD/AGREE + reward shaping(+1/-2)
  - ReasoningBank: LLM-judge 성공/실패 판정 → 궤적당 최대 3개 아이템 추출, **단순 append** (pruning/merge 없음 — 의도적 단순화). 실패 궤적에서 예방적 교훈 추출이 핵심 차별화 (AWM은 실패 추가 시 오히려 성능 하락)
- → **하나의 공통 추상화**(`MemoryEvolution` 연산 로그)로 세 방법론을 모두 표현 가능. 우리 설계의 핵심 통찰.

### 그룹 3 — 평가: LongMemEval (+LoCoMo, DMR)
- LongMemEval_S(~115k tokens)/M(~1.5M)/Oracle, 500문항, GPT-4o judge(스냅샷 `gpt-4o-2024-08-06` 하드코딩).
- V2(2026-05)는 agentic experience memory로 확장 — 그룹 2 시스템 평가에 적합.

## 3. 메커니즘 상세 비교

### 3.1 Write path (ingestion)

| 시스템 | 단위 | LLM 호출/단위 | 처리 방식 | 핵심 연산 |
|---|---|---|---|---|
| MemoryOS | dialogue page | 배치성: STM 방출 시 page당 1–2 + 배치 1, heat≥5 시 +2 (병렬) | sync (배치 몰림) | FIFO 방출→topic 분할→segment 병합(F_score≥0.6)→heat 승격 |
| A-Mem | note | **2 고정** (구성 Ps1 + 진화 Ps3) | sync (턴 지연 큼, issue #21) | note 생성→top-5 이웃 링크→이웃 context/tags 재작성 |
| Zep/Graphiti | episode | **4–6+** | async 가능, SEMAPHORE_LIMIT | entity 추출→resolution→edge 추출→temporal 파싱→invalidation→community |
| Nemori | episode (2–25 msgs) | ~3/episode (분절 배치 1 + 서사 1 + 예측 1 + 증류 1) — LoCoMo 전체 373 calls (**최강 baseline 대비 -59.5%**) | **완전 async** (asyncpg) | boundary 탐지(σ=0.7)→3인칭 서사(시간 절대화)→예측→원본과 대조해 gap만 증류→(opt) episode merge |
| ACE | task trajectory | 3-role: Generator/Reflector(≤3회 반복)/Curator | offline/online 양쪽 | reflection→delta ops→결정론적 merge→(opt-in) embedding dedup 0.90 |
| ReasoningBank | task trajectory | 2 (judge 판정 t=0.0 + 추출 t=1.0, ≤3 items) | 태스크 종료 시 | self-judge 성공/실패→양방향 전략 증류→append only. MaTTS 시 self-contrast(≤5 items) |
| G-Memory | MAS trajectory | 궤적당 1–2 + 주기적 finetune/merge | 태스크 종료 시 | sparsify→저장→5개마다 insight finetune→20개마다 FINCH merge |

### 3.2 Read path (retrieval)

| 시스템 | 검색 방식 | rerank | read-path LLM | 반환 형태 |
|---|---|---|---|---|
| MemoryOS | 3-way 병렬 dense (STM+MTM top-5 seg+LPM top-k) | 없음 | 0 (생성 제외) | page + profile + knowledge |
| A-Mem | 단일 dense top-k (+1-hop 링크 보강) | 없음 | **0** | note 텍스트 |
| Zep/Graphiti | **hybrid: BM25+cosine(+BFS)** | **RRF/MMR/cross-encoder/node_distance/episode_mentions** | 0 (레시피에 따라) | edge facts + node summaries + community |
| Nemori | dense 이원 (episodic k=10 + semantic 2k=20, narrative 임베딩) | 없음 | **0** (Search 787ms) | episode 서사(상위 2개는 원본 첨부) + semantic facts |
| ACE | **retrieval 없음** — playbook 전체를 컨텍스트에 주입 (bullet 단위 사용 추적) | — | 0 | playbook 텍스트 |
| ReasoningBank | dense **top-k=1** (gemini-embedding-001; k≥2는 오히려 성능 하락) | 없음 | 0 | 전략 아이템 → system prompt 주입 |
| G-Memory | Chroma dense + query graph k-hop(=1) | LLM relevance scoring (2×topk 호출) | 2×topk + role projection | insights(役별 재작성) + 궤적 |

**관찰**: read-path에 LLM을 쓰는 건 G-Memory뿐이고 나머지는 전부 저지연 벡터 검색. ACE는 아예 검색 자체를 포기하고 "작은 메모리를 통째로 넣기"를 택함 — 메모리 크기가 작을 때(playbook)만 유효한 전략.

### 3.3 시간(temporality) 처리

| 시스템 | 방식 |
|---|---|
| Zep | **bi-temporal** (valid/invalid + transaction time), edge invalidation은 삭제가 아닌 무효화 (append-only 감사 가능) |
| MemoryOS | timestamp + recency 지수감쇠(코드 τ=24h), profile은 timestamp append/merge |
| A-Mem | timestamp 필드만 (추론 없음) |
| Nemori | episode 경계 = 1급 개념 + **서사 생성 시 상대시간을 절대시간으로 변환** ("yesterday"→날짜 병기) — temporal reasoning 우위(+15~48% 상대)의 직접 원인 |
| ACE/RB/G-Memory | 태스크 단위라 시간 개념 희박 (helpful/harmful, reward score가 대체) |

LongMemEval temporal-reasoning에서 Zep이 +48% 상대개선을 보인 반면 knowledge-update/verbatim recall에서는 **퇴화**(-3~-18%) — 그래프 추상화가 원문 디테일을 잃는 tradeoff의 실증.

## 4. 벤치마크 성적 종합 (출처 성격 주의)

### LongMemEval_S (accuracy)

| 시스템 | 모델 | 점수 | full-context 대비 | latency | 출처 성격 |
|---|---|---|---|---|---|
| full-context | gpt-4o | 60.2% | — | 28.9s | 원논문 |
| **Zep** | gpt-4o | **71.2%** | +11.0%p | **2.58s (-90%)** | 자체 논문 |
| Zep | gpt-4o-mini | 63.8% | +8.4%p | 3.20s | 자체 논문 |
| **Nemori** | gpt-4o-mini | 64.2% | +9.2%p | 컨텍스트 95% 절감 (3.7~4.8k tokens) | 자체 논문 |
| Nemori | gpt-4.1-mini | **74.6%** | +9.0%p | — | 자체 논문 |
| (LoCoMo 참고) Nemori | gpt-4o-mini | LLM-judge 73.0 (Full Context 72.3 상회, Mem0 61.3 대비 +19%) | 총 latency 3.05s (MemoryOS 15.2s, LangMem 22.1s) | 구축 비용 -59.5% calls | 자체 논문 (baseline 7종 동일 judge) |
| Mem0 (참고) | — | 94.4% | — | 6.8k tokens/call | **벤더 자체 발표 (미검증)** |
| Supermemory (참고) | — | 95% | — | 720 tokens | **벤더 자체 발표 (미검증)** |

### LoCoMo (F1, GPT-4o-mini)

| 시스템 | Single-hop | Multi-hop | Temporal | Open-domain |
|---|---|---|---|---|
| MemoryOS | **35.27** | 41.15 | **20.02** | **48.62** |
| A-Mem (보고치) | 27.02 | **45.85** | 12.14 | 44.65 |
| A-Mem* (MemoryOS 재현치) | 22.61 | 33.23 | 8.04 | 34.13 |
| MemGPT | 26.65 | 25.52 | 9.15 | 41.04 |

### Agent 태스크

| 시스템 | 벤치마크 | 결과 |
|---|---|---|
| ACE | AppWorld (DeepSeek-V3.1) | ReAct 42.4% → offline 59.4% / online 59.5% (GPT-4.1 기반 IBM CUGA 60.3%와 동급) |
| ACE | FiNER/Formula | 69.1% → 81.9%; 적응 latency **-86.9%** (vs GEPA/DC) |
| G-Memory | ALFWorld (AutoGen) | 85.82% (커뮤니티 재현 67–76%) |
| G-Memory | 5개 벤치 | +3~+21%p (GPT-4o-mini, Qwen2.5-7B/14B) |
| ReasoningBank | WebArena (Gemini-2.5-flash) | 40.5% → **48.8%** (+8.3%p), step 9.7→8.3; pro 46.7→53.9 |
| ReasoningBank | SWE-Bench-Verified | flash 34.2→**38.8%**, pro 54.0→57.4; step 최대 -16% |
| ReasoningBank | MaTTS (Shopping, k=5) | 49.7 → **55.1** (parallel) / 54.5 (sequential); 메모리 없는 TTS는 39~42 정체 |

### ⚠️ 재현성 신뢰도 등급 (조사 결과 기반)

| 등급 | 시스템 | 근거 |
|---|---|---|
| 상 | LongMemEval(벤치), ACE | eval 코드+데이터 공개, judge 프로토콜 명세. 단 ACE는 AppWorld 코드 미공개 |
| 중 | MemoryOS, Zep | eval 파이프라인 공개. 단 MemoryOS는 prompt 불일치(#50)·파라미터 미문서화, Zep은 LoCoMo 84% 주장이 제3자 재측정에서 58.44%로 반박된 전례 |
| 하 | A-Mem, G-Memory | A-Mem: L2/cosine 버그(#24), 인덱스 버그(#32), 타 논문 재현치가 보고치 대비 크게 낮음. G-Memory: 커뮤니티 재현 -10~-18%p, eval 데이터 라벨 버그, HotpotQA 데이터 미공개 |

**교훈**: 우리 재현 목표는 "논문 절대 수치 도달"이 아니라 **(1) 동일 프로토콜에서 baseline 대비 향상 방향 재현, (2) multi-run 평균±편차 보고, (3) judge/프롬프트/데이터 버전 pin 고정**이어야 한다.

## 5. 구현 관점 코드 품질/이식성 요약

| 시스템 | 저장소 실체 | 이식 난이도 | 소형(≤3B) 모델 증거 |
|---|---|---|---|
| MemoryOS | JSON+FAISS (ChromaDB판 별도), 코어 4벌 복제 드리프트 | 중 — 코어 로직 단순, heat 공식 이식 용이 | Qwen2.5-3B/7B 본문 실험 ✔ |
| A-Mem | ChromaDB(기본 in-memory) + dict | 하(쉬움) — 단 알려진 버그 3개 수정 필요 | 1B–3B 실험 ✔ (단 strict JSON 실패 모드) |
| Zep/Graphiti | Neo4j/FalkorDB/Neptune (Kuzu deprecated) | 상 — 파이프라인 10단계, 구조화 출력 의존 | ✘ 공식적으로 비권장 (PR #1227 이후에도) |
| ACE | 텍스트 playbook + faiss dedup | 하(쉬움) — 3 프롬프트 + 결정론 merge | DeepSeek-V3.1(대형 오픈)만 실증, 0.5B는 미검증 |
| ReasoningBank | JSON 파일 + 사전계산 임베딩 JSON (벡터 DB 없음) | **하(가장 쉬움)** — 프롬프트 4종 + cosine top-1 | 커뮤니티: Qwen3-1.7B에서 +8.0pp (통계 비유의) — **0.5B 재현의 최우선 후보** |
| G-Memory | networkx pickle + JSON + Chroma | 중 — 실험 하네스에 강결합, 서비스화 안 됨 | Qwen2.5-7B/14B ✔ (0.5B 미검증) |

## 6. 설계로 가져갈 핵심 시사점 (통합)

1. **차별화는 write-path**: retrieval은 공통 인프라(hybrid search + pluggable rerank)로 통일하고, 방법론별 차이는 write-path의 "조직화 전략(organizer)"으로 캡슐화한다.
2. **연산 기반 진화의 공통 추상화**: ACE(delta ops)·G-Memory(rule ops)·ReasoningBank(전략 추가)·A-Mem(이웃 재작성)을 `ADD/UPDATE/MERGE/DELETE/INVALIDATE` 연산 로그 하나로 수렴 가능. append-only 로그 + 스냅샷 재구성으로 감사 가능성(Zep의 bi-temporal 교훈)까지 확보.
3. **Verbatim 보존 레이어 필수**: Zep의 single-session-assistant 퇴화가 보여주듯 추상화 레이어(그래프/요약)와 별개로 **원문 episode를 항상 보존**하고 retrieval이 양쪽 모두에 닿아야 한다.
4. **소형 모델 대응 = 구조화 출력 방어**: 0.5B급의 최대 실패 모드는 strict JSON 미준수(A-Mem evolution 무력화, Graphiti ingestion 실패). → (a) JSON schema 단순화, (b) constrained decoding(vLLM guided_json/outlines), (c) 실패 시 재시도+강등 규칙, (d) 추출 태스크용 SFT(0.5B 파인튜닝 — 학습 계획과 연결)로 4중 방어.
5. **비용 프로파일이 곧 방법론 선택**: write당 LLM 호출이 A-Mem 2회 ↔ Zep 4–6회 ↔ MemoryOS 배치성 4.9회/응답. 우리 벤치 하네스는 정확도와 함께 **calls/tokens/latency를 1급 메트릭**으로 기록해야 한다 (MemoryOS Table 3, ACE Table 4 형식).
6. **평가 프로토콜 엄격화**: judge 스냅샷 pin(`gpt-4o-2024-08-06`), reading method(`con`)+history format(`json`) 통일(이것만 10%p 차이), cleaned 데이터 버전 명시, abstention 30문항 처리 규칙, multi-run 보고. Zep-LoCoMo 논란의 반면교사.
