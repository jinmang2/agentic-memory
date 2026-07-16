# Zep / Graphiti 리서치 노트

> 출처: 병렬 리서치 세션 (2026-07-16). arXiv:2501.13956, github.com/getzep/graphiti, github.com/getzep/zep-papers 기반.

## 논문 정보

"Zep: A Temporal Knowledge Graph Architecture for Agent Memory", Rasmussen, Paliychuk, Beauvais, Ryan, Chalef (Zep AI), arXiv:2501.13956 (2025-01). Zep 제품의 core memory engine이 오픈소스 **Graphiti**(github.com/getzep/graphiti, Apache-2.0)로 분리 배포되어 있음.

## A. 핵심 개념

### A.1 세 개의 subgraph로 구성된 temporal knowledge graph

Zep/Graphiti의 지식 그래프 `G = (N, E, φ)`는 세 개의 subgraph로 구성된다.

- **Episode subgraph (G_e)**: 원본으로 ingest된 데이터(대화 메시지 / 텍스트 / JSON)를 손실 없이 그대로 보관하는 provenance 레이어. Episodic edge가 각 episode를 그 안에서 언급된 semantic entity로 연결한다.
- **Semantic entity subgraph (G_s)**: dedup(중복 제거)된 entity node와, entity 간 관계/사실을 나타내는 entity edge로 구성.
- **Community subgraph (G_c)**: 강하게 연결된 entity들의 클러스터를 나타내는 community node. 각 community node는 LLM이 생성한 요약(summary)과 embedding을 가진다.

### A.2 Bi-temporal 모델

엣지(fact)마다 **두 개의 독립적 시간축**을 추적한다.

- **Transaction time (T′)**: `t′created`, `t′expired` — 시스템이 그 사실을 언제 알게/폐기하게 되었는지에 대한 감사(audit) 기록.
- **Event/valid time (T)**: `t_valid`, `t_invalid` — 그 사실이 실세계에서 실제로 유효했던 기간.

이 이중 시간 모델 덕분에 "그 시점에 우리가 무엇을 믿고 있었는가"와 "그 시점에 실제로 무엇이 참이었는가"를 분리해서 질의할 수 있다.

**Edge invalidation**: 새 edge가 추가될 때, LLM이 의미적으로 관련된 기존 edge들과 새 edge를 비교해 모순을 탐지한다. 모순이 발견되면 기존 edge를 무효화하되(delete하지 않고) `t_invalid`를 새 edge의 `t_valid`로 설정한다 — 즉 "최신 정보가 이긴다"는 원칙이며, 과거 사실의 이력은 그대로 보존된다(append-only에 가까운 감사 가능한 그래프).

### A.3 Entity resolution (중복 제거) 파이프라인

1. entity 이름을 embedding space(1024-dim, cosine similarity)에 투영
2. 이름/요약에 대한 full-text search
3. 후보 entity들을 episode context와 함께 LLM에 넘겨 dedup 판단
4. 중복으로 판정된 entity들을 병합하며 통합 이름+요약을 새로 생성

### A.4 Community detection

**Leiden이 아니라 label propagation**을 사용한다. 선택 이유는 새 노드가 들어올 때마다 그래프 전체를 다시 계산하지 않고, 이웃 노드들 중 다수(plurality)에 속한 커뮤니티에 신규 노드를 즉시 배정하는 **점진적(incremental) 확장**이 가능하기 때문이다. drift를 막기 위해 주기적으로 전체 재계산(refresh)을 수행한다. Community summary는 멤버 노드 요약들을 map-reduce 방식으로 반복 요약해서 생성하며, community node 자체도 이름이 embedding되어 유사도 검색 대상이 된다.

### A.5 Ingestion 파이프라인 (`add_episode`) — 약 10단계

1. Episode 수신 (actor, timestamp, context)
2. Entity 추출 — 현재 메시지 + 직전 4개 메시지를 함께 처리, 화자(speaker) 자동 추출, hallucination을 줄이기 위한 "reflection" 기법 적용
3. Entity embedding 후 기존 노드 대비 cosine + full-text 검색
4. Entity resolution — LLM 기반 dedup + 요약 병합
5. 그래프 반영은 **사전 정의된(predefined) Cypher 쿼리**로 수행 (LLM이 Cypher를 직접 생성하지 않음 — 신뢰성/안전성 목적)
6. Resolve된 entity 쌍 사이의 fact/edge 추출 (관계 유형 + temporal 속성)
7. Fact embedding 후 동일 entity-pair 범위 내에서 hybrid search로 dedup
8. Temporal extraction — 절대/상대 시간 표현을 파싱해 `t_valid`/`t_invalid` 채움
9. Edge invalidation — 관련 기존 edge들과의 모순 여부를 LLM이 검사
10. Community update — 신규 entity에 대한 label propagation 기반 동적 확장, 주기적 전체 refresh

Episode 하나를 추가할 때마다 (추출, dedup, temporal parsing, invalidation, 필요 시 community refresh 등) **최소 4~6회 이상의 LLM 호출**이 발생하는 구조로, 논문 자체는 정확한 호출 횟수를 명시하지 않는다.

## B. 공식 코드 분석 (github.com/getzep/graphiti)

**License: Apache License 2.0.**

### B.1 리포지토리 구조

```
graphiti/
├── graphiti_core/    # 핵심 엔진 (pip 패키지명: graphiti-core)
├── examples/         # 퀵스타트
├── mcp_server/       # 공식 MCP 서버
├── server/           # FastAPI 기반 REST 서비스
├── tests/
├── spec/
└── docker-compose.yml
```

`graphiti_core/` 내부:

- `graphiti.py` — 메인 `Graphiti` 클래스 (엔트리포인트, 예: `add_episode`)
- `nodes.py` — `EntityNode`, `EpisodicNode`, `CommunityNode` 정의
- `edges.py` — `EntityEdge`, `EpisodicEdge`, `CommunityEdge` 정의
- `graph_queries.py` — 사전 정의된 Cypher/쿼리 빌더
- `errors.py`, `helpers.py`, `decorators.py`, `graphiti_types.py`, `tracer.py`
- `search/` — 검색 구현체, 특히 `search_config_recipes.py` (아래 B.3 참고)
- `llm_client/` — LLM provider 추상화 (OpenAI, Anthropic, Gemini, Groq, OpenAI-호환/로컬)
- `embedder/` — 임베딩 provider 추상화 (OpenAI, Azure OpenAI, Voyage, Gemini)
- `cross_encoder/` — cross-encoder reranker 추상화 (OpenAI, Gemini reranker)
- `driver/` — 그래프 DB 백엔드에 대한 `GraphDriver` 추상화 레이어
- `prompts/` — 추출/dedup/invalidation용 LLM 프롬프트 템플릿
- `models/`, `namespaces/`, `migrations/`, `telemetry/`, `utils/`

### B.2 지원 그래프 DB (`driver/` 추상화 경유)

- **Neo4j 5.26+** (기본값)
- **FalkorDB 1.1.2+** (Redis 기반; Python 3.12+ 필요한 임베디드 "FalkorDB Lite"도 존재)
- **Amazon Neptune** (Database Cluster 또는 Analytics Graph) — full-text search를 위해 Amazon OpenSearch Serverless 병행 필요
- **Kuzu 0.11.2** — **deprecated 표시**, upstream 프로젝트 자체가 유지보수 중단 상태

설치: `pip install graphiti-core`, extras로 `[falkordb]`, `[falkordblite]`, `[neptune]`, `[kuzu]`, `[anthropic]`, `[groq]`, `[google-genai]` 제공.

### B.3 LLM/embedder/reranker provider 및 로컬 모델 지원

기본 provider는 OpenAI, Anthropic, Google Gemini, Groq, Azure OpenAI, 그리고 OpenAI-호환 엔드포인트(DeepSeek, Together, **Ollama, vLLM, llama.cpp, LM Studio**)까지 포괄한다. 다만 공식 문서는 OpenAI/Anthropic/Gemini를 신뢰성 측면에서 명시적으로 권장하는데, entity/edge 추출 및 dedup 파이프라인이 **구조화(JSON) 출력**에 강하게 의존하기 때문에 소형/로컬 모델은 스키마에 맞지 않는 JSON을 자주 출력해 ingestion이 실패하는 경우가 많다고 밝히고 있다.

2026년 병합된 PR #1227("feat(mcp): add local model support with openai_generic provider and reranker config")에서 `openai_generic` provider와 로컬 모델 튜닝용 기본값(기존 8K → **16K max tokens**)이 추가되어 로컬 모델 호환성이 개선되었다 — 즉 이 PR 이전 버전은 Ollama/vLLM 사용 시 불안정했다고 봐야 한다.

### B.4 검색 레시피 (`graphiti_core/search/search_config_recipes.py`)

Hybrid search = **BM25(full-text) + cosine similarity**, 필요 시 **BFS 그래프 순회**를 추가하고, pluggable reranker로 재정렬한다. 소스에서 확인된 preset들:

- `COMBINED_HYBRID_SEARCH_RRF` — edge/node/community 전체에 BM25+cosine, RRF(Reciprocal Rank Fusion) rerank(episode 포함)
- `COMBINED_HYBRID_SEARCH_MMR` — 동일 retrieval, edge/node/community는 MMR(mmr_lambda=1) rerank, episode는 RRF 유지
- `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` — edge/node에 BM25+cosine+BFS, 전체에 cross-encoder rerank
- `EDGE_HYBRID_SEARCH_{RRF,MMR,NODE_DISTANCE,EPISODE_MENTIONS,CROSS_ENCODER}` (edge 전용 5종)
- `NODE_HYBRID_SEARCH_{RRF,MMR,NODE_DISTANCE,EPISODE_MENTIONS,CROSS_ENCODER}` (node 전용 5종)
- `COMMUNITY_HYBRID_SEARCH_{RRF,MMR,CROSS_ENCODER}` (community 전용 3종)

Reranker 옵션 정리: **RRF**, **MMR**, **cross-encoder**, **node_distance**(특정 entity, 예: "Kendra"로부터의 그래프 거리 기준 재정렬), **episode_mentions**(언급 빈도 기반).

### B.5 Custom entity/edge types

개발자가 **Pydantic 모델**로 커스텀 entity/edge type을 정의해 `add_episode`/Graphiti 설정에 전달할 수 있다. MCP 문서에 언급된 기본 entity type: `Preference`, `Requirement`, `Procedure`, `Location`, `Event`, `Person`, `Organization`, `Document`, `Topic`, `Object` — 호출 단위로 `excluded_entity_types`를 통해 제외 가능.

### B.6 공식 MCP 서버 (`mcp_server/`)

**확실히 검증된 핵심 tool 집합**:
- `add_memory` — episode ingest(text/JSON/message), `reference_time`, `excluded_entity_types`, `custom_extraction_instructions` 지원
- `search_memory_nodes` — entity 검색, `entity_types`/`center_node_uuid` 필터
- `search_memory_facts` — edge 검색, `edge_types`/`center_node_uuid`/`valid_at`·`invalid_at` 날짜 범위 필터
- `get_episodes`, `delete_episode`, `clear_graph`, `get_status`

일부 문서 추출 과정에서 `add_triplet`, `summarize_saga`, `build_communities`, `get_episode_entities`, `delete_entity_edge`, `get_entity_edge`, `search_nodes` 같은 추가 tool 명이 나타났으나 **신뢰도가 낮음**(추출 아티팩트일 가능성) — 최종 인용 전 `mcp_server/src` 소스 직접 확인 필요. 위 핵심 6~7개 tool은 다수 소스에서 교차 확인됨.

**Transport**: HTTP가 기본(`/mcp/` 엔드포인트, 포트 8000), stdio도 지원(Claude Desktop류 클라이언트용). 일부 서드파티 문서에서 SSE(`/sse?group_id=...`)도 언급됨.

**설정**: CLI 인자 > 환경변수 > `config.yaml` 순으로 우선순위. 주요 환경변수: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `AZURE_OPENAI_API_KEY`/`_ENDPOINT`/`_DEPLOYMENT`, `FALKORDB_URI`, `NEO4J_URI`, `SEMAPHORE_LIMIT`(기본 10), `GRAPHITI_TELEMETRY_ENABLED`. `.env`는 `mcp_server/` 디렉토리에 위치해야 함(`docker/`가 아님). `group_id`는 하나의 배포에서 여러 지식그래프를 격리하는 네임스페이스(기본값 "main").

기본 embedder는 OpenAI `text-embedding-3-small`로 비교적 확실히 확인됨. (일부 2026년 중순 fetch 결과에서 MCP 서버 기본 LLM이 "gpt-5.5"로 나타났으나 요약 도구의 아티팩트일 가능성이 있어 — 실제 신규 기본값인지 여부는 최종 보고서 인용 전 재확인 필요.) Docker compose 기본 DB는 FalkorDB(단일 컨테이너 결합형, MCP+FalkorDB), 대안으로 Neo4j(`bolt://neo4j:7687`, 브라우저 UI `:7474`) 구성 제공. FalkorDB 결합형 구성에서는 웹 UI가 `:3000`.

Docker compose 옵션 3종: FalkorDB 결합형(단일 컨테이너), Neo4j(별도 컨테이너), FalkorDB 분리형(외부/클라우드 FalkorDB용).

**동시성 가이드(커뮤니티 문서)**: OpenAI Tier 1(3 RPM) → `SEMAPHORE_LIMIT` 1~2, Tier 3(500 RPM) → 10~15, Anthropic 기본(50 RPM) → 5~8. episode 하나당 다수의 LLM 호출이 발생하므로 실제 동시 요청 수는 semaphore 값보다 커진다는 점에 유의.

**Telemetry**: 익명 PostHog 사용 통계(UUID, OS, 버전, 설정 선택값)만 수집하며 API 키/PII/그래프 콘텐츠는 수집하지 않는다고 명시. `GRAPHITI_TELEMETRY_ENABLED=false`로 opt-out 가능.

## C. 성능 특성 (arxiv.org/html/2501.13956v1)

### C.1 DMR (Deep Memory Retrieval)

500개의 multi-session 대화(세션당 약 60개 메시지), single-turn fact-retrieval 질문.

| Memory | Model | Score |
|---|---|---|
| Recursive Summarization | gpt-4-turbo | 35.3% |
| Conversation Summaries | gpt-4-turbo | 78.6% |
| MemGPT | gpt-4-turbo | 93.4% |
| Full-conversation (baseline) | gpt-4-turbo | 94.4% |
| **Zep** | gpt-4-turbo | **94.8%** |
| Conversation Summaries | gpt-4o-mini | 88.0% |
| Full-conversation | gpt-4o-mini | 98.0% |
| **Zep** | gpt-4o-mini | **98.2%** |

논문 스스로 DMR의 한계(single-turn만 존재, 모호한 표현, 낮은 enterprise 대표성)를 지적하며 이것이 LongMemEval 평가를 도입한 동기라고 밝힌다.

### C.2 LongMemEval (LME)

평균 약 115K 토큰 대화, 2024년 12월~2025년 1월 사이 평가.

| Memory | Model | Accuracy | Latency (mean) | Latency IQR | 평균 context 토큰 |
|---|---|---|---|---|---|
| Full-context | gpt-4o-mini | 55.4% | 31.3s | 8.76s | 115k |
| **Zep** | gpt-4o-mini | **63.8%** | **3.20s** | **1.31s** | **1.6k** |
| Full-context | gpt-4o | 60.2% | 28.9s | 6.01s | 115k |
| **Zep** | gpt-4o | **71.2%** | **2.58s** | **0.684s** | **1.6k** |

헤드라인 수치: full-context 대비 **최대 +18.5%p 정확도**, **최대 90% 지연시간 감소**. context가 약 72배(115k→1.6k 토큰) 압축되는 것이 지연시간 개선의 주된 원인.

**카테고리별 분해 (gpt-4o-mini, full-context→Zep)**:
- single-session-preference: 30.0%→53.3% (+77.7% 상대개선)
- single-session-assistant: 81.8%→**75.0% (−9.06%, 퇴화)**
- temporal-reasoning: 36.5%→54.1% (+48.2%)
- multi-session: 40.6%→47.4% (+16.7%)
- knowledge-update: 76.9%→**74.4% (−3.36%, 퇴화)**
- single-session-user: 81.4%→92.9% (+14.1%)

**gpt-4o**: single-session-preference 20.0%→56.7%(+184%); single-session-assistant 94.6%→**80.4%(−17.7%, 퇴화)**; temporal-reasoning 45.1%→62.4%(+38.4%); multi-session 44.3%→57.9%(+30.7%); knowledge-update 78.2%→83.3%(+6.52%); single-session-user 81.4%→92.9%(+14.1%).

논문은 **single-session-assistant 카테고리의 퇴화를 명시적으로 known weakness**로 인정한다 — 정답이 하나의 세션 안에 verbatim으로 존재하고 모델이 그것을 그대로 recall해야 하는 경우, 그래프로의 abstraction 과정에서 미세한/verbatim 디테일이 손실되어 full-context보다 성능이 떨어질 수 있다.

**사용 모델**: 그래프 구축에는 gpt-4o-mini-2024-07-18, 응답 생성에는 gpt-4o 계열, retrieval/rerank 임베딩은 BGE-m3. 검색 방식(cosine/BM25/BFS)이나 reranker 선택에 대한 정식 ablation은 논문에 없으며, episode당 LLM 호출 수나 비용에 대한 명시적 분석도 없음.

### C.3 ⚠️ LoCoMo 벤치마크 관련 커뮤니티 반박 (재현성 이슈)

`getzep/zep-papers` 이슈 #5 "Revisiting Zep's 84% LoCoMo Claim: Corrected Evaluation & 58.44% Accuracy"에서, 이는 arXiv 논문 자체의 DMR/LongMemEval 수치가 아니라 **이후 별도로 마케팅된 LoCoMo 벤치마크 84% 주장**에 대한 것으로, Mem0 팀이 다음을 지적한다.

1. Category 5(adversarial) 질문을 분모에서는 제외하면서 정답 처리는 분자에 포함시켜 **약 25.6%p 부풀려짐**
2. Zep에 유리한 prompt/retrieval 템플릿 불일치
3. 단일 실행 보고 vs Mem0의 10-run 평균 방법론

Mem0의 표준화된 프로토콜 하에서 Zep의 LoCoMo 점수는 **58.44% ± 0.20%**로 재측정됨. 조사 시점 기준 이 이슈에 대한 메인테이너 공식 답변은 확인되지 않음. 이는 arXiv 논문 자체의 수치와는 별개이지만, Zep의 벤치마크 주장에 최소 한 차례의 실증된 부풀리기 사례가 있었다는 신뢰성 관련 근거로 시스템 설계 문서에 반영할 가치가 있다.

## D. 재현 관점

### D.1 재현에 필요한 것

- DMR/LongMemEval 평가 코드는 core `graphiti` 리포지토리가 아니라 별도 리포지토리 `github.com/getzep/zep-papers`에 있으며(`locomo_eval/zep_locomo_eval.py` 등, `benchmarks/` 하위에 LoCoMo/LongMemEval 하네스로 추정되는 구조 존재), 재현하려면 이 자매 리포지토리를 함께 pull해야 함.
- LongMemEval 데이터셋은 크기 변형이 있음: **LongMemEval_S**(~115K 토큰/문제, 논문에서 사용된 버전)와 **LongMemEval_M**(~1.5M 토큰/문제). 논문은 작은 버전만 보고하므로 대규모(_M) 재현은 저자 스스로도 검증하지 않은 상태.

### D.2 알려진 GitHub 이슈/함정

- **#683**: 과거 "해결됨" 처리된 #282의 회귀 — Neo4j `TypeError: Property values can only be of primitive types`. LLM이 nested dict/object 속성을 출력할 때 발생(Neo4j는 primitive 또는 primitive 배열만 허용).
- **#1021**: 일부 환경에서 Neo4j에 첫 episode 이후로는 episode가 조용히 저장되지 않는 문제 — open 상태, 확정 원인 미상.
- **#1116**: OpenAI provider가 `api_base` 설정을 무시하고 실제 OpenAI 엔드포인트로 fallback되어, Ollama/로컬 LLM을 가리키려 할 때 401 에러 발생 — provider 라우팅 버그.
- **#868**: "Cannot get minimal example to work with Ollama" — 로컬 모델 사용 시 일반적 마찰 보고.
- **PR #1227**(병합됨, `openai_generic` provider + reranker 설정 + 로컬 모델용 기본 max-tokens 16K 상향) — 이 PR 이전 버전은 로컬 모델 지원이 미흡했다고 봐야 함.

### D.3 로컬/소형 모델로 교체하기

- 전체 추출/dedup/temporal-parsing 파이프라인이 신뢰성 있는 JSON-schema 준수 출력에 의존한다. 공식 문서는 소형/로컬 모델에서 이 부분이 자주 실패한다고 명시하며, "하드웨어가 감당할 수 있는 가장 강력한 로컬 모델을 사용하라"는 실무 가이드를 제시한다. `openai_generic` 호환 shim을 쓰더라도 약한 모델에서는 ingestion 실패를 각오해야 함.
- `SEMAPHORE_LIMIT` 튜닝이 429 방지에 필수적이며, episode 하나당 순차/병렬로 여러 LLM 호출이 발생하므로 ingestion 비용은 "메시지당 1회 호출"보다 훨씬 빠르게 증가한다 — 특히 GPT-4급 모델 사용 시 운영 비용에 실질적 영향을 미치는 요인으로, 시스템 설계 문서에 명시할 가치가 있다.

### D.4 재현성 종합 평가

arXiv 논문 자체의 DMR/LongMemEval 수치는 평가 코드가 공개되어 있어 재현 시도는 가능하나, LoCoMo에서처럼 제3자에 의한 엄밀한 재검증은 아직 이루어지지 않은 것으로 보인다. 따라서 "vendor-reported, 그럴듯하지만 제3자 재현이 검증되지 않은 수치"로 취급하고, LoCoMo 논란(58.44% vs 84%)을 Zep 벤치마크 주장의 신뢰도에 대한 명시적 카운터-에비던스로 함께 인용하는 것을 권장한다.

## 우리 프로젝트에 주는 시사점

1. Graphiti는 "그래프"를 실제 프로덕션급 그래프 DB(Neo4j/FalkorDB/Neptune) 위에 구축하고, LLM이 직접 Cypher를 생성하지 않도록 predefined query로 안전장치를 둔 점이 특징 — G-Memory(networkx+pickle)보다 훨씬 무거운 인프라 의존성을 갖는 대신 프로덕션 배포 관점에서는 더 성숙함.
2. bi-temporal 모델(valid_at/invalid_at + transaction time)은 "사실의 갱신"을 삭제가 아닌 append-only invalidation으로 다루는 설계로, 감사 가능성(auditability)이 필요한 우리 시스템에도 참고할 가치가 있음.
3. episode당 4~6회 이상의 LLM 호출이 드는 ingestion 비용 구조는 실서비스 적용 시 비용 모델링에 반드시 반영해야 함.
4. 공식 MCP 서버가 이미 존재하므로, 우리가 MCP 인터페이스를 설계할 때 `add_memory`/`search_memory_nodes`/`search_memory_facts` 같은 tool 분리(엔티티 검색 vs 사실/엣지 검색을 별도 tool로 노출) 패턴을 참고할 수 있음.
5. 로컬/소형 모델 호환성은 2026년 PR #1227 기준으로도 여전히 "권장하지 않음" 수준이며, 구조화 출력 신뢰성이 약한 모델에서는 ingestion 파이프라인 전체가 취약해진다는 점을 우리 소형 모델 호환성 설계에 반영해야 함.
6. single-session-assistant/knowledge-update 카테고리에서의 성능 퇴화(그래프 추상화로 인한 verbatim 디테일 손실)는 우리 시스템에서도 "그래프 압축 vs raw 텍스트 보존"의 트레이드오프를 설계할 때 참고할 실증 사례.
7. LoCoMo 벤치마크 수치 논란은 우리가 자체 벤치마크를 보고할 때 평가 프로토콜(분모/분자 정의, multi-run 평균, prompt 동일성)을 처음부터 엄격하게 문서화해야 한다는 반면교사.
