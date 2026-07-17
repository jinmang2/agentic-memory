# 기술 스펙 상세 검토 — Retrieval/Rerank, Graph, Latency, Cold-start, Storage

> 각 축마다: 조사에서 확인된 사실 → 우리 선택 → capability 프로파일별 구성.
> 원칙: 방법론은 전부 구현, capability로 선택 (`docs/01-capability-system.md`).

## 1. Retrieval & Rerank

### 1.1 조사에서 확인된 사실

- 8개 시스템 중 7개의 기본 검색은 **dense cosine top-k** 단 하나. hybrid를 갖춘 것은 Graphiti뿐 (BM25+cosine+BFS, 프리셋 15종+).
- Reranker 실증 스펙트럼 (Graphiti): **RRF**(무비용), **MMR**(다양성, λ 파라미터), **cross-encoder**(고비용 고정밀), **node_distance**(그래프 중심성), **episode_mentions**(빈도).
- LongMemEval 논문 자체 실험: retriever(flat-bm25/contriever/stella/gte) × granularity(turn/session) × index expansion(요약/keyphrase/userfact) 조합이 10%p 이상 좌우. **time-aware query pruning만으로 recall +6.8~11.3%p**.
- ReasoningBank의 반직관적 발견: **top-k=1이 최적**, k≥2는 노이즈/충돌로 성능 하락 — "많이 가져오기"가 항상 좋지 않음. 반면 Nemori는 k=10에서 포화.
- A-Mem 교훈: 임베딩 대상에 메타데이터 포함(`concat(content, keywords, tags, context)`), Nemori 교훈: 원문이 아닌 **생성된 서사를 임베딩**하는 게 우세 (76.9 vs 76.4).

### 1.2 우리 스펙

**검색 파이프라인 (고정된 4단계, 각 단계가 어댑터):**

```
query → [1 QueryExpansion] → [2 Recall] → [3 Fusion] → [4 Rerank] → MemoryBundle
```

1. **QueryExpansion** (optional): time-range 추출(LongMemEval temporal pruning 이식), 키워드 추출. LLM 불필요한 규칙 기반 우선.
2. **Recall** (병렬 실행, 소스별):
   - `DenseRecall`: 벡터 검색 (모든 store 공통)
   - `LexicalRecall`: BM25/FTS5 (sqlite FTS5 / tantivy / Neo4j fulltext)
   - `GraphRecall`: seed 노드에서 BFS/k-hop (graph store 활성 시)
   - 각 recall은 `candidate_k = 3×final_k` 반환
3. **Fusion**: RRF 기본 (rank 기반이라 스코어 정규화 불필요)
4. **Rerank** (capability-gated):
   - `NoopReranker` → `MMRReranker`(다양성) → `LLMReranker`(listwise, 소형 LLM 1콜) → `CrossEncoderReranker`(bge-reranker-v2-m3, GPU)

**임베딩 정책:**
- 임베딩 대상 = 방법론이 결정 (A-Mem식 메타데이터 concat / Nemori식 서사 / raw). `Embeddable` 프로토콜로 각 메모리 타입이 `embedding_text()` 구현.
- 기본 모델: `lite`=bge-small-en-v1.5 또는 multilingual-e5-small (CPU, 384d) / `standard`=bge-m3 (GPU, 1024d, Zep과 동일 계열) / `full`=API (text-embedding-3-small, 1536d).
- **차원 변경 = 컬렉션 재빌드**임을 1급 제약으로 문서화 (Nemori/Qdrant 교훈). 인덱스 메타에 모델명+차원 기록, 불일치 시 명시적 에러.

**top-k 정책**: 메모리 타입별 기본값 분리 — episodic k=10, semantic k=20 (Nemori), strategy k=1~2 (ReasoningBank), facts/edges k=10 (Zep). 전부 config 노출.

## 2. Graph

### 2.1 조사에서 확인된 사실

- 그래프 구현 스펙트럼: **networkx+pickle** (G-Memory) ↔ **프로덕션 그래프 DB** (Graphiti: Neo4j 5.26+/FalkorDB/Neptune; Kuzu는 upstream 중단으로 deprecated).
- Graphiti의 안전장치: LLM이 Cypher를 직접 생성하지 않고 **사전 정의된 쿼리**만 실행.
- Graphiti known issues: nested dict 속성 시 Neo4j TypeError(#683), silent episode drop(#1021), 로컬 LLM 라우팅 버그(#1116).
- bi-temporal (valid_at/invalid_at + transaction time) + **edge invalidation은 삭제가 아닌 무효화** — 감사 가능성.
- community detection은 Leiden이 아닌 **label propagation** (점진적 배정 가능) + 주기적 전체 refresh.
- 그래프 추상화의 비용: verbatim recall 퇴화 (Zep single-session-assistant -9~-18%).

### 2.2 우리 스펙

**GraphStore 인터페이스** (전 구현 공통):

```python
class GraphStore(Protocol):
    def upsert_node(self, node: Node) -> str
    def upsert_edge(self, edge: Edge) -> str          # bi-temporal 필드 포함
    def invalidate_edge(self, edge_id, t_invalid)      # 삭제 아닌 무효화
    def neighbors(self, node_id, hops=1, edge_filter=None) -> list[Node]
    def search_nodes(self, embedding, k) -> list[Node]
    def communities(self) -> list[Community]
    def run_named_query(self, name: str, params) -> Any   # 사전 정의 쿼리만
```

**구현 우선순위:**

| 구현 | 백엔드 | 프로파일 | 비고 |
|---|---|---|---|
| `SqliteGraphStore` | SQLite (nodes/edges 테이블 + 재귀 CTE k-hop) | lite 기본 | networkx 인메모리 캐시 병용. G-Memory 수준엔 충분 |
| `KuzuGraphStore` | Kuzu embedded | lite/standard 옵션 | upstream 중단 리스크 명시 (Graphiti도 deprecated 처리) |
| `Neo4jGraphStore` | Neo4j (Docker) | full | Graphiti 호환 스키마 지향. 속성은 primitive만 (교훈 #683) |
| `FalkorDBGraphStore` | FalkorDB (Redis) | full 옵션 | Graphiti MCP compose 기본값과 동일 |

**스키마 (Zep 3-subgraph 차용 + 단순화):**
- `Episode` (원문, 불변) / `Entity` (dedup 대상) / `Fact` (entity 간 edge, bi-temporal) / `Community` (label propagation).
- 모든 파생 노드는 `source_episode_ids` provenance 유지 (Nemori `source_episode_id` 교훈).
- **원문 episode는 어떤 방법론에서도 삭제/수정 금지** — verbatim 퇴화 방어의 1차 수단.

## 3. Latency

### 3.1 조사에서 확인된 실측

| 시스템 | read latency | write 비용 |
|---|---|---|
| Zep | search+생성 2.58~3.20s (full-context 28.9~31.3s) | episode당 LLM 4–6+ |
| Nemori | search 787ms, 총 3.05s | episode당 ~3 (LoCoMo 전체 373 calls) |
| MemoryOS | **search 9.9s** (계층 순회 비용), 총 15.2s | 응답당 평균 4.9 calls (배치 몰림) |
| LangMem | search 19.8s | 920 calls |
| A-Mem | search 947ms, 총 2.87s | note당 동기 2 (턴 블로킹, issue #21) |
| ACE | retrieval 없음 (playbook 주입) | 적응 latency GEPA 대비 -82%, DC 대비 -91% |

### 3.2 우리 스펙

**Latency budget (목표, lite 프로파일 / 로컬 0.6B LLM 기준):**

| 경로 | 목표 | 근거 |
|---|---|---|
| read: recall(dense+lexical 병렬) | < 100ms @ 100k items | sqlite-vec/LanceDB 실측으로 검증 |
| read: rerank (MMR/RRF) | < 20ms | 순수 연산 |
| read: rerank (LLM/cross-encoder) | < 1.5s | capability-gated, 기본 off in lite |
| read 총합 (p95) | **< 2s** | Zep/Nemori 수준 |
| write: enqueue | < 10ms (동기 구간) | fire-and-forget |
| write: 조직화 완료 | best-effort (background) | 아래 비동기 설계 |

**비동기 write 설계 (모든 방법론 공통 골격):**

```
add(episode) ──sync──> raw store 기록 + buffer
                └─async─> OrganizerQueue → [방법론별 Organizer 파이프라인]
                                            (LLM 추출/링크/증류/그래프 반영)
```

- **즉시 검색 가능성 보장**: raw episode는 enqueue 직후부터 dense/lexical 검색에 노출. 조직화된 표현(entity/semantic/strategy)은 준비되는 대로 추가 노출. → "write가 느려도 read는 항상 동작".
- 큐는 SQLite 기반 단일 프로세스 큐(lite) / Redis(full). 워커 수 = min(2, cpu_cores-2) in lite.
- **LLM 호출 수를 방법론별 스펙으로 명문화**하고 벤치 하네스가 calls/tokens/latency를 자동 기록 (MemoryOS Table 3 / Nemori Table 3–4 형식 재현).

## 4. Cold-start

### 4.1 문제 정의 (조사 기반)

세 종류의 cold-start가 있고 각각 대응이 다름:

1. **시스템 콜드스타트** — 프로세스 기동 시 모델/인덱스 로딩.
   - lite에서 embedder(수백 MB)+LLM(0.6B, ~1.2GB) 로딩이 수십 초 가능. GPU 6GB에 둘 다 상주시켜야 함.
2. **메모리 콜드스타트** — 새 사용자/에이전트는 메모리가 비어 있어 초기 이득 0.
   - ReasoningBank/G-Memory류는 태스크 수십 개를 겪어야 insight가 쌓임 (G-Memory: threshold 5개부터 finetune 시작).
   - ACE offline 모드가 정확히 이 문제의 해법: 사전 학습 셋으로 playbook을 미리 구축(warm-start) 후 배포.
3. **벤치마크 콜드스타트** — LongMemEval 평가는 히스토리 전체(115k~1.5M tokens)를 먼저 ingest해야 함. Zep식 4–6 calls/episode면 500 세션 ingest에 수천 콜.

### 4.2 우리 스펙

- **시스템**: 모델 서버 분리 상주 (vLLM/llama.cpp 데몬, OpenAI-compatible). 라이브러리가 모델을 직접 로드하지 않음 → 프로세스 재시작해도 콜드스타트 없음. MCP 서버도 같은 데몬을 바라봄. `capabilities` 감지에 endpoint 헬스체크 포함.
- **메모리**: 모든 방법론에 `warm_start(corpus)` API를 공통 제공 — ACE offline 학습, ReasoningBank 사전 궤적 증류, 대화 히스토리 백필(backfill)을 같은 진입점으로. 백필은 배치 세그멘테이션(Nemori `BATCH_SEGMENTATION_PROMPT` 방식)으로 콜 수 절감.
- **벤치마크**: ingest 결과물(구축된 메모리 상태)을 **아티팩트로 직렬화/캐시** — 같은 히스토리에 대해 retrieval/rerank/생성 설정만 바꿔 재평가할 때 재-ingest 없이 재사용. (LongMemEval 그리드 서치가 수천 콜로 폭발하는 것의 방어책.)

## 5. Memory Storage

### 5.1 조사에서 확인된 선택지

| 시스템 | 저장소 | 평가 |
|---|---|---|
| MemoryOS | JSON + FAISS (별도 ChromaDB판) | 단순하지만 코어 4벌 복제 드리프트 |
| A-Mem | ChromaDB (기본 in-memory!) + dict | 프로세스 종료 시 유실 위험 |
| Zep/Graphiti | Neo4j/FalkorDB + 내장 fulltext | 프로덕션급, 무거움 |
| Nemori | **PostgreSQL(tsvector) + Qdrant 듀얼** | 프로덕션급 async, Docker 필수 |
| ACE | 텍스트 파일 (playbook) + jsonl 로그 | 충분 (메모리가 작음) |
| ReasoningBank | JSON 파일 + 임베딩 JSON | 연구 프로토타입 |
| G-Memory | Chroma + networkx pickle | 연구 프로토타입 |

### 5.2 우리 스펙

> **정책 (2026-07-17 사용자 지시로 확정)**: PC 제약을 이유로 실 스택 배선을
> 생략하지 않는다. **naive in-python 구현(브루트포스 numpy 등)은 런타임 기본값
> 금지** — 이 스터디의 목적 자체가 실제 백엔드 기술을 다뤄보는 것. 원 시스템의
> 엔진이 너무 무거우면 **같은 계열의 경량 실물로 대체**한다(예: Milvus →
> Milvus Lite/Qdrant local/LanceDB; Qdrant 서버 → qdrant-client local 모드;
> ChromaDB in-memory → PersistentClient+cosine 명시). 현재 배선된 어댑터:
> 벡터 `QdrantVectorStore`(local 모드, Nemori 계열) / `ChromaVectorStore`(#24
> cosine 교정판, A-Mem 계열) / `LanceDBVectorStore` / `SqliteVecStore`(vec0);
> 그래프 `KuzuGraphStore`(임베디드 실 그래프 엔진, lite/standard 기본) /
> `Neo4jGraphStore`(bolt 서비스 감지 시, full 기본) / `SqliteGraphStore`(최후 폴백);
> 문서 `PostgresDocStore`(pgserver 임베디드 실 PostgreSQL, tsvector lexical —
> Nemori 스택) / `SqliteDocStore`(단일 파일 기본).
> `NumpyVectorStore`는 후보에서 제외, 테스트 픽스처로만 잔존.

**단일 파일 SQLite를 lite의 진실 소스로:**

```
~/.agentic_memory/{namespace}/
├── memory.db          # SQLite: episodes, notes, entities, facts(bi-temporal),
│                      #   strategies, playbook_bullets, evolution_log(append-only),
│                      #   FTS5 가상 테이블(lexical), 큐 테이블
├── vectors/           # sqlite-vec 테이블(기본) 또는 LanceDB 디렉토리
└── artifacts/         # 벤치 ingest 캐시, playbook 스냅샷
```

- **evolution_log가 핵심 테이블**: 모든 방법론의 메모리 변경을 `(op, target_id, payload, actor, t_transaction)` append-only 행으로 기록. ACE delta ops / G-Memory rule ops / A-Mem evolution / Zep invalidation을 전부 이 로그 위의 재생(replay)으로 표현 → 디버깅/감사/스냅샷 복원 공짜.
- 벡터: `sqlite-vec`(기본, 단일 파일 유지) / `LanceDB`(standard, 컬럼형+버전닝) / `Qdrant`(full, Nemori 호환) 어댑터.
- **스토리지 예산**: A-Mem 실측 1K notes=1.46MB 선형 — 임베딩 384d float32 기준. 우리 목표 규모(개인 PC, ≤1M items)에서 벡터 ~1.5GB, 디스크 여유(600GB) 대비 무의미한 수준. RAM이 병목이므로 **벡터 인덱스는 mmap 기반**(sqlite-vec/Lance 모두 해당)을 조건으로 채택.
- 컬렉션 네임스페이스: `{user_id}/{agent_id}/{memory_type}` (Nemori 멀티테넌시 + Graphiti group_id 패턴).

## 6. 소형 모델(0.5B) 대응 스펙 — 횡단 관심사

조사에서 확인된 실패 모드와 4중 방어:

| 실패 모드 | 실증 | 방어 |
|---|---|---|
| strict JSON 미준수 | A-Mem evolution 무력화(1–3B), Graphiti ingestion 실패(공식 문서 경고) | ① 스키마 최소화(필드 수 ≤5, 중첩 금지) ② vLLM `guided_json`/outlines constrained decoding ③ 파싱 실패 시 재시도 1회 → 필드 단위 fallback → **명시적 drop 카운터** (silent skip 금지 — A-Mem 교훈) |
| 추출 품질 저하 | G-Memory/커뮤니티 재현 수치 하락 | ④ **역할별 모델 티어링**: boundary 탐지·keyword 추출은 0.5B, 증류·reflection은 상위 티어(4B/API)로 라우팅 가능하게 — `llm.{extract,distill,judge}` 분리 설정 |
| judge 신뢰도 | LongMemEval judge는 yes/no 단순 판정이지만 preference/abstention은 민감 | judge는 기본 API pin(`gpt-4o-2024-08-06`), 로컬 judge 사용 시 API judge와의 합치율 검증 리포트 의무화 |
| 학습으로 보완 | ReasoningBank-slm(Qwen3-1.7B) 사례 | 추출/분절/증류 태스크용 SFT 데이터셋을 대형 모델 distillation으로 구축 → 0.5B LoRA 학습 (roadmap Phase 4) |
