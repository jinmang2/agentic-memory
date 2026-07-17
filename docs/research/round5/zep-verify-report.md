# Zep-graph 충실도 검증 보고서

- 대상: `src/agmem/organizers/zep_graph.py`, `src/agmem/stores/sqlite_graph.py`, 배선(`src/agmem/retrieval/pipeline.py`, `scripts/exp_locomo_conv0.py`)
- 기준 1: Zep 논문 arXiv:2501.13956 v1 (PDF 원문 직접 추출·인용)
- 기준 2: Graphiti 공식 코드 github.com/getzep/graphiti, **HEAD `36918ce` (2026-07-17, 오늘)** shallow clone 후 코드 직접 대조
- 결론 요약: **docs/10의 ○(골격, 측정 금지) 등급과 "4-way 미포함" 판단은 유지가 옳다.** 누락 리스트 ①~⑤는 전부 여전히 유효하나, ②·③은 upstream이 그 사이 변경되어 서술 업데이트가 필요하고, 리스트에 없는 신규 누락 4건 + 배선 버그 1건을 발견했다.

---

## 1. 구현된 부분 4건 — 항목별 판정

### 1-1. Entity 추출 프롬프트 의미론 — **부분 일치 (◑)**

우리 (`zep_graph.py:54-60`): "Extract the distinct real-world entities … include the speaker; at most 5", 이전 메시지 컨텍스트 없음, 자유형 type 문자열.

논문 §2.2.1:
> "the system processes both the current message content and the last n messages to provide context … n = 4, providing two complete conversation turns" … "the speaker is automatically extracted as an entity. Following initial entity extraction, we employ a reflection technique inspired by reflexion[12]"

upstream `prompts/extract_nodes.py:130`:
> "1. **Speaker Extraction**: Always extract the speaker (the part before the colon `:` in each dialogue line) as the first entity node."

- 일치: 스피커 추출 지시, "관계/행동은 노드로 만들지 말 것"의 취지(우리는 명시 안 하지만 fact 단계 분리로 구조상 유사).
- 불일치:
  - **이전 메시지 컨텍스트 부재** — 논문 n=4, 현 upstream은 `add_episode`에서 `last_n=RELEVANT_SCHEMA_LIMIT`(=10) previous episodes를 `<PREVIOUS MESSAGES>`로 전달(`graphiti.py:1088-1090`). 우리는 단일 메시지만 보므로 대명사 해소("he/she" → 이름)가 불가능. upstream 프롬프트는 "Pronoun references … should be disambiguated"를 명시(`extract_nodes.py:115`).
  - **`maxItems: 5` 상한은 우리 임의 추가** — 논문·upstream 어디에도 개수 상한 없음.
  - 현 upstream 프롬프트는 방대한 negative 규칙(대명사·추상개념·bare noun 금지, "Nisha's dad" 소유자 한정, 구체성 규칙)과 entity type ontology 분류(`entity_type_id`)를 갖는데 우리는 전무. 다만 이 negative 규칙 대부분은 2025~26년에 추가된 것으로 논문 v1 부록(§6.1.1)에는 없음 — 논문 기준으로는 감점 폭이 작다.
  - 논문의 **reflexion(누락 entity 재확인) 단계**는 우리도 없지만, **현 upstream 코드에서도 제거됨**(`grep reflexion` 0건) → 미구현이 현 upstream과는 오히려 일치. 우선순위 낮음.

### 1-2. 임베딩 기반 resolution — **핵심 의미론 불일치 (✘) — docs/10 ②가 정확히 지적한 그 지점**

우리 (`zep_graph.py:106-110`): `name(:summary)` 임베딩 → vec top-1 → **score ≥ 0.85면 무조건 기존 노드로 병합, LLM 판정 없음**.

논문 §2.2.1:
> "the system embeds each entity name into a 1024-dimensional vector space. This embedding enables the retrieval of similar nodes through cosine similarity search … The system also performs a separate full-text search on existing entity names and summaries to identify additional candidate nodes. These candidate nodes, together with the episode context, are then processed through an LLM using our entity resolution prompt. When the system identifies a duplicate entity, it generates an updated name and summary."

현 upstream `utils/maintenance/node_operations.py`:
- `NODE_DEDUP_CANDIDATE_LIMIT = 15`, `NODE_DEDUP_COSINE_MIN_SCORE = 0.6` (line 64-65)
- `resolve_extracted_nodes` docstring: "Resolve nodes with semantic retrieval first, then deterministic and LLM dedup." (line 635)
- 흐름: ① cosine 후보 수집(이름만 임베딩, top 15, ≥0.6) → ② `_resolve_with_similarity` 결정적 판정 — 정규화 exact-name 일치, 또는 entropy gate 통과 시 MinHash/LSH fuzzy(`_FUZZY_JACCARD_THRESHOLD = 0.9`) → ③ 미해결분만 `dedupe_nodes.nodes` 프롬프트로 LLM 판정(`duplicate_candidate_id`, -1 = 신규; "Java 언어 vs Java 섬" 같은 동명이의 구분 예시 포함).

판정:
- **임베딩 유사도만으로 자동 병합하는 경로는 논문에도 현 upstream에도 존재하지 않는다.** 임베딩은 어디까지나 *후보 수집*이고, 병합 결정은 (a) 결정적 exact/fuzzy 문자열 일치이거나 (b) LLM이다. 우리 0.85 top-1 자동 병합은 "Java(언어) vs Java(섬)"류 오병합과, 표기가 다른 동일 entity("NYC" vs "New York City", cos < 0.85 가능) 분열을 모두 막지 못한다.
- docs/10 ②의 "fulltext 후보" 서술은 **논문 v1 기준으로는 정확**하나, **현 upstream은 fulltext 후보를 버리고 cosine 후보 + 결정적 exact/fuzzy로 대체**했다(아래 §3-② 참조).
- 추가: 우리는 dup 판정 시 기존 노드의 name/summary 갱신도 없음. 논문은 "generates an updated name and summary", 현 upstream은 `_promote_resolved_node` + `extract_attributes_from_nodes`(summary 재생성)를 수행.
- 미세 차이: 논문·upstream은 **entity name만** 임베딩("embeds each entity name"; upstream `_semantic_candidate_search`는 `node.name`으로 질의). 우리는 `"{name}: {summary}"`를 임베딩 — threshold의 의미 자체가 달라진다.

### 1-3. Bi-temporal fact 표현 — **부분 일치 (◑) + 문서-코드 불일치 발견**

논문 §2.2.3:
> "the system tracks four timestamps: t′created and t′expired ∈ T′ monitor when facts are created or invalidated in the system, while tvalid and tinvalid ∈ T track the temporal range during which facts held true."

- 스키마는 일치: `sqlite_graph.py` `graph_edges`에 `created_at / expired_at / valid_at / invalid_at` 4-타임스탬프 모두 존재. 물리 삭제 없음도 일치.
- **그러나 `expired_at`은 어디서도 기록되지 않는 죽은 컬럼이다.** `invalidate_edge`(line 87-90)는 `invalid_at`만 UPDATE. upstream `edge_operations.py:569-570`은 invalidation 시 두 축을 모두 기록한다:
  ```python
  edge.invalid_at = resolved_edge.valid_at
  edge.expired_at = edge.expired_at if edge.expired_at is not None else utc_now()
  ```
  따라서 `sqlite_graph.py` docstring의 *"both 'what was true then' and 'what we believed then' stay queryable"* 주장은 현재 **거짓**(T′ 축 소실). 문서-코드 불일치.
- `valid_at`은 항상 `ep.timestamp`(`zep_graph.py:134`) — 시간표현 파싱 부재(docs/10 ③)로 fact-시간 축 T가 사실상 ingestion-시간으로 퇴화. "2주 전에 이직했다" → valid_at이 발화 시점이 되는 오류. 논문이 명시한 상대/절대 시간표현 해석("I started my new job two weeks ago")이 핵심 차별점인데 없음.

### 1-4. LLM invalidation 의미론 — **취지 일치, 범위·가드 불일치 (◑)**

- 프롬프트 취지 일치: 우리 CONTRA_PROMPT "Which of the existing facts does it contradict (i.e. can no longer be true if the new fact is true)? Usually none." ↔ upstream `dedupe_edges.resolve_edge`의 contradiction 절반("Determine which facts the NEW FACT contradicts").
- `invalid_at := 새 fact의 ts` 설정은, 우리 valid_at이 항상 ep.ts이므로 결과적으로 논문의 "setting their tinvalid to the tvalid of the invalidating edge"와 수치상 일치(단 valid_at 자체가 부정확하므로 반쪽 일치).
- 불일치 3건:
  1. **후보 범위**: 우리는 같은 entity-pair의 edge만(`edges_between`) 검사. 논문은 "compare new edges against **semantically related** existing edges", 현 upstream은 same-pair 후보(duplicate용)와 **별도로 그래프 전역 `EDGE_HYBRID_SEARCH_RRF`(fact 텍스트로 BM25+cosine+RRF) invalidation 후보**를 수집해 continuous idx로 한 콜에 넘긴다(`edge_operations.py:392-430`). 같은 pair 밖의 모순(예: "Alice lives in Paris" vs "Alice–Berlin LIVES_IN")은 우리 구조로는 절대 못 잡는다.
  2. **시간 중첩 가드 부재**: 논문 "When the system identifies **temporally overlapping** contradictions, it invalidates…". upstream `resolve_edge_contradictions`(line 538-573)는 (기존 edge가 새 edge valid 이전에 이미 invalid) 또는 (새 edge가 기존 edge valid 이전에 invalid)이면 invalidation을 건너뛴다. 우리는 LLM이 지목하면 무조건 invalidate — 과거에 이미 끝난 사실을 소급 invalidate하는 오류 가능.
  3. **duplicate 판정과 미통합**: upstream은 한 콜에서 `duplicate_facts` + `contradicted_facts`를 동시 판정하고, "duplicate이면서 동시에 contradicted(제목만 갱신된 경우)"까지 다룬다. 우리는 contradiction만 있고 dedup이 아예 없음(→ §3-⑤).

---

## 2. docs/10 누락 리스트 ①~⑤ 재검증 (2026-07-17 upstream 기준)

| # | docs/10 항목 | 판정 | 비고 |
|---|---|---|---|
| ① | community subgraph (label propagation) | **유효** | upstream `community_operations.py:93-137` `label_propagation`(plurality, tie→큰 community, 수렴까지 반복) 그대로 존재. 추가로 **동적 확장** `update_community`(line 325, `add_episode(update_communities=True)` 시 노드별 이웃 plurality로 편입) + 주기적 full refresh(`build_communities`), map-reduce pair 요약(`summarize_pair`) — 논문 §2.2.4와 일치. docs/10의 "주기 refresh" 계획에 **동적 단일-스텝 확장**도 추가할 것 |
| ② | resolution의 LLM 판정 + fulltext 후보 | **유효하나 서술 업데이트 필요** | "LLM 판정 누락"은 여전히 핵심 갭. 단 **현 upstream은 fulltext 후보를 제거**하고 cosine 후보(15개, ≥0.6) → 결정적 exact/fuzzy(MinHash Jaccard 0.9 + entropy gate) → 미해결분 LLM의 3단계로 바뀜. 논문 v1 재현이 목적이면 fulltext+cosine, 현 upstream 재현이 목적이면 cosine+결정적+LLM. 어느 쪽이든 "임베딩 점수 단독 자동병합"은 근거 없음 |
| ③ | 시간표현 파싱 (t_valid/t_invalid) | **유효, 구현 형태 변경** | 별도 `extract_edge_dates` 프롬프트는 사라졌고, 현 upstream은 **edge 추출 프롬프트에 valid_at/invalid_at 필드를 통합**(REFERENCE_TIME으로 상대표현 해석, "ongoing이면 valid_at=episode ts", "연도만 있으면 1월 1일" 등 규칙 명문화, `extract_edges.py:168-175`) + 미설정 시 `extract_timestamps` 소형모델(fallback, `ModelSize.small`) 콜. 우리 구현 시 fact 추출 콜에 통합하는 편이 콜 수 절약 면에서도 upstream-일치 |
| ④ | GraphRecall(BFS) 파이프라인 배선 | **유효 — 최우선 유지 (동의)** | 상세는 §4 |
| ⑤ | fact dedup (hybrid) | **유효** | upstream 3겹: (a) 배치 내 `(src,dst,normalized fact)` exact dedup(`edge_operations.py:344-358`), (b) same-pair 기존 edge + `EDGE_HYBRID_SEARCH_RRF`(edge_uuids 필터로 same-pair 한정) 후보, (c) `resolve_edge` LLM이 duplicate 판정 → **기존 edge 재사용 + `episodes.append(episode.uuid)` provenance 누적**. 논문 §2.2.2 "The hybrid search for relevant edges is constrained to edges existing between the same entity pairs" 그대로. 우리는 동일 fact가 매번 새 edge로 무한 증식 |

**우선순위 조정 제안**: docs/10 순서(1 GraphRecall → 2 resolution → 3 temporal → 4 dedup → 5 community)는 대체로 유지하되, ⑤ dedup과 ③ temporal은 **한 콜에 통합 가능**(현 upstream이 정확히 그렇게 함: fact 추출 시 valid_at/invalid_at 동시 추출, resolve_edge 한 콜에서 dup+contradiction 동시 판정)하므로 2-3-4를 "write-path 1회 재설계"로 묶는 것이 콜 예산상 유리.

---

## 3. 신규 발견 (docs/10에 없는 알고리즘 갭)

1. **[배선 버그] invalidated fact가 검색에 그대로 노출된다.** `memory.py:237-242` INVALIDATE는 doc item에 `invalid_at`만 써넣고 **vec 인덱스에서 제거하지 않으며**, `pipeline.py`의 `_hydrate`/render 어디에도 invalid_at 필터·표시가 없다. 즉 모순으로 무효화된 옛 fact가 현재 fact와 나란히, 무효 표시 없이 프롬프트에 들어간다. Zep의 constructor χ는 "for each ei ∈ Es, χ returns the fact and **tvalid, tinvalid** fields"(논문 §3) — 컨텍스트 템플릿도 "FACT (Date range: from - to)". temporal 카테고리를 측정하는 순간 이 갭이 정면으로 결과를 오염시킨다. 최소 수정: 검색 시 invalid 제외 또는 date-range 병기.
2. **previous-episodes 컨텍스트 전무** (§1-1). 논문 n=4 / 현 upstream last 10. entity 추출·대명사 해소·resolution·fact 추출 전 단계 품질에 직결. LoCoMo류 대화 데이터에서 "he/she" 발화가 많아 실질 영향 큼.
3. **resolution 시 노드 name/summary 갱신 없음** (§1-2). 논문 "generates an updated name and summary". 우리 노드 summary는 최초 추출 시점 한 줄에서 영원히 동결 — entity 검색 채널("entities")의 임베딩 텍스트 품질이 시간이 갈수록 upstream 대비 열화.
4. **expired_at 미기록 → bi-temporal의 T′ 축 소실 + `sqlite_graph.py` docstring 허위** (§1-3). 한 줄 수정(invalidate 시 `expired_at=now`)으로 해결 가능.
5. (minor) 논문의 hyper-edge("the same fact can be extracted multiple times between different entities") 미지원, predicate 표기 SCREAMING_SNAKE_CASE(upstream) vs snake_case(우리), fact `maxItems: 5` 임의 상한.
6. (참고, 갭 아님) 현 upstream에는 논문 이후 추가물이 있음 — combined nodes+edges 단일 콜 추출(`combined_extraction.py`, bulk 경로), SagaNode, entity attributes ontology. 논문 재현 범위 밖이므로 추적만.

---

## 4. Read-path 배선 검증: 현 설정 vs Zep 실제 검색

현 설정 (`exp_locomo_conv0.py:144-145`): `("episodic","facts","entities")`, k=10, keyword_queries=False → `pipeline.py`에서 **facts/entities는 dense cosine 단독**(lexical은 episodic에만), RRF는 사실상 단일 랭킹 통과, reranker=None, BFS 없음, communities 없음.

Zep 실제 (논문 §3.1-3.2 + upstream `search/`):
- 검색 3함수: ϕ_cos + ϕ_bm25(Okapi BM25, Lucene) + **ϕ_bfs**("identifying additional nodes and edges within n-hops … using recent episodes as seeds", upstream `MAX_SEARCH_DEPTH = 3`, `EdgeSearchMethod.bfs`/`node_bfs_search`).
- 검색 필드: edge는 fact 텍스트, node는 entity name, community는 community name.
- reranker: RRF / MMR / episode-mentions / node-distance / cross-encoder. upstream 기본: 단순 `search()`는 `EDGE_HYBRID_SEARCH_RRF`, 고급 `search_()` 기본값은 `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`. `DEFAULT_SEARCH_LIMIT = 10`.
- constructor: facts(+date range) / entity name: summary / community summary 템플릿.

**거리 평가: 현 배선은 Zep 검색이 아니라 "그래프 산출물을 평면 벡터 RAG로 읽는 것"이다.** 3개 검색 함수 중 1개(cos)만, reranker 5종 중 0종, 3-tuple(edges, nodes, communities) 중 2개를 dense로만. 특히 Zep의 검색 우위 주장 근거인 ϕ_bfs(그래프 구조 활용)와 hybrid가 전부 빠져 있어, 이 설정으로 측정하면 Zep 방법론이 아닌 것을 Zep이라 부르게 된다. docs/10의 "측정 금지" 판단 재확인.

**충실한 최소 read-path (측정 해금 조건)**:
1. facts: fact 텍스트에 cosine + BM25(현 `search_lexical`을 facts에도) → RRF (upstream 기본 recipe와 동일)
2. entities: entity name에 cosine + BM25 → RRF
3. **BFS**: 최근 episodes가 언급한 entity 노드를 seed로 `sqlite_graph.neighbors`(이미 recursive CTE 존재, hops≤3) → 그 노드들의 active edge를 후보에 합류 — `recall.py`에 GraphRecall로 배선(docs/10 계획 1과 일치)
4. rerank: 최소 RRF(3-way 융합)로 논문의 ρ 성립. cross-encoder는 선택(논문 실험은 BGE-m3 계열 사용 — 로컬 프로파일과도 호환 가능)
5. constructor: fact에 `(Date range: {valid_at} - {invalid_at})` 병기 + invalidated 처리(§3-1), entity는 `NAME: summary`
6. (community 구현 후) community name cosine+BM25 채널 추가

---

## 5. 권고 요약

1. docs/10 ②·③ 서술 업데이트: ② "fulltext 후보"에 "(논문 v1; 현 upstream은 cosine 후보 15/0.6 + exact/fuzzy MinHash + LLM escalation)" 병기, ③ "별도 콜"이 아니라 "fact 추출 프롬프트 통합 + fallback 콜"로.
2. 신규 갭 4건(§3-1~4)을 docs/10 누락 리스트에 ⑥~⑨로 추가. 특히 **⑥ invalidated fact 검색 노출**은 zep 측정 해금의 전제조건이자 한나절 수정거리이므로 우선순위 1.5로 삽입 권고.
3. `sqlite_graph.py` docstring의 "what we believed then queryable" 주장은 `invalidate_edge`에 `expired_at=now` 추가 전까지 수정 또는 구현.
4. 측정 금지 유지. 해금 조건: §4의 최소 read-path(1-5) + write-path의 resolution LLM 판정·temporal 통합 추출·fact dedup.

## 부록: 근거 위치
- 우리: `zep_graph.py:106-110`(0.85 자동병합), `:134`(valid_at=ep.ts), `:141-156`(same-pair contradiction), `sqlite_graph.py:87-90`(expired_at 미설정), `memory.py:237-242`(INVALIDATE), `pipeline.py:44-56`(dense-only), `exp_locomo_conv0.py:144-145`
- upstream(36918ce): `node_operations.py:64-65,418-450,627-708`, `dedup_helpers.py:31-36,220-260`, `edge_operations.py:344-430,538-573,576-621,688-776`, `extract_edges.py:94-176,242-301`, `dedupe_edges.py:43-100`, `dedupe_nodes.py:117-179`, `community_operations.py:93-137,259-331`, `search_config.py:29-35`, `search_utils.py:67`, `search_config_recipes.py:34-146`, `graphiti.py:1088-1090,1184-1190,1545-1610`
- 논문: §2.2.1(추출·resolution·reflexion·n=4), §2.2.2(hyper-edge·same-pair dedup), §2.2.3(4-timestamp·invalidation), §2.2.4(label propagation·동적 확장), §3-3.2(ϕ/ρ/χ·BFS·reranker), 부록 §6.1.2/6.1.4(프롬프트 원문)
