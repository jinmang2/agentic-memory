# 7차: Zep community 구현 + read-path 결함 2건 (2026-07-27)

> 범위: 6차 감사가 남긴 "다음" 목록의 2번(Zep 재등급)과 C3(다음: Zep). 해금 조건으로
> 남아 있던 것은 community(label propagation + 동적 확장) 하나였고, 그것을 구현하면서
> read-path를 실제로 실행해 본 결과 **측정 이전에 이미 죽어 있던 채널 2건**을 발견했다.
> 업스트림은 당일 raw 재다운로드로 대조:
> `getzep/graphiti:graphiti_core/{utils/maintenance/community_operations.py,
> prompts/summarize_nodes.py, graphiti.py, graph_queries.py, utils/text_utils.py,
> search/{search.py,search_utils.py,search_config.py,search_config_recipes.py,search_helpers.py}}`.

## 판정 요약

| # | 항목 | 성격 | 저장된 수치에 영향 | 조치 |
|---|---|---|---|---|
| R1 | entity 아이템에 `content`가 없어 **렌더가 빈 문자열** | read-path 결함 | 없음 (zep 측정 금지) | **수정** |
| R2 | 같은 원인으로 **entities BM25 채널이 빈 인덱스** | read-path 결함 | 없음 | **수정** |
| R3 | community subgraph 부재 (round-5 ①) | 미구현 | 없음 | **구현** |
| U1 | 업스트림 label propagation이 **무한 루프**에 빠지는 입력 존재 | upstream 결함 | — | 재현 불가(정지 조건 추가) |
| U2 | 우리 zep read 설정이 업스트림 **두 레시피의 혼종** | 재현 대상 미결정 | 없음 | **해소** — 논문 §4.1이 확정, 레시피 표로 전량 구현 |
| R4 | φ_bfs가 랭킹 채널이 아니라 flat append(GraphRecall) | 충실도 결함 | 없음 | **수정** (채널화) |
| R5 | reranker가 후보 수 ≤ k일 때 **실행되지 않음** | 파이프라인 결함 | 없음 (측정런 전부 Noop) | **수정** |

---

## R1/R2. entity 채널이 두 방향 모두 비어 있었다

### 무엇이 틀어졌나

`ZepGraphOrganizer`가 내보내는 entity 아이템 payload는 이랬다:

```python
{"id", "name", "summary", "entity_type", "source_episode_ids", "embedding_text"}
```

`content` 키가 없다. 그런데 이 값을 읽는 곳이 둘이다:

1. **렌더** — `retrieval/steps.py::_DictItem.__init__`이 `self.content = data.get("content", "")`.
   `render()`는 `f"{head}{self.content}{stamp}"`를 첫 줄로 만든다.
2. **BM25 인덱스** — `stores/sqlite_doc.py::put_item`이 `str(data.get("content") or "")`를
   `items_fts`에 넣는다. 빈 문자열이면 **행 자체를 넣지 않는다**.

실측:

```
>>> _DictItem({"id":"e1","name":"Melanie","summary":"a painter in Seattle"}).render()
''
```

즉 `memory_types=(..., "entities")`로 검색하면 번들에 `- `(빈 불릿)만 k개 들어가고,
`lexical_types`에 `entities`를 넣어도 **매칭될 문서가 0건**이다.

### 피해 범위

zep_graph는 docs/09/10에서 측정 금지(○)라 저장된 수치는 없다. 다만 이것이 의미하는 바가
중요하다 — round-5 zep 보고서 §4가 정의한 "충실한 최소 read-path" 6항목 중 2번
(`entities: cosine + BM25 → RRF`)과 5번(`entity는 NAME: summary`)이 **코드상 존재했지만
실행되지 않고 있었다**. 해금 조건을 체크리스트로만 봤다면 통과했을 것이다.

### 수정과 업스트림 근거

업스트림은 두 채널의 텍스트가 **서로 다르다**:

| 채널 | 업스트림 | 근거 |
|---|---|---|
| dense | entity **name**만 임베딩 | `_semantic_candidate_search`가 `node.name`으로 질의 |
| BM25 | **(name, summary)** 복합 인덱스 | `graph_queries.py:134` `node_name_and_summary` |
| 렌더(χ) | `{entity_name, summary}` | `search_helpers.py::search_results_to_context_string` |

따라서 `content = "name: summary"`(BM25 + 렌더 동시 충족), `embedding_text = name`(dense).
merge 판정으로 name/summary가 갱신되는 UPDATE 경로에도 같이 실었다 — 안 그러면 병합 후
아이템의 렌더 텍스트가 옛 이름으로 굳는다.

community는 이 셋이 한 필드로 안 모인다(BM25는 name만, χ는 name+summary). 그래서
`put_item`에 **`lexical_text`** 를 추가했다 — `embedding_text`가 dense 채널에 대해 하는 일을
lexical 채널에 대해 하는 필드다. 없으면 BM25 텍스트가 "렌더가 원하는 것"에 영구히 묶인다.

**회귀 테스트**: `test_zep_entity_items_carry_content_for_render_and_bm25` —
`content`/`embedding_text`가 서로 다른 값임을 단언하고, `search_lexical_items`가 비어 있지
않음과 렌더 문자열을 함께 확인한다.

---

## R3. community subgraph 구현 (round-5 ① 해소)

### 업스트림 구조

| 단계 | 업스트림 | 우리 |
|---|---|---|
| 클러스터링 | `label_propagation(projection)` — projection은 `{node: [(neighbor, edge_count)]}` | `zep_graph/community.py::label_propagation` 전사 |
| 요약 | `build_community`: 멤버 summary들을 **map-reduce 페어 요약**(앞 절반 × 뒷 절반, 홀수는 이월) | 동일 (`_reduce_summaries`) |
| 이름 | `generate_summary_description(summary)` — 요약을 설명하는 **한 문장** | 동일 (`_describe`) |
| 저장 | Community 노드 + `HAS_MEMBER` 엣지, name을 임베딩 | graph에 동일 구조, 단 **MemoryOp 경유** |
| 전체 재계산 | `Graphiti.build_communities()` = `remove_communities()` 후 전량 재생성 | `flush_buffer` (기본 on) |
| 점진 확장 | `add_episode(update_communities=True)` (기본 **False**) | `update_communities=True` (기본 False, 업스트림과 동일) |

프롬프트 2종은 업스트림 원문에서 옮겼다. 특히 `summarize_pair`의 negative 규칙
("mentioned/described/stated ... 같은 filler 동사 금지")은 유지했다 — N개 요약을
map-reduce로 접을 때 "여러 주제를 논의한다"류로 수렴하는 것을 막는 장치라서 뺄 수 없다.

### op 로그 경유 (6차 B3의 교훈 적용)

community도 **organizer가 스토어에 직접 쓰지 않는다.** `ADD(communities, ...)` payload에
`member_ids`를 실어 `AgenticMemory._apply_graph`가 노드 upsert + 멤버십 전량 교체를 수행하고,
사라진 community는 `DELETE` op → `_apply_one`이 그래프에서도 제거한다. fact 엣지에
`subject_id`/`object_id`를 실은 것과 같은 이유 — **적용 시점에 복원 불가능한 정보는 op가
들고 있어야 로그만으로 그래프가 재생된다.**

**회귀 테스트**: `test_zep_communities_are_built_at_flush_and_replay_from_the_log` —
zep을 모르는 passthrough 메모리에 op만 재생해서 community 수·멤버 수가 일치함을 단언.

### 의도적 편차 1건: 멤버십 해시 id

업스트림은 `build_communities()`마다 community를 **전부 지우고 새 uuid로 다시 만든다**.
우리는 id를 `sha1(sorted(member_ids))`로 유도한다. 이유 둘:

1. community 하나당 LLM 콜이 `len(members)-1 + 1`회다. 파티션이 안 움직였는데 재계산하면
   그 비용을 통째로 재지출한다. (`flush()`와 `consolidate()`가 모두 `flush_buffer`를 부르므로
   한 런에서 2회 호출이 정상 경로다.)
2. append-only 로그에 flush마다 community 수 × (DELETE + ADD)가 쌓인다.

멤버가 같으면 스킵하므로 2회차 flush는 **op 0건 · LLM 콜 0회**다(테스트로 단언).
점진 확장은 community의 기존 id를 유지하므로 해시 불변식이 깨지는데, 그것이 곧 다음 전체
재계산이 그 community를 다시 만드는 트리거가 된다 — 논문이 주기적 재계산을 두는 이유
(drift 교정)와 같은 방향이라 그대로 뒀다.

### 비용

LoCoMo conv0 규모에서 entity가 n개면 전체 재계산 1회에 대략 **n + (클러스터 수)** 콜이다.
고립 노드도 자기 자신만의 community가 되어 naming 콜 1회를 쓴다(업스트림 동일 —
`get_by_group_ids`가 모든 entity를 projection에 넣고, 1-노드 클러스터는 페어 요약 없이
naming만 한다). 측정 전에 예산을 잡을 때 **write 비용이 대략 2배가 된다**고 봐야 한다.

---

## U1. 업스트림 label propagation은 특정 입력에서 영원히 끝나지 않는다

### 메커니즘

```python
while True:                        # 종료 조건: no_change 뿐
    for uuid, neighbors in projection.items():
        ...
        community_lst.sort(reverse=True)
        candidate_rank, community_candidate = community_lst[0] if community_lst else (0, -1)
        if community_candidate != -1 and candidate_rank > 1:
            new_community = community_candidate
        else:
            new_community = max(community_candidate, curr_community)
        new_community_map[uuid] = new_community      # ← 갱신이 SYNCHRONOUS
    community_map = new_community_map
```

모든 노드를 **같은 스냅샷**에서 재배정한다(synchronous label propagation). 이 변종은
수렴이 보장되지 않고 2-cycle에 빠질 수 있다는 것이 알려진 성질이다.

가장 작은 반례: **엣지 2개로 연결된 2-노드 컴포넌트.**

```
projection = {"c": {"d": 2}, "d": {"c": 2}}     # c=label 2, d=label 3
round 1: c는 d의 label(3)을 weight 2로 봄 → 2 > 1 → c := 3
         d는 c의 label(2)을 weight 2로 봄 → 2 > 1 → d := 2
round 2: 원위치. 이후 무한 반복.
```

엣지가 1개면 `candidate_rank > 1` 게이트에 걸려 `max(candidate, curr)` 경로로 빠지므로
수렴한다. 즉 **"두 엔티티 사이에 fact가 2개 이상이고 다른 연결이 없는" 컴포넌트**가
정확히 함정이며, 대화 데이터에서 흔한 모양이다.

### 우리 처리

`while True` 대신 **2-cycle 감지**(새 배정 == 2라운드 전 배정)와 `max_rounds=100` 백스톱을
두고, 둘 다 warning으로 남긴다. 정지해도 **병합을 발명하지는 않는다** — 사이클의 어느
상태에서도 두 노드의 label이 다르므로 각자 싱글턴 community가 된다. 이것이 "발표된 규칙이
이 컴포넌트를 클러스터링하지 못한다"는 정직한 독해다(업스트림은 아예 답을 내지 못하므로,
합치는 쪽으로 정하면 업스트림에 없는 파티션을 만드는 것이 된다).

### 업스트림 docstring이 코드와 다른 점 2가지 (재현함)

1. *"Ties are broken by going to the largest community"* — 실제로는 `(count, label)` 튜플을
   내림차순 정렬해 **label 숫자가 큰 쪽**을 고른다. label은 열거 인덱스라 크기와 무관하다.
2. *"take on the community of the plurality of its neighbors"* — weight 1짜리 plurality는
   **이기지 못한다**(`candidate_rank > 1`). 리프 노드는 대개 자기 label을 유지한다.

**회귀 테스트**: `test_label_propagation_reproduces_upstream_weight_gate_and_isolates` —
weight 1 쌍/고립 노드/weight 2 진동을 한 projection에 넣고 파티션을 단언한다.

---

## U2. 우리 zep read 설정은 업스트림 두 레시피의 혼종이었다 → 논문이 확정, 전량 구현

`search_config_recipes.py`를 grep한 결과가 결정적이다:

```
grep -n "bfs" search_config_recipes.py
80:  # ... and bfs with cross_encoder reranking          ← COMBINED_HYBRID_SEARCH_CROSS_ENCODER
86:  EdgeSearchMethod.bfs,
94:  NodeSearchMethod.bfs,
148: EdgeSearchMethod.bfs,    ← EDGE_HYBRID_SEARCH_CROSS_ENCODER
193: NodeSearchMethod.bfs,    ← NODE_HYBRID_SEARCH_CROSS_ENCODER
```

**BFS는 cross-encoder 레시피에만 있다.** RRF 레시피(`COMBINED_HYBRID_SEARCH_RRF`,
`EDGE_HYBRID_SEARCH_RRF`)의 `search_methods`는 `[bm25, cosine_similarity]` 둘뿐이다.

우리 `zep_graph` config는 RRF 융합(= RRF 레시피)에 `GraphRecall`(= BFS 채널) 을 얹은
상태다. 게다가 GraphRecall은 **랭킹 채널이 아니라** 최상위 점수의 0.9배로 flat하게
append한다 — 업스트림에서 BFS는 다른 채널과 동등하게 융합된다.

시딩 자체는 충실하다: 업스트림 `search()`도 명시적 origin이 없으면 **다른 채널이 이미
찾은 노드**에서 origin을 만든다(`search.py:540` `origin_node_uuids = [node.uuid for result
in search_results for node in result]`). 우리 시드(검색된 entity 히트)와 같다.

### 논문이 이미 답을 갖고 있었다 (§4.1)

처음에는 "측정 착수 시점의 결정 사항"으로 유예했다. **틀렸다** — 논문 본문에 답이 있다:

> **4.1 Choice of models** — "Our experimental implementation employs the **BGE-m3 models
> from BAAI for both reranking and embedding tasks**."

BGE-m3의 reranker(`BAAI/bge-reranker-v2-m3`)는 **cross-encoder**이고, 업스트림도 정확히 그렇게
배선한다 (`cross_encoder/bge_reranker_client.py`: `CrossEncoder('BAAI/bge-reranker-v2-m3')`).
따라서 DMR·LongMemEval 수치는 **cross-encoder 레시피**에서 나왔고, 그 레시피가 곧 BFS 채널을
가진 유일한 계열이다. 즉 A/B 중 하나를 고르는 문제가 아니라 **A가 논문의 operating point,
B(RRF)가 업스트림의 기본값**이라는 서로 다른 두 사실이었다.

동시에 논문 §4는 스코프를 스스로 제한한다:

> "While these experiments demonstrate key retrieval capabilities of Graphiti, they
> represent a **subset** of the system's full search functionality."

그리고 §3.1–3.2는 검색함수 3종(φ_cos, φ_bm25, φ_bfs)과 reranker 5종(RRF, MMR,
episode-mentions, node-distance, cross-encoder)을 **전부 시스템 구성요소로** 제시한다.
논문에는 이들에 대한 ablation이 없으므로, 어떤 레시피도 "논문의 ablation"을 주장할 수 없고
**"논문의 operating point" / "업스트림 기본값" / "논문이 서술하지만 측정하지 않은 메커니즘"**
세 부류로만 말할 수 있다.

### 조치: 레시피를 프리셋 표로 (Nemori·MemoryOS 계보와 같은 구조)

`organizers/zep_graph/search.py`에 `ZEP_SEARCH_RECIPES`를 신설했다. 업스트림
`search_config_recipes.py`를 필드 단위로 전사한 데이터 표이고, `SearchRecipe.config_kwargs()`가
`AgmemConfig` 필드로 환산한다. 읽기 경로만 담으므로 **한 번의 ingest를 여러 레시피로 측정**할
수 있다.

| 레시피 | 업스트림 상수 | BM25 | BFS | reranker | 성격 |
|---|---|---|---|---|---|
| `cross_encoder` (기본) | `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` | 3종 | facts·entities | `bge-reranker-v2-m3` | **논문 §4 operating point** |
| `rrf` | `COMBINED_HYBRID_SEARCH_RRF` | 3종 | 없음 | RRF(=Noop) | 업스트림 기본값 |
| `mmr` | `COMBINED_HYBRID_SEARCH_MMR` | 3종 | 없음 | MMR(λ=1) | §3.2, 미측정 |
| `edge_episode_mentions` | `EDGE_HYBRID_SEARCH_EPISODE_MENTIONS` | facts | 없음 | episode-mentions | §3.2, 미측정 |
| `node_distance` | `NODE_HYBRID_SEARCH_NODE_DISTANCE` | entities | 없음 | node-distance | §3.2, 미측정 |
| `edge_rrf` | `EDGE_HYBRID_SEARCH_RRF` | facts | 없음 | RRF | 업스트림 `search()` |

주의할 전사 디테일 3건:
1. **communities에는 어느 레시피에도 BFS가 없다** — `CommunitySearchMethod`에 `bfs` 멤버가
   아예 없다. cross-encoder 레시피도 edges/nodes에만 BFS를 준다.
2. 업스트림 MMR 레시피는 `mmr_lambda=1`, 즉 **다양성 항이 꺼진 상태**로 출하된다.
3. `episodic`(원문 메시지)은 어떤 레시피에도 없다 — Zep의 컨텍스트 템플릿에 원문 섹션이
   없으므로, 넣으면 docs/10이 재현 인용 금지로 표시한 혼합 조건이 된다.

### R4. φ_bfs를 랭킹 채널로 (GraphRecall 대체)

논문 §3.1은 BFS를 cosine·BM25와 **나란한 검색 함수**로 두고, 셋이 함께 reranker ρ에 들어간다.
우리 `GraphRecall`은 후처리 스텝이라 최상위 점수의 0.9배로 flat하게 append했다 — 랭킹이 아니라
상수였다. `retrieval/bfs.py`를 신설해 **채널**로 만들었다:

- origin: 업스트림 `search()`와 동일하게 **같은 타입의 다른 채널 결과**에서 유도
  (facts는 찾은 엣지의 **subject 노드만** — upstream `source_node_uuid`; entities는 찾은 노드
  자신). 논문이 언급한 "recent episodes as seeds"는 업스트림의 명시적 origin 경로라 미재현.
- 정렬: hop 거리 우선(가까울수록 "contextually similar" — 논문의 근거 그대로), 동순위는
  content 유도 키. 업스트림은 드라이버 행 순서라 결정적이지 않은데, RRF는 **rank만** 쓰므로
  그대로 두면 같은 질의가 런마다 달라진다.
- `graph_expansion_cap` 기본값을 **0으로 내렸다**. GraphRecall은 φ_bfs의 대역이었고, 채널과
  동시에 켜면 같은 엣지를 두 메커니즘으로 이중 서빙한다.

**회귀 테스트**: `test_bfs_channel_depth_reaches_the_second_ring` — A-B-C 체인에서 e2의 벡터를
지워 **그래프 경로만으로 도달 가능**하게 만든 뒤, depth 1은 e1만, depth 2는 e2까지 끌어옴을
단언한다(벡터를 안 지우면 dense가 둘 다 반환해 BFS 없이도 통과한다).
`test_graph_recall_is_off_unless_asked_for`가 기본 비활성을 고정한다.

### R5. reranker가 후보 수 ≤ k이면 실행되지 않았다

`if self.reranker is not None and len(fused) > type_k:` — "버릴 게 있을 때만 rerank한다"는
전제였다. reranker는 **순서도** 결정하고, 순서는 절단 후에도 살아남는다:
`MemoryBundle.render`가 번들 전체를 점수로 정렬하므로, rerank를 건너뛴 타입은 RRF 점수를
유지하고 rerank된 타입은 relevance 점수를 갖는데 그 둘이 한 예산에서 비교된다. 업스트림은
무조건 rerank 후 절단한다. 게이트를 `len(fused) > 1`(0·1개는 재배열 대상 자체가 없음)로 고쳤다.

**저장 수치 무영향**: 측정된 모든 런이 profile `lite` → `NoopReranker`로 해석되고, 그 rerank는
절단이다. 즉 이 결함은 reranker를 실제로 쓰는 순간(=지금 추가한 레시피들) 발현할 잠복 결함이었다.

**회귀 테스트**: `test_reranker_runs_even_when_candidates_fit_in_k` — fact 3개·k=3(아무것도
버리지 않음)에서 episode-mentions 순서가 3>2>1로 나옴을 단언.

### 새로 구현한 reranker 2종 (논문 §3.2)

| | 논문 서술 | 구현 |
|---|---|---|
| `EpisodeMentionsReranker` | "prioritizes results based on the frequency of entity or fact mentions within a conversation" | 업스트림은 `MENTIONS` 엣지 수, 우리는 `len(source_episode_ids)` — 같은 양(그 리스트가 곧 어떤 에피소드가 이 항목을 만들었는지다). 동순위는 fusion 순서 유지(안정 정렬) |
| `NodeDistanceReranker` | "reorders results based on their graph distance from a designated centroid node" | 그래프는 생성 시 주입(프레임워크 핸들), centroid는 질의별이라 `search(center_node_id=...)`로 전달 — 업스트림 `center_node_uuid`. centroid 없으면 no-op(업스트림도 동일) |

둘 다 capability-free라 `RERANKER_CANDIDATES`에서 **MMR 아래**에 뒀다. 위에 두면 프로필 기본값이
불가용할 때 `resolve()`가 이들을 "최선의 가용 reranker"로 골라버린다 — node-distance는 centroid
없이는 no-op이므로 조용한 성능 저하가 된다.

reranker 생성자 주입은 `AgenticMemory._build_reranker` 하나로 모았다: LLM reranker는 structured
caller, node-distance는 graph store·namespace, 나머지는 `AgmemConfig.reranker_params`
(cross-encoder의 `model_name`, MMR의 `lambda_`). 해석된 클래스가 받지 않는 파라미터는 경고 후
무시한다 — 리졸버가 config가 쓰인 클래스와 다른 것으로 degrade했을 수 있기 때문이다.

---

## 남은 Zep 갭

1. **측정 실행** — 해금 조건은 모두 충족. 단 `cross_encoder` 레시피는 `sentence_transformers` +
   BGE reranker 모델(~2.2GB)이 필요하고, capability 미충족 시 리졸버가 조용히 다른 reranker로
   내려간다. 런 스탬프의 `degradations`와 `search_recipe`를 **반드시 확인**할 것
2. hyper-edge(같은 fact가 여러 entity 쌍에서 추출되는 경우) 미지원
3. "recent episodes as seeds" BFS origin 변종 (업스트림 명시적-origin 경로)
4. entity attributes ontology / SagaNode / combined 단일콜 추출 — 논문 이후 업스트림 추가물,
   추적만
5. predicate 표기 `snake_case`(우리) vs `SCREAMING_SNAKE_CASE`(업스트림)

## 검증

```
327 passed, 1 skipped                        (pytest)
97 files already formatted                   (ruff format --check src/ tests/ scripts/)
Found 53 errors  (src/ tests/)               ← HEAD(49) 대비 +4, 전부 memoryos/organizer.py
Found 7 errors   (scripts/)                  ← HEAD와 동일
```

`+4`는 이번 작업분이 아니라 **6차 C1/C2(MemoryOS LPM 재구현·계보 프리셋)** 가 남긴 것이고,
전부 이 리포지토리가 이미 전역적으로 안고 있는 규칙(`UP017 datetime.UTC`,
`C408 dict()`)의 추가 사례다. 이번 라운드가 만든 신규 lint는 0건
(`ruff check src/agmem/organizers/zep_graph/ tests/test_organizers_phase3.py` → clean).

## 변경 파일

```
src/agmem/core/types.py                       MEMORY_TYPES += "communities"
src/agmem/config.py                           bfs_types · bfs_max_depth · reranker_params ·
                                              graph_expansion_{cap→0,hops}
src/agmem/memory.py                           _apply_graph(communities) · DELETE 시 그래프 제거 ·
                                              _build_reranker · search(center_node_id) · 배선
src/agmem/organizers/zep_graph/community.py   (신규) label propagation + 요약 프롬프트 2종
src/agmem/organizers/zep_graph/search.py      (신규) ZEP_SEARCH_RECIPES 6종 + config 환산
src/agmem/organizers/zep_graph/organizer.py   community 빌드/확장 · entity content · produces
src/agmem/retrieval/bfs.py                    (신규) φ_bfs 랭킹 채널 2종
src/agmem/retrieval/rerank.py                 EpisodeMentions · NodeDistance · 시그니처 확장
src/agmem/retrieval/{pipeline,steps}.py       BFS 채널 배선 · rerank 게이트 · meta/centroid 전달
src/agmem/stores/sqlite_doc.py                lexical_text
src/agmem/stores/{sqlite,kuzu,neo4j}_graph.py community 테이블/메서드 (3벌 모두)
scripts/exp_locomo_conv0.py                   레시피 기반 zep config 4종 + run(recipe=) + 스탬프
tests/{test_organizers_phase3,test_pipeline_p0,test_capabilities,test_lifecycle}.py
```
