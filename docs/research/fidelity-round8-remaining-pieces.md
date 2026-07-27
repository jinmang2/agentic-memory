# 8차: Zep·MemoryOS·ACE 잔여 항목 전량 구현 (2026-07-27)

> 범위: 7차가 남긴 "남은 Zep 갭"과 docs/10의 MemoryOS·ACE 누락 열거. 구현 전 업스트림을 당일
> raw로 재확인했다:
> `getzep/graphiti:graphiti_core/prompts/extract_edges.py`,
> `BAI-LAB/MemoryOS:memoryos-pypi/{short_term,mid_term,updater,retriever,memoryos,prompts,utils}.py`,
> `BAI-LAB/MemoryOS:eval/main_loco_parse.py`, `ace-agent/ace:ace/ace.py`·`playbook_utils.py`·
> `ace/prompts/curator.py`, 논문 arXiv:2501.13956 §3.1–3.3·§6.1.3, arXiv:2506.06326 §3.3,
> arXiv:2510.04618 §3.

## 판정 요약

| # | 항목 | 성격 | 저장된 수치에 영향 | 조치 |
|---|---|---|---|---|
| Z1 | predicate가 `snake_case` (업스트림·논문은 SCREAMING_SNAKE_CASE) | 충실도 이탈 | 없음 (zep 미측정) | **수정** |
| Z2 | hyper-edge 미지원 + subject==object 무가드 | 누락 | 없음 | **수정** |
| Z3 | "recent episodes as seeds" BFS origin 변종 | 누락 | 없음 | **구현** |
| M1 | STM이 1-page FIFO 롤링이 아니라 **전량 배치 flush** | **구조적 이탈** | **있음** (콜 수·세그먼트 경계) | **수정** |
| M2 | dialogue chain(continuity + meta_info) 전무 | 누락 | **있음** (콜 수) | **구현** |
| M3 | STM이 QA 시점에 주입되지 않음 (round-5 N2) | 누락 채널 | 있음 | **구현** |
| M4 | assistant-knowledge 채널 부재 | 누락 채널 | 있음 | **구현** |
| M5 | eval 계보의 2콜 프로필 갱신 | 계보 미분리 | 없음 (eval 미측정) | **구현** |
| M6 | agent persona (역할극 system 프롬프트) | 누락 | — | **구현(기본 off)** |
| M7 | MTM 검색이 1단계(세그먼트 요약)라 **업스트림에 없는 요약 채널**을 주입 | 채널 이탈 | 있음 | **수정** |
| U3 | 업스트림이 `agent_response` 빈 page를 **조용히 버린다** | upstream 결함 | — | 미재현 |
| A1 | ACE curator에 token budget·stats·progress 부재 | 누락 | 없음 | **구현** |
| A2 | ACE environment feedback 2분기 부재 | 누락 | 없음 | **구현** |
| A3 | multi-round reflection이 "누락"으로 분류돼 있었다 | **분류 오류** | — | 계약 밖으로 재분류 |

**docs/09 수치 해석 정정 필요** — 아래 M1/M2 절 참조.

---

# A. Zep

## Z1. predicate 표기

논문 부록 §6.1.3: *"The relation_type should be a concise, **all-caps** description of the
fact (e.g., LOVES, IS_FRIENDS_WITH, WORKS_FOR)."* 현 업스트림도 동일
(`extract_edges.py:34` `relation_type` 필드 description: "in SCREAMING_SNAKE_CASE").
우리 프롬프트는 `snake_case_relation`을 요구하고 있었다.

표기가 장식이 아닌 이유: predicate는 그래프에서 **엣지의 정체성 일부**다. 한 메시지에서
`lives_in`, 다음 메시지에서 `LIVES_IN`이 나오면 서로 다른 관계 타입으로 읽힌다.

수정: 프롬프트를 고치고 **`_relation_type()`으로 정규화**한다(비영숫자→`_`, upper, 빈 값은
`RELATED_TO`). 프롬프트만으로는 부족하다 — 0.6B 모델은 표기 지시를 무시하는 빈도가 높아
그래프가 혼재된 상태로 남는다.

## Z2. hyper-edge와 DISTINCT 노드 가드

논문 §2.2.2: *"the same fact can be extracted multiple times between different entities,
enabling Graphiti to model complex multi-entity facts through an implementation of
**hyper-edges**."*

즉 hyper-edge는 **별도 자료구조가 아니다** — 3개 이상 엔티티가 걸린 사실을 **쌍마다 한 번씩**
같은 statement로 내보내면 그것이 hyper-edge다. 우리 dedup은 이미 same-pair 한정이라
(업스트림의 제약과 동일) 구조적으로 막히지 않았지만, 프롬프트가 그 가능성을 **알려주지
않았다**. 업스트림 프롬프트의 guideline을 전사해 넣었다.

같이 발견: **subject == object 가드가 없었다.** 논문 §6.1.3 "two DISTINCT nodes"이고,
그래프 위험도 있다 — `edges_between(x, x)`는 양방향 매칭 절 때문에 자기 엣지를 **두 번**
반환하므로, 다음 메시지에서 자기 자신이 중복 후보로 올라온다. 드롭한다.

## Z3. recent episodes를 BFS seed로

논문 §3.1: *"φ_bfs can accept nodes as parameters for the search … This functionality proves
particularly valuable when using **recent episodes as seeds** for the breadth-first search,
allowing the system to incorporate recently mentioned entities and relationships into the
retrieved context."*

업스트림 API는 `search(..., bfs_origin_node_uuids=[...])`이고, **명시 origin이 주어지면
파생 origin을 쓰지 않는다** (`search.py`: `if ... and bfs_origin_node_uuids is None:` 안에서만
파생). 그대로 옮겼다:

- `RetrievalPipeline.search(bfs_origin_ids=...)` → `AgenticMemory.search(bfs_origin_ids=...)`
- `AgenticMemory.recent_episode_entity_ids(n_episodes=4, limit=20)` — 최근 n개 에피소드가
  언급한 엔티티 노드, **최신 에피소드 우선** 정렬. 멤버십은 엔티티 아이템의
  `source_episode_ids`에서 얻는다(업스트림은 `MENTIONS` 엣지를 걷지만, 그쪽은 에피소드가
  그래프 노드이고 우리는 doc store에 있다 — 같은 관계, 다른 저장소).

**회귀 테스트**: `test_zep_predicate_is_screaming_snake_case_and_self_loops_drop`,
`test_zep_recent_episode_entities_seed_the_bfs_channel`.

---

# B. MemoryOS

## M1. STM은 배치가 아니라 1-page FIFO 롤링 윈도우다 (round-5 N2 절반)

### 업스트림 확정

```python
# memoryos.py:242-246
if self.short_term_memory.is_full():          # len(memory) >= max_capacity
    self.updater.process_short_term_to_mid_term()
self.short_term_memory.add_qa_pair(qa_pair)

# updater.py:100-103
while self.short_term_memory.is_full():
    qa = self.short_term_memory.pop_oldest()
```

`is_full()`이 `>=`이므로 **pop 한 번이면 조건이 풀린다** → 방출은 매번 **정확히 1페이지**.
STM은 `capacity - 1` 페이지를 상시 유지한다. 논문도 같다 (arXiv:2506.06326 §3.3:
"the **oldest dialogue page** is transferred from the STM to the MTM according to the FIFO
principle").

우리는 `stm_capacity` 페이지가 차면 **전량을 한 배치로** 내보내고 버퍼를 비웠다.

### 한 번에 세 가지가 달라진다

| | 업스트림 | 우리(구) |
|---|---|---|
| TOPIC(multi-summary) 콜 | **페이지당 1회** (워밍업 이후) | `capacity` 페이지당 1회 |
| 세그먼트 경계 | 페이지 단위로 순차 삽입 | 배치 경계로 절단 |
| QA 시점 STM | `capacity - 1` 페이지 상주 | **비어 있음** |

### docs/09 비용 해석 정정 (중요)

docs/09은 "MemoryOS가 **배치 설계 덕에** organizer 호출 91회로 가장 저렴 (A-Mem/Nemori의
1/9)"이라고 적었다. 그 배치 설계는 **MemoryOS의 것이 아니라 우리 구현의 것**이었다.
업스트림 단위로 세면 419턴 ≈ 209페이지에서:

```
TOPIC        ≈ 209 - 10 = 199회   (페이지당 1회, 워밍업 10 제외)
continuity   ≈ 199회              (M2)
meta_info    ≈ 199회              (M2)
LPM 승격     별도
합계         ≈ 600회+   vs 저장된 91회
```

즉 **파레토 그래프의 "MemoryOS가 가장 저렴" 결론은 구현 이탈의 산물**이다. 방법론 자체는
A-Mem/Nemori보다 **더** 비싸다. docs/09에 캐비앗을 병기했다.

### 수정

`on_message`가 가장 오래된 페이지 하나만 방출하고 나머지를 버퍼에 남긴다. `flush_buffer`는
**기본 no-op** — 업스트림은 STM을 드레인하지 않고 QA에 주입한다(M3). 구 동작은
`flush_stm_on_drain=True`로 유지했다(docs/09 런 재현용).

eval 계보(`stm_capacity=1`)에서는 방출 후 STM이 비므로 이 구분이 없다.

**회귀 테스트**: `test_memoryos_stm_rolls_one_page_and_stays_resident` — 3페이지에서 1페이지만
나가고 2페이지가 상주하며, `flush_buffer`가 비우지 않음을 단언.

## M2. dialogue chain (continuity + meta_info)

업스트림은 방출 페이지마다 **LLM 2콜**:

1. `check_conversation_continuity(prev_page, curr_page)` — "Return ONLY 'true' or 'false'"
2. `generate_page_meta_info(last_meta, curr_page)` — 이전 요약을 이어받아 1~2문장 갱신

그리고 QA 시 페이지 옆에 `Conversation chain overview: {meta_info}`로 주입한다
(`memoryos.py:277`, `main_loco_parse.py`도 동일).

전사: `CONTINUITY_PROMPT`/`META_INFO_PROMPT`(업스트림 원문), 드롭 시 "비연속/요약 없음"으로
degrade — 업스트림도 `response.strip().lower() == "true"`이므로 파싱 실패는 False다.

**저장 지점**: 업스트림은 meta_info를 **page**에 달고 페이지마다 주입한다. 우리 MTM 아이템은
세그먼트라 **저장은 세그먼트에** 하지만, M7의 2단계 검색이 세그먼트를 page로 치환할 때
meta_info를 함께 실어 보내므로 **주입 지점은 업스트림과 같다**(page 옆). 저장 지점이 달라도
값은 같다 — 업스트림 `_update_linked_pages_meta_info`가 체인 전체를 최신 요약으로 덮어쓰므로
한 체인의 모든 페이지가 같은 값을 갖는다.

`dialogue_chain=False`로 끌 수 있다. 이 조직자의 write 비용에서 가장 큰 항이고, docs/09 런은
이것 없이 산출됐으므로 그 런을 재현하려면 꺼야 한다.

**회귀 테스트**: `test_memoryos_dialogue_chain_summarizes_and_renders` — 첫 페이지에는 연속성
콜이 없고, 두 번째 페이지에서 체인이 이어지며, 렌더에 "Conversation chain overview:"가 나옴.

## M3. STM을 QA에 주입 (round-5 N2 나머지 절반)

업스트림 `get_response`는 `short_term_memory.get_all()`로 `history_text`를 만들어 프롬프트
맨 앞에 넣는다. 검색 채널로 대체할 수 없다 — 요점이 유사도가 아니라 **최근성**이다.

`Organizer.recent_context() -> str` 훅 신설(기본 `""`). 스토어를 건드리지 않는 유일한 훅이고,
"색인되지도 검색되지도 않지만 매 질문에 주입되는 원문 창"이라는 계약이다.
`bench/locomo.answer`가 활성 조직자 전체에서 모아 컨텍스트 맨 앞에 붙인다.

## M4. assistant-knowledge 채널

업스트림 두 계보가 다르다 (round-5 §3 표와 일치):

| | user knowledge | assistant knowledge |
|---|---|---|
| pypi | top-20 검색 | top-20 검색 |
| eval (논문 수치) | 검색 결과 주입 | **전량 주입** (`get_assistant_knowledge()`) |

우리 `semantic` 검색 채널이 이미 두 종류를 색인·검색하므로 pypi 쪽은 커버된다. eval 계보가
추가하는 것은 **assistant knowledge 전량**이고, 그것을 `answer()`가 별도 섹션으로 주입한다.

> **9차 정정**: 그 주입에 **계보 게이트가 없었다.** 즉 pypi config도 top-20 검색과 전량 덤프를
> 동시에 받았다 — 이 절이 구분한 두 계보가 실제로는 한 런에 겹쳐 있었다. `answer(
> assistant_knowledge_mode="retrieved"|"full")`로 분리, 기본 `"retrieved"` — round-9 §L2a.

## M5. eval 계보의 2콜 프로필 갱신

```python
# eval/main_loco_parse.py:53-58
result = gpt_personality_analysis(un_analyzed, client)      # ← 옛 프로필을 보지 않는다
if old_profile:
    updated_profile = gpt_update_profile(old_profile, result["profile"], client)  # ← 2번째 콜
else:
    updated_profile = result["profile"]
```

pypi는 옛 프로필을 분석 프롬프트에 접어 넣고 **1콜**로 끝낸다. 차이가 의미적이다: eval 형태에서
**분석은 이전 주장을 수정할 수 없고 merge만 할 수 있으며**, merge는 분석 결과를 "대화"가 아니라
"New Analysis Data"로 본다.

`MEMORYOS_PRESETS`에 `profile_update="single"|"two_call"` 추가, `UPDATE_PROFILE_PROMPT` 전사.
merge 콜이 드롭되면 분석 결과를 프로필로 쓴다(업스트림의 `updated_profile = new_profile`
분기와 같은 방향 — merge를 잃는 것이 분석을 잃는 일이 되면 안 된다).

**회귀 테스트**: `test_memoryos_eval_lineage_merges_the_profile_in_a_second_call` — eval 계보는
merge 프롬프트가 정확히 1번 나오고 그 안에 1차 분석이 "Old User Profile"로 들어가며, pypi
계보는 merge 프롬프트가 0번.

## M6. agent persona (기본 off)

eval 드라이버의 system 프롬프트는 역할극이다: *"You are role-playing as {speaker_b} … Here are
some of your character traits and knowledge: {assistant_knowledge}"* + 극단적 간결 답변 지시
+ few-shot 1개.

`PERSONA_SYSTEM_PROMPT`로 전사하고 `answer(persona=...)`/`evaluate(persona=...)`로 노출했으나
**기본 off**다. 이유는 충실도가 아니라 이 하네스의 성질이다 — 모든 config가 같은
`ANSWER_PROMPT`로 답하기 때문에 점수 차이를 메모리 시스템에 귀속시킬 수 있고,
방법론별 답변 프롬프트는 그 귀속을 깬다. 켜면 그 런은 **업스트림 수치와는 비교 가능하고
docs/09의 다른 config와는 비교 불가능**해진다 — 그 교환을 명시적 인자로 만들어 뒀다.

## M7. MTM 검색은 2단계이고, 세그먼트 요약은 컨텍스트가 아니다

### 업스트림

```python
# mid_term.py search_sessions
index = faiss.IndexFlatIP(dim); index.add(summary_embeddings_np)      # ① 세션 = summary 임베딩
distances, indices = index.search(query_arr_np, min(top_k_sessions, ...))
...
if session_relevance_score >= segment_similarity_threshold:           # 0.1
    for page in session["details"]:
        page_sim_score = float(np.dot(page_embedding, query_vec))     # ② 세션 안의 page 채점
        if page_sim_score >= page_similarity_threshold:               # 0.1
            matched_pages_in_session.append(...)
```
그리고 `Retriever._retrieve_mid_term_context`가 **모든 매칭 세션의 page를 전역 heap**으로 모아
`retrieval_queue_capacity`개만 남긴다. **세션 summary는 프롬프트에 들어가지 않는다** — 매칭
키일 뿐이다.

우리는 세그먼트 요약을 1단계로 검색해서 **그 요약을 그대로 주입**했다. round-5 §3이 이미
"업스트림에 없는 채널 — 미문서화"로 지목한 지점이다.

### 부수 확인: 검색 단계의 keyword 항은 죽어 있다

`search_sessions`의 세션 점수는 `semantic_sim + keyword_alpha * s_topic_keywords`인데
바로 위에 `query_keywords = set()  # Keywords extraction removed`가 있다. 빈 집합이라
`s_topic_keywords`는 **항상 0**이다. 즉 **검색에서 keyword 항은 사문**이고, keyword가 실제로
쓰이는 곳은 병합 F_score뿐이다. (A-MAC의 죽은 N/R, MemoryOS eval의 죽은 R_recency와 같은 계열.)
우리 1단계는 dense + BM25(FTS)인데 BM25는 upstream 검색에 대응물이 없다 — `lexical_types`에
`pages`를 넣지 않는 현 config가 결과적으로 정합적이다.

### 수정: `MemoryOSPageRecall` read step

`pages` 타입에 등록되는 스텝으로, 매칭된 세그먼트를 **그 안의 page들로 치환**한다.
`cap`=`retrieval_queue_capacity`, `threshold`=`page_similarity_threshold`(0.1).

> **9차 정정**: 여기 적었던 `cap`(10)은 **계보 표시가 빠진 값이었다.** 10은 eval 드라이버가
> 넘기는 값(`main_loco_parse.py:237`)이고 pypi는 7이다(`memoryos.py:38`, `Retriever` 기본값).
> 전역 기본이 10이라 pypi 프리셋이 eval 계보의 큐로 읽고 있었다 — round-9 §L2b.

이를 위해 세그먼트가 자기 **page 구조**를 들고 있어야 한다 — `_segment_add`/병합 경로가
`page_units: [[episode_id, ...], ...]`를 payload에 싣는다. 평평한 `source_episode_ids`로는
한 교환이 어디서 끝나는지 알 수 없고, 그룹핑은 organizer의 몫(`_pages`)이기 때문이다.

편차 2건:
1. **page 임베딩이 없다.** 업스트림은 page를 한 문자열로 임베딩해(`f"User: {u} Assiant: {a}"`,
   오타 upstream) 질의와 내적한다. 우리 page는 아이템이 아니라 벡터가 없으므로 **멤버 메시지
   벡터 중 최대 cosine**으로 채점한다 — 같은 텍스트를 반으로 나눠 잰 것이고, 질의마다 page마다
   임베더를 부르지 않아도 된다.
2. **heat 되먹임 경로.** 업스트림은 매칭 page를 가진 세션의 N_visit을 올린다. 치환 후 서빙되는
   id는 page(선두 메시지 id)이므로, `on_retrieval`이 `_unit_pages`(unit→segment 역인덱스)로
   되돌려 세그먼트를 올린다. 이걸 안 하면 **round-5 N1이 복원한 되먹임 루프가 2단계 도입과
   함께 조용히 죽는다.** LFU 카운터도 세그먼트 키로 교정했다(page 키였다면 어떤 축출도 읽지
   않는 카운터가 된다).

`page_units`가 없는 `pages` 아이템(스토어에 직접 쓴 행, 이 변경 이전 세그먼트)은 **그대로
통과**시킨다 — "page가 있는데 하나도 임계를 못 넘었다"(드롭, 업스트림 정합)와 구분하기
위해서다. 스텝이 organizer가 아니라 **타입**에 걸린다는 모듈 원칙도 이 구분으로 지켜진다.

`page_recall_cap=0`으로 끄면 구 동작(요약 주입)으로 돌아간다.

**회귀 테스트**: `test_memoryos_page_recall_serves_pages_not_segment_summaries` — 렌더에 원문
page가 있고 요약 텍스트("SUMMARY-ONLY-TEXT")는 없으며, page id가 서빙됐는데도 세그먼트의
`n_visit`/`_access`가 올라감을 단언.

**주의(테스트·소형 임베더)**: FakeEmbedder처럼 해시 기반 임베더에서는 무관한 질의의 cosine이
≈0이라 임계 0.1에 전부 걸려 **빈 결과**가 된다. 업스트림 동작 그대로지만, page를 기대하는
테스트는 page 본문과 겹치는 질의를 써야 한다.

## U3. 업스트림 결함: `agent_response`가 빈 page는 조용히 사라진다

```python
# updater.py:101-104
qa = self.short_term_memory.pop_oldest()
if qa and qa.get("user_input") and qa.get("agent_response"):
    evicted_qas.append(qa)
```

pop은 했는데 조건이 거짓이면 **아무 데도 들어가지 않는다**. LoCoMo 드라이버에서 세션이
speaker A 발화로 끝나면 그 page의 `agent_response`가 `""`이므로 정확히 이 경로다.
`evicted_qas`가 비면 조기 return이라 요약 콜도 없다.

**미재현** — 원문 손실 금지 원칙에 정면으로 걸린다. 우리는 반쪽 페이지도 그대로 MTM에 넣는다.

---

---

# C. ACE (docs/10 행이 낡아 있었다)

round-5가 지적한 ACE 항목 중 **read 계약(전체 playbook 주입)·curator 전체-뷰·intra-batch
dedup·렌더 포맷 단일화는 이미 해소돼 있었고 docs/10 행만 갱신되지 않았다.**
`memory.get_playbook()`은 `list_items` 전체 스캔이고 그 docstring이 "이 전체 렌더가 방법론의
read 계약"임을 명시한다(top-k로 바꾸지 말 것). `ACEOrganizer._current_playbook`도 전체를
반환한다. 즉 "read 계약 고정 전 측정 보류" 사유가 이미 없었다.

## A1. token budget + stats + progress (round-5 §3.5)

업스트림 curator 프롬프트의 "Training Context" 블록:

```
- Total token budget: {token_budget} tokens      # ace.py:127 playbook_token_budget=80000
- Training progress: Sample {current_step} out of {total_samples}
**Current Playbook Stats:**
{playbook_stats}                                  # playbook_utils.get_playbook_stats
```

전체 playbook을 주입하는 이상 성장을 억제하는 유일한 수단이 **"예산을 알려주는 것"**이다.
그래서 이것은 프롬프트 입력이지 **절단이 아니다** — bullet을 버려서 예산에 맞추는 것이 곧
논문이 진단한 context collapse다. `playbook_stats()`는 업스트림의 세 버킷을 전사했다:
high-performing(helpful>5 and harmful<2), problematic(harmful>=helpful and harmful>0),
unused(카운터 0). `total_samples`는 데이터셋 크기라 메모리 계층이 알 수 없으므로 선택 인자로
두고, 없으면 progress 줄이 step만 남는다. 토큰 수는 chars/4 추정 — 프롬프트 한 줄이 유일한
소비처인 숫자 때문에 토크나이저를 하드 의존성으로 만들 이유가 없다.

## A2. environment feedback 2분기 (round-5 §3.7)

업스트림은 `is_correct`로 갈라 "Predicted answer matches ground truth" /
"...does not match ground truth"를 reflector에 주입한다. 우리 `outcome`은 자유형 문자열이므로,
그 문자열이 두 상태 중 하나를 지칭할 때만 업스트림 문구로 치환하고 그 외에는 그대로 통과시킨다
(업스트림도 `REFLECTOR_PROMPT_NO_GT` 분기를 갖는 no-ground-truth 변형이다).

## A3. multi-round reflection은 구현하지 않는다 — organizer 계약 밖이다

docs/10은 이것을 "누락"으로 적어 왔지만, 업스트림 루프를 읽으면 분류가 틀렸다
(`ace.py:501-543`):

```python
for round_num in range(max_num_rounds):          # 3
    reflection_content, bullet_tags, _ = self.reflector.reflect(...)
    gen_response, bullet_ids, _ = self.generator.generate(...)   # ← 답변 재생성
    final_answer = extract_answer(gen_response)
    if data_processor.answer_is_correct(final_answer, target):   # ← 교정되면 종료
        break
```

라운드마다 **trajectory가 바뀐다** — reflect의 산출물을 generator에 다시 먹여 답을 새로 만들고,
정답이 되면 멈추는 **추론 시점 교정 루프**다. 우리 `on_task_end`는 완료된 trajectory를 사후에
받으므로 재생성을 할 수 없고, 고정된 trajectory를 3번 reflect하는 것은 **업스트림에 없는
메커니즘을 발명하는 것**이다. 이 루프는 메모리 계층이 아니라 **생성기(에이전트)**가 소유한다.
docs/10 행에서 "누락"이 아니라 "계약 밖"으로 재분류했다.

같은 이유로 offline 모드(train/val 분할·multi-epoch·val 기반 best 선택)도 하네스 기능이지
organizer 기능이 아니다.

**회귀 테스트**: `test_ace_curator_gets_the_token_budget_stats_and_progress` — stats 세 버킷과
섹션 카운트, 프롬프트의 예산·progress 줄, 그리고 `outcome="failure"`가 업스트림 문구로
치환됨을 단언.

---

# 검증

```
334 passed, 1 skipped                        (pytest)
97 files already formatted                   (ruff format --check src/ tests/ scripts/)
Found 53 errors  (src/ tests/)               ← 8차 시작 시점과 동일, 신규 0
Found 7 errors   (scripts/)                  ← 동일
```

# 재측정 대상 갱신

`results/locomo-conv0-memoryos.json`은 이제 **두 세대 낡았다**: 6차 B1/C1(page 단위·LPM
재구현)에 이어 8차 M1~M4가 write 경로와 read 컨텍스트를 모두 바꿨다. 비교 가능한 재현을
원하면 `dialogue_chain=False` + `flush_stm_on_drain=True`로 구 배선을 재구성해야 한다.

# 변경 파일

```
src/agmem/bench/locomo.py                     recent_context 주입 · assistant knowledge 전량
                                              · PERSONA_SYSTEM_PROMPT · answer/evaluate(persona=)
src/agmem/memory.py                           search(bfs_origin_ids) · recent_episode_entity_ids
src/agmem/organizers/base.py                  recent_context() 훅
src/agmem/organizers/memoryos/organizer.py    롤링 방출 · dialogue chain · profile_update 계보
                                              · flush_stm_on_drain · recent_context
src/agmem/organizers/zep_graph/organizer.py   _relation_type · hyper-edge 가이드 · DISTINCT 가드
src/agmem/config.py                           page_recall_cap · page_recall_threshold
src/agmem/retrieval/pipeline.py               bfs_origin_ids (명시 origin이 파생을 대체)
                                              · ReadContext에 query_embedding/vector_store
src/agmem/retrieval/steps.py                  meta_info 렌더 · MemoryOSPageRecall (2단계 검색)
src/agmem/organizers/ace/organizer.py         token budget · playbook_stats · progress ·
                                              environment feedback 2분기
tests/{test_organizers_phase3,test_lifecycle,test_ace}.py
```
