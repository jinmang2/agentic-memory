# G-Memory 충실도 검증 리포트 (2026-07-17)

## 0. 검증 소스

- **로컬**: `/home/jinmang2/agentic_memory/src/agmem/organizers/gmemory.py` (198줄), `docs/10-fidelity-audit.md` 43행, `docs/research/g-memory.md`
- **논문**: arXiv:2506.07398 PDF 전문 텍스트 추출 성공 (24p, §3 정의 + §4.1–4.3 메서드 전체 확보. arXiv HTML v1/v2는 404)
- **공식 코드**: `github.com/bingreeky/GMemory` clone 성공 (commit `7b581c5`, 2026-04-09). 핵심 파일: `mas/memory/mas_memory/GMemory.py` (892줄: `GMemory`/`TaskLayer`/`InsightsManager`), `mas/memory/mas_memory/prompt.py`, `mas/memory/common.py` (`StateChain`/`MASMessage`)
- **선행 주의**: 논문 §4.3의 insight 업데이트는 "요약 함수 J로 신규 insight 생성 + supporting query set Ω_k 갱신"으로 기술되지만, 공식 코드는 Reflexion 계열 critique finetune(ADD/EDIT/REMOVE/**AGREE**) + backward reward + FINCH merge로 구현되어 **논문↔코드 자체가 어긋남**. 우리 구현은 코드 계보를 따르므로 아래 대조는 공식 코드를 1차 기준, 논문을 구조 기준으로 삼는다.

등급 표기: ✅ CONFIRMED(주장대로) / ⚠️ PARTIAL(구현됐으나 의미 축약) / ❌ REFUTED

---

## 1. 3-tier 그래프 구조 검증 (논문 §3 vs 업스트림 vs 우리)

| 계층 | 논문 정의 | 업스트림 코드 | 우리 구현 |
|---|---|---|---|
| Insight Graph | `G_insight=(I,E_i)`, 노드 ι_k=(내용 κ_k, **supporting query set Ω_k**), 하이퍼엣지 (ι_m,ι_n,q_j) | `InsightsManager`: JSON 리스트 `{rule, score, positive_correlation_tasks, negative_correlation_tasks}` — Ω_k는 correlation_tasks로 구현, 엣지는 미구현 | `kind="insight"` strategies 레코드 `{content, score}` — **Ω_k(correlation tasks) 자체가 없음** |
| Query Graph | `G_query=(Q,E_q)`, 노드 q_i=(Q_i, 상태 Ψ_i∈{Failed,Resolved}, G_inter), 엣지=의미 관계, Eq.(5) 1-hop 확장 | `TaskLayer`: `nx.Graph` pickle, **cos sim ≥ 0.7이면 엣지**, `retrieve_related_task(cutoff=hop=1)` | 없음 (임베딩 검색으로 대체, 헤더 docstring에 편차 명기) |
| Interaction Graph | utterance 노드 (A_i, m_i), temporal 엣지 | `StateChain`: state별 `nx.DiGraph`, 노드=`AgentMessage`, `edge_type='spatial'`, `node_link_data` 직렬화 | 없음 (`list[dict]`를 json.dumps로 평문화) |

→ docs/10의 누락 3항목(query graph k-hop / FINCH merge / StateChain interaction graph)은 **전부 실재하는 누락으로 CONFIRMED**. 단, insight 계층도 "그래프"성(Ω_k support set, 하이퍼엣지)이 빠져 있어 upstream보다 한 단계 더 평탄하다(아래 §4-신규).

---

## 2. Per-claim 판정 — docs/10 "구현됨" 3항목

### 2.1 궤적 sparsify — ⚠️ PARTIAL (구현 표기는 정당하나 의미 축약 큼)

업스트림 `_extract_mas_message`는 3단계다:

1. **기계적 필터**: `if state_chain.get_state(state_id).graph.get('reward', 0) < 0: state_chain.pop_state(state_id)` — 음수 reward state를 StateChain에서 제거 후 숫자 제거(`re.sub(r'\d+', '', trajectory)`)
2. **LLM key-step 추출**: `extract_true_traj_*` 프롬프트(ALFWorld few-shot 예시 포함, "Strictly follow the original trajectory; absolutely no steps that are not in the trajectory should be added", **temperature=0.1**)
3. **실패 태스크 전용**: `_detect_mistakes` 별도 LLM 콜 → `fail_reason` 필드 ("Identify the inconsistency between the final state of the target ... and the required task goal")

우리(`SPARSIFY_PROMPT`, gmemory.py:54–62): 단일 콜로 `key_steps`+`mistakes` 동시 산출. 차이:
- 기계적 reward<0 필터 없음 (StateChain 부재의 귀결 — docs/10 누락 항목에 포섭됨)
- **mistakes 의미가 다름**: 업스트림 `fail_reason`은 *실패 태스크에만* 붙는 최종상태 불일치 분석. 우리는 성공/실패 무관 "pruned detours". 업스트림에서 fail_reason은 comparative finetune 프롬프트의 `{fail_reason}` 슬롯으로 주입되는 핵심 입력인데, 우리 finetune 프롬프트에는 대응 슬롯이 없음
- 업스트림 few-shot·"no invented steps" 제약·temperature 0.1 미반영 (우리는 `ctx.llm.call("distill", ...)` 온도 미지정)

### 2.2 insight ADD/EDIT/REMOVE + reward — ⚠️ PARTIAL (연산 골격은 맞으나 점수 의미론 대부분 상이)

업스트림 (`_parse_rules` 정규식 `r'((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): ...'` + `_update_rules`):

| 연산 | 업스트림 의미론 | 우리 |
|---|---|---|
| ADD | 신규 rule, **score=2**, `positive_correlation_tasks=relative_tasks` | ADD, **score=0.0**, correlation 없음 — 값 상이 |
| EDIT | rule 교체 + **score += 1** + positive tasks 갱신; 기존 rule과 중복 텍스트면 AGREE로 강등 | UPDATE(content만), 점수 무변 — 축약 |
| REMOVE | **soft**: score -= 1 (리스트 full(≥10)이면 -= 3), negative tasks 기록. 실제 삭제는 `clear_insights()`(score ≤ 0 프루닝)가 담당 | **hard DELETE 즉시 수행** — 의미 상이 |
| AGREE | score += 1 + positive tasks 갱신 | **연산 자체 부재** |

finetune 구조 차이:
- 업스트림: 트리거 `memory_size >= 5(start_insights_threshold) and memory_size % 5(rounds_per_insights) == 0`, **랜덤 메모리 포인트 5개**(insights_point_num) 앵커 → 포인트당 succ 3 + fail 1 회수 → **(성공,실패) 비교쌍 프롬프트** + **성공 5개 청크 프롬프트** 별도, 프롬프트당 "Do at most **4** operations and each existing rule can only get a maximum of 1 operation", rule 형식 강제 **"XXX, because XXX"**, 리스트 full 시 suffix로 ADD 억제, 관련 insight는 `_find_related_insights`(correlation task 교집합 ≥ len/2)로 선별
- 우리: **현재 task 1개 앵커**, insight ≤10 + 성공/실패 혼합 궤적 10개를 한 프롬프트에, **≤6 ops**, 형식 제약 없음, cap 초과 시 최저 score insight hard delete (docstring에 FINCH 대체로 명기된 의도적 편차)
- 트리거 카운터: 업스트림은 영속 `memory_size`(Chroma 레코드 수), 우리는 인스턴스 변수 `_task_count` — **프로세스 재시작 시 케이던스 리셋**. 기본값(5)에서는 발화 시점 동일하나 start threshold 부재로 설정 변경 시 어긋남

reward(backward): 업스트림 `backward(reward)`는 `insights_cache`(해당 task에 서빙된 insight만)에 `-2 if reward == False else 1` 적용 후 **즉시 `clear_insights()`로 score ≤ 0 전부 프루닝**. 우리 `backward()`/`Memory.report_feedback`은 +1.0/-2.0으로 **값은 정확히 일치**하나 (memory.py:280–284), score ≤ 0 프루닝이 어디에도 없음 → §5 참조.

### 2.3 projection/backward — ✅ CONFIRMED (부분집합으로서 정확)

- `project_insights(role, insights)`: 업스트림 `project_insights_system/user_prompt`("adapt these general insights into personalized insights tailored to the given role")와 의미 일치. 출력 형식만 numbered list → JSON (구조화 출력 프로젝트 규약, 무해). 단 업스트림의 **`project_insights_with_traj` 변형**(성공 궤적 동봉)은 미구현·docs/10 미기재
- backward 보상값 +1/-2 정확 (논문 미명시 상수 — 코드가 유일 근거이며 일치)
- 우리 프롬프트의 "(drop insights irrelevant to that role)"은 업스트림에 없는 지시(업스트림은 적응만 요구, 탈락 지시 없음) — 사소한 추가 의미

---

## 3. 검색 플로우 대조 (논문 §4.1–4.2 / 업스트림 `retrieve_memory`)

업스트림 파이프라인 (우리에게 **전부 부재** — docs/10에 부분만 기재됨):

1. **Coarse retrieval + k-hop** (Eq.4–5): `retrieve_related_task(node_num=max((s+f)//2,1), hop=1)` → Chroma 유사도 top 노드 + `nx.single_source_shortest_path_length(cutoff=1)` 이웃. 논문 감도분석: "1-hop expansion ... In contrast, 2-hop and 3-hop settings often degrade performance"
2. **성공/실패 분리 채널**: `successful_topk=2, failed_topk=1`(2배 과회수), label 필터 유사도 보충, **threshold=0.3** 필터
3. **LLM relevance rating** (Eq.7 R_LLM): 성공 궤적마다 `generative_task_user_prompt`로 1–10 점수 → 정렬 후 top-k. 실패 궤적은 직취
4. **insight 회수 = support-set 투표** (Eq.6 Π_Q→I): `query_insights_with_score` — 관련 task(succ 4+fail 2)+현재 query의 `positive_correlation_tasks` 교집합 **카운트 투표**로 정렬, top 10. 임베딩 유사도가 아님
5. `project_insights`로 role별 주입, 서빙된 insight는 `insights_cache`에 기록 → backward 대상

우리: `pipeline.search(memory_types=("strategies",))` 단일 벡터 검색. 1–4 전부 없음. docs/10은 이 중 1(k-hop)만 누락으로 기재.

---

## 4. docs/10 누락 리스트 정확성 판정

**기재된 3항목은 모두 정확** (✅): query graph k-hop(§1,§3-1), FINCH merge(업스트림 `merge_insights`: `memory_size % 20 == 0`마다 task 임베딩 FINCH cosine 클러스터링 → 클러스터별 rule을 LLM 병합(batch 10, `limited_number = (len(batch)//3)//3` — 업스트림 자체의 이중 나눗셈 quirk로 매우 공격적 압축) → **전체 insight 리스트 clear 후 score=2로 재생성**), StateChain interaction graph(§1).

**그러나 불완전** — 미기재 누락 (신규 발견, 중요도순):

1. **[높음] 검색 경로 의미론 전체**: 성공/실패 분리 채널(2/1, threshold 0.3), LLM relevance rating(R_LLM), insight support-set 투표(Π_Q→I). G-Memory 검색 우위 주장의 실질이 여기 있는데 "k-hop"이라는 한 단어로만 대표됨. Zep 행이 "GraphRecall 파이프라인 배선"을 별도 항목화한 것과 비교하면 비대칭
2. **[높음] insight 점수 의미론**: AGREE 연산, ADD init 2 / EDIT +1 / REMOVE soft(-1/-3), **score ≤ 0 프루닝(clear_insights)**, positive/negative_correlation_tasks(= 논문 Ω_k — insight "그래프"성의 실체)
3. **[중간] `_detect_mistakes` fail_reason** (실패 분석 전용 콜 + comparative finetune 입력)
4. **[중간] finetune 구조**: 랜덤 포인트 앵커링, (성공,실패) 비교쌍 프롬프트, 4-op cap/rule당 1-op, "XXX, because XXX" 형식
5. **[낮음] `project_insights_with_traj` 변형**, 온도 0.1, 트리거 영속 카운터

"측정 가능? ✘ MAS 벤치 미구축으로 제외"는 **올바름** — bench/에 gmemory config 없음, LoCoMo 4-way에 미포함 확인. 등급 ○(골격) 판정 자체는 유지 타당(오히려 근거가 기재보다 많음).

---

## 5. 배선(wiring) 점검 — 신규 발견 포함

**emit 타입**: `GMemoryOrganizer`는 전부 `target_type="strategies"`로 emit하며 payload `kind`로 구분 — `kind="trajectory"`(score 1.0/-2.0) + `kind="insight"`(score 0.0). `core/types.py:35` 주석대로 strategies는 **ReasoningBank 아이템과 공유 타입**. 트리거는 `Memory.add_task_result` → `on_task_end` (memory.py:169) — 배선 정상.

MAS 벤치가 생겼을 때 흐름 시뮬레이션:

- ✅ `mem.search(query, memory_types=("strategies",))`로 회수는 됨 (pipeline.py:44–57, 벡터 검색+RRF, 비-episodic이라 lexical 없음)
- ⚠️ **W-1: kind 미분리** — insight와 condensed trajectory가 단일 랭킹에서 경쟁, 성공/실패 궤적 미분리, ReasoningBank과 gmemory를 같은 namespace에서 쓰면 서로 섞임. 업스트림의 (succ 2, fail 1, insight 10) 3채널 반환 구조와 불일치
- ⚠️ **W-2: reward 루프가 read-path에 무영향 (write-only)** — `report_feedback`이 score를 +1/-2로 갱신하지만(값 자체는 upstream 일치), **score는 검색 랭킹·필터 어디에도 쓰이지 않고**, score ≤ 0 프루닝도 없음. 유일한 소비처는 finetune 시 cap 초과 축출의 min 선택(gmemory.py:176)과 프롬프트 내 표기. 업스트림은 backward 직후 `clear_insights()`로 score ≤ 0 insight를 즉시 제거하므로 reward가 서빙 집합을 실제로 바꿈. 현재 구조에서는 -2를 아무리 받아도 해당 insight가 계속 서빙됨
- 🐛 **W-3: REMOVE된 insight가 검색에 유령으로 잔존** — `_apply_ops` DELETE는 doc를 `{"id", "deleted": True}`로 덮어쓰지만(memory.py:254–255) **벡터는 vec store에 남고**, `pipeline._hydrate`(pipeline.py:80–96)에는 `deleted` 필터가 없음 → REMOVE된 insight가 빈 content의 ScoredItem으로 계속 반환되며 k 슬롯을 소모. organizer 내부 `_fetch`(gmemory.py:136)만 필터함. gmemory 한정이 아닌 파이프라인 공통 결함이나, DELETE를 실사용하는 organizer가 gmemory(와 MemoryOS LFU)라 여기서 실질 발현
- ⚠️ **W-4: `report_feedback`이 trajectory에도 적용** — id가 strategies에 있으면 kind 무관하게 ±점수 (memory.py:279–284). 업스트림 backward는 insights_cache(insight 한정) 대상. 실해는 작으나 의미 이탈
- ⚠️ **W-5: `project_insights`/`backward`(organizer 메서드) 호출자 부재** — 파이프라인·벤치 어디서도 호출 안 됨. projection은 MAS 하네스가 직접 불러야 하는 public API로 존재(설계상 예정된 상태), backward는 `report_feedback`이 로직을 중복 구현(값 일치 확인)

**결론**: 벤치가 생기면 "돌아는 가지만" 업스트림 검색 의미론(분리 채널·투표·rating·k-hop)과 reward 폐루프가 재현되지 않으며, W-3은 명백한 버그.

---

## 6. 권고 (우선순위순)

1. **[P0/버그] W-3 수정**: DELETE 시 vec store에서도 제거하거나 `_hydrate`에 `deleted` 필터 추가. gmemory 외 MemoryOS LFU 축출에도 동일 영향 — 전 파이프라인 회귀 확인 필요
2. **[P0/문서] docs/10 G-Memory 행 보강**: 누락 칸에 "검색 경로(성공/실패 분리+LLM rating+insight support-set 투표)", "AGREE+soft score(≤0 프루닝)+correlation task set", "_detect_mistakes"를 추가. 현행 3항목만으로는 ○ 등급의 근거가 과소 기재
3. **[P1] reward 폐루프 복원**: 최소한 backward/report_feedback 후 score ≤ 0 insight 무효화(업스트림 `clear_insights` 대응). 이것 없이는 논문의 핵심 주장("reward-shaped insight evolution")이 우리 쪽에서 검증 불능
4. **[P1] 점수 의미론 정합**: ADD init 2 / EDIT +1 / REMOVE soft(-1, full 시 -3) / AGREE 도입, 또는 hard-delete 편차를 헤더 docstring에 명기(현재 FINCH 편차만 명기됨)
5. **[P2] MAS 벤치 대비 검색 구성**: kind별 분리 검색(succ 2 / fail 1 / insight 10, threshold 0.3) config화, ReasoningBank과의 strategies 타입 공유 시 namespace 또는 kind 필터 격리
6. **[P2] 트리거를 영속 카운트 기반으로** (+ start_insights_threshold 도입), sparsify/finetune 온도 0.1, `_detect_mistakes` 대응 실패-사유 슬롯 추가
7. **[P3] 문서에 논문↔공식코드 불일치 명기**: 논문 §4.3(J 요약+Ω_k 갱신)과 코드(critique finetune+FINCH+backward)가 다르며 우리는 코드 계보를 따른다는 것 — 향후 감사 때 기준 혼선 방지
8. MAS 벤치 미구축으로 측정 제외한 현행 판단은 **유지** (올바름)

## 부록: 업스트림 하이퍼파라미터 (코드 기준, commit 7b581c5)

hop=1 · start_insights_threshold=5 · rounds_per_insights=5 · insights_point_num=5 · merge 주기 20 · query-graph edge sim ≥ 0.7 · retrieval threshold 0.3 · successful_topk 2 / failed_topk 1 / insight_topk 10 · finetune 회수 succ 3/fail 1 (insight 투표용은 succ 4/fail 2) · MAX_RULE_THRESHOLD 10 · REMOVE 강도 1/3(full) · ADD init score 2 · backward +1/-2 · 프롬프트당 최대 4 ops · sparsify temperature 0.1. 논문: 1-hop 최적(2-hop PDDL 49.79%로 저하), k ∈ {1,2} 사용.
