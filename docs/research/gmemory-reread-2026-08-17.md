# G-Memory 재조사 (2026-08-17) — 논문·공식코드·우리 포트 전면 대조

> **이 문서의 지위**: G-Memory를 다시 볼 때 **여기부터** 읽는다. 기존 문서와의 관계는
> - `docs/research/g-memory.md` — 1차 리서치 노트(2026-07-16). 사실관계는 유효, 이 문서가 상위 집합.
> - `docs/research/round5/gmemory-verify-report.md` — **2026-07-17 시점 스냅샷.** 권고 W-1~W-5는
>   이후 round-12(커밋 `6fad7bb`)에서 대부분 조치됨. 그 문서의 "우리 구현" 열은 **현행이 아니다**.
> - `docs/16-abstraction-study.md` 세션 5 — 논문 형식론 ↔ upstream ↔ 우리 3열 대조표.
> - `docs/research/upstream-defect-catalog.md` §6 — 결함 원장(GM-1~GM-15). 이 문서가 GM-12~15의 근거.
>
> 소스: 로컬 클론 `~/.agmem/upstream/GMemory` @ `7b581c5` (LICENSE 없음 — 동작·상수만 재현, 원문 미복제),
> 논문 arXiv:2506.07398 v2. **HTML 판(`arxiv.org/html/2506.07398v2`)은 2026-08-17 기준 열린다** —
> `g-memory.md` line 76과 round-5 §0의 "HTML 404"는 2026-07 시점 사실이었고 지금은 무효.

---

## 0. 요약

1. **논문의 그래프 3계층 중 우리에게 없는 것은 interaction graph 하나뿐이고, 그건 upstream에서도
   읽히지 않는다.** 발화 노드·엣지를 읽는 코드가 repo 전체에 없다(§6 GM-14). 나머지 두 계층은
   우리 쪽에 이름만 바뀌어 존재한다(§7).
2. **MAS 계층(AutoGen/DyLAN/MacNet, agent, reasoning, envs)은 방법론이 아니라 벤치마크 하네스다.**
   미포트가 정당하되, 그 대가로 **G-Memory는 우리 벤치에서 측정 불가**다 — LoCoMo/LongMemEval에는
   성공/실패 라벨이 붙는 task 루프가 없어 label→backward→insight 진화라는 핵심 신호를 돌릴 무대가 없다.
3. **신규 결함 4건**(§6): DyLAN에서 backward 미호출 / 식(5) 1-hop 결과를 통째로 버리는 분기 /
   interaction graph write-only / key-step 추출 전 궤적의 모든 숫자 제거.
   ①②는 **논문 대표 메커니즘이 공식 코드에서 부분적으로 죽어 있다**는 유형이다.
4. 이 방법의 "finetune"은 **모델 파인튜닝이 아니라 ExpeL 계보의 규칙 리스트 LLM 비평 루프**다(§4).

---

## 1. 논문 (arXiv:2506.07398 v2)

Zhang Guibin · Fu Muxin · Wan Guancheng · Yu Miao · Wang Kun · Yan Shuicheng. v1 2025-06-09 / v2 2025-06-16.
NeurIPS 2025 poster.

문제제기(초록): 기존 MAS 메모리는 ① **에이전트 간 협업 궤적을 완전히 무시**하고
② **cross-trial·agent-specific 커스터마이즈가 없다** — 단일 에이전트용 메모리 대비 후진적.

### 1.1 3계층 (§3)

| 계층 | 정의 |
|---|---|
| **𝒢_inter** (Interaction/Utterance Graph) | 노드 = atomic utterance (A_i 발화 에이전트, m_i 내용), 엣지 = 발화 간 시간 관계 |
| **𝒢_query** (Query Graph) | 노드 `q_i ≜ (Q_i, Ψ_i, 𝒢_inter^(Q_i))`, **Ψ ∈ {Failed, Resolved}**, 엣지 = 질의 간 의미 관계 |
| **𝒢_insight** (Insight Graph) | 노드 `ι_k = (κ_k 내용, Ω_k supporting query set)`, 엣지 = **하이퍼엣지 (ι_m, ι_n, q_j)** — 질의 q_j를 매개로 insight m이 n을 맥락화 |

### 1.2 읽기 — bi-directional memory traversal (§4.1–4.2)

- **식(4)** coarse: 새 질의 Q 임베딩 → query graph top-k 유사 질의
- **식(5)** expansion: 그 노드들의 **1-hop 이웃**으로 확장 → 후보 𝒬̃
- **식(6)** upward Π_Q→I: **Ω_k ∩ 𝒬̃ ≠ ∅** 인 insight를 끌어올림 (임베딩 유사도가 **아님**)
- **식(7)** downward: R_LLM이 과거 질의 관련도를 채점해 top-M 선택 → **S_LLM(LLM graph sparsifier)** 이
  각 𝒢_inter에서 core subgraph만 남김
- **식(8)** role-aware filter Φ → 에이전트별 Mem_i 초기화

### 1.3 쓰기 — hierarchy memory update (§4.3)

- interaction 층: 이번 태스크의 발화 그래프 저장
- **식(9)**: query 노드 + 엣지 추가
- **식(10)**: `ι_new = (𝒥(𝒢_inter^(Q), Ψ), {q_new})` — **요약함수 𝒥** 로 신규 insight 생성
- **식(11)**: 사용된 insight의 `Ω_k ← Ω_k ∪ {q_new}`

### 1.4 실험·수치

- 벤치 5종: HotpotQA·FEVER(지식 QA) / ALFWorld·SciWorld(embodied) / PDDL(planning)
- MAS 3종: AutoGen, DyLAN, MacNet — **프레임워크 수정 없이** 외부 메모리 모듈로 부착
- 백본 3종: Qwen2.5-7B / Qwen2.5-14B / GPT-4o-mini
- 향상: embodied **+20.89%p**, 지식 QA **+10.12%p**까지.
  GPT-4o-mini 평균 AutoGen 57.18%(+8.91) / DyLAN 50.88%(+7.76) / MacNet 51.95%(+9.94)
- ablation: fine-grained interaction 제거 시 −4.47(AutoGen)/−3.82(DyLAN),
  insight 제거 시 −3.95/−3.39
- 민감도: **1-hop 최적**("2-hop, 3-hop은 종종 성능을 떨어뜨린다"), **k ∈ {1,2}**, k=5는 크게 악화.
  ALFWorld 85.82% / PDDL 55.24%(AutoGen)가 1-hop 피크
- 비용: PDDL +10.32%를 **1.4×10⁶ 토큰**으로 (MetaGPT-M은 2.2×10⁶에 +4.07%).
  **LLM 콜 수는 논문 어디에도 없다** — 총 토큰만 보고(§5.3, Fig 3/7)
- 논문은 **insight score / reward / ADD·EDIT·REMOVE·AGREE / 클러스터 병합을 한 번도 언급하지 않는다**
  (식10–11의 생성·Ω 갱신만) — GM-1의 근거

---

## 2. 공식 코드 지도

repo는 **G-Memory 라이브러리가 아니라 MAS 벤치마크 하네스**다. G-Memory는
`mas/memory/mas_memory/` 의 플러그인 7종 중 하나(나머지 chatdev, metagpt, voyager, generative,
memorybank, empty — `mas/module_map.py`).

### 2.1 논문 용어 ↔ 코드 실명

| 논문 | 코드 | 실체 |
|---|---|---|
| 𝒢_insight | `InsightsManager` (`GMemory.py:467-892`) | **JSON 리스트** `{rule, score, positive_correlation_tasks, negative_correlation_tasks}`. Ω_k = positive_correlation_tasks. **하이퍼엣지 없음** |
| 𝒢_query | `TaskLayer` (`:352-462`) | `networkx.Graph` + pickle 사이드카(`{namespace}_graph.pkl`). 노드 = **task 문자열 그 자체** |
| 𝒢_inter | `StateChain`/`MASMessage` (`common.py:53-191`) | state별 `nx.DiGraph`, 노드=`AgentMessage`, 엣지 `edge_type='spatial'`, `node_link_data`로 직렬화해 Chroma 메타데이터에 문자열 저장. **읽는 코드 없음(GM-14)** |
| 식(4) top-k | `TaskLayer.retrieve_related_task` (`:404-423`) | Chroma `similarity_search_with_score` |
| 식(5) 1-hop | 같은 함수, `nx.single_source_shortest_path_length(cutoff=hop)` | hop 기본 1, CLI `--hop` |
| 식(6) Π_Q→I | `query_insights_with_score` (`:490-506`) | SUCC 4 + FAIL 2 태스크 + 현재 질의 → `_find_related_insights` **카운트 투표** |
| 식(7) S_LLM | `_extract_mas_message` (`:244-281`) | **그래프 sparsifier가 아니라 텍스트 요약.** 기계적 reward<0 state 제거 + LLM key-step 추출 |
| 식(7) R_LLM | `retrieve_memory` (`:219-232`) | 성공 궤적마다 1–10점 LLM 채점 → 정렬 → top-k |
| 식(8) Φ | `project_insights` (`:304-350`) | role별 재작성 LLM 콜. **`--use_projector` 켜야만 동작** (autogen.py:235) |
| 식(10) 𝒥 | **부재** | `finetune_insights`(§4)로 대체 |
| 식(11) Ω 갱신 | `_update_rules` (`:808-878`) | ADD/EDIT/AGREE 시 `positive_correlation_tasks` 합집합 |

### 2.2 읽기 한 번의 실제 흐름 (`retrieve_memory`, `:189-241`)

```
질의 task
 → _retrieve_memory_raw(2×succ_topk, 2×fail_topk, 2×insight_topk, threshold)   # 2배 과회수(:216)
     ├ TaskLayer: Chroma top-node_num(:137 = max((s+f)//2,1)) + 1-hop 이웃 → task 문자열 리스트
     ├ 각 문자열을 similarity_search(k=1)[0]로 doc 환원(:140) → label로 성공/실패 분리
     ├ 부족하면 label-필터 벡터검색으로 "보충"(:150-164)  ← 실제로는 그래프 결과를 폐기(GM-13)
     ├ 진짜 cosine 재정렬 + threshold 컷(:122-131, :167-169)
     └ query_insights_with_score: correlation 카운트 투표 top-2×insight_topk(:184)
 → 성공 궤적마다 R_LLM 1–10점 → top succ_topk(:219-232)
 → 실패 궤적은 채점 없이 앞에서 자름(:235)
 → insights_cache = 서빙된 rule 문자열(:239)  ← backward 대상
```

**insight는 한 번도 임베딩되지 않는다.** 회수 채널은 오직 "과거 task 문자열 집합과의 교집합
카운트"다. 우리 포트가 insight에 `embedding_text=None`을 넣는 이유(`organizer.py:625`).

거리 공간 주의: 엣지 게이트(`:390-392`)는 `1 − squared_l2` 위의 0.7 = **실효 cosine 0.85**(GM-2)인데,
같은 함수 안의 retrieval threshold(`:127`)는 `mas/memory/utils.py`의 **진짜 cosine**이다. 한 파일에
두 종류의 유사도가 섞여 있다.

### 2.3 쉬핑된 operating point (= 논문 수치를 낸 값)

`tasks/run.py:128-131` — **함수 시그니처 기본값(2/1/10/0.3)이 아니다**:

```
successful_topk=1, failed_topk=0, insights_topk=3, threshold=0.0, hop=1
```

그리고 세 MAS 워크플로 전부 `successful_trajectories, _, insights = retrieve_memory(...)` —
**실패 궤적 리스트를 버린다**(autogen.py:108 / dylan.py:180 / graph_mas.py:104). 실패 궤적은 읽기
시 에이전트에게 절대 보이지 않고 finetune 입력으로만 쓰인다(GM-5).

기타 상수: `start_insights_threshold=5`, `rounds_per_insights=5`, merge 20태스크마다,
`insights_point_num=5`, `MAX_RULE_THRESHOLD=10`, sparsify temperature 0.1(`:273`),
임베더 all-MiniLM-L6-v2(run.py:74), `random.seed(42)`(run.py:119),
`llm_config: max_token 512, temperature 0.1`(configs/configs.yaml).
**작업 디렉토리 삭제는 주석 처리**(run.py:146-147)라 메모리가 실행 간 누적된다.

---

## 3. MAS·에이전트·환경 계층 (우리가 안 가져온 부분)

### 3.1 에이전트/추론 — 매우 얇다

- `mas/agents/base.py` (53줄): `Agent = {name, profile(role), system_instruction, reasoning, memory=None}`.
  `response()`는 system+user 2메시지 1콜. **에이전트별 메모리는 항상 `None`** — 메모리는 MAS 전체가
  공유하는 `meta_memory` 하나뿐.
- `mas/reasoning/reasoning_modules.py` (34줄): `ReasoningIO` 하나. `--reasoning io` 외 선택지 없음.
  CoT/ToT 류 없음.
- `mas/mas.py` (35줄): `MetaMAS = {agents_team, env, meta_memory}` + `hire/schedule`.

### 3.2 MAS 워크플로 3종

| | 구성(기본값, `tasks/configs.yaml`) | 협업 구조 |
|---|---|---|
| **AutoGen** (`autogen/autogen.py`, 247줄) | 에이전트 2개: `solver` + `ground_truth` | 사실상 단일 에이전트 루프. 같은 액션 3연속이면(`_solver_stuck`, :205-219) `ground_truth`가 그 스텝만 대행. **에이전트 간 대화 없음**, `add_agent_node(..., upstream_agent_ids=[])`(:184)라 발화 그래프에 **엣지가 생기지 않음** |
| **DyLAN** (`dylan/dylan.py`, 426줄) | `node_num=2 × round_num=2` 뉴런 그리드 + decision 에이전트 | 층간 완전연결, importance 전파(lr 0.01), ranker, consensus 조기종료. 진짜 다중 에이전트 토론. **backward 미호출(GM-12)** |
| **MacNet** (`macnet/graph_mas.py`, 376줄) | `graph_type=Random, node_num=3, use_critic=False` | 그래프 위상 따라 upstream 출력을 받아 전파 |

논문의 "MAS"는 **2~4 에이전트 규모**다.

### 3.3 환경 4종 (+ HotpotQA 미포함)

`tasks/envs/` — `alfworld_env.py`(max_steps 30, few-shot 1) / `fever_env.py`(12, 3) /
`sciworld_env.py`(30, 0) / `pddl_env/`(pddlgym 벤더링 — repo 797파일 중 대부분이 이것).
**HotpotQA는 코드·데이터 모두 미포함**(이슈 #12/#15/#21). SciWorld `test.jsonl`은 90샘플
(공식 test split ~1800).

### 3.4 "failure"의 정확한 정의 — 두 층위

1. **태스크 레벨 label** — `env.feedback() → (reward, done, message)`의 `done`이 그대로
   `label: bool`로 저장된다(`memory_base.py:73-82`). ALFWorld는 `self.done`,
   FEVER는 `reward == 1`(`fever_env.py:113-118`), SciWorld는 progress_rate/done 일치 검사 후
   `done`(`sciworld_env.py:93-101`). 피드백 문장(`"You failed the task."`)은 `task_description`에
   append되어 이후 프롬프트에 그대로 실린다. 이 label이 ① 성공/실패 채널 분리 ② `backward` 부호
   ③ 실패 시 `_detect_mistakes` 추가 콜을 결정한다. **논문의 Ψ ∈ {Failed, Resolved}가 이것.**
2. **스텝 레벨 reward** — ALFWorld 기준 `think:` 액션이면 **−1**, 관측이 `"Nothing happens."`면
   **−1**, 그 외 0/1(`alfworld_env.py:62-70`). sparsification이 `reward < 0`인 state를
   제거한다(`GMemory.py:249-251`). 즉 **"희소화"의 실체는 '생각 스텝과 헛발질 스텝 삭제'** 이며,
   논문이 말한 LLM graph sparsifier가 아니라 규칙 필터 + 요약 프롬프트다.

우리 쪽엔 ①만 있다(`outcome`). ②는 `trajectory: list[dict]`에 step reward가 실려 올 수는 있으나
계약이 없고 organizer도 읽지 않는다.

---

## 4. "finetune"의 정체

**모델 파인튜닝이 아니다.** 가중치는 전혀 건드리지 않는다. `InsightsManager.finetune_insights`
(`:647-676`)는 **규칙 리스트를 LLM 비평으로 편집하는 루프**이며, 출처는 README가 명시한
**ExpeL**(LeapLabTHU/ExpeL — "Experiential Learning", Reflexion 계보) 프롬프트 디자인이다.
데이터셋 환경·베이스라인은 **AgentSquare** 유래(README Acknowledgement).

트리거: `memory_size >= 5 and memory_size % 5 == 0`(`:106`). 1회 이벤트는

```
5회 반복 (insights_point_num):
  랜덤 저장 태스크 1개를 앵커로 선택(:654)         ← 현재 태스크가 아니다
  앵커와 유사한 성공 3 + 실패 1 회수(:649, 663-665)
  앵커 자신을 자기 라벨 쪽에 append (중복 제거 없음, :666-669)
  보여줄 rule = correlation 겹침 >= len(tasks)/2 (:672)
  ├ (성공,실패) 쌍마다 comparative 프롬프트 1콜(:719-729)   ← 실패의 fail_reason이 슬롯
  └ 성공 궤적 5개 청크(random_divide_list)마다 success 프롬프트 1콜(:711, 734-745)
  응답을 정규식 파싱: (REMOVE|EDIT|ADD|AGREE)( \d+)?: <rule>  (:793)
  프롬프트당 최대 4연산, 기존 rule당 최대 1연산 (UP:312)
  rule 형식 강제: "XXX, because XXX"
score <= 0 프루닝(clear_insights, :584-586) → JSON 저장
```

연산 의미론(`_update_rules`, `:808-878`): **ADD score=2 신규 / EDIT 교체 +1 / AGREE +1 /
REMOVE −1(리스트 >=10이면 −3)**. cap 10은 **soft** — ADD는 무조건 실행되고, 초과 시 프롬프트
suffix("stop ADD unless VERY insightful")만 붙는다(GM-6).

여기에 태스크 종료 시 `backward(final_done)`가 **그 태스크에서 실제로 서빙된 rule에만**
+1 / −2를 적용한다(`:292-297`). 매칭은 **부분문자열**(`:578` `if insight in inner_insight['rule']`)이라
한 rule 텍스트가 다른 rule의 부분문자열이면 동시 가점된다.

20태스크마다 `merge_insights`(`:508-549`): task 임베딩 FINCH 클러스터링 → 클러스터별 rule LLM 병합
(`limited_number = (len(batch)//3)//3` 이중 나눗셈 quirk, GM-3) → **insight 리스트를 통째로 clear한 뒤
score=2로 재생성**. 즉 20태스크마다 모든 보상 이력이 리셋된다. (우리는 미포트)

---

## 5. 태스크 1개당 LLM 콜 예산 (하네스 operating point, projector off)

| 시점 | 콜 수 |
|---|---|
| 읽기 R_LLM 채점 | 2 (2×succ_topk 후보) |
| 읽기 role projection | 0 — `--use_projector` 켜면 role 수만큼(AutoGen 2) |
| 쓰기 sparsify | 1 (+실패면 `_detect_mistakes` 1) |
| 5태스크마다 finetune | ~10 (5앵커 × 약 2콜) → 태스크당 평균 ~2 |
| 20태스크마다 merge | 클러스터 수 × ⌈rule/10⌉ |

**평균 태스크당 5~6콜**이 메모리 유지비. 논문은 이를 콜 수로 보고하지 않고 총 토큰(1.4×10⁶)으로만
보고한다 — 트랙5에서 확인한 "비용은 콜이 아니라 토큰" 함정과 같은 구간이다.

---

## 6. 신규 결함 4건 (원장 GM-12 ~ GM-15)

전부 2026-08-17 재조사에서 확인. 검증 방법을 함께 적어 재확인 가능하게 남긴다.

### GM-12 · DyLAN에서는 reward shaping이 한 번도 돌지 않는다

`backward` 호출부는 **`autogen.py:201`과 `graph_mas.py:197` 두 곳뿐**이다.
`dylan/dylan.py`의 `schedule`은 `save_task_context(label=final_done, feedback=...)` 직후
`return final_reward, final_done`으로 끝난다(`:262-264`).

- 검증: `grep -rn 'backward' --include='*.py' .` → GMemory 내부 정의 2건, memory_base 추상 1건,
  호출 2건(autogen, graph_mas). dylan 0건.
- 영향: **논문 Table 1/2의 DyLAN 컬럼 전체가 insight score +1/−2와 score<=0 프루닝이 한 번도
  발화하지 않은 상태의 수치**다. ablation의 "insight 기여분"도 DyLAN에선 *정적* 규칙 리스트의
  기여분이며, 세 MAS 프레임워크가 같은 메모리를 썼다는 전제가 깨진다.
- 우리 쪽: 무해(우리는 `on_feedback` 하나로 통일). 인용 시 "DyLAN 수치는 reward-shaping 없는 arm"이라
  명시해야 한다.

### GM-13 · 식(5) 1-hop 확장 결과를 통째로 버리는 분기

`GMemory.py:150-164`:

```python
if len(true_tasks_doc) < successful_topk:
    true_tasks_doc = self.main_memory.similarity_search(...)   # 그래프 결과를 재바인딩으로 폐기
    for doc in true_tasks_doc:
        if doc not in true_tasks_doc:      # 자기 자신에 대한 멤버십 검사 — 항상 False
            true_tasks_doc.append(doc)
```

주석은 "부족분을 유사도 증강으로 채운다"지만 **재바인딩이라 그래프에서 온 문서를 전부 버리고**
평범한 label-필터 벡터 검색으로 교체한다. 뒤따르는 루프는 자기 리스트에 대한 멤버십 검사라 죽은 코드다.
실패 채널(`:158-164`)도 동일 형태.

- 발화 조건: 하네스 값에서 `successful_topk`는 2배된 **2**, 그래프 경로는
  `related_point_num = max((2+0)//2, 1) = 1`이므로 top-1 노드 + 그 1-hop 이웃뿐. 성공 라벨 문서가
  2개 미만이면 폐기 → **그래프에 엣지가 없는 초기 구간에서는 100% 폐기**되고, 엣지 게이트가
  실효 cos 0.85(GM-2)로 빡빡해 엣지는 드물게 생긴다.
- 영향: 논문의 **핵심 민감도 축(1-hop vs 2/3-hop)** 이 상당 비율의 호출에서 아무 효과도 갖지 못한다.
  "1-hop이 최적, 2-hop은 악화"라는 관찰의 해석에 직접 영향.
- 부수: 살아남은 경우에도 task 문자열 → doc 환원이 `similarity_search(task_main, k=1)[0]`(`:140`)라
  정확 조회가 아니다(근사 문자열이 다른 doc를 물어올 수 있음).
- 우리 쪽: `TaskGraphExpansion`은 폐기 분기가 없다 — 이웃을 진짜 cosine으로 재채점해 threshold 컷만
  적용한다. **의도치 않게 upstream보다 식(5)에 충실**하다. 재현 arm을 만들 거라면 이 분기를
  재현해야 한다.

### GM-14 · interaction graph는 write-only — 노드·엣지를 읽는 코드가 없다

`StateChain`의 노드(`AgentMessage`)와 `edge_type='spatial'` 엣지를 **읽는 코드가 repo 전체에 없다.**
유일한 소비처인 `_extract_mas_message`는 `state.graph.get('reward')` / `state.graph['action']` /
`state.graph['observation']`만 읽는데(`:250-255`), 이는 `move_state`가 붙인 **그래프 수준 속성**이지
발화 노드가 아니다.

- 검증: `grep -rn '\.nodes|\.edges|node_link_data|has_node' mas/ tasks/mas_workflow/ tasks/run.py`
  → `.nodes` 3건은 전부 `TaskLayer.graph`(= query graph, FINCH 클러스터링용, `:432,453,462`),
  `has_node`는 write 시 검증(`common.py:84`), `node_link_data`는 직렬화(`common.py:127`).
- 게다가 AutoGen은 `upstream_agent_ids=[]`(`autogen.py:184`)라 **엣지 자체가 생성되지 않는다** —
  대표 표(AutoGen)의 "interaction graph"는 단일 노드 그래프의 나열이다. DyLAN/MacNet은 진짜
  upstream id를 넘기지만(`dylan.py:235`, `graph_mas.py:164`) 역시 아무도 읽지 않는다.
- 즉 **바닥 계층의 실체는 "reward<0 스텝을 뺀 선형 action/observation 로그"** 다. 직렬화된 그래프는
  Chroma 메타데이터에 문자열로 쌓이고 매 회수 때 `MASMessage.from_dict`가 파싱만 한다(`common.py:190`).
- 영향: 논문 초록의 첫 번째 문제제기("협업 궤적을 무시한다")에 대한 해법이 코드에선 **저장되되
  사용되지 않는다**. ablation의 "fine-grained interaction 제거 시 −4.47"은 *그래프* 제거가 아니라
  *조건부 궤적 텍스트* 제거의 효과다.
- 우리 쪽: 미포트가 손실 0임을 이 항목이 증명한다. docs/16 세션 5의 "발화 그래프는 upstream도
  텍스트로 평탄화하므로 손실 아님"에 **"애초에 읽히지도 않는다"** 를 추가한다.

### GM-15 · key-step 추출 전 궤적의 모든 숫자를 제거한다

`GMemory.py:261` — `trajectory = re.sub(r'\d+', '', trajectory)` → `clean_traj`(`:262`)가
key-step 추출 프롬프트의 입력이 된다(`:270`). `go to drawer 3` → `go to drawer `.

- ALFWorld/PDDL은 객체 인덱스가 의미의 전부인 환경이고, few-shot 예시 자체가 `bowl 1`, `desk 2`,
  `drawer (1-6)` 같은 인덱스로 쓰여 있다(prompt.py의 `extract_true_traj_user_prompt`).
- 산출물 `key_steps`는 읽기 시 그대로 프롬프트에 주입된다(`format.py:60-65`,
  `autogen.py:116` / `dylan.py:188` / `graph_mas.py:112`).
- 실패 궤적에도 적용된다 — `_detect_mistakes`도 `clean_traj`를 입력으로 받는다(`:285`).
- 우리 쪽: 미재현(의도적).

### 보조 관찰 (원장 등재까지는 아님)

- **sparsify 비대칭**: `task_trajectory` 치환은 `label == True`일 때만(`:257-258`). 실패 궤적은
  원문 그대로 저장되어 comparative 프롬프트의 `task2_trajectory` 슬롯에 raw로 들어간다.
- **backward 부분문자열 매칭**(`:578`): rule A가 rule B의 부분문자열이면 A 서빙 시 B도 가점.
- **backward의 O(n) 쓰기**: 서빙 insight 하나마다 `clear_insights()` + JSON 전체 재기록(`:581-582`).
- **작업 디렉토리 미삭제**(run.py:146-147): 이전 실행의 그래프/insight/Chroma가 누적된 상태로
  다음 실행이 시작된다 — 재현 편차(#22)의 후보 원인.

---

## 7. 우리 포트 지도 — "그래프가 안 보이는" 이유

없는 게 아니라 **이름이 바뀌어 아이템 필드로 들어가 있다.**

| 논문 개념 | 우리 위치 | 형태 |
|---|---|---|
| Query graph 노드/엣지 (식9) | `organizers/gmemory/organizer.py:370-409` (`on_task_end`) | pickle 사이드카 대신 **아이템 payload의 인접 리스트** `task_edges`. 이웃에 `UPDATE`로 역엣지를 붙여 무방향 유지. 게이트는 **0.85 true cosine**(`:399`), top-10 후보 내 |
| 식(5) 1-hop + 재랭크 | `retrieval/steps.py:881` `TaskGraphExpansion` | 이웃을 질의와의 진짜 cosine으로 재채점 후 threshold 컷. **hop은 1로 하드코딩**(upstream은 `--hop`) |
| 식(6) Π_Q→I | 같은 스텝 후반부 | 서빙된 궤적의 **full task 문자열** ∩ `positive_correlation_tasks` 카운트 정렬 → `insight_cap` |
| Ω_k | 아이템 필드 `positive_correlation_tasks` / `negative_correlation_tasks` | round-5 시점엔 없었고 round-12에서 추가 |
| Insight 하이퍼엣지 | 없음 | **upstream에도 없음** — 논문에만 존재 |
| 𝒢_inter | 없음 (`list[dict]` → `json.dumps` 평문) | upstream도 읽지 않음(GM-14) |
| 식(7) S_LLM | `SPARSIFY_PROMPT` + 실패 시 `MISTAKES_PROMPT` 2콜(`:325-344`) | upstream 2콜 구조와 콜 수 일치 |
| 식(7) R_LLM 리랭크 | **미포트** | 읽기 측 유일한 LLM 신호 누락 |
| 식(8) Φ | `organizer.py:662` `project_insights()` | 구현됨, **호출자 없음**(MAS 루프가 부를 public API) |
| finetune | `organizer.py:441-596` | 5 랜덤 앵커 × (compare-pair + success-chunk), 점수 상수 동일, 구조화 JSON ops |
| backward | `organizer.py:674` `on_feedback` | +1/−2, **서빙 insight에만**, 궤적 불변, <=0 프루닝 |
| FINCH merge | **미포트**(이연) | GM-3 |
| operating point | `organizer.py:85-90` `GMEMORY_READ_RECIPE` | `{1, 0, 3, 0.0}` — Zep 레시피 표와 같은 방식 |
| 설정 | `config.py:115-127` | `task_graph_expansion_cap=5`, `task_graph_insight_cap=10`(0이면 스텝 비활성) |
| 테스트 | `tests/test_organizers_phase3.py:723-890` 7개 | 엣지 게이트 0.85, 반복 태스크 스킵, 서빙 insight 한정 피드백, read recipe 상수, hop 확장 |
| 결함 재현 | `scripts/repro/defects/repro_gmemory_threshold.py` | GM-2 Tier-0 |

의도적 **비재현**: rule을 id로 주소지정(AGREE `-1` 오가점 구조적으로 불가, GM-4),
EDIT→AGREE 강등 없음, 숫자 제거 없음(GM-15), RNG는 organizer 소유 시드(`finetune_seed=0`).

---

## 8. 미포트 목록과 판단

| 미포트 | 이유 | 판단 |
|---|---|---|
| MAS 하네스 전체(워크플로 3종, agent, reasoning, envs, pddlgym) | 메모리 방법론이 아님 | **정당.** 단 이것 없이는 측정 무대가 없다 |
| interaction graph | 평탄화 | **손실 0** — GM-14가 증명 |
| R_LLM 리랭크(식7 상단) | 읽기 측 LLM 비용 | **재고 여지** — 읽기 경로에서 유일하게 빠진 LLM 신호 |
| FINCH merge | 이연 | 정당(20태스크마다 전체 리셋, 재현 가치 대비 위험 큼) |
| role projection 호출자 | MAS 루프가 부를 API | 구현 있음, **호출자 없음 = 미측정** |
| 스텝 레벨 reward 필터(think/헛발질 제거) | 궤적 계약에 step reward 없음 | 계약 확장이 필요한 유일 항목 |
| hop 설정화 | 1 하드코딩 | 논문 민감도 축이 hop이므로 열어두는 편이 나음(소규모) |

---

## 9. 다시 볼 때 체크리스트

1. **측정 얘기가 나오면 먼저**: 우리 벤치(LoCoMo/LongMemEval)에는 task 성공/실패 라벨 루프가 없다.
   G-Memory를 측정하려면 MAS 벤치를 세우거나(비용 큼) ALFWorld류 환경을 붙여야 한다.
   "코드가 있다 = 측정했다"로 오독 금지(`longmemeval-implemented-never-run` 교훈과 동형).
2. **인용 시 주의**: DyLAN 수치는 reward-shaping 없는 arm(GM-12) · 모든 그래프 밀도는 실효 cos 0.85
   기준(GM-2) · 실패 궤적은 읽기 시 보이지 않음(GM-5) · 1-hop 민감도 해석은 GM-13과 함께 읽을 것.
3. **재현 arm을 만든다면** 재현해야 할 결함: GM-13(폐기 분기), GM-15(숫자 제거), GM-4(AGREE −1),
   GM-3(FINCH 전체 리셋). 우리 현행 포트는 이들을 고쳐 놓았으므로 upstream-faithful arm이 아니다.
4. **라이선스**: LICENSE 파일 없음(2026-07-27 GitHub API `license: null` 재확인). 동작·상수만 재현,
   원문 미복제 원칙 유지.
5. 남은 열린 질문: R_LLM 리랭크의 기여분(측정 불가 상태) · `hop` 설정화 여부 ·
   step reward 계약 도입 여부.
