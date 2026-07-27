# 10차: 추적만 하던 항목 정리 — MaTTS 구현, eval 계보 read 완성, Zep 분류 확정 (2026-07-27)

> 범위: 9차 이후 남은 "추적만/미세 항목/훅만 존재"로 분류돼 있던 것들. 분류가 맞는지부터
> 확인하고, 틀린 것은 고치고 맞는 것은 근거를 붙였다.
> 재확인 소스: `google-research/reasoning-bank:WebArena/{induce_memory,induce_scaling,
> memory_management}.py`·`prompts/memory_instruction.py`,
> `BAI-LAB/MemoryOS:eval/{mid_term_memory,retrieval_and_answer,dynamic_update,utils}.py`,
> `getzep/graphiti` 트리, 논문 arXiv:2509.25140 §3.3 · arXiv:2501.13956 §2.2·§6.1.

## 판정 요약

| # | 추적 항목 | 기록돼 있던 분류 | 실제 | 조치 |
|---|---|---|---|---|
| R1 | ReasoningBank MaTTS | "훅만 존재", 등급 ● | **훅이 없다.** 등급도 틀렸다 | **구현** + 등급 ●→◑⁺ |
| R2 | MaTTS sequential | (미분류) | 에이전트 소유 — 계약 밖 | 근거 첨부, 미구현 확정 |
| M8 | MemoryOS "eval 계보 검색 상수 — 미세 항목" | 미세 | **미세가 아니다**: 살아 있는 LLM keyword 채널 | **구현** |
| M9 | 1단계 segment 게이트 | (미인지) | 두 계보 모두 적용, 우리는 미적용 | **구현** |
| Z4 | Zep SagaNode·attribute ontology·combined 추출 | "논문 이후 업스트림 추가물 — 추적만" | **맞다** | 근거 확정, 추적 유지 |

---

# A. ReasoningBank

## R1. "MaTTS 훅만 존재"는 사실이 아니었다 — 훅이 없었다

docs/10은 ReasoningBank를 **●(핵심 메커니즘 전부)**로 두고 누락 칸에 "MaTTS — 훅만 존재"라고
적어 왔다. 검색하면 `matts|contrast|scaling`이 이 패키지 전체에서 **`__init__.py` docstring 한
줄**에만 나온다. 훅은 없었고, `on_task_end`는 궤적을 **하나만** 받으므로 여러 궤적을 받을
진입점 자체가 없었다.

논문 §3.3과 공식 코드가 경계를 명확히 그어 준다. 업스트림은 단일 궤적 유도
(`induce_memory.py` + `SUCCESSFUL_SI`/`FAILED_SI`)와 **스케일링 유도**
(`induce_scaling.py` + `PARALLEL_SI`)를 **별도 모듈**로 둔다. 후자는 한 질의의 여러 궤적을
한 프롬프트에 넣고 self-contrast로 메모리를 뽑는다 — 즉 **메모리 계층의 일**이다.
논문도 같다: *"self-contrast across multiple trajectories curates reliable memory."*

세 상수가 단일 궤적 경로와 다르고 전부 업스트림 값이다:

| | 단일 궤적 | MaTTS parallel |
|---|---|---|
| 아이템 상한 | 3 (`SUCCESSFUL_SI`) | **5** (`PARALLEL_SI`) |
| content 길이 | 1-3 문장 | **1-5 문장** |
| 온도 | 1.0 (`induce_memory.py:164`) | **0.7** (`induce_scaling.py:196`) |

**조치**: `Organizer.on_scaled_task_end(trajectories, task, ctx)` 훅 신설(기본 no-op),
`ReasoningBankOrganizer`가 `PARALLEL_SI` 전사본으로 구현, `AgenticMemory.add_scaled_task_result`
로 노출. 궤적 1개면 단일 경로로 폴백한다 — 하나짜리 집합을 대조하는 것은 그 메커니즘이 아니다.
온도는 organizer 상수가 아니라 role 설정이므로 docstring에 명시만 했다.

### 업스트림 결함: 정답 라벨이 프롬프트에 도달하지 않는다

`induce_scaling.main()`은 궤적마다 `status`(success/fail)를 계산해 `get_info`에 넘기지만,
그 필드를 **읽는 곳이 없다**. "## Correctness Signal" 블록을 만드는 `format_examples`
헬퍼는 `main()`에서 **호출되지 않는다**. 즉 parallel 유도가 실제로 쓰는 신호는 라벨이 아니라
**섞임 그 자체**("Some trajectories may be successful, and others may have failed")다.

그래서 우리 훅에도 **궤적별 outcome 인자를 두지 않았다** — 두면 메커니즘에 없는 채널을
만드는 것이다. (MemoryOS의 죽은 R_recency, A-MAC의 죽은 N/R과 같은 계열: 재현하되 고치지
않는다.) 아이템은 `outcome="contrast"`로 적재한다. 섞인 집합에서 나온 것이라 success도
failure도 참이 아니고, 업스트림 parallel 뱅크에도 per-item 라벨이 없다.

**등급 재산정 ● → ◑⁺.** 논문 제목이 내건 주장의 절반이 메모리 측에서 비어 있었는데 ●였다.
이제 parallel은 구현됐고 sequential은 계약 밖(R2)이므로 ◑⁺.

## R2. sequential scaling은 구현하지 않는다 — 에이전트가 소유한다

`SEQUENTIAL_PROMPT`(memory_instruction.py:132)는 **에이전트에게** 하는 말이다:
*"Let's carefully re-examine the previous trajectory ... Output must stay in the same
`<think>...</think><action>` format as previous trajectories."* 즉 궤적을 다시 쓰게 하는
self-refinement 루프이고, 메모리 계층은 그 결과 궤적을 평소대로 `on_task_end`로 볼 뿐이다.

**ACE multi-round reflection(8차 §A3)이 놓인 것과 정확히 같은 선**이다. 그때 세운 기준
("재생성을 하면 생성기의 것")을 그대로 적용하면 sequential은 계약 밖이다.

---

# B. MemoryOS

## M8. "eval 계보 검색 상수 — 미세 항목"은 미세하지 않았다

8차 §M7이 *"검색 단계의 keyword 항은 죽어 있다"*고 확정하고 그것을 "A-MAC의 죽은 N/R,
MemoryOS eval의 죽은 R_recency와 같은 계열"로 분류했다. **그 확인은 `memoryos-pypi`
한 벌만 보고 내린 것이다.** eval 계보에서는 살아 있다:

```python
# eval/mid_term_memory.py:200  (pypi는 같은 자리에서 query_keywords = set())
query_keywords = llm_extract_keywords(query, client)     # ← 질의당 LLM 콜 1회, 최대 3개
...
s_top = 0.5 * (len(overlap)/len(query_keywords) + len(overlap)/len(session_keywords))
overall = lambda_t * (dist + alpha * s_top)               # alpha=1.0
```

즉 **논문 LoCoMo 수치를 낸 계보는 read 시점에 질문당 LLM 콜을 하나 더 쓰고**, 그 키워드
겹침(merge와 같은 containment mean)을 세그먼트 점수에 더한다. docs/10이 이를 "page 임베딩
텍스트 등 미세 항목"으로 적어 온 것은 과소 기재였고, 그대로 두면 `memoryos_eval` 프리셋이
"논문 수치를 재현 가능"하다는 주장을 지탱하지 못한다.

**조치**: `ReadContext.query_keywords` 신설 → `MemoryOSPageRecall._relevance`가
`cos + alpha * containment_mean`을 계산. 키워드가 비면 순수 cosine이 되므로 **한 스텝이 두
계보를 모두 담는다**(pypi는 구조적으로 빈 집합). 추출은 파이프라인이 아니라 벤치에서 한다 —
검색 계층에는 LLM이 없고, A-Mem의 keyword query 재작성이 같은 이유로 벤치에 있다.

`lambda_t`(검색 시점 recency)는 **두 계보 모두 죽어 있다**: eval 파일이 `lambda_t = 1`로
두고 decay 줄을 주석 처리했다. 재현하되 고치지 않는다.

## M9. 1단계 segment 게이트가 통째로 빠져 있었다

두 계보 모두 `if session_relevance_score >= segment_similarity_threshold:` 안에서만 page를
채점한다. 우리 `MemoryOSPageRecall`은 이 게이트 없이 융합 랭킹으로 들어온 세그먼트를 전부
펼치고 있었다. 요약이 질의와 무관해도 안의 page가 맞으면 서빙됐다는 뜻이고, 업스트림은
그 경우 **아무것도 반환하지 않는다**.

`page_recall_segment_threshold`(기본 0.1, 두 드라이버 값) 신설. 점수는 융합 랭크가 아니라
**세그먼트 자신의 summary 벡터**로 다시 계산한다 — 업스트림이 매칭하는 대상이 그것이고,
융합 점수는 그것이 아니기 때문이다.

## 계보 knob 통합 (9차 L2의 후속)

9차가 `assistant_knowledge_mode`를 분리했는데, M8이 두 번째 read 계보 knob를 추가하게 됐다.
독립 knob 두 개는 9차가 적발한 결함(어느 upstream에도 없는 조합)을 다시 만드는 구조이므로
**`memoryos_lineage="pypi"|"eval"` 하나로 합쳤다.** assistant-knowledge 전량 주입과 질의
키워드 추출이 함께 움직인다. `page_recall_cap`(7/10)만 생성 시점 config라 별도이고,
실험 테이블의 `memoryos_eval` 항목이 셋을 한자리에서 편다.

---

# C. Zep — "추적만" 분류는 옳았다

논문(arXiv:2501.13956)은 **v1이 유일본, 2025-01-20 제출**이고 세 항목 모두 없다:

- **SagaNode**: 논문에 없음. 업스트림에는 `prompts/summarize_sagas.py` + 드라이버 3벌
  (`saga_node_ops.py`)로 존재 → 논문 이후 추가물.
- **entity attribute ontology**: 논문에 없음(§6.1.1 엔티티 추출에 속성 스키마 없음).
  업스트림 `utils/maintenance/attribute_utils.py` → 추가물.
- **combined 단일콜 추출**: 논문 §2.2 *"extracts semantic entities and facts in **separate
  stages**"*, §6.1이 엔티티/사실/시간 프롬프트를 각각 둔다. 업스트림
  `prompts/extract_nodes_and_edges.py` + `utils/maintenance/combined_extraction.py`는
  **논문과 반대 방향의 최적화** → 추가물.

우리 구현은 분리 추출이므로 **논문 쪽에 서 있다.** 세 항목은 추적만 유지하고, docs/10 행에
"논문 v1에 없음 확인(10차)"을 명기했다.

---

# 검증

```
339 passed, 1 skipped     (9차 336 + 신규 3)
97 files already formatted
Found 53 errors (src/tests) / 7 (scripts)   ← 기준선 동일, 신규 0
```

신규 회귀 3건:
- `test_matts_contrasts_the_trajectory_set_instead_of_judging_each` (R1)
- `test_a_single_attempt_is_not_a_contrast_and_falls_back` (R1)
- `test_the_segment_keyword_term_is_dead_in_pypi_and_live_in_the_eval_lineage` (M8/M9)

기존 테스트 1건 수정: `test_memoryos_dialogue_chain_summarizes_and_renders`의 stub 요약이
질의어를 공유하도록 바꿨다. M9의 1단계 게이트가 실제로 무는 것을 보여 주는 변경이다 —
요약이 질의와 겹치지 않으면 안의 page가 완벽히 맞아도 서빙되지 않는다(업스트림 동작).

# 변경 파일

```
src/agmem/organizers/base.py                      on_scaled_task_end 훅
src/agmem/organizers/reasoning_bank/organizer.py  PARALLEL_SI 전사 · on_scaled_task_end · _emit
src/agmem/memory.py                               add_scaled_task_result · search(query_keywords=)
src/agmem/core/types.py                           StrategyItem.outcome 어휘에 contrast
src/agmem/retrieval/steps.py                      ReadContext.query_keywords · _relevance
                                                  · segment 게이트 · _cosine 헬퍼
src/agmem/retrieval/pipeline.py                   query_keywords · page_recall_segment_threshold
src/agmem/config.py                               page_recall_segment_threshold
src/agmem/bench/locomo.py                         memoryos_lineage (2 knob 통합) · eval 키워드 추출
scripts/exp_locomo_conv0.py                       memoryos_lineage 배선
tests/{test_organizers,test_organizers_phase3,test_locomo}.py
```
