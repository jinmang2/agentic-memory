# 9차: 8차 재검증 — 계보 혼합 2건과 낡은 계약 3건 (2026-07-27)

> 범위: 8차가 "측정 외 마무리"를 선언했으므로, 그 선언 자체를 검증했다. 방법은 8차와 같다 —
> 업스트림 raw를 당일 다시 받아 주장 하나하나를 대조하고, 그 다음 **우리 코드가 그 주장대로
> 되어 있는지**를 따로 확인했다.
> 재확인한 소스: `BAI-LAB/MemoryOS:memoryos-pypi/{short_term,updater,memoryos,mid_term,
> retriever,utils}.py`·`eval/{main_loco_parse,retrieval_and_answer}.py`,
> `getzep/graphiti:graphiti_core/prompts/extract_edges.py`·`search/{search,
> search_config_recipes}.py`, `ace-agent/ace:ace/ace.py`·`playbook_utils.py`·
> `ace/prompts/curator.py`.

## 8차 주장 검증 결과: 업스트림 대조는 전부 CONFIRMED

| 8차 주장 | 업스트림 근거 | 판정 |
|---|---|---|
| M1 STM 1-page 롤링 | `short_term.py:31` `is_full()` = `>=`; `updater.py:101` `while is_full(): pop_oldest()` | ✓ |
| M2 페이지당 continuity+meta 2콜 | `updater.py:130,142` (양쪽 분기 모두) | ✓ |
| M2 주입 지점 = 검색된 page 옆 | `memoryos.py:277` `Conversation chain overview:` | ✓ |
| M3 STM 전량 QA 주입 | `memoryos.py:269-273` `history_text` | ✓ |
| M4 pypi=top-20 / eval=전량 | `retriever.py:86` vs `main_loco_parse.py:103` | ✓ |
| M5 eval 2콜 프로필 | `main_loco_parse.py:47-53` | ✓ |
| M7 2단계 검색, summary 미주입 | `mid_term.py:302-342` + `retriever.py:47-68` | ✓ |
| M7 검색 keyword 항 사문 | `mid_term.py:292` `query_keywords = set()` | ✓ |
| U3 빈 `agent_response` page 소실 | `updater.py:103` | ✓ |
| Z1 SCREAMING_SNAKE_CASE | `extract_edges.py:34,166` | ✓ |
| Z3 명시 origin이 파생을 대체 | `search.py:332,540` `and bfs_origin_node_uuids is None` | ✓ |
| A1 budget 80000 + stats 3버킷 | `ace.py:127`; `playbook_utils.py:240-245` | ✓ |
| A2 environment feedback 2문구 | `ace.py:507,554` | ✓ |
| A3 multi-round은 생성기 소유 | `ace.py:501-543` — reflect → **재생성** → 정답 재확인 → break | ✓ |
| 7차 레시피 전사 | `search_config_recipes.py:34,60,86,94` (RRF에 BFS 없음, MMR λ=1) | ✓ |

REFUTED 없음. **논문·공식코드 대조 자체는 8차가 옳았다.**

## 그럼에도 "측정만 남았다"가 아니었던 이유

8차는 read 경로를 크게 바꾸면서 **그 변경이 기존 config·전역 상수와 만나는 지점**을 보지
않았다. 아래 5건은 전부 그 접합부에 있다.

---

## L1. 재측정용 config에 업스트림에 없는 raw 채널이 남아 있었다 (측정 영향)

`memoryos`/`memoryos_eval` 둘 다 `memory_types = ("episodic", "pages", "semantic")`였다.
업스트림 `get_response`의 컨텍스트는 STM history + 검색된 MTM page + profile/knowledge뿐이고,
**전체 원문에 대한 검색 채널이 없다.**

이 프로젝트는 2차 재감사에서 정확히 같은 이유로 `amem`/`nemori`를 순수화하고 구 설정을
`amem_mixed`/`nemori_mixed`로 분리했다(docs/10). **MemoryOS만 그 처리를 받지 않았다.**

그리고 8차가 이걸 악화시켰다: M7 이전에는 `pages`가 요약을 서빙했으므로 원문 경로가
하나(episodic)였지만, 이제 M7이 원문 page를, M3가 원문 STM을 넣으므로 **원문이 세 경로로
들어간다** — 업스트림은 둘이다.

**조치**: `memoryos`/`memoryos_eval`을 `("pages", "semantic")`으로 순수화. `lexical_types=()`
도 함께 — 업스트림 MTM 검색은 dense 단독이고(FAISS IP over summary embedding, keyword 항은
사문) BM25에 대응물이 없다. 구 배선은 **`memoryos_mixed`** 신설로 보존했다(docs/09 런 재현용:
raw 채널 + `flush_stm_on_drain=True` + `dialogue_chain=False` + `page_recall_cap=0`).

## L2. pypi 계보 config가 eval 계보의 read 경로로 읽고 있었다 (측정 영향)

`MEMORYOS_PRESETS` 주석은 *"provenance is per-source and **never mixed** within a preset"*
인데, **프리셋 밖에 있던 두 값**이 그 규칙을 깨고 있었다.

### L2a. assistant-knowledge 전량 주입이 무조건 실행됐다

`bench/locomo.answer`가 `kind == "assistant_knowledge"` 아이템을 **모든 config에서** 전량
주입했다. 바로 위 주석이 *"eval 계보가 전량 주입, pypi는 top-20 검색"*이라고 스스로 적어
놓은 채로. 결과적으로 기본 `memoryos`(=pypi) 런은 **top-20 검색 + 전량 덤프를 동시에** 받았다.

**조치**: `answer(assistant_knowledge_mode="retrieved"|"full")` 신설, 기본 `"retrieved"`.
`persona=`와 같은 형태의 명시 인자다 — 한 계보에만 속하는 주입은 상수가 아니라 인자여야
하고, 계보를 섞은 런은 어느 쪽과도 비교할 수 없다. `evaluate`를 거쳐 실험 스크립트까지 배선.

### L2b. `page_recall_cap = 10`은 eval 드라이버 값이었다

업스트림 pypi는 `retrieval_queue_capacity=7`(`memoryos.py:38`, `Retriever` 기본값 7)이고,
10은 논문 수치를 낸 하네스가 넘기는 값이다(`eval/main_loco_parse.py:237`
`RetrievalAndAnswer(..., queue_capacity=10)`). 8차 문서는 이를 "`retrieval_queue_capacity`
(10)"이라고 계보 표시 없이 적었다. 게다가 이건 프리셋 키가 아니라 `AgmemConfig` 전역이라
계보별로 갈리지도 않았다.

**조치**: 기본값을 **7**(pypi, 기본 프리셋)로 내리고, `memoryos_eval` config가 10을 명시.
`page_similarity_threshold=0.1`은 두 계보 동일이라 무변경. 두 값 모두 결과 JSON에 스탬프한다
(`page_recall_cap`, `assistant_knowledge_mode`) — 안 그러면 저장된 런이 어느 계보로 읽었는지
말할 수 없다.

## L3. 코드와 정면으로 어긋나는 docstring 2건

8차가 ACE에 대해 지적한 *"docs/10 행이 낡아 있었다"*와 **같은 결함이 8차 자신의 산출물에**
있었다.

- `organizers/memoryos/organizer.py` 모듈 docstring — 네 문장이 거짓이었다:
  "STM flushes as a whole batch … round-5 N2 — **still open**", "STM is not retained for
  injection at QA time", "`meta_info` … the Retriever's separate assistant-knowledge
  channel", 그리고 **"the eval lineage's two-call profile update … we follow pypi in both
  presets"** — 같은 파일 100줄 아래 `MEMORYOS_PRESETS["eval"]["profile_update"] = "two_call"`
  이 이를 반박한다. 전부 8차가 구현한 것들이다.
- `bench/harness.py` — "Loaders: LoCoMo …; **LongMemEval not yet implemented**". LongMemEval
  포팅(273207d)과 그 결함 5건 수정(8c5f3d6) 이후로 낡았다.

**조치**: 둘 다 현재 코드에 맞게 다시 씀. MemoryOS 잔여 갭 목록도 실제로 남은 것
(eval 계보 검색 상수 미세항, 업스트림 Retriever의 3채널 동시 실행)만 남겼다.

## L4. `recent_context()`가 벤치마크 한쪽에만 배선돼 있었다

M3가 신설한 `Organizer.recent_context()` 훅의 소비자는 `bench/locomo.answer` 하나였다.
`bench/longmemeval.answer`는 호출하지 않았다 — MemoryOS를 LongMemEval에서 재면 M3가 복원한
STM 채널이 **조용히 사라지고, degradation 스탬프에도 남지 않는다.**

**조치**: `longmemeval.answer`의 **검색 경로에만** 배선(명시 `history`는 메모리를 전혀 읽지
않는 full-context 베이스라인이므로 제외). 컨텍스트 맨 앞에 붙인다 — LoCoMo와 같은 이유이고,
`max_history_tokens`가 꼬리부터 자르기 때문이기도 하다.
회귀 테스트: `test_the_verbatim_recency_window_reaches_this_benchmark_too`.

## L5. 첫 페이지 continuity 콜 (콜 수만)

업스트림은 첫 페이지에도 continuity를 묻는다 — 빈 이전 페이지를 상대로 묻고, **답을
버린다**(`if is_continuous and temp_last_page_in_batch`에서 두 번째 항이 `None`). 우리는 그
콜을 건너뛴다. 동작은 같고 대화당 1콜이 적다. **미문서 편차였던 것을 문서화**했다(구현
변경 없음) — 업스트림과 콜 수를 비교할 때 continuity는 pages가 아니라 pages-1로 읽어야 한다.

---

# 검증

```
336 passed, 1 skipped     (pytest; 8차 334 + 신규 2)
97 files already formatted (ruff format --check src/ tests/ scripts/)
Found 53 errors (src/ tests/) / 7 errors (scripts/)   ← 8차와 동일, 신규 0
```

신규 회귀 테스트 2건:
- `test_full_assistant_knowledge_is_the_eval_lineage_only` (L2a)
- `test_the_verbatim_recency_window_reaches_this_benchmark_too` (L4)

# 재측정 대상 갱신

`results/locomo-conv0-memoryos.json`은 이제 **세 세대 낡았다**(6차 B1/C1 → 8차 M1~M7 →
9차 L1/L2). 그 파일을 재현하려면 새 `memoryos_mixed` config를 쓴다. 새 `memoryos`/
`memoryos_eval`은 그 파일과 **비교 대상이 아니라 대체 대상**이다 — read 채널 구성과 계보가
둘 다 다르다.

# 변경 파일

```
src/agmem/config.py                          page_recall_cap 10 -> 7 (+계보 근거)
src/agmem/bench/locomo.py                    assistant_knowledge_mode (answer/evaluate)
src/agmem/bench/longmemeval.py               recent_context 주입 (검색 경로 한정)
src/agmem/bench/harness.py                   LongMemEval 로더 존재 반영
src/agmem/organizers/memoryos/organizer.py   모듈 docstring 재작성 · continuity 편차 명시
scripts/exp_locomo_conv0.py                  memoryos/_eval 순수화 · memoryos_mixed 신설
                                             · 계보 knob 배선 + 결과 스탬프
tests/{test_locomo,test_longmemeval}.py      회귀 2건
```
