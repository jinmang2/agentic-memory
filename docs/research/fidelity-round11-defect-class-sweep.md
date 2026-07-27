# 11차: 8~10차가 매번 우연히 걸린 두 결함 유형의 전수 조사 (2026-07-27)

> 동기: 8·9·10차가 각각 **직전 라운드의 주장에서** 오류를 찾아냈는데, 세 번 모두 같은 두
> 자리였다. 우연이 세 번이면 유형이다. 새 organizer를 고르는 대신 **유형을 정의하고 전수로
> 훑었다.**
>
> - **(a) 업스트림이 여러 벌인데 한 벌만 보고 내린 판정** — 8차가 `memoryos-pypi`만 보고
>   "검색 keyword 항은 사문"이라 확정한 것(10차 §M8이 eval에서 반증).
> - **(b) 자기가 머리에 얹힌 코드와 어긋나는 문서** — 8차의 ACE 행, 9차의 memoryos·harness
>   docstring, 10차의 RB "훅만 존재".
>
> 방법: 전 organizer 모듈 docstring에서 "누락/편차/미구현/구현 위치" 형태의 **검증 가능한
> 문장을 전부 뽑아 코드와 1:1 대조**했고, 업스트림이 여러 벌인 방법론은 **나머지 벌을 마저
> 받아** 판정이 벌마다 갈리는지 확인했다.

## 결과

| # | 유형 | 대상 | 판정 |
|---|---|---|---|
| S1 | (a) | MemoryOS **세 번째 벌** `memoryos-chromadb` | **미모델링 — keyword 항이 여기서도 살아 있고 공식이 또 다르다** |
| S2 | (b) | G-Memory "clean-room reimplementation" | **거짓** — 같은 docstring이 두 줄 위에서 반박 |
| S3 | (b) | A-Mem "retrieval/pipeline.py에 구현" | **낡음** — `retrieval/steps.py` |
| S4 | (b) | Zep "flush_buffer가 full refresh" | 조건부 서술 누락 (경미) |
| — | (b) | A-Mem 버그픽스 4건, Nemori 편차 6건, ACE 편차 3건, Zep 나머지 | **전부 코드와 일치** |

즉 **class-(b)는 이번으로 전 organizer 소진**했고, class-(a)에서 한 건이 더 나왔다.

---

## S1. MemoryOS는 두 벌이 아니라 **세 벌**이고, 10차 판정도 아직 부분이었다

`BAI-LAB/MemoryOS` 최상위에는 `memoryos-pypi`, `memoryos-chromadb`, `eval/`이 있다.
round-5가 이미 *"업스트림 코어 3벌(pypi/chromadb/eval)이 서로 다름"*이라고 적어 뒀는데
`MEMORYOS_PRESETS`는 두 벌만 모델링한다. 10차가 keyword 항을 pypi/eval 두 벌로 갈랐지만,
세 번째 벌을 보면 **갈래가 하나 더 있다**:

| 벌 | 질의 키워드 | 겹침 공식 |
|---|---|---|
| `memoryos-pypi` | `query_keywords = set()` — 사문, read LLM 콜 **없음** | (해당 없음) |
| `memoryos-chromadb` | `extract_keywords_from_multi_summary(query)` → **write용 multi-summary 프롬프트를 질의에 적용**, 개수 상한 없음 | **Jaccard** (`intersection/union`) |
| `eval/` (논문 수치) | `llm_extract_keywords(query)` — 전용 프롬프트, **최대 3개** | **containment mean** |

10차가 넣은 `_relevance`는 **containment mean을 하드코딩**하고 있었다. 즉 chromadb 계보로
키워드를 켜면 그 벌에 없는 공식이 돌아간다 — 8차가 pypi만 보고 일반화한 것과 **정확히 같은
실수를 한 단계 아래에서** 반복할 뻔했다.

**조치**: `MemoryOSPageRecall(keyword_similarity=...)` + `AgmemConfig.page_recall_keyword_
similarity` 신설. **한 벌은 read와 merge에 같은 공식을 쓰므로** organizer의
`_keyword_overlap`과 반드시 일치해야 하고, 두 구현이 서로 다른 모듈에 있으므로 회귀
테스트로 못박았다(`test_the_read_and_merge_keyword_formulas_are_the_same_function`).

**남긴 것(명시적 추적)**: `chromadb` 벌 전체를 프리셋으로 모델링하는 일. 확인된 차이는
위 두 칸과 `RECENCY_TAU_HOURS = 24`(우리 기본값과 동일), `retrieval_queue_capacity=7`
(pypi와 동일)이다. 나머지 상수 대조는 하지 않았다 — **"미세 항목"이라고 부르지 않는다.**
10차가 그 표현으로 넘어간 자리에서 LLM 콜 하나가 나왔다.

## S2. G-Memory의 "clean-room reimplementation"은 사실이 아니었다

organizer docstring:

> *"No official license upstream: this is a **clean-room reimplementation** from the paper
> + published research notes."*

같은 docstring이 두 줄 위에서 반박한다: *"Score semantics follow **the official code**
(round-5): ADD init 2, EDIT/AGREE +1, REMOVE soft −1 (−3 full), prune at ≤0"*. 그리고
round-5 보고서(`round5/gmemory-verify-report.md`)는 *"공식 코드: github.com/bingreeky/GMemory
clone 성공 (commit 7b581c5)"*, *"우리 구현은 **코드 계보**를 따르므로 아래 대조는 공식 코드를
1차 기준으로 삼는다"*라고 적혀 있다. docs/10의 G-Memory 행도 "우리는 코드 계보"다.

코드를 읽고 상수를 가져온 것이 감사 기록에 남아 있는데 docstring만 독립성을 주장하고 있었다.
게다가 **라이선스가 실제로 없다**(2026-07-27 재확인, GitHub API `license: null`) — 그러면
provenance 문장은 더더욱 정확해야 한다.

**조치**: 문장을 사실대로 다시 썼다 — clean-room이 아니고, 공식 코드를 1차 참조로 읽었으며
(논문과 코드가 §4.3에서 어긋나므로 그럴 수밖에 없었고), 재현한 것은 소스 텍스트가 아니라
동작과 상수라는 것, 그리고 업스트림에 라이선스가 없다는 것을 함께 적었다.

> 이건 코드 결함이 아니라 **기록의 정확성 문제**이고, 판단이 필요한 지점이라 별도로
> 보고했다. 부정확한 provenance 주석은 없는 것보다 나쁘다.

## S3. A-Mem read 경로 포인터가 낡아 있었다

*"Read-path counterpart ... is implemented in retrieval/pipeline.py."* → `LinkExpansion`은
`retrieval/steps.py`에 있다. read 경로 플러그인화 때 옮겨졌고, **같은 리팩터가
`--expand-links` 어블레이션을 한동안 무효 토글로 만들었다**(6차 A1). 9차의 `harness.py`
"LongMemEval not yet implemented"와 같은 계열이라, 옮긴 이력까지 포함해 정정했다.

## S4. Zep flush_buffer (경미)

모듈 docstring이 *"``flush_buffer`` runs a full label-propagation refresh"*라고만 적었는데
실제로는 `community_refresh`가 켜져 있고 그래프가 변경됐을 때만 돈다(메서드 자체
docstring은 정확). 조건을 모듈 docstring에도 넣었다.

## 대조했고 이상 없던 것

- **A-Mem** 버그픽스 4건(ID 기반 이웃 / 진짜 cosine / evolution 실패 명시 드롭 /
  metadata-enriched 질의)과 `actions` 빈 배열 → 두 효과 폴백(`organizer.py:245-247`) — 전부 코드와 일치.
- **Nemori** 편차 6건 중 가장 falsifiable한 것: *"on buffer_max the whole buffer INCLUDING
  the newest message is flushed"* → `stages.py:92-93` `if len(buffer) >= self.buffer_max:
  return [buffer], []` — 일치.
- **ACE** 편차 3건, **Zep** 나머지 서술(3단계 resolution·통합 temporal·op 경유 그래프 쓰기·
  entity 임베딩 name-only) — 일치.

---

# 검증

```
340 passed, 1 skipped     (10차 339 + 신규 1)
97 files already formatted
Found 53 errors (src/tests) / 7 (scripts)   ← 기준선 동일, 신규 0
```

# 변경 파일

```
src/agmem/retrieval/steps.py                   MemoryOSPageRecall.keyword_similarity (3벌 문서화)
src/agmem/retrieval/pipeline.py                page_recall_keyword_similarity 배선
src/agmem/config.py                            page_recall_keyword_similarity
src/agmem/memory.py                            같은 배선
src/agmem/organizers/gmemory/organizer.py      provenance 정정 (clean-room 주장 철회)
src/agmem/organizers/amem/organizer.py         read 경로 포인터 정정
src/agmem/organizers/zep_graph/organizer.py    flush_buffer 조건 명시
tests/test_organizers_phase3.py                read/merge 공식 일치 회귀
```
