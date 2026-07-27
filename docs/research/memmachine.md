# MemMachine 조사 — 1차 대조(2026-07-27) + 구현 시 2차 대조

> arXiv:2604.04853 / `github.com/MemMachine/MemMachine` (**Apache-2.0**, 3,341★,
> 최종 푸시 2026-07-26 — 조사 당일 기준 활성).
> 이 문서는 이 프로젝트의 작업 모드(**논문 원문 → official 코드 → 우리 구현**)의 첫 두 단계다.
> 당일 raw로 읽은 파일: `packages/server/src/memmachine_server/episodic_memory/`의
> `episodic_memory.py`·`short_term_memory/`·`long_term_memory/`·`event_memory/`
> (`segmenter/text_segmenter.py`, `deriver/text_deriver.py`)·`declarative_memory/`.

> **⚠ 2026-07-27 구현 착수 시 2차 대조에서 §1.1의 중심 근거가 틀린 것으로 확인됐다.**
> 요약: `TextSegmenter`/`TextDeriver`는 **`event` 백엔드**의 부품이고, 공개 LoCoMo
> 수치를 낸 하네스는 **`declarative` 백엔드 + STM 없음**으로 배선한다. "write 경로에
> LLM 콜 0회"라는 결론 자체는 유지되며 오히려 더 강해지지만(요약 콜조차 없다),
> **근거로 든 파일이 그 수치의 경로가 아니었다.** 상세는 아래 §4, 확정 배선은 §5 프리셋 표.
> 1차 대조는 clone 없이 파일 단위로 읽었고, 2차는 `18f1211`(2026-07-20)을 clone해 읽었다.

## 0. 왜 이 논문이 후보 1순위인가

| 기준 | MemMachine | RecMem | SAGE / Memory Worth / GRAVITY |
|---|---|---|---|
| official 코드 | **Apache-2.0, 활성** | 공개(라이선스 미확인) | **없음** |
| 3자 대조 가능 | ✔ | ✔ | ✘ (2자만) |
| 비교표에서의 자리 | **추출 축의 반대 극단** | 또 하나의 추출형 3-tier | policy 멤버 / read 기법 |
| 부수 소득 | **공식 LoCoMo·LongMemEval 하네스 동봉** | — | — |

G-Memory에서 겪은 무라이선스 문제(11차 §S2)가 없다는 점도 크다.

## 1. 확정된 사실

### 1.1 write 경로에 **메시지당 LLM 콜이 0회**다 (논문 주장, 코드로 확인)

> **정정(§4.1):** 아래 `TextSegmenter`/`TextDeriver` 근거는 `event` 백엔드 전용이고,
> 그나마 둘 다 그 백엔드의 **기본값이 아니다**(기본은 passthrough + whole_text).
> 공개 수치의 경로인 `declarative` 백엔드는 세그먼터 단계 자체가 없다.
> "LLM 콜 0회" 결론은 두 백엔드 모두에서 참이다.

논문의 *"stores entire conversational episodes and reduces lossy LLM-based extraction"*은
수사가 아니다. event-memory write 경로 전체에서 언어모델 호출이 없다:

- **분절 = 순수 기계적**. `TextSegmenter`는 langchain `RecursiveCharacterTextSplitter`
  (`chunk_size=500`, `chunk_overlap=0`)이고, separator 목록에 전각 물음표·표의문자 마침표·
  zero-width space까지 넣은 다국어 우선순위 리스트다. LLM 경계 판정이 **없다**.
- **파생 = 문장 추출 + 포맷팅**. `TextDeriver`는 `extract_sentences`를 쓰고, 임베딩 앵커를
  `[{timestamp}] {producer}: {json.dumps(text)}` 형태로 만든다. fact 추출이 **없다**.
- 우리가 읽은 7개 코어 파일 중 언어모델 참조가 있는 것은 `short_term_memory.py`
  **하나뿐**이고, 그것도 용도가 **비동기 STM 요약**이다(`llm_model`, `summary_prompt_system`,
  `summary_prompt_user`).

**이것이 이 논문의 값어치다.** A-Mem은 turn당 2콜을 써서 추출하고, MemMachine은 사실상
추출하지 않는다. 우리 비교표에서 `passthrough`가 하한, A-Mem이 상한인데 **그 사이에 실물이
없었다.** MemMachine이 그 자리다.

### 1.2 우리 기록 정정 — contextualized retrieval은 **대칭이 아니다**

`memory-component-taxonomy.md` §2.4와 `write-path-critics.md` §4.4는 이 확장을
*"주변 ±1~2 turn"*이라고 적었다. 코드는 비대칭이다:

```python
# event_memory.py:450-451
max_backward_segments = expand_context // 3
max_forward_segments = expand_context - max_backward_segments
```

즉 예산의 **1/3만 뒤로, 2/3를 앞으로** 쓴다. 대화에서 답이 질문 뒤에 온다는 사전지식을
넣은 것이고, "±N"으로 옮기면 그 사전지식이 사라진다. 두 문서를 정정했다.

### 1.3 논문의 3-tier와 배포 코드의 계층이 **이름부터 다르다**

논문: short-term / long-term episodic / **profile**.
코드: `episodic_memory/{short_term_memory, long_term_memory, event_memory,
declarative_memory}` + **최상위 별도 `semantic_memory/`**(cluster_manager,
cluster_splitter, cluster_store, config_store).

"profile"이라는 패키지는 없고, 대신 **클러스터링 서브시스템**이 있다. G-Memory에서 본
논문↔코드 괴리와 같은 계열이므로, 구현 전에 **어느 계보를 재현하는지 먼저 못박아야 한다**
(Nemori·MemoryOS에서 프리셋으로 푼 그 문제).

### 1.4 부수 소득 — 공식 LoCoMo·LongMemEval 하네스가 동봉돼 있다

> **정정(§4.3):** 하네스는 **두 벌**이다. `evaluation/episodic_memory/`는 README가
> 스스로 "legacy"라 적었고 **`18f1211`에서 실행 자체가 불가능**하며(5-튜플을 3개로
> 언팩), 권장 경로는 `evaluation/retrieval_agent/`다. 두 벌의 operating point가 다르다.

```
evaluation/episodic_memory/{locomo,longmemeval}_{ingest,search,evaluate}.py
evaluation/episodic_memory/{llm_judge,generate_scores}.py
```

우리 `bench/locomo.py`·`bench/longmemeval.py`는 6차(A2/A3/B5)와 8c5f3d6에서 **결함 5건**이
나온 층이다. **네 번째 독립 참조**가 생기는 셈이고, 이 리포는 두 벤치를 모두 커버한다.
조직자 구현과 무관하게 이것만으로도 대조 가치가 있다.

### 1.5 논문의 주장 중 우리에게 가장 아픈 것

초록: *"retrieval-stage optimizations (depth tuning, formatting, prompt design)
outperformed ingestion-stage improvements."*

이 프로젝트는 8차까지 거의 전적으로 write 경로를 팠고, read 경로는 7~10차에 와서야
채널 단위로 정리됐다. 이 주장이 맞다면 우리 비용-정확도 파레토의 해석이 달라진다.
**측정 없이는 확인할 수 없는 주장**이므로, 지금은 "구현해서 측정 대기열에 올려두는" 것까지가
가능한 범위다.

보고 수치: LoCoMo 0.9169(gpt-4.1-mini) / LongMemEval-S 93.0% / HotpotQA-hard 93.2% /
WikiMultiHop 92.6%, input 토큰 Mem0 대비 약 −80%.

> 인용 캐비앗은 유효하다(`write-path-critics.md` §4.4): "80% 절감"은 memory-only 경로
> 기준이고 retrieval 단계를 따로 떼지 않았다. agent mode는 8.57M이다.

## 2. 우리 계약에의 매핑 (구현 계획)

> **정정(§4.1):** 아래 표의 첫 두 행은 `event` 백엔드 기준이고, 기본 계보인 `declarative`
> 에는 분절 단계가 없다. 실제 구현된 매핑은 §5 프리셋 표.

| MemMachine | 우리 자리 | 비고 |
|---|---|---|
| `TextSegmenter` (기계적 분절) | organizer `on_message` | LLM 없음 — 우리 organizer 중 처음 |
| `TextDeriver` (문장 단위 파생) | 파생 메모리 타입 ADD op | 새 타입 1종(segment/derivative) |
| STM 요약(비동기) | `on_message` 내 요약 콜 | 유일한 write LLM |
| profile / semantic cluster | 계보 결정 필요 (§1.3) | **먼저 확정할 것** |
| contextualized expansion | `retrieval/steps.py` ReadStep | 비대칭 1/3·2/3 (§1.2) |
| Retrieval Agent (3-way 라우팅) | `policies/` read-side 멤버 | taxonomy §2.4의 기존 판정 유지 |

**표면적 경고**: read 쪽이 무겁다. 확장 스텝은 우리 레지스트리에 그대로 들어가지만,
Retrieval Agent는 policy 계층의 **첫 read-side 멤버**라 계약을 새로 뽑아야 한다.
9~11차가 반복해서 보여준 것이 "새 배선이 기존 config·전역 상수와 만나는 접합부에서 결함이
난다"이므로, 구현 직후 9차식 접합부 점검을 붙인다.

## 3. 착수 순서 (제안 → 1·2·3 완료 2026-07-27)

> 계보는 **배포 코드**로 확정, write organizer(`organizers/memmachine/`)와 read step
> (`MemMachineContextualize`)까지 구현됐다. 4·5는 여전히 별건이다.

1. §1.3 계보 확정 — 논문(3-tier) vs 배포 코드(4+1) 중 무엇을 재현하는지. 프리셋 표부터.
2. write 경로(분절+파생+STM 요약) organizer 구현 + 3자 대조.
3. contextualized expansion ReadStep(비대칭 1/3·2/3).
4. Retrieval Agent는 **분리**해서 별건 — policy read-side 계약이 선행이다.
5. 부수: 동봉된 공식 LoCoMo·LongMemEval 하네스를 우리 `bench/`와 대조(조직자와 독립).

## 4. 2차 대조 — 구현하면서 잡힌 것 (2026-07-27, clone `18f1211`)

1차 대조는 GitHub에서 파일을 골라 읽었고, 2차는 리포를 clone해 **배선을 따라갔다**.
차이는 그 자체로 교훈이다: 파일을 읽으면 "이 코드가 무엇을 하는지"는 알 수 있지만
**"어느 코드가 그 수치를 냈는지"는 호출부를 따라가야만 나온다.**
[[audit-defect-classes]]의 class-(a)가 정확히 이 자리다.

### 4.1 공개 LoCoMo 수치의 백엔드는 `declarative`이고 STM은 아예 없다

`evaluation/utils/agent_utils.py::init_memmachine_params` (L383~):

```python
long_term_memory = LongTermMemory(
    LongTermMemoryParams(
        session_id=...,
        vector_graph_store=...,
        embedder=...,
        reranker=...,
        message_sentence_chunking=message_sentence_chunking,
    )
)  # -> DeclarativeBackendParams
memory = EpisodicMemory(
    EpisodicMemoryParams(..., long_term_memory=long_term_memory, short_term_memory=None)
)
```

`LongTermMemoryParams`는 `backend`로 판별하는 discriminated union이고, 여기엔
`vector_graph_store`가 들어가므로 **`declarative` 변종**이다. `short_term_memory=None`
이므로 **요약 콜조차 없다.** §1.1이 근거로 든 `TextSegmenter`(chunk 500)와
`SentenceTextDeriver`는 `event` 백엔드 소속이고, 그 백엔드에서도
`EventLongTermMemoryConf`의 기본값은 `segmenter=passthrough` / `deriver=whole_text`다
(`common/configuration/episodic_config.py` L238-245). **어느 공개 수치도 그 둘을 통과하지
않는다.**

`declarative`의 write 경로 실체(`declarative_memory.py::_derive_derivatives`):
메시지 1건 → derivative 1건, 내용은 `f"{source}: {content}"`. 임베딩 앵커를 만드는 게
전부고, `message_sentence_chunking=True`면 문장당 1건이 된다.

### 4.2 read 경로가 무겁다는 경고는 맞았고, 비대칭은 **양쪽 백엔드에 다 있다**

`declarative_memory.py::search_scored` L398-400이 `event_memory.py` L450-451과 같은 식이다:

```python
expand_context = min(max(0, expand_context), max_num_episodes - 1)
max_backward_episodes = expand_context // 3
max_forward_episodes = expand_context - max_backward_episodes
```

§1.2의 정정(±N이 아니라 1/3·2/3)은 유효하고, 적용 범위가 event 전용이 아니라 **공통**이다.
추가로 잡힌 것 3가지:

- **넘칠 때의 우선순위도 비대칭이다.** `_weighted_index_proximity`는 앞쪽 이웃을
  `(d-0.5)/2`, 뒤쪽을 `d`로 매긴다 — 같은 거리면 앞쪽이 항상 이긴다.
- **reranker가 점수를 매기는 대상은 아이템이 아니라 "조립된 컨텍스트 문자열"이다**
  (`_score_episode_contexts`). 그래서 declarative는 reranker가 **필수**고(`Optional`이 아님),
  eval config는 Cohere `rerank-v3-5`(Bedrock)를 쓴다. 우리 read step이 파이프라인
  reranker를 **한 번 더** 호출하는 이유가 이것이다.
- **벡터 검색은 derivative에 대해 `min(5 * limit, 200)`으로 과인출**하고, 그 뒤에
  episode 단위로 dedup한다. 즉 "derivative k"와 "episode limit"은 다른 숫자다.

### 4.3 하네스가 두 벌이고, 논문 수치 쪽은 지금 실행되지 않는다

- `evaluation/episodic_memory/` — `README`가 스스로 **legacy**라 적었다. operating point는
  `locomo_search.py`의 `query_memory(query=question, limit=30, expand_context=3)`.
  그런데 `locomo_{ingest,search,delete}.py` 셋 다 `init_memmachine_params`의 **5-튜플을
  3개로 언팩**한다 → `18f1211`에서 `ValueError`. 즉 **논문 수치를 낸 코드가 현재 HEAD에서
  깨져 있다.**
- `evaluation/retrieval_agent/` — 권장 경로. 5개로 정확히 언팩하고, 질의는 query agent를
  거치며 `QueryParam`의 기본값은 `limit=20, expand_context=0`이다.

⇒ **"MemMachine의 LoCoMo operating point"라고 쓸 때는 어느 하네스인지 반드시 박을 것.**
우리 쪽은 두 점을 둘 다 config로 등록했다(`memmachine` = legacy 30/3,
`memmachine_library` = 라이브러리 기본 20/0).

### 4.4 기타 확정 사실

- **STM 용량은 turn 수가 아니라 문자 수**다(`message_capacity`, 기본 64000,
  `sample_configs`는 500). 요약 최대 길이는 `capacity/2/8`을 100 단위 올림.
- 축출은 "이미 요약된 것부터 앞에서 버리고, 그래도 넘치면 **남은 버퍼 전체**를 다시 요약"
  이다. 요약 없이 버려지는 에피소드는 없다.
- `extract_sentences`는 **set을 반환**한다 — 문장 순서가 사라지고 중복 문장은 1건이 된다.
  `sentence_text` deriver를 쓰는 순간 "앵커 수 ≠ 문장 수"가 된다.
- 날짜 포맷이 두 계보에서 다르다: declarative는 `strftime("%A, %B %d, %Y")`(0 패딩),
  event 앵커는 babel CLDR `full`(0 패딩 없음). 같은 시각이 다르게 찍힌다.

## 5. 우리 구현 — 확정 프리셋 표

`src/agmem/organizers/memmachine/organizer.py`의 `MEMMACHINE_PRESETS`.
계보 선택은 **배포 코드**로 확정(2026-07-27): 논문의 3-tier는 대조할 코드가 없어 2자 대조가 되고,
`profile` 패키지는 코드에 존재하지 않는다.

| | `declarative` (기본) | `event` |
|---|---|---|
| upstream | `declarative_memory.py` (+ 공개 LoCoMo 수치) | `event_memory/` |
| 분절 | **단계 없음** | `passthrough`(기본) / `text`(chunk 500) |
| 파생 | 메시지당 1건 `"{src}: {content}"` | segment당 1건 `[{full date}] {src}: {json}` |
| 문장 분할 | `message_sentence_chunking` | `deriver="sentence_text"` |
| STM | **없음**(하네스가 `None`) | 없음(동일) |
| reranker | **필수**, 컨텍스트 문자열 채점 | 선택 |
| write LLM 콜 | **0** | **0** (STM 켜면 축출당 1) |

read 경로는 프리셋이 아니라 **레시피**다(`AgmemConfig.memmachine_*`, [[zep-search-recipes-are-config]]와 같은 취급):
`expand_context`/`context_limit`의 기본값은 **라이브러리 기본(0 / 20)**이고, legacy 하네스의
3 / 30은 실험 config에서 명시적으로 준다. MemoryOS `page_recall_cap` 사고와 같은 이유다 —
기본값이 조용히 eval 계보의 숫자를 물고 있으면 "기본 설정 실행"이 전부 eval 계보 실행이 된다.

미구현/분리 항목: Retrieval Agent(3-way 라우팅)는 policy read-side 계약이 선행이라 별건이고,
`semantic_memory/` 클러스터링도 이번 범위 밖이다.

## 6. 참고

- 논문: [arXiv:2604.04853](https://arxiv.org/abs/2604.04853)
- 코드: [github.com/MemMachine/MemMachine](https://github.com/MemMachine/MemMachine) (Apache-2.0)
- 기존 기록: `write-path-critics.md` §4.4(인용 캐비앗), `memory-component-taxonomy.md` §2.4
  (구조 판정 = mechanism, Retrieval Agent = policy)
