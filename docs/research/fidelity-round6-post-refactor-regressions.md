# 6차 충실도 감사: round-5 이후 리팩터 회귀 + 미검증 배선 (2026-07-27)

> 방법: 라운드 3~5가 논문·공식 코드 대조를 이미 끝냈으므로, 이번엔 **그 이후 변경**만
> 표적으로 삼았다 — mechanism/policy 분리(1419e3c), organizer 패키지화(b302343),
> read-path 플러그인화(3b39c7d), A-MAC 게이트(07d3167), LongMemEval 포팅(273207d),
> lifecycle 수정(0995236·c168ff8·3349ac9). 여기에 **round-5가 자기 세션에서 제안하고
> 같은 세션에서 구현해 독립 검증을 못 받은 항목**(P2/구-P3 일괄 반영)을 더했다.
> 업스트림은 당일 raw로 재다운로드해 대조:
> `snap-research/locomo:task_eval/evaluation.py`,
> `WujiangXu/AgenticMemory:utils.py`,
> `xiaowu0162/LongMemEval:src/evaluation/print_qa_metrics.py`,
> `BAI-LAB/MemoryOS:memoryos-pypi/{mid_term,short_term}.py`.
> 대상 HEAD: adf613d.

## 판정 요약

| # | 항목 | 성격 | 지금 저장된 수치에 영향 | 조치 |
|---|---|---|---|---|
| A1 | `--expand-links`가 무효 토글 | 리팩터 회귀 (죽은 속성) | 없음 (산출물이 리팩터보다 앞섬) | **수정** |
| A2 | wujiang 모드 BLEU-1이 `ours`/`wujiang` 혼종 | 지표 배선 누락 | **있음** (`bleu1` 열) | **수정** |
| A3 | `bleu1`을 "공식 채점기 미러"로 표기 | 출처 오기 | 없음 | **문서 정정** |
| B1 | MemoryOS `L_interaction`·`stm_capacity` 단위 불일치 | 충실도 드리프트 | **있음** (승격 빈도·콜 수) | **수정** |
| B2 | RRF 점수를 타입 내 생성·타입 간 비교 | 잠복 편향 | 없음 (예산 미바인딩) | **수정** |
| B3 | Zep이 evolution log를 우회해 그래프에 직접 쓰기 | 불변식 위반 | 없음 (측정 금지 상태) | **수정** |
| B4 | `evaluate(workers>1)` 무부작용 주장이 A-Mem 한정 | 잘못된 보증 | 없음 (현 config workers=1) | **가드 추가** |
| B5 | LongMemEval `overall` 모집단이 업스트림보다 넓음 | 소소한 이탈 | 없음 (공식 데이터에선 동일) | **수정** |

**재검증 후 이상 없음**: A-Mem write 파이프라인(should_evolve 게이팅·ID 기반 이웃 지정·
tags 2단 op), Nemori v4 스테이지(`EpisodeMerger` K_e=5 / `ThreeWayIntegrator` K_m=5·τ=0.70,
전 실패경로가 세그먼트/팩트를 잃지 않는 fallback), A-MAC 릴리스 결함 4건 처리,
locomo `normalize_answer` 정규식 + nltk PorterStemmer 대응, LongMemEval judge 5분기·
모델 핀·abstention 이중계상.

---

# A. 리팩터가 만든 회귀

## A1. `--expand-links`가 아무것도 하지 않는 토글이 되어 있었다

### 배경

A-Mem의 read 경로 핵심은 **1-hop 링크 확장**이다(논문 Fig.2 "linked memories automatically
accessed"). 어블레이션 Table 3에서 링크 생성이 최대 기여 항목(+11.7)이므로, 재현 하네스에
on/off 토글이 있는 것이 맞다.

### 무엇이 틀어졌나

`3b39c7d`(read-path post-step 플러그인화) 이전에는 확장 캡이 `RetrievalPipeline`의 속성이었고,
재현 스크립트가 생성 후에 그것을 덮어썼다:

```python
mem.pipeline.link_expansion_cap = 5 if args.expand_links == "on" else 0
```

플러그인화 이후 캡은 **생성자에서 `AgmemConfig.link_expansion_cap`을 읽어 스텝 레지스트리를
만들 때** 소비된다. 즉 위 대입문은 **아무도 읽지 않는 새 속성을 하나 만들 뿐**이었다.

실측:

```
link_expansion_cap attr exists on pipeline? False
read_steps: {'experiences': ..., 'notes': 'LinkExpansion', ...}
after setting to 0 → LinkExpansion cap= 5      # 여전히 켜짐
```

결과적으로 `--expand-links on`과 `off`가 **동일한 실행**(cap=5)이 되고, 파일명·스탬프에는
`expand-off`가 박혔다.

### 피해 범위

저장된 `results/repro/*_expand-off_*.json`은 `git_sha = e2e7ebe`이고
`git merge-base --is-ancestor e2e7ebe 3b39c7d` → 참. 즉 **산출물은 리팩터 이전이라 유효**하다.
문제는 그 어블레이션이 그 시점 이후 **재실행 불가**였고, 재실행하면 조용히 거짓 라벨이
붙었다는 것.

### 수정

`AgmemConfig(link_expansion_cap=5 if args.expand_links == "on" else 0)`. 검증:

```
expand-links=on  -> notes step = LinkExpansion cap=5
expand-links=off -> notes step = None
```

`docs/14`의 CLI 절에 무효 기간을 명기했다.

---

## A2. `eval_mode="wujiang"`의 BLEU-1이 혼종 지표였다

### 배경

`--eval-mode`는 "A-Mem LoCoMo eval"이라 불리는 서로 다른 채점기 셋 중 어느 것을 쓰는지
고르는 스위치다. `wujiang`은 WujiangXu/A-Mem 재현 리포의 채점 규칙을 그대로 쓰겠다는 뜻이다.

### 무엇이 틀어졌나

`locomo.evaluate`가 gold·F1·cat5 프롬프트만 갈아끼우고 BLEU는 `ours` 함수를 그대로 썼다:

```python
if wujiang:
    f1, b1 = token_f1_wujiang(pred, gold), bleu1(pred, gold)   # ← BLEU가 ours
```

업스트림 `utils.calculate_metrics`는 **F1과 BLEU에 서로 다른 토크나이저**를 쓴다:

| 지표 | 업스트림 토크나이저 | 계산 |
|---|---|---|
| F1 | `simple_tokenize` (lower + `.,!?`→space), **set 화** | set 기반 P/R 조화평균 |
| BLEU-1 | **`nltk.word_tokenize(lower)`** | `sentence_bleu(weights=(1,0,0,0), SmoothingFunction().method1)` |

우리 `bleu1`은 locomo `normalize`(구두점 제거 + 관사 `a|an|the|and` 제거 + **Porter stemming**)
위에서 계산한다. 완전히 다른 전처리다.

### 격차 크기

`bleu1_wujiang` 신설 후 업스트림 호출식과 직접 대조 (5케이스 전부 `1e-12` 이내 일치):

| pred / gold | 기존(`ours` 경로) | 업스트림 |
|---|---|---|
| `the running shoes` / `running shoe` | **1.0000** | **0.3333** |
| `Melanie, a painter.` / `Melanie is a painter` | 0.6065 | 0.6000 |
| `she loves hiking` / `loves hiking` | 0.6667 | 0.6667 |
| `7 May 2023` / `7 May 2023` | 1.0000 | 1.0000 |
| `no match here` / `completely different` | 0.0000 | 0.0000 |

첫 행이 문제의 전형이다 — `the` 제거 + `shoes→shoe` stemming이 **완전 일치**를 만들어낸다.
업스트림 기준으로는 3토큰 중 1토큰만 맞은 0.3333이다.

### 수정

`locomo.bleu1_wujiang` 신설. **실제 nltk를 호출**하고 재구현하지 않는다 — `word_tokenize`는
punkt 문장 분할 + Treebank 토크나이저라, 손으로 근사하면 이 프로젝트가 금지한
"naive in-python fallback"(docs/03 §5) 그 자체가 된다. `agmem._porter`(완전히 명세된
알고리즘의 전사)와는 성격이 다르다.

nltk가 없으면 **지표를 생략**한다 — `None` 반환 → `agg`가 `bleu1` 키 자체를 뺀다.
0.0으로 보고하지도, `ours` 값으로 대체하지도 않는다. `combine_aggs`와 run-summary도
"없음"을 0.0으로 평균내지 않도록 함께 고쳤다. `pyproject`에 `[eval] = ["nltk>=3.9"]` 추가.

### 남은 캐비앗

`results/repro/*_wujiang_*.json`의 `bleu1`(3시드: 31.33 / 31.93 / 32.33)은 혼종값이다.
`docs/14 §9-3b`에 인용 금지로 명기. **F1은 무영향** — `token_f1_wujiang`은 처음부터
분리돼 있었고, 모든 헤드라인 수치는 F1 기반이다.

---

## A3. `bleu1`을 "공식 채점기 미러"로 표기한 것은 오기

`snap-research/locomo`에는 **BLEU가 존재하지 않는다.** 확인 방법 두 가지:

1. GitHub 코드검색 `repo:snap-research/locomo bleu` → `total_count: 0`
2. `task_eval/{evaluation,evaluate_qa,evaluation_stats}.py`를 raw로 받아 grep → 0건

실제 지표는 셋이다:

```
evaluation.py:75   def normalize_answer(s)      # regex.sub(r'\b(a|an|the|and)\b', ' ', ...)
evaluation.py:127  f1_score                     # [ps.stem(w) for w in ...], Counter 겹침
evaluation.py:148  def rougel_score(...)        # 이름과 달리 scores["rouge-1"]["f"]를 반환
```

즉 `normalize_answer` 정규식과 nltk PorterStemmer 대응은 **우리가 맞았고**(`mode="original"`
잔차는 기존에 문서화된 유예 건), `bleu1`만 대응물이 없는 우리 지표였다.

`docs/14` 두 곳을 정정:
- §4 표: "multiset F1 + BLEU-1" → "multiset F1. **BLEU는 없음** — EM / stemmed token-F1 /
  `rougel_score`(rouge-1 F) 3종"
- §5.1 표: "`token_f1`/`bleu1` | 공식 채점기 미러" → 두 행으로 분리, `bleu1`은 "**우리 지표**"

---

# B. 배선·불변식 항목

> **2026-07-27 후속 (같은 날 "교정" 지시)**: 아래 B1~B3은 최초 보고 시 "동작을 바꾸므로
> 기록만" 으로 유예했으나, 지시에 따라 **전부 수정**했다. 각 절 말미의 「수정」 항목이
> 실제 반영 내용이고, 유예 사유였던 서술은 「최초 판단」으로 남겨 둔다 — 무엇을 왜
> 뒤집었는지가 다음 라운드의 판단 근거가 되기 때문이다.
> **결과적으로 MemoryOS의 동작이 바뀌었다**: `results/locomo-conv0-memoryos.json`은
> 이제 구 배선(메시지 단위) 산출물이며 재측정 대상이다.

## B1. MemoryOS heat의 `L_interaction`이 다른 단위를 센다

### 메커니즘

MemoryOS는 STM → MTM → LPM 3계층이다. MTM 세그먼트마다 **heat**을 유지하고, 임계값 τ를
넘으면 LPM으로 승격한다(= 프로필/지식 추출 LLM 콜 발생 + heat 리셋). 용량 초과 시엔
최저-heat 세그먼트를 축출한다.

```
H_segment = α·N_visit + β·L_interaction + γ·R_recency
```

계수는 pypi 코어와 일치한다 (`mid_term.py:21-24`: `HEAT_ALPHA=1.0, HEAT_BETA=1.0,
HEAT_GAMMA=1, RECENCY_TAU_HOURS=24`), τ=5.0. **이 부분은 문제 없다.**

### 정확히 뭐가 다른가

업스트림 (`memoryos-pypi/mid_term.py`):

```python
:162   "L_interaction": len(processed_details),          # detail = page
:271   target_session["L_interaction"] += len(pages_to_insert)
```

그리고 page의 정의 (`memoryos-pypi/short_term.py:18`):

```python
def add_qa_pair(self, qa_pair):
    ...  qa_pair.get('user_input')  /  agent_response
```

**1 page = 1 대화 교환 = 발화 2개.** 공개 API가 `add_memory(user_input, agent_response)`이므로
구조적으로 그렇다.

우리 (`organizers/memoryos/organizer.py`):

```python
self._heat[segment_id] = {"n_visit": 0, "length": len(members), ...}
#                                               ^^^^^^^ members: list[Episode] = 메시지
h["length"] += len(members)     # 병합 경로도 동일
```

`Episode`는 파사드의 쓰기 단위, 즉 **메시지 1개**다. `bench/locomo.ingest`가 턴마다
`add_message`를 부른다.

### 결과

같은 대화 길이에서 `L_interaction`이 **약 2배** → heat도 약 2배. `R_recency ≈ 1`(막 만든
세그먼트)이므로 τ=5 승격 조건은:

| | 승격에 필요한 분량 |
|---|---|
| 업스트림 | 4 page ≈ **8 발화** |
| 우리 | **4 메시지** |

**승격이 절반 분량에서 발동한다.** 저장된 산출물에서 근거가 보인다
(`results/locomo-conv0-memoryos.json`, 419턴):

```
n_turns = 419,  stm_capacity = 10  →  배치 41회  →  TOPIC 콜 41회
llm_budget["distill"]["calls"] = 91
∴ LPM 승격 콜 ≈ 50회  (배치 수 41보다 많다)
```

업스트림 단위였다면 10-메시지 배치를 토픽으로 쪼갠 그룹이 4 page(8발화)에 도달하는 일
자체가 드물다. 파급은 두 갈래다:

1. **write 비용** — 승격마다 `PROFILE_PROMPT` LLM 콜 1회
2. **read 컨텍스트** — `bench/locomo.answer`가 `kind="profile"` 팩트를 **모든 QA 프롬프트에
   무조건 주입**한다(round-5 memoryos §3, 업스트림 eval 정합). 프로필이 많아지면 주입량이
   그만큼 커진다

### 최초 판단 (뒤집힘)

"`len(members)/2`는 땜빵이고, 진짜 원인은 파사드의 쓰기 단위가 메시지라는 구조다. LoCoMo는
speaker A / speaker B 대화라 user·assistant 쌍이 명확하지도 않으니, organizer가 메시지를
임의로 짝지으면 파이프라인 어디에도 없는 대화 모델을 혼자 발명하는 셈이다."

이 판단은 **틀렸다.** 짝짓기 규칙을 발명할 필요가 없었다 — 업스트림이 이미 LoCoMo용
드라이버에 그 규칙을 가지고 있다 (`eval/main_loco_parse.py:169-183`):

```python
if speaker == speaker_a:
    processed.append({"user_input": text, "agent_response": "", "timestamp": timestamp})
else:
    if processed:
        processed[-1]["agent_response"] = text
```

즉 **처음 등장한 화자가 페이지를 열고, 다른 화자의 발화는 진행 중인 페이지에 붙는다.**
발명이 아니라 이식이다. 그리고 우리 `bench/locomo.ingest`는 이미
`meta={"speaker": speaker, ...}`를 실어 보내므로 필요한 정보도 갖고 있었다.

### 추가 발견: `stm_capacity`도 같은 단위 오류

수정 중 확인: `ShortTermMemory(max_capacity=10)`은 **`add_qa_pair` 엔트리의 deque**이고
`is_full()`은 `len(memory) >= max_capacity`다. 즉 `short_term_capacity=10`도 **10 page
(≈20 발화)** 다. 우리는 10 **메시지**에서 flush하고 있었으므로, TOPIC 프롬프트에 들어가는
배치가 업스트림의 절반이었다. `L_interaction`만 고치면 이 상수는 여전히 틀린 단위로
남는다 — 하나의 결함이지 둘이 아니다.

### 수정

`_page_key` / `_pages` 신설 (업스트림 드라이버 전사), 그리고 페이지를 세는 지점을 전부 통일:

| 지점 | 이전 | 이후 |
|---|---|---|
| STM flush 트리거 | `len(self._stm) >= stm_capacity` (메시지) | `len(self._pages(self._stm)) >= stm_capacity` (페이지) |
| TOPIC 프롬프트 인덱스 | 메시지 `[i] content` | 페이지 `[i] A: ... \| B: ...` |
| 그룹 스키마 필드 | `message_indexes` | `page_indexes` |
| 신규 세그먼트 heat | `"length": len(members)` | `"length": n_pages` |
| 병합 세그먼트 heat | `h["length"] += len(members)` | `h["length"] += n_pages` |

`members`(=provenance용 episode 목록)는 선택된 페이지들을 flatten해서 그대로 유지하므로
`source_episode_ids`는 변하지 않는다. 인덱스는 `sorted(set(...))`으로 중복 제거 —
같은 인덱스가 두 번 오면 내용 증가 없이 `L_interaction`만 부풀기 때문이다.

의도적 편차 2건을 docstring에 명시했다:
1. 업스트림은 같은 페이지에 다른 화자의 두 번째 발화가 오면 `agent_response`를
   **덮어써서 첫 발화를 잃는다**. 우리는 append한다(원문 손실 금지 원칙).
2. 업스트림은 형성된 pair를 한 번에 받지만 우리는 두 반쪽을 별도 `add_message`로 받으므로,
   flush가 한 교환의 중간에 떨어져 세그먼트가 갈릴 수 있다.

**회귀 테스트**: `test_memoryos_counts_pages_not_messages` — A/B/A/B 스트림에
`stm_capacity=2`를 걸고 (a) 메시지 2개(i=1)에서 flush되지 **않고** (b) 두 번째 A(i=2)에서
flush되며 (c) heat `length == 2`(3개 메시지를 담은 2페이지)임을 단언한다. (a)가 구
배선과의 판별 조건이다.

**결과**: `results/locomo-conv0-memoryos.json`은 구 배선 산출물이 되었다. 재측정 대상.

---

## B2. RRF 점수를 타입 안에서 만들고 타입 밖에서 비교한다

### 메커니즘

`RetrievalPipeline.search`는 memory type마다 **따로** 검색한다. 타입별로 채널을 모아
RRF로 융합한다:

```python
for memory_type in memory_types:
    rankings = [vector_store.search(...)]                    # dense 채널
    if memory_type == "episodic":
        rankings.append(doc_store.search_lexical(...))        # BM25 채널
    elif memory_type in self.lexical_types:
        rankings.append(doc_store.search_lexical_items(...))
    fused = rrf_fuse(rankings)
```

RRF는 점수가 아니라 **순위**만 쓴다: `score = Σ 1/(60 + rank + 1)`. 채널 하나당 1위가
`1/61 = 0.0164`씩 더해진다. 따라서 **점수의 상한이 채널 개수에 비례**한다.

| 타입 | 채널 | 1위 점수 상한 |
|---|---|---|
| `episodic` | dense + BM25 | **0.0328** |
| `notes` / `episodes` / `semantic` / `pages` | dense만 | **0.0164** |

`AgmemConfig.lexical_types` 기본값이 `("episodic",)`이라 **raw 에피소드만 2채널**이다.

### 문제

`MemoryBundle.render`가 이 점수를 **타입 구분 없이 한 줄로 세워** 예산을 자른다:

```python
for scored in sorted(self.items, key=lambda s: s.score, reverse=True):
    if used + len(text) > budget_chars and selected:
        break        # 예산 초과하는 첫 항목에서 전부 중단
```

타입 안에서만 의미 있는 숫자를 타입 간 비교에 쓰는 것이다. 재현 (episodic 12건 + notes 6건,
k=5씩, 결정적 해시 임베더):

```
rank  type      RRF score   channels
0     episodic  0.03227     2
1     episodic  0.03178     2
2     episodic  0.03101     2
3     episodic  0.03062     2
4     episodic  0.03021     2
5     notes     0.01639     1     ← notes 1위조차 episodic 꼴찌보다 아래
6     notes     0.01613     1
...
모든 episodic이 모든 notes보다 위인가?: True
```

**두 채널에 모두 걸린 raw 에피소드는 파생 메모리보다 무조건 위**다. 관련성이 아니라 채널
개수가 만든 순서다. dense 20위 + BM25 20위인 에피소드(`2/81 = 0.0247`)도 notes 1위
(`0.0164`)를 이긴다.

### 영향 지점은 정확히 하나

- **섹션 순서**: `type_order = dict.fromkeys(...)` = 번들 삽입 순 = `memory_types` 순. **무관**
- **섹션 내 정렬**: 같은 타입끼리 정렬 → 스케일 동일. **무관**
- **예산 컷**: 전역 정렬. ← **여기만 영향**

즉 **예산이 걸리지 않으면 효과가 0**이다. 순서만 다르고 전부 선택되기 때문이다.

### 실제 피해 범위

측정된 런에서 예산은 한 번도 걸리지 않았다:

```
캡처된 질문 수: 12,314 | 최대 번들 chars: 11,663 | 예산(24,000) 초과: 0건
```

(`budget_tokens=6000` × `CHARS_PER_TOKEN=4` = 24,000자)

저장된 mixed 산출물 4종은 retrieval capture 이전이라 번들 크기를 확인할 수 없다:

```
amem         memory_types=['episodic', 'notes']                f1=23.25
memoryos     memory_types=['episodic', 'pages', 'semantic']    f1=20.90
nemori       memory_types=['episodic', 'episodes', 'semantic'] f1=18.97
passthrough  memory_types=['episodic']                         f1=22.85
```

이들은 구 4-way 런으로, docs/10이 이미 "혼합(raw RAG 포함) 조건 → 논문 재현으로 인용 금지"로
표시해 둔 것이다. 항목 수(k=10+10)와 LoCoMo 턴 길이로 보면 예산이 걸리지 않았을 가능성이
높지만 **미측정**이다.

**결론: 지금 틀어진 수치는 없는 잠복 결함.** `budget_tokens`를 낮추거나, 파생 항목이 긴
방법론(Nemori 서사 + r=2 원문 첨부)으로 mixed 어블레이션을 돌리는 순간 발현한다.

### 최초 판단 (뒤집힘)

"타입별 정규화를 넣으면 mixed config의 랭킹이 바뀐다. **문서화된 편향**을 **문서화되지
않은 불연속**으로 바꾸는 거래라 판단이 필요하다."

이 우려는 **과대평가였다.** 정규화는 `rrf_fuse` 한 호출 안의 모든 점수를 같은 상수로 나누는
**단조 재스케일**이라 타입 내 순서를 바꿀 수 없다. 따라서:

- **순수 config는 비트 동일** — `notes` 단독, 또는 `episodes`/`semantic`처럼 채널 수가 같은
  타입들만 모이므로 공통 상수로 나눠도 전역 정렬이 그대로다.
- 바뀌는 것은 **혼합 검색의 예산 선택 순서뿐**이고, 예산은 측정된 12,314질문 전부에서
  걸리지 않았다. 즉 재랭킹이 실제로 뒤집을 수 있는 저장 수치가 없다.

"불연속" 리스크가 걸려 있다고 본 대상이 실은 존재하지 않았다 — deferral 근거가 없는 비교를
보호하고 있던 셈이고, 이는 이전 라운드(`verify-deferral-rationales`)에서 이미 두 번 겪은
패턴이다.

### 수정

`rrf_fuse`가 채널 수로 나눈다:

```python
channels = len(rankings)
return sorted(((i, s / channels) for i, s in fused.items()), key=..., reverse=True)
```

호출자가 하나뿐이지만 **`rrf_fuse` 안**에 둔 이유: 정규화되지 않은 RRF는 "모든 후보가 같은
랭킹 집합을 통과한다"는 전제 위에서만 옳고, 그 전제를 깨는 것이 타입별 융합이다. 미래의
두 번째 호출자가 이 문제를 다시 발견하는 대신 비교 가능한 점수를 상속받도록 했다.

검증 (수정 전/후 같은 입력):

```
[전] 0 episodic 0.03227 | ... | 4 episodic 0.03021 | 5 notes 0.01639
     모든 episodic이 모든 notes보다 위인가?: True
[후] 0 notes 0.01639 | 1 episodic 0.01613 | 2 notes 0.01613 | ...
     모든 episodic이 모든 notes보다 위인가?: False
```

---

## B3. Zep만 evolution log를 우회해 스토어에 직접 쓴다

### 위반된 불변식

이 프로젝트의 전제다 (`organizers/base.py`, docs/04 §2):

> organizer는 스토어를 직접 건드리지 않는다. 변경은 `MemoryOp`로 **반환**하고, 파사드가
> **append-only 로그에 먼저 기록한 뒤** 적용한다.

여기서 나오는 성질 셋:

1. 로그만으로 상태를 **재생**할 수 있다
2. 로그가 **먼저**라, 적용이 실패해도 이력은 남는다
3. 상태 변경 지점이 `_apply_one` **하나**다

### 정확히 뭐가 다른가

`organizers/zep_graph/organizer.py`가 훅 안에서 그래프에 **즉시** 쓴다:

```
:259   graph.upsert_node(dup, ...)              # 엔티티 병합 (LLM 판정 후)
:275   graph.upsert_node(node_id, ...)          # 신규 엔티티
:400   graph.invalidate_edge(contradicted_id, valid_at)   # 모순 팩트 무효화
:411   graph.upsert_edge(edge_id, ...)          # 신규 팩트
:432   graph.invalidate_edge(edge_id, invalid_at)
```

그리고 `memory.py`의 `_apply_one`은 `graph_store`를 **한 번도 참조하지 않는다**(grep 확인 —
파사드에서 `graph_store`는 생성·ctx 주입·pipeline 주입·close에만 등장). 파사드가 아는 것은
`ADD(entities)` / `UPDATE(entities)` / `ADD(facts)` / `INVALIDATE(facts)` op뿐이고, 이들은
doc_store와 vector_store에만 적용된다.

### 결과 셋

**① 로그로 그래프를 복원할 수 없다.**
evolution log를 리플레이하면 doc/vector는 재건되지만 **노드·엣지·엣지 유효기간은 재건되지
않는다.** `retrieval/steps.py`의 `GraphRecall`은 `ctx.graph_store.edges_for_nodes(...)`를
읽으므로, 복원된 메모리에서 **Zep의 read 경로가 조용히 평범한 벡터 RAG로 퇴화한다.**
`GraphRecall.run`은 `graph_store is None`일 때만 no-op이고, 빈 그래프는 그냥 빈 결과라
오류도 경고도 없다.

**② 실패 시 doc/graph 불일치.**
`_resolve_entity`는 그래프에 노드를 쓰고 `ops` 리스트에 ADD를 담아 반환한다. 이후 fact
추출 단계에서 예외가 나면 ops는 **적용되지 않지만**(비동기 워커는 `logger.exception` 후
계속) **노드는 이미 그래프에 있다.** 그래프가 doc보다 앞서면 `GraphRecall`이 doc_store에
없는 fact id를 끌어오고 `get_items`가 조용히 적게 반환한다.

**③ 감사 추적이 불완전하다.**
"이 메모리가 무엇을 했는가"의 정본이 evolution log라는 것이 프로젝트 전제인데, Zep에 대해
거짓이다. `AgenticMemory.log`로 tail해도 그래프 변경은 보이지 않는다.

### 실제 피해 범위

Zep은 docs/09/10에서 **측정 금지(○ 골격)** 상태이므로 지금 틀어진 수치는 없다. "버그"라기보다
**해금 전에 갚아야 할 설계 부채**다.

### 수정

op 어휘는 건드리지 않았다 — **기존 `entities`/`facts` op에 그래프가 필요한 필드를 싣고,
`AgenticMemory._apply_graph`가 적용**한다. 새 op 타입도, 새 스토어 계약도 없다.

| | 이전 | 이후 |
|---|---|---|
| 노드 쓰기 | organizer의 `graph.upsert_node` | `_apply_graph`가 `entities` ADD/UPDATE에서 |
| 엣지 쓰기 | organizer의 `graph.upsert_edge` | `_apply_graph`가 `facts` ADD/UPDATE에서 |
| 엣지 무효화 | organizer의 `graph.invalidate_edge` | `_apply_one`의 `INVALIDATE` 분기에서 |
| payload | `subject`/`object` (이름) | + `subject_id`/`object_id` (노드 id), entity UPDATE에 `entity_type` |

payload에 **id**를 실은 이유: 적용 시점에는 이름을 노드로 되돌릴 수 없다(동명이인이 가능하고,
resolution은 이미 organizer에서 끝났다). `entity_type`을 병합 UPDATE에 실은 이유: 그 op가
구동하는 upsert가 노드 타입을 기본값으로 되돌리지 않게 하기 위해서다.

두 스토어 메서드가 모두 id 기준 full-row upsert이므로 **리플레이는 수렴한다**(중복 생성 없음).
`upsert_edge`는 full-row replace라 `invalid_at`을 지우므로, 이미 무효화된 fact는 upsert 직후
다시 stamp한다.

organizer 쪽 변화는 하나뿐이다: 같은 `on_message` 안에서 결정된 엣지가 아직
`edges_between`에 보이지 않으므로, 호출-로컬 `pending` 맵(무순서 쌍 키 — `edges_between`의
양방향 매칭과 일치)이 그 창을 메운다. 모순 판정으로 무효화된 엣지는 로컬 뷰에도
`invalid_at`을 찍어, 같은 메시지의 뒤 fact가 이미 나가는 중인 엣지를 다시 반박하지 않게 한다.

**회귀 테스트**: `test_zep_graph_is_rebuildable_from_the_evolution_log` — zep 메모리에서
엔티티/팩트 op만 뽑아 **passthrough 메모리**(Zep이 뭔지 모르는)에 `_apply_one`으로 재생하고,
노드/엣지 수와 엣지 내용이 원본과 일치함을 단언한다. 수정 전에는 재생 결과가 빈 그래프였다.

`docs/10`의 Zep 완성 계획 6번은 해결로 갱신.

---

## B4. `evaluate(workers>1)`의 "부작용 없음"은 A-Mem 한정이었다

기존 docstring:

> every store read path is lock-guarded and **A-Mem's `on_retrieval` is a no-op**, so QA is
> side-effect-free … the aggregates are therefore IDENTICAL to the sequential path

앞 절은 참이지만, 뒷 절("따라서 항상 동일")은 **활성 방법론의 성질**이지 이 함수의 성질이
아니다. round-5가 조회 통지 훅을 신설한 뒤로:

- `MemoryOSOrganizer.on_retrieval` → `self._heat[id]["n_visit"] += 1`, `last_access` 갱신
- `GMemoryOrganizer.on_retrieval` → `self._served.update(...)`

둘 다 평범한 dict/set 변경이고, `AgenticMemory.search`가 이를 **동기적으로** 부르므로
`ThreadPoolExecutor`의 모든 워커에서 동시에 일어난다. 손실 갱신이 생기고, 그 heat에 의존하는
후속 상태가 worker 수에 따라 달라진다.

**수정**: `workers > 1`이고 `on_retrieval`을 오버라이드한 organizer가 활성이면 경고를 낸다
(`organizers.base.overrides`로 판별 — 무조건 no-op 상속과 실제 구현을 구분하는 기존 헬퍼).
현재 config는 A-Mem 재현만 `workers=8`이고 나머지는 1이라 **잠복 상태**다.

---

## B5. LongMemEval `overall`의 모집단이 업스트림보다 넓었다

업스트림 `print_qa_metrics.py`를 verbatim으로 받아 확인:

```python
all_acc, task_acc, abstention_acc = [], [], []
type2acc = {t: [] for t in ['single-session-user', ..., 'knowledge-update']}
for entry in in_data:
    assert entry['autoeval_label']['model'] == 'gpt-4o-2024-08-06'
    type2acc[ref_entry['question_type']].append(1 if entry['autoeval_label']['label'] else 0)
    if '_abs' in entry['question_id']:
        abstention_acc.append(1 if entry['autoeval_label']['label'] else 0)

for k, v in type2acc.items():
    all_acc += v                       # ← all_acc는 6개 버킷의 연결
    task_acc.append(np.mean(v))
```

`all_acc`는 **알려진 6타입 버킷을 이어붙인 것**이라, 그 밖의 타입은 도달할 수 없다(사실
업스트림은 그런 행에서 `KeyError`가 난다). 우리 `aggregate`는 모든 행을 `everything`에
넣고 있었다.

**수정**: `everything = [hit for t in ordered for hit in by_type[t]]`. 미지 타입은 `by_type`의
`extra`로 남아 조용히 사라지지 않되, 두 헤드라인 수치에는 들어가지 않는다. 공식 데이터에선
두 정의가 일치한다.

검증:

```
rows: multi-session(True), multi-session_abs(False), made-up-type(False)
→ by_type: {'multi-session': 50.0 (n=2), 'made-up-type': 0.0 (n=1)}
   task_averaged: 50.0,  overall: 50.0,  abstention: 0.0 (n=1),  n: 2
```

**반대로, abstention 이중 계상은 우리 포팅이 옳다** — 위 원문에서 보이듯 모든 행이 자기
타입 버킷에 들어가고, `_abs` 행은 **추가로** abstention 버킷에도 들어간다. 즉 두 헤드라인
수치는 abstention 질문을 포함한다.

---

# 검증

```
313 passed, 1 skipped                        (pytest)
94 files already formatted                   (ruff format --check src/ tests/ scripts/)
Found 49 errors  (src/ tests/)               ← HEAD 기준선과 동일, 신규 0
Found 7 errors   (scripts/)                  ← HEAD 기준선과 동일, 신규 0
```

`bleu1_wujiang`은 업스트림 호출식(`sentence_bleu([word_tokenize(gold.lower())],
word_tokenize(pred.lower()), weights=(1,0,0,0), smoothing_function=SmoothingFunction().method1)`)과
5케이스 전부 `1e-12` 이내 일치를 확인했다.

# 변경 파일

```
docs/10-fidelity-audit.md                   6차 감사 블록 + MemoryOS 행 + Zep 해금조건 6번
docs/14-amem-reproduction.md                §4 표 · §5.1 표 · §9-3b 캐비앗 · CLI 주석
pyproject.toml                              [eval] = ["nltk>=3.9"]
scripts/exp_amem_repro.py                   A1 배선 · combine_aggs · run_summary
src/agmem/bench/locomo.py                   bleu1_wujiang · agg · workers 가드 · docstring
src/agmem/bench/longmemeval.py              aggregate overall 모집단
src/agmem/organizers/memoryos/organizer.py  B1 기록
src/agmem/organizers/zep_graph/organizer.py B3 기록
src/agmem/retrieval/pipeline.py             B2 기록
```

# 다음

1. **MemoryOS 재측정** — B1으로 write 경로가 바뀌었다. `results/locomo-conv0-memoryos.json`은
   구 배선(메시지 단위) 산출물이므로 승격 횟수·프로필 수·F1 전부 재산출 대상
2. **Zep 재등급 검토** — B3으로 "골격(○)" 근거 중 감사 추적성 항목이 해소됐다. 남은 해금
   조건은 community(label propagation) 하나
3. round-5 잔여 미구현(Zep community, G-Memory FINCH·MAS 채널, RB MaTTS, MemoryOS STM
   롤링·LPM 프로필 문서 교체)은 그대로 — 각 벤치 착수 시점

---

# C. 후속 구현 (감사 범위 밖 — "계속 구현" 지시)

## C1. MemoryOS LPM 재구현 (docs/10 M1 해소)

### 업스트림 구조

`_trigger_profile_and_knowledge_update_if_needed`(`memoryos-pypi/memoryos.py:126-220`)는
**가장 뜨거운 세그먼트 하나**(`self.mid_term_memory.heap[0]`)를 보고, 그 세션의
**미분석 페이지만** 뽑아 두 LLM 태스크를 병렬 실행한다. LPM은 append 로그가 아니라 **3종
스토어**다:

| 스토어 | 업스트림 | 갱신 방식 |
|---|---|---|
| 사용자 프로필 | 단일 문서 | `update_user_profile(user_id, new_data, merge=False)` — **전체 교체**. 프롬프트가 기존 프로필을 입력으로 받아 갱신본 전문을 출력 |
| user private knowledge | `deque(maxlen=100)` | 줄 단위 append, 초과 시 가장 오래된 것 자동 폐기 |
| assistant knowledge | **별도** `deque(maxlen=100)` | 동일 |

갱신 후: 세션의 **모든** 페이지에 `analyzed=True`(업스트림 주석이 이 선택의 모호함을 스스로
표시), `N_visit=0`, `L_interaction=0`, heat 재계산, `last_visit_time` 갱신. 프로필은
`len(strip()) >= 30`이고 `"none"`이 아닐 때만 교체하되, 미달이어도 **지식은 저장하고 heat는
리셋**한다. 두 태스크 중 하나라도 예외면 **리셋 없이 return**(재시도).

### 우리가 하던 것

`_promote_to_lpm`이 세그먼트 summary를 프롬프트에 넣고 `profile_facts` 배열을 받아
`kind="profile"` semantic 항목으로 **append**. 문서 교체도, 지식 분리도, 용량 제한도,
analyzed 마킹도 없었고, 승격 대상은 "방금 쓴 세그먼트"였다.

### 수정

- **프로필 문서**: 고정 id `PROFILE_ITEM_ID = "memoryos:user_profile"` 아래 ADD로 전체 교체
  (`base.cursor_op`와 같은 근거 — 항목의 상태 전체가 그 문서이므로 full replace가 정확하고,
  최초 기록 시에도 동작). 프롬프트는 `PERSONALITY_ANALYSIS_*` 축약 — 기존 프로필을 입력으로
  받아 갱신본 전문을 출력하고, 심리 모델 5차원은 원문대로, 나머지 "90 dimensions"는 업스트림
  자체가 고정 스키마가 아닌 자유 열거라 모델에 위임.
- **지식 FIFO 2종**: `kind="user_knowledge"` / `"assistant_knowledge"`, 각각
  `knowledge_capacity`(기본 100). 초과분은 **DELETE op로 축출**(업스트림은 조용히 버리지만
  우리는 로그에 남긴다). `"none"`/`"- none"`/`"- none."` 줄 필터 전사.
- **analyzed 마킹**: `self._analyzed`(episode id). 미분석 유닛만 프롬프트에 들어가고, 갱신 후
  세그먼트의 **전체** 유닛을 마킹(업스트림 동일). 없으면 뜨거운 세그먼트가 flush마다 같은
  페이지를 다시 분석해 비용만 재지출한다.
- **승격 대상**: 그룹 루프 안이 아니라 루프 뒤에서 `max(self._heat, key=self._segment_heat)`
  하나. 주기는 업스트림(페이지마다) vs 우리(STM flush마다)로 다르지만, heat은 MTM에 페이지가
  닿을 때만 움직이므로 flush 사이의 추가 검사는 새 후보를 찾을 수 없다 — 유일한 실차이는 한
  flush에서 여러 세그먼트가 τ를 넘는 경우다(docstring에 명시).
- **실패 처리**: 두 콜 중 하나라도 drop이면 `[]` 반환 + heat 보존(업스트림 try/except 대응).
- **bench 주입**: `bench/locomo.answer`의 프로필 섹션이 `- {fact}` 불릿 + `[-100:]` 캡에서
  **문서 원문 주입**으로 변경. 불릿과 캡은 append 시절의 산물이고, 단일 문서에 캡을 씌우면
  업스트림에 대응물 없는 절단이 된다.

**회귀 테스트 2건**: `test_memoryos_eviction_creates_segment_and_promotes`(프로필/지식 3종이
모두 나오고 프로필은 고정 id), `test_memoryos_profile_is_replaced_and_knowledge_fifo_evicts`
(두 번 승격 후 스토어에 남는 문서는 최신본 1개, `"None"` 줄 필터, capacity=2에서 3번째
항목이 첫 항목을 DELETE로 밀어냄).

### 남은 MemoryOS 갭

STM recency 주입(round-5 N2 — 배치 flush로 QA 시점 STM이 비어 있음), dialogue chain
`meta_info`, Retriever의 assistant-knowledge 채널, eval-core 상수 계보
(heat 0.8/0.8/1e-4·Dice keyword·STM cap=1), agent persona.

## C2. MemoryOS 계보 프리셋 (`fidelity="pypi"|"eval"`)

### 왜 필요한가

"MemoryOS 공식 코드"는 **두 벌**이고 상수가 다르다. `memoryos-pypi/`는 유지보수되는
라이브러리, `eval/`은 **논문의 LoCoMo 수치를 만든 하네스**다. 우리는 pypi 계보만 갖고
있었으므로, 재현 대상인 발표 수치가 사정권 밖이었다. Nemori의 `fidelity`와 같은 구조로
분리했다 — 프리셋 안에서 출처를 섞지 않는다.

| | `pypi` (기본) | `eval` (논문 수치) |
|---|---|---|
| heat 계수 α/β/γ | 1.0 / 1.0 / 1.0 | **0.8 / 0.8 / 1e-4** |
| R_recency | live 재계산 (`compute_time_decay`) | **저장값**, 검색 히트에만 갱신 |
| 키워드 항 | Jaccard `\|∩\|/\|∪\|` | **containment 평균** `0.5(\|∩\|/\|A\| + \|∩\|/\|B\|)` |
| STM 용량 | 10 page | **1 page** |
| 축출 | 최저-heat (논문 서술) | **access-count LFU** (`evict_lfu`) |
| 공유 | θ=0.6, τ=5.0, MTM 2000 | 동일 |

키워드 항은 집합 크기가 다르면 containment 평균이 Jaccard보다 **항상 크다** — 같은
θ=0.6에서 eval 계보가 더 쉽게 병합된다. (예: `{trip,paris}` vs `{trip,paris,museum,budget}`
→ Jaccard 0.50, containment 평균 0.75.)

### eval 코어 결함 5건 (A-Mem 선례대로 명시)

| # | 내용 | 재현? |
|---|---|---|
| E1 | `insert_pages_into_session`의 마지막 `L_interaction += len(pages)`가 **merge 분기 밖**에 있어, 점수 미달로 새 세션을 만든 경우에도 **탈락한 후보**의 heat이 오른다. pypi에는 없음 | **미재현** — 내용을 하나도 안 받은 세그먼트에 heat을 주는 것이라 승격 순서를 재현이 아니라 손상시킨다 |
| E2 | `R_recency`가 죽어 있다. 저장값(초기 1.0, 검색 시에만 갱신)에 γ=1e-4 → 한 interaction(β=0.8)의 **1/8000**. 어떤 비교도 못 바꾼다 | **재현** — 발표 operating point 자체다. `recency="stored"`로 죽어 있음을 설정에 드러냄 |
| E3 | 페이지 임베딩 텍스트가 경로마다 다르다. `add_session`은 `f"User: {u} Assiant: {a}"`(오타 upstream), merge 분기는 `f"用户: {u}"` — 언어 접두사가 다르고 **assistant 발화가 통째로 빠진다** | **미재현** — 설정이 아니라 불일치 |
| E4 | merge 분기가 삽입 페이지 전부에 **세그먼트 키워드**를 찍는다(`add_session`은 페이지별 LLM 추출) | **미재현** |
| E5 | `MidTermMemory.max_capacity` 클래스 기본값이 7이지만 드라이버가 2000을 넘겨 **기본값이 사문** | 2000 채택 |

E2는 A-MAC의 죽은 N/R과 같은 계열이다 — 논문은 recency를 heat의 한 축으로 제시하지만
실행 코드에서는 비교에 영향을 줄 수 없다.

### 배선

`MEMORYOS_PRESETS` + `fidelity` 인자(명시 kwarg가 프리셋을 덮어씀, Nemori와 동일 규칙),
`_keyword_overlap`/`_segment_heat`가 계보별 분기, `self._access`(승격이 리셋하지 않는
LFU 카운터 — upstream `access_frequency` 대응) 신설, `self.params`/`self.fidelity` 스탬핑
표면 추가. 벤치에 `memoryos_eval` config 추가.

**LFU 축출에 대해 알아둘 것**: 카운터는 검색 히트에만 오르므로 ingest-후-eval 벤치에서는
ingest 내내 전부 0이고 `min()`이 첫 키를 반환한다 — **LFU가 아니라 삽입순 FIFO로 퇴화**한다.
docstring에 명시.

**회귀 테스트 2건**: 계보 분리(가중치·recency·키워드·STM·축출 + 공유 상수 + kwarg 우선),
그리고 eval 계보에서 recency만 다른 두 세그먼트의 heat 차가 1e-4로 한 페이지 heat의
1/1000 미만임을 단언.

## C3. 다음: Zep

남은 것은 round-5 ①(community subgraph — label propagation + 동적 확장)과 read 경로
(현 `GraphRecall`은 최소 형태; 업스트림 Zep은 BFS ϕ + hybrid + reranker 조합). B3으로
감사 추적성은 해소됐으므로, community를 넣으면 측정 금지 해제 조건이 전부 채워진다.
