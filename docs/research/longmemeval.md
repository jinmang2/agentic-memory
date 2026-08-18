# LongMemEval 정본 조사 — 논문 · 공식 코드 · 데이터 · 방법론별 보고 현황 · 지표 비판

> **이 문서의 지위**: LongMemEval에 관한 이 리포의 **정본**이다.
> [`ace-longmemeval.md`](ace-longmemeval.md)는 2026-07-16의 ACE+LME 합본 1차 조사이고,
> 그 §2/§3이 담은 LME 내용은 이 문서가 **대체**한다(그 문서의 §1 ACE 부분은 계속 유효).
> 아래 §0에 1차본에서 **틀린 것으로 판명된 항목**을 모아두었다.
>
> | | |
> |---|---|
> | 조사일 | 2026-08-17 |
> | 공식 코드 | `github.com/xiaowu0162/LongMemEval` @ **`9e0b455f4ef0e2ab8f2e582289761153549043fc`** (2026-05-11), MIT. 로컬 `~/.agmem/upstream/longmemeval` |
> | 후속 벤치 | `github.com/xiaowu0162/LongMemEval-V2` @ **`2cc8c540bdb87fe6761629b585e727e1c4704520`** (2026-08-09). 로컬 `~/.agmem/upstream/longmemeval-v2` |
> | 논문 | arXiv:2410.10813 (ICLR 2025) / V2는 arXiv:2605.12493 |
> | 데이터 | `xiaowu0162/longmemeval-cleaned` (`_s_cleaned` sha256 `d6f21ea9…`, `_oracle` sha256 `821a2034…`), 폐기본 `xiaowu0162/longmemeval`도 대조용으로 취득 |
> | 이 문서의 모든 실측 | 우리 포트 `src/agmem/bench/longmemeval.py`의 로더로 산출, 재현 스크립트는 §9에 목록 |

---

## 0. 1차 조사본(2026-07-16/08-07)에서 교정된 것

| # | 1차본 서술 | 실측/재확인 결과 |
|---|---|---|
| C1 | `_s` 토큰 중앙값 **121,086**, 전체 60.5M (4.045 chars/token) | **틀림.** upstream이 실제로 쓰는 `o200k_base`로 재면 **4.610 chars/token**, 중앙값 **113,840**, 최대 114,991, 전체 **55.9M** |
| C2 | "우리 실측 121K라 128K 창에 아슬아슬" | **약화.** upstream 상한 126,200(=128K−800−1000)을 넘는 인스턴스 **0/500**. 창의 89%를 채울 뿐 절단은 없다. 애초에 구축 시 `enforce_json_length=115000`으로 맞춰진 값 |
| C3 | "`_M` 최적 구성: GPT-4o 65.7%, **Llama-70B 72.0%**" | **오귀속.** 논문 Table 3의 `Value=Round, K=V+fact` 행에서 65.7은 GPT-4o top-5, **72.0도 GPT-4o(top-10)**다. Llama-3.1-70B는 .638/.682 |
| C4 | "폐기본 데이터라 논문 수치와 대조 불가" | **`_s`에만 해당.** `longmemeval_oracle`은 폐기본과 cleaned가 **sha256 동일**(`821a2034…`)이라 논문의 oracle 수치(.870 / .924)는 지금 데이터로 직접 대조 가능하다 |
| C5 | 우리 포트 `evidence_session_ids` docstring: "abstention은 비어 있다" | **틀림.** abstention 30문항 전부 `answer_session_ids`가 1개씩 있고, `has_answer` 턴만 0이다. 코드 수정 완료 |
| C6 | "논문 GPT-4o `_s` = 60.6" (단일 값으로 인용) | **불완전.** Figure 3b는 CoN 없이 .606 / **CoN 켜면 .640**이다. 논문이 권장하는 세팅은 후자다 |

---

## 1. 벤치마크 — 논문

### 1.1 무엇을 재는가

"needle in a haystack"의 **대화판**이다. 바늘은 사람이 만든 evidence statement, 건초더미는 무관한 대화 세션이다. 결정적 설계는 **haystack이 test time에 조립된다**는 것 — 같은 500문항으로 115K(`_s`) / 1.5M(`_m`) / evidence만(`oracle`) 세 난이도를 만든다. 벤치마크라기보다 **벤치마크 생성기**다(`data/custom_history/sample_haystack_and_timestamp.py`가 그 생성기).

논문 Table 1의 위치 선정: 기존 벤치(MSC, DuLeMon, MemoryBank, PerLTQA, LoCoMo, DialSim) 중 **Knowledge Update를 가진 것은 LongMemEval뿐**이고, 5축(IE/MR/KU/TR/ABS)을 모두 가진 것도 이것뿐이다. 컨텍스트 깊이는 115k / 1.5M로 LoCoMo(10k)의 10~150배.

### 1.2 통합 프레임워크 — 3단계 4 control point (논문 §4)

논문의 진짜 기여는 500문항이 아니라 **메모리 시스템을 4축으로 분해한 것**이다. 이후 모든 실험이 이 축의 ablation이고, Table 2가 9개 시스템을 이 축에 투영한다.

```
[Indexing]                        [Retrieval]                 [Reading]
 CP1 Value  무엇을 저장하나         CP3 Query  무엇으로 찾나     CP4 Reading  어떻게 읽나
 CP2 Key    무엇으로 색인하나       (+ 검색 구조: flat/PPR/계층)
```

- **CP1 Value**: `session` vs `round`(user+assistant 1쌍). 논문 결론 **round 분해가 낫다** — Figure 5. 다만 Table 3을 보면 **검색 지표는 session이 더 좋고(R@10 .783 vs .692) QA는 round가 더 좋다**(GPT-4o top-10 .676 vs .670, K=V+fact에서 .700 vs .720). 검색을 잘하는 것과 답을 잘 하는 것이 어긋나는 첫 지점.
- **CP2 Key**: `K=V` / `K=fact` / `K=summary` / `K=keyphrase`, 그리고 join mode `separate`(별도 (k,v) 추가) / `merge`(원래 키에 합침) / `replace`(원래 키 대체). 결론 **`K = V + fact` (merge)**. session 기준 R@10 .783 → **.862**.
  - **`replace`는 손해**다(Table 10의 RM 행). 원문을 버리고 요약/사실만 저장하는 구조(LD-Agent, ChatGPT, Coze — 그리고 우리 리포의 Mem0류 fact-only 구조)가 정보를 잃는다는 게 §C.2의 지적이다.
  - `K=keyphrase` 단독은 재앙: round 기준 R@5 **.282** (K=V의 .582 대비 반토막).
- **CP3 Query**: **time-aware query expansion**. 질문에서 시간 범위를 LLM으로 추출해 그 구간 밖 세션을 뒤로 밀어낸다. TR 부분집합 R@10 .721→.797. 단 **GPT-4o로 추출할 때만** 이득이고 Llama-3.1-8B로 하면 하락(.711)한다 — Table 11의 실패 사례가 원인을 보여준다: 8B는 시간 언급이 없는 질문에도 범위를 만들어내는 **false positive**를 낸다.
- **CP4 Reading**: 아래 §1.3.

### 1.3 CoN(Chain-of-Note)이란 무엇인가 — 정밀 정의

**원 개념** (Yu et al. 2023): RAG에서 검색 결과에 노이즈가 섞였을 때, 곧바로 답하게 하지 말고 **문서마다 reading note를 먼저 쓰게** 한 뒤 그 노트로 추론하게 한다. 긴 문맥 독해를 두 개의 쉬운 작업으로 쪼갠다 — ① 중요한 디테일 **베껴 쓰기**, ② 짧아진 노트로 **추론하기**.

LongMemEval에는 이게 **두 구현**으로 들어와 있고, 재현에서 자주 뒤섞인다:

| alias (`run_generation.sh`) | 플래그 | 하는 일 | LLM 콜/문항 | 출력 상한 |
|---|---|---|---|---|
| `direct` | `--cot false` | 그냥 답하라 | 1 | 500 |
| **`con`** (논문 권장) | `--cot true` | **한 프롬프트 안에서** "먼저 관련 정보를 전부 추출하고 그 다음 추론하라" | 1 | 800 |
| `con-separate` | `--cot true --con true` | **청크마다 별도 LLM 콜**로 노트를 뽑고 원문을 `{"session_summary": note}`로 **치환**한 뒤 답하라 | 1 + 청크수 | 800 (노트는 500) |

논문 부록이 스스로 *"a small variation of CoN"*이라고 적는다 — 즉 권장 세팅 `con`은 사실상 **구조화된 CoT 한 방**이지 원논문의 문서별 노트 생성이 아니다. 진짜 CoN에 가까운 건 `con-separate`이고 이건 40세션이면 **문항당 41콜**이다.

`con` 프롬프트 원문 (`run_generation.py:55`):
```
I will give you several history chats between you and a user. Please answer the question
based on the relevant chat history. Answer the question step by step: first extract all the
relevant information, and then reason over the information to get the answer.
```

**JSON vs NL 포맷**도 CP4에 붙는다. `json`은 세션을 `json.dumps([{role, content}, …])`로, `nl`은 `"user: …\n\nassistant: …"`로 편다. §5.5의 결론이 미묘하다: **CoN 없이는 JSON이 NL을 일관되게 못 이긴다. CoN과 함께일 때만 JSON이 항상 이득**이다. 상호작용 최대 **10pp**.
**[실측 2026-08-17, docs/20]** oracle × mini 4셀에서 방향이 그대로 나온다 — json−nl이 con에서 **+2.40pp**
[−0.40,+5.40], direct에서 **+0.20pp** [−2.20,+2.60]. 각 셀은 500문항에서 미분리이고 DiD 구간은 안 샀으므로
**정합이지 확인은 아니다.** 두 포맷 모두 upstream과 바이트 동일(prompt_rediff D·E, 각 500/500)이라
차이는 포맷의 것이지 우리 렌더링의 것이 아니다.

> **논문이 말하지 않는 상호작용**: Figure 3b에서 Llama-3.1-70B는 oracle에서 CoN이 **+10.4pp**(.744→.848)인데 `_s`에서는 **−4.8pp**(.334→.286)다. Phi-3.5도 oracle −0.8 / `_s` −1.8. **CoN 이득의 부호가 컨텍스트 길이에 따라 뒤집히는데 본문은 "최대 10pp 이득"만 말한다.** 이건 우리가 싸게 재현할 수 있는 1순위 질문이다(§8).
>
> **[실측 2026-08-17, docs/20]** oracle 길이에서 부호는 **양수**다 — mini +4.40pp, luna +3.20pp(overall,
> 페어링 CI 둘 다 0 제외). 논문의 GPT-4o oracle 갭 +5.4pp를 우리 둘이 감싼다. **역전은 `_s`가 있어야
> 보이고**, 그 절반이 이 문서를 쓰는 시점에 도는 중이다.

### 1.4 질문 유형과 실측 분포

| 능력 | question_type | cleaned `_s` 실측 문항수 | 요구 사항 |
|---|---|---|---|
| IE | `single-session-user` | 70 | 사용자가 한 세션에서 말한 디테일 회상 |
| IE | `single-session-assistant` | 56 | **어시스턴트가** 말한 것 회상 |
| IE | `single-session-preference` | 30 | 정답이 아니라 **rubric**으로 채점 |
| MR | `multi-session` | 133 | 여러 세션 정보의 집계·비교 |
| TR | `temporal-reasoning` | 133 | 명시 시간 표현 + 타임스탬프 추론 |
| KU | `knowledge-update` | 78 | 값이 바뀌었을 때 최신값 |
| ABS | (교차절단면, `_abs`) | 30 | false-premise 질문에 기권 |

**타입 간 4.4배 불균형**(133 vs 30)이고, 이것이 "정확도가 두 개"(§2.6)를 실질적 문제로 만든다.
데이터 내부 태스크명 ↔ 공식명 매핑(README): `single_hop`→SSU, `implicit_preference_v2`→SSP, `assistant_previnfo`→SSA, `two_hop`/`multi_session_synthesis`→MS, `temp_reasoning_{implicit,explicit}`→TR, `knowledge_update`→KU.

### 1.5 데이터 구축 파이프라인

1. **164개 사용자 attribute**를 5범주(demographic / lifestyle / situational context / life events / belongings)로 **수작업** 정의 (Table 5)
2. Llama-3-70B가 attribute 기반 배경 문단 생성
3. LLM이 (질문, 답) 초안 → **사람이 전량 필터·재작성**
4. 사람이 답을 timestamp 붙은 **evidence statement**로 분해
5. LLM self-chat으로 evidence session 시뮬레이션 (사용자가 정보를 **간접적으로** 노출) → 사람이 검수·편집
6. haystack 컴파일: filler를 **simulated 0.5 / ShareGPT 0.25 / UltraChat 0.25**로 섞음 (`sample_haystack_and_timestamp.py:99`)
7. 타임스탬프는 순서를 따라 무작위 배정, evidence의 사전 정의 시각이 앵커, 기본 2023년 5월

총 400 human-hours(구축) + 150(상용 시스템 연구).
**실측 교차검증**: cleaned `_s`의 세션 id 접두어를 세면 `sharegpt` 6,179 / `ultrachat` 6,008 / `answer` 948 / 나머지(해시형 = simulated) 10,732 — 27.0 / 26.2 / 46.8%로 설계 비율과 정합한다.

### 1.6 논문 수치 전량 (HTML 파싱, 표 원문)

**Figure 3b — long-context LLM (폐기본 `_s` 기준)**

| Model | Size | Oracle | `_S` | Drop | | Oracle(CoN) | `_S`(CoN) | Drop |
|---|---|---|---|---|---|---|---|---|
| GPT-4o | — | .870 | **.606** | 30.3% | | .924 | **.640** | 30.7% |
| Llama-3.1 Instruct | 70B | .744 | .334 | 55.1% | | .848 | .286 | 66.3% |
| Llama-3.1 Instruct | 8B | .710 | .454 | 36.1% | | .710 | .420 | 40.8% |
| Phi-3 128k Instruct | 14B | .702 | .380 | 45.9% | | .722 | .344 | 52.4% |
| Phi-3.5 Mini Instruct | 4B | .660 | .342 | 48.1% | | .652 | .324 | 50.3% |

**Figure 3a — 상용 시스템** (Offline Reading GPT-4o .9184 / ChatGPT GPT-4o .5773 · GPT-4o-mini **.7113** / Coze GPT-4o .3299 · GPT-3.5 .2474)
⚠️ **97문항, 세션 3–6개짜리 축소 세팅**이고 abstention **미포함**(당시 데이터셋에 없었음), TR 일부와 SSA는 **제외**(부록 B). 500문항 표와 같은 축에 놓으면 안 된다. Table 7(사람 채점)이 그 근거를 보여준다 — ChatGPT는 mini가 4o를 IE에서 1.000 vs 0.688로 이긴다.

**Table 3 — `_M` 키 설계** (Stella V5 1.5B 검색, 상위 발췌)

| Key Design | R@5 | NDCG@5 | R@10 | NDCG@10 | GPT-4o top-5 | top-10 | L3.1-70B top-5 | top-10 |
|---|---|---|---|---|---|---|---|---|
| Round, K=V | .582 | .481 | .692 | .512 | .615 | .670 | .600 | .624 |
| Round, K=keyphrase | .282 | .159 | .392 | .303 | .425 | .489 | .404 | .450 |
| **Round, K=V+fact** | .644 | .498 | .784 | .536 | **.657** | **.720** | .638 | .682 |
| Session, K=V | .706 | .617 | .783 | .638 | .670 | .676 | .592 | .570 |
| **Session, K=V+fact** | **.732** | **.620** | **.862** | **.652** | .714 | .700 | .588 | .584 |

**Table 4 — 시간 인식 질의 확장 (TR 부분집합)**: `K=V+fact` session R@10 .721 → **.797**(GPT-4o 확장) / .711(Llama-8B 확장, **하락**).
**Table 6 — judge 메타평가**: 평균 .98(GPT-4o 답변)/.97(Llama-8B 답변). **최저는 single-session-preference .90과 abstention .90.**
**Table 8 — 소형 모델**: `_S` LC Direct가 Llama-3.2-3B **0.008**, 1B 0.010, Qwen2.5-7B 0.128. 오라클에서는 .522/.386/.282 — 긴 컨텍스트에서 지시 따르기가 붕괴한다.
**Table 9 — 검색기 비교**: Contriever가 Stella V5 1.5B와 대등하거나 낫다(session K=V R@10 .823 vs .794). BM25는 .710.
**§E.5 오류 분석**: 전체의 **15–19%가 "검색은 맞았는데 생성이 틀림"**(오류 중 40–50%).

---

## 2. 공식 코드 — 파일별 정독

### 2.1 스냅샷과 구성

```
src/generation/run_generation.{py,sh}      QA 생성 (full-context / RAG)
src/retrieval/run_retrieval.{py,sh}        인덱싱 + 검색 + 검색지표
src/retrieval/{eval_utils,index_expansion_utils}.py
src/index_expansion/batch_expansion_*.py   오프라인 키 확장 6종
src/index_expansion/temp_query_search_pruning.py  시간 인식 질의 확장
src/evaluation/{evaluate_qa,print_qa_metrics,print_retrieval_metrics}.py
data/custom_history/sample_haystack_and_timestamp.py   haystack 생성기
requirements-lite.txt (평가만) / requirements-full.txt (검색+생성)
```

우리 포트가 인용한 file:line은 이 SHA에서 **전부 일치**한다(§4).

### 2.2 생성 경로 `run_generation.py`

`prepare_prompt`(:46-282)이 전부다. retriever_type 6종(`orig-{turn,session}`, `oracle-{turn,session}`, `flat-{turn,session}`, `no-retrieval`)에 따라 chunk를 모으고 → topk 자르고(:171-174) → `has_answer` 제거(:177-191) → **날짜 정렬**(:225) → 포맷(:234-259) → **tiktoken 절단**(:266-279) → 템플릿 삽입(:280).

핵심 상수: `gen_length` = 800(cot) / 500(direct) (:341-342), `max_retrieval_length = model_max_length − gen_length − 1000` (:343) = gpt-4o에서 **126,200**.

### 2.3 검색 경로 `run_retrieval.py` — **가장 큰 발견이 여기 있다**

**LME-A1 · 인덱스에 user 턴만 들어간다.**
`process_item_flat_index`(:202-229): session 단위는 `' '.join(content for interact in data if interact['role']=='user')`(:206), turn 단위는 `if turn['role']=='user'`(:214). **assistant 턴은 색인에서 완전히 빠진다.** 논문 §5.1이 *"we only keep the user-side utterances"*라고 한 줄 적지만, 그 귀결은 적지 않는다.

**LME-A2 · 그 귀결: SSA 타입이 검색 평가에서 사실상 삭제된다.**
:209는 evidence 세션이라도 user 쪽에 `has_answer`가 없으면 id를 `answer_`→`noans_`로 **강등**하고, :396-402의 집계는 (a) 모든 `_abs` 문항과 (b) **user 쪽 `has_answer` 턴이 하나도 없는 문항**을 통째로 뺀다.

실측(cleaned `_s`):

| | |
|---|---|
| evidence 세션 총계 | 948 |
| 그중 **user 턴으로 도달 가능** | 803 (**145개, 15.3%가 검색 불가**) |
| user 쪽 evidence 턴이 0인 문항 | **72** (abstention 21 + 비-abstention 51) |
| 그 51건의 정체 | **전부 `single-session-assistant`** (해당 타입 56문항 중 **51건 = 91%**) |
| **검색 지표의 실제 분모** | 500 − 30(abstention) − 51 = **419** |

⇒ **논문 Table 3·4·9·10의 모든 recall/NDCG는 419문항에서 계산됐고, SSA 타입은 91%가 빠져 있다.** README는 abstention 30건 제외만 밝힌다.

**LME-A3 · 검색 골드가 세션 id의 부분문자열이다.**
`correct_docs = [d for d in corpus_ids if "answer" in d]`(:272) — `answer_session_ids` 필드를 쓰지 않는다. 실측상 이 릴리스에서는 두 정의가 **500/500 일치**하므로 잠복 상태지만, filler id에 "answer"가 들어가는 순간 골드가 오염된다(현재 filler 접두어는 `sharegpt`/`ultrachat`/해시).

**LME-A4 · "oracle"이라는 말이 한 리포에서 두 가지를 가리킨다.**
`--retriever oracle`(:278-285)은 **완벽한 랭킹**(정답 문서를 앞으로)이고, `longmemeval_oracle.json`은 **evidence만 담긴 데이터 파일**이다. 서로 다른 실험이다.

**LME-A5 · 인덱스 캐시가 없다.** dense 검색이 질문마다 **코퍼스 전체를 재임베딩**한다(:130-163). BM25도 질문마다 `BM25Okapi(corpus)`를 새로 만든다(:109). 정확성 문제는 아니지만 "GPU 필수"의 진짜 이유다.

**LME-A6 · `check_args`의 오타.** :53의 목록에 `'session_userfact'`(언더스코어)가 들어 있는데 실제 choice는 `session-userfact`(하이픈)다. 그래서 이 방법에 대해서만 granularity 검증이 **조용히 건너뛰어진다**.

**LME-A7 · 집계 루프의 `except: continue`**(:406)가 지표 계산 오류를 삼킨다.

### 2.4 인덱스 확장 `index_expansion_utils.py`

`resolve_expansion`(:17-80)이 join mode를 구현한다.

- `separate`: 확장 텍스트를 **같은 세션 id로** 새 코퍼스 항목에 추가(:50-52).
- `merge`: `expansion + ' ' + 원문`(:60). **확장 항목마다 원문을 복제**한다.
- `replace`: 원문을 확장으로 갈아치움(:67).

**LME-A8 · `split-*` 모드는 recall@k의 의미를 바꾼다.** `split-merge`에서 세션에 fact가 10개면 **같은 id를 가진 코퍼스 항목이 10개** 생긴다. `evaluate_retrieval`은 `recalled_docs = set(corpus_ids[idx] for idx in rankings[:k])`(eval_utils.py:25)라 상위 k 슬롯을 같은 세션이 여러 개 잡아먹는다. 논문 본표(Table 3)는 비-split `merge`(KM)를 쓰므로 본 수치엔 안 닿지만, Table 10의 RM/KM 비교를 읽을 때 필요한 사실이다.

**LME-A9 · 확장 실패는 빈 문자열로 조용히 대체된다**(:23-24, :30-31, :36-38). 캐시에 `None`이 얼마나 있는지는 로그에 남지 않는다.

### 2.5 시간 인식 질의 확장 `temp_query_search_pruning.py`

**LME-A10 · 릴리스 상태로 실행되지 않는다.** :88이 `json.load(sys.argv[1])` — `json.load`는 파일 객체를 받는데 문자열 경로를 넘긴다. 첫 줄에서 `AttributeError`로 죽는다. **논문 Table 4를 만든 아티팩트가 그대로는 돌지 않는다.**

**LME-A11 · pruning이 아니라 re-ranking이다.** 범위 밖 세션을 지우지 않고 **랭킹 뒤로 밀 뿐**(:172, `left + right`)이다. 논문 서술("narrow down the search space")과 코드가 다르다. 검색 공간은 그대로이므로 비용 절감도 없다.

**LME-A12 · 지표 출력이 집합(set)이라 이름이 사라진다.** :191, :193이 `print({round(np.mean(...), 4) for k in ...})` — dict가 아니라 **set comprehension**이라 어느 지표인지 알 수 없고 값이 같으면 중복 제거된다.

기타: API 키/조직이 `YOUR_API_KEY`/`YOUR_ORGANZATION`(오타) 하드코딩, 모델은 주석 토글로 전환, 범위 확장 ±2일 하드코딩, `if out_data['start'] in query_date: return {}`라는 정체불명의 hack(:80-81).

### 2.6 채점 `evaluate_qa.py` / `print_qa_metrics.py` / `print_retrieval_metrics.py`

judge: `gpt-4o-2024-08-06`, `temperature=0`, `max_tokens=10`, 판정은 `'yes' in eval_response.lower()`(:113). 타입별 5분기(`get_anscheck_prompt`:24-43)이며 미지 타입은 `raise NotImplementedError`(:39).

5분기의 규칙 차이(모두 원문 전사):
- 기본 3타입(SSU/SSA/MS): "정답의 **부분집합만** 담으면 no"
- `temporal-reasoning`: 위 + "**날짜 off-by-one은 감점 금지**"
- `knowledge-update`: **"부분집합이면 no" 문장을 뺌** + "옛 정보를 같이 말해도 갱신된 답이 맞으면 correct"
- `single-session-preference`: 정답이 아니라 **Rubric** 대조, "rubric 전부를 반영할 필요 없음"
- abstention: 질문 유형 **무시**, "모른다고 제대로 말했는가"만 봄. `'_abs' in question_id`(**substring**, :101)로 분기

**LME-A13 · 정확도가 셋이다.** `evaluate_qa.py:130`이 자체 `Accuracy`를 찍고, `print_qa_metrics.py`가 `Task-averaged`(6타입 평균의 평균, :31)와 `Overall`(전체 문항 평균, :32)을 찍는다. 타입 불균형 4.4배라 뒤 둘은 같지 않다.

**LME-A14 · abstention 이중계상.** 모든 엔트리가 자기 `question_type` 버킷에 들어가고, `_abs`면 **추가로** abstention 버킷에도 들어간다(:21-23). 30문항이 두 정확도 안에 그대로 남는다.

**LME-A15 · judge가 `assert`로 강제된다.** `assert entry['autoeval_label']['model'] == 'gpt-4o-2024-08-06'`(:20). 다른 judge로 매긴 결과는 공식 집계기가 **읽기를 거부**한다.

**LME-A16 · subset은 채점 자체가 안 된다.** 6타입 중 하나라도 판정 행이 0이면 `np.mean([])`로 `Task-averaged Accuracy: nan`이 나온다. 실행으로 확인함.

**LME-A17 · README 명령이 코드와 다르다.** README는 `python3 print_qa_metrics.py gpt-4o hyp.log ref.json`(3인자)인데 스크립트는 `len(sys.argv) != 3`(=2인자)에서 usage 출력 후 종료한다. 로그 파일명도 README는 `[hyp].log`, 코드는 `[hyp].eval-results-{model}`(:56).

**LME-A18 · 부분 실행이 완주처럼 집계된다.** `run_generation.py:376-378`은 API 예외를 `continue`로 삼키고, `evaluate_qa.py:92-94`는 ref에 없는 항목을 skip한다. **어디에도 n==500 검증이 없다.**

**LME-A19 · `print_retrieval_metrics.py`의 죽은 코드.** `task2type`/`type2acc`(:14-24)가 만들어지고 한 번도 쓰이지 않으며, 실제 출력은 `try/except: pass`(:32-35, :39-42)라 키가 없으면 **조용히 아무것도 안 찍는다**.

### 2.6a 자기 검증 — 무엇을 실행했고 무엇을 읽기만 했나

2026-08-17 1차 보고에서 A10을 **읽기만 하고 실행하지 않은 채** "실행되지 않는다"고 적었다. 실행해 보니
**결론은 맞았고 실패 지점은 하나 더 있었다.** 이 절은 그 재검을 기록한다.

```
① README가 지시하는 그대로 (README.md:228-229, `cd src/index_expansion`):
   ModuleNotFoundError: No module named 'src'        ← temp_query_search_pruning.py:8
   (파일이 `from src.retrieval.eval_utils import …`를 하는데 README는 하위 디렉터리에서 실행시킨다)

② PYTHONPATH를 리포 루트로 고쳐 import를 통과시킨 뒤:
   AttributeError: 'str' object has no attribute 'read'   ← :88 `json.load(sys.argv[1])`
```

즉 이 스크립트는 **두 개의 독립된 이유로** 실행되지 않는다. `git log -L 88,88`은 그 줄이 최초 커밋
`6a92d1a`(2024-10-12)에 그대로 들어왔고 이후 한 번도 수정되지 않았음을 보여준다.

### 2.6b 이슈 트래커 대조 — 무엇이 이미 신고됐고, 저자가 뭐라 했고, 안 고쳐졌나

공식 리포의 issues/PR **54건 전수 조회**(GitHub API, 2026-08-17). 이 문서의 발견 중 여럿은
**우리가 처음 찾은 것이 아니다.** 출처를 밝히는 편이 발견을 주장하는 것보다 강하다.

| 우리 번호 | 이미 신고됨 | 저자 반응 | 현재 상태 |
|---|---|---|---|
| **A1/A2** (user-only 인덱싱 → SSA 삭제) | **ISS #7** (2025-04-27, closed) — "recall@all bug due to empty correct_docs" | **인정**: *"the `has_answer` annotations lie in assistant turns… most of the methods in this repo drop the assistant turns during indexing, the correct docs could be empty. **For now, I have filtered out these questions**"* | 필터 추가로 종결 (2025-04-26 커밋) |
| **A8** (merge가 코퍼스 수를 바꿈) | **ISS #5** (2025-04-02, open) — 코퍼스 501 → **380**(−24%) | 원인 지목만: *"trace your code and make sure `cur_item_expansions` is not an empty list"* | **미수정** |
| **A8 심화** (확장 없는 항목이 **버려짐**) | **ISS #9 코멘트** (2025-08-14) — "sessions/turns without key expansion are discarded… unfair comparison" | **인정**: *"This is indeed possible and a fallback case that just uses the unexpanded item should be added."* | **12개월째 미수정** (`index_expansion_utils.py`는 커밋이 `6a92d1a` 하나뿐) |
| **A15/A17** (judge assert + README 명령) | **PR #47** (2026-05-30, **open**) — 우리와 같은 두 결함을 같은 SHA `9e0b455`에서 지목 | 무응답 | **미머지** |
| CoN 정의 | **ISS #9** (2025-07-23, open) | **답변**: *"`con-separate` was not used in the paper… Effectively, Chain-of-Note is just a different chain-of-thought prompt."* | 문서 미반영 |
| Table 9 재현 실패 | **ISS #27** (2026-03-06, open) — BM25 session K=V로 `recall_any` 0.752/0.802를 얻어 논문 0.634/0.710과 불일치 | 무응답 | **5개월째 미응답** |
| 데이터 결함 | **15건** (#11 #12 #13 #15 #19 #20 #21 #22 #26 #37 #38 #39 #40 #41 #54), **12건 open** | 일부만 반영 | 진행 중 |

**ISS #16 (2025-09-29)이 우리 실측을 외부에서 재현했다.** 제3자가 붙인 실행 로그에
`Ignored 30 instances due to abstention` + **`Additionally ignored 51 instances due to no target
turns from the user side`**가 그대로 찍혀 있다. 우리가 데이터에서 센 51과 **정확히 같다**.

### 2.6c 🔴 논문의 검색 수치는 지금 코드로 재현되지 않는다 (LME-A20)

A1/A2에 시간축을 넣으면 더 강한 사실이 나온다. `git log -- src/retrieval/run_retrieval.py`는 커밋이
셋뿐이다: `6a92d1a`(2024-10-12, 최초) → `cf920ec`, `b60a5b7`(**둘 다 2025-04-26**). 그리고 이슈 #7
당시 SHA(`8a44eb1`)의 집계부를 꺼내 보면 필터가 **없다** — abstention만 제외한다.

```python
# 8a44eb1 (논문 시점 코드)
averaged_results[t][k] = np.mean([... for x in results if '_abs' not in x['question_id']])
```

`correct_docs`가 빈 리스트일 때 `eval_utils.evaluate_retrieval`이 무엇을 반환하는지가 관건이다:

```python
recall_any = float(any(doc in recalled for doc in []))   # → 0.0
recall_all = float(all(doc in recalled for doc in []))   # → 1.0   (vacuous truth)
ndcg       = 0.0                                          # ideal_dcg == 0 → return 0.
```

그리고 `print_retrieval_metrics.py:30,37`이 보고하는 이름은 **`recall_all@k`와 `ndcg_any@k`**다.

⇒ **논문의 Recall 열은 evidence가 user 턴에 없는 ~51개 인스턴스에 대해 강제로 1.0을 받은 값이고,
NDCG 열은 같은 인스턴스에서 0을 받은 값이다.** 필터는 논문 발표 **뒤**(2025-04-26)에 추가됐으므로
오늘 코드는 그 인스턴스들을 아예 빼고 419개로 평균한다 — **같은 명령이 다른 분모, 다른 값을 낸다.**
산술 예시(측정이 아니라 예시): 나머지 419개의 참 recall이 0.60이라면 논문 시점 값은
`(51·1.0 + 419·0.60)/470 = 0.643`, 오늘 값은 `0.60` — **약 4pp의 계통 편차**다.
ISS #27이 5개월째 답을 못 받고 있는 재현 실패가 이 구조와 무관하지 않다.

> 이 항목은 우리가 **실행으로 확인하지 못했다** — 검색 재현은 GPU와 dense 모델이 필요하다.
> 확인된 것은 ① 코드 경로 ② git 이력 ③ 빈 `correct_docs`의 반환값 ④ 51이라는 실측 개수이고,
> 편차의 **부호는 연역되지만 크기는 측정되지 않았다.** 이 구분을 흐리지 말 것.

### 2.7 구축 스크립트 `sample_haystack_and_timestamp.py`

filler 비율 상수(:99), `enforce_json_length`(`_s`=115000)로 haystack을 토큰 길이에 맞춰 패킹(:109-111), 타임스탬프는 evidence 앵커 주변으로 무작위 생성(:34-88). 즉 **`_s`가 115K 근처인 것은 우연이 아니라 제약**이다 — 그래서 "128K 창에 아슬아슬하다"는 서술은 설계 의도를 사후 관찰로 착각한 것이다(§0 C2).

---

## 3. 데이터 — 실측과 릴리스 계보

### 3.1 규모 (o200k_base 실측 · `_m`은 2026-08-18 취득)

| | `_s` (cleaned) | `oracle` | **`_m` (cleaned)** |
|---|---|---|---|
| 인스턴스 | 500 | 500 | **500** |
| 세션/인스턴스 | p50 **48** (38–62) | p50 **2** (1–6) | p50 **476** (460–490) |
| 턴/인스턴스 | 평균 **494** | 평균 **21.9** | 평균 **4,894** (4,586–5,229) |
| 렌더 토큰/인스턴스 | p50 **113,840**, 최대 114,991 | p50 **6,127**, 최대 23,770 | p50 **1,107,053**, 최대 1,189,008 † |
| 전체 토큰 | **55.9M** | **3.06M** | **553.9M** † |
| 전체 턴 | **246,750** | 10,960 | **2,446,993** |
| 전체 세션 | 23,867 | 948 | **237,655** |
| chars/token | **4.610** | — | 4.610 가정 † |
| upstream 상한(126,200) 초과 | **0/500** | 0/500 | **500/500 (중앙값 8.8배)** |
| 파일 크기 | 277,383,467 B | 15,388,478 B | **2,737,100,077 B** |

† `_m` 토큰은 `_s`에서 실측한 4.610 chars/token 환산값이다(o200k 실측은 `LME_EXACT_TOKENS=1`).
sha256 `9d79e552…`. 재현: `scripts/repro/lme_audit/m_stats.py`.

`_s`는 LoCoMo 전체 캠페인(5,882턴)의 **42배** 물량이고, `_m`은 다시 그 **9.9배**(LoCoMo의 416배)다.
oracle의 948 세션은 `_s`의 evidence 세션 948과 정확히 일치한다(교차검증).

**`_m`의 경계 검증.** `_m`에는 대조할 발표 인스턴스 수치가 없다. 대신 `_m`은 `_s`와 **같은 500문항**을
담는다 — 스트리밍 로더가 뱉은 500개의 `(question_id, question, answer, question_type)` 집합이 `_s`의
집합과 **완전히 일치**했고, 타입 분포도 동일하며(56/70/133/78/133/30, abstention 30), 병렬 haystack
배열이 어긋난 인스턴스는 0건이다. 잘못 쪼개는 스캐너는 이 집합에 착지하지 못한다.

**세 가지 귀결.**
1. **`_m`에 full-context arm은 존재할 수 없다.** 중앙값 인스턴스가 128K 창의 **8.8배**다. `_s`가
   창에 들어가도록 패킹된 데이터(§2.7)인 것과 정반대이고, 그래서 `_m`에서는 검색이 선택지가 아니라
   전제다. **§9.4 약점 0이 지목한 "안 잰 체제"의 정확한 의미가 이것이다.**
2. **D3가 여기서 발동한다.** upstream 상한 초과가 `_s` 0/500 → `_m` **500/500**이다. 캡을
   chars/4로 환산하던 구 코드는 `_m`의 **모든** 인스턴스를 13% 더 잘랐을 것이다. §10.3이 "받기 전에
   고치라"고 한 이유가 실측으로 확인됐다.
3. **evidence 위치는 여전히 균등하다**: p10 0.08 / p50 0.48 / p90 0.91 / 평균 0.49 (`_s`: 0.11 /
   0.54 / 0.92 / 0.52). haystack이 10배가 돼도 증거가 앞이나 뒤로 쏠리지 않는다 — 즉 `_m`의
   난이도 증가는 위치 편향이 아니라 **순수한 distractor 희석**이다(evidence 세션은 여전히 평균 2개,
   이제 476개 중 2개).

### 3.2 릴리스 계보 — 폐기본 vs cleaned

2025-09-19에 `xiaowu0162/longmemeval`이 폐기되고 `longmemeval-cleaned`가 같은 날 공개됐다("removes noisy history sessions that interfere with the answer correctness"). 두 릴리스를 **모두 받아 diff**한 결과:

| | |
|---|---|
| `longmemeval_oracle` | **sha256 동일** (`821a2034…`) — 손대지 않았다 |
| `_s` 크기 | 278,025,796 → 277,383,467 bytes |
| id / question / answer / question_type / question_date / answer_session_ids | **전부 무변경** (500/500) |
| 세션 **추가** | 0 |
| 세션 **편집** | 0 |
| 세션 **삭제** | **1,243개** (454/500 인스턴스, 평균 2.74, 최대 8) |
| **evidence 세션 삭제·편집** | **0건** |
| 세션/인스턴스 | p50 50 → 48, 총 25,112 → 23,867 (**−5.0%**) |

⇒ cleaning은 **순수 삭제**였고 **distractor filler만** 걷어냈다. 따라서
1. **cleaned 점수는 구조적으로 폐기본보다 높다.** 논문의 `_s` 60.6/64.0은 오늘 데이터 기준 **하한**이다.
2. **oracle 계열 수치만 통시 비교가 가능하다.** 논문의 .870/.924는 지금 바이트로 잰 값이다.
3. `_s` 수치를 인용하는 모든 발표는 **릴리스를 명시해야** 한다. 실제로는 아무도 명시하지 않는다(§5).

### 3.3 evidence 위치와 절단 실험

evidence 세션의 정규화 위치(0=가장 오래된, 1=가장 최근)는 **균등**하다: p10 0.11 / p50 0.54 / p90 0.92 / 평균 0.52.

세션 캡을 걸었을 때 evidence를 잃는 인스턴스 비율:

| cap | 앞에서 자르기 `[:cap]` | 뒤에서 자르기 `[-cap:]` (upstream 방식) |
|---|---|---|
| 20 | 79.2% | 74.4% |
| 30 | 62.0% | 54.6% |
| 40 | 30.0% | 26.4% |
| **50** (`run_generation.sh` 기본값) | 2.4% | **0.8%** |
| 60 | 0% | 0% |

⇒ **비용은 세션 수로 줄일 수 없다. 인스턴스 수로만 줄일 수 있다.** 그리고 인스턴스로 줄이면 공식 집계기가 `nan`을 뱉는다(LME-A16). 두 사실이 합쳐져 **"싸게 subset으로 돌려본다"는 설계가 원천 봉쇄**된다.

`_s`에서 세션 50개를 넘는 인스턴스는 **104/500**이므로, 쉘 스크립트 기본값 `topk=50`으로 돌리면 upstream 규칙(꼬리)에서도 4건이 evidence를 잃는다.

---

### 3.4 데이터 자체의 결함 — 우리가 직접 검증한 3건

공식 트래커의 데이터 결함 신고 15건 중 3건을 **우리가 가진 파일에서 직접 확인**했다.

**① 중복 filler 세션 (ISS #54, 2026-08-16, open)** — 신고와 **13/13 완전 일치**.
같은 `haystack_session_id`가 한 인스턴스 안에 두 번 나오고, **본문은 바이트 동일, 날짜만 다르다**.
세션 출현 23,867 → 고유 23,854. 전부 non-answer filler다.
신고자는 원인을 `sample_haystack_and_timestamp.py:190-200`의 uniqueness guard 없는 `random.choice`로
지목한다. 우리 검증 스크립트: `scripts/repro/lme_audit/dup_sessions.py`.
영향: `(question_id, session_id)`를 문서 신원으로 쓰는 어댑터는 한 벌을 버리고, 안 그런 어댑터는
검색 슬롯 두 칸을 같은 내용에 쓴다 — **시스템마다 다른 코퍼스를 보게 된다.**

**② gold 산술 오류 (ISS #41, 2026-04-28, open)** — 확인됨.
`370a8ff4` "How many weeks had passed since I recovered from the flu when I went on my 10th jog outdoors?"
gold = **`15`**. 그런데 evidence 세션 두 개의 날짜가 `2023/01/19`(독감 회복)와 `2023/04/10`(10번째 조깅)이다.
**81일 = 11.57주.** temporal judge는 "**날짜 off-by-one**"만 봐주므로 3.4주 오차는 구제되지 않는다 —
**정답을 정확히 계산한 시스템이 오답 처리된다.**

**③ 불완전한 turn-level gold (ISS #22, 2025-12-08, open)** — 확인됨.
`51a45a95` "Where did I redeem a $5 coupon on coffee creamer?" gold = `Target`.
evidence 세션 12턴 중 `has_answer` 표시는 **딱 1개**이고, 그 턴의 원문은
*"I actually redeemed a $5 coupon on coffee creamer last Sunday…"* — **`Target`이라는 말이 없다.**
Target은 같은 세션의 **다른** 턴(Cartwheel 앱 언급)에서만 나온다. 즉 turn-level recall gold를 100%
맞혀도 답을 도출할 수 없다. `has_answer`를 학습/평가 신호로 쓰는 모든 설계가 이 오라벨을 상속한다.

> 이 세 건은 **`_s`의 cleaned 릴리스**에 남아 있다. §3.2가 보인 대로 cleaning은 filler 삭제만 했고
> gold와 라벨은 손대지 않았기 때문이다.

## 4. 우리 포트 — 상태와 재대조

`src/agmem/bench/longmemeval.py` + `tests/test_longmemeval.py` 39개(전부 통과).
드라이버는 **배선 완료**: `scripts/repro/exp_lme_reading.py`가 §8.5의 가드 6종을 하드코드로 박고
oracle 4 arm을 완주시켰다(2026-08-17, [`docs/20`](../20-lme-reading.md)). 이 리포는 이제 자체 LME
수치를 갖는다 — 단 **oracle 한정**이고 `_s`는 아직 하나도 없다.

배선 중 발견한 결함 2건(둘 다 수정, `b4812d0`):
- `answer`/`judge_answer`가 `temperature=0`·`max_tokens=N`을 **호출 오버라이드로 고정**하고 있었다.
  `LLMClient.chat`은 오버라이드를 RoleConfig보다 우선해 그대로 전송하므로, `max_completion_tokens`를
  요구하고 temperature를 거부하는 gpt-5.6-luna에서 **400으로 죽는다** — 계획된 4 arm 중 2개가
  애초에 실행 불가였다. 이제 캡 키는 `RoleConfig.max_tokens_key`에서 오고, fixed-sampling 모델에서는
  temperature를 빼고 그 사실을 stamp에 이탈로 남긴다(D6).
- oracle 정렬 부재(§4.1 C)는 이 절의 예고대로 34/500 이탈이었고, `sort_haystack_by_date` 적용 후
  `prompt_rediff.py` config D가 **500/500 바이트 동일**을 보고한다.

### 4.1 프롬프트 바이트 대조 (2026-08-17)

upstream `prepare_prompt`을 클론에서 그대로 떼어내(orig-session / json / useronly=false / cot=true, tiktoken 절단 포함) 우리 `render_sessions`+`ANSWER_PROMPT_CON`과 실데이터로 대조:

| 설정 | 결과 |
|---|---|
| `_s`, topk ≥ 세션수 | **60/60 바이트 동일** ✅ |
| `_s`, topk=50 | 60개 중 **18개 불일치** (upstream은 꼬리 `[-topk:]`, 우리 `_capped`는 머리) |
| `oracle`, 정렬 없이 | 500개 중 **34개 불일치** (길이 동일 = 순수 재정렬) |

⇒ 드라이버 필수 가드 3개: **oracle은 정렬 후 렌더**, **세션 캡 금지**, topk는 전량.

### 4.2 집계기 대조

공식 `print_qa_metrics.py`를 **셸아웃 실행**해 우리 `aggregate`와 50회 무작위 대조(타입 불균등 + abstention 25%): task_averaged / overall / abstention 전부 **최대 편차 0.000000pp**.

### 4.3 남은 이탈 (전부 문서화됨)

- **D1 순서**: upstream은 검색 결과를 날짜 정렬 후 포맷. 우리 `MemoryBundle.render`는 융합점수 순. full-context 경로는 upstream과 동일(정렬됨).
- **D2 ingest 단위**: upstream은 ingest가 없다(haystack 위에서 검색). 우리는 턴마다 `add_message`.
- **D3 토큰 환산**: 우리 `max_history_tokens`는 `MemoryBundle.CHARS_PER_TOKEN=4`로 환산 → 실측 4.61 대비 **~13% 과절단**. 캡이 **opt-in**이라 안 넘기면 무해하지만, "latent"의 정확한 뜻은 이것이다 — **`_s`에서 upstream 값 126,200을 그대로 넘기면 유효 캡이 126,200×4 = 504,800자 ≈ 109.5K 토큰이 되어 중앙값 인스턴스(113.8K 토큰 ≈ 524,800자)부터 절단된다.** upstream에서 이 절단은 no-op(초과 0/500)이므로 `_s`/oracle의 바이트 충실은 **캡을 넘기지 않는 것**이다. 캡이 실제로 필요한 변종은 `_m`뿐이고, 그래서 D3 수정이 `_m` 취득에 선행한다(§10.3).
- `con-separate` 미포팅(조용한 강등 대신 raise).

---

## 5. 방법론별 LongMemEval 보고 현황

`~/.agmem/upstream` 클론 10벌 전수 grep + 걸린 것은 코드 원문 정독.

### 5.1 비교 가능성 행렬

| 우리 organizer | LME 보고 | **릴리스** | **reader** | **judge** | 검색 예산 | 공식 프로토콜 준수 |
|---|---|---|---|---|---|---|
| `amem` | ✗ | — | — | — | — | — |
| `memoryos` | ✗ | — | — | — | — | — |
| `gmemory` | ✗ | — | — | — | — | — |
| `reasoning_bank` | ✗ | — | — | — | — | — |
| `ace` | ✗ | — | — | — | — | — |
| `amac` | ✗ | — | — | — | — | — |
| `nemori` | 64.2 / 74.6 | **폐기본** | gpt-4o-mini / gpt-4.1-mini | gpt-4o-mini 또는 gpt-4.1-mini | top-10 ep + top-20 sem | ❌ 4중 이탈 |
| `zep_graph` | 63.8 / **71.2** | **폐기본** (2025-01) | gpt-4o-mini / gpt-4o | 미공개 | 미공개 | ❌ 불명 |
| `mem0` | **94.4** | cleaned | **gpt-5** | **gpt-5** | **top-200** | ❌ 5중 이탈 |
| `memmachine` | **93.0** | cleaned | **gpt-5-mini** | **gpt-4o-mini** | **k=100** | ❌ judge 이탈 |

**같은 이름의 "LongMemEval 정확도"가 릴리스·reader·judge·검색예산 네 축에서 전부 다르다.** 공식 `print_qa_metrics.py`는 이 넷 중 어느 것도 통과시키지 않는다(judge assert 하나만으로도 전부 거부).

### 5.2 시스템별 상세

**Nemori** (`evaluation/longmemeval/`)
- 데이터 안내가 **폐기 repo 경로**를 가리키고 그 URL은 **404**다. 위치와 검증 방법을 명시한다(최상위
  README가 아니라서 찾기 어렵다):
  - 파일: **`evaluation/README.md:55`** (클론 SHA `d2a6dff6`, 2026-04-16), 원문
    `wget https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s.json`
  - 검증: `curl -sIL -o /dev/null -w "%{http_code}"` → `…/longmemeval_s.json` **404**,
    확장자 없는 `…/longmemeval_s` **200**. 폐기 repo의 파일명은 `longmemeval_{s,m,oracle}`로
    **확장자가 없다**(HF API `siblings` 조회로 확인).
  - **라이브 확인**: `raw.githubusercontent.com/nemori-ai/nemori/main/evaluation/README.md`를
    2026-08-17에 받아도 **같은 줄이 그대로**다 — 우리 클론이 낡아서가 아니다.
  - 덤: 최상위 `README.md:175`는 `evaluation/longmemeval/readme.md`를 가리키는데 **그 파일은 없다**
    (해당 디렉터리엔 `add.py`/`config.json`/`evals.py`/`search.py`뿐).
- `evaluation/longmemeval/evals.py` judge 이탈 4건 (전부 file:line 확인):
  ① **모델이 핀이 아니고, 리포 안에서 세 곳이 서로 다르다** — 클래스 기본 `model="gpt-4o-mini"`(`:32`),
     argparse 기본 `default="gpt-4.1-mini"`(`:326`), 그 help 문구는 `"(default: gpt-4o-mini)"`(`:327`).
  ② **abstention 분기가 아예 없다** — 프롬프트 상수는 `TEMPORAL_REASONING`(`:45`),
     `KNOWLEDGE_UPDATE`(`:59`), `SINGLE_SESSION_PREFERENCE`(`:73`), `DEFAULT`(`:87`) **넷뿐**이고
     `:119-133`의 분기는 abstention을 보지 않는다. 30문항이 DEFAULT로 떨어져
     "explanation을 정답으로 담았는가"를 묻게 된다. (Mem0와 달리 인라인 규칙조차 없다.)
  ③ upstream에 없는 `system_prompt` 추가(`:115`).
  ④ 자유 텍스트+substring이 아니라 `beta.chat.completions.parse(..., response_format=Grade)`(`:138-144`).
- 보고는 `overall_accuracy` + `accuracy_by_type`(`:265-266`)이고 **task-averaged는 없다**.
- config: `gpt-4.1-mini`, `gemini-embedding-001`, episodes top-10 / semantic top-20.

**Zep / Graphiti** (arXiv:2501.13956)
- `_s` 기준 baseline 55.4(mini) / 60.2(4o) → Zep 63.8 / 71.2. 레이턴시 31.3s→3.20s, 28.9s→2.58s.
- 타입별 상대 개선: preference +77.7%, TR +48.2%, MS +16.7%, SSU +14.1%, **KU −3.36%, SSA −9.06%**.
  **[실측 2026-08-18]** 최대 항목인 preference의 정체가 우리 `_s` 측정으로 설명된다: full-context
  baseline의 preference가 **3.33**이다(§9.3a). 그 위에서의 +77.7%는 preference를 잘해서 나오는
  수치가 아니라 **해당 세션을 리더 앞에 놓기만 하면 나오는 수치**다. 우리 검색 arm은 같은 조건에서
  3.33 → **53.33**을 write LLM 콜 0개로 얻는다. **두 타입에서 메모리가 손해**다 — 그리고 SSA 손해는 §2.3의 user-only 인덱싱 문제와 같은 뿌리일 가능성이 높다(어시스턴트 발화를 저장하지 않는 설계).
- 저자 스스로 "노트북(보스턴)↔AWS us-west-2 왕복이라 Zep 쪽에만 네트워크 레이턴시가 더해졌다"고 명시.
- 리포의 `tests/evals`는 `longmemeval_oracle`을 gpt-4.1-mini로 쓰는 **그래프 구축 평가**이고 논문 수치 경로가 아니다.

**Mem0** (docs + `mem0-memory-benchmarks/benchmarks/longmemeval/`)
- **94.4** / 문항당 평균 6,787 토큰. 타입별: SSU 98.6 · SSA 98.2 · SSP 96.7 · KU 93.6 · TR 97.0 · MS 88.0.
- 하네스: `longmemeval_s_cleaned.json` 자동 다운로드 + 500문항 검증, `--answerer-model` 기본 **gpt-5**, `--judge-model` 기본 **gpt-5**, `--top-k` 기본 **200**.
- **judge 프롬프트가 타입별 분기 없이 단일 템플릿**이다: `get_judge_prompt`(prompts.py)이 `question_type`과 `question_id`를 인자로 받고 **둘 다 쓰지 않으며**, `JUDGE_PROMPT`에 넣는 것은 question/answer/response/question_date뿐이다.
  ⚠️ **정정(2026-08-17 1차 보고)**: 이를 "abstention 처리가 없다"고 적었으나 **틀렸다.** 단일 템플릿 **안에** abstention 규칙이 인라인으로 들어 있다(prompts.py:323-326). 다만 그 규칙이 upstream보다 관대하다 — *"The key test: does the response REFUSE to answer the question? If yes, it matches an abstention ground truth, period."* "abstention 분기가 아예 없다"가 맞는 것은 **Nemori**다.
- 그리고 그 프롬프트가 관대함을 명시적으로 주문한다(prompts.py:269, 원문):
  > *"IMPORTANT BIAS CHECK: You have a tendency to say 'no' too quickly. Before concluding 'no', you MUST verify the answer is truly wrong, not just differently worded. When in doubt, lean toward 'yes'."*
- abstention 판정도 `question_id.endswith("_abs")`로 upstream의 substring과 다르다(현 데이터에선 잠복).
- 자체 문서에 두 개의 중요한 자백: **"Scores reflect Mem0's managed platform, which includes proprietary optimizations not available in the open-source SDK"**, **"Scores carry a ±1 point confidence interval due to judge inconsistency."**
- 그리고 벤치 자체에 대한 자백: *"Some benchmarks today, particularly smaller ones like LoCoMo and LongMemEval, can be materially improved by aggressive retrieval strategies, larger context windows, or frontier models. That does not necessarily mean the underlying memory system has gotten better."*
- ⚠️ 이 타입별 수치는 우리 1차 조사본이 기록한 값(SSU 94.3 / SSP 46.4 / KU 98.2 / TR 76.7 / MS 96.7)과 **다르다**. 벤더 수치는 버전 표기 없이 갱신된다.

**MemMachine** (arXiv:2604.04853, `evaluation/episodic_memory/`)
- `longmemeval_s_cleaned.json` 명시, **93.0** (구성 C15, k=100, answer LLM **gpt-5-mini**), judge **gpt-4o-mini**.
- 타입별: SSU 1.000 · SSA .982 · SSP .933 · TR .917 · KU .949 · MS .872.
- **6축 ablation 기여도**: 검색 깊이 k 20→30 **+4.2%** · 컨텍스트 포맷 **+2.0%** · 검색 프롬프트 개선 **+1.8%** · user-query bias 보정 **+1.4%** · **문장 청킹(ingest) +0.8%** · **모델 교체 GPT-5→GPT-5-mini +2.6%**.
  ⇒ **읽기·검색 하이퍼파라미터가 ingest 개선을 5배 이상 압도하고, 모델을 바꾼 것만으로 ingest 개선의 3배가 움직인다.**
- 코드: `get_anscheck_prompt`을 공식과 동일하게 전사(abstention 포함), 다만 `evaluate_responses`의 기본값이 `exclude_abstention=True`이고 호출부만 `False`로 넘긴다. 점수는 `llm_score` 단순 평균(overall만).
- 저자 자백: "cross-system comparisons mix re-run results and published numbers, which may differ in preprocessing, prompt settings, or infrastructure."

### 5.3 참고 — 최신 발표치 (전부 자체 보고, 우리가 검증하지 않음)

| 시스템 | 점수 | 조건 |
|---|---|---|
| Mastra Observational Memory | full-ctx gpt-4o **60.20** → OM gpt-4o **84.23** → gemini-3-pro 93.27 → **gpt-5-mini 94.87** | 평균 컨텍스트 ~30K. **자체 oracle 측정 82.4** (논문은 .870/.924) |
| OMEGA | 95.4 (466/500) | GPT-4.1 |
| Supermemory | ~95 | Recall@15 + aggregation |
| EverMemOS | 83.0 | `_s` |
| MemForest | 79.8 | `_s`, 30B reader |
| TiMem | 76.88(4o-mini) / 78.96(4o) | `_s` |

Mastra 사례가 결정적이다: **메모리 시스템을 고정한 채 reader만 gpt-4o→gpt-5-mini로 바꿔 84.23 → 94.87 (+10.6pp).**
그리고 **자기 시스템이 oracle을 이겼다**(84.23 vs 82.4)고 보고하는데, evidence만 받은 상한을 넘었다는 건 시스템이 좋다는 신호이기 전에 **oracle 읽기 세팅이 약하다**는 신호다(논문 Figure 6이 읽기 전략만으로 10pp 움직인다고 이미 말한다). 실제로 그들의 oracle 82.4는 논문의 .870(direct)/.924(CoN)보다 4.6–10pp 낮다.

---

## 6. 이 지표의 문제 — P1~P15

**P1 · 측정 정의가 잠기지 않았다.** 정확도가 셋이고(LME-A13) 타입 불균형 4.4배라 서로 다르다. **어느 쪽인지 밝힌 발표 수치를 하나도 찾지 못했다.**

**P2 · abstention은 별도 타입이 아니라 겹친 절단면이다**(LME-A14).
⚠️ **정정**: 1차 보고와 원장 C-4는 이를 "두 헤드라인 **안에서** 이중 계상된다"고 적었는데, **코드를 다시
읽으면 그렇지 않다.** `all_acc`는 6개 타입 버킷에서만 누적되고(`print_qa_metrics.py:28`),
`task_acc`도 6개 타입 평균의 평균이다(:29). **한 행이 한 헤드라인 안에서 두 번 세어지지는 않는다.**
실제 위험은 다른 데 있다: ① abstention 30문항은 **빠지지 않고** 두 헤드라인에 그대로 들어가므로,
"모른다"를 남발해 기권을 따먹는 시스템이 **일반 타입 점수에서 이득을 본다**. ② 별도 줄로 출력되는
탓에 독자는 7번째 타입(=헤드라인에서 제외된 것)으로 오독하기 쉽다. 원장 C-4의 표현은 교정 대상이다.

**P3 · judge가 아무도 안 지키는 핀이다.** 공식은 `assert`로 강제(LME-A15)하는데 실제로는 Nemori=4o-mini/4.1-mini, Mem0=**gpt-5**, MemMachine=`gpt-4o-mini`. 더 나쁜 것은 **judge 프롬프트까지 바뀐다**는 점이다 — Mem0는 타입별 분기를 없애고 "애매하면 yes로 기울여라"를 명시했다. 이건 프로토콜 디테일이 아니라 **채점 기준의 변경**이다. 논문 자신도 preference/abstention에서 인간 합치율이 .90이고(Table 6), Mem0는 judge 불일치로 **±1점**의 불확실성을 인정한다.
**[실측 2026-08-17, docs/20]** 우리 4 arm을 mini judge로 재채점하면 전부 **−1.2~−2.0pp**(일치율 96.2~98.0%)로
싼 judge가 더 엄격하다. 즉 **±1점은 낙관적이고 2점이 실측치**다. 그런데 이동이 계통적이라 **스프레드는
12.57→12.63 / 15.40→15.00, arm 순위는 동일**하다. ⇒ **레벨은 judge 의존, 대조는 judge 불변.**
이 벤치를 순위표로 읽으면 틀리고 메커니즘 분리로 읽으면 값을 한다는 §6 종합의 정량적 근거다.
재채점은 판정 콜만 사면 되므로 **arm당 2센트**다(`scripts/repro/lme_rejudge.py`).

**P4 · 릴리스가 잠기지 않았다.** `_s`에서 filler 1,243개가 삭제됐고(§3.2) distractor만 빠졌으므로 cleaned 점수는 구조적으로 높다. Nemori(폐기본)와 Mem0/MemMachine(cleaned)를 같은 표에 놓는 것은 정의상 틀렸다. **oracle만 통시 비교가 가능하다.**

**P5 · reader 모델이 메모리를 압도한다.** Mastra: 메모리 고정, reader만 교체해 **+10.6pp**. MemMachine: ablation에서 **모델 교체 +2.6%p vs ingest 개선 전체 +0.8%p**. 리더보드 상단 90%대의 상당 부분이 메모리가 아니라 모델의 몫이고, 그 기여를 분리해 보고하는 시스템은 MemMachine 하나뿐이다.

**P6 · 검색 예산이 잠기지 않았다.** 논문 실험은 top-5/top-10, Mem0는 **top-200**, MemMachine은 k=100. Mem0의 ablation은 아니지만 MemMachine의 ablation이 **k를 20→30으로 올린 것만으로 +4.2%p**라고 알려준다. top-200이면 그 축이 얼마나 더 남았는지는 아무도 보고하지 않는다.

**P7 · 검색 지표는 SSA 타입을 91% 삭제한 채 계산된다**(LME-A1/A2). 논문의 모든 recall/NDCG는 419문항 위의 값이고, README는 abstention 30건 제외만 밝힌다. 그런데 SSA는 이 벤치가 **"기존 벤치와 다르다"고 내세운 능력**이다(LoCoMo가 어시스턴트 발화 회상을 평가하지 않는다는 게 논문 서론의 차별점 주장이다). **차별점으로 내세운 축이 자기 검색 평가에서 빠져 있다.**

**P8 · 구성 타당도 — "메모리"가 아니라 "긴 컨텍스트 독해"일 수 있다.** `_s`는 실측 113,840토큰으로 128K 창 **안에** 들어가고, 그건 우연이 아니라 `enforce_json_length=115000`이라는 **설계 제약**이다(§2.7). 창이 200K~1M인 2026년 모델에게 `_s`는 "메모리가 필요한 문제"가 아니다. Mastra가 oracle을 넘겼다고 보고하는 것도 같은 방향의 증거다.

**P9 · 해상도가 낮다.** 500문항 이진 채점의 McNemar 분간 한계는 실측 불일치율 기준 **5.3–8.0pp**다. 우리 LoCoMo 1,540문항으로도 안 갈리는 인접 격차를 500문항이 가릴 수 없다. **새 벤치를 더해 순위를 확정한다는 설계는 방향이 반대다.**

**P10 · 싸게 줄일 방법이 없다.** 세션을 자르면 evidence를 잃고(§3.3), 인스턴스를 줄이면 공식 집계기가 `nan`을 낸다(LME-A16). 전량 아니면 비교 불가라는 구조가 재현 비용을 고정시킨다 — `_s` full-context 1 arm이 55.9M 입력 토큰이다.

**P11 · 재현 아티팩트가 실행되지 않거나 침묵한다.** 시간 인식 질의 확장 스크립트는 **첫 줄에서 죽고**(LME-A10), 그 지표 출력은 이름이 사라진 set이며(LME-A12), README의 집계 명령은 인자 수가 틀렸고(LME-A17), 부분 실행은 완주처럼 집계된다(LME-A18). 논문 Table 4를 재현하려면 코드를 고쳐야 한다.

**P12 · 포화, 그리고 리더보드 없는 리더보드.** 상단이 93~95%에 몰려 있다.
그리고 V1에는 **공식 리더보드가 없다** — 2025-09의 ISS #14 *"Is there a leaderboard…?"*는 답이 없고,
그 자리를 **이슈 트래커가 대신하고 있다.** `[Implementation Sharing]` 형식으로 자기 점수를 올리는
이슈가 최소 12건이고(#28 ~99% · #29 96.2% · #31 92.3% R@5 · #32 **99.9% R@5** · #34 93.6% ·
#35 96.6% R@5 · #42 · #43 89.0% · #44 96.6% Any@5 · #46 99.4% R@10 · #48 97% R@5 · #49 92.0% ·
#51 · #53 82.2%), **검증 절차가 없다.** ISS #41은 자기 시스템이
*"achieves 100% accuracy on all 499 remaining valid LME-S questions"*라고 적는다.

**P13 · 데이터 결함이 미해결로 쌓인다.** 트래커의 데이터 품질 신고 **15건 중 12건이 open**이고,
우리가 검증한 3건은 전부 현재 cleaned 릴리스에 살아 있다(§3.4): 중복 filler 13건, gold 산술 오류
(11.57주를 15로), turn-level gold가 답을 담지 않는 사례. 상단이 93~95%인 벤치에서 이 정도 결함률은
**측정 상한 자체를 불확실하게** 만든다 — 남은 5~7%p가 시스템 실패인지 라벨 오류인지 구분되지 않는다.

**P14 · 코드가 사실상 동결됐다.** `git log` 기준 **마지막 코드 변경은 2025-04-26**이고 그 뒤 커밋
(2025-09~2026-05)은 전부 README 수정이다. 저자가 *"a fallback case … should be added"*라고 인정한
확장 드롭 버그(ISS #9, 2025-08-14)는 **12개월째** 그대로이고, 같은 결함 둘을 고치는 PR #47은
**2.5개월째 미머지**다. "유지보수되는 벤치마크"라는 전제는 코드에 대해서는 성립하지 않는다.

**P15 · 발견의 출처를 밝혀야 한다.** 위 P1~P14 중 A1/A2·A8·A15/A17·CoN 정의는 **우리가 처음 찾은 것이
아니다**(§2.6b). 이미 신고됐고, 일부는 저자가 인정했고, 대부분 안 고쳐졌다. 이 사실 자체가 P14의
증거이며, 동시에 우리 보고의 신뢰도를 높인다 — 독립적으로 같은 지점에 도달했고 제3자 로그(ISS #16)가
우리 실측 51을 재현한다. 남은 신호는 총점이 아니라 **특정 타입**에 있다 — Mem0조차 MS 88.0 / KU 93.6이 최저이고, KU가 낮은 이유를 스스로 "ADD-only 구조라 옛 사실이 덮이지 않고 남는다"고 밝힌다. 우리 원장의 write-path 관심사와 정확히 겹치는 지점이다.

> **종합**: LongMemEval의 설계(4 control point, KU/ABS 도입, 길이 확장성)는 여전히 이 분야에서 가장 잘 정돈된 프레임이다. 무너진 것은 **비교 가능성**이다 — 릴리스·judge·reader·검색예산 네 축이 전부 자유변수인데 아무도 고정하지 않는다. 그래서 이 벤치의 숫자를 **랭킹**으로 읽으면 거의 항상 틀리고, **메커니즘 분리**(같은 조건에서 한 축만 바꿔 부호를 보는 것)로 읽으면 여전히 값을 한다.

---

## 7. LongMemEval-V2 — 저자가 스스로 고친 것들

arXiv:2605.12493 (2026-05), 별개 벤치마크. 451문항, 최대 500 trajectory/haystack, **최대 115M 토큰**, web/enterprise 두 도메인, small/medium 두 티어. 대화 기억 → **에이전트 경험 기억**으로 전환했다.

5축: static state recall / dynamic state tracking / workflow knowledge / environment gotchas / **premise awareness**(= V1 abstention의 후신).

**V1 비판에 대한 직접적 대응이 코드에 있다:**

| V1의 문제 | V2의 조치 |
|---|---|
| P3 judge 자유변수 | 채점 대부분이 **결정론적 매처**(`norm_phrase_set_match`, `mc_choice_match`, `extract_boxed_answer`)로 바뀌고 LLM judge는 abstention/gotchas 등 일부에만 남는다 |
| abstention 채점이 관대함 (V1: "정보가 불완전하다고만 말해도 정답") | **엄격해짐**: *"If the model's final answer is just UNKNOWN / cannot determine **without identifying the flaw**, grade 0"*, 모순된 답도 0 |
| P5 reader 모델이 지배 | 논문 실험이 **고정 reader(Qwen3.5-9B)** + 고정 임베더(Qwen3-Embedding-8B) |
| P6 검색 예산 자유 | 하네스가 `--memory-context-max-tokens`를 **강제** |
| `has_answer` 등 골드 누출 | 백엔드는 `query` 시 **벤치 메타데이터를 전혀 못 받는다** — question id·type·gold answer·평가 설정을 하네스가 비공개로 유지하고, 넘겨주는 것은 run-local 난수 id뿐 |
| 정확도만 보는 리더보드 | **LAFS** — (정확도, 레이턴시) Pareto frontier 기반 점수. 고정 참조 frontier 대비 **gain**으로 제출을 평가 |

LAFS 참조 frontier (논문 본표 하드코딩, `leaderboard/compute_lafs.py`):

| 티어 | RAG(slice+notes) | Codex | AgentRunbook-R | AgentRunbook-C |
|---|---|---|---|---|
| small | 51.0 @ 0.2s | 69.9 @ 177.2s | 58.6 @ 26.9s | **74.9 @ 108.3s** |
| medium | 45.9 @ 0.3s | 68.7 @ 185.8s | 57.0 @ 25.8s | **70.1 @ 139.9s** |

`T_MIN=1.0s`, `T_MAX=200.0s`. judge 기본값은 `gpt-5.2`(reasoning `medium`).

**시사점**: V2는 우리가 §6에서 지적한 P3·P5·P6와 누출 문제를 **저자가 동의하고 설계로 막은** 결과물이다. 즉 §6의 비판은 사후 트집이 아니라 벤치 저자와 같은 진단이다. 다만 V2는 웹 에이전트 궤적 도메인이라 **대화형 메모리 시스템(우리 organizer 대부분)을 그대로 얹을 수 없다**. V1은 여전히 대화형 메모리의 사실상 표준으로 남아 있다.

### 7.1 baseline 3종 — 특히 AgentRunbook이 무엇인가

V2의 haystack은 대화가 아니라 **웹 에이전트 궤적**이라 "메모리"의 의미가 달라진다. 리포가 구현한
백엔드는 `no_retrieval` / `rag` / `agentrunbook_r` / `codex` / `agentrunbook_c` / `agentrunbook_c_v2`다
(`memory_modules/*.py`의 `@register_memory`).

**AgentRunbook-R (Retrieval)** — `memory_modules/agentrunbook_r.py` (3,175줄).
insert 시 LLM이 궤적 하나를 **노트 두 개**로 증류한다. 결정적으로 **다운스트림 질문을 모르는 상태**에서
쓴다(`support.py`의 `NOTE_GENERATION_SYSTEM_PROMPT`: *"Assume these notes will later be retrieved for
unknown future questions. You do not know the downstream question in advance."*):

- `procedure_note` — 재사용 가능한 **핵심 워크플로**, 4~8 bullet
- `hint_note` — 오래 가는 고가치 사실·함정, 6~12 bullet

프롬프트 규칙이 엄격하다 — *"관측된 실행으로 직접 뒷받침되지 않는 사실은 절대 쓰지 말 것"*,
*"라벨·버튼·메뉴·모듈은 원문 문자열 그대로(paraphrase 금지)"*, *"실패한 실행을 성공한 것처럼 쓰지 말 것,
대신 hint_note에 무엇이 걸림돌이었는지 남길 것"*. query 시엔 structured query bundle 생성 →
임베딩 검색 → 선택적 LLM 리랭크. 컨트롤러 기본값은 `Qwen/Qwen3.5-9B`.

**AgentRunbook-C (Codex/agentic)** — `memory_modules/agentrunbook_c.py`.
**임베딩 인덱스가 없다.** 코딩 에이전트(Codex 바이너리)를 `INSTRUCTION.md` + `question.json` +
`trajectories/` + 조사 스크립트(`render_trajectory_summary.py`, `inspect_trajectory.py`)가 든 디렉터리에
떨어뜨리고 **파일시스템을 직접 탐색**하게 한다(`rg`, `find` 필요). query 프롬프트가
*"you must explore trajectories/ before returning your final result"*라고 못박는다.

**Codex (vanilla)** — 같은 에이전트, runbook 스캐폴딩 없음.

성적(LAFS 참조 frontier, small 티어):

| 방법 | 정확도 | 질의 레이턴시 | 성격 |
|---|---|---|---|
| RAG (query→slice+notes) | 51.0 | **0.2s** | 임베딩 검색 |
| AgentRunbook-R | 58.6 | 26.9s | write-time 증류 + 임베딩 검색 + 리랭크 |
| Codex (vanilla) | 69.9 | 177.2s | agentic 파일 탐색 |
| **AgentRunbook-C** | **74.9** | 108.3s | write-time 증류 + agentic 파일 탐색 |

> **이 표가 §9 주장 C의 세 번째 독립 증거다.** V2에서 이기는 "메모리 시스템"은 인덱스를 잘 만든 쪽이
> 아니라 **읽을 때 잘 뒤지는 쪽**이다 — 임베딩 검색 대비 **+16.3pp**, 대신 레이턴시 **400배**.
> LAFS가 존재하는 이유가 정확히 이것이다: 정확도만 보면 제일 느린 agentic 방법이 항상 이긴다.

**우리 리포와의 연결**: AgentRunbook-R의 write 경로(질문을 모른 채 LLM이 procedure+hint로 증류)는
우리가 Track 5에서 구현·측정한 **ACE playbook과 구조적으로 같은 물건**이다. 우리 결론은 "ACE playbook이
우리 dedup에서도 상류 기본값에서도 무학습과 분리되지 않았다"였고, V2에서는 같은 형태가 baseline 2위다.
**같은 메커니즘이 도메인에 따라 값을 하기도 안 하기도 한다** — 대조로 쓸 수 있는 발견이다.

---

## 8. 실행 설계와 견적

### 8.1 지출 전에 이미 확정된 것 ($0)

- full-context 경로는 upstream과 **바이트 동일**(§4.1) — 충실성은 증명 끝
- 집계기는 공식과 **0.000000pp** 일치(§4.2)
- oracle은 논문과 **같은 바이트** — 유일하게 정당한 논문 대조점(§3.2)

### 8.2 시세 (2026-07-30 인하 반영, per 1M in/out)

| 모델 | in | out | 비고 |
|---|---|---|---|
| gpt-4o-mini | $0.15 | $0.60 | 레지스트리 등록됨. Zep/Nemori 기준선 모델 |
| gpt-4o-2024-08-06 | $2.50 | $10.00 | **judge 핀**. 논문 reader |
| gpt-5.6-luna | $0.20 | $1.20 | 등록됨 |
| gpt-5.6-terra | $2.00 | $12.00 | **미등록** |
| gpt-5.6-sol | $5.00 | $30.00 | **미등록** |

### 8.3 arm당 비용 (실측 토큰 기준, judge $0.60/arm 포함)

| reader | oracle (3.06M in) | `_s` (55.9M in) |
|---|---|---|
| gpt-4o-mini | $1.17 | **$9.11** |
| gpt-5.6-luna | $1.44 | **$12.02** |
| gpt-4o-2024-08-06 | $10.15 | $142.4 |
| gpt-5.6-terra | $8.96 | $114.8 |
| gpt-5.6-sol | $21.0 | $285.4 |

### 8.4 검증이 지목한 실험 (cap $30)

| # | 런 | 견적 | 실측 | 결과 |
|---|---|---|---|---|
| R1 | oracle × gpt-4o-mini × CoN | $1.17 | **$1.05** | task-avg **83.89** / overall 83.60 |
| R1′ | oracle × gpt-4o-mini × direct | ~$1.1 | **$0.76** | task-avg **79.57** / overall 79.20 |
| R2 | oracle × luna × CoN | $1.44 | **$1.02** | task-avg **92.14** / overall 94.60 |
| R2′ | oracle × luna × direct | ~$1.4 | **$0.91** | task-avg **89.96** / overall 91.40 |
| **S1** | `_s` × gpt-4o-mini × CoN (full-context) | $9.11 | **$9.07** | task-avg **58.40** / overall 60.40 |
| **S3** | `_s` × gpt-4o-mini × CoN, **검색 top-50** | — | **$2.62** | task-avg **80.81** / overall **81.60** |
| R4 | `_s` × luna × CoN | $12.02 | — | 미승인 (**P8 직접 시험**이 여기 남아 있다) |

oracle 4 arm은 2026-08-17에 완주했고(500/500 × 4, judge 핀 `gpt-4o-2024-08-06`, 합계 **$3.74**,
wall-clock 약 13분), `_s` 2 arm은 2026-08-18에 완주했다(합계 **$11.69**, S1 38분 / S3 2.7시간 —
S3는 임베딩 246,750콜이 시간을 지배한다). 스모크 실측이 dry-run 견적의 1.20배였고 본런은 견적 이하로 들어왔다 —
견적이 상한(완성 토큰을 `max_tokens`로 계산)이기 때문이다. 판정과 페어링 통계는
[`docs/20`](../20-lme-reading.md)이 정본이고, `direct` 조건이 붙었으므로 §1.3의 **CoN 부호**도
oracle 길이에서는 답이 나왔다(양수, 두 리더 모두). 부호 **역전**은 `_s`가 있어야 보이므로 미측정.
`_s`를 돌릴 수 있는 프런티어 모델은 캡 $30 안에서 **luna뿐**이다(terra $114.8, sol $285.4).

### 8.5 드라이버 필수 요건

1. oracle 렌더 전 **날짜 정렬** (§4.1 C)
2. **세션 캡 금지**, topk 전량 (§3.3)
3. **n==500 완주 검증** (LME-A18 대응)
4. 인스턴스당 **fresh memory** (haystack 혼입 방지)
5. `--dry-run` 콜 원장 + `--max-spend-usd` + 캠페인 아티팩트 캡처
6. **`_s`/oracle에서 `max_history_tokens` 금지** — upstream 126,200을 넘기면 chars/4 환산(D3) 탓에
   중앙값 인스턴스부터 절단돼 **이탈을 막는 게 아니라 만든다**. upstream의 절단은 no-op(0/500)이므로
   무캡이 충실이다 (§4.3 D3)

---

## 9. 발표 주장과 근거 사슬 (2026-08-17 확정)

> **리뷰어를 위한 안내.** 이 절의 모든 셀은 셋 중 하나로 태그돼 있다 — **[실측]** 우리가 이 리포에서
> 잰 것, **[연역]** 코드/데이터에서 논리적으로 따라 나오지만 실행으로 확인하지 않은 것,
> **[인용]** 남의 자체 보고. 이 구분을 흐리는 문장이 있으면 그게 결함이다.
> 가장 공격받기 쉬운 지점을 §9.4에 먼저 적어 두었다.
>
> ⚠️ **번호 주의**: §0의 `C1~C6`은 1차 조사본의 **교정 항목** 번호이고, 이 절의 `C1~C6`은
> **근거 사슬** 번호다. 서로 다른 네임스페이스이며 같은 번호끼리 아무 관계도 없다.

### 9.1 주장 (한 문장, 2026-08-18 갱신)

> **점수는 답변자에게 무엇이 도달했는가에서 난다. 그리고 그것을 사는 데 write 지출이 필요했던
> 사례를 우리는 아직 찾지 못했다 — `_s`에서 검색은 21.20pp를 사는데, 그 검색은 ingest LLM 콜
> 0개로 얻어진다. 그래서 LongMemEval 순위는 메모리 설계가 아니라 read 경로를 정렬한다 —
> V2가 reader를 고정하고 LAFS를 도입한 이유가 이것이다.**

**전반부는 이제 양의 주장이고, 후반부는 여전히 부정형이다.** 이 구분이 이 절 전체의 설계다:
"read 경로가 점수를 정한다"는 페어링 CI가 0을 제외하는 실측 두 건(C4 12.57~15.40pp, C6 +21.20pp)이
지탱한다. "write 지출은 그걸 사지 않는다"는 **우리가 잰 범위 안에서 반례가 없다**는 형태로만 성립하고,
그 범위는 §9.4가 명시한다.

⚠️ **2026-08-17 판의 첫 문장은 "write 지출이 점수를 산다는 증거가 없다"였다.** 그 문장 자체는 지금도
참이지만 **앞에 세우면 오독을 부른다** — 실제로 이 캠페인 안에서 "그럼 메모리는 무용한가"로 읽혔다.
C6이 그 오독의 반례다: 메모리를 **검색으로** 붙이면 21점이 붙는다. 무용한 것으로 밝혀진 적이 없는 것은
**검색**이고, 값을 아직 못 증명한 것은 **write에 LLM을 쓰는 부분**이다.

### 9.2 근거 사슬

| | 주장 | 근거 | 태그 |
|---|---|---|---|
| **C1** | write 지출이 점수를 사지 않는다 | LoCoMo 5-arm(`docs/18` :27-33): **ρ(write 콜, J) = −0.60**, ρ(ingest $, J) = −0.60. 극단값 — Zep은 Nemori A보다 write 콜 **7.7배**(27,449 vs 3,579)를 쓰고 J가 **24.87pp 낮다** | **[실측]** |
| **C2** | 점수는 read 경로가 정한다 | cross-arm ρ(read 컨텍스트, J) = **+0.90**. 더 강한 것은 **arm 내부** 두 건(`docs/18` :159-165, :34-40): A-Mem read-path 질의 재작성 **+5.26 J**, 링크확장으로 컨텍스트 **+74% → +1.36 J** | **[실측]** |
| **C3** | 남의 실측도 같은 비율 | MemMachine ablation(LongMemEval): 검색깊이 +4.2 · 포맷 +2.0 · 검색프롬프트 +1.8 · bias보정 +1.4 vs **ingest 개선 전체 +0.8**, 모델 교체 +2.6 | **[인용]** |
| **C3′** | V2 baseline도 같은 방향 | 파일 탐색(AgentRunbook-C 74.9) > 임베딩 검색(R 58.6), 레이턴시 400배 (§7.1) | **[인용]**(벤치 저자) |
| **C4** | 메모리를 상수로 고정해도 점수가 움직인다 | oracle 4 arm({mini, luna} × {con, direct}), 같은 500문항·같은 바이트·판정 핀 동일: **task-averaged 스프레드 12.57pp / overall 15.40pp**(79.57–92.14 / 79.20–94.60). 리더 교체가 +11~12pp(overall), 읽기 방식이 +3~4pp — 둘 다 페어링 CI가 0을 제외한다. Zep이 자기 메모리 시스템 전체로 주장하는 폭이 +11.0pp(4o) / +8.4pp(mini)다. 정본: [`docs/20`](../20-lme-reading.md) | **[실측]** |
| **C6** | **read 경로는 점수를 사고, 그 값은 write 지출 없이 얻어진다** | `_s` 같은 500문항, 같은 리더/프롬프트/judge: full-context **60.40** → **검색 top-50 81.60**, 페어링 **+21.20pp [+16.60,+25.60]**, McNemar 132/26, p<1e-16. 이 arm의 ingest LLM 콜은 **0**(passthrough, 임베딩 $1.12)이고 비용은 full-context의 **1/3**, 컨텍스트는 **1/9**. SSU·SSA·abstention은 oracle 수치와 **정확히 동일**하게 복귀 | **[실측]** |
| **C5** | 필드가 같은 진단에 도달 | V2가 reader·예산 고정, 결정론 채점, 메타데이터 비공개, LAFS (§7) | **[연역]**(설계 증거) |

### 9.3 남은 실험 하나 — C4 ✅ **완료 (2026-08-17)**

C1~C3는 전부 **arm이 서로 다른** 비교라 "메모리가 달라서 그런 것 아니냐"는 반론이 가능하다.
C4가 그 반론을 원천 차단한다:

> **`longmemeval_oracle`은 evidence 세션만 담긴 데이터다. 검색이 완벽한 조건, 즉 메모리가 상수로
> 고정된 조건이다. 거기서 관측되는 변동은 정의상 메모리 탓이 아니다.**

설계: `oracle × {gpt-4o-mini, gpt-5.6-luna} × {CoN, direct}` = **4런**, 견적 $5 / **실측 $3.74**,
workers=8로 **13분**. 판정 기준은 실행 전에 [`docs/20`](../20-lme-reading.md)에 사전등록했다
(≥5pp 성립 / 2–5pp 약한 성립 / <2pp 불성립).

**결과: 스프레드 task-averaged 12.57pp, overall 15.40pp — 성립.** 분해하면 리더 교체가 +11~12pp,
읽기 방식이 +3~4pp이고 페어링 CI는 하나(luna의 task-averaged con−direct, +2.18 [−0.99, +5.57])를
빼고 전부 0을 제외한다. 타입별로는 두 single-session 회상 타입이 모든 arm에서 포화(97~100)라
움직임이 **multi-session(+20.3pp)·temporal-reasoning(+24.8pp)·preference(+13.3pp)** 에 몰린다 —
검색이 완벽할 때 남는 것은 **찾기가 아니라 찾아온 것을 다루기**이고, 그 부분은 메모리 시스템의
소유가 아니다.

### 9.3a `_s` 두 arm — C6 (2026-08-18, $11.69)

oracle은 **천장 조건**이라 거기서 나온 것은 그대로 일반화되지 않는다. 그 경계를 우리 손으로 확인했다.

| `_s` × mini × con | task-avg | overall | 비용 | 컨텍스트(중앙값) |
|---|---|---|---|---|
| full context | 58.40 | 60.40 | $9.07 | 517,430자 |
| **검색 top-50** | **80.81** | **81.60** | **$2.62** | 55,108자 |
| (oracle 천장) | 83.89 | 83.60 | $1.05 | ~27,000자 |

1. **전문 읽기의 낙폭은 논문과 정합**: oracle → `_s`가 −23.20pp(−27.8% 상대), 논문 GPT-4o는
   .924→.640(−30.7% 상대). **다른 리더로 재현된 셈**이고, `_s`는 128K 창에 들어가도록 패킹된
   데이터라(§2.7) **절단 효과가 아니다** — 우리는 캡을 걸지 않는다.
2. **한 타입은 나빠지는 게 아니라 무너진다**: single-session-preference **63.33 → 3.33**.
   육안 확인 결과 모델이 선호를 잘못 쓰는 게 아니라 **개인화를 멈추고** 히스토리가 없는 것처럼
   일반 답을 한다. **113K에서 먼저 죽는 것은 사실 회상(SSU 84.29·SSA 92.86)이 아니라
   리더 자신의 컨텍스트를 답에 반영하는 능력이다.**
   ⇒ Zep이 최대 상대개선을 보고하는 타입이 정확히 preference(**+77.7%**, §5.2)인데,
   **그 baseline이 3%다.** preference를 잘해서 나오는 수치가 아니라 해당 세션을 리더 앞에
   놓기만 하면 나오는 수치다.
3. **검색이 그 낙폭의 91%를 되찾는다** — write LLM 콜 0으로. 이것이 C6이다.

**oracle만 봤다면 정반대 결론이 나왔다.** oracle에서 검색을 얹으면 **−3.00pp**(천장 조건의 세금)이고
`_s`에서 같은 메커니즘이 **+21.20pp**다. **부호가 "검색이 필요한가"에 따라 뒤집힌다.** 이 사실 자체가
§9.4에 새 약점으로 들어간다.

미측정으로 남은 것: `_s` × luna(P8, ~$12.5)와 §1.3의 **CoN 부호 역전**(`_s`에서 direct arm이 필요),
그리고 **LLM write를 쓰는 organizer × `_s`** — 최저가 Nemori도 ingest만 $50~70이라(턴 246,750개)
현 캡 밖이다. 전부 미승인.

### 9.4 이 주장의 약점 (먼저 적어 둔다 · 2026-08-18 재평가)

0. **🔴 새 약점 — 측정한 체제 밖으로 일반화하지 말 것.** 같은 검색 메커니즘이 oracle에서 **−3.00pp**,
   `_s`에서 **+21.20pp**다. **부호가 뒤집힌다.** 우리가 oracle에서 먼저 잰 "검색세"를 그대로 밀었으면
   정확히 틀린 결론을 발표할 뻔했다. 그래서 이 절의 모든 문장에는 **어느 haystack 길이에서 잰 것인지**가
   붙어야 하고, 붙지 않은 문장은 결함이다. 아직 안 잰 체제: `_m`(1.5M 토큰, 컨텍스트에 **안 들어감**) —
   write 경로가 존재하는 이유가 정확히 그 체제인데 **우리는 한 번도 그 체제를 재지 않았다.**
1. **n=5, 단일 시드였다 — 시드 쪽은 이제 실측됐다.** ρ=−0.60도 +0.90도 유의하지 않다(n=5에서
   p≈0.08~0.28). 다만 "단일 시드"의 크기는 잰다: 같은 arm 재실행이 **overall +0.40pp
   [−1.40,+2.20]**, 500문항 중 20개만 판정이 갈린다(docs/20). 그리고 judge 교체는 **−1.2~−2.0pp**.
   ⇒ **2pp 이하의 LongMemEval 주장은 노이즈와 구분되지 않는다.** C4(12.57~15.40)와 C6(21.20)은
   그 6~30배라 살아남고, cross-arm ρ는 여전히 **예시**로만 쓴다.
2. **각 arm이 자기 lineage의 operating point로 읽는다.** `docs/18`이 *"다른 행과 비교하기 전에 알아야 할
   가장 중요한 것"*이라고 명시한 조건이다. ⇒ cross-arm 상관은 **예시**로만 쓰고, 무게는 arm 내부
   증거(+5.26 / +1.36)와 C4에 실어야 한다.
3. **Zep 행이 낮은 이유가 write 때문이 아닐 수 있다.** `docs/18`의 후보 설명은 *"참여자 발화가 아니라
   추상을 건넨 유일한 arm"* — write 실패가 아니라 **read 산출물의 성질**이다. 이건 C1의 반례가 아니라
   **C2의 사례로 재배치**하는 것이 정직하다.
4. **C2의 cross-arm 상관은 검색 품질과 교락돼 있다.** 컨텍스트가 큰 arm이 검색도 잘했을 수 있다.
   교락이 없는 것은 arm 내부 두 건뿐이고, 그 둘이 말하는 것은 "컨텍스트를 늘리는 것(+1.36)보다
   질의를 다루는 방식(+5.26)이 4배 세다"이다 — **컨텍스트 크기 자체가 답이라는 주장은 하지 않는다.**
   C6이 이 방향을 강화한다: `_s`에서 **컨텍스트를 9배 줄인 쪽이 21점 이겼다.**
5. **C6은 "검색"의 값이지 "메모리 시스템"의 값이 아니다.** 우리가 잰 것은 임베딩 인덱스 + 하이브리드
   검색 top-50이고, 그건 어떤 organizer도 하지 않은 일이 아니라 **모든 organizer가 공통으로 깔고 가는
   바닥**이다. 그러니 C6은 "메모리 무용"의 반례이면서 동시에 **"write에 LLM을 얹는 부분이 이 바닥 위에
   무엇을 더하는가"를 여전히 미측정으로 남긴다.** 그 측정은 organizer × `_s`이고 최저가가 $50~70이다.
6. **우리 검색은 upstream보다 유리하다.** upstream 인덱스는 user 턴만 담는데(LME-A1) 우리는 전 턴을
   담는다. SSA 98.21이 그 결과이고, upstream의 검색 평가는 그 타입을 아예 채점하지 못한다(P7).
   따라서 C6의 절대값을 upstream 검색 baseline과 나란히 놓으면 안 된다.

### 9.5 발표 구조

```
1. 다들 LongMemEval 점수로 메모리 순위를 매긴다          (현황: §5 비교가능성 행렬)
2. 그 점수의 축을 분해하면 메모리의 몫이 작다            (C3 + C1/C2)
3. 메모리를 완벽히 고정해도 이만큼 움직인다              (C4 ← 실측: 12.6~15.4pp, docs/20)
3b. 그리고 검색은 21점을 사는데 write 콜 0개로 산다      (C6 ← 실측: _s 60.40 → 81.60)
4. 게다가 애초에 비교가 불가능하다                       (§6 P1~P15, 4축)
5. 필드도 같은 결론에 도달했다                           (§7 V2가 고친 것들)
6. 그리고 아무도 안 재는 축이 있다 — write 비용          (§3.1: _s ingest = LoCoMo의 42배)
```

새로 돌릴 것은 3번 하나뿐이었고, **2026-08-17에 돌았다**($3.74). 3b는 계획에 없던 줄인데
**`_s`가 만들어 줬다**(2026-08-18, $11.69) — 그리고 이 줄이 발표의 논조를 바꾼다.
3번만 있으면 "메모리는 별 값이 없다"로 오독되지만, 3b가 붙으면 주장이 정확해진다:
**값을 하는 것은 검색이고, 값을 아직 증명하지 못한 것은 write에 LLM을 얹는 부분이다.**
[연역]으로 남은 것은 C5(V2 설계 증거)뿐이다.

---

## 10. 재현 스크립트 · 실행 환경 · 미해결

### 10.1 재현 스크립트

이 문서의 **감사 실측**은 전부 `$0`이며 `scripts/repro/lme_audit/`의 10종으로 산출했다(같은
디렉터리의 README가 실행법과 필요 입력을 담는다). CI에는 넣지 않는다 — 로컬 데이터셋과 upstream
클론이 필요하다. **유료 실측**(§8.4 R1~R2′)은 `scripts/repro/exp_lme_reading.py`가 돌리고
`scripts/repro/lme_c4_analysis.py`가 사전등록된 판정 규칙을 집행한다(둘 다 아티팩트 계약은
`run_status.py`가 읽는 형식).

`lme_stats.py`(규모·절단) · `lme_tokens.py`(o200k 토큰) · `oracle_stats.py`(oracle 규모·비정렬) ·
`cleaning_diff.py`(릴리스 diff) · `prompt_rediff.py`(프롬프트 바이트 대조) · `rediff.py`(집계기 대조) ·
`retrieval_gold_audit.py`(user-only 인덱싱 여파) · `abs_evidence.py`(abstention evidence) ·
`dup_sessions.py`(ISS #54) · `gold_issue_check.py`(ISS #41·#22)

### 10.2 이 머신에서 돌 수 있는가 — 실측 (2026-08-17)

| 항목 | 값 | 판정 |
|---|---|---|
| 디스크 | 500 GB 여유 | 무관 |
| **RAM** | **총 7.9 GB, 가용 3.0~4.5 GB, swap 3.1 GB 사용 중** | **유일한 병목** |
| CPU | 4 코어 (런은 API 대기 지배) | 무관 |
| `oracle` 로드 | peak RSS 72 MB / 1.1초 → 스트리밍 **35 MB / 0.7초** | ✅ 여유 |
| **`_s` 로드** | ~~2.42 GB / 35초~~ → 스트리밍 **124 MB / 10.3초** | ✅ (2026-08-18 이후) |
| **`_m` 로드** | 구 경로 **≥24 GB 투영 → 불가**, 스트리밍 **193 MB** | ✅ (스트리밍 한정) |
| 동시성 | `bench/locomo.py:843`의 ThreadPoolExecutor 패턴 재사용 | ✅ |
| 예상 wall-clock | oracle 500문항 × (생성 1 + judge 1), workers=8 → **런당 ~10분** | ✅ |

**2026-08-18에 바뀐 것 — 로더가 스트리밍이 됐다.** `json.loads(path.read_text())`의 2.42 GB는
파싱 결과가 아니라 **디코딩된 `str` 사본**이 대부분이었다(이 코퍼스는 non-BMP 문자를 포함해
277 MB 파일이 ~1.1 GB `str`가 된다). 실제 파싱된 인스턴스 리스트는 **468 MB**다. 따라서:

| | 구 경로 peak | `iter_longmemeval` peak |
|---|---|---|
| `oracle` (15 MB) | 72 MB | **35 MB** |
| `_s` (277 MB) | 2.42 GB | **124 MB** |
| `_m` (2,737 MB) | **≥24 GB (투영)** | **193 MB** |

`_m`의 구 경로 투영치는 이 머신의 RAM+swap 합(15.9 GB)을 넘는다 — **`_m`은 스트리밍이
아니면 로드 자체가 불가능하다.** `load_longmemeval`은 시그니처를 유지한 채 같은 이득을
받지만(`_s` 468 MB), **`_m`의 리스트는 ~4.6 GB라 여전히 안 들어간다.** `_m` 호출자는
리스트가 아니라 이터레이터를 써야 한다.

**여전히 유효한 규칙**: ① 프로세스가 아니라 **스레드**로 병렬화(메모리 공유)
② 인스턴스 **deep copy 금지**(`prompt_rediff.py`가 하는 짓 — 감사 스크립트라 괜찮지만
드라이버에선 금지) ③ 로드한 원본 리스트를 필요 이상으로 붙들지 말 것.

**드라이버는 ③을 두 곳에서 어기고 있었고(2026-08-18 수정)**, 둘 다 `_s`에서는 무해하고
`_m`에서는 치명적인 같은 모양의 결함이었다: 인스턴스 500개를 리스트로 전량 보유(`_m` ~4.6 GB),
그리고 스탬프용 `sha256(data_path.read_bytes())`(단일 2.74 GB 할당). 지금은 유계 큐 뒤의
producer 스레드가 인스턴스를 하나씩 흘려보내고, 해시는 청크로 읽는다. **바이트 불변 증명**:
`_s` dry run이 유료 arm의 `prompt_sha256`을 5/5 재현하고, oracle/`_s` 스탬프가 각각
`821a2034` / `d6f21ea9`로 동일하다.

**행당 벽시계는 비용과 따로 본다.** `_s` × 검색 top-50 arm의 실측은 **행당 156초**
(500행 = 78,038 행-초 → workers=5에서 **4.3시간**)였고, 지배 항은 API 대기가 아니라
**ingest의 턴당 임베딩 왕복**이다(`_s` 494턴 → 466콜/인스턴스). `_ingest_episode`가
동기 인덱싱을 계약으로 못박고 있어(docs/04 §2) 배치화는 프레임워크 write 의미론 변경이다.
⇒ `_m`(인스턴스당 ~4,947턴)은 **비용이 아니라 벽시계가 제약**이다.

### 10.3 미해결

- ~~드라이버 배선~~ **완료** (`scripts/repro/exp_lme_reading.py`, §8.5 가드 6종 하드코드)
- terra/sol 레지스트리 등록 (견적 확정에 필요)
- 원장 C-4 갱신 3건 — §3.2(릴리스 계보), LME-A1/A2(SSA 91% 삭제), **§6 P2의 "이중계상" 표현 교정**.
  원장은 레인 A 소유라 여기서 고치지 않는다
- ~~`_m`(2.74 GB) 미취득~~ **완료 (2026-08-18)**. D3를 먼저 고쳤고(4 → 실측 4.610, 그리고 토크나이저를
  넘기면 정확 절단), 취득 후 감사로 D3가 실제로 `_m` **500/500**에서 발동함을 확인했다(`_s`는 0/500).
  로더·드라이버는 스트리밍이 됐다(§10.2). 남은 것은 **측정**이고, 그 제약은 비용이 아니라 벽시계다
- **LME-A20의 크기 미측정** — 부호는 연역되지만 크기는 검색 재현(GPU)이 필요하다
- **`_s` × luna 미측정**(P8, ~$12.5)과 **`_s` × direct 미측정**(§1.3 CoN 부호 역전이 여기 있다)
- **organizer × `_s` 미측정** — 이것이 남은 가장 큰 구멍이다. 우리 LME 수치에는 아직 어떤 organizer도
  들어간 적이 없고, `_s`에 얹으면 ingest만 Nemori $50~70 / A-Mem ~$110 / Zep ~$210이다(턴 246,750개,
  2026-08-17 CountingLLM 실측을 22.5배로 환산). **C6이 깔아둔 검색 바닥 위에 LLM write가 무엇을
  더하는지는 그래서 여전히 미측정이다.**
- **ingest 임베딩이 턴당 1 왕복이다** — `_ingest_episode`가 "호출자가 돌아오는 순간 검색 가능"을
  계약으로 잡고 있어(docs/04 §2) 동기 인덱싱이고, 그래서 `_s` 한 행이 466 왕복 = 156초다.
  배치화하면 **같은 벡터·같은 점수에 왕복만 준다**(점수에 영향 없음, 벽시계에만 영향). 다만 write
  의미론 변경이라 측정 직전에 손댈 것은 아니다. `_m`의 벽시계 제약이 전부 여기서 나온다
- **organizer × `_m`은 현 캡 밖이다** — `_s`의 $50~70(Nemori)이 턴 2,446,993개에서 **10배**가 된다.
  `_m`에서 write 경로를 재는 것은 $500~700짜리이고, 검색 arm은 ~$12다. **"write 경로가 존재하는
  이유인 체제"에서 정작 write 경로만 못 재는 구조**이며, 이것을 §9.4에 약점으로 명시할 것
- **LLM 클라이언트에 전송 재시도가 없다** — `_s` full-context에서 500행 중 16행이 연결/타임아웃으로
  죽었고(3.2%), 재개 프로세스가 그 비용을 갚았다. 임베더는 2026-08-17에 재시도를 얻었다
- Mem0/Supermemory/OMEGA 등 벤더 수치의 독립 검증은 하지 않았고 할 계획도 없다(§5.3은 전부 자체 보고 태그)
