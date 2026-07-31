# Mem0 업스트림 스터디 — write-path 콜 구조, eval 프로토콜, 포트 스코프

> 조사: 2026-07-31, Phase 2 Stage 1 Task 8. 정적 판독만 수행했고 업스트림 코드를 실행하거나
> API를 호출하지 않았다. 논문 arXiv:2504.19413 (Chhikara, Khant, Aryan, Singh, Yadav, 2025-04).
>
> 표기 규약(`zep-graphiti.md`와 동일): **검증 ✓** = 이 조사에서 코드를 직접 읽어 확인한 사실
> (file:line 인용 포함) · **1차 소스 ○** = 논문/README/docstring이 주장하는 바로, 코드 대조까지는
> 하지 않았거나 코드가 공개되지 않아 대조가 불가능한 것.

## 핀

| 리포 | 경로 | SHA | 날짜 |
|---|---|---|---|
| `mem0ai/mem0` (본체) | `~/.agmem/upstream/mem0` | `760dca6f391277d79c3c7d2096c1bf1d037526c3` | 2026-07-30 |
| ↳ 태그 `v0.1.94` (논문 시점 라이브러리) | 같은 클론 내 태그 | `07ddd7cb4bd67962cf9a988d7b5c3f3920fad2d4` | **2025-04-26** |
| ↳ 태그 `evaluation-archive` (논문 하네스 보존본) | 같은 클론 내 태그 | `931d579ba56cf6ef0c50cdb0495ce79a41d15def` | 2026-06-13 |
| `mem0ai/memory-benchmarks` (현행 eval) | `~/.agmem/upstream/mem0-memory-benchmarks` | `4b61c5d31b9c668a12b4f5e78064248a02c82d2b` | 2026-05-14 |

`--depth 50` 얕은 클론이라 태그 3종은 `git fetch --depth 1 origin refs/tags/<tag>`로 개별 확보했다.
`v0.1.94`는 arXiv 제출(2025-04) 직전 릴리스로, 논문 시점 OSS 라이브러리를 대표하는 태그로 골랐다.
(같은 리포에 `v` 접두 유무가 다른 두 태그 계열이 공존한다 — 접두 없는 `0.1.94`는 **2024-03-05**의
전혀 다른 커밋이다. 태그로 논문 시점을 지목할 때 반드시 `v` 붙은 쪽을 써야 한다. 검증 ✓)

---

## ① 계보 — 논문 수치를 만든 코드는 어느 것인가

이 리포는 단일 라이브러리가 아니라 **플랫폼 리포**다. 최소 네 갈래의 쓰기 경로가 한 트리에 공존한다.

| # | 계보 | 엔트리포인트 | write 경로 소재 |
|---|---|---|---|
| L1 | OSS 라이브러리 | `mem0.Memory` (`mem0/memory/main.py:462`) | 리포 안 (공개) |
| L2 | 관리형 플랫폼 SDK | `mem0.MemoryClient` (`mem0/client/main.py`) | **서버 측, 비공개** |
| L3 | self-host REST 서버 | `server/main.py:374` | L1을 감싼 얇은 래퍼 |
| L4 | TS SDK / CLI / 플러그인 | `mem0-ts/`, `cli/`, `integrations/` | 조사 범위 밖 |

**논문의 LoCoMo 수치는 L2(플랫폼)에서 나왔다. 검증 ✓** — 근거는 논문 하네스 자체다.
`evaluation-archive` 태그가 보존한 `evaluation/README.md`는 `:3`·`:6`에서 arXiv:2504.19413의
"code and dataset for our paper"임을 명시하고 인용 BibTeX까지 싣는데, 그 하네스의 ingest 진입점
`evaluation/src/memzero/add.py:47-51`은 `MemoryClient(api_key=…, org_id=…, project_id=…)`를
생성하고 `:69-71`에서 `self.mem0_client.add(message, user_id=…, version="v2", …)`를 호출한다.
검색 측 `evaluation/src/memzero/search.py:20-24`도 동일하게 `MemoryClient`다.
그리고 `mem0/client/main.py:117`은 `self.host = host or "https://api.mem0.ai"`,
`:217`은 `self.client.post("/v3/memories/add/", json=payload)` — **순수 HTTP 클라이언트**이며
추출·판단 로직이 클라이언트 측에 전혀 없다.

즉 **논문 Table 1·2의 Mem0 행을 만든 write 경로는 공개되어 있지 않다.** 리포에서 읽을 수 있는
두 단계 write(L1)는 같은 시기에 존재했지만 그 수치를 만든 코드가 아니다. 이것은 이 프로젝트의
`audit-defect-classes` 중 **"여러 벌 중 한 벌만 보기"**의 교과서적 사례이고, 우리 4-way 비교표에
반드시 각주로 달려야 한다.

현행 벤치마크 리포도 같은 결론을 강화한다. `results/`에 LoCoMo 결과 파일은
`results/platform/locomo_results.json`·`locomo_top50_results.json` 둘뿐이고,
`results/oss/`에는 LongMemEval 4종만 있고 **LoCoMo가 아예 없다**(검증 ✓). 공개된 Mem0 LoCoMo
수치 중 OSS 코드로 산출된 것은 과거·현재 통틀어 확인되지 않는다.

부수적으로: 본체 리포의 `evaluation/`는 지금 **초기화되지 않은 서브모듈**이다
(`.gitmodules` → `mem0ai/memory-benchmarks`, `git submodule status`가 `-4b61c5d…`로 미체크아웃
표시, 디렉토리는 비어 있음. 검증 ✓). 핀된 SHA는 memory-benchmarks의 현재 HEAD와 정확히 일치한다.
`mem0`만 클론해서 `evaluation/`을 열면 빈 디렉토리를 보게 되므로, 이 계보 질문 자체가 눈에
띄지 않는 구조라는 점을 기록해 둔다.

---

## ② write-path 콜 구조

### ②-1 논문 시점 OSS(`v0.1.94`) — 2단계, 배치 판단

`Memory._add_to_vector_store` (`mem0/memory/main.py:203-339` @ v0.1.94). 검증 ✓

1. **Phase 1 — fact 추출, LLM 1콜.** `get_fact_retrieval_messages(parsed_messages)`
   (`mem0/memory/utils.py:6` @ v0.1.94)가 `FACT_RETRIEVAL_PROMPT`
   (`mem0/configs/prompts.py:14` @ v0.1.94)를 system으로, `f"Input:\n{parsed_messages}"`를 user로
   묶어 반환하고, `:221-227`이 `response_format={"type": "json_object"}`로 1회 호출한다.
   파싱은 `json.loads(response)["facts"]` (`:231`), 실패 시 빈 리스트(`:232-234`).
2. **retrieval — LLM 0콜, embedder F콜.** 추출된 fact F개 각각에 대해
   `embedding_model.embed(new_mem, "add")` 1회(`:239`) 후 그 벡터로
   `vector_store.search(query=new_mem, vectors=…, limit=5, filters=filters)` (`:241-246`).
   **similarity threshold 인자가 없다** — top-5는 유사도와 무관하게 전부 후보에 들어간다(검증 ✓).
   결과를 id 기준으로 dedup(`:249-252`)하므로 판단 콜에 들어가는 기존 메모리 수는
   **fact별 5개의 합집합**(최대 5F, 실제로는 훨씬 적음)이다.
3. **UUID→정수 매핑.** `temp_uuid_mapping` (`:255-259`) — 주석이 "handling UUID hallucinations"라고
   밝힌 대로, LLM에는 `"0","1",…`만 보여주고 응답의 정수를 실제 UUID로 되돌린다.
4. **Phase 2 — 갱신 판단, LLM 1콜(전 fact 배치).** `get_update_memory_messages(retrieved_old_memory,
   new_retrieved_facts, custom_update_memory_prompt)` (`:261-263`)가 만든 단일 프롬프트를
   `:266-269`에서 **한 번** 호출한다. **fact당 1콜이 아니라 전체 fact 1콜이다**(검증 ✓).
   프롬프트 본문은 `DEFAULT_UPDATE_MEMORY_PROMPT` (`mem0/configs/prompts.py:61` @ v0.1.94):
   "You can perform four operations: (1) add … (2) update … (3) delete … (4) no change."
5. **적용.** `:283-327`이 `resp["event"]`로 분기해 `_create_memory`/`_update_memory`/`_delete_memory`를
   부른다. 이벤트 어휘는 **`ADD`/`UPDATE`/`DELETE`/`NONE`** — 우리가 쓰는 이름은 NOOP이지만
   업스트림 문자열은 `NONE`이다(`:326`). 이 문자열이 포트에서 그대로 재현되어야 한다.

**"~2 LLM calls/turn" — CONFIRMED (조건부). 검증 ✓** `add()` 1회당 정확히 LLM 2콜이며,
추출된 fact 개수 F와 무관하다(F=0이어도 판단 콜은 나간다 — `retrieved_old_memory`가 비고
`new_retrieved_facts`가 빈 리스트여도 `:261`은 무조건 실행된다). embedder는 별도로
**fact당 1콜**이고, ADD/UPDATE의 `text`가 추출 fact 문자열과 다르면
(`_create_memory:648-651`, `_update_memory:730-733`의 `existing_embeddings` 미스) **추가 embed 1콜**이
더 붙는다. 갱신 판단 프롬프트가 "keep the fact which has the most information" 식으로 문장을
**다시 쓰도록** 지시하므로 이 미스는 예외가 아니라 상시로 봐야 한다.

호출 수 견적을 위한 산술(논문 하네스 기준): 하네스는 `batch_size=2`로 메시지 2개씩 묶어 add하고
(`evaluation/src/memzero/add.py:46`, `:80-83`), 같은 세션을 화자별로 **두 번** 적재한다
(`:110-127`, 두 번째는 role을 뒤집은 `messages_reverse`). 메시지 M개 세션이면
add 호출 = 2 × ⌈M/2⌉ ≈ M회, LLM 콜 = **2M회 → 메시지 1개당 2콜**. 배치(÷2)와 이중 적재(×2)가
상쇄되어 "턴당 2콜"이라는 숫자는 유지되지만, **이유는 논문 서술과 다르다**. 견적서에는
"턴당 2콜"이 아니라 "add 호출당 2콜, 하네스 배치·이중적재 반영 시 메시지당 2콜"로 적어야 한다.

**"vector variant is self-contained without the graph store" — CONFIRMED. 검증 ✓**
`add()` (`:176-183` @ v0.1.94)는 `_add_to_vector_store`와 `_add_to_graph`를 `ThreadPoolExecutor`에
**항상 함께** 제출하지만, `_add_to_graph` (`:341-350`)는 `if self.enable_graph:`가 거짓이면
빈 리스트를 즉시 반환한다. 벡터 경로는 그래프 결과를 읽지 않고, 반환값도
`{"results": vector_store_result}`로 분리된다(`:195-201`). 그래프는 **가산적 부가 경로**이지
벡터 경로의 의존물이 아니다. Mem0-g 제외 전제는 유효하다.

### ②-2 현행 HEAD(`760dca6`) — **2단계가 사라졌다**

`Memory._add_to_vector_store` (`mem0/memory/main.py:849-1176` @ HEAD)는
`# === V3 PHASED BATCH PIPELINE ===` (`:886`)로 완전히 대체되었다. 검증 ✓

- Phase 0: `db.get_last_messages(session_scope, limit=10)` (`:890`).
- Phase 1: `embedding_model.embed(parsed_messages, "search")` 1콜(`:895`) +
  `vector_store.search(…, top_k=10, filters=search_filters)` (`:896-901`). threshold 없음.
- Phase 2: **LLM 1콜뿐**(`:925-932`). system은 `ADDITIVE_EXTRACTION_PROMPT`(`:912`),
  user는 `generate_additive_extraction_prompt(...)`(`:918-923`).
- Phase 3~8: 배치 임베딩(`:964`), md5 해시 dedup(`:990-994`), 배치 insert, 엔티티 링킹
  (`:1056-1160`, 세만틱 매치 임계 `score >= 0.95` at `:1122`), 메시지 저장.
- 반환: `{"id": r[0], "memory": r[1], "event": "ADD"}` — **`"ADD"`가 리터럴로 고정**(`:1165-1168`).

`ADDITIVE_EXTRACTION_PROMPT` (`mem0/configs/prompts.py:468`)는 이를 스스로 못박는다:
*"Your sole operation is ADD."* (`:472`) 그리고 빌더 docstring(`:1029-1032`)도
*"The LLM will produce only ADD operations"*.

**따라서 현행 OSS write는 턴당 LLM 1콜, ADD 전용이다.** UPDATE/DELETE/NONE 판단은 존재하지 않는다.
`DEFAULT_UPDATE_MEMORY_PROMPT`와 `get_update_memory_messages`는 여전히
`mem0/configs/prompts.py:176`/`:406`에 남아 있으나 **라이브러리 어디서도 호출되지 않는다** —
전 리포 grep에서 참조처는 `tests/configs/test_prompts.py`뿐이다(검증 ✓). 같은 방식으로
`FACT_RETRIEVAL_PROMPT`/`USER_MEMORY_EXTRACTION_PROMPT`/`AGENT_MEMORY_EXTRACTION_PROMPT`와
이들을 묶는 `mem0/memory/utils.py:15` `get_fact_retrieval_messages`,
`:31` `get_fact_retrieval_messages_legacy`도 **호출자가 없다**(검증 ✓).

async 경로(`AsyncMemory._add_to_vector_store`, `:2477-2809` @ HEAD)는 sync와 **동일한 V3 파이프라인**이며
`asyncio.to_thread` 래핑만 다르다(`:2521`의 동일 주석, `:2548`의 동일 system prompt). v0.1.94의
async(`:935-1099`)도 sync와 콜 수가 같고 fact별 retrieval만 `asyncio.gather`로 병렬화한
차이뿐이다(`:973-986`). **sync/async 사이의 콜 수 분기는 두 버전 모두 없다**(검증 ✓) — 즉
"한 벌 안 여러 경로" 결함은 sync/async 축에서는 발견되지 않았고, 대신 **버전 축**에서 나타난다.

---

## ③ eval 프로토콜 vs 우리 Mem0-J

우리 구현: `src/agmem/bench/locomo.py` — 모듈 주석 `:113` "Mem0-standard binary LLM judge
(cat 1-4 only)", 프롬프트 `:121`, 판정 함수 `judge_answer` `:431-444`.

비교 대상은 **두 개**다. 논문 하네스(`evaluation-archive`)와 현행 벤치마크 리포는 서로 다른
프로토콜이며, 우리가 재현 대상으로 삼아야 하는 것은 **논문 하네스** 쪽이다.

### ③-1 논문 하네스(`evaluation/metrics/llm_judge.py`) vs 우리

| 항목 | 논문 하네스 (검증 ✓) | 우리 `locomo.py` (검증 ✓) | 판정 |
|---|---|---|---|
| judge 모델 | `gpt-4o-mini` 하드코딩 (`llm_judge.py:42`) | `--judge-model`로 주입 | 설정 차이 (Task 3에서 분리됨) |
| **temperature** | **0.0** (`llm_judge.py:52`) | **0.1** (`src/agmem/llm/client.py:35` 기본값) | **불일치 — 각주 필요** |
| 출력 형식 | `response_format={"type":"json_object"}`, 키 `"label"` (`:51`, `:54`) | JSON schema `{"label": enum[CORRECT,WRONG]}` (`locomo.py:115-119`) | 동등(우리가 더 강제적) |
| 라벨 파싱 | `label == "CORRECT"` → 1 (`:55`) | `.strip().upper() == "CORRECT"` (`:444`) | 우리가 더 관대(대소문자·공백) |
| 추론 요구 | "First, provide a short (one sentence) explanation … then finish with CORRECT or WRONG" — **CoT를 먼저 쓰게 한 뒤** JSON으로 라벨만 반환하라는 **자기모순 지시** | 설명 요구 없음, 라벨만 | **불일치 — 각주 필요** |
| cat5 | `int(category) == 5: continue` (`llm_judge.py:88`, `evals.py:23`) — 분자·분모 모두에서 제외 | `cat_num in (1,2,3,4)`일 때만 판정(`locomo.py:783-784`) | **일치** |
| cat3 전처리 | 없음 | 세미콜론 앞부분만 사용 (`locomo.py:249`) | **불일치 — 우리가 추가한 것** |
| 루브릭 | 4문단 산문: "be generous", 같은 topic이면 CORRECT, 날짜는 같은 시기면 형식 무관 | 4개 불릿: 패러프레이즈·부분정답·추가디테일·날짜 14일/기간 50% | 의미상 근접, **문면은 다름** |
| F1/BLEU | `metrics/utils.py` — A-Mem `WujiangXu/AgenticMemory/utils.py`에서 **차용했다고 docstring이 명시**(`:1-13`), **set 기반** 토큰 F1(`:143-152`), stemming·관사 제거 없음 | `eval_mode="wujiang"` 경로가 동일 계보 | **일치(우리 기록과 부합)** |

가장 중요한 항목은 **cat3 세미콜론 전처리**다. 우리는 `gold_for` (`locomo.py:249`)에서 cat3 gold를
`;` 앞부분으로 자르는데, **논문 하네스에는 이 처리가 없다**(`evals.py:17-28`은 `item["answer"]`를
그대로 `str()`으로 넘긴다). 흥미롭게도 **현행** 벤치마크 리포에는 있다
(`benchmarks/locomo/prompts.py:315-322` `preprocess_answer`, cat3 & `";" in answer`). 즉 우리 처리는
현행 Mem0와는 일치하고 논문 하네스와는 불일치한다 — 비교표에서 어느 쪽 수치와 나란히 놓느냐에
따라 각주 문면이 달라진다.

temperature 0.1 vs 0.0은 우리 클라이언트 기본값이 판정에도 그대로 적용된 결과이며, 판정 재현성을
떨어뜨린다. **judge 호출에 한해 0.0을 강제하는 것이 옳다**(코드 변경은 이 태스크 범위 밖 —
포트 플랜의 후보 항목으로 남긴다).

### ③-2 현행 벤치마크 리포 — 논문 프로토콜과 단절

`memory-benchmarks@4b61c5d`의 LoCoMo judge는 논문 것과 **다른 물건**이다(검증 ✓).

- 프롬프트가 4문단 산문 → **7개 번호 규칙**으로 확장(`benchmarks/locomo/prompts.py:218-234`).
  규칙 1 "PARTIAL CREDIT: … AT LEAST ONE correct item … Only mark WRONG if NONE",
  규칙 3 "Never penalize for being more detailed", 규칙 7 "Only mark WRONG when … genuinely
  different or incorrect understanding" — 논문판보다 **일관되게 더 관대**하다.
- answer 생성 프롬프트가 논문의 8줄 지시 + "answer should be less than 5-6 words"
  (`evaluation/prompts.py` `ANSWER_PROMPT`)에서 **7단계 CoT 레시피**로 교체
  (`benchmarks/locomo/prompts.py:40-97`), 길이 제한도 사라졌다. 컨텍스트는
  `ANSWERER_MEMORY_LIMIT = 200`개 메모리를 **시간순 정렬**해 넣는다(`:101`, `:172`).
- 기본 모델이 `gpt-5`(answerer·judge 모두, `run.py:682-683`), 논문은 `gpt-4o-mini`.
- cat5 제외는 유지(`CATEGORIES_TO_EVALUATE = [1, 2, 3, 4]`, `prompts.py:33`).
- **`--with-evidence` 플래그**(`run.py:708`): gold evidence를 judge에 함께 넣고, 주입되는 규칙이
  *"Use evidence only to ACCEPT answers, never to reject them more strictly"*
  (`prompts.py:212`)라고 명시한다 — **한 방향으로만 점수를 올리는 노브**다.

⇒ 현행 리포 수치와 논문 수치는 **같은 이름의 서로 다른 측정**이다. 4-way 비교표에서 Mem0 수치를
인용할 때 어느 하네스인지 반드시 명시해야 한다.

LoCoMo 카테고리 분포는 우리 데이터셋으로 교차 확인했다:
`~/.agmem/datasets/locomo10.json` 실측 `{1:282, 2:321, 3:96, 4:841, 5:446}`, 합 1986 —
`benchmarks/locomo/prompts.py:9-14` docstring의 숫자와 **정확히 일치**(검증 ✓).

---

## ④ evolution-op 매핑 (포트 설계 입력)

논문 시점 OSS(`v0.1.94`) 기준. 우리 op log 어휘로의 대응과, **각 op가 실제로 남기는 부수효과**까지
적는다(포트에서 재현해야 할 것은 라벨이 아니라 부수효과다).

| 업스트림 event | 우리 op | 적용 코드 | 저장 부수효과 (검증 ✓) |
|---|---|---|---|
| `ADD` | `ADD` | `main.py:289-301` → `_create_memory:646-665` | 새 uuid4, `data`/`hash`(md5)/`created_at` 기록, `vector_store.insert`, history 행 `("ADD", old=None)` |
| `UPDATE` | `UPDATE` | `:302-316` → `_update_memory:708-749` | **id 유지**, `created_at`은 **기존 값 보존**, `updated_at` 신규, `hash` 재계산, 벡터 **재삽입**, history 행 `("UPDATE", old=prev_value)`. 반환 dict에 `previous_memory` 포함(`:314`) |
| `DELETE` | `DELETE` | `:317-325` → `_delete_memory:751-758` | **하드 삭제**(`vector_store.delete`), history 행 `("DELETE", new=None, is_deleted=1)` — 감사 로그에만 흔적이 남고 벡터는 사라진다 |
| `NONE` | `NOOP` | `:326-327` | **아무것도 하지 않고 로그만**. `returned_memories`에도 넣지 않는다 → **API 반환값으로는 NOOP이 관측되지 않는다** |

포트 시 반드시 재현해야 할 비대칭 3가지:

1. **`NONE`은 반환되지 않는다.** 호출자는 "판단이 있었으나 아무 일도 없었다"와 "판단 자체가
   없었다"를 구별할 수 없다. 우리 op log는 NOOP을 **기록**하므로 이것은 의도적 확장이며,
   업스트림 대비 관측 가능성이 늘어난 지점으로 문서화해야 한다.
2. **빈 text는 조용히 버려진다.** `:286-288` — `if not resp.get("text"): continue`.
   event가 무엇이든 무시된다(DELETE에도 text가 없으면 버려짐).
3. **환각 id는 조용히 삼켜진다.** UPDATE/DELETE는 `temp_uuid_mapping[resp["id"]]`를
   인덱싱하는데(`:304`, `:311`, `:318`, `:321`), LLM이 범위 밖 정수나 실제 UUID를 뱉으면 `KeyError`가
   나고 `:328-329`의 내부 `except`가 로그만 남기고 넘어간다. **실패한 op이 실패로 집계되지 않는다.**
   우리 A-Mem 포트가 이미 hallucinated-ID 필터를 갖고 있으므로, Mem0 포트에서도 같은 자리를
   `discarded` 카운터로 승격해야 한다(우리 organizer 계약의 silent-discard 원칙과 동일).

추가로 `UPDATE`의 `created_at` 보존(`_update_memory:720`)은 사소해 보이지만 **시간 기반 read/랭킹을
쓰는 우리 쪽에서 순서를 바꾼다** — 갱신된 노트가 최신으로 떠오르지 않는다. 재현 대상으로 명시.

---

## ⑤ 결함 후보

레저 행이 아니라 **후보**다. 각 항목은 코드 인용을 동반하며, 채택 여부는 별도 판단 대상이다.

**(a) 여러 벌 중 한 벌만 보기**

- **M0-C1 — 논문 수치의 write 경로가 비공개.** 논문 하네스가 `MemoryClient`(HTTP → `api.mem0.ai`)만
  사용하므로(`evaluation/src/memzero/add.py:47-71`, `search.py:20-24`, `mem0/client/main.py:117,217`),
  Table 1·2의 Mem0 행은 **어떤 공개 코드로도 재현 불가**하다. 리포에 있는 두 단계 write(L1)를
  읽고 "이것이 논문의 코드"라고 적으면 오기다. **영향: CITATION(강)** — 우리 비교표의 Mem0 행은
  "vendor-hosted, 비공개 write 경로"로 표기되어야 한다.
- **M0-C2 — 현행 리포에 OSS LoCoMo 결과가 없다.** `results/oss/`는 LongMemEval 4종뿐,
  LoCoMo는 `results/platform/`에만 존재. OSS 코드로 산출된 Mem0 LoCoMo 공개 수치는 확인되지 않음.
- **M0-C3 — `evaluation/`이 빈 서브모듈.** `.gitmodules` + `git submodule status` `-4b61c5d…`.
  본체만 클론하면 논문 하네스가 보이지 않아 C1을 발견할 기회 자체가 사라진다. **영향: —(절차)**

**(b) docstring/프롬프트 vs 코드**

- **M0-C4 — `add()` docstring이 현행 코드에 없는 기능을 약속한다. (강)**
  `mem0/memory/main.py:765-767`: *"an LLM is used to extract key facts from 'messages' and decide
  whether to **add, update, or delete** related memories."* 그러나 같은 파일의 V3 파이프라인은
  `event: "ADD"`를 리터럴로 고정하고(`:1165-1168`), 시스템 프롬프트는
  *"Your sole operation is ADD"*(`prompts.py:468`)라고 못박는다. **docstring이 두 버전 전의 동작을
  서술하고 있다.** 포트 근거를 docstring에서 읽으면 존재하지 않는 경로를 구현하게 된다.
- **M0-C5 — 프롬프트가 선언한 입력 중 절반을 OSS 호출자가 채우지 않는다. (강)**
  `ADDITIVE_EXTRACTION_PROMPT`는 `## Summary`, `## Recently Extracted Memories`,
  `## Observation Date`를 1급 입력으로 설명하고 특히
  *"Observation Date … This is your ONLY temporal anchor"*라고 강조한다(`prompts.py:524-526`).
  그런데 OSS 호출자는 `existing_memories`·`new_messages`·`last_k_messages`·`custom_instructions`
  **4개만** 넘긴다(`main.py:918-923`). 결과적으로 `generate_additive_extraction_prompt`
  (`prompts.py:1016-1062`)는 `## Summary`를 빈 문자열로,
  `## Recently Extracted Memories`를 `[]`로 렌더링한다.
  ⇒ 브리프가 물은 **"rolling-summary input"은 OSS 경로에 존재하지 않는다**(플랫폼 빌더에는 있고,
  빌더 docstring `:962`가 *"Ported from platform/backend/shared/core/utils/prompt_builder.py"*라고
  출처를 밝힌다 — **포팅하면서 배선을 빠뜨린 것**).
- **M0-C6 — Observation Date가 구조적으로 항상 오늘이다. (강, 측정 영향 있음)**
  C5의 따름정리. `_resolve_dates(current_date=None, observation_date=None)`
  (`prompts.py:1007-1014`)는 둘 다 없으면 `datetime.now(timezone.utc).date()`로 채우고,
  OSS 호출자는 `timestamp`를 넘기지 않는다. 더욱이 `Memory.add()`는 `timestamp` 인자를 받으면
  **예외를 던진다** — `:787-788` *"Platform-only temporal parameter. Not supported in OSS."*
  ⇒ OSS 사용자는 올바른 관측 시점을 **넘길 방법이 없고**, 프롬프트는 그 틀린 앵커를
  "유일한 시간 기준"으로 신뢰하라고 지시한다. 2023년 대화(LoCoMo)를 2026년에 적재하면
  "last week"이 3년 어긋난 절대 날짜로 고정된다. **temporal(cat2) 카테고리에 직접 타격.**
- **M0-C7 — `calculate_metrics`의 두 반환 분기가 키 집합이 다르다.**
  `evaluation/metrics/utils.py:119-135`(빈 입력)은 `rouge1_f`·`bert_f1`·`meteor`·`sbert_similarity`를
  포함한 12키를 반환하지만, `:137-164`(정상 경로)는 `exact_match`·`f1`·`bleu1..4`의 7키만 반환한다.
  docstring은 양쪽 다 *"comprehensive evaluation metrics"*. 소비자가 `["rougeL_f"]`를 읽으면
  **빈 입력에서만 동작하고 정상 입력에서 KeyError**가 난다. `evals.py`는 `["f1"]`만 읽어 실사용
  경로는 무사하다. **영향: —(잠재)**

**(c) 죽은 코드 / 읽히지 않는 설정**

- **M0-C8 — 2단계 write의 잔해가 통째로 dead.** `DEFAULT_UPDATE_MEMORY_PROMPT`(`prompts.py:176`),
  `get_update_memory_messages`(`:406`), `FACT_RETRIEVAL_PROMPT`(`:15`),
  `USER_MEMORY_EXTRACTION_PROMPT`(`:63`), `AGENT_MEMORY_EXTRACTION_PROMPT`(`:124`),
  `utils.get_fact_retrieval_messages`(`:15`), `get_fact_retrieval_messages_legacy`(`:31`) —
  전 리포 grep 결과 라이브러리 호출자 0, 참조는 `tests/` 뿐. **테스트가 죽은 코드를 살아 있게
  보이도록 지탱하고 있다**(`tests/configs/test_prompts.py:19`가 `DEFAULT_UPDATE_MEMORY_PROMPT`로
  시작하는지까지 단언한다).
- **M0-C9 — `uuid_mapping`이 assign 후 미사용.** `main.py:904-908`(sync), `:2540-2544`(async).
  주석은 `# Map UUIDs to integers (anti-hallucination)`인데, 그 매핑을 되돌릴 소비자
  (=갱신 판단 응답 처리)가 삭제되어 **쓰이지 않는다**. C4·C8과 같은 뿌리.
- **M0-C10 — `linked_memory_ids`가 요청되지만 저장되지 않는다.** 시스템 프롬프트가 링킹을
  상세히 지시하고(`prompts.py:513`) 빌더 docstring도 *"with optional linked_memory_ids"*
  (`:1031`)라고 하지만, `_add_to_vector_store`가 각 항목에서 읽는 필드는 `text`(`:986`)와
  `attributed_to`(`:1006`)뿐이다. **LLM이 생성한 링크는 파싱 즉시 버려진다.**
- **M0-C11 — 논문 하네스의 `process_questions_parallel`이 호출되지 않는다.**
  `evaluation/src/memzero/search.py:198` 정의, `run_experiments.py:43-49`는
  `process_data_file`만 부른다. 또한 `run_experiments.py:29`의 `--top_k` 기본값은 **30**인데
  `MemorySearch.__init__` 기본값은 **10**(`search.py:19`) — Makefile이 `--top_k 30`을 명시하므로
  (`evaluation/Makefile:7,13`) 공표 경로는 30이지만, 인자 없이 직접 부르면 조용히 10이 된다.
- **M0-C12 — 문서 수치 불일치(경미).** `memory-benchmarks/README.md`의 벤치마크 표는 LoCoMo를
  "~300 questions"로 적지만, 같은 리포 `benchmarks/locomo/prompts.py:9-14`의 카테고리별 수
  (282+321+96+841+446)는 **1986**이고 이쪽이 실측과 일치한다(위 ③ 교차 확인).

**(d) 측정 관대성 노브**

- **M0-C13 — `--with-evidence`는 단방향이다.** `benchmarks/locomo/run.py:708`,
  주입 규칙 `prompts.py:211-212`: *"Use evidence only to ACCEPT answers, never to reject them
  more strictly"*, 그리고 gold와 어긋나도 evidence가 뒷받침하면 CORRECT로 하라고 명시
  (*"The gold answer may be wrong or oversimplified"*). 점수를 올리는 방향으로만 작동하는
  스위치이므로, 이 플래그가 켜진 수치는 꺼진 수치와 **비교 불가**다.

---

## ⑥ 포트 스코프 제안

**vector-only: 확정(CONFIRMED).** ②에서 검증한 대로 그래프 경로는 가산적이고 벡터 경로가
그래프 산출물을 읽지 않는다. Mem0-g는 별도 트랙으로 미루고, 이번 포트는 벡터 변형만 다룬다.

**어느 벌을 포트할 것인가 — `v0.1.94`의 2단계 write.** 근거:

1. 우리 트랙의 목적은 *"conversational verification track"* + **비용 독립측정**이다.
   ADD 전용 V3(HEAD)는 measure할 evolution-op이 하나뿐이라 우리 op-log 어휘와 겹치는 면적이
   거의 없다. 2단계 write는 ADD/UPDATE/DELETE/NOOP 네 op을 모두 준다.
2. 논문 수치와 대조 가능한 유일한 공개 구조가 이쪽이다(수치 자체는 비공개 경로 산출물이라
   재현 대상이 아니라 **구조 대조** 대상이다 — M0-C1).
3. HEAD의 V3는 플랫폼 빌더의 **미완성 백포트**다(M0-C5·C6). 그 상태를 재현하면 우리 쪽에
   업스트림 배선 누락을 그대로 옮겨오게 된다.

**설정 노브(포트에서 노출할 것):**

| 노브 | 업스트림 값 | 비고 |
|---|---|---|
| fact 추출 프롬프트 | `FACT_RETRIEVAL_PROMPT` (v0.1.94 `prompts.py:14`) | `custom_fact_extraction_prompt`로 교체 가능(`main.py:215-217`) |
| 갱신 판단 프롬프트 | `DEFAULT_UPDATE_MEMORY_PROMPT` (`prompts.py:61`) | `custom_update_memory_prompt`로 교체 가능 |
| 판단용 retrieval k | fact당 **5**, 합집합, **threshold 없음** (`main.py:244`) | 우리 쪽 threshold 기본값이 끼어들지 않도록 명시적 무효화 필요 |
| 판단 배치 | 전 fact **1콜** | fact당 콜로 바꾸는 옵션은 **만들지 말 것**(업스트림에 없음) |
| event 어휘 | `ADD`/`UPDATE`/`DELETE`/**`NONE`** | 문자열 `NONE` 그대로. 우리 로그에서 NOOP으로 매핑 |
| ingest 배치 크기 | 하네스 `batch_size=2` (`evaluation/src/memzero/add.py:46`) | 라이브러리 기본값 아님 — **하네스 파라미터**로 분리 |
| 화자 이중 적재 | 하네스가 세션당 2회 (`add.py:110-127`) | 비용 산정에 ×2로 반영 |
| judge temperature | **0.0** (`llm_judge.py:52`) | 우리 기본 0.1과 다름 — judge 경로만 0.0 강제 권고 |

**견적용 콜 수(우리 CountingLLM 계약 기준):**

- organizer LLM: `add()` **1회당 정확히 2콜** (fact 수 F와 무관, F=0이어도 2콜).
- embedder: fact당 1콜 + ADD/UPDATE의 text가 추출 fact와 다를 때마다 1콜.
  갱신 프롬프트가 재작성을 지시하므로 **상한 2F로 잡는 것이 안전**하다.
- LoCoMo 세션(메시지 M개), 논문 하네스 형상 재현 시:
  add 호출 ≈ M회(=2×⌈M/2⌉), organizer LLM ≈ **2M콜**, embedder ≈ 2F_total콜.
  화자 이중 적재를 끄면 정확히 절반.

**포트 플랜(별도 태스크)이 처리해야 할 열린 항목:**

1. 환각 id로 인한 UPDATE/DELETE 유실을 `discarded` 카운터로 승격 (④-3, M0-C-없음: 이건 우리 계약).
2. `NONE`을 op log에 기록할지 — 업스트림은 반환하지 않는다(④-1). 기록하는 쪽을 권고하되
   업스트림 대비 확장임을 코드 주석에 명시.
3. `UPDATE`의 `created_at` 보존이 우리 시간 기반 read와 충돌하는지 확인(④ 말미).
4. judge temperature 0.0 고정.
5. cat3 세미콜론 전처리를 어느 하네스에 맞출지 결정 (③-1 — 논문 하네스에는 없고 현행에는 있음).

---

## 우리 프로젝트에 주는 시사점

1. **"논문 리포"와 "논문 코드"는 다를 수 있다.** Mem0는 논문 리포를 공개했지만 그 안의 실행 경로가
   비공개 서비스를 가리킨다. `upstream-rediff-audit-technique`(공식 함수를 떼어내 출력 비교)이
   여기서는 **적용 불가**이며, 그 사실 자체가 비교표의 각주가 되어야 한다.
2. **서브모듈 미초기화가 계보 질문을 은폐한다.** 앞으로 업스트림을 뜰 때 `.gitmodules` 확인을
   체크리스트에 넣는다. 이번에는 `evaluation-archive` 태그가 있어 논문 하네스를 복원할 수
   있었지만, 태그 이름을 몰랐다면 막혔을 것이다.
3. **버전 축의 "여러 벌"**. 지금까지 적발한 사례는 대부분 같은 시점의 여러 리포/여러 경로였는데,
   Mem0는 **같은 리포의 시간 축**에서 write 경로가 통째로 교체된 사례다(2단계 → ADD 전용).
   docstring·프롬프트·테스트가 전부 그 교체를 반쯤만 따라가서(M0-C4·C8·C9·C10) 코드를 읽는
   사람이 잘못된 결론에 도달하기 쉽다. 업스트림 스터디 시 **태그를 찍어 시점을 고정**하는 절차의
   가치를 보여준다.
4. **"LLM 2콜"은 싸다는 뜻이 아니다.** 하네스가 화자별로 세션을 두 번 적재하므로 실측 비용은
   2배다. 우리가 write-path 비용을 독립 측정할 때 **라이브러리 콜 수와 하네스 형상을 분리해
   보고**해야 한다는 점을 다시 확인시켜 준다.
