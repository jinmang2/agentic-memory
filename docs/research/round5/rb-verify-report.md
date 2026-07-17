# ReasoningBank 충실도 재검증 리포트 (2026-07-17)

- 대상: `/home/jinmang2/agentic_memory/src/agmem/organizers/reasoning_bank.py`
- 기준: arXiv:2509.25140 + github.com/google-research/reasoning-bank (main, 오늘자 raw 소스 직접 대조)
- 참조 감사: `docs/research/fidelity-deep-audit.md` §3 (커밋 a3f6962)
- upstream 파일 위치 정정: 감사가 인용한 경로들은 리포 루트가 아니라 **`WebArena/` 하위**임
  (`WebArena/prompts/memory_instruction.py`, `WebArena/induce_memory.py`, `WebArena/memory_management.py`,
  `WebArena/run.py`, `WebArena/agents/legacy/agent.py`, `WebArena/autoeval/clients.py`).
  라인 번호는 감사 인용과 대체로 일치(파일 내용 동일 추정, 커밋 6개·변동 없음).

## 0. 로컬 변경 여부 (감사 이후)

- `reasoning_bank.py`: **변경 없음** (`git diff a3f6962..HEAD` 빈 출력). P3 항목 12–14 중 코드 반영된 것은 온도 기본값뿐.
- `scripts/exp_locomo_conv0.py`: round-4(91b4d28)에서 role 기본 온도가
  `extract 0.1 / distill 0.1 / judge 0.0 / generate 0.0`으로 변경됨 (`:37-38`).
  감사의 "4 role 전부 0.1" 서술은 **현시점 구식**.
- LoCoMo 실험 config에 `reasoning_bank`은 **아예 없음** (`known` dict `:131-146`) — 감사의
  "LoCoMo 무관" 판단과 일관. RB는 현재 어떤 벤치에서도 실측되지 않음.
- round-4 검증 문서(`fidelity-round4-verification.md`)는 RB를 다루지 않음 → P3는 그대로 open.

## 1. 감사 주장 (a)–(f) 재검증

### (a) 검색 단위: 아이템 vs top-1 경험 — **CONFIRMED / 로컬 미수정**
- upstream `run.py:177-193`: `select_memory(n=1, ...)` 후 top-1 경험의 `memory_items` **전부**를 메모리 파일에 기록.
  ```python
  res = select_memory(n=1, reasoning_bank=reasoning_bank, cur_query=cur_query, ...)
  for item in res:
      for i in item["memory_items"]:
          mem_items.append(i)
  ```
- upstream `memory_management.py:172-215 screening()`: 현재 질의를 **과거 태스크 질의 임베딩 캐시**와 대조
  (경험=태스크 단위, 아이템 아님). 유사도는 instruction-aware 질의 임베딩:
  > `task = "Given the prior web navigation queries, your task is to analyze a current query's intent and select relevant prior queries that could help resolve it."`
  gemini-embedding-001, 3072차원, L2 정규화 cosine ×100.
- 로컬: `core/types.py:144-145` `embedding_text() = f"{self.title}\n{self.description}"` → `memory.py:_apply_one`이
  이 텍스트를 임베딩 → 파이프라인 일반 경로로 **아이템 단위 k개** 회수. 변경 없음. → P3-13 open.

### (b) 온도 role 분리 — **CONFIRMED / 로컬 부분 수정**
- upstream 판정: `autoeval/clients.py:50, 83, 148` 모두 `temperature=0` (LM_Client/CLAUDE_Client/gemini).
- upstream 추출: `induce_memory.py:164-166`
  > `llm_client.one_step_chat(trajectory, system_msg=SUCCESSFUL_SI, temperature=1.0)` (FAILED_SI 동일).
- 로컬 현황: 기본 role 온도가 judge **0.0**(수정됨), distill **0.1**(미수정 — upstream 1.0).
  단 RB 실험 config 자체가 없으므로 "실측 config 위반"은 현재 성립 불가 — 에이전트 벤치 config 작성 시
  `distill 1.0` 지정 필요. → P3-12 절반만 해소.

### (c) judge 이유 미전달 — **CONFIRMED / 로컬 미수정**
- upstream `induce_memory.py:145-162`: autoeval thoughts를 궤적 말미에 첨부.
  ```python
  trajectory += f"\n\nThe task {status_label} because: {autoeval_thoughts}"
  ```
- 로컬 `reasoning_bank.py:113-119`: `JUDGE_SCHEMA`가 `reason`을 받지만 `verdict["reason"]`은
  이후 **어디에도 사용되지 않음**. 변경 없음. → P3-14 open.

### (d) 주입 포맷 — **CONFIRMED / 로컬 미수정 (+감사 서술 1건 교정 필요)**
- upstream `agents/legacy/agent.py:132-137`: **system 프롬프트**에 지시문+아이템 원문 markdown 첨부.
  > `sys_msg += "\n\n" + "Below are some memory items that I accumulated from past interaction from the environment that may be helpful to solve the task. You can use it when you feel it's relevant. In each step, please first explicitly discuss if you want to use each memory item or not, and then take action."`
- 로컬: `MemoryBundle.render()`(유저측 컨텍스트, 1600토큰 예산) + 지시문 없음. 변경 없음.
- **감사 교정**: 감사는 우리 주입을 `StrategyItem.render()`(title+description+content, `types.py:147-148`)로
  기술했으나, 실제 검색 경로는 doc-store dict를 `_DictItem`으로 감싸므로 (`pipeline.py:80-96, 139-160`)
  렌더는 `"{title}: {content}"` — **description은 read 경로에서 탈락**하고 `StrategyItem.render()`는
  검색 경로에서 죽은 코드. (역설적으로 논문 App A.2 "title and content"에는 이쪽이 더 가깝지만,
  공식 코드는 raw markdown 전체를 주입 — 아래 §3-1 참조.)

### (e) content 길이 1-3 vs 1-5 — **CONFIRMED / 로컬 미수정**
- upstream `memory_instruction.py:35` (SUCCESSFUL_SI), `:59` (FAILED_SI):
  > `## Content <1-3 sentences describing the insights learned ...>`
  "1-5 sentences"는 `:88` **PARALLEL_SI(MaTTS)**에만 존재 — 감사의 지적 그대로.
- 로컬 `reasoning_bank.py:70, 84`: 여전히 `"content": "1-5 sentences"`. → P3-14 open.

### (f) 궤적 6000자 절단 — **CONFIRMED / 로컬 미수정**
- upstream `induce_memory.py:70-76 format_trajectory`: 전체 `<think>...</think>\n<action>...</action>` 페어를
  **무절단** 연결. `:142-143`에서 `"**Query:** {query}\n\n**Trajectory:**\n{trajectory}"` 형식.
- 로컬 `reasoning_bank.py:87-93`: JSON dump + `max_chars=6000` head/tail 절단. 변경 없음.

## 2. 감사 주장의 유효성 총평

(a)–(f) 전부 upstream 기준으로 **CONFIRMED**(감사가 정확했음). 로컬 코드는 감사 이후 organizer 무변경이며,
유일한 진전은 실험 스크립트 기본 judge 온도 0.0 (단 RB config 부재로 실효 없음).
감사 표에서 부정확했던 것은 (d)의 "우리 주입=render()=title+description+content" 한 줄뿐 (§1-d 교정).

## 3. 신규 발견 (감사에 없던 항목)

1. **upstream은 추출 출력을 파싱하지 않고 통째로 저장**: `induce_memory.py:184`
   `"memory_items": generated_memory_item.split("\n\n")` — markdown 구조 파싱 없이 문단 분할.
   따라서 실주입물에는 `## Description`은 물론 추출 LLM의 "why" **서두 추론 텍스트까지 포함**될 수 있음.
   논문 App A.2의 "title and content" 서술과 공식 코드가 서로 불일치 — 재현 시 어느 쪽을 따를지
   명시적 결정 필요(공식 코드 = 아이템 markdown 원문 전체 권장).
2. **top-1 미스 시 빈 주입**: `memory_management.py:159-170` — top-1 유사 태스크의 `task_id`가
   bank에 없으면(추출 실패/미수행 태스크) rank-2로 **폴백하지 않고 빈 결과** 반환. 질의 임베딩 캐시는
   `screening():189-197`에서 bank 적재 여부와 무관하게 매 태스크 append되므로 이 상황이 실제로 발생.
   우리 아이템-단위 k-NN은 항상 뭔가를 돌려줌 — 경험 단위 모드 구현 시 이 "miss→무주입" 의미론도 결정 대상.
3. **임베딩 비대칭**: 캐시에 기록되는 질의 벡터는 무-instruction(`RETRIEVAL_DOCUMENT`), 랭킹용 현재 질의
   벡터는 instruction 부가 — 문서/질의 비대칭 임베딩 (`memory_management.py:186-208`).
4. **페르소나 부재**: upstream 프롬프트는 `"You are an expert in web navigation."`으로 시작 (도메인 특화).
   로컬 프롬프트에는 페르소나 없음 — WebArena 재현 시 도메인 페르소나 주입 파라미터 필요.
5. **description "one sentence" 제약 누락**: upstream `## Description <one sentence summary ...>`;
   로컬은 "description must state WHEN to apply"만 있고 1문장 제약 없음.
6. **추출 프롬프트 배치**: upstream은 SI를 **system 메시지**, 궤적을 user 메시지로 분리
   (`one_step_chat(trajectory, system_msg=SUCCESSFUL_SI, ...)`); 로컬은 단일 user 프롬프트 (감사 표에
   있었으나 system/user 분리 자체는 명시 안 됨).
7. **upstream에는 조직기 내 judge가 없음**: 정오 신호는 `--criteria gt`(환경 보상) 또는 별도 autoeval
   파이프라인 (`induce_memory.py:124-129`). 우리 `JUDGE_PROMPT`는 자체 구성물(감사도 "타당한 확장"으로
   분류) — 단 에이전트 벤치에서는 gt reward 우선 사용이 원방법론에 충실.
8. **at most 3 강제 방식**: upstream은 프롬프트 지시뿐, 코드 강제 없음(문단 분할이라 3개 초과 저장 가능);
   로컬은 `maxItems 3` + 슬라이스로 더 엄격. 방향성 무해.
9. **MaTTS 자산 확인**: upstream `PARALLEL_SI`(≤5)/`PARALLEL_AWM_SI`/`SEQUENTIAL_PROMPT`
   (`memory_instruction.py:63-142`) + `induce_scaling.py`/`pipeline_scaling.py`. 로컬 grep 결과 MaTTS 관련
   심볼 0건 — 미구현(공지됨) 그대로.
10. **에이전트 실행 온도**: upstream 에이전트 자체는 `temperature=0.7` (`run.py:213`) — 에이전트 벤치
    config 작성 시 참고.

## 4. 플로우 배선 점검 (요청 항목 6)

- 쓰기: `reasoning_bank.py:138-145` ADD payload에 `embedding_text` 포함 → `memory.py:_apply_one:233-236`이
  `embedding_text`(title\ndescription)를 임베딩해 `memory_type="strategies"`로 vec 등록. **정상 배선**.
- 읽기: `pipeline.search`는 "strategies"에 특례 없음(일반 경로: vec.search → fuse → `doc.get_items(ids,
  "strategies")` → `_DictItem`). notes(링크 확장)/episodes(원문 첨부)와 달리 별도 처리 없음 — 의도대로.
- 단 `_DictItem.render`가 description을 렌더하지 않아 (§1-d) **저장은 되지만 주입에서 소실**.
  `StrategyItem.render()`는 read 경로 미사용(사실상 dead code) — 배선상 유일한 실질 결함.
- `outcome`(success/failure) 필드는 저장만 되고 read에서 미사용 — upstream도 구분 주입하지 않으므로 일치.
- 현재 LoCoMo config에 RB가 없어 write/read 어느 쪽도 실측 경로에 안 탐 — 감사 결론과 일치.

## 5. 권고 (P3 갱신안 — 에이전트 벤치 착수 전)

1. **경험 단위 검색 모드** (P3-13 유지): 태스크 질의를 검색 키로 임베딩 저장, top-1 경험의 아이템 묶음
   주입. upstream의 "top-1 miss → 무주입" 의미론 채택 여부를 config 플래그로 명시. instruction-aware
   질의 임베딩은 임베더 의존이므로 옵션 처리.
2. **주입 포맷**: system 프롬프트 배치 + "explicitly discuss each memory item" 지시문 문자열 추가.
   렌더는 공식 코드 기준(아이템 markdown 전체 = title+description+content)으로 통일하거나, 최소한
   `_DictItem.render`에 description 포함 — 현재의 소실이 가장 시급한 실코드 결함.
3. **프롬프트 정합** (P3-14): content "1-3 sentences"로, description에 "one sentence" 제약 추가,
   judge/gt 이유를 `"The task {succeeded|failed} because: {reason}"`으로 궤적 말미 첨부,
   도메인 페르소나 파라미터화, SI를 system 메시지로 분리.
4. **온도** (P3-12 잔여): 에이전트 벤치 config에 `distill 1.0` 명시 (judge 0.0은 기본값으로 해결됨).
5. **절단 정책**: 6000자 절단 제거 또는 모델 컨텍스트 기반 대폭 상향 (upstream 무절단).
6. MaTTS(PARALLEL_SI ≤5 / SEQUENTIAL re-examine)는 스케일링 실험 시점에 별도 구현.
7. 문서: `fidelity-deep-audit.md` §3 주입 포맷 행의 "우리=render() title+description+content" 서술을
   "_DictItem: title+content (description 소실)"로 교정하고, upstream 경로가 `WebArena/` 하위임을 병기.
