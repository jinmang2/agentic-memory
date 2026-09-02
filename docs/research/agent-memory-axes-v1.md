# 에이전트 메모리의 "네 축" — v1이 덮는 것, 비어 있는 것, 문헌·제품·수요가 가리키는 것 (2026-09-02)

v1(코딩 에이전트의 경험 메모리) 착수 전 조사의 정본. 계기는 한 문장이다: **"지금 구현한 `experience`
organizer는 에이전트 메모리의 네 축 중 하나만 다룬다"** — 그런데 그 "네 축"이 무엇인지 이 리포 어디에도 정의되어
있지 않았다(§1에서 확인). 그래서 질문을 셋으로 쪼갰다.

1. 문헌과 제품이 에이전트 메모리를 어떤 축으로 나누는가, 그중 "넷"인 것은 무엇인가. (§1)
2. 그 축 위에서 우리 스택은 무엇을 갖고 있고 `experience`는 정확히 어디까지인가. (§2)
3. 비어 있는 축마다 **통제된 증거**(무기억·무학습 대조군이 같은 하네스에 있는 것)가 있는가, 제품은 무엇을
   출하하는가, 사용자는 무엇을 요구하는가. (§3–§5)

그 답으로 v1의 범위와 리팩터 순서를 정한다(§6–§7). 조사는 레인 다섯 개(문헌 분류·연구 증거·제품·수요·리포 대응)로
병렬 수행했고, 원 보고서는 세션 스크래치패드에 있으며 이 문서가 그 종합이다. 이 리포의 감사 규칙을 그대로 쓴다:
숫자에는 조건을 붙이고, 자기보고와 통제 측정을 구분하고, 확인 못 한 것은 그렇게 적는다(§8).

---

## 0. 한 장 요약

- **"네 축"의 단일 정설은 없다.** 4성분 분해가 최소 여섯 가지이고 서로 다른 대상을 자른다(내용 종류 / 능력 / 파이프라인
  단계 / 저장 구조). 이 문서는 **기능 4축(작업·에피소드·의미·절차, CoALA 계보)** 을 작업 프레임으로 채택하고, 나머지
  프레임(MemoryAgentBench 4역량, 시스템 4축, LME-V2 5능력, 8월 피드백 공백 A–D)을 그 위에 교차 대응시킨다(§1.3).
- **`experience`는 기능 4축 중 절차 축 하나, 파이프라인 3단계 중 write 하나만 한다.** 산출은 `runbooks` 타입 하나,
  op는 `ADD`·`NOOP`뿐, 읽기 스텝 없음, 원문 궤적은 파사드가 버린다. 게다가 세션을 organizer로 흘려보내는 배선
  (`as_task_trajectory()` 호출자)이 `src/` 안에 0개다(§2.4).
- **문헌의 통제 증거는 비대칭이다.** 에피소드 축은 "원문 궤적에 접근 가능하게 하는 것"에 강한 통제 증거가 있고(LME-V2
  무검색 1.3% → 기성 코딩 에이전트 69.9%), 절차 축은 한 자릿수 pp의 이득이 스킬-과제 정합이 좋을 때만 나오며, 의미
  축은 효용은 미검증이고 위험(오염·유출)만 실증되었고, 작업 기억 축은 벤더 자기보고 외에 통제 증거가 약하다(§3.5).
- **제품은 마크다운·파일시스템·grep으로 수렴했고, 절차 메모리는 Skills로, 의미 메모리는 AGENTS.md류 지시 파일로
  흡수되는 중이다.** 에피소드 축(세션 원문의 관련성 검색)이 제품 공백이고, 작업 기억 축은 기능이 없어서가 아니라 **압축 통제권이
  사용자에게 없어서** 불만이 가장 큰 자리다(§4, §5).
- **수요는 "기억"이 아니라 "이식성"과 "압축 통제권"으로 표현된다.** 정규화하면 메모리 이슈 비중은 오히려 줄고
  (1.92%→0.85%), 최강 신호는 AGENTS.md 지원(6,556 반응), episodic 수요는 아직 언어화되지 않았다(§5).
- **v1의 정직한 베이스라인은 "무기억"이 아니라 "세션 로그 원문 + grep 허용 에이전트"다.** organizer가 그것을 못 이기면
  값을 못 하는 것이고, 그 비교가 성립하려면 원문 궤적 보존과 탐색기 읽기 경로가 먼저다(§6).
- **리팩터는 세 묶음이다**: 궤적 보존·세션 배선(파사드), 기능 축·갱신 정책을 타입 메타데이터로(어휘), 탐색기·지연
  1급값(읽기 경로). OpenRouter 스모크 전에 첫 묶음이 끝나야 스모크가 재는 것이 있다(§7).

---

## 1. "네 축"은 무엇인가

### 1.1 리포 안에는 없다

`docs/`, `README.md`, 메모리 노트를 전수 검색한 결과 "네 축"이라는 표현은 세 곳에 나오는데 전부 다른 뜻이다.

| 위치 | "네 축"의 뜻 | 에이전트 메모리의 분해인가 |
|---|---|---|
| `docs/research/longmemeval.md` §1.2 | LongMemEval 논문의 4 control point (Value / Key / Query / Reading) | 아니다. **읽기 파이프라인**의 설계 변수다 |
| `docs/21-lme-findings.md`, `longmemeval.md` §6 | 같은 이름의 정확도가 릴리스·reader·judge·검색예산 네 축에서 다르다 | 아니다. **비교 가능성**의 자유변수다 |
| `docs/05-api-design.md` §2 | Nemori `fidelity=` 프리셋의 segmenter·episode merge·semantic integration·consolidation 4축 | 아니다. **한 organizer의 쓰기 정책** 스위치다 |

8월 피드백(`docs/_external/2026-08-07-claude-feedback.md`)에는 4분류가 둘 있는데, 그중 어느 것도 "네 축"이라 불리지
않았다: §4 끝의 "메모리 에이전트에 필요한 **네 능력**은 정확한 검색·test-time 학습·장거리 이해·충돌 해소"(출처는
MemoryAgentBench v1 초록, §1.2; 같은 문장이 곧바로 "구조 조직화는 대부분의 팀이 아예 테스트하지 않는 **다섯 번째** 공백"이라고
이어지므로 그 문서조차 넷으로 닫지 않았다), 그리고 §2의 구조적 공백 A–D(기질·읽기 경로·RL·보안, E는 골드 오류). §3의 기능
분류는 **셋**(factual / experience / procedural)이고 작업 기억이 없다. 즉 사용자가 들은 "네 축"은 리포 밖 어휘이며,
아래 후보 중 하나다.

### 1.2 문헌의 4성분 분해 — 후보 여섯 개 (레인: 문헌 분류, arXiv 16건 원문 확인)

| 순위 | 프레임 | 출처 | 성분 | 무엇을 자르나 | `experience`의 위치 |
|---|---|---|---|---|---|
| 1 | **CoALA** | arXiv:2309.02427 (Sumers et al. 2023) | working / episodic / semantic / procedural | 저장 **내용의 인지적 종류** | procedural 하나 |
| 2 | **MemoryAgentBench** | arXiv:2507.05257 (v1 2025-07 / v3 2026-03) | accurate retrieval / test-time learning / long-range understanding / **conflict resolution(v1) → selective forgetting(v3)** | 메모리 시스템의 **역량** | test-time learning 하나 |
| 3 | **Agent Memory: Characterization…** | arXiv:2606.06448 (2026-06) | construction / storage / retrieval / mutability | 시스템 **파이프라인 단계** | construction 하나 — "four axes"를 문자 그대로 쓰는 유일한 논문이지만, 이 프레임에서 organizer가 한 축만 하는 것은 결함이 아니라 정상 |
| 4 | LangChain 문서 | docs.langchain.com (memory) | short-term + semantic / episodic / procedural | CoALA 재포장 | procedural |
| 5 | Anatomy of Agentic Memory | arXiv:2602.19320 (2026-02) | lightweight semantic / entity-centric / episodic-reflective / structured-hierarchical | 저장소 **구조** | 해당 없음(runbook은 어느 구조도 아님) |
| 6 | Rethinking Memory in LLM based Agents | arXiv:2505.00675 (제목 개정됨, 2025-12) | 연구 주제 4종 | 연구 아젠다 | 해당 없음 |

후보에서 뺀 것: **Memory in the Age of AI Agents**(arXiv:2512.13564, 47인 공저)는 forms × functions × dynamics **3축 × 3값**이다.
넷이 아니지만 functions 축의 **factual vs experiential vs working** 구분이 v1 방향과 가장 직접 맞물린다. MemOS(2507.03724)의
plaintext / activation / parametric은 **기질 3축**이고 8월 피드백 공백 A가 이것이다. arXiv:2603.07670(우리 `memory-component-
taxonomy.md`가 채택한 서베이)은 temporal scope / substrate / control policy **3축**이며 write–manage–read 루프를 형식화한다.

실무 도구는 학술 분류를 대체로 채택하지 않았다. LangChain만 3분할을 문서화했고, Mem0는 열거형에 셋을 두고
procedural 하나만 구현("not implemented" 명시), Google ADK·OpenAI Agents SDK·Anthropic memory tool은 인지 종류 없이
저장소 인터페이스만 제공한다. **"업계 표준 4분할"은 실재하지 않는다.**

### 1.3 이 문서의 작업 프레임: 기능 4축, 그리고 교차 대응표

기능 4축(CoALA 계보)을 채택하는 이유는 셋이다. 가장 널리 통용되고, "organizer가 하나만 한다"는 진술이 결함 지적으로
읽히는 프레임이며(시스템 4축에서는 같은 문장이 단순 위치 서술이 된다, §1.2 3순위), 8월 피드백 §3의 갱신 정책 차이(factual=supersede / experience=append-only /
procedural=abstract)를 그대로 얹을 수 있다. 다른 프레임이 담은 구분은 아래 교차 대응표로 옮겨 두어 잃지 않는다.

| 기능 축 | CoALA 정의 | MemoryAgentBench 역량 | LME-V2 5능력 (2605.12493) | 8월 피드백 | Claude Code 자동 메모리 타입 | runbook 필드(계보) | 갱신 정책 |
|---|---|---|---|---|---|---|---|
| **작업(working)** | 현재 결정 주기의 활성 정보 | long-range understanding(부분) | — | (없음) | (없음) | (없음, 컴팩션은 별도) | 세션 내 휘발 |
| **에피소드(episodic)** | 과거 경험·궤적 그 자체 | accurate retrieval | static state recall, dynamic state tracking | experience | project(부분) | (세션 원문이 Codex Phase 1 입력) | append-only, 불변 |
| **의미(semantic)** | 세계·사용자·환경에 대한 사실 | conflict resolution / selective forgetting | environment gotchas, premise awareness(부분) | factual | user, project, reference | reusable_knowledge, preference_signals, references (Codex) | 최신값 우선, supersede |
| **절차(procedural)** | 수행 규칙·스킬·워크플로 | test-time learning | workflow knowledge | procedural | feedback | procedure (AgentRunbook-R), failures (Codex) | 여러 사례에서 추상화 |

읽는 법: LME-V2의 다섯 능력은 기능 4축 중 **에피소드 둘 + 의미 둘 + 절차 하나**로 갈라진다. `experience` runbook은
그중 **workflow knowledge 하나**를 정면으로, environment gotchas를 `failures`로 부분적으로 덮는다. static/dynamic state
recall과 premise awareness는 runbook으로는 닿지 않는다(§3.2 b-1, §6).

---

## 2. 우리가 가진 것 (레인: 리포 대응, 커밋 5ad1dbf 기준, 전부 file:line 확인)

### 2.1 어휘: 타입 14개는 평평하고, 기능 축을 선언한 코드는 없다

`MEMORY_TYPES`(`src/agmem/core/types.py:34-49`) 14개를 기능 축에 배정하면 다음과 같다. **배정은 각 organizer가 인용하는
논문의 성격으로 이 조사가 붙인 것이고, 코드에는 축 개념이 없다.**

| 기능 축 | 타입 (쓰는 주체) |
|---|---|
| 에피소드 | `episodic`(파사드 자신, `memory.py:570-598`), `episodes`(nemori), `pages`(memoryos), `derivatives`(memmachine 앵커) |
| 의미 | `semantic`(nemori·memoryos·mem0·memmachine_profile), `notes`(amem), `entities`·`facts`·`communities`(zep_graph) |
| 절차 | `strategies`(reasoning_bank·gmemory), `experiences`(reasoning_bank), `playbook`(ace), **`runbooks`(experience, v1 신설)** |
| 작업 | (없음) |
| 부기 | `state`(consolidate 커서, 메모리 아님) |

`OpType` 8개(`core/ops.py:20-31`) 중 실제 발행 현황: `TAG`는 **발행자 0**, `LINK`는 amem만, `MERGE`는 nemori만. 8월 피드백
공백 A의 지적(이 어휘는 plaintext 기질에만 완전하다)은 여전히 참이며 v1 범위 밖이다.

### 2.2 커버리지 행렬: 기능 4축 × write / manage / read

| | write | manage (통합·갱신·망각) | read |
|---|---|---|---|
| **작업 기억** | 없음. 파사드에 컨텍스트 개념 자체가 없다 | 없음. 컴팩션·요약 트리거 없음 | 부분. `recent_context()`(`organizers/base.py:188`)는 인자 없는 문자열 반환이고 구현은 MemoryOS·MemMachine STM 둘, 호출자는 벤치 둘뿐(`bench/locomo.py:663`, `bench/longmemeval.py:666`). 훅·MCP 경로에서는 아무도 부르지 않는다 |
| **에피소드** | 있음. 모든 원문은 불변 `episodic`으로 저장(`memory.py:570-598`, `Episode`는 frozen). **단 태스크 궤적은 저장되지 않는다**(`memory.py:495-498`) | 부분. Nemori 서사 분절+MERGE, MemoryOS 페이지·eviction, MemMachine 앵커 | 있음. dense+BM25 RRF(`retrieval/pipeline.py:126-180`), 원문 첨부·앵커 되돌림. **JIT 파일 탐색(rg/find)은 없다** — `src/agmem` 안 `subprocess` 호출 지점 3곳(`capabilities/detect.py:91`, `bench/stamp.py:52`, `hooks/daemon.py:133`) 전부 무관 |
| **의미** | 있음(가장 두꺼움) | 있음. Mem0 ADD/UPDATE/DELETE/NOOP, Zep bi-temporal INVALIDATE, A-Mem 진화·LINK, 커서 기반 지연 consolidate | 있음. 타입별 채널·BFS·링크 확장·그래프 리콜·노드 거리 리랭크 |
| **절차** | 있음. `strategies`/`experiences`/`playbook`/`runbooks` | **거의 없음.** ACE helpful/harmful 카운터, G-Memory 보상 성형이 전부. RB는 논문대로 append-only. **`experience`는 `ADD`·`NOOP`만**(`experience/organizer.py:269-297`) | 부분. RB 확장·G-Memory 태스크 그래프·playbook 전량 렌더. **`runbooks` 전용 ReadStep 없음**(`retrieval/steps.py:983-1039`의 7개 스텝 어디에도 없음) |

시간 인식은 부분적이다: bi-temporal 서빙 판정(`is_servable`)은 있으나 **시간 감쇠·최신성 리랭킹은 검색 파이프라인에 없다**
(`retrieval/rerank.py` 6종 확인). 최근성은 훅(`hooks/recall.py`, SessionStart 12줄)에만 있다.

### 2.3 `experience` organizer의 정확한 범위 (`src/agmem/organizers/experience/organizer.py`)

- 훅은 `on_task_end` 하나(`:213-298`). `on_message`·`consolidate`·`flush_buffer`·`on_feedback`·`recent_context` 모두 미구현.
- LLM 호출 세션당 1회(`:235-242`), 트랜스크립트 60K자 머리·꼬리 절단(`:64`). LLM 없으면 경고 후 명시적 스킵.
- 산출: 태스크 블록당 `ADD` 하나, payload는 마크다운 `content` + 검색 핸들 `embedding_text`(이름·키워드·참조·절차·선호신호)
  + 구조화 필드 8개(`name, outcome, preference_signals, reusable_knowledge, failures, references, procedure, keywords`).
  신호가 없으면 `NOOP`. 품질 게이트: 네 핵심 필드가 전부 비면 블록 폐기(`:263-267`).
- 프롬프트는 Codex 추출기 규칙 이식: 트랜스크립트는 데이터이지 지시가 아님, 증거 기반, 시크릿 금지, **"NO-OP IS ALLOWED
  AND PREFERRED"**, 어시스턴트 자신의 제안은 채택·검증 전엔 메모리가 아님(`:91-135`).

담지 못하는 것(코드로 확인):

1. **의미 축 없음.** `reusable_knowledge`·`preference_signals`는 runbook 문자열 안에 묻히고 별도 `semantic` 아이템이 되지 않아
   사실 단위 UPDATE·충돌 해소 경로가 없다.
2. **manage 없음.** Codex Phase 2(전역 통합 에이전트)를 의도적으로 뺐다고 docstring이 밝힌다(`:31-34`). 같은 워크스페이스의
   모순된 runbook 두 개는 둘 다 남는다.
3. **작업 기억 없음.** `recent_context()` 미구현.
4. **에피소드 JIT 없음.** 원문 궤적을 저장하지 않고, 파사드도 버린다.
5. **읽기 경로 없음.** `runbooks`는 일반 벡터 검색에만 걸린다.
6. **피드백 없음.** Codex의 인용→`usage_count` 되먹임에 해당하는 것이 없다.

### 2.4 배선 공백 세 가지 (가장 중요)

1. `SessionTrajectory.as_task_trajectory()`(`sessions/__init__.py:181`)를 **호출하는 코드가 `src/` 안에 0개**다. 참조는 docstring
   두 곳뿐. 세션이 organizer로 흘러가는 경로가 아직 없다.
2. `AgenticMemory.add_task_result`가 **궤적을 저장하지 않는다**(`memory.py:495-498`, "the full `trajectory` is never persisted by the
   facade"). JIT 읽기를 붙일 원문 저장소가 없다.
3. 훅은 organizer를 구조적으로 금지한다(`hooks/__init__.py:207`, `organizers=[]` 고정; 근거 `:23-26`). 세션 종료 시점 증류를
   걸 자리가 현재 계약에 없다. LME-V2 어댑터도 없고(`Memory.insert/query` 래퍼 부재), reader만 레지스트리에 있다
   (`bench/registry.py:44-48`).

### 2.5 제품층이 이미 서 있는 자리

훅 셋 중 읽기 둘(`recall` SessionStart 최근성 12줄 / `recall_prompt` UserPromptSubmit top-5)은 `hookSpecificOutput.additionalContext`
**주입**이고 파일이 아니며(`hooks/__init__.py:92`), `capture`는 쓰기 전용으로 컨텍스트를 전혀 내지 않는다(`hooks/capture.py:14-17`, 한 훅은 한 방향). 데몬은 `agmem-mcp --transport http`
프로세스 그 자체이다. `docs/research/product-memory-landscape.md` §5.2가 주장하는 "유일한 칸"은 쓰기 0콜 + 로컬 임베딩 + 두 호스트 한
스토어 + 측정 하네스이며, organizer 교체 가능성은 "연구 플랫폼의 가치이지 제품 가치가 아니다"라고 스스로 적는다.

---

## 3. 축별 연구 증거 (레인: 연구 증거, arXiv ID 전부 원문 확인, 자기보고/통제 구분)

판정 기준. "에이전트 설정"은 대화형 QA가 아니라 환경과 상호작용하며 과제를 푸는 설정을 뜻한다. "통제된"은 무기억·무학습 대조군이
**같은 하네스·같은 모델**에서 짝지어 측정된 것.

### 3.1 작업 기억 / 컨텍스트 관리 — 통제 증거 약함

| 항목 | 출처 | 주장·수치 | 대조군 | 성격 |
|---|---|---|---|---|
| Sleep-time compute | 2504.13171 | 동일 정확도에 테스트 연산 ~5×↓, 다중질의 시 질의당 비용 2.5×↓ | 연산량 일치 비교 | 수학 과제 주력, 자체 측정 |
| Anthropic context engineering | 블로그 2025-09-29 | compaction·구조화 노트·서브에이전트 격리·JIT 검색 | 없음 | 방법론 문서, 수치 없음 |
| Anthropic memory tool + context editing | 제품 발표 2025-09-29 | +39%(둘 다), +29%(editing만), 토큰 84%↓ | 비공개 | **벤더 내부 평가, 재현 불가** |
| Manus context engineering | 블로그 2025-07-18 | KV 캐시 적중률 최우선, 파일시스템을 컨텍스트로, todo.md 재낭독 | 없음 | 운영 경험담 |
| ACON | 2510.00615 | 토큰 26–54%↓, 소형 모델 최대 +46% | **무압축 대조군 없음** | 압축끼리 비교 |
| 재귀 압축의 실행 불안정성 | 2608.06503 | 재귀 압축이 차단 행동·탐색 반복·런 간 불안정을 키움 | 쌍 폐루프 연장(TRACE) | **부정 결과**, 저자도 preliminary |
| Cartridges / at Scale | 2506.06266, 2606.04557 | KV 카트리지 메모리 38.6×↓; 카트리지 혼합은 순진하게 하면 붕괴 | — | 가중치 수준, API 모델엔 비적용. "섞으면 붕괴"는 세션 간 지식 병합 위험의 유비 |

판정: 압축 기법 자체의 통제 효용 증거는 약하고 반대 증거가 있다. 다만 **"에이전트가 파일을 직접 관리하게 하라"**는
명제에는 교차 시나리오 증거가 있다(AutoMEM 2606.04315, §3.2). 이것은 압축 기법이 아니라 **통제권 배치**의 문제다.
SleepGate(2603.14517)는 4층 793K 파라미터 토이 트랜스포머 실험이라 에이전트 근거로 쓸 수 없다.

### 3.2 에피소드 (원문 궤적) — 통제된 긍정 증거 있음, 단 "접근 가능성"의 가치

| 항목 | 출처 | 수치 | 대조군 |
|---|---|---|---|
| **LongMemEval-V2** | 2605.12493 | small: 무검색 **1.3** / slice 42.8 / slice+notes 51.0 / AgentRunbook-R 58.6@27s / **기성 Codex 69.9@177s** / AgentRunbook-C 74.9@108s | **무검색 1.3% 명시** |
| GAM (JIT vs AOT) | 2511.18423 | HotpotQA 224K F1: GAM 64.56 / RAG 51.84 / **Mem0 31.74**; RULER multi-hop 93.2 vs 0–53.8 | long-LLM·RAG를 memory-free로 명시 | 자체 측정 |
| MemoryArena | 2602.16313 | LoCoMo 포화 에이전트가 상호의존 에이전트 과제에서 무너짐 | 미명시 | 수치는 초록에 없음 |
| **Code Isn't Memory** | 2606.22417 | 하네스·모델 고정, 3시드, 누출 감사, 인덱스 절제 → 통계적으로 분리된 resolve 이득, solve당 비용↓ | **인덱스 제거 동일 하네스** | 가장 엄격한 설계, 우리가 따라야 할 본보기 |
| Memory Transfer Learning | 2604.14004 | 6 코딩 벤치 평균 +3.7%; **저수준 궤적은 부정 전이**, 고수준 통찰만 일반화 | 4 표현 간 비교 | |
| Subtask-level memory | 2602.21611 | SWE-bench Verified **+4.7pp**(vanilla 대비), Gemini 2.5 Pro +6.8pp; 인스턴스 단위 기억은 입도 불일치로 오검색 | **vanilla + 인스턴스 기억 둘 다** | |
| SWE-ContextBench | 2602.08316 | 잘 요약·검색된 경험은 정확도↑ 비용↓; **걸러지지 않은 컨텍스트는 이득 없거나 부정적** | | |
| SWE-Exp | 2507.23361 | SWE-bench Verified 73.0 | **짝지은 무기억 수치 없음** | 하네스 차이 섞임 |
| AutoMEM | 2606.04315 | 8 메모리 시스템 + 파일 하네스, 5 시나리오. **평문 파일을 도구로 자율 관리하는 하네스가 교차 순위 1위** | | 수치는 초록에 없음 |
| Letta filesystem | 블로그 2025-08 | gpt-4o-mini + 파일 저장만으로 LoCoMo 74.0 (Mem0 그래프 68.5) | | 벤더 측정 |
| **EvoMemBench** | 2605.18421 | 15개 메모리 방법 vs 강한 장문 컨텍스트, 표준 프로토콜. **장문 컨텍스트가 여전히 매우 경쟁력 있고, 모든 설정에서 일관되게 이기는 단일 메모리 형태는 없다** | 표준 프로토콜 | 우리 결론과 가장 직접 맞물림 |

판정: **읽기 경로가 68pp를 움직이고, 설계된 메모리가 기성 코딩 에이전트를 이기는 폭은 +5.0pp(small)/+1.4pp(medium)다.**
이 세 건(LME-V2, Code Isn't Memory, Subtask-level)이 증명하는 것은 "과거 궤적에 접근 가능하게 하는 것"의 가치이지
"과거 궤적에서 지식을 증류해 저장하는 것"의 가치가 아니다. 우리 v0 결론(`docs/22`)과 같은 방향의 외부 증거가 셋 늘었다.

### 3.3 의미 / 사실 — 효용 미검증, 위험 실증 (비대칭)

효용 쪽:

- Mem0(2504.19413)는 전체 컨텍스트 대조군을 포함하지만 이득은 LoCoMo에 묶여 있고, GAM의 224K에서는 RAG보다 20점 낮다.
- Zep(2501.13956)의 주 비교군은 MemGPT 하나, DMR 이득 1.4pp. 우리 기존 결론(읽기 레시피가 operating point, `zep-graphiti.md`)과 모순 없음.
- MemoryAgentBench(2507.05257): **네 역량을 모두 숙달한 방법은 없다.** Mem0 검색 28%, Cognee 33.5%로 단순 임베딩 RAG(83%)보다
  크게 낮음. conflict resolution 다중 홉은 모든 시스템 최대 6%.
- 갱신 충돌은 미해결: 신선도 규칙이 **명시된** 데이터셋(FactConsolidation)에서도 22개 시스템 전부 multi-hop 최대 7%(2606.01435).
  같은 논문이 증거 추출과 정책 실행을 분리하면 262K에서 27–41%까지 오르고, 이득 대부분이 분리 자체에서 온다고 보고.
- 제어 평면(2606.15903): supersede/release/purge를 **변형 시점 훅**에 두면 의도 인식 삭제 78–85% 회복, 전체 91.7–93.2%,
  385케이스당 $0.17. 회상 경로는 그대로.

위험 쪽:

| 항목 | 출처 | 수치 |
|---|---|---|
| CIMemories (contextual integrity) | 2511.14937 | 속성 단위 위반 최대 **69%**; GPT-5 과제 1→40개에 0.1%→9.6%, 같은 프롬프트 5회 25.1%; 프라이버시 프롬프팅은 과일반화로 실패 |
| AgentPoison | 2407.12784 | 오염률 <0.1%로 ASR 80%+, 정상 성능 영향 <1% |
| MINJA | 2503.03704 | **질의와 출력 관찰만으로** 메모리 주입 |
| MemMorph | 2605.26154 | 레코드 **3개**로 최대 85.9% |
| 메모리 오염 체계 연구 | 2606.04329 | 쓰기 채널 4·취약점 9·공격 6종; **더 공격적으로 쓰는 설계일수록 취약**, 기존 프롬프트 인젝션 방어는 미커버 |
| TMA-NM | 2606.24322 | 내용·계보 기반 방어는 세탁(에이전트 자신의 요약 등)으로 최대 68% 뚫림 |
| OWASP ASI06 | 2025-12-09 | Memory & Context Poisoning 공식 등재; 세션 리셋으로 사라지지 않고 며칠–몇 주 뒤 발화 |

판정: 의미 축은 **"효용은 미검증, 위험은 실증"** 상태다. 코딩 세션 로그에는 이슈 본문·의존성 README·웹 결과가 섞이므로
쓰기 채널 자체가 간접 주입 경로다. 8월 피드백 공백 D가 문헌에서 더 무거워졌다.

### 3.4 절차 / 스킬 — 부분적, 한 자릿수 pp, 정합 조건 의존

| 항목 | 출처 | 수치 | 대조군 |
|---|---|---|---|
| **ReasoningBank** | 2509.25140 | WebArena No Memory 40.5→48.8(flash, +8.3) / 46.7→53.9(pro) / 41.7→46.3(claude); SWE-bench Verified 54.0→**57.4**(+3.4) | **No Memory 명시** |
| Dynamic Cheatsheet | 2504.07952 | Game of 24 10%→99%, AIME 2×, GPQA +9 | 무기억 명시 | 이득 대부분이 **검증된 코드 조각 재사용** |
| AFTER (procedural memory) | 2606.23127 | 정제 1회에 +3.7~6.7점, 교차 모델 73.1%; 일부 스킬은 직무 특화되어 전이 시 효과 상실 | 4 통제 설정 | 절차 메모리 정면 통제 평가 유일 |
| Training-Free GRPO | 2510.08191 | AIME24 80.0→82.7, AIME25 67.9→73.3, WebWalkerQA 63.2→67.8, 학습 비용 ~$18 | 무학습 명시 | runbook과 개념적으로 가장 가까움 |
| **스킬 사용의 취약성** | 2604.04323 | 34,000개 스킬에서 자율 검색하게 하면 설정이 현실적일수록 이득이 일관되게 감소하고, **가장 어려운 시나리오에서는 통과율이 무스킬 베이스라인에 근접**; 질의 특화 정제로 Terminal-Bench 2.0 57.7→65.5 | 무스킬 | v1에 가장 뼈아픈 결과 |
| SkillJuror | 2606.11543 | 접촉 자원 1.18→3.85(3배)인데 결과 +4.1%(410 시행 중 17개) | 짝지은 시행 | **행동은 크게, 결과는 조금** — 우리 트랙5(구체성 9.4→52.8%인데 정확도 하락)와 같은 형태 |
| 대규모 스킬 평가 | 2606.17819 | 500 스킬·1,000 과제·19 구성; 모델별 지시 충실도 차이가 이득 차이로 | | |
| AWM, SkillWeaver | 2409.07429, 2504.07079 | 상대 +24.6~51.1%(AWM Mind2Web·WebArena), +31.8~54.3%(SkillWeaver) | 대조군 정의 불명확 | 스킬이 환경에 맞게 합성된 이상 조건 |
| Voyager | 2305.16291 | 고유 아이템 3.3×, 이동 거리 2.3×, 마일스톤 15.3× 빠름 | 스킬 라이브러리 절제 초록에 없음 | 코드 스킬 라이브러리의 원형 |
| ACE | 2510.04618 | 에이전트 +10.6%, 금융 +8.6% | **무학습 대조군 없음** | 공개 재현 실패 보고 없음 → **우리 페어링 런(441, base 48.24 vs online 46.71 / nodedup 48.98 / retry 45.80)이 현재 가장 강한 부정 증거** |
| Auto-Dreamer | 2605.20616 | GRPO 오프라인 통합기; ScienceWorld에서 베이스라인 +7점, 활성 메모리 은행 1/12, ALFWorld·WebArena 재학습 없이 전이 | 고정·RL·프롬프트 메모리 베이스라인 | 자체 측정, RL은 우리 하드웨어 밖 |
| ExpeL, MemSkill, MemRL, Mem-α, Memory-R1 | 각 ID | 초록 수준 정량치 없음 또는 자체 측정 | | 인용 시 본문 표 확인 필요 |

판정: 이득은 실재하되 **한 자릿수 pp이고, 스킬-과제 정합이 오라클에 가까울 때만** 난다. 코딩 세션이 쌓이면 항목이
수만 단위가 되므로 우리 v1은 2604.04323의 "현실 조건" 쪽에 있다. **저장(쓰기)이 아니라 검색·정제(읽기)가 병목이다.**

### 3.5 축별 판정 요약

| 기능 축 | 에이전트 설정 통제 증거 | 핵심 근거 | v1 함의 |
|---|---|---|---|
| 에피소드 | **있음** (접근 가능성) | LME-V2 1.3→69.9, Code Isn't Memory, Subtask +4.7pp | 원문 보존 + 탐색기가 1순위 |
| 절차 | **부분** (한 자릿수 pp) | RB +3.4~8.3, AFTER +3.7~6.7, 2604.04323 붕괴 | runbook은 유지하되 읽기·정제를 측정 대상으로 |
| 작업 기억 | **약함** | 벤더 자기보고, ACON 무대조군, 2608.06503 부정 | 압축 기법 대신 통제권(파일 관리)으로 접근 |
| 의미 | **효용 통제증거 없음 / 위험 실증** | MemoryAgentBench, 2606.01435 7%, CIMemories 69%, MemMorph 3건 85.9% | 갱신은 결정적 신호에, 출처 결속·문맥 게이팅은 설계 시점에 |

---

## 4. 제품 지형: 코딩 에이전트가 출하하는 메모리 (레인: 제품, 공식 문서·체인지로그·저장소 직접 확인, `[2]`=2차)

`docs/research/product-memory-landscape.md`(2026-09-02, Codex 소스 해부 + Claude Code 관찰 + 애드온 5종)를 이 문서는 대체하지 않고
**기능 4축으로 재투영**하고 제품 범위를 20여 종으로 넓힌다.

### 4.1 세 줄 요약

1. **2026년은 자동 메모리(모델이 스스로 쓰는 의미 계층)가 호스트 기본 기능으로 진입한 해다.** Copilot(2026-01 프리뷰, 03 Pro 기본), Claude Code
   (2026-02, v2.1.59 `[2]`), Qwen Code(v0.16.2 기본), Codex, Warp, Augment. **정확히 반대로 간 곳은 Cursor 하나**(2.1에서 Memories 제거, 공식
   Rules 문서에 단어 자체가 없음; 제거 사유의 1차 출처는 미발견).
2. **저장 형태는 거의 전부 마크다운 + 프롬프트 주입이다.** 1군 호스트 중 자체 메모리 저장소로 벡터·그래프 DB를 쓰는 곳을 하나도 확인하지 못했다.
   서버 사이드로 간 Copilot·Warp·Devin·Amp도 벡터라는 문서 근거가 없다(Amp는 PostgreSQL 명시). 검색은 상시 주입 아니면 파일명·설명 기반 JIT 읽기.
3. **어떤 제품도 자기 메모리 기능의 수치를 발표하지 않았다.** 애블레이션·리콜·비용 회계 전무. 숫자를 내는 쪽은 애드온 벤더뿐이고 그 숫자(LoCoMo)는
   mem0↔Zep 상호 반박 중(mem0#3944, zep-papers#5)이라 단서 없이 인용 불가.

### 4.2 기능 4축 매트릭스 (핵심 제품)

| 제품 | 작업(W) | 에피소드(E) | 의미(S) | 절차(P) | 특기 |
|---|---|---|---|---|---|
| **Claude Code** | 자동 컴팩션, `/compact`, `/autocompact <n>`; 압축 후 루트 CLAUDE.md 재주입, 호출 스킬 5,000토큰까지 재주입 | 트랜스크립트 저장·재개·`/rewind`; `cleanupPeriodDays` 후 삭제되나 **메모리 디렉터리는 삭제 제외** | Auto memory(`~/.claude/projects/<p>/memory/`, 타입 user/feedback/project/reference, MEMORY.md **첫 200줄/25KB만 로드**, 주제 파일은 JIT), 저장소 단위·**머신 로컬**; CLAUDE.md 계층 | Skills; **`/verify`·`/run-skill-generator`가 성공한 실행 레시피를 저장소에 커밋되는 SKILL.md로 자동 기록**(v2.1.200+) | AGENTS.md를 **읽지 않는다고 문서에 명시한 유일한 제품**, 대신 `/init`·`/import`(v2.1.213+)로 남의 파일을 가장 넓게 흡수 |
| **Codex** | 컴팩션 `[2]` | `codex resume`/`fork`, rollout 파일 | Memories(`~/.codex/memories/`), 백그라운드 2단계·**2모델**(extract/consolidation), **전역 스코프**(프로젝트 단위 아님), 레이트리밋 잔량 낮으면 생성 스킵, `disable_on_external_context` | AGENTS.md | 문서 스스로 "recall layer일 뿐 규칙의 출처가 아니다"; `/import`가 Claude Code·Cursor 설정+최근 30일 50대화 `[2]` |
| **Cursor** | 미확인 | 미확인 | Rules 4종 + AGENTS.md("simple alternative") | Rules 적용 모드 4종 | Memories 제거(1차: 문서에서 소멸; 2차 출처는 상충) |
| **Antigravity** | 문서에 컴팩션 페이지 없음 | `/resume`, Artifacts | Rules(`~/.gemini/GEMINI.md`, `.agents/rules`, 12,000자) | Workflows(12,000자), **사용자가 요청하면 에이전트가 대화 이력에서 워크플로 생성** | 출시 블로그는 knowledge base를 "core primitive"라 했는데 **2.0 문서 내비게이션에 knowledge/memory 페이지 0개** — 제거인지 미문서화인지 판정 불가 |
| **Copilot coding agent** | VS Code/CLI `/compact` `[2]`, 요약 후 유실 불만 이슈 | — | **Copilot Memory**: 서버 사이드(추론), 저장소 스코프 자동 캡처, **28일 만료**, 사용 전 코드베이스 대조 검증, 저장소 사실은 **모든 기여자에게 보임** | `applyTo` instructions | 2026-01-15 프리뷰 → 06-02 Enterprise |
| **Windsurf → Devin Desktop** | — | — | Memories(`~/.codeium/windsurf/memories/`, 워크스페이스·머신 로컬); **벤더 문서가 자동 Memories보다 Rules/AGENTS.md를 쓰라고 권고** | Workflows(수동 호출만, 12,000자) | 2026-06 리브랜딩 `[2]` |
| **Devin** | 미공개 | Worklog·세션 재개 | Knowledge: **자동 제안 + 사람 승인 게이트**, 3단 스코프 | Playbooks(REST API) | UI/API 객체, 저장소 마크다운 아님 |
| **Warp** | — | — | Agent Memory(Research Preview): 대화 종료 후 비동기 추출, Personal/**Agent-owned**/Team 3종 스토어, **충돌 해소·출처 추적·주입 시 인용을 문서화한 유일한 제품** | Notebooks·Workflows | AGENTS.md 선호, WARP.md 우선 |
| **Augment** | Context Engine | — | Memories: 자동 포착 + **Memory Review 승인 게이트**("nothing gets stored without sign-off"), 개인→워크스페이스 Rule 승격 | Rules 3모드, 문자 상한 24,576/49,512 | |
| **OpenHands** | `LLMSummarizingCondenser`(`max_size=120`, `keep_first=4`) | **append-only EventLog**, LLM에 보내는 View만 압축 | AGENTS.md 전체 상시 주입, microagents | 키워드 트리거 microagents → `.agents/skills/` | **W와 E를 구조적으로 분리한 유일한 사례** |
| **Amp** | **자동 컴팩션 제거(2025-10-23) → Handoff**(보조 모델이 새 스레드용 프롬프트 초안) | 스레드 서버 영속(PostgreSQL) | AGENTS.md만 | — | 3자는 다시 컴팩션한다고 서술(상충) |
| **Qwen Code** | `/compress`·`/compress-fast`(규칙 기반, LLM 없음), 자동 압축 warn/auto/hard 3단 | 체크포인트, `--resume`, **`@`멘션으로 과거 세션 요약 주입** | 자동 메모리 기본(v0.16.2), `pinned/`, `/dream` 정리; **Team Memory opt-in + 쓰기 전 시크릿 스캔** | `/learn`, `/loop` | 오픈소스 중 가장 완성도 높은 자동 메모리 스택 |
| **Kiro** | 미확인 | 미확인 | Steering 4모드(`always/fileMatch/manual/auto`), CLI는 모드 무시하고 전부 로드 | Specs 3단계, Agent hooks | |
| **Zed** | 자동·수동 `/compact` | Thread History, @멘션 | 파일명 9종 인식하되 **첫 매치만**(`.cursorrules`가 AGENTS.md를 가리는 함정) | Skills `[2]` | 자동 기록 없음, 가장 순수한 사용자 큐레이션 |
| **Cline / Roo** | Auto Compact(비Anthropic은 규칙 폴백) / Condensing(원본 보존) | 태스크 영속·shadow git | `.clinerules` / `.roo/rules` + AGENTS.md | **Memory Bank는 제품 내장이 아니라 커뮤니티 프롬프트 관례** | 컨덴싱 롤아웃 반발 이슈 다수 |
| Gemini CLI | `/compress` | 체크포인트(opt-in) | GEMINI.md, `save_memory` append | — | **2026-06-18 개인 사용자 처리 중단**(엔터프라이즈 유지), 저장소 미아카이브 |
| Kimi / Trae / Factory / Junie / Replit / Aider / Mistral Vibe | (원 보고서 §1 2군 참조) | | AGENTS.md 계층 각자 | | Kimi의 "Memory System"은 마케팅뿐, 이슈 #1283 미응답; Factory는 "메모리는 직접 만들어 쓰는 것"; Junie CLI는 첫 실행 시 타 에이전트 파일 임포트 제안 |

애드온(mem0/OpenMemory, Letta MemFS, Zep/Graphiti, Supermemory, Basic Memory, claude-mem, gbrain, 공식 memory MCP)은 원 보고서 §2. 벡터·그래프는
전부 이쪽에만 있고, claude-mem만 훅 5종으로 **완전 자동** 캡처+요약 재주입을 한다. "OpenMemory"는 무관한 프로젝트 3~4개가 같은 이름을 쓴다.

### 4.3 매트릭스를 가로지르는 경향 8개

1. 마크다운 우선·벡터 없음(1군 전부). 2. 지시 파일이 AGENTS.md로 수렴(네이티브 12개 제품; 자체 규약 고수는 Claude Code·Kiro·Antigravity·Qwen·Replit·
Aider·Cline). 3. 자동 메모리가 호스트 기본(Cursor만 역행). 4. 자동 기록이 **침묵 기록**(Claude Code·Codex·Qwen·Warp·Copilot·Windsurf)과 **승인 게이트**
(Augment·Devin)로 갈림. 5. 호스트 간 이관이 기능으로 출시(Codex·Claude Code `/import`, Junie, Augment, Antigravity CLI, Zed, Factory) — **단 대상은 전부
손으로 쓴 계층**. 6. 절차 메모리가 "스킬" 마크다운 폴더로 수렴, 파일 상한이 12,000자 부근으로 모임. 7. **컴팩션은 보편적이면서 가장 미움받는 기능**
(Cline·Roo·Copilot 반발, Amp는 제거). 8. 컨텍스트 예산을 문자·이벤트 수로 못 박는 관행(200줄/25KB, 80,000/40,000자, 120 이벤트 등).

### 4.4 어떤 제품도 출하하지 않은 것 (v1이 겨냥할 빈칸)

| # | 공백 | 기능 축 | 가장 가까운 예외 |
|---|---|---|---|
| 1 | **자동 메모리의 크로스 호스트 이식성** | S | 없음. 이관 기능은 전부 손으로 쓴 계층 대상 |
| 2 | 자동 메모리의 크로스 머신 이식성 | S | 서버 사이드 4곳(Copilot·Warp·Devin·Augment)은 벤더 종속 |
| 3 | **메모리 효과 수치 공개** (애블레이션·리콜·오주입률) | 전부 | 없음 — **가장 큰 빈칸** |
| 4 | 메모리 쓰기 비용 회계 | 전부 | Codex 레이트리밋 게이트, Warp "런타임 비용 없음" |
| 5 | **에피소드 메모리를 관련성으로 검색** | E | Qwen `@`과거세션(사용자 지목), 애드온 claude-mem |
| 6 | **실패로부터 자동으로 플레이북 학습** | P | Claude Code `/verify`(성공만), Antigravity(요청 시), Devin(승인) |
| 7 | 망각·감쇠 | manage | Copilot 28일, Qwen `/dream`; Claude Code는 오히려 삭제 제외 = 무한 누적 기본 |
| 8 | 자동 메모리의 팀 공유 기본 경로 | S | Qwen team-memory(opt-in+시크릿 스캔), Warp Team, Copilot 저장소 사실(프라이버시 문제) |
| 9 | 충돌 해소 | manage | Warp만 문서화 |
| 10 | 주입 출처 추적·인용 | read | Warp만; Kiro·Copilot 부분 |

우리 `product-memory-landscape.md` §5.2의 "유일한 칸"(쓰기 0콜·로컬 임베딩·두 호스트 한 스토어·측정 하네스)은 이 표의 #1·#3에 정확히 대응한다.
반면 우리가 **없는 것** 중 이 표가 부각하는 것은 #5(에피소드 관련성 검색 — 벡터는 있으나 세션 원문 저장이 없음)와 #6(실패 학습 — runbook의 `failures`
필드가 후보)이다. 그리고 #7·#9는 §3.3의 결론(결정적 신호·변형 시점 훅)과 합쳐 읽어야 한다.

---

## 5. 수요와 트렌드 (레인: 수요, GitHub 수치는 `gh api`로 2026-09-02 직접 계량)

### 5.1 통념을 뒤집는 것 셋

1. **"메모리 수요 폭증"은 정규화하면 성립하지 않는다.** anthropics/claude-code에서 제목에 `CLAUDE.md`·`auto memory`가 든
   이슈는 52→633건(12배)이지만 전체 이슈가 2,702→54,697건(20배)이라 비중은 1.92%(2025 H1) → 1.11% → 1.16% → **0.85%**
   (2026-07~09)로 감소.
2. **"memory" 키워드 카운트는 오염되어 있다.** 제목에 `memory`가 든 1,383건 중 반응 상위 10개 가운데 7개가 RAM 누수·OOM.
3. **최강 신호는 기억이 아니라 이식성이다.** 저장소 전체 반응 1위가 `Support AGENTS.md`(#6235, **6,556 반응**/389 댓글),
   2위(3,286)의 두 배. 재발 요청 #34235(118 반응)도 열려 있음.

### 5.2 유형별 수요

| 축 | 대표 신호 | 관측 |
|---|---|---|
| 작업 기억 | codex #4106 압축 파라미터 통제(118 반응), claude-code #27242 압축 후 이전 컨텍스트 열람 불가(84), #6354 압축 후 CLAUDE.md 망각(30, 1년 open) | 반응 수 최대. 단 요구는 "기억하라"가 아니라 **"압축을 통제하게 해달라"** |
| 의미 | #6235, #34235, #14467 조직 공유 CLAUDE.md(43), #2544 규칙 무시(45, 15개월 open) | 절대 1위. 형태는 **파일 규약의 이식성**이지 검색 기억이 아님 |
| 절차 | Skills로 흡수 중. 제목 `skill` 이슈 **2,153건** vs `auto memory` 135건. 공식 문서도 다단계 절차는 skill로 옮기라고 라우팅 | |
| 에피소드 | #25739 기기 간 이식(39), #14228 claude.ai 메모리 연동(50), #59111 TTL·salience(3) | **가장 약함.** 고반응 이슈 없음 — 사용자가 이 요구를 언어화하는 법을 모를 가능성 있음(추론) |

### 5.3 운영자·엔터프라이즈

- OWASP Agentic Top 10 ASI06(2025-12-09) 공식 등재. Claude Code 소스 유출(2026-03-31) 후 "클론 저장소의 CLAUDE.md가 압축을 통과해
  세탁된다"는 권고가 보안팀 체크리스트에 포함(2차).
- 시크릿: #59094(.env 무리댁션 읽기), GitGuardian 2026: Claude Code 관여 커밋 유출률 3.2% vs 기준 1.5%(2차, 원보고서 미검증).
- SAP 조직 메모리 배포 논문(2608.00122): 큐레이션 메모리 1,144건·링크 7,863개를 보고하면서도 **검색 적합도·과제 성공률은 측정
  못 했다고 인정**. 우리 v1의 자리(통제 측정)가 여기서 비어 있다.
- 메모리가 청구 항목이 됨: Vertex AI 2026-01-28부터 1,000건당 $0.25, Bedrock AgentCore 별도 미터(2차). Cline Memory Bank 사용자
  월 $30→$230(Discussion #1727).
- 감사 가능성: Claude Code 메모리는 평문 마크다운·`/memory`·`modified` 타임스탬프(v2.1.214+). 그러나 #82056(인덱스가 통째로
  로드됐는지 세션이 알 수 없음, 47 댓글), #47959(Auto Dream이 하루 23파일 무동의 삭제).

### 5.4 트렌드 (검증됨 / 2차 구분)

| 트렌드 | 근거 | 판정 |
|---|---|---|
| 컨텍스트 엔지니어링 | Anthropic 2025-09(select/compress/order/isolate/format), LangChain 4전략, Manus | 확립 |
| **마크다운·파일시스템이 기본 기질** | Letta MemFS 기본값에 벡터 인덱스 없음, Claude Code 자동 메모리 = 평문 + MEMORY.md 인덱스(첫 200줄/25KB만 로드) | 확립 |
| AGENTS.md 표준화 | 2025-08 공개 명세, 2025-12 Linux Foundation AAIF 기증, 20+ 도구; "60,000개 저장소"는 2차·방법론 미검증 | 확립(수치 미검증) |
| **AGENTS.md 효율 1차 측정** | arXiv:2601.20404 (ICSE JAWs 2026): 저장소 10개·PR 124개, 실행시간 중앙값 −28.64%, 출력 토큰 −16.58%, 완료 행동 동등 | 검증 |
| Skills = 배포 가능한 절차 메모리 | agentskills.io 명세(2025-12-18), 32개 도구 호환, anthropics/skills 62K★(2차) | 확립 |
| 배경 통합(sleep-time/dreaming) | Letta sleep-time; Claude Code **Auto Dream**은 `autoDreamEnabled`·`/dream` 실재하나 공식 문서·llms.txt에 미등재(직접 확인) | 존재 확실, 문서화 안 됨 |
| 벤치 이동 | LoCoMo → LongMemEval → LME-V2 / MemoryAgentBench(ICLR 2026) / MemoryArena(ICML 2026) | 검증 |
| 신뢰 하락 | Stack Overflow 2026: 채택 84%, 정확성 신뢰 29%(2024년 40%), 최대 불만 "거의 맞지만 딱 맞지 않음" 66% | 2차 |

### 5.5 반대 신호

- **Cursor는 2.1에서 Memories를 제거하고 Rules로 접었다**(포럼 2025-11/12; 스태프는 Custom Modes 제거만 확인, Memories 공식 설명 미발견).
  같은 시기 Anthropic은 auto memory·Auto Dream을 확대. 두 제품이 반대로 움직였다.
- 실무자 비판: 자동 메모리는 "too specific or too general, almost never on point"이고 코드베이스를 안다는 거짓 안정감을 줌(Zoeller 2026-04).
- 낡은 기억은 빈 기억보다 나쁘다: STALE(arXiv:2605.06527)이 정식화, #59111이 TTL 요청.
- 규칙은 주입돼도 준수되지 않는다: #2544(15개월 open), 공식 문서 스스로 "context, not enforced configuration", 강제하려면 훅.
- 커뮤니티 관심이 얕다: Ask HN 메모리 스레드 7 points / 10 댓글, 같은 저장소 요금제 이슈는 1,491 댓글.
- 벤더 자기 수치 신뢰 문제: Mem0 "State of AI Agent Memory 2026"(LoCoMo 92.5·LME 94.4)은 전부 자사 측정. **Mem0 LME 94.4와 LME-V2
  최고 72.5(small·medium 평균; 티어별로는 74.9/70.1, §3.2)는 다른 벤치이므로 나란히 놓으면 안 된다.**

### 5.6 우리 서사에 걸리는 것

"메모리 수요가 폭증한다"고 쓰면 정규화 수치로 반박당한다. 방어 가능한 문장은 **"수요는 파일 규약의 이식성과 압축
통제권으로 표현되었고, episodic 수요는 아직 언어화되지 않았다"**이고, 따라서 v1은 설문이 아니라 **도그푸드 측정으로 수요를
만들어 보여야** 한다(v1 계획서 Phase 3, 이중 호스트 도그푸딩). Claude Code 자동 메모리의 4타입(user/feedback/project/reference)은 기능 4축과 일치하지
않고 작업 기억에 해당하는 타입이 없다(§1.3 표).

---

## 6. 종합: v1이 runbook 너머로 덮어야 하는 것 (근거 순)

정직한 베이스라인부터 바꾼다. **v1의 대조군은 "무기억"이 아니라 "세션 로그를 원문 그대로 두고 grep·파일 읽기를 허용한
에이전트"다.** LME-V2에서 그것만으로 69.9%가 났고(설계된 메모리의 추가는 +5.0pp), AutoMEM에서 파일 하네스가 8개 시스템을
이겼으며, EvoMemBench에서 장문 컨텍스트가 여전히 경쟁력 있다. organizer가 이 대조군을 못 이기면 값을 못 하는 것이고,
그 비교는 우리 v0 결론과 정합적으로 **읽기 경로가 주 실험 팔**이 되어야 성립한다. 경쟁하는 두 번째 정직한 베이스라인은 Anthropic memory tool
문서가 권장하는 **다중 세션 개발 패턴**이다(초기화 세션에서 진행 로그·기능 체크리스트를 세팅하고, 매 세션이 그것을 읽고 시작해 종료 전 갱신하며,
"코드를 썼을 때가 아니라 종단간 검증이 끝났을 때 완료로 표시"). 이것은 모델 주도 파일 메모리이며 organizer도 벡터도 없다. 도그푸드 비교에서
"원문 + grep"과 이 패턴을 각각 0번·0′번 아암으로 두어야 "네이티브보다 나은가"라는 질문이 실제 네이티브를 상대로 성립한다.

| # | 덮어야 할 것 | 기능 축 | 근거 | 현재 상태 (§2) |
|---|---|---|---|---|
| 1 | **원문 궤적 보존 + 포인터** (JIT, AOT 아님) | 에피소드 | GAM, LME-V2 AgentRunbook-C(인덱스 없음, rg/find), Code Isn't Memory | `add_task_result`가 궤적 폐기 |
| 2 | **탐색기 읽기 경로** (모델이 rg/find로 원문을 뒤짐), 벡터는 후보 좁히기 | 에피소드·read | LME-V2 small: AgentRunbook-C 74.9 vs -R 58.6(+16.3pp, 지연 108s vs 27s ≈ 4×), vs RAG 51.0(지연 0.2s 대비 수백 배, `longmemeval.md` §7은 "400배"로 적음) → 지연 1급값 | 없음 |
| 3 | LME-V2 5능력 중 runbook 밖의 넷: static state recall(빌드·테스트 명령·구조), dynamic state tracking(브랜치·이미 지출한 실험), environment gotchas, premise awareness | 에피소드·의미 | 2605.12493 | runbook은 workflow knowledge만 |
| 4 | **입도를 세션이 아니라 하위 과제로**, 항목마다 소속 단계 표시 | 절차·에피소드 | 2602.21611 +4.7pp, 인스턴스 단위는 오검색 | 프롬프트가 태스크 블록으로 쪼개지만 단계 라벨 없음 |
| 5 | **추상화 등급을 항목 속성으로**, 등급별 분리 측정 | 절차 | 2604.14004 저수준 부정 전이, AFTER 직무 특화 손실 | `procedure`·`failures`가 저수준으로 흐르기 쉬움 |
| 6 | "잘 뽑았는가"와 "잘 골라 왔는가"를 **분리 측정** | read | 2604.04323 붕괴→정제로 회복, SWE-ContextBench, 트랙5 미분리 항목 | 측정 설계 없음 |
| 7 | 신선도·갱신 충돌은 **결정적 신호**(커밋 해시·mtime)에 결속, LLM에게 추적시키지 않음; 변형 시점 훅 | 의미·manage | 2606.01435 7%, 2606.15903 91.7–93.2%@$0.17 | manage 없음, 시간 감쇠 없음 |
| 8 | **쓰기 채널 신뢰 경계**: 항목마다 출처를 쓰기 시점에 결속(origin binding) | 의미·write | 2606.04329, 2606.24322 세탁 68%, MemMorph 3건 | 프롬프트의 "데이터이지 지시가 아님" 문구뿐 |
| 9 | **프로젝트 간 유출 게이팅**(출력 시점 문맥별) | 의미·read | CIMemories 69% | 시크릿 제거는 입력 위생만(b694cfc), 그것도 패턴 9개(`sessions/__init__.py:55-67`)이며 주석 스스로 "완전하지 않다, Phase 3에서 정책을 정한다"고 선언(`:25-27`) |
| 10 | 비용은 토큰으로, **누적 기록량에 따른 열화 곡선** | 전 축 | 트랙5 nodedup 5.9×, 2606.30306 §6.4.3 "baseline-beats-memory" | 단발 비교만 |
| 11 | 작업 기억은 압축 기법이 아니라 **통제권 배치**(압축 전 자동 저장, 압축 후 복원 가능한 원문) | 작업 | AutoMEM, #27242·#28107 수요, 2608.06503 | 없음 |
| 12 | 인용→사용량 되먹임(Codex `usage_count` 대응) | manage | Codex 소스, SkillJuror(사용 계측 ≠ 효용) | `on_feedback` 미구현 |

**하지 않는 것**(이 조사로 확정): 기질 축(KV/파라메트릭)과 RL 정책은 하드웨어 밖 그대로. 새 대화형 organizer 없음.
의미 축의 "더 똑똑한 추출"은 근거 없음 — 대신 결정적 갱신·출처 결속·게이팅이라는 **위험 쪽 설계**만 한다.

---

## 7. 리팩터 함의 — OpenRouter 스모크 전에 무엇이 끝나야 하나

리포 대응 레인이 찾은 압력 지점 15개를 세 묶음으로 나눈다. 코드 제안이 아니라 "현재 계약이 무엇을 가정하고 어디서
어긋나는가"이며, 순서는 §6의 근거 순서를 따른다.

### 7.1 묶음 A — 궤적 보존과 세션 배선 (스모크 전 필수)

| 지점 | 위치 | 왜 어긋나나 |
|---|---|---|
| `add_task_result`가 궤적을 버림 | `memory.py:495-498` | JIT 읽기의 원문 저장소가 없다. 세션 원문을 `to_episodes()`로 넣는 배선도 없다 |
| `as_task_trajectory()` 호출자 0개 | `sessions/__init__.py:181` | 세션 → organizer 진입점이 MCP 도구(`mcp/server.py:221`)뿐이고 그것은 모델이 부를 때만 동작 |
| 훅이 organizer를 금지 | `hooks/__init__.py:207`, 근거 `:23-26` | 키 입력마다 LLM 불가라는 근거는 옳지만, **세션 종료(또는 데몬 배치) 시점의 1콜 증류**를 걸 자리가 없다 |
| 세션 경계 훅 없음 | `organizers/base.py:100-259` 12훅 | `experience`는 세션을 태스크 하나로 취급하고 프롬프트가 다시 쪼갠다. `flush_buffer`·`consolidate`는 호출자가 불러야 한다 |
| `experience`가 admission 사정권 밖 | `gated.py:39-44` | "어떤 세션을 증류할 가치가 있는가"라는 세션 단위 admission 자리가 없다 |
| LME-V2 어댑터 없음 | `bench/registry.py:44-48`만 | `Memory.insert(trajectory)/query(text,image)` 래퍼 부재 |

스모크가 재야 하는 것은 "세션 하나가 들어가 runbook이 나오고, 그 runbook과 원문이 같은 스토어에 포인터로 묶이는가"다.
첫 두 줄이 없으면 스모크는 LLM 호출 1회가 JSON을 뱉는지만 본다.

### 7.2 묶음 B — 어휘: 기능 축과 갱신 정책을 타입 메타데이터로

| 지점 | 위치 | 왜 어긋나나 |
|---|---|---|
| `MEMORY_TYPES`가 평평한 튜플 | `core/types.py:34-49` | 14개가 한 평면. 기능 축도, 갱신 정책(supersede / append-only / abstract)도 표현 못 함. `state`가 주석으로만 구분됨 |
| `TAG` op가 죽어 있음 | `core/ops.py:30`, `memory.py:802-806` | 경험 메모리의 라벨(성공/실패·워크스페이스·추상화 등급·하위 과제)이 필요한데 어휘는 있고 발행자가 없다 |
| `experience`에 manage 없음 | `experience/organizer.py:269-297` | 모순 runbook 병합·무효화 경로 없음. §6 #7의 결정적 신호 결속이 들어갈 자리 |
| `produces`가 읽기 순서이고 load-bearing | `base.py:79-84`, `memory.py:926-951` | 확장 스텝이 번들 내 중복 id를 제거하므로 순서가 결과를 바꾸고, `default_memory_types`가 이것으로 구동되며 `playbook`은 의도적으로 제외됨. 타입 어휘를 건드리는 묶음 B의 **회귀 위험이 여기에 있다** |
| `observes_store_on_message`가 메시지 훅에만 | `base.py:86-98` | 태스크 훅에 같은 선언이 없어 세션 단위 organizer의 스토어 읽기 여부를 파사드가 모름 |

### 7.3 묶음 C — 읽기 경로: 탐색기와 지연

| 지점 | 위치 | 왜 어긋나나 |
|---|---|---|
| `runbooks` ReadStep 없음 | `retrieval/steps.py:983-1039` | 절차 특유의 읽기(절차 전문 서빙·실패 회피 우선)가 없고 일반 벡터 검색만 |
| `query_strategy`가 문자열 질의만 가정 | `policies/retrieval.py:307-345`, `planned.py:31-33` | 탐색기는 `search` 콜러블이 아니라 셸 도구가 필요해 이 seam에 그대로 안 들어감 — 8월 피드백 공백 B의 `Researcher` 계약 |
| 지연이 1급값 아님 | `memory.py:975-980`, `planned.py:72-102` | LAFS(정확도·지연 Pareto)가 요구하는 질의별 벽시계 필드 없음 |
| 컨텍스트 예산이 렌더에만 | `core/types.py:260-311` | 문자당 4토큰 근사, 한 번도 바인딩 안 됨. LME-V2는 `--memory-context-max-tokens` 강제 |
| `recent_context()`가 인자 없음 | `base.py:188-202` | 작업 기억은 "지금 컨텍스트에 무엇이 있는가"를 입력으로 받아야 함 |

### 7.4 순서 제안

1. 묶음 A 전부 → OpenRouter 스모크(견적·승인 후) → 스모크가 "세션 → runbook + 원문 포인터"를 실제로 재는지 확인.
2. 묶음 C의 탐색기 + 지연 → `agmem-rag` / `agmem-explorer` 아암이 성립(v1 계획 Phase 2).
3. 묶음 B는 A·C가 드러내는 필요에 맞춰 최소로. 특히 갱신 정책은 §6 #7(결정적 신호)로 한정하고 LLM 통합 에이전트(Codex Phase 2)는 측정 뒤.

---

## 8. 검증 상태와 모순

### 8.1 검증 등급

- **직접 확인**: arXiv 16건(분류 레인)+30여 건(연구 레인)의 제목·날짜·초록; GitHub 이슈 번호·반응·댓글·기간별 카운트(`gh api`);
  Claude Code 공식 메모리 문서 전문; OWASP ASI06 등재일; LME-V2 수치표; AGENTS.md 효율 논문 수치; SAP 논문 수치; 리포 file:line 전부.
- **2차 출처만**: AGENTS.md 60,000 저장소, Skills 스타·스킬 수, Mem0/Zep/Letta 스타(출처 간 불일치), Vertex/Bedrock 단가, Stack Overflow·
  LangChain 설문, GitGuardian 수치, 오염 ASR 80–99% 요약치, MemoryArena "40–60%", Letta 5×/2.5×, Codex `/import` 30일 채팅.
- **초록에 수치 없음**(인용 시 본문 표 필요): ExpeL, MemSkill, Mem-α, Memory-R1, MemRL, MemoryArena, Evo-Memory, AutoMEM, MINJA.
- **미확인**: arXiv:2404.13501의 write/manage/read 3연산 표현(HTML 404); Cursor Memories 제거의 공식 사유; Auto Dream의 공식 지위.
- **추론**: §2.1 타입→축 배정; "episodic 수요가 가장 약하다"(반응 분포에서 도출); §7의 압력 지점 중 4·8·10·11·13은 파라미터 모양과
  호출자 목록에서 읽은 해석.

### 8.2 정정한 전제

- **LAFS는 베이스라인 방법이 아니라** LME-V2 리더보드의 프런티어 이득 지표다(레인 지시문의 오기, v1 계획서 §3 Phase 2 표의
  "LAFS 참조 frontier"는 옳은 용법).
- `on_turn`·`on_session_end` 훅은 이 리포에 **없다**. 실제 훅 12개는 §2.3.
- **MemoryAgentBench 네 번째 역량은 판본마다 다르다**: v1(2025-07) conflict resolution, v3(2026-03) selective forgetting. 8월 피드백의
  "네 능력"은 v1 어휘다. 인용 시 판본 명시.
- arXiv:2505.00675 제목이 "…Representations, Operations, and Emerging Topics"로 바뀜.
- SleepGate(2603.14517)는 토이 트랜스포머 실험.
- LME-V2 성적표는 레인마다 다른 집계를 옮겼다: 연구 레인은 **티어별**(small 74.9/69.9/58.6/51.0, medium 70.1/68.7/57.0/45.9), 수요·분류 레인은
  **small·medium 평균**(72.5/69.3/48.5). 둘은 같은 표의 다른 열이며(v1 계획서 §1.1도 같은 확인), 이 문서는 티어별 수치를 기본으로 쓴다.

### 8.3 모순 (매끄럽게 넘기지 않음)

1. 개별 방법 논문은 예외 없이 자기가 이겼다고 하지만, **표준 프로토콜로 함께 돌린 연구일수록**(EvoMemBench, AutoMEM, MemoryAgentBench)
   메모리 시스템에 불리하다.
2. 같은 방법이 벤치에 따라 뒤집힌다(Mem0: LoCoMo 강 / HotpotQA 224K에서 RAG보다 −20). **LoCoMo 계열 수치를 에이전트 성능의 대리로 쓰면 안 된다.**
3. 행동 변화 ≠ 결과 변화(SkillJuror 3배 vs +4.1%; 우리 트랙5).
4. 압축은 도움이자 해악(ACON vs 2608.06503).
5. 절차 메모리의 이상 조건(AWM 상대 +24.6~51.1%, SkillWeaver 상대 +31.8~54.3%)과 현실 조건(34K 스킬 자율 검색, 최난이도에서 무스킬 수준)이 다르며 **v1은 후자에 있다**.
6. 파일시스템이 벤치에서 이기는데 상업적으로는 메모리 계층(Mem0 $24M)이 앞선다.
7. 수요는 크지만 비중은 준다(#6235 6,556 반응 vs 점유율 1.92→0.85%). 메모리는 한 번 크게 요구되고 파일 규약으로 해결된 뒤 관심이 이동한 주제일 수 있다.
8. Cursor는 제거, Anthropic은 확대. AGENTS.md는 효율 −28.64%인데 규칙 준수는 여전히 실패.
9. ACE 공개 재현 실패 보고가 없다 — 우리 페어링 런이 공개 문헌 공백을 메우는 위치에 있다.

---

## 9. 이 문서가 바꾸는 것

- v1 계획서(`docs/_internal/plans/2026-09-02-v1-experience-memory.md`)의 Phase 1 남은 조각 순서: **원문 궤적 보존·세션 배선(§7.1)이
  explorer보다 먼저**, 그리고 스모크는 그 뒤.
- 측정 아암의 대조군 정의: `agmem-rag`가 아니라 **"원문 + grep 에이전트"** 가 0번 아암이다. `agmem-experience`는 그것 대비 +pp로 읽는다.
- runbook 스키마에 붙일 속성 셋: 하위 과제 단계, 추상화 등급, 출처 결속(세션 id·스텝 범위·커밋 해시). 전부 `TAG`/payload로 표현 가능.
- 서사: "네 축 중 하나"는 참이되, 문헌이 말하는 것은 **나머지 셋을 채우라가 아니라 에피소드 축(원문 접근)을 먼저 세우고 의미 축은
  위험 쪽만 설계하라**다.

## 부록 — 원 보고서

`docs/_internal/research/2026-09-02-v1-axes-lanes/` (비커밋) — `lane-taxonomy.md`(분류, 159줄) · `lane-research.md`(연구 증거, 570줄) · `lane-products.md`(제품, 434줄) ·
`lane-needs.md`(수요, 195줄, 재현 명령 부록 포함) · `lane-repo.md`(리포 대응, 585줄). 이 문서에 옮기지 않은 세부(개별 이슈
번호 전체, 논문별 대조군 서술 전문)는 그쪽에 있다.
