# 컴포넌트 분류 조사 → 모듈 설계 (2026-07-26)

> 계기: A-MAC 게이트를 `organizers/admission.py`에 놓은 것이 옳은지 사용자가 문제 제기.
> `organizers/`에 organizer가 아닌 모듈(`nemori_stages.py`, `admission.py`)이 섞여 있었다.
> 질문: **"호스트 방법론을 수정하는 cross-cutting 컴포넌트"가 실재하는 반복 범주인가, A-MAC 단발인가?**
> 방법: 후보 논문들을 *구조* 관점(자체 시스템인가 modifier인가 / 어느 lifecycle seam인가)으로
> 1차 소스 조사. RecMem은 공식 코드까지 통독.
> 결론: 범주는 실재하고, **문헌이 이미 이름을 갖고 있다** — 그 어휘를 그대로 채택했다.

---

## 1. 문헌의 분해를 채택한다 (arXiv:2603.07670)

서베이 "Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers"는
메모리를 **write–manage–read**로 모델링한다. read 연산자 `R(M, x)`, update 연산자
`U(M, x, a, o, r)`, 연산 집합 **store / retrieve / update / summarize / discard**. 그리고 두 층위를
명시적으로 분리한다:

| 서베이 층위 | 내용 | 우리 대응 |
|---|---|---|
| **mechanism** (§4) | 시스템이 *무엇인지*. 5 계열: context compression, retrieval stores, reflection, hierarchical memory, learned control | `organizers/` (`Organizer`) |
| **control policy** (§3.3) | 연산이 *언제/어떻게* 발동하는지 지배하는 heuristic / prompted self-control / learned 규칙. 서베이 표현: **"mechanism 선택과 직교하는 cross-cutting 차원"** — 같은 store를 다른 policy로 구동할 수 있다 | `policies/` |

우리가 범주를 발명할 필요가 없었다. `Organizer`는 mechanism이고, A-MAC 류는 control policy다.

부수 확인 — arXiv:2607.08032("What to Keep, What to Forget: A Rate–Distortion View of Memory
Compaction")도 같은 층위 구분을 전제한다: compaction을 pretraining/prompt/decode/serving/
within-task/between-task **여러 stage에 걸친 add-on 계열**로 다루고 자체 메모리 타입을 정의하지
않으며, COMPACT-Bench를 "post-hoc optimization"으로 위치시킨다.

---

## 2. 조사 결과: 범주별 소속 (구조 판정)

**판정 기준(운영적)**: *policy는 메모리 타입을 선언하지 않고 `MemoryOp`도 발행하지 않는다.*
저장 상태 + 그것을 되읽는 경로를 소유하면 mechanism.

### 2.1 mechanism (→ 자체 organizer)

| 논문 | 근거 | 상태 |
|---|---|---|
| A-Mem, Nemori, MemoryOS, Zep/Graphiti, ACE, ReasoningBank, G-Memory | 기존 구현 | 구현됨 |
| **RecMem** (2605.16045, ACL'26 Findings) | **공식 코드 통독**: 자체 embedding/vector store/LLM 계층 + subconscious·episodic·semantic **3-tier**. `SubconsciousMemory` docstring이 buffer가 *query time 검색 소스*임을 명시 → 3 tier 전부 읽힌다 | **재분류** |
| **GRAVITY** (2605.01688) | 논문 전문: anchor 3종을 **offline build phase**에서 생성(entity=incremental batch update+offline consolidation, event=4W1O 튜플→temporal trace, topic=cross-session 식별+요약)하고 "anchor knowledge base를 **standalone 파일로 저장**". query time은 retrieval+query expansion+injection | **재분류** |
| LightMem (2510.18866) | Light1 sensory(LLMLingua-2 토큰 압축 + topic 분절) → Light2 short-term(`{topic, message turns}`, 버퍼 256–1024 토큰에서 요약 트리거) → Light3 long-term(`{topic, embedding, user_i, model_i}` + sleep-time offline consolidation). **자체 entry 구조 보유, 압축 게이트도 자체 파이프라인 내장이고 외부 시스템용 미들웨어로 제시되지 않음** | 후보 |
| HyMem (2602.13933) | dual-granular 저장 + 2-tier 자체 retrieval + reflection | 후보 |
| MIRIX (2507.07957) | 6 메모리 타입(Core/Episodic/Semantic/Procedural/Resource/Knowledge Vault) + multi-agent 제어 | 미구현 |
| MemOS (2507.03724) | memory OS, MemCube 단위. MIRIX/Mem0/Zep/Memobase/MemU/Supermemory를 baseline으로 비교 | 미구현 |
| **MemMachine** (2604.04853, Apache-2.0) | 자체 3-tier(short-term / long-term episodic / profile) + 자체 **Retrieval Agent**. §2.4 참조 | 미구현 |

### 2.4 MemMachine — 추출 축의 반대 극단 (비교표에서 가장 쓸모 있는 지점)

**구조 판정: mechanism.** 자체 3 tier(short-term, long-term episodic, profile)와 자체 read 경로를
소유한다. 그런데 방향이 A-Mem과 정반대다:

- **write**: "ground-truth-preserving architecture that **stores entire conversational episodes and
  reduces lossy LLM-based extraction**". 메시지별 fact 추출이 **없다**. write-path LLM은 STM 요약과
  profile 추출에만 쓴다. 인덱싱은 문장 단위.
- **read**: nucleus match를 주변 ±1~2 turn으로 확장하는 *contextualized retrieval* → dedup·시간순
  정렬 + 선택적 cross-encoder rerank.
- **Retrieval Agent**: 질의를 direct retrieval / parallel decomposition / iterative chain-of-query
  중 하나로 **적응적 라우팅**한다.

수치: LoCoMo 0.9169(gpt-4.1-mini, agent mode) / LongMemEval-S 93.0%(gpt-5-mini).
input 토큰 LoCoMo 4.20M vs Mem0 19.21M (−78%), memory mode 4.20M / agent mode 8.57M.

**우리에게 주는 것 3가지**:
1. **비용-정확도 파레토의 반대쪽 끝점**이다. A-Mem은 turn당 2콜을 써서 추출하고, MemMachine은
   사실상 추출하지 않는다. 8-시스템 비교표(Phase 5)의 축을 정의해주는 대조군이다.
2. **우리 설계 규칙을 외부에서 검증해준다.** `core/types.py`의 "raw Episode는 불변, organizer는
   파생만 — verbatim-loss 방어"가 MemMachine의 ground-truth preservation과 같은 주장이다. 즉 우리
   `passthrough` baseline이 이 계열의 하한이고, MemMachine은 그 위에 STM 요약 + profile만 얹은 형태다.
3. **Retrieval Agent는 read-path control policy이고 우리에게 자리가 없다.** 적응적 질의 라우팅은
   메모리 타입을 선언하지 않고 op도 발행하지 않는다 → 판정 기준상 policy인데, MemMachine 안에서는
   mechanism 내부 부품이다. **mechanism이 policy를 내장할 수 있다**는 사례이므로, 우리가 이걸
   구현한다면 `policies/`의 read-side 멤버로 뽑아내는 게 맞다(Memory Worth와 같은 자리).

인용 캐비앗은 `write-path-critics.md` §4.4에 이미 기록됨 — retrieved vs total 간극을 스스로
분리하지 않았고 "80% 절감"은 memory-only 경로 기준이다.

### 2.4.1 "consolidation"은 최소 3가지 다른 것을 가리킨다

비교표를 쓸 때 가장 헷갈리는 용어다. 우리 코드 기준으로 확정된 사실:

| 논문/코드의 "consolidation" | 실제로 하는 일 | 시점 | 우리 훅 |
|---|---|---|---|
| **우리 `Organizer.consolidate(ctx)`** | cursor로 재개되는 **지연 관리 패스**. `AgenticMemory.consolidate()`로만 호출 | offline | 그 자체. **구현한 방법론은 Nemori 하나** |
| **Nemori** | semantic 통합. inline(v4 `ThreeWayIntegrator`)과 deferred(`SemanticOfflineConsolidator`, `consolidation="semantic_offline"`)의 **두 갈래** = 시점 ablation 축 | 둘 다 | 둘 다 (inline은 write 경로, deferred는 `consolidate()`) |
| **Zep/Graphiti** | **consolidation이 아니다.** entity resolution(같은 실체 병합) + bi-temporal invalidation(기존 fact 무효화)이고 전부 `on_message` 안이다. `consolidate()`를 구현하지 않는다. community detection만 offline 성격인데 TODO 미구현 | inline | `on_message` |
| **RecMem** | recurrence 게이트가 트리거하는 buffer→episodic **승격** | **inline** (`add_memory` 안, `rec_mem.py:337-411`) | `on_message` |
| **LightMem** | sleep-time offline 재구성 | offline | `consolidate()` |
| **MemoryOS** | STM→MTM→LPM heat 승격 + LFU eviction | inline | `on_message` |

⇒ 정리: 서베이의 연산 어휘로 보면 "consolidation"은 단일 연산이 아니고 **summarize + update +
discard의 묶음**이다. 비교할 때 갈라야 할 축은 두 개다 — **언제**(inline / deferred-offline) ×
**무엇을**(요약 / 통합 / 승격 / 무효화·삭제). "consolidation을 한다"만으로는 아무 것도 비교되지 않는다.

내가 RecMem을 "`consolidate()` 훅이 그 자리"라고 잘못 배치한 원인도 이것이다 — 논문이
"consolidation"이라 부르니 offline 패스라고 가정했지만 실제로는 per-message write 경로였다.

### 2.4.2 `on_message` vs `on_task_end` — 무엇이 실제로 달라지는가

파사드 코드(`memory.py:227-283`)로 확인한 비대칭:

| | `add_message` → `on_message` | `add_task_result` → `on_task_end` |
|---|---|---|
| 훅 시그니처 | `(episode, ctx)` | `(trajectory, outcome, task, ctx)` — **Episode를 받지 않는다** |
| 저장되는 raw | 메시지 **전문**이 Episode로 영구 보존 | `content=task` 문자열 + `meta={outcome, agent_id, steps}` 뿐. **trajectory는 파사드가 저장하지 않는다** |
| 정보 소실 위험 | 낮음 — organizer가 손실 추출해도 원문이 남아 재파생 가능 | **높음 — `on_task_end`가 뽑지 못한 것은 영구 소실.** 기회가 한 번이다 |
| `warm_start` 백필 | 지원 (corpus를 `on_message`로 replay) | **불가** — corpus는 `Episode` 리스트이므로 task 기반 organizer는 아무 것도 만들지 않는다 |
| admission policy | 적용 가능 | **적용 불가** (§2.5) |
| 우리 organizer | amem, nemori, memoryos, zep_graph, passthrough | ace, gmemory, reasoning_bank |

**비교 설계에 주는 제약(중요)**: LoCoMo/LongMemEval은 대화 벤치 = `on_message` 계열만 측정한다.
task 기반 3종은 그 벤치에서 **아무 것도 생산하지 않는다**. 즉 Phase 5의 "8-시스템 비교표"는
**하나의 벤치로 8개를 나란히 놓을 수 없다** — 대화 벤치(5종) + 에이전트 태스크 벤치(3종)로 갈리고,
파레토 곡선도 축이 다르므로 따로 그려야 한다. 이건 구현 문제가 아니라 방법론들이 소비하는 입력
단위가 다르다는 사실이다.

**LightMem이 주는 설계 시사점**: write 트리거 granularity가 **turn이 아니라 topic-segmented
group**이다. A-MAC/SAGE는 turn 단위 admission이고 LightMem의 sensory 압축은 **토큰 단위**다.
즉 policy를 분류할 때 "어느 연산인가"만으로는 부족하고 **granularity**가 두 번째 축이다 —
우리 `AdmissionGated`는 episode(turn) 단위이므로 토큰/그룹 단위 정책은 같은 자리에 못 들어간다.

### 2.2 control policy (→ `policies/`)

| 논문 | 지배하는 연산 | 근거 |
|---|---|---|
| **A-MAC** (2603.04549, ICLR'26) | **store** (admission) | 5-feature 가중합 게이트. 자체 메모리 타입 없음. 감사: `amac-admission-gate.md` |
| **SAGE** (2605.30711) | **store** (admission) | 제목이 "A Novelty Gate". vMF 밀도 추정 + adaptive threshold로 ADD/NOOP/LLM-merge 라우팅. **"a drop-in binary gate for A-Mem"**으로 명시, LLM 콜 16–18% 절감. 자체 메모리 타입 없음 |
| **Memory Worth** (2604.12007, "When to Forget") | **discard** + **retrieve** 억제 | "lightweight, theoretically grounded foundation for staleness detection, retrieval suppression, and deprecation decisions". **메모리 유닛당 스칼라 카운터 2개**만 필요하고 "retrieval과 episode outcome을 이미 로깅하는 아키텍처에 추가 가능" |
| **Mem-α** (2509.25911) | **store / update / discard** (learned) | RL이 학습하는 것은 `memory_insert`/`memory_update`/`memory_delete` **호출 정책**이다. 논문이 직접: "our memory architecture is modular and **decoupled from the reinforcement learning framework**. Researchers can seamlessly substitute alternative memory designs." retriever·generator는 frozen이고 **write 정책만 gradient를 받는다** |

⇒ **범주는 실재하며 4 멤버 / 3 seam.** 두 가지가 결정적이다:

1. **SAGE는 A-MAC과 같은 seam(store)·같은 호스트(A-Mem)**다 → `policies/admission.py`가 단발
   모듈이 아니라 최소 2 멤버가 들어올 자리임이 확정.
2. **Mem-α는 서베이 §3.3의 heuristic/prompted/**learned** 3분류 중 learned 변종**이고, 자기 논문이
   "메모리 아키텍처와 분리돼 교체 가능"하다고 명시한다 → policy를 mechanism과 분리하는 축이
   우리 발명이 아니라 **논문들이 스스로 채택한 축**임을 확인해준다.

즉 policy는 최소 3축으로 분류된다: **어느 연산**(store/retrieve/update/discard) ×
**granularity**(토큰 / turn / topic-group) × **종류**(heuristic / prompted / learned).

Memory Worth는 우리 구조에 이미 착지점이 있다: `on_retrieval` 훅이 served (item_id, type, score)를
넘기고 MemoryOS의 N_visit / G-Memory의 served 캐시가 그걸로 돌아간다. "retrieval을 이미 로깅하는
아키텍처"가 곧 우리다.

### 2.3 methodology-internal stage (→ 소유자의 서브패키지)

Nemori의 `PerMessageBoundary` / `BatchPartitioner` / `EpisodeMerger` / `AppendIntegrator` /
`ThreeWayIntegrator` / `DedupIdReuseIntegrator`. 이들은 **Nemori 논문 자신의 메커니즘**이고
fidelity 스위치 때문에 교체 가능한 클래스로 분리돼 있을 뿐이다 — 2026-07-21 감사의 N1 수정이
바로 `ThreeWayIntegrator`가 experimental이 아니라 **충실 코어에 속한다**는 판정이었다(docs/10).
policy도 아니고 독립 mechanism도 아니므로 별도 패키지가 아니라 소유자 밑에 둔다.

---

## 2.5 검증: "policies/에 뒀다"는 일반성 주장이 실제로 참인가

사용자 지적 — policy 패키지에 둔다는 건 **다른 organizer에도 적용된다는 주장**인데, 최초 구현은
`AMemOrganizer(admission=...)` 생성자 인자였다. 즉 **주장은 했고 구현은 A-Mem 하나였다.**
우리 organizer 8종을 진입점 기준으로 실제 확인한 결과:

| organizer | 진입점 | 게이트 적용 | 근거 |
|---|---|---|---|
| `amem` | on_message | ✅ 유효 | turn당 LLM 2콜(Ps1+Ps2/Ps3) → 거부 시 0콜 |
| `zep_graph` | on_message | ✅ 유효 | turn당 entity 추출 콜 → 거부 시 절약 |
| `memoryos` | on_message | ✅ 유효 | turn당 STM page + capacity 소모 |
| `passthrough` | on_message | ⚠️ **무의미** | `on_message`가 `[]`를 반환하고 **facade가 organizer 실행 전에 raw episode를 이미 저장**(write-then-organize, `memory.py:247→248`). 거부해도 바뀌는 게 없다 |
| `nemori` | on_message | ⚠️ **메커니즘이 변한다** | 메시지를 버퍼링하고 스트림 위에서 episode boundary를 탐지한다. 중간 메시지를 떨구면 **분절 자체가 달라진다** = 비용 최적화가 아니라 ablation. 허용하되 그렇게 보고해야 함 |
| `ace` | on_task_end | ❌ **적용 불가** | `on_message`가 없다 |
| `gmemory` | on_task_end | ❌ **적용 불가** | 동일 |
| `reasoning_bank` | on_task_end | ❌ **적용 불가** | 동일 |

⇒ **판정: 범주 수준의 일반성은 참(4 멤버·3 seam), 그러나 A-MAC 개별 정책의 적용 범위는
message 기반 organizer로 한정된다.** task 기반 3종에는 원리적으로 적용 불가다 — episode 키 게이트가
trajectory에 대해 판정할 것이 없다. 이건 `policies/`의 한계가 아니라 **admission이라는 seam의
한계**다(Memory Worth의 discard/suppression은 쓰기 방식과 무관하게 검색된 유닛을 지배하므로
task 기반에도 적용된다).

**구현 교정**: 생성자 인자를 없애고 wrapper로 바꿨다 — `organizers/gated.py::AdmissionGated`.

    AdmissionGated(AMemOrganizer(), AdmissionGate())

얻은 것 3가지:
1. **한 번 구현으로 모든 message 기반 organizer에 적용** — 일반성 주장이 말이 아니라 코드가 됐다.
2. **mechanism이 policy를 모른다** — `organizers/amem/`에 `policies` import가 0개다. 직교성의
   실제 증거이고, 애초에 `policies/`를 만든 이유가 이것이다.
3. **합성이 명시적** — nemori를 게이팅하는 것이 숨은 파라미터가 아니라 조립 시점의 선택이 된다.

`warm_start`를 별도로 게이팅해야 했던 이유가 하나 있다: `NemoriOrganizer`와 `MemoryOSOrganizer`가
**둘 다 `warm_start`를 오버라이드**하므로, `on_message`에만 게이트를 두면 정확히 그 둘에서
우회된다. 그래서 wrapper는 corpus를 먼저 필터하고 넘긴다.

**알려진 gap** (`ChainedConsumer`의 기존 "Known gap"과 같은 성격으로 문서화): 
`ChainedConsumer(AdmissionGated(x))` 조합에서 `base.overrides()`가 wrapper를 보므로, 자체 `retire`
정책을 가진 wrapped(MemoryOS)가 chained 일반 경로로 라우팅된다. admission 논문들은 모두 직접
메시지 경로를 대상으로 하므로 이 조합은 **지원이 아니라 범위 외**로 둔다.

## 2.6 memory type은 방법론 전용이 아니다 — 세 번째 어휘 충돌 (2026-07-26 추가)

§2.4.1이 "consolidation"이 3가지를 가리킨다고 정리했는데, **같은 처방이 필요한 용어가 하나 더
있었다**: memory type 자체다. 우리 8 organizer의 `produces`를 실제로 세어보면 두 타입이 공유된다.

| type | 생산자 | 실제로 담는 것 |
|---|---|---|
| `semantic` | Nemori | predict-calibrate로 증류한 fact |
| | MemoryOS | LPM profile fact (`kind="profile"`) |
| `strategies` | ReasoningBank | 궤적에서 증류한 전략 아이템 |
| | G-Memory | trajectory(`kind="trajectory"`) + insight rule(`kind="insight"`) |

타입만으로 소유자를 알 수 없는데 **store 질의는 타입 키로만 이뤄지므로**, 조합 설정에서 조용히
서로를 침범한다. 실제로 발견된 두 자리:

1. **Nemori의 semantic 통합기** — `ThreeWayIntegrator`가 `memory_type="semantic"`로 후보를 검색해
   LLM에게 merge/conflict를 물으므로 MemoryOS의 profile fact를 INVALIDATE할 수 있었고,
   `DedupIdReuseIntegrator`는 top-1 id를 재사용하므로 **내용을 통째로 덮어쓸** 수 있었다.
   `SemanticOfflineConsolidator`는 *선택* 측에 `op.actor` 가드가 이미 있었는데 *후보* 측엔 없었다.
2. **`report_feedback`** — `target_type`으로 분기해 `strategies`면 G-Memory의 +1/−2를 적용했다.
   ReasoningBank는 논문상 append-only(피드백 루프 없음)인데도 점수가 붙었다. 반대로 G-Memory의
   served-insight 게이트(round-5 W-4)는 호출자 없는 `backward()` 안에만 있어 실경로엔 없었다.

**처방**: 새 어휘를 발명하지 않고 **이미 있는 `actor`를 쓴다.** op에 이미 실려 있고 evolution log가
기록하던 것이므로, 파사드가 ADD 적용 시 아이템에도 함께 남기면 소유권 질의가 가능해진다
(`memory.py::_apply_one`). 두 자리를 그것으로 고쳤다 — `nemori/stages.py::own_items`,
그리고 피드백을 organizer의 `on_feedback`으로 팬아웃(docs/04 §3.4). `actor`가 없는 과거 아이템은
자기 것으로 취급하므로 기존 스토어·측정치 해석은 불변이고, 실제로 측정된 run은 전부 단일
organizer라 **수치 영향은 0**이다.

**남는 판단**: 타입을 쪼개는(`semantic_profile` 신설) 쪽이 더 근본적이지만
`results/locomo-conv0-memoryos.json`을 포함한 저장 아티팩트의 타입 키가 바뀌므로 하지 않았다.
`kind`가 이미 read 경로(`bench/locomo.py`)의 디스크리미네이터로 쓰이고 있어 어휘가 셋(`type` /
`kind` / `actor`)으로 늘어난 상태다 — 이 축 정리는 Phase 5 비교표를 그리기 전에 한 번 더 봐야 한다.

## 3. 채택한 레이아웃

```
src/agmem/
├── organizers/          # mechanism. 루트의 평범한 모듈 = 프레임워크, 방법론은 전부 패키지
│   ├── base.py          #   [프레임워크] Organizer contract
│   ├── gated.py         #   [프레임워크] AdmissionGated: policy를 임의 organizer에 적용
│   ├── __init__.py      #   [프레임워크] name→class registry
│   ├── amem/            #   방법론 1개 = 패키지 1개 (organizer.py + __init__.py 재수출)
│   ├── nemori/          #   내부 스테이지를 가진 예시: organizer.py + stages.py
│   ├── memoryos/  zep_graph/  ace/  reasoning_bank/  gmemory/  passthrough/
│   └── experimental/    #   의미적 격리(논문 재현 아님) — 위 위치 규칙과 직교
└── policies/            # control policy. 메모리 타입 미선언 + MemoryOp 미발행
    └── admission.py     #   store 게이트: A-MAC(구현) / SAGE(후보)
```

**규칙이 "루트엔 Organizer 서브클래스만"에서 "루트엔 프레임워크만"으로 바뀐 이유**: 전자는
`base.py`/`__init__.py`를 예외로 뺐고, **예외 목록이야말로 루트가 처음 섞인 원인**이었다. 그리고
그 규칙은 `gated.py`를 통과시키면서 범주를 다시 섞었다 — 합성 어댑터는 프레임워크이지 방법론이
아니다. 위치 기반 규칙은 예외가 0개이고, 방법론이 내부 모듈을 갖게 될 때 들어갈 자리가 이미 있다.
단일 파일 방법론까지 패키지가 되는 ceremony 비용은 있으나, `__init__.py` 재수출로 **호출부 40여 곳
전부 무변경**이다.

**`gated.py`가 `organizers/` 루트에 있는 이유**: 그것은 `Organizer`이면서 **방법론이 아니다** —
논문을 구현하지 않고 임의 방법론에 policy를 꽂는 합성 어댑터이므로 `base.py`와 같은 프레임워크
층위다. 판정 기준과 모순되지 않는다: **결정 로직은 `policies/`가 소유하고, 그 결정을 라이프사이클에
꽂는 어댑터는 프레임워크**다. stores/embedders가 Protocol과 어댑터로 갈리는 것과 같은 구조다.

설계 판단 3가지:

1. **`policies/base.py`를 아직 만들지 않는다.** 구현 멤버가 1개인 상태에서 공통 인터페이스를
   만들면 예제 하나에서 추상을 추측하는 것이다. SAGE(같은 seam)가 들어올 때 강제로 뽑아낸다.
2. **`retrieval/steps.py`는 policy가 아니다.** ReadStep 4종은 memory type 키로 등록되고 그 타입을
   생산한 mechanism에 속한다(A-Mem 링크확장, Nemori source 첨부) → mechanism의 부품이다.
   반면 Memory Worth처럼 *누가 생산했는지 무관하게* 호스트가 검색한 것을 억제하는 것은 policy다.
3. **`organizers/nemori/__init__.py`가 `NemoriOrganizer`를 재수출**하므로
   `from agmem.organizers.nemori import NemoriOrganizer`가 단일 모듈 시절과 동일하게 해석된다 —
   import 6곳 무변경.

**불변식을 테스트로 강제** (`tests/test_organizers.py`, 이름은 실제 함수명):

| 테스트 | 강제하는 것 |
|---|---|
| `test_organizers_root_is_framework_only_and_methodologies_are_packages` | 루트=프레임워크, 방법론=패키지 (§3의 위치 규칙) |
| `test_methodology_packages_keep_their_single_module_import_path` | 패키지화가 import 경로를 바꾸지 않음 |
| `test_policies_declare_no_memory_type_and_emit_no_ops` | policy 판정 기준 |
| `test_no_mechanism_imports_the_policies_package` | 직교성 — `gated.py` 외 어떤 organizer도 `policies`를 import하지 않음 |
| `test_feedback_is_owned_by_the_producing_organizer` | 공유 타입에서 피드백 규칙이 생산자 소유임 (§2.6) |

문서 규칙이 아니라 실행되는 규칙이다.

---

## 4. 이전 진술 교정

세션 중 제시했던 배치 표에 오류 2건이 있었고 여기서 교정한다. 원인은 이전 세션의 조사 문서
(`write-path-critics.md` §5)를 1차 소스 확인 없이 인용한 것이다.

| 대상 | 틀린 진술 | 확정 사실 |
|---|---|---|
| GRAVITY | "read-path 기법이라 `retrieval/steps.py` ReadStep 레지스트리에 그대로 들어간다" | **틀림.** anchor는 offline build phase 산출물이고 standalone 저장된다 ⇒ 자체 organizer(새 메모리 타입 3종 + offline 단계는 `consolidate()`) **＋** read step. ReadStep만으로는 절반도 안 된다 |
| RecMem | "`consolidate()` 훅이 그 자리" | **틀림.** 자체 3-tier 시스템 ⇒ 자체 organizer. recurrence 게이트는 `add_memory`의 per-message write 경로 안에 있고, buffer 자체가 검색 소스다 |

RecMem `add_memory`의 실제 3단계(공식 코드 `rec_mem.py:337-411`):
1. episodic에서 유사 에피소드 검색 → `merge_with_epi_thresh` 초과면 **merge + semantic refinement**(LLM)
2. 아니면 subconscious를 `gating_raw_topk`로 검색, `score > min_relevant_score` 히트 수 +1이
   `min_consolidation_cnt` 이상이면 `_consolidate_memory`(LLM)
3. 아니면 **LLM 없이** subconscious에 적재

⇒ write-path 비판 3편 중 **A-MAC만 진짜 cross-cutting policy**다(SAGE·Memory Worth가 그 범주의
동료). 이것이 A-MAC이 `organizers/`에 있으면 안 됐던 이유를 독립적으로 확인해준다.

## 5. 미조사 (필요 시 후속)

MemOS, MIRIX, EM-LLM, H-MEM, MemoryBank, LiCoMemory, Memobase — 전부 자체 시스템으로 보이므로
범주 판정에 영향 없을 것으로 예상하나 미확인. 서베이 2604.16548(메모리 lifecycle 보안/거버넌스)은
retention/decay/eviction 정책을 다루므로 policy 범주 확장 후보.

[[write-path-critics]] [[amac-admission-gate]]
