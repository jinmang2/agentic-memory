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

**불변식을 테스트로 강제**: `tests/test_organizers.py::test_organizers_package_root_holds_only_organizers`
가 `organizers/` 직하 모듈 전부가 Organizer를 정의하는지 검사하고,
`::test_policies_declare_no_memory_type_and_emit_no_ops`가 policy 판정 기준을 검사한다.
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
