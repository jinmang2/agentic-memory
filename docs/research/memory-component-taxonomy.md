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
| LightMem (2510.18866) | sensory→short-term→long-term 3단계 자체 파이프라인, offline consolidation이 long-term 단계의 정의적 구성요소 | 후보 |
| HyMem (2602.13933) | dual-granular 저장 + 2-tier 자체 retrieval + reflection | 후보 |

### 2.2 control policy (→ `policies/`)

| 논문 | 지배하는 연산 | 근거 |
|---|---|---|
| **A-MAC** (2603.04549, ICLR'26) | **store** (admission) | 5-feature 가중합 게이트. 자체 메모리 타입 없음. 감사: `amac-admission-gate.md` |
| **SAGE** (2605.30711) | **store** (admission) | 제목이 "A Novelty Gate". vMF 밀도 추정 + adaptive threshold로 ADD/NOOP/LLM-merge 라우팅. **"a drop-in binary gate for A-Mem"**으로 명시, LLM 콜 16–18% 절감. 자체 메모리 타입 없음 |
| **Memory Worth** (2604.12007, "When to Forget") | **discard** + **retrieve** 억제 | "lightweight, theoretically grounded foundation for staleness detection, retrieval suppression, and deprecation decisions". **메모리 유닛당 스칼라 카운터 2개**만 필요하고 "retrieval과 episode outcome을 이미 로깅하는 아키텍처에 추가 가능" |

⇒ **범주는 실재하며 ≥3 멤버 / ≥2 seam.** 특히 SAGE는 **A-MAC과 같은 seam(store)·같은 호스트
(A-Mem)**여서, `policies/admission.py`가 단발 모듈이 아니라 최소 2 멤버가 들어올 자리임이 확정된다.

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

## 3. 채택한 레이아웃

```
src/agmem/
├── organizers/          # mechanism. 루트에는 Organizer 서브클래스만
│   ├── base.py          #   (예외 2: contract)
│   ├── __init__.py      #   (예외 2: name→class registry)
│   ├── amem.py  memoryos.py  zep_graph.py  ace.py  reasoning_bank.py
│   ├── gmemory.py  passthrough.py
│   ├── nemori/          # 스테이지를 소유하는 방법론 → 서브패키지
│   │   ├── __init__.py  #   NemoriOrganizer 재수출
│   │   ├── organizer.py
│   │   └── stages.py
│   └── experimental/    # 논문 재현이 아닌 합성 (격리 유지)
└── policies/            # control policy. 메모리 타입 미선언 + MemoryOp 미발행
    └── admission.py     #   store 게이트: A-MAC(구현) / SAGE(후보)
```

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
