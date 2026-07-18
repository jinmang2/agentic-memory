# Nemori 라이프사이클 재설계 — management-agnostic 계약 + chaining (2026-07-18)

> 근거 조사: `docs/research/write-path-lifecycle-survey.md` (deep-research, 21클레임 3-vote 검증).
> 스터디 노트: `docs/11-nemori-study.md`. 관련 감사: docs/10, docs/research/fidelity-round5.
> 사용자 결정: ① 마이그레이션 범위 = Nemori+MemoryOS 먼저(나머지 6개는 base 디폴트로 무변경),
> ② 계약 = A안(이벤트 구독 + consolidate 훅), ③ consolidate 트리거 = 명시적 API만,
> ④ §1 인라인/유예 명시 + INVALIDATE 원칙 접합, §2 threshold 3분할 + dedup/consolidation 축 분리.

## 0. 목표와 비목표

**목표**
1. Nemori를 논문 v1 / 논문 v4 / upstream repo / 우리 mixing 네 구성으로 같은 코드에서
   재현·조합할 수 있는 fidelity 스위치 제공 (docs/05 로드맵 `fidelity="paper"`의 구체화).
2. organizer 계약을 distillation(메모리 단위 형성)과 management(dedup/merge/conflict/
   promote/evict)가 서로를 모른 채 조합되는 management-agnostic 구조로 확장하고,
   한 organizer의 산출물이 다른 organizer의 입력이 되는 chaining을 1급으로 지원.
3. 조합 검증 2종: MemoryOS×Nemori(에피소드를 정제된 대화 이력으로) + A-Mem×Nemori
   (에피소드를 note 원문으로) — consumer 2개로 계약의 management-agnostic을 검증.

**비목표 (이번 공사 제외)**
- 나머지 5개 organizer의 새 훅 활용 마이그레이션 (base 디폴트 no-op으로 기존 동작 보존).
- consolidate의 자동 트리거(유휴 디바운스/스케줄러) — 효과 검증 후 후속 공사.
- LongMemEval 하네스, 스토리지 엔진 변경 (MemoryOp 원칙·profile 축 유지).

## 1. 라이프사이클 계약 (organizers/base.py, memory.py)

### 1.1 훅의 위상(位相) 구분 — 인라인 vs 유예

계약은 두 실행 위상을 명시적으로 구분한다. 연구조사 결론(§4)에 따라 **인라인 위상은
방법론 충실 재현의 자리**(Nemori v4 Alg.1, Mem0, Zep이 요구), **유예 위상은 dual-phase
관리의 자리**(LightMem/MOOM/Letta/ACE-lazy 패턴)다. 어느 쪽도 다른 쪽으로 강제하지 않는다.

| 위상 | 훅 | 트리거 | 용도 |
|---|---|---|---|
| 인라인 (ingest 경로) | `on_message`, `on_task_end`, `on_retrieval`, `on_memory_event`, `flush_buffer` | 메시지/태스크/검색/타 organizer 이벤트 | distill + 인라인 integrate (방법론 원형) |
| 유예 (명시 호출) | `consolidate` | `AgenticMemory.consolidate()` 공개 API — 벤치가 인제스트 종료 후·세션 사이에 호출 | 배치 dedup/merge/conflict, 재조직 (결정론 유지) |

### 1.2 이벤트 구독 (chaining)

```python
@dataclass
class MemoryEvent:
    source: str        # 발생 organizer name
    op: OpType         # ADD | UPDATE | MERGE 만 전파
    target_type: str   # "episodes", "semantic", "pages", ...
    target_id: str     # MERGE면 "살아남는" 항목의 id
    payload: dict      # 적용된 op payload 그대로 (재조회 없음)
    supersedes: tuple[str, ...] = ()  # MERGE가 흡수한 같은 타입 항목 id들

class Organizer:
    consumes: tuple[str, ...] = ()   # 구독할 target_type. 인스턴스 설정으로 변경 가능

    def on_memory_event(self, ev: MemoryEvent, ctx: OrganizerContext) -> list[MemoryOp]:
        return []

    def consolidate(self, ctx: OrganizerContext) -> list[MemoryOp]:
        return []
```

퍼사드 배선 규칙:
- `_apply_ops(ops, actor)` 완료 후, ADD/UPDATE/MERGE op를 MemoryEvent로 변환해
  `target_type ∈ consumes`인 **다른** organizer에 리스트 순서대로 전달 (자기 이벤트 제외).
- **depth=1**: `on_memory_event`가 반환한 op는 로그·적용되지만 재전파되지 않는다
  (upstream `_on_episode_created`도 1단계; 다단계 요구가 생기면 그때 depth 파라미터화).
- DELETE/INVALIDATE는 전파하지 않는다 — 관리 결과에 관리가 연쇄되는 사이클 원천 차단.
  **supersession 정보는 INVALIDATE 이벤트가 아니라 MERGE 이벤트의 `supersedes` 필드로
  원자적으로 전달**된다: 발생 organizer는 [MERGE(승계 id, 병합 payload) + 흡수된 id들의
  INVALIDATE]를 같은 배치로 반환하고, 퍼사드는 MERGE op에서 같은 배치의 동일-타입
  INVALIDATE 대상 id들을 `supersedes`로 채운다. consumer는 이벤트 하나로 "새 단위 +
  무엇이 사라졌는지"를 함께 안다 (옛 page 미정리 문제의 해결).
- 전달 순서: 배치 내 op 순서 → organizer 리스트 순서. **chained 구성은 distiller를
  manager보다 앞에 배치**해야 warm_start/flush에서도 이벤트가 인과 순서로 흐른다
  (flush()의 flush_buffer 순회도 리스트 순서).
- 멱등성 요구: INVALIDATE는 기존 `invalid_at`을 보존(최초 시각 유지)하고, 벡터 delete는
  없는 id에 안전해야 한다 (이중 무효화·재처리 대비 — 스토어 계약 테스트에 추가).
- 이벤트 전달은 organizer 작업과 동일한 dispatch 경로(sync/워커)를 탄다 — `sync_write=True`
  재현 원칙 유지.
- `AgenticMemory.consolidate()`: 등록 순서대로 각 organizer의 `consolidate`를 호출·적용.
  반환 op도 evolution log를 통과 (감사성).

### 1.3 관리 연산의 INVALIDATE 원칙

기존 원칙(memory.py: 물리 DELETE는 용량 축출 전용, round-5 X1: 삭제 시 벡터 동반 제거)에
관리 연산을 접합한다:

- **supersession/conflict 해소는 INVALIDATE + ADD**로 표현한다 (DELETE 아님).
  구 항목은 `invalid_at`(+`superseded_by`)이 찍힌 채 로그와 doc에 남는다 — bi-temporal
  감사 추적 보존, 재생 가능성 유지.
- **INVALIDATE 시 검색 제외 보장**: Zep facts는 무효화 후에도 validity 구간과 함께
  렌더되지만(bi-temporal 질의가 목적), semantic/episodes는 무효화되면 서빙에서 빠져야
  한다. 규칙 — `_apply_one(INVALIDATE)`는 대상 타입이 bi-temporal 렌더 타입(facts)이
  아니면 **벡터도 제거**한다(doc/log에는 남음). ghost-hit 방지(X1)와 동일 계열 규칙.
- DELETE는 종전대로 용량 축출(MemoryOS heat, G-Memory REMOVE)과 점수 프루닝 전용.

### 1.4 consolidate 진행 포인터 (커서)

유예 위상의 "어디서 재개하는가"를 evolution log로 해결한다 — 새 인프라 없이:

- **커서 = evolution log 시퀀스 번호** (append-only 로그의 단조 증가 seq). organizer별
  커서를 doc store의 `target_type="state"` 항목(`consolidate:{organizer}`)으로 영속화한다.
- `consolidate(ctx)`는 ① 커서 읽기 → ② 커서 이후 로그에서 자기 관심 타입의 신규/변경
  항목을 수집 → ③ 관리 op 배치(INVALIDATE+ADD/UPDATE) 반환 → ④ **마지막 op로
  커서 전진**(UPDATE, target_type="state")을 반환한다. 커서 전진 자체가 로그에 남아
  consolidation 이력이 감사 가능하다 (state 항목은 embedding_text 없음 — 벡터 미생성).
- 크래시 의미론: op 적용 전에 실패하면 커서가 안 전진했으므로 다음 호출이 재처리한다.
  재처리는 멱등에 가깝게: INVALIDATE는 멱등(§1.2), ADD 중복은 consolidator가 스토어
  현재 상태와 대조(dedup)하므로 수렴한다.
- 위상 분류 명확화(혼동 방지): **MemoryOS heat 승격은 인라인 위상**이다 — 원논문이
  방출 캐스케이드 내 동작으로 정의하므로 consolidate로 옮기지 않는다. Mem0의 비동기
  요약은 메모리 관리가 아니라 컨텍스트 준비라 범위 밖. 이 훅의 준거는 LightMem의
  sleep-time 재조직(항목별 update queue를 우리는 로그 커서로 단순화)과 Letta의 유휴
  재조직(빈도 결정은 호출자 몫 — 우리는 명시적 API)이다.

## 2. Nemori 재구조화 — fidelity 프리셋 + 스위치

### 2.1 스위치 축 (4개, 직교)

인라인 축 3개 + 유예 축 1개. **dedup(인라인)과 consolidation(유예)은 별개 축** —
조합 가능(예: `append` + `semantic_offline`, `dedup` + `semantic_offline`).

```python
NemoriOrganizer(
    fidelity="v1" | "v4" | "upstream" | None,   # 프리셋; None=전부 수동
    segmenter="per_message" | "batch",           # 인라인: 경계 검출
    episode_merge="off" | "llm",                 # 인라인: §3.2.3 병합
    semantic_integration="append" | "dedup" | "llm3way",  # 인라인: 통합
    consolidation="off" | "semantic_offline",    # 유예: consolidate() 훅
    # + 프리셋이 채우는 세부 파라미터들 (2.2), 개별 오버라이드 허용
)
```

### 2.2 프리셋별 threshold 3분할 (출처 혼용 금지)

연구 검증 결과 논문값과 코드값이 다르다 (τ=0.70은 v4 논문, 0.85는 upstream 코드 생성자
기본값). 프리셋은 각자의 출처 값만 쓴다:

| 파라미터 | `v1` (논문 2025-08) | `v4` (논문 2026-04) | `upstream` (main d2a6dff) |
|---|---|---|---|
| 경계 | per_message: σ_boundary=0.7, β_max=25 | batch: w=20 | batch: batch_threshold=20, buffer_min=2, buffer_max=25, >80msg chunk 분할 |
| 병합 | off | llm: K_e=5, LLM idx∈{1..K_e}∪{-1} (P_sel→P_int), 유사도 임계 없음(cos top-K만) | llm: similarity_threshold=0.85, merge_top_k=5, 2-프롬프트(decision→content) |
| 통합 | append | llm3way: τ=0.70 필터 → top-K_m=5 → δ∈{new,merge,conflict} (P_con) | append (참고: PR#19 dedup = top-1, 0.85, ID 재사용 — 병합 전이므로 프리셋 밖 옵션) |
| 유예 | off | off | off |

- 잔여 미해결: **>1h 갭 병합 금지의 소재** (논문/코드 부재 주장이 모두 기각됨) — 구현 첫
  단계에서 upstream `llm/prompts.py`를 직접 확인해 `llm` 병합 프롬프트에 반영하고 이 표를
  갱신한다.
- 현행 구현 = `v1` 프리셋과 동치가 되도록 마이그레이션 (기존 실험 config `nemori` 불변).

### 2.3 내부 구조

`_flush_segment`를 스테이지 객체 합성으로 재구성. 인라인 순서는 v4 Alg.1 충실:
**narrate → episode merge → predict-calibrate → semantic integrate**.

- `Segmenter`: `PerMessageBoundary`(현행 BOUNDARY_PROMPT 경로) / `BatchPartitioner`
  (upstream BATCH_SEGMENTATION 계열 프롬프트 이식, w/threshold 파라미터).
- `EpisodeMerger`: 후보 검색(vec, top-K_e) → 판정 프롬프트 → 병합문 프롬프트.
  스토리지 반영은 **MERGE op(신규 병합 에피소드) + 구 에피소드 INVALIDATE**
  (§1.3 원칙; upstream의 물리 삭제와 서빙 동작은 동일하되 로그에 남음).
- `SemanticIntegrator`: `Append`(현행) / `DedupIdReuse`(PR#19식: top-1 유사도≥임계 시
  기존 ID로 UPDATE) / `ThreeWay`(τ 필터→top-K_m→LLM 3분기; merge→기존들 INVALIDATE+통합문
  ADD, conflict→기존들 INVALIDATE+신규 ADD).
- `SemanticOffline` (**우리 추가 — upstream/논문 모두 부재**): `consolidate()`에서 실행.
  semantic 전수(또는 마지막 consolidate 이후 증분)를 유사도 클러스터링 → 클러스터별 LLM
  merge/conflict 판정 → INVALIDATE+ADD 배치. LightMem/MOOM dual-phase의 Nemori 이식이며,
  `llm3way`(인라인) vs `append+semantic_offline`(유예)를 같은 코드로 ablation 비교한다.
- 프롬프트는 upstream 원문 계보를 docs/11 §4.3 표에 이어 기록 (P_sel/P_int/P_con 축약 이식).

## 3. Manager 마이그레이션 (MemoryOS·A-Mem) + Nemori 조합

- `MemoryOSOrganizer(input="messages" | "episodes")`:
  - `"messages"`(기본): 현행 동작 그대로.
  - `"episodes"`: `consumes=("episodes",)`, `on_message`는 no-op,
    `on_memory_event(episode_created)`가 STM append를 대체 — Nemori 서사 에피소드가
    MemoryOS page 단위가 되어 F_score 분절·heat·승격·축출 관리를 받는다
    (스터디 제안: "Nemori를 MemoryOS의 대화 이력 정제 레이어로").
- semantic 타입 충돌 방지: MemoryOS LPM은 `kind="profile"`(현행), Nemori는
  `kind="fact"`를 명시해 출처 구분. 검색 채널은 변경 없음.
- **MERGE/supersedes 처리 (manager 공통 규칙)**: manager는 소비한 단위의 역매핑
  (`unit_id → 자기 산출물 id`; MemoryOS는 episode→page, 인메모리 dict — `_heat`와 동일한
  재시작 휘발성, 문서화된 기존 한계)을 유지한다. `supersedes`가 담긴 이벤트 수신 시:
  - 흡수된 단위가 아직 STM/버퍼에 있으면 → 버퍼에서 교체(승계 payload로 대체).
  - 이미 page화됐으면 → 해당 page의 source가 **전부** superseded일 때만 page INVALIDATE,
    일부면 유지(요약문에서 부분 제거는 LLM 재작성 비용 대비 이득 없음 — 중복은 heat가
    흡수). 이중 무효화는 §1.2 멱등 규칙이 방어.
  - 승계 단위(target_id)의 UPDATE/MERGE 이벤트를 이미 page화한 뒤 받으면 → 무시(문서화된
    staleness 허용; STM에 있으면 교체).
- 1차 chained 실험은 `fidelity="v1"`(merge off)로 시작해 이 경로를 배제하고, v4-chained는
  supersedes 규칙 검증 후 ablation으로 둔다.
- **A-Mem도 동일 패턴으로 이번 공사에 포함**: `AMemOrganizer(input="messages" | "episodes")`.
  `"episodes"`면 `consumes=("episodes",)`, `on_message` no-op, 에피소드가 note 원문이 되어
  A-Mem의 링크·이웃 진화 관리를 받는다 (방법론 무변경 — 입력 전환만). manager 공통 규칙
  적용: 흡수된 에피소드의 note는 supersedes 수신 시 INVALIDATE(역매핑 episode→note).
  consumer 2개(MemoryOS·A-Mem)로 계약의 management-agnostic을 검증한다.
- 조합 config: `organizers=[NemoriOrganizer(fidelity="v1"), MemoryOSOrganizer(input="episodes")]`
  (A-Mem 조합은 `[NemoriOrganizer(fidelity="v1"), AMemOrganizer(input="episodes")]`).
- 나머지 5개 organizer(zep_graph/ace/reasoning_bank/gmemory/passthrough): 무변경 (신규 훅
  base 디폴트).

## 4. 효율/확장성

- `batch` segmenter는 경계 LLM 콜을 메시지당 1회→세그먼트당 1회로 절감 (~1/w).
  merger +1~2콜/에피소드, llm3way +1콜/fact 배치, consolidate는 호출 시에만.
- `BudgetTracker`에 위상 태그(ingest/merge/integrate/consolidate)를 추가해 방법론 비교의
  1급 비용 메트릭으로 계측 (docs/04 llm/budget.py 원칙의 확장).
- 하드웨어 제약(RTX/WSL)은 설계에 반영하지 않는다 — 실험은 가용 capa를 따른다 (사용자 합의).

## 5. 검증 계획

- 단위: 이벤트 전파(consumes 매칭·자기이벤트 제외·depth=1·DELETE/INVALIDATE 비전파),
  MERGE `supersedes` 채움과 manager 처리(STM 교체 / 전부-superseded page INVALIDATE /
  부분 유지), INVALIDATE 멱등(invalid_at 최초 보존)·벡터 delete 멱등(스토어 계약),
  INVALIDATE 벡터 제거 규칙(facts 예외), consolidate 커서(영속·재개·크래시 재처리 수렴),
  프리셋 4종 스모크(스위치→스테이지 해석), chained MemoryOS·A-Mem(input="episodes")
  시나리오, consolidate() 공개 API.
- 회귀: 기존 조직자 테스트 전부 통과 (`nemori`=v1 동치, `memoryos` 기본 동작 불변).
- LoCoMo conv0 configs (스위치 명시):

  | config | segmenter | merge | integration | consolidation | 비고 |
  |---|---|---|---|---|---|
  | `nemori` | per_message | off | append | off | =v1, 현행 결과와 연속성 |
  | `nemori_v4` | batch(w=20) | llm(K_e=5) | llm3way(τ=0.70, K_m=5) | off | 논문 v4 재현 |
  | `nemori_upstream` | batch(20/2/25) | llm(0.85/5) | append | off | repo main 재현 |
  | `nemori_mix` | batch(w=20) | llm(K_e=5) | append | semantic_offline | 우리 기여: 인라인 vs 유예 통합 비교축 |
  | `nemori_memoryos` | per_message | off | append | off | +MemoryOS(input="episodes") chained |
  | `nemori_amem` | per_message | off | append | off | +A-Mem(input="episodes") chained |

  결과 스탬프에 fidelity 스위치 전체 기록 (docs/01 재현성 원칙).
  `nemori_mix`↔`nemori_v4` 비교가 "같은 관리를 인라인으로 vs 유예로" ablation이 된다.

## 6. 열린 질문 / 후속

1. >1h 갭 병합 금지 소재 확정 (구현 1단계에서 upstream prompts.py 확인 후 §2.2 갱신).
2. consolidate 자동 트리거(유휴 디바운스/스케줄러) — 효과 검증 후.
3. 나머지 6개 organizer의 새 훅 활용 (예: ACE lazy refine→consolidate, Zep community
   refresh→consolidate) — 점진 이관.
4. PR #19 병합 추적 (2026-07-18 기준 open).
