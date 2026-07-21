# Organizer Experimental Split — 설계 (2026-07-21)

> **부분 대체 (2026-07-21 머지 전 충실도 리뷰, docs/10 N1):** 아래 §2·§3.4·§4 표는
> `ThreeWayIntegrator`도 `experimental/nemori_mixing.py`로 옮긴다고 기술하나, 이후
> 리뷰가 이를 **코어(`nemori_stages.py`) 잔류**로 되돌렸다 — v4 §3.3.3 P_con 자체가
> 논문 메커니즘이기 때문. experimental엔 `SemanticOfflineConsolidator`만 남는다.
> ThreeWay 배치에 관한 한 이 스펙보다 docs/10을 따를 것.

브랜치: `refactor/organizer-experimental-split`. 베이스라인: **125 passed, 1 skipped**.

## 목표

organizer 계층을 **논문 충실 코어**와 **실험적 합성(experimental)**으로 분리한다.
핵심 불변식: **논문 그대로의 로직(또는 controllable bug-fix — Nemori 프리셋 방식)은
온전히 남고, 기존 구축물을 훼손하지 않는다. 전 과정 동작 보존(no behavior change) —
검증 기준은 125-test suite green 유지.** 격상/측정은 이 리팩터 이후 단계.

## 무엇이 experimental인가 (범위 확정)

- **크로스-organizer 체인**: `input="episodes"` 소비자 모드 (amem, memoryos) +
  `nemori_amem`, `nemori_memoryos` config. 논문 근거 없는 우리 합성.
- **nemori 내부 our-mixing**: `semantic_integration="llm3way"`(ThreeWayIntegrator),
  `consolidation="semantic_offline"`(SemanticOfflineConsolidator) — Nemori 논문·upstream
  둘 다에 없음 (`nemori_mix` config).

experimental 아님 (유지): nemori `v1/v4/upstream` 프리셋(전부 논문 버전),
`append`/`dedup` 인테그레이터(baseline), `_mixed` raw-RAG config(ablation 대조군),
골격 organizer(zep/gmemory — 별도 '미완성' 범주).

## 설계: 어댑터 추출

### 관심사 분리
- **충실 organizer** = "유닛으로 메모리 구성 + 자기 라이프사이클 관리". 공개 진입점만
  노출, **체이닝 인지 제거**: `input=`/`consumes`/`on_memory_event`/`MemoryEvent` import
  삭제 → messages-only 순수체.
- **어댑터**(experimental/) = "상위 organizer 이벤트 구독 → 유닛/은퇴 호출로 번역".
  합성 glue(구독, payload→Episode 재구성=text seam, op 디스패치, 소스-id 매핑) 전부.

### `experimental/chained.py` — `ChainedConsumer(Organizer)`
```
ChainedConsumer(wrapped: Organizer, source_type: str = "episodes")
  name     = wrapped.name           # actor 귀속 보존 (테스트가 actor=="memoryos" 확인)
  consumes = (source_type,)
  on_message(ep, ctx) -> []         # raw 스트림 무시
  on_memory_event(ev, ctx):
    unit = Episode(content=payload["content"], role="episode",
                   id=ev.target_id, meta={"date": payload["timestamp"]})  # seam 격리
    supersede: wrapped.retire(ids, ctx) 있으면 위임          # MemoryOS
               없으면 어댑터 추적 생성물 INVALIDATE            # A-Mem generic
    UPDATE:    wrapped.patch_unit(unit) 있으면 호출, 없으면 stale  # 동작 보존
    ADD/MERGE: ops = wrapped.on_message(unit, ctx)
               retire 프로토콜 없으면 _produced[ev.target_id] = ops[0].target_id/type
    return ...
```

### 충실 organizer가 노출하는 프로토콜 (전부 organizer 자기 로직 — 신규 아님)
| organizer | 노출 | 어댑터가 쓰는 법 |
|---|---|---|
| **A-Mem** | `on_message`만 | 어댑터가 `_produced` 매핑 + generic `INVALIDATE(notes, id)` 소유. amem.py에서 체인 코드 **완전 제거** |
| **MemoryOS** | `on_message` + `retire(ids, ctx)`(구 `_retire` 승격) + `patch_unit(unit)`(구 UPDATE-in-place) | 어댑터가 위임. 역인덱스·페이징은 페이징과 얽혀 MemoryOS 잔류(정직) |

A-Mem은 supersede→생성물 1:1 INVALIDATE라 generic. MemoryOS는 page가 다중 소스라
"전 소스 흡수 시에만 invalidate" — 내부 역인덱스 필요 → 커스텀 `retire`.

### `experimental/nemori_mixing.py`
`ThreeWayIntegrator`, `SemanticOfflineConsolidator`를 `nemori_stages.py`에서 이동.
nemori.py는 experimental 프리셋 선택 시에만 import. **순수 재배치 = 동작 보존.**

## 이동 표면 (전부 동작 보존)
| 파일 | 변경 |
|---|---|
| `experimental/__init__.py`, `chained.py`, `nemori_mixing.py` | 신규 |
| `amem.py` | `input=`/`on_memory_event`/`_episode_notes`/`consumes` 제거, `_ingest`는 `on_message` 뒤로 유지 |
| `memoryos.py` | `input=`/`on_memory_event`/`consumes` 제거; `_retire`→공개 `retire()`, UPDATE-patch→공개 `patch_unit()` |
| `nemori_stages.py` | ThreeWay/SemanticOffline → experimental로 이동 (재export shim 없음) |
| `nemori.py` | 두 클래스 import 출처 변경 |
| `exp_locomo_conv0.py` | `nemori_amem`/`nemori_memoryos` → `ChainedConsumer(...)`; experimental config 그룹 분리 주석 |
| `tests/test_lifecycle.py` | Task 12/13 테스트를 새 어댑터 API로 갱신(동일 단언) |
| `docs/13 §5`, `docs/10` | experimental 경계 명시 |

## 검증 (cleared-session이 재확인할 것)
1. `uv run pytest -q` → **125 passed, 1 skipped** 유지 (동작 보존 증명).
2. 충실 organizer 순수성: `grep -n "input\|consumes\|on_memory_event\|MemoryEvent" amem.py memoryos.py` →
   체이닝 흔적 0 (organizer 자기 메서드 `retire`/`patch_unit`은 허용).
3. experimental 경계: `input="episodes"` 문자열은 코드베이스에서 사라지고,
   체인은 `ChainedConsumer(X(), "episodes")`로만 표현.
4. nemori v1/v4/upstream 프리셋 동작 불변 (프리셋 테스트 green).

## 격상 경로 (이 리팩터 밖)
experimental/의 항목은 LoCoMo E2E 실측 후 이득 확인 시 seam 제대로 고쳐 core 승격,
아니면 격리 유지/삭제. 지금은 경계만 세운다.
