# Write-path 라이프사이클 연구조사 — distillation/management 훅 관점 (2026-07-18)

> 목적: `on_message`/`on_task_end` 2훅 계약이 Nemori 및 타깃/최신 방법론들의 write-path를
> 담기에 충분한지 검증하고, management-agnostic 계약 + chaining 설계의 근거를 수집.
> 방법: deep-research 하네스 (24 소스 fetch → 120 클레임 추출 → 상위 25건 3-vote 적대 검증,
> 21 확정 / 4 기각). 검증 예산이 Nemori에 집중됨 — Nemori 관련은 **검증됨(✓)**,
> 그 외 시스템은 1차 소스 인용 기반이나 **미검증(○)** 표기. 조사일 2026-07-18.

---

## 1. Nemori 3개 변종의 write-path (전부 검증 ✓)

### 1.1 공통 구조: 인라인 캐스케이드 + 명시적 chaining

세 변종(논문 v1, 논문 v4, upstream main) 모두 **버퍼/경계 트리거의 완전 인라인 캐스케이드**이며
distillation과 management가 한 패스에 결합. **sleep-time/offline/background consolidation은
논문·코드 어디에도 없음** (v4 원문 전수 grep 0건). upstream은 오히려 semantic 생성을
EventBus fire-and-forget에서 **직접 await로 되돌림** — merge가 참조 에피소드를 지우는
FK race 제거 목적 ("await directly instead of fire-and-forget via EventBus").

chaining은 Nemori의 1급 구조다: v4 초록이 스스로 "two cascading modules: Episodic Memory
Integration → Semantic Knowledge Distillation"로 정의하고, upstream은
`_on_episode_created` 훅으로 (병합 반영된) 에피소드를 semantic 층에 넘긴다
(`episode = merged_ep  # Use merged episode for downstream`).

주의: **"Nemori가 distill/manage를 분리(management-agnostic)했다"는 클레임은 기각(1-2)** —
Algorithm 1은 둘을 한 패스에 결합. management-agnostic 프레임은 Nemori의 것이 아니라
우리 라이브러리의 설계 목표다 (§4의 다른 근거들로 정당화).

### 1.2 버전별 차이 (v4는 2026-04-16, 제목도 "What Deserves Memory: Adaptive Memory
Distillation for LLM Agents"로 개제 — arXiv에 ACL'26 표기는 없음)

| 모듈 | v1 (2025-08) | v4 (2026-04) | upstream main (HEAD d2a6dff) |
|---|---|---|---|
| 경계 검출 | per-message f_θ, 트리거 `(b∧c>σ)∨(\|M\|≥β_max)`, σ=0.7, β_max=25 | **Local Message Partitioning**: 버퍼가 w=20 도달 시 LLM 일괄 분할 `O←f_LLM(P_par∥B_t)` (w∈[5,40] 성능 ±1%) — f_θ 언급 0회 | buffer_size_min=2에서 async `_process()` 스폰, **batch_threshold=20** 이상이면 배치 분할(80msg 초과는 chunk), 미만이면 단일 그룹. buffer_size_max=25, 명시적 flush() |
| 에피소드 병합 | 없음 | **§3.2.3**: cos top-K_e=5 후보 → LLM이 idx∈{1..K_e}∪{-1} 선택(P_sel) → 병합문 생성(P_int). 서사 생성 직후·semantic 증류 직전 인라인 (Alg.1 line 8-9) | merger.check_and_merge: Qdrant top-k=5(+1 self 제외) → 2-프롬프트 LLM(decision→content), 생성자 기본 similarity 0.85. 병합 시 구 에피소드를 PG+Qdrant 양쪽에서 삭제 |
| semantic 통합 | append-only (merge/conflict/dedup/evict/decay 언급 0건) | **§3.3.3**: τ=0.70 필터 후 top-K_m=5 대상 LLM 3분기 δ∈{new, merge, conflict} (P_con). merge=통합문으로 대체, conflict=구 항목 purge 후 대체 | **append-only** (fresh uuid4라 ON CONFLICT 불발). PR #19(2026-06-03, 미병합·리뷰 0건): top-1 0.85 임베딩 유사도 ID 재사용 dedup만 — LLM 분기 없음, `enable_semantic_dedup=True` 플래그, `_on_episode_created` 내 인라인 |
| 저장소 | (명세 없음, dual vector store) | (명세 없음) | PR #13(2026-03-25 병합, 33커밋): **PostgreSQL 16**(episodes/semantic_memories/message_buffer 테이블, tsvector/GIN, agent_id 멀티테넌트) + **Qdrant**(gRPC, pgvector 대체). 전부 async |

**미해결**: >1h 갭 병합 금지의 소재 — "논문에 없다"와 "코드에 없다" 클레임이 **둘 다 기각**되어
소재가 진짜 열린 상태. docs/11(3차 감사)은 upstream MERGE_DECISION 프롬프트에 있다고
기록했으므로, 구현 시 upstream `llm/prompts.py`에서 직접 재확인할 것. 0.85도 논문엔 없고
코드 생성자 기본값으로만 확인됨 (논문의 유일한 유사도 임계는 τ=0.70).

## 2. 타깃 논문들의 write-path (미검증 ○, 1차 소스 인용; MemoryOS STM 항목만 ✓)

| 시스템 | 트리거 | 라이프사이클 | distill/manage 결합 | chaining |
|---|---|---|---|---|
| MemoryOS | per-turn STM append; STM 용량(7p) 초과 시 FIFO 방출 ✓; MTM heat>τ=5 승격; 용량 초과 시 최저 heat 축출 | append→segment(F_score=cos+Jaccard, θ=0.6)→promote/evict. LPM도 100개 FIFO, 승격 후 L_interaction 리셋(재승격 방지) | 인라인 결합. **단 N_visit은 retrieval-time 피드백** — read 경로가 write 신호를 만든다 | STM→MTM→LPM 3계층 캐스케이드 |
| Zep/Graphiti | per-episode 인라인 (최근 n메시지 컨텍스트) | extract→resolve(dedup, 동일 entity쌍 스코프)→bi-temporal invalidate | 인라인 결합. **예외: community 층은 인라인 label propagation이 휴리스틱일 뿐, 주기적 full refresh 필요** — 유일한 deferred 요구 지점. (Mem0 논문의 관찰: Zep 인제스트 직후 검색이 부실하고 수 시간 뒤 개선 — 백그라운드 처리 존재 방증) | episodes→entities→communities 3계층 |
| ACE | 온라인(쿼리당 delta 1개) / 오프라인(≤5 epoch) 겸용 | Generator→**Reflector(distill)→Curator(integrate)** 역할 분리; integrate는 비-LLM 결정적 병합(id별 카운터) | **grow-and-refine(dedup/prune)을 명시적으로 분리**: "proactively (after each delta) or lazily (only when the context window is exceeded)" — 관리 패스가 설정 가능한 별도 훅 | Reflector 출력→Curator 입력 |
| ReasoningBank | task-end | judge→distill(성공+실패 모두, ≤3items)→append | 관리 부재가 설계 의도 ("minimal consolidation strategy: directly added without additional pruning") | MaTTS: 스케일링 궤적들→self-contrast→증류 |
| A-Mem | per-session note 생성 (LLM 태그 추출) | note→link→이웃 진화(update) | 인라인 결합. release/soft-delete 원시연산 없음 | 신규 note→이웃 note 진화 |
| G-Memory | (이번 조사에서 클레임 미확보 — 기존 docs/research/g-memory.md 참조: task-end sparsify→insight, reward 프루닝) | — | — | — |

## 3. 최신 시스템: dual-phase(inline+deferred)가 2025-26의 수렴 패턴 (미검증 ○)

- **Mem0**: 메시지쌍 단위 extract→update가 인라인 한 패스 (fact별 top-s=10 검색 후 LLM이
  ADD/UPDATE/DELETE/NOOP 툴콜). 단 **대화 요약 갱신은 비동기 별도 모듈**. Mem0g는 충돌
  triplet을 soft-invalidate. LETHE 벤치(arXiv 2606.15903)의 관찰: 이 라우터(infer=True)는
  삭제 정밀도를 붕괴시킴(68.3→43.6%) — 인라인 결합의 실증적 약점.
- **LightMem** (arXiv 2510.18866): **분리의 대표 사례**. 온라인은 soft update로 LTM에 삽입만,
  재조직/dedup/충돌해소는 전부 오프라인 "sleep-time" 단계로 유예. 토픽 분절은
  attention∩similarity 경계, 버퍼 토큰 임계에서 요약. 항목별 update queue로 병렬 consolidation.
  A-MEM 대비 +2.7~9.65% 정확도에 토큰 32~117배 절감 — **유예 관리가 정확도를 희생하지 않음**.
- **Letta sleep-time** (arXiv 2504.13171): 주 에이전트에서 메모리 편집 도구를 **제거**하고
  별도 sleep-time 에이전트가 유휴기에 재조직 (빈도 파라미터로 비용-품질 트레이드). 동일
  컨텍스트 10쿼리 시 쿼리당 비용 2.5배 절감 — 유예 consolidation의 상각 논리.
- **LangMem**: `create_memory_manager`(순수 추출, 저장 없음)와 `create_memory_store_manager`
  (저장 통합)를 **팩토리 수준에서 분리**. `ReflectionExecutor`가 after_seconds 디바운스로
  백그라운드 실행 — 유효 트리거는 "활동 후 정지(quiescence)". enable_inserts/updates/deletes
  플래그로 통합 패스의 관리 동작을 토글. hot-path(인라인) vs subconscious(백그라운드) 2모드.
- **SeCom**: 세그먼트 단위 저장, 관리 단계 전무 (압축은 read-side denoising).
- **Memp**: Build(궤적→절차 메모리 2단계 granularity)와 Update(t개 태스크마다 배치;
  vanilla/validation/reflexion 중 reflexion 최고)가 **명시적 분리 단계**.
- **Memory 서베이** (arXiv 2512.13564, 2025-12): 라이프사이클을 **Formation / Evolution
  (=Consolidation·Updating·Forgetting) / Retrieval** 3과정으로 공식화 — distill vs manage
  분리 프레임과 정합. MOOM·LightMem의 dual-phase를 eventual-consistency 설계로 명명하고,
  §7.8.2에서 "다음 세대는 online-only를 넘어 수면 유사 오프라인 consolidation 구간 필요"라고
  전망.
- **LETHE 벤치** (arXiv 2606.15903, 2026-06): LLM 배치 지점(inscribe-time 추출 vs
  mutation-time 관리 훅)이 **상보적** — inscribe-time은 정규화 회복(100%)·의도적 삭제 0%,
  mutation-time 훅은 의도적 삭제 78-85% 회복. Letta+mutation 훅 65.5→76.1%. **distill-time과
  manage-time 훅을 분리·체이닝 가능하게 설계하라는 직접적 근거.**

미커버: MemOS, MIRIX, MemoryBank, EM-LLM, H-MEM, G-Memory (fetch 예산 초과 — 필요 시 후속 조사).

## 4. 종합: 계약 설계에 주는 시사점

1. **2훅(on_message/on_task_end)은 "트리거"로는 충분하나 "역할"로는 불충분.** 조사된 모든
   트리거는 {per-message, buffer/경계, task-end, retrieval-time, 유휴/유예}로 수렴하고, 앞의
   셋은 기존 훅+organizer 내부 버퍼로 표현 가능(검증된 Nemori/MemoryOS가 그 증거). 빠진 것은
   (a) **retrieval-time 피드백** — 이미 `on_retrieval`로 보유, (b) **유예 consolidation 훅** —
   타깃 논문 재현에는 필수가 아니나(검증된 시스템 중 필수는 없음; Zep community refresh만
   부분 예외) 2025-26 수렴 패턴(LightMem/MOOM/Letta/LangMem/ACE-lazy/Memp)이자 우리
   mixing(consolidate)의 자리, (c) **chaining** — 한 organizer의 산출물이 다른 organizer의
   입력이 되는 경로 (Nemori `_on_episode_created`, Zep 3계층, LightMem 3단 캐스케이드,
   ACE Reflector→Curator가 전부 이 모양).
2. **management-agnostic의 의미**: distiller는 자기 산출물(메모리 단위)을 누가 어떻게 관리
   (dedup/merge/conflict/promote/evict)하는지 모르고, manager는 단위의 출처를 모른다.
   근거: 서베이의 Formation/Evolution 분리, LangMem의 extract/store-integrate 팩토리 분리,
   ACE의 proactive/lazy refine 스위치, LETHE의 배치 지점 상보성.
3. **인라인 결합도 보존해야 함**: Nemori v4·Mem0·Zep은 통합 판정이 인라인이어야 충실 재현
   (v4 Alg.1 순서: narrate→merge→distill→integrate). 계약은 "인라인 통합"과 "유예 통합"을
   모두 표현해야지 유예로 강제하면 안 됨.
4. **읽기→쓰기 피드백은 이미 옳게 설계됨**: MemoryOS N_visit이 retrieval-time에 갱신되는 것을
   round-5에서 `on_retrieval`로 복원한 것이 조사 결과와 일치.

## 5. 우리 Nemori fidelity 스위치에 직결되는 사실 (전부 ✓)

- v1 프리셋 = 지금 구현 (per-message σ=0.7/β_max=25, merger 없음, append-only). 정확.
- v4 프리셋 = batch 분할(w=20) + merger(K_e=5, P_sel/P_int) + 3분기 통합(K_m=5, τ=0.70).
  **τ=0.70이 논문값, 0.85는 코드 생성자 기본값** — 프리셋 분리 시 값 구분할 것.
- upstream 프리셋 = batch(batch_threshold=20, min=2, max=25, >80 chunk) + merger(0.85/top-5)
  + **append-only** (PR #19 병합 전까지). #19가 병합되면 top-1 0.85 ID 재사용 dedup 추가.
- 저장소: upstream은 PG16+Qdrant지만 이는 우리 profile 축(full)과 직교 — organizer가
  스토리지를 지정하지 않는 MemoryOp 원칙 유지가 타당 (§1.1의 FK race가 오히려 우리
  로그-선행 append 설계의 정당성을 보여줌).

## 6. 남은 열린 질문

1. >1h 갭 병합 금지의 정확한 소재 (구현 시 upstream prompts.py 직접 확인).
2. G-Memory/MemOS/MIRIX 등 미커버 시스템의 라이프사이클 (후속 조사 필요 시).
3. PR #19 병합 여부 추적 (2026-07-18 기준 open, 리뷰 0).

소스: arxiv.org/abs/2508.03341 (v1/v4 HTML), github.com/nemori-ai/nemori (+PR #13, #19),
arxiv 2506.06326 (MemoryOS), 2501.13956 (Zep), 2510.04618 (ACE), 2509.25140 (ReasoningBank),
2504.19413 (Mem0), 2512.13564 (서베이), 2606.15903 (LETHE), 2510.18866 (LightMem),
2504.13171 + letta.com (sleep-time), langchain-ai.github.io/langmem, neo4j.com (Graphiti),
2502.05589 (SeCom), 2508.06433 (Memp).
