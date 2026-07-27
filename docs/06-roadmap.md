# 개발 로드맵

> 목표: ① 8개 방법론 직접 구현 ② MCP 배포 ③ 0.5B급 소형 모델로 PC에서 table 재현 ④ 필요 시 보조모델 학습.
> 각 Phase는 "동작하는 수직 슬라이스"를 끝점으로 함. 기간은 스터디 병행 기준 러프 추정.

## Phase 0 — 스캐폴딩 (1주) — 2026-07-16 대부분 완료

- [x] `uv init` + 패키지 구조 (`docs/04-architecture.md` 레이아웃) — Python 3.12 pin
- [x] `capabilities/` 감지·리졸버 + profile 3종 (`docs/01-capability-system.md`)
- [x] `core/types.py`, `core/ops.py` (MemoryOp + EvolutionLog)
- [x] `stores/sqlite_doc.py` + `sqlite_vec.py` (FTS5 포함) + `numpy_vec.py` fallback + 테스트 26개 통과
- [x] retrieval v0: Dense+Lexical → RRF (pipeline 골격)
- [x] LLM 데몬 셋업: llama.cpp **CUDA 빌드(compute 7.5 직접 컴파일)** + Qwen3-0.6B-Q8_0, `scripts/serve-llm.sh`(:8080, GPU 전 레이어 오프로드). structured output 스모크 **3/3 유효 JSON, 1.6s/call** (CPU 스왑 병목 143s → GPU 90×). `llm/client.py` 역할별 라우팅 완료
- [x] `llm/structured.py`: guided_json + 재시도 + drop 카운터 (0.5B 대응의 토대)
- [x] 실호스트 스모크: capability 감지(RTX 2060/7.8GB) + 강등 로그 + 한국어 add→search 확인

**완료 기준 달성**: `AgenticMemory(organizers=["passthrough"])` add→search 동작 확인.
참고: 호스트에 qdrant(6333)·redis(6379) 포트 활성 감지됨 — full 프로파일 테스트에 활용 가능.

## Phase 1 — 첫 방법론 2종 + 추상화 검증 (2주) — 2026-07-16 완료

구현 난이도 "하"이면서 설계 공간의 양극단인 둘을 먼저:

- [x] **ReasoningBank organizer** (self-judge→성공/실패 증류→append; field-level fallback으로 깨진 아이템만 스킵)
- [x] **A-Mem organizer** (노트+링크+진화 배치 호출; **버그 수정판**: 이웃을 인덱스가 아닌 ID로 참조(#32), cosine 보장(#24), 환각 ID 필터, silent skip 금지. fidelity="paper"는 하이퍼파라미터만 재현)
- [x] retrieval 파이프라인 v1 (Dense+Lexical → RRF → MMR; vector store에 `get()` 추가)
- [x] 비동기 워커 + `flush()` (raw episode는 항상 동기 인덱싱 — read는 write를 기다리지 않음)
- [x] `bench/harness.py` 골격: multi-run mean±std + calls/tokens/latency + 조건 스탬프(commit/profile/embedder)
- [x] **0.5B 방어층 실전 검증**: Qwen3-0.6B E2E에서 top-level 배열 반환 실패 모드 발견 → 스키마 유도 배열 코어싱 추가 → **drop 0회** (notes 2 + strategies 1, 5.7s/6 LLM calls)

**완료 기준 달성**: MemoryOp 추상화가 대화형(A-Mem)·태스크형(ReasoningBank) 양극단을 누수 없이 수용. 테스트 39개 통과.
발견 사항: 0.6B judge는 실패 궤적을 success로 오판하는 사례 확인 — judge role의 상위 모델 라우팅(티어링) 필요성 실증.

## Phase 2 — 벤치마크 하네스 + 1차 재현 (2–3주) — 진행 중

- [x] **LoCoMo** 파이프라인 (F1/BLEU-1 — judge 불필요라 가장 저렴한 시작점)
- [x] 1차 재현 (conv0, 로컬 0.6B 단독): passthrough F1 22.85 vs A-Mem(수정판) 23.25.
      비용 841 calls/946s vs 0 calls/6.3s. multi-hop은 오히려 -1.6 — organizer 모델
      품질 종속성 실증. 상세: `docs/08-amem-implementation-review.md` §4
- [x] LoCoMo conv0 4-way 완료 (passthrough/A-Mem/Nemori/MemoryOS — docs/09-results-summary.md)
- [~] **LongMemEval_S** 파이프라인 — **라이브러리 배선 완료**(`bench/longmemeval.py`, 2026-07-26.
      같은 날 공식 소스 재대조 감사로 5건 수정 — 아래 §2026-07-26 감사).
      judge pin `gpt-4o-2024-08-06`는 `judge_answer`가 첫 콜 전에 강제(공식 집계기가 assert하는 값),
      reading `con` + history `json`은 공식 프롬프트 원문 그대로, full-context 베이스라인
      (`render_sessions`)까지 포함. **미작성**: CLI 드라이버와 ingest 아티팩트 캐시 —
      인스턴스마다 메모리를 새로 만드는 구조라 500질문 = 500 ingest이고, 측정 승인 전에는
      드라이버를 쓸 이유가 없다. 데이터 버전(cleaned 여부)은 결과와 함께 기록 필요.
- [ ] 1차 재현 실험 (PC, 0.6B extract + API judge):
  - A-Mem × LoCoMo — 원논문 GPT-4o-mini 수치와 방향성 비교
  - ReasoningBank류 × 간단 태스크(수학/코딩 스트림, reasoning-bank-slm 프로토콜 참고)
- [ ] 재현 리포트 v1: 절대치가 아닌 **baseline 대비 향상 + 3-run 편차** 보고

**완료 기준**: `agmem-bench` 한 줄로 표가 나오고, 결과에 전체 실험 조건이 스탬프됨.

## Phase 3 — 나머지 방법론 + MCP 배포 (3–4주) — 2026-07-16 완료 (Claude Code dogfooding 제외)

- [x] **Nemori organizer** (분절→서사→predict-calibrate; 시간 절대화 포함)
- [x] **MemoryOS organizer** (STM/MTM/LPM, heat, LFU)
- [x] **ACE organizer** (3-role, delta ops, dedup 0.90 — 기본 on으로 원논문 함정 회피)
- [x] **Zep-graph organizer** (community detection은 TODO) (graph store 위: entity resolution→bi-temporal fact→invalidation→label propagation community) — 가장 무거우므로 마지막
- [x] **G-Memory organizer** (클린룸 재구현; MAS 예제는 추후) (MAS 훅; AutoGen 예제 1개)
- [x] rerank 어댑터 완성 (LLMReranker, CrossEncoder — capability-gated)
- [x] **MCP 서버 배포**: stdio + HTTP, 6 tools, agmem-mcp entry point (Claude Code dogfooding은 사용자 리뷰 시)

**완료 기준**: 7개 organizer 전부 동일 API로 구동 + MCP로 Claude Code에서 실사용.

## Phase 4 — 소형 모델 학습 + 심화 재현 (3–4주, 선택 확장)

- [~] **SFT 데이터 생성**: 파이프라인 구현 완료(train/distill_data.py) — 실행은 API teacher 키 확보 후: 대형 모델(API)로 분절/추출/증류 태스크의 입출력 쌍 생성
      (LoCoMo/LongMemEval 히스토리 + 자체 대화 로그 소스)
- [~] **Qwen3-0.6B QLoRA**: 트레이너 구현 완료(train/sft_lora.py, train extra) — 실행 대기 (RTX 2060 6GB: 4bit base + LoRA r=16, batch 1 + grad accum — 가능성 확인됨)
      태스크별 어댑터: ① boundary 탐지 ② note/entity 추출(JSON) ③ 전략 증류
- [ ] 평가: 구조화 출력 준수율 / 추출 품질(vs API 모델) / 최종 벤치 점수 개선폭
- [ ] **LongMemEval 2차 재현**: 학습된 0.5B extract 모델 vs 프롬프트만 0.6B vs API 3-way 비교
      → "0.5B로 어디까지 가능한가" 표 — 이 스터디의 고유 기여물
- [ ] (여력 시) MaTTS 재현: parallel k=3 self-contrast, 소형 모델에서의 비대칭성 관찰

## Phase 5 — 종합 (1–2주)

- [ ] 방법론 교차 조합 실험 (Nemori+ReasoningBank 스택 등)
- [ ] 최종 리포트: 8-시스템 비교표(자체 측정) + 비용-정확도 파레토 곡선
- [ ] 블로그/스터디 발표 자료

## 리스크와 대응

| 리스크 | 근거 | 대응 |
|---|---|---|
| 0.5B 구조화 출력 실패 | A-Mem/Graphiti 실증 | Phase 0의 4중 방어 우선 구축; extract만 4B-AWQ 강등 경로 |
| 원논문 수치 미도달 | G-Memory -10~18%p, A-Mem* 사례 | 목표를 "방향성+편차 보고"로 명시 (Phase 2) |
| WSL RAM 7.8GB 병목 | 실측 | `.wslconfig` 상향; 서버형 store는 full 프로파일로 격리; mmap 벡터 |
| judge 비용 누적 | LongMemEval 그리드 폭발 | ingest 캐시 + LoCoMo(F1) 우선 + judge 호출 상한 설정 |
| G-Memory LICENSE 부재 | 조사 확인 | 코드 복사 금지, 논문 기반 클린룸 재구현 + 출처 명기 |
| 스코프 폭발 | 8 방법론 × 벤치 × 학습 | Phase별 완료 기준 엄수; Zep-graph/G-Memory는 후순위 배치 |

## 즉시 다음 액션 (2026-07-26 갱신)

> 이전 판(Phase 0 부트스트랩 3단계)은 오래 전 완료돼 삭제. 아래가 현재 재개 지점이다.
> 브랜치: **`main` 단일**. 테스트 263 passed / 1 skipped.
>
> **2026-07-26 사용자 지시 — 실험 전면 보류**: 로컬(0.5B)이든 API든 측정을 하지 않는다. 현재
> 작업 모드는 ①새 논문 구현 ②API 배선 refactor 검토이고, 각 항목마다 **논문 원문 → official
> 코드 → 우리 구현** 3자 대조를 반복한다(A-Mem에서 5차까지 돌린 그 패턴). 배선이 정말 이상
>없다고 확정된 최후에 측정을 재개한다. 따라서 아래 1·2번은 **배선 완료 상태로 대기**시켜 둔다.

### 미실측 — 배선은 끝났고 유료 실행만 남음 (지시에 따라 대기)
1. **Nemori v4 Table 7 실측** (`docs/13-amem-study.md` §5.1). config 2종이 대기 중:
   ```bash
   uv run python scripts/exp_locomo_conv0.py --configs nemori_amem_k nemori_amem_k_batched
   ```
   판별 기준: **저장공간 45~64% 감소 밴드**에 들어가는 granularity는 하나뿐이다. 노트 수와
   write 토큰을 A-Mem 단독(turn당 1노트)과 대조할 것. temporal 카테고리는 `K`에 timestamp가
   없다는 전제를 반드시 붙여 해석.
2. **라이프사이클 config E2E 미실측** (Phase 2 잔여): `nemori_v4` / `nemori_upstream` /
   `nemori_mix` / `nemori_memoryos` / `nemori_amem`. 유닛 테스트까지만 검증된 상태.

### 구현 후보 (근거: `docs/research/write-path-critics.md` §5)
3. [x] **A-MAC admission gate** (1순위) — **2026-07-26 구현 완료**, 감사 문서
   `docs/research/amac-admission-gate.md`. `src/agmem/policies/admission.py` +
   적용 wrapper `organizers/gated.py::AdmissionGated`, 테스트 62건. 게이트를 붙이지 않으면
   A-Mem은 전부 저장(= baseline)이므로 기존 config 동작 무변경.
   **적용 범위 검증됨**(taxonomy §2.5): message 기반(`amem`/`zep_graph`/`memoryos`)에 유효,
   `passthrough`는 무의미, `nemori`는 분절이 바뀌므로 ablation으로 보고할 것, task 기반
   (`ace`/`gmemory`/`reasoning_bank`)은 `on_message`가 없어 **적용 불가**.
   **측정 재개 시 재튜닝이 선행 조건**: 논문 weight `[0.1,0.1,0.1,0.1,0.6]`/θ=0.55는 공식 코드의
   결함 2건(N≡1.0·R≡0.0 상수화, Type Prior 부분문자열 매칭) 위에서 맞춰진 값이라 디버그된
   feature에 전이되지 않는다. 자세한 건 감사 문서 §7.
   성공 기준은 여전히 논문 F1(admission 결정 F1, oracle 라벨, N=225)이 **아니라** "answer 품질
   유지하며 노트 수·write 토큰 감소"이며, `AdmissionStats`가 그 관측 장비다.
3b. [x] **MemMachine** (2026-07-27 1순위로 재정렬 → **당일 구현 완료**) — 조사·2차 대조
   `docs/research/memmachine.md`. `src/agmem/organizers/memmachine/` +
   read step `MemMachineContextualize`(`retrieval/steps.py`) + 새 메모리 타입 `derivatives`,
   테스트 17건. 후보 1순위 근거는 **official 코드 유무를 1차 필터로 세운 것**이고
   (SAGE·Memory Worth·GRAVITY는 코드 미공개라 3자 대조 불가), 자리는 **추출 축의 반대 극단**:
   write 경로 LLM 콜이 **메시지당 0회**라 `passthrough`(하한)와 A-Mem(turn당 2콜) 사이의
   첫 실물 중간점이다.
   **계보 확정 = 배포 코드**(논문의 3-tier 중 `profile`은 코드에 없다). 배포 코드 안에서도
   백엔드가 둘이라 `MEMMACHINE_PRESETS`로 갈랐다 — `declarative`(공개 LoCoMo 수치의 경로,
   기본)와 `event`. read operating point는 프리셋이 아니라 레시피(`AgmemConfig.memmachine_*`)로,
   기본값은 **라이브러리 기본 0/20**이고 legacy 하네스의 3/30은 `memmachine` config가 명시한다
   (MemoryOS `page_recall_cap` 사고와 같은 처방).
   **미측정, 그리고 지금 상태로는 논문 0.9169와 비교 금지**: 그 수치는
   text-embedding-3-small + Cohere `rerank-v3-5`가 **조립된 에피소드 컨텍스트**를 채점한
   결과이고, 텍스트를 채점할 수 없는 reranker(프로파일 `lite`의 Noop)에서는 컨텍스트가
   fusion 순서를 그대로 유지한다.
   부수 발견: 동봉 하네스는 **두 벌**이고 논문 수치를 낸 `evaluation/episodic_memory/` 쪽은
   `18f1211`에서 실행 불가(5-튜플을 3개로 언팩) — "네 번째 독립 참조"로 쓰려면 이 사실을
   전제해야 한다.
4. **GRAVITY** (2순위) — **2026-07-26 재분류: read-path 기법이 아니다.** anchor 3종은 offline
   build phase 산출물이고 standalone 저장되므로 **자체 organizer**(새 메모리 타입 3종 + offline
   단계는 `consolidate()`) **＋** read step 양쪽이 필요하다. ReadStep 레지스트리만으로는 절반도
   안 된다. 코드 미공개라 anchor 3종 프롬프트는 자체 설계 필요.
5. **RecMem** (3순위) — **2026-07-26 재분류: `consolidate()` 게이트가 아니다.** 공식 코드에 자체
   embedding/vector store/LLM + subconscious·episodic·semantic 3-tier가 있고 buffer 자체가 검색
   소스다 ⇒ **자체 organizer**. recurrence 게이트는 per-message write 경로 안에 있다.
6. **SAGE** (신규, 2605.30711) — A-MAC과 **같은 seam·같은 호스트**의 control policy
   ("drop-in binary gate for A-Mem", LLM 콜 16–18% 절감). vMF 밀도 novelty + adaptive threshold로
   ADD/NOOP/LLM-merge 라우팅. `policies/admission.py`의 두 번째 멤버가 될 자리이고, 이때 policy
   공통 contract를 뽑아내면 된다.
7. **Memory Worth** (신규, 2604.12007) — discard + retrieval 억제를 지배하는 policy. 메모리
   유닛당 스칼라 카운터 2개만 필요하고 "retrieval을 이미 로깅하는 아키텍처"를 전제하는데
   `on_retrieval` 훅이 정확히 그것이다. **admission과 달리 task 기반 organizer에도 적용된다**
   (쓰기 방식과 무관하게 검색된 유닛을 지배하므로).
8. **Mem-α** (신규, 2509.25911) — policy 범주의 **learned** 변종. RL이 `memory_insert`/`update`/
   `delete` 호출 정책을 학습하고 논문이 "메모리 아키텍처는 RL 프레임워크와 분리돼 교체 가능"하다고
   명시 — 우리 mechanism/policy 분리가 문헌의 축임을 확인해주는 사례. 학습 인프라가 필요해 Phase 4
   (0.5B 학습)와 묶어야 한다.

> 분류 근거와 조사 전문: `docs/research/memory-component-taxonomy.md`. 새 논문을 어디에 넣을지는
> docs/04 §1.1의 판정 기준(메모리 타입 미선언 + MemoryOp 미발행 → policy)으로 결정한다.

### 별도 판단이 필요한 미수정 버그 1건 (코드에 문서화됨)

> **9·10은 2026-07-26 해소.** 둘 다 "고치면 수치가 바뀐다"를 이유로 미뤄뒀는데, **보호하려던 비교가
> 실재하지 않았다.** 미루기 전에 확인했어야 할 것을 이번에 확인한 결과다:
>
> - #9(`ChainedConsumer`가 `flush_buffer`를 wrapped에 미전달 → 체이닝된 MemoryOS의 부분 STM tail
>   영구 미방출)의 근거는 "`nemori_memoryos` 수치 변동"이었으나, `results/`에 **chained variant의
>   저장 결과가 하나도 없다**(conv0는 amem/memoryos/nemori/passthrough 4종뿐). 전달하도록 고쳤고
>   순서가 load-bearing이다 — 어댑터의 `_pending`을 먼저 흘려보낸 뒤 wrapped를 비워야 방금 넘긴
>   유닛을 놓치지 않는다.
> - #10(`warm_start`가 ingest ADD op를 남기지 않음)의 근거는 "과거 run과 op 카운트 비교가 깨진다"
>   였으나, **`warm_start`는 scripts/·bench/ 어디에서도 호출되지 않는다**(src와 테스트에만 존재).
>   비교 대상 run 자체가 없다. 이제 `{"role", "warm_start": True}` payload로 기록하므로 로그가
>   스토어의 완전한 기록이 되고(재생으로 복원 가능), 백필과 실트래픽은 여전히 구분된다.
>
> 회귀 테스트 2건이 `tests/test_lifecycle.py`에 있고, **수정 전 상태에서 실제로 실패함을 확인**했다.

11. **doc store와 vector store의 키 단위 불일치** (2026-07-26 발견). `items` 테이블 PK는
   `(id, memory_type)`인데 **모든 벡터 백엔드는 `item_id` 단독 upsert**(numpy/sqlite-vec의
   `ON CONFLICT(item_id)`, chroma/qdrant/lance upsert-by-id)이고 `memory_type`은 필터 메타데이터로만
   저장한다. 같은 id를 두 타입으로 쓰면 벡터 인덱스에서 나중 것이 앞의 것을 덮어써 한쪽이 조용히
   검색 불가가 된다. 실 run의 id는 uuid4라 충돌 확률이 사실상 0이므로 **측정 리스크는 없다**;
   고치려면 5개 백엔드 스키마를 모두 건드리고 재색인이 필요하므로 별도 결정으로 남긴다.
   **이 건은 9·10과 달리 근거를 재확인해도 유지된다** (2026-07-26): 코드베이스에서 유일한
   비-uuid4 `target_id`는 consolidate 커서(`consolidate:{name}`, type `state`)인데, 그 payload에
   `content`/`embedding_text`가 없어 `_apply_one`이 벡터를 만들지 않으므로 애초에 벡터 인덱스에
   도달하지 않는다. 확률 논증이 조용히 썩지 않도록
   `test_consolidate_cursor_is_the_only_deterministic_id_and_never_gets_a_vector`가 전제를 고정한다.

### 2026-07-26 LongMemEval 배선 (측정 없음)

공식 저장소를 클론해 `evaluate_qa.py` / `print_qa_metrics.py` / `run_generation.py`를 통독하고
포팅했다. **논문만 읽었으면 놓쳤을 것 4건**이 나왔고, 그게 이 작업의 실제 산출물이다:

1. **정확도 정의가 둘** — task-averaged(타입 평균의 평균) vs overall(질문 평균). 타입별 문항
   수가 불균등해 값이 다르다. `aggregate()`가 **항상 둘 다** 반환한다.
2. **abstention은 교차 절단면** — 자기 question_type에도 계수되고 abstention 버킷에도 들어간다.
3. **judge 모델이 assert로 고정** — `check_judge_model`이 500콜 지출 전에 실패시킨다.
4. **`_abs` 판정이 부분문자열** — 조사 1차본의 "`_abs`로 끝나면"은 오류였고
   `docs/research/ace-longmemeval.md`에서 교정했다.

부수: 미지의 question_type에 upstream이 `raise NotImplementedError`하는 것(기본 템플릿 fallback
없음)도 그대로 옮겼다 — knowledge-update 분기는 다른 분기의 "부분집합이면 no" 규칙을 빼고 있어서
오분기하면 KU가 테스트하는 행동이 감점된다.

명시적 이탈 2건: **D1** 컨텍스트 정렬(upstream은 검색 청크를 시간순 정렬, 우리 번들은 융합 점수
순 — organizer 산출물엔 비교 가능한 날짜가 없는 경우가 많다. full-context 경로는 시간순으로
upstream과 일치), **D2** ingest 단위(upstream은 ingest가 없고 haystack을 검색한다; 우리는 turn당
`add_message`로 LoCoMo와 같은 write 경로를 태운다). `con-separate` reading method는 세션별 노트
추출 LLM 패스가 필요해 **포팅하지 않고 raise**한다 — 조용히 `con`으로 강등되면 라벨과 수치가
어긋난다.

테스트 23건이 이 계약을 고정한다(`tests/test_longmemeval.py`). `has_answer` 누출 테스트는
**실제로 누출을 주입해 실패하는지 확인**했다 — 첫 판은 `list_items("episodic")`이 항상 `[]`를
반환해(raw episode는 별도 테이블) vacuous였다.

#### 2026-07-26 감사 (공식 소스 재대조) — 5건 수정, 테스트 31건

포팅 직후 공식 저장소를 다시 받아 대조했다. 프롬프트는 **judge 5분기 + abstention = 8케이스,
answer `con`/`direct` 2종 모두 바이트 동일**이었고(공식 `get_anscheck_prompt`를 떼어내 출력 비교),
`run_generation.sh`의 `con` alias가 `--con`이 아니라 `--cot true`라는 함정도 통과해 있었다.
반면 다음 5건이 나왔다. 앞의 4건은 **수정 전 코드에서 회귀 테스트가 실제로 실패함을 확인**했다.

1. **`render_sessions`가 `has_answer` 골드 라벨을 유출**했다 — upstream은 포맷 직전에 전 turn에서
   `pop`한다(run_generation.py:177-191). 재현 대상이 정확히 full-context 베이스라인(60.6%)인데
   정답이 든 turn을 표시해 주는 셈이라 **수치가 위로 왜곡**된다. ingest 경로엔 누출 테스트가
   있었지만 이 경로엔 없어서 조용히 통과했다. upstream과 달리 **복사본에 strip**한다 — 인스턴스의
   turn-level recall 골드를 파괴하면 안 되기 때문(그 불변식도 테스트로 고정).
2. **`check_judge_model`에 호출부가 없었다.** "500콜 지출 전에 실패시킨다"는 주장에 실제 경로가
   없었다. `judge_answer`가 `configured_judge_model`로 role 설정을 읽어 첫 콜 전에 강제한다.
   모델명을 읽을 수 없는 클라이언트는 "검사 불가"이지 "통과"가 아니며, `enforce_pin=False`로
   의도적 off-pin judge(disagreement 연구)를 남긴다.
3. **full-context 경로에 길이 상한이 없었다** — `budget_tokens`는 검색 번들에만 걸린다.
   upstream은 `model_max_length - gen_length - 1000`으로 자른다(:343, :266-279). 토크나이저가
   없으므로 상한은 opt-in(`max_history_tokens`)이고, 그 산술은 `upstream_max_history_tokens`로
   노출했다. `_m`은 이게 없으면 ~1.5M 토큰이 그대로 API로 간다.
4. **`max_sessions=0`이 전체 haystack을 돌렸다** (`sessions[:max_sessions or len(...)]`).
   가장 작은 ablation 지점이 가장 큰 run이 되는 종류의 버그라 `_capped`로 `None`만 무제한으로.
5. **`task_averaged`가 이중 반올림**이었다(타입별 pct를 반올림한 뒤 평균). upstream은 raw 평균을
   한 번만 반올림한다(:31). 현실적 분포에서 67.52 vs 67.51 — 0.01pp지만 "어느 accuracy인지
   모르는 수치는 비교 불가"를 논지로 삼은 모듈에서 스스로 만든 drift다.

인용 오타 1건도 교정: `print_qa_metrics.py:15` → 실제 `:17`.

### 2026-07-26 재점검에서 처리한 것 (수치 영향 0 — 측정 run은 전부 단일 organizer)

- **공유 memory type 오염 차단.** `semantic`(Nemori+MemoryOS)·`strategies`(RB+G-Memory)는 타입만으로
  소유자를 알 수 없어 조합 설정에서 서로를 침범했다. 파사드가 ADD 적용 시 op의 `actor`를 아이템에
  기록하고, Nemori 통합기의 후보 검색(`own_items`)과 피드백 라우팅이 그것으로 판정한다.
  `actor` 없는 과거 아이템은 자기 것으로 취급 → 기존 스토어 해석 불변. 조사: taxonomy §2.6.
- **`report_feedback` → `Organizer.on_feedback` 팬아웃.** 호출자 없는 `GMemoryOrganizer.backward()`가
  round-5 W-4(served-insight 게이트)를 품은 채 죽어 있었고, `_served`는 clear되지 않아 프로세스
  수명 내내 누적됐으며, 실경로인 파사드는 그 게이트 없이 RB 아이템에도 reward를 적용했다.
  **대가**: 소유 organizer가 비활성이면 피드백이 no-op(0 반환)이 된다.
- **`MEMORY_TYPES` 누락 보정** — `experiences`(RB가 계속 발행 중이던 것)와 `state`를 추가.
  `core/ops.py`가 "target_type은 MEMORY_TYPES 중 하나"라고 선언하면서 검증이 없어 드리프트가
  잡히지 않았다. `test_stores.py`가 `produces` 대비 전수 검사한다.
- **TOML이 read-path 노브를 읽게 함** (`[retrieval]`) — "config로 ablatable"이 Python API에서만
  참이었다. 재현 런북은 TOML 경로를 쓴다.
- **문서 교정**: docs/04 §3이 제거된 `input="episodes"` 생성자 모드를 여전히 서술하고 있었다
  (docs/10·13은 이미 갱신됨). 훅 매트릭스를 실제 구현 기준으로 재작성하고
  `on_retrieval`/`on_feedback`/`flush_buffer`를 추가, `search()`가 되먹임으로 쓰기도 한다는 사실을
  §2 Read에 명시. docs/05에 합성 API(`AdmissionGated`/`ChainedConsumer`)와
  `default_memory_types` 추가.

**남긴 판단 3건** (전부 "기록하고 지금은 건드리지 않는다" — 이유가 각각 다름):

① **타입 분할 보류.** `semantic_profile` 신설이 `actor` 필터보다 근본적이지만 저장 아티팩트의
타입 키가 바뀐다. 어휘가 `type`/`kind`/`actor` 셋으로 늘어난 상태이고, **Phase 5 비교표를 그리기
전에 한 번에 정리**하는 게 맞다. `actor` 가드가 실피해는 이미 막고 있으므로 급하지 않다.

② **`search()`의 write-큐 우회.** `on_retrieval` op를 호출자 스레드에서 적용하므로 organizer의
in-memory 상태(MemoryOS `_heat`)에 워커 스레드와의 경합이 있다. **락으로 고치면 안 된다** —
`_apply_from_all`을 잠그면 워커가 LLM 호출 중 락을 쥐고 있는 동안 `search()`가 블록되어, read가
write를 기다리지 않는다는 async write의 설계 목적 자체가 깨진다. 자료구조는 손상되지 않고 heat
카운터 갱신 1건 손실 수준이라 문서화만 했다(docs/04 §2).

③ **`profile` 용어 3중 충돌** (2026-07-26 재점검에서 제기, 미해소). `AgmemConfig.profile`
(lite/standard/full) / MemoryOS `kind="profile"`(사용자 프로필 fact) / `stats()["profile"]`·
`bench/harness.py`의 재현 스탬프가 전부 같은 단어다. §2.4.1이 "consolidation"에, §2.6이 memory
type에 한 것과 **같은 처방이 필요한 세 번째 용어**인데 아직 축을 가르지 않았다. 어느 쪽을 고쳐도
비용이 있다 — `AgmemConfig.profile`은 TOML `[profile]`과 저장된 결과 스탬프에 박혀 있고,
`kind="profile"`은 아티팩트와 `bench/locomo.py`의 read 경로에 박혀 있다. ①과 같은 타이밍
(Phase 5 어휘 정리)에 함께 결정한다.

### 문서 교정 잔여
12. `docs/09-results-summary.md`의 conv0 4-way 표는 0.6B 시절 수치이고 read-path P0 수정 전에
   측정된 것 — 재측정 보류 중임이 표 위에 명기돼 있다.

### 조사 후 종결된 리스크 (재개하지 말 것)
13. **vendored Porter가 nltk와 다르다 → 영향 실측 0으로 종결** (2026-07-26, A-MAC 감사 부산물).
    `snap-research/locomo`의 `normalize_answer`와 `rouge_score` 둘 다 nltk 기본 모드
    (`NLTK_EXTENSIONS`)를 쓰는데 `agmem/_porter.py`는 Porter(1980) 원문 조건이라, step 1c
    `Y→I`에서 `cry`/`cried`가 병합되지 않는 등 실제 차이가 있다. `PorterStemmer(mode=...)`로
    파라미터화했고 기본값은 `"original"`(기존 동작) 유지 — `bench/locomo.py`가 명시적으로 지정.
    **저장된 `results/repro/*.records.jsonl` 8파일 11,914 질문을 두 모드로 오프라인 재채점한
    결과 F1이 바뀌는 질문이 0건**(감사 문서 §4 표). 발표 수치 무영향이므로 기본값 전환도
    재측정도 불필요. 잔여 gap(nltk 불규칙 pool·step5a 2글자 cvc·step2, 692어휘 중 10단어)은
    같은 이유로 낮은 우선순위의 별건.

> **2026-07-26 처리 완료**: (구 8번) "A-Mem read = cosine+BM25 hybrid" 오류를 `docs/13`·`08`·
> `03`·`14`에서 교정. `docs/14`는 같은 문서 안에서 자기모순(45행은 "BM25 없음", 349행은
> "robust=BM25 hybrid")이었다. 부수 교정: 이 오해가 낳은 **잘못된 캐비앗**("우리 amem은 read 채널이
> upstream과 다르다")도 뒤집었다 — 양쪽 다 순수 dense이므로 실제 차이는 keyword-query 재작성과
> 링크 캡이 per-hit이 아니라 global 5라는 두 가지다. 3b39c7d로 사라진 심볼
> (`_expand_links`/`_attach_sources` + `pipeline.py:181` 행 참조)도 함께 갱신.
