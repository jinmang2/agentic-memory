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
- [ ] **LongMemEval_S** 파이프라인 (judge pin `gpt-4o-2024-08-06`, reading `con`+`json` 고정, cleaned 버전 명시, ingest 아티팩트 캐시)
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
> 브랜치: **`main` 단일**. 테스트 254 passed / 1 skipped.
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

### 별도 판단이 필요한 미수정 버그 3건 (코드에 문서화됨)
9. `ChainedConsumer`가 `flush_buffer`를 **wrapped에** 전달하지 않아 체이닝된 MemoryOS의 부분 STM
   tail이 영구 미방출 (`experimental/chained.py` "Known gap"). 고치면 `nemori_memoryos` 수치 변동.
   (어댑터 자신의 `_pending` 배치는 정상 방출되므로 `nemori_amem_k_batched`의 마지막 배치는
   안전하다 — 2026-07-26 확인.)
10. `warm_start`가 raw episode의 ingest ADD op를 evolution log에 남기지 않음 (`add_message`는 남김,
   `memory.py:warm_start` docstring). 고치면 과거 run과 op 카운트 비교가 깨진다.
11. **doc store와 vector store의 키 단위 불일치** (2026-07-26 발견). `items` 테이블 PK는
   `(id, memory_type)`인데 **모든 벡터 백엔드는 `item_id` 단독 upsert**(numpy/sqlite-vec의
   `ON CONFLICT(item_id)`, chroma/qdrant/lance upsert-by-id)이고 `memory_type`은 필터 메타데이터로만
   저장한다. 같은 id를 두 타입으로 쓰면 벡터 인덱스에서 나중 것이 앞의 것을 덮어써 한쪽이 조용히
   검색 불가가 된다. 실 run의 id는 uuid4라 충돌 확률이 사실상 0이므로 **측정 리스크는 없다**;
   고치려면 5개 백엔드 스키마를 모두 건드리고 재색인이 필요하므로 별도 결정으로 남긴다.

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
