# Nemori 스터디 가이드 — 논문·공식 코드·우리 구현 (2026-07-17)

> 대상: arXiv:2508.03341 (v1 초기 형식화 + v4 ACL 형식화), github.com/nemori-ai/nemori
> (현행 main = 이슈 #13의 async/Qdrant 리라이트, LoCoMo V5 83.05% 버전).
> 우리 구현: `src/agmem/organizers/nemori.py` + `src/agmem/retrieval/pipeline.py`
> + `src/agmem/bench/locomo.py`. 커밋 e7c5f8f(3차 감사 반영) 기준.
> 근거 감사: docs/research/fidelity-deep-audit.md(1차), fidelity-round3-paper-code-forensics.md(3차).

---

## 1. 논문이 풀려는 문제

장기 대화 에이전트의 메모리를 "무엇을 저장할 단위로 삼을 것인가"부터 다시 설계한다.
기존 접근(메시지 단위 RAG, 요약 압축)의 문제:

- 메시지 단위는 **인간이 기억하는 단위가 아니다** — 사람은 "사건(episode)"으로 기억함
- 요약은 정보 손실이 크고, 시간 표현("어제", "다음 주")이 상대적이라 나중에 무의미해짐
- 모든 것을 저장하면 중복이 쌓여 검색이 오염됨

Nemori의 대답이 **두 원리**:

1. **Two-Step Alignment (표현 정렬)** — ① 대화를 인간의 사건 단위로 **분절**(boundary
   detection)하고 ② 각 세그먼트를 **시간이 절대화된 3인칭 서사**로 재작성(episode
   generation). "yesterday" → "March 15, 2024 (yesterday)"처럼 원 표현 뒤 괄호 병기.
2. **Predict-Calibrate (자유에너지/surprise 원리)** — 새 에피소드가 오면 기존 semantic
   지식으로 그 내용을 **예측**해 보고, 예측이 빗나간 **gap만** 원자적 semantic fact로
   증류. 이미 아는 것은 다시 저장하지 않음 → 중복 억제.
   결정적 디테일: 캘리브레이션의 ground truth는 **생성된 서사 ζ가 아니라 원본 대화
   M**이다 (서사화 과정의 정보 손실이 semantic 추출에 전파되지 않도록).

메모리는 2계층: **episodic**(서사) + **semantic**(원자 문장). QA 시 둘 다 검색.

## 2. v1 vs v4 — 형식화가 바뀌었다 (중요)

| 항목 | v1 (2025-08) | v4 (ACL, 2026) |
|---|---|---|
| 경계 검출 | per-message `f_θ(m_{t+1}, M) → (b, c)`, 트리거 `(b ∧ c>σ_boundary) ∨ (\|M\|≥β_max)`, σ=0.7, β_max=25 | **폐기** → 관측창 w=20 단위 **배치 분할**(Local Message Partitioning). σ/β 표기 삭제 |
| 에피소드 병합 | 없음 | **신규 §3.2.3**: 새 에피소드마다 top-K_e=5 유사 후보 → LLM 병합 판정(같은 사건, **>1h 갭이면 금지**) |
| semantic 통합 | append-only | **신규 §3.3.3**: new/merge/conflict 3분기(K_m=5, τ=0.70) — redundancy 회피 |
| 검색 | episodic k=10 + semantic m=2k=20, r=2 원문 첨부 | 동일 (Table 6에 r=2 명문화) |
| LoCoMo 결과 | 0.744 (gpt-4o-mini) | **80.8** vs Full-Context 80.6 (gpt-4.1-mini), temporal +15.9% over A-MEM |

**어블레이션 반전 (논문 최대 약점)**: v1은 semantic 제거 시 -3.9점(episodic 제거가
더 아픔) ↔ v4는 semantic 제거 시 **-25.1%**(semantic이 더 중요). 같은 방법·같은
데이터셋인데 버전 간 핵심 결론이 뒤집힘. v4의 semantic 기여 주장은 §3.3.3 통합
모듈(중복 억제)과 결합된 수치라는 점에 유의.

**그 외 알아둘 약점**:
- Predict-Calibrate의 직접추출 대비 이득이 gpt-4o-mini +25% → gpt-4.1-mini +14.4%로
  축소 — **모델이 약할수록 PC 이득이 크고, 강할수록 얇아짐**. 뒤집으면 아주 약한
  모델(우리 0.6B)에선 예측문 자체가 환각이라 gap 추출이 왜곡될 수 있음.
- v4가 "w=5~40에서 ±1%"라고 스스로 보고 — 적응적 경계 검출이 고정 청킹 대비 거의
  이득이 없다는 자기반박. Two-Step의 1단계 실익은 사실 얇다.
- r=2 원문 첨부는 부분적 raw-RAG — "서사가 좋다"는 주장을 스스로 희석.
- temporal 우위는 저장 형식뿐 아니라 **답변 프롬프트의 날짜 계산 지시**의 기여가
  섞여 있음(분리 검증 없음). judge도 gpt-4o-mini 자기평가.
- 이슈 #8: 외부 사용자가 0.388 vs ~0.83 재현 실패 보고 — 프롬프트·설정 민감도 높음.

## 3. 공식 코드 아키텍처 (현행 main)

```
nemori/
  config.py                  # buffer 2-25, batch_threshold=20, merge 0.85/top-5,
                             # search episodes=10 semantic=10(eval은 20)
  core/memory_system.py      # 오케스트레이션: >=batch_threshold면 배치 분할,
                             # 에피소드마다 merger.check_and_merge()
  llm/prompts.py             # 모든 프롬프트 (475줄)
  llm/generators/
    segmenter.py             # BatchSegmenter: 80msg chunk, t=0.2, topic 라벨 출력
    episode.py               # 에피소드 생성 (few-shot, 괄호 병기 스타일)
    semantic.py              # enable_pc && 기존 semantic 있음 → predict-correct,
                             # 없으면 direct extraction (에피소드를 입력으로)
    merger.py                # MERGE_DECISION(>1h 금지) + MERGE_CONTENT
  evaluation/locomo/
    add.py                   # 전체 대화 push 후 flush → 배치 분할 발동
    search.py                # episodic 10 + semantic 20, r=2 Source Messages,
                             # 답변 프롬프트(8개 timestamp 지시+7단계), t=0.0
```

핵심 관찰: **현행 main에는 per-message 경계 검출 코드가 아예 없다** (배치 전용).
우리가 재현한 per-message 방식은 v1 논문의 형식화다. 저장은 PostgreSQL+Qdrant,
에피소드 임베딩은 `title + " " + content`, semantic은 문장 단독.

## 4. 우리 구현 워크스루

### 4.1 write 경로 — `NemoriOrganizer` (nemori.py)

메시지 하나가 들어오면 (`on_message`, :185):

```
buffer에 추가
├─ len < buffer_min(2)        → 대기
├─ len ≥ buffer_max(25)       → 버퍼 전체를 세그먼트로 flush   (v1 β_max)
└─ 그 외 → BOUNDARY_PROMPT LLM 콜 (:70)
    boundary=true ∧ confidence ≥ 0.7(σ_boundary)
    → buffer[:-1]을 세그먼트로 flush, 최신 메시지는 다음 에피소드의 시작으로 잔류
```

세그먼트 flush (`_flush_segment`, :231):

1. **에피소드 생성** — `EPISODE_PROMPT`(:92): title(10-20단어) + 3인칭 과거형 서사
   (결정·감정·계획 포함) + **상대시간 → "원 표현 (절대날짜)" 괄호 병기** + 발생시각
   분석(timestamp). 실패 시 기계적 폴백(title=첫 8단어, narrative=원문) — 세그먼트를
   잃지 않음.
2. **ADD episodes** — payload에 title/narrative/timestamp/`source_episode_ids`
   (원본 메시지 ID들 — read 경로의 r=2 원문 첨부가 이걸 소비).
   임베딩 텍스트는 `title\nnarrative` (upstream `title + " " + content` 등가).
3. **Predict-Calibrate** (`_predict_calibrate`, :263):
   - 에피소드 임베딩으로 semantic top-10 검색 (upstream search_top_k_semantic=10)
   - **cold start**: 검색된 semantic이 없으면 예측 생략, `DIRECT_EXTRACT_PROMPT`(:158)로
     **에피소드(title+narrative)에서** 직접 추출 (upstream direct 경로와 동일 입력)
   - 있으면: `PREDICT_PROMPT`(:113) — **title만** + 지식 문장으로 내용 예측
     → `CALIBRATE_PROMPT`(:142) — 예측 vs **원본 대화**(타임스탬프 제거한
     `role: content` 포맷, upstream original_messages와 동일) 대조, gap만 추출
   - 4-test(6개월 지속/구체성/효용/독립성) + 고가치 7카테고리 + 저가치 금지 +
     **시간·날짜 정보 금지**(`_FOUR_TESTS`, :128) — upstream 문구 축약 이식
4. **ADD semantic** — fact 문장별 개별 저장(문장 단독 임베딩), append-only.

인제스트 종료 시 `flush_buffer`(:222)가 잔여 버퍼를 마지막 세그먼트로 처리.

### 4.2 read 경로 — `RetrievalPipeline` + bench

```
질문 → 임베딩 → 타입별 검색: episodes k=10 (dense) + semantic k=20 (dense)
     → RRF(단일 랭킹이라 순서 유지) → hydrate
     → episodes 상위 r=2건에 _attach_sources():                (pipeline.py)
        source_episode_ids로 원본 메시지를 "Source Messages:"로 부착
     → MemoryBundle.render(budget 6000tok):                    (types.py)
        점수순으로 예산 내 선별 후 "Episodic Memories:" / "Semantic Memories:"
        섹션으로 그룹 렌더 (upstream search.py의 컨텍스트 형태)
     → ANSWER_PROMPT (bench/locomo.py): 타임스탬프 기반 절대날짜 계산 지시 +
        최신 우선 + 최단 span, generate role t=0.0
```

실험 config (`scripts/exp_locomo_conv0.py`):
- `nemori`: `("episodes", "semantic")`, k={episodes:10, semantic:20} — **방법론 순수**
  (upstream eval과 동일 채널)
- `nemori_mixed`: raw episodic 10건을 추가 검색하는 구 설정 — ablation용, 논문 재현 아님

### 4.3 우리 프롬프트 5종의 계보

| 우리 프롬프트 | upstream 원본 | 보존한 핵심 | 의도적 축약 |
|---|---|---|---|
| BOUNDARY (:70) | BATCH_SEGMENTATION (online 재구성) | topic/intent/temporal/structural 신호, 30분 갭, 관련도<30%, 2-15msg, "when in doubt split", strict 톤 | 배치의 topic 라벨 출력(우리는 boolean+confidence — v1 형식) |
| EPISODE (:92) | EPISODE_GENERATION | title 10-20단어, 3인칭 스토리(결정·감정·계획), 괄호 병기 + 예시, 발생시각 분석, ISO | Time Analysis 3단계 블록을 2문장으로 압축 |
| PREDICT (:113) | PREDICTION | title-only 입력, 4대 초점, "content not style", ignore 목록 | GOOD/BAD 예시 |
| CALIBRATE (:142) | EXTRACT_KNOWLEDGE_FROM_COMPARISON | 4-test(6개월), 고가치 7카테고리, 저가치 금지, 시간 금지, 디테일 포함, quality>quantity | GOOD/BAD 예시 문장들 |
| DIRECT (:158) | SEMANTIC_GENERATION | 동일 4-test·카테고리, **에피소드 입력** | 8개 예시 |

## 5. upstream과의 잔여 편차 (전부 의도적·문서화됨)

| 편차 | 이유 | 영향 |
|---|---|---|
| **per-message 경계 (v1) vs 배치 분할 (현행 리포/v4)** | 우리는 스트리밍 조직화가 설계 목표(온라인 에이전트 메모리). v1 형식화가 이에 부합 | LLM 콜 ~N배 비용. v4의 "w 민감도 ±1%" 근거상 품질 차이는 작을 것 |
| **에피소드 병합 없음** | LoCoMo는 세션 간격이 수일~수개월 → upstream의 >1h 금지 게이트가 세션 간 병합을 대부분 차단. LongMemEval 단계에서 구현 예정 | 동일 세션 내 쪼개진 사건 미통합 → 에피소드 수 증가(비용), 점수 영향 제한적 |
| **semantic 통합(v4 §3.3.3) 없음 — append-only** | 현행 리포 main도 append-only(dedup PR #19 미병합). v4 재현 시 최우선 구현 대상 | 장기 인제스트에서 중복 fact 누적 → v4의 semantic 기여 재현 불가 |
| confidence 0.7 게이트 | v1 σ_boundary. 현행 리포에 대응물 없음(배치라서) | v1 재현 관점에선 정확 |
| 저장소: MemoryOp 로그 + SQLite/FTS5 (vs PostgreSQL+Qdrant) | 프로젝트 공통 인프라 | 없음 (검색은 동일하게 dense) |
| 평가: token-F1/BLEU (vs LLM judge) | judge LLM 비용 회피 + 결정론 (docs/06) | 논문 절대치와 비교 불가, 방법 간 상대 비교 유효 |
| 임베더/LLM: e5-small / Qwen3-0.6B (vs text-embedding-3-small 또는 gemini / gpt-4.1-mini) | 로컬 재현 (docs/07) | §6 캐비앗 참조 |
| 답변 프롬프트: 공통 1종 (vs Nemori 전용 8-지시 프롬프트) | 방법론 간 공정 비교 프레임. 시간 계산 지시는 공통 프롬프트에 반영함 | 논문 대비 단순하나 전 방법 동일 조건 |

## 6. 결과 해석 캐비앗 (0.6B 재현의 한계)

1. **Predict-Calibrate가 0.6B에서 불리할 수 있음**: PC 이득은 모델이 약할수록 크다는
   게 논문 패턴이지만, 그 "약함"의 하한 밖(0.6B)에서는 예측문 자체가 환각 → gap이
   왜곡됨. `nemori` 결과가 나쁘면 "PC의 실패"가 아니라 "0.6B 예측의 실패"일 수 있음.
   분리하려면 direct-extraction-only ablation config가 필요.
2. **temporal 카테고리**: 서사의 괄호 병기와 답변 프롬프트의 날짜 계산 모두 0.6B의
   날짜 산술 능력에 의존 — 논문의 temporal 우위가 재현되지 않아도 방법론 반증이 아님.
3. **append-only semantic**: 대화가 길어질수록 중복 fact가 semantic top-20을 오염
   → v4식 semantic 기여를 하한 추정하게 됨.
4. 논문 수치(0.744/80.8)는 LLM judge 기준 — 우리 F1/BLEU와 축이 다름.

## 7. 실행

```bash
scripts/serve-llm.sh                                   # 로컬 Qwen3-0.6B (:8080)
uv run python scripts/exp_locomo_conv0.py --configs nemori           # 순수
uv run python scripts/exp_locomo_conv0.py --configs nemori_mixed     # +raw RAG
uv run pytest tests/test_organizers.py -k nemori -q                  # 단위 테스트
```

결과는 `results/locomo-conv0-nemori.json` (stamp에 k/budget/keyword 설정 기록).

## 8. 더 읽기

- 1차 심층 감사: `docs/research/fidelity-deep-audit.md` §2 (Nemori 라인 대조)
- 3차 감사(논문 재분석 + 구절 포렌식): `docs/research/fidelity-round3-paper-code-forensics.md` §2
- 노트: `docs/research/nemori-reasoningbank.md`
- upstream: github.com/nemori-ai/nemori (`llm/prompts.py`가 사실상 방법론 명세서)
