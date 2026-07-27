# A-Mem write-path 비판·재현 논문군 조사 (2026-07-26)

> 목적: A-Mem 리뷰의 "write-path 비용이 절감률 계산에서 빠졌다"는 논지를 뒷받침할 1차 소스를
> 확보하고, 우리가 구현할 가치가 있는 기법을 선별.
> 방법: 각 논문 abstract/HTML 전문 직접 fetch. 수치는 전부 1차 소스 대조. 조사일 2026-07-26.
> 미커버(예산): HyMem(2602.13933), H-MEM, MemOS, MIRIX, MemoryBank, EM-LLM.
> 조사 중 발견한 후속 후보: SAGE(2605.30711, memory evolution novelty gate),
> "When to Forget: A Memory Governance Primitive"(2604.12007), LiCoMemory, Memobase.

---

## 1. 한눈에 보기

| 논문 | arXiv | 채택 | 경로 | 공식 코드 | A-Mem에 대한 주장 |
|---|---|---|---|---|---|
| **A-MAC** | 2603.04549 | ICLR'26 (MemAgents) | **write (admission)** | [GuilinDev/Adaptive_Memory_…](https://github.com/GuilinDev/Adaptive_Memory_Admission_Control_LLM_Agents) | A-Mem은 recall 1.000 / precision 0.371 = 전부 저장 |
| **RecMem** | 2605.16045 | ACL'26 Findings | **write (consolidation)** | [CaiusDai/RecMem](https://github.com/CaiusDai/RecMem) | construction 토큰 1,459.9K → 193.2K (-86.8%) |
| **GRAVITY** | 2605.01688 | (arXiv) | **read (generation-time)** | 미공개 | A-Mem build cost가 자기 10배 |
| **MemMachine** | 2604.04853 | (arXiv) | write 최소화 | [MemMachine/MemMachine](https://github.com/MemMachine/MemMachine), Apache-2.0 | LLM 추출 자체를 줄여야 함 (ground-truth 보존) |
| **LightMem** | 2510.18866 | (arXiv) | write (sleep-time 분리) | — | A-MEM 대비 +2.7~9.65%p에 토큰 32~117배 절감 |

LightMem은 `docs/research/write-path-lifecycle-survey.md` §3에서 이미 조사됨 — 여기서는 GRAVITY의
build-cost 표에 함께 등장하는 부분만 교차 확인했다.

---

## 2. 가장 중요한 발견: 비용 비판의 출처는 사실상 2개 측정뿐

"A-Mem write-path가 비싸다"는 주장이 여러 논문에 등장하지만 **독립 측정은 두 건**이다.

| 출처 | A-Mem input | output | total | LLM calls | 시간 | 독립 측정? |
|---|---|---|---|---|---|---|
| Nemori v4 Table 3 (gpt-4o-mini) | 912.6K | 236.8K | 1,149.4K | 1,175.5 | — | ✅ |
| GRAVITY Table 3 | 912.6K | 236.8K | 1,149.4K | — | 6,060.7s | ❌ **Nemori 수치와 동일** |
| RecMem (gpt-4.1-mini) | — | — | 1,459.9K | — | — | ✅ |
| **우리 실측** (gpt-4o-mini, 10 conv) | **728.4K** | **205.0K** | **933.4K** | **1,175.3** | — | ✅ |

**GRAVITY의 A-Mem 행은 Nemori Table 3과 토큰 단위까지 동일하다** — 재측정이 아니라 재인용이다.
따라서 "A-Mem이 GRAVITY의 10배"는 독립 근거가 아니고, 실질 독립 측정은 Nemori와 RecMem 둘이다.

**우리 실측이 세 번째 독립 측정이 되며, call 수가 Nemori와 3자리까지 일치한다** (1,175.3 vs
1,175.5, per-conversation). 원자료: `results/repro/gpt-4o-mini_all_ingest_seed1.json`
(extract 5,882콜 / 871,726 in, distill 5,871콜 / 6,411,995 in, 10 conv, $2.32).

- **call 수 일치**는 우리 포트의 write-path 구조(turn당 Ps1 1회 + Ps2+Ps3 1회, 첫 노트는 이웃 없어
  스킵)가 upstream과 동일하다는 강한 증거다.
- **토큰은 우리가 18.8% 낮다** (933.4K vs 1,149.4K). 우리 프롬프트가 논문 원문을 압축한 판본이라
  방향과 크기가 정합적이다. 즉 우리 수치를 "A-Mem 비용"으로 인용할 때는 이 하한 성격을 밝혀야 한다.

⇒ 리뷰에서 쓸 수 있는 문장: A-Mem의 write 비용은 대화당 ~10⁶ 토큰 / ~1,175 LLM 호출 규모이고,
이는 **서로 독립인 세 측정(Nemori·RecMem·우리)에서 같은 자리수로 재현된다.** 반면 논문 §4.3의
85~93% 절감률은 read 컨텍스트만 센 수치다.

---

## 3. 두 번째 발견: A-Mem의 LoCoMo 점수가 논문마다 제각각

| 출처 | 백본 | 지표 | A-Mem 점수 |
|---|---|---|---|
| Nemori v4 Table 2 | gpt-4o-mini | LLM-judge | 52.5 |
| Nemori v4 Table 2 | gpt-4.1-mini | LLM-judge | 61.4 |
| RecMem | gpt-4.1-mini | accuracy | 68.83 |
| GRAVITY Table 1 | (미확인) | accuracy | 65.3 |
| Mem0 Table 1 | — | LLM-judge (자체 재실행, temp 0) | 논문 대비 낮게 보고 |
| **우리 재현** | gpt-4o-mini | F1 / J | **34.80 / 50.67** |

같은 시스템·같은 벤치마크인데 52.5 ~ 68.83까지 벌어진다. 우리 J 50.67은 Nemori의 gpt-4o-mini
52.5에 가장 가깝다(같은 백본·같은 judge 계열). **가장 방어 가능한 대조는 동일 백본끼리다** —
RecMem/GRAVITY 수치는 gpt-4.1-mini거나 백본 미확인이므로 우리 수치와 직접 비교 금지.
[[locomo-published-numbers-verified]]와 정합.

---

## 4. 논문별 메커니즘 (구현 판단용)

### 4.1 A-MAC — write-path admission gate ⭐구현 후보 1
메모리 가치를 5개 factor로 분해하고 **가중합 > threshold일 때만 저장**:

| factor | 계산 | LLM 필요? |
|---|---|---|
| Utility 𝒰 | "미래에 유용할 확률" 1콜, temp 0 | **✅ 1회** |
| Confidence 𝒞 | `max_s ROUGE-L(m, s)`, s = 이전 turn (환각 전파 차단) | ❌ 규칙 |
| Novelty 𝒩 | `1 − max cos(φ(m), φ(m'))`, Sentence-BERT | ❌ 규칙 |
| Recency ℛ | `exp(−λ·τ)`, λ=0.01/h (반감기 ~69h) | ❌ 규칙 |
| Type prior 𝒯 | POS 단서 기반 패턴 매칭 (선호·정체성 = 높은 지속성) | ❌ 규칙 |

정책 학습: **5-fold CV + weight grid search**, threshold ∈ [0.3, 0.6], F1 최대화.
백본: Sentence-BERT + Qwen 2.5 (로컬).

LoCoMo 결과:

| 방법 | Precision | Recall | F1 | Latency(ms) |
|---|---|---|---|---|
| Random | 0.278 | 0.278 | 0.278 | <1 |
| MemGPT | 0.316 | 0.333 | 0.324 | 2,765 |
| MemoryBank | 0.368 | 0.583 | 0.452 | 2,843 |
| Equal Weights | 0.362 | 0.694 | 0.476 | 2,916 |
| A-mem | 0.371 | **1.000** | 0.541 | 3,831 |
| **A-MAC** | **0.417** | 0.972 | **0.583** | **2,644** |

> ⚠️ **이 표는 QA F1이 아니다.** admission 결정의 precision/recall/F1이고 **N=225 샘플**이다.
> A-Mem 논문 Table 1의 answer F1과 같은 축이 아니므로 섞어 인용하면 안 된다. 그리고 A-mem의
> recall 1.000은 성능이 아니라 **정의상 전부 저장하기 때문에 자동으로 1**이다 — "A-Mem은 아무것도
> 안 버린다"는 사실의 재표현이지 A-MAC이 A-Mem을 이겼다는 근거로는 약하다.
>
> 그래도 우리에게 값진 이유: precision 0.371이라는 수치가 **"저장된 노트의 63%는 한 번도 참조되지
> 않는다"는 정량 근거**이고, 이건 우리가 지금 못 재는 축이다.

### 4.2 RecMem — recurrence-based consolidation ⭐구현 후보 2
turn마다 LLM을 부르지 않는다. 새 unit이 들어오면 (1) top-k 유사 interaction 검색 →
(2) `θ_sim` 넘는 것만 필터 → (3) 개수가 `θ_count` 초과할 때**만** LLM 호출해 시간순 episodic 요약
생성 후 semantic 정제. "의미적으로 유사한 interaction의 **지속적 재발(sustained recurrence)**이
관측될 때만 LLM을 부른다."

| 벤치 | 백본 | RecMem | Full Context | A-Mem | Mem0 | MemoryOS |
|---|---|---|---|---|---|---|
| LoCoMo | gpt-4.1-mini | 81.10 | **84.18** | 68.83 | 62.92 | 67.60 |
| LongMemEval-S | gpt-4.1-mini | **76.80** | — | 71.60 | 71.20 | 74.4 |

construction 토큰: LoCoMo 193.2K (A-Mem 1,459.9K 대비 -86.8%), LongMemEval-S 365.49K (-71.1%).
**LoCoMo에서 Full Context가 여전히 최강**(84.18 > 81.10)이라는 점을 논문이 스스로 보고한다 —
우리 리뷰의 "메모리 시스템 이득은 강한 baseline 앞에서 줄어든다" 논지와 정합.

### 4.3 GRAVITY — read-path, architecture-agnostic ⭐구현 후보 3 (우리 구조에 딱 맞음)
**write/read 어느 것도 고치지 않고 generation-time에 프롬프트만 증강**한다. 호스트 시스템이 검색을
끝낸 뒤 3종 anchor를 붙인다:
- **Entity anchor**: 속성·타입 관계·타임라인·요약을 가진 동적 프로필
- **Event anchor**: 4W1O(Who/What/When/Where/Outcome) 튜플을 시간순 trace로 연결
- **Topic anchor**: 세션 간 요약 (거시 서사)

anchor는 **한 번 만들어 모든 호스트에서 재사용**한다. 호스트별 LoCoMo 개선(accuracy):
ZEP 54.7→61.8, Mem0 51.0→59.1, **A-Mem 65.3→70.9 (+5.6)**, LiCoMemory 55.7→66.2,
LightMem 70.1→75.8. 평균 +7.5%p. 약한 호스트가 더 크게 오르고 강한 호스트도 +3.8~5.7 (수확체감).

build cost 192.6K 토큰 / 556.8s. **코드 미공개.**

### 4.4 MemMachine — LLM 추출 자체를 줄이는 반대 방향
raw 대화 episode를 **손실 추출 없이 그대로** 저장하고 문장 단위로 인덱싱. write-path LLM은 STM
요약과 profile 추출에만 쓴다(메시지별 fact 추출 없음). read는 nucleus match를 주변 세그먼트로
확장(contextualized retrieval) 후 dedup·시간순 정렬 + 선택적 cross-encoder rerank.
**확장은 비대칭이다** — 예산의 1/3만 뒤로, 2/3를 앞으로 쓴다(`event_memory.py:450-451`
`max_backward = expand_context // 3`). 이 문서가 적었던 "±1~2 turn"은 틀렸다(11차 이후
당일 코드 확인, `memmachine.md` §1.2).

LoCoMo 0.9169 (gpt-4.1-mini, agent mode) / LongMemEval-S 93.0% (gpt-5-mini).
토큰: LoCoMo input 4.20M vs Mem0 19.21M (-78%).

> ⚠️ 우리 리뷰가 인용한 "retrieved vs total pipeline 간극 명시"는 **이 논문이 그 간극을 스스로
> 깔끔히 분리한 것은 아니다.** memory mode 4.20M와 agent mode 8.57M(라우팅 오버헤드 포함)을
> 함께 보고하지만, "80% 절감" 문구는 memory-only 경로 기준이고 retrieval 단계만 따로 떼지 않았다.
> 즉 이 논문은 **간극의 사례이자 부분적 반례**로 인용하는 게 정확하다.

구조 판정(2026-07-26 추가): **mechanism**이다 — 자체 3-tier + 자체 Retrieval Agent. 그리고 A-Mem의
**추출 축 반대 극단**이라 비교표의 끝점이 된다. 상세는 `memory-component-taxonomy.md` §2.4.

---

## 5. 구현 후보 판단

| 후보 | 우리에게 주는 것 | 배선 난이도 | 판단 |
|---|---|---|---|
| **A-MAC admission gate** | "A-Mem은 전부 저장한다"의 **대안**을 실측 가능하게 함. 규칙 4개 + LLM 1콜이라 우리 organizer 계약에 그대로 맞음. precision 축(참조되지 않는 노트 비율)을 처음으로 측정 가능 | 낮음 — `AMemOrganizer.on_message` 앞단 게이트. Novelty는 이미 embedder 보유, Recency/Type은 순수 규칙, Confidence는 ROUGE-L | ~~1순위~~ **완료 (2026-07-26)** — `policies/admission.py`, 감사 `amac-admission-gate.md` |
| **GRAVITY anchors** | ~~read-path 기법이고 ReadStep 플러그인 구조에 정확히 들어맞는다~~ → **아래 교정 참조**. A-Mem 호스트에서 +5.6%p 보고 | 중~높음 — anchor 3종 build가 offline write 패스 + read 주입. 코드 미공개라 프롬프트를 우리가 설계해야 함 | **2순위** |
| **RecMem recurrence gate** | ~~`consolidate()` 훅이 이미 그 자리~~ → **아래 교정 참조**. LightMem과 같은 "유예/선택적 LLM 호출" 계열. 코드 공개 | 중 — 트리거는 단순하지만 3-tier 전체를 배선해야 함 | 3순위 |
| **MemMachine** | ~~인용만~~ → **구현 후보 1순위로 재분류(2026-07-27)**. 근거는 세 가지다: (a) **Apache-2.0 활성 리포**라 3자 대조가 되고 G-Memory식 무라이선스 문제가 없다, (b) write 경로에 **메시지당 LLM 콜이 0회**임을 코드로 확인 — A-Mem(turn당 2콜)과 `passthrough` 사이에 **실물 중간점이 처음 생긴다**, (c) 공식 LoCoMo·LongMemEval 하네스가 동봉돼 우리 `bench/`의 네 번째 독립 참조가 된다. §4.4 인용 캐비앗은 여전히 유효 | 중 — write는 가볍지만(기계적 분절) read가 무겁다; Retrieval Agent는 policy의 첫 read-side 멤버라 별건 | **1순위** (조사: `memmachine.md`) |
| **LightMem** | 이미 조사됨. `consolidate()` = sleep-time 자리 | 중 | 보류 |

**A-MAC이 1순위인 이유**: 우리 리뷰의 핵심 지적("A-Mem은 전부 저장하고 전부 인라인 처리한다")에
대해 **가장 값싸게 대조군을 만들 수 있는** 기법이다. 5개 factor 중 4개가 LLM-free라 write 비용이
turn당 2콜 → 사실상 그대로거나 오히려 감소하고(저장 안 하면 Ps2+Ps3 스킵), 우리가 지금 전혀 못 재는
"저장된 노트의 참조율"을 측정 축으로 추가한다.

**단, A-MAC 재현 시 반드시 분리할 것**: 논문의 F1 0.583은 admission 결정 F1(N=225)이다. 우리가
측정할 것은 LoCoMo **answer** F1/J이며, 두 수치는 비교 대상이 아니다. A-MAC 배선의 성공 기준은
"answer 품질을 유지하면서 노트 수와 write 토큰을 줄이는가"로 세워야 한다.

### ⚠️ 2026-07-26 교정 — 위 표의 GRAVITY·RecMem 배치 판단은 틀렸다

이 표는 abstract 수준 조사로 seam을 추정했고, 이후 **논문 전문 + RecMem 공식 코드 통독**으로
둘 다 오분류였음이 확인됐다. 전문은 `memory-component-taxonomy.md` §2·§4.

- **GRAVITY는 read-path 기법이 아니다.** anchor 3종은 **offline build phase** 산출물이다 —
  entity는 *incremental batch update + offline consolidation* 2단계, event는 4W1O 튜플 →
  temporal trace, topic은 cross-session 식별 + 요약. 게다가 "anchor knowledge base를 **standalone
  파일로 저장**"한다. query time에 하는 일은 anchor retrieval + query expansion + prompt injection.
  ⇒ **자체 organizer**(새 메모리 타입 3종, offline 단계는 `consolidate()`) **＋** read step.
  ReadStep 레지스트리만으로는 절반도 커버되지 않는다.
- **RecMem은 게이트가 아니라 자체 시스템이다.** 공식 코드(`CaiusDai/RecMem`)에 자체 embedding/
  vector store/LLM 계층과 subconscious·episodic·semantic **3-tier**가 있고, `SubconsciousMemory`
  docstring이 buffer가 *query time 검색 소스*임을 명시한다. recurrence 게이트는 `consolidate()`가
  아니라 `add_memory`의 per-message write 경로 안에 있다(`rec_mem.py:337-411`: ①episodic merge
  ②`min_relevant_score`/`min_consolidation_cnt` 게이트 ③LLM 없이 buffer 적재). ⇒ **자체 organizer**.

⇒ 따라서 이 §5의 3개 후보 중 **A-MAC만 cross-cutting control policy**이고, 그 범주의 동료는
GRAVITY/RecMem이 아니라 **SAGE**(2605.30711, "drop-in binary gate for A-Mem", 같은 store seam)와
**Memory Worth**(2604.12007, discard + retrieval 억제)다. 이 사실이 `agmem/policies/` 패키지 신설의
근거가 됐다(docs/04 §1.1).

---

## 6. 남은 열린 질문

1. GRAVITY의 A-Mem baseline 65.3이 어느 백본인지 미확인 (Table 1 캡션 재확인 필요).
2. A-MAC의 "225 샘플"이 LoCoMo의 어느 부분집합인지, admission ground-truth를 어떻게 만들었는지
   불명 — 재현하려면 이 라벨 정의가 핵심인데 abstract/HTML에서 확인 못 함.
3. HyMem(2602.13933)의 "A-Mem inference 796.7s = 8종 중 최느림" 주장 미검증.
4. RecMem의 memory unit 정의와 우리 episodes/semantic의 매핑.

출처: [2603.04549](https://arxiv.org/abs/2603.04549) (A-MAC),
[2605.16045](https://arxiv.org/html/2605.16045v1) (RecMem),
[2605.01688](https://arxiv.org/html/2605.01688) (GRAVITY),
[2604.04853](https://arxiv.org/html/2604.04853v1) (MemMachine),
[2510.18866](https://arxiv.org/abs/2510.18866) (LightMem),
`results/repro/gpt-4o-mini_all_ingest_seed1.json` (우리 실측).
