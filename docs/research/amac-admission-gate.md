# A-MAC 감사 + 구현 (2026-07-26)

> 대상: **A-MAC** — "Adaptive Memory Admission Control for LLM Agents Using Weighted Feature
> Scoring" (arXiv:2603.04549, ICLR'26 MemAgents 제출).
> `docs/research/write-path-critics.md` §5의 구현 후보 **1순위**.
> 방법: 논문 HTML 전문 + **공식 코드 전수 통독**(github.com/GuilinDev/Adaptive_Memory_Admission_Control_LLM_Agents,
> 단일 커밋 `40407ae`, MIT) + 주장 경험적 재현. 수치는 전부 1차 소스.
> 산출 코드: `src/agmem/policies/admission.py` + 적용 wrapper `organizers/gated.py`,
> 테스트 `tests/test_admission_gate.py` (62건).
> **측정은 하지 않았다** — 사용자 지시(배선 확정 전 실험 금지)에 따라 유닛 검증까지만.

---

## 1. 메커니즘 (논문 원문)

    S(m) = w_U·U(m) + w_C·C(m) + w_N·N(m) + w_R·R(m) + w_T·T(m),   admit iff S(m) ≥ θ
    s.t.  w_i ≥ 0,  Σ w_i = 1

| feature | 논문 정의 | LLM 필요 |
|---|---|---|
| **U** Future Utility | temperature 0 LLM 1콜, "actionable / 후속질문 지지 / 지속적 제약·선호" 판정 | ✅ 1콜 |
| **C** Factual Confidence | `C(m) = max_{s∈Support(m)} ROUGE-L(m, s)` | ✗ |
| **N** Semantic Novelty | `N(m) = 1 − max_{m'∈M} cos(φ(m), φ(m'))`, SBERT | ✗ |
| **R** Temporal Recency | `R(m) = exp(−λ·τ(m))`, **λ=0.01/h (반감기 ≈69h)** | ✗ |
| **T** Content Type Prior | "rule-based pattern matching with **part-of-speech cues**" | ✗ |

탐색: weight 조합 grid + **θ ∈ [0.3, 0.6]**, 5-fold CV로 F1 최대화.
평가: LoCoMo, "30 conversations / 약 1,500 candidate memory", 70/15/15 split, **test N=225**.
보고 결과 (README Key Results = 논문 Table 1):

| Method | Precision | Recall | F1 | Latency (ms) |
|---|---|---|---|---|
| **A-MAC (Ours)** | **0.417** | 0.972 | **0.583** | **2644** |
| A-mem | 0.371 | **1.000** | 0.541 | 3831 |
| Equal Weights | 0.362 | 0.694 | 0.476 | 2916 |
| MemoryBank | 0.368 | 0.583 | 0.452 | 2843 |
| MemGPT | 0.316 | 0.333 | 0.324 | 2765 |
| Random | 0.278 | 0.278 | 0.278 | <1 |

학습된 weight `[U,C,N,R,T] = [0.1, 0.1, 0.1, 0.1, 0.6]`, `θ = 0.55`. ablation 결론 = **Type Prior 지배**.

**F1의 정체 (인용 시 필수 캐비앗)**: `data_loader.get_memory_candidates`가 라벨을
`is_referenced = turn.dia_id ∈ (모든 QA의 evidence 합집합)`으로 만든다. 즉 **admission 결정
F1**이고 answer F1이 아니다 — A-Mem 논문 Table 1 옆에 놓으면 안 된다. A-mem의 recall 1.000도
성과가 아니라 **전부 저장하므로 정의상 1**이다. 값진 건 precision 0.371 = "저장된 노트의 62.9%는
어떤 질문도 인용하지 않는다"는 정량 근거. 또한 이 라벨은 **미래 QA에서 역산한 oracle**이라 배포
시점에 존재하지 않는다.

---

## 2. 공식 코드 감사 — 결함 4건 (전부 경험적 확인)

### 결함 1. 5개 feature 중 **N과 R은 상수**다 (가장 중대)

`optimize_weights_cv.py:extract_features_for_candidates` (실제 feature 행렬을 만드는 유일한 경로):

```python
current_time = time.time()
...
c = extractors['confidence'].score(memory, history)
n = extractors['novelty'].score(memory, [])          # ← existing_memories 가 빈 리스트 하드코딩
r = extractors['recency'].score(memory, current_time) # ← wall clock
t = extractors['type_prior'].score(memory)
```

- **N ≡ 1.0.** `NoveltyExtractor.score`는 `if not existing_memories: return 1.0`. 빈 리스트를
  넘기므로 전 후보가 1.0이고 **SBERT는 한 번도 호출되지 않는다.** README가 광고하는
  `all-MiniLM-L6-v2`는 보고 수치에 기여분 0.
- **R ≡ 0.0.** LoCoMo 타임스탬프는 `data_loader`가 `2023-05-01` 기준으로 생성하고 비교 기준은
  `time.time()`이다. 실측:

      age_hours = 28,378  →  R = exp(-0.01 × 28,378) = 5.7e-124
      (half-life 69.3h, TTL(0.1) 230.3h)

  float 언더플로 수준이라 전 후보가 동일하게 0.

⇒ 신호를 가진 feature는 **5개가 아니라 3개(U, C, T)**다. "Type Prior가 지배적"이라는 ablation
결론은 나머지 둘이 상수인 상태에서 얻어진 것이다.

### 결함 2. Type Prior가 키워드를 **부분문자열**로 매칭한다

`features/type_prior.py:classify` → `if keyword in content`. `fact` 키워드 집합이
`{'is','are','was','were',...}`이므로:

```
'is' in 'this'  → True     'is' in 'sister' → True     'is' in 'island' → True
```

실측 (공식 코드 그대로 실행):

| turn | classify | prior |
|---|---|---|
| `Hey! How was your weekend?` | fact | 0.70 |
| `It was good, thanks for asking.` | fact | 0.70 |
| `I went to the beach with my sister on Saturday.` | **fact** | 0.70 |
| `My name is Caroline and I live in Boston.` | identity | 0.90 |
| `I prefer tea over coffee.` | preference | 0.90 |
| `Sounds great!` | unknown | 0.50 |
| `Ugh.` / `Wow.` | unknown | 0.50 |

3행이 `fact`가 되는 이유는 **"s-is-ter"의 `is`**다. 즉 T는 의미 분류가 아니라
"실질 내용어를 포함하는가"의 대리 지표가 된다.

### 결함 3. "5-fold CV"가 **train fold로 학습하지 않는다**

```python
for train_idx, val_idx in kf.split(features):
    X_train, X_val = features[train_idx], features[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    result = evaluate_weights_threshold(X_val, y_val, weights, threshold)  # X_train/y_train 미사용
```

`X_train`/`y_train`은 죽은 지역변수다. weight·θ 선택이 **평가 대상과 같은 데이터에서** 이뤄진다.
후보 집합도 논문이 말한 grid가 아니라 `1(equal) + 100(random) + 5(feature-emphasized)` = 106개
랜덤 서치다. θ 목록 `[0.3 … 0.6]`은 논문의 구간과 일치한다(유일하게 일치하는 항목).

### 결함 4. 논문의 "part-of-speech cues"가 릴리스에 **없다**

기본 `TypePriorExtractor`는 키워드 집합뿐이고, 비기본 서브클래스
`AdvancedTypePriorExtractor`가 정규식 3종을 더한 게 전부다. 코드에 `TODO: Could integrate with
spaCy NER`가 남아 있다. POS 태거는 의존성에도 없다.

### 파생 결과: 보고된 operating point의 실제 의미

`[0.1,0.1,0.1,0.1,0.6]`, θ=0.55에 결함 1(N=1.0, R=0.0)을 대입하면
`S = 0.1·U + 0.1·C + 0.1 + 0.6·T`. LLM 없는 설정(U=0.5)에서:

| T | 판정 |
|---|---|
| 0.9 (preference/identity) | **모든 C에서 ADMIT** |
| 0.7 (fact) | **모든 C에서 ADMIT** |
| 0.5 (unknown) | C ≥ 1.00 필요 → 사실상 불가 |
| 0.3 / 0.2 | C ≥ 2.20 / 2.80 필요 → 불가능 |

⇒ 발표된 A-MAC은 **"bare interjection이 아니면 admit"** 규칙으로 환원된다. 결함 2 때문에 거의
모든 실질 turn이 `fact` 이상을 받으므로 **recall 0.972**가 나오고, precision 0.417은 라벨
기저율(A-mem의 0.371)을 조금 넘는 수준이다. 즉 **보고된 recall은 부분문자열 버그의 산물**이다.

---

## 3. 우리 구현 (`policies/admission.py` + `organizers/gated.py`)

    AdmissionGated(AMemOrganizer(), AdmissionGate())

wrapper가 `on_message`를 가로채므로 거부된 turn은 wrapped organizer에 **도달하지 않고**, A-Mem의
**2콜(Ps1 + Ps2/Ps3)이 0콜**이 된다. 게이트를 붙이지 않은 `AMemOrganizer()`가
**A-Mem 논문 동작(전부 저장)** = 측정 baseline이다.

**처음엔 `AMemOrganizer(admission=...)` 생성자 인자였고, 그건 틀린 배치였다.** policy 패키지에
둔다는 것은 다른 organizer에도 적용된다는 주장인데 생성자 인자는 그 주장을 A-Mem 하나로 한정하고
mechanism이 policy를 import하게 만든다. wrapper로 바꾼 뒤 `organizers/amem/organizer.py`의 policy import는
0개이고, 적용 범위는 taxonomy §2.5에서 organizer 8종 전수 검증했다 — message 기반 3종 유효,
`passthrough` 무의미, `nemori`는 분절이 변하므로 ablation, task 기반 3종은 적용 불가.

### 의도적 편차 (전부 코드 docstring에 근거 명시)

| 항목 | upstream | 우리 | 이유 |
|---|---|---|---|
| N 비교 대상 | `[]` (상수 1.0) | 실 vector index top-1 | 결함 1. `VectorStore.search`가 cosine·내림차순을 계약으로 보장하므로 top-1 = 공식의 `max`. 전수 스캔 1회 → ANN 1회, embedder는 retrieval과 동일 |
| T 매칭 | 부분문자열 | **`matching="word"` 기본** (전체 토큰/구) | 결함 2. `matching="substring"`으로 upstream 정확 재현 가능 |
| C 스테머 | `rouge_score`(nltk 기본) | vendored Porter `mode="nltk"` | 의존성 3개(absl-py/nltk/six)를 feature 1개에 붙이지 않음. §4 참조 |
| U 출력 | bare float + regex 파싱 | `StructuredCaller` JSON | guided-json·1회 재시도·명시적 drop을 이미 신뢰하는 경로로 통일 |
| U 역할 | ollama `qwen2.5` 직접 | 전용 role `"admit"`, `phase="admit"` | `call()`에 temperature override가 없음 → 논문의 temp 0을 `RoleConfig`로 표현. role 없으면 1회 경고 후 명시적 degradation (A-Mem `extract`의 temp 0.7을 몰래 빌려쓰지 않음) |
| POS cues | 없음(논문에만) | 없음 | 결함 4. 없는 것을 발명하지 않는다 |

### 충실하게 남긴 것

- weight 제약(`w_i ≥ 0`, `Σ = 1`) 강제, 논문 operating point가 기본값(`PAPER_WEIGHTS`/`PAPER_THRESHOLD`).
- `TYPE_PRIORS`·`TYPE_KEYWORDS` **축자 전사**. 타이 해소 순서까지 보존 — 릴리스가
  `max(type_scores.items(), ...)`로 첫 최대값을 취하므로 **dict 선언 순서가 load-bearing**이다.
- C의 span 사전필터(생짜 단어 겹침 ≥1)와 `max` 집계.
- R의 `exp(−λ·age)`와 λ=0.01 기본값.

### 새로 넣은 것 1: **exact utility short-circuit**

U는 `[0,1]`이므로 `S ∈ [base, base + w_U]`. 이 구간이 θ 한쪽에 완전히 있으면 U는 판정을 바꿀 수
없다 → **LLM 콜 생략**. 근사가 아니라 정확하다.

비용 산술 (이게 `use_utility=False` 기본값의 근거):

- 게이트 없음: turn당 2콜.
- 게이트 + U: `1 + 2(1−r)` (r = 거부율) → **r > 0.5 일 때만 이득**.
- 게이트, U 없음: `2(1−r)` → 절감 `2r`, 게이트는 완전 LLM-free.

단, short-circuit은 **전면 생략이 아니다**. 논문 weight에서 straddle 구간은 `base ∈ [0.45, 0.55)`
이고, `unknown`(T=0.5) + 낮은 C가 정확히 거기 떨어진다. 즉 중간 점수 turn은 여전히 U를 지불한다
(`test_paper_weights_still_call_the_llm_inside_the_straddling_band`가 이걸 고정).

### 새로 넣은 것 2: `AdmissionStats` — 미참조율 축의 관측 장비

`seen / admitted / rejected / admit_rate / utility_calls / utility_skipped / feature_means /
type_counts`. write-path-critics 조사가 "우리가 전혀 못 재는 축"이라 지적한 **저장 노트 미참조율**을
재려면 admit rate와 feature 분포가 영속 기록돼야 한다. U의 평균은 분모가 `seen`이 아니라
`utility_observations`다 — short-circuit된 후보가 평균을 0으로 끌어내리지 않도록.

`AdmissionDecision.as_dict()`는 flat JSON 1행이라 full-artifact-capture 요건의 per-decision
기록에 그대로 들어간다.

### 발견: **R은 streaming 게이트에서 구조적으로 무신호다**

배선 버그가 아니다. turn은 도착 즉시 판정되므로 age ≈ 0 → **R ≡ 1.0**. upstream은 전 코퍼스를
하나의 `time.time()`에 대해 채점해서 R을 "몇 번째 세션인가"로 바꿔 *변동*을 얻지만, λ가 월 단위
스팬을 언더플로시켜 그 변동도 파괴한다(결함 1). **두 설정 모두 R은 상수이고 상수값만 다르다.**
충실하게 구현하고 `decay_rate`로 노출하고 stats에 실어서, 죽어 있다는 사실이 가정이 아니라
관측이 되게 했다.

### 발견: C는 hallucination이 아니라 **어휘 반복**을 잰다

논문은 C를 hallucination 방어로 프레이밍한다("well-supported by conversation evidence"). 그런데
릴리스가 채점하는 candidate는 `candidate_to_memory`가 `content = dialogue_turn.text`로 두는
**축자 transcript turn**이고, 우리도 게이트 시점에는 같다. 축자 turn이 transcript에 의해
"미지지"일 수 없으므로 C가 실제로 재는 것은 **이전 turn과의 어휘 반복** — confidence보다
redundancy에 가깝다. (노트 *구성 후*에 게이트를 두면 이 프레이밍이 성립하지만, 그러면 절감하려는
LLM 콜을 이미 써버린다. 이 트레이드오프가 게이트 위치를 결정했다.)

---

## 4. 부수 발견: vendored Porter가 nltk와 다르다 — **영향은 실측 0**

C를 `rouge_score`와 대조하다 발견. `rouge_score`는 `nltk.stem.porter.PorterStemmer()`를 기본
모드(`NLTK_EXTENSIONS`)로 쓰고, **snap-research/locomo의 `normalize_answer`도 마찬가지**다. 반면
`agmem/_porter.py`는 Porter(1980) 원문 조건을 구현했다. 즉 **이미 측정에 쓰인 F1/BLEU-1 경로**의
충실도 문제다.

step 1c `Y → I` 조건 차이:

| | 조건 | `saturday` | `cry` | `cried` | `enjoy` |
|---|---|---|---|---|---|
| Porter 1980 (`mode="original"`) | `(*v*)` 스템에 모음 있음 | saturdai | **cry** | cri | enjoi |
| nltk 기본 (`mode="nltk"`) | `(*c and not c)` y 앞이 자음 | saturday | **cri** | cri | enjoy |

`cry`/`cried`가 원문 조건에서 **병합되지 않는다** — 토큰 겹침 점수가 달라질 수 있는 실제 차이다.

**조치**: `PorterStemmer(mode=...)`로 파라미터화. 기본값은 `"original"`(= 기존 동작) 유지하고
`bench/locomo.py`가 명시적으로 그걸 지정 — 저장된 run 아티팩트의 재현성을 깨지 않기 위해.
`admission.py`는 `mode="nltk"`.

**영향 정량화 (무료 오프라인 재채점, LLM 0콜)**: 저장된 `results/repro/*.records.jsonl` 8개
파일의 `(pred, gold)`를 두 모드로 재채점.

| records 파일 | N | F1(original) | F1(nltk) | Δ | 변동 질문 수 |
|---|---|---|---|---|---|
| all_k10_ours_expand-on run1 seed1 | 1986 | 34.147 | 34.147 | +0.000 | **0** |
| all_k10_ours_expand-on run1 seed2 | 1986 | 35.027 | 35.027 | +0.000 | **0** |
| all_k10_ours_expand-on run1 seed3 | 1986 | 35.206 | 35.206 | +0.000 | **0** |
| all_k10_wujiang_expand-off run1 seed1 | 1986 | 34.920 | 34.920 | +0.000 | **0** |
| all_k10_wujiang_expand-off run1 seed2 | 1986 | 35.635 | 35.635 | +0.000 | **0** |
| all_k10_wujiang_expand-off run1 seed3 | 1986 | 35.747 | 35.747 | +0.000 | **0** |
| conv0_k10_ours_expand-off run1 | 199 | 36.109 | 36.109 | +0.000 | **0** |
| conv0_k10_wujiang_expand-off run1 | 199 | 36.876 | 36.876 | +0.000 | **0** |

**총 12,314 질문에서 F1이 바뀌는 질문이 1건도 없다.** 스테머 모드는 원리적으론 실제 차이지만
우리가 발표한 어떤 수치에도 영향이 없다 ⇒ **열려 있던 결정이 아니라 닫힌 리스크**다.

(검증 부수효과: `ours` 모드 재채점이 저장된 `f1` 필드와 max drift 4.8e-04로 일치 = 저장값 라운딩.
`wujiang` 모드는 max drift 1.0 — 예상된 결과로, upstream 자체 지표(stopword 부분점수)를 쓰는
별도 채점 경로다. `locomo-eval-fidelity` 메모의 anomaly와 정합.)

**남은 gap (별건, 미해결)**: `mode="nltk"`도 실제 nltk와 **완전 등가는 아니다**. 692단어 LoCoMo
어휘에서 10단어가 다르고 원인은 step1c가 아닌 3가지다 — ① nltk의 16개 불규칙 pool(`sky`→sky,
`outings`→outing), ② step 5a `_ends_cvc`에 추가된 2글자 `vc` 케이스(그래서 nltk는
`are`/`ate`/`use`를 보존), ③ `emotionally`가 도달하는 step 2 차이. 완전 등가화는 별도 과제로 남긴다
(영향이 0으로 측정됐으므로 우선순위 낮음).

C에 대해서는 이 잔차가 무해하다: candidate와 transcript span을 **같은 토크나이저**로 양쪽 처리하므로
스템 클래스의 일관된 재라벨링은 LCS를 바꾸지 않는다.

---

## 5. 레이어링 수정

**모듈 위치 (2026-07-26 재배치)**: 처음 `organizers/admission.py`에 뒀는데, A-MAC은 메모리 타입을
선언하지도 `MemoryOp`를 발행하지도 않으므로 방법론이 아니라 **control policy**다 →
`policies/admission.py`로 이동. 분류 근거는 문헌 자체의 mechanism/policy 구분이고
(`memory-component-taxonomy.md`), 이 재배치 과정에서 GRAVITY·RecMem을 policy로 봤던 이전 판단도
오분류로 교정됐다(둘 다 자체 메모리 계층 소유 → mechanism). `organizers/` 루트에 organizer만
남는다는 불변식은 `test_organizers.py::test_organizers_package_root_holds_only_organizers`가 강제한다.

`_porter.py`를 `agmem/bench/` → `agmem/` 루트로 이동. `organizers`가 `bench`를 import하면
`bench.locomo → agmem.memory → organizers` 순환이 닫힌다. 임포터 2곳(`bench/locomo.py`,
`tests/test_locomo_eval.py`) 갱신.

---

## 6. 검증 상태

- `tests/test_admission_gate.py` **54건 통과**. 전체 스위트 **245 passed / 1 skipped**
  (게이트 전 189). `ruff format` clean.
- **조립된 파사드(`AgenticMemory.add_message`) 관통 검증** — 조직자 단위가 아니라 실 write path.
  5 turn(선호·감탄사·정체성·감탄사·"…with my sister…")을 흘려서:

  | 설정 | 노트 | LLM 콜 | admit rate |
  |---|---|---|---|
  | `admission=None` (A-Mem 논문 동작) | 5 | **9** (extract 5 + distill 4) | 1.00 |
  | A-MAC `matching="word"` | 2 | **3** | 0.40 |
  | A-MAC `matching="substring"` (릴리스) | 3 | **5** | 0.60 |

  ungated의 distill이 5가 아니라 4인 건 **첫 노트에 이웃이 없어 Ps2/Ps3가 스킵**되기 때문
  (write-path-critics §2의 call 수 구조와 일치). word→substring에서 노트가 2→3으로 늘어난 항목이
  정확히 **"s-is-ter" turn**이다 = 결함 2가 우리 데이터에서도 admit rate를 올린다는 직접 증거
  (`test_substring_matching_admits_more_which_is_defect_2_in_our_own_pipeline`).
- ROUGE-L은 **실제 `rouge_score` 패키지**를 일회용 venv에 설치해 12쌍의 기준값을 생성하고
  테스트에 고정 — 의존성 없이 upstream 지표에 대한 상시 대조가 된다. 토큰화·F-measure 전부 일치.
- 두 스테머 모드를 실 nltk(기본/`ORIGINAL_ALGORITHM`)와 692단어로 대조 → §4의 잔차 10/4건 규명.
- 결함 1·2는 공식 코드를 그대로 실행해 재현(§2 표들이 그 출력).

## 7. 다음 (측정 재개 시)

1. **재튜닝이 선행 조건.** 논문 weight/θ는 결함 1·2 위에서 맞춰진 값이라 디버그된 feature에
   전이되지 않는다. 라벨 1회 패스(= `is_referenced` 산출은 LLM 0콜, 저렴)로 grid를 다시 잡을 것.
   `matching="substring"` vs `"word"` 두 조건을 같은 grid로 돌리면 "발표 recall이 버그의 산물"을
   우리 데이터로 확증/반증할 수 있다.
2. **성공 기준은 논문 F1이 아니다.** admission 결정 F1(N=225, oracle 라벨)이 아니라
   **"answer 품질 유지하며 노트 수·write 토큰 감소"**로 세운다 (write-path-critics §5 지시).
   대조군은 `admission=None`(A-Mem 전부 저장), 지표는 `AdmissionStats` + 기존 F1/J.
3. R은 ablation에서 **제외하거나 λ를 대화 스팬에 맞춰 재설정**해야 의미가 생긴다. 현 λ=0.01/h는
   LoCoMo에서 무조건 상수다.

[[write-path-critics]] · 다음 후보: GRAVITY anchors(2순위, `retrieval/steps.py` ReadStep),
RecMem recurrence gate(3순위, `consolidate()`)
