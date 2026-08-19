# X4 — 경계 문서화: MemoryOp의 substrate 한계, read 경로의 재해석, 비용-품질 평면

2026-08-19. 확장 계층 X4 스트림의 산출물 (`docs/_internal/plans/2026-08-07-expansion-layer-design.md` §2.X4).
지출 $0 — 이 노트의 모든 수치는 기존 커밋 문서·아티팩트의 인용이며, 새 측정은 하나도 없다.
입력: 외부 피드백(`docs/_external/2026-08-07-claude-feedback.md`, 이하 "피드백"), docs/16(추상화 스터디),
docs/18(LoCoMo 5-way), docs/20(LongMemEval), docs/19(FiNER), `src/agmem/organizers/base.py`,
`src/agmem/core/ops.py`. 피드백 문서가 요약한 외부 논문 내용은 **원문 미대조 2차 출처**이며 본문에서
매번 그렇게 표기한다 — 발표 수치는 1차 출처 대조 후에만 사실로 쓰는 이 레포의 규율(docs/17의
증거 기준이 그 성문화다)을 스터디 백로그에도 적용한 것이다.

한 줄 요약: **이 프레임워크의 완전함은 plaintext substrate 위에서의 완전함이고, 그 경계는 결함이
아니라 측정 가능한 사실이다.** 경계 바깥(activation/parametric memory)은 왜 바깥인지를 op 계약의
성질로 진술하고, 경계 안쪽에서는 "write 경로가 사는 점수"라는 통념을 우리 실측이 어디까지
뒤집었는지를 비용 축과 함께 고정한다.

---

## 1. MemoryOp 어휘의 경계 — 무엇이 원리적으로 표현되지 않는가

### 1.1 계약이 실제로 요구하는 세 성질

op 어휘는 `OpType` 8종이다: ADD / UPDATE / MERGE / DELETE / INVALIDATE / LINK / TAG / NOOP
(`src/agmem/core/ops.py:20-31`; 플랜 §2.X4가 "6종(+NOOP)"이라 부른 것의 현행 정확한 형태 —
TAG가 하나 더 있다). organizer는 store를 직접 만지지 않고 op 리스트를 반환하며, facade가
append-only 로그에 기록한 뒤 적용한다 (`organizers/base.py:1-21`, docs/04 §2). 이 배선이 성립하려면
substrate가 세 성질을 가져야 한다:

1. **이산 주소성** — 모든 op는 `target_type`/`target_id`로 특정되는 항목 하나를 겨냥한다.
2. **결정론적 재적용** — 로그 replay가 store를 재구성한다는 감사 보증은 op 적용이 순수 함수일
   때만 성립한다 (Zep replay 사건이 이 보증의 실측 사례 — docs/16 세션 4 발견 2).
3. **저장-인출 동형** — 쓴 항목은 그 항목 그대로 검색 파이프라인에 등판한다. `produces` 선언이
   "callers search what the methodology produced"를 잇는 다리다 (`base.py:75-84`).

plaintext 항목(노트·fact·episode·bullet·그래프 엣지)은 셋 다 만족한다. 9개 방법론이 전부 이 안에
있고, 접힘 손실 실측 0건이 docs/16의 종합 판정이다. 경계는 그 바깥에서 시작한다.

### 1.2 경계 사례 1 — activation memory (Cartridges 류; 피드백 §2.2, 2차 출처)

Cartridges(arXiv:2506.06266)는 코퍼스별 KV cache를 오프라인 학습(self-study)해 추론 시 로드한다고
보고된다. 이것이 어휘에 안 들어가는 이유는 op 하나하나가 막혀서가 아니라 **세 성질이 동시에
깨지기 때문**이다:

- 쓰기 단위가 이산 항목이 아니라 연속 텐서 전체다. "이번 세션에서 무엇이 ADD됐는가"에 대응하는
  `target_id`가 존재하지 않는다 (성질 1 위반).
- 쓰기가 gradient 최적화라 같은 로그를 재생해도 같은 KV가 나온다는 보증이 없다 (성질 2 위반).
- 읽기가 검색이 아니라 디코딩 시 로드다 — `produces` → 검색 파이프라인이라는 다리가 이어질 곳이
  없다 (성질 3 위반).

### 1.3 경계 사례 2 — parametric memory (knowledge editing / LoRA-as-memory; 피드백 §2.3, 2차 출처)

가중치 쓰기(ROME/MEMIT 류, LoRA 모듈)는 UPDATE처럼 *보이지만*, 우리 UPDATE의 의미론과 두 지점에서
어긋난다:

- **INVALIDATE가 원리적으로 불가능하다.** 우리 INVALIDATE는 "never physically remove"
  (`ops.py:28`) — 물리 삭제 대신 bi-temporal 마킹이고, T′축(트랜잭션 타임라인)은 append-only
  로그가 공짜로 제공한다 (docs/16 세션 4 발견 3). 가중치 갱신에는 되돌려 마킹할 원본이 없다.
- **저장≠인출이 substrate 성질로 등장한다.** 피드백 §2.3이 인용하는 Titans 평가(학습 손실 100%
  수렴, free-form 인출 0–40%; 2차 출처)는 "기억했다"와 "떠올린다"가 분리되는 사례다. 우리
  어휘는 성질 3(저장-인출 동형)을 전제하므로 이 분리 자체를 표현할 자리가 없다 — 아래 §7 긴장
  (2)에서 우리 데이터의 동형 현상(기권 = plaintext에서의 인출 실패)과 대조한다.

### 1.4 경계의 정확한 위치 — 이미 관찰된 압력 두 건

경계는 추측이 아니라 이미 두 번 밀려본 자리다. 둘 다 substrate 경계가 아니라 **read 추상화의
경계**였고, 프레임워크 안에서 수용됐다:

- ACE의 read는 검색이 아니라 전량 주입이라 read-step 밖 별도 경로(`get_playbook`)가 필요했다 —
  "읽기 추상화의 첫 예외 사례" (docs/16 세션 6).
- "read가 방법론 전체인 arm"(GAM식 passthrough + agentic read)은 기계 부재가 아니라 arm 부재다 —
  read 레시피는 config가 전체 공급 가능함을 `zep_cross_encoder`가 이미 증명했다 (플랜 §1 보정 2).
  구현은 X5 스케치(§9)로만 둔다.

따라서 정직한 진술은 이렇다: **MemoryOp 어휘는 plaintext substrate에 대해 (9개 방법론 실측
기준) 접힘 손실 없이 완전하고, activation·parametric substrate에 대해서는 §1.1의 세 성질이
깨지므로 확장이 아니라 별도 계약이 필요하다.** 이 진술 자체가 피드백 공백 A의 반영처다 —
MemCube(MemOS)가 주장만 하고 증명하지 않은 "범위 자인"을 우리는 계약 성질로 내렸다.

---

## 2. read 경로의 재해석 — full-context, passthrough, 그리고 72.90의 출처

### 2.1 먼저 교정: "우리 측정 Full-context 72.90"은 우리 측정이 아니다

docs/17 B-3 측정 칸은 "below **our measured** Full-context 72.90"이라 쓰고, 피드백도 이를 "자체
측정"으로 재인용했다. 이번 $0 검증 결과:

- 이 레포의 `results/`에는 LoCoMo full-context run 아티팩트가 **존재하지 않는다** (summary 전수
  스캔, 2026-08-19). full-context arm은 LongMemEval 하네스(`scripts/repro/exp_lme_reading.py`)에만
  있다.
- 72.90은 **Mem0 논문(arXiv:2504.19413) Table 2의 Full-context overall J**와 소수 둘째 자리까지
  일치한다 — 2026-07-23 arXiv 원문 직접 대조로 검증된 값이다 (세션 기록; Table 2에서
  Full-context 72.90 > Mem0ᵍ 68.44 > … > A-Mem 48.38).

∴ 이 노트는 72.90을 **Mem0의 발표 수치**로만 인용한다. 우리 67.60(docs/18)과 한 문장에 놓는
비교는 하네스·응답 모델·judge 프롬프트 판본이 다른 교차 비교이므로 방향 참고 이상으로 쓰지
않는다. docs/17 B-3의 "our measured" 표기는 교정 대상이지만 docs/17은 레인 A 소유라 여기서
고치지 않는다 — 플랜 §4 순차화 큐 등재 후보로 기록해 둔다.

### 2.2 그 대신, 같은 명제의 우리 소유 실측이 이미 있다 (LongMemEval, docs/20)

"write 경로가 아니라 reader에게 무엇이 닿는가가 점수를 정한다"는 명제는 발표 수치 이월 없이
우리 하네스로 세 번 측정됐다. 전부 docs/20의 실측이고, 플랜 42-43행의 경고대로 **컨텍스트
예산·비용 축을 붙여서만** 인용한다:

| 실측 | 수치 | 컨텍스트/비용 축 |
|---|---|---|
| C4: 메모리를 바이트 고정하고 읽기만 변경 | task-avg 스프레드 **12.57pp** / overall **15.40pp** (docs/20 §Results) | oracle, 4 arm 합계 $3.74 |
| C6: `_s`에서 full-context vs 검색 top-50 | 검색이 **+21.20pp** [+16.60, +25.60], write LLM 콜 **0** (docs/20 §Plain retrieval) | 프롬프트 중앙값 517,430 → 55,108 chars, $9.07 → $2.62 |
| 같은 기제, oracle에서 | 검색이 **−3.00pp** [−6.60, +0.40] — 부호 반전, 미분리 (docs/20 §What a retrieval layer costs) | haystack 중앙값 6.1K tok, 이미 다 들어감 |
| C7: 같은 read 경로, 코퍼스 9.9배 | `_s`−`_m` = **+8.60pp** [+5.00, +12.20] — 검색의 이득은 코퍼스 길이에 상수가 아니다 (docs/20 §`_m`) | `_m` 중앙값은 128K 윈도의 8.8배 — full-context arm 자체가 불가능 |

여기서 full-context의 정확한 지위가 나온다: **full-context는 "무압축 극한점"이되, 그 점이 최강인
것은 haystack이 컨텍스트에 들어가는 동안만이다.** oracle(6.1K)에서는 검색이 세금이고(−3pp),
`_s`(114K, 들어가긴 함)에서는 전문을 읽는 쪽이 23.20pp를 잃으며(83.60→60.40, docs/20 §`_s`),
`_m`(1.11M)에서는 full-context라는 arm이 정의되지 않는다. 피드백 §6의 "LongMemEval은 컨텍스트
윈도 테스트다"라는 주장(ledger C-4와 함께 서술하라는 플랜 §7 매트릭스 지시)에 대한 우리 판정은
그래서 이중적이다: `_s` 한정으로는 절반만 맞고(윈도에 *들어가는데도* 23점을 잃으므로 윈도
관리가 아니라 읽기 능력의 문제), `_m`에서는 성립하지 않는다(윈도가 아예 없다).

### 2.3 "write-path의 손실은 커버리지 손실" — LoCoMo 쪽 분해 (docs/18)

`J ≈ (1 − abstain) × acc`는 다섯 arm 전부에서 성립한다 (docs/18 §Where the deficits come from):

| arm | 기권률 | 답했을 때 정답률 | J |
|---|---|---|---|
| Nemori arm A | 19.5% | 84.0% | 67.60 |
| Nemori arm B | 20.6% | 82.9% | 65.78 |
| A-Mem (global-5 시점) | 26.9% | 82.0% | 59.87 |
| Zep `cross_encoder` | 32.2% | **62.5%** | 42.73 |
| Mem0 | **55.6%** | 71.7% | 31.82 |

Mem0의 대 Nemori 35.8pp 적자 중 **약 26pp가 커버리지, 약 10pp가 정확도**이고, 검색 자체는 한
번도 실패하지 않았다(1,986문항 전부 30개 fact 반환, 빈 결과 0 — docs/18). 즉 write 경로가 잃는
것은 "찾기"가 아니라 **reader에게 답을 나를 수 있는 단위를 남겼는가**다. read쪽 개입도 같은
통화로 지불된다: A-Mem의 keyword rewrite 제거는 기권 26.9→21.6%로 +5.26 J를 샀고(docs/18 §query
rewrite), per-hit 캡 개방은 기권을 거의 안 움직이고(26.9→25.1%) 폭 효과로 +1.36 J를 샀다(같은
문서 §link-expansion). 예외 하나는 기록해야 공정하다: **Zep은 유일하게 정확도 축에서도
실패한다**(답했을 때 62.5%, 5개 arm 최저) — 참가자 원문이 아닌 추상(트리플·요약)을 서빙하는
유일한 arm이라는 후보 설명과 함께, 분리 실험은 미실행 (docs/18).

피드백 §2.1의 GAM AOT/JIT 프레임(2차 출처)은 이 관찰의 이론화로 읽는다: AOT 압축(우리 write
경로 전부)은 미래 쿼리 분포를 모른 채 버릴 것을 정하는 결정이고, 그 비용이 위 표의 기권률
열이다. 다만 GAM의 자기 수치(RULER 90%+ 등)는 원문 미대조라 이 노트에서 사실로 쓰지 않는다.

---

## 3. 비용-품질 평면 — "단문 인용 금지"의 정식 서술

플랜 §1 보정 4가 금지한 것: "full-context가 모든 arm을 이긴다"를 비용 축 없이 인용하는 것.
정식 서술은 평면 위에서만 한다. 전 수치 docs/18 Results·read-path 표, docs/20, docs/19.

**LoCoMo 5-way (같은 하네스·같은 judge, 단일 seed):**

| arm | J | write 콜 | 총 $ | 문항당 서빙 tok |
|---|---|---|---|---|
| Nemori A | 67.60 | 3,579 | 2.24 | 3,574–4,409 |
| Nemori B | 65.78 | 2,759 | 1.82 | 〃 |
| A-Mem (헤드라인) | 61.23 | 11,754 | 3.90 | 3,322 |
| Zep `cross_encoder` | 42.73 | 27,449 | 5.09 | 1,086 |
| Mem0 | 31.82 | 5,890 | 2.17 | **837** |

이 평면에서 읽히는 것 세 가지:

1. **write 지출과 J의 상관은 음수다** — ρ(write 콜, J) = **−0.60** (docs/20 §organizer × `_s`의
   집계). 최다 지출 arm(Zep, 27,449콜 $5.09)이 하위 2위다.
2. **비용의 단위는 콜이 아니라 토큰이다** — ACE nodedup은 online 대비 콜 +2개(1,323→1,325)에
   비용 5.9배($1.461→$8.633)다 (docs/19 결과 표). 콜 수 회계는 이 축을 통째로 놓친다.
3. **read쪽 지출은 같은 평면에서 훨씬 싸게 움직인다** — LME `_s`에서 전문 읽기 $9.07/60.40 대
   검색 top-50 $2.62/81.60 (docs/20): 1/3 비용으로 +21.2pp. 반대 방향의 실측도 함께: A-Mem
   per-hit 캡은 컨텍스트 +74%를 내고 +1.36 J — "충실성으로는 필요, 배포로는 나쁜 거래"
   (docs/18 §link-expansion).

무압축 극한(full-context)의 비용 축은 우리 실측으로 LME에만 있다: 문항당 프롬프트 중앙값
517,430 chars vs 검색 55,108 chars (docs/20). LoCoMo 쪽 무압축 극한점은 우리 run이 없으므로
평면에 찍지 않는다 (§2.1).

---

## 4. 9개 방법론 store의 기능 축 재분류 (피드백 §3 반영)

피드백 §3의 축: factual(선언 지식, 최신값 우선) / experience(과거 행동·결과, append-only) /
procedural(방법 지식, 사례에서 추상화). 분류 근거는 각 organizer의 `produces` 선언과 docs/16의
세션별 대조. A-MAC은 store를 소유하지 않는 policy라 축 밖(docs/16 세션 8), passthrough는 원문
무가공이라 축의 원점이다.

| organizer | produces | 기능 축 | 실측된 갱신 정책 (docs/18 evolution log 외) |
|---|---|---|---|
| Mem0 | `semantic` | factual | ADD 5,654 · **NOOP 26,209** · UPDATE 1,077 · DELETE 227 |
| Zep | `facts`,`entities`,`communities` | factual (시간 축 포함) | UPDATE 5,268 · **INVALIDATE 1,293** (supersede 마킹) |
| MemMachine | `derivatives` (+profile `semantic`) | factual (문장 추출·프로필) | 발표 경로는 write LLM 0콜 (docs/16 세션 7) |
| A-Mem | `notes` | **혼합**: 단위는 턴(경험), 내용은 선언적 재서술+링크 | ADD 5,882 · LINK 5,866 · UPDATE 16,342 |
| Nemori | `episodes`,`semantic` | experience(서사) + factual(예측오차 증류) | MERGE 223 + INVALIDATE 223 (supersedes 전파) |
| MemoryOS | `pages`,`semantic` | experience(대화 페이지) + factual(프로필/KB) | STM→MTM→LPM 승격, lfu 퇴출 (docs/16 세션 3) |
| ReasoningBank | `experiences`,`strategies` | experience + **procedural** | **append-only ADD — 논문의 명시적 설계** (docs/16 세션 6) |
| ACE | `playbook` | procedural | ADD-only + 카운터 UPDATE + dedup 상시 (docs/16 세션 6) |
| G-Memory | `strategies` | procedural | ADD/EDIT/REMOVE ±점수, ≤0 prune (docs/16 세션 5) |

이 표에서 감사 가치가 있는 것은 분류 자체보다 **피드백 §3의 예측("축마다 갱신 정책이 다르다")이
우리 실측과 어긋나는 자리**다:

- 예측이 맞는 곳: factual store들은 실제로 UPDATE/supersede 계열이 살아 있다 (Mem0 UPDATE
  1,077, Zep INVALIDATE 1,293). experience인 RB experiences는 실제로 append-only다.
- **어긋나는 곳: procedural = "추상화로 갱신"이라는 예측에 RB strategies가 반례다** — 논문
  스스로 consolidation을 "a simple addition operation"으로 못박았다 (docs/16 세션 6). 같은
  procedural인 G-Memory는 점수 기반 EDIT/REMOVE를 돌린다. 즉 기능 축은 갱신 정책을 결정하지
  않고, 방법론의 자기 설계가 결정한다 — 축은 서술 도구이지 규범이 아니다.
- **축 미분리의 비용은 이미 한 번 실측됐다**: G-Memory의 backward ±점수가 같은 `strategies`
  타입을 쓰는 RB(append-only, feedback 무개념)에 새던 사건 — `on_feedback`이 organizer 소유가
  된 직접 원인이다 (`base.py:148-165`, docs/16 세션 5·6). 피드백 §10 체크리스트 1("종류를
  분리하라")의 우리 쪽 실증이 바로 이 사건이다.
- procedural 쌍(RB/ACE)의 교차 링크: 두 store는 같은 벤치에서 나란히 측정됐고 **둘 다 무학습
  base와 미분리**다 — FiNER 441 페어링에서 base 48.24 / ACE online 46.71 / RB 48.24→48.24
  (docs/19 결과 표·§rb). procedural 축의 존재가 이득의 존재를 함의하지 않는다는 실측점.

---

## 5. sleep-time 대응 — 훅 계약이 이미 구현한 분리 (피드백 §5 반영)

Letta 패턴(피드백 §5, 2차 출처): primary agent에게 메모리 편집 도구를 주지 않고, 편집 도구는
sleep-time agent에 붙인다. 우리 organizer 훅 계약과의 구조 대응:

| Letta 패턴 | 우리 계약 | 근거 |
|---|---|---|
| primary agent는 메모리 편집 불가 | 답변 경로는 op를 내지 않는다 — 검색 후 유일한 write-back 훅 `on_retrieval`은 "Must be cheap: no LLM calls here" | `base.py:138-145` |
| 편집 도구는 별도 에이전트 소유 | 모든 mutation은 organizer 훅이 반환한 op로만, facade가 로그 후 적용 | `base.py:1-21` |
| 유휴 시간 consolidation | `consolidate()` — "deferred management pass", 커서 복구로 재시작 내성 | `base.py:173-177` |
| 세션 경계에서의 정리 | `flush_buffer()` — end-of-ingestion drain | `base.py:179-186` |

방법론 단에서도 같은 갈림이 이미 노출돼 있다: Nemori semantic은 upstream 동기 인라인이 기본이고
`consolidation="semantic_offline"`이 우리 확장이다 (docs/16 세션 2) — 정확히 "쓰기를 언제로
미루는가"라는 sleep-time 축의 config화다.

**대응이 아닌 것도 명시한다.** Letta의 sleep-time은 스케줄링(비동기, 유휴 시간)이고 우리
`consolidate()`는 호출 시점이 caller 소유인 동기 패스다 — 대응은 **권한 분리의 구조**(누가
쓸 수 있는가)이지 실행 시점의 비동기성이 아니다. 피드백 §5의 "offline policy improvement"
해석과 Auto-Dreamer(GRPO consolidation 학습)는 스터디 백로그(§8)로만 간다.

---

## 6. 피드백 §10 체크리스트 7항 × 우리 프레임워크 대조표

| # | 체크리스트 | 우리 상태 | 근거 |
|---|---|---|---|
| 1 | 메모리 종류 분리 | **부분 충족** — `produces` 타입은 분리되나 기능 축과 1:1이 아니고, 미분리 비용은 strategies 공유 사건으로 실측 후 `on_feedback` 소유권으로 교정 | §4; `base.py:148-165` |
| 2 | 쓰기를 읽기에서 분리 | 충족 — 훅 계약이 구조로 강제 | §5 |
| 3 | 비가역 연산 회피 | **설계는 제공, 강제는 안 함** — INVALIDATE("never physically remove")·MERGE+supersedes·append-only 로그가 있으나, DELETE도 어휘에 있고 Mem0 충실성이 227회 사용 | `ops.py:28`; docs/18 |
| 4 | provenance 필수화 | 충족 — ops.jsonl이 writer·시점 전수 기록, `MemoryEvent.source` (X2의 계보 확인 요건은 별도 — 플랜 §2.X2) | `base.py:34-48` |
| 5 | 쓰기 경로 = 신뢰 경계 | **미구현** — X3(포이즈닝)은 파일럿 견적까지만, 본런은 상품단 이후로 확정 | 플랜 §5 결정 ③ |
| 6 | 공개 벤치 점수로 고르지 마라 | 이 레포의 존재 이유 | docs/17 전체, X1 |
| 7 | write 비용 포함 회계 | 충족+확장 — ingest $ 별도 계상에 더해 "비용은 콜이 아니라 토큰" 실측 | docs/18 Results; docs/19 (§3-2) |

---

## 7. 다섯 긴장(피드백 §9) × 우리 실측점 — 골격

각 긴장마다 우리 데이터의 점 하나 이상. 상세 곡선은 X6 노트(압축-왜곡 실측, 병렬 스트림)가
소유하고 여기는 자리만 고정한다.

1. **압축 vs 보존** — Mem0 46.0자 × 5,427개, 문항당 837 tok, 기권 55.6% / Nemori 서사 episode,
   3,574–4,409 tok, 기권 19.5% (docs/18). "더 많이 저장하고 더 적게 기억한다"(more memories,
   less memory)가 이 축의 우리 쪽 한 줄이다.
2. **저장 ≠ 인출** — parametric에서는 substrate 성질(§1.3, 2차 출처)이지만, plaintext에서도
   기능적 동형이 나타난다: Mem0는 검색이 매번 30개를 반환하는데도 55.6%를 기권한다 — 저장은
   됐고 인출 *단위*가 답을 못 나른다 (docs/18). LME `_m`의 8.60pp 감쇠(docs/20)는 같은 축의
   코퍼스 길이 의존 버전.
3. **쓰기 경로의 비가역성** — INVALIDATE·supersedes·append-only 로그가 우리 쪽 답이고(§6 행 3),
   비가역 op의 실사용도 실측돼 있다 (Mem0 DELETE 227, docs/18).
4. **long context가 메모리를 대체하는가** — 우리 판정은 §2.2: 들어가는 동안만, 그리고 들어가도
   23.20pp를 잃으면서 (docs/20). `_m`(1.11M tok)에서 대체론은 정의부터 실패한다.
5. **손으로 짠 정책 → 학습된 정책** — RL은 명시적 비채택(플랜 §2 비채택 절: RTX 2060 6GB +
   로컬 장시간 추론 금지). 대체물이 X2 오라클 상한(사후 분포로 학습 가능 여지를 정량화)이고,
   A-MAC 재튜닝도 측정이므로 보류 유지 (docs/16 세션 8).

---

## 8. 스터디 백로그 — 신규는 GAM 하나, 나머지는 순서만

기존 `docs/research/` 스터디가 커버하지 않는 신규 항목은 **GAM(arXiv:2511.18423) 하나**다
(플랜 §7 매트릭스 §2.1행). 나머지는 피드백 §11의 11편을 백로그로 이관하고 우선순위만 확정한다
(플랜 §7 §11행): **GAM → rate-distortion(2607.08032) → Cartridges(2506.06266 + 2508.17032) →
서베이(2603.07670) → Auto-Dreamer → 나머지**.

각 항목이 이 레포의 어느 실측과 만나는지 — 전부 원문 미대조 상태의 예약이며, 스터디 시점에
원문 대조가 선행 조건이다:

- **GAM**: AOT/JIT 프레임 ↔ §2.3의 커버리지 분해; Researcher ↔ X5 스케치. 서베이(2603.07670)의
  write–manage–read 3D taxonomy는 docs 상위 프레임 후보 (플랜 §7 §0–1행).
- **rate-distortion(2607.08032)**: Mem0 46.0자/5,427 vs Nemori episode의 실측점과 연결 —
  X6 노트의 이론 축.
- **Cartridges(+기계적 해석 2508.17032)**: §1.2 경계 사례의 원문 검증.
- **Auto-Dreamer**: §5의 "consolidation을 학습" 갈래 — provenance 기반 region rewriting이라는
  갱신 의미론이 우리 op 어휘와 어떻게 대응/비대응하는지가 스터디 질문.

---

## 9. X5 스케치 — 스케치만 (플랜 §2.X5의 요약, 구현·견적 없음)

passthrough store(원문 turn 그대로, ingest는 임베딩만 = write LLM 0콜) + read 레시피 확장
(질문 분해→다중 검색→통합, 문항당 콜 수 config 캡). 주는 것: GAM의 AOT vs JIT 주장을 우리
하네스·동일 judge로 독립 검증 + full-context와 write-path 사이의 빈 구간 점. C6(+21.2pp가 이미
write 0콜)이 이 arm의 하한을 시사하지만, agentic read가 그 위에 무엇을 더하는지는 미측정.
착수 조건은 플랜 그대로: X1·X6 결과의 가치 판정 → 견적 → 승인. 여기까지가 이 노트의 전부다.

---

## 10. 이 노트가 확립하지 않는 것

1. **경계 바깥의 어떤 주장도 검증하지 않았다.** §1.2–1.3, §8의 외부 논문 내용은 전부 피드백
   문서 경유 2차 출처다. 이 노트가 확립한 것은 "우리 계약의 세 성질이 저들의 보고된 메커니즘과
   양립하지 않는다"까지이고, 저들의 보고 자체의 사실성은 스터디 백로그의 몫이다.
2. **"write 경로는 가치가 없다"를 확립하지 않는다.** C6의 +21.2pp는 write *LLM 콜*이 0인
   arm이지 메모리 시스템 무용론이 아니며, organizer×`_s`는 측정하지 않기로 한 상태다
   (docs/20 §organizer — 후보 confound와 4개의 선행 null이 그 결정의 근거). LoCoMo 5-way의
   ρ=−0.60은 5점 단일 seed 상관이다.
3. **full-context 72.90을 우리 좌표계의 점으로 쓸 수 없다** (§2.1). LoCoMo 무압축 극한점은
   우리 run이 없고, 이 노트는 그 공백을 메우지 않았다 — 메우는 것은 지출이 필요한 별도 결정이다.
4. **기능 축 재분류(§4)는 서술이지 실험이 아니다.** 같은 arm에서 축 하나만 움직인 비교(예:
   Nemori episodes만/semantic만)는 X6의 1급 증거 규율에 따라 그쪽 노트가 소유한다.
5. **sleep-time 대응(§5)은 구조 대응이지 성능 주장이 아니다.** 분리가 latency·신뢰성을
   개선한다는 Letta 쪽 주장을 우리는 측정한 적이 없다.
6. **단일 seed 한계는 이 노트의 모든 LoCoMo 인용에 이월된다** (docs/18 각주 7·9·10: 상위 3 arm
   순위는 이 정밀도에서 미해결).
