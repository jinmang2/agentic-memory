# 3차 충실도 감사: 논문 재분석 + 구절 단위 코드 포렌식 (2026-07-17)

> 방법: 병렬 감사 4건 — 논문 전용 분석 2건(A-Mem arXiv:2502.12110 v11, Nemori
> arXiv:2508.03341 v1+v4 전문 정독), 코드·이슈 포렌식 2건(upstream 당일 소스 전문
> + GitHub 이슈 트래커 전수 조사). 대상 HEAD: f49c5cf (P0/P1/P2 수정 반영 후).
> Nemori upstream 원본은 스크래치 디렉터리에 보관됨(세션 한정).

## 요약 판정

| 시스템 | 코어 알고리즘 | 프롬프트 충실도 | 벤치 구성 | 종합 |
|---|---|---|---|---|
| A-Mem | ● (재현 필수 11항목 전부 반영) | ◑⁺ (의미 보존 축약, 필드 구조 일부 변형) | ● (P2 순수화 후) | **●⁻** |
| Nemori | ● (v1 기준) / ◑ (v4 기준 — 통합 모듈 2개 부재) | **◑ (구절 격차 다수 — 이번 신규 발견)** | ◑⁺ (답변 프롬프트 과소) | **◑⁺** |

2차 감사와 달라진 점: 코어 메커니즘·검색 설정은 확정적으로 충실 판정. 대신
**구절 단위 대조에서 프롬프트 디테일 격차**(특히 Nemori)와 **docstring 이슈번호
오배정**(A-Mem)이 신규 발견됨.

---

# 1. A-Mem

## 1.1 논문 핵심 포인트 (재분석)

- 3 메커니즘: Ps1 노트 구성(식1-2), Ps2 링크 생성(식4-6: 코사인 top-k 이웃 →
  LLM 관계 판단), Ps3 진화(식7: 이웃의 context/tags 갱신·대체).
- 식(3) `e_i = f_enc[concat(c_i, K_i, G_i, X_i)]` — 메타데이터 풍부 임베딩이
  헤드라인 공식. **agiresearch 라이브러리판은 이 식을 미준수**(content만 임베딩);
  WujiangXu 논문 재현판과 우리 구현이 준수.
- 읽기 경로: Fig.2 캡션 "linked memories automatically accessed" = 1-hop 링크
  확장. 어블레이션(Table 3, multi-hop F1): w/o LG&ME 9.65 → LG만 21.35 → full
  27.02. **링크 생성 +11.7이 최대 기여, 진화 +5.67** — 논문 서술("evolution
  critical")과 달리 델타상 LG가 더 큼.

## 1.2 논문 자체의 약점

1. **카테고리별 k 스윕(App A.5, k=10~50) = eval 오버피팅.** 테스트셋에
   하이퍼파라미터를 맞춤. (우리는 k=10 고정 — 오염 없음)
2. **cat5 gold 노출.** 공식 eval은 adversarial에서 정답과 "Not mentioned"를
   이지선다 제시 (WujiangXu #8에서 외부 지적 존재). 우리는 미재현(더 엄격).
3. 어블레이션이 multi-hop 단일 카테고리·단일 런·분산 없음.
4. **링크 효용이 메타데이터 임베딩과 분리 검증 안 됨** — "메타 임베딩만, 링크
   없이" 조건이 어블레이션에 없음.
5. 재현성 붕괴: 공식 리포 2개가 서로 다르고 둘 다 결함(아래 1.4).
6. DialSim F1 3.45 vs 2.55 — 지표 분해능 의심.

## 1.3 재현 체크리스트 vs 우리 구현

필수 11항목(Ps1 구성 / 식3 임베딩 / 이웃 k=5 코사인 / 단일 진화 콜 / strengthen
tags / update_neighbor / notes 단독 k=10 / 1-hop 확장 / 키워드 질의 / 메타데이터
렌더 / 예산) **전부 반영 확인** — 근거는 2차 감사와 amem-paper 보고의 매핑 표
(`amem.py`, `pipeline.py:71-119`, `bench/locomo.py:33-40`, `exp:101`).

## 1.4 신규 발견 (코드 포렌식)

| # | 심각도 | 내용 | 위치 |
|---|---|---|---|
| A1 | **높음(문서 정확성)** | docstring 이슈번호 오배정: "score 의미 반전; 랭킹 유지"는 **#23**인데 #24로 인용(#24는 L2-vs-cosine — 비정규화 임베딩에선 랭킹 영향 가능). "#10=silent skip"도 오배정(#10은 add_note가 analyze_content를 안 부르는 문제; silent skip은 이슈번호 없는 광역 try/except). deep-audit §5의 #24 서술도 같은 혼동 | `amem.py:11-14` |
| A2 | 중간 | **should_evolve 게이팅 의미차**: upstream은 should_evolve=True + actions에 "strengthen"이 있을 때만 링크 추가. 우리는 connections를 should_evolve와 무관하게 항상 적용 → 링크가 더 자주 생성됨 | `amem.py:147-154` |
| A3 | 중간 | 양방향 링크(upstream 단방향) — 1-hop 확장 폭 확대. cap=5로 억제, docstring에 인정됨 | `amem.py:152-154` |
| A4 | 낮음 | EVOLVE 프롬프트 구조차: actions enum 부재, 새 노트 제시가 keywords 대신 tags, 이웃 텍스트에 timestamp/keywords 생략 + content 200자 절단 | `amem.py:75-94,133-137` |
| A5 | 낮음 | write 이웃검색 질의가 enriched(embedding_text) — upstream은 raw content | `amem.py:124` |
| A6 | 낮음 | max_tokens=300 (upstream 1000) — neighbor_updates 다수 시 JSON 절단 → 파싱 실패 → drop으로 진화 적용률 저하 가능 | `exp_locomo_conv0.py:29` |
| A7 | 낮음 | pipeline 주석의 근거함수 오귀속: eval read-path는 `find_related_memories_raw`(search_agentic은 라이브러리판 전용·eval 미사용) | `pipeline.py:6,98` |
| A8 | 정보 | "paper-reproduction repo is self-consistent" 과장 — plain `memory_layer.py`는 `import re` 누락으로 메타데이터 전량 빈 값(WujiangXu #24 "all metrics zero"·#5로 외부 코러보), robust판만 정상 | `amem.py:8-9` |

이슈 재검증: agiresearch #32(index-vs-id)·#11(id 미검증) HEAD에 잔존 → 우리 수정
유효. #25(enriched 임베딩 제안)가 우리 embedding_text 선택을 지지.
WujiangXu #16/#21이 find_related_memories_raw의 cap 의미(전역 vs per-hit)가
upstream에서도 애매함을 확인 → 우리 전역 cap=5 선택은 방어 가능.

## 1.5 검증된 프롬프트

- NOTE_PROMPT: 의미충실 축약("nouns/verbs 초점", "don't be too redundant",
  "intended audience" 문구 누락 — 영향 낮음).
- KEYWORD_QUERY_PROMPT: **verbatim 확인** ('cosmos' quirk 포함).

---

# 2. Nemori

## 2.1 논문 핵심 포인트 (v1/v4 재분석)

- 두 원리: Two-Step Alignment(경계 검출 + 시간 절대화 서사), Predict-Calibrate
  (기존 지식으로 예측 → **원본 대화 M 기준** gap만 증류).
- **v1→v4 방법론 개정**: per-message f_θ(σ=0.7, β_max=25) → 관측창 w=20 배치
  분할로 교체(σ/β 표기 삭제). 신규 정식 모듈 2개: 에피소드 병합(§3.2.3, K_e=5,
  >1h 금지), semantic new/merge/conflict 통합(§3.3.3, K_m=5, τ=0.70).
- v4 수치: LoCoMo(gpt-4.1-mini) 80.8 vs Full-Context 80.6, temporal +15.9% over
  A-MEM.

## 2.2 논문 자체의 약점

1. **어블레이션 결론이 버전 간 반전 (최대 red flag)**: v1은 episodic 제거가 더
   아픔(semantic 기여 +3.9) ↔ v4는 semantic 제거가 더 아픔(-25.1%). 같은
   방법·데이터셋에서 핵심 결론이 뒤집힘.
2. **Predict-Calibrate 이득은 모델이 약할수록 큼**: 직접추출 대비 gpt-4o-mini
   +25% → gpt-4.1-mini +14.4%. 강한 모델에선 semantic 전체 기여가 +3.9점.
   → **0.6B에서는 예측문 환각으로 PC가 직접추출보다 나쁠 수 있음** (우리 결과
   해석에 필수 캐비앗).
3. v4의 "w=5~40에서 ±1%" 주장은 적응적 경계 검출의 실익이 작다는 자기반박 —
   Two-Step의 절반이 흔들림.
4. r=2 원문 첨부 = 부분적 raw-RAG; temporal 우위도 답변 프롬프트의 날짜 계산
   지시 기여와 분리 안 됨.
5. judge가 gpt-4o-mini 자기평가(self-preference 위험). 이슈 #8에서 타 사용자
   재현 실패(0.388 vs ~0.83) 보고 — 프롬프트 민감도 높음.

## 2.3 재현 체크리스트 vs 우리 구현

조직화 코어 + read 경로 A~I 항목(적응 경계 / 서사+시간 절대화 / timestamp /
PC의 GT=원본 M / cold-start / m=2k=20 / r=2 / σ·β 파라미터 / 예산) **전부 반영**.
미구현: **v4 신규 모듈 2개** — 에피소드 병합(J), semantic 통합(K).
K는 v4가 semantic 기여의 핵심으로 내세운 redundancy 회피 장치 → **v4 재현
목표 시 최우선 구현 대상** (어블레이션 반전 근거).

## 2.4 신규 발견 (코드 포렌식 — 구절 단위)

| # | 심각도 | 내용 | 위치 |
|---|---|---|---|
| N1 | **높음** | **답변 프롬프트 과소**: upstream `search.py`는 timestamp 추론 8개 지시(워크드 예시 "4 May 2022 last year→2021", 모순 시 최신 우선, 캐릭터명≠유저), 7단계 접근, "<5-6 words", **temp 0.0**. 우리는 최단 span 한 줄 + temp 0.1. temporal(cat2)·multi-hop(cat1) 직접 타격. 단 이 프롬프트는 벤치 공통이라 상대 비교는 유지 — "공통 최소 프롬프트 vs 방법론별 공식 프롬프트" 프레임 결정 필요 | `bench/locomo.py:42-51` |
| N2 | **높음** | **에피소드 생성 프롬프트 격차**: title "(10-20 words)" 스펙, 상대시간 "원 표현 뒤 괄호 병기" 스타일 + few-shot 예시(하이킹), Time Analysis 블록(Primary/Secondary/Fallback/ISO), content 요구(결정·감정·계획·"coherent story"·시간 hour 단위), timestamp "실제 발생시각 분석"(우리는 첫 메시지 복사), boundary_reason(topic 라벨) 주입 — 전부/대부분 누락. **논문 헤드라인인 temporal anchoring이 프롬프트 수준에서 약화됨** | `nemori.py:82-94` |
| N3 | 중간 | 컨텍스트 렌더 구조: upstream은 "Episodic Memories:"/"Semantic Memories:" 2섹션 분리, `- [{created_at}] {content}`, source messages에 role 접두사 + 4칸 들여쓰기. 우리는 점수순 인터리브 단일 리스트, role 접두사 없음 | `pipeline.py:141-155`, `types.py:190` |
| N4 | 중간 | CALIBRATE/DIRECT 프롬프트: 7개 HIGH-VALUE 카테고리, LOW-VALUE 목록, GOOD/BAD 예시, "include ALL specific details", "quality over quantity", **시간정보 절대 금지**(우리는 상대시간만 금지) 누락 → 저가치 semantic 노이즈 위험 | `nemori.py:109-140` |
| N5 | 중간 | 에피소드 병합 부재 — upstream **eval에서 활성**(0.85/top-5/>1h 금지). 단 LoCoMo는 세션 간격이 수일~수개월이라 >1h 게이트가 세션 간 병합을 대부분 차단 → 점수 영향 제한적, 에피소드 수·비용 격차는 실재. docstring "논문 부재" 서술에 "리포 eval에선 기본 on" 추가 필요 | `nemori.py:17` |
| N6 | 낮음 | 경계 프롬프트: Structural Signals("speaking of which" 등), Intent Transition 상세, topic 라벨 출력, "HIGH SENSITIVITY" 톤 누락. confidence 0.7은 upstream에 대응물 없는 우리 고유 파라미터(v1 논문 형식) | `nemori.py:66-79` |
| N7 | 낮음 | PREDICT 프롬프트: 4대 초점(Core Facts/Key Decisions/Knowledge Exchange/Logical Flow)·IGNORE 목록 누락. 입력(title only + knowledge)은 일치 | `nemori.py:97-106` |
| N8 | 낮음 | DIRECT 입력 소스차: upstream은 에피소드 title+content, 우리는 raw segment | `nemori.py:246` |
| N9 | 낮음 | 온도: upstream 세그멘테이션 0.2 / answer 0.0, 우리 전역 0.1 | `exp:28-30` |
| N10 | 정보 | upstream main은 per-message 경계 코드가 **아예 없음**(배치 전용). CALIBRATE 입력도 upstream은 타임스탬프 없는 `role: text`, 우리는 `[ts] role: content`. semantic dedup PR #19는 미병합 — main 기준 격차 아님 | — |

확정 사실(재확인): 검색 채널은 upstream search_method="vector"(dense 전용)와
우리 episodes/semantic dense 전용이 **일치**. 에피소드 임베딩 텍스트
(title+content ≈ title\nnarrative) 실질 동일. PC 게이팅 로직 일치.

---

# 3. 종합: 결과 해석 프레임과 수정 권고

## 3.1 결과 해석에 영향을 주는 괴리 순위 (통합)

1. **0.6B 로컬 LLM + e5-small** — 논문 절대치 비교 불가는 기지 사항이나, 이번
   분석으로 구체화: PC(예측 환각)·temporal anchoring(날짜 산술) 둘 다 LLM
   품질 직결 → **Nemori의 두 win-reason이 모두 저품질 모델에서 검증되는 구조**.
   상대 비교로만 해석.
2. **답변 프롬프트 프레임(N1)** — 공통 최소 프롬프트는 방법 간 공정하나,
   temporal 카테고리에서 전 방법이 논문 세팅보다 불리. 방법론별 "공식 답변
   프롬프트" 모드를 옵션으로 둘지 결정 필요.
3. **Nemori 에피소드 프롬프트(N2)** — 서사 품질·시간 병기 스타일이 upstream보다
   빈약 → Nemori 수치가 구조적으로 저평가될 수 있음.
4. **A-Mem 링크 생성률 차이(A2+A3)** — 게이팅 완화 + 양방향으로 링크가 upstream
   보다 많음 → 1-hop 확장 강도가 다름. multi-hop 수치 해석 시 캐비앗.
5. **semantic 통합(K)·병합(J) 부재** — v4 재현 목표 시 K 최우선.

## 3.2 수정 권고 (우선순위)

**P-doc (즉시, 코드 동작 무변경)**
1. `amem.py` docstring 이슈번호 정정: #24→#23(score 반전), #10 인용 제거(광역
   try/except로 서술), "self-consistent" → "robust판 기준"(A1, A8).
2. `pipeline.py` 주석 search_agentic → find_related_memories_raw(A7).
3. deep-audit §5의 #24 서술에 정오표 추가.

**P-w (write 경로 — 재인제스트 필요)**
4. Nemori EPISODE_PROMPT 보강(N2): title 10-20단어, 괄호 병기 스타일 + 미니
   예시, timestamp "발생시각 분석", content 요구(결정·감정·계획).
5. Nemori CALIBRATE/DIRECT 보강(N4): 고가치 카테고리·저가치 예시·시간 금지.
6. A-Mem should_evolve 게이팅 정합(A2) + max_tokens 상향(A6).

**P-r (read 경로 — 재인제스트 불필요)**
7. 렌더 2섹션 분리 + source role 접두사(N3).
8. 답변 프롬프트: 공통 프롬프트에 시간 계산 지시 추가 또는 방법론별 공식
   프롬프트 모드 옵션(N1) — 프레임 결정 필요.

**P-later**
9. v4 semantic new/merge/conflict 통합(K) — v4 재현 시 최우선.
10. 에피소드 병합(J) — LongMemEval 단계(기존 계획 유지).

## 3.3 이전 감사와의 이견 정리

- 2차 감사의 "A-Mem read 경로 불충실" 판정은 70ba537로 해소 — 현재 코어는 ●.
  캐비앗을 "read 부재"에서 "양방향 링크·게이팅 완화·eval 프레임 차이"로 교체.
- deep-audit이 merging 영향을 "중간"으로 본 것은 >1h 게이트 분석으로 하향
  (LoCoMo 점수 영향 제한적, 비용·구조 격차는 실재).
- deep-audit §5의 이슈 #24 서술은 #23과 혼동 — 본 문서 1.4 A1로 교정.
