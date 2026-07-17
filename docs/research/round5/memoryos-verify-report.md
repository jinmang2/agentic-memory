# MemoryOS 충실도 검증 보고서 (2026-07-17)

대상: `/home/jinmang2/agentic_memory/src/agmem/organizers/memoryos.py` (198줄)
기준: 논문 arXiv:2506.06326 v1 (repo 동봉 `Paper-MemoryOS.pdf`, 9쪽) + 공식 코드
github.com/BAI-LAB/MemoryOS (2026-07-17 `git clone --depth 1` 재다운로드).

업스트림에는 **코어가 3벌** 있고 서로 다르다 (docs/02의 "코어 4벌 복제 드리프트" 지적과 일치):

| 판 | 경로 | 특징 |
|---|---|---|
| pypi판 (주력) | `memoryos-pypi/` (`memoryos/`와 동일 모듈 구성) | Jaccard, heat α=β=1.0 γ=1, τ=24h |
| chromadb판 | `memoryos-chromadb/` | pypi판과 상수·공식 동일 (mid_term.py L22–25, L227, L229 확인) |
| **eval판 (LoCoMo 논문 수치 산출)** | `eval/` | **Dice 유사 keyword 항, heat α=β=0.8 γ=0.0001**, STM cap=1 |

이하 인용은 클론 사본 `scratchpad/MemoryOS/` 기준.

---

## 1. 클레임별 판정

### C1. STM capacity 10 + FIFO→MTM 흐름 — **부분 REFUTED (배치 단위가 다름, 미문서화)**

- capacity 10 자체는 CONFIRMED: pypi `memoryos.py:35` `short_term_capacity=10`, `short_term.py:10` `max_capacity=10`. 우리 `stm_capacity=10` 일치.
- **흐름은 다름.** 업스트림 pypi `memoryos.py:242-246`:
  ```python
  if self.short_term_memory.is_full():
      self.updater.process_short_term_to_mid_term()
  self.short_term_memory.add_qa_pair(qa_pair)
  ```
  `updater.py:102-103` `while self.short_term_memory.is_full(): qa = self.short_term_memory.pop_oldest()` — `is_full()`은 `len >= max_capacity`이므로 **가득 찬 뒤에는 매 add마다 가장 오래된 1개 페이지만 방출**된다(진짜 FIFO 롤링 윈도우, STM은 계속 9~10개 유지). 논문도 동일: "the **oldest dialogue page** is transferred from the STM to the MTM according to the FIFO principle" (§3.3).
  eval판은 아예 `max_capacity=1`로 QA쌍 하나마다 즉시 방출 (`main_loco_parse.py:233`).
- 우리 구현 `memoryos.py:88-93`: 10개가 쌓이면 **STM 전체를 한 번에 비우고** (`batch, self._stm = self._stm, []`) 10개 배치를 topic 분할. 결과:
  1. 배치 크기 10 vs 업스트림 1 — multi-summary가 받는 입력 단위가 다름.
  2. **QA 시점에 STM이 비어 있어** "최근 문맥 무조건 주입" 채널이 사라짐 (§4 참조).
  3. 우리 Episode는 **메시지 단위**, 업스트림 page는 **QA쌍 단위** — STM 10 = 업스트림의 절반 창.
- docstring의 "issue #65 silent data loss" 언급: 현재 업스트림 pypi판은 이미 수정됨 (`memoryos.py:240-241` "FIX: Migrate old entries BEFORE adding…"). docstring이 옛 상태를 서술.

### C2. F_score = cosine + Jaccard, 임계 0.6 — **업스트림 공식 CONFIRMED / 우리 단순화는 정확히 문서화됨, 단 부수 효과 미기술**

- 논문 식(3): `Fscore = cos(es, ep) + FJacard(Ks, Kp)`, `FJacard = |Ks∩Kp| / |Ks∪Kp|`.
- pypi `mid_term.py:208-220` (insert 시):
  ```python
  semantic_sim = float(np.dot(existing_summary_vec, new_summary_vec))
  ...
  s_topic_keywords = intersection / union
  overall_score = semantic_sim + keyword_similarity_alpha * s_topic_keywords   # alpha=1.0
  ```
  임계: `memoryos.py:40` `mid_term_similarity_threshold=0.6` → `insert_pages_into_session(similarity_threshold=0.6)`. (updater 자체 기본값은 0.5이지만 파사드가 0.6으로 덮음. eval판도 0.6, `main_loco_parse.py:236`.)
- **eval판은 Jaccard가 아니라 Dice 평균 변형**: `eval/mid_term_memory.py:148-152`
  `s_top = 0.5 * (|overlap|/|A| + |overlap|/|B|)` — 논문 수치는 이 판으로 산출됨. "Jaccard 항"이라는 docs/10 표현은 논문·pypi 기준으로는 맞고 eval판 기준으론 근사.
- 우리 `memoryos.py:131-133`: `ctx.vec.search(..., k=1)` cosine만, 임계 0.6. docs/10의 "F_score의 Jaccard 항 부재(cos만 사용)" 캐비앗은 **정확**.
- **미기술 부수 효과 2건**:
  1. 업스트림 점수는 [0,2] 범위(cos+Jaccard)에서 0.6과 비교, 우리는 [−1,1] cosine 단독으로 0.6과 비교 → **실질 병합 기준이 더 엄격**해 세그먼트 분열(fragmentation) 방향으로 편향.
  2. Jaccard 부재의 근본 원인: **keyword 추출 자체가 없음.** 업스트림은 multi-summary가 theme별 `keywords`를 반환하고(`prompts.py:74-77`) page/summary keyword를 저장·병합·검색에 씀. 우리 TOPIC_PROMPT/스키마에는 keywords 필드가 없다. Jaccard 항 복원 시 프롬프트·스키마·저장까지 같이 필요.

### C3. Heat 공식 (N_visit, L_interaction, R_recency, 상수) — **공식 형태 CONFIRMED / "공식 동일" 주장은 write-path 한정, read-path 결손으로 실질 REFUTED**

- pypi/chromadb `mid_term.py:21-36`:
  ```python
  HEAT_ALPHA = 1.0; HEAT_BETA = 1.0; HEAT_GAMMA = 1; RECENCY_TAU_HOURS = 24
  return alpha * N_visit + beta * L_interaction + gamma * R_recency
  ```
  `utils.py:228-237` `R_recency = exp(-Δhours/24)`. 우리 `_segment_heat` (`memoryos.py:78-84`) `n_visit + length + exp(-hours/24)` — **수식·상수 동일. CONFIRMED.**
- 단, 판별 편차 주의: 논문은 µ=1e7초(≈2778h, §3.3), **eval판은 α=β=0.8, γ=0.0001** (`eval/mid_term_memory.py:24`). "코드 기본값" 기준으로만 동일하다는 한정을 docs에 명기할 것.
- **[신규·중요] N_visit이 우리 구현에서 영원히 0.** 업스트림은 **검색 시** `session["N_visit"] += 1`, `last_visit_time` 갱신, `access_count_lfu` 증가 (`mid_term.py:346-351`; 논문 §3.4 "After retrieval, the segment's visit counter Nvisit and recency factor Rrecency are updated"). 우리 organizer는 read-path 훅이 없어 heat = length + recency뿐이다. 즉 **"자주 조회되는 세그먼트가 뜨거워진다"는 논문의 핵심 피드백 루프가 없음.** docs/10에 미기재.

### C4. MTM→LPM 승격 임계 — **CONFIRMED (값) / 트리거 시점·단위는 다름 (미문서화)**

- 논문 §3.3: "Segments with heat exceeding a threshold τ (i.e., 5) are transferred to LPM."
  pypi `memoryos.py:26` `H_PROFILE_UPDATE_THRESHOLD = 5.0`; eval `main_loco_parse.py:23` `H_THRESHOLD = 5.0`. 우리 `heat_threshold=5.0`. 값 일치.
- 차이:
  1. **트리거 시점**: 업스트림은 **매 add_memory마다 heap 최상단(전역 최고열) 세그먼트**를 검사 (`memoryos.py:126-140`, eval `main_loco_parse.py:240-244`도 매 dialog마다). 우리는 **flush 배치에서 건드린 세그먼트만** insert 직후 검사 (`memoryos.py:157-158`).
  2. **분석 대상**: 업스트림은 `analyzed=False`인 page만 분석하고 전 page에 `analyzed=True` 마킹 (`memoryos.py:148, 207-209`). 우리는 analyzed 플래그가 없어 세그먼트 `content[:4000]` 전체를 매번 재분석 → 재승격 시 중복 fact 추출 가능.
  3. **리셋**: 업스트림 성공 시 `N_visit=0; L_interaction=0; last_visit_time=now` (`memoryos.py:211-215`; 논문은 L_interaction만 리셋 언급 — 코드가 더 공격적). 우리 `_promote_to_lpm`은 `n_visit=0, length=0` (`memoryos.py:181-182`) — 코드와 일치. 단 우리는 **LLM 호출 실패(result None)여도 먼저 리셋**해 해당 세그먼트 승격 기회가 소실됨(업스트림은 실패 시 리셋 없이 return, `memoryos.py:175-180`).
  4. 승격 cadence 체감 차: 업스트림은 L_interaction이 page 1개씩 증가해 ~4-5 page 축적 후 승격. 우리는 10-메시지 배치가 한 topic이면 length=10, heat≈11로 **생성 즉시 승격**되는 경우가 지배적.

### C5. LFU eviction — **REFUTED (라벨), 동작은 논문 준수 / 코드 비준수**

- 업스트림 코드의 eviction은 **진짜 LFU**: `mid_term.py:71-75` `lfu_sid = min(self.access_frequency, key=...)` — 검색 히트 횟수(`access_count_lfu`) 최소 세그먼트 제거. capacity 기본 **2000** (`memoryos.py:36`, eval도 2000).
- 논문 §3.3은 반대로 "segments with the **lowest heat** are evicted"라고 서술 — 코드와 논문이 서로 불일치(업스트림 자체 드리프트).
- 우리 `memoryos.py:161-165`: `min(self._heat, key=self._segment_heat)` — **최저 heat 제거**. 즉 **논문에는 충실, 공식 코드의 LFU와는 다름**. docs/10 "LFU" 표기는 코드 기준으로 부정확("lowest-heat eviction, 논문 준수/코드의 access-frequency LFU와 상이"로 교정 권고). 어차피 우리는 read-path 접근 카운트가 없어 진짜 LFU는 구현 불가능한 구조.
- **[신규] capacity 기본값 불일치**: 업스트림 2000 vs 우리 `mtm_capacity=200` (`memoryos.py:65`). docs/05조차 "capacity 10/2000/100"으로 스펙을 적어놓고 코드는 200. conv0 규모(세그먼트 ~수십)에선 실측 영향 없음이지만 스펙 문서와 코드가 어긋남.

### C6. Profile 갱신 cadence·프롬프트 — **구조 CONFIRMED(승격→LPM), 내용은 대폭 단순화 — docs/10의 "90-dim trait, agent persona 누락" 표기는 정확하나 불완전**

업스트림 승격 1회 = **병렬 LLM 2콜** (`memoryos.py:169-172`):
1. `gpt_user_profile_analysis` — **90차원 성격/관심 태그 프롬프트** (`prompts.py:91-168` "update the user profile based on the 90 personality preference dimensions… Dimension ( Level(High/Medium/Low) )")에 **기존 프로필 전문을 넣어 통합된 새 프로필 전문을 출력**, `update_user_profile(merge=False)`로 **문서 교체** (`memoryos.py:190`). 30자 미만이면 스킵 (`memoryos.py:186-188`).
2. `gpt_knowledge_extraction` — 【User Private Data】/【Assistant Knowledge】 2섹션 (`prompts.py:180-196`), 줄 단위로 분해해 User KB·Assistant LTM에 append. "None"/"- None" 줄 필터 (`memoryos.py:195-204`).
   User KB·Assistant KB는 **deque(maxlen=100) FIFO** (`long_term.py:18-19`; 논문 "fixed-size queue (i.e., 100), FIFO").

우리 구현 (`memoryos.py:177-197`): **단일 PROFILE_PROMPT 1콜**로 "profile_facts" 리스트를 뽑아 semantic 항목으로 ADD. 판정:
- "none"/"n/a" 필터는 업스트림 준수 (CONFIRMED, 주석의 long_term.py 대응도 맞음).
- docs/10에 이미 명기: 90-dim trait ✔, agent persona ✔.
- **미명기 3건**: ① 업스트림은 **통합 프로필 문서를 교체-갱신**(상태 유지형)인데 우리는 **원자적 fact append**(누적형) — 모순 fact 덮어쓰기가 없음. ② user KB **capacity 100 FIFO 부재** (우리 semantic은 무제한). ③ 병렬 2콜 → 1콜로 콜 수/프롬프트 구조가 다름 (assistant knowledge 채널 자체가 없으니 파생 결과이긴 함).

### C7. dialogue chain meta — **누락 CONFIRMED (docs/10 정확)**

업스트림: 매 page마다 continuity 판정 LLM콜 (`prompts.py:209-214` "Return ONLY 'true' or 'false'") + meta-summary 생성/전파 (`prompts.py:217-232`, `updater.py:130-151`, `_update_linked_pages_meta_info`), QA 시 "Conversation chain overview: {meta_info}"로 주입 (`memoryos.py:277`). 우리는 전무 — docs/10 표기와 일치. (참고: 이 기능은 page당 LLM 1–2콜을 추가하므로, 미구현이 docs/09의 "organizer 91콜" 저비용 수치를 낳은 주요 요인 중 하나. 비용 비교 각주 권고.)

---

## 2. 신규 발견 (docs/10 미기재 목록)

| # | 심각도 | 내용 |
|---|---|---|
| N1 | **높음** | **read-path heat 피드백 부재**: 검색 시 N_visit/last_visit_time/access_count 갱신(논문 §3.4 명시, `mid_term.py:346-351`)이 없어 heat가 length+recency로 퇴화. LFU도 구조적으로 불가능. |
| N2 | **높음** | **STM 방출 단위**: 업스트림 1-page FIFO 롤링(STM 상시 유지) vs 우리 10개 전량 flush(STM 소거). QA 시 최근-문맥 채널 소실과 직결. |
| N3 | 중간 | **keyword 파이프라인 전무**: multi-summary의 theme별 keywords 추출/저장이 없음 — Jaccard 항 부재의 근본 원인이며 복원 전제조건. |
| N4 | 중간 | **병합 시 세그먼트 요약 의미 변화**: 업스트림은 병합 시 기존 session summary·embedding **불변**(page만 append, `mid_term.py:268-273`); 우리는 summary를 content에 이어붙이고 `embedding_text=content[-2000:]`로 재임베딩 (`memoryos.py:137-146`) → 세그먼트 표류(drift) 특성이 다름. |
| N5 | 중간 | **승격 실패 시에도 heat 리셋** (`memoryos.py:181-183` 리셋이 `result is None` 검사보다 먼저) — 업스트림은 실패 시 리셋 없이 반환. |
| N6 | 낮음 | mtm_capacity 기본 200 vs 업스트림/docs05 스펙 2000 (C5). |
| N7 | 낮음 | 프롬프트 문구가 업스트림과 전면 상이(자체 작성): MULTI_SUMMARY의 "maximum of two themes" 제한, theme/keywords/content 포맷, 90-dim 목록, KNOWLEDGE_EXTRACTION 2섹션 포맷 모두 미이식. round-3에서 A-Mem/Nemori만 upstream-faithful화되었고 MemoryOS 프롬프트는 대상이 아니었음. |
| N8 | 낮음 | docstring의 "issue #65 silent data loss" 서술은 이미 업스트림에서 수정된 상태(`memoryos.py:240-246`) — 현행 대비 우위 주장으로는 낡음. |
| N9 | 참고 | 업스트림 quirk: multi-summary가 theme 2개를 내면 **같은 page 배치를 두 세션에 중복 삽입** (`updater.py:174-185`, 매 theme마다 `pages_to_insert=current_batch_pages`). 우리는 message_indexes로 분할 — 우리 쪽이 더 정합적이나 "충실 재현"은 아님(의도적 편차로 문서화 권고). |
| N10 | 참고 | 판별 드리프트: eval판 heat 상수(0.8/0.8/0.0001)·Dice keyword 항은 pypi판·논문과 다름. "코드 기본값 동일" 주장은 pypi/chromadb판 한정임을 docs에 명기. |

---

## 3. 검색(retrieval) 배선 대조

### 업스트림 QA 시점 (LoCoMo eval판, `main_loco_parse.py:257-285` + `retrieval_and_answer.py`)

| 채널 | 방식 | 개수 |
|---|---|---|
| STM | **질의 무관 전량 주입** (recency) | eval cap=1 → 마지막 QA쌍 (pypi판이면 10) |
| MTM | 2단계: 세션 요약 매칭(cos+keyword, 임계 0.1) → page cosine(임계 0.1) → 전역 heap top | **retrieval_queue=10 pages** (pypi 기본 7) + page별 meta_info 주입 |
| User KB | cosine top-k, 임계 0.1 | **top_k=10** (`long_term_memory.py:86`; 논문 §3.4 "top-10 entries"; pypi판은 top-20/임계 0.01) |
| User Profile | **검색 없이 전문 무조건 주입** (`main_loco_parse.py:95-98`) | 1 문서 |
| Assistant Knowledge | eval판 **전량 주입** (`get_assistant_knowledge`, `main_loco_parse.py:103-106`; pypi판은 top-20 검색) | 전부 |

부가: 검색 성공 시 N_visit/heat 갱신(위 N1), 프롬프트에 speaker 치환(`re.sub('user'→speaker_a)`) 등.

### 우리 exp config (`scripts/exp_locomo_conv0.py:142`)

`"memoryos": (["memoryos"], ("episodic", "pages", "semantic"), 10, False, None, None)` — 세 채널 각 k=10, dense(+episodic만 lexical) RRF, budget 6000 tokens.

대응 관계 평가:
- **episodic k=10 ≈ MTM page 검색(queue 10)**: 개수 일치. 단 우리는 1단계 flat 검색(전체 raw 메시지 대상)이고 업스트림은 세션→page 2단계 + meta_info 부착. raw 대화 텍스트가 컨텍스트에 들어간다는 점에서 기능적 등가에 가까움. A-Mem/Nemori에서 "raw episodic 혼입"으로 순수화했던 것과 달리, MemoryOS는 **업스트림도 QA 시 raw page 텍스트를 주입**하므로 episodic 채널 유지가 방법론 오염이 아님 — 단 이 논거를 docs에 명시할 것.
- **semantic k=10 ≈ User KB top-10**: 개수·방식(cosine) 일치 (논문·eval판 기준 정확 일치. pypi판 20과는 다름).
- **pages(세그먼트 요약) k=10**: **업스트림에 없는 채널.** 업스트림은 세그먼트 요약을 1단계 매칭에만 쓰고 프롬프트에 주입하지 않는다. 우리는 요약을 직접 주입 — 정보 손실 보상 측면의 의도적 편차로 볼 수 있으나 미문서화.
- **누락 채널 3종**: ① User Profile 전문 무조건 주입(우리는 통합 프로필 자체가 없음 — C6), ② Assistant Knowledge, ③ **STM 최근-문맥 무조건 주입**(N2로 인해 구조적으로 불가; LoCoMo 후반 세션 질문엔 episodic 유사도 검색이 부분 보상하지만 recency 채널은 아님).
- 승격 즉시성(C4-4) 때문에 실제로는 대부분의 세그먼트가 빠르게 semantic으로도 복제되어, 채널 간 중복이 업스트림보다 큼.

**결론**: k=10·User-KB top-10은 업스트림 eval과 맞지만, "프로필 전문 주입 + STM recency 주입 부재"는 MemoryOS의 personalization 주장과 직결되는 read-path 결손이다. docs/09 캐비앗을 "F_score Jaccard 항 부재"에서 "**Jaccard 항 + profile 전문/STM recency 주입 + read-path heat 피드백 부재**"로 확장해야 수치 해석이 공정하다.

---

## 4. 권고

1. **docs/10 MemoryOS 행 교정** — 누락란을 다음으로 확장: "F_score Jaccard(keyword 파이프라인 전무), dialogue chain meta, 90-dim trait/프로필 문서 교체형 갱신, agent persona, **read-path heat 피드백(N_visit)**, **STM recency 주입**, user-KB cap 100". 등급 ◑ 유지는 타당하나 근거 목록이 현재보다 넓어야 함.
2. **docs/09 캐비앗 확장** (§3 결론 문구). 아울러 "LFU" 표기를 "lowest-heat eviction(논문 준수, 코드의 access-count LFU와 상이)"로 정정.
3. 충실도 개선 우선순위 (P0→P2):
   - P0: keyword 추출(topic 스키마에 keywords 추가) + F_score = cos + Jaccard, 임계 0.6 그대로 — 저비용(추가 LLM콜 0).
   - P0: QA-time 프로필 주입 — semantic(kind=profile) 상위 항목 또는 통합 프로필 문서를 무조건 컨텍스트에 포함하는 채널.
   - P1: read-path 훅으로 N_visit/last_access 갱신(파이프라인이 organizer에 조회 결과를 통지하는 경로 필요 — 현 구조상 설계 변경).
   - P1: STM 방출을 1-page FIFO 롤링으로 변경하거나, 롤링 유지가 어려우면 최근 n 메시지 recency 채널을 검색에 추가.
   - P2: 승격 실패 시 리셋 순서 수정(`result is None`이면 리셋하지 않음 — 1줄 이동), mtm_capacity 200→2000, analyzed 플래그 도입.
4. docstring 정리: issue #65 서술을 "업스트림도 이후 수정(add 전 migrate)"으로 갱신, "code defaults: capacity 10, θ=0.6, τ=5"에 "mtm 2000(우리 200), heat 상수는 pypi판 기준(eval판 0.8/0.8/1e-4)" 각주.
5. 4-way 수치 재해석: 다른 방법론과 달리 MemoryOS는 raw episodic 채널이 오염이 아니므로 config 순수화 재측정은 불필요. 다만 프로필 주입 부재는 single-hop/open-domain 카테고리에 불리하게 작용했을 수 있음 — P0 수정 후 재측정 대상.
