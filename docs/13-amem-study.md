# A-Mem 스터디 가이드 — 논문·공식 코드·우리 구현 (2026-07-21)

> A-MEM: Agentic Memory for LLM Agents (arXiv:2502.12110, **NeurIPS 2025**).
> 저자: Wujiang Xu(1저자) … Yongfeng Zhang(마지막 저자), Rutgers + AIOS Foundation.
> 이 문서는 docs/11(Nemori 스터디)의 자매편. 근거: 논문 §3/App B/§4 verbatim 재독
> (2026-07-21) + 공식 3벌 원문 정독 + `src/agmem/organizers/amem.py`,
> `src/agmem/retrieval/pipeline.py`. 선행 포렌식: docs/08, docs/10,
> docs/research/fidelity-round3(§1)·round4(§1).

---

## 1. 논문이 풀려는 문제

기존 메모리 시스템은 저장·검색은 되지만 **조직화(organization)가 고정적**이다 — graph
DB조차 사전 정의 스키마에 종속. A-Mem의 해법은 **Zettelkasten**(원자적 노트 + 자유 연결):
LLM이 스스로 노트를 만들고, 이웃과 연결하고, 이웃을 갱신하게 한다. "predetermined
operations 없이 long-term memory 유지."

핵심 통찰 3가지:
1. **차별화는 전부 write-path에 있다.** read는 순수 dense top-k(+링크 확장), LLM 0회.
2. **메타데이터-풍부 임베딩**(식3)이 검색 품질을 견인.
3. **링크 생성이 이득의 대부분**, evolution은 보정 — ablation이 뒷받침(§4).

---

## 2. 논문 formalization — 식(1)–(10) verbatim

메모리 노트 컬렉션 `ℳ = {m₁,…,m_N}`. 각 노트:

**(1)** `m_i = {c_i, t_i, K_i, G_i, X_i, e_i, L_i}`
- `c_i` 원본 상호작용 내용, `t_i` 타임스탬프, `K_i` LLM 키워드, `G_i` LLM 태그,
  `X_i` LLM 맥락 서술, `e_i` 임베딩, `L_i` 연결된 메모리 집합.

### Ps1 — Note Construction (§3.1)
**(2)** `K_i, G_i, X_i ← LLM(c_i ‖ t_i ‖ P_s1)` — 키워드/태그/맥락을 LLM이 생성.
**(3)** `e_i = f_enc[concat(c_i, K_i, G_i, X_i)]` — **content+keywords+tags+context 연결
임베딩**. `f_enc` = all-MiniLM-L6-v2. (헤드라인 공식. agiresearch 라이브러리판은 이걸
**위반**하고 content만 임베딩 — 아래 §3.)

### Ps2 — Link Generation (§3.2)
**(4)** `s_{n,j} = (e_n·e_j)/(|e_n||e_j|)` — 코사인 유사도.
**(5)** `ℳ_near^n = {m_j | rank(s_{n,j}) ≤ k, m_j ∈ ℳ}` — top-k 이웃.
**(6)** `L_i ← LLM(m_n ‖ ℳ_near^n ‖ P_s2)` — LLM이 이웃 중 연결 판단. `L_i = {m_i,…,m_k}`.

### Ps3 — Memory Evolution (§3.3)
**(7)** `m_j* ← LLM(m_n ‖ ℳ_near^n\m_j ‖ m_j ‖ P_s3)` — 새 노트가 이웃 `m_j`를 진화.
"The evolved memory m_j* **then replaces** the original memory m_j." (컬렉션에서 교체.)

### Read — Retrieve Relative Memory (§3.4)
**(8)** `e_q = f_enc(q)`  **(9)** `s_{q,i} = (e_q·e_i)/(|e_q||e_i|)`
**(10)** `ℳ_retrieved = {m_i | rank(s_{q,i}) ≤ k}` — dense top-k. **LLM 호출 없음.**
Fig.2 캡션: "linked memories … are also **automatically accessed**" = 1-hop 링크 확장.

### 논문 프롬프트 (App B, verbatim 핵심)
- **B.1 (P_s1)**: keywords "focus on nouns, verbs, key concepts / most→least important /
  exclude speaker name·time / ≥3, don't be redundant"; context "one sentence: main
  topic·key points·intended audience"; tags "broad categories: domain·format·type / ≥3".
- **B.2–B.3 (P_s2/P_s3, 결합 프롬프트)** JSON 스키마:
  ```json
  {"should_evolve": true/false,
   "actions": ["strengthen", "merge", "prune"],
   "suggested_connections": ["neighbor_memory_ids"],
   "tags_to_update": ["tag_1", ...],
   "new_context_neighborhood": ["new context", ...],
   "new_tags_neighborhood": [["tag_1", ...], ...]}
  ```
  → 논문 actions 어휘는 **strengthen / merge / prune**. 이웃 갱신은 별도 필드
  `new_context_neighborhood`(이웃 전체 배열 positional) + `new_tags_neighborhood`.

### 실험 셋업·ablation (§4)
- 백본 6종: GPT-4o-mini/4o, Qwen2.5-1.5b/3b, Llama3.2-1b/3b. **최소 1B — sub-1B 없음.**
- 임베더 all-minilm-l6-v2 전 실험 공통. **QA k=10** ("we primarily employ k=10").
- LoCoMo 7,512 QA, 5카테고리, 대화 평균 9K 토큰·최대 35세션.
- **Ablation (Table 3, GPT-4o-mini, F1):**

  | 조건 | Multi | Temporal | Open | Single | Adversarial |
  |---|---|---|---|---|---|
  | w/o LG & ME | 9.65 | 24.55 | 7.77 | 13.28 | 15.32 |
  | w/o ME (LG만) | 21.35 | 31.24 | 10.13 | 39.17 | 44.16 |
  | **Full** | **27.02** | **45.85** | **12.14** | **44.65** | **50.03** |

  읽는 법: **Link Generation이 대부분** (single 13.28→39.17 **+25.9**, adversarial
  +28.8, multi +11.7). **Memory Evolution의 최대 기여는 temporal** (31.24→45.85
  **+14.6**), multi는 +5.67. 논문 본문도 LG="critical foundation",
  ME="essential refinements"로 델타와 정합.
- **k-sweep (Table 8)**: 카테고리별 k를 10~50에서 튜닝 (GPT-4o-mini: multi=40,
  temporal=40, open=50, single=50, adversarial=40). LoCoMo는 train/test 분리가 없으므로
  **eval-set에 하이퍼파라미터를 맞춘 것** — 우리는 k=10 고정(오염 없음).

---

## 3. 공식 코드 아키텍처 (3벌 체제 — 혼동 주의)

**저자 2명이 각자 냈다.** 1저자 개인 2벌 + 마지막 저자 랩 org(AIOS) 1벌:

| repo | 소유 | 정체 | 재현 적합성 |
|---|---|---|---|
| `WujiangXu/A-mem` (구 `AgenticMemory`) | 1저자 개인 | **논문 재현 전용**. 논문 수치를 낸 판본 | ✅ robust 경로 |
| `WujiangXu/A-mem-sys` | 1저자 개인 | 시스템판(openai/ollama/sglang), 최소 버그 | ◑ 프로덕션(#23/#24 잔존) |
| `agiresearch/A-mem` | 마지막 저자 org(AIOS) | 라이브러리판(ChromaDB). **arXiv 공식 링크** | ❌ 3버그+Ps1 dead |

**어느 판이 실제로 뭘 하나 (원문 정독 결과):**
- **논문 LoCoMo 수치 = `WujiangXu/A-mem`의 robust 경로** (`memory_layer_robust.py` +
  `test_advanced_robust.py`). 저장소는 **ChromaDB가 아니라 in-memory cosine
  `SimpleEmbeddingRetriever` + BM25 hybrid**. → **#23/#24(ChromaDB L2/distance 버그)는
  논문 수치에 무관** (그 버그는 agiresearch/A-mem-sys 계보에만 존재).
- plain `memory_layer.py`는 `import re` 누락으로 `analyze_content`가 매번 NameError →
  메타데이터 전량 폴백(빈 keywords/tags, context="General"). **robust판만 정상.**
- `agiresearch/A-mem`: `add_note`가 `analyze_content`를 **아예 호출 안 함**(Ps1 사문,
  `memory_system.py:233-264`) + `find_related_memories`가 ID 아닌 **순위 index** 반환
  (`:308`, 이슈 #32) + ChromaDB L2(#24) + score=distance(#23). eval read는
  `find_related_memories_raw`(1-hop, per-hit 캡 → k=10이면 이웃 ~100개).
- `A-mem-sys`: **#32·Ps1 수정됨**(실 doc_id 반환, Ps1 live), 단 #23/#24 잔존.

이슈 #23/#24/#32는 2026-07 현재 **전부 open**(#32는 2026-07-13 신규 등록 — 우리 round-3
발견을 사후 코러보).

---

## 4. 우리 구현 워크스루

### 4.1 write 경로 — `AMemOrganizer` (`organizers/amem.py`)

메시지당 파이프라인 (`_ingest`, `amem.py:188`):
```
1. Ps1 노트 구성      (LLM 1콜, NOTE_PROMPT → keywords/context/tags JSON)   amem.py:202-217
2. 이웃 top-k 검색     (식3 metadata-concat 임베딩 질의, store계층 cosine)   amem.py:221-226
3. Ps2+Ps3 배치 진화   (LLM 1콜, 이웃 전체 한 프롬프트 — EVOLVE_PROMPT)      amem.py:237-248
  → [ADD(note), LINK(단방향), UPDATE(이웃 context/tags 재임베딩)]           amem.py:262-325
```
스토리지를 직접 만지지 않고 **`MemoryOp(ADD/LINK/UPDATE/INVALIDATE)` 리스트를 반환** →
append-only 로그로 **replay·audit 가능**. 원논문의 "evolution이 뭘 바꿨는지 추적 불가"
문제를 구조적으로 해소 (원논문은 식7에서 m_j를 in-place replace).

### 4.2 read 경로 — `RetrievalPipeline._expand_links` (`retrieval/pipeline.py:181`)
dense top-k(식10) → 검색된 노트의 링크 이웃 1-hop 확장(`find_related_memories_raw` 대응).
링크는 단방향(upstream 정합). **캡 의미 편차**: upstream은 per-hit, 우리는 **전역 cap=5**
(`pipeline.py:181-210`).

### 4.3 우리 프롬프트 2종의 계보
- `NOTE_PROMPT` (`amem.py:77`): B.1 의미충실 축약("nouns/verbs" 초점,
  speaker/time 제외, ≥3). "intended audience" 문구는 누락(영향 낮음).
- `EVOLVE_PROMPT` (`amem.py:89`): B.2–B.3 결합 프롬프트를 배치 1콜로. **actions 어휘를
  논문 strengthen/merge/prune → 우리 `strengthen`/`update_neighbor`로 재매핑**
  (merge/prune는 우리 UPDATE/INVALIDATE MemoryOp로 흡수). 이웃 갱신을 논문의 positional
  `new_context_neighborhood` 배열 대신 **ID 기반 `neighbor_updates`**로(#32 수정의 귀결).

---

## 5. 논문/upstream과의 잔여 편차 (전부 의도적·문서화)

| 편차 | 원본 | 우리 | 근거 |
|---|---|---|---|
| 이웃 참조 | agiresearch: 순위 index(#32) | **ID 기반** + 환각 ID 필터 | 버그픽스, 테스트 고정 |
| 유사도 | 라이브러리판: L2(#24)/score반전(#23) | store계층 **cosine 보장** | 버그픽스 |
| Ps1 실행 | agiresearch: 미호출(사문) | 항상 1콜 실행 | 버그픽스 |
| 이웃 검색 질의 | 코드 양쪽: `note.content`만 | **enriched embedding_text**(식3 충실) | 논문충실/코드비충실 |
| evolution 실패 | 광역 try/except silent skip | 명시적 **drop 카운터** + 로그 | 0.5B 방어 |
| 링크 방향 | upstream 단방향 | 단방향(정합) | round-3 수정 |
| 링크 확장 캡 | upstream per-hit(~k²) | **전역 cap=5** | pipeline docstring |
| actions 어휘 | 논문 strengthen/merge/prune | strengthen/update_neighbor(+MemoryOp) | §4.3 |
| 빈 actions | upstream no-op | **양 효과 폴백**(소형모델) | 우리 고유 |
| in-place replace | 식7 교체 | **MemoryOp 반환**(replay 가능) | 설계 차별점 |

**미감사 잔여** ⚠️: `AMemOrganizer(input="episodes")` chained-manager 모드
(`amem.py:127-186`, config `nemori_amem`)는 round-3/4 포렌식 **이후** 추가(commit
e44e300). 논문/공식코드에 없는 우리 확장 — 다음 감사에서 편차 항목화 필요.

---

## 6. 결과 해석 캐비앗

1. **0.6B 로컬 LLM은 논문 검증 범위(최소 1B) 아래** → 절대치 비교 불가, 상대 비교만.
   링크 품질이 organizer 모델 능력에 종속(논문 ablation의 백본 의존성과 정합).
2. **논문 수치를 "재현"으로 인용 금지**: 논문 수치는 in-memory cosine+BM25 hybrid 경로
   산출. 우리 `amem` config는 notes dense-only(+keyword_queries) — read 채널이 다름.
   (`exp:208 AMEM_STORE=ChromaVectorStore`는 "라이브러리판 저장소 계보"이지 논문 수치
   계보가 아님 — docs/03 §5 전제 정정 대상.)
3. **강한 baseline 중요**: passthrough(raw + hybrid BM25+dense+RRF)만으로 F1 22.85 —
   많은 메모리 논문이 약한 naive-RAG와 비교한다는 비판적 관점.

---

## 7. 실행

```bash
bash scripts/serve-llm.sh                 # 로컬 LLM 서버
uv run python scripts/exp_locomo_conv0.py --configs passthrough amem
# amem = notes-only, k=10, keyword_queries, write온도 0.7/0.7, Chroma(cosine)
# amem_mixed = raw episodic 혼입(ablation용, 논문 재현 아님)
# nemori_amem = Nemori 에피소드 → A-Mem 노트 체이닝(미감사 모드)
```
테스트: `tests/test_organizers.py` — `test_amem_note_link_and_evolution`,
`test_amem_hallucinated_neighbor_ids_ignored`, `test_amem_degrades_without_llm`.

---

## 8. 더 읽기
- 발표용 리뷰: docs/08 / 충실도 등급표: docs/10
- 포렌식 원장: docs/research/fidelity-round3(§1) · round4(§1)
- 코드: `src/agmem/organizers/amem.py` · `src/agmem/retrieval/pipeline.py:181`
