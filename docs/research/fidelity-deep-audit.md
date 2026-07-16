# 충실도 심층 감사: A-Mem · Nemori · ReasoningBank (라인 단위 대조)

> 작성: 2026-07-16. 측정 재개 전 필수 감사.
> 대조 소스 (모두 **오늘자 upstream main 브랜치 실제 소스**를 raw로 내려받아 확인):
>
> | 시스템 | 논문 | 공식 코드 (확인 시점 main) |
> |---|---|---|
> | A-Mem | arXiv:2502.12110 v11 (NeurIPS'25) | github.com/agiresearch/A-mem (라이브러리판, pushed 2025-12-12) + github.com/WujiangXu/A-mem (**논문 재현판**, pushed 2026-03-05) |
> | Nemori | arXiv:2508.03341 v1 + v4(ACL 형식화) | github.com/nemori-ai/nemori (pushed 2026-04-16) |
> | ReasoningBank | arXiv:2509.25140 | github.com/google-research/reasoning-bank (pushed 2026-07-10) |
>
> 우리 측: `src/agmem/organizers/{amem,nemori,reasoning_bank}.py`, `src/agmem/retrieval/pipeline.py`,
> `src/agmem/core/types.py`, `src/agmem/memory.py`, `src/agmem/bench/locomo.py`, `scripts/exp_locomo_conv0.py`.
>
> 판정 기호: **일치**(의미 동일) / **단순화**(축약이나 의미 보존) / **누락**(기능 부재) / **변형**(다른 동작).
> 영향 등급: 낮음 / 중간 / **높음**.

핵심 결론 요약 (상세는 각 절):

1. **A-Mem: write 경로는 충실, read 경로가 불충실.** 우리가 놓친 것은 (a) **질의 시 1-hop 링크 확장**(공식 eval의 `find_related_memories_raw`가 하는 일이자 논문 Fig.2 캡션의 "linked memories automatically accessed"), (b) 공식 eval의 **LLM 키워드 질의 생성**, (c) 검색 컨텍스트에 context/keywords/tags 메타데이터 미표시. 링크를 만들기만 하고 읽을 때 안 쓰므로 **Link Generation 모듈의 효용이 벤치마크에서 실현되지 않음** (논문 ablation에서 LG 제거가 가장 큰 하락 요인).
2. **Nemori: 조직화 코어는 충실, QA 검색 설정이 불충실.** (a) semantic top-`m=2k`(=20) 대신 10, (b) **top-2 에피소드에 원문 메시지 첨부(r=2) 누락**, (c) 1600토큰 렌더 예산이 upstream 대비 크게 작아 검색 결과 대부분이 잘림, (d) episode merging 누락(v4 논문에서는 정식 모듈), (e) cold-start 직접 추출 경로 누락.
3. **ReasoningBank: 코어 일치, 검색 단위와 온도 설정이 다름.** upstream은 "현재 질의 ↔ **과거 태스크 질의**" 유사도로 top-1 **경험**을 골라 그 경험의 아이템(≤3개)을 전부 주입. 우리는 아이템 단위 검색. 실험 스크립트가 모든 role을 t=0.1로 통일해 논문의 judge 0.0/추출 1.0 분리가 실측에서 실현되지 않음.
4. **우리 버그픽스 3건은 모두 upstream에서 오늘도 open이며 방향이 맞다.** 단 #24(L2)는 "랭킹 붕괴"가 아니라 "score 의미 반전"이 정확한 서술이고, #32는 라이브러리판(agiresearch)에만 실존하며 **논문 재현판(WujiangXu)에서는 인덱스 체계가 자기일관적이라 버그가 아니다** — 문서 서술을 교정할 것. 추가로 WujiangXu 재현판의 신규 결함(`import re` 부재로 Ps1 메타데이터가 사실상 전부 빈 값 폴백)을 발견함.

---

## 1. A-Mem (arXiv:2502.12110 v11)

### 1.1 파이프라인 단계별 대조 — write 경로

| 단계 | 논문/공식코드 동작 | 우리 구현 동작 | 판정 | 벤치마크 영향 |
|---|---|---|---|---|
| **Ps1 노트 구성** | 논문 Appendix B.1: keywords(중요도순, 화자·시간 제외, ≥3) / context(1문장: topic·key points·audience) / tags(domain·format·type, ≥3). 코드 동일: agiresearch `memory_system.py:159-231 analyze_content()`, WujiangXu `memory_layer.py:308-401` | `amem.py:58-65 NOTE_PROMPT` — 세 필드·화자/시간 제외·≥3·중요도순 모두 유지, 문구만 축약 | **단순화** | 낮음 |
| **category 속성** | 코드에 `category`(기본 "Uncategorized") 존재하나 LLM이 생성하지 않고 프롬프트/스키마에도 없음 (`memory_system.py:71`) — 사실상 사문 | `Note`에 category 없음 (`core/types.py:56-72`) | **일치**(사문 속성 미이식) | 없음 |
| **임베딩 구성 e=concat** | 논문 식(3): `e_i = f_enc[concat(c_i, K_i, G_i, X_i)]` = content, keywords, tags, context 순. **코드는 리포마다 다름**: WujiangXu `memory_layer.py:722` `"content:"+content+" context:"+context+" keywords: "+... +" tags: "+...` (라벨 포함, 순서 c,X,K,G); agiresearch `memory_system.py:258`+`retrievers.py:63-83`은 **content만** 임베딩(메타데이터는 Chroma metadata로만 저장 — 논문 식(3) 위반) | `core/types.py:69-72 Note.embedding_text()` — content, keywords, tags, context를 `" \n"`으로 연결. **논문 식(3)의 순서와 정확히 일치**, 라벨 없음 | **일치**(논문 기준; upstream 두 코드끼리도 서로 불일치) | 낮음 (라벨 유무는 임베딩 모델에 미미) |
| **이웃 검색 (진화용)** | k=5 고정: agiresearch `memory_system.py:605`, WujiangXu `memory_layer.py:755` — 둘 다 질의 텍스트는 **note.content만** (`find_related_memories(note.content, k=5)`) | `amem.py:115-117` k=5(top_k), 질의 벡터는 `note.embedding_text()`(메타데이터 포함) | **변형**(질의에 메타데이터 포함 — 아마 우세하나 미검증) | 낮음 |
| **Ps2 링크 생성 / Ps3 진화 — 콜 구조** | 논문은 식(6) Ps2와 식(7) Ps3을 개념상 분리하나, **Ps2 프롬프트(App B.2)는 Ps3(App B.3)의 접두부이며 출력 포맷조차 없음**. 공식 코드 양쪽 모두 **단일 process_memory 콜 1회**로 링크+진화를 함께 결정 (agiresearch `memory_system.py:590-727`, WujiangXu `memory_layer.py:753-857`). WujiangXu robust판만 3콜로 분리 (`memory_layer_robust.py:463-536`) | `amem.py:69-85 EVOLVE_PROMPT` 단일 콜 | **일치**(공식 코드 기준. "Ps2+Ps3 병합"은 우리가 만든 게 아니라 upstream 원형) | 없음 |
| **strengthen 액션** | `suggested_connections`를 **새 노트의 links에 추가**(단방향) + **`tags_to_update`로 새 노트의 tags 교체** (agiresearch `memory_system.py:679-683`, WujiangXu `memory_layer.py:832-836`) | `amem.py:139-145` connections → 새 노트 LINK + **역방향 LINK도 추가**(양방향). **새 노트 tags 갱신(`tags_to_update`) 미적용** | **부분 누락 + 변형** | 중간: tags는 임베딩 텍스트 성분이므로 새 노트의 검색 표현이 진화하지 않음. 양방향 링크는 upstream에 없는 확장(1-hop 확장 구현 시 회수 범위가 넓어짐 — 명시할 것) |
| **update_neighbor 액션** | 이웃 순서대로 positional 배열(`new_context_neighborhood`, `new_tags_neighborhood`)로 context/tags 교체. keywords는 불변 (양 리포 동일) | `amem.py:147-164` ID 기반 `neighbor_updates`(id, new_context, new_tags), keywords 보존, 갱신 시 embedding_text 재계산 | **변형**(ID 기반 재설계 — §1.3 버그픽스 검증 참조) | 낮음 (의미 보존) |
| **진화 결과의 인덱스 반영** | **양 리포 모두 이웃 갱신이 검색 인덱스에 즉시 반영되지 않음** — 객체(dict)만 수정. `consolidate_memories()`(evo 100회마다 인덱스 전체 재구축, agiresearch `memory_system.py:260-286`)가 유일한 재동기화 지점 | `memory.py:219-232 _apply_one()` UPDATE 시 즉시 재임베딩·재색인 | **변형**(논문 식(7) "m_j*가 m_j를 교체"의 의미를 우리가 더 정확히 구현) | 없음~우세 |
| **consolidation (evo_threshold=100)** | 위와 같음 — "통합"이 아니라 **stale 인덱스 재빌드 장치**. 메모리 병합/삭제 없음 | 없음 | **누락이나 불필요** — 우리는 즉시 재색인이라 재빌드할 stale 상태가 없음 | 없음 |
| **진화 실패 처리** | 예외 → `(False, note)` 반환, 카운트/로그 없이 계속 (agiresearch `memory_system.py:720-727`) | `amem.py:135-136` None → drop 카운터 (`llm/structured.py:65-71,108`) + 노트는 저장 | **변형(개선)** | 없음 (관측성 향상) |

### 1.2 파이프라인 단계별 대조 — read(QA) 경로 ← **주요 격차**

공식 LoCoMo eval은 WujiangXu `test_advanced.py`의 `advancedMemAgent`가 기준.

| 단계 | 논문/공식코드 동작 | 우리 구현 동작 | 판정 | 벤치마크 영향 |
|---|---|---|---|---|
| **질의 생성** | `test_advanced.py:95-129`: 질문 → **LLM이 키워드 생성**(`generate_query_llm`) → 키워드 문자열로 검색 | `bench/locomo.py:124-127` 질문 원문으로 검색 | **누락** | 중간: 키워드화는 노이즈 제거+임베딩 정합 목적. 질문 원문 검색은 단문 질의라 e5 계열에선 무난하나 공식 절차와 다름 |
| **검색 대상/방식** | notes 컬렉션 단일, cosine top-k (retrieve_k, 스윕; 논문 4.2는 k=10 기본). RRF/lexical 없음 | `exp_locomo_conv0.py:93` `("episodic","notes")` 각 k=10, dense+RRF(+episodic은 FTS5 lexical 융합, `pipeline.py:34-58`) | **변형** | 중간: **raw episodic 스트림이 함께 검색되어 A-Mem 단독 성능이 아니라 "passthrough+notes" 혼합 성능이 측정됨.** 방법론 비교 목적이면 notes 단독 조건도 필요 |
| **1-hop 링크 확장** | `find_related_memories_raw()` — top-k 각 hit의 `links` 이웃을 컨텍스트에 이어붙임 (WujiangXu `memory_layer.py:877-898`, agiresearch `memory_system.py:315-344`). 논문 Fig.2 캡션: "When related memory is retrieved, similar memories that are linked within the same box are also automatically accessed" | **없음** — `pipeline.py`는 links를 전혀 따라가지 않음. LINK op는 저장만 되고 (`memory.py:239-246`) 읽기에서 소비되지 않음 | **누락 (최대 격차)** | **높음**: 링크 생성(write 2번째 LLM 콜)의 비용은 내면서 효용은 0. 논문 ablation(w/o LG가 최대 하락)이 주장하는 핵심 이득이 우리 측정에 반영 안 됨. 특히 multi-hop 카테고리 직결 |
| **컨텍스트 포맷** | `talk start time:{ts}\t memory content:...\t memory context:...\t memory keywords:...\t memory tags:...` — 메타데이터 전부 노출 (`memory_layer.py:891`) | `MemoryBundle.render()`(`core/types.py:190-203`) + `_DictItem.render()`(`pipeline.py:80-89`) — **content만** (title 있으면 title:content). context/keywords/tags 미표시. timestamp는 LoCoMo ingest가 content에 "(date) speaker:" 접두를 넣어(`bench/locomo.py:117`) 간접 포함 | **누락(부분)** | 중간: context 문장은 답변 근거로 쓰일 수 있는 LLM 생성 요약. temporal은 date 접두 덕에 방어됨 |
| **답변 프롬프트** | 카테고리별 분기 (`test_advanced.py:144-196`): cat2 "Use DATE... approximate date", cat3/4 "exact words from the context", **cat5는 gold 답과 'Not mentioned...'를 이지선다로 제시**(+전용 temperature) | 단일 프롬프트 (`bench/locomo.py:23-32`) — 최단 span, 없으면 "No information available" | **변형** | 중간(비교 프레임): cat5 이지선다는 gold 노출이라 upstream에 유리한 세팅. **우리 방식이 더 엄격** — 절대치 비교 시 캐비앗 필수, 재현이 목적이면 옵션으로 분리 |
| **top-k (QA)** | 논문 4.2: k=10 기본(모델별 카테고리 조정 표 App A.5) | k=10 | **일치** | — |

### 1.3 우리 버그픽스 3건 재검증 (upstream 오늘자 기준)

| 픽스 | 우리 주장 (docs/08:35-74, amem.py:7-13) | 오늘자 upstream 사실관계 | 재판정 |
|---|---|---|---|
| **#24/#23 L2↔cosine** | "reference ChromaDB collection used L2 while treating scores as similarity" | agiresearch `retrievers.py:59-61` — `get_or_create_collection`에 `hnsw:space` 미지정 → **Chroma 기본 L2 그대로, issue #24/#23 여전히 open**. 단 정밀하게는: Chroma가 거리 오름차순으로 정렬해 주므로 **top-k 랭킹 자체는 붕괴하지 않고**, `score` 필드에 거리가 담겨 의미가 반전된 것(#23)이 실체이며 그 score는 eval에서 재정렬에 쓰이지 않음. **논문 수치를 낸 WujiangXu판은 처음부터 sklearn cosine** (`memory_layer.py:605`) | **방향 맞음, 서술 교정 필요**: "검색 품질 저하"가 아니라 "score 의미 반전(라이브러리판 한정), 논문 재현판은 무관". 우리 cosine 보장(`stores/sqlite_vec.py:39`, `numpy_vec.py:73`)은 논문 식(4)와 일치 — 유지 |
| **#32 인덱스 vs ID** | "neighbors are addressed by note ID, not result-list index" | **agiresearch판에 실존, open**: `memory_system.py:288-313`이 순위 인덱스(0..k-1)를 반환하고 `process_memory():687-716`이 `noteslist=list(self.memories.values())`에 그 인덱스로 접근 → **항상 가장 오래된 k개 노트를 갱신**하는 진짜 버그. 링크도 "memory index:i" 텍스트를 본 LLM이 0,1..을 반환 → `memories.get("0")` 실패로 링크 확장 무효. **그러나 WujiangXu 재현판은 버그 아님**: `SimpleEmbeddingRetriever.search()`가 corpus 인덱스를 반환하고 corpus 추가순 == memories 삽입순이라 `memory_layer.py:851-856`의 인덱스 접근이 자기일관적(링크도 인덱스로 일관, `find_related_memories_raw:893-894`도 인덱스 해석) | **방향 맞음, 범위 한정 필요**: "논문 수치가 이 버그 위에서 나왔다"고 말하면 과함 — 재현판은 정합. 라이브러리판 사용자에게만 실버그. 우리 ID 기반+환각 ID 필터(`amem.py:138-151`, 테스트 `test_organizers.py:133`)는 안전한 재설계 |
| **#10 silent skip** | "evolution failure is an explicit drop, never a silent skip" | 진화 실패 시 무기록 계속은 양 리포 모두 사실 (`memory_system.py:720-727` 등). issue #10(노트 구성 LLM 미호출)은 #13으로 부분 수정됐으나 **여전히 open**. **신규 발견**: WujiangXu `memory_layer.py`는 top-level에 `import re`가 없어(`1-19행 임포트 목록`) `analyze_content:380`의 `re.sub`가 NameError → except 경유로 **모든 노트가 빈 keywords/context/tags 폴백** — 현재 main 그대로 실행하면 Ps1이 사실상 무효(robust판 `memory_layer_robust.py:15`에서만 수정됨) | **맞음, 유지**. 우리 drop 카운터(`structured.py:108`, stats 노출 `memory.py:314`)는 순수 개선. WujiangXu 신규 결함은 "재현 시 robust판인지 확인" 체크리스트에 추가할 것 |

### 1.4 하이퍼파라미터 대조

| 항목 | 논문/코드 | 우리 | 판정 |
|---|---|---|---|
| 진화용 이웃 k | 5 (코드 하드코딩) | 5 (`amem.py:91`) | 일치 |
| QA top-k | 10 (논문 4.2) | 10 | 일치 |
| evo_threshold(재색인 주기) | 100 | 해당 없음(즉시 재색인) | 무영향 |
| 임베더 | all-MiniLM-L6-v2 | multilingual-e5-small (`exp_locomo_conv0.py:89`) | 변형 — 전 조건 공통이므로 상대 비교엔 무방, 절대치 비교 시 명기 |

---

## 2. Nemori (arXiv:2508.03341)

주의: v1(우리가 따른 형식화)과 v4(ACL 판)는 **방법론 자체가 개정**됨.
v4는 관측창(w) 단위 배치 분할(3.2.1), 에피소드 병합(3.2.3 Associative Memory Integration),
semantic 통합(new/merge/conflict, 3.3.3)을 정식 모듈로 승격 — 현 리포 구현과 일치.
우리 구현은 v1 형식화 기준. 아래 표는 양쪽 모두 표기.

### 2.1 경계 검출 (boundary alignment)

| 단계 | 논문/공식코드 | 우리 구현 | 판정 | 영향 |
|---|---|---|---|---|
| 트리거 형식 | **v1 §3.1.1**: 메시지마다 f_θ(m_{t+1}, M) → (b, c); T = (b ∧ c>σ_boundary) ∨ (\|M\|≥β_max); σ=0.7, β_max=25 (v1 §4 구현 세부) | `nemori.py:141-171` 메시지마다 LLM 콜, `boundary ∧ confidence≥0.7`, buffer_max=25 | **일치 (v1)** | — |
| 트리거 형식 (리포/v4) | 리포는 per-message 검출기가 **없음**: `core/memory_system.py:105-113` — 버퍼가 batch_threshold(20) 이상이면 `BATCH_SEGMENTATION_PROMPT`(1..N 메시지를 에피소드 그룹으로 일괄 분할), 미만이면 통짜 1그룹. v4 3.2.1도 관측창 w 채움 → 일괄 분할 | per-message | **변형 (리포/v4 대비)** — 의도적·문서화됨 (`nemori.py:14-17`) | 중간: per-message는 LLM 콜 수 ~N배. 분할 품질은 배치가 전역 문맥을 봐서 유리할 수 있음. 비용 비교표에 반드시 명기 |
| 경계 판단 기준 | `prompts.py:241-308 BATCH_SEGMENTATION`: ①topic change(최우선) ②intent transition ③temporal markers + **"time gap > 30 minutes"** ④structural signals ⑤relevance<30% / **"When in doubt, split"** / **"2-15 messages per episode"** | `nemori.py:65-77 BOUNDARY_PROMPT`: topic/intent/temporal marker/relatedness<30%는 있음. **30분 갭 규칙·"when in doubt split"·에피소드 길이 가이드 없음** | **단순화(부분 누락)** | 중간: LoCoMo는 세션 간 날짜가 크게 뛰므로 30분 규칙이 세션 경계를 기계적으로 잡아줌. 우리는 LLM이 "long time gaps" 문구만으로 추론해야 함 |
| 경계 시 버퍼 처리 | v1: M을 세그먼트로 넘기고 **m_{t+1}이 새 버퍼 시작** | `nemori.py:167-170` — 동일 (`buffer[:-1]` flush, 최신 메시지 잔류) | **일치** | — |
| β_max 도달 시 | v1 수식상 \|M\|≥β_max이면 M만 flush, m_{t+1}은 새 버퍼 | `nemori.py:153-155` — **최신 메시지 포함 전체 25개를 flush** | **변형(경미)** | 낮음 |
| 실패 처리 | (리포엔 해당 경로 없음) | verdict None → 경계 아님 취급 + drop 카운트 (`nemori.py:165-166`) | 개선 | — |

### 2.2 에피소드 생성 (representation alignment)

| 단계 | 논문/공식코드 | 우리 구현 | 판정 | 영향 |
|---|---|---|---|---|
| 서사 생성 | `prompts.py:11-53 EPISODE_GENERATION`: title(10-20단어, 검색성) / content(3인칭 서사, **모든 중요 정보**, 시간은 연·월·일·시 단위, **상대시간 → 절대날짜를 원문 뒤 괄호로**) / **timestamp 필드(ISO)** — `episode.py:64-69`에서 파싱해 `created_at`으로 사용, boundary_reason 주입 | `nemori.py:80-90 EPISODE_PROMPT`: title/narrative, 상대시간 절대화 지시(괄호 병기 방식은 아님) | **단순화 + timestamp 필드 누락** | 중간: upstream은 검색 결과를 `- [{created_at}] {content}`로 렌더 (`evaluation/locomo/search.py:83-85`) — 에피소드의 **사건 시각**이 컨텍스트에 노출됨. 우리는 narrative 내 날짜 문장에만 의존 |
| 임베딩 | `episode.py:60` `f"{title} {content}"` (v4 식: f_emb(c‖N)) | `nemori.py:207` `f"{title}\n{narrative}"` | **일치** | — |
| 생성 실패 폴백 | `episode.py:131-143` title="Conversation (N messages)", content=원문 | `nemori.py:194-199` title=첫 8단어, narrative=원문 | **일치(등가)** | — |
| **에피소드 병합** | 리포 기본 on (`config.py:85-87`, threshold 0.85, top-5 후보): 새 에피소드마다 유사 후보 검색 → LLM merge 판정(같은 사건·**>1시간 갭이면 금지**, `prompts.py:381-417`) → 병합문 생성·원본 2건 삭제·병합본 저장 (`memory_system.py:132-151`, `merger.py:37-67`). **v4 3.2.3에서 정식 모듈(K_e=5)** — 관측창 절단으로 쪼개진 같은 사건을 복원하는 장치 | 없음 (`nemori.py:18` 문서화) | **누락** | v1 기준 낮음(논문에 없음), **v4/리포 기준 중간**: LoCoMo는 한 세션이 여러 버퍼로 잘릴 때 같은 세션 사건이 중복 에피소드로 남아 검색 slot을 낭비. 배치 인제스트 벤치에서 실효 있음 |

### 2.3 predict-calibrate (semantic 생성)

| 단계 | 논문/공식코드 | 우리 구현 | 판정 | 영향 |
|---|---|---|---|---|
| 지식 검색(예측용) | v1 식: Retrieve(embed(ξ⊕ζ), K, m, σ_s); 리포 `memory_system.py:167-176` — **episode 임베딩**으로 semantic top-k(config `search_top_k_semantic`, 기본 10; v1 실험 σ_s=0.0) | `nemori.py:216-221` — episode(title+narrative) 임베딩으로 semantic top-10 | **일치** | — |
| **cold-start 분기** | `semantic.py:51-54`: 기존 semantic이 **없으면 predict-calibrate 대신 직접 추출**(`SEMANTIC_GENERATION_PROMPT`, 4 tests+고가치 카테고리) | 없음 — knowledge="(none)"으로 그대로 예측 (`nemori.py:221`) | **누락** | 낮음~중간: "(none)" 대비 gap 추출은 사실상 전체 추출과 유사하게 동작하나, 예측문이 환각 지식을 담으면 그만큼 gap이 줄어 초기 지식 형성이 빈약해질 수 있음 |
| 예측 | `prompts.py:57-89 PREDICTION_PROMPT`: 입력 = **title만** + 지식문장; "ACTUAL CONTENT and KNOWLEDGE... not the writing style"; 출력 자유 서술(JSON 아님, `semantic.py:87-91`) | `nemori.py:93-102 PREDICT_PROMPT`: title+지식, "Predict actual facts and knowledge, not writing style", 출력 JSON | **일치(핵심) / 단순화(포맷)** | 낮음 |
| **캘리브레이션 기준** | v1 §3.2.2: "ground truth is **not** the generated episodic narrative ζ, but the original, unprocessed Segmented Conversation M". 리포 `semantic.py:93-101` — `episode.source_messages` 원문 사용 | `nemori.py:229-233` — RAW `seg_text` 사용 | **일치** ✔ (사용자 우려 항목 — 우리가 올바르게 함) | — |
| 4-test 필터 | `EXTRACT_KNOWLEDGE_FROM_COMPARISON` (`prompts.py:93-156`): Persistence(6개월)/Specificity/Utility/Independence + 고가치 7카테고리·저가치 예시·**"DO NOT include time/date information in the statement"**·present tense·atomic | `nemori.py:105-121 CALIBRATE_PROMPT`: 4 tests 전부, present tense, atomic, **"no relative time expressions"**(절대시간은 허용 — upstream은 시간 자체 금지) | **일치(핵심) / 단순화**(카테고리·예시 생략, 시간 규칙 완화) | 낮음 (우리 완화가 temporal 카테고리엔 오히려 유리할 수도 — 단 upstream과 다름은 명기) |
| semantic 저장 | `semantic.py:59-71` 문장별 임베딩 + **memory_type 분류**(identity/preference/... `_classify_type:135-151`, 키워드 규칙) | `nemori.py:239-250` 문장별 ADD, 분류 없음 | **단순화** | 없음 (분류는 검색에 미사용) |
| semantic 통합(v4) | v4 3.3.3: new/merge/conflict 3분기(K_m=5, τ=0.70). **현 리포에는 미구현**(`semantic_similarity_threshold`가 config에만 있고 미사용 — append-only) | 없음 (append-only) | **일치(리포 기준) / 누락(v4 기준)** | v4 재현 목표 시 필요 |

### 2.4 QA 검색·답변 ← **주요 격차**

공식 LoCoMo eval: `evaluation/locomo/search.py`.

| 단계 | 논문/공식코드 | 우리 구현 | 판정 | 영향 |
|---|---|---|---|---|
| 검색량 | **episodic top-k=10 + semantic top-m=2k=20** (논문 v1 §4: "we retrieve top-k episodic and top-m=2k semantic... k=10 (thus m=20)"; eval 기본값 `search.py:218-219`) | `exp_locomo_conv0.py:94` `("episodic","episodes","semantic")` **각 k=10** | **변형** | 중간: semantic이 절반. 대신 raw episodic 10개가 추가로 섞임(방법론 순수성 훼손 — A-Mem과 동일 이슈) |
| **원문 첨부 (r=2)** | "only the top-2 episodic memories include their original conversation text" (논문 v1 §4, v4 §4.1 r=2); `search.py:77-94 format_memory_lines(include_original_limit=2)` — 상위 2개 에피소드 밑에 `Source Messages:` 원문 대화 나열 | **없음** — 에피소드는 narrative만 렌더 (`pipeline.py:80-89`) | **누락** | **높음**: 서사화 과정의 정보 손실(정확한 수치·표현)을 상위 2건에 한해 원문으로 보정하는 장치. exact-span 계열 답(F1)에 직접 영향. v4 Table 6이 이 설정을 기본으로 못박음 |
| 렌더 예산 | 제한 없음 — 10 narrative + 20 semantic + 원문 2건 전부 프롬프트에 (수천 토큰) | `bench/locomo.py:126-128` **budget 1600토큰** — 서사 10건이면 이미 초과, 하위 항목 대량 탈락 (`core/types.py:190-203`) | **변형** | **높음**: 실효 검색량이 upstream의 수분의 1. Nemori처럼 항목이 긴 방법론일수록 불리 — 방법론 간 비교 공정성도 훼손(passthrough의 짧은 raw 메시지가 예산 내 더 많이 들어감) |
| 답변 프롬프트 | `search.py:25-67`: 타임스탬프 기반 상대시간 → 절대날짜 계산 지시 + 단계별 접근(step by step) + "less than 5-6 words", **temperature 0.0** (`search.py:165-169`) | `bench/locomo.py:23-32` 최단 span, CoT 없음; generate role t=0.1 (`exp_locomo_conv0.py:28-29`) | **변형** | 중간: 특히 temporal 카테고리(시간 계산 지시 부재). 단 우리 프롬프트는 전 방법론 공통이므로 상대 비교는 유지, 절대치·논문 수치 비교는 불가 |
| 평가 지표 | 리포는 LLM judge(`evaluation/locomo/metrics/llm_judge.py`); 논문 LLM score | 토큰 F1/BLEU-1 (`bench/locomo.py:89-108`) | **변형(의도적, docs/06)** | 절대치 비교 불가 — 이미 문서화됨 |
| 임베더 | text-embedding-3-small | multilingual-e5-small | 변형(공통 조건) | 낮음 |

---

## 3. ReasoningBank (arXiv:2509.25140)

| 단계 | 논문/공식코드 | 우리 구현 | 판정 | 영향 |
|---|---|---|---|---|
| **판정(judge)** | LLM-as-a-judge, 궤적+질의 → Success/Failure 범주 출력, **t=0.0** (논문 App A.2; `autoeval/clients.py:50,83,148` temperature=0). **judge의 판정 이유(thoughts)를 추출 입력 궤적에 첨부** — "The task succeeded/failed because: ..." (`induce_memory.py:146-162`) | `reasoning_bank.py:112-119` self-judge(JSON bool+reason). **reason을 추출 프롬프트에 미전달**. outcome이 명시되면 judge 생략(코드상 타당한 확장) | **일치(핵심) / 부분 누락(이유 전달)** | 중간: 실패 이유가 추출 LLM의 반성 품질을 끌어올리는 장치 |
| **추출 프롬프트** | `prompts/memory_instruction.py:15-61 SUCCESSFUL_SI/FAILED_SI`: **system 메시지**로 주입, "first think why", **at most 3 items**, no repeat, "Prefer concrete, actionable procedures... Do not embed specific product names, queries, or literal string contents", 출력 **Markdown**(# Memory Item / ## Title / ## Description(1문장, when or when NOT) / ## Content **1-3 sentences**) | `reasoning_bank.py:58-84`: user 프롬프트 단일, 규칙 전부 포함(literal strings 금지·when/when-not·no duplicates), JSON 스키마 maxItems 3, content "1-5 sentences" | **단순화**(포맷 JSON화·content 길이 1-5로 완화 — 1-5는 PARALLEL_SI(MaTTS)의 수치) | 낮음 |
| **추출 온도** | **t=1.0** (논문 App A.2 "temperature 1.0"; `induce_memory.py:164-166`) | 코드가 role 분리를 지원한다고 문서화(`reasoning_bank.py:7-9`)하나 **실험 config는 4 role 전부 t=0.1** (`exp_locomo_conv0.py:28-29 make_roles`) | **변형(실측 config)** | 중간(에이전트 벤치 시): 추출 다양성 감소. judge 0.0도 미준수(0.1) |
| 궤적 포맷 | `<think>...</think><action>...</action>` 페어 전체 + autoeval 이유 (`induce_memory.py:70-76,142-143`) | JSON dump 스텝, **6000자 head/tail 절단** (`reasoning_bank.py:87-93`) | **변형** | 낮음(현재)~중간(실제 웹 궤적은 6000자 초과 상시) |
| 저장/관리 | append-only, dedup/prune 없음 — 논문 각주 2 "deliberately keep the memory usage pipeline simple" | append-only ADD (`reasoning_bank.py:129-146`) | **일치** | — |
| **검색** | **k=1 (기본)**: 현재 질의 임베딩(gemini-embedding-001, instruction-aware) ↔ **과거 태스크 질의 임베딩** 유사도로 top-1 **경험(task)** 선택 → 그 경험의 memory_items(≤3) 전부 주입 (`memory_management.py:138-215`, `run.py:177-191`; 논문 App A.2 "top-k most similar experiences (default k=1)") | 검색 단위가 **아이템**: `StrategyItem.embedding_text()=title\ndescription` (`core/types.py:144-145`)을 질의와 대조, k개 아이템 회수 | **변형** | 중간: upstream은 "비슷한 과제를 했던 경험 1건의 교훈 묶음", 우리는 "비슷해 보이는 교훈 k개". 논문 §3.2 본문("retrieves relevant memories")과는 양립하나 실측 절차와 다름 — 재현 시 경험 단위 모드 필요 |
| **주입 포맷** | **system 프롬프트**에 "Below are some memory items that I accumulated... please first explicitly discuss if you want to use each memory item or not, and then take action" + 아이템 markdown (`agents/legacy/agent.py:132-137`). 논문 App A.2: "each item represented by its **title and content**" (description 제외) | `StrategyItem.render()` = title+**description**+content (`core/types.py:147-148`), MemoryBundle로 유저측 컨텍스트, 사용 지시문 없음 | **변형** | 낮음~중간 |
| max items | 3 | 3 (`maxItems`+슬라이스, `reasoning_bank.py:34,130`) | **일치** | — |
| MaTTS | parallel self-contrast(≤5 items)/sequential re-examine (`memory_instruction.py:63-142`) | 미구현 (docs/10에 기재) | **누락(공지됨)** | LoCoMo 무관, 에이전트 벤치 시 필요 |

---

## 4. 판정 요약 — 충실도 등급 재산정

기존 docs/10 등급 대비 재산정 (●충실 / ◑부분 / ○골격):

| 시스템 | 기존 | **재산정** | 사유 |
|---|---|---|---|
| **A-Mem** | ● | **◑** | write 경로는 ● 수준(잔여: strengthen의 새 노트 tags 미갱신)이나, **read 경로에서 1-hop 링크 확장 누락**으로 링크 생성 모듈의 효용이 측정에 전혀 반영되지 않음. 공식 eval의 키워드 질의·메타데이터 컨텍스트도 부재. "링크를 만들되 쓰지 않는" 구현은 A-Mem의 핵심 주장(Zettelkasten 연결망) 검증으로 부적합 — read 경로 수정 전 측정치는 "A-Mem write + 일반 RAG read"로 라벨해야 함 |
| **Nemori** | ◑ | **◑ (측정 보류 권고)** | 조직화 코어(경계 v1 형식·서사·시간 절대화·predict-calibrate·RAW 캘리브레이션)는 충실. 그러나 QA 검색이 **2k semantic·r=2 원문 첨부·무제한 예산**이라는 공식 설정과 3중으로 어긋나며, 특히 1600토큰 예산은 Nemori처럼 항목이 긴 방법론에 구조적으로 불리. merging·30분 규칙·timestamp 필드도 누락. 기존 "측정됨(merging 부재 명기)" 판단은 **검색 설정 격차를 놓친 과대평가** — 검색 수정 후 재측정 필요 |
| **ReasoningBank** | ● | **◑⁺ (LoCoMo 무관, 에이전트 벤치 전 수정)** | 판정→분기 추출→≤3 append-only 코어는 일치. 그러나 (a) 검색 단위(경험 vs 아이템), (b) 온도 role 분리 미적용(실험 config 전부 0.1), (c) judge 이유 미전달, (d) 주입 포맷이 논문 절차와 다름. 대화 벤치에는 미사용이므로 현 결과 무영향, 에이전트 벤치 착수 전 수정 필수 |

**docs/09 기측정 결과의 유효성**: 4-way 비교의 **상대 순위**는 "동일한(우리) read 파이프라인 위에서의 write 조직화 비교"로는 여전히 유의미하나, **각 방법론의 논문 대비 재현으로 해석해서는 안 됨** — 특히 A-Mem·Nemori는 read 경로가 논문 설정보다 불리하게 측정됨. 결과 문서에 이 프레임을 명기할 것.

## 5. "우리 버그픽스" 재검증 결과 (한줄 요약)

| 픽스 | 맞았나 | 과했나 | upstream이 고쳤나 |
|---|---|---|---|
| #24 L2/cosine | 맞음 (논문 식(4)는 cosine) | **서술 과함** — 라이브러리판의 score 의미 반전이 실체, 랭킹 붕괴 아님. 논문 재현판(WujiangXu)은 원래 cosine | 아니오 (open, 2026-07 기준) |
| #32 index-vs-id | 맞음 (agiresearch판 실버그: 최고(最古) k개 노트를 오갱신 + 링크 ID 무효) | **범위 한정 필요** — WujiangXu 재현판은 인덱스 체계가 자기일관적이라 논문 수치는 이 버그와 무관 | 아니오 (open) |
| silent skip (#10 계열) | 맞음 | 아니오 — 순수 관측성 개선 | 부분(#13) 후 정체; 재현판엔 **신규 결함 발견**: `memory_layer.py` `import re` 부재로 Ps1 메타데이터 전량 빈 값 폴백 (robust판만 정상) |

추가 교정 사항: `amem.py:7-13` 모듈 docstring과 docs/08의 버그 서술을 위 표 기준으로 완화·정밀화할 것 (특히 "reference ChromaDB ... treating scores as similarity" → "score 필드 의미 반전, 랭킹은 유지").

## 6. 측정 재개 전 필수 수정 목록 (우선순위순)

**P0 — 이번 재측정 전 필수 (read 경로, 결과에 직접 영향)**

1. **A-Mem 1-hop 링크 확장**: 질의 hit의 `links`를 따라 이웃 노트를 컨텍스트에 추가 (upstream `find_related_memories_raw` 등가).
   위치: `src/agmem/retrieval/pipeline.py`(memory_type=="notes"일 때 hydrate 후 링크 팔로우) 또는 amem 전용 recall 단계. 총량 상한(upstream은 전체 k 상한) 규칙 결정 필요 — 우리 양방향 링크는 확장 폭이 upstream(단방향)보다 넓으므로 상한 필수.
2. **Nemori 검색 설정 정합**: episodic(에피소드) k=10 + semantic m=2k=20, **top-2 에피소드에 source_episode_ids로 원문 메시지 첨부**.
   위치: `src/agmem/bench/locomo.py answer()`에 per-type k 지원(또는 `pipeline.search`에 memory_type별 k dict), 원문 첨부는 hydrate 시 `doc.get_episodes(source_episode_ids)`.
3. **렌더 예산 상향/타입별 예산**: 1600토큰 → 방법론별 공식 설정에 준하는 수준(최소 4-6k) 또는 "검색된 항목 전부" 모드. 방법론 간 공정성 문제이므로 4-way 전체 재측정 대상.
   위치: `src/agmem/core/types.py MemoryBundle.render`, `bench/locomo.py`.
4. **컨텍스트 렌더에 메타데이터 노출**: notes는 context(+tags), episodes/semantic은 timestamp 병기.
   위치: `src/agmem/retrieval/pipeline.py _DictItem.render`.

**P1 — write 경로 (재인제스트 필요, P0와 같은 라운드에 처리 권장)**

5. **A-Mem strengthen의 `tags_to_update` 적용**: 진화 콜 응답에 새 노트 태그 갱신 필드 추가, 새 노트 embedding_text 재계산.
   위치: `src/agmem/organizers/amem.py` EVOLVE_SCHEMA/EVOLVE_PROMPT/on_message.
6. **Nemori 경계 프롬프트 보강**: 30분 시간 갭 규칙 + "when in doubt, split" + 2-15 메시지 가이드.
   위치: `src/agmem/organizers/nemori.py BOUNDARY_PROMPT`.
7. **Nemori cold-start 직접 추출**: 검색된 semantic이 0건이면 predict 생략하고 직접 추출 프롬프트(4 tests 동일) 사용.
   위치: `src/agmem/organizers/nemori.py _predict_calibrate`.
8. **Nemori 에피소드 timestamp 필드**: EPISODE_SCHEMA에 timestamp 추가, payload에 저장, 렌더에 사용 (항목 4와 연동).

**P2 — 벤치 프레임/문서 (코드 아님)**

9. docs/09·10에 본 감사 결론 반영: A-Mem 등급 ◑ 강등, "read 경로 수정 전 수치는 write-only 비교" 캐비앗, 버그 서술 교정(§5).
10. A-Mem 공식 eval 절차 옵션화 검토: LLM 키워드 질의 생성, notes 단독 검색 조건(현재 episodic 혼합은 passthrough 성분 오염). **cat5 이지선다(gold 노출)는 재현하지 않되 upstream 수치와 비교 시 반드시 명기.**
11. **Nemori 에피소드 병합**: LongMemEval/배치 인제스트 단계 진입 시 구현 (v4 3.2.3, MERGE_DECISION/MERGE_CONTENT, >1h 금지 규칙, 원본 대체).

**P3 — ReasoningBank (에이전트 벤치 착수 전)**

12. 온도 role 분리 실측 반영: judge 0.0 / distill 1.0 (`scripts/` 실험 config).
13. 검색을 경험(task) 단위 모드로 지원: 태스크 질의 임베딩 저장 → top-1 경험의 아이템 묶음 주입 + system 프롬프트 주입 지시문("explicitly discuss if you want to use each memory item").
14. judge 이유(reason)를 추출 프롬프트 궤적 말미에 첨부; content 길이 규칙 1-3문장으로 정합; 궤적 절단 정책 재검토.
