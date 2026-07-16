# Nemori vs ReasoningBank: 심층 리서치 리포트

> 조사 기준일: 2026-07-16. 소스: arXiv/ACL Anthology/OpenReview 논문 원문 + GitHub 공식 저장소 코드 직접 분석.

---

# 1. Nemori: Self-Organizing Agent Memory Inspired by Cognitive Science

**논문 이력**: arXiv:2508.03341. 이 논문은 **버전에 따라 제목이 바뀌었다**. v1(2025-08-05, "Nemori: Self-Organizing Agent Memory Inspired by Cognitive Science")로 처음 공개된 후, ACL 2026 Long Paper로 채택되면서 v4(제목: **"What Deserves Memory: Adaptive Memory Distillation for LLM Agents"**, aclanthology.org/2026.acl-long.1607)로 확장·재구성되었다. 저자: Jiayan Nan(Tongji University, 교신저자), Wenquan Ma(SUFE), Wenlong Wu(Beihang), Yize Chen(Tanka AI). v4에서는 baseline이 대폭 확장되고(A-MEM, MemoryOS 추가), "third-party management integration"이라는 새 실험축이 추가되었다. 아래는 v1과 v4(ACL) 양쪽 내용을 종합한 것이다.

**공식 코드**: github.com/nemori-ai/nemori (MIT License, 207 stars, 2026-07-16 기준). 주의: 현재 main 브랜치는 논문 정합용으로 **완전히 재작성**되었고, 예전 MVP는 `legacy-mvp` 브랜치로 분리되어 있음("This release is a complete rewrite ... not compatible with the previous MVP").

## A. 핵심 개념

Nemori의 문제의식: 기존 MAG(Memory-Augmented Generation) 시스템은 두 가지 축에서 실패한다 — 입력 청크(x, 메모리 단위 granularity)와 조직화 함수(f, 메모리 구축·진화 메커니즘). 이를 `y = f(x)`로 정식화하고, 두 원칙으로 해결한다.

### 1) Two-Step Alignment Principle (Event Segmentation Theory 기반)

- **Boundary Alignment (경계 탐지)**: 메시지 버퍼 `B_u`에 메시지가 쌓이면, 새 메시지 `m_{t+1}`마다 LLM 기반 경계 탐지기 `f_θ`가 `(b_boundary, c_boundary)` — 불리언 판단과 신뢰도 점수 — 를 출력. 트리거 조건: `T = (b_boundary ∧ c_boundary > σ_boundary) ∨ (|M| ≥ β_max)`. 즉 **고신뢰도 의미적 전환이 감지되거나 버퍼가 최대 크기에 도달**하면 세그먼트를 확정. 판단 근거는 문맥 일관성(의미 유사도), 시간적 마커("by the way" 등), 사용자 의도 전환(정보 요청→의사결정 등). 실제 프롬프트(`nemori/llm/prompts.py`의 `BATCH_SEGMENTATION_PROMPT`)는 배치 방식으로 메시지 그룹 전체를 한 번에 분석해 `{"episodes": [{"indices": [...], "topic": "..."}]}` JSON을 반환하며, 토픽 전환/의도 전환/시간 마커(30분 이상 gap)/구조적 신호/콘텐츠 관련성(<30% 시 분할)을 명시적 기준으로 제시. 원칙은 "When in doubt, split", 에피소드당 2~15개 메시지 권장.
- **Representation Alignment (서사 생성)**: 세그먼트 `M`을 LLM 기반 Episode Generator `g_φ`가 `e=(ξ, ζ)` — 제목(ξ)과 3인칭 서사(ζ) — 로 변환. `EPISODE_GENERATION_PROMPT`는 특히 **시간 정보의 절대화**를 강조("yesterday" → 정확한 연/월/일/시로 변환해 괄호 병기). 이 시간-앵커링이 Temporal Reasoning 우위의 핵심 근거.

### 2) Predict-Calibrate Principle (Free-Energy Principle 기반) — semantic memory 증류, 3단계

- **Stage 1 Prediction**: 새 에피소드 제목 `ξ`로 기존 semantic memory DB `K`에서 관련 지식 `K_relevant = Retrieve(embed(ξ⊕ζ), K, m, σ_s)` 검색 → LLM Episode Predictor `h_ψ`가 제목과 관련 지식만으로 **원본을 보지 않고 내용을 예측**(`ê`). `PREDICTION_PROMPT`는 "문체가 아닌 실제 지식/사실을 예측하라"고 명시적으로 지시.
- **Stage 2 Calibration**: 예측 `ê`를 (생성된 서사 ζ가 아닌!) **원본 미가공 대화 M**과 비교하여 "예측 오차(prediction gap)" — 기존 지식이 예측하지 못한 새롭거나 놀라운 정보 — 를 LLM Semantic Knowledge Distiller `r_ω`가 추출: `K_new = r_ω(ê, M)`. `EXTRACT_KNOWLEDGE_FROM_COMPARISON_PROMPT`는 Persistence/Specificity/Utility/Independence 4대 테스트를 통과하는 지식만 추출하도록 강제(예: "Caroline's favorite book is 'Becoming Nicole'"은 채택, "The user thanked the assistant"는 기각). 각 statement는 self-contained/atomic, 현재시제, 시간정보 미포함.
- **Stage 3 Integration**: `K_new`를 semantic memory DB에 통합.

핵심 통찰: **"예측-실패에서 배운다"** — 이미 아는 것을 재추출(수동적 rule-based extraction)하는 게 아니라, 모르는 것(prediction gap)만 능동적으로 학습. Free-energy principle의 surprise 개념을 LLM 파이프라인으로 조작화한 것.

### 3) Unified Retrieval

episodic/semantic memory 모두 동일한 `Retrieve(q, D, m, σ_s)` — dense vector cosine similarity 기반 3단계(유사도 계산→후보 선정→임계값 필터링). 기본 설정은 top-k episodic + top-2k semantic(비율 고정, m=2k), 상위 2개 에피소드(r=2)만 원본 대화 텍스트 첨부.

### 4) v4(ACL)에서의 재프레이밍

v4 논문은 이를 **"experience의 future utility 평가를 predictability 문제로 환원"**으로 재프레이밍하고, memory 시스템을 retrieval-time / management-time / distillation-time 3단계로 분류하는 taxonomy(Table 1)를 추가. 또한 Nemori를 **third-party 메모리 시스템(A-MEM, MemoryOS)의 "distillation kernel"로 꽂아넣는** 실험을 추가해 "management-agnostic" 설계 철학을 강조.

## B. 공식 코드 분석

### 저장소 구조 (`nemori-ai/nemori`, Python 3.10+)

```
nemori/
├── api/facade.py         # NemoriMemory 비동기 퍼사드 클래스
├── config.py              # MemoryConfig 데이터클래스 (하이퍼파라미터)
├── core/memory_system.py  # MemorySystem 오케스트레이터
├── db/                    # buffer_store, episode_store, semantic_store (PostgreSQL)
│                          #  + qdrant_store.py, connection.py, migrations.py
├── domain/models.py       # Message, Episode, SemanticMemory, HealthResult 데이터클래스
├── llm/                   # client.py, orchestrator.py, prompts.py(PromptTemplates), generators/
├── search/unified.py      # 벡터+텍스트 하이브리드 통합 검색
├── services/              # embedding.py, event_bus.py
└── utils/                 # 이미지 압축(멀티모달용)
evaluation/{locomo, longmemeval}/  # 벤치마크 스크립트
docker/init-extensions.sql
docker-compose.yml
```

### 데이터 모델 (`nemori/domain/models.py`)

- `Episode(user_id, title, content, source_messages, agent_id, embedding, metadata, created_at, updated_at)`
- `SemanticMemory(user_id, content, memory_type, source_episode_id, confidence, embedding, ...)` — semantic memory가 `source_episode_id`로 provenance 유지.
- `Message`는 OpenAI content array 표준의 멀티모달 콘텐츠(`text` / `image_url` parts) 지원.

### 저장소/임베딩 선택

**듀얼 백엔드** — PostgreSQL 16(메타데이터, tsvector/GIN 텍스트 검색, 메시지 버퍼링, asyncpg 완전 비동기)와 Qdrant(모든 벡터 저장/유사도 검색, gRPC 클라이언트, 임베딩 차원 자동 적응). Docker Compose로 두 서비스 기동(PostgreSQL 5432, Qdrant 6333/6334). LLM/임베딩은 OpenAI SDK 호환 인터페이스를 통해 OpenRouter(권장, 단일 키) 또는 OpenAI 직접 연동. 논문 실험은 `text-embedding-3-small`.

### 핵심 하이퍼파라미터 (`nemori/config.py` `MemoryConfig` 실제 기본값)

```python
llm_model = "gpt-4o-mini"
embedding_model = "text-embedding-3-small"; embedding_dimension = 1536
buffer_size_min = 2; buffer_size_max = 25       # β_max
enable_batch_segmentation = True; batch_threshold = 20
episode_min_messages = 2; episode_max_messages = 25
enable_semantic_memory = True; enable_prediction_correction = True
semantic_similarity_threshold = 0.85
enable_episode_merging = True; merge_similarity_threshold = 0.85; merge_top_k = 5
search_top_k_episodes = 10; search_top_k_semantic = 10
llm_max_concurrent = 10; llm_timeout = 30.0; llm_retries = 3
```

논문 실험 설정(v1): `σ_s=0.0, σ_boundary=0.7, β_max=25`, k=10(m=2k=20). v4(ACL): `τ=0.70, K_e=K_m=5, K_s=10`으로 표기 변경. 코드에는 논문 수식에 없는 **Episode Merging** 기능도 존재 — 시간적으로 겹치는 유사 에피소드를 LLM(`MERGE_DECISION_PROMPT`, `MERGE_CONTENT_PROMPT`)이 병합 판단(같은 이벤트/1시간 이내만 병합).

### 의존성 / 라이선스 / MCP

- 런타임: `asyncpg, openai, tiktoken, pydantic, python-dotenv, Pillow, qdrant-client`
- eval extra: `nltk, bert-score, rouge-score, pandas, spacy`
- 라이선스: **MIT**
- **MCP 서버 지원 없음** (코드 검색 결과 없음)

### 멀티테넌시/멀티모달

`agent_id` 기반 워크스페이스 격리(에피소드/semantic/vector collection 네임스페이스 분리). `add_multimodal_message()`로 이미지 압축 후 저장(2026-03-24 릴리스 추가).

### 릴리스 타임라인

- 2025-07-10 MVP(episodic memory generation)
- 2025-09-26 전체 오픈소스 공개(episodic+semantic 전체)
- 2025-10-28 세그멘터 업그레이드 + 토큰 카운팅
- 2026-03-24 완전 비동기 재작성(PostgreSQL+Qdrant 듀얼백엔드, OpenRouter, 멀티모달, Docker Compose)

## C. 성능 특성

### LoCoMo (1,540개 질문, 10개 대화, 평균 24K 토큰) — v4(ACL) 최종 표 기준

Baseline 7종: Full Context, RAG-4096, LangMem, Zep, Mem0, A-MEM, MemoryOS. Judge: gpt-4o-mini.

| Backbone | 방법 | Overall LLM-judge | F1 | BLEU-1 |
|---|---|---|---|---|
| gpt-4.1-mini | Full Context | 80.6 | – | – |
| gpt-4.1-mini | LangMem (최강 baseline) | 73.4 | 47.6 | 40.0 |
| gpt-4.1-mini | Zep | 61.6 | 36.9 | 30.9 |
| gpt-4.1-mini | Mem0 | 66.3 | 43.5 | 36.5 |
| gpt-4.1-mini | A-MEM | 61.4 | 39.4 | 33.2 |
| gpt-4.1-mini | MemoryOS | 60.6 | 39.9 | 32.5 |
| gpt-4.1-mini | **Nemori** | **80.8** | **52.1** | **45.0** |
| gpt-4o-mini | Full Context | 72.3 | 46.2 | 37.8 |
| gpt-4o-mini | Mem0 (최강 baseline) | 61.3 | 41.5 | 34.2 |
| gpt-4o-mini | Zep | 58.5 | 37.5 | 30.9 |
| gpt-4o-mini | A-MEM | 52.5 | 32.4 | 27.0 |
| gpt-4o-mini | MemoryOS | 54.5 | 39.9 | 31.9 |
| gpt-4o-mini | **Nemori** | **73.0** | **50.3** | **39.7** |

- gpt-4.1-mini: LangMem 대비 **+10.1%**, Full Context 근소 상회(80.8 vs 80.6). gpt-4o-mini: Mem0 대비 **+19.1%**.
- **Temporal Reasoning**이 최대 강점: 77.3(gpt-4.1-mini, A-MEM 대비 +15.9%), 67.6(gpt-4o-mini, Zep 대비 +14.8%). "reasoning during memory formation"(메모리 형성 시 시간 절대화) 설계 효과.
- Open Domain 카테고리는 최강 baseline 대비 소폭 열세(사전지식 의존 특성).

### 메모리 구축 비용 (Table 3, gpt-4o-mini)

| 방법 | LLM점수 | LLM Calls | Input(k) | Output(k) | Total(k) |
|---|---|---|---|---|---|
| LangMem | 51.3 | 920.6 | 898.3 | 112.0 | 1010.2 |
| Mem0 | 61.3 | 1602.2 | 1483.4 | 210.0 | 1693.4 |
| A-MEM | 52.5 | 1175.5 | 912.6 | 236.8 | 1149.4 |
| MemoryOS | 54.5 | 1016.1 | 404.5 | 122.0 | 526.5 |
| **Nemori** | **73.0** | **373.2** | **277.2** | **45.7** | **322.9** |

LLM 호출 **-59.5%**, 토큰 **-38.7%**(최강 baseline 대비). 메시지 단위가 아닌 **에피소드 단위 처리**가 원인.

### 응답 생성 비용/지연시간 (Table 4, gpt-4o-mini)

| 방법 | LLM점수 | 평균토큰 | Search(ms) | Total(ms) |
|---|---|---|---|---|
| FullContext | 72.3 | 23,653 | – | 5,806 |
| LangMem | 51.3 | 125 | 19,829 | 22,082 |
| Zep | 58.5 | 2,247 | 522 | 3,255 |
| Mem0 | 61.3 | 1,027 | 784 | 3,539 |
| A-MEM | 52.5 | 2,614 | 947 | 2,867 |
| MemoryOS | 54.5 | 1,560 | 9,910 | 15,220 |
| **Nemori** | **73.0** | 2,745 | 787 | **3,053** |

Full Context 대비 **토큰 88% 절감, 지연 47% 감소**하면서 정확도 우세. MemoryOS는 검색에만 9.9초(계층 구조의 대가), LangMem은 검색 19.8초.

### Ablation

- `w/o Nemori`(프레임워크 제거) → 0.6점으로 붕괴(구조 필수성).
- **Predict-Calibrate 직접 검증**: `w/o e`(예측오차 증류 semantic만) vs `Nemori-s`(직접 추출 semantic만) = gpt-4o-mini에서 65.0 vs 52.0(**+25.0%**), gpt-4.1-mini에서 74.9 vs 65.5(**+14.4%**).
- episodic 제거 손실 > semantic 제거 손실(상호보완적이되 episodic이 더 핵심).
- Retrieval 전략 ablation(Table 6): narrative 임베딩이 raw 임베딩보다 일관되게 우세(76.9 vs 76.4).
- k=10에서 성능 포화(diminishing returns). 관찰 윈도우 w=5~40에서 ±1% 안정.

### LongMemEval_S (500개 대화, 평균 105K 토큰)

Full Context 대비 gpt-4o-mini **+16.7%**(64.2 vs 55.0), gpt-4.1-mini **+13.7%**(74.6 vs 65.6) — LoCoMo의 +1.0%/+0.2%보다 훨씬 커서, **컨텍스트가 길수록 우위 확대** 입증. single-session-preference에서 극적(6.7→46.7, 16.7→86.7). 반면 single-session-assistant와 knowledge-update는 Full Context 우세(세밀 디테일 손실, 논문이 인정하는 한계). 사용 토큰 3.7~4.8K로 **95~96% 컨텍스트 절감**.

### Third-party 통합 (Table 7, v4 신규)

Nemori의 증류 지식 `K`를 A-MEM/MemoryOS 입력으로 raw messages 대신 사용 시, 저장공간 **45~64% 감소** + core 점수 **+1.9%~+6.1%** 개선(평균은 ±4% 유지).

### Small-model 근거

논문은 gpt-4o-mini/gpt-4.1-mini만 사용. Full Context와의 격차가 **약한 모델(gpt-4o-mini)에서 더 크다**는 패턴 관찰 — "메모리 시스템의 가치는 모델 능력이 제한적일수록, 태스크가 복잡할수록 커진다"는 해석.

## D. 재현 관점

- 재현 진입점: `evaluation/locomo/{add,search,evals,generate_scores}.py` 순차 실행. `pyproject.toml` `[project.scripts]`에 `nemori-add-locomo`, `nemori-eval-locomo`, `nemori-add-longmemeval` 등 CLI 엔트리포인트 등록.
- Judge 모델: **gpt-4o-mini**(LoCoMo/LongMemEval 공통, LongMemEval은 Zep 방식의 태스크별 프롬프트). 모든 baseline 동일 judge.
- **Mem0/Zep은 상용 API**로 메모리 컨텍스트만 취득, 답변 생성은 gpt-4o-mini/gpt-4.1-mini로 통일 — 이 두 baseline은 API 키가 필요하고 완전한 오프라인 재현 불가.
- 인프라: Docker Compose(PostgreSQL 16 + Qdrant) 필수. 임베딩 모델 변경 시 Qdrant 컬렉션 재생성 필요.
- 함정:
  1. 2026-03-24 재작성 이전 커밋은 논문 실험 당시 코드와 다를 수 있어 커밋/태그 확인 필요.
  2. v1과 v4 논문의 하이퍼파라미터 표기가 다름(`σ_boundary=0.7` vs `τ=0.70`, `K_e/K_m/K_s`) — 재현 대상 버전을 명시해야 함.
  3. LoCoMo 점수 스케일이 v1(0~1)과 v4(0~100)로 다르게 표기됨.

---

# 2. ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory

**논문**: arXiv:2509.25140 (Google Cloud AI Research). 저자: Siru Ouyang(UIUC), Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang(Yale), Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han(UIUC), Chen-Yu Lee, Tomas Pfister. v1 제출 2025-09-29, v2 개정 2026-03-16. **ICLR 2026 채택**(openreview.net/forum?id=jL7fwchScm).

**공식 코드**: github.com/google-research/reasoning-bank (Apache-2.0, 438 stars). README 명시: **"This is not an officially supported Google product... intended for demonstration purposes only, not for production."**

## A. 핵심 개념

**문제의식**: 지속 배치된 LLM 에이전트가 태스크 스트림에서 축적 경험으로부터 학습하지 못함. 기존 메모리(Synapse=raw trajectory, AWM=성공 워크플로우만)는 (1) 고차원 전이가능 추론 패턴 증류 실패, (2) **성공 경험에만 치중해 실패의 교훈을 방치**.

**Test-time learning 세팅**: 쿼리 스트림 `Q={q_1,...,q_N}`이 순차 도착, ground-truth 없음 — 에이전트는 자기 트래젝토리와 self-verification만으로 진화해야 함. 에이전트 정책 `π_L(·|M,A)` — 메모리 모듈 M은 관련 메모리를 system instruction으로 주입.

**메모리 스키마**: 각 memory item = `{title, description, content}`
- title: 전략/추론 패턴의 간결한 식별자
- description: 한 문장 요약, **언제/언제 쓰지 말아야 하는지** 포함
- content: 증류된 추론 단계·의사결정 근거·운영 통찰, 1~5문장
- human-interpretable + machine-usable 양립 설계.

**Closed-loop 3단계**:

1. **Memory Retrieval**: 태스크 쿼리를 **gemini-embedding-001**(Vertex AI)로 임베딩, cosine 유사도 top-k 검색(**기본 k=1**) → title+content 형식으로 system prompt에 주입. 주입 지시문: "Below are some memory items that I accumulated from past interaction... In each step, please first explicitly discuss if you want to use each memory item or not, and then take action."
2. **Memory Construction**: 완료 후 **LLM-as-a-judge**(에이전트와 동일 backbone, **temperature=0.0**)가 쿼리+트래젝토리+최종 상태를 보고 Success/Failure 이진 판정(ground-truth 불사용) → 성공/실패 각각 다른 프롬프트로 **최대 3개** memory item 추출(**temperature=1.0**). `SUCCESSFUL_SI`: "왜 성공했는지 먼저 사고 → 전이가능 전략 요약". `FAILED_SI`: "왜 실패했는지 반성 → 예방적 교훈/복구 절차 도출". 두 프롬프트 모두 "구체적 상품명/쿼리/리터럴 문자열 임베드 금지, 추상 원칙보다 구체적 실행 절차 우선, 중복 금지" 강제. 출력은 `# Memory Item i / ## Title / ## Description / ## Content` Markdown 고정 포맷.
3. **Memory Consolidation**: 단순 추가(pruning/merging 없음) — "ReasoningBank 자체의 기여를 분리하기 위한 의도적 단순화".

**MaTTS (Memory-aware Test-Time Scaling)** — scaling factor k:
- **Parallel Scaling**: 같은 쿼리에 대해 검색 메모리 하에서 k개 트래젝토리 병렬 생성 → **self-contrast**(`PARALLEL_SI` 프롬프트, 최대 5개 item): 성공/실패 트래젝토리를 직접 비교·대조하여 일관 패턴은 채택, spurious 해법은 필터. 최종 답은 Best-of-N — 동일 backbone LLM이 N개 트래젝토리를 한 번에 보고 최선 선택.
- **Sequential Scaling**: 단일 트래젝토리를 완료 후 **self-refinement** 반복(`SEQUENTIAL_PROMPT`: "이전 트래젝토리를 재검토, 올바른 요소를 썼는지·응답이 쿼리를 해결하는지 확인, `<think>...</think><action></action>` 포맷 유지"). 중간 노트도 메모리 신호로 활용.
- **핵심 통찰(양방향 시너지)**: 좋은 메모리 → 스케일링을 유망 경로로 유도, 풍부한 rollout → 대조 신호로 더 좋은 메모리 합성. 이 선순환이 "memory-driven experience scaling"이라는 새 스케일링 축.

## B. 공식 코드 분석

### 저장소 구조 (`google-research/reasoning-bank`, Python≥3.13)

```
main.py                       # placeholder ("Hello from reasoning-bank!")
WebArena/
├── agents/{basic,legacy}/    # BrowserGym 통합 웹 에이전트
├── autoeval/                  # LLM-as-a-judge 정답성 판정
├── config_files/              # WebArena 태스크 config 생성 (generate_config_files.py)
├── prompts/autoeval_prompts.py
├── prompts/memory_instruction.py  # 전체 프롬프트 원문 공개
├── induce_memory.py, induce_scaling.py
├── memory_management.py, pipeline_memory.py, pipeline_scaling.py
├── run.py, run.sh, webarena_patch.py, reeval_memevol_prompt.py
SWE-Bench/
├── run.sh, compute_stats.py
third_party/                   # 패치된 webarena 트리 + mini-swe-agent
```

`memory_instruction.py`에 공개된 프롬프트: `SUCCESSFUL_SI`, `FAILED_SI`, `PARALLEL_SI`, `PARALLEL_AWM_SI`(AWM baseline 재현용), `SEQUENTIAL_PROMPT`/`SEQUENTIAL_FOLLOWING_PROMPT`, `AWM_INSTRUCTION`/`AWM_EXAMPLE`.

### 의존성 (`pyproject.toml`)

- `browsergym-core/experiments/webarena==0.14.1` (버전 고정, "Must match MemEvol_cleaned for correct reward calculation" 주석)
- `playwright>=1.44.0`, `anthropic>=0.64.0`, `google-cloud-aiplatform>=1.108.0`, `google-genai>=1.21.1`
- `langchain>=0.3.26`(+anthropic/community/openai), `openai>=2.26.0`, `torch>=2.7.0`

### LLM 지원

GPT(OpenAI API 직접), Gemini(gemini-2.5-flash/pro)와 Claude(claude-3-7-sonnet@20250219)는 **Vertex AI 경유**(`gcloud auth application-default login` + `GOOGLE_CLOUD_PROJECT/LOCATION`, `GOOGLE_GENAI_USE_VERTEXAI=True`).

### 웹 환경

BrowserGym 위에 구축, WebArena Docker 환경(gasse/webarena-setup) 별도 필요. **패치된 webarena 트리를 `third_party/webarena/`에 동봉**(shopping split 주석 수정, wishlist eval 수정, `fill('','')` 가드, `retry_with_force=True` 클릭 안정화) — `PYTHONPATH`에 `third_party` prepend로 pip 설치본을 shadow(namespace package라 코드 수정 불필요).

### SWE-Bench

mini-swe-agent 기반, **Bash-only 환경**(툴 없음, 스캐폴드 없음, 순수 ReAct 루프). `mini-extra swebench --model gemini-2.5-flash --subset verified --split test --workers 1`로 실행, 평가는 `sb-cli`.

### 메모리 저장 포맷

**JSON 파일** — entry = `{task query, original trajectory, memory items}`. 쿼리 임베딩은 사전계산되어 별도 JSON 저장. run별 메모리 풀 영속화. 벡터 DB 없음.

### 라이선스 / MCP / 커뮤니티

- 라이선스: **Apache-2.0**. **MCP 지원 없음**.
- 커뮤니티 구현: budprat/ReasoningBank(비공식 재구현), Lanerra/reasoning-bank-slm(소형모델 실험), langchain4j Discussion #4366(기능 제안).
- **재현 흔적**: `WebArena/run.sh`에 저자 개인 경로(`/home/junyann_google_com/projects/reasoning-bank/.venv`)와 GCP 프로젝트(`zifengw-research`)가 하드코딩되어 남아있음 — 내부 스크립트를 거의 그대로 공개한 것으로, 재현 시 치환 필수.

## C. 성능 특성

### WebArena (684개 인스턴스: Shopping 187 / Admin 182 / Gitlab 180 / Reddit 106 / Multi 29)

Success Rate(SR↑)/평균 Step(↓), baseline: No Memory, Synapse, AWM:

| Backbone | 방법 | Overall SR | Overall Step |
|---|---|---|---|
| Gemini-2.5-flash | No Memory | 40.5 | 9.7 |
| Gemini-2.5-flash | Synapse | 42.1 | 9.2 |
| Gemini-2.5-flash | AWM | 44.1 | 9.0 |
| Gemini-2.5-flash | **ReasoningBank** | **48.8** (+8.3 vs NoMem) | **8.3** |
| Gemini-2.5-pro | No Memory | 46.7 | 8.8 |
| Gemini-2.5-pro | **ReasoningBank** | **53.9** (+7.2) | **7.4** |
| Claude-3.7-sonnet | No Memory | 41.7 | 8.0 |
| Claude-3.7-sonnet | **ReasoningBank** | **46.3** (+4.6) | **7.3** |

Multi subset(cross-website 전이)에서 AWM은 No Memory보다 **하락**(Gemini-flash: 10.3→3.4)하는 반면 ReasoningBank는 상승(10.3→13.8; Gemini-pro 6.9→13.8) — workflow형 메모리의 일반화 실패를 대조.

### SWE-Bench-Verified (500개, mini-swe-agent)

| Backbone | 방법 | Resolve Rate | Step |
|---|---|---|---|
| Gemini-2.5-flash | No Memory | 34.2 | 30.3 |
| Gemini-2.5-flash | Synapse | 35.4 | 30.7 |
| Gemini-2.5-flash | **ReasoningBank** | **38.8** | **27.5** |
| Gemini-2.5-pro | No Memory | 54.0 | 21.1 |
| Gemini-2.5-pro | **ReasoningBank** | **57.4** | **19.8** |

### Mind2Web (cross-task 252 / cross-website 177 / cross-domain 912)

모든 세팅에서 개선, **cross-domain(최고 일반화 난도)에서 격차 최대** — Gemini-flash EA 35.8→40.6, SR 1.0→1.6; Gemini-pro EA 37.9→42.8, SR 1.4→1.7.

### 효율성 (step 감소)

- WebArena에서 No Memory 대비 최대 1.4 step, 다른 메모리 baseline 대비 1.6 step 감소. SWE-Bench에서 2.8/1.3 step 절감(전체 상대 기준 최대 16.0% 감소).
- 성공/실패 분리 분석(Table 4): **step 감소가 성공 케이스에서 훨씬 큼**(Shopping: 6.8→4.7, -2.1 step, 26.9% 상대감소) — "실패를 조기 포기시켜서"가 아니라 "성공 경로를 더 효율적으로 찾아서" 효율 개선.

### 실패 트래젝토리 활용 (Figure 7, WebArena-Shopping, Gemini-2.5-flash)

- Synapse 40.6→41.7(미미), AWM 44.4→42.2(**하락**)
- **ReasoningBank 46.5(성공만)→49.7(성공+실패)** — 실패를 노이즈가 아닌 건설적 신호로 변환하는 유일한 설계.

### MaTTS (WebArena-Shopping, Gemini-2.5-flash)

- Parallel: k=1의 49.7 → k=5의 **55.1**; Sequential: 49.7 → **54.5**. 메모리 없는 TTS는 39.0~42.2 사이에서 요동(이득 미미).
- MaTTS vs vanilla TTS(w/o aggregation): k=5에서 55.1 vs 52.4(parallel), 54.5 vs 51.9(sequential) — 대조 신호 집계가 핵심.
- k=3 스냅샷의 BoN: No memory 39.0→40.6, Synapse→42.8, AWM→45.5, **ReasoningBank 49.7→52.4** — 좋은 메모리일수록 스케일링을 잘 흡수.
- Pass@1: 약한 메모리는 스케일링이 오히려 악화(Synapse 40.6→40.1, AWM 44.4→41.2), ReasoningBank만 49.7→50.8 개선 — **비대칭성** 입증.
- Pass@k 분석: MaTTS는 k=2에서 51.3, k=5에서 62.1(메모리 없는 스케일링은 52.4) — 샘플 효율과 성장 지속성 모두 우세.
- 검색 개수 ablation: k=1이 최적(49.7), k=2/3/4는 46.0/45.5/44.4로 하락 — **과다 검색은 노이즈/충돌 유발, 품질>양**.

### Emergent behavior (정성적, Figure 6)

memory item이 execution-oriented 규칙 → adaptive self-reflection(식별자 재검증) → adaptive check(검색/필터 체계적 활용) → compositional strategy(교차참조·재평가)로 **RL 학습 동역학과 유사하게 진화**.

### Small-model 근거

논문 본체는 Gemini-2.5-flash/pro, Claude-3.7-sonnet만. **커뮤니티 실험**(Lanerra/reasoning-bank-slm): Qwen3-1.7B, competition_math(MATH Level 3-4)에서 40.0% → 48.0%(**+8.0pp, +20% 상대개선**; 16개 개선/8개 퇴행). 단 95% CI에서 통계적 비유의(구간 겹침)를 저자 스스로 명시. 축적된 223개 메모리 중 94.6%가 성공 기반(원 논문보다 실패 비중 훨씬 낮음).

## D. 재현 관점

- **에이전트 스캐폴드**: WebArena/Mind2Web은 **BrowserGym**(0.14.1 버전 고정) + **ReAct** 스타일 루프(SeeAct 아님). observation은 텍스트 accessibility tree(실제로는 길이 문제로 thinking process를 관찰 근사로 사용 — AWM 논문 방식 답습). 에이전트 decoding temperature=**0.7**, max step=**30**(WebArena).
- SWE-Bench는 **mini-swe-agent**(bash-only) 기반.
- **Judge**: correctness 이진 판정은 **에이전트와 동일 backbone**, temperature=**0.0**. 메모리 추출은 temperature=**1.0**, 트래젝토리당 최대 3개(parallel scaling 시 최대 5개). WebArena 최종 평가는 벤치마크 기본 프로토콜(LLM fuzzy matching + exact string matching).
- **임베딩**: gemini-embedding-001(Vertex AI), cosine similarity, 기본 top-k=1.
- **재현 pitfalls**:
  1. **Map 도메인 제외**(웹사이트 이슈, Miyai et al. 2025 프로토콜) — 이를 놓치면 overall 숫자 불일치.
  2. **AWM은 SWE-Bench baseline에서 제외**(open-ended bash 액션 공간에서 고정 워크플로우 추출 불가 — 논문 각주 명시).
  3. Gemini/Claude 모두 **Vertex AI ADC 인증** 필요 — OpenAI 키만으로는 GPT 계열만 가능.
  4. `run.sh`의 하드코딩된 저자 GCP 프로젝트/경로 치환 필수.
  5. `third_party/webarena` 패치 트리를 PYTHONPATH에 prepend하지 않으면 평가 안정성 차이로 수치가 달라질 수 있음(shopping 주석 수정, wishlist eval 수정 포함).
  6. browsergym==0.14.1 고정을 어기면 reward 계산이 달라짐(pyproject 주석 경고).
  7. 순차 스트리밍 세팅이므로 **태스크 순서가 결과에 영향** — 메모리 풀이 run별로 축적되므로 병렬 실행/재시작 시 상태 관리 주의.

---

# 3. 두 시스템 비교 요약

| 항목 | Nemori | ReasoningBank |
|---|---|---|
| 대상 | 장기 대화(멀티턴) 메모리 | 에이전틱 태스크(웹/SWE) 실행 메모리 |
| 메모리 단위 | Episode(제목+3인칭 서사) + Semantic fact | Memory item(title/description/content) |
| 핵심 메커니즘 | 경계탐지 세그멘테이션 + 예측오차(surprise) 증류 | LLM-judge 성공/실패 판정 + 양방향 증류 |
| 실패 활용 | 해당 없음(대화에 성공/실패 개념 부재; "예측 실패"가 유사 역할) | **핵심 차별화** — 실패에서 예방 교훈 |
| 스케일링 축 | 없음(단일 패스) | MaTTS(parallel self-contrast / sequential self-refine) |
| 검색 | cosine, k=10 episodic + 20 semantic | cosine, k=1 (많으면 오히려 악화) |
| Consolidation | Episode merging(LLM 판단) 있음 | 단순 append(의도적) |
| 저장소 | PostgreSQL + Qdrant(프로덕션급) | JSON 파일(연구 프로토타입) |
| 라이선스 | MIT | Apache-2.0 |
| 코드 성숙도 | 활발한 유지보수, Docker 배포 가능 | "not for production" 명시 데모 |
| MCP | 없음 | 없음 |
| 벤치마크 | LoCoMo 73.0/80.8, LongMemEval_S 64.2/74.6 | WebArena 48.8/53.9/46.3, SWE-V 38.8/57.4 |
| Backbone | gpt-4o-mini, gpt-4.1-mini | Gemini-2.5-flash/pro, Claude-3.7-sonnet |

**시스템 설계 관점의 시사점**: 두 시스템은 상호보완적이다. Nemori는 "무엇을 기억할 가치가 있는가"(distillation-time utility assessment)를, ReasoningBank는 "어떤 형태의 기억이 행동을 개선하는가"(strategy-level reasoning content + 실패 학습)를 다룬다. ReasoningBank 논문 스스로 Appendix D에서 episodic traces(Nemori류)와의 결합을 미래 방향으로 명시하고 있어, "Nemori식 에피소드 분할/예측오차 증류를 하부 레이어로, ReasoningBank식 전략 item + MaTTS를 상부 태스크 레이어로" 조합하는 설계가 자연스럽다.

---

## 참고 소스

- Nemori: [arXiv:2508.03341](https://arxiv.org/abs/2508.03341) / [v4 HTML](https://arxiv.org/html/2508.03341v4) / [ACL 2026](https://aclanthology.org/2026.acl-long.1607/) / [github.com/nemori-ai/nemori](https://github.com/nemori-ai/nemori)
- ReasoningBank: [arXiv:2509.25140](https://arxiv.org/abs/2509.25140) / [OpenReview ICLR 2026](https://openreview.net/forum?id=jL7fwchScm) / [github.com/google-research/reasoning-bank](https://github.com/google-research/reasoning-bank) / [Lanerra/reasoning-bank-slm](https://github.com/Lanerra/reasoning-bank-slm)

## 코드 인용 근거 파일 경로

- `nemori/llm/prompts.py` — BATCH_SEGMENTATION_PROMPT, EPISODE_GENERATION_PROMPT, PREDICTION_PROMPT, EXTRACT_KNOWLEDGE_FROM_COMPARISON_PROMPT, MERGE_DECISION_PROMPT
- `nemori/config.py` — MemoryConfig 기본값
- `nemori/domain/models.py` — Episode/SemanticMemory 데이터클래스
- `WebArena/prompts/memory_instruction.py` — SUCCESSFUL_SI/FAILED_SI/PARALLEL_SI/SEQUENTIAL_PROMPT
- `WebArena/run.sh`, `SWE-Bench/run.sh` — 실행 스크립트/환경 변수
- 양 저장소의 `pyproject.toml` — 의존성/버전 고정
