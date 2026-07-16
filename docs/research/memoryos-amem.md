# Agentic Memory 시스템 심층 조사 보고서: MemoryOS vs A-Mem

조사 기준일: 2026-07-16. 출처: arXiv 원문(HTML), GitHub 공식 리포(raw 소스 코드 직접 분석), GitHub API(이슈/메타데이터), PyPI, 웹 검색.

---

# 1. MemoryOS — "Memory OS of AI Agent"

- **논문**: arXiv 2506.06326 (Jiazheng Kang, Mingming Ji, Zhe Zhao, Ting Bai — BUPT BAI-LAB), **EMNLP 2025 Main Conference Oral** (ACL Anthology: 2025.emnlp-main.1318)
- **리포**: github.com/BAI-LAB/MemoryOS — ★1,513 / fork 152 (2026-07 기준), 최근 push 2026-07-07 (활발히 유지보수 중)
- **라이선스**: Apache 2.0 / **PyPI**: `memoryos-pro` v0.1.0 / Docker: `ghcr.io/bai-lab/memoryos:latest`
- 2026-01-15에 BAI-LAB이 AI Memory Survey 논문도 발표. README 기준 병렬화로 "5x faster" 업그레이드 공지.

## A. 핵심 개념

**문제의식**: LLM의 고정 context window와 빈약한 memory 관리로 인한 long-term memory 부재 + personalization 한계. OS의 메모리 계층(페이지, 세그먼트, 힙)을 차용해 해결.

**아키텍처**: 4개 모듈 — Storage / Updating / Retrieval / Generation. 저장소는 3계층:

1. **STM (Short-Term Memory)**: dialogue page 단위 저장. page = `{query, response, timestamp, meta_chain(meta_info)}`. 연속 페이지 간 의미적 연속성을 LLM으로 판정해 dialogue chain을 구성(주제가 바뀌면 chain 리셋). 용량 초과 시 **FIFO**로 MTM으로 방출.
2. **MTM (Mid-Term Memory)**: segmented paging. 방출된 page 배치를 **multi-topic summary**로 주제별 분할 후, 기존 segment(session)와의 유사도 `F_score = cos(e_s, e_p) + F_Jaccard(K_s, K_p)`가 **θ=0.6** 이상이면 해당 segment에 병합, 아니면 신규 segment 생성.
   - **Heat 공식**: `Heat = α·N_visit + β·L_interaction + γ·R_recency`, α=β=γ=1.
     - N_visit: 검색 hit 횟수, L_interaction: segment 내 page 수, R_recency: 지수 시간감쇠 `exp(−Δt/μ)` (논문 μ=1e7초; **코드에서는 `RECENCY_TAU_HOURS = 24`**)
   - Heat ≥ **τ=5** 인 segment는 LPM 반영(profile/knowledge 분석)이 트리거되고 분석 후 `N_visit=0, L_interaction=0`으로 리셋. 용량 초과 시 **LFU eviction** (`evict_lfu()`).
3. **LPM (Long-Term Personal Memory)**: 이중 persona 구조.
   - User: 정적 profile(덮어쓰기 대신 timestamp 붙여 append/merge) + knowledge base(**FIFO deque, max 100**) + **90차원 trait vector**(성격/성향/관심사 — GitHub issue #59에서 "왜 90차원?"이라는 질문이 미해결로 남아있음)
   - Agent: 고정 role setting + 동적 assistant knowledge(FIFO deque, max 100)

**Retrieval(질의 시)**: STM 전체 page + MTM에서 top-5 segment → 그 안에서 top page 선별 + LPM에서 user/assistant knowledge top-k. 모두 dense embedding cosine 유사도. 3개 소스를 `ThreadPoolExecutor(max_workers=3)`로 **병렬 검색**.

## B. 공식 코드 분석

**리포 구조** (동일 코어가 4벌 복제되어 있음 — 주의):

```
MemoryOS/
├── memoryos-pypi/        # PyPI 배포판(코어): memoryos.py, short_term.py, mid_term.py,
│                         #   long_term.py, retriever.py, updater.py, prompts.py, utils.py, test.py
├── memoryos-chromadb/    # ChromaDB 백엔드판 (+ storage_provider.py)
├── memoryos-mcp/         # MCP 서버: server_new.py, config.json, mcp.json + memoryos/ 코어 복제
├── memoryos-playground/  # Flask 데모 (memdemo/app.py)
├── eval/                 # 재현 스크립트: evalution_loco.py, main_loco_parse.py,
│                         #   retrieval_and_answer.py, dynamic_update.py, locomo10.json(데이터 포함)
├── Dockerfile, Paper-MemoryOS.pdf, README.md, readme_cn.md
```

**핵심 클래스와 기본 하이퍼파라미터** (`memoryos-pypi/memoryos.py`의 `Memoryos.__init__` 실측값):

| 파라미터 | 기본값 |
|---|---|
| `short_term_capacity` | **10** (deque maxlen) |
| `mid_term_capacity` | **2000** |
| `long_term_knowledge_capacity` | **100** |
| `retrieval_queue_capacity` | **7** |
| `mid_term_heat_threshold` (`H_PROFILE_UPDATE_THRESHOLD`) | **5.0** |
| `mid_term_similarity_threshold` | **0.6** |
| `llm_model` | `"gpt-4o-mini"` |
| `embedding_model_name` | `"all-MiniLM-L6-v2"` (bge-m3 지정 시 `use_fp16:True` 자동, Qwen3-Embedding 지원) |

- `mid_term.py`: `HEAT_ALPHA=1.0, HEAT_BETA=1.0, HEAT_GAMMA=1, RECENCY_TAU_HOURS=24`, `compute_segment_heat()`, min-heap(음수 heat)으로 hottest segment 관리, `evict_lfu()`, `insert_pages_into_session()` (유사도 임계 0.6).
- `retriever.py` `retrieve_context()` 기본값: `segment_similarity_threshold=0.1, page_similarity_threshold=0.1, knowledge_threshold=0.01, top_k_sessions=5, top_k_knowledge=20`.
- `long_term.py`: FAISS **IndexFlatIP** + 벡터 정규화(=cosine), threshold 0.1, top-k 5, deque 100. 빈 문자열/"none" 류 지식 거부 검증 포함.
- `short_term.py`: `deque(maxlen=max_capacity)`, `is_full()`, `pop_oldest()`, JSON 영속화, 스레드 락.

**스토리지 백엔드**: 코어(pypi판)는 **JSON 파일 영속화 + 인메모리 FAISS 인덱스**. `memoryos-chromadb/`판은 ChromaDB(`storage_provider.py`). 스레드 락으로 동시성 보호(최근 커밋에서 race condition 수정 다수).

**LLM 호출 지점** (write 경로가 무겁고 계단식):

- `add_memory()` → STM 가득 찰 때 `Updater.process_short_term_to_mid_term()`:
  1. `check_conversation_continuity()` — page당 1회
  2. `generate_page_meta_info()` — chain meta 생성/전파 page당 1회
  3. `gpt_generate_multi_summary()` — 방출 배치당 1회 (주제별 요약 JSON)
- Heat ≥ 5.0 트리거 시: `gpt_user_profile_analysis()` + `gpt_knowledge_extraction()` — 2회 (ThreadPoolExecutor 2-worker **병렬**)
- `get_response()`: 최종 생성 1회 (`temperature=0.7, max_tokens=1500`)
- 논문 자체 집계: **응답당 평균 4.9 LLM calls**

**의존성**(`memoryos-pypi/requirements.txt`): numpy==1.24.*, sentence-transformers==5.0.0, transformers>=4.51.0, FlagEmbedding>=1.2.9, **faiss-gpu>=1.7.0**, openai, httpx[socks], flask, python-dotenv.

**MCP 서버**: **공식 존재** (`memoryos-mcp/server_new.py`, `python server_new.py --config config.json`). 툴 3종: `add_memory`, `retrieve_memory`, `get_user_profile`. Claude Desktop/Cline/Cursor 연동 문서화됨.

## C. 성능 특성

**LoCoMo (GPT-4o-mini), F1 / BLEU-1** — 논문 Table 2 (A-Mem\*는 저자들의 A-Mem 재현치):

| Method | Single-hop | Multi-hop | Temporal | Open-domain |
|---|---|---|---|---|
| TiM | 16.25/13.12 | 18.43/17.35 | 8.35/7.32 | 23.74/22.05 |
| MemoryBank | 5.00/4.77 | 9.68/6.99 | 5.56/5.94 | 6.61/5.16 |
| MemGPT | 26.65/17.72 | 25.52/19.44 | 9.15/7.44 | 41.04/34.34 |
| A-Mem (보고치) | 27.02/20.09 | 45.85/36.67 | 12.14/12.00 | 44.65/37.06 |
| **A-Mem\* (재현치)** | 22.61/15.25 | 33.23/29.11 | 8.04/7.81 | 34.13/27.73 |
| **MemoryOS** | **35.27/25.22** | 41.15/30.76 | **20.02/16.52** | **48.62/42.99** |

평균 +49.11% F1, +46.18% BLEU-1 (vs 차상위). Multi-hop에서만 A-Mem 보고치가 우위인 점 주목.

**LoCoMo (Qwen2.5-3B)** — 소형 모델 증거 (F1):

| Method | Single-hop | Multi-hop | Temporal | Open-domain |
|---|---|---|---|---|
| TiM | 4.37 | 2.54 | 6.20 | 6.35 |
| MemoryBank | 3.60 | 1.72 | 6.63 | 4.11 |
| MemGPT | 5.07 | 2.94 | 7.04 | 7.26 |
| A-Mem | 12.57 | **27.59** | 7.12 | 17.23 |
| **MemoryOS** | **23.26** | 21.44 | **10.18** | **26.23** |

3B에서도 동작하며 프레임워크 이득이 유지됨(단 multi-hop은 A-Mem이 우세).

**GVD 벤치마크** (10일치 대화, 15명 유저; **DeepSeek-R1이 judge**로 자동 채점 — Accuracy(0/1), Correctness(0/0.5/1), Coherence(0/0.5/1)):

- GPT-4o-mini: MemoryOS **93.3/91.2/92.3** vs A-Mem 90.4/86.5/91.4, MemGPT 87.9/83.2/89.6, TiM 84.5/78.8/90.8, MemoryBank 78.4/73.3/91.2
- Qwen2.5-7B: MemoryOS **91.8/82.3/90.5** vs A-Mem 87.2/79.5/87.8, MemGPT 85.1/80.2/86.9

**효율 (Table 3)**: MemoryOS **3,874 tokens / 4.9 calls / F1 36.23** vs MemGPT 16,977/4.3/29.13, A-Mem\* 2,712/**13.0**/26.55, TiM 1,274/2.6/18.01, MemoryBank 432/3.0/6.84. 즉 MemoryOS는 토큰은 중간·호출 수는 적당·정확도 최고 포지션. 쓰기(write)는 STM 방출 시점에 몰리는 **배치성 sync 처리**이며, LPM 갱신·retrieval은 ThreadPool 병렬화(README에서 5x 속도 개선 언급). Ablation: **MTM 제거 시 성능 하락 최대**, LPM 제거가 그다음, dialogue chain 기여는 미미.

## D. 재현 관점

- **필요물**: `eval/` 디렉토리에 LoCoMo 파이프라인 완비(`evalution_loco.py`, `locomo10.json` 데이터 동봉, `main_loco_parse.py`, `retrieval_and_answer.py`). LoCoMo는 F1/BLEU-1 문자열 매칭이라 judge 불필요, GVD는 DeepSeek-R1 judge API 필요. GPU는 embedding(bge-m3/faiss-gpu)용으로 1장이면 충분, 나머지는 OpenAI API 비용.
- **알려진 함정 (GitHub issues)**:
  - #51 "LoCoMo 재현 파라미터 설정?" / #42 "재현 결과가 좋지 않음" — 재현 파라미터 문서화 부족
  - #50 **eval에 쓰인 prompt와 `prompts.py`의 prompt가 불일치** (closed이나 재현 시 확인 필수)
  - #49 Table 3의 token/call 집계 방식 질문(open), #44 MemGPT baseline 재현 방법 미문서화(open)
  - #59 90차원 trait vector의 근거 불명(open)
  - 동시성: #55/#73 race condition 락 수정, #65 STM 만석 시 silent data loss 수정 — **구버전 사용 시 데이터 유실 버그 존재했음**
  - 보안: playground의 path traversal(CWE-22) 다수, 하드코딩 API key(#70) — 데모 코드 품질 주의
  - 코어 코드가 4벌 복제(pypi/chromadb/mcp/playground)되어 **버전 간 드리프트** 위험(#73이 실제 사례)
- 커뮤니티: Milvus 백엔드 제안(#75), litellm provider PR(#74), PostgreSQL 문의(#48).

---

# 2. A-Mem — "A-MEM: Agentic Memory for LLM Agents"

- **논문**: arXiv 2502.12110 (Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang — Rutgers agiresearch), v1 2025-02-17 → **v11 2025-10-08**까지 개정. **NeurIPS 2025 채택**.
- **리포 3개 체제** (혼동 주의):
  - `WujiangXu/A-mem` (구 WujiangXu/AgenticMemory): **논문 재현 전용** ★919, 최근 push 2026-03-05
  - `agiresearch/A-mem`: 라이브러리판 ★1,113 / fork 119, MIT, 최근 push 2025-12-12
  - `WujiangXu/A-mem-sys`: **"공식 시스템"판** ★371, openai/ollama/**sglang/openrouter** 백엔드 지원, push 2026-03-15
- **라이선스**: MIT / PyPI 패키지명 `agentic-memory` v0.0.1 (pyproject.toml)

## A. 핵심 개념

**문제의식**: 기존 memory 시스템은 저장/검색은 되지만 **조직화(organization)가 고정적**이라 태스크 다양성에 적응 못 함(graph DB 방식도 사전 정의 스키마에 종속). **Zettelkasten**(카드 노트 + 자유 링크) 방식으로 LLM 자신이 메모리를 조직하게 함.

**Memory note 구조** (m_i, 7요소): `c_i`(원본 내용), `t_i`(timestamp), `K_i`(LLM 생성 keywords), `G_i`(LLM 생성 tags), `X_i`(LLM 생성 context 한 줄 설명), `e_i`(embedding), `L_i`(링크 집합). 임베딩은 **`e_i = f_enc[concat(c_i, K_i, G_i, X_i)]`** — 메타데이터를 임베딩에 포함시키는 것이 특징.

**3단계 write 파이프라인**:

1. **Note construction (프롬프트 Ps1)**: content+timestamp를 넣고 JSON으로 keywords(≥3, 중요도순, 화자명/시간 제외), context(주제·논점·대상을 한 문장), tags(≥3, 도메인/형식/유형) 생성.
2. **Link generation (Ps2)**: 신규 노트 임베딩과 전체 메모리의 cosine 유사도로 **top-k 이웃**을 뽑고, LLM이 `L_i ← LLM(m_n ‖ M_near ‖ Ps2)`로 연결 여부 판단.
3. **Memory evolution (Ps3)**: `m_j* ← LLM(m_n ‖ M_near\m_j ‖ m_j ‖ Ps3)` — 이웃 노트의 context/tags를 **재작성**할지 결정. JSON 출력: `should_evolve(bool)`, `actions(["strengthen","update_neighbor"])`, `suggested_connections`, `tags_to_update`, `new_context_neighborhood`, `new_tags_neighborhood`. (논문 서술상의 merge/prune은 코드에는 strengthen/update_neighbor만 구현.)

**Retrieval(질의 시)**: 순수 dense 검색. `e_q = f_enc(q)` → cosine top-k. **그래프 다중 홉 순회 없음** — 링크는 `search_agentic()`에서 검색된 노트의 `links`를 따라 이웃을 k까지 추가하는 1-hop 보강 수준. 읽기 경로에 LLM 호출 0회.

## B. 공식 코드 분석

**agiresearch/A-mem 구조**:

```
agentic_memory/
├── memory_system.py   (32.7KB) — MemoryNote, AgenticMemorySystem, 프롬프트 전문
├── retrievers.py      (10.7KB) — ChromaRetriever, PersistentChromaRetriever, CopiedChromaRetriever
└── llm_controller.py  (3.8KB)  — LLMController, OpenAIController, OllamaController
examples/sovereign_memory.py, tests/ (pytest), pyproject.toml, requirements.txt
```

**핵심 클래스/기본값** (소스 실측):

- `MemoryNote(content, id=uuid, keywords, links, retrieval_count=0, timestamp, last_accessed, context, evolution_history, category, tags)`
- `AgenticMemorySystem(model_name='all-MiniLM-L6-v2', llm_backend='openai', llm_model='gpt-4', evo_threshold=100)`
  - `add_note()` → `analyze_content()`(LLM 호출 1: Ps1) → `process_memory()`: `find_related_memories(note.content, k=5)`(임베딩 검색) + evolution 판정(LLM 호출 1: Ps3, **OpenAI strict json_schema 강제**) → ChromaDB에 문서 추가
  - evolution 발생 횟수가 `evo_threshold=100`의 배수가 될 때마다 `consolidate_memories()` — ChromaDB 컬렉션 전체 재구축
- `ChromaRetriever(collection_name="memories", model_name="all-MiniLM-L6-v2")` — chromadb + `SentenceTransformerEmbeddingFunction`, `search(query, k=5)`
- `LLMController`: OpenAI(`temperature=0.7, max_tokens=1000`, system prompt "You must respond with a JSON object.") 또는 Ollama(litellm 경유)

**스토리지**: **ChromaDB** (기본 in-memory `chromadb.Client`; `PersistentChromaRetriever`는 `~/.chromadb` 영속화) + 파이썬 dict `self.memories`(노트 원본은 사실상 인메모리 — 프로세스 종료 시 노트 객체 자체는 pickle 안 하면 유실). 임베딩: **all-MiniLM-L6-v2 (384-dim)** 고정에 가까움.

**LLM 호출 수**: **write당 2회** (note 구성 1 + evolution 판정 1; 이웃별 개별 호출이 아니라 이웃 5개를 한 프롬프트에 넣어 일괄 판정), **read당 0회**. MemoryOS 논문의 효율 표에서 A-Mem이 13.0 calls로 집계된 것은 QA 파이프라인 전체(주기적 consolidation 포함) 기준으로 보임 — 집계 기준 차이에 유의.

**의존성**: sentence-transformers>=2.2.2, chromadb>=0.4.22, rank_bm25>=0.2.2(임포트만 되고 hybrid 경로는 사실상 미사용), nltk, litellm>=1.16.11, scikit-learn, openai, ollama. **공식 MCP 서버 없음** (커뮤니티 Rust 포트 제안 issue #28, Cortex 등 파생 존재).

## C. 성능 특성

**LoCoMo F1 (논문 Table 1)**:

| Method | Multi-hop | Temporal | Open-domain | Single-hop | Adversarial |
|---|---|---|---|---|---|
| GPT-4o-mini: LoCoMo(무기억) | 25.02 | 18.41 | 12.04 | 40.36 | 69.23 |
| GPT-4o-mini: ReadAgent | 9.15 | 12.60 | 5.31 | 9.67 | 9.81 |
| GPT-4o-mini: MemoryBank | 5.00 | 9.68 | 5.56 | 6.61 | 7.36 |
| GPT-4o-mini: MemGPT | 26.65 | 25.52 | 9.15 | 41.04 | 43.29 |
| **GPT-4o-mini: A-Mem** | **27.02** | **45.85** | **12.14** | **44.65** | 50.03 |
| GPT-4o: A-Mem | 32.86 | 39.41 | 17.10 | 48.43 | 36.35 |
| GPT-4o: MemGPT | 30.36 | 17.29 | 12.24 | 60.16 | 34.96 |

- **Temporal reasoning에서 이득 최대** (18.41→45.85). Adversarial은 무기억 baseline(69.23)이 오히려 최고 — 메모리 시스템이 오답 유도 질문에 취약함을 시사.
- **DialSim** (TV쇼 ~350K tokens): A-Mem F1 3.45 / BLEU-1 3.37 / SBERT 19.51 vs LoCoMo 2.55/3.13/15.76, MemGPT 1.18/1.07/8.54 (절대값 자체가 매우 낮음).
- **Ablation (GPT-4o-mini F1, Multi-hop)**: w/o LG&ME 9.65 → w/o ME 21.35 → full 27.02. Link Generation이 대부분의 이득, Evolution이 추가 보정.
- **백본 6종**: GPT-4o(-mini), Qwen2.5-1.5B/3B, Llama3.2-1B/3B 등 + 부록에서 DeepSeek-R1-32B, Claude 3/3.5 Haiku. **1B~3B 소형 모델에서도 동작**이 핵심 셀링포인트 — 단, MemoryOS 논문의 교차 검증에서 Qwen2.5-3B일 때 A-Mem 27.59(MH)로 multi-hop만 강하고 나머지는 급락. 소형 모델은 Ps3의 strict JSON(중첩 배열 길이 제약)을 자주 못 지켜 evolution이 무력화되는 실패 모드 예상(코드에 JSONDecodeError 시 조용히 skip하는 fallback 존재).
- **하이퍼파라미터**: 논문 실험은 top-k ∈ {10,20,30,40,50} — k 증가 시 개선 후 정체/하락. 코드 기본값은 k=5.
- **비용/규모**: 메모리 op당 ~1,200 tokens (baseline 16,900 대비 85-93% 절감), op당 <$0.0003. 스토리지 선형: 1K notes=1.46MB → 1M=1,464MB, 검색 0.31μs→3.70μs(임베딩 인덱스만의 시간).
- **Latency 주의점**: write가 동기 LLM 2회라 대화 루프에 붙이면 턴당 지연 큼(issue #21 "왜 이렇게 느린가"). `evo_threshold`마다 전체 컬렉션 재구축도 스파이크 유발.

## D. 재현 관점

- **필요물**: 재현은 `WujiangXu/A-mem` 리포(README에 명시: "이 리포는 논문 재현 전용"). LoCoMo(7,512 QA, 5 카테고리) + DialSim 데이터, F1/BLEU-1/ROUGE/METEOR/SBERT 메트릭(judge LLM 불필요), OpenAI API 비용 소액. 임베딩은 CPU로도 가능(MiniLM).
- **알려진 버그/함정 (agiresearch/A-mem issues — 재현 신뢰성에 직결)**:
  - **#10 "Absence of LLM for note elements"**: 한때 라이브러리판에서 note 구성 시 LLM을 실제로 호출하지 않는 상태였음(#13에서 수정) — 버전에 따라 논문 방법과 코드가 불일치했음
  - **#24/#23**: ChromaDB 컬렉션이 **cosine이 아닌 L2 distance**로 생성됨 + search score의 의미가 반전(거리를 유사도로 오용) — 검색 품질 저하, 논문과 구현 불일치 (open)
  - **#32**: `find_related_memories()`가 memory ID가 아니라 **결과 순위 인덱스**를 반환 → evolution이 **엉뚱한 노트를 갱신**할 수 있는 버그 (open)
  - #7 `search()` 내 동일 `retriever.search` 중복 호출, #9 `model_name` 파라미터 미사용이었음(closed), #11 ID 검증 부재, #14 retriever 중복
  - #16 "Is this project dead?" — agiresearch판 유지보수 정체(2025-12 이후 push 없음); 실사용은 A-mem-sys 권장
- **논란/교차 검증**: (1) MemoryOS 논문의 재현치 **A-Mem\***가 보고치 대비 전 카테고리에서 유의하게 낮음(예: MH 45.85→33.23). (2) **Mem0 논문**(arXiv 2504.19413)도 LLM-judge 기반으로 A-Mem을 독립 평가하며 다른(낮은) 수치 보고. (3) alphaXiv에서 baseline(Mem0) 특성화가 피상적이라는 공개 비판 코멘트. (4) 2026년 벤치마크 리뷰 글들에서 "A-Mem의 85-93% 절감은 태스크 비용이 아닌 op당 토큰 기준"이라는 지적. → **보고 수치의 재현 가능성은 보수적으로 봐야 함**.

---

# 3. 시스템 설계 관점 종합 (요약 비교)

| 축 | MemoryOS | A-Mem |
|---|---|---|
| 조직화 원리 | OS식 계층 (STM→MTM→LPM), 구조 고정 | Zettelkasten 평면 노트 + LLM이 링크/진화 결정 |
| write 비용 | 배치성 (STM 방출 시 page당 1-2 + 배치 1 + heat 트리거 시 2) | 노트당 고정 2 LLM calls (동기) |
| read 비용 | LLM 1 (생성) + 3-way 병렬 임베딩 검색 | LLM 0 + 단일 cosine top-k |
| 스토리지 | JSON+FAISS (ChromaDB판 별도) | ChromaDB (기본 in-memory) |
| 임베딩 기본값 | all-MiniLM-L6-v2 (bge-m3/Qwen3 지원) | all-MiniLM-L6-v2 고정에 가까움 |
| 강점 | temporal/single-hop, 사용자 profile 지속성, MCP 공식 지원 | multi-hop (링크 덕), 소형 모델(1-3B) 실험 커버리지, 구조 단순 |
| 약점 | 코드 4벌 복제 드리프트, 재현 파라미터 문서화 부족 | L2/cosine 버그·인덱스 버그 등 구현-논문 불일치, 수치 재현 논란 |
| 소형 모델 증거 | Qwen2.5-3B/7B 본문 실험 (프레임워크 이득 유지) | 1B-3B 백본 실험 다수 (단 strict JSON evolution이 병목 위험) |
| 유지보수 (2026-07) | 활발 (push 2026-07, EMNLP'25 Oral) | 재현 리포만 유지, 라이브러리판 정체 (NeurIPS'25) |

**설계 문서용 핵심 시사점**:

1. 두 시스템 모두 검색은 단순 dense cosine + top-k이며 차별화는 전적으로 **write-path의 LLM 기반 조직화**에 있음.
2. MemoryOS의 heat 기반 승격(방문수+길이+recency 감쇠, τ=5)은 저비용 휴리스틱으로 이식 가치가 높고, A-Mem의 evolution은 이득 대비 버그 표면적과 지연이 커서 이웃 일괄 판정(1 call) 형태로만 채택할 만함.
3. 양쪽 모두 벤치마크 수치는 상호 재현 실패 사례(A-Mem\*, Mem0 재평가)가 있으므로 자체 재현 시 LoCoMo F1/BLEU 파이프라인을 고정하고 eval prompt 버전(MemoryOS issue #50)까지 핀 고정해야 함.

---

## 주요 출처

- MemoryOS 논문: https://arxiv.org/abs/2506.06326 (HTML: https://arxiv.org/html/2506.06326v1)
- ACL Anthology (EMNLP 2025 Oral): https://aclanthology.org/2025.emnlp-main.1318/
- MemoryOS 리포: https://github.com/BAI-LAB/MemoryOS
- A-Mem 논문: https://arxiv.org/abs/2502.12110 (HTML: https://arxiv.org/html/2502.12110)
- A-Mem 리포: https://github.com/agiresearch/A-mem / https://github.com/WujiangXu/A-mem / https://github.com/WujiangXu/A-mem-sys
- Mem0 논문 (교차 검증): https://arxiv.org/pdf/2504.19413
