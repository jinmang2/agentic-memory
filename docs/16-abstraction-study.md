# 추상화 스터디 — 논문 원문 ↔ upstream ↔ 우리 코드 3자 대조 (2026-07-27)

목적: 방법론이 8종+로 늘어나며 "초기에 생각한 논문 구현 형태와 거리가 생겼는가"를 검증한다.
방식: 우리 기존 기록(docs/08~15, research/*)을 **경유하지 않고**, 논문 원문(arXiv HTML)·
upstream 클론(`~/.agmem/upstream/`)·우리 코드를 세션마다 직접 읽어 대조한다.
각 세션은 ① 논문의 자기 명명(verbatim) ② 3자 대조표 ③ 발견 ④ 추상화 평가 순서.

거리가 생기는 방식은 세 종류로 분류한다:
- **훅 사영**: 논문 라이프사이클을 `Organizer` 훅 10개로 쪼개 넣은 것
- **mechanism vs policy 분리**: 방법론 일부를 organizer가 아닌 policy wrapper로 뺀 것 (docs/04 §1.1)
- **facade 소유권 이전**: 직접 store 쓰기 → `MemoryOp` 선언 + facade 적용 (docs/04 §2)

---

## 세션 1: A-Mem (arXiv:2502.12110) — 기록 기반 (원문 재페치 생략, docs/13이 원문 대조 완료본)

- 논문 Ps1(Note Construction)/Ps2(Link Generation)/Ps3(Memory Evolution)/Read(§3.4) ↔
  `organizers/amem/organizer.py`의 `on_message` 내 2 LLM콜 + `retrieval/steps.py::LinkExpansion`.
- 매핑은 1:1. 논문 세 가지 쓰기 효과 = ADD/LINK/UPDATE op 정확 대응. 훅은 `on_message` 하나만 사용.
- 우리가 논문 쪽으로 교정한 것: 이웃 검색 쿼리 = `embedding_text()`(식 (3) 충실; 양쪽 공식 코드는 raw
  content), 이웃 ID 지칭(#32), 진짜 cosine(#23/#24). upstream을 따라간 것: 단방향 링크,
  add-then-evolve 2-op 분리.
- 잔여 편차(라운드 12에서 "1건"이 과소집계로 판명, 실제 집합): ① read 링크 확장 cap 의미론 —
  upstream은 히트당 k개(k=10이면 ~100, #16/#21 스스로도 미확정), 우리는 전역 5개
  (`link_expansion_cap`) + 중복 링크 dedup(upstream은 중복을 두 번 서빙하며 cap 슬롯 소모);
  ② 진화된 이웃의 즉시 재임베딩 — upstream은 `consolidate_memories()`까지 stale 인덱스
  (fidelity-deep-audit §1.1); ③ 첫 노트 evolution 콜 스킵 — robust 계보를 따름, plain
  (발표수치) 대비 대화당 콜 -1. 라운드 12에서 편차였다가 해소된 것: LINK 순서(sorted-set →
  upstream의 삽입순서+중복 보존), 새 노트 tag 가드(무조건 적용으로 plain 재현).
- 평가: **추상화가 가장 잘 맞는 케이스.** 비용은 "구현이 write(organizer)/read(step) 두 파일로
  갈라진 것"뿐. `produces`→`default_memory_types` 다리는 A-Mem에선 무해.

## 세션 2: Nemori (arXiv:2508.03341) — 전면 직접 대조

논문 자기 명명(현행판): §3.2 **Episodic Memory Integration**(3.2.1 Local Message Partitioning /
3.2.2 Narrative Episode Generation / 3.2.3 Associative Memory Integration), §3.3 **Semantic
Knowledge Distillation**(3.3.1 Anticipatory Schema Synthesis / 3.3.2 Prediction Error
Distillation / 3.3.3 Agnostic Knowledge Consolidation), §3.4 Response Generation(τ=0.70).
원리: Structure/Representation/Distillation Prior.

| 논문 | upstream (`~/.agmem/upstream/nemori`) | 우리 |
|---|---|---|
| 3.2.1 Partitioning | `BatchSegmenter.segment` + `BATCH_SEGMENTATION_PROMPT` | `BatchPartitioner`(v4·upstream) / `PerMessageBoundary`(v1 기본) |
| 3.2.2 Narrative Gen | `EpisodeGenerator` + `EPISODE_GENERATION_PROMPT` | `EPISODE_PROMPT`, `_flush_segment` 1단계 |
| 3.2.3 Assoc. Integration | `merger.check_and_merge` (hard delete 후 재저장) | `EpisodeMerger` → **MERGE op + supersedes** |
| 3.3.1 Anticipatory Schema | `SemanticGenerator._prediction_correction` 1단계 (title+지식) | `PREDICT_PROMPT` (동일 입력) |
| 3.3.2 Prediction Error | 같은 함수 2단계, **raw source_messages** 대조 | CALIBRATE 단계, raw segment 대조 (동일) |
| 3.3.3 Consolidation | **없음** (append-only, `ON CONFLICT(id)`뿐, merge/conflict grep 0건) | `ThreeWayIntegrator`/`DedupIdReuseIntegrator` |
| §3.4 dual-mode 반환 | `UnifiedSearch` + search.py raw 부착 | type-키 read step + `AttachSources(top_r=2)` |

라이프사이클: upstream `add_messages`(버퍼, `buffer_size_min=2`, asyncio fire-and-forget) /
`flush()` ↔ 우리 `on_message` / `flush_buffer` 1:1. semantic은 upstream 동기 인라인
(`_on_episode_created`) = 우리 기본; `consolidation="semantic_offline"`은 우리 확장.

발견:
1. **기본 프리셋 ≠ 현행 논문**: no-arg `fidelity="v1"`은 구판 per-message f_θ. 현행 arXiv §3.2.1은
   배치 분할. 의도된 계보 보존이나 API 표면에서 안 보임.
2. **논문 §3.3.3은 upstream 배포 코드에 부재** — 구현체는 우리 `ThreeWayIntegrator`가 유일.
   (A-Mem "Ps1 사멸"과 같은 논문-코드 괴리 유형.)
3. 병합: upstream hard delete vs 우리 MERGE+supersedes(파생상태 retire 전파) — 의미 동일,
   감사성·체이닝 안전성 상위 호환.

평가: `flush_buffer`/`consolidate`/MERGE/2-type `produces` 첫 가동, 접힘 손실 없음. 개선 후보는
표면: stages 클래스 docstring에 논문 §번호 병기, fidelity 프리셋의 판본 계보 가시화.

## 세션 3: MemoryOS (arXiv:2506.06326) — 전면 직접 대조

논문 자기 명명: 4모듈 **Memory Storage / Updating / Retrieval / Response Generation**(§3.1).
STM(dialogue page {Q,R,T}, dialogue chain) / MTM(segment, F_score=cos+F_Jaccard, θ=0.6) /
LPM(User Persona{Static profile, User KB, User Traits} + Agent Persona). Updating: STM→MTM
"dialogue-chain-based FIFO", MTM→LPM Heat=α·N_visit+β·L_interaction+γ·R_recency (α=β=γ=1,
R=exp(−Δt/μ), μ=1e7, τ=5), 퇴출 "segments with the lowest heat are evicted"(§3.3).

upstream은 5벌 체제(pypi/chromadb/mcp/playground/eval). 대응: `ShortTermMemory.add_qa_pair` ↔
페이지 조립+`recent_context()`; **paper "segment" = 코드 "session"**(`add_session`);
`Updater.process_short_term_to_mid_term` ↔ `_evict_to_mtm`+`TOPIC_PROMPT`;
`compute_segment_heat`→스레드 병렬 profile/knowledge 추출 ↔ `_promote_to_lpm`;
`search_sessions`의 N_visit 증가 ↔ `on_retrieval` 훅.

발견:
1. **논문 상수 ≠ 논문 수치의 상수** (1차 소스 확정): 논문 α=β=γ=1·μ=1e7 / pypi (1,1,1)+24h 감쇠
   (`RECENCY_TAU_HOURS=24`) / eval 하네스 (0.8, 0.8, 0.0001)+stored recency. LoCoMo 표는 eval
   상수의 산물. 우리 프리셋 2종이 이 갈림을 보존.
2. **[결함·미수정] 우리 `MEMORYOS_PRESETS`가 존재하지 않는 퇴출 정책 차이를 기록**:
   pypi=`"lowest_heat"` / eval=`"lfu"`로 두 계보가 다르다고 주장하나, 실측 결과 **양쪽 모두
   `evict_lfu`** (pypi `mid_term.py:177`, eval `mid_term_memory.py:120`; access_frequency 최소
   세션 삭제). "lowest heat 퇴출"은 논문 문장에만 존재. 즉 우리 pypi 프리셋은 논문 정책을
   구현하면서 코드 계보 라벨을 닮. 완화: MTM capacity 2000(eval 하네스도 2000 명시 전달)이라
   벤치마크 규모에서 퇴출 미발화 → 수치 불활성. 조치안: pypi 라벨을 `"lfu"`로 교정하고
   lowest_heat는 "paper" 개념으로 강등 + 모듈 docstring의 "계보 간 퇴출 차이" 문장 삭제.
   유형: audit-defect-classes의 "코드와 어긋난 자기 docstring".
3. upstream LFU는 ingest-only 실행에서 전원 0 → 삽입순 FIFO로 퇴화 (N_visit 불활성 결론의
   퇴출 버전).

평가: 훅 표면 최대 사용(on_message/on_retrieval/recent_context/flush_buffer/retire/patch_unit).
논문의 4모듈 분해가 우리 프레임워크 분해와 동형이라 접힘 손실 없음. 실제 위험은 훅이 아니라
**단위**(page vs message, 2026-07-27 audit B1에서 교정 완료).

## 세션 4: Zep/Graphiti (arXiv:2501.13956) — 전면 직접 대조

논문 자기 명명: 서브그래프 3계층 𝒢_e(episode)/𝒢_s(semantic entity)/𝒢_c(community, label
propagation). 구축: entity 추출(n=4 문맥)→resolution, fact edge, **bi-temporal** T(t_valid,
t_invalid)/T′(t′_created, t′_expired), 모순 시 t_invalid←새 fact의 t_valid. 검색(§3):
**f(α)=χ(ρ(φ(α)))**, φ 3종(cos/bm25/bfs), ρ 5종(RRF/MMR/episode-mentions/node-distance/
cross-encoder). §4.1 BGE-m3 → 발표 수치 operating point는 cross-encoder 레시피.

대응: `extract_nodes`/`resolve_extracted_nodes` ↔ on_message 1-2단계;
`extract_edges`(temporal 통합; **temporal_operations.py는 현 main에서 소멸**) ↔ fact+valid_at/
invalid_at; `resolve_edge_contradictions` ↔ 같은쌍 1콜 → INVALIDATE op;
`community_operations` ↔ `community.py`+`flush_buffer`; `search_config_recipes.py` 프리셋 표 ↔
`search.py::ZEP_SEARCH_RECIPES`(DEFAULT="cross_encoder"); ρ 구현 ↔ `retrieval/{fusion,rerank,
bfs}.py`; φ_bfs ↔ `GraphRecall` step.

발견:
1. **"write는 하나, read는 메뉴"**: read 경로가 데이터(레시피 표)로 출하되는 첫 방법론. upstream
   기본 search()=RRF, 논문 수치=cross-encoder. 한 ingest에 여러 레시피 측정 가능 구조 보존.
2. **계약 위반 전력**: 유일하게 store 직접 쓰던 organizer → replay 시 빈 그래프, GraphRecall
   조용한 RAG 퇴화. 2026-07-27 audit B3에서 op 페이로드(`entity_type`, `subject_id`/`object_id`)
   + facade `_apply_graph`로 교정. 교훈: 새 store 종류 도입 시 계약이 규율로만 지켜짐(구조적
   강제 없음).
3. **bi-temporal의 T′축은 프레임워크가 공짜 제공**: append-only evolution log가 곧 트랜잭션
   타임라인. 방법론은 T축(INVALIDATE+valid_at/invalid_at)만 구현하면 됨.
4. **upstream은 논문을 앞질러 감**: 현 main에 논문에 없는 SagaNode(`summarize_saga`),
   `combined_extraction.extract_nodes_and_edges`(노드+엣지 단일 콜). 우리 포트는 혼합 스냅샷
   (resolution=현행, 나머지=논문 시점), saga 미포트.
5. **`produces` 순서 최초 load-bearing**: ("facts","entities","communities") — GraphRecall의
   dedup이 facts 선순서를 요구 (base.py 주석 명시).
6. GraphRecall 미결 편차 2건(문서화된 open decision): 평평한 점수 append vs 융합 채널;
   RRF+BFS는 upstream 레시피 2개의 혼합. `graph_expansion_cap=0`으로 순수 RRF 도달 가능.

평가: 압박 축은 훅이 아니라 store 종류(graph_store 슬롯)·op 표현력(INVALIDATE)·read 정책화
(레시피 표). 셋 다 수용됐으나 ②는 한 번 부러진 뒤의 수용.

## 세션 5: G-Memory (arXiv:2506.07398) — 전면 직접 대조 (PDF 추출)

논문 자기 명명(§3–4, PDF 직접 추출 — HTML 변환 실패로 pypdf 사용):
3계층 그래프 — **Interaction Graph (Utterance Graph)** 𝒢^(Q)_inter(발화 노드 (A_i, m_i), 시간/영감
엣지) / **Query Graph** 𝒢_query(노드 (Q_i, Ψ_i, 𝒢_inter), Ψ∈{Failed, Resolved}) / **Insight Graph**
𝒢_insight(노드 ι_k=(κ_k, Ω_k) — 내용+supporting query set, hyper-edge (ι_m, ι_n, q_j)).
읽기: §4.1 **Coarse-grained Memory Retrieval**(식4 top-k cosine MiniLM, 식5 **1-hop expansion**),
§4.2 **Bi-directional Memory Traversal** — upward 𝒢_query→𝒢_insight 프로젝터 **Π_Q→I**(식6, Ω_k
교집합), downward **LLM graph sparsifier S_LLM**(식7, core subgraph), role-aware filter **Φ**가
각 agent Mem_i 초기화. 쓰기: §4.3 **Hierarchy Memory Update** — 식9 query 노드+엣지, 식10
**요약함수 J(𝒢_inter, Ψ)** 로 새 insight, 식11 Ω_k ← Ω_k∪{q_new}. 운용점: 1-hop, k∈{1,2}.

upstream(`~/.agmem/upstream/GMemory`, bingreeky/GMemory — **라이선스 파일 없음**): repo는 MAS
벤치마크 하네스이고 G-Memory는 `mas/memory/mas_memory/GMemory.py`의 플러그인 1종
(voyager/memorybank/generative/metagpt/chatdev 베이스라인과 병렬).

| 논문 | upstream 실측 | 우리 |
|---|---|---|
| 𝒢_inter 저장 + S_LLM | `_extract_mas_message` (trajectory condensation — 그래프 아님, 텍스트) | `on_task_end`의 SPARSIFY(`key_steps`/`mistakes`) |
| 𝒢_query + 식4/식5 | `TaskLayer`(networkx) `retrieve_related_task(hop=1)` + Chroma | **임베딩 검색으로 근사 — hop expansion 미구현 (TODO)** |
| Π_Q→I (식6) | `query_insights_with_score`: SUCC 4/FAIL 2 과제 검색 후 연관 insight **카운트 점수화** | 유사 형상 (round-5 상수 준수) |
| Φ role 필터 | `project_insights(role)` | `project_insights(role)` 동일 |
| **J + Ω_k (식10–11)** | **부재.** `finetune_insights` — 정규식(`:793`)으로 **ADD/EDIT/REMOVE/AGREE** 파싱, 점수(ADD 2, EDIT/AGREE +1, REMOVE −1/−3, ≤0 prune), `backward(reward)` +1/−2, FINCH `cluster_tasks`+`_merge_rules` | 코드 쪽을 따름 — 구조화 JSON으로 동일 op, 점수 동일; FINCH 이연, cap은 ADD 억제+soft REMOVE |

발견:
1. **논문-코드 중앙 메커니즘 불일치 1차 소스 재확정**: 논문 §4.3의 J·Ω 형식론은 코드에 없고,
   코드는 Reflexion풍 critique finetune. 우리 포트는 코드를 따르며, organizer docstring이
   "clean-room 아님"을 명시 (라이선스 부재 → 동작·상수만 재현, 원문 미복제).
2. **3계층이 우리 쪽에서 단일 `strategies` 타입으로 평탄화**: 발화 그래프는 upstream도 텍스트로
   평탄화하므로 손실 아님. 그러나 query graph의 1-hop expansion(식5, 논문 민감도 분석까지 있는
   운용점)은 **우리 read 경로에 없음** — 현재 포트 중 가장 큰 fidelity 갭. graph_store가 이미
   있으므로(Zep) 과제 그래프 배선은 가능(TODO 기록 있음).
3. **feedback 소유권 문제의 원인 제공자**: G-Memory의 backward +1/−2가 `strategies` 타입을
   공유하는 ReasoningBank(append-only, feedback 무개념)에 새던 버그가 `on_feedback` 훅을
   organizer 소유로 만든 계기 (base.py 문서화 완료).

평가: 첫 task-loop 방법론(on_task_end/on_retrieval/on_feedback 3종 가동). 훅 사영은 자연스럽고,
긴장은 "그래프 계층 vs 평탄 타입"에 있음. 결론: 갭 2건(hop expansion, task graph)은 추상화의
한계가 아니라 미완 구현 — 프레임워크는 이미 수용 수단을 가짐.

## 세션 6: ACE (arXiv:2510.04618) + ReasoningBank (arXiv:2509.25140) — task-end 계열

### ACE — 논문 자기 명명

3역할 **Generator / Reflector / Curator**(§3). 컨텍스트는 **bullets**(id + helpful/harmful
카운터, §3.1), Reflector가 증류한 **delta contexts**를 Curator가 통합 — **incremental delta
updates**("monolithic rewrite 대체"), 병합은 **deterministic non-LLM**. §3.2 **grow-and-refine**
(새 bullet append + 기존 in-place 갱신 + 임베딩 dedup). 실패 모드 명명: **brevity bias**,
**context collapse**(§2.2). 적응 모드: offline/online, multi-epoch.

| 논문 | upstream(`ace-agent/ace`) | 우리 |
|---|---|---|
| playbook/bullets | **텍스트 표현** `playbook_utils.parse/format_playbook_line` | `Bullet` dataclass + `bullets` 타입 |
| Curator delta ops | `apply_curator_operations` — **ADD-only** | ADD-only 유지 (의도적 동행) |
| grow-and-refine dedup | **opt-in이고 deps 없으면 조용히 스킵** (재현 함정, docs/research/ace-longmemeval.md §D) | 임계 0.90 **상시 ON** |
| helpful/harmful | `update_bullet_counts(bullet_tags)` — Generator가 인용한 bullet에 귀속 | 인용 신호 부재 → `report_feedback()`가 정확 경로, 카운터는 UPDATE op로 감사 가능 |
| read | 전체 playbook 주입 | `get_playbook()` — top-k 검색 금지 계약 (round-5) |

발견: 논문 §3.2가 말하는 풍부한 갱신(dedup 상시, MERGE/DELETE 함의)보다 upstream이 얇음
(ADD-only + 조용한 dedup 스킵) — A-Mem Ps1/Nemori §3.3.3과 같은 "논문>코드" 유형. ACE의
read는 검색이 아니라 **전량 주입**이라 read-step 추상화 밖의 별도 facade 경로(`get_playbook`)가
필요했음 — 읽기 추상화의 첫 예외 사례. Generator의 multi-round reflection은 메모리 레이어가
아닌 agent 루프로 판정 (round-8 §A3 선례).

### ReasoningBank — 논문 자기 명명

**memory item** = (title, description, content)(§3.2 Memory Schema). 파이프라인 3단계
**memory retrieval → memory extraction → memory consolidation**. extraction은
**LLM-as-a-judge**가 success/failure 라벨 후 성공/실패 **다른 추출 전략**; consolidation은
"**a simple addition operation**" — append-only가 논문의 명시적 설계. §3.3 **MaTTS**:
**parallel scaling**(k개 trajectory 간 **self-contrast**) / **sequential scaling**(단일 trajectory
**self-refinement**, 중간 노트도 메모리 신호).

| 논문 | upstream(`google-research/reasoning-bank`) | 우리 |
|---|---|---|
| retrieval (top-k, 시스템 지시 주입) | `memory_management.select_memory` — top-1 **experience** 주입 | `experiences` k=1 + `ExpandExperiences` read step (miss→무주입 의미론 보존) |
| judge + extraction | `pipeline_memory` + `induce_memory`(t=1.0), autoeval 사유를 traj에 append | `on_task_end`: judge(t=0.0)→증류 ≤3, 사유 append 동일 |
| consolidation | 단순 추가 | append-only ADD, no prune/merge (설계 그대로) |
| MaTTS parallel | `induce_scaling`(PARALLEL_SI, t=0.7, ≤5개) | **전용 훅 `on_scaled_task_end`** — 프롬프트/버짓/온도가 달라 별도 훅이 정당 |
| MaTTS sequential | SEQUENTIAL_PROMPT — **agent가 자기 trajectory 재작성** | **의도적 부재** — generator 루프 판정, 정제된 traj는 일반 on_task_end로 유입 |

발견: RB는 append-only가 **논문의 명시적 설계**라 feedback 루프가 없음 — G-Memory와
`strategies` 타입을 공유하다 backward ±점수가 새던 사건이 `on_feedback` organizer-소유화의
직접 원인 (세션 5 발견 3과 동일 사건의 반대편). `on_scaled_task_end`는 우리 훅 중 유일하게
단일 방법론 전용인데, base.py가 그 존재 이유(대조가 독립 메커니즘)를 명시 — 훅 인플레이션
경계 사례로 종합에서 재론.

## 세션 7: MemMachine (arXiv:2604.04853) — organizer+policy 복합의 첫 사례

논문 자기 명명: 계층 **Short-term memory**(§4.3, LLM 요약) / **Long-term memory**(§4.4,
**Sentence Extraction** — NLTK Punkt, 문장 임베딩) / **Profile Memory (Semantic Memory)**(§3.2·
§4.7 — 논문 스스로 별칭 병기). 읽기: **Memory Search**(§4.5.2), **Contextualization**(§4.6 —
"nucleus episode를 이웃으로 확장해 episode cluster 형성"), **Multi-Query Reranking**(§5.5 —
리랭커가 "검색에 쓰인 모든 쿼리의 연결"을 받음), **Retrieval Agent**(§5: ToolSelectAgent /
ChainOfQuery ≤3회 / SplitQuery 2–6분해 / direct leaf). 비용 주장 §6.1 verbatim: "LLMs are NOT
used for per-message fact extraction, memory deduplication, or routine memory management."
발표 수치: LoCoMo 0.9169(agent mode)/0.9123(memory mode), gpt-4.1-mini.

upstream 재검증(`~/.agmem/upstream/MemMachine`, 클론 직독):
- **profile 패키지 부재 확정** — `memmachine_server/`에는 `episodic_memory/{short_term,long_term,
  declarative_memory,event_memory}` + 최상위 `semantic_memory/`. 논문 "Profile Memory" = 코드
  `semantic_memory` = 우리 `profile.py` — **한 물건, 세 이름**.
- **QueryPolicy 사문 재확정**: `grep "policy\." retrieval_agent/` 0건. 6개 필드 전부 관통만 하고
  아무도 안 읽음 → 우리는 live 값 2개(max_attempts, confidence)만 생성자 인자로 (사문 knob
  계열: A-MAC X_train, MemoryOS 사문 키워드 항과 동일 가족).
- SplitQuery "2–6"(논문) vs "1–6"(우리 기록): 코드 프롬프트가 "1-6 lines total (if split: 2-6)"
  — 논문은 split 경우만 서술. **결함 아님**, 논문의 반올림.
- 발표 수치 경로 재확정: eval 하네스는 declarative + STM=None → **write 경로 LLM 0콜**.
  segmenter/deriver는 event 백엔드 비기본 옵션, 발표 수치 무관 (docs/research/memmachine.md
  §1.1 교정 유지).

우리 배치: write=`MemMachineOrganizer`(produces=("derivatives",), declarative 계보, STM은
`recent_context()`) + profile은 `profile.py` + read 제어=`policies/retrieval.py`의
`QueryStrategy` 4종(Direct/Split/ChainOfQuery/ToolSelect, `retrieval/planned.py::PlannedSearch`로
부착 — 호출부 배선 금지) + `MemMachineContextualize` read step(§4.6).

**추상화 평가 — 이 세션의 핵심**: MemMachine은 organizer/policy/read-step/recent_context 네
조각으로 갈라진 첫 방법론인데, 그 절단선이 **논문 자신의 장 구분(§4 Memory vs §5 Retrieval
Agent)과 일치**한다. retrieval agent는 메모리 타입을 소유하지 않고 op를 내지 않고 임의의
search callable 위에서 동작 — policies/ 소속 판정의 operational test를 논문 구조가 스스로
통과시켜 준 사례. mechanism vs policy 분리가 "우리 임의 재분류"가 아니라 방법론 자체의 자연
관절일 수 있음을 보여줌 (종합에서 A-MAC과 함께 재론).

## 세션 8: A-MAC (arXiv:2603.04549) — write-path policy

논문 자기 명명: §3.2 "Interpretable Memory Value Signals" 5종 — **Utility 𝒰 / Confidence 𝒞 /
Novelty 𝒩 / Recency ℛ / Type Prior 𝒯**. 식1(§3.1) S(m)=Σw_i·f_i(m), **admit iff S(m) ≥ θ**.
적합: §3.3 5-fold CV + grid search, θ∈[0.3,0.6], θ*=0.55(Table 3). 동기: Table 1 — A-Mem
recall 1.000/precision 0.371 ("전부 저장, 63%는 인용 안 됨"). §3.2 Type Prior는
"part-of-speech cues" 사용 주장.

공식 repo(`~/.agmem/upstream/amac`, GuilinDev/... 단일 커밋, MIT) 직접 재검증:
- 결함1 재확정: `optimize_weights_cv.py:78` **`novelty.score(memory, [])`** — N≡1.0;
  `recency.py:45` `current_time=time.time()` vs LoCoMo 2023 타임스탬프 — R≡0.0. 5특징 중 3개만 신호.
- 결함3 재확정: `:181` `X_train` 할당, `:185` **`evaluate(X_val)`만 호출** — train fold 사문,
  가중치 선택이 평가 데이터 위에서 수행됨.
- 결함4 방증: 논문은 POS cues 주장, 릴리스는 키워드 substring + `TODO: spaCy` — 결함2(substring
  포화)와 결합해 발표 운용점은 "감탄사가 아니면 admit"으로 환원.
- **신규 확인**: 가중치 벡터 [0.1,0.1,0.1,0.1,0.6]은 **논문에 미공개** (Table 2는 상대 중요도만)
  — 출처는 릴리스 코드. "발표 가중치"라는 우리 표현은 "릴리스 가중치"가 정확.

우리 배치: `policies/admission.py::AdmissionGate` — organizer 앞 wrapper(`AdmissionGated`),
기각 시 host LLM 0콜. 결함 1·2를 교정한 특징 재유도 + 릴리스 운용점도 verbatim 도달 가능
(기본값). 재튜닝은 측정이므로 보류 상태 유지.

추상화 평가: A-MAC은 논문 제목부터 "admission control" — **스스로 정책임을 자인**하는
방법론. 어떤 메모리 타입도 소유하지 않고 임의 organizer 앞에 붙는 wrapper 형태가 논문의
의도(A-Mem 앞 게이트)와 정확히 일치. mechanism vs policy 분리의 write쪽 정당성 사례.

---

# 종합: 추상화 평가 총괄 (2026-07-27)

## A. 훅 매트릭스 (전 세션 실측 재구성)

| | on_msg | task_end | scaled | on_retr | feedback | consol | flush | recent_ctx | retire/patch | produces |
|---|---|---|---|---|---|---|---|---|---|---|
| A-Mem | ● | | | | | | | | | notes |
| Nemori | ● | | | | | ○(옵션) | ● | | | episodes, semantic |
| MemoryOS | ● | | | ● | | | ● | ● | ● | pages, semantic |
| Zep | ● | | | | | | ● | | | facts, entities, communities |
| G-Memory | | ● | | ● | ● | | | | | strategies |
| ACE | | ● | | | ● | | | | | bullets |
| ReasoningBank | | ● | ● | | | | | | | strategies, experiences |
| MemMachine | ● | | | | | | | ● | | derivatives |
| A-MAC | (훅 없음 — write gate wrapper) | | | | | | | | | — |

모든 훅이 사용자 2+ — 예외는 `on_scaled_task_end`(RB 전용, 대조=독립 메커니즘 근거 문서화)와
`retire`/`patch_unit`(체이닝 전용). 훅 인플레이션 없음 판정.

## B. 반복 패턴 4가지

1. **논문 ≠ 공식 코드가 원칙, 예외가 아님.** A-Mem Ps1 사멸(공식 2/3벌) · Nemori §3.3.3 부재 ·
   MemoryOS 논문상수≠수치상수, lowest-heat 퇴출은 어느 코드에도 없음 · G-Memory §4.3 J/Ω 부재
   (코드는 regex finetune) · ACE grow-and-refine 얇음(ADD-only, dedup 조용한 스킵) · A-MAC
   가중치 논문 미공개+결함 위 적합. 유일한 역방향: Zep — 코드가 논문을 앞서감(saga,
   combined extraction). **"논문에 충실"은 단일 목표가 될 수 없고, 계보 고정(프리셋 표)이
   유일하게 방어 가능한 태도** — 우리 프레임워크의 실질 기여점.
2. **사문 knob 가족**: A-MAC `X_train` · MemMachine `QueryPolicy` 6필드 · MemoryOS(eval) 사문
   항들. "설정된 것처럼 읽히지만 읽히지 않는 값"은 upstream 감사의 상수 항목이 되어야 함.
3. **명명 3세대**: 논문 수사 ↔ upstream 엔지니어링 명 ↔ 우리 mechanism 명 (segment=session,
   Profile Memory=semantic_memory=profile.py, Local Message Partitioning=BatchSegmenter=
   BatchPartitioner). 사용자가 체감한 "논문과의 거리"의 대부분은 의미론이 아닌 이 명명 거리.
   조치: 각 organizer/stage docstring에 논문 §번호·원문 용어 병기 (부분적으로만 되어 있음).
4. **움직이는 upstream**: Nemori v1/v4, MemoryOS 5벌, Zep 상시 진화, A-Mem 3벌. fidelity
   프리셋/레시피 표가 이 문제의 일반해로 자리잡음 (NEMORI/MEMORYOS_PRESETS,
   ZEP_SEARCH_RECIPES, MemMachine 백엔드 2종).

## C. 세 추상화 축 최종 판정

- **훅 사영**: 접힘 손실 실측 0건. 경계 원칙 "generator 루프 vs 메모리 레이어"가 ACE
  multi-round reflection과 MaTTS sequential에 일관 적용됨. 비용은 "방법론이 여러 파일로
  갈라짐"(최대 사례 MemMachine 4조각 — 단, 절단선이 논문 자신의 §4/§5 구분과 일치).
- **facade 소유권 (MemoryOp)**: Zep bi-temporal T′축 공짜 제공, Nemori MERGE+supersedes가
  upstream hard-delete의 상위 호환, A-Mem류 버그 재현 불가 구조. 유일 위반(Zep 직접 쓰기)은
  교정 완료. 남은 약점: 계약이 구조가 아닌 규율로 강제됨.
- **mechanism vs policy**: MemMachine(논문 §5가 스스로 분리)과 A-MAC(제목이 자인)이 양쪽에서
  정당성 입증. read쪽 정책화(Zep 레시피, MemMachine QueryStrategy)와 write쪽(A-MAC 게이트)이
  같은 operational test(타입 무소유·op 무발행·임의 host)로 수렴.
- read 경로 관찰된 비용 2건: `produces` 순서 load-bearing(Zep), ACE 전량 주입이 read-step
  밖 별도 경로(`get_playbook`) 요구.

## D. 미결/조치 목록 (2026-07-27 승인 후 실행분 반영)

- [x] **MemoryOS 프리셋 eviction 라벨 교정** — pypi 프리셋 `"lfu"`로, docstring의 "계보 간 퇴출
  차이" 주장 제거, DELETE reason을 실정책명으로, 테스트 갱신 (`test_organizers_phase3.py`)
- [x] A-MAC "published weights" → "release's weights" 3개소 교정 + θ만 논문 공개(Table 3)라는
  각주 (`policies/admission.py`)
- [x] **G-Memory query graph 구현** — `on_task_end`가 유사도 ≥0.7·top-10(upstream
  `TaskLayer.add_task_node` 상수)으로 `task_edges` 무방향 인접(백엣지 UPDATE) 구축, insight에
  `positive/negative_correlation_tasks` 기록(ADD/EDIT/AGREE/REMOVE 의미론 upstream 동일),
  read는 `TaskGraphExpansion`(retrieval/steps.py — 식5 1-hop + 식6 count-scored insight recall,
  `task_graph_expansion_cap` config로 ablatable, RB store에는 불활성). 그래프는 사이드카 없이
  op 로그를 탐. 미포트로 명시 잔존: upstream read의 후보별 LLM importance rerank, FINCH.
  테스트 `test_gmemory_task_graph_edges_and_hop_expansion` 추가.
- [x] Nemori stages/organizer에 논문 § 병기 (§3.2.1/3.2.2/3.2.3/3.3.1/3.3.2/3.3.3 + v1
  formalism 주석)
- [x] Zep upstream 스냅샷 계보 기록 (saga·combined_extraction 미포트) — zep_graph docstring
- [ ] GraphRecall 융합 방식 결정 (측정 재개 시)

검증: 전체 스위트 388 passed·1 skipped, 수정 파일 ruff E/F 청정(잔여 E501 1건은 기존
MERGE_DECISION_PROMPT 문자열), format 청정.

upstream 클론 보관: `~/.agmem/upstream/{AgenticMemory,nemori,MemoryOS,graphiti,GMemory,ace,
reasoning-bank,MemMachine,amac}` — 이후 감사 시 재클론 불필요.
