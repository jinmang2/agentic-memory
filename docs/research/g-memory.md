# G-Memory 리서치 노트 (원자료)

> 출처: 병렬 리서치 세션 (2026-07-16). arXiv:2506.07398, github.com/bingreeky/GMemory 기반.

## 논문 정보

"G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems", Zhang, Guibin; Fu, Muxin; et al., arXiv:2506.07398 (2025-06). **NeurIPS 2025 poster 채택.**

## 1. 핵심 개념

조직 기억(organizational memory) 이론에 기반한 3-tier graph hierarchy:

- **Insight Graph** — 상위 레벨, cross-trial/cross-task 일반화 규칙("insights"). 코드상으로는 그래프가 아니라 `{rule, score, positive_correlation_tasks, negative_correlation_tasks}` 레코드의 JSON 리스트.
- **Query Graph** — 중간 레벨. 과거 task query 간 유사도 그래프. retrieval 시 k-hop 이웃 확장에 사용.
- **Interaction Graph** — 세밀한 레벨. task별 multi-agent 협업 궤적(에이전트 메시지의 directed graph 체인)을 압축 저장.

**Retrieval = bi-directional traversal**: 임베딩 기반 coarse retrieval + query graph k-hop 확장. 위쪽(insight)으로는 전략적 지침, 아래쪽(interaction trajectory)으로는 LLM 기반 sparsification/condensation을 거친 구체 궤적. 추가로 **role-specific memory injection**: insight를 에이전트 role별로 LLM이 재작성(projection)해서 주입.

**Update flow**: task 완료(성공/실패 라벨) → 궤적 sparsify(실패 sub-step 제거) → 저장 → 신규 레코드가 일정량 쌓이면 insight 레이어 "fine-tune"(LLM이 성공/실패 궤적 쌍을 비교해 REMOVE/EDIT/ADD/AGREE 연산 제안) + 주기적 "merge"(task 임베딩 FINCH clustering 후 클러스터별 규칙 통합).

## 2. 코드 분석 (github.com/bingreeky/GMemory)

- 260 stars, 33 forks, 이슈 11개 open. **LICENSE 파일 없음** (재사용 시 법적 모호성).
- 논문 용어와 코드 실명이 다름:
  - `GMemory` (mas/memory/mas_memory/GMemory.py) — 최상위 클래스, `MASMemoryBase` 상속.
  - `TaskLayer` = Query Graph: `networkx.Graph`, pickle 저장(`{namespace}_graph.pkl`). cosine sim ≥ 0.7이면 edge 추가. `retrieve_related_task()`가 `nx.single_source_shortest_path_length(cutoff=hop)`, 기본 hop=1.
  - `InsightsManager` = Insight Graph: flat JSON 규칙 리스트(`insights.json`). `merge_insights`(FINCH clustering), `finetune_insights`(LLM critique), `_parse_rules`(정규식 `REMOVE|EDIT|ADD|AGREE`), `backward(insight, reward)` — 성공 +1 / 실패 -2 reward shaping.
  - Interaction Graph = `StateChain`/`MASMessage` (mas/memory/common.py): `networkx.DiGraph` 리스트, 노드는 `AgentMessage`, edge_type='spatial'. `node_link_data`로 직렬화.
  - 궤적/task 메타데이터는 **Chroma** vector store(`langchain_chroma`)에 저장. 즉 retrieval = Chroma 벡터 검색 + pickled networkx 그래프 순회의 결합.
  - `project_insights(raw_insights, role, task_traj)` — role별 insight 재작성 LLM 호출.

### 디렉토리 구조

```
GMemory/
├── configs/configs.yaml          # max_token=512, temperature=0.1
├── data/{alfworld,fever,pddl,sciworld}/
├── mas/
│   ├── llm.py                    # OpenAI-compatible endpoint (.env OPENAI_API_BASE/KEY)
│   ├── memory/
│   │   ├── common.py             # StateChain, MASMessage, AgentMessage
│   │   └── mas_memory/
│   │       ├── memory_base.py    # MASMemoryBase (add/retrieve/update/backward)
│   │       ├── GMemory.py        # GMemory, TaskLayer, InsightsManager
│   │       ├── prompt.py         # 모든 프롬프트 템플릿
│   │       └── chatdev.py 등     # 베이스라인 메모리 재구현 (Voyager, MemoryBank, Generative Agents...)
│   └── reasoning/
├── tasks/run.py                  # --task --mas_memory g-memory --mas_type autogen ...
└── requirements.txt
```

### LLM 호출 패턴 (task 생애주기당)

1. `_extract_mas_message` — 궤적 sparsification 1회 (+실패 시 `_detect_mistakes` 1회)
2. `retrieve_memory` — 후보별 relevance scoring, 2×successful_topk회
3. `project_insights` — role당 1회
4. 주기적 finetune (5개 레코드마다): point당 최대 3회 + batch당 len/5회
5. 20개 레코드마다 merge: 클러스터당 1회

### 주요 하이퍼파라미터

hop=1, start_insights_threshold=5, rounds_per_insights=5, insights_point_num=5, query-graph edge threshold=0.7 (하드코딩), successful_topk=2, failed_topk=1, insight_topk=10, retrieval threshold=0.3, MAX_RULE_THRESHOLD=10/cluster.

### 의존성

langchain 0.3.25, langchain_chroma, openai, sentence_transformers 3.4.1, finch_clust, networkx 3.3, alfworld 0.3.5, scienceworld 1.2.2, gym, camel 0.1.2. MAS 프레임워크: **AutoGen, DyLAN, MacNet** (수정 없이 external memory module로 hook).

## 3. 성능

- 벤치마크 5종: HotpotQA, FEVER (지식), ALFWorld, SciWorld (embodied), PDDL (planning).
- 백본 3종: **GPT-4o-mini, Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct** — small-model 친화적 스토리.
- 향상폭: ALFWorld +11.20~20.89% (예: 58.21%→79.10%; Table 3 AutoGen+ALFWorld 85.82%), SciWorld +9.53~13.78% (progress rate 기준, ~60%), PDDL +4.02~10.32%, HotpotQA +3.73~10.12%, FEVER +3.32~9.11%.
- latency/token 비용의 정량 테이블은 접근 가능한 소스에서 미확인 ("marginal token overhead" 주장만).
- 전체 Table 1/3 그리드는 PDF 직접 확인 필요 (arXiv HTML 404, PDF 10MB 초과).

## 4. 재현 관점 — ⚠️ 커뮤니티에서 재현 어려움이 공식적으로 보고됨

- **#22**: AutoGen+Qwen2.5-14B(vLLM)로 ALFWorld 67–76% (논문 85.82%) — 3 run 편차 큼. SciWorld ave_reward 0.44–0.56 vs 논문 ~60%. 저자 답변은 "LLM/메모리 랜덤성" 언급 후 미해결.
- **데이터 라벨 버그**: SciWorld 일부 에피소드가 score=100인데 won=false (test.jsonl subgoal 매칭 경직성) — 쉬핑된 eval 데이터 자체 품질 문제.
- **#25**: FEVER/PDDL/SciWorld 재현 발산 (Qwen2.5-14B) — open, 무응답.
- **HotpotQA eval 데이터/환경 미포함** (#12, #15, #21). SciWorld test.jsonl은 뒤늦게 추가됐고 **90 샘플뿐** (공식 test split ~1800 대비, 사유 미설명).
- **AlfWorld 버전 민감성** (0.3.4 vs 0.3.5)이 성공률에 영향 (#9).
- langchain API drift로 코드 깨짐 (#17).
- **LICENSE 없음**.
- **standalone memory service 없음** — 실험 하네스에 강결합된 in-process 라이브러리. MCP/서비스화하려면 상당한 통합 작업 필요 (#24에서 요청만 존재).
- 로컬 소형 모델 스위칭은 `OPENAI_API_BASE`만 바꾸면 됨(vLLM 확인, #22) — 단 성능은 논문 수치보다 낮게 재현되는 경향.

## 우리 프로젝트에 주는 시사점

1. "graph"라고 해도 실제로는 networkx + JSON + Chroma 수준의 경량 구현 — 우리 embedded-first 원칙과 부합. 재구현 난이도 중간.
2. reward-shaped insight list (ADD/EDIT/REMOVE/AGREE 연산)는 ACE의 delta update와 구조적으로 유사 → 통합 추상화 후보.
3. 재현 목표는 "논문 수치 도달"이 아니라 "baseline 대비 향상 재현 + 편차 리포트"로 잡아야 현실적.
4. eval 데이터 검증 파이프라인(라벨 버그 감지)을 우리 하네스에 내장할 것.
5. role-specific projection, k-hop query graph는 MCP 도구 설계 시 파라미터로 노출 가치 있음.
