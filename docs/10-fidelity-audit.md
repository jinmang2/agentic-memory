# 구현 충실도 감사 (2026-07-16, 자기 감사)

> 규율: **부분 구현은 벤치마크 대상에서 제외**하거나 결과표에 fidelity 등급을 병기한다.
> 미구현 요소가 그 방법론의 핵심 주장과 연결되면 측정 전 반드시 구현한다.

등급: ●충실(핵심 메커니즘 전부) ◑부분(주변부 누락) ○골격(핵심 일부 누락 — 측정 금지)

| organizer | 등급 | 구현됨 | **누락 (논문 대비)** | 측정 가능? |
|---|---|---|---|---|
| A-Mem | ● | 2콜 write(Ps1/Ps3), metadata-concat 임베딩, top-k링크, 이웃 배치 진화 + 버그수정 | k스윕 실험용 옵션 일부 | ✔ (측정됨) |
| ReasoningBank | ● | self-judge, 성공/실패 프롬프트, ≤3 items, k=1 | **MaTTS**(parallel self-contrast/sequential) — 훅만 존재 | ✔ (LoCoMo엔 해당無) |
| Nemori | ◑ | boundary(σ=0.7)/서사/시간절대화/predict-calibrate 3단계 | episode merging, 배치 세그멘테이션 모드 | ✔ (측정됨, merging 부재 명기) |
| MemoryOS | ◑ | STM/MTM/heat(공식 동일)/LFU/profile 승격 | **F_score의 Jaccard 항**(cos만 사용), dialogue chain meta, 90-dim trait, agent persona | ✔ (F_score 단순화 명기) |
| ACE | ◑ | reflect→curate ADD, helpful/harmful, dedup 0.90 상시 | **multi-round reflection**(≤3), offline 모드(train/val 분리) | ✔ (online 모드만) |
| **Zep-graph** | **○** | entity 추출, 임베딩 resolution, bi-temporal fact, LLM invalidation | **① community subgraph(label propagation) ② resolution의 LLM 판정+fulltext 후보 ③ 시간표현 파싱(t_valid/t_invalid) ④ GraphRecall(BFS) 파이프라인 배선 ⑤ fact dedup(hybrid)** | **✘ 측정 금지 — 4-way에 미포함 (올바름)** |
| G-Memory | ○ | 궤적 sparsify, insight ADD/EDIT/REMOVE+reward, projection/backward | **query graph k-hop, FINCH merge, StateChain interaction graph** | ✘ MAS 벤치 자체가 미구축 |

## 측정 결과(docs/09)의 유효성

- 4-way(passthrough/A-Mem/Nemori/MemoryOS)는 ●/◑ 등급만 포함 — 결론 유지.
  단 MemoryOS 수치에는 "F_score Jaccard 항 부재" 캐비앳을 docs/09에 병기할 것.
- call 수는 방법론 구조가 결정(0.5B 영향은 재시도 ≤1%) — 비용 비교는 유효.

## Zep 완성 계획 (측정 전 필수, 우선순위순)

1. **GraphRecall 배선**: `retrieval/recall.py`에 graph BFS recall 추가 + zep 설정 시 파이프라인 활성 (핵심 — Zep의 검색 우위 주장이 hybrid+graph에서 나옴)
2. resolution 2단계화: 임베딩 후보 → LLM dedup 판정 (Graphiti 파이프라인 §A.3)
3. temporal extraction 콜 추가: 시간표현 → t_valid/t_invalid (temporal 카테고리 성능과 직결)
4. fact dedup (동일 entity-pair 범위 hybrid)
5. community: label propagation + 주기 refresh (LongMemEval 단계에서)
