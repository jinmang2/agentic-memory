# 구현 충실도 감사 (2026-07-16, 자기 감사)

> **2026-07-17 5차 검증 (docs/research/fidelity-round5-other-organizers.md)**: 나머지
> organizer 5종(RB/MemoryOS/ACE/Zep/G-Memory)을 논문+공식 코드 당일 클론으로 대조.
> 등급은 전부 유지되나 **아래 표의 누락 칸은 과소 기재로 판명** — 교정 목록은 round-5
> 문서 §2·§4 참조. 특히: MemoryOS "LFU"는 "최저-heat 축출(논문 준수/코드 비준수)"로
> 정정; ACE는 read 계약(playbook 전체 주입) 미고정; 파이프라인 공통 결함 X1(DELETE
> 유령 벡터)·X2(INVALIDATE 검색 노출)·X3(strategies description 소실) 발견.
> 배선 자체는 실엔진 4종 매트릭스(40 tests × sqlite-vec/LanceDB/Qdrant/Chroma) 통과.

> **2026-07-17 4차 검증 (docs/research/fidelity-round4-verification.md)**: round-3
> 수정 커밋(e7c5f8f)을 업스트림 당일 재다운로드 소스로 독립 재대조 — 24개 클레임
> 전수 CONFIRMED, REFUTED 없음. 신규 발견(업스트림 write 온도 0.7, 콜드스타트
> 프롬프트 규칙차, 링크 캡 per-hit 의미차 등)과 문서 교정을 일괄 반영. 판정:
> **A-Mem ●⁻ / Nemori ●⁻ (v1+리포 eval 기준; v4 통합 모듈 2개는 계획된 미구현)**.
> 온도·콜드스타트 프레임은 upstream 충실로 확정, 측정은 API 전환 결정 전까지 보류.

> **2026-07-16 2차 재감사 (P0/P1 수정 커밋 70ba537 이후, upstream 당일 소스 재대조)**:
> 병렬 감사 2건으로 A-Mem·Nemori를 재검증. 판정:
>
> - **A-Mem ◑ → ◑⁺ (코드 ●)**: P0-1(1-hop 링크 확장) P0-3(예산 6000) P1-5(strengthen
>   `new_note_tags`) 정확 반영 확인. 잔여: [높음] raw episodic 채널 혼입(→ `amem` config를
>   notes-only로 순수화, 구 설정은 `amem_mixed`로 분리), [중간] LLM 키워드 질의 생성 부재
>   (→ `keyword_queries` 옵션으로 구현, `amem` config 기본 on), [낮음] 렌더 keywords 미노출.
> - **Nemori ◑ → ◑⁺**: P0-2(episodic 10/semantic 2k=20) r=2 원문 첨부·cold-start·30분
>   갭·timestamp 모두 정확 반영 확인. 잔여: [높음] raw episodic 채널 혼입(→ `nemori` config를
>   episodes+semantic으로 순수화, 구 설정은 `nemori_mixed`), [중간] episode merging 부재
>   (upstream 리포 기본 on — LongMemEval 단계 P2-11 유지), [중간] per-message vs 배치
>   분할 구조 차이(문서화된 의도적 편차, 콜 수 ~N배 캐비앗).
> - 기존 4-way 수치는 혼합(raw RAG 포함) 조건 측정치이므로 "논문 재현"으로 인용 금지 —
>   순수 config 재측정 후 교체할 것.

> **2026-07-16 갱신**: 심층 감사(docs/research/fidelity-deep-audit.md)로 대체됨.
> 재산정: A-Mem ●→◑(read 링크 확장 누락 — P0-1로 수정), Nemori ◑ 측정 보류(검색 설정
> 불일치 — P0-2/3으로 수정), ReasoningBank ●→◑⁺(검색 단위·온도 분리 — P3).
> 아래 표는 1차 자기감사 기록으로 보존.

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
