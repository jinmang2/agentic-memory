# 5차 충실도 검증: 나머지 organizer 5종 + 파이프라인 배선 (2026-07-17)

> 방법: 병렬 검증 5건 — ReasoningBank / MemoryOS / ACE / Zep-graph / G-Memory 각각
> **논문 + 공식 코드 당일 재다운로드(클론)** 후 구절·라인 단위 대조. 배선(wiring)은
> 로컬에서 별도 검증: organizer 전체 테스트 40건 × 실엔진 4종(sqlite-vec / LanceDB /
> Qdrant local / Chroma) 매트릭스 **전부 통과** (`scripts/test-engine-matrix.sh`).
> 상세 보고서 5건: `docs/research/round5/*.md` (본 문서는 종합).
> 대상 HEAD: 0bd5037.

## 종합 판정

| organizer | docs/10 등급 | round-5 판정 | 한 줄 요약 |
|---|---|---|---|
| ReasoningBank | ◑⁺ | **유지** | deep-audit §3 (a)~(f) 전부 재확인·전부 미수정(open). 신규 10건 + read 경로 실결함 1건(description 소실) |
| MemoryOS | ◑ | **유지 (누락 목록 대폭 확장)** | 코어 상수·공식은 일치하나 read-path heat 피드백(N_visit) 부재로 heat가 퇴화. 업스트림 코어 3벌 드리프트 확인 |
| ACE | ◑ | **유지 (read 계약 미고정 플래그)** | write 경로 충실(upstream도 ADD-only 확인). 단 ACE는 top-k 검색이 아니라 **playbook 전체 주입** 방법론 — read 계약 고정 전 측정 불가 |
| Zep-graph | ○ 측정금지 | **유지 (누락 ⑥~⑨ 추가)** | 기존 ①~⑤ 유효. 0.85 임베딩 자동병합은 논문·코드 어디에도 없는 경로. invalidated fact 검색 노출 버그 |
| G-Memory | ○ 측정제외 | **유지 (근거 확장)** | 논문↔공식코드 자체가 불일치(§4.3 J-요약 vs critique finetune) — 우리는 코드 계보. 검색 의미론·점수 의미론 대부분 미이식 |

## 1. 파이프라인 공통 결함 (전 방법론 횡단 — 최우선)

| # | 결함 | 발현 지점 | 수정 |
|---|---|---|---|
| X1 | **DELETE 유령 벡터**: `_apply_one`의 DELETE가 doc만 `{"deleted": true}`로 덮고 **vec 인덱스는 방치**, `pipeline._hydrate`에 deleted 필터 없음 → 삭제된 항목이 빈 content의 ScoredItem으로 k 슬롯 소모 | G-Memory REMOVE, MemoryOS LFU 축출 | vec 제거 또는 hydrate 필터 (P0) |
| X2 | **INVALIDATE된 fact 검색 노출**: invalid_at이 doc에만 기록되고 검색·렌더 어디서도 필터/표시 안 됨 → 무효화된 옛 fact가 현행 fact와 나란히 주입. Zep 논문의 χ는 fact에 date range 병기가 명세 | Zep temporal 전체 | 검색 시 invalid 제외 또는 date-range 병기 (Zep 측정 해금 전제) |
| X3 | **strategies description 소실**: read 경로가 `_DictItem.render`(title+content)라 description이 주입에서 탈락. `StrategyItem.render()`는 데드 코드 | ReasoningBank (역설적으로 논문 App A.2 "title and content"에는 부합, 공식 코드의 markdown 전문 주입과는 불일치) | 렌더 계약 결정 후 통일 |
| X4 | zep_graph의 `SqliteGraphStore(":memory:")` 조직자 소유 — data_dir 있어도 그래프 미영속, capability graph_store 슬롯 미연결 | Zep | Zep 완성 단계에서 슬롯 연결 |

## 2. 방법론별 핵심 발견 (상세는 round5/ 각 보고서)

**ReasoningBank** (`round5/rb-verify-report.md`)
- deep-audit (a)~(f) 전부 CONFIRMED, 전부 미수정. 온도만 절반 해소(judge 0.0 기본값; RB 실험 config 자체가 아직 없음).
- 신규: upstream은 추출 출력을 파싱 없이 `split("\n\n")`로 통째 저장(논문 "title and content" 서술과 코드 불일치); top-1 경험 miss 시 **무주입** 의미론; 문서/질의 비대칭 임베딩; 도메인 페르소나·"one sentence" description 제약 누락; 에이전트 실행 온도 0.7.

**MemoryOS** (`round5/memoryos-verify-report.md`)
- 업스트림 코어 3벌(pypi/chromadb/eval)이 서로 다름 — **논문 수치는 eval판**(Dice keyword 항, heat 0.8/0.8/1e-4, STM cap=1). 우리 상수는 pypi판 기준 일치.
- **[높음] read-path heat 피드백 부재**: 업스트림은 검색 시 N_visit·last_visit·access_count 갱신(논문 §3.4 명시). 우리는 N_visit이 영원히 0 → heat = length+recency로 퇴화, LFU 구조적 불가.
- **[높음] STM 방출 단위**: 업스트림 1-page FIFO 롤링(STM 상시 유지, QA 시 전량 주입) vs 우리 10개 전량 flush(QA 시 STM 빈 상태 — recency 채널 소실).
- eviction 라벨 정정: 우리는 최저-heat 제거 = **논문 준수 / 공식 코드(access-count LFU)와는 상이**.
- keyword 파이프라인 전무(=Jaccard 부재의 근본 원인), 승격 실패 시에도 heat 리셋(업스트림은 보존), mtm_capacity 200 vs 2000.
- read 채널 평가: episodic k=10·semantic k=10은 업스트림 eval과 정합(**MemoryOS는 raw episodic 채널이 오염이 아님** — 업스트림도 raw page 주입). 누락: 프로필 전문 무조건 주입, Assistant Knowledge, STM recency.

**ACE** (`round5/ace-verify-report.md`) — 공식 리포 github.com/ace-agent/ace 확인·클론 대조
- write 경로: ADD-only는 **upstream도 동일**(코드 주석 확인), dedup 0.90 일치, multi-round(코드 3/논문 ablation 5)·offline(train/val+multi-epoch+best 선택) 누락 재확인.
- **[중요] read 계약**: 공식은 **playbook 전체를 Generator 프롬프트에 주입**(선택은 LLM이 bullet_ids 인용으로), 크기 관리는 token budget 80k+dedup. 우리 `get_playbook()`은 `vec.search(embed("playbook"), k=200)` **근사**이고, top-k 검색 경로와의 계약이 미고정 — 측정 전 필수 결정.
- **[중요] Curator 부분 뷰**: 공식은 전체 playbook+stats를 주는데 우리는 task-유사 top-30만 → paraphrase 중복 ADD 위험.
- Reflector 태깅 대상이 "실사용 bullet"(공식: Generator가 인용한 id만)이 아니라 "검색된 bullet"; intra-batch dedup 공백; 렌더 포맷 이중화.

**Zep-graph** (`round5/zep-verify-report.md`) — Graphiti HEAD 36918ce 대조
- **0.85 임베딩 top-1 자동 병합은 논문에도 현 upstream에도 없는 경로** — 임베딩은 후보 수집용이고 병합 결정은 exact/fuzzy(MinHash 0.9)+LLM. 우리 방식은 동명이의 오병합과 표기 변이 분열을 둘 다 못 막음.
- 누락 ①~⑤ 전부 유효, 단 ②(현 upstream은 fulltext 후보 → cosine 15/0.6+결정적+LLM 3단계로 변경)·③(별도 콜 → fact 추출 프롬프트 통합+소형모델 fallback) 서술 갱신 필요.
- 신규 ⑥ invalidated 검색 노출(X2), ⑦ previous-episodes 컨텍스트 전무(논문 n=4/현행 10 — 대명사 해소 불가), ⑧ resolution 시 name/summary 갱신 없음(summary 동결), ⑨ `expired_at` 죽은 컬럼(bi-temporal T′축 소실 — sqlite_graph docstring 주장은 현재 허위).
- read 경로: 현 배선은 "그래프 산출물을 평면 벡터 RAG로 읽는 것" — ϕ_bfs·hybrid·reranker 전부 부재. 측정 해금 조건(최소 read-path 5항목)을 보고서에 명세.

**G-Memory** (`round5/gmemory-verify-report.md`) — bingreeky/GMemory 7b581c5 대조
- **논문 §4.3과 공식 코드가 서로 불일치**(J-요약+Ω_k vs critique finetune+FINCH+backward) — 우리는 코드 계보를 따름 (문서에 명기할 것).
- docs/10 누락 3건 유효 + 미기재: 검색 의미론 전체(성공/실패 분리 채널 2/1·threshold 0.3·LLM relevance rating·insight support-set 투표), 점수 의미론(ADD init 2/EDIT +1/REMOVE soft −1·−3/AGREE/score≤0 프루닝), `_detect_mistakes` fail_reason, 랜덤 포인트 앵커 finetune 구조.
- 배선: reward 폐루프가 write-only(score가 랭킹·필터에 미사용, 프루닝 없음 — −2를 받아도 계속 서빙), backward/projection 호출자 부재(MAS 하니스용 public API로 예정된 상태), ReasoningBank과 strategies 타입 공유 시 혼입.

## 3. 배선 총평

- **스토리지 계층은 건전**: 전 organizer 플로우가 실엔진 4종에서 동일 동작(40×4 통과). MemoryOp → doc/vec 이중 기록, embedding_text 계약, memory_type 필터 모두 정상.
- **읽기 계층이 약점**: 공통 결함 X1~X3에 더해, 방법론별 read 의미론(ACE 전체 주입, MemoryOS 프로필/STM 채널, Zep hybrid+BFS, G-Memory 분리 채널+투표)이 "벡터 top-k"라는 단일 프레임으로 평탄화되어 있음. A-Mem/Nemori에서 round-3~4가 했던 read-path 정합 작업이 나머지 방법론에는 아직 적용되지 않은 상태.
- **피드백 루프 부재 패턴**: N_visit(MemoryOS)·reward 프루닝(G-Memory)·helpful/harmful 실사용 귀속(ACE) 모두 "검색 결과가 메모리 상태를 갱신하는" 경로가 없어 생기는 같은 계열의 갭 — 파이프라인에 조회 통지 훅 하나를 설계하면 셋을 함께 풀 수 있음.

## 4. 수정 우선순위 제안

**P0 (버그 — 방법론 무관)**: X1 DELETE 유령 벡터, X2 INVALIDATE 노출(최소 필터), X3 description 렌더.
**P1 (측정 예정 방법론의 read 정합)**: MemoryOS 프로필 주입 채널+keyword/Jaccard 복원(+docs/09 캐비앗 확장), ACE read 계약 고정(get_playbook full-scan화+ curator 전체 뷰).
**P2 (측정 보류 방법론)**: Zep write 재설계(resolution LLM화+temporal 통합 추출+dedup, 이후 read 5항목), G-Memory 점수·검색 의미론 정합 — 각 벤치 착수 시점에.
**P-doc**: docs/10 각 행 교정(본 문서 §2의 라벨 정정·누락 확장 반영), deep-audit §3 주입 포맷 행 교정, memoryos/sqlite_graph docstring 갱신.
