# 개발 로드맵

> 목표: ① 8개 방법론 직접 구현 ② MCP 배포 ③ 0.5B급 소형 모델로 PC에서 table 재현 ④ 필요 시 보조모델 학습.
> 각 Phase는 "동작하는 수직 슬라이스"를 끝점으로 함. 기간은 스터디 병행 기준 러프 추정.

## Phase 0 — 스캐폴딩 (1주) — 2026-07-16 대부분 완료

- [x] `uv init` + 패키지 구조 (`docs/04-architecture.md` 레이아웃) — Python 3.12 pin
- [x] `capabilities/` 감지·리졸버 + profile 3종 (`docs/01-capability-system.md`)
- [x] `core/types.py`, `core/ops.py` (MemoryOp + EvolutionLog)
- [x] `stores/sqlite_doc.py` + `sqlite_vec.py` (FTS5 포함) + `numpy_vec.py` fallback + 테스트 26개 통과
- [x] retrieval v0: Dense+Lexical → RRF (pipeline 골격)
- [x] LLM 데몬 셋업: llama.cpp **CUDA 빌드(compute 7.5 직접 컴파일)** + Qwen3-0.6B-Q8_0, `scripts/serve-llm.sh`(:8080, GPU 전 레이어 오프로드). structured output 스모크 **3/3 유효 JSON, 1.6s/call** (CPU 스왑 병목 143s → GPU 90×). `llm/client.py` 역할별 라우팅 완료
- [x] `llm/structured.py`: guided_json + 재시도 + drop 카운터 (0.5B 대응의 토대)
- [x] 실호스트 스모크: capability 감지(RTX 2060/7.8GB) + 강등 로그 + 한국어 add→search 확인

**완료 기준 달성**: `AgenticMemory(organizers=["passthrough"])` add→search 동작 확인.
참고: 호스트에 qdrant(6333)·redis(6379) 포트 활성 감지됨 — full 프로파일 테스트에 활용 가능.

## Phase 1 — 첫 방법론 2종 + 추상화 검증 (2주) — 2026-07-16 완료

구현 난이도 "하"이면서 설계 공간의 양극단인 둘을 먼저:

- [x] **ReasoningBank organizer** (self-judge→성공/실패 증류→append; field-level fallback으로 깨진 아이템만 스킵)
- [x] **A-Mem organizer** (노트+링크+진화 배치 호출; **버그 수정판**: 이웃을 인덱스가 아닌 ID로 참조(#32), cosine 보장(#24), 환각 ID 필터, silent skip 금지. fidelity="paper"는 하이퍼파라미터만 재현)
- [x] retrieval 파이프라인 v1 (Dense+Lexical → RRF → MMR; vector store에 `get()` 추가)
- [x] 비동기 워커 + `flush()` (raw episode는 항상 동기 인덱싱 — read는 write를 기다리지 않음)
- [x] `bench/harness.py` 골격: multi-run mean±std + calls/tokens/latency + 조건 스탬프(commit/profile/embedder)
- [x] **0.5B 방어층 실전 검증**: Qwen3-0.6B E2E에서 top-level 배열 반환 실패 모드 발견 → 스키마 유도 배열 코어싱 추가 → **drop 0회** (notes 2 + strategies 1, 5.7s/6 LLM calls)

**완료 기준 달성**: MemoryOp 추상화가 대화형(A-Mem)·태스크형(ReasoningBank) 양극단을 누수 없이 수용. 테스트 39개 통과.
발견 사항: 0.6B judge는 실패 궤적을 success로 오판하는 사례 확인 — judge role의 상위 모델 라우팅(티어링) 필요성 실증.

## Phase 2 — 벤치마크 하네스 + 1차 재현 (2–3주)

- [ ] **LoCoMo** 파이프라인 (F1/BLEU-1 — judge 불필요라 가장 저렴한 시작점)
- [ ] **LongMemEval_S** 파이프라인 (judge pin `gpt-4o-2024-08-06`, reading `con`+`json` 고정, cleaned 버전 명시, ingest 아티팩트 캐시)
- [ ] 1차 재현 실험 (PC, 0.6B extract + API judge):
  - A-Mem × LoCoMo — 원논문 GPT-4o-mini 수치와 방향성 비교
  - ReasoningBank류 × 간단 태스크(수학/코딩 스트림, reasoning-bank-slm 프로토콜 참고)
- [ ] 재현 리포트 v1: 절대치가 아닌 **baseline 대비 향상 + 3-run 편차** 보고

**완료 기준**: `agmem-bench` 한 줄로 표가 나오고, 결과에 전체 실험 조건이 스탬프됨.

## Phase 3 — 나머지 방법론 + MCP 배포 (3–4주)

- [ ] **Nemori organizer** (분절→서사→predict-calibrate; 시간 절대화 포함)
- [ ] **MemoryOS organizer** (STM/MTM/LPM, heat, LFU)
- [ ] **ACE organizer** (3-role, delta ops, dedup 0.90 — 기본 on으로 원논문 함정 회피)
- [ ] **Zep-graph organizer** (graph store 위: entity resolution→bi-temporal fact→invalidation→label propagation community) — 가장 무거우므로 마지막
- [ ] **G-Memory organizer** (MAS 훅; AutoGen 예제 1개)
- [ ] rerank 어댑터 완성 (LLMReranker, CrossEncoder — capability-gated)
- [ ] **MCP 서버 배포**: stdio + HTTP, Claude Code 연동 실사용 (본인 코딩 세션에 물려 dogfooding)

**완료 기준**: 7개 organizer 전부 동일 API로 구동 + MCP로 Claude Code에서 실사용.

## Phase 4 — 소형 모델 학습 + 심화 재현 (3–4주, 선택 확장)

- [ ] **SFT 데이터 생성**: 대형 모델(API)로 분절/추출/증류 태스크의 입출력 쌍 생성
      (LoCoMo/LongMemEval 히스토리 + 자체 대화 로그 소스)
- [ ] **Qwen3-0.6B QLoRA** (RTX 2060 6GB: 4bit base + LoRA r=16, batch 1 + grad accum — 가능성 확인됨)
      태스크별 어댑터: ① boundary 탐지 ② note/entity 추출(JSON) ③ 전략 증류
- [ ] 평가: 구조화 출력 준수율 / 추출 품질(vs API 모델) / 최종 벤치 점수 개선폭
- [ ] **LongMemEval 2차 재현**: 학습된 0.5B extract 모델 vs 프롬프트만 0.6B vs API 3-way 비교
      → "0.5B로 어디까지 가능한가" 표 — 이 스터디의 고유 기여물
- [ ] (여력 시) MaTTS 재현: parallel k=3 self-contrast, 소형 모델에서의 비대칭성 관찰

## Phase 5 — 종합 (1–2주)

- [ ] 방법론 교차 조합 실험 (Nemori+ReasoningBank 스택 등)
- [ ] 최종 리포트: 8-시스템 비교표(자체 측정) + 비용-정확도 파레토 곡선
- [ ] 블로그/스터디 발표 자료

## 리스크와 대응

| 리스크 | 근거 | 대응 |
|---|---|---|
| 0.5B 구조화 출력 실패 | A-Mem/Graphiti 실증 | Phase 0의 4중 방어 우선 구축; extract만 4B-AWQ 강등 경로 |
| 원논문 수치 미도달 | G-Memory -10~18%p, A-Mem* 사례 | 목표를 "방향성+편차 보고"로 명시 (Phase 2) |
| WSL RAM 7.8GB 병목 | 실측 | `.wslconfig` 상향; 서버형 store는 full 프로파일로 격리; mmap 벡터 |
| judge 비용 누적 | LongMemEval 그리드 폭발 | ingest 캐시 + LoCoMo(F1) 우선 + judge 호출 상한 설정 |
| G-Memory LICENSE 부재 | 조사 확인 | 코드 복사 금지, 논문 기반 클린룸 재구현 + 출처 명기 |
| 스코프 폭발 | 8 방법론 × 벤치 × 학습 | Phase별 완료 기준 엄수; Zep-graph/G-Memory는 후순위 배치 |

## 즉시 다음 액션 (Phase 0 시작점)

1. `uv init agentic_memory && uv add pydantic sqlite-vec sentence-transformers openai fastmcp`
2. llama.cpp 설치 + Qwen3-0.6B GGUF 다운로드, OpenAI-compatible 서버 기동 확인
3. `capabilities/detect.py` 작성 (이 문서의 환경 실측값이 첫 테스트 케이스)
