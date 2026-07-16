# A-Mem 구현체 리뷰 (발표용)

> A-MEM: Agentic Memory for LLM Agents (arXiv:2502.12110, **NeurIPS 2025**)
> 리뷰 기준일 2026-07-16. 근거: 공식 리포 소스 직접 분석 + 자체 재구현/재현 실험
> (github.com/jinmang2/agentic-memory).

## 1. 논문이 주장하는 것

- 기존 메모리 시스템은 저장/검색은 되지만 **조직화(organization)가 고정적** — graph DB 방식도 사전 정의 스키마에 종속.
- 해법: **Zettelkasten**. LLM 스스로 메모리를 조직하게 한다.
  - **Note construction (Ps1)**: 메시지 → keywords/tags/context를 LLM이 생성
  - **Link generation (Ps2)**: 임베딩 top-k 이웃 중 LLM이 연결 판단
  - **Memory evolution (Ps3)**: 새 노트가 이웃 노트의 context/tags를 **재작성**
- 임베딩 특징: `e = f_enc[concat(content, keywords, tags, context)]` — 메타데이터 포함.
- read path는 LLM 0회 (순수 dense top-k + 1-hop 링크 보강). **차별화는 전부 write-path.**

## 2. 공식 구현의 실체

### 2.1 리포 3벌 체제 (혼동 주의)

| 리포 | 용도 | 상태 (2026-07) |
|---|---|---|
| `WujiangXu/A-mem` | **논문 재현 전용** (README 명시) | ★919, 2026-03 마지막 push |
| `agiresearch/A-mem` | 라이브러리판 | ★1,113, 2025-12 이후 정체 (#16 "Is this project dead?") |
| `WujiangXu/A-mem-sys` | 시스템판 (openai/ollama/sglang) | ★371, 실사용 권장 |

### 2.2 실제 코드 스펙 (agiresearch판 실측)

- `AgenticMemorySystem(model_name='all-MiniLM-L6-v2', llm_model='gpt-4', evo_threshold=100)`
- 저장소: ChromaDB (**기본 in-memory** — 프로세스 종료 시 노트 유실 위험) + python dict
- write당 LLM **2회** (Ps1 구성 1 + Ps3 진화 1; 이웃 5개를 한 프롬프트에 배치), read당 **0회**
- evolution 100회마다 컬렉션 전체 재구축 (latency 스파이크)
- 논문 top-k ∈ {10..50} 실험, 코드 기본값은 **k=5**

### 2.3 알려진 버그 — 논문-구현 불일치 (발표 하이라이트)

| 이슈 | 내용 | 상태 |
|---|---|---|
| **#24/#23** | ChromaDB 컬렉션이 cosine이 아닌 **L2 distance**로 생성 + 거리를 유사도로 오용 | open |
| **#32** | `find_related_memories()`가 노트 ID가 아닌 **결과 순위 인덱스** 반환 → evolution이 엉뚱한 노트를 갱신 가능 | open |
| **#10** | 한때 note 구성 시 LLM을 실제로 호출하지 않았음 (#13에서 수정) — 버전에 따라 논문 방법 미실행 | closed |
| #7/#14 | 동일 검색 중복 호출, retriever 중복 | — |

### 2.4 수치 재현성 논란

- MemoryOS 논문의 재현치 **A-Mem\***: 보고치 대비 전 카테고리 하락 (Multi-hop 45.85 → 33.23)
- **Mem0 논문**(2504.19413)의 독립 재평가도 더 낮은 수치 보고
- "85–93% 토큰 절감"은 태스크 비용이 아닌 **op당 토큰** 기준이라는 커뮤니티 지적
- 종합: **링크/진화의 이득 방향은 재현되지만 절대 수치는 보수적으로 봐야 함**
  (ablation에서 Link Generation이 이득의 대부분: w/o LG&ME 9.65 → full 27.02 F1)

## 3. 우리 재구현 (agmem)

### 3.1 설계: 방법론을 Organizer 플러그인으로

A-Mem 전체가 `organizers/amem.py` 하나 (~160줄). 스토리지를 직접 만지지 않고
**MemoryOp(ADD/LINK/UPDATE) 리스트를 반환** → append-only evolution_log에 기록 후 반영.
전 과정이 감사/재생(replay) 가능 — 원논문의 "evolution이 뭘 바꿨는지 추적 불가" 문제 해소.

```
on_message(ep):
  1. Ps1 note 구성   (LLM 1회, JSON: keywords/context/tags)
  2. 이웃 top-5 검색  (metadata-concat 임베딩, cosine 보장)
  3. Ps2+Ps3 배치 호출 (LLM 1회, JSON: connections/neighbor_updates)
  → [ADD(note), LINK(양방향), UPDATE(이웃 context/tags 재임베딩)]
```

### 3.2 원본 대비 의도적 수정 (fidelity="paper"는 하이퍼파라미터만 재현)

| 원본 문제 | 우리 처리 |
|---|---|
| L2를 유사도로 오용 (#24) | vector store 계층에서 cosine 보장 (sqlite-vec `distance_metric=cosine`) |
| 인덱스로 이웃 참조 (#32) | **ID로 참조** + LLM이 환각한 ID는 필터 (테스트로 고정) |
| evolution 실패 시 silent skip | 명시적 **drop 카운터** + 로그 (0.5B 대응 4중 방어의 일부) |
| UPDATE가 노트 전체 덮어쓰기 위험 | 기존 아이템에 **병합**(merge) 시맨틱 |
| in-memory 유실 | SQLite 단일 파일 영속화 + FTS5 lexical 검색 공짜 획득 |

### 3.3 0.5B급 소형 모델 실전 (Qwen3-0.6B, RTX 2060)

- 노트 구성/진화 파이프라인 정상 구동: **drop 0회**, 호출당 ~1.4s
- 발견한 실패 모드: strict JSON 대신 **top-level 배열 반환** → 스키마 유도 코어싱으로 방어
  (원논문도 1–3B에서 Ps3 strict JSON 실패로 evolution 무력화 — 우리는 방어층으로 흡수)

## 4. LoCoMo 1차 재현 (conv0, 로컬 0.6B 단독)

> ⏳ 실험 진행 중 — 완료 후 갱신. 조건: LoCoMo conv 0 (세션 19개, ~400턴),
> passthrough(raw episode + hybrid retrieval) vs A-Mem(수정판), 동일 0.6B answer 모델,
> k=10, F1/BLEU-1 (judge 불필요), 비용(calls/tokens/latency) 병기.

| config | Overall F1 | BLEU-1 | ingest 비용 | 비고 |
|---|---|---|---|---|
| passthrough | (진행 중) | | LLM 0 calls | 대조군 |
| A-Mem (fixed) | (진행 중) | | 턴당 2 calls | |

해석 가이드 (사전 등록): 목표는 논문 절대 수치가 아니라 **(a) 동일 조건에서
A-Mem 조직화가 passthrough 대비 주는 델타의 방향, (b) 그 델타의 비용**(턴당 LLM 2회).
0.6B answer 모델의 절대 성능은 낮을 수밖에 없음 — 이 스터디의 관심사는
"작은 모델일수록 메모리 시스템의 상대 이득이 커지는가"(Nemori 관찰)의 검증.

## 5. 발표 톡킹 포인트 제안

1. **아이디어는 우아하고 코드는 부실** — Zettelkasten 발상과 read-path 0-call 설계는
   차용 가치가 높지만, 공식 구현은 L2/인덱스 버그가 열린 채 유지보수 정체.
   "논문 채택 ≠ 구현 신뢰성"의 전형적 사례.
2. **이득의 원천은 Link Generation** (ablation) — evolution은 보정 수준. 재구현 시
   evolution을 이웃 배치 1콜로 단순화해도 무방.
3. **temporal reasoning 이득이 최대** (18.41→45.85) 인데 이는 노트 metadata가
   timestamp를 담기 때문 — Nemori의 "시간 절대화"가 이 관찰의 발전형.
4. **write 동기 2콜이 UX 병목** (issue #21 "왜 이렇게 느린가") — 우리는 async 워커로
   분리하고 raw episode는 즉시 검색 가능하게 설계.
5. 재현 시 체크리스트: 어떤 리포인가(3벌), cosine 여부, evolution 실제 호출 여부,
   k 값(논문 10–50 vs 코드 5), 수치 출처(보고치 vs A-Mem* vs Mem0 재평가).
