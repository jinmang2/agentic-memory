# 코드 컨벤션 — clean code 원칙 (2026-07-18 확립)

> 배경: 전 코드베이스 클린업(사용자 요청)에서 확립. `self.doc`, `vec`, `m_title` 류의
> 축약 이름과 public docstring 공백이 누적되어 있었다. 이 문서가 이후 모든 코드의 기준이며,
> 리뷰 게이트에서 이 문서 위반은 Important로 취급한다.

## 1. 이름 — "축약보다 한 단어 더"

**원칙: 이름은 타입이 아니라 역할을 말한다. 도메인에 정착된 토큰 외의 축약 금지.**

### 1.1 허용되는 짧은 이름 (도메인 정착 토큰)

| 이름 | 의미 | 근거 |
|---|---|---|
| `ctx` | `OrganizerContext` | 전 organizer 훅 시그니처에 정착 |
| `op` / `ops` | `MemoryOp` (목록) | 코어 도메인 모델 자체가 Op |
| `k`, `top_k` | 검색 상한 | IR 관례 |
| `llm` | 역할 라우팅 LLM 클라이언트 | 보편 약어 |
| `i`, `n` | 한 줄 comprehension/enumerate 인덱스 | 관례 (여러 줄 루프에서는 금지) |

### 1.2 금지 → 교정 매핑 (이번 클린업에서 일괄 적용)

| 금지 | 교정 | 비고 |
|---|---|---|
| `self.doc`, `ctx.doc` | `doc_store` | "doc"은 문서/독스트링과 중의적 |
| `self.vec`, `ctx.vec` | `vector_store` | |
| `self.graph`, `ctx.graph` | `graph_store` | 스토어 3종 표기 통일 |
| 훅 파라미터 `ep` | `episode` | 시그니처 가독성 |
| `emb` | `embedding` (역할 접두 권장: `query_embedding`) | |
| `m_title`/`m_narr`/`m_ts` | `merged_title`/`merged_narrative`/`merged_timestamp` | |
| `ep_ts` | `episode_timestamp` | |
| `fid`/`hid` | `fact_id`/`hit_id` | |
| `cand`/`cands` | `candidate`/`candidates` | |
| `cal`/`gen`/`evo` | `calibration`/`generated`/`evolution_verdict` | LLM 응답 변수는 내용을 말할 것 |
| `seg`/`srcs`/`idxs`/`kw` | `segment`/`source_ids`/`indexes`/`keywords` | |

- 새 축약이 필요해 보이면: 그 이름이 이 표의 1.1에 들어갈 자격(코드베이스 전역 정착)이
  있는지 물을 것. 아니면 풀어 쓴다.
- 불리언은 서술형(`is_`/`has_`/`needs_`), 컬렉션은 복수형, 매핑은 `x_by_y` 또는 `x_to_y`
  (`_page_sources: dict[page_id, set[unit_id]]`처럼 값 구조가 자명하지 않으면 주석으로 명시).

## 2. Docstring — public 표면은 전부, "무엇+왜 중요한가"

- **module**: 첫 단락에 이 모듈이 시스템에서 맡는 역할과 설계 근거(스펙/논문/감사 문서
  참조 포함 — 기존 관례 유지).
- **public class/함수/메서드**: docstring 필수. 한 줄이라도 좋으나 시그니처를 반복하는
  문장("Return the count")은 금지 — **호출자가 알아야 할 계약**(부작용, 실패 시 동작,
  단위, 순서 보장)을 쓴다. 예: `ops_since`는 "seq 오름차순, seq > 인자, limit 초과분은
  절단됨(호출자가 페이지네이션)"까지가 계약이다.
- **Protocol 스텁**: 클래스 docstring에 계약을 모으고, 메서드는 비자명한 것만 한 줄.
- `_private` 헬퍼: docstring 선택이나, 알고리즘 단계(스펙 §번호)를 구현하면 참조를 남긴다.
- 언어: docstring/식별자는 영어(기존 관례), 설계 문서는 한국어(docs/).
- 인라인 주석 문화 유지: 감사 근거(`round-5 X1`), 논문 좌표(`§3.2.3`), upstream 대비
  편차는 주석으로 남기는 것이 이 리포의 강점 — 지우지 말 것. 단 "다음 줄이 뭘 하는지"
  반복하는 주석은 금지.

## 3. 구조 원칙 (기존 설계 불변식의 재확인)

1. **Organizer는 store에 쓰지 않는다** — 읽기는 `ctx.*_store`, 쓰기는 MemoryOp 반환만
   (docs/04 §2). 위반은 Critical.
2. **로그 선행**: 모든 변이는 evolution log에 append 후 적용. 커서/상태도 op로 표현.
3. **유실 금지**: LLM 실패 경로는 항상 명시적 폴백(기계적 에피소드, leftover 세그먼트,
   Append 폴백)을 갖는다. 조용한 drop 금지 — drop 카운터 또는 logger.warning.
4. **매직 넘버 금지**: 상수는 모듈 상단 명명 상수 또는 config 파라미터로. 출처가 있는
   값(논문/코드)은 주석에 출처 좌표를 단다 (예: `tau=0.70  # v4 §4.1`).
5. **명시적 저하(degradation)**: 기능이 환경 때문에 꺼질 때는 1회 경고 + 문서화된 동작.
6. **테스트는 퍼사드 행동 단위**: 내부 호출 순서가 아니라 산출 op/저장 상태를 단언.
   테스트 더블은 `tests/helpers.py`의 StubLLM/FakeEmbedder를 재사용.

## 4. 포매팅·도구

- `ruff format` — **line-length 100** (pyproject `[tool.ruff]`에 고정; 주석 길이는 제한
  없음). `ruff check`는 커밋 전 통과.
- import 순서: stdlib → third-party → agmem (ruff isort 규칙).
- 커밋: 논리 단위, 메시지는 `type(scope): 요지` — 리네이밍처럼 기계적 대량 변경은
  행동 변화 없는 단독 커밋으로 분리한다 (리뷰 가능성).

## 5. 적용 이력

- 2026-07-18: §1.2 매핑 전면 적용(스토어 속성 3종, 훅 파라미터, 지역 축약), public
  docstring 공백 해소, README/docs 현행화. 이후 신규 코드는 본 문서를 따른다.
