# 4차 충실도 검증: round-3 수정 확인 + 업스트림 독립 재대조 (2026-07-17)

> 방법: 병렬 검증 2건 — A-Mem(C1~C12)·Nemori(N1~N12) 클레임 체크리스트를 만들어
> **업스트림 원본을 당일 raw로 재다운로드**해 구절 단위 대조. 대상 HEAD: 1b8d035
> (round-3 수정 커밋 e7c5f8f 반영 후). 소스: agiresearch/A-mem
> (`agentic_memory/memory_system.py`, `retrievers.py`) + WujiangXu/AgenticMemory
> (`memory_layer.py`, `memory_layer_robust.py`, `test_advanced.py`) + GitHub 이슈
> #23/#24/#32 + nemori-ai/nemori (`prompts.py`, `config.py`, `semantic.py`,
> `episode.py`, `segmenter.py`, `orchestrator.py`, `search.py`) + arXiv v1 HTML.

## 판정 요약

**round-3 권고의 코드 항목은 전부 정확히 반영되었고, 24개 클레임 전수 재대조에서
REFUTED 없음.** A-Mem 게이팅/단방향 링크/tags_to_update, Nemori 프롬프트 5종의
핵심 구절(30분 갭, "when in doubt split", 2-15msg, 4-test, 7카테고리, 시간 금지,
괄호 병기+예시), read 경로(1-hop 확장, r=2, 2섹션 렌더, k=10/20, verbatim 키워드
질의)가 모두 업스트림 원문으로 뒷받침됨. 이슈 #23/#24/#32는 2026-07 현재도 전부
open이며 배정이 정확함.

미반영이었던 것: round-3 P-doc 3(deep-audit §5 정오표), docs/08 버그 서술 교정.
→ 본 라운드에서 신규 발견과 함께 일괄 반영(아래 §4).

## 1. A-Mem 신규 발견

| # | 내용 | 조치 |
|---|---|---|
| R4-A1 | **agiresearch add_note는 analyze_content를 아예 호출하지 않음** — Ps1 메타데이터가 생성자 기본값 그대로. WujiangXu plain판의 `import re` 결함과 합치면 **공식 두 판본 모두 Ps1이 사실상 사문** (robust판만 정상). WujiangXu 폴백의 context는 빈 값이 아니라 `"General"` | docstring·docs/08 반영 |
| R4-A2 | `# paper default k=5` 오기 — 논문은 QA top-k=10("we primarily employ k=10"), k=5는 **코드** 하드코딩값 | 주석 정정, `fidelity` 데드 파라미터 제거 |
| R4-A3 | 이웃 검색 질의가 enriched `embedding_text()` — 업스트림 코드 양쪽은 `note.content`만. 논문 식(3) 관점에선 우리가 논문 충실/코드 비충실 | docstring 이탈 목록에 추가 |
| R4-A4 | `actions` 빈 값 폴백(양 효과 유지)은 우리 고유 동작 — 업스트림은 no-op | docstring에 명시 |
| R4-A5 | **1-hop 확장 캡 의미차**: 업스트림은 per-hit(agiresearch 히트당 k개, WujiangXu off-by-one으로 k+1) → eval k=10이면 링크 이웃 최대 ~100개. 우리는 글로벌 5 | pipeline docstring 보강 + 결과 캐비앗 유지 (동작 유지 — WujiangXu #16/#21이 업스트림 자체의 모호함을 확인) |
| R4-A6 | update_neighbor 프롬프트 과제 차이: 업스트림은 "전체 이웃 배열, 무변경 시 원본 반복" positional, 우리는 "변경분만" ID 기반 — #32 수정의 귀결로 유지 | 기록만 |
| R4-A7 | 업스트림 write 경로 온도는 `get_completion` 기본 **0.7** (WujiangXu; 답변도 cat1-4 0.7, cat5 0.5) | exp config에 A-Mem write 0.7 반영 |
| R4-A8 | 첫 노트 처리: agiresearch는 이웃 0이면 진화 스킵(우리와 동일), WujiangXu는 호출 | 기록만 |

## 2. Nemori 신규 발견

| # | 내용 | 조치 |
|---|---|---|
| R4-N1 | **업스트림 온도: episode 생성·예측·지식 추출이 전부 클라이언트 기본 0.7** (orchestrator `LLMRequest` 기본값, 미오버라이드; max_tokens 2000). 명시 온도는 segmentation 0.2 / 답변 0.0뿐. round-3 N9는 이 0.7을 놓침 | exp config: nemori extract 0.2, distill 0.7 + max_tokens 2000 |
| R4-N2 | **콜드스타트 SEMANTIC_GENERATION은 comparison과 다른 프롬프트**: 6카테고리(Beliefs & Values 없음), **시간·날짜 금지 없음**, GOOD 예시에 날짜 포함("joined Amazon in August 2020"). 기존 우리 구현은 시간 금지 포함 규칙을 재사용 → 초기 semantic의 날짜 소실 위험(temporal 영향) | DIRECT_EXTRACT_PROMPT를 업스트림 규칙으로 분리 |
| R4-N3 | r=2 원문 첨부에 role 접두사 누락 (업스트림 `role: content`). LoCoMo에선 content의 "(date) speaker:" 접두 덕에 실측 영향 없음 | `_attach_sources`에 role 접두사 추가 |
| R4-N4 | 경계 신호 오배치: "speaking of which"/"quick question"은 업스트림 **Structural**, temporal은 "earlier/before/by the way/oh right/also"+30분 갭 | BOUNDARY_PROMPT 재분류 |
| R4-N5 | 에피소드 예시 의역: 원문은 "the user expressed interest in going hiking on the upcoming weekend (March 16, 2024)" | EPISODE_PROMPT 원문 정합 |
| R4-N6 | 업스트림 timestamp 파싱 실패 폴백은 `datetime.now()` — 자기 프롬프트("not the current time")와 모순. 우리는 첫 메시지 날짜(논문 의도에 더 충실) | docstring에 기록 |
| R4-N7 | β_max 도달 시 최신 포함 전체 flush는 v1 수식(M만 flush)과의 편차 — docstring 미기재였음. buffer_size_min=2·search_top_k_semantic=10은 논문이 아니라 **repo config** 출처 | docstring·주석 정정 |
| R4-N8 | 렌더 라인 포맷 차이(업스트림 `- [created_at] content` vs 우리 `title: content (ts)`), 예산 선별(업스트림은 top-k 전량 포함) | 기록만 (공정성 프레임 유지) |

## 3. 프레임 결정 (2026-07-17)

- **온도·콜드스타트는 upstream 충실로 확정** — write 경로는 방법론별 업스트림 온도
  (A-Mem 0.7/0.7, Nemori 0.2/0.7)를 exp config `role_overrides`로 반영.
- **답변 경로는 공통 프레임 유지** — ANSWER_PROMPT(시간 계산 지시 포함)·generate
  t=0.0은 전 방법론 공통. A-Mem 업스트림의 답변 0.7·cat5 이지선다는 미채택(더 엄격).
- 0.6B 로컬 측정은 하드웨어 제약으로 중단. 다음 단계: **최저가 API로 전환하기 전에
  방법론별 LLM 콜 수·토큰량을 역산해 비용 상한을 추정**하고, 지표(F1/BLEU vs LLM
  judge) 검증을 선행한다. 이때 R4-A7/N1의 업스트림 온도가 그대로 적용된다.

## 4. 본 라운드 반영 내역

- `organizers/amem.py`: docstring 이탈 목록 확장(R4-A1/A3/A4), `fidelity` 파라미터
  제거, k=5 출처 정정 (R4-A2)
- `organizers/nemori.py`: BOUNDARY 신호 재분류(R4-N4), EPISODE 예시 원문(R4-N5),
  DIRECT/CALIBRATE 규칙 분리(R4-N2), docstring 편차 추가(R4-N6/N7)
- `retrieval/pipeline.py`: 소스 메시지 role 접두사(R4-N3), 링크 캡 서술 보강(R4-A5)
- `scripts/exp_locomo_conv0.py`: 방법론별 `role_overrides` 온도(R4-A7/N1), stamp에
  기록, docstring 갱신
- `docs/08`: 이슈 표 정밀화(#23/#24 분리, #10 open, 라이브러리판 한정 명시), 단방향
  링크 반영, fidelity 언급 제거
- `docs/research/fidelity-deep-audit.md` §5: 정오표 추가 (round-3 P-doc 3 이행)
- `docs/11`: DIRECT 프롬프트 계보·β_max 편차·라인 참조 갱신

측정 규율: 프롬프트·온도가 바뀌었으므로 **기존 수치와의 비교 불가, 전 config
재인제스트 후 재측정** (측정은 API 전환 결정 전까지 보류 유지).
