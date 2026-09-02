# v0 요약 — 아젠틱 메모리, 재현과 충실성

## v0가 무엇이었나

아홉 가지 방법론(A-Mem · Nemori · Mem0 · MemoryOS · Zep/Graphiti · G-Memory · ACE · ReasoningBank · MemMachine)을 **발표 논문 대조 하에서** 재구현했다. 결함 원장은 세 등급(Tier A 5건 · Tier B 8건 · Tier C 10건, 항목 헤딩 기준)이고, 12라운드 감사의 검증 패스는 96개 판정 중 94건 확정·2건 반박이며 반박 2건은 철회했다(docs/17). 측정 하네스 (LoCoMo 10-conversation + LongMemEval 500-question), 통합 정책 단계 (A-MAC admission gate + MemMachine retrieval agent). 제품 레이어 세 집합: MCP 서버 (`agmem-mcp`), 훅 계약 3개 (recall / recall_prompt / capture), 데몬 1개 (idle 퇴출).

## 증명한 것

| 주장 | 수치·조건 | 출처 |
|---|---|---|
| **LoCoMo 상단(A>B) 분리 불가** (X1) | Nemori arm A 67.60 · arm B 65.78 · A-Mem 61.23 (J, gpt-4o-mini, LLM judge, n=1,540); 골드 감사 오류 문항을 빼고 재채점해도 A–B 순서는 판별 불가 | docs/18 헤드라인 표 + docs/research/locomo-gold-audit-replay.md |
| **LME C4 성립**: 메모리 고정, reader/prompt만 바꿔도 12.57~15.40pp | gpt-4o-mini 83.89 → gpt-5.6-luna 92.14 (task-avg), 5건 사전등록 판정 규칙 통과 | docs/20 Results · §9.1–2 |
| **LME C6**: 검색만으로 write 콜 0개, +21.20pp | `_s`(gpt-4o-mini) full-context overall 60.40 → retrieval 81.60, CI [+16.60, +25.60]; oracle 83.60 대비 −2.00pp | docs/20 `_s` 절 |
| **X2**: 뺄셈형 정책 상한 ≈ 0 | 기여도 0인 항목 절단해도 4 arm 모두 ΔJ ∈ [−0.39, +0.20], McNemar p>0.6 | docs/research/x2-write-oracle.md 결과표 |
| **ACE/ReasoningBank FiNER 무학습** | base 48.24, ACE online/nodedup/retry 모두 48.24와 미분리 (CI 0 포함), ReasoningBank도 48.24 | docs/19 Results · ReasoningBank 절 |
| **데몬 전환** | capture 훅: 인프로세스 임베더 시절 웜 12.9~19 s·콜드 56 s → 데몬 없음 0.8~1.4 s(doc store만)·데몬 웜 0.46~0.58 s; recall_prompt 0.8~1.6 s (2026-09-02, 실제 임베더, 부하 중 머신) | docs/05 §2.3.1, 커밋 d63a7fd |

## 증명하지 못한 것 / 하지 않은 것

- **organizer × `_s`** — 배선·견적·스모크 완료, 본런은 지출 미승인(docs/20 §10.2)
- **`_s` × luna** — full-context만 실측(task-avg 86.19 / overall 89.20, docs/20); luna 위의 retrieval 아암과 paired 비교는 미실행
- **poisoning robustness** — 미측정
- **substrate axis** — KV cache 대 parametric memory, 미측정
- **RL policies** — 미측정
- **external release** (PyPI · Hugging Face) — Python API와 MCP만, 바이너리 패키징 미포함

## 결론 한 줄

쓰기 경로 방법론은 무학습·무조직화 기준선과 분리되지 않았고(ACE·ReasoningBank on FiNER, X2 감산 오라클 상한 ≈ 0, LoCoMo 상단 밴드 판별 불가), 점수는 읽기 경로에 있다(C4·C6). 그리고 호스트(Claude Code·Codex)는 벡터 없이 마크다운 + grep 메모리를 네이티브로 출하한다(docs/research/product-memory-landscape.md).

## v1의 첫 문장

v1은 도메인을 대화 QA에서 **에이전트 궤적**으로 옮기고, 우리 스택이 호스트 네이티브 메모리보다 나은지를 **LongMemEval-V2와 우리 자신의 세션**에서 잰다 (docs/_internal/plans/2026-09-02-v1-experience-memory.md).

---

## 찾지 못한 수치 및 포인터

- **LoCoMo 5-arm 하드웨어 비용** — compute 시간만 있고 USD 환산 없음
- **Zep X2 재실행** — 절단 완료·스모크 통과, 재실행 미완료 ($0.36 예산)
- **LongMemEval organizer N/A 분해** (write vs read 실패) — capture 개선 2026-08-19 이후 적용 안 됨
- **ACE/RB null의 일반화** — single-task·single-seed, FiNER/gpt-4o-mini만
