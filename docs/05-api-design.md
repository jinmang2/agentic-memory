# API 설계 — Python API & MCP 서버

## 1. Python 공개 API (`agmem.AgenticMemory`)

```python
from agmem import AgenticMemory

mem = AgenticMemory(
    namespace="jinmang2/coding-agent",
    organizers=["nemori", "reasoning_bank"],   # 방법론 조합 가능 (스택형)
    profile="lite",                            # lite | standard | full
    config="agmem.toml",                       # 선택 오버라이드
    sync_write=False,                          # True = 원논문 재현 모드
)

# ---- write ----
mem.add_message(role="user", content="...", timestamp=..., meta={...})
mem.add_task_result(trajectory=[...], outcome="success",   # ReasoningBank/ACE/G-Memory 경로
                    task="...", agent_id="planner")
mem.warm_start(corpus)                         # cold-start 해소: 백필/offline 학습 공통 진입점
mem.flush()                                    # 큐 드레인 대기 (테스트/벤치용)

# ---- read ----
bundle = mem.search("사용자가 선호하는 여행지?",
                    memory_types=["episodic", "semantic", "facts"],
                    k=10, time_range=("2025-01-01", None))
bundle.render(budget_tokens=1600)              # 프롬프트 주입용 텍스트
bundle.items                                   # 구조화 접근 (provenance 포함)

# ---- introspection / evolution ----
mem.log.tail(20)                               # evolution_log (append-only 연산 로그)
mem.snapshot("before-exp3") / mem.restore("before-exp3")   # 로그 재생 기반
mem.stats()                                    # 항목 수, LLM calls/tokens 누계, 저장 용량
mem.capabilities()                             # 감지 결과 + 활성 어댑터 + 강등 이력
```

설계 결정:

1. **organizers는 리스트** — Nemori(대화 증류) + ReasoningBank(전략)처럼 상호보완 조합이 조사에서 자연스럽다고 확인됨 (ReasoningBank Appendix D도 명시). 각 organizer는 자기 memory_type에만 연산.
2. **모든 반환에 provenance** — `source_episode_ids`를 강제해 "이 답의 근거 원문"을 항상 추적 가능 (Nemori/Zep 교훈).
3. **원논문 재현 모드** — `sync_write=True` + `fidelity="paper"` 옵션이면 각 organizer가 원 구현의 기본값(예: A-Mem k=5, evo_threshold=100 / MemoryOS capacity 10/2000/100, heat τ=5)과 알려진 버그 수정 여부까지 스위칭. 재현 실험과 개선 실험을 같은 코드로 분리 실행.

## 2. MCP 서버

### 2.1 도구 설계 (Graphiti 패턴 참조 + 확장)

Graphiti 공식 서버의 검증된 패턴(`add_memory` / `search_memory_nodes` / `search_memory_facts` 분리)을 따르되, 방법론 통합 특성을 반영:

| tool | 파라미터 | 반환 | 비고 |
|---|---|---|---|
| `add_memory` | content, role?, timestamp?, namespace?, meta? | ack + queued op id | 비동기; 즉시 raw 검색 가능 |
| `add_task_result` | trajectory, outcome, task, agent_id? | ack | 전략 증류 경로 (RB/ACE) |
| `search_memory` | query, memory_types?, k?, time_range?, budget_tokens? | rendered context + items(+provenance) | 통합 검색 (기본 진입점) |
| `search_facts` | query, entity?, valid_at?, k? | bi-temporal fact 목록 | graph 활성 시 (Zep 패턴) |
| `get_profile` | namespace | user profile/persona | MemoryOS LPM 패턴 |
| `get_playbook` | section? | ACE playbook (bullet+카운터) | ACE 활성 시 |
| `report_feedback` | memory_ids, helpful: bool | ack | ACE helpful/harmful, G-Memory reward backward |
| `memory_stats` | namespace? | 항목 수/비용 누계/활성 프로파일 | 운영 가시성 |
| `admin_snapshot` / `admin_clear` | name / confirm | ack | 파괴적 연산은 별도 tool + confirm 강제 |

설계 결정:

- **검색을 하나의 `search_memory`로 통합** (Graphiti처럼 nodes/facts를 쪼개는 대신) — LLM 에이전트가 도구를 고르는 부담을 줄임. 단 `search_facts`는 temporal 질의라는 명확한 용도가 있어 별도 유지.
- **`report_feedback`이 차별점** — ACE/G-Memory/ReasoningBank 계열의 "사용 결과가 메모리를 개선"하는 루프를 MCP 레벨로 노출. 에이전트가 답변 후 어떤 메모리가 유효했는지 되먹임.
- 파괴적 도구(`admin_*`)는 read/write 도구와 분리하고 기본 비활성 플래그(`--enable-admin-tools`).

### 2.2 전송/배포

- **stdio** (Claude Desktop/Code, Cursor) + **streamable HTTP** (`:8765/mcp`) 겸용 — FastMCP로 구현.
- namespace = Graphiti `group_id` 패턴 (기본 `"main"`), 클라이언트별 격리.
- 설정 우선순위: CLI 인자 > 환경변수 > `agmem.toml` (Graphiti와 동일 규칙).
- `SEMAPHORE_LIMIT` 상당의 `worker_concurrency` 노출 (로컬 LLM 데몬 보호).
- 배포 3형태: ① `uvx agmem-mcp` (로컬 stdio) ② Docker compose (agmem + vLLM) ③ 기존 Claude Code 세션 연동 예제.

### 2.3 Claude Code 등록 예시

```json
{
  "mcpServers": {
    "agmem": {
      "command": "uvx",
      "args": ["agmem-mcp", "--profile", "lite", "--namespace", "jinmang2"],
      "env": { "AGMEM_LLM_ENDPOINT": "http://localhost:8000/v1" }
    }
  }
}
```

## 3. 벤치 CLI

```bash
agmem-bench longmemeval --variant s --organizer nemori --profile lite \
    --runs 3 --judge gpt-4o-2024-08-06 --reading-method con --history-format json
agmem-bench locomo --organizer memoryos --fidelity paper --sync-write
agmem-bench report results/           # 표 생성: accuracy + calls/tokens/latency ± std
```

- ingest 아티팩트 캐시(`artifacts/`)로 설정 그리드 재평가 시 재-ingest 생략.
- 모든 결과에 `{profile, commit, model, judge, dataset_version, runs}` 스탬프 — 재현성 규율 (Zep-LoCoMo 논란 반면교사).
