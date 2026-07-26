# API 설계 — Python API & MCP 서버

## 1. Python 공개 API (`agmem.AgenticMemory`)

```python
from agmem import AgenticMemory
from agmem.config import AgmemConfig

mem = AgenticMemory(
    namespace="jinmang2/coding-agent",
    organizers=["nemori", "reasoning_bank"],   # 방법론 조합 가능 (스택형)
    config=AgmemConfig(profile="lite",         # lite | standard | full
                       sync_write=False),      # 또는 "agmem.toml" 경로
)
```

`profile=`은 `AgmemConfig(profile=...)`의 축약이고 **`config=`가 없을 때만** 쓴다
(`AgenticMemory(profile="standard")`). 둘 다 주면서 값이 다르면 `ValueError`다 — 예전엔
`profile=`이 조용히 버려져서 슬롯 해석과 `resolved_embed_model`이 config 쪽 profile로 가는데
`stats()`/`capabilities()` 어디에도 요청한 profile의 흔적이 남지 않았다. 라이브러리에는
적용할 우선순위 규칙이 없으므로(config 객체는 하나고 병합 순서가 없다) 불일치는 호출자 버그다.
CLI 우선순위가 정의된 곳은 MCP 서버뿐이다 (§2.2).

`close()`는 스토어 핸들과 write 워커를 모두 정리하므로 **필수**이고, `with` 형태를 지원한다:

```python
with AgenticMemory(organizers=["amem"], config=cfg) as mem:
    ...
```

`organizers`는 **레지스트리 이름과 `Organizer` 인스턴스를 섞어 받는다.** 합성(policy 부착,
체이닝)은 인스턴스로만 표현되므로 — 생성자 인자로 두면 policy가 그 mechanism 하나에서만
도달 가능해진다(docs/04 §1.2) — 이 형태가 유일한 진입점이다:

```python
from agmem.organizers.amem import AMemOrganizer
from agmem.organizers.gated import AdmissionGated
from agmem.organizers.experimental.chained import ChainedConsumer
from agmem.policies import AdmissionGate

mem = AgenticMemory(organizers=[
    AdmissionGated(AMemOrganizer(), AdmissionGate()),   # write-admission policy 부착
    ChainedConsumer(AMemOrganizer(), "semantic"),       # 상류 organizer 출력을 입력으로 (experimental)
])

# ---- write ----
mem.add_message(content="...", role="user", timestamp=..., meta={...})
mem.add_task_result(trajectory=[...], outcome="success",   # ReasoningBank/ACE/G-Memory 경로
                    task="...", agent_id="planner")
mem.warm_start(corpus)                         # cold-start 해소: 백필/offline 학습 공통 진입점
mem.flush()                                    # 큐 드레인 대기 (테스트/벤치용)
mem.consolidate()                              # 유예 위상 명시 트리거: organizer별 consolidate()
                                                #   일괄 호출(dedup/merge/재조직), 적용된 op 수 반환

# ---- read ----
bundle = mem.search("사용자가 선호하는 여행지?",
                    memory_types=["episodic", "semantic", "facts"], k=10)
                                               # 생략 시 default_memory_types =
                                               #   episodic + 활성 organizer들의 produces
                                               # 주의: search는 on_retrieval 되먹임으로
                                               #   쓰기도 한다 (docs/04 §2 Read)
bundle.render(budget_tokens=1600)              # 프롬프트 주입용 텍스트
bundle.items                                   # 구조화 접근 (provenance 포함)

# ---- feedback / introspection ----
mem.report_feedback([...], helpful=True)       # 사용 결과 되먹임 → 각 organizer의 on_feedback
                                               #   (ACE 카운터 / G-Memory reward). 규칙은
                                               #   생산자 소유이므로 해당 organizer가
                                               #   비활성이면 0을 반환하는 no-op
mem.get_playbook()                             # ACE playbook 렌더
mem.log.tail(20)                               # evolution_log (append-only 연산 로그)
mem.stats()                                    # 항목 수, LLM calls/tokens 누계
mem.capabilities()                             # 감지 결과 + 활성 어댑터 + 강등 이력
```

미구현(로드맵): `search(time_range=...)` temporal 필터, `mem.snapshot()/restore()` 로그 재생 복원.

설계 결정:

1. **organizers는 리스트** — Nemori(대화 증류) + ReasoningBank(전략)처럼 상호보완 조합이 조사에서 자연스럽다고 확인됨 (ReasoningBank Appendix D도 명시). 각 organizer는 자기 memory_type에만 연산.
2. **모든 반환에 provenance** — `source_episode_ids`를 강제해 "이 답의 근거 원문"을 항상 추적 가능 (Nemori/Zep 교훈).
3. **원논문 재현 모드** — `sync_write=True`면 organizer가 동기 적용되어 원 구현 재현에 가깝다.
   `fidelity=` 스위치는 **Nemori에서 구현됨**: `NemoriOrganizer(fidelity="v1"|"v4"|"upstream")`이
   segmenter(`per_message`/`batch`)·episode merge(`off`/`llm`)·semantic integration
   (`append`/`dedup`/`llm3way`)·consolidation(`off`/`semantic_offline`) 4축을 프리셋으로
   묶고 개별 kwarg로 오버라이드 가능 (docs/11 §4, 스펙:
   `docs/superpowers/specs/2026-07-18-nemori-lifecycle-redesign-design.md`). 나머지
   organizer로의 `fidelity=` 확장은 로드맵.
4. **라이프사이클 훅 2종 (인라인/유예)** — `on_memory_event`(다른 organizer의 ADD/UPDATE/
   MERGE를 `consumes` 구독으로 수신, chaining)와 `consolidate`(명시적 `mem.consolidate()`
   호출로만 실행되는 배치 dedup/merge 훅, evolution log seq 커서로 재개)가 매 organizer에
   기본 no-op으로 제공된다. **구독 측을 native로 구현한 방법론은 없다** — 구 `input="episodes"`
   생성자 모드는 제거됐고, 소비는 전부 `experimental.ChainedConsumer` 어댑터가 담당한다
   (방출 측 Nemori는 그대로). 훅별 실제 구현 매트릭스는 docs/04 §3.1.
5. **LLM 비용의 위상별 계측** — `ctx.llm.call(..., phase=...)`은 예산 항목을
   `f"{role}/{phase}"`로 태깅해(`llm/structured.py`) segment/narrate/merge/integrate/
   predict_calibrate/consolidate 단계별 calls/tokens를 역할(role)과 별개로 분리 집계한다
   (`mem.stats()["llm"]`에 노출, docs/04 §4 비용 원칙의 확장).

## 2. MCP 서버

### 2.1 도구 설계 (Graphiti 패턴 참조 + 확장)

Graphiti 공식 서버의 검증된 패턴(`add_memory` / `search_memory_nodes` / `search_memory_facts` 분리)을 따르되, 방법론 통합 특성을 반영:

| tool | 파라미터 | 반환 | 비고 |
|---|---|---|---|
| `add_memory` | content, role?, timestamp? | ack | 비동기 큐 적재 |
| `add_task_result` | task, outcome, trajectory_json?, agent_id? | ack | 전략 증류 경로 (RB/ACE/G-Memory) |
| `search_memory` | query, memory_types?, k?, budget_tokens? | rendered context + items(+provenance) | 통합 검색 (기본 진입점). memory_types는 콤마 구분 문자열 |
| `get_playbook` | section? | ACE playbook (bullet+카운터) | ACE 활성 시 |
| `report_feedback` | memory_ids, helpful: bool | ack | ACE helpful/harmful, G-Memory reward backward |
| `memory_stats` | — | 항목 수/비용 누계/활성 프로파일 | 운영 가시성 |
| `admin_snapshot_log` / `admin_flush` | n / — | 최근 연산 로그 / 큐 드레인 | `--enable-admin-tools`로만 등록 |

로드맵(미구현 도구): `search_facts`(bi-temporal fact 질의, Zep 패턴), `get_profile`(MemoryOS LPM 패턴).

설계 결정:

- **검색을 하나의 `search_memory`로 통합** (Graphiti처럼 nodes/facts를 쪼개는 대신) — LLM 에이전트가 도구를 고르는 부담을 줄임. `search_facts`는 temporal 질의라는 명확한 용도가 있어 도입 시에도 별도 도구로 유지 예정.
- **`report_feedback`이 차별점** — ACE/G-Memory/ReasoningBank 계열의 "사용 결과가 메모리를 개선"하는 루프를 MCP 레벨로 노출. 에이전트가 답변 후 어떤 메모리가 유효했는지 되먹임.
- 파괴적 도구(`admin_*`)는 read/write 도구와 분리하고 기본 비활성 플래그(`--enable-admin-tools`).

### 2.2 전송/배포

- **stdio** (Claude Desktop/Code, Cursor) + **streamable HTTP** (`:8765/mcp`) 겸용 — FastMCP로 구현.
- namespace = Graphiti `group_id` 패턴 (기본 `"main"`), 클라이언트별 격리.
- 설정 우선순위: CLI 인자 > 환경변수 > `agmem.toml` (Graphiti와 동일 규칙). 이 규칙이 실제로
  적용되는 곳은 `mcp/server.py::main`이다 — `--config`를 주면 `--profile`이 통째로 무시돼
  TOML의 profile로 뜨면서 로그엔 플래그 값을 찍고 있었다(규칙의 정반대). 이제 `--profile`이
  주어지면 로드한 config의 profile을 덮어쓰고 그 사실을 로그에 남긴다.
- `agmem.toml`이 읽는 테이블: `[profile]` `[storage]` `[embed]` `[override]` `[write]`
  `[retrieval]` `[llm.<role>]`. `[retrieval]`은 read-path 스텝의 노브
  (`lexical_types` / `link_expansion_cap` / `attach_sources_top_r` / `graph_expansion_cap`)를
  받는다 — `retrieval/steps.py`가 upstream 이탈을 "config로 ablatable"이라 주장하는데 그게
  Python API에서만 참이고 TOML(재현 런북이 쓰는 경로)에선 조용히 무시되고 있었다.
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

## 3. 벤치 실행

현재 진입점은 스크립트다 (`agmem-bench` CLI는 로드맵; pyproject 엔트리포인트는 `agmem-mcp`뿐):

```bash
uv run python scripts/exp_locomo_conv0.py      # LoCoMo conv-0: 방법론별 설정 그리드 실행
```

하니스는 `bench/harness.py`(멀티런 + mean/std 집계), 로더는 `bench/locomo.py`와 `bench/longmemeval.py`.

LongMemEval은 **문자열 지표가 없다** — LLM judge가 유일한 점수원이고 질문 타입별 프롬프트가 5분기다.
따라서 판정 없는 run은 점수가 아예 없고 hypothesis만 남는다(`run_instance(judge=True)` 기본값).
CLI 드라이버는 **의도적으로 미작성** — 인스턴스당 메모리를 새로 만드는 구조라 500질문 실행이
곧 500 ingest이고, 측정 승인 전에는 그 비용을 지불할 수 없다. 라이브러리 수준의 배선
(load/ingest/answer/judge/aggregate + full-context 베이스라인)은 완료돼 있다.

- ingest 아티팩트 캐시(`artifacts/`)로 설정 그리드 재평가 시 재-ingest 생략.
- 모든 결과에 `{profile, commit, model, judge, dataset_version, runs}` 스탬프 — 재현성 규율 (Zep-LoCoMo 논란 반면교사).
