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
mem.add_session(traj, outcome="success")       # 코딩 에이전트 세션 1건: 원문 보존 + 증류
mem.research("how do I run the bench?")     # 탐색기 읽기 경로: 파일 뷰를 모델이 grep·read, 인용·지연 동반
mem.warm_start(corpus)                         # cold-start 해소: 백필/offline 학습 공통 진입점
mem.flush()                                    # 큐 드레인 대기 (테스트/벤치용)
mem.consolidate()                              # 유예 위상 명시 트리거: 큐 드레인 + 버퍼 드레인
                                                #   후 organizer별 consolidate() 일괄 호출
                                                #   (dedup/merge/재조직), 적용된 op 수 반환

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
mem.get_playbook()                             # ACE playbook 렌더. playbook을 produces하는
                                                #   organizer가 비활성이면 "" — 읽기도 쓰기와
                                                #   같은 생산자 소유 규칙을 따른다 (docs/04 §3.4)
mem.log.tail(20)                               # evolution_log (append-only 연산 로그)
mem.stats()                                    # 항목 수, LLM calls/tokens 누계
mem.capabilities()                             # 감지 결과 + 활성 어댑터 + 강등 이력
```

`add_session(traj, *, outcome, persist_steps, distill, force, batch_size, admit)`는 `agmem.sessions.SessionTrajectory`
하나를 받아 **원문과 증류물을 함께** 남긴다. `admit`은 세션 단위 admission 정책(`agmem.sessions.SessionAdmission` 또는
같은 모양의 콜러블)으로, 거부 사유를 돌려주면 저장도 증류도 하지 않고 `SessionIngest.admitted=False`로 알린다.
`experience` organizer는 runbook마다 결정적 라벨(`outcome:`·`host:`·`cwd:`·`cited:`·`tasks:`)을 `TAG`로 발행하고, 읽기 쪽에서는
`AttachCitedSteps`가 상위 runbook 히트에 인용 스텝 원문을 Source Messages로 붙인다. `add_task_result`는 태스크 한 줄만 저장하고 궤적을
버리지만(벤치 하네스와 MCP 도구가 넘기는 궤적에는 가리킬 만한 영속 id가 없기 때문에 그대로 둔다),
세션 로그에는 호스트·세션 id·스텝 위치라는 지속적인 신원이 있으므로 스텝마다 `Episode` 하나를
결정적 id(`SessionTrajectory.episode_id`)로 저장하고, organizer가 쓴 runbook이 자기가 읽은 스텝을
`source_episode_ids`로 되짚을 수 있게 한다(docs/research/agent-memory-axes-v1.md §7.1).

- **`on_message` 팬아웃은 하지 않는다.** 세션 스텝은 대화 턴이 아니다. 도구 호출과 그 출력은 한
  에이전트의 작업 기록이지 사용자가 시스템에 건넨 발화가 아니며, 대화형 방법론(A-Mem·Nemori·
  MemoryOS)은 그것을 발화처럼 분절·요약하면서 스텝당 모델 호출을 지불하게 된다. 세션을 소비하는
  방법론은 `on_task_end`로 받고, 원문 episode는 나중에 id로 되읽기 위해 존재한다.
- **멱등이다.** 첫 스텝의 id가 이미 doc store에 있으면 저장도 증류도 건너뛰고
  `already_ingested=True`로 돌아온다. 데몬의 백필이 같은 파일을 다시 훑기 때문에 필요하고,
  두 번째 증류는 두 번째 청구서이기 때문에 중요하다. `force=True`면 다시 저장하고
  (`add_episode`가 INSERT OR REPLACE라 id가 늘지 않는다) 다시 증류한다.
- **저장이 dispatch보다 먼저 끝난다.** 그래야 스텝 dict에 실린 `episode_id` 포인터가 organizer가
  볼 시점에 스토어에서 해소된다.

CLI 진입점은 `python -m agmem.sessions ingest`다. `--dry-run`(doc store만 열고 아무것도 쓰지 않음)과
`--no-distill`(원문만 저장)은 $0 경로이고, 증류는 `--limit N`(N ≤ 20)을 명시해야만 실행되며 설정에
LLM 역할이 없으면 조용히 건너뛰는 대신 종료 코드 2로 거부한다.

`research(query, *, root, refresh, max_steps, budget_tokens)`는 **두 번째 읽기 경로**이고 `search`의 모드가 아니다.
`agmem.explore.export_workspace`가 스토어를 파일 뷰(`sessions/<host>/<id>.md` 원문 트랜스크립트·`runbooks/<id>.md`·
`messages/<YYYY-MM>.md`·`INDEX.md`, 결정적·증분·관리 대상 외 파일 불변)로 내보내고, `Explorer`가 JSON 액션 루프
(`search`=`rg`/`grep`, `list`, `read`, `final`)로 그 디렉터리를 뒤진 뒤 file:line 인용이 붙은 컨텍스트를 돌려준다. 이것이
docs/research/agent-memory-axes-v1.md §6이 정한 v1의 정직한 대조군("세션 원문 + grep 허용 에이전트", LME-V2 AgentRunbook-C
방식)이며, 벡터 경로와 **서로 대체되지 않는다**: `explore` LLM 역할이 없으면 `RuntimeError`로 거부하지 `search()`로 내려가지
않고, 탐색기가 답을 못 내면 `degraded` 사유와 빈 컨텍스트를 돌려준다. 지연은 1급값이다 — `ResearchResult.latency_s`와 스텝별
초, 그리고 같은 이유로 `search(metrics=...)`·`PlannedSearch`도 `latency_s`를 기록한다(LME-V2 LAFS가 정확도를 지연 대비로
채점). 경로는 전부 워크스페이스 안으로 해석되며(`..`·절대경로는 관측으로 거부), 도구는 셸 없이 argv로 실행된다. MCP 도구는
`research_memory`, CLI는 `python -m agmem.explore export|ask`(`ask`는 `--max-steps ≤ 12`, 역할 없으면 종료 2).

미구현(로드맵): `search(time_range=...)` temporal 필터, `mem.snapshot()/restore()` 로그 재생 복원.

설계 결정:

1. **organizers는 리스트** — Nemori(대화 증류) + ReasoningBank(전략)처럼 상호보완 조합이 조사에서 자연스럽다고 확인됨 (ReasoningBank Appendix D도 명시). 각 organizer는 자기 memory_type에만 연산.
2. **모든 반환에 provenance** — `source_episode_ids`를 강제해 "이 답의 근거 원문"을 항상 추적 가능 (Nemori/Zep 교훈).
3. **원논문 재현 모드** — `sync_write=True`면 organizer가 동기 적용되어 원 구현 재현에 가깝다.
   `fidelity=` 스위치는 **Nemori에서 구현됨**: `NemoriOrganizer(fidelity="v1"|"v4"|"upstream")`이
   segmenter(`per_message`/`batch`)·episode merge(`off`/`llm`)·semantic integration
   (`append`/`dedup`/`llm3way`)·consolidation(`off`/`semantic_offline`) 4축을 프리셋으로
   묶고 개별 kwarg로 오버라이드 가능 (docs/11 §4, 스펙:
   `docs/_internal/specs/2026-07-18-nemori-lifecycle-redesign-design.md` — gitignored 내부 문서). 나머지
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
- namespace = Graphiti `group_id` 패턴 (기본 `"main"`). **2026-09-02부터 모든 도구가 선택 인자
  `namespace`를 받는다** — 생략하면 서버가 기동한 기본 namespace. 서버 하나가 namespace를 지연
  개방해 프로세스 수명 동안 유지하며, 두 번째부터는 첫 번째의 임베더를 공유한다(핸드셰이크 ~10초 중
  ~4초가 모델 생성이므로 namespace 추가 비용은 스토어 3개 여는 ~1초). 그 전에는 namespace가 기동
  플래그뿐이라 데몬 하나 = 메모리 한 통이었고, 프로젝트를 나누려면 데몬을 여러 개 띄워야 했다
  (issue #2). namespace는 데이터 디렉터리 아래 경로 한 조각이 되므로 들어오는 자리에서 검증한다
  (`agmem.env.validate_namespace`: `/`·`..`·선행 `.`·공백 거부).
- `agmem.toml`이 읽는 테이블에 `[llm_options]`(`guided_json`)가 포함된다 — 코드는 읽고 있었지만
  문서 목록·`load_config` docstring·예시 파일 어디에도 없어 발견 불가능한 스위치였다. 두 실험
  스크립트가 Python에서 `use_guided_json=False`로 도는데 TOML 경로엔 맞출 방법이 없었다.
- 설정 우선순위: CLI 인자 > 환경변수 > `agmem.toml` > 기본값 (Graphiti와 동일 규칙). 이 규칙이
  실제로 적용되는 곳은 `mcp/server.py::main`이다 — `--config`를 주면 `--profile`이 통째로 무시돼
  TOML의 profile로 뜨면서 로그엔 플래그 값을 찍고 있었다(규칙의 정반대). 이제 `--profile`이
  주어지면 로드한 config의 profile을 덮어쓰고 그 사실을 로그에 남긴다.
- **환경변수 층은 2026-09-02까지 서버에 존재하지 않았다.** 위 규칙을 문서가 주장하는 동안
  `server.py`는 `os.environ`을 한 번도 읽지 않았고, `AGMEM_NAMESPACE`/`AGMEM_DATA_DIR`는 훅만
  읽었다. 지금은 세 변수를 서버와 훅이 같은 코드(`agmem.env`)로 읽는다:

  | 변수 | 서버 플래그 | 뜻 | 기본값 |
  |---|---|---|---|
  | `AGMEM_NAMESPACE` | `--namespace` | 기본 namespace | `main` |
  | `AGMEM_DATA_DIR` | `--data-dir` | 스토어 루트 | `[storage].data_dir`, 없으면 `~/.agmem/data` |
  | `AGMEM_CONFIG` | `--config` | `agmem.toml` 경로 | 없음 (프로파일 기본값) |
  | `AGMEM_DAEMON_URL` | `--host`/`--port` | 훅이 찾는 상주 서버 | `http://127.0.0.1:8765` |

  **한 번 export하면 두 층이 같은 스토어를 연다**는 것이 이 표의 목적이다. 훅은 하네스가 인자를
  주지 않으므로 환경변수가 유일한 경로이고, 서버는 플래그로 덮어쓸 수 있다.
- `agmem.toml`이 읽는 테이블: `[profile]` `[storage]` `[embed]` `[override]` `[write]`
  `[retrieval]` `[llm.<role>]`. `[retrieval]`은 read-path 스텝의 노브
  (`lexical_types` / `link_expansion_cap` / `attach_sources_top_r` / `graph_expansion_cap`)를
  받는다 — `retrieval/steps.py`가 upstream 이탈을 "config로 ablatable"이라 주장하는데 그게
  Python API에서만 참이고 TOML(재현 런북이 쓰는 경로)에선 조용히 무시되고 있었다.
- ~~`SEMAPHORE_LIMIT` 상당의 `worker_concurrency` 노출~~ (2026-08-19 정정: 이 이름의 노브는 어디에도 구현되지 않았다. 실재하는 동시성 노브는 repro 스크립트의 `--workers`다 — `scripts/repro/ingest_parallel.py`의 대화 병렬 워커 수 = in-flight LLM 콜 상한, `scripts/repro/exp_lme_reading.py`의 질문 병렬 워커 수(기본 8).)
- 배포 3형태: ① `uvx agmem-mcp` (로컬 stdio) ② Docker compose (agmem + vLLM) ③ 기존 Claude Code 세션 연동 예제.

### 2.3 Claude Code 등록 예시

```json
{
  "mcpServers": {
    "agmem": {
      "command": "/absolute/path/to/agentic_memory/.venv/bin/agmem-mcp",
      "args": ["--profile", "lite"],
      "env": { "AGMEM_NAMESPACE": "main", "AGMEM_DATA_DIR": "/home/you/.agmem/data" }
    }
  }
}
```

**namespace는 플래그가 아니라 환경변수로 준다.** §2.4의 훅이 같은 변수를 읽으므로 두 층에 같은
값을 두 번 쓰는 대신 한 곳(셸 프로파일, 또는 위처럼 양쪽 등록 블록)에 한 번 쓴다. 2026-09-02
이전의 이 예시는 `--namespace jinmang2`였고 §2.4는 훅 기본값 `claude-code`를 안내했다 — **문서를
그대로 따르면 두 층이 서로 다른 스토어를 열었다**(issue #2). 어느 쪽도 에러를 내지 않고, recall은
그냥 빈 결과를 냈다.

**이 블록은 2026-08-08에 실제로 stdio 위에서 구동해 확인한 형태다**(2026-09-02에 위와 같이 갱신,
`scripts/smoke_product_stack.py`가 양쪽 기본값이 한 스토어에 닿는지 확인한다). 이전 판은 두 군데가
틀려 있었고 둘 다 조용히 틀리는 종류였다.

- `uvx agmem-mcp`은 **이 리포의 코드를 실행하지 않는다** — uvx는 PyPI에서 `agmem`을 받아 온다.
  로컬 설치를 가리키려면 venv의 콘솔 스크립트를 **절대경로**로 줘야 한다. MCP 클라이언트는 리포를
  cwd로 갖지 않으므로 `agmem-mcp`이라고만 쓰면 클라이언트 PATH에 있을 때만 우연히 동작한다.
- `"env": {"AGMEM_LLM_ENDPOINT": ...}`의 그 변수는 **`src/` 어디에서도 읽히지 않는다.** LLM
  엔드포인트는 §2.2의 `[llm.<role>]` 테이블에서 오므로 `--config /path/to/agmem.toml`로 준다.
  존재하지 않는 변수를 설정하는 것은 실패하지 않고 무시되므로, 따라 한 사람은 엔드포인트를 줬다고
  믿는 서버를 얻는다.

기동 비용: 핸드셰이크까지 **9.4–9.8초**(2026-08-08 실측, lite 프로파일, 모델 캐시 있음). 대부분이
임베더고, `SentenceTransformerEmbedder`가 캐시 우선으로 로드하기 전에는 15.5–16.0초였다.

### 2.3.1 데몬 — 훅이 찾는 상주 프로세스 (2026-09-02)

`agmem-mcp --transport http`가 그대로 데몬이다. MCP `/mcp` 외에 훅용 평문 HTTP 라우트를 연다:
`GET /health`(기본 namespace·열린 namespace·`pending_embed` 수·유휴 시간), `POST /hooks/capture`(`add_memory`와
같은 쓰기), `POST /hooks/recall`(`search_memory`와 같은 검색, 항목 텍스트 반환), `POST /hooks/preserve`(압축 직전
트랜스크립트 보존), `POST /hooks/distill`(끝난 세션의 보존 + 증류, 스레드에서 처리하고 즉시 `queued` 반환). 훅은 stdlib
`urllib`만으로 이것들을 부르므로 torch도 MCP SDK도 import하지 않는다.

- **라이프사이클(결정 A)**: SessionStart의 `recall` 훅이 `/health` 실패 시 `python -m agmem.mcp.server --transport
  http --idle-timeout 1800`을 detach로 띄운다. 유휴 30분이면 스스로 종료. 호스트 둘(Claude Code·Codex)이 같은
  데몬을 공유한다. `AGMEM_NO_DAEMON=1`이면 어떤 훅도 띄우지 않는다.
- **부재 시**: `capture`는 에피소드를 doc store에만 쓰고(0.2초, 임베더 안 열음) 데몬을 요청한 뒤 exit 0. 데몬은
  기동 시와 매 `--backfill-period`(기본 60초)마다 "doc store에는 있고 vector store에는 없는" 에피소드를 임베딩한다.
  `recall_prompt`는 doc store만 열어 BM25(runbook → 지난 턴, 프로젝트 게이팅, 4자 이상 토큰 겹침 필터)로 답하고
  헤더에 어느 경로가 답했는지 적는다(2026-09-05; 데몬 기동이 약 20초라 그 사이의 프롬프트가 조용히 빈손이었다 —
  `docs/23` §8). **어떤 훅도 인프로세스로 모델을 올리지 않는다** — 그것이 이슈 #2 §1의 문제 자체였다.
- **기동 시간**: 훅이 띄운 데몬이 `/health`에 답하기까지 약 20초(2026-09-05 실측, lite, 모델 캐시 있음; 모델 적재 +
  기동 backfill). 콜드 데몬의 첫 벡터 질의 자체는 0.17초라 예열은 필요 없다. 그 20초 동안 `/health`는 실패하므로 그 사이에 발화한
  훅이 두 번째 데몬을 띄우던 경합이 있었다(2026-09-05 도그푸드 로그: 진 쪽이 kuzu 파일 잠금에서 트레이스백으로 죽는다). 이제
  `ensure_running`은 스폰을 임시 디렉터리의 마커(`agmem-daemon-<port>.spawn`)에 적고, 60초 안의 마커는 "뜨는 중"으로 믿고 다시
  띄우지 않는다. 데몬의 stdout/stderr는 `AGMEM_DAEMON_LOG`가
  가리키는 파일에 붙는다(없으면 버림) — 데몬을 띄우는 훅(recall·capture·preserve·distill)이 이 변수를 데몬에 넘긴다.
- 루프백 전용, 인증 없음. `AGMEM_DAEMON_URL`을 루프백 밖으로 두지 말 것.

### 2.4 Claude Code 훅 등록

MCP는 도구라서 모델이 **부르기로 결정해야** 동작한다. 훅은 결정 없이 발화하며, 그게 자동 캡처의
전제다. `~/.claude/settings.json`(또는 프로젝트 `.claude/settings.json`):

```json
{
  "hooks": {
    "SessionStart": [
      { "hooks": [{ "type": "command",
                    "command": "/absolute/path/to/.venv/bin/python -m agmem.hooks.recall",
                    "timeout": 10 }] }
    ],
    "UserPromptSubmit": [
      { "hooks": [{ "type": "command",
                    "command": "/absolute/path/to/.venv/bin/python -m agmem.hooks.recall_prompt",
                    "timeout": 5 },
                  { "type": "command",
                    "command": "/absolute/path/to/.venv/bin/python -m agmem.hooks.capture",
                    "async": true }] }
    ],
    "PreCompact": [
      { "hooks": [{ "type": "command",
                    "command": "/absolute/path/to/.venv/bin/python -m agmem.hooks.preserve",
                    "timeout": 10 }] }
    ],
    "SessionEnd": [
      { "hooks": [{ "type": "command",
                    "command": "/absolute/path/to/.venv/bin/python -m agmem.hooks.distill",
                    "timeout": 5 }] }
    ]
  }
}
```

훅은 다섯이다. **`distill`(SessionEnd, 2026-09-05 신설)** 은 끝난 세션의 트랜스크립트를 데몬에 넘기고 즉시 종료하며, 데몬이
백그라운드에서 `add_session(distill=True)`로 원문 보존 + `experience` 증류 1콜을 수행한다(`[llm.distill]`이 없으면 원문만 남고
증류는 명시적으로 건너뛴다). 훅이 띄우는 데몬은 `--organizers experience`로 뜬다. 나머지 넷: `recall`(SessionStart — **이
프로젝트의 최신 live runbook 5건**(이름·outcome·stage·키워드, 세션 요약은 세션당 한 번)을 먼저, 그 뒤에 사용자의 최근 턴 12줄;
`source=compact`이면 **이 세션이 압축 전에 말한 턴**을 대신 돌려준다),
**`recall_prompt`(UserPromptSubmit, 프롬프트를 질의로 데몬에서 top-5를 주입 — 2026-09-02 신설; 데몬이 없으면 §2.3.1의 BM25 폴백)**, `capture`(UserPromptSubmit, 비동기),
**`preserve`(PreCompact, 2026-09-05 신설 — 압축 직전 트랜스크립트 원문을 세션 id 아래 에피소드로 보존, 모델 호출 없음; 데몬이 없으면
스풀에 적고 데몬이 다음 기동 때 처리)**. 같은 이벤트에서 recall_prompt가 capture보다 앞에 와야 자기 프롬프트를 자기에게
되돌려주지 않는다. Codex의 `~/.codex/hooks.json`도 같은 계약이라 그대로 붙는다.

읽기 훅과 MCP `search_memory`는 세션의 cwd로 **프로젝트 게이팅**한다: 다른 프로젝트 트리에서 쓰인 항목(origin)은 서빙하지
않고, cwd를 모르는 항목은 통과한다(`docs/research/agent-memory-axes-v1.md` §6 #9).

- 네임스페이스·저장 위치·설정 파일은 `AGMEM_NAMESPACE`/`AGMEM_DATA_DIR`/`AGMEM_CONFIG`로 준다
  (기본 `main`, `~/.agmem/data`, 없음). **서버와 같은 변수, 같은 기본값**(§2.2 표, `agmem.env`).
  훅 기본 namespace는 2026-09-02까지 `claude-code`였다 — 서버의 `main`과 달라서, 둘 다 기본값으로
  켜면 서로 못 보는 스토어 두 개가 생겼다(issue #2). 훅에는 플래그가 없다: 하네스가 인자를 주지
  않고, 같은 것을 말하는 두 번째 방법은 어긋나는 두 번째 방법이다.
- `AGMEM_CONFIG`는 서버의 `--config`와 같은 파일을 훅에도 적용한다(임베더·스토어 오버라이드).
  단 **훅 프로세스 자신은 organizer를 무시하고 항상 비운다** — 훅은 키 입력마다 도는 것이라 인프로세스로 LLM을
  부르지 않는다. LLM이 도는 곳은 둘뿐이다: 모델이 부르기로 결정한 MCP 도구, 그리고 훅이 띄운 데몬이
  SessionEnd마다 하는 `experience` 증류 1콜(설정에 `[llm.distill]`이 있을 때만; 없으면 원문 보존만). 서버를 손으로
  띄울 때의 기본 `--organizers nemori,reasoning_bank`와 훅이 띄울 때의 `experience`는 그래서 다르고, LLM 엔드포인트가
  없으면 어느 쪽도 과금되지 않는다.
- **우리 머신의 도그푸드 설정(2026-09-05, `docs/23` §8)**: `~/.claude/settings.json`의 `env`에 `AGMEM_CONFIG=~/.agmem/agmem.toml`,
  `AGMEM_DAEMON_LOG=~/.agmem/daemon.log`, 그리고 증류 키(`OPENROUTER_API_KEY`)를 두고, 그 toml은 다음과 같다.

  ```toml
  [profile]
  name = "lite"

  [storage]
  data_dir = "~/.agmem/data"

  [write]
  distill_max_calls = 8               # 한 프롬프트(60K자)를 넘는 세션은 최대 8구간으로 나눠 증류 (기본 1)

  [llm.distill]                       # SessionEnd 증류 (없으면 원문 보존만)
  endpoint = "https://openrouter.ai/api/v1"
  model = "qwen/qwen3.5-9b"
  api_key = "env:OPENROUTER_API_KEY"  # 값이 아니라 변수 이름 — 데몬이 기동할 때 푼다
  temperature = 0.1
  max_tokens = 4096                   # 2048이면 runbook JSON이 잘린다 (docs/23 §4)

  [llm.explore]                       # MCP research_memory의 탐색 에이전트
  endpoint = "https://openrouter.ai/api/v1"
  model = "qwen/qwen3.5-9b"
  api_key = "env:OPENROUTER_API_KEY"
  temperature = 0.0
  max_tokens = 2048
  ```

  9B 기준 증류 비용은 콜당 $0.002 안팎이다. `distill_max_calls`가 1이면 긴 세션은 머리 2/3·꼬리 1/3만 보고 한 번 부르고,
  N이면 렌더가 필요로 하는 만큼(최대 N)의 연속 구간으로 나눠 한 구간에 한 번씩 부른다 — 스텝 라벨은 세션 전체 번호라 어느 구간의
  인용이든 같은 에피소드를 가리키고, 구간 밖 인용은 거부된다(`docs/23` §6: 4일짜리 세션이 1콜에서 6% 가시, 16구간 ≈ $0.03).
  훅 설정은 새 세션부터 적용되며, 현재 세션에서 바로 쓰려면 `/hooks`를 한 번 연다.
  recall은 SQLite 문서 스토어를 직접 열므로 doc_store를 다른 것으로 오버라이드한 설정은 거부한다
  (옆에 빈 SQLite를 만들어 "기억 없음"으로 보이는 대신 로그에 남기고 exit 0).
- **예산 실측(2026-08-08, 인프로세스)**: recall **0.18초**(doc store만), capture **10.8초**(임베더를 프로세스마다
  올리던 때). 2026-09-02 재측정은 웜 12.9~19초, 콜드 캐시 56초였고, 그래서 §2.3.1의 데몬으로 갔다. 데몬 경로의
  수치는 `scripts/smoke_product_stack.py --daemon`이 출력한다.
- 진단은 `AGMEM_HOOK_LOG=/path/to/log`(훅 자신), `AGMEM_DAEMON_LOG`(훅이 띄운 데몬). 훅은 모든 실패 경로에서 exit 0이므로 로그를 켜지 않으면
  고장이 침묵으로 나타난다 — 세션을 망가뜨리지 않기 위한 설계이고, 그 대가다.
- **교차 검증됨**: capture가 쓴 에피소드가 MCP `search_memory`로 조회된다. 두 층이 한 스토어를
  공유한다는 주장은 각 층의 테스트로는 확인되지 않아 `scripts/smoke_product_stack.py`로 구동해
  확인한다. 2026-09-02부터 스모크는 **어느 쪽에도 namespace를 알려주지 않고** 돌려서 기본값끼리
  같은 디렉터리에 닿는지를 판정에 포함한다 — 이전 판은 양쪽에 같은 값을 명시해서 issue #2의
  불일치를 구조적으로 볼 수 없었다.
- **테스트 밀폐화(2026-09-02)**: `tests/test_hooks.py`는 `AGMEM_CONFIG`로 `FakeEmbedder`를 강제한다.
  그 전엔 훅이 lite 프로파일을 못박아 주입할 이음매가 없었고, 픽스처가 HOME을 임시 경로로 바꿔
  HF 캐시까지 안 보였으므로 **실행마다 471MB 모델을 내려받고** 있었다(이날 120초 타임아웃으로
  2건 ERROR). 실제 모델 경로는 스모크 스크립트가 맡는다.

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

> **(2026-08-19 정정)** 위 문단은 더 이상 사실이 아니다: 드라이버는 이후
> `scripts/repro/exp_lme_reading.py`로 작성됐고 승인된 예산 안에서 실제로 실행됐다 —
> oracle 4개 arm과 retrieval/context arm들의 측정 결과는 `docs/20-lme-reading.md`,
> 아티팩트는 `results/repro/gpt-4o-mini_lme_*`. organizer(메모리 시스템) arm만이
> 여전히 실행된 적 없다 (docs/20의 의도적 보류).

- ingest 아티팩트 캐시(`artifacts/`)로 설정 그리드 재평가 시 재-ingest 생략.
- 모든 결과에 `{profile, commit, model, judge, dataset_version, runs}` 스탬프 — 재현성 규율 (Zep-LoCoMo 논란 반면교사).
