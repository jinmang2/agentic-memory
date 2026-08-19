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
- namespace = Graphiti `group_id` 패턴 (기본 `"main"`), 클라이언트별 격리.
- `agmem.toml`이 읽는 테이블에 `[llm_options]`(`guided_json`)가 포함된다 — 코드는 읽고 있었지만
  문서 목록·`load_config` docstring·예시 파일 어디에도 없어 발견 불가능한 스위치였다. 두 실험
  스크립트가 Python에서 `use_guided_json=False`로 도는데 TOML 경로엔 맞출 방법이 없었다.
- 설정 우선순위: CLI 인자 > 환경변수 > `agmem.toml` (Graphiti와 동일 규칙). 이 규칙이 실제로
  적용되는 곳은 `mcp/server.py::main`이다 — `--config`를 주면 `--profile`이 통째로 무시돼
  TOML의 profile로 뜨면서 로그엔 플래그 값을 찍고 있었다(규칙의 정반대). 이제 `--profile`이
  주어지면 로드한 config의 profile을 덮어쓰고 그 사실을 로그에 남긴다.
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
      "args": ["--profile", "lite", "--namespace", "jinmang2"]
    }
  }
}
```

**이 블록은 2026-08-08에 실제로 stdio 위에서 구동해 확인한 형태다.** 이전 판은 두 군데가 틀려
있었고 둘 다 조용히 틀리는 종류였다.

- `uvx agmem-mcp`은 **이 리포의 코드를 실행하지 않는다** — uvx는 PyPI에서 `agmem`을 받아 온다.
  로컬 설치를 가리키려면 venv의 콘솔 스크립트를 **절대경로**로 줘야 한다. MCP 클라이언트는 리포를
  cwd로 갖지 않으므로 `agmem-mcp`이라고만 쓰면 클라이언트 PATH에 있을 때만 우연히 동작한다.
- `"env": {"AGMEM_LLM_ENDPOINT": ...}`의 그 변수는 **`src/` 어디에서도 읽히지 않는다.** LLM
  엔드포인트는 §2.2의 `[llm.<role>]` 테이블에서 오므로 `--config /path/to/agmem.toml`로 준다.
  존재하지 않는 변수를 설정하는 것은 실패하지 않고 무시되므로, 따라 한 사람은 엔드포인트를 줬다고
  믿는 서버를 얻는다.

기동 비용: 핸드셰이크까지 **9.4–9.8초**(2026-08-08 실측, lite 프로파일, 모델 캐시 있음). 대부분이
임베더고, `SentenceTransformerEmbedder`가 캐시 우선으로 로드하기 전에는 15.5–16.0초였다.

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
                    "command": "/absolute/path/to/.venv/bin/python -m agmem.hooks.capture",
                    "async": true }] }
    ]
  }
}
```

- 네임스페이스·저장 위치는 `AGMEM_NAMESPACE`/`AGMEM_DATA_DIR`로 준다(기본 `claude-code`,
  `~/.agmem/data`). 이 둘은 §2.3의 그 변수와 달리 **실제로 읽힌다**(`hooks/__init__.py`).
- **예산 실측(2026-08-08)**: recall **0.18초**(블로킹이라 `timeout` 안에 반드시 들어와야 함,
  doc store만 열기 때문에 이 값), capture **10.8초**(임베더 필요 — `async: true`가 필수인 이유).
- 진단은 `AGMEM_HOOK_LOG=/path/to/log`. 훅은 모든 실패 경로에서 exit 0이므로 로그를 켜지 않으면
  고장이 침묵으로 나타난다 — 세션을 망가뜨리지 않기 위한 설계이고, 그 대가다.
- **교차 검증됨**: capture가 쓴 에피소드가 MCP `search_memory`로 조회된다(같은 namespace·data-dir
  기준). 두 층이 한 스토어를 공유한다는 주장은 각 층의 테스트로는 확인되지 않아 별도로 구동해 확인했다.

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
