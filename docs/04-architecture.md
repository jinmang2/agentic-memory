# 모듈 구조 설계

> 핵심 아이디어: **retrieval은 공통 인프라, 방법론 차이는 write-path의 `Organizer`로 캡슐화,
> 모든 메모리 변경은 append-only `evolution_log` 연산으로 표현.**

## 1. 패키지 레이아웃

```
agentic_memory/                      # 패키지명: agmem (pip install agmem)
├── pyproject.toml                   # uv 관리, dependency-groups: dev/embed/backends (+train extra)
├── src/agmem/
│   ├── capabilities/                # §01 문서의 capability detection
│   │   ├── detect.py                # HostCapabilities 감지 (+캐시 ~/.agmem)
│   │   ├── requires.py              # Requires 선언, CapabilityWarning
│   │   └── resolver.py              # override > profile > capability 매칭 → 구현 선택
│   │
│   ├── core/                        # 방법론 독립적 도메인 모델
│   │   ├── types.py                 # Episode, Note, Entity, Fact(bi-temporal),
│   │   │                            #   StrategyItem, Bullet, MemoryBundle
│   │   └── ops.py                   # MemoryOp = ADD|UPDATE|MERGE|DELETE|INVALIDATE|LINK|TAG
│   │                                #   + EvolutionLog Protocol (append-only)
│   │
│   ├── stores/                      # 저장 어댑터 (전부 동일 인터페이스)
│   │   ├── base.py                  # DocStore / VectorStore Protocol
│   │   ├── sqlite_doc.py            # episodes/notes/... + FTS5 + evolution_log (lite/standard)
│   │   ├── postgres_doc.py          # embedded PostgreSQL(pgserver) + tsvector (full)
│   │   ├── sqlite_vec.py            # lite 기본
│   │   ├── lance_vec.py             # standard
│   │   ├── qdrant_vec.py            # full
│   │   ├── chroma_vec.py            # 원논문 재현용 (A-Mem/G-Memory fidelity)
│   │   ├── numpy_vec.py             # 테스트 전용 (런타임 후보 제외)
│   │   ├── sqlite_graph.py          # 재귀 CTE k-hop (최후 수단)
│   │   ├── kuzu_graph.py            # embedded 실물 그래프 (lite/standard 기본)
│   │   └── neo4j_graph.py           # full (Zep/Graphiti 자체 엔진)
│   │
│   ├── llm/                         # 모델 접근 (전부 OpenAI-compatible)
│   │   ├── client.py                # 역할별 라우팅: extract/distill/judge/rerank/generate
│   │   ├── structured.py            # guided_json + 재시도 + drop 카운터
│   │   └── budget.py                # calls/tokens/latency 계측 (1급 메트릭)
│   │
│   ├── embed/
│   │   ├── base.py                  # Embedder Protocol + Embeddable(embedding_text())
│   │   ├── st_embedder.py           # sentence-transformers (e5-small/bge-m3 등)
│   │   └── fake.py                  # 결정적 해시 임베더 (테스트 전용)
│   │
│   ├── retrieval/                   # §03 문서의 파이프라인
│   │   ├── pipeline.py              # Recall(dense/lexical/graph) → Fusion → Rerank
│   │   │                            #   + 링크/그래프/experience 확장, bi-temporal 렌더
│   │   ├── fusion.py                # RRF
│   │   └── rerank.py                # Noop / MMR / LLMReranker / CrossEncoder
│   │
│   ├── organizers/                  # ★ mechanism = 방법론 = Organizer 플러그인
│   │   │                            #   ── 규칙: 루트의 평범한 모듈 = 프레임워크,
│   │   │                            #   방법론은 전부 자기 패키지. 예외 없음.
│   │   │                            #   테스트로 강제: test_organizers.py
│   │   │                            #   ::test_organizers_root_is_framework_only_...
│   │   ├── base.py                  # [프레임워크] Organizer Protocol:
│   │   │                            #   on_message / on_task_end → list[MemoryOp]
│   │   │                            #   (+ warm_start, on_retrieval, on_feedback,
│   │   │                            #   flush_buffer, retire/patch_unit,
│   │   │                            #   on_memory_event(consumes 구독)/consolidate — §2)
│   │   ├── gated.py                 # [프레임워크] AdmissionGated: policies/의 결정을
│   │   │                            #   임의 organizer에 적용하는 합성 어댑터
│   │   ├── __init__.py              # [프레임워크] name→class 레지스트리
│   │   │
│   │   │                            # ── 방법론: 각 패키지가 organizer.py + __init__.py
│   │   │                            #   재수출(기존 import 경로 그대로) + 필요 시 내부 모듈
│   │   ├── passthrough/             # no-op baseline (raw episode만)
│   │   ├── amem/                    # 노트 구성→링크→이웃 진화 (버그 수정판 명시)
│   │   ├── memoryos/                # STM/MTM/LPM + heat 승격 + LFU eviction
│   │   ├── nemori/                  # 내부 스테이지를 가진 예시
│   │   │   ├── __init__.py          #   NemoriOrganizer 재수출
│   │   │   ├── organizer.py         #   boundary 분절→서사(시간 절대화)→predict-calibrate 증류
│   │   │   └── stages.py            #   Segmenter/EpisodeMerger/Integrator/Consolidator
│   │   │                            #   (fidelity 스위치, docs/11 §4)
│   │   ├── memmachine/              # 기계적 derivative 인덱싱 (write LLM 콜 0회).
│   │   │                            #   배포 코드 계보, MEMMACHINE_PRESETS로 declarative/event
│   │   │                            #   분리 — 읽기는 retrieval/steps.py의 Contextualize
│   │   ├── zep_graph/               # entity 추출→resolution→fact→invalidation
│   │   ├── ace/                     # Generator/Reflector/Curator + playbook delta
│   │   ├── reasoning_bank/          # self-judge→성공/실패 증류→append (+MaTTS 훅)
│   │   ├── gmemory/                 # MAS 궤적 sparsify→insight, reward 기반 프루닝
│   │   └── experimental/            # 논문 재현이 아닌 합성 (의미적 격리 — 위 위치 규칙과
│   │                                #   직교. ChainedConsumer + Nemori experimental 스테이지)
│   │
│   ├── policies/                    # ★ control policy = mechanism과 직교하는 cross-cutting 규칙
│   │   │                            #   소속 판정: 메모리 타입 미선언 + MemoryOp 미발행.
│   │   │                            #   근거/조사: docs/research/memory-component-taxonomy.md
│   │   └── admission.py             # store 연산 게이트: A-MAC(2603.04549) 구현,
│   │                                #   SAGE(2605.30711)가 같은 seam의 다음 후보.
│   │                                #   적용은 organizers/gated.py wrapper 경유 —
│   │                                #   어떤 mechanism도 이 패키지를 import하지 않는다
│   │                                #   (organizers 중 gated.py만 import)
│   │
│   ├── memory.py                    # AgenticMemory 퍼사드 (05 문서의 공개 API)
│   │                                #   + 비동기 write 워커 (내장 스레드+큐, sync_write=False 시)
│   ├── config.py                    # TOML 로딩 + profile 프리셋
│   │
│   ├── mcp/                         # MCP 서버 (05 문서)
│   │   └── server.py                # FastMCP, stdio + streamable HTTP
│   │
│   ├── bench/                       # 평가 하네스
│   │   ├── harness.py               # multi-run + mean/std 집계, 재현성 스탬프
│   │   └── locomo.py                # LoCoMo 로더, F1/BLEU-1 (judge 불필요 경로)
│   │
│   └── train/                       # 0.5B 보조모델 학습 (roadmap Phase 4)
│       ├── distill_data.py          # 대형 모델로 추출/분절/증류 SFT 데이터 생성
│       └── sft_lora.py              # peft QLoRA (RTX 2060 6GB 타깃)
│
├── scripts/                         # 실험 진입점 (exp_locomo_conv0.py 등)
├── tests/                           # pytest
└── docs/                            # 본 문서들
    └── 12-code-conventions.md       # 이름/docstring/구조 컨벤션 (리뷰 게이트 기준)
```

로드맵(미구현 모듈): `core/namespace.py`, `retrieval/expand.py`(time-range 추출), `embed/api_embedder.py`, `bench/judges.py·report.py`, `train/eval_extract.py`, graph store용 `QueueStore`/`GraphStore` 공통 Protocol.

`bench/longmemeval.py`는 구현됨 — LoCoMo와 **구조가 다르다**: LoCoMo 샘플 1개 = 대화 1개 + 질문 다수(메모리 1개로 전부 처리)인데, LongMemEval 인스턴스 1개 = **질문 1개 + 자기 haystack ~40세션**이라 충실한 실행은 질문마다 메모리를 새로 만든다(`_s` 기준 500개). 그래서 `ingest`가 corpus가 아니라 instance를 받고, 인스턴스별 메모리 수명은 호출자가 소유한다(`run_instance`).

### 1.1 mechanism vs control policy — 새 논문을 어디에 넣을지 정하는 규칙

`organizers/`와 `policies/`의 분리는 우리 편의가 아니라 **문헌 자체의 분해**를 따른다. 서베이
arXiv:2603.07670은 메모리 시스템을 write–manage–read로 모델링하고(read 연산자 `R(M,x)`, update
연산자 `U(M,x,a,o,r)`, 연산 집합 *store / retrieve / update / summarize / discard*) 두 층위를
명시적으로 구분한다:

- **mechanism** (§4) — 시스템이 *무엇인지*. 자기 메모리 표현과 읽기 경로를 소유한다. → `Organizer`
- **control policy** (§3.3) — 그 연산이 *언제/어떻게 발동하는지*를 지배하는 heuristic/prompted/
  learned 규칙. 서베이가 "mechanism 선택과 **직교하는 cross-cutting 차원**"이라 부르는 것. → `policies/`

**판정 기준(운영적)**: *policy는 메모리 타입을 선언하지 않고 `MemoryOp`도 발행하지 않는다.*
저장 상태와 그것을 되읽는 방법을 소유하면 mechanism이다. 이 기준으로 조사 단계에서 잘못
분류했던 두 건을 교정했다 — RecMem(2605.16045)과 GRAVITY(2605.01688)는 각자 메모리 계층을
소유하므로 **policy가 아니라 mechanism**이다(`docs/research/memory-component-taxonomy.md` §2).

**세 번째 범주**: 한 방법론이 소유한 내부 스테이지(Nemori의 boundary/merge/integration 전략).
이것은 그 방법론의 논문 메커니즘이므로 별도 패키지가 아니라 **소유자 패키지 안**에 둔다
(`organizers/nemori/stages.py`).

### 1.2 `organizers/` 위치 규칙 — 예외 없는 형태

**루트의 평범한 모듈 = 프레임워크. 방법론은 전부 자기 패키지.**

| 위치 | 내용 |
|---|---|
| `organizers/base.py` | Organizer contract |
| `organizers/gated.py` | 합성 어댑터(policy 적용) |
| `organizers/__init__.py` | name→class 레지스트리 |
| `organizers/<methodology>/` | 방법론 1개. `organizer.py` + `__init__.py` 재수출 (+내부 모듈) |
| `organizers/experimental/` | 의미적 격리(논문 재현 아님). 위 규칙과 **직교** |

이전 판의 규칙은 "루트는 Organizer 서브클래스만"이었고 `base.py`/`__init__.py`를 **예외로
빼뒀다** — 그런데 예외 목록이야말로 루트가 처음 섞인 원인이었다. 게다가 그 규칙은 `gated.py`를
통과시키면서 범주를 다시 섞었다(합성 어댑터는 프레임워크이지 방법론이 아니다). 그래서 규칙을
**위치 기반·예외 없음**으로 바꿨다: 루트의 평범한 모듈은 프레임워크이고, 논문을 구현하는 것은
전부 자기 패키지를 가진다 — 내부 스테이지가 생길 자리도 그 안이다(Nemori가 이미 그 형태).

단일 파일 방법론까지 패키지로 만드는 ceremony 비용이 있지만, `__init__.py`가 organizer를
재수출하므로 **호출부 40여 곳이 전부 무변경**이고(`from agmem.organizers.amem import
AMemOrganizer` 그대로 해석), 방법론이 내부 모듈을 갖게 될 때 루트를 다시 오염시키지 않는다.
`test_methodology_packages_keep_their_single_module_import_path`가 경로 보존을 고정한다.

**policy를 mechanism에 붙이는 방법**: 생성자 인자가 아니라 wrapper
(`organizers/gated.py::AdmissionGated`). 생성자 인자로 두면 policy가 그 mechanism 하나에서만
도달 가능해져 "cross-cutting"이 말뿐이 되고, mechanism이 policy를 import하게 된다. wrapper면
한 번 구현으로 모든 message 기반 organizer에 적용되고, **어떤 mechanism도 `policies`를 import하지
않는다** — `organizers/` 안에서 `policies`를 import하는 모듈은 어댑터인 `gated.py` 단 하나이고
(`policies`는 `organizers.base`의 `OrganizerContext`만 역참조한다), 이것이 직교성의 실제 증거다. **단 적용 범위는 정책의 seam이 정한다**: admission은
`on_message`가 있는 organizer에만 의미가 있고 task 기반(ACE/G-Memory/ReasoningBank)에는
원리적으로 적용 불가다. 검증된 매트릭스는 `gated.py` 모듈 docstring과 taxonomy 문서 §2.5.

## 2. 데이터 흐름

### Write (모든 방법론 공통 골격)

```
add_message(msg) / add_task_result(traj)
  │  sync: DocStore에 raw episode 기록 (+즉시 검색 노출), queue enqueue, <10ms 반환
  ▼
worker: Organizer.on_message/on_task_end 실행 (LLM 호출 발생 지점)
  │  산출물 = list[MemoryOp]
  ▼
EvolutionLog.append(ops) → 각 store에 반영 (vector upsert, graph upsert/invalidate, ...)
  │  applied ADD/UPDATE/MERGE op → MemoryEvent(source, op, target_type, target_id,
  │  payload, supersedes) 로 변환
  ▼
_propagate_events: target_type ∈ consumes 인 다른 organizer에 순서대로 전달
  │  (자기 자신 제외, depth=1 — 응답 op는 적용되지만 재전파 안 됨;
  │  DELETE/INVALIDATE는 전파하지 않음 — supersedes는 MERGE에 실려 원자적으로 전달)
  ▼
(명시적) AgenticMemory.consolidate() → 등록 순서대로 Organizer.consolidate(ctx) 호출
  │  각자 evolution log seq 커서(target_type="state", id="consolidate:{name}")를 읽어
  │  이후분만 배치 처리(dedup/merge/재조직) → INVALIDATE+ADD/UPDATE 반환, 마지막에 커서 전진
```

- Organizer는 store를 직접 만지지 않고 **MemoryOp만 반환** → 방법론 코드와 스토리지가 완전 분리, 로그 재생으로 상태 복원/디버깅 가능.
- 동기 모드(`sync_write=True`)도 지원 — 원논문 재현 실험은 동기로 돌려 원 구현과 조건을 맞춘다.
- **인라인 vs 유예 위상**: `on_message`/`on_task_end`/`on_retrieval`/`on_feedback`/
  `on_memory_event`/`flush_buffer`는 인라인(ingest 경로에서 즉시 실행, 방법론 원형 재현의 자리)이고
  `consolidate`는 유예(명시적 API 호출로만 실행, 배치 dedup/merge/재조직의 자리) —
  두 위상은 서로를 강제하지 않는다 (스펙 §1.1).
- INVALIDATE는 기존 `invalid_at`을 보존(최초 시각 유지)하며, bi-temporal 렌더 타입
  (`memory.py::BITEMPORAL_TYPES` = `facts`)이 아니면 벡터도 함께 제거한다 — semantic/
  episodes 등은 무효화 즉시 검색에서 빠지고, facts는 validity 구간과 함께 계속 렌더된다.

### Read

```
search(query, memory_types=[...], k=...)
  → retrieval.pipeline (타입별: dense+lexical recall → RRF → rerank → hydrate → 확장)
  → MemoryBundle { episodes, facts, semantic, strategies, playbook, provenance }
```

- QueryExpansion(time-range 추출 등, docs/03 §1.2)은 로드맵 — 현재 파이프라인은 recall부터 시작.
- recall은 memory_type별로 순차 실행(스레드/async 병렬화 없음); dense는 항상, lexical은
  `episodic`과 `lexical_types`에 포함된 타입만 BM25/FTS 채널을 추가로 RRF 융합한다.
- `memory_types`를 생략하면 `default_memory_types` — `episodic` + 활성 organizer들의
  `produces`를 선언 순서대로 dedup한 것 — 이 쓰인다. `produces`의 **순서가 load-bearing**이다:
  확장 스텝이 이미 번들에 있는 id를 제외하므로, 다른 타입의 스텝이 끌어오는 타입이 먼저 와야
  한다(zep_graph의 facts→entities).

**read→write 되먹임 (읽기가 쓰기이기도 하다)**: `search()`는 서빙된 `(item_id, memory_type,
score)`를 각 organizer의 `on_retrieval`에 넘기고, 반환된 op를 **호출자 스레드에서 동기 적용한
뒤** 리턴한다. 즉 그 op는 진행 중인 검색이 아니라 *다음* 검색에 반영된다. 두 가지가 따라온다:

- async 모드에서 다른 모든 쓰기는 워커 큐를 거치지만 **이 경로만 큐를 우회한다.** store는
  RLock으로 보호되나 organizer의 in-memory 상태는 아니다 — 워커가 `on_message`에서
  MemoryOS `_heat`를 갱신하는 동안 호출자가 `on_retrieval`에서 같은 dict를 건드릴 수 있다.
  손실되는 것은 heat 카운터 갱신 한 건 수준이고 자료구조는 깨지지 않는다.
- 그래서 `on_retrieval`은 **싸야 한다**(LLM 호출 금지 — base docstring의 계약).

- `MemoryBundle.render(budget_tokens)` — 타입별 우선순위/토큰 예산으로 프롬프트 주입용 텍스트 생성 (Zep의 context block, ReasoningBank의 system prompt 주입 형식 지원).

#### 서빙 가능성 판정은 한 곳에서만 — `retrieval/steps.py::is_servable`

id로 아이템을 끌어오는 read 경로는 **전부** 두 가지를 걸러야 한다: DELETE가 남긴 톰스톤
(`{"deleted": True}`)과, bi-temporal 타입이 아닌데 INVALIDATE된 아이템. 이 판정이 스텝마다
복제돼 있었고 그래서 갈라졌다 — `_hydrate`/`ExpandExperiences`/`GraphRecall`은 각자 `deleted`만
검사했고 `LinkExpansion`은 **둘 다 없었다**. 결과:

- `ChainedConsumer`가 은퇴시킨 A-Mem 노트(`notes`에 INVALIDATE, §3.3의 문서화된 조합)가
  인바운드 링크를 타고 원문 그대로 되살아났다.
- DELETE된 노트는 빈 ghost hit(`content=""`)으로 서빙됐다 — `_apply_one`과 `_hydrate`가 이미
  고쳤던 round-5 X1과 같은 계열인데 이 스텝만 그 수정에서 빠져 있었다.
- 그리고 dense 경로는 INVALIDATE 시 벡터를 지우므로 멀쩡했지만 **lexical 채널은 id로 다시
  끌어오므로** `[retrieval] lexical_types`에 파생 타입을 넣는 순간 같은 아이템이 되살아났다.

`facts`는 예외로 남는다 — `BITEMPORAL_TYPES`(이제 `core/types.py` 소유: write 쪽 `_apply_one`과
read 쪽 `is_servable`이 같은 목록을 봐야 하고 retrieval은 파사드를 import할 수 없다)에 속하므로
무효화 후에도 validity 구간과 함께 계속 렌더된다. 그래서 이 판정은 단순 `deleted` 검사가 될 수 없다.

## 3. 방법론 → 공통 추상화 매핑 검증

### 3.1 훅 구현 매트릭스

`Organizer`의 훅은 전부 base에 no-op 디폴트가 있으므로, ✓는 **서브클래스가 실제로 오버라이드한
것**을 뜻한다(`base.overrides()`가 판정하는 것과 같은 의미).

| 방법론 | on_message | on_task_end | on_retrieval | on_feedback | consolidate | flush_buffer | retire / patch_unit | warm_start |
|---|---|---|---|---|---|---|---|---|
| passthrough | ✓ (no-op 반환) | — | — | — | — | — | — | — |
| A-Mem | ✓ | — | — | — | — | — | — | — |
| Nemori | ✓ | — | — | — | ✓ | ✓ | — | ✓ |
| MemoryOS | ✓ | — | ✓ | — | — | ✓ | ✓ | ✓ |
| Zep-graph | ✓ | — | — | — | — | — | — | — |
| ACE | — | ✓ | — | ✓ | — | — | — | — |
| ReasoningBank | — | ✓ | — | — | — | — | — | — |
| G-Memory | — | ✓ | ✓ | ✓ | — | — | — | — |

읽는 법 세 가지:

- **`on_message` 계열 5종과 `on_task_end` 계열 3종은 입력 단위가 다르다** — 대화 벤치
  (LoCoMo/LongMemEval)는 앞의 5종만 측정하고 뒤의 3종은 아무 것도 생산하지 않는다. Phase 5
  비교표를 하나의 벤치로 그릴 수 없는 이유이고, admission policy가 뒤의 3종에 적용 불가인
  이유이기도 하다(§1.2, taxonomy §2.5).
- **`consolidate`를 구현한 방법론은 Nemori 하나뿐이다.** 여러 논문이 "consolidation"이라 부르는
  것은 대부분 `on_message` 안의 인라인 승격/병합이다 — 용어가 최소 3가지를 가리키므로
  taxonomy §2.4.1의 **언제**(inline/deferred) × **무엇을**(요약/통합/승격/무효화) 두 축으로
  갈라서 봐야 한다.
- **`on_memory_event`/`consumes`를 native로 구현한 방법론은 없다.** 체이닝은 전부
  `experimental.ChainedConsumer` 어댑터가 담당한다(§3.3).

### 3.2 MemoryOp / store 요구

| 방법론 | 사용하는 MemoryOp | 쓰는 memory type | 필요 store |
|---|---|---|---|
| passthrough | — (raw episode는 파사드가 기록) | — | doc+vec |
| A-Mem | ADD(note), LINK, UPDATE(이웃) | `notes` | doc+vec |
| Nemori | ADD(episode/semantic), MERGE(episode/semantic)+INVALIDATE(supersedes) | `episodes`, `semantic` | doc+vec |
| MemoryOS | ADD(page/segment), MERGE, ADD(profile fact), DELETE(LFU) | `pages`, `semantic` | doc+vec |
| Zep-graph | ADD(entity/fact), MERGE(dedup), INVALIDATE(모순) | `entities`, `facts` | doc+vec+graph |
| ACE | ADD(bullet), UPDATE(카운터), MERGE(dedup) | `playbook` | doc+vec(dedup용) |
| ReasoningBank | ADD(strategy), ADD(experience) | `strategies`, `experiences` | doc+vec |
| G-Memory | ADD(traj/insight), UPDATE(reward), DELETE(prune) | `strategies` | doc+vec+graph |
| MemMachine | ADD(derivative) — LLM 콜 없음 | `derivatives` | doc+vec |

→ 9개 organizer 모두 `(훅 × MemoryOp 7종 × store 3종)` 안에 들어감. 추상화 누수 없음을 Phase 1에서
A-Mem/ReasoningBank로 먼저 검증.

**memory type은 방법론 전용이 아니다**: `semantic`을 Nemori와 MemoryOS가, `strategies`를
ReasoningBank와 G-Memory가 공유한다. 타입만으로 소유자를 알 수 없으므로 파사드가 ADD 적용 시
op의 `actor`를 아이템에 함께 기록하고(`memory.py::_apply_one`), 소유권이 필요한 곳은 그것으로
판정한다 — Nemori의 semantic 통합기가 후보를 자기 것으로 한정하는 것
(`nemori/stages.py::own_items`)과 피드백이 생산자에게 팬아웃되는 것(§3.4)이 그 두 자리다.
`actor` 필드가 없는 과거 아이템은 자기 것으로 취급되므로 기존 스토어의 해석은 불변이다.

### 3.3 체이닝은 어댑터가 담당한다 (native 구독 아님)

구 `input="episodes"` 생성자 모드는 **제거**됐다. A-Mem/MemoryOS는 이제 `consumes=()`이고
`on_memory_event`를 구현하지 않는다 — 대신 `experimental.ChainedConsumer`가 wrapped organizer를
감싸 `consumes=(source_type,)`를 선언하고, 상류 이벤트를 `Episode`로 평탄화해 wrapped의 평범한
`on_message`에 먹인다. 방출 측(Nemori)은 그대로다: episode merge 시 MERGE(신규 병합)+
INVALIDATE(흡수된 구 episode)를 같은 배치로 반환하고 MERGE op의 `payload["supersedes"]`에 흡수
id를 명시한다.

    ChainedConsumer(AMemOrganizer(), "semantic")   # Nemori v4 Table 7
    ChainedConsumer(MemoryOSOrganizer(), "episodes")

이 배치의 이유는 §1.2와 같다 — **합성 어댑터는 프레임워크이지 방법론이 아니고**, 게다가 이 조합은
논문 재현이 아니므로 `experimental/`에 격리된다. wrapped가 `retire`/`patch_unit`을 오버라이드하면
자기 은퇴 정책을 쓰고(MemoryOS), 아니면 어댑터의 일반 1:1 INVALIDATE로 떨어진다.

### 3.4 피드백은 생산자 소유다

`report_feedback`은 `target_type`으로 분기하지 않고 각 organizer의 `on_feedback`으로 팬아웃한다.
타입 분기는 공유 타입에서 곧바로 오염됐다 — ReasoningBank 아이템(논문상 append-only, 피드백 루프
자체가 없음)이 `strategies`를 공유한다는 이유로 G-Memory의 +1/−2를 받았고, 반대로 G-Memory의
served-insight 게이트(round-5 W-4)는 호출자 없는 `backward()` 안에만 있어 실경로에서 빠져 있었다.
팬아웃 이후 "이 규칙은 누구 것인가"가 "어느 organizer가 활성인가"와 같은 질문이 된다. 대가는
명시적이다: **소유 organizer가 활성이 아니면 피드백은 0을 반환하는 no-op**이다.

### 3.5 op 의미와 위상 경계 — "구현이 지키던 것"을 구조로 옮긴 두 건

파사드 감사에서 나온 두 건은 **당시 실경로가 없었다.** 그래서 넘길 뻔했는데, 둘 다 안전한
이유가 "우연히 아무도 그렇게 안 해서"였다 — 즉 다음 방법론이 재발시킬 수 있는 상태였다.

- **`UPDATE`는 더 이상 upsert가 아니다.** 없는 id에 UPDATE하면 내용도 provenance도 없는 파편이
  생기고 그게 검색에 서빙됐다. upsert가 필요했던 유일한 자리는 `base.cursor_op`(첫 전진 시 커서
  행이 없음)인데, 커서의 상태는 `seq` 전부라 **전체 치환이 곧 올바른 의미**다 — 그래서 `cursor_op`은
  `ADD`가 됐고 UPDATE는 "대상이 있어야 한다"로 좁혀졌다(없으면 warning 후 미적용, op는 로그에
  남으므로 이력 손실 없음). **`MERGE`는 upsert를 유지한다** — 병합 결과는 새 id에 쓰이므로
  (Nemori의 MERGE(신규)+INVALIDATE(흡수)) 대상 부재가 정상 경로다.
- **`on_retrieval` 반환 op는 전파하지 않는다.** `base.py`의 "must be cheap: no LLM calls here"는
  훅 본인에게만 걸리는 계약이었다 — 반환 op가 MemoryEvent가 되면 구독자의 `on_memory_event`가
  임의 작업(`ChainedConsumer`는 wrapped의 `on_message` = LLM)을 읽기 경로에서 돌린다. `memoryos`/
  `gmemory`가 둘 다 `[]`를 반환해서 발화하지 않았을 뿐이므로, 읽기 경로의 비용 상한을 구조로
  옮겼다(`_apply_from_all(propagate=False)`). op 자체는 그대로 적용된다.

  **"관측 불가"가 손실이 아닌 이유**: 구독자를 구현한 건 `ChainedConsumer` 하나뿐인데
  (`gated.py`는 위임), 그 핸들러는 `payload["content"]`를 읽어 wrapped 유닛을 덮어쓴다. read-path
  op에는 `content`가 없으므로(heat 카운터는 `{"heat": 3}`) 전파하면 누적 중이던 내용이 `""`로
  **파괴된다** — 실측: `{'pg1': 'user booked a flight to Paris'}` → `{'pg1': ''}`. 즉 현재 구독자는
  read-path op를 의미 있게 소비할 수 없다. 그래도 남는 위험은 드롭이 **조용하다**는 것이므로,
  read-path op의 `target_type`이 실제로 누군가의 `consumes`에 있으면 warning을 남긴다
  (`memory.py::_warn_if_subscribed`) — UPDATE를 좁힐 때와 같은 원칙이다. 어떤 방법론이 진짜로
  전파를 필요로 하면 write-path 팬아웃에서 물려받는 게 아니라 명시적 결정이어야 한다.

기록만 한 것: **`warm_start`는 프로덕션 호출자가 없다.** 큐 드레인은 `consolidate`와 맞추려고
넣었지만 이 훅 전체가 테스트에서만 불린다 — 결함이 아니라 배선 상태다.

## 4. 프로세스 토폴로지

```
[LLM 데몬]  vLLM/llama.cpp (Qwen3-0.6B 등, OpenAI-compatible :8000)  ← 상주
[MCP 서버]  agmem.mcp (stdio 또는 :8765)      ─┐
[Python API] import agmem                      ├─ 같은 memory.db/vectors 공유
[벤치 하네스] scripts/exp_locomo_conv0.py 등   ─┘
[워커]      memory.py 내장 백그라운드 스레드 (sync_write=False 시)
```

- lite에서는 전부 단일 머신·단일 DB 파일. full에서는 store들이 서버형으로 바뀔 뿐 토폴로지 동일.
