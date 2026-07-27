# 구현 충실도 감사 (2026-07-16, 자기 감사)

> **2026-07-27 11차 — 8~10차가 매번 우연히 걸린 두 결함 유형의 전수 조사**
> (상세: `docs/research/fidelity-round11-defect-class-sweep.md`)
>
> 8·9·10차가 각각 직전 라운드의 주장에서 오류를 찾았는데 **세 번 모두 같은 두 자리**였다.
> (a) 업스트림이 여러 벌인데 한 벌만 보고 내린 판정, (b) 자기가 머리에 얹힌 코드와 어긋나는
> 문서. 새 organizer를 고르는 대신 유형을 정의하고 **전 organizer docstring의 검증 가능한
> 문장을 코드와 1:1 대조**했다.
>
> - **[S1] MemoryOS는 두 벌이 아니라 세 벌이고, 10차 판정도 부분이었다.**
>   `memoryos-chromadb`에서도 검색 keyword 항이 **살아 있고**, 질의 키워드를 **write용
>   multi-summary 프롬프트를 질의에 적용**해 뽑으며(개수 상한 없음), 겹침을 **Jaccard**로
>   센다 — eval의 전용 프롬프트(≤3개) + containment mean과 다르다. 10차가 넣은 `_relevance`는
>   containment mean을 **하드코딩**하고 있었으므로, 8차가 pypi만 보고 일반화한 것과 같은
>   실수를 한 단계 아래에서 반복할 뻔했다. → `MemoryOSPageRecall(keyword_similarity=)` +
>   `AgmemConfig.page_recall_keyword_similarity`. **한 벌은 read와 merge에 같은 공식을 쓰므로**
>   organizer의 `_keyword_overlap`과 일치해야 하고, 두 구현이 다른 모듈에 있어 회귀 테스트로
>   못박았다. **남은 추적**: `chromadb` 벌 전체의 프리셋화(확인된 차이는 위 두 칸 +
>   τ=24h·queue 7로 우리와 동일) — "미세 항목"이라 부르지 않는다. 10차가 그 표현으로 넘어간
>   자리에서 LLM 콜 하나가 나왔다.
> - **[S2] G-Memory의 "clean-room reimplementation" 주장은 사실이 아니었다.** 같은 docstring이
>   두 줄 위에서 *"Score semantics follow **the official code**"*라고 적고, round-5 보고서는
>   `bingreeky/GMemory` clone(commit 7b581c5)을 1차 기준으로 삼았다고 명시한다. 업스트림에
>   **라이선스가 없는 것은 사실**(2026-07-27 재확인)이므로 provenance 문장은 더 정확해야 한다.
>   → 사실대로 재작성(clean-room 아님, 공식 코드를 1차 참조, 재현 대상은 소스가 아니라 동작·상수,
>   업스트림 무라이선스).
> - **[S3] A-Mem read 경로 포인터가 낡아 있었다** — "retrieval/pipeline.py에 구현" → 실제로는
>   `retrieval/steps.py`의 `LinkExpansion`. 9차의 `harness.py`와 같은 계열이고, 같은 리팩터가
>   `--expand-links`를 한동안 무효 토글로 만들었다(6차 A1). → 이력까지 포함해 정정.
> - **[S4] Zep**: 모듈 docstring의 "flush_buffer가 full refresh"에 조건(`community_refresh` +
>   그래프 변경) 누락 — 명시.
> - **이상 없음(전수 대조)**: A-Mem 버그픽스 4건 + 빈 `actions` 폴백, Nemori 편차 6건
>   (`buffer_max`에서 최신 메시지 포함 flush 포함), ACE 편차 3건, Zep 나머지 서술.
>   **class-(b)는 전 organizer 소진.**
>
> **2026-07-27 10차 — "추적만/미세/훅만 존재" 3분류의 재검증**
> (상세: `docs/research/fidelity-round10-tracked-gaps.md`; 업스트림 당일 raw:
> `google-research/reasoning-bank:WebArena/{induce_memory,induce_scaling}.py`·
> `prompts/memory_instruction.py`, MemoryOS `eval/{mid_term_memory,retrieval_and_answer,
> utils}.py`, graphiti 트리, 논문 arXiv:2509.25140 §3.3 · arXiv:2501.13956 §2.2·§6.1)
>
> - **[R1] ReasoningBank "MaTTS — 훅만 존재"는 사실이 아니었다. 훅이 없었다.**
>   `matts|contrast|scaling` 검색 결과가 패키지 전체에서 `__init__.py` docstring 한 줄뿐이고,
>   `on_task_end`는 궤적을 하나만 받으므로 진입점 자체가 없었다. 논문 §3.3과 공식 코드는 경계를
>   명확히 긋는다 — 업스트림은 단일 궤적 유도(`induce_memory.py`)와 **스케일링 유도**
>   (`induce_scaling.py` + `PARALLEL_SI`)를 별도 모듈로 두고, 후자는 한 질의의 여러 궤적을 한
>   프롬프트에 넣어 self-contrast로 메모리를 뽑는다(= 메모리 계층의 일). 상수 셋이 단일 경로와
>   다르다: **아이템 5개**(3 아님), content **1-5문장**(1-3 아님), **t=0.7**(1.0 아님).
>   → `Organizer.on_scaled_task_end` 훅 신설 + RB 구현 + `AgenticMemory.add_scaled_task_result`.
>   궤적 1개는 단일 경로로 폴백. **업스트림 결함**: `induce_scaling.main`이 궤적별 정답 라벨을
>   계산하고도 프롬프트에 넣지 않는다(`format_examples`가 호출되지 않음) — 실제 신호는 라벨이
>   아니라 **섞임 그 자체**이므로 우리 훅도 outcome 인자를 두지 않았고, 아이템은
>   `outcome="contrast"`로 적재한다. **등급 ● → ◑⁺** (논문 제목이 내건 주장의 절반이 비어
>   있었는데 ●였다).
> - **[R2] MaTTS sequential은 계약 밖이 맞다.** `SEQUENTIAL_PROMPT`는 **에이전트에게** 궤적을
>   다시 쓰라고 지시한다("Output must stay in the same `<think>...<action>` format") — ACE
>   multi-round(8차 §A3)와 같은 선. 미구현 확정, 근거 첨부.
> - **[M8] MemoryOS "eval 계보 검색 상수 — 미세 항목"은 미세하지 않았다.** 8차 §M7의
>   "검색 keyword 항은 사문"은 **pypi 한 벌만 보고 내린 결론**이다. eval 계보는
>   `query_keywords = llm_extract_keywords(query, client)` — **질문당 LLM 콜 1회**(최대 3개)를
>   쓰고 그 containment mean을 `alpha=1.0`으로 세그먼트 점수에 더한다. 논문 수치를 낸 계보의
>   read 경로가 그만큼 비어 있었다. → `ReadContext.query_keywords` + `_relevance` 신설(키워드가
>   비면 순수 cosine이라 한 스텝이 두 계보를 담는다). 추출은 검색 계층에 LLM이 없으므로 벤치에서
>   한다(A-Mem keyword query와 같은 이유). 검색 시점 recency `lambda_t`는 **두 계보 모두 죽어
>   있다**(eval이 `lambda_t = 1`, decay 줄 주석) — 재현하되 고치지 않는다.
> - **[M9] 1단계 segment 게이트가 통째로 빠져 있었다.** 두 계보 모두
>   `if session_relevance_score >= segment_similarity_threshold:` 안에서만 page를 채점하는데,
>   우리는 융합 랭킹으로 들어온 세그먼트를 전부 펼쳤다. → `page_recall_segment_threshold`(0.1)
>   신설, 점수는 융합 랭크가 아니라 **세그먼트 자신의 summary 벡터**로 재계산(업스트림이 매칭하는
>   대상이 그것이다).
> - **계보 knob 통합**: 9차가 만든 `assistant_knowledge_mode`에 M8이 두 번째 read knob을 더하게
>   되어, 독립 knob이 다시 "어느 upstream에도 없는 조합"을 만들 수 있었다 →
>   **`memoryos_lineage="pypi"|"eval"` 하나로 통합**.
> - **[Z4] Zep "추적만" 분류는 옳았다.** 논문은 **v1이 유일본(2025-01-20)**이고 SagaNode·entity
>   attribute ontology가 없으며, §2.2는 추출을 *"separate stages"*로 못박는다 — 업스트림의
>   `summarize_sagas.py`·`attribute_utils.py`·`combined_extraction.py`는 전부 논문 이후 추가물
>   (마지막 것은 논문과 반대 방향의 최적화). 우리 분리 추출이 논문 쪽이다. 추적 유지.
>
> **2026-07-27 9차 — 8차의 "측정만 남았다" 재검증: 계보 혼합 2건 + 낡은 계약 3건**
> (상세: `docs/research/fidelity-round9-lineage-mixing.md`; 업스트림 당일 raw 재대조:
> MemoryOS `{short_term,updater,memoryos,mid_term,retriever,utils}.py`·
> `eval/{main_loco_parse,retrieval_and_answer}.py`, graphiti `extract_edges.py`·
> `search/{search,search_config_recipes}.py`, ACE `ace.py`·`playbook_utils.py`·`curator.py`)
>
> **8차의 업스트림 대조 주장 15건은 전수 CONFIRMED, REFUTED 0.** 문제는 대조가 아니라 **그
> 변경이 기존 config·전역 상수와 만나는 접합부**였고, 거기에 5건이 있었다.
>
> - **[L1] 재측정용 config가 순수하지 않았다.** `memoryos`/`memoryos_eval`이
>   `("episodic", "pages", "semantic")` — 업스트림 QA 컨텍스트에는 **원문 검색 채널이 없다**
>   (STM history + 검색된 MTM page + profile/knowledge뿐). 2차 재감사가 `amem`/`nemori`를
>   같은 이유로 순수화하고 `*_mixed`로 분리했는데 MemoryOS만 그 처리를 못 받았다. 게다가 8차가
>   악화시켰다 — M7(원문 page 서빙) + M3(원문 STM 주입)로 **원문 경로가 셋**이 됐다(업스트림은
>   둘). → `("pages", "semantic")` + `lexical_types=()`로 순수화, 구 배선은 **`memoryos_mixed`**
>   신설로 보존.
> - **[L2] pypi 계보 config가 eval 계보의 read 경로로 읽고 있었다.** `MEMORYOS_PRESETS`의
>   "never mixed within a preset" 규칙을 **프리셋 밖의 두 값**이 깨고 있었다. (a)
>   assistant-knowledge **전량 주입이 무조건** 실행돼 pypi 런이 top-20 검색과 전량 덤프를 동시에
>   받았다 → `answer(assistant_knowledge_mode=)` 기본 `"retrieved"`. (b) `page_recall_cap=10`은
>   **eval 드라이버 값**이고 pypi는 7이다(`memoryos.py:38`; eval은
>   `RetrievalAndAnswer(queue_capacity=10)`) → 기본 7, `memoryos_eval`이 10을 명시. 둘 다 결과
>   JSON에 스탬프.
> - **[L3] 8차 산출물의 docstring 2건이 코드와 정면으로 어긋났다** — 8차가 ACE에 대해 지적한
>   "행이 낡아 있었다"와 같은 결함. `memoryos/organizer.py` 모듈 docstring은 M1~M5를 **여전히
>   미구현으로** 적고 있었고("still open", "we follow pypi in both presets"), 후자는 같은 파일
>   `MEMORYOS_PRESETS["eval"]["profile_update"] = "two_call"`이 반박한다. `bench/harness.py`는
>   "LongMemEval not yet implemented"였다(273207d 이후 낡음). → 둘 다 재작성.
> - **[L4] `recent_context()` 훅이 벤치마크 한쪽에만 배선돼 있었다.** `longmemeval.answer`가
>   호출하지 않아, MemoryOS를 LongMemEval에서 재면 M3가 복원한 STM 채널이 **degradation
>   스탬프도 없이** 사라졌다. → 검색 경로 한정으로 배선(full-context 베이스라인 제외).
> - **[L5] 첫 페이지 continuity 콜**: 업스트림은 빈 이전 페이지를 상대로 묻고 답을 버린다.
>   우리는 건너뛴다 — 동작 동일, 대화당 1콜 차이. **미문서 편차였던 것을 문서화**(코드 무변경).
>
> **⚠️ `results/locomo-conv0-memoryos.json`은 세 세대 낡았다**(6차 → 8차 → 9차). 재현은
> `memoryos_mixed`; 새 `memoryos`/`memoryos_eval`은 그 파일의 비교 대상이 아니라 **대체 대상**.
>
> **2026-07-27 8차 — Zep·MemoryOS 잔여 항목 전량 구현**
> (상세: `docs/research/fidelity-round8-remaining-pieces.md`; 업스트림 당일 raw 재확인:
> graphiti `extract_edges.py`, MemoryOS `memoryos-pypi/{short_term,updater,retriever,
> memoryos,prompts,utils}.py`·`eval/main_loco_parse.py`, 두 논문 §3.1–3.3·§6.1.3)
>
> **Zep**: predicate를 SCREAMING_SNAKE_CASE로(논문 §6.1.3 + upstream `relation_type`;
> 프롬프트만으로는 0.6B가 무시하므로 `_relation_type`으로 정규화 — predicate는 엣지 정체성의
> 일부라 `lives_in`/`LIVES_IN`이 두 관계 타입이 된다) / hyper-edge 가이드라인 전사 +
> **subject==object 가드**(§6.1.3 "two DISTINCT nodes"; `edges_between(x,x)`가 자기 엣지를 두 번
> 반환해 다음 메시지에서 자기 중복 후보가 되는 위험도 함께) / **"recent episodes as seeds"**
> BFS origin 변종(`search(bfs_origin_ids=...)` + `recent_episode_entity_ids` — upstream처럼
> 명시 origin이 파생 origin을 **대체**).
>
> **MemoryOS**: **[M1] STM이 1-page FIFO 롤링 윈도우다.** `is_full()`이 `>=`이므로 upstream의
> `while is_full(): pop_oldest()`는 **정확히 1페이지**만 방출하고 STM은 `capacity-1` 페이지를
> 상시 유지한다(논문 §3.3도 "the oldest dialogue page … FIFO"). 우리는 전량 배치 flush였고,
> 그 하나로 **TOPIC 콜 빈도·세그먼트 경계·QA 시점 STM 잔존**이 동시에 틀어졌다.
> **[M2]** dialogue chain(페이지당 continuity + meta_info 2콜, "Conversation chain overview:"로
> 주입) 구현 — meta_info 부착 지점만 세그먼트로 이동(업스트림이 체인 전체를 최신 요약으로
> 덮으므로 값은 동일). **[M3]** `Organizer.recent_context()` 훅 신설로 STM을 QA에 주입
> (round-5 N2 해소). **[M4]** assistant-knowledge 전량 주입(eval 계보). **[M5]** eval 계보의
> 2콜 프로필 갱신(`profile_update` 프리셋 키). **[M6]** 역할극 persona 프롬프트 —
> **기본 off**(방법론별 답변 프롬프트는 config 간 점수 귀속을 깬다).
> **[M7] MTM 검색이 2단계다.** 업스트림은 세션을 summary 임베딩으로 매칭한 뒤 **그 안의
> page를 채점해 전역 top-`retrieval_queue_capacity` page**를 주입하며, **세션 summary는
> 프롬프트에 넣지 않는다**(매칭 키일 뿐). 우리는 요약을 그대로 주입하고 있었다 — round-5 §3이
> "업스트림에 없는 채널"로 지목한 그 지점. `MemoryOSPageRecall` 스텝 신설(세그먼트가
> `page_units`를 payload에 싣도록 함께 변경), heat 되먹임은 서빙된 page id를 `_unit_pages`로
> 세그먼트로 되돌려 유지 — 안 그러면 round-5 N1이 복원한 루프가 2단계 도입과 함께 죽는다.
> 부수 확인: 업스트림 `search_sessions`의 keyword 항은 `query_keywords = set()`이라 **항상 0**
> (검색에서 keyword는 사문, 병합 F_score에서만 유효).
>
> **ACE**: docs/10의 ACE 행이 **낡아 있었다** — round-5가 지적한 "read 계약(전체 주입)",
> "curator 전체-뷰", intra-batch dedup, 렌더 포맷은 이미 해소돼 있었고 행만 갱신되지 않았다.
> 8차가 추가한 것은 **[A1] token budget 80k + playbook stats + progress**를 curator
> 프롬프트에 넣은 것(전체 playbook을 주입하는 이상 성장 억제의 유일한 수단이 "예산을
> 알려주는 것"이고, 예산으로 **절단하지 않는다** — bullet을 버리는 것이 곧 논문이 말하는
> context collapse다)과 **[A2] environment feedback 2분기**(`outcome`이 success/failure를
> 지칭하면 업스트림 문구로 치환)다.
> **[A3] multi-round reflection은 구현하지 않았고, 그것이 옳다**: 업스트림 루프
> (`ace.py:501-543`)는 reflect → **답변 재생성** → 정답 재확인 → 교정되면 break다. 즉 라운드마다
> trajectory가 **바뀐다**. 우리 `on_task_end`는 완료된 trajectory를 사후에 받으므로 재생성을
> 할 수 없고, 같은 trajectory를 3번 reflect하는 것은 업스트림에 없는 메커니즘을 발명하는 것이다.
> 이 루프는 메모리 계층이 아니라 **생성기**가 소유한다. → 행의 "누락"에서 "계약 밖"으로 재분류.
> **ACE 등급 ◑ → ●⁻, 측정 보류 해제.**
>
> **[U3] upstream 결함**: `process_short_term_to_mid_term`이 `agent_response`가 빈 page를
> pop만 하고 **아무 데도 넣지 않는다**(세션이 speaker A로 끝나면 발생). 원문 손실 금지 원칙상
> 미재현.
>
> **⚠️ docs/09 비용 결론 정정**: "MemoryOS가 배치 설계 덕에 91콜로 가장 저렴"의 그 배치는
> **MemoryOS의 것이 아니라 우리 구현의 것**이었다. 업스트림 단위로는 페이지당 TOPIC 1 +
> continuity 1 + meta_info 1 ≈ 600콜+ 규모이므로 **방법론 자체는 A-Mem/Nemori보다 비싸다.**
> `results/locomo-conv0-memoryos.json`은 6차(B1/C1)에 이어 8차로 **두 세대 낡았다** — 구 배선
> 재현은 `dialogue_chain=False` + `flush_stm_on_drain=True`.
>
> **2026-07-27 7차 — Zep community 구현 + read-path 결함 2건**
> (상세: `docs/research/fidelity-round7-zep-communities.md`; 업스트림 당일 raw 재대조:
> getzep/graphiti `community_operations.py`·`summarize_nodes.py`·`graph_queries.py`·
> `search/{search,search_utils,search_config_recipes,search_helpers}.py`)
>
> - **[R1/R2] entity 채널이 두 방향 모두 죽어 있었다.** organizer의 entity payload에
>   `content`가 없어서 (a) `_DictItem.render()`가 **빈 문자열**을 반환하고 — 검색된 entity가
>   QA 프롬프트에 `- ` 빈 불릿으로 들어갔다 — (b) `sqlite_doc.put_item`이 `content`로
>   FTS 행을 만들므로 **entities BM25 인덱스가 비어 있었다**. round-5 zep §4의 해금 조건
>   2·5번이 코드로는 존재하되 실행되지 않던 상태. 업스트림은 dense=name,
>   BM25=(name, summary), χ={entity_name, summary}로 **채널마다 텍스트가 다르므로**
>   `content="name: summary"` + `embedding_text=name`으로 분리하고, community처럼 셋이
>   한 필드에 안 모이는 경우를 위해 `lexical_text`(= `embedding_text`의 lexical 판)를 신설.
> - **[R3] community subgraph 구현 완료** (round-5 ①, Zep 완성 계획 5번). label propagation
>   전사 + map-reduce 페어 요약 + 동적 확장, **전부 MemoryOp 경유**(6차 B3 계약 준수 —
>   passthrough 재생 테스트로 단언). graph store 3벌(sqlite/kuzu/neo4j) 모두 지원.
> - **[U1] 업스트림 label propagation은 특정 입력에서 무한 루프다.** synchronous 갱신이라
>   **엣지 2개로 연결된 2-노드 컴포넌트**(대화 데이터에서 흔함)에서 두 노드가 매 라운드
>   label을 맞바꾼다. upstream `while True`에는 상한이 없다. 2-cycle 감지로 정지하되
>   병합을 발명하지 않는다(사이클 어느 상태에서도 두 label이 다르므로 싱글턴).
> - **[U2] 해소 — 논문 §4.1이 재현 대상을 확정한다.** "BGE-m3 models from BAAI for both
>   **reranking** and embedding tasks" → 실험 수치는 **cross-encoder 레시피**(=BFS 채널을 가진
>   유일한 계열)에서 나왔다. RRF 레시피는 "열등한 변종"이 아니라 **upstream `search()`의
>   기본값**이라는 별개 사실. 논문에 검색함수/reranker ablation은 없으므로
>   "논문 operating point / upstream 기본값 / 논문이 서술만 한 메커니즘" 3부류로만 말할 수 있다.
>   → `ZEP_SEARCH_RECIPES` 6종(`organizers/zep_graph/search.py`)으로 전사, 기본값은
>   `cross_encoder`. Nemori·MemoryOS의 계보 프리셋과 같은 구조이고 읽기 경로만 담으므로
>   **한 번의 ingest를 여러 레시피로 측정**할 수 있다.
> - **[R4] φ_bfs를 랭킹 채널로 승격.** 논문 §3.1은 BFS를 cosine·BM25와 나란한 검색 함수로
>   두는데, `GraphRecall`은 후처리로 최상위 점수의 0.9배를 flat하게 붙이고 있었다 —
>   랭킹이 아니라 상수. `retrieval/bfs.py` 신설(origin은 upstream처럼 같은 타입의 다른 채널
>   결과에서 유도), `graph_expansion_cap` 기본값 0으로 내려 이중 서빙 차단.
> - **[R5] reranker가 후보 수 ≤ k이면 실행되지 않았다.** 게이트가 `len(fused) > type_k`라
>   "버릴 게 없으면" 건너뛰었는데, reranker는 **순서도** 정하고 순서는 절단 후에도
>   `render`의 전역 점수 정렬에 남는다. `len(fused) > 1`로 수정. 측정 런은 전부
>   `NoopReranker`였으므로 **저장 수치 무영향**, reranker를 실제로 쓰는 순간 발현할 잠복 결함.
> - **reranker 2종 신규**(논문 §3.2): `EpisodeMentionsReranker`(mention 빈도 =
>   `len(source_episode_ids)`), `NodeDistanceReranker`(centroid로부터의 그래프 거리 —
>   `search(center_node_id=...)`). 둘 다 capability-free라 `RERANKER_CANDIDATES`에서 MMR 아래.
> - **Zep 등급 ○ → ◑⁺, 측정 금지 해제.** 남은 것은 실행뿐이며, `cross_encoder` 레시피는
>   BGE reranker 모델이 필요하므로 런 스탬프의 `degradations`·`search_recipe`를 확인해야 한다.
>
> **2026-07-27 6차 감사 — round-5 이후 리팩터 회귀 점검**
> (상세: `docs/research/fidelity-round6-post-refactor-regressions.md`; 업스트림 당일 raw 재대조:
> snap-research/locomo `task_eval/evaluation.py`, WujiangXu `utils.py`,
> xiaowu0162/LongMemEval `print_qa_metrics.py`, BAI-LAB/MemoryOS `mid_term.py`·
> `short_term.py`). 라운드 3~5가 논문·공식코드 대조를 이미 끝냈으므로, 이번엔 **그
> 이후 변경(mechanism/policy 분리, read-path 플러그인화, A-MAC, LongMemEval 포팅)이
> 만든 회귀**와 **round-5가 자기 세션에서 구현하고 독립 검증을 못 받은 부분**만 봤다.
>
> **수정 완료 (코드)**
> - **[A1] `--expand-links`가 무효 토글이었음.** `exp_amem_repro.py`가
>   `mem.pipeline.link_expansion_cap`(3b39c7d에서 사라진 속성)에 대입 → on/off 양쪽이
>   `LinkExpansion(cap=5)`로 실행. 실측으로 확인(`hasattr(...) == False`). 이제
>   `AgmemConfig.link_expansion_cap`로 배선. **sha e2e7ebe 이하 산출물은 리팩터 이전이라
>   기존 수치 무영향**, 단 그 어블레이션은 그동안 재실행 불가 상태였다.
> - **[A2] `eval_mode="wujiang"`의 BLEU-1이 혼종이었음.** gold·F1·cat5만 업스트림으로
>   갈고 BLEU는 `ours` 정규화(관사 제거+Porter stem)를 썼다. 업스트림은 BLEU에
>   `nltk.word_tokenize` + `sentence_bleu((1,0,0,0), method1)`을 쓴다. `bleu1_wujiang`
>   신설(실제 nltk 호출, 선택 의존성 `[eval]`), 없으면 **지표 생략**(0.0 보고 금지).
>   기존 `results/repro/*_wujiang_*.json`의 `bleu1`은 혼종값 — docs/14 §9-3b 캐비앗.
> - **[A3] snap-research/locomo에는 BLEU가 아예 없다** (리포 전량 검색 0건; 지표는
>   EM / stemmed token-F1 / 이름과 달리 rouge-1 F를 반환하는 `rougel_score`). `locomo.bleu1`을
>   "공식 채점기 미러"로 적은 docs/14 두 곳을 "우리 지표"로 정정.
> - **[B4] `evaluate(workers>1)`의 "부작용 없음"이 A-Mem 한정.** MemoryOS `on_retrieval`은
>   `_heat`를, G-Memory는 `_served`를 풀 스레드에서 변경한다. `on_retrieval`을 오버라이드한
>   organizer가 활성일 때 경고를 내도록 가드 추가(현재 config는 전부 workers=1이라 잠복).
> - **[B5] LongMemEval `overall` 모집단.** 업스트림 `all_acc`는 알려진 6타입 버킷의 연결이라
>   미지 타입은 들어갈 수 없다(사실 KeyError). 우리도 동일하게 좁힘 — 공식 데이터에선 동일값.
>   (반대로 **abstention 이중 계상은 우리 포팅이 맞다** — 원문 verbatim 확인.)
>
> **수정 완료 (후속 "교정" 지시 — 최초엔 기록만으로 유예했다가 전부 반영)**
> - **[B1] MemoryOS가 page가 아니라 메시지를 세고 있었다.** 업스트림의 단위는 page
>   (=`add_qa_pair` 한 교환)이고 **`L_interaction`과 `stm_capacity` 둘 다** 그 단위다
>   (`short_term.py`는 pair deque, `is_full()`은 `>= max_capacity`). 메시지로 세면 heat ~2배
>   → τ=5 LPM 승격이 절반 분량에서 발동하고, STM 배치도 절반 크기였다. 유예 사유("짝짓기
>   규칙을 발명해야 한다")는 **틀렸다** — 업스트림 LoCoMo 드라이버
>   (`eval/main_loco_parse.py`)에 규칙이 이미 있고(첫 화자가 페이지를 열고 다른 화자가 붙음),
>   우리 ingest는 이미 `meta["speaker"]`를 싣고 있었다. `_pages`로 전사, 세는 지점 5곳을
>   페이지로 통일, 스키마 `message_indexes`→`page_indexes`.
>   → **`results/locomo-conv0-memoryos.json`은 구 배선 산출물, 재측정 대상.**
> - **[B2] RRF 점수를 타입 내에서 만들고 타입 간에 비교했다.** `episodic`만 dense+BM25
>   2채널이라 최대 2배 스코어 → `MemoryBundle.render`의 전역 정렬·예산 컷에서 raw가 파생
>   메모리를 통째로 밀어냄(실측: episodic 0.0328 vs notes 0.0164). `rrf_fuse`가 채널 수로
>   나누도록 수정. 유예 사유("mixed 랭킹 불연속")도 **과대평가** — 공통 상수 나눗셈은 단조
>   재스케일이라 순수 config는 비트 동일이고, 바뀔 수 있는 유일한 지점인 예산 컷은 측정된
>   12,314질문 전부에서 걸리지 않았다. 즉 보호 대상이던 비교가 실재하지 않았다.
> - **[B3] Zep이 유일하게 스토어에 직접 썼다.** `upsert_node`/`upsert_edge`/
>   `invalidate_edge`가 evolution log를 우회 → 로그 리플레이로 그래프 복원 불가(복원 시
>   GraphRecall이 조용히 벡터 RAG로 퇴화), 예외 시 doc/graph 불일치. op 어휘는 유지한 채
>   payload에 `subject_id`/`object_id`·`entity_type`을 싣고 `AgenticMemory._apply_graph`가
>   적용하도록 이관. 회귀 테스트가 **passthrough 메모리에 op만 재생해 그래프가 복원됨**을
>   단언한다. → Zep 완성 계획 6번 해결.
>
> **재검증 후 이상 없음**: A-Mem write 파이프라인(게이팅·ID 기반 이웃·2단 tags op), Nemori v4
> 스테이지(K_e=5/K_m=5/τ=0.70, 전 실패경로 무손실 fallback), A-MAC 릴리스 결함 4건 처리,
> locomo `normalize_answer` 정규식·nltk PorterStemmer 대응, LongMemEval judge 5분기·모델 핀.

> **2026-07-21 머지 전 충실도 리뷰** (refactor/organizer-experimental-split 대상):
> 병렬 독립 에이전트 3종이 nemori/memoryos/amem을 **논문(arXiv)+공식 레포 실시간
> 대조+인용 이슈 원문**으로 재검증, 우리 코드에 닿는 주장은 전부 재확인. 기준선
> **125 passed/1 skipped 불변**(리팩터는 동작 보존). 판정 요약:
>
> - **A-Mem ● 확정.** docstring이 근거로 든 버그픽스 이슈(#32 인덱스→ID, #23 score
>   반전, #24 L2 vs cosine)를 **이슈 원문과 대조해 전부 CONFIRMED**. eq.(3) 메타데이터
>   임베딩은 공식 코드(content-only)보다 논문에 더 충실. **[A1 수정]** `_ingest`의
>   neighbor 업데이트에서 doc_store 미반환 히트가 게이트를 통과해 `by_id` KeyError를
>   낼 수 있던 엣지를 `valid_ids = {실제 반환 노트}` 로 차단.
> - **Nemori v1/v4 ● · upstream ◑.** **[N1 수정]** `ThreeWayIntegrator`는 v4 §3.3.3
>   P_con(new/merge/conflict) **논문 메커니즘**이므로 experimental이 아니라 충실 코어
>   (`organizers/nemori/stages.py`)에 있어야 맞다 — 이전 커밋(2164acb)이 이를 "논문 밖"으로
>   오분류했던 것을 바로잡아 코어로 되돌리고, 진짜 우리 발명인 `SemanticOfflineConsolidator`
>   (유예/오프라인 스케줄링)만 experimental에 남김. v4 프리셋이 더 이상 experimental을
>   import하지 않음. **[N2 해소]** upstream 프리셋의 `merge_time_gap_hours=1.0`(">1h 병합
>   금지")은 **upstream `nemori/llm/prompts.py`의 `MERGE_DECISION_PROMPT`에 실제로 존재**
>   함을 raw 파일 직접 대조로 확인("Do NOT merge if: ... separated by significant time
>   gaps (>1 hour)"). 즉 docs/11(3차 감사) 기록이 옳았고, write-path-survey §1.2의
>   "미해결"과 리뷰 에이전트의 merger.py-only 확인(프롬프트 미확인)은 **모두
>   상위 소스로 정정**. 코드 무변경(테스트 `time_gap_hours == 1.0` 유지).
> - **MemoryOS ◑ 확정(정직한 ◑).** MTM/heat/F_score 층은 ●급 — 상수(cap 10/2000, θ=0.6,
>   τ=5, heat 계수 1/1/1)가 pypi 코어와 일치, Jaccard F_score 정확, 논문 충실 최저-heat
>   축출을 코드의 실제 access-count LFU와 올바르게 구분. **[M2 수정]** "read-path
>   feedback 부재로 N_visit=0" docstring은 낡음 — `on_retrieval`은 배선돼 있고
>   memoryos config가 `pages`를 조회하므로 루프는 **살아 있으며**, LoCoMo conv0가
>   ingest-후-eval 구조라 회수 시점 갱신이 이미 끝난 승격/축출에 되먹임되지 않아
>   *효과만* 무해함을 명확화. **[M1 기록·미구현]** LPM이 append-only semantic facts로,
>   upstream의 단일 진화형 프로필 **문서 교체**(`update_user_profile(merge=False)`) +
>   FIFO(100) 지식 deque를 결여 — 논문 간판 메커니즘의 실제 갭. 이는 동작을 바꾸는
>   **충실도 격상**이라 이 리팩터(동작 보존)에서 구현하지 않고 E2E 실측 후 별건으로
>   추적. (부수: recency τ가 논문 μ≈1e7s가 아닌 코드값 24h — 저영향, 미문서였던 점 명기.)
> - **재현 캐비앗(docs/09에 병기 대상):** A-Mem eq.(3) 임베딩≠발표수치의 content-only,
>   empty-actions→양효과 폴백(소형모델 evolution 과다), read 링크확장 전역캡5 vs
>   upstream per-hit. 어떤 수치도 이 캐비앗 없이 "논문 재현"으로 인용 금지.

> **2026-07-21 experimental 경계 분리** (spec
> `docs/superpowers/specs/2026-07-21-organizer-experimental-split-design.md`): 논문·공식코드에
> 대응물이 없는 **크로스-organizer 합성**을 `organizers/experimental/`로 격리 — 충실
> organizer(amem/memoryos)에서 구 `input="episodes"` chained-manager 코드를 제거하고
> `experimental.ChainedConsumer` 어댑터로 추출(`nemori_amem`/`nemori_memoryos`). Nemori의
> 유예/오프라인 스케줄링(`semantic_offline` SemanticOfflineConsolidator)만
> `experimental/nemori_mixing.py`로 이동 — 그것이 재사용하는 `ThreeWayIntegrator`(v4
> §3.3.3 논문 메커니즘)는 충실 코어(`organizers/nemori/stages.py`)에 유지([N1 수정]). **전 과정 동작
> 보존**(125 passed/1 skipped 불변). 충실 코어 등급 무변경 — 논문 그대로의 로직은 온전.
> A-Mem 논문 재독은 docs/13(스터디 가이드)로 별도 정리. experimental 항목의 격상은 LoCoMo
> E2E 실측 후.

> **2026-07-18 Nemori 라이프사이클 재설계** (스펙:
> `docs/superpowers/specs/2026-07-18-nemori-lifecycle-redesign-design.md`): 아래 표 Nemori 행의
> "episode merging, 배치 세그멘테이션 모드" 누락은 `fidelity="v4"|"upstream"` 스위치
> (`EpisodeMerger`, `BatchPartitioner`, `organizers/nemori/stages.py`)로 해소됨 — 등급/측정 판정은 실측
> 전까지 재산정하지 않음(표는 v1 디폴트 기준 그대로 유지).

> **2026-07-17 round-5 P2/구-P3 일괄 반영** (상세: fidelity-round5-other-organizers.md
> §4): 조회 통지 훅(on_retrieval — MemoryOS N_visit/G-Memory served 캐시), G-Memory
> 점수 의미론+reward 폐루프, Zep write 재구축(3단계 resolution·temporal 통합·dedup·
> n=4 컨텍스트)+hybrid/GraphRecall read, RB 경험 단위 검색+프롬프트 정합. 스토어
> 슬롯 확장: graph=Kuzu(임베디드)/Neo4j(서비스 감지), doc=pgserver 임베디드
> PostgreSQL. Zep은 여전히 측정 금지(community 부재·재측정 전 검증 필요)이나
> "골격(○)" 근거였던 write-path 갭 대부분 해소 — 다음 감사에서 재등급 대상.

> **2026-07-17 5차 검증 (docs/research/fidelity-round5-other-organizers.md)**: 나머지
> organizer 5종(RB/MemoryOS/ACE/Zep/G-Memory)을 논문+공식 코드 당일 클론으로 대조.
> 등급은 전부 유지되나 **아래 표의 누락 칸은 과소 기재로 판명** — 교정 목록은 round-5
> 문서 §2·§4 참조. 특히: MemoryOS "LFU"는 "최저-heat 축출(논문 준수/코드 비준수)"로
> 정정; ACE는 read 계약(playbook 전체 주입) 미고정; 파이프라인 공통 결함 X1(DELETE
> 유령 벡터)·X2(INVALIDATE 검색 노출)·X3(strategies description 소실) 발견.
> 배선 자체는 실엔진 4종 매트릭스(40 tests × sqlite-vec/LanceDB/Qdrant/Chroma) 통과.

> **2026-07-17 4차 검증 (docs/research/fidelity-round4-verification.md)**: round-3
> 수정 커밋(e7c5f8f)을 업스트림 당일 재다운로드 소스로 독립 재대조 — 24개 클레임
> 전수 CONFIRMED, REFUTED 없음. 신규 발견(업스트림 write 온도 0.7, 콜드스타트
> 프롬프트 규칙차, 링크 캡 per-hit 의미차 등)과 문서 교정을 일괄 반영. 판정:
> **A-Mem ●⁻ / Nemori ●⁻ (v1+리포 eval 기준; v4 통합 모듈 2개는 계획된 미구현)**.
> 온도·콜드스타트 프레임은 upstream 충실로 확정, 측정은 API 전환 결정 전까지 보류.

> **2026-07-16 2차 재감사 (P0/P1 수정 커밋 70ba537 이후, upstream 당일 소스 재대조)**:
> 병렬 감사 2건으로 A-Mem·Nemori를 재검증. 판정:
>
> - **A-Mem ◑ → ◑⁺ (코드 ●)**: P0-1(1-hop 링크 확장) P0-3(예산 6000) P1-5(strengthen
>   `new_note_tags`) 정확 반영 확인. 잔여: [높음] raw episodic 채널 혼입(→ `amem` config를
>   notes-only로 순수화, 구 설정은 `amem_mixed`로 분리), [중간] LLM 키워드 질의 생성 부재
>   (→ `keyword_queries` 옵션으로 구현, `amem` config 기본 on), [낮음] 렌더 keywords 미노출.
> - **Nemori ◑ → ◑⁺**: P0-2(episodic 10/semantic 2k=20) r=2 원문 첨부·cold-start·30분
>   갭·timestamp 모두 정확 반영 확인. 잔여: [높음] raw episodic 채널 혼입(→ `nemori` config를
>   episodes+semantic으로 순수화, 구 설정은 `nemori_mixed`), [중간] episode merging 부재
>   (upstream 리포 기본 on — LongMemEval 단계 P2-11 유지), [중간] per-message vs 배치
>   분할 구조 차이(문서화된 의도적 편차, 콜 수 ~N배 캐비앗).
> - 기존 4-way 수치는 혼합(raw RAG 포함) 조건 측정치이므로 "논문 재현"으로 인용 금지 —
>   순수 config 재측정 후 교체할 것.

> **2026-07-16 갱신**: 심층 감사(docs/research/fidelity-deep-audit.md)로 대체됨.
> 재산정: A-Mem ●→◑(read 링크 확장 누락 — P0-1로 수정), Nemori ◑ 측정 보류(검색 설정
> 불일치 — P0-2/3으로 수정), ReasoningBank ●→◑⁺(검색 단위·온도 분리 — P3).
> 아래 표는 1차 자기감사 기록으로 보존.

> 규율: **부분 구현은 벤치마크 대상에서 제외**하거나 결과표에 fidelity 등급을 병기한다.
> 미구현 요소가 그 방법론의 핵심 주장과 연결되면 측정 전 반드시 구현한다.

등급: ●충실(핵심 메커니즘 전부) ◑부분(주변부 누락) ○골격(핵심 일부 누락 — 측정 금지)

| organizer | 등급 | 구현됨 | **누락 (논문 대비)** | 측정 가능? |
|---|---|---|---|---|
| A-Mem | ● | 2콜 write(Ps1/Ps3), metadata-concat 임베딩, top-k링크, 이웃 배치 진화 + 버그수정 | k스윕 실험용 옵션 일부 | ✔ (측정됨) |
| ReasoningBank | ◑⁺ | self-judge, 성공/실패 프롬프트, ≤3 items, k=1 / **10차: MaTTS parallel self-contrast**(`on_scaled_task_end` 훅 + `PARALLEL_SI` 전사 — ≤5 items·1-5문장, upstream `induce_scaling.py`; 궤적별 정답 라벨은 업스트림도 프롬프트에 넣지 않으므로 outcome 인자 없음, 아이템은 `outcome="contrast"`) | **MaTTS sequential은 계약 밖**(`SEQUENTIAL_PROMPT`가 에이전트에게 궤적을 다시 쓰라고 지시 — ACE multi-round와 같은 선, 10차 §R2), upstream 유도 온도(단일 1.0 / scaled 0.7)는 role 설정이라 config 몫 | ✔ (LoCoMo엔 해당無) |
| Nemori | ◑ | boundary(σ=0.7)/서사/시간절대화/predict-calibrate 3단계 | episode merging, 배치 세그멘테이션 모드 | ✔ (측정됨, merging 부재 명기) |
| MemoryOS | ●⁻ | STM/MTM/heat 공식·상수, `L_interaction`·`stm_capacity` 모두 upstream 단위인 **page**(6차 B1) / **계보 프리셋 `fidelity="pypi"\|"eval"`** — 논문 LoCoMo 수치를 낸 eval 하네스(heat 0.8/0.8/1e-4, 저장형 recency, containment-평균 키워드, STM cap=1, LFU)를 재현 가능(6차 C2, eval 결함 5건 명시) / **LPM 3종: 프로필 문서 교체형 갱신 + user·assistant 지식 FIFO(100) + analyzed 마킹 + 최고-heat 세그먼트 선택**(M1 해소) / **8차: STM 1-page FIFO 롤링**(round-5 N2 해소) + **dialogue chain**(페이지당 continuity+meta_info 2콜, 주입 포함) + **STM의 QA 주입**(`recent_context()` 훅) + **assistant-knowledge 전량 주입** + **eval 계보 2콜 프로필 갱신** + persona 프롬프트(기본 off) + **2단계 page 검색**(`MemoryOSPageRecall`: 세그먼트 매칭 → 그 안의 page 채점 → 전역 top-N page 서빙, 요약은 매칭 키로만)  / **9차: read 채널 순수화**(`episodic` 제거 — 업스트림 QA 컨텍스트에 원문 검색 채널이 없다) + **read 경로 계보 분리**(`page_recall_cap` pypi 7 / eval 10, `assistant_knowledge_mode` retrieved/full)  / **10차: eval 계보 read 완성**(질의 keyword 추출 → 세그먼트 점수의 containment 항 — pypi는 이 항이 사문, `memoryos_lineage` 하나로 계보 통합) + **1단계 segment 게이트**(`segment_similarity_threshold`, 세그먼트 summary 벡터로 재채점) | **`memoryos-chromadb` 벌 전체의 프리셋화**(11차 S1 — 세 번째 upstream 벌. 확인된 차이: 질의 키워드를 write용 multi-summary 프롬프트로 뽑고 겹침을 Jaccard로 셈; τ=24h·queue 7은 우리와 동일. 나머지 상수 미대조 — 미세하다고 부르지 않는다), 검색 시점 recency `lambda_t`(세 벌 모두 사문 — 재현), page 임베딩 텍스트 미세차, 업스트림 Retriever의 3채널 동시 실행(무영향) | ✔ **재측정 필요 — 6·8·9·10차로 네 세대 변경, 비용 결론 정정 대상. 구 배선은 `memoryos_mixed`** |
| ACE | ●⁻ | reflect→curate ADD(upstream도 ADD-only 확인), helpful/harmful, dedup 0.90 상시 + intra-batch, **read 계약 = `get_playbook()` 전체 스캔**(round-5 §2·§3.1 해소), **curator 전체-뷰**(§3.2 해소), 렌더 포맷 단일화(§3.6 해소) / **8차: token budget 80k + playbook stats + progress를 curator 프롬프트에**(§3.5), **environment feedback 2분기**(§3.7 — `outcome`이 success/failure를 지칭하면 업스트림 문구로 치환, 아니면 no-GT 변형) | **multi-round reflection은 organizer 계약 밖**(업스트림 루프는 reflect→**재생성**→정답 재확인이라 생성기가 소유; 고정된 trajectory를 3번 reflect하는 것은 업스트림 메커니즘이 아니다 — 8차 §A3), offline 모드(train/val+multi-epoch, 하네스 기능), reflector가 "실사용 bullet"이 아니라 전체 playbook을 봄(§3.3 — 실사용 경로는 `report_feedback`) | ✔ **측정 가능** (read 계약 고정 완료) |
| **Zep-graph** | **●⁻** | round-5 write 재구축(3단계 resolution·temporal 통합 추출·fact dedup·previous-episodes n=4) + 6차 B3(그래프 쓰기 op 로그 경유) + **7차: community subgraph 전량**(label propagation 전사·map-reduce 페어 요약·동적 확장·op 경유 영속화, graph store 3벌) · **entity 채널 복구**(`content` 부재로 렌더·BM25 동시 공백) · **read 경로 전량**: φ_cos/φ_bm25/**φ_bfs 3채널** + reranker 5종 + `ZEP_SEARCH_RECIPES` 6종(기본 = 논문 §4.1 cross-encoder) / **8차: predicate SCREAMING_SNAKE_CASE + hyper-edge 가이드·DISTINCT 가드 + recent-episodes BFS seeding** | entity attributes ontology · SagaNode · combined 단일콜 추출 (모두 논문 이후 업스트림 추가물 — 추적만) | **✔ 측정 가능** — 단 `cross_encoder`는 BGE reranker 모델 필요, 스탬프의 `degradations`/`search_recipe` 확인 필수 |
| G-Memory | ○ | 궤적 sparsify, insight ADD/EDIT/REMOVE+reward(값 일치), projection/backward | **query graph k-hop, FINCH merge, StateChain**, + round-5 추가: 검색 의미론 전체(성공/실패 분리 채널·LLM rating·support-set 투표), 점수 의미론(AGREE/soft REMOVE/score≤0 프루닝/Ω_k), _detect_mistakes, reward 폐루프(read-path 미반영) | ✘ MAS 벤치 자체가 미구축. 논문 §4.3↔공식코드 불일치 — 우리는 코드 계보 |

## 측정 결과(docs/09)의 유효성

- 4-way(passthrough/A-Mem/Nemori/MemoryOS)는 ●/◑ 등급만 포함 — 결론 유지.
  단 MemoryOS 수치에는 "F_score Jaccard 항 부재" 캐비앳을 docs/09에 병기할 것.
- call 수는 방법론 구조가 결정(0.5B 영향은 재시도 ≤1%) — 비용 비교는 유효.

## Zep 완성 계획 (측정 전 필수, 우선순위순)

> 2026-07-27 7차 기준 1~5 전부 완료, 측정 금지 해제. 남은 것은 실행이다.

1. ~~**GraphRecall 배선**~~ → **φ_bfs 채널로 대체 완료**(7차 R4/U2). BFS는 논문 §3.1의 세 검색
   함수 중 하나이므로 후처리 스텝이 아니라 랭킹 채널이어야 한다(`retrieval/bfs.py`).
   레시피별 on/off는 `ZEP_SEARCH_RECIPES`가 결정하고, `GraphRecall`은 기본 비활성
2. ~~resolution 2단계화~~ — 완료(round-5, 임베딩 후보 ≥0.6 → exact-name → LLM 판정)
3. ~~temporal extraction~~ — 완료(round-5, fact 추출 프롬프트에 valid_at/invalid_at 통합 —
   현 upstream과 동일한 형태)
4. ~~fact dedup~~ — 완료(round-5, same-pair 후보 + duplicate/contradiction 단일 콜)
5. ~~community: label propagation + 주기 refresh~~ — **완료(7차)**: `flush_buffer`에서 전체
   재계산, `update_communities=True`로 동적 단일-스텝 확장. 업스트림 label propagation이
   무한 루프에 빠지는 입력(엣지 2개짜리 2-노드 컴포넌트)이 있어 2-cycle 감지로 정지 —
   7차 §U1
6. ~~**그래프 쓰기를 MemoryOp로 표현**~~ (6차 감사 B3) — **완료**: 기존 entities/facts op의
   payload에 `subject_id`/`object_id`·`entity_type`을 실어 `AgenticMemory._apply_graph`가
   적용. organizer에는 `edges_between` 읽기만 남았고, 로그 재생으로 그래프가 복원됨을
   `test_zep_graph_is_rebuildable_from_the_evolution_log`가 단언한다.
