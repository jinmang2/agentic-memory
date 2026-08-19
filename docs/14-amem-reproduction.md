# A-Mem × LoCoMo 재현 가이드 (2026-07-23)

> 목표: **A-Mem(arXiv:2502.12110, NeurIPS'25)의 LoCoMo 수치를 우리 재구현으로 재현**하고,
> 재현 과정을 누구나 다시 돌려 검증할 수 있도록 코드·설정·명령·비용을 전부 문서화한다.
> 자매편: docs/13(A-Mem 스터디 — 논문/공식코드/우리구현 대조), docs/11(Nemori).
> 근거 코드: 업스트림 클론 `WujiangXu/AgenticMemory`(이하 "업스트림 클론"),
> 우리 `src/agmem/bench/locomo.py`, `scripts/exp_amem_repro.py`, `scripts/repro/*.sh`.

이 문서는 **fidelity-sensitive**하다. 업스트림 수치를 재현하려는 값(토크나이저·F1·온도·k·
cat5 MCQ)은 업스트림 클론 코드에 **file:line으로 못박아** 대조했다.

---

## 1. 재현 사다리 (reproduction ladder)

각 단(rung)은 **딱 하나의 gap**을 격리한다. 위에서 아래로 내려갈수록 "논문 수치"에서
"우리가 실제로 배선한 것"으로 이동한다.

| Rung | 무엇 | 격리하는 gap | 스크립트 |
|---|---|---|---|
| **0 published** | 논문 Table 1의 A-Mem 수치 (§3) | (기준점) | — |
| **1a upstream@ours** | 업스트림 A-Mem **자기 코드**를 gpt-4o-mini로 실행 | 논문(0) ↔ 업스트림코드 재실행 gap (Mem0 재실행 발산·백본·데이터) | `scripts/repro/phase1a_upstream.sh` (외부 코드, **문서 전용**) |
| **1b our-reimpl@aligned** | **우리 organizer**를 업스트림-정렬 설정으로 실행 | 업스트림코드(1a) ↔ 우리 재구현(1b) gap — 배선 충실도 | `scripts/repro/phase1b.sh` |
| **2 our-production** | 우리 organizer를 **우리 프로덕션** 설정(ours 메트릭 + J-judge + 링크확장) | 재현용(1b) ↔ 우리가 실제로 배포할 형태(2) gap | `scripts/repro/phase2.sh` |
| **3 sensitivity** | k-sweep(10–50) · evolution ablation | 하이퍼파라미터·설계요소 민감도 | `scripts/repro/phase3_ksweep.sh`, `phase3_ablation.sh` |

**핵심 논리**: 1a와 1b는 *같은 설정*(MiniLM 임베더, notes-only flat cosine, k=10,
set-based F1, cat5 MCQ)에서 돌린다. 두 수치의 차이 = 순수 재구현 gap. 1b와 2의 차이 =
"충실 재현"과 "우리 프로덕션 배선"의 차이.

---

## 2. 확정된 업스트림 설정 (file:line 대조)

전부 업스트림 클론에서 직접 확인. `scripts/repro/phase1a_upstream.sh`가 이 값들로 외부
코드를 돌리는 커맨드를 문서화한다.

| 항목 | 값 | 업스트림 근거 (file:line) |
|---|---|---|
| write 온도 | **0.7** | `memory_layer.py:43,88,137,209` — `get_completion(..., temperature=0.7)` 기본값 |
| answer 온도 (cat1–4) | **0.7** | `test_advanced.py:146` — `temperature = 0.7` |
| answer 온도 (cat5) | **0.5** | `test_advanced.py:160,415` — `temperature = self.temperature_c5`, CLI 기본 `--temperature_c5 0.5` |
| retrieval k | **10** | `test_advanced.py:417` — `--retrieve_k` 기본 10; `answer_question:134` `k=self.retrieve_k` |
| 임베더 | **all-MiniLM-L6-v2** | `test_advanced.py:33,41` — `SentenceTransformer('all-MiniLM-L6-v2')` |
| read 경로 | **순수 in-memory sklearn cosine** (BM25 없음) | `memory_layer.py:678,744` `self.retriever = SimpleEmbeddingRetriever(...)`; class `:554`, cosine `:605` `cosine_similarity(...)`; `find_related_memories_raw:877` = cosine + 1-hop 링크 확장 |
| 데이터셋 | `data/locomo10.json`, **1,986 QA** | cat1=282, cat2=321, cat3=96, cat4=841, cat5=446 |
| gold | cat5=`adversarial_answer`, else `answer` | `load_dataset.py:17-21` `QA.final_answer` |
| 메트릭 | **set-based token F1** (dedup) | `utils.py:129-145` `calculate_metrics` |
| 토크나이저 | lowercase + `. , ! ?`→space + split, **stemming 없음** | `utils.py:34-38` `simple_tokenize` |
| cat5 생성 | 2지선다 MCQ (gold vs "Not mentioned in the conversation", 랜덤순서) | `test_advanced.py:146-160` |
| 채점 대상 | 전 5개 카테고리 (cat5도 동일 F1로) | `test_advanced.py:259,320` `allow_categories=[1,2,3,4,5]` |
| J-score | **없음** | `utils.py` — LLM judge 미존재 |

> ⚠️ 주의: 업스트림 클론의 `test_advanced.py`는 plain `memory_layer.py`를 쓴다. **논문
> 수치는 `*_robust.py` 경로**(`memory_layer_robust.py` + `test_advanced_robust.py`)에서
> 나온다 — docs/13 §3. rung 1a는 두 경로 모두 커맨드를 문서화한다.
> robust의 차이는 Ps1 live와 방어적 파싱이지 **retrieval 채널이 아니다**: 두 경로 모두
> `SimpleEmbeddingRetriever` 순수 cosine이고, `HybridRetriever`(`memory_layer.py:403`)는
> 어느 eval 경로에서도 호출되지 않는 dead code다(이 표 "read 경로" 행과 일치).

### cat5 MCQ 재현성 처리 (우리의 의도적 편차)

업스트림 `test_advanced.py:149-154`는 **시드 없는** `random.random() < 0.5`로 두 옵션
순서를 정한다 → 재실행이 non-reproducible. 우리는 **질문의 md5 해시**로 코인을 뽑아 순서를
byte-stable하게 고정한다(`cat5_options`, `locomo.py`). 옵션 자체(gold vs "Not mentioned
in the conversation")·프롬프트 문구·채점(raw reply를 gold와 set-F1)은 업스트림과 동일.
letter로 답하는 모델을 위해 `(a)`/`(b)` → 옵션 텍스트 resolver도 추가(`resolve_cat5_reply`).

---

## 3. A-Mem 논문 Table 1 (gpt-4o-mini) 재현 타깃

LoCoMo, F1 (BLEU-1은 참고용 2번째 수치 — docs/08/13의 선행 분석 기준):

| 방법 | Multi (cat1) | Temporal (cat2) | Open (cat3) | Single (cat4) | Adversarial (cat5) |
|---|---|---|---|---|---|
| **LoCoMo-baseline** | 25.02 | 18.41 | 12.04 | 40.36 | 69.23 |
| **A-Mem** | 27.02 | 45.85 | 12.14 | 44.65 | 50.03 |

읽는 법(docs/13 §2): A-Mem의 최대 이득은 **Temporal(+27.4)**, baseline은 오히려
**Adversarial(69.23)**이 높음 — cat5는 "언급 안 됨"을 잘 맞추는 baseline이 강하다.

---

## 4. 세 개의 eval 코드베이스 구분 (혼동 주의)

"A-Mem LoCoMo eval"이라 불리는 코드가 **셋** 있고, 채점 규칙이 서로 다르다. 우리
`--eval-mode`가 어디에 대응하는지가 재현의 핵심이다.

| 코드베이스 | 채점 규칙 | 우리 대응 |
|---|---|---|
| **snap-research/locomo** | SQuAD-style `normalize_answer`: lowercase + 구두점제거 + **관사(a/an/the/and) 제거 + Porter stemming**, multiset F1. **BLEU는 없음** — 지표는 EM / stemmed token-F1 / `rougel_score`(이름과 달리 rouge-1 F 반환) 3종 | `--eval-mode ours` (`locomo.normalize`/`token_f1`). `locomo.bleu1`은 **공식 대응물이 없는 우리 지표** |
| **WujiangXu/A-Mem** (`utils.calculate_metrics`) | F1은 **set-based** token F1, lowercase + `.,!?`→space, **stemming/관사제거 없음**. BLEU-1은 **다른 토크나이저** — `nltk.word_tokenize(lower)` + `sentence_bleu(weights=(1,0,0,0), SmoothingFunction().method1)` | `--eval-mode wujiang` (`locomo.token_f1_wujiang` / `locomo.bleu1_wujiang`) |
| **Mem0-J** | LLM binary judge (CORRECT/WRONG), cat1–4만 | `--judge` (ours 모드 전용, `locomo.judge_answer`) |

**왜 우리의 이전 eval 수정(commit e74534a)은 snap-research를 겨냥했나**: 우리 원래
`normalize`/`token_f1`은 snap-research/locomo의 공식 채점기를 미러링한 것이라 stemming과
관사 제거가 들어간다. 그건 **snap-research 재현**엔 맞지만 **A-Mem(WujiangXu) 재현엔
과하게 관대**하다(stem이 부분일치를 늘림). 그래서 A-Mem 재현용으로 `wujiang` 모드를 새로
추가해 **set-based·no-stem** 규칙을 정확히 미러링한다. 두 모드는 완전히 분리되어 있고
`ours` 경로는 e74534a 그대로 불변이다.

---

## 5. 이 브랜치(`feat/locomo-eval-fidelity` — 이후 `main`으로 병합, 브랜치는 삭제됨)의 코드 변경 전량

### 5.1 commit e74534a — snap-research eval 충실도 수정 (선행)

| 파일:심볼 | 무엇 | 왜 |
|---|---|---|
| `locomo.py:normalize` | lowercase+구두점제거에 **관사(a/an/the/and) 제거 + Porter stemming** 추가 | snap-research `normalize_answer` 정합 |
| `locomo.py:token_f1` | multiset(Counter) 겹침 기반 F1 | snap-research `f1_score` 미러 |
| `locomo.py:bleu1` | 같은 `normalize` 위의 unigram precision + brevity penalty | **우리 지표** — snap-research/locomo에는 BLEU가 없음(리포 전량 검색 0건). "공식 채점기 미러"라는 종전 표기는 오류 |
| `locomo.py:gold_for` | cat3 gold `"A; B; C"` → 첫 alias `"A"` truncation | snap-research가 primary answer로 채점 |
| `locomo.py:ANSWER_PROMPT_NO_ABSTAIN` | cat5는 abstention 문장 제거한 프롬프트 | cat5는 답을 강제(거부 불가) |
| `locomo.py:judge_answer` + `evaluate(judge=)` | Mem0-style binary J-score (cat1–4, opt-in) | Mem0 판정 재현 |
| `bench/_porter.py` (현재 `src/agmem/_porter.py`) | Porter stemmer 구현 | stemming 의존성 없이 |

### 5.2 이번 commit — A-Mem 재현 하네스 + wujiang eval 모드

| 파일:심볼 | 무엇 | 왜 |
|---|---|---|
| `src/agmem/_env.py:load_env_local` | 의존성 없는 `.env.local` KEY=VALUE 로더, 기존 env 미덮어씀 | repo-root `.env.local`의 `OPENAI_API_KEY`를 pip dep 없이 |
| `locomo.py:_tok_wujiang` | 업스트림 `simple_tokenize` 미러 (`utils.py:34-38`) | wujiang F1 토크나이저 |
| `locomo.py:token_f1_wujiang` | **set-based** F1 미러 (`utils.py:129-145`) | A-Mem 정확 재현 채점 |
| `locomo.py:gold_for_wujiang` | cat5=adversarial, else answer, **cat3 truncation 없음** (`load_dataset.py:17-21`) | 업스트림 gold 정합 |
| `locomo.py:cat5_options`/`CAT5_MCQ_PROMPT`/`resolve_cat5_reply` | cat5 2지선다 MCQ, md5-seeded 순서, letter/text resolver (`test_advanced.py:146-160`) | cat5 생성 재현(순서는 재현성 위해 시드 고정) |
| `locomo.py:answer(eval_mode,gold,cat5_temperature)` | wujiang cat5는 MCQ 프롬프트+0.5온도, 그 외 불변 | 모드 분기 |
| `locomo.py:evaluate(eval_mode,cat5_temperature)` | wujiang이면 gold/F1/cat5를 wujiang 경로로, **judge 강제 off** | 모드 스레딩 |
| `scripts/exp_amem_repro.py` | 재현 하네스 (아래 §6 CLI) | 재현 사다리 구동 |
| `scripts/repro/*.sh` | rung별 실행 스크립트 6종 | "모든 실험 돌리는 script" |

### 5.3 이번 commit — 전체 run-artifact capture + write-once/read-sweep

| 파일:심볼 | 무엇 | 왜 |
|---|---|---|
| `llm/client.py:LLMClient(trace_path)`/`_trace` | 콜당 full-I/O JSONL sink(프롬프트·응답 truncate 없음, 성공/실패 모두), sink 미설정 시 무동작 | re-spend insurance(산출물 4). backward-compatible |
| `bench/locomo.py:answer(capture=)`/`evaluate(capture_retrieval=)` | 질문별 검색 detail(raw/rewrite query·k·types·`{id,type,score,text}`)을 record `retrieval` 필드로 | retrieval chunk 보존(산출물 2). 비캡처 경로 불변 |
| `stores/sqlite_doc.py:list_episodes` | episodic 전량 열거(`list_items` 파생 타입과 대칭) | 메모리 스냅샷 episodic 라인(산출물 5) |
| `scripts/exp_amem_repro.py` | trace/memory 배선·`dump_memory_snapshot`·`memory_capacity`·`cost_usd`·`git_sha`·`timing`·enriched stamp·ingest-only 산출물 | 5-산출물 매니페스트 |
| `scripts/repro/*.sh` | smoke/phase1b/phase2/ksweep를 ingest-once + eval-only(read-sweep)로 | 유료 write 1회, eval 무한 재실행 |
| `.gitignore` | `*.llm-trace.jsonl`/`*.memory.jsonl`/`stores/` 무시(durable-on-disk) | repo bloat 방지, 소형 산출물은 커밋 |
| `tests/test_repro_artifacts.py` | trace full I/O·retrieval capture·memory snapshot·write-once→eval-only 0 write콜 | fake만, 유료 콜 0 |

`exp_locomo_conv0.py`는 **손대지 않았다**(로컬 Qwen, answer t=0.0 그대로).

---

## 6. 실행 방법

### 사전 준비
1. **`.env.local`** (repo-root)에 `OPENAI_API_KEY=sk-...` — 이미 `.gitignore`됨
   (`git check-ignore .env.local`로 확인). 값은 절대 커밋/출력 금지.
2. **임베더 다운로드**: 최초 실행 시 `all-MiniLM-L6-v2`(~90MB)가 자동 다운로드된다.
3. `uv`로 실행(모든 스크립트가 `uv run python ...`).

### CLI (`scripts/exp_amem_repro.py`)

```
--model         (기본 gpt-4o-mini)
--endpoint      (기본 https://api.openai.com/v1); api_key는 OPENAI_API_KEY env
--embedder      (기본 all-MiniLM-L6-v2)
--conv          (기본 0; 'all' 또는 정수 인덱스 0-9)
--k             (기본 10)
--eval-mode     (wujiang|ours, 기본 wujiang)
--expand-links  (off|on, 기본 off) — A-Mem 1-hop 노트 링크 확장 토글
                (주의: read-path 플러그인화(3b39c7d) 이후 한동안 무효 토글이었다 —
                 구 코드가 `mem.pipeline.link_expansion_cap`이라는 죽은 속성에 대입해
                 on/off 양쪽이 cap=5로 돌았다. 지금은 `AgmemConfig.link_expansion_cap`로
                 배선됨. sha e2e7ebe 이하의 `expand-off` 산출물은 리팩터 이전이라 유효)
--judge         (store_true; ours 모드에서만 적용)
--runs          (기본 1) — answer 경로 반복해 answer-temp mean±std (write 경로는 불변)
--workers       (기본 1) — QA 동시 실행 워커 수. store는 read-only·전 store 락 보호·A-Mem
                on_retrieval no-op이라 결과는 workers=1과 비트동일(테스트 보증). eval 처리량용
--tag-suffix    (기본 "") — 출력 tag에 접미(예 _seed1). 서로 다른 store로 반복 full run 시
                산출물 덮어쓰기 방지 (헤드라인 K=3에서 사용)
--data-dir      (스토어 영속화 경로; --ingest-only/--eval-only의 전제)
--ingest-only   (ingest+영속화만, QA 없음; 완료 시 `.ingest_complete.json` sentinel 기록)
--eval-only     (영속화된 스토어 reload 후 QA만; sentinel 없으면/불완전하면 거부)
```

> **⚠️ --runs vs 헤드라인 K회 ingest**: `--runs`는 **answer 경로만** 반복한다(고정 note 그래프
> 위에서 재채점) → answer 온도(0.7/cat5 0.5) 분산만 측정. **재현 변동의 지배원인은 write 경로
> (temp 0.7 → note 그래프 draw)**인데 이건 `--runs`로 안 잡힌다. rung 1b/2 헤드라인 수치는
> `phase1b_headline.sh`/`phase2_headline.sh`(K=3 **독립 ingest**, seed별 fresh store)로 뽑아
> `aggregate_headline.py`가 per-category mean±std를 낸다. multi-run cost는 요약 JSON의 top-level
> `cost_usd`가 **전 run 합계**(실제 지출), `runs[]`에 per-run `cost_usd`/`llm_budget`도 기록.
>
> **헤드라인 비용은 정직 분리**: 집계 JSON은 `eval_cost_usd`(답변/재채점 지출) + `ingest_cost_usd`
> (`--ingest-summaries`로 ingest 요약 넘길 때) + `campaign_cost_usd`(=ingest+eval)로 나눠 보고한다.
> eval-only 합을 "total"로 라벨하지 않는다(그러면 ingest 지출이 누락). phase1b/phase2 헤드라인은
> **seed store를 공유**하므로 두 `campaign_cost_usd`를 더하면 공유 ingest 이중계산 — `ingest_note`가
> 이를 경고한다. 또 집계기는 seed들의 model/k/eval_mode/expand가 다르면 평균을 **거부**한다.

출력: 실행당 **다섯 산출물**(요약 `<tag>.json` + `<tag>.records.jsonl` + `logs/` +
`<tag>.llm-trace.jsonl` + `<tag>.memory.jsonl`). 요약 JSON은 config stamp + metrics +
`cost_usd` + `timing` + `memory_capacity` + 나머지 산출물 포인터를 담는다. 전체 스키마·git
정책·오프라인 re-score·write-once/read-sweep은 아래 **"산출물 & 영속성 — 재현성 계약"** 참조.

### 각 phase

```bash
bash scripts/repro/smoke.sh            # conv0, wujiang + ours+judge (~$0.35)
bash scripts/repro/phase1b.sh          # rung 1b: full, our-reimpl@aligned, wujiang (단일 ingest)
bash scripts/repro/phase2.sh           # rung 2: full, our-production (ours+judge+expand, 단일)
bash scripts/repro/phase1b_headline.sh # rung 1b HEADLINE: K=3 독립 ingest → mean±std (~$4.8)
bash scripts/repro/phase2_headline.sh  # rung 2 HEADLINE: K=3 (seed store 공유 시 eval만 ~$2.3)
bash scripts/repro/phase3_ksweep.sh    # rung 3: k∈{10,20,30,40,50}, ingest-once + eval-sweep
bash scripts/repro/phase3_ablation.sh  # rung 3: Full vs w/o-evolution (아래 주의)
# scripts/repro/phase1a_upstream.sh    # rung 1a: 업스트림 자기 코드 (문서 전용, 실행 X)
```

> **동시성 (2축, 둘 다 결과 불변 · wall-clock만 단축)**:
> - **eval QA 동시성** `WORKERS`(기본 8): 질문은 read-only store 위에서 독립 → 비트동일
>   (`WORKERS=16 bash scripts/repro/phase1b.sh`).
> - **ingest 대화 동시성** `INGEST_WORKERS`(기본 4, 헤드라인 스크립트): `scripts/repro/ingest_parallel.py`가
>   대화를 병렬 ingest한다. **대화 간은 공유 상태 0**(각 대화 = 별도 namespace store, 순차 하네스도
>   대화를 fresh하게 build) → 순차와 **바이트 동일**. **대화 내부 turn은 병렬 불가**(evolution이
>   노트 그래프 read-modify-write라 turn N이 1..N-1에 의존 — 순서 바뀌면 그래프가 달라짐). 즉
>   병렬화는 딱 안전한 축(대화)에서만 한다. `ingest_parallel.py`는 `--conv all --ingest-only`의
>   **drop-in**: per-conv 워커(`--no-sentinel`)를 `--workers`개 동시 실행 후 **통합 sentinel +
>   `<model>_all_ingest<sfx>.json` 1개**를 써서 `--eval-only`/aggregator가 그대로 동작.
>   - **rate-limit 제어**: 대화 워커는 한 번에 API 콜 1개만 in-flight → 동시 콜 ≈ `--workers`.
>     4로 시작, 429 없으면 상향. SDK 429 백오프(max_retries=2) + 오케스트레이터 conv-단위 재시도
>     (`--retries`, 재시도 전 부분 store wipe)로 이중 방어. RAM ≈ `workers × ~1GB`.
>   - **비용 불변**: 병렬은 콜 수·토큰·비용을 안 바꾼다(벽시계만). 콜 수·비용은 여전히 요약 JSON의
>     `llm_budget`/`cost_usd`에서 확인.
> **헤드라인 K**: `K=5 bash scripts/repro/phase1b_headline.sh`로 ingest draw 수 조절.
> **ingest 완료 가드**: phase 스크립트는 `<store>/.ingest_complete.json` sentinel로만 ingest를
> skip한다(bare dir 아님). 죽은/부분 ingest store는 자동 `rm -rf` 후 clean 재-ingest → truncated
> store로 micro-average하는 사고 방지.

> **evolution ablation 주의**: 우리 `AMemOrganizer`는 evolution을 **LLM의 per-note
> `should_evolve` 판정**(`amem.py:209`)으로 게이팅하며, **생성자 스위치가 없다**. 즉
> `AMemOrganizer(evolve=False)` 같은 "evolution 끄기" 토글이 현재 **미구현**이다. 충실한
> ablation을 하려면 (1) evolution 비활성 시 `EVOLVE_PROMPT` 콜을 건너뛰고 ADD+LINK만
> emit하는 스위치를 추가하고, (2) `exp_amem_repro.py`에 `--no-evolution` 플래그를 배선해야
> 한다. `phase3_ablation.sh`는 현재 Full arm만 돌리고 이 gap을 명시한다 → **follow-up 필요**.

### 산출물 & 영속성 (Artifacts & persistence) — 재현성 계약

USER HARD REQUIREMENT: **한 번의 (유료) 실행이 모든 것을 durable하게 남겨** 어떤 분석도
유료 재실행을 강요하지 않는다 — "time, intermediate values, retrieval chunks, memory
capacity, all of it, down to the smallest thing". 한 실행은 **다섯 가지 산출물**을 남긴다.
모두 `results/repro/`(프로젝트 실디스크, 세션 간 durable)에 쓰이고 **절대 scratchpad(/tmp,
휘발성)에는 쓰지 않는다**.

| # | 산출물 | 경로 | git? | 내용 |
|---|---|---|---|---|
| 1 | 요약/매니페스트 JSON | `results/repro/<tag>.json` | ✅ commit | config stamp(model/embedder/k/eval_mode/temps/expand + **git_sha·dataset_path·utc_started/finished·cost_rates·cat5_seed**) + per-category/overall metrics + `llm_budget`(role별 calls/tokens_in/out/latency) + **`cost_usd`**(tokens×gpt-4o-mini rate, rate는 stamp에 자기기술) + `timing`(ingest_s/eval_s/total_s) + `drops` + `memory_capacity` + 나머지 4개 산출물 포인터(`records_file`/`llm_trace_file`/`memory_file`). lean 유지. |
| 2 | per-question 감사 추적 | `results/repro/<tag>.records.jsonl` | ✅ commit | 질문 1개당 JSON 1줄. `run`·`conv` 태그 + `q`/`gold`/`pred`/`cat`/`f1`(+judge 시 `j`) + **`retrieval`**(그 질문이 실제로 무엇을 검색했나: raw `query`, keyword-rewrite된 `rewritten_query`, `k`, `memory_types`, 그리고 `retrieved`=`{id,memory_type,score,text}` 리스트 — context에 실제로 들어간 청크 텍스트). **모든 conv·run·질문**. |
| 3 | 실행 로그 | `results/repro/logs/<script>_<UTC>.log` | ✅ commit | `scripts/repro/*.sh`가 `tee`로 콘솔+파일 동시 기록. `.gitignore` 전역 `*.log`를 `!results/repro/logs/*.log`로 un-ignore해 **커밋**됨. |
| 4 | **전체 LLM I/O trace** | `results/repro/<tag>.llm-trace.jsonl` | ⛔ durable-on-disk (gitignored) | 모든 `LlamaClient.chat()` 콜당 JSON 1줄: `{ts_iso, role, budget_key, model, messages(FULL 프롬프트), response_text(FULL), tokens_in, tokens_out, latency_ms, error}`. 성공·실패 **둘 다** 기록, **truncate 없음**. **이 파일 하나로 오프라인 replay/re-score를 API 콜 0으로** 할 수 있다(re-spend insurance). |
| 5 | **메모리 스냅샷** | `results/repro/<tag>.memory.jsonl` | ⛔ durable-on-disk (gitignored) | ingest 후 저장된 **모든** 아이템 1개당 JSON 1줄: episodic(원 turn) + 파생 타입(notes 등). `{conv, memory_type, ...item dict 전부}`(content/tags/links/keywords/context/metadata/kind). 용량·capacity는 요약 JSON의 `memory_capacity`(per-type 개수·total·바이트·store dir 바이트)에도 요약. |

추가로 `--data-dir`로 실행하면 **영속 store**가 `results/repro/stores/<store>/<namespace>/`
(SQLite/vector/graph DB dir)에 남는다 — 이것도 durable-on-disk(gitignored).

**git 정책(명시)**: 작은 것(요약 `.json`·`.records.jsonl`·`logs/`)은 **커밋**한다. 무거운
것(`*.llm-trace.jsonl`·`*.memory.jsonl`·`stores/`)은 full run에서 거대해질 수 있어 **커밋하지
않되 실디스크에 durable하게 남긴다**(사라지지 않음 — 단지 repo bloat 방지). 규칙은
`.gitignore`에 `results/repro/*.llm-trace.jsonl` / `results/repro/*.memory.jsonl` /
`results/repro/stores/`로 못박혀 있다.

**오프라인 re-score 방법 (API 콜 0)**: `<tag>.llm-trace.jsonl`의 각 줄이 그 콜의 완전한
프롬프트+응답이다. cat1-4는 `role=="generate"` 줄의 `response_text`가 답, cat5는 같은 줄에
MCQ 응답이 들어있다. 채점기(`locomo.token_f1_wujiang`/`token_f1`)를 이 응답들에 다시 돌리면
어떤 메트릭·판정도 유료 재실행 없이 재계산된다. 검색이 무엇을 넣었는지는
`<tag>.records.jsonl`의 `retrieval` 필드에 이미 있다.

### write-once / read-sweep (재실행 비용 제거)

`--ingest-only --data-dir <store>`로 **딱 한 번** ingest(=유료 write path: 노트 추출 Ps1 +
evolution Ps3)해 store를 영속화하고, 이후 `--eval-only --data-dir <store>`로 **몇 번이든**
QA만 재실행한다. eval-only는 ingest·consolidate를 **건너뛰고** 영속 namespace를 reopen하며,
**write-path LLM 콜(extract-note/distill)을 0회** 낸다(조직자 `on_message`가 호출되지 않음 —
`AMemOrganizer`는 `on_message`만 write; `on_retrieval`/`consolidate`는 no-op). eval 시에는
읽기 경로 콜만 난다: keyword-rewrite(`extract`) + answer(`generate`). 구조적 증명은
`tests/test_repro_artifacts.py::test_write_once_then_eval_only_issues_zero_write_calls`
(fake embedder+LLM, 유료 콜 0).

- **`--k`는 검색-시점 파라미터**다(`locomo.evaluate(k=...)`가 search에서 적용). ingest에
  baked되지 **않으므로** eval-only에서만 바꿔도 옳다(노트·링크는 k와 무관, 질문당 몇 개를
  가져오느냐만 달라짐). → phase3 k-sweep은 ingest 1회 + eval-only 5회.
- **`--expand-links`도 read-시점**(`AgmemConfig.link_expansion_cap` → `LinkExpansion` 스텝
  등록 여부/캡으로 반영). 마찬가지로
  eval-only에서 on/off 가능.
- **`--eval-mode`/`--judge`도 채점-시점**. 따라서 같은 store 하나로 wujiang·ours 둘 다 평가
  가능 → smoke.sh는 ingest 1회 + eval-only 2회, phase1b/phase2는 같은
  `results/repro/stores/full_all_seed<seed>` store를 공유(존재하면 ingest skip;
  `SEED` env로 선택, 기본 1 — 2026-08-19에 seed 스토어 3벌 체제로 경로 갱신).

`<tag>` = `<model>_<conv>_k<k>_<eval-mode>_expand-<on|off>_run<runs>` (ingest-only는
`<model>_<conv>_ingest`).

---

## 7. 비용 추정 (gpt-4o-mini)

| 항목 | 대략 비용 |
|---|---|
| smoke (conv0, wujiang + ours+judge) | ~$0.35 |
| rung 1b (full, wujiang) | ~$1.6 |
| rung 2 (full, ours+judge+expand) | ~$3.2 |
| rung 1–2 합계 | **~$3.2** (1b는 2에 포함되는 write를 공유하지 않으므로 별도면 ~$4.8) |
| + k-sweep(ingest 1회 + 5×answer) | ~$5.4 |
| + ablation | ~$3.2 |
| **합계 + 버퍼** | **$10–15 권장** |

숫자는 1,986 QA 기준 write(Ps1+Ps3 2콜/노트) + answer(1콜/QA) + keyword rewrite(1콜/QA)
개략치. 실측 cost는 결과 JSON의 `llm_budget`(role별 tokens_in/out)에서 확인.

---

## 8. 결과

> **(2026-08-19)** 이 섹션의 placeholder 표(⬜)는 이 문서 안에서 끝내 채워지지 않았고,
> 이제 채우지 않는다: 사다리 실행 이후의 fidelity 수정들이 write 경로를 바꿔 rung 구획
> 그대로의 표가 더 이상 정본이 될 수 없기 때문이다. **A-Mem의 측정 정본은
> `docs/18-locomo-4way.md`의 5-way 표다** — 전체 10-conversation, 단일 하네스·단일 judge,
> per-category 표와 read-path ablation(키워드 rewrite, per-hit 링크 확장 캡) 포함.
> rung 0(논문 발표치: Multi 27.02 / Temporal 45.85 / Open 12.14 / Single 44.65 /
> Adversarial 50.03)과의 대조·해석도 그 문서의 각주 규칙을 따라 인용할 것.

---

## 9. 알려진 confounder / 왜 EXACT 재현이 기대되지 않나

1. **Mem0 baseline 재실행 발산**: 논문 baseline 수치는 Mem0의 특정 시점 재실행 산출.
   재실행 시 LLM 비결정성으로 발산 — rung 0↔1a 차이의 상당 부분.
2. **온도·비결정성**: write 0.7 / answer 0.7 / cat5 0.5 모두 t>0 → 같은 설정도 run마다
   흔들림. `--runs N`으로 mean±std 측정 권장.
3. **메트릭 차이**: snap-research(stem+관사) vs WujiangXu(set-based, no-stem)는 **같은
   답에도 다른 F1**을 준다. rung 비교 시 반드시 같은 `--eval-mode`로.
3b. **[정정] 저장된 wujiang BLEU-1은 혼종이다**: `eval_mode="wujiang"`은 gold·F1·cat5만
   업스트림으로 갈아끼우고 BLEU-1은 `ours` 정규화(구두점 제거 + 관사 제거 + Porter
   stemming)를 쓰고 있었다. 업스트림은 BLEU에 **다른 토크나이저**(`nltk.word_tokenize`)를
   쓴다. 현재는 `locomo.bleu1_wujiang`이 실제 nltk를 호출하고, nltk가 없으면 지표를
   **생략**한다(0.0으로 보고하지 않음 — `agg`가 키 자체를 뺀다). 이 함수 이전에 기록된
   wujiang 산출물(`results/repro/*_wujiang_*.json`의 `bleu1`)은 **`ours`/`wujiang` 혼종
   값**이므로 논문 BLEU-1과 대조 금지. **F1은 영향 없음** — 그 경로는 처음부터 분리돼
   있었고 헤드라인 수치는 전부 F1 기반이다.
4. **QA 개수 7,512 vs 1,986**: 논문/일부 분석이 인용하는 7,512는 다른 카운팅(대화·turn
   단위 등). 우리 `data/locomo10.json`은 **1,986 QA**(cat1=282/cat2=321/cat3=96/cat4=841/
   cat5=446). 절대 수치 인용 시 어느 카운트인지 명시.
5. **read 채널 차이**: 업스트림 plain은 SimpleEmbeddingRetriever cosine + per-hit 1-hop
   링크(캡 없음 → k=10이면 이웃 ~100). 우리 `LinkExpansion`(`retrieval/steps.py`)은 **전역
   cap=5**. `--expand-links on/off`로 토글하되 캡 의미가 달라 multi-hop 수치는 편차.
6. **robust vs plain**: 논문 수치는 robust(Ps1 live) 경로이고, 우리 rung 1b는 notes-only
   dense + keyword-query다. **채널 종류 자체는 양쪽 다 순수 cosine으로 일치**하므로
   (종전 서술의 "robust=BM25 hybrid"는 오류) 남는 차이는 keyword-query 재작성과 링크 캡
   의미(위 5번)다. 그래도 절대 재현이 아니라 **상대 gap 측정**이 목적(docs/13 §6 캐비앗).
7. **backbone**: 논문은 최소 1B. gpt-4o-mini는 재현 범위 안이지만 로컬 0.6B(exp_locomo_conv0)
   와는 다른 리그 — 두 실험을 섞어 비교 금지.

---

## 9b. 아티팩트 실태 — 이 캠페인에서 복구 불가능한 두 가지 (2026-08-18 감사)

`scripts/repro/audit_artifact_pointers.py`(무지출)가 커밋된 요약 445개의 아티팩트 포인터를 전수
검사하면서 드러난 것. **어느 발표 수치도 여기에 의존하지 않는다** — 점수는 records에서, 용량은
요약의 `memory_capacity`에서 나온다. 그럼에도 적는 이유는 이 리포의 never-delete 규칙이
"아티팩트는 디스크에 영속한다"고 말하기 때문이고, 두 항목 모두 그 약속과 어긋난다.

**(1) 3-seed 캠페인의 per-conv 메모리 스냅샷 33건이 없다.** seed1/2/3 × conv0–9의
`*.memory.jsonl`이 디스크에 없는데, **각 요약이 그 파일의 바이트 수를 기록하고 있다**
(755 KB ~ 1.19 MB). 기록이 있다는 건 파일이 존재했다는 뜻이다. 언제 어떻게 사라졌는지는
남은 기록으로 특정할 수 없고, 재생성은 유료 ingest 재실행을 뜻하므로 **하지 않는다.**
같은 캠페인의 `.records.jsonl`과 요약은 전부 온전하다.

**(2) 3-seed 캠페인 합본 요약의 `op_counts`는 영구히 빈칸이다.** `merge_ingest_summaries`가
이 블록을 통째로 누락하던 결함(Track 2에서 수정)의 잔재로, 수정 이전에 쓰인 합본 6개가
"write 경로가 아무것도 안 했다"고 기록하고 있었다. 그중 **3개는 2026-08-18에 복원했다** —
per-conv 요약이 온전해서, 하네스 자신의 병합 함수를 그 위에 다시 돌렸다(합산을 새로 구현하지
않았다; 두 번째 합산은 한 리포의 두 파일이 같은 런을 두고 다투게 되는 방식이다):

| 합본 | 복원된 `op_counts` |
|---|---|
| `gpt-4o-mini_all_ingest_e3s` | ADD:episodic 5,882 / ADD:notes 5,882 / LINK:notes 5,866 / UPDATE:notes 16,342 |
| `..._nemori_upstream_all_ingest_e3sA` | ADD:episodic 5,882 / ADD:episodes 555 / ADD:semantic 1,926 / MERGE·INVALIDATE:episodes 223 |
| `..._nemori_merge085_all_ingest_e3sB` | ADD:episodic 5,882 / ADD:episodes 763 / ADD:semantic 1,745 / MERGE·INVALIDATE:episodes 22 |

교차검증: `ADD:episodic` 5,882 = 캠페인 총 턴수, armA `semantic` 1,926 = 임베더 진단이 기록한
값, 그리고 **armA 223 병합 vs armB 22** 는 B-3의 결론(필터가 죽은 arm이 후보를 자유롭게 병합)이
예측하는 방향 그대로다.

나머지 **3개(seed1/2/3)는 복원할 수 없다.** 계획 메모는 "A-Mem·Nemori 두 arm은 per-conv에
op_counts가 있다"고 적어두었으나 seed 캠페인에는 **per-conv에도 없다** — 그 런들은 op-log
아티팩트가 도입되기 전에 돌았기 때문이다. 원천이 없으므로 합산할 것이 없고, 빈칸이 정확한 상태다.

## 10. 더 읽기
- docs/13 — A-Mem 스터디(논문 formalization·3벌 코드·우리 구현 워크스루)
- docs/08 / docs/10 — 발표용 리뷰 · 충실도 등급표
- 코드: `src/agmem/bench/locomo.py` · `scripts/exp_amem_repro.py` · `scripts/repro/*.sh`
- 업스트림 근거: `WujiangXu/AgenticMemory` (`utils.py`, `test_advanced.py`, `memory_layer.py`, `load_dataset.py`)
