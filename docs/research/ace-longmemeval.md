# 조사 보고서: ACE (Agentic Context Engineering) & LongMemEval

> 작성일: 2026-07-16 / 조사 범위: arXiv 논문, 공식 GitHub 코드, 2026년 중반까지의 벤치마크 결과

---

# 1. ACE — Agentic Context Engineering

## A. 핵심 개념

**논문**: "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"
(arXiv:2510.04618, 2025년 10월 제출, v3 리비전 2026년 3월, ICLR 2026, 32페이지, CC-BY 4.0)

**저자**: Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, Kunle Olukotun (Stanford / SambaNova Systems)

### 문제의식

기존 context adaptation 기법(프롬프트 최적화, ICL, 요약 기반 memory)의 두 가지 병리:

- **Brevity bias**: LLM이 컨텍스트를 반복 재작성(rewrite)할 때 "간결함"을 선호하게 되어, 실전에 유용한 도메인 디테일(엣지 케이스, 공식, 실패 패턴)이 점진적으로 소실됨.
- **Context collapse**: 모놀리식(monolithic) 전체 재작성을 반복하면 컨텍스트 정보량이 시간이 갈수록 붕괴하는 현상.

### 핵심 아이디어

컨텍스트를 "진화하는 플레이북(evolving playbook)"으로 취급. 산출물은 한 문단짜리 프롬프트가 아니라 섹션별로 구조화된 itemized bullet 목록.

각 bullet 포맷:
```
[section_slug-00000] helpful=X harmful=Y :: content
```
고유 ID + helpful/harmful 카운터(메타데이터) + 콘텐츠로 구성.

### 3-role 아키텍처

- **Generator**: 새 쿼리에 대해 reasoning trajectory를 생성하며, 유효했던 전략과 반복되는 실패 패턴을 표면화. Playbook과 이전 reflection을 입력받아 answer + 사용한 bullet_ids를 출력.
- **Reflector**: 실행 트레이스를 비평하여 성공/실패로부터 구체적 insight를 추출. 평가(evaluation)/insight 추출을 curation과 분리하는 것이 핵심 설계. 각 bullet에 `helpful`/`harmful`/`neutral` 태그 부여. 여러 라운드 반복 정제 가능(`max_num_rounds`).
- **Curator**: Reflector의 lesson을 구조화된 delta 항목(ADD/UPDATE/MERGE/DELETE operation)으로 합성, **비-LLM 결정론적 로직(deterministic merge)**으로 기존 playbook에 병합.

### Delta update vs 모놀리식 재작성

Curator는 전체 playbook을 재생성하지 않고 "현재 playbook에 없는 새 insight만" JSON operation list로 출력 → 국소적(localized) 편집만 반영되어 기존 지식 보존 + fine-grained retrieval + 점진적 적응 가능.

### Grow-and-refine 메커니즘

- 새 bullet은 append, 기존 bullet은 helpful/harmful 카운터만 in-place 업데이트.
- 중복 제거는 **semantic embedding 유사도** 비교로 수행.
- latency/accuracy trade-off에 따라 proactive(매 스텝) 또는 lazy(주기적) 실행.

### Offline / Online 두 모드 모두 지원

- **Offline**: 시스템 프롬프트 최적화 — `mode='offline'`, train/val/test 분리, 학습된 playbook을 고정 배포.
- **Online**: 에이전트 메모리처럼 실시간으로 playbook이 진화 — `mode='online'`, ground-truth 없이(`no_ground_truth=True`) environment feedback만으로 self-supervised 적응 가능.

## B. 공식 코드 분석

**저장소**: `github.com/ace-agent/ace` (공식, Apache-2.0, ★1.2k / forks 154, Python 100%, 2025년 11월 공개, 활발히 유지보수 중)

### 모듈 구조

```
ace/
├── ace/ace.py                        # ACE 메인 오케스트레이터 (train/eval loop)
├── ace/ace_batch.py                  # 배치 처리 버전
├── ace/core/generator.py             # Generator 클래스
├── ace/core/reflector.py             # Reflector 클래스
├── ace/core/curator.py               # Curator 클래스 (playbook operation 적용)
├── ace/core/bulletpoint_analyzer.py  # 임베딩 기반 dedup/merge
├── ace/prompts/generator.py          # GENERATOR_PROMPT
├── ace/prompts/reflector.py          # REFLECTOR_PROMPT, REFLECTOR_PROMPT_NO_GT
├── ace/prompts/curator.py            # CURATOR_PROMPT, CURATOR_PROMPT_NO_GT
├── llm.py                            # timed_llm_call() — provider-agnostic 호출 래퍼
├── playbook_utils.py                 # extract_json_from_text(), apply_curator_operations()
├── eval/finance/{data_processor.py, run.py}   # FiNER / XBRL Formula
├── eval/mind2web/, eval/mind2web2/   # 웹 에이전트 벤치마크
└── pyproject.toml / uv.lock          # uv 기반 의존성 관리
```

주: 레포에 **AppWorld 평가 코드는 미포함** — 공개 eval 태스크는 finance(FiNER/Formula)와 mind2web/mind2web2.

### Bullet 파싱 정규식 (`bulletpoint_analyzer.py`)

```python
r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)'
```

### 프롬프트 상세

**Curator 프롬프트** (`ace/prompts/curator.py`, `CURATOR_PROMPT`):
- "You are a master curator of knowledge... Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook... Do NOT regenerate the entire playbook — only provide the additions needed."
- 입력: `token_budget`, `current_step/total_samples`, `playbook_stats`, `recent_reflection`, `current_playbook`, `question_context`
- 출력 (순수 JSON): `{"reasoning": "...", "operations": [{"type": "ADD", "section": "formulas_and_calculations", "content": "..."}]}`
- Ground-truth 유무에 따라 `CURATOR_PROMPT` / `CURATOR_PROMPT_NO_GT` 분기.

**Reflector 프롬프트** (`ace/prompts/reflector.py`):
- 입력: question / reasoning trace / predicted answer / ground truth / environment feedback / 사용된 bullet 목록
- 출력 JSON 필드: `reasoning`, `error_identification`, `root_cause_analysis`, `correct_approach`, `key_insight`, `bullet_tags` (`[{"id": "calc-00001", "tag": "helpful"}]`)

**Generator 프롬프트** (`ace/prompts/generator.py`):
- 입력: playbook + reflection + question + context
- 출력: `{"reasoning": ..., "bullet_ids": [...], "final_answer": ...}`
- bullet_id fallback 정규식: `r'\[([a-z]{3,}-\d{5})\]'`

### 중복 제거/병합 (`BulletpointAnalyzer`, opt-in: `use_bulletpoint_analyzer`)

1. `sentence-transformers` (`all-mpnet-base-v2` 기본)로 bullet content 임베딩
2. `faiss.normalize_L2` 정규화 후 cosine similarity matrix 계산
3. threshold (기본 **0.90**) 이상 bullet 그룹핑 (`_find_similar_groups`)
4. 그룹을 LLM (temperature=0.3)으로 병합 — helpful/harmful 카운트 합산, 첫 bullet ID 유지 (`_merge_bullets_with_llm`)
5. 파싱 실패 시 첫 bullet 유지 fallback

### LLM 호출 패턴

- `llm.py`의 `timed_llm_call(client, provider, model, prompt, role, call_id, max_tokens, log_dir, use_json_mode)`
- 3개 role 독립 호출, 매 스텝 `detailed_llm_logs/`에 상세 로그
- Provider: `sambanova`(기본), `together`, `openai`, `commonstack`
- 기본 모델: 3-role 모두 **`DeepSeek-V3.1`**

### 주요 config 기본값

| 파라미터 | 기본값 |
|---|---|
| `num_epochs` | 1 |
| `max_num_rounds` (오답 시 reflection 반복) | 3 |
| `curator_frequency` | 1 |
| `playbook_token_budget` | 80000 |
| `online_eval_frequency` | 15 |
| `test_workers` | 20 |
| `bulletpoint_analyzer_threshold` | 0.9 |

### 출력 아티팩트

`final_playbook.txt`, `best_playbook.txt`(offline 전용), `bullet_usage_log.jsonl`, `curator_operations_diff.jsonl`, `intermediate_playbooks/`, `detailed_llm_logs/`, `run_config.json`, `final_results.json`

## C. 성능

### AppWorld (ReAct 백본, DeepSeek-V3.1)

| Method | Test-Normal TGC | Test-Challenge TGC | Average |
|---|---|---|---|
| ReAct (baseline) | 63.7% | 41.5% | 42.4% |
| ReAct + ICL | 64.3% | 46.0% | 46.0% |
| ReAct + GEPA | 64.9% | 46.0% | 46.4% |
| ReAct + ACE (labels, offline) | **76.2%** | **57.3%** | **59.4%** |
| ReAct + ACE (no labels) | 75.0% | 54.4% | 57.2% |
| ReAct + Dynamic Cheatsheet (online) | 65.5% | 52.3% | 51.9% |
| ReAct + ACE (online) | 69.6% | **66.0%** | **59.5%** |

**리더보드 비교**: ACE offline 59.4%는 GPT-4.1 기반 프로덕션 에이전트 **IBM CUGA**(60.3%)와 동급이며, test-challenge split에서는 더 작은 오픈소스 모델(DeepSeek-V3.1)로 CUGA를 **능가**(+8.4%p TGC).

### Finance (FiNER / XBRL Formula)

| Method | FiNER | Formula | Average |
|---|---|---|---|
| Base LLM | 70.7% | 67.5% | 69.1% |
| ICL | 72.3% | 67.0% | 69.6% |
| MIPROv2 | 72.4% | 69.5% | 70.9% |
| GEPA | 73.5% | 71.5% | 72.5% |
| ACE (labels) | **78.3%** | **85.5%** | **81.9%** |
| ACE (no labels) | 71.1% | 83.0% | 77.1% |
| DC (online) | 74.2% | 69.5% | 71.8% |
| ACE (online) | 76.7% | 76.5% | 76.6% |

평균 개선: agent tasks **+10.6%p**, finance **+8.6%p**.

### Latency / Cost 절감 (논문 Table 4, README)

- **Offline (AppWorld, vs GEPA)**: latency **-82.3%** (53,898s → 9,517s), rollout **-75.1%** (1,434 → 357)
- **Online (FiNER, vs Dynamic Cheatsheet)**: latency **-91.5%** (65,104s → 5,503s), token 비용 **-83.6%** ($17.7 → $2.9)
- README Key Features의 "**86.9% lower adaptation latency on average**"는 위 두 세팅 평균치 — -86.9% 수치는 실재함.

### Ablation (Table 3)

| Configuration | Test-Normal | Test-Challenge | Average |
|---|---|---|---|
| ACE w/o Reflector + multi-epoch | 70.8% | 55.9% | 55.1% |
| ACE w/o multi-epoch | 72.0% | 54.9% | 56.8% |
| Full ACE | **76.2%** | **57.3%** | **59.4%** |

Reflector 분리와 multi-epoch refinement 모두 유의미하게 기여.

### 소형 오픈모델 호환성

전체 실험이 DeepSeek-V3.1(오픈 가중치)로 수행 — GPT-4.1 상용 에이전트와 동급/우위. "works with smaller open models" 확인됨.

## D. 재현 관점

- 설치: `uv sync` + `.env`에 provider API 키. SambaNova가 기본 provider지만 OpenAI/Together로 교체 가능.
- **AppWorld 실험 코드가 공개 레포에 없음** — 재현하려면 논문 부록의 프롬프트/설정을 별도 구현해야 함. 공개 재현 가능 범위는 FiNER/Formula(데이터 포함: `eval/finance/data/*.jsonl`)와 mind2web.
- Curator는 현재 **ADD operation만 완전 지원** (코드 주석: "Currently only ADD operations are fully supported... You can add support for UPDATE, MERGE, DELETE operations here"). 논문의 MERGE/DELETE는 `bulletpoint_analyzer.py`의 별도 후처리로 보완되는 구조.
- Dedup은 **opt-in이며 기본값 off** (`use_bulletpoint_analyzer=False`) — 기본 실행에서는 grow-and-refine의 "refine" 축이 비활성. 논문 효과 재현 시 반드시 플래그를 켤 것.
- `sentence-transformers`/`faiss-cpu`는 optional dependency — 미설치 시 dedup이 **경고만 출력하고 조용히 스킵**됨(재현 함정).

---

# 2. LongMemEval

**논문**: "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory"
(arXiv:2410.10813, **ICLR 2025**, 저자 Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu)

**저장소**: `github.com/xiaowu0162/LongMemEval` (MIT License)

## A. 벤치마크 설계

### 5대 핵심 능력

1. **Information Extraction (IE)** — 긴 히스토리에서 정확한 세부정보 검색 (user/assistant 발화 모두)
2. **Multi-Session Reasoning (MR)** — 여러 세션에 흩어진 정보의 집계·비교·합성
3. **Knowledge Updates (KU)** — 사용자 정보 변경 감지 및 최신값 반영
4. **Temporal Reasoning (TR)** — 명시적 시간 언급 + 타임스탬프 메타데이터에 대한 추론
5. **Abstention (ABS)** — 히스토리에 없는 정보를 묻는 질문에 "모른다"고 답하는 능력

### 질문 타입 (총 500문항)

`single-session-user`, `single-session-assistant`, `single-session-preference`, `multi-session`, `temporal-reasoning`, `knowledge-update`.
`question_id`가 `_abs`로 끝나면 abstention 질문(**30문항**, 기존 질문의 false-premise 변형).

### 데이터 내 태스크명 ↔ 공식명 매핑 (README)

| 데이터 내 이름 | 공식 명칭 |
|---|---|
| single_hop | single-session-user |
| implicit_preference_v2 | single-session-preference |
| assistant_previnfo | single-session-assistant |
| two_hop / multi_session_synthesis | multi-session |
| temp_reasoning_implicit / temp_reasoning_explicit | temporal-reasoning |
| knowledge_update | knowledge-update |

### 구축 방법 (attribute-controlled pipeline)

1. **164개 사용자 attribute**를 5개 카테고리(demographic, lifestyle, life events, situational context, belongings)로 수작업 정의
2. LLM으로 attribute 기반 사용자 배경 문단 생성
3. 전문가가 seed question 필터링/재작성
4. 답변을 timestamp 있는 evidence statement로 수작업 분해
5. Self-chatting으로 evidence session 생성 (정보를 간접적으로 노출)
6. 사람이 evidence 포함 여부 검수/편집
7. Filler session(ShareGPT + UltraChat 소스)과 evidence session을 섞어 timestamped history 컴파일

### 3가지 variant

| 파일 | 규모 | 설명 |
|---|---|---|
| `longmemeval_s.json` | **~115k tokens** (Llama-3 tokenizer 기준, ~40 sessions) | 128k 컨텍스트 모델에 맞춤 |
| `longmemeval_m.json` | **~500 sessions** (~1.5M tokens 상당) | long-context 한계 초과, retrieval 필수 |
| `longmemeval_oracle.json` | evidence session만 | 완벽한 retrieval 상한선 |

배포: HuggingFace `xiaowu0162/longmemeval-cleaned` (2025/09 cleaned 버전 — 정답 간섭 세션 정제).

### 인스턴스 필드

`question_id`, `question_type`, `question`, `answer`, `question_date`, `haystack_session_ids`, `haystack_dates`, `haystack_sessions` (각 turn은 `{"role", "content"}`, evidence turn엔 `has_answer: true` 라벨 — turn-level recall 평가용), `answer_session_ids` (session-level recall 평가용).

## B. 평가 프로토콜 (코드 분석)

### 저장소 구조

```
LongMemEval/
├── data/custom_history/            # custom history 구축 corpus
│   ├── 1_attr_bg/, 2_questions/, 5_filler_sess/, 6_session_cache/
├── src/retrieval/run_retrieval.sh  # 메모리 인덱싱+검색
├── src/generation/run_generation.sh # (retrieval-augmented) QA
├── src/evaluation/
│   ├── evaluate_qa.py              # LLM judge 평가
│   ├── print_qa_metrics.py         # 정확도 집계
│   └── print_retrieval_metrics.py  # recall@k 등
├── src/index_expansion/            # key expansion, temporal query pruning
└── src/utils/serve_vllm.sh         # 로컬 오픈모델 서빙
```

### Judge 프롬프트 (`src/evaluation/evaluate_qa.py`의 `get_anscheck_prompt()`, 소스 원문 확인 — 질문 타입별 5분기)

- **`single-session-user` / `single-session-assistant` / `multi-session`**:
  "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer... If the response only contains a subset of the information required by the answer, answer no... Answer yes or no only."
- **`temporal-reasoning`**: 위와 동일 + "do not penalize off-by-one errors for the number of days... predicting 19 days when the answer is 18, the model's response is still correct."
- **`knowledge-update`**: "If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer."
- **`single-session-preference`** (rubric 기반): "The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly."
- **abstention**: "Please answer yes if the model correctly identifies the question as unanswerable."

### Judge 실행 설정

- `model_zoo = {'gpt-4o': 'gpt-4o-2024-08-06', 'gpt-4o-mini': 'gpt-4o-mini-2024-07-18', 'llama-3.1-70b-instruct': 로컬 vLLM(localhost:8001)}`
- `temperature=0`, `max_tokens=10`, yes/no 바이너리 판정 (`'yes' in response.lower()`), `backoff.expo` 재시도
- 논문 주장: 인간 전문가와 **>97% 합치율**

### 실행 흐름

1. 시스템 출력을 `{question_id, hypothesis}` jsonl로 저장
2. `python3 evaluate_qa.py gpt-4o hyp_file ref_file`
3. `.eval-results-gpt-4o` 로그에 `autoeval_label` 추가
4. `print_qa_metrics.py`로 question_type별 집계

### 메모리 시스템 플러그인 (unified index)

`bash run_retrieval.sh IN_FILE RETRIEVER GRANULARITY`

- **GRANULARITY**: `turn` 또는 `session` (value 단위)
- **RETRIEVER**: `flat-bm25`, `flat-contriever`, `flat-stella`(Stella-en-1.5B-v5), `flat-gte`(gte-Qwen2-7B-instruct); dense는 multi-GPU 지원
- **Index expansion** (key 단위 확장): `session-summ`, `session-keyphrase`, `session-userfact`, `turn-keyphrase`, `turn-userfact` × join mode `separate`/`merge`/`replace` — (key, value) 분리 설계로 "fact를 key로, 원문 세션을 value로" 같은 구성 가능
- **Time-aware query expansion**: `src/index_expansion/temp_query_search_pruning.py` — 쿼리에서 시간 범위 추론 → 검색 공간 pruning (recall +6.8~11.3%p)
- Retrieval 평가 시 abstention 30문항은 항상 제외 (정답 위치가 없으므로)

### 논문 베이스라인 수치

- Oracle (evidence만) GPT-4o: **~87%**
- LongMemEval_S full-context: GPT-4o **60.6%** (오라클 대비 -30%p), Llama-3.1-70B 33.4%, Llama-3.1-8B 45.4%
- 상용 시스템(축소 세팅): ChatGPT(GPT-4o) 57.7%, Coze 33.0%
- LongMemEval_M + 최적 retrieval 구성(round decomposition + key=value+fact): GPT-4o 65.7%, Llama-70B 72.0%
- 핵심 메시지: long-context LLM도 지속 상호작용 기억에서 **30~60%p 성능 하락**
- Reading method는 `con`(extract-then-reason, Chain-of-Note 스타일) + history format `json` 권장 — 이 설정만으로 최대 10%p 차이

### SOTA / 리더보드 정리 (2026년 중반 기준)

> ⚠️ 출처 성격 주의: Zep/Nemori는 arXiv 논문 자체 보고치, Mem0/Supermemory는 **벤더 자체 발표치**(judge 버전·방법론 상이 가능, 제3자 미검증). 직접 비교 시 주의.

| 시스템 | 출처 | Overall | 비고 |
|---|---|---|---|
| Full-context GPT-4o | 원 논문 | ~60.6% | S variant 기준선 |
| **Zep** (Graphiti, gpt-4o) | arXiv:2501.13956 | **71.2%** (베이스 60.2% 대비 +18.5%) | latency 28.9s→2.58s (-90%); 카테고리별: ss-user 92.9 / ss-assistant 80.4 / preference 56.7 / multi-session 57.9 / KU 83.3 / TR 62.4 |
| Zep (gpt-4o-mini) | 〃 | 63.8% (+15.2%) | latency 31.3s→3.20s |
| **Nemori** (gpt-4o-mini) | arXiv:2508.03341 | **64.2%** (full-context 55.0% 대비) | 컨텍스트 95~96% 절감; preference 46.7 / TR 61.7 / multi-session 51.1 |
| Nemori (gpt-4.1-mini) | 〃 | **74.6%** (full-context 65.6%) | preference 86.7 / TR 72.2 |
| **Mem0** (token-efficient algorithm, 2026) | mem0.ai/research (자체) | **94.4%** | 평균 6,787 tokens/call (full-context 25k+ 대비 3~4배 절감); ss-user 94.3 / ss-assistant 98.6 / preference 46.4 / KU 98.2 / TR 76.7 / multi-session 96.7 |
| **Supermemory** (2026/05) | supermemory.ai 자체 | **95%** (Recall@15+aggregation) | mean 720 tokens (99.4% 절감); ss-assistant 100 / KU 99 / TR 91 / multi-session 93 |
| MemoryOS | arXiv:2506.06326 | LongMemEval 수치 미보고 | 주로 LoCoMo 사용 (F1 +49.11%, BLEU-1 +46.18%) |
| MemForest | 2026 arXiv | 70.4% (4B reader) / 79.8% (30B reader) | |
| Memoria | 2026 | ~88.78% (LongMemEval_S) | retrieval 강점 |
| EM-LLM | — | LongMemEval 공식 수치 미확인 | 주로 LongBench/∞-Bench 계열 평가 |

### LongMemEval-V2 (후속 벤치마크)

- arXiv:2605.12493, 2026/05 공개, `github.com/xiaowu0162/LongMemEval-V2`
- 대화 기억 → **에이전트 경험 기억(agentic experience memory)**으로 전환
- WebArena/WorkArena(++) 궤적(599+941개)에서 **451문항** 수작업 큐레이션, 컨텍스트 **25M~115M tokens**
- 새 5축: static state recall, dynamic state tracking, workflow knowledge, environment gotchas, premise awareness
- 궤적 미제공 시 frontier LLM 정확도 **14.1%**에 불과 — 경험 메모리 필요성 입증
- 참고치(LME-V2-Small): AgentRunbook-C 74.9%, Vanilla Codex 69.9%, AgentRunbook-R 58.6%, Simple RAG 51.0%
- V1은 여전히 대화형 메모리 표준으로 활발히 사용 중 (별개 벤치마크로 취급)

## C. 재현 관점

- **컴퓨팅/비용**: S(115k tokens)는 128k 컨텍스트 모델 단일 pass로 가능(빠르고 저렴). M(~500 sessions)은 retrieval 필수 — dense retriever(Stella 1.5B / GTE-Qwen2-7B)는 GPU 필요(multi-GPU 병렬 내장), BM25는 CPU만으로 가능. 환경은 CUDA 12.1 + torch 2.3.1 기준 테스트됨.
- **Judge 비용**: 500문항 × 1콜, `max_tokens=10`이라 출력 비용은 미미 — GPT-4o 기준 1회 평가에 수 달러 수준. 다만 retrieval 그리드 서치(retriever 4종 × granularity 2종 × expansion 5종 × join 3종)를 돌리면 generation+judge 콜이 수천~수만 회로 누적.
- **Judge 버전 drift**: 코드가 `gpt-4o-2024-08-06` 스냅샷을 하드코딩 — deprecate되면 재현 불가/판정 분포 변동. ">97% 인간 합치율"도 해당 스냅샷 기준. 벤더 블로그 수치들 간 비교가 어려운 주 원인 중 하나.
- **로컬 judge 가능성**: `llama-3.1-70b-instruct`(vLLM, `src/utils/serve_vllm.sh`)가 이미 `model_zoo`에 공식 등록되어 있어 로컬 judge 경로 존재. Judge 태스크가 단순 yes/no 판정이라 70B급이면 근사 가능하나, 소형(8B급) judge는 preference/abstention 타입에서 신뢰도 별도 검증 권장.
- **환경 분리**: `requirements-lite.txt`(평가만) / `requirements-full.txt`(retrieval+generation 포함) 이원화 — 자체 메모리 시스템만 평가할 땐 lite 환경으로 무거운 의존성 회피 가능.
- **기타 함정**:
  1. cleaned 버전(2025/09)과 원본 버전의 수치가 다를 수 있음 — 어떤 버전인지 명시 필요.
  2. `longmemeval_oracle.json`의 `haystack_session_ids`는 timestamp 정렬이 안 되어 있음 (s/m은 정렬됨).
  3. reading method(`direct`/`con`/`con-separate`)와 history format(`json`/`nl`) 선택만으로 최대 10%p 차이 — 비교 실험 시 반드시 통일할 것.

---

# 주요 참고 경로 요약

## ACE
- 논문: https://arxiv.org/abs/2510.04618
- 공식 코드: https://github.com/ace-agent/ace
  - `ace/prompts/{curator,reflector,generator}.py` — 프롬프트 원문
  - `ace/core/bulletpoint_analyzer.py` — semantic dedup/merge
  - `ace/ace.py`, `playbook_utils.py` — 오케스트레이션/playbook 연산
- SambaNova 공개 블로그: https://sambanova.ai/blog/ace-open-sourced-on-github

## LongMemEval
- 논문: https://arxiv.org/abs/2410.10813
- 공식 코드: https://github.com/xiaowu0162/LongMemEval
  - `src/evaluation/evaluate_qa.py` — judge 프롬프트 원문 (`get_anscheck_prompt()`)
  - `src/retrieval/run_retrieval.sh`, `src/generation/run_generation.sh`
  - `src/index_expansion/temp_query_search_pruning.py`
- 데이터: HuggingFace `xiaowu0162/longmemeval-cleaned`
- 후속: https://github.com/xiaowu0162/LongMemEval-V2 (arXiv:2605.12493)

## SOTA 출처
- Zep: https://arxiv.org/abs/2501.13956
- Nemori: https://arxiv.org/abs/2508.03341 / https://github.com/nemori-ai/nemori
- Mem0: https://mem0.ai/research (자체 발표)
- Supermemory: https://supermemory.ai/research/longmembench/ (자체 발표)
- MemoryOS: https://arxiv.org/abs/2506.06326
