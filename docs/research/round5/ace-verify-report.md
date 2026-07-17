# ACE 구현 충실도 검증 보고서

- **대상**: `/home/jinmang2/agentic_memory/src/agmem/organizers/ace.py` (+ `src/agmem/core/types.py` Bullet, `src/agmem/memory.py` playbook 경로)
- **기준 논문**: "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models" (arXiv:2510.04618, v3 HTML 확인)
- **공식 코드**: `github.com/ace-agent/ace` — **공식 저장소 확인됨** (README가 arXiv 2510.04618을 자기 논문으로 링크, SambaNova 블로그 "ACE Open-Sourced on GitHub"가 공지, 저자진 Stanford/SambaNova). 검증을 위해 scratchpad에 shallow clone 후 소스 직접 대조함 (`ace/ace.py`, `ace/core/{generator,reflector,curator,bulletpoint_analyzer}.py`, `ace/prompts/*.py`, `playbook_utils.py`).
- **참고**: 로컬 선행 조사 `docs/research/ace-longmemeval.md`의 upstream 분석 내용은 이번 독립 대조에서 모두 사실로 재확인됨.

---

## 1. 항목별 판정 (docs/10 기준 + 지시된 핵심 루프)

### 1.1 Generator / Reflector / Curator 3-role — **일치 (구조적 각색 1건 caveat)**

논문(§method): *"The Generator produces reasoning trajectories; the Reflector distills concrete insights from successes and errors; and the Curator integrates these insights into structured context updates."*

- 우리 구현은 Reflector(`REFLECT_PROMPT`)+Curator(`CURATE_PROMPT`)를 organizer로 두고, Generator 역할은 호스트 에이전트가 수행하는 구조. 메모리 라이브러리 관점에서 합리적 각색.
- **caveat**: 공식 파이프라인에서 reflection은 Generator의 **재생성(regenerate)에 즉시 피드백**된다(`ace/ace.py:530` `generator.generate(..., reflection=reflection_content)`). 우리는 post-hoc organizer라 reflection→재생성 피드백 루프가 구조적으로 불가능. 이 점은 multi-round 누락(docs/10 기재)과 동전의 양면.

### 1.2 Reflection이 trajectory에서 helpful/harmful 교훈 추출 — **대체로 일치 (스키마 단순화)**

- 공식 `REFLECTOR_PROMPT` 출력 필드: `reasoning, error_identification, root_cause_analysis, correct_approach, key_insight, bullet_tags` (tag ∈ helpful/harmful/neutral). 우리: `key_insight, lessons, bullet_tags` — `bullet_tags` enum과 `key_insight`는 동일, `error_identification/root_cause/correct_approach`는 `lessons` 배열로 축약. `lessons`는 upstream에 없는 우리 발명 필드.
- 공식 Curator는 **reflection 전문(모든 필드)**을 `recent_reflection`으로 받는 반면, 우리 Curator는 `key_insight+lessons`만 받음 — 정보 손실 소폭.
- 공식은 정답 여부(`is_correct`)로 분기: 오답이면 최대 `max_num_rounds` 반복, 정답이면 helpful 태깅용 reflection 1회. 우리는 `outcome` 문자열만 넘기고 분기 없음(단일 라운드 고정).

### 1.3 Curator delta ADD → 섹션별 bullet playbook — **일치**

- 공식 코드 주석(`ace/core/curator.py:212-213`): *"Currently only ADD operations are fully supported. Note: You can add support for UPDATE, MERGE, DELETE operations here"* — **upstream도 ADD-only**. 우리 ADD-only는 공식 코드와 동등 (논문 서술의 MERGE/DELETE는 upstream에서 BulletpointAnalyzer 후처리로 대체).
- 프롬프트 핵심 문구 일치: 공식 *"Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook ... Do NOT regenerate the entire playbook"* ↔ 우리 `CURATE_PROMPT` *"Identify ONLY the NEW insights that are MISSING from the current playbook. Do NOT regenerate or rephrase existing bullets."*
- 소차이: 공식은 섹션이 고정 초기 템플릿에 있고 미지의 섹션은 `others`로 폴백(`playbook_utils.py:148-151`); 우리는 자유 섹션 + `general` 폴백. 공식은 op 개수 상한 없음, 우리는 `max_ops=5` 상한(우리 추가 안전장치, 무해).

### 1.4 Bullet 메타데이터(helpful/harmful 카운터) 갱신 — **일치 (귀속 대상 divergence 1건, §3.3)**

논문: *"consisting of (1) metadata, including a unique identifier and counters tracking how often it was marked helpful or harmful; and (2) content..."*

- `Bullet` dataclass(`types.py:152-167`): id/section/helpful/harmful — 논문 메타데이터 구조 충족. 렌더 포맷 `[section-xxxxx] helpful=X harmful=Y :: content`는 공식 `format_bullet()`의 `[{bullet_id}] helpful={h} harmful={hm} :: {content}`와 동형.
- 카운터 갱신 경로 2개: reflection `bullet_tags` → UPDATE op(ace.py:146-157, 5자 축약 id 복원 로직 포함), `report_feedback()`(memory.py:265) — 공식 `update_bullet_counts()`와 의미 일치.

### 1.5 Embedding dedup threshold 0.90 — **일치 (동작 방식 2건 차이, §3.4)**

- 공식: `bulletpoint_analyzer_threshold` 기본 **0.90**(`ace/ace.py:134`), sentence-transformers(all-mpnet-base-v2)+faiss cosine, **opt-in 기본 off**(`use_bulletpoint_analyzer=False`, `ace/ace.py:41,133`), 켜면 curator 적용 직후 전체 playbook 대상 pairwise 그룹핑 후 **LLM merge(카운터 합산, 첫 ID 유지)**.
- 우리: 동일 threshold 0.90, **상시-on**(모듈 docstring에 의도적 편차로 명시 — upstream의 "의존성 미설치 시 조용히 스킵" 재현 함정 회피), 신규 bullet을 기존 스토어와 1:1 비교해 **skip(드롭)**.
- 상시-on은 문서화된 의도적 편차로 인정. skip-vs-merge는 신규 bullet 카운터가 0/0이라 정보 손실 미미.

### 1.6 Grow-and-refine (lazy vs proactive) — **일치 (proactive 변형)**

논문: *"In grow-and-refine, bullets with new identifiers are appended, while existing bullets are updated in place... A de-duplication step then prunes redundancy by comparing bullets via semantic embeddings"*; *"This refinement can be performed proactively (after each delta) or lazily (only when the context window is exceeded)."*

- 공식 코드도 (켰을 때) 매 curator 후 실행 = proactive만 구현, lazy 트리거(컨텍스트 초과 시) 미구현. 우리 상시 dedup = proactive 변형으로 논문과 합치. **lazy 트리거는 우리도 upstream도 없음** — 미구현 지적 불필요.

### 1.7 Multi-round reflection — **docs/10 누락 지적 정확 (수치 각주 1건)**

- 공식 `ace/ace.py:501` `for round_num in range(max_num_rounds)` — 오답 시 reflect→카운터갱신→재생성→정답판정 루프, 기본값 `max_num_rounds=3`(`ace/ace.py:123`). docs/10의 "≤3"은 **코드 기본값 기준으로 정확**.
- 각주: 논문 v3 HTML의 ablation 서술은 refinement round 상한을 5로 설정했다고 읽힘(*"the Reflector critiques these traces to extract lessons, optionally refining them across multiple iterations"* + 상한 5 언급). 코드 기본값(3)과 논문 실험 설정(5)이 다를 수 있으니 docs/10에 "코드 기본 3, 논문 ablation은 최대 5" 병기 권장.

### 1.8 Offline 모드 — **docs/10 누락 지적 정확 (범위 보강 권장)**

- 공식은 `mode ∈ {offline, online, eval_only}`(`ace/ace.py:190`), offline은 train/val 필수(`ace/ace.py:193-194`). 논문: offline = *"optimized on the training split and evaluated on the test split"*, online = *"for each sample, the model first predicts with the current context, then updates its context based on that sample"*. 우리 on_task_end 흐름 = online 모드와 동형 — docs/10의 "online 모드만" 표기 정확.
- 보강: offline 누락에는 train/val 분리뿐 아니라 **multi-epoch(`num_epochs`, ablation Table 3에서 유의미 기여) + val 기반 best_playbook 선택**도 포함됨을 docs/10에 명시할 것.

---

## 2. 핵심 질문: full-playbook 주입 vs top-k retrieval

**논문/공식 코드는 playbook 전체를 Generator 프롬프트에 주입한다. top-k retrieval이 아니다.**

- 공식 `ace/ace.py:465-467`: `self.generator.generate(question=..., playbook=self.playbook, ...)` — `self.playbook`은 **단일 문자열 전체**. `GENERATOR_PROMPT`는 `**Playbook:**\n{}` 슬롯에 전체를 넣고, "fine-grained retrieval"은 Generator가 프롬프트 내에서 관련 bullet의 `bullet_ids`를 골라 인용하는 방식(LLM-내부 선택)으로 구현됨.
- 크기 관리는 retrieval이 아니라 **token budget**(기본 `playbook_token_budget=80000`, curator 프롬프트에 예산·진행도·통계를 알려 성장 억제) + dedup으로 수행.
- 논문의 병리 진단(brevity bias/context collapse) 자체가 "긴 컨텍스트 모델에 포괄적 playbook을 통째로 유지·주입"을 전제: *"prevents collapse with structured, incremental updates that preserve detailed knowledge and scale with long-context models."*

**우리 read 경로 진단**:

| 경로 | 동작 | 판정 |
|---|---|---|
| `AgenticMemory.get_playbook()` (memory.py:288, MCP `get_playbook` 툴로 노출) | "전체 렌더" 의도. 그러나 구현이 `vec.search(embed("playbook"), k=200)` — **문자열 "playbook"의 임베딩으로 유사도 검색해 상위 200개**를 가져오는 근사 | 의도는 논문-faithful하나 구현이 근사치: (a) 200개 초과 시 조용히 절단, (b) 벡터스토어가 유사도와 무관하게 k개를 다 돌려준다는 보장에 의존. `SqliteDocStore`에 type별 전체 나열 API가 없어서 생긴 우회 |
| `AgenticMemory.search(memory_types=["playbook"])` (retrieval pipeline) | 쿼리 기반 **top-k** bullet 반환 | **논문 사용법과 다름**. 호스트가 이 경로로 playbook을 소비하면 ACE의 전제(전체 playbook 제공, 선택은 Generator LLM이 수행)가 깨짐 |

**플래그**: top-k retrieval 경로 자체는 라이브러리 일반 기능이라 존재해도 되지만, **"ACE 방법론으로 벤치마크/사용할 때는 get_playbook(전체)을 컨텍스트에 주입하는 것이 논문 프로토콜"**임이 어디에도 강제/문서화되어 있지 않다. bench 하니스에는 ace/playbook 참조가 전혀 없어(grep 무히트) 현재 측정 경로가 어느 쪽인지 코드로 확인 불가 — ACE를 측정에 올리기 전에 반드시 read-path 계약을 고정해야 함.

---

## 3. 신규 발견 (docs/10 미기재 mismatch)

### 3.1 [중요] Full-playbook 주입 vs top-k retrieval — §2 참조
docs/10은 write 경로(reflect→curate)만 다루고 read 경로 계약을 다루지 않음. 논문의 핵심 주장(포괄적 playbook 보존이 성능 원천)과 직결되므로, docs/10의 규율("미구현 요소가 핵심 주장과 연결되면 측정 전 구현")에 따라 **측정 전 처리 필요**.

### 3.2 [중요] Curator가 전체 playbook 대신 task-유사 top-30만 봄
- 우리 `_current_playbook(ctx, task, k=30)`(ace.py:103-107): **task 임베딩과 유사한 상위 30개**만 curator 프롬프트의 "Current playbook"으로 제공.
- 공식 `CURATOR_PROMPT`는 `current_playbook`에 **전체 playbook**(+`playbook_stats`, `token_budget`, 진행도)을 제공.
- 결과: "MISSING인 것만 ADD"라는 LLM-수준 중복 억제가 부분 뷰에서 판정됨 → task와 임베딩이 먼 기존 bullet과의 중복 ADD 증가. 0.90 embedding dedup이 near-verbatim 중복은 걸러주지만 paraphrase 수준(0.90 미만) 중복은 통과.

### 3.3 [중간] Reflector 태깅 대상이 "실제 사용된 bullet"이 아니라 "retrieval된 bullet"
- 공식: Generator가 출력에서 인용한 `bullet_ids`만 추출해(`extract_playbook_bullets(self.playbook, bullet_ids)`) *"Part of Playbook that's used by the generator"*로 Reflector에 제공 — 카운터는 **실사용 bullet에만** 귀속.
- 우리: task 임베딩 top-30(k=30, curation용과 동일 집합)을 "available bullets"로 주고 trajectory에서 추정 태깅. 프롬프트의 "if you can tell from the trajectory; else omit"이 완화 장치이고 `report_feedback()`이 실사용 기반 경로를 별도 제공하지만, on_task_end 경로에서는 **Generator에게 보인 적 없는 bullet에 카운터가 쌓일 수 있음**.

### 3.4 [경미] intra-batch dedup 공백
- 우리 dedup은 신규 bullet마다 **벡터스토어**를 조회하는데, 같은 on_task_end 배치의 앞선 ADD op는 아직 스토어에 반영 전(`memory.py:169`에서 배치 일괄 apply). 한 reflection에서 나온 유사 lesson 2개가 모두 통과 가능.
- 공식 BulletpointAnalyzer는 curator 적용 **후** 전체 playbook에 pairwise 유사도를 돌려 intra-batch 중복도 포착.

### 3.5 [경미] Playbook token budget / stats 부재
- 공식 curator는 `token_budget`(기본 80,000)·`current_step/total_samples`·`playbook_stats`를 받아 성장을 조절하고, 매 스텝 `count_tokens(self.playbook)`를 추적. 우리는 playbook 크기 상한/예산 개념이 없음(무한 성장은 dedup만이 억제). full-playbook 주입(§3.1)을 도입하면 budget 관리가 전제 조건이 됨.

### 3.6 [경미] 렌더 포맷 이중화
- organizer 내부 `_render_playbook`(ace.py:114-116)은 `[xxxxx] helpful=...`(섹션 없는 5자 id), `Bullet.render`/`memory.get_playbook`은 `[section-xxxxx] helpful=...`. 공식 포맷은 `[sec-00001] helpful=... ::`. 두 렌더가 다르면 reflector가 에코하는 id 형태가 경로마다 달라짐 — ace.py의 `startswith` 복원 로직이 organizer 경로는 흡수하지만, 호스트가 `get_playbook` 렌더의 `section-xxxxx`를 그대로 `report_feedback`에 넘기면 full-id 매칭 실패. 포맷 단일화 권장.

### 3.7 [정보] 정답 여부 기반 분기 부재
- 공식은 environment feedback을 "Predicted answer does/does not match ground truth"로 명시 주입하고 정답/오답 분기. 우리는 자유형 `outcome` 문자열 — no-GT online 세팅(공식 `CURATOR_PROMPT_NO_GT`/`REFLECTOR_PROMPT_NO_GT` 분기 존재)과 유사한 사용법이므로 결격은 아니나, docs/10에 "no-GT online 변형에 해당"이라고 성격을 명시하면 정확해짐.

---

## 4. docs/10 판정 자체에 대한 평가

| docs/10 주장 | 판정 |
|---|---|
| 구현됨: reflect→curate ADD | **정확** (upstream도 ADD-only — 코드 주석으로 확인) |
| 구현됨: helpful/harmful | **정확** |
| 구현됨: dedup 0.90 상시 | **정확** (상시-on은 upstream opt-in 대비 의도적 강화, 모듈 docstring에 문서화됨) |
| 누락: multi-round reflection(≤3) | **정확** (코드 기본 3; 논문 v3 ablation은 5 언급 — 각주 권장) |
| 누락: offline 모드(train/val) | **정확** (+multi-epoch, val 기반 best_playbook 선택도 포함시킬 것) |
| 등급 ◑ | **조건부 타당** — write 경로만 보면 ◑이 맞으나, §3.1(read 경로)·§3.2(curator 부분 뷰)는 논문 핵심 주장(전체 playbook 보존·주입)과 연결됨. read-path 계약을 고정하지 않으면 "측정 가능 ✔(online만)"이 흔들림 |

## 5. 권고 (우선순위순)

1. **read-path 계약 고정**: ACE 사용/측정 시 컨텍스트 주입은 `get_playbook()`(전체)로 한다고 문서화하고, bench 하니스가 실제로 그 경로를 쓰도록 배선. `SqliteDocStore`에 type별 전체 나열 API를 추가해 `get_playbook`의 `vec.search(embed("playbook"), k=200)` 근사를 진짜 full scan으로 교체.
2. **Curator 입력을 전체 playbook로**: `_current_playbook(k=30)` 대신 전체(또는 budget 내 전체) + 간단한 stats 제공. token budget 필드(기본 80k) 동반 도입.
3. **intra-batch dedup**: 같은 배치에서 이미 통과한 신규 bullet 임베딩을 로컬 리스트로 유지해 함께 비교 (몇 줄 수정).
4. **렌더 포맷 단일화**: organizer `_render_playbook`을 `Bullet.render`(`[section-xxxxx]`)로 통일하고 id 복원 로직이 `section-` 접두를 벗기게 보강.
5. **docs/10 갱신**: ACE 행에 §3.1~3.3을 누락 칼럼에 추가하고, multi-round 수치 각주(코드 3/논문 5), offline 범위(multi-epoch+val 선택), "우리 online은 no-GT 변형" 성격 명시.
6. (선택) 오답/정답 분기: `outcome`에 구조화된 success 플래그를 받으면 공식과 동일한 environment feedback 문구 주입 가능.

## 출처
- 논문: https://arxiv.org/abs/2510.04618 (v3 HTML에서 method 인용)
- 공식 코드: https://github.com/ace-agent/ace (로컬 clone: scratchpad/ace-official — `ace/ace.py`, `ace/core/*.py`, `ace/prompts/*.py`, `playbook_utils.py` 라인 인용)
- 공식성 근거: https://sambanova.ai/blog/ace-open-sourced-on-github
- 로컬: `src/agmem/organizers/ace.py`, `src/agmem/core/types.py:152-167`, `src/agmem/memory.py:159-172,260-302`, `src/agmem/mcp/server.py:76-86`, `docs/10-fidelity-audit.md:41`, `docs/research/ace-longmemeval.md`
