# v1 이후: 추가 측정·연구 질문·구현 후보 (2026-09-08)

> 후속 구현: E0/I6의 결과 검증·manifest 준비 도구와 사용 절차는 [docs/25](25-measurement-preparation.md)에 기록한다.
> 진단·반복 설계·고정 저장본·오프라인 회계의 후속 구현은 [docs/26](26-measurement-followup.md)에 기록한다.
> 아래의 “미구현/이번 조사”는 조사 시점의 범위이며, 이후 구현 여부는 해당 문서에서 확인한다.

## 1. 지금의 결론과 범위

**먼저 기존 결과를 분해하고 제품 검색의 계약을 맞춘 뒤, 증류의 효용과 실제 코딩 작업 전이를 따로 검증하는 것이 좋다.**
새 organizer를 더 구현해야 한다는 근거는 이번 조사에서 나오지 않았다. v0의 9개 방법론 재현은 종료된 연구이고,
v1의 원문 보존·탐색기·훅 연결·새 세션 도그푸딩도 이미 구현·확인됐다. 후속은 품질과 일반화의 문제다.

이 문서는 코드·저장 산출물·1차 문헌을 대조한 **조사 결과와 제안**이다. 아래 I/E/R 항목을 구현하거나 유료 실험을 실행한 보고서가 아니다.
이번에 실행한 것은 기존 결과 집계와 문서 검증이며, 제품 Python 코드·설정·스토어는 바꾸지 않았다.
`$0`는 새 외부 모델/API 과금이 없다는 뜻이다. 로컬 계산·검토 시간까지 없다는 뜻은 아니다.

| 근거 등급 | 이 문서에서 뜻하는 것 |
|---|---|
| 확인 | 현재 코드 또는 저장된 결과에서 직접 확인한 사실 |
| 기존 관찰 | 과거 도그푸딩 보고에 있으나 이번에 라이브 스토어로 재현하지 않은 사실 |
| 가설·제안 | 실험으로 판별해야 할 설명이나 아직 구현하지 않은 후보 |

현재 상태 정본은 [docs/23](23-v1-experience-memory.md)이다. [기존 로드맵](06-roadmap.md),
[착수 전 네 축 조사](research/agent-memory-axes-v1.md), 내부 계획서의 날짜별 기록은 역사 자료다.
내부 HTML 지도는 로컬 비추적 파일이므로, 다른 머신에서도 읽을 후속 범위는 이 문서에 둔다.

## 2. 기존 결과에서 확인한 것

### 2.1 실제 평가 산출물

공통 경로는 `results/lme_v2/full/`이다. 아래 여섯 디렉터리에 각각 240문항의 `per_question.jsonl`과
`aggregated_metrics.json`, `prompt_rows.jsonl`, 실행 설정·견적이 있다. 6×240은 **서로 다른 1,440문항이 아니라 같은 240문항의 반복 평가**다.

| 디렉터리 | 정답 / 전체 | 정답률 | reader+탐색 비용 |
|---|---:|---:|---:|
| `agmem_raw_vector_web_small` | 62/240 | 25.8% | $0.409 |
| `agmem_experience_vector_web_small` | 65/240 | 27.1% | $0.330 |
| `agmem_raw_explorer_web_small` | 87/240 | 36.2% | $1.505 |
| `agmem_raw_explorer_web_small_rerun` | 96/240 | 40.0% | $1.491 |
| `agmem_experience_explorer_web_small` | 71/240 | 29.6% | $1.593 |
| `agmem_experience_vector_web_small_b4` | 57/240 | 23.8% | $0.335 |

비용은 [`lme_v2_summary.py`](../scripts/repro/lme_v2_summary.py)의 당시 단가와 저장 토큰을 사용한 계산값이다.
청구서 총액이 아니며 judge·최초 저장본 구축은 제외된다. 따라서 다음 실행의 가격이나 지출 승인을 대신하지 않는다.
`summary.json`과 각 아암의 집계값을 대조하고 기존 스크립트도 새로 실행했다.

```bash
uv run --no-sync python scripts/repro/lme_v2_summary.py results/lme_v2/full \
  --json /tmp/agmem-lme-v2-summary.json
```

이 명령은 저장 결과만 읽는다. 로컬 upstream이 있으면 기존 LAFS 계산도 출력하지만,
web-only 점을 논문의 web+enterprise 참조선과 비교한 숫자는 공식 동일조건 순위로 사용하지 않는다.
`results/lme_v2/`는 조사 시점에 미추적 상태였다. fresh clone에는 없을 수 있으므로 파일 존재 여부부터 확인해야 한다.

### 2.2 능력별 분모와 정정

출처: 두 `agmem_raw_explorer_web_small*` 디렉터리의 `aggregated_metrics.json`.
앞 네 행은 non-abstention 범주이고, abstention은 세 하위 범주의 합이다.

| 범주 | 1차 | 재실행 |
|---|---:|---:|
| static | 32/60 = 53.3% | 26/60 = 43.3% |
| dynamic | 14/51 = 27.5% | 17/51 = 33.3% |
| procedure | 18/42 = 42.9% | 21/42 = 50.0% |
| gotchas | 6/15 = 40.0% | 5/15 = 33.3% |
| static-abs | 10/31 = 32.3% | 13/31 = 41.9% |
| dynamic-abs | 5/21 = 23.8% | 7/21 = 33.3% |
| procedure-abs | 2/20 = 10.0% | 7/20 = 35.0% |
| abstention 전체 | 17/72 = 23.6% | 27/72 = 37.5% |
| non-abstention 전체 | 70/168 = 41.7% | 69/168 = 41.1% |

기존 `abs 24·35`의 35는 전체 abstention 비율이 아니었다. 전체 점수의 +9 정답은 abstention +10, 일반 문항 −1로 분해된다.
**추론**: 재실행의 전체 개선을 곧바로 사실 검색·절차 지식 향상이라고 부를 수 없다. abstention 판단과 증거 회수를 나눠 봐야 한다.
`is_unknown` 횟수와 abstention 정답 수는 다른 필드다. 문자열 `unknown` 개수만 세어 premise awareness 점수로 대체하면 안 된다.

87/240→96/240은 정확히 3.75pp 차이다. 두 실행의 불일치는 24:33, 대응 부호검정 p≈0.289다.
두 번만으로 “분산 ±4pp”나 신뢰구간을 확정할 수 없고, 같은 문항을 재사용했으므로 독립 데이터셋 두 개도 아니다.
raw+explorer와 raw+vector의 대응 검정은 두 실행 모두 p≤0.003이지만, 이는 이 모델·데이터·하네스 조건에 한정된다.

### 2.3 무엇이 없고 무엇이 이미 있는가

| 항목 | 확인된 상태 | 후속의 정확한 범위 |
|---|---|---|
| web small의 다섯 능력 | 이미 측정됨 | 범주별 대응 분석·반복·일반화 |
| enterprise | `results/lme_v2/agmem_raw_vector_enterprise_small/`에 raw 저장본과 runtime input 존재, 평가 결과 없음 | 실제 채점 및 필요 시 experience 저장본 구축 |
| 20문항 smoke | 별도 `*_eval20*` 산출물 존재 | 풀런과 섞거나 독립 240문항 근거로 세지 않기 |
| 독립 네이티브 대조군 | 논문 참조 수치는 있으나 자체 재실행은 없음 | 같은 조건의 baseline 실행 |
| 증류 explorer 반복 | 한 번만 평가 | 같은 저장본 반복과 새 증류 반복을 분리 |
| 중간 reader checkpoint | experience+explorer에 `reader_outputs.jsonl` 240행 존재 | 다른 아암에 이 파일이 없다고 평가 누락으로 판정하지 않기 |
| 원문·단계·출처·압축 보존 | 구현되어 있음 | 새로 만드는 대신 품질·효용·경계 조건 검증 |

## 3. 추가 측정 범위

우선순위는 연구 기여의 크기만이 아니라, 현재 불확실성을 얼마나 적은 변경으로 줄이는지도 반영한 제안이다.

| ID | 질문과 최소 대조 | 필요한 자료·코드 | 지표·완료 조건 | 새 모델 호출 |
|---|---|---|---|---|
| E0 | 기존 6아암의 차이는 어느 범주에서 나는가? | `per_question.jsonl`을 question ID로 대응, 기존 summary 확장 | ID 중복·누락 거부, 분모·정답·뒤집힘·효과 크기 출력. 범주별 탐색 분석과 사전 지정 주분석 구분 | 없음 |
| E1 | 탐색 실패는 예산·도구·모델 중 어디에 있는가? | `query_traces/`, `memory_post_query_metadata`, 저장 문맥 | 빈 문맥·degraded·step cap을 별개로 집계하고 겹침 표 출력. 범주별 오류 목록, p50/p95, 토큰 기여 | 없음 |
| E2 | 벡터는 증거를 못 찾는가, reader가 못 쓰는가? | `prompt_rows.jsonl`과 공개 gold, 사람이 확인한 근거 위치 | 정답 증거 포함/미포함 × 정답/오답. 문자열 정답 포함은 근거 포함의 대리임을 표시 | 기존 답 분석은 없음; oracle-context 재답변은 있음 |
| E3 | role 정책을 맞춰도 유용한 recall을 유지하는가? | 동일 스토어 복제본·고정 질의, I1 | 도구 트래픽 비율, eligible 후보 recall@k, 반환 개수, p95. runbook은 보존, 타 프로젝트 결과 0 | 로컬 검색·수동 라벨은 없음; end-to-end 답변 비교는 있음 |
| E4 | 증류의 검색·탐색 이득이 반복되는가? | raw/experience × vector/explorer, 같은 reader·질의·예산 | 원래 저장본 고정 반복을 먼저 실시. 이후 독립 증류 저장본으로 write 변동 분리. 절차와 abstention을 별도 보고 | 있음 |
| E5 | 기억이 쌓일 때 어디부터 열화되는가? | I6, 고정 정답 근거 + 중첩 distractor 집합 | 예: 1×/2×/4×/8× 규모. recall@k·빈 결과·p95·저장량부터, 정확도는 후속. distractor 순서 효과 분리 | 로컬 검색 곡선은 없음; explorer·reader 곡선은 있음 |
| E6 | web 결과가 enterprise에서도 성립하는가? | 기존 enterprise raw 저장본, 같은 아암 설정 | 211문항 결과, 범주별 분모, web와 동일한 평가 조건. 두 도메인이 있을 때만 공식 결합 지표 | 있음 |
| E7 | 성능 격차는 모델·이미지·탐색 예산 중 무엇인가? | I5/I8, 텍스트·이미지 on/off, 탐색 스텝 한 요인씩 | 고정 reader에서 controller·embedding·이미지를 분리. query latency와 전체 응답시간을 따로 기록 | 있음 |
| E8 | 경험 메모리가 실제 코딩을 돕는가? | 시간순 train/test 태스크, 동일 코드 스냅샷·도구·모델 | 무경험 / 원문 / runbook / 원문+runbook. 테스트 통과, 해결 시간, 호출 수, 같은 실수 반복률, 비용/해결 | 하네스·정적 테스트는 없음; agent 실행은 있음 |

E0의 문항 bootstrap은 **고정 실행의 문항 불확실성**만 추정한다. 실행 간 변동은 E4의 독립 반복이 필요하다.
E1에서 step cap에 닿은 실행이 항상 빈 문맥인 것은 아니고, 빈 문맥이 항상 오답인 것도 아니다.
E2에서 gold를 retrieval 입력이나 explorer workspace에 넣으면 평가 누출이다. gold는 사후 분석기만 읽는다.
E5에서 정답 근거까지 함께 제거하면 “기억 누적 열화”와 “근거 부재”가 섞인다. 검색·읽기 곡선과 쓰기·증류 누적 실험도 분리한다.
E8은 답변 QA가 아니라 실제 테스트로 판정할 수 있는 작은 코딩 태스크부터 시작한다. 세션을 무작위로 나눠 같은 과제의 정답이 양쪽에 들어가게 하지 않는다.

## 4. 구현할 코드 후보

아래는 새로 발견한 구현 공백과 기존 기능의 확장을 구분한 목록이다. 자동 실행 목록이 아니다.
제품 수정과 벤치마크 아암 수정은 분리한다. 특히 사용자 턴만 주입하는 제품 정책을 궤적 벤치마크에 그대로 적용하면 도구 행동 증거를 잃는다.

### I1. prompt recall의 role 정책과 top-k 충족 — 우선

**확인**: [`hooks_recall`](../src/agmem/mcp/server.py)(553행 이후)은 검색 결과를 그대로 반환한다.
[`recall_prompt.fallback_items`](../src/agmem/hooks/recall_prompt.py)는 에피소드를 사용자 턴으로 제한한다.
**제안**: hook의 episodic 후보만 같은 role 정책을 적용하고 runbook·파생 메모리는 보존한다. 일반 MCP 원문 검색은 별도 계약이다.
단순히 top-5 뒤에서 필터링하면 0건이 될 수 있으므로, 적격 후보 단계의 필터 또는 상한 있는 후보 확장이 필요하다.
[`AgenticMemory.search`](../src/agmem/memory.py)의 프로젝트 필터도 검색 뒤에 있어 같은 후보 고갈 현상을 함께 검토할 가치가 있다.
수용 조건: user/assistant/tool_result/runbook 혼합 fixture, 상위 후보가 전부 도구 턴인 fixture, 타 프로젝트 fixture에서
적격 결과·반환 수·빈 결과를 검증한다. `tests/test_daemon.py`와 `tests/test_hooks.py`의 실제 경로를 재사용한다. 로컬 테스트 가능.

### I2. namespace 경로 일치 — 가설 검증 후 작은 수정

**확인**: `recall_prompt.request_body`(123행 이후)는 query/k/cwd만 보내고, 서버는 namespace도 받는다.
fallback의 `open_doc_store`는 환경에서 namespace를 해석한다.
**가설**: 기존 데몬의 기본 namespace와 hook 환경이 다를 때 두 경로가 달라질 수 있다. 일반적인 같은 환경 기동에서는 문제를 증명하지 못했다.
수용 조건: 두 namespace를 가진 데몬과 다른 `AGMEM_NAMESPACE` hook 환경을 구성해 실제 결과 차이를 먼저 재현한다.
필요한 경우 기존 `resolve_namespace`를 재사용하며 fast path에 store open을 추가하지 않는다. 재현 안 되면 버그 목록에서 제외한다.

### I3. 증류 stage·과제 입도 평가와 개선 — 라벨 품질

**확인**: [`experience/organizer.py`](../src/agmem/organizers/experience/organizer.py)에 stage enum과 prompt가 이미 있다.
34건 중 `other` 20건은 이전 도그푸딩 관찰이며 현재 스토어 전체 비율은 아니다.
**제안**: 가시 구간만 보고 사람이 부여한 소규모 라벨셋을 먼저 만들고 혼합 과제·정말 분류 불가한 과제를 포함한다.
stage macro-F1·`other` precision·중복 과제 비율·인용 근거 커버리지를 함께 본다. `other` 비율만 낮추는 것이 목표가 아니다.
수용 조건: enum·인용 검증의 기존 동작을 유지하고, 고정 holdout에서 라벨 품질을 비교한다.
`success`만 보고 `verify`로 강제 보정하는 규칙은 단계와 결과를 혼동하므로 채택하지 않는다. 프롬프트 문장 자체를 고정하는 테스트도 피한다.
스키마·파서 테스트는 무료, 새 모델의 라벨 품질 검증은 별도 모델 호출 또는 명시적으로 마련한 로컬 모델이 필요하다.

### I4. 실제 노출과 명시적 feedback의 연결 — 제품 효용

**확인**: `on_retrieval` 카운터와 [`report_feedback`](../src/agmem/mcp/server.py)(408행 이후)는 이미 구현되어 있다.
반면 `recall_prompt.render`는 id 대신 잘린 text를 주입한다. “검색됨”, “모델에게 보임”, “실제로 도움이 됨”은 서로 다른 사건이다.
**제안**: 주입한 memory ID·버전·시점을 추적하고 명시적 feedback을 해당 노출에 연결한다. 기존 카운터·MCP를 재사용한다.
수용 조건: 렌더 예산 밖으로 잘린 후보가 노출로 기록되지 않고, 동일 이벤트 재전송이 중복 가산되지 않으며,
정확한 namespace·memory ID의 helpful/harmful만 바뀐다. 사용자가 말하지 않은 성공을 자동 추론해 reward로 만들지 않는다.

### I5. read policy를 LME-V2에서도 측정 가능하게 배선 — 기존 기능 재사용

**확인**: [`retrieval/planned.py`](../src/agmem/retrieval/planned.py)의 `searcher_for`와 query strategy들은 존재한다.
하지만 현재 [`AgmemMemory.query`](../src/agmem/bench/lme_v2.py)(369행 이후)의 vector 분기는 `self.mem.search`를 직접 호출한다.
따라서 TOML 설정만 켜면 LME-V2까지 같은 정책이 적용된다고 가정하면 안 된다.
**제안**: 정책 사용 여부를 명시한 별도 아암과 metrics 배선을 만들고 기존 raw+vector 기준선은 고정한다.
수용 조건: StubLLM으로 split-query 호출·결과 결합·LLM call/token metrics를 검증하고, off일 때 기존 단일 검색 결과를 유지한다.
정책 배선은 무료 테스트, 새 정책의 정확도·지연 검증은 별도 모델 호출이다.

### I6. 결과 manifest·범주 집계·누적량 sweep — 연구 인프라

**확인**: [`lme_v2_summary.py`](../scripts/repro/lme_v2_summary.py)는 전체 아암 집계·대응 부호검정을 이미 제공한다.
`AgmemMemory`는 save/load를 지원한다. 새 분석기를 통째로 만들 이유가 없다.
**제안**: 범주별 분모·누락 ID 검증, frozen 설정/코드/데이터 checksum, nested haystack 계획을 기존 스크립트 옆에 추가한다.
기존 summary는 ID 교집합으로 비교하므로, 부분 실패 아암을 정상 240문항 비교로 오인하지 않는 사전 검사가 필요하다.
수용 조건: 중복·누락 문항을 감지하고, 고정 근거는 모든 누적 단계에 존재하며, 같은 seed의 계획이 재현된다.
로컬 역할·출처·레이턴시 sweep은 LLM 없는 설정으로 돌리고, paid run 명령 생성과 실행을 분리한다.
큰 로컬 산출물을 저장소에 일괄 추가하는 대신, 공유할 집계 manifest와 원본 보관 경로·checksum을 정한다.

### I7. 실제 지출 회계와 예산 제어 — 유료 실험 전

**확인**: [`estimate/run`](../src/agmem/bench/lme_v2.py)(662–735, 777행 이후)의 `--max-usd`는 **시작 전 추정치 검사**다.
judge는 견적 합계에서 빠지고, structured-output 재시도나 실제 누적 지출을 이 gate가 차단하지 않는다.
**제안**: reader·explorer·distill·judge별 사용량, retry, cache 가격을 구분하고 run 단위로 합산한다.
실행 중 상한을 만들려면 호출 전 최대 비용 예약과 완료 후 정산, 진행 중 병렬 요청까지 포함한 예산 계약이 필요하다.
수용 조건: 가짜 provider에서 judge·retry까지 합산되고, 예산이 부족한 다음 호출을 보내지 않으며 checkpoint 재개가 이중 집계하지 않는다.
이미 시작한 외부 호출을 소급 취소해 과금 0으로 만들 수 있다고 약속하지 않는다. 회계 코드·가짜 provider 검증은 무료다.

### I8. 이미지 아암 — 후순위의 독립 가설

**확인**: `AgmemMemory.query(query, query_image)`는 이미지 입력을 사용하지 않고 text item만 반환한다.
**제안**: 텍스트 아암을 유지한 채 필요한 screenshot을 선택하는 별도 아암을 만든다.
공식 [backend 계약](https://github.com/xiaowu0162/LongMemEval-V2#implementing-your-method)은 text/image item을 허용한다.
수용 조건: 존재하는 이미지 경로·예산 적용·reader 전달을 작은 fixture로 확인한다. 모든 스크린샷을 무조건 주입하지 않는다.
시각 정보의 효용 측정은 모델 호출·이미지 데이터 확보가 필요한 별도 실험이며, 정확도가 오른다고 미리 가정하지 않는다.

## 5. 연구 질문과 1차 근거

2026-09-08에 아래 논문의 arXiv 초록·버전 정보와 공식 benchmark README를 다시 확인했다.
논문 보고는 우리 코드의 효과 증명이 아니다. 아래 실험 설계는 그 근거에서 도출한 **agmem용 제안**이다.

| ID | 질문 | 1차 근거와 범위 | agmem에서 구별할 실험 |
|---|---|---|---|
| R1 | 경험은 사실 회상인가, 잘못된 전제의 감지인가? | [LongMemEval-V2](https://arxiv.org/abs/2605.12493v1), 2026-05-12. 다섯 능력과 evidence gathering, 정확도·지연 평가 | E0/E2의 일반·abstention 분해. 무검색·정답 근거·검색 문맥을 나누되 gold는 평가자만 접근 |
| R2 | 누적될수록 무엇을 기억하고 버려야 하나? | [MemoryAgentBench v4](https://arxiv.org/abs/2507.05257v4), 2026-06-28 개정. incremental retrieval·learning·understanding·forgetting | E5의 고정 근거+distractor와 시간순 업데이트. 저장량 효과와 정보 갱신 효과 분리 |
| R3 | 오래된 runbook이 새 정정을 이기는가? | [Memora](https://arxiv.org/abs/2604.20006v1), 2026-04-21, ACL 2026 Findings. 무효 기억 재사용을 벌점화하는 FAMA 제안 | 같은 과제의 이전 절차→정정 절차→질문 스트림. stale retrieval rate와 실제 잘못된 행동을 분리; invalidation on/off 대조 |
| R4 | runbook이 새 태스크·역할·모델로 전이되는가? | [AFTER](https://arxiv.org/abs/2606.23127v1), 2026-06-22. task/role/model transfer를 분리한 절차 메모리 평가 | 같은 과제 반복과 다른 과제 holdout을 구분. I3의 과제 입도·추상화 수준을 한 요인씩 바꾸고 테스트 기반 성공률 측정 |
| R5 | 코드 검색과 경험 기억의 기여를 구분했는가? | [Code Isn't Memory](https://arxiv.org/abs/2606.22417v1), 2026-06-21. 고정 모델·하네스의 구조적 코드 인덱스 제거 실험 | E8에서 코드 접근·인덱스는 고정하고 과거 경험 유무만 변경. 필요하면 인덱스 유무를 별도 교차 요인으로 추가 |
| R6 | 원문보다 runbook이 덜 쓰이는 원인은 검색인가, 지식 품질인가? | 위 LME-V2의 raw slice/notes/파일 탐색 비교와 로컬 `runbook 인용 1/240` 관찰 | 동일 후보·같은 예산에서 runbook을 제시한 실험과 자율 검색 실험을 분리. 노출·인용·실제 해결 개선을 각각 기록 |

R3의 deterministic invalidation 테스트가 통과해도 자연어 정정 이해가 입증되지는 않는다.
R4의 label 개선과 task-success 개선도 별개다. R5의 구조적 인덱스는 현재 저장소에서 사실을 찾는 도구이며,
다른 세션의 실패·정정을 기억하는 장치와 같은 것으로 취급하지 않는다. 이 논문은 원문 세션 메모리의 우열을 직접 증명하지 않는다.

### 네이티브 baseline의 의미를 둘로 나누기

LME-V2의 `codex` baseline은 benchmark의 파일 탐색 아암이다. 이를 실제 제품의 지속 메모리 on/off 실험과 동일시하지 않는다.
공식 [재현 안내](https://github.com/xiaowu0162/LongMemEval-V2#setup-environment)는 고정 reader로 Qwen3.5-9B를 사용한다.
따라서 논문과의 격차를 단순히 “우리 reader가 9B여서”라고 설명할 근거는 없다. controller·embedding·이미지·설정 차이를 분리해야 한다.
자체 baseline 실행 시 모델·바이너리·upstream revision을 고정하고, 제품 비교에서는 호스트 기본 메모리와 agmem의 과거 기록 접근 범위를 따로 기록한다.
이번 조사에서 호스트 최신 기능이나 네이티브 on/off 실험을 새로 검증한 것은 아니다.

## 6. 추천 순서와 중단 기준

1. **무료 근거 정리**: E0/E1/E2와 I6의 최소 집계부터. 기존 아암·문항·설정을 식별할 manifest와 실패 목록을 확보한다.
2. **제품 경계**: I1의 role/top-k와 I2의 namespace 가설을 로컬 fixture로 검증한다. 일반 원문 검색과 제품 주입의 계약을 섞지 않는다.
3. **작은 연구 질문 하나**: E4의 증류 효용 또는 E5의 누적 열화 중 하나를 주실험으로 정한다. I7을 먼저 준비하고 reader·judge·retry 비용을 포함한다.
4. **일반화**: enterprise(E6), 실제 코딩(E8), 절차 전이(R4)로 넓힌다. 이미지·더 큰 모델·여러 query strategy를 한꺼번에 바꾸지 않는다.

**제안하는 stop rule**: 로컬 기능은 선언한 경계·회귀 테스트와 실제 hook 경로가 맞으면 종료한다.
연구는 사전 지정한 문항 집합·반복 수·예산에 도달하면 성공·실패와 무관하게 집계한다. 유의성이 나올 때까지 반복하지 않는다.
여러 능력·아암을 탐색하면 다중 비교임을 보고하고, 다음 confirmatory run의 주지표를 먼저 정한다.
정확도·지연·비용 간 허용 교환과 실용적 최소 개선량은 실행 전에 정한다. 지금 데이터만으로 임의 임계치를 “검증된 기준”으로 만들지 않는다.

이번 요청은 조사와 문서 정합성 수정이다. 신규 라이브 실험·코드 기능 구현·커밋·푸시는 수행하지 않았다.

## 7. 이번 정정과 검증

| 정정 대상 | 조치 |
|---|---|
| 도그푸딩 미관찰/진행 vs 완료 | docs/23과 로컬 지도 모두 2026-09-06 완료로 정리 |
| “네 능력 미측정” | web small 측정 완료, 일반화·enterprise 미검증으로 분리 |
| `abs 24·35` | 전체 abstention 17/72→27/72로 정정 |
| “분산 ±4pp” | 관찰된 두 실행 차이 3.75pp, 분산·CI 미확정 |
| 비용 표·LAFS 해석 | 저장 사용량 기준 비용 정렬, judge/구축 제외와 도메인 비교 한계 명시 |
| 훅 넷·세션당 한 콜 | 모듈 다섯 개, 구간·재시도·탐색 호출을 구분 |
| 과거 계획의 열린 TODO | 시점 안내와 현재 정본 링크 추가, 과거 기록 자체는 보존 |

검증: 기존 집계 스크립트로 6개 아암의 240문항 집계·대응 검정을 재실행했다. 제품 Python 코드 변경은 없으므로 전체 테스트 스위트를 다시 돌린 결과는 주장하지 않는다.
로컬 지도는 Chromium으로 375/768/1280px에서 실제 렌더링과 JavaScript 오류·페이지 가로 넘침·완료 상태를 검사했다.
원본 실험 파일·개인 세션·스토어를 변경하거나 유료 채점을 재실행하지 않았다. 외부에 게시된 지도 아티팩트는 갱신하지 않았다.
