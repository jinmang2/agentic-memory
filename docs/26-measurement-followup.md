# 측정 준비 후속: 진단·반복 설계·저장본·비용 회계

2026-09-08 착수, 2026-09-09 최종 검증. [이전 준비 도구](25-measurement-preparation.md)의 후속 구현이다. 실제 측정은 실행하지 않는다.

## 무엇을 비교할 것인가

주질문은 **같은 읽기 방식에서 experience 저장본이 raw 저장본보다 정확도를 높이는가**다.
vector와 explorer는 별도 대조로 보고, 두 방식의 결과를 독립 표본처럼 합치지 않는다.

| 실험 | 고정할 것 | 반복 단위 | 주지표와 해석 |
|---|---|---|---|
| D0 기존 결과 진단 | 기존 문항·정답 판정·실행 결과 | 반복 없음 | step cap·빈 문맥·degraded와 오답의 겹침. 원인 규명 또는 새 성능 측정으로 해석하지 않음 |
| E4a 고정 저장본 | raw/experience 저장본, 문항, read 방식, reader/judge 설정 | 저장본을 재사용하는 검색+reader 실행 3회 | 각 read 방식의 experience−raw 정확도 차이. retrieval/controller와 reader 변동을 함께 포함 |
| E4b 독립 쓰기 | 원본 궤적·문항·쓰기 설정·읽기 설정 | 쓰기부터 새로 만드는 전체 실행 3회 | 쓰기 변동까지 포함한 차이. E4a와 반복을 합치지 않음 |
| E4c reader-only | 문항별 reader 입력 문맥·순서·프롬프트·이미지 참조 | 검색하지 않고 reader만 3회 | 고정된 문맥에서의 reader 변동. prompt snapshot replay가 연결되기 전에는 실행하지 않음 |

3회는 초기 반복 설계이며 검정력 분석으로 결정한 수가 아니다. 3회 평균만으로 일반화나 유의성을 주장하지 않는다.
반복을 늘릴 경우 결과를 보고 유리할 때 멈추지 않도록 실행 전 변경된 반복 수와 이유를 기록한다.
고정 저장본이라고 reader-only가 되지는 않는다. explorer의 재탐색은 별도 변동을 만든다.

## 사전에 고정하는 분석 계약

- 주지표: web/small 전체 240문항에서 `score_bool`의 평균. 실패하거나 기권한 문항을 분모에서 제외하지 않는다.
- 주대조: vector 내 experience−raw, explorer 내 experience−raw의 두 대조. 다른 비교는 탐색 분석이다.
- 보조 지표: abstention/non-abstention 정확도, 범주별 분모·정답·기권, 빈 문맥·degraded·step cap, 지연과 토큰의 관측 커버리지.
- 저장된 단일 실행의 대응 부호검정은 그 실행의 문항 차이를 설명한다. 반복 간 변동을 추정하는 검정으로 대체하지 않는다.
- 두 주대조를 유의성 판정에 사용할 경우 family-wise 0.05에서 Holm 보정을 사전에 적용한다. 현재 audit의 p값은 보정 전 탐색 출력이다.
- 효과 방향·크기와 각 반복 값을 먼저 보고한다. `240 × 반복 수`를 독립 문항 수로 취급하지 않는다.
- gold나 사람이 표시한 근거는 사후 분석에만 사용한다. reader/retrieval 입력과 저장본에 추가하지 않는다.

## 실행 조건의 통제

기존 네 아암의 `run_args.json`과 `runtime_inputs/memory_config.json`에서 확인한 기준은 다음과 같다.

- reader `qwen/qwen3.5-9b`, judge `gpt-5.2`; 이름만으로 provider 내부 revision을 고정했다고 보지 않는다.
- reader temperature 0.6, top_p 0.95, top_k 20, thinking 활성, max_completion_tokens 20000.
- 하네스 memory_context_max_tokens 200000, reader 동시 요청 4, prompt-build worker 1.
- agmem vector 검색 top_k 10(설정에 없을 때 adapter 기본값), budget_tokens 12000, explorer_budget_tokens 4000, max_steps 8.
- 하네스 top_k는 reader 샘플링 설정이다. agmem 검색 top_k와 혼동하지 않는다.
- shuffle_questions_seed는 기존 결과에서 null이다. 순서와 seed를 명시적으로 고정할 때는 새 설정 변경으로 기록한다.

query strategy는 manifest의 실험 선언이다. 현재 LME-V2 vector 구현은 직접 `mem.search`를 호출한다.
새 전략을 선언해도 실행 경로에 자동 적용되지 않는다. 전략별 runtime 배선과 StubLLM 테스트가 갖춰지기 전에는 그 아암을 실행하지 않는다.
기존 직접 검색과 explorer 재탐색의 기준 recipe에는 각각 실제 경로에 맞는 선언만 사용한다.

## 수용 기준

1. 저장 결과 진단은 엄격 audit를 통과한 문항에만 붙고, 관측이 없는 신호는 미확인이다. 신호 조합별 정답/오답 및 범주별 분모가 전체 문항 수와 일치한다.
2. 고정 저장본 recipe는 저장본 식별자와 파일 목록을 요구한다. 아암 간 공유가 명시되고 파일 변경은 verify에서 거부된다. prepare/verify는 DB를 열거나 쓰지 않는다.
3. 실행 설정과 query strategy가 manifest에 남는다. 같은 파일에 다른 이름만 붙인 것을 독립 쓰기 반복으로 보고하지 않는다.
4. 비용은 component별 한 번 합산한다. retry는 중복 가산하지 않는 별도 subtotal이다. 미확인 사용량/가격은 예산 허용 근거가 될 수 없다.
5. CLI는 오프라인 fixture에서 성공/실패/덮어쓰기 거부를 검증한다. 원본 결과·저장본은 수정하지 않는다.

## 코드 관리

진단, 준비 명세, 회계는 서로의 실행 상태를 변경하지 않는 모듈로 유지한다. 공통 JSON 경계를 재사용하며 새 의존성은 추가하지 않는다.
원본 데이터·DB·설정은 Git에 추가하지 않는다. 작은 공개 recipe와 fixture만 버전 관리하고 실제 로컬 해시·진단 보고서는 `.omx/artifacts/`에 보관한다.

## CLI와 파일

```bash
uv run --no-sync python -m agmem.bench.lme_v2_tools diagnose results/lme_v2/full --output /tmp/lme-diagnosis.json
uv run --no-sync python -m agmem.bench.lme_v2_tools costs experiments/lme_v2_followup/cost-ledger.fixture.json --output /tmp/lme-costs.json
uv run --no-sync python -m agmem.bench.lme_v2_tools prepare experiments/lme_v2_followup/fixed-store.recipe.json --output /tmp/lme-fixed-manifest.json
uv run --no-sync python -m agmem.bench.lme_v2_tools verify /tmp/lme-fixed-manifest.json
```

출력 파일은 새 경로를 지정해야 한다. 기존 파일은 덮어쓰지 않는다. `costs`는 보고서 생성 성공이면 종료 코드 0이며,
예산 판단은 JSON의 `decision.status`를 읽는다. 이 명령은 모델 호출을 승인하거나 실행하지 않는다.
입력 오류는 종료 코드 2, `verify` 무결성 실패도 2다.

[예제 디렉터리](../experiments/lme_v2_followup/README.md)에 분석 계약, 실행 설정, 고정 저장본/독립 쓰기 recipe, reader-only 설계와 합성 비용 fixture를 둔다.
recipe의 데이터·저장본·로컬 TOML 경로는 기존 체크아웃 기준이므로 다른 환경에서는 실제 경로로 조정한다.
`source_files`의 짧은 예제 목록만으로 전체 구현을 고정했다고 보지 않는다. 실제 준비에서는 organizer·retriever·provider adapter와 upstream 하네스까지 포함한 파일 목록을 사용한다.

## 비용 기록 위치와 연결 계약

현재 비용 API는 합성 event를 받아 검증하는 독립 기반이다. 다음 위치에 연결될 계약을 분리해 둔다.

| 기록 위치 | component | 연결 시 계약 |
|---|---|---|
| reader 요청·응답 | reader | 요청 시 최대 과금 예약, 응답/오류에서 실제 과금 정산 |
| explorer/controller·query planning | retrieval | reader와 구분한 호출 ID, 각 실제 시도의 비용 기록 |
| organizer/distill | write | 독립 build ID와 요청 ID, 저장본 재사용과 새 쓰기를 구분 |
| embedding | embedding | build/query 목적을 식별하고 batch의 실제 사용량 기록 |
| evaluator | judge | reader와 별도 event ID, judge 재시도도 포함 |
| transport/structured-output retry | 원래 component | 새로운 event ID와 base_event_id; retry subtotal은 총액에 다시 더하지 않음 |

금액 단위는 정수 microUSD(1 USD = 1,000,000 microUSD)다. 현재 입력은 가격표를 자동 조회하거나 토큰 수에서 금액을 추정하지 않는다.
합성 fixture의 숫자는 provider 가격이 아니다. 실제 연결에서는 provider 청구값 또는 버전이 고정된 가격 계산의 출처를 함께 기록해야 한다.
실패한 요청도 청구될 수 있으므로 오류를 무료로 처리하지 않는다. 가격/과금 여부를 알 수 없으면 미확인으로 유지한다.

예산은 완료된 비용과 진행 중 예약을 함께 고려한다. 예약 한도를 넘은 실제 정산은 숨기지 않고 초과로 보고한다.
이미 시작한 외부 호출의 과금을 소급 차단할 수는 없다. fixture 검증은 현재 하네스의 `--max-usd`를 실시간 지출 제한으로 바꾸지 않는다.

기존 recipe 견적의 6항목은 **서로 겹치지 않는 추정액**으로 채운다. reader/retrieval/write/embedding/judge 기본 추정액은 재시도를 제외하고 retries에 그 추정액을 한 번 넣는다.
반면 실제 event ledger는 재시도까지 해당 component에 귀속하고 retry subtotal을 별도 표시한다. 두 표현을 옮길 때 같은 재시도 금액을 두 번 합산하지 않는다.
명시된 실제 청구액이 있으면 토큰 수가 미확인이어도 금액 자체는 알려진 값이다. 토큰 커버리지와 비용 커버리지를 구분한다.

## 준비 스키마의 해석

recipe v1의 기존 fresh 입력은 계속 받으며 새 저장본·아암 설정을 선택적으로 추가한다. 생성 manifest는 v2다.
이전 manifest v1은 새 필드가 없으므로 새 recipe로 다시 준비해 비교한다. 기존 파일을 조용히 새 버전으로 덮어쓰지 않는다.

- `fixed_stores`: `id`, 선언된 `write`, snapshot `path`. 같은 저장본을 쓰는 아암은 같은 `fixed_store_id`를 참조한다.
- 아암의 `query_strategy`, `settings_paths`: 정책 식별자와 설정 파일 해시. 명세를 만들었다고 runtime 정책이 연결된 것은 아니다.
- fixed snapshot은 현재 agmem `main` namespace의 다섯 파일 레이아웃을 지원한다: `memory_config.json`, `agmem_state.json`, `agmem/main/{memory.db,vectors.db,graph.kuzu}`.
- 누락·빈 파일·심볼릭 링크·추가 파일은 거부한다. 이 제한은 다른 namespace나 다른 저장 엔진의 임의 디렉터리를 검증했다고 오인하지 않도록 하기 위한 지원 범위다.
- 해시는 읽은 바이트의 동일성을 증명한다. writer를 잠그거나 DB 일관성을 검사하지 않는다. 실행 준비에는 쓰기가 멈춘 일관된 snapshot이 필요하다.
- `write`는 recipe의 선언이며 builder 실행 이력의 독립 인증이 아니다. 저장본 provenance와 reader/judge의 실제 runtime 설정 일치는 별도로 확인한다.

## 기존 결과에서 확인한 진단

실제 저장 결과 6아암 × 240문항을 읽어 1,440개 문항-아암 진단을 생성했다. 다음 숫자는 새 실행 결과가 아니다.

| 아암 | 상한 소진 | 빈 문맥·degraded·상한 소진이 함께 관측된 문항 | 그중 오답 |
|---|---:|---:|---:|
| raw explorer 첫 실행 | 200/240 | 41 | 41 |
| raw explorer 재실행 | 200/240 | 41 | 41 |
| experience explorer | 200/240 | 44 | 44 |

저장된 steps에는 최종 답변 단계가 포함된다. `steps > max_steps`는 상한 이후 강제 답변,
`steps == max_steps`에서 degraded가 있으면 상한 소진 후 강제 답변 실패다.
반대로 같은 단계 수에서 degraded가 명시적 null이면 마지막 허용 단계에 정상 답변한 것이다.
기존 docs/23의 159건은 raw의 강제 최종 답변 성공만 센 값이며, 실패 41건을 포함한 상한 소진은 200건으로 정정했다.
이 해석은 현재 저장 메타데이터를 만든 Explorer 계약에 따른다. 다른 producer의 단계 수 정의에 그대로 적용하지 않는다.

세 vector 아암은 빈 문맥 0/240이지만 step/degraded 필드는 관측되지 않는다. 이를 step cap 없음·정상 탐색으로 해석하지 않는다.
정답 근거 포함 여부는 모든 문항에서 미확인이다. 문맥이 비어 있지 않거나 인용이 있다는 사실만으로 근거가 충분하다고 판정하지 않는다.
범주별 비교는 `by_arm_category`, 관측 분모는 `observed`, 서로 겹치지 않는 조합은 `exclusive`를 사용한다.
`by_category`는 여러 아암을 합친 기술 집계이므로 독립 문항 수로 사용하지 않는다.

합성 사용 예제 `cost-ledger.fixture.json`은 5개 component 합계 70 microUSD, 그중 재시도 10 microUSD를 보고한다.
`budget-reservation.fixture.json`은 reader만 대상으로 한 시뮬레이션이다. 한도 100에서 미정산 예약 60과 실제 정산 80이 함께 있으면
총 노출 140으로 판단해 `breached` 정산과 최종 `blocked`를 보고한다. 이 숫자는 모두 테스트 값이다.

## 검증 기록

- 준비·진단·회계·CLI 통합 테스트 89개 통과. 기존 adapter/stamp/hook 관련 테스트 110개 통과.
- 새 패키지·관련 테스트 28개 파일의 Ruff check/format, basedpyright 오류 0개, no-excuse 규칙 검사 통과.
- CLI 테스트는 네트워크 연결을 거부하는 subprocess에서 실행했다. 실제 진단 CLI는 기존 결과만 읽었다.
- 실제 Explorer에 StubLLM을 연결해 max_steps=2에서 정상 최종 답변은 steps2, 강제 최종 답변은 steps3, 강제 답변 실패는 steps2/degraded=max_steps임을 확인했다.
- 개발 중 집계 의미가 바뀐 중간 보고서는 최종 근거로 쓰지 않는다. 로컬 진단의 최종 근거는 `.omx/artifacts/measurement-followup-20260908/diagnosis-corrected.json`이다.

모듈 소유는 `diagnostics*`(진단·집계), `recipe*`/`store_inventory`/`prepare`/`manifest`(입력과 준비 명세),
`costs*`(파싱·회계·보고), `budget`(예약 상태 전이), `__main__`(CLI 출력)로 나눴다.
타입 우회나 새 의존성 없이 공통 JSON 경계를 재사용한다. recipe/prepare/budget은 200–250 pure LOC 구간이므로
다음 기능 확장에서 입력 스키마나 inventory 책임이 커지면 해당 책임을 분리한다. 이번 배치에서는 전체 adapter를 리팩터링하지 않았다.

- 독립 코드 리뷰 APPROVE: 중간 예약 거절을 최종 승인으로 오인하던 회계 오류를 회귀 테스트로 고쳤고, 재검토에서 남은 차단 사항은 없었다.

- 최종 고정 저장본·독립 쓰기 명세는 각각 240문항·100궤적·12개 예정 job과 코드/하네스 156개 파일을 포함하며, 두 verification 모두 `valid: true`다. 고정 저장본은 2개 × 5파일이다. 예정 job 디렉터리는 생성하지 않았다.
- 로컬 최종 파일: `.omx/artifacts/measurement-followup-20260908/{fixed-store,fresh-write}.manifest-final.json` 및 대응 `.verification-final.json`. 비용 6항목은 미확인으로 남겼다.
