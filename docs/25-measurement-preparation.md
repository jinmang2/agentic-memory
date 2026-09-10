# 측정 준비 도구와 후속 실행 계획

> 첫 준비 배치의 기록이다. 고정 저장본·query strategy 명세, 진단·비용 CLI를 포함한 후속 기능은 [docs/26](26-measurement-followup.md)을 따른다.

2026-09-08. [후속 조사](24-next-research-and-implementation.md)의 E0/I6을 우선 구현한다.
이번 배치는 결과 검증과 실험 입력 고정에 집중한다. 모델 호출이나 실제 벤치마크 실행은 포함하지 않는다.

## 작업 순서

| 단계 | 이번 산출물 | 완료 판정 |
|---|---|---|
| 결과 검증 | 엄격한 결과 로더, 범주별 분모와 대응 부호검정 | 중복·누락 ID, 분류 변경, 집계 불일치를 거부하는 fixture 테스트 |
| 실험 준비 | recipe → manifest, 파일 SHA256, 아암·반복·출력 위치 목록 | 동일 입력에서 동일 계획 생성, 입력 변경·누락·출력 충돌 검출 |
| 통합 | `prepare`, `verify`, `audit` CLI와 사용 절차 | 네트워크가 차단된 실제 Python 프로세스에서 정상·오류 경로 검증 |

제품 recall 정책, query strategy 배선, stage 품질 개선은 별도 변경으로 남긴다.
이미 900줄 가까운 `lme_v2.py`에 준비 기능을 더하지 않고 `agmem.bench.lme_v2_tools`에 분리한다.
기존 summary의 정상 입력과 기본 출력은 유지하며 새 검증은 명시적으로 호출한다.
새 라이브러리, 저장소 전체 스타일 변경, 기존 결과·스토어 변경은 도입하지 않는다.

## CLI

저장소의 core 개발 환경에서 실행한다. upstream 하네스나 모델을 실행하는 명령은 아니다.

```bash
uv run --no-sync python -m agmem.bench.lme_v2_tools --help
uv run --no-sync python -m agmem.bench.lme_v2_tools prepare recipe.json --output manifest.json
uv run --no-sync python -m agmem.bench.lme_v2_tools verify manifest.json
uv run --no-sync python -m agmem.bench.lme_v2_tools audit results/lme_v2/full --output audit.json
```

`--output`을 생략하면 JSON을 표준 출력으로 보낸다. 지정한 출력 파일이 이미 있으면 실패한다.
입력 오류와 무결성 실패는 종료 코드 2로 보고한다. CLI에는 실험을 시작하는 `run` 명령이 없다.
실제 결과 디렉터리는 로컬 산출물이며 새 clone에 없을 수 있다. CI는 합성 fixture만 사용한다.

## Recipe 작성

다음 예시는 저장소 루트의 `recipe.json`을 기준으로 한다. `data_root` 아래에는 공개 데이터의
`questions.jsonl`, `trajectories.jsonl`, `haystacks/lme_v2_small.json`이 있어야 한다.
경로는 recipe 파일의 위치를 기준으로 해석하며 `~`도 확장한다.

```json
{
  "schema_version": 1,
  "study": "distillation-preparation",
  "domain": "web",
  "tier": "small",
  "data_root": "~/.agmem/datasets/longmemeval-v2",
  "config_paths": ["docs/_internal/configs/agmem.lme-v2.toml"],
  "source_files": ["src/agmem/bench/lme_v2.py", "pyproject.toml", "uv.lock"],
  "reader": "qwen/qwen3.5-9b",
  "judge": "gpt-5.2",
  "arms": [
    {"name": "raw_vector", "write": "raw", "read": "vector"},
    {"name": "experience_vector", "write": "experience", "read": "vector"}
  ],
  "repeats": 3,
  "output_root": "results/lme_v2/planned-distillation",
  "costs": {
    "reader": null,
    "retrieval": null,
    "write": null,
    "embedding": null,
    "judge": null,
    "retries": null
  }
}
```

위 config 경로는 기존 로컬 설정이며 Git 무시 대상이다. 다른 체크아웃에서는 실제 사용할 설정 경로로 지정한다.
`source_files`는 자동 전체 수집이 아닌 명시적 목록이다. 위 세 파일만으로 전체 코드가 고정되는 것은 아니다.
비교에 영향을 주는 organizer·retriever·upstream 하네스 소스와 lockfile도 목록에 넣는다.
CLI override나 모델 파라미터를 별도 JSON으로 관리한다면 그 파일도 `config_paths`에 포함한다.
이 도구가 설정 파일을 실행하거나 reader/judge 선언과 실제 provider 설정의 일치까지 판정하는 것은 아니다.

반복은 현재 새 저장본을 만드는 `fresh` 계획만 다룬다. 같은 저장본의 reader 반복과 독립 증류 반복은 다른 실험이다.
고정 저장본 재사용은 저장본 ID·fingerprint·아암 간 공유 관계를 추가한 별도 변경에서 지원해야 한다.
manifest의 job 목록은 실행기가 아닌 검토용 목록이며 출력 디렉터리를 만들지 않는다.

`costs`는 **전체 recipe의 모든 아암·반복을 합한 사용자 입력 추정 USD**다. 단일 실행 가격을 자동으로 반복 수만큼 곱하지 않는다.
누락 키와 `null`은 미확인이고, 확인된 무료 항목만 `0`으로 적는다. 한 항목이라도 미확인이면 전체 추정액은 `null`이다.
숫자만 채웠다고 API의 실제 청구액이나 예산 차단이 검증된 것은 아니다.

## 해석 경계

- manifest는 준비 시점의 입력과 계획을 식별한다. 모델 제공자의 내부 버전이나 추론 결정성을 고정하지 않는다.
- 파일 fingerprint 일치는 입력 무결성 검사다. 비용 승인, 모델 품질 검증, 전체 실행 준비 완료를 의미하지 않는다.
- 비용의 미확인 항목은 `null`로 남긴다. 일부 알려진 비용의 합을 전체 비용으로 부르지 않는다.
- 기존 `lme_v2 run --max-usd`는 추정치 사전 검사다. judge·재시도까지 포함한 실시간 지출 차단이 아니다.
- 범주별 대응 검정은 같은 문항과 분류가 유지된 결과만 비교한다. 탐색적 다중 비교이며 새로운 독립 실행을 대체하지 않는다.
- `score_bool`에 따른 정답과 `is_unknown`에 따른 응답 기권은 별도 값이다. abstention 문제에서 둘이 함께 참일 수 있다.
- 데이터·설정 파일의 해시를 기록하되 질문 정답, 프롬프트, API 키를 manifest에 복사하지 않는다.

## 다음 변경 단위

1. **실패 원인 분석(E1/E2)**: 현재 검증된 결과를 입력으로 탐색 step cap·빈 문맥·degraded의 겹침과 근거 포함 여부를 분리한다.
2. **제품 recall(I1/I2)**: role/top-k와 namespace 가설을 로컬 fixture로 검증한다. 벤치마크 궤적에서 도구 턴을 지우는 정책과 혼동하지 않는다.
3. **측정 실행 기반(I5/I7)**: query strategy를 명시한 아암과 judge·retry의 실제 사용량 회계를 별도 구현한다.
4. **실험(E4/E5 이후)**: 저장본 고정 반복과 독립 증류 반복을 분리하고 주지표·반복 수·예산을 사전에 정한 뒤 실행한다.

새 요인을 한 번에 여러 개 바꾸지 않는다. 각 변경은 독립 테스트와 리뷰가 가능한 크기로 유지한다.

## 구현과 검증 기록 (2026-09-08)

- `src/agmem/bench/lme_v2_tools/`에 JSON 입력 경계, recipe/manifest, prepare/verify, 결과 파서/aggregate 검증, audit, CLI를 역할별로 분리했다. 기존 summary·adapter·hook·stamp는 수정하지 않았다. 새 의존성은 없다.
- 모든 새 JSON 입력에서 중첩 객체의 중복 키를 거부한다. ID·분류 불일치, 불완전한 결과, 비정상 수치, fingerprint 변경, 출력 덮어쓰기도 검증한다.
- 새 테스트 51개와 기존 관련 테스트 110개가 통과했다. 새 코드·테스트의 Ruff format/check, basedpyright, no-excuse 정적 검사도 통과했다.
- CLI 테스트는 네트워크 연결을 거부하는 subprocess에서 prepare/verify/audit와 실패 경로를 확인한다.
- 실제 로컬 입력으로 240문항·100궤적·4아암 × 3반복의 12개 fresh job 명세를 준비했다. 결과 디렉터리를 만들거나 측정을 실행하지 않았다.
- 기존 저장 결과 6아암 × 240문항을 엄격 감사하고 15개 대응 비교를 생성했다. 이는 기존 결과의 재검증이며 새 품질 측정이 아니다.
- 최종 `manifest-final.json`은 코드 141개 파일을 포함하며 `verification-final.json`에서 `valid: true`를 확인했다. `audit-final.json`과 로컬 recipe/manifest 증거는 Git 무시 대상 `.omx/artifacts/measurement-preparation-20260908/`에 보관한다. 비용 6항목은 미확인 상태로 유지한다.
- 독립 코드 리뷰: APPROVE, 지적 사항 0개. 리뷰어가 새 테스트 51개, compileall, LSP 오류 0개, 중복 키 실패 재현과 CLI help를 별도로 확인했다.
