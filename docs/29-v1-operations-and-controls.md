# v1 기억 확인·교정·복구

이 문서는 로컬 제품 경로의 관리 방법이다. 기능 검증과 실제 코딩 효용 측정은 구분한다.
명령의 namespace와 data-dir는 실제 사용할 저장소로 지정해야 한다. 아래는 예시다.

## 기억 상태 확인과 사용자 제어

```bash
uv run --no-sync python -m agmem.manage --namespace main --data-dir /path/to/data status
uv run --no-sync python -m agmem.manage --namespace main --data-dir /path/to/data inspect --type runbooks --id MEMORY_ID
uv run --no-sync python -m agmem.manage --namespace main --data-dir /path/to/data disable --type runbooks --id MEMORY_ID --reason '더 이상 유효하지 않음'
uv run --no-sync python -m agmem.manage --namespace main --data-dir /path/to/data correct --type runbooks --id MEMORY_ID --content '현재 적용할 정확한 내용' --reason '사용자 교정'
uv run --no-sync python -m agmem.manage --namespace main --data-dir /path/to/data restore --type runbooks --id MEMORY_ID --reason '검토 후 복원'
```

`episodic`을 지정하면 원문에서 수집한 개별 기억에도 제어를 적용할 수 있다.
runbook을 교정했다고 그 출처인 모든 원문이나 다른 runbook까지 자동으로 바뀌지는 않는다.
inspect의 출처 ID로 관련 항목을 확인하고 필요한 항목을 명시적으로 교정·비활성화한다.
자동으로 연관 기억 전체를 수정하면 유효한 다른 사실까지 훼손할 수 있다.

- `disable`: 자동 주입에서 제외한다. 삭제하지 않으므로 inspect로 확인할 수 있다.
- `correct`: 현재 전달할 내용을 바꾸고 원본과 변경 사유를 보존한다. 사용자 교정은 모델의 사실 검증과 다르다.
- `restore`: 명시적으로 검토한 항목의 자동 주입을 복원한다. 이미 은퇴·삭제된 항목까지 무조건 활성화하지 않는다.
- harmful 피드백은 자동 주입 제외에 반영한다. helpful 카운터를 임의 검색 가중치로 바꾸지는 않는다.

MCP에서도 `inspect_memory(memory_type, memory_id, namespace)`와
`update_memory(memory_type, memory_id, action, content?, reason?, namespace)`를 제공한다.
`action`은 `disable`, `restore`, `correct`이며 `correct`에는 비어 있지 않은 내용이 필요하다.
이 도구는 명시적인 사용자 교정·관리 동작에 사용한다.

## 품질 정보를 읽는 방법

| 필드 | 뜻 | 뜻하지 않는 것 |
|---|---|---|
| `quality.citation.status` | 인용 단계가 실제 입력과 맞는지 | 기억 내용이 참이라는 증명 |
| `quality.source_coverage.model_visible_ratio` | 전체 단계 중 입력에 일부라도 나타난 단계 비율 | 각 단계 전문을 읽은 비율 |
| `rendered_char_ratio`, `has_clipped_steps` | 렌더링된 입력 길이와 단계 내부 잘림 정보 | 모델이 내용을 이해한 정도 |
| `cited_ratio` | 인용한 단계의 비율 | 읽은 범위나 유용성 점수 |
| `quality.fact_basis.outcome_basis` | 모델이 붙인 결과 라벨이라는 출처 | 독립적인 테스트 성공 판정 |

인용 누락·무효 상태는 표시하되 기존 세션 출처 연결은 남긴다.
따라서 `whole_session_fallback`은 정확한 문장별 인용이 아니다.
기존 항목에 품질 필드가 없으면 과거 데이터의 상태가 미상인 것이며, 소급해서 검증됐다고 표시하지 않는다.

## 증류와 작업 복구 상태

| 상태 | 처리 |
|---|---|
| `started` | 처리 중이거나 이전 실행이 중단된 상태. 같은 인스턴스의 진행 중 요청은 중복 실행하지 않음 |
| `completed` | organizer가 전체 처리를 마침. 유용성·정확성 보증은 아님 |
| `skipped` | 유효한 응답에서 남길 신호가 없다고 판단한 상태 |
| `partial` | 일부 결과는 있으나 처리되지 않은 부분이 있어 검토 필요. 이전 결과를 보존함 |
| `failed` | 처리 실패. 원문 버전을 완료로 간주하지 않고 재시도 허용 |

부분 결과와 과거 상태 없는 데이터의 강제 재증류는 명시적인 복구 동작이어야 한다.
Python API의 `add_session(trajectory, force=True)`는 모델 설정이 있으면 비용을 발생시킬 수 있다.
준비·상태 확인 명령은 이 강제 재증류를 자동으로 실행하지 않는다.

디스크 대기열의 `*.processing`은 처리 중 또는 재시도 대상이고, `*.bad`에는 해석 불가·원문 파일 없음 등의 사유가 남는다.
파일 경로를 보관하는 작업 큐이므로 원문 파일을 옮기거나 삭제하면 자동 복구할 수 없다.
접수와 처리 성공은 별개다. 큐가 보존되어도 모델 제공자 장애나 설정 누락 자체가 해결되는 것은 아니다.

POSIX 파일 잠금으로 같은 대기열의 소비를 직렬화한다. 프로세스 중단 후 재실행될 수 있는 at-least-once 처리이며,
외부 모델 호출과 로컬 저장을 하나의 트랜잭션으로 묶은 exactly-once 과금 보장은 아니다.

## 실행 상태 진단

```bash
uv run --no-sync python -m agmem.manage --namespace main doctor
uv run --no-sync python -m agmem.manage --namespace main doctor --daemon-url http://127.0.0.1:8765
uv run --no-sync python scripts/smoke_product_stack.py --hermetic --daemon
```

`doctor`는 데몬을 시작하거나 저장소를 생성하지 않는다. 응답이 없으면 `unavailable`, 소스·설정·저장 경로가 다르면
`mismatch`와 차이 항목을 반환하고 exit code 2로 끝난다. 일치하면 `ok`/exit 0이다.
소스와 설정 지문은 daemon 시작 시 고정한 값과 비교한다. 설정 내용이나 API key는 출력하지 않는다.
namespace가 기본값과 다르면 경고하되, 명시적 namespace 요청이 가능한 다중 공간 운영을 오류로 단정하지 않는다.

`status`는 보존·증류 queue의 `queued/processing/bad` 수와 최근 증류 상태·사유를 함께 보여준다.
교정 후 `control.index_refresh=pending`은 문서·키워드 내용은 바뀌었지만 오프라인 CLI가 벡터를 재생성하지 않았다는 뜻이다.
`--hermetic --daemon` smoke는 임시 저장소, FakeEmbedder와 로컬 모델 대역으로 실행하며 외부 API 호출을 하지 않는다.
