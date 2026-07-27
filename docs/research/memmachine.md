# MemMachine 조사 — 구현 착수 전 1차 대조 (2026-07-27)

> arXiv:2604.04853 / `github.com/MemMachine/MemMachine` (**Apache-2.0**, 3,341★,
> 최종 푸시 2026-07-26 — 조사 당일 기준 활성).
> 이 문서는 이 프로젝트의 작업 모드(**논문 원문 → official 코드 → 우리 구현**)의 첫 두 단계다.
> 당일 raw로 읽은 파일: `packages/server/src/memmachine_server/episodic_memory/`의
> `episodic_memory.py`·`short_term_memory/`·`long_term_memory/`·`event_memory/`
> (`segmenter/text_segmenter.py`, `deriver/text_deriver.py`)·`declarative_memory/`.

## 0. 왜 이 논문이 후보 1순위인가

| 기준 | MemMachine | RecMem | SAGE / Memory Worth / GRAVITY |
|---|---|---|---|
| official 코드 | **Apache-2.0, 활성** | 공개(라이선스 미확인) | **없음** |
| 3자 대조 가능 | ✔ | ✔ | ✘ (2자만) |
| 비교표에서의 자리 | **추출 축의 반대 극단** | 또 하나의 추출형 3-tier | policy 멤버 / read 기법 |
| 부수 소득 | **공식 LoCoMo·LongMemEval 하네스 동봉** | — | — |

G-Memory에서 겪은 무라이선스 문제(11차 §S2)가 없다는 점도 크다.

## 1. 확정된 사실

### 1.1 write 경로에 **메시지당 LLM 콜이 0회**다 (논문 주장, 코드로 확인)

논문의 *"stores entire conversational episodes and reduces lossy LLM-based extraction"*은
수사가 아니다. event-memory write 경로 전체에서 언어모델 호출이 없다:

- **분절 = 순수 기계적**. `TextSegmenter`는 langchain `RecursiveCharacterTextSplitter`
  (`chunk_size=500`, `chunk_overlap=0`)이고, separator 목록에 전각 물음표·표의문자 마침표·
  zero-width space까지 넣은 다국어 우선순위 리스트다. LLM 경계 판정이 **없다**.
- **파생 = 문장 추출 + 포맷팅**. `TextDeriver`는 `extract_sentences`를 쓰고, 임베딩 앵커를
  `[{timestamp}] {producer}: {json.dumps(text)}` 형태로 만든다. fact 추출이 **없다**.
- 우리가 읽은 7개 코어 파일 중 언어모델 참조가 있는 것은 `short_term_memory.py`
  **하나뿐**이고, 그것도 용도가 **비동기 STM 요약**이다(`llm_model`, `summary_prompt_system`,
  `summary_prompt_user`).

**이것이 이 논문의 값어치다.** A-Mem은 turn당 2콜을 써서 추출하고, MemMachine은 사실상
추출하지 않는다. 우리 비교표에서 `passthrough`가 하한, A-Mem이 상한인데 **그 사이에 실물이
없었다.** MemMachine이 그 자리다.

### 1.2 우리 기록 정정 — contextualized retrieval은 **대칭이 아니다**

`memory-component-taxonomy.md` §2.4와 `write-path-critics.md` §4.4는 이 확장을
*"주변 ±1~2 turn"*이라고 적었다. 코드는 비대칭이다:

```python
# event_memory.py:450-451
max_backward_segments = expand_context // 3
max_forward_segments  = expand_context - max_backward_segments
```

즉 예산의 **1/3만 뒤로, 2/3를 앞으로** 쓴다. 대화에서 답이 질문 뒤에 온다는 사전지식을
넣은 것이고, "±N"으로 옮기면 그 사전지식이 사라진다. 두 문서를 정정했다.

### 1.3 논문의 3-tier와 배포 코드의 계층이 **이름부터 다르다**

논문: short-term / long-term episodic / **profile**.
코드: `episodic_memory/{short_term_memory, long_term_memory, event_memory,
declarative_memory}` + **최상위 별도 `semantic_memory/`**(cluster_manager,
cluster_splitter, cluster_store, config_store).

"profile"이라는 패키지는 없고, 대신 **클러스터링 서브시스템**이 있다. G-Memory에서 본
논문↔코드 괴리와 같은 계열이므로, 구현 전에 **어느 계보를 재현하는지 먼저 못박아야 한다**
(Nemori·MemoryOS에서 프리셋으로 푼 그 문제).

### 1.4 부수 소득 — 공식 LoCoMo·LongMemEval 하네스가 동봉돼 있다

```
evaluation/episodic_memory/{locomo,longmemeval}_{ingest,search,evaluate}.py
evaluation/episodic_memory/{llm_judge,generate_scores}.py
```

우리 `bench/locomo.py`·`bench/longmemeval.py`는 6차(A2/A3/B5)와 8c5f3d6에서 **결함 5건**이
나온 층이다. **네 번째 독립 참조**가 생기는 셈이고, 이 리포는 두 벤치를 모두 커버한다.
조직자 구현과 무관하게 이것만으로도 대조 가치가 있다.

### 1.5 논문의 주장 중 우리에게 가장 아픈 것

초록: *"retrieval-stage optimizations (depth tuning, formatting, prompt design)
outperformed ingestion-stage improvements."*

이 프로젝트는 8차까지 거의 전적으로 write 경로를 팠고, read 경로는 7~10차에 와서야
채널 단위로 정리됐다. 이 주장이 맞다면 우리 비용-정확도 파레토의 해석이 달라진다.
**측정 없이는 확인할 수 없는 주장**이므로, 지금은 "구현해서 측정 대기열에 올려두는" 것까지가
가능한 범위다.

보고 수치: LoCoMo 0.9169(gpt-4.1-mini) / LongMemEval-S 93.0% / HotpotQA-hard 93.2% /
WikiMultiHop 92.6%, input 토큰 Mem0 대비 약 −80%.

> 인용 캐비앗은 유효하다(`write-path-critics.md` §4.4): "80% 절감"은 memory-only 경로
> 기준이고 retrieval 단계를 따로 떼지 않았다. agent mode는 8.57M이다.

## 2. 우리 계약에의 매핑 (구현 계획)

| MemMachine | 우리 자리 | 비고 |
|---|---|---|
| `TextSegmenter` (기계적 분절) | organizer `on_message` | LLM 없음 — 우리 organizer 중 처음 |
| `TextDeriver` (문장 단위 파생) | 파생 메모리 타입 ADD op | 새 타입 1종(segment/derivative) |
| STM 요약(비동기) | `on_message` 내 요약 콜 | 유일한 write LLM |
| profile / semantic cluster | 계보 결정 필요 (§1.3) | **먼저 확정할 것** |
| contextualized expansion | `retrieval/steps.py` ReadStep | 비대칭 1/3·2/3 (§1.2) |
| Retrieval Agent (3-way 라우팅) | `policies/` read-side 멤버 | taxonomy §2.4의 기존 판정 유지 |

**표면적 경고**: read 쪽이 무겁다. 확장 스텝은 우리 레지스트리에 그대로 들어가지만,
Retrieval Agent는 policy 계층의 **첫 read-side 멤버**라 계약을 새로 뽑아야 한다.
9~11차가 반복해서 보여준 것이 "새 배선이 기존 config·전역 상수와 만나는 접합부에서 결함이
난다"이므로, 구현 직후 9차식 접합부 점검을 붙인다.

## 3. 착수 순서 (제안)

1. §1.3 계보 확정 — 논문(3-tier) vs 배포 코드(4+1) 중 무엇을 재현하는지. 프리셋 표부터.
2. write 경로(분절+파생+STM 요약) organizer 구현 + 3자 대조.
3. contextualized expansion ReadStep(비대칭 1/3·2/3).
4. Retrieval Agent는 **분리**해서 별건 — policy read-side 계약이 선행이다.
5. 부수: 동봉된 공식 LoCoMo·LongMemEval 하네스를 우리 `bench/`와 대조(조직자와 독립).

## 4. 참고

- 논문: [arXiv:2604.04853](https://arxiv.org/abs/2604.04853)
- 코드: [github.com/MemMachine/MemMachine](https://github.com/MemMachine/MemMachine) (Apache-2.0)
- 기존 기록: `write-path-critics.md` §4.4(인용 캐비앗), `memory-component-taxonomy.md` §2.4
  (구조 판정 = mechanism, Retrieval Agent = policy)
