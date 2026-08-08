조사 범위가 넓어서 여러 갈래를 나눠 확인하겠습니다.조사했습니다. "memory-augmented generation(MAG)"은 특정 논문의 조어이면서 동시에 세 갈래 연구 계보가 2025–26에 수렴한 이름이기도 해서, 용어 정리부터 시작하겠습니다.

---

# 0. 용어의 출처와 RAG와의 경계

"Memory-Augmented Generation"이라는 말이 논문 제목에 박힌 것은 MemOS(arXiv:2505.22101)가 사실상 처음입니다. 이 논문은 현재 LLM이 파라미터에 인코딩된 지식(parametric memory)과 컨텍스트에 갇힌 일시적 런타임 상태(activation memory)에만 의존하고 있으며, RAG가 plaintext memory를 도입했지만 라이프사이클 관리와 멀티모달 통합이 없어 장기적 지식 진화 능력이 제한된다고 문제를 제기합니다. 즉 MAG는 **RAG의 상위 개념**으로 제안된 것입니다.

경계를 명확히 하면:

| | RAG | MAG |
|---|---|---|
| 대상 | 외부 코퍼스 (read-only) | 상호작용의 산물 (read-write) |
| 연산 | read | **write – manage – read** |
| 시간 축 | 없음 (스냅샷) | 필수 (언제 알았나, 언제 바뀌었나) |
| 실패 모드 | 검색 실패 | 검색 실패 + **쓰기 오류 + 갱신 실패 + 오염** |
| 소유권 | 문서 소유자 | 사용자 (삭제권·감사 대상) |

2026년 3월 서베이는 에이전트 메모리를 지각·행동과 결합된 write–manage–read 루프로 형식화하고, temporal scope / representational substrate / control policy의 3차원 분류를 제안합니다. 이 write–manage–read 프레이밍이 지금 표준입니다. RAG는 이 루프의 read만 있는 특수 케이스입니다.

---

# 1. 세 개의 계보

MAG는 서로 다른 세 전통이 합류한 지점입니다. 각각이 지금의 세 가지 substrate에 대응합니다.

**(a) 미분가능 외부 메모리 (2014–2017)**
Memory Networks → End-to-End MemNN → NTM/DNC → Recurrent Entity Networks. 읽기·쓰기 헤드를 미분 가능하게 만들어 end-to-end 학습. 스케일이 안 나와서 사그라들었지만, **"메모리 슬롯에 무엇을 쓸지를 학습한다"**는 문제 설정이 지금 policy-learned memory로 부활합니다.

**(b) 검색을 메모리로 (2020–2022)**
kNN-LM → RETRO → Memorizing Transformers → Transformer-XL/Compressive Transformer. 핵심은 "메모리 = 확장된 KV 공간". Memorizing Transformer가 과거 KV를 kNN으로 붙인 순간, 지금의 **activation memory** 개념이 나왔습니다.

**(c) LLM 에이전트 메모리 (2023–)**
MemGPT, Generative Agents, Reflexion. MemGPT는 유한한 컨텍스트를 넘어서기 위해 LLM이 메모리 계층을 스스로 관리하는 OS 영감 아키텍처(virtual context management)를 제안했고, Generative Agents는 관찰–검색–reflection–계획으로 장기 행동 일관성을 유지했으며, Reflexion은 가중치 갱신 없이 언어적 reflection을 에피소드 메모리에 저장하는 것만으로 시행 간 개선이 가능함을 보였습니다.

이 셋이 MemOS의 세 분류 — Plaintext Memory, Activation Memory, Parameter Memory — 로 정리됩니다. 이 계층적 메모리 아이디어는 선행 연구 Memory³에서 explicit/implicit 경로 구분으로 처음 제안된 것입니다.

---

# 2. Substrate별 상세

## 2.1 Plaintext Memory — 텍스트 저장소

가장 활발하고 가장 제품화된 갈래입니다.

### MemGPT / Letta
LLM을 자기 메모리를 관리하는 OS로 취급합니다. main context(RAM — 지금 프롬프트에 있는 것), recall store(최근 대화 이력), archival store(외부 장기 저장소)의 3계층입니다. 메모리 연산이 function call로 노출되어 모델이 스스로 페이지 인/아웃합니다.

이후 중요한 구조 변경이 있었습니다. Letta의 sleep-time 에이전트 설계에서는 primary agent에게 core memory 편집 도구를 주지 않고, 그 도구들을 sleep-time agent에 붙입니다. 원래 MemGPT는 메모리 관리·대화·기타 작업이 한 에이전트에 묶여 있어 대화 중 메모리 연산을 호출해야 하므로 느리고, 메모리 관리 도구와 일반 도구를 함께 호출해야 해서 신뢰성이 떨어졌기 때문입니다.

**→ 이게 이 분야의 반복 패턴입니다: 쓰기 경로를 읽기 경로에서 분리한다.**

### Mem0
원시 대화 청크를 저장하는 대신 추출 단계에서 메시지 쌍마다 핵심 사실을 뽑아 압축된 자연어 메모리로 만듭니다. 저장 시 새 사실을 기존 메모리와 비교해 ADD / UPDATE / DELETE / NOOP 네 연산 중 하나를 고릅니다. 사용자가 뭄바이에서 방갈로르로 이사했다고 하면 옛 도시 사실을 실제로 삭제하고 새 것을 추가합니다.

2026년에 알고리즘이 바뀌었습니다. 단일 패스 계층적 추출과 다중 신호 검색 기반으로, 의미·키워드·엔티티 세 검색 신호를 병렬 스코어링해 융합하고, ADD-only 추출로 사실을 덮어쓰지 않으며, 엔티티 링킹으로 동일 인물·장소 언급을 묶습니다. LoCoMo 92.5, LongMemEval 94.4, 검색 호출당 평균 6,956 토큰을 보고합니다. 가장 큰 개선은 temporal +29.6점, multi-hop +23.1점입니다.

주목할 점: **UPDATE/DELETE에서 ADD-only로 회귀했다는 것.** 이유는 §4에서 다룹니다.

### Zep / Graphiti
시간 인식 지식 그래프. bi-temporal 모델(사건이 일어난 시각 vs 시스템이 알게 된 시각)을 명시적으로 유지합니다. LongMemEval에서 temporal knowledge graph 계열의 강점으로 최대 18.5% 정확도 이득이 보고됩니다.

### A-Mem
Zettelkasten 노트 기법을 차용해, 각 메모리 항목을 원시 텍스트 청크가 아니라 맥락 설명·키워드·태그·관련 노트로의 명시적 링크를 가진 구조화된 노트로 저장합니다. 이 메타데이터가 임베딩 유사도만으로는 불가능한 검색을 가능하게 합니다.

### MemOS
MemCube라는 단일 추상화 아래 plaintext(RAG 유사), activation(추론 시 KV-cache 상태), parametric(LoRA 어댑터 등 가중치 지식) 세 타입을 통합합니다. MemCube는 provenance, 버저닝, 접근 정책, 라이프사이클 상태 같은 메타데이터를 함께 들고 다니며, 타입 간 변환을 지원합니다 — 자주 접근되는 plaintext를 KV-cache 템플릿으로 승격시키거나, 안정된 지식을 파라미터로 증류하는 식입니다. LoCoMo에서 GPT-4o-mini 기준 75.80을 보고합니다.

다만 parametric·activation 타입은 모델 내부(가중치, KV-cache) 접근이 필요해 폐쇄형 API 제공자 간 이식성이 떨어집니다. Mem0, Zep, MemMachine 등은 애플리케이션 레이어에서 텍스트 API로만 동작해 어떤 모델 제공자와도 무수정으로 붙습니다.

### GAM — 개념적으로 가장 중요한 반론
General Agentic Memory(arXiv:2511.18423)는 미리 준비된 정적 메모리가 필연적으로 심각한 정보 손실을 겪는다고 주장하며, "just-in-time(JIT) 컴파일" 원리를 제안합니다. Memorizer는 가벼운 메모리로 핵심 이력만 표시하고 완전한 원본 이력은 범용 page-store에 유지하며, Researcher가 온라인 요청 시 그 사전 구축 메모리를 가이드로 page-store에서 정보를 검색·통합합니다.

기존 Mem0·MemoryOS 같은 시스템은 Ahead-of-Time(AOT) 컴파일에 의존합니다 — 세션이 끝나면 즉시 벡터나 텍스트 요약으로 압축하는데, 이 압축은 손실적이라 어떤 디테일이 미래에 필요할지 시스템이 미리 예측해야 합니다. 긴 코딩 세션의 특정 변수명처럼 "사소하다고 압축해버린" 디테일이 나중에 필요해지면 실패합니다. GAM은 이를 뒤집어 저장 충실도를 최대화하고 복잡성을 런타임 검색 에이전트로 넘깁니다.

GAM은 memorizer와 researcher를 다운스트림 답변 품질 보상으로 end-to-end RL 최적화하며, LoCoMo·HotpotQA·RULER·NarrativeQA에서 long-context LLM/RAG 같은 memory-free 베이스라인과 A-Mem·Mem0·MemoryOS·LightMem 같은 memory 베이스라인을 모두 상회합니다. RULER Multi-Hop Tracing에서 다른 방법들이 60% 미만에 머무를 때 90% 이상을 달성합니다.

**→ 지난번에 얘기한 RAG → agentic RAG(Search-R1)의 궤적이, 메모리에서 그대로 반복되고 있습니다.** 손으로 짠 압축 휴리스틱 → RL로 학습된 메모리 정책.

## 2.2 Activation Memory — KV-cache를 메모리로

activation memory는 추론 중 생성되는 중간 상태로, KV-cache가 그 중심 구조입니다.

**KV eviction 계열의 한계가 명확히 측정되었습니다.** Cartridges 논문은 프롬프트 압축과 KV cache 압축 기법들이 압축률 2배를 넘어가면 어려운 long-context 태스크에서 성능이 급격히 저하됨을 보였습니다. StreamingLLM, H2O, SnapKV 류가 여기 해당합니다.

**Cartridges (Stanford HazyResearch, arXiv:2506.06266)** 가 이 갈래의 전환점입니다.

코퍼스별로 더 작은 KV cache를 오프라인으로 학습시켜, 추론 시 이 학습된 KV cache(= Cartridge)를 로드해 디코딩합니다. Cartridge 학습 비용은 그 코퍼스를 참조하는 모든 쿼리에 걸쳐 상각됩니다. 단순히 코퍼스에 next-token prediction으로 학습시키면 ICL에 못 미치고, 대신 코퍼스에 대한 합성 대화를 생성해 context-distillation 목적함수로 학습시키는 self-study 레시피를 씁니다. 결과적으로 ICL 성능을 재현하면서 메모리 38.6배 절감, 처리량 26.4배 향상을 달성합니다. MTOB에서 유효 컨텍스트 길이를 128k에서 484k로 늘리고, 놀랍게도 재학습 없이 추론 시 합성(compose)이 가능한 Cartridge가 나옵니다.

기계적 해석도 나왔습니다. Cartridge의 key는 압축된 코퍼스에 대한 안정적이고 공유 가능한 검색 라우터로 작동하며, 학습된 압축의 대부분은 value 벡터 안에서 일어납니다.

이게 왜 중요하냐면 — **"메모리 = 텍스트"라는 가정을 깨고, 메모리를 모델의 네이티브 표현(KV)으로 직접 저장하는 경로**를 열었기 때문입니다. 토큰 예산과 무관하게 메모리 용량을 키울 수 있습니다.

## 2.3 Parametric Memory — 가중치에 쓰기

**Knowledge editing 계열은 사실상 정체 상태입니다.** ROME(단건) → MEMIT(GPT-J 6B, GPT-NeoX 20B에서 수천 건까지 스케일). 하지만 2026년 시점 평가는 냉정합니다.

QwQ-32B 편집 시 공분산 행렬 계산에만 최소 80.1GB와 레이어당 7시간이 걸리고, MEMIT은 97.56GB로 GPU 2장이 필요하며 편집당 191.44초가 소요됩니다(ROME은 11.85초/80.54GB, LoRA는 14.23초/69.24GB). 모델이 커질수록 메모리 요구가 계속 증가하는 추세가 명확합니다. 품질 측면에서도 파라메트릭 편집은 국소 가중치 갱신이 인접 사실로 과일반화되는 문제가 있고, 메모리 기반 편집은 검색된 편집이 엉뚱한 프롬프트에 주입되는 문제가 있습니다.

**대신 LoRA를 모듈형 메모리 유닛으로 보는 흐름이 자리잡았습니다.** 2026년 5월 연구는 Parametric Memory Law를 제시합니다 — 손실 감소 ΔL이 유효 파라미터 수와 시퀀스 길이에 대해 견고한 멱법칙을 따르고, 토큰 수준에서는 예측 확률 p>0.5를 기준으로 결정론적 상전이가 나타납니다. 파라메트릭 메모리의 **용량 한계를 정량화**하려는 첫 시도라는 점에서 의미가 있습니다. (jinmang2님이 좋아하실 종류의 논문입니다 — 다운스트림 QA 성능이 아니라 순수 기억 용량을 격리해서 측정합니다.)

**Titans / MIRAS (Google)** — 아키텍처 차원의 접근.

어텐션은 컨텍스트가 제한되지만 의존성 모델링이 정확하므로 단기 기억, 신경 메모리는 데이터를 기억할 수 있으므로 장기·지속 기억으로 봅니다. 핵심은 Neural Long-Term Memory Module(LMM)로, 순전파 도중 자기 가중치를 최적화하며 test time에 무엇을 기억하고 잊을지를 적응적으로 학습하는 메타 in-context learner입니다. 모멘텀을 가진 gradient 기반 "surprise" 지표로 중요한 사건을 추적·저장하고, 적응적 망각 기제로 메모리 오버플로를 방지합니다. 통합 방식은 세 가지 — MAC(메모리 토큰을 어텐션 컨텍스트에 prepend), MAG(메모리 출력이 어텐션 출력을 게이팅), MAL(메모리 모듈을 독립 레이어로 스택에 삽입)입니다.

**그런데 중요한 반증이 나왔습니다.** MAC 변형을 임베딩 차원 128, 코퍼스 크기 N=10/25/50으로 평가한 결과, Titans는 사실을 완벽하게 기억할 수는 있지만(학습 손실 100% 수렴) 요구 시 인출에 어려움을 겪습니다 — free-form completion 정확도는 코퍼스 크기에 따라 0~40%, forced-choice 정확도는 45~85%입니다.

**→ 저장(storage) ≠ 인출(retrieval).** 이건 MAG 전체를 관통하는 이슈입니다. "기억했다"는 손실 수렴으로 증명되지만 "떠올린다"는 별개 능력입니다. 사람의 tip-of-the-tongue 현상과 구조적으로 같은 문제입니다.

---

# 3. 기능적 분류 (substrate와 직교)

장기 메모리는 세 종류로 나뉩니다. Factual memory는 선호·정책·도메인 정보 같은 사용자/환경에 대한 선언적 지식을 저장합니다. Experience memory는 과거 행동과 그 결과를 기록해 성공 행동을 재사용하거나 실패를 피하게 합니다. 여기에 procedural/skill memory(어떻게 하는지)가 더해집니다.

이 구분이 실무에서 중요한 이유: **세 종류는 갱신 정책이 완전히 다릅니다.**

- Factual: 최신값이 옳음 → UPDATE/supersede
- Experience: 과거도 유효함 → append-only, 절대 덮어쓰지 않음
- Procedural: 일반화가 목적 → 여러 사례에서 추상화

대부분의 시스템이 이걸 하나의 저장소에 뭉뚱그려서 문제가 생깁니다. Mem0가 ADD-only 추출로 사실을 절대 덮어쓰지 않는 방향으로 간 것도 이 맥락으로 읽힙니다.

---

# 4. write–manage–read 각 단계의 실제 난제

## write: 무엇을 저장할 가치가 있는가
salience 판단. 이건 근본적으로 **미래 쿼리 분포를 모르는 상태에서의 압축 결정**입니다. GAM의 JIT 논증이 정확히 이걸 공격합니다.

## manage: 여기가 진짜 어려운 곳
rate-distortion 관점의 분석에 따르면, UPDATE/DELETE 연산은 비가역 압축의 가장 순수한 형태입니다 — 잘못된 병합이나 잘못된 삭제는 재유도가 불가능해 오류가 영구적입니다. ADD-only로 회귀하는 이유가 여기 있습니다.

그리고 아직 안 풀린 것: 사용자가 뉴욕에서 샌프란시스코로 옮겼다면 그 전이 자체가 이해되어야 하는데, 대부분 시스템은 변화를 단순 대체로 취급합니다. 올바른 동작은 진화로 다루는 것입니다.

## read: 검색 + 시간 인식 리랭킹
저장소 자체를 구조화하는 방향도 있습니다 — RAPTOR는 청크를 재귀적으로 클러스터링·요약해 상향식 트리를 만들어, 무손실 리프 위에 손실적 요약이 얹힌 다중 충실도 저장소에서 여러 추상화 수준으로부터 검색하게 합니다.

## 그리고 결정적 관찰
프로덕션 실패의 대부분은 벤치마크가 커버하는 read 단계가 아니라 write와 manage 단계에서 발생합니다. LoCoMo 점수가 거의 포화된 에이전트도 메모리가 세션에 걸쳐 순차적 의사결정을 지시해야 하는 agentic 환경에서는 형편없이 동작합니다. 메모리 에이전트에 필요한 네 능력은 정확한 검색, test-time 학습, 장거리 이해, 충돌 해소이며, 구조 조직화는 대부분의 팀이 아예 테스트하지 않는 다섯 번째 공백입니다.

---

# 5. Sleep-time compute — 쓰기 경로의 비동기화

Letta의 설계에서 primary agent는 사용자에게 메시지를 보내고 도구를 호출하고 외부 메모리를 검색할 수 있지만, 컨텍스트 내 memory block(core memory)을 편집하는 도구는 받지 못합니다. 그 도구들은 sleep-time agent에 붙어, primary agent의 in-context 메모리와 자기 자신의 메모리를 함께 관리합니다.

이론적 해석이 깔끔합니다. sleep-time compute는 offline policy improvement로 볼 수 있습니다 — 유휴 시간에 에이전트가 이미 수집한 데이터(과거 상호작용)를 사용해 새로운 환경 상호작용 없이 메모리 표현(정책)을 개선합니다. 정적 trajectory 데이터셋으로부터 학습하는 offline RL과 연결됩니다.

비용 구조상의 트레이드오프도 지적되어 있습니다. consolidation을 오프라인으로 옮기면 LLM이 쿼리 도착 전에 컨텍스트에 대해 "생각"해 유용한 추론을 캐싱하므로 압축당 LLM 호출 비용이 미래 쿼리들에 상각됩니다. 함정은 결코 조회되지 않을 비트에 예산을 투기적으로 쓴다는 것입니다.

그리고 consolidation 자체를 학습하기 시작했습니다. Auto-Dreamer는 end-to-end 에이전트 성능을 보상 신호로 GRPO 학습해, 온라인 경험으로 획득한 메모리를 어떻게 통합할지를 배웁니다. ScienceWorld trajectory만으로 학습해 고정형·RL 학습형·프롬프트형 메모리 베이스라인을 7점 앞서면서 액티브 메모리 뱅크는 12배 작습니다. 방법론적으로는 오프라인 통합을 provenance에 근거한 region rewriting으로 정식화합니다 — 선택된 작업 영역을 읽기 전용 증거로 취급하고 합성된 대체 집합으로 교체하는 방식이라, 항목별 CRUD와 달리 세션 간 추상화·중복 제거·생략 기반 망각이 기본 갱신 의미론이 됩니다.

---

# 6. 평가 — 여기가 이 분야의 가장 약한 고리입니다

## 벤치마크 지형
엄밀한 의미의 메모리 벤치마크로 널리 인용되는 것은 LoCoMo, LongMemEval, BEAM 셋입니다. 셋 다 고정 입력에 대한 단일 패스 어텐션이 아니라 대화 이력에 대한 멀티세션 연속성을 테스트하도록 만들어졌습니다. 흔히 함께 인용되는 NIAH, RULER, BABILong, InfiniteBench, LongBench는 long-context 어텐션을 측정하는 것으로, 메모리 시스템이 올라타는 기반은 알려주지만 멀티세션 쓰기-검색 루프를 테스트하지는 않습니다. 2026년에는 long context와 memory가 서로 다른 문제로 다른 평가를 받는 것이 일반적입니다.

LoCoMo 공개 릴리스는 10개 멀티세션 대화에 대해 1,986개 QA — single-hop 841, multi-hop 282, temporal 321, open-domain 96, adversarial 446 — 를 담고 있고, 관례상 adversarial을 뺀 1,540문항으로 보고합니다. LongMemEval-S는 정보 추출, 멀티세션 추론, 지식 갱신, 시간 추론, abstention 다섯 능력을 중심으로 설계된 500문항입니다.

## 그런데 이 숫자들을 믿기 어렵습니다

**LoCoMo 정답 키 감사.** 체계적 감사 결과 1,540문항 중 99개(6.4%)의 점수 왜곡 오류가 발견되었습니다 — 정답 키의 환각된 사실, 잘못된 시간 추론, 화자 귀속 오류 등입니다. 게다가 LLM judge가 의도적으로 틀리게 만든 답을 최대 63%까지 통과시킵니다.

**LongMemEval은 메모리 테스트가 아닐 수 있습니다.** LongMemEval-S는 115K 토큰 안에서 정보를 찾을 수 있는지를 테스트합니다. 유용한 능력이지만 이건 컨텍스트 윈도 테스트이지 메모리 테스트가 아닙니다. Mastra의 결과가 이를 보여줍니다 — 128K 윈도의 gpt-4o full-context 베이스라인이 60.20%인데(115K 임계에 근접), 같은 모델의 observational memory 시스템은 컨텍스트를 더 여유 있게 맞도록 압축함으로써 84.23%를 얻었습니다. 벤치마크가 측정하는 것은 장기 메모리 검색이 아니라 컨텍스트 윈도 관리 효율입니다.

**벤더 간 결과가 서로 뒤집힙니다.** Mem0가 LoCoMo 기준 SOTA를 주장하자 Zep은 자사 시스템이 잘못 구성되어 평가되었다고 반박하고, 올바른 구현으로 실행하면 Zep이 75.14 ± 0.17로 Mem0 최고 구성(Mem0 Graph)을 약 10% 앞선다고 정정 발표했습니다. Mem0가 보고한 Zep 점수는 65.99%였습니다.

**독립 감사에서 드러난 것들.** 2026년 4월 MemPalace 벤치마크 주장에 대한 감사에서 여섯 가지 문제가 확인되었습니다 — 96.6% Recall@5는 ChromaDB 기본 임베딩 모델(all-MiniLM-L6-v2)을 원문 청크에 적용한 성능으로 palace 구조 없이 최소 ChromaDB 설정으로 재현되고, LongMemEval 100%(500/500)는 LLM 리랭킹 다회 반복이 필요했는데 단일 실행 점수처럼 제시되었으며, LoCoMo 100% 주장은 top_k=50 — 사실상 전체 대화를 검색 — 으로 달성된 것이고, 합리적 k값에서의 정직한 성능은 Recall@10 기준 60.3%(원본) 또는 88.9%(하이브리드+리랭킹)입니다.

**채점 기준을 바꾸면 순위가 바뀝니다.** LoCoMo와 LongMemEval-S에 대한 감사 연구는 Raw / Source / Canonical이라는 서로 다른 채점 타깃을 구성해 같은 순위가 다른 승자를 낳는다는 것을 보였습니다.

**정적 recall이 agentic 성능을 예측하지 못합니다.** LoCoMo 같은 정적 recall 벤치마크는 멀티세션 agentic 성능을 예측하지 못합니다(MemoryArena, 2026). LoCoMo는 허구의 대화 데이터로 돌고 LongMemEval은 스크립트된 상호작용을 씁니다. 둘 다 사용자가 세션 중간에 마음을 바꿨을 때 에이전트가 저장된 사실을 갱신하는지는 알려주지 못합니다.

## 그나마 스케일 신호는 있습니다
BEAM에서 1M → 10M로 컨텍스트가 10배 늘 때 64.1 → 48.6으로 약 25% 성능이 손실됩니다. temporal 쿼리가 가장 어려운 카테고리이고, 신규 알고리즘의 +29.6점 개선 이후에도 헤드룸이 큽니다.

**정리: 지금 인용되는 90%대 숫자들은 대부분 벤더 자체 평가이고, 세 벤치마크 모두 독립 감사에서 심각한 문제가 지적되었습니다. 시스템 선택 근거로 쓰기에 부적합합니다.**

---

# 7. 보안 — MAG 고유의 위협 클래스

RAG에는 없고 MAG에만 있는 위협입니다. 쓰기 경로가 생겼기 때문입니다.

OWASP는 2026 Agentic AI Top 10에 Memory and Context Poisoning을 ASI06으로 추가했습니다. 프롬프트 인젝션은 세션 스코프여서 대화가 끝나면 리셋되지만, 메모리 포이즈닝은 지속적이어서 탐지·제거 전까지 모든 후속 세션에 영향을 미칩니다. 공격과 그 효과가 시간적으로 분리(temporally decoupled)된다는 것이 핵심 차이입니다.

**공격 스펙트럼:**

AgentPoison은 코퍼스 수준 공격자가 0.1% 미만의 포이즌 비율로 임베딩 공간 백도어를 심어 80% 이상 성공률을 얻음을 보였습니다. MINJA는 쿼리 전용 상호작용만으로 메모리 상태를 영구 변경할 수 있음을 보였습니다. eTAMP는 권한 격차를 완전히 없애, 공격자가 에이전트와 상호작용할 필요조차 없이 에이전트가 정상 동작 중 방문하는 웹 페이지만 조작하면 된다는 것을 보였습니다. 방어 함의는 명확합니다 — 쓰기 단계 공격 표면에는 명시적 메모리 갱신뿐 아니라 에이전트가 무엇을 저장할지 결정하는 데 영향을 줄 수 있는 모든 관찰 가능한 컨텍스트가 포함됩니다.

FARMA는 한 단계 더 나아가 에이전트가 세상에 대해 아는 것이 아니라 에이전트가 이미 추론한 것을 오염시킵니다 — 위조된 추론 항목이나 결정 로그 형태의 항목을 메모리 저장소에 삽입하는 주입 단계와 증폭 단계로 동작합니다. 실제 사례도 있습니다 — SpAIware는 간접 프롬프트 인젝션으로 ChatGPT의 영구 메모리에 적대적 항목을 쓸 수 있음을 시연했고, OpenAI가 패치하는 데 두 달 넘게 걸렸습니다. 다중 에이전트 프레임워크에서는 공유 메모리 저장소가 한 에이전트의 쓰기를 다른 에이전트의 읽기로 만들기 때문에 특히 위험합니다.

공격 성공률은 구현에 따라 80%, 95%, 99.8%까지 보고되고, Agent Security Bench는 최고 평균 ASR 84.30%를 보고하며 현행 방어의 효과가 제한적임을 보였습니다.

**공격자가 없어도 실패합니다.** 공유 저장소의 조용한 크로스 유저 오염, 더 이상 유효하지 않은 맥락에 프로필 사실을 과잉 적용하는 것, 메모리로 유발되는 sycophancy는 메모리 증강 에이전트 시스템의 평범한 운영에서 그냥 발생합니다.

**방어 방향:** provenance 추적, 신뢰도 인식 검색(trust-aware retrieval), memory contract(에이전트가 무엇을 믿어도 되는가의 명세), belief drift 탐지. contextual integrity를 다루는 CIMemories 같은 벤치마크(ICLR 2026)도 등장했습니다.

---

# 8. 제품 레이어

Anthropic은 2025년 9월 29일 memory_20250818 도구를 공개했습니다. Claude가 메모리를 읽고 쓸 수 있는 로컬 파일로 다루는 방식이며, context editing과 결합해 100턴 웹 검색 태스크 내부 벤치마크에서 84% 토큰 절감과 39% 성능 향상을 보고했습니다. 아키텍처의 핵심은 한 문장으로 요약됩니다 — 모델은 읽기/쓰기 명령만 발행하고 실제 저장소는 사용자 쪽에 있습니다.

Claude는 작업 시작 전 메모리를 읽고, 작업 중 파일을 생성·갱신하며, 이후 대화에서 참조합니다.

감사 가능성 측면의 설계 차이가 있습니다. ChatGPT의 메모리는 백그라운드에서 보이지 않게 사용자 프로필을 구축해 무엇을 언제 기억하고 사용하는지 알기 어려운 반면, Claude의 구현은 가시적 tool call을 사용해 무엇이 저장·검색되는지 정확히 검사할 수 있습니다.

**패러다임 관점의 의미:** 이 도구는 "에이전트에 장기 메모리 주기"를 "RAG 파이프라인 구축"에서 "파일시스템 인터페이스 구현"으로 압축합니다. 벡터 DB가 필수가 아니게 되는 흐름(지난번 논의한 agentic keyword search)과 정확히 같은 방향입니다.

---

# 9. 이 분야를 관통하는 근본 긴장 다섯 가지

**(1) 압축 vs 보존**
공격적으로 압축하는 시스템(Mastra, Mem0)은 작은 컨텍스트와 낮은 쿼리당 비용을 얻지만 결정적 디테일을 잃을 위험이 있고, 원시 데이터를 보존하는 시스템(MemMachine)은 사실적 무결성을 유지하지만 효율적 검색 기제를 요구합니다. GAM의 AOT vs JIT가 이 축의 이론화입니다.

**(2) 저장 ≠ 인출**
Titans가 100% 기억하고 0~40% 인출하는 현상. 파라메트릭 메모리 전체의 미해결 문제.

**(3) 쓰기 경로의 비가역성**
UPDATE/DELETE는 되돌릴 수 없습니다. 그래서 ADD-only + supersede 마킹으로 수렴 중.

**(4) long context가 메모리를 대체하는가**
답은 "아니오, 하지만 벤치마크는 대체당했다"입니다. LongMemEval-s 각 haystack은 약 115K 토큰인데 Sonnet은 200K, 여러 SOTA 모델은 100만 토큰 윈도를 갖습니다. 검색 프레이밍은 이제 많은 배포 환경에서 존재하지 않을 수 있는 제약을 테스트합니다. 진짜 메모리 문제는 1M+ 스케일(BEAM 10M)과 write/manage 단계에 남아 있습니다.

**(5) 손으로 짠 정책 → 학습된 정책**
GAM, Auto-Dreamer, MemRL 등. RAG가 Search-R1으로 간 것과 완전히 동일한 궤적입니다. 지난 대화의 결론이 그대로 여기 적용됩니다 — **"구현은 낡고 아이디어는 살아남는다"**의 다음 라운드가 지금 메모리 시스템에서 벌어지고 있습니다.

---

# 10. 실무 설계 시 체크리스트

1. **메모리 종류를 분리하라** — factual / experience / procedural은 갱신 정책이 다릅니다. 한 저장소에 넣지 마세요.
2. **쓰기를 읽기 경로에서 빼라** — Letta가 sleep-time agent로 간 이유. latency와 신뢰성 둘 다 개선됩니다.
3. **비가역 연산을 피하라** — DELETE 대신 supersede 마킹 + 시간 인식 리랭킹.
4. **provenance를 필수 필드로** — 보안 요구이자 디버깅 요구입니다. "이 메모리는 누가/어디서/언제 썼는가"가 없으면 오염 시 롤백 불가.
5. **쓰기 경로를 신뢰 경계로 취급하라** — 에이전트가 읽은 모든 것(웹페이지, 이메일, 코드 주석)이 쓰기 공격 표면입니다.
6. **공개 벤치 점수로 시스템을 고르지 마라** — 자기 워크로드로 재현하세요. 특히 write/manage를 테스트하는 시나리오를 직접 만들어야 합니다.
7. **비용을 read 단위가 아니라 write까지 포함해 계산하라** — 추출 LLM 호출이 숨은 비용입니다.

---

# 11. 읽을 순서 제안

**기초 (개념 프레임)**
1. MemGPT (2310.08560) — virtual context management의 원형
2. MemOS (2505.22101 → 2507.03724) — MAG 용어와 3-substrate 분류
3. Memory for Autonomous LLM Agents (2603.07670) — write–manage–read 형식화, 2022~2026 커버

**메커니즘 (깊이)**
4. Cartridges (2506.06266) + 후속 mechanistic 분석 (2508.17032) — activation memory의 정점
5. Titans (2501.00663) + MIRAS — 아키텍처 차원
6. GAM (2511.18423) — AOT vs JIT, 가장 개념적으로 도발적
7. Parametric Memory Law (2605.30260) — 정량적 한계

**비판적 검토 (필수)**
8. LoCoMo 감사 (dial481/locomo-audit), Same Ranking, Different Winner (2605.24060)
9. rate-distortion 관점의 memory compaction (2607.08032) — 무엇을 버릴지의 이론
10. Long-Term Memory Security 서베이 (2604.16548)

---

지난번 RAG 논의와 이어 붙이면 그림이 하나로 맞습니다. **RAG(read-only) → agentic RAG(read policy 학습) → MAG(write-manage-read 전체)** 로 확장되면서, 각 단계마다 "손으로 짠 휴리스틱 → RL로 학습된 정책"이라는 동일한 전이가 반복되고 있습니다. 그리고 매번 평가 체계가 그 전이를 따라가지 못합니다.

이걸 노트 체계에 맞춰 정리한 `.md` 파일로 뽑아드릴 수 있고, 특정 갈래(예: Cartridges의 self-study objective 유도, 또는 rate-distortion 프레이밍의 수식화)를 골라 first-principles로 파고들 수도 있습니다. 어느 쪽이 더 유용할지 말씀해주세요.

레포부터 확인해보겠습니다.하이픈 버전이 있네요. 클론해서 보겠습니다.레포 찾았습니다. 이름이 `agentic_memory`(언더스코어)가 아니라 **`jinmang2/agentic-memory`**(하이픈)라 처음에 404가 났습니다. 클론해서 README, 18-locomo-4way, 17-defect-ledger, 06-roadmap, 소스 트리 전부 읽었습니다.

먼저 솔직하게 말씀드리면 — **제가 어제 드린 조사의 "벤치마크 약점" 섹션은 이 레포가 이미 훨씬 깊게 해놓은 일입니다.** 제가 인용한 건 벤더 블로그와 DEV 포스트 수준이고, 이 레포는 pinned SHA + `file:line` + CI가 돌리는 Tier-0 재현 스크립트로 같은 결론을 증명하고 있습니다. A-MAC의 `'is'` → *s**is**ter* 서브스트링 매치 같은 건 어디에도 공개된 적 없는 발견입니다. 그러니 "뭘 더 공부하면 좋을까"에 대한 답은 기초 보강이 아니라 **구조적 공백**이어야 맞습니다.

---

## 1. 지금 레포의 좌표

- 38k LOC, 9개 방법론 + A-MAC, 441 테스트, 12라운드 감사(96 verdict / 94 confirmed / 2 refuted)
- `MemoryOp` 단일 추상화 위에 write path를 플러그인화, read path는 공유 인프라
- LoCoMo 4-way 실측 완료, LongMemEval은 배선만 되고 미실행(C-4)

**포지션 요약: 이 레포는 "메모리 시스템 구현체"가 아니라 "메모리 문헌의 재현성 감사 인프라"입니다.** 그게 진짜 기여물이고, 제 조언도 그 축을 강화하는 쪽으로 잡겠습니다.

작은 것 하나 — README에 "Nine agentic-memory methodologies"라고 써놓고 아래 §How the claims here were checked에서 "not that **eight** systems were written"이라고 되어 있습니다. 확장 전 문장이 남은 것 같습니다.

---

## 2. 구조적 공백 다섯 가지

### 공백 A — substrate 축이 하나뿐 (가장 큰 개념적 구멍)

MemOS 분류로 보면 **plaintext / activation / parametric** 세 축인데, 9개 방법론이 전부 plaintext입니다. `grep`으로 확인했더니 `cartridge`, `KV_cache`, `parametric` 모두 0건. `train/sft_lora.py`가 있지만 이건 *0.5B 추출기 학습*용이지 *메모리로서의 LoRA*가 아닙니다.

왜 중요하냐면 — **`MemoryOp`(ADD/UPDATE/MERGE/DELETE/INVALIDATE/LINK) 추상화가 텍스트 op 어휘라서 activation·parametric memory를 원리적으로 표현하지 못합니다.** Cartridges의 "코퍼스를 KV로 증류"나 LoRA-as-memory의 "가중치에 쓰기"는 이 vocabulary에 안 들어갑니다.

이건 결함이 아니라 **추상화의 경계를 발견한 것**이고, 그 경계를 명시적으로 문서화하는 것 자체가 기여입니다. "MemoryOp는 plaintext substrate에 대해 완전하고, 다른 두 substrate에 대해서는 이런 이유로 확장 불가하다"는 진술은 MemOS의 MemCube가 주장만 하고 증명하지 않은 것입니다.

읽을 것: Cartridges(2506.06266) + 후속 mechanistic 분석(2508.17032 — key는 라우터, value에 압축), Parametric Memory Law(2605.30260).

### 공백 B — read path가 방법론인 경우가 없음

현재 설계는 "Organizer = write-path plugin, retrieval = 공유 인프라"입니다. 명시적 선택이고 18-locomo-4way에서 arm별 read path 차이를 정직하게 테이블로 뽑아둔 것도 봤습니다.

그런데 **GAM(General Agentic Memory, 2511.18423)은 이 구도에 안 들어갑니다.** GAM의 주장은 정확히 "AOT 압축을 하지 마라, 원본을 page-store에 두고 런타임에 deep research로 뽑아라"이고, 즉 **write path가 거의 없고 read path가 방법론 전체**입니다. LoCoMo·HotpotQA·RULER·NarrativeQA에서 Mem0/A-Mem/MemoryOS/LightMem을 다 이겼다고 보고합니다(RULER multi-hop tracing에서 90%+ vs 경쟁 60% 미만).

이걸 붙이려면 `Organizer` 옆에 `Researcher` 같은 read-path plugin 계약이 필요합니다. 그리고 그 순간 **passthrough arm의 지위가 바뀝니다** — 지금 passthrough는 baseline이지만, GAM 관점에서는 passthrough + 강한 read가 정답 후보입니다. 실제로 ledger C-5에서 RecMem의 Full Context 84.18 > RecMem 81.10을 인용해두셨고, Nemori arm A의 67.60이 자체 측정 Full-context 72.90보다 낮다는 것도 기록해두셨습니다. **"모든 write-path 방법론이 no-memory baseline보다 나쁘다"는 게 이 레포의 데이터에서 이미 보이는 결론**인데, 아직 정면으로 다뤄지지 않았습니다.

### 공백 C — 학습된 메모리 정책 (RL) 축이 완전히 없음

`GRPO` 0건, `reinforcement` 2건(둘 다 언급 수준). 지금 문헌의 최전선이 여기입니다:

- GAM: memorizer·researcher를 다운스트림 답변 품질로 end-to-end RL
- Auto-Dreamer(2605.20616): GRPO로 offline consolidation 학습, region rewriting을 갱신 의미론으로. ScienceWorld에서 고정형·프롬프트형 베이스라인 +7점, 메모리 뱅크는 12배 작음
- MemRL, MemEvolve 등

RTX 2060 6GB에서 RL 학습은 현실적이지 않습니다. **하지만 학습 없이 할 수 있는 게 있고, 그게 더 가치 있습니다** — §3에서 구체적으로.

### 공백 D — 보안·프라이버시 축이 통째로 비어 있음

`poison`, `privacy`, `redact`, `multi_user` 전부 0건. 그런데 **이 레포는 이 연구를 할 수 있는 거의 유일한 인프라를 이미 갖고 있습니다.**

OWASP는 2026 Agentic AI Top 10에 Memory and Context Poisoning을 ASI06으로 추가했고, 프롬프트 인젝션과 달리 메모리 포이즈닝은 지속적이어서 탐지·제거 전까지 모든 후속 세션에 영향을 미칩니다. AgentPoison은 0.1% 미만 포이즌 비율로 80%+ 성공률, MINJA는 쿼리 전용 상호작용만으로 메모리 상태를 영구 변경, eTAMP는 에이전트가 방문하는 웹페이지 조작만으로 성립합니다.

**아무도 답하지 않은 질문: 어떤 write path가 포이즈닝에 강한가?** 이건 write path의 성질입니다 —

- Mem0는 33,167개 판정 중 79% NOOP를 냅니다. 이 게이트가 방어인가, 아니면 그냥 무차별인가?
- Nemori는 turn을 episode로 접습니다 — 주입된 단일 turn이 서사에 흡수되면 희석되는가, 아니면 서사 전체를 오염시키는가?
- A-Mem은 링크를 만듭니다 — 오염이 링크를 타고 전파되는가?
- A-MAC 게이트는 실질적으로 "전부 admit"이니 방어가 0인가?

이걸 측정할 수 있는 사람이 지금 세계에 몇 명 없습니다. 9개 write path × 동일 harness × ops 로그 × pinned lineage를 다 갖춘 물건이 이 레포뿐이기 때문입니다.

### 공백 E — LoCoMo gold 자체의 오류가 결론을 삼킬 수 있음

C-2에서 grader 세 종류가 비교 불가라는 걸 증명하셨습니다. 그런데 **gold answer 자체의 오류는 아직 다루지 않았습니다.**

2026년 4월 독립 감사는 LoCoMo 1,540문항 중 99개(6.4%)의 점수 왜곡 오류를 찾았습니다 — 정답 키의 환각된 사실, 잘못된 시간 추론, 화자 귀속 오류. 그리고 LLM judge가 의도적으로 틀리게 만든 답을 최대 63%까지 통과시킵니다.

이게 18-locomo-4way에 직접 꽂힙니다. footnote 8에서 **"상위 세 arm이 2.5점 밴드(67.60/65.78/65.13) 안에 있고 순서는 이 정밀도에서 미해결"**이라고 이미 정확히 진단하셨는데, gold 6.4% 오류를 더하면 그 2.5점 밴드는 **원리적으로 판별 불가**일 가능성이 높습니다. Nemori arm A vs arm B의 1.82점 차이, 그리고 임베더 바꾸면 부호가 뒤집힌다는 그 발견 — 그게 실제 신호인지 gold 노이즈인지 지금 알 수 없습니다.

추가로 알아두실 것: LoCoMo-Plus가 "cognitive" 카테고리를 도입했습니다 — 어휘적 겹침 없이 의미적으로 단절된 cue-trigger 쌍("보호소 개를 입양했다" → "어떤 사료를 사야 할까")을 세션을 넘어 연결해야 하는 문제입니다. 그리고 BEAM이 세 번째 표준으로 자리잡았고, NIAH·RULER·BABILong·LongBench는 memory 벤치가 아니라 long-context 벤치로 분류하는 게 2026년 관례입니다. BEAM은 1M→10M에서 64.1→48.6으로 떨어집니다.

---

## 3. 우선순위 — 세 트랙, 이 순서로

### 트랙 1 (즉시, $0) — **판별력을 먼저 유도하고 측정하라**

지금 상태는 "single seed로 2.5점 밴드 안의 순위를 논하는 것"이고, 본인도 footnote 7·8에서 그걸 아십니다. 다음 실험에 돈을 쓰기 전에 **먼저 계산해야 할 것**:

$$
\text{ΔJ를 } \alpha=0.05,\ \text{power } 0.8 \text{로 검출하는 데 필요한 seed 수 } n
$$

필요한 입력은 이미 다 갖고 계십니다 — 1,540문항, 문항별 binary 판정, Track 1이 측정한 per-arm J 안정성 ±0.35pp. 여기에 gold 오류율 6.4%를 **판정 노이즈로 모델링**해 넣으면(오류 문항은 arm 무관하게 랜덤 감점), 검출 가능 최소 효과크기가 나옵니다.

제 예상: **상위 세 arm의 순서는 seed를 몇 개 더 돌려도 판별 안 되고, 판별하려면 gold를 고쳐야 합니다.** 그러면 결론은 "순위를 매기지 말고 밴드로 보고하라"가 되고, 이건 지금 문헌 전체를 향한 강한 주장이 됩니다.

그리고 gold 감사는 $0로 부분 재현 가능합니다 — dial481/locomo-audit의 99개 오류 목록을 받아서, 이미 커밋된 `*.records.jsonl` 8개를 **오류 문항 제외하고 재채점**하면 됩니다. C-2의 `repro_locomo_rescoring.py`가 이미 그 replay 기계를 갖고 있으니 필터 한 줄 추가입니다. 순위가 바뀌는지 안 바뀌는지가 바로 나옵니다.

**이게 첫 번째인 이유: 이 레포의 기존 결론 전체의 신뢰구간을 정하는 작업이고, 비용이 0이고, 결과와 무관하게 발표 가치가 있습니다.**

### 트랙 2 (다음, 저비용) — **write policy의 오라클 상한**

RL 학습 없이 "메모리 정책을 학습하면 얼마나 좋아지는가"의 **상한**을 구할 수 있습니다. Adaptive-RAG의 silver label 트릭과 정확히 같은 발상입니다.

이미 갖고 계신 재료:
- `*.ops.jsonl` — arm별 모든 write 결정(ADD/UPDATE/MERGE/DELETE/NOOP)
- `*.records.jsonl` — 문항별 검색된 청크와 판정 결과
- 1,986문항의 evidence turn 라벨(LoCoMo가 turn ID를 annotate합니다)

절차:
1. 각 write 결정에 대해 **사후적으로** "이 결정이 나중에 어떤 문항의 정답에 기여했는가"를 역산
2. 기여 0인 write를 전부 제거한 가상 store를 만들고 재채점 → **오라클 상한**
3. 실제 J와의 갭 = "학습 가능한 여지"

이게 나오면 세 가지가 동시에 답해집니다:
- Mem0의 79% NOOP는 좋은 판단인가 나쁜 판단인가 (오라클과 비교)
- A-MAC 게이트가 실제로 얼마나 손해인가 (A-1에서 "전부 admit"임을 증명하셨으니, 오라클 대비 손실을 정량화 가능)
- 각 arm의 55.6%/26.9%/19.5% abstention 중 몇 %가 **write 실패**이고 몇 %가 **read 실패**인지 분리

**추가 API 비용 거의 0입니다** (재채점만). 그리고 이건 어떤 논문도 안 한 계산입니다 — 다들 write policy를 학습하려 하지, 학습의 상한을 먼저 재지 않습니다.

### 트랙 3 (확장) — **write path의 포이즈닝 강건성**

트랙 1·2가 기존 결론을 굳히는 일이라면 이건 새 영토입니다.

최소 설계:
- LoCoMo 대화에 MINJA 스타일 주입 turn을 삽입(대화당 1~3개, 0.05% 수준)
- 9개 arm 전부 동일 프로토콜로 ingest
- 측정 두 가지: **주입 성공률**(오염 항목이 store에 남았는가) × **공격 성공률**(retrieval에 올라와 답을 바꿨는가)
- 대조군: 주입 없는 기존 run (이미 있음)

가설이 명확해서 실험이 재밌습니다. 압축이 강한 arm(Nemori의 서사 접기)은 주입 성공률이 낮지만 한 번 통과하면 오염 반경이 클 것이고, 원자적 arm(Mem0의 46자 fact)은 반대일 것입니다. **압축 vs 보존 트레이드오프의 세 번째 축**이 되고, 이건 지금 아무 논문에도 없습니다.

비용은 트랙 2보다 크지만(재-ingest 필요) 대화 1~2개로 파일럿이 가능하고, 이 레포는 이미 그런 파일럿 게이트를 갖고 있습니다(커밋 `58930b1` "a pilot gate that cannot become the run it guards against").

---

## 4. 읽을거리 — 지금 레포에 없는 것만

| 공백 | 문헌 | 왜 |
|---|---|---|
| A (substrate) | Cartridges 2506.06266 + 2508.17032 | activation memory, self-study, MemoryOp의 경계 |
| A | Parametric Memory Law 2605.30260 | LoRA를 메모리로 볼 때의 용량 멱법칙 |
| B (read=방법론) | GAM 2511.18423 | AOT vs JIT, passthrough의 재해석 |
| C (학습된 정책) | Auto-Dreamer 2605.20616 | region rewriting, GRPO consolidation |
| C | Letta sleep-time compute | write/read 경로 분리의 이론적 정당화 |
| D (보안) | OWASP ASI06, MINJA, AgentPoison, eTAMP, 서베이 2604.16548 | 트랙 3의 기반 |
| E (평가) | dial481/locomo-audit, LoCoMo-Plus, BEAM, MemoryArena | 트랙 1의 기반 |
| 이론 | rate-distortion 관점 2607.08032 | 아래 참조 |
| 지도 | 서베이 2603.07670 | write–manage–read 3D taxonomy — 레포 문서의 상위 프레임으로 유용 |

마지막 것 하나 따로 짚겠습니다. **2607.08032(rate-distortion view of memory compaction)이 jinmang2님 기존 노트 라인과 정확히 이어집니다.** roofline과 α-β 모델을 유도하시던 그 작업의 메모리 버전입니다. 그리고 이 레포는 **그 이론을 검증할 실측 데이터를 이미 갖고 있습니다** — Mem0의 46.0자 atomic fact 5,427개 vs Nemori의 서사 episode, 그리고 각각의 J. 압축률 대 왜곡의 실측점이 손에 있는 상태입니다.

$$
J(\text{arm}) \approx (1 - p_{\text{abstain}}) \times p_{\text{correct}|\text{answered}}
$$

이 분해는 이미 18-locomo-4way에서 하셨고 네 arm 전부에서 성립함을 확인하셨습니다. 여기서 한 발 더 나가면 — $p_{\text{abstain}}$을 **압축률의 함수**로 모델링하고, 왜 Mem0가 "더 많이 저장하고 더 적게 기억하는지"(more memories, less memory)를 정보이론으로 유도할 수 있습니다. 지금 갖고 계신 게 딱 그 유도에 필요한 데이터입니다.

---

## 5. 한 문장으로

이 레포의 약점은 지식이 아니라 **축의 개수**입니다. plaintext substrate × write path × LoCoMo라는 하나의 평면 위에서 극단적으로 깊게 파셨고, 그 깊이는 공개 문헌 어디에도 없는 수준입니다. 다음 성장은 더 깊이가 아니라 **직교하는 축을 하나 세우는 것** — 그리고 그 후보 중 비용 대비 효과가 압도적인 게 트랙 1(판별력 계산 + gold 재채점)입니다. $0이고, 기존 결론 전체의 지위를 확정하고, 결과가 어느 쪽으로 나오든 발표거리가 됩니다.

트랙 1의 검정력 계산이나 gold 필터 replay 스크립트는 지금 바로 같이 짜드릴 수 있습니다. 어느 트랙부터 잡으실지 말씀해주세요.