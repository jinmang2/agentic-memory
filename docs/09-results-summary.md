# LoCoMo conv0 4-way 재현 결과 (2026-07-16)

> 조건: LoCoMo conv0 (세션 19, 423턴, QA 199), **전 역할 Qwen3-0.6B-Q8_0** (RTX 2060,
> llama.cpp), e5-small 임베더, k=10, F1/BLEU-1. 단일 run 예비 결과.
> 원자료: `results/locomo-conv0-*.json` (per-question 레코드 포함).

| config | Overall F1 | BLEU-1 | ingest | organizer LLM calls | drops |
|---|---|---|---|---|---|
| **passthrough** (hybrid retrieval) | **22.85** | 18.40 | 6.3s | 0 | 0 |
| A-Mem (수정판) | **23.25** | 19.04 | 946s | 841 | 4 |
| Nemori | 18.97 | 14.76 | 639s | 912 | 5 |
| MemoryOS | 20.90 | 16.70 | 143s | 91 | 1 |

카테고리별 F1:

| 카테고리 | passthrough | A-Mem | Nemori | MemoryOS |
|---|---|---|---|---|
| single-hop | 20.44 | **22.84** | 16.17 | 17.88 |
| multi-hop | 15.97 | 14.39 | 13.16 | **16.32** |
| temporal | 44.66 | 44.43 | **45.71** | 43.25 |
| open-domain | 8.11 | 8.11 | 7.67 | **11.09** |
| adversarial | **18.02** | 17.40 | 9.17 | 13.65 |

## 핵심 발견 (스터디의 중심 질문: "0.5B로 어디까지")

1. **0.6B가 organizer까지 맡으면 어떤 방법론도 raw hybrid retrieval을 의미 있게 못 이긴다.**
   A-Mem +0.40이 최선, Nemori는 -3.88. 논문들의 이득은 organizer 모델 품질에 강하게
   종속된다는 실증 — 각 논문의 최소 백본(1B~4o-mini)과 정합적.
2. **추상화의 비용이 그대로 재현됨**: Nemori의 adversarial 붕괴(18.02→9.17)는 파생
   서사가 raw 디테일을 밀어내 "없는 정보에 없다고 답하기"가 어려워지는 패턴 —
   Zep 논문이 인정한 single-session-assistant 퇴화와 동일 계열. 반면 Nemori는
   temporal에서 유일하게 baseline 상회(45.71) — **시간 절대화 설계는 0.6B에서도 유효**.
3. **비용 스펙트럼**: MemoryOS가 배치 설계 덕에 organizer 호출 91회로 가장 저렴
   (A-Mem/Nemori의 1/9). 정확도-비용 파레토에서 passthrough가 지배적, MemoryOS가
   조직화 중 최선.
4. **방어층은 완성 단계**: 총 1,844 organizer 호출에서 drop 10회(0.5%). 0.6B로
   파이프라인 자체는 무너지지 않는다 — 품질이 병목이지 형식이 아님.
   → Phase 4(추출 태스크 SFT)의 가설을 지지: 형식은 해결됐으니 품질을 학습으로.

## 다음 검증 (전체 리뷰 시 논의)

- answer만 API 모델로 교체 → organizer 품질 효과 분리 (A-Mem 발표 후속)
- extract/distill만 4B-AWQ 티어링 → "organizer 품질 종속" 가설 직접 검증
- conv 1–9 확장 + 3-run 편차, Phase 4 SFT 후 재측정 (동일 하네스)
