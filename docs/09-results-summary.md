# LoCoMo conv0 4-way 재현 결과 (2026-07-16)

> **⚠ SUPERSEDED (2026-08-19).** 여기서 "보류 중"이던 재측정은 이후 수행됐다:
> 전체 10-conversation 5-way(A-Mem·Nemori·Mem0·Zep 포함, gpt-4o-mini + LLM judge)는
> `docs/18-locomo-4way.md`가 정본이다. 이 문서는 당시 로컬 0.6B 예비 결과의 기록으로 보존.

> 조건: LoCoMo conv0 (세션 19, 423턴, QA 199), **전 역할 Qwen3-0.6B-Q8_0** (RTX 2060,
> llama.cpp), e5-small 임베더, k=10, F1/BLEU-1. 단일 run 예비 결과.
> 원자료: `results/locomo-conv0-*.json` (per-question 레코드 포함).
> **Fidelity**: 측정 4종은 충실도 ●/◑ 등급만 포함 (docs/10-fidelity-audit.md).
> MemoryOS는 측정 당시 segment 매칭 F_score에서 Jaccard 항이 빠진 cosine 단독 구현이었음
> (round-5 P1에서 cos+Jaccard로 복원됨, memoryos.py) — 아래 수치 해석 시 유의.
> **MemoryOS 행은 이후 세 세대 낡았다**: 6차(page 단위 계수·LPM 3-스토어 재구현),
> 8차(STM 1-page 롤링·dialogue chain·STM/assistant-knowledge QA 주입·2단계 page 검색),
> 9차(read 채널 순수화 + 계보 knob 분리)가 write 경로와 read 컨텍스트를 모두 바꿨다. 특히
> **결론 3(비용)은 정정됐다** — 아래 해당 항목 참조.
> 구 배선 재현은 **`memoryos_mixed` config**(`scripts/exp_locomo_conv0.py`)로 한다: raw
> episodic 채널 + `flush_stm_on_drain=True` + `dialogue_chain=False` + `page_recall_cap=0`.
> 새 `memoryos`/`memoryos_eval`은 이 행의 비교 대상이 아니라 **대체 대상**이다 — read 채널
> 구성(업스트림에 없던 원문 검색 채널 제거)과 계보(pypi cap 7 / eval cap 10 + 전량 주입)가
> 둘 다 다르다.
> Zep-graph·G-Memory는 골격(○) 단계라 측정에서 의도적으로 제외.
> **⚠️ 2026-07-16 심층 감사 후 재해석** (docs/research/fidelity-deep-audit.md):
> 이 표는 "동일한 우리 read 파이프라인 위 write 조직화 비교"이며 **논문 재현이 아님**.
> A-Mem은 read에서 1-hop 링크 확장 누락, Nemori는 검색 설정 3중 불일치(semantic 20→10,
> r=2 원문 첨부 누락, 1600토큰 예산) 상태로 측정됨 — P0 수정 완료, 재측정은 보류 중.
> **2026-07-18 라이프사이클 재설계** (스펙:
> `docs/_internal/specs/2026-07-18-nemori-lifecycle-redesign-design.md` — gitignored 내부 문서)로 `nemori_v4` /
> `nemori_upstream` / `nemori_mix` / `nemori_memoryos` / `nemori_amem` 5개 config가
> `scripts/exp_locomo_conv0.py`에 추가됨(기존 `nemori`=v1 동치는 불변) — **아직 실측 대기**,
> 이 표에는 반영되지 않았다.
> **⚠️ 2026-07-21 충실도 리뷰 재현 캐비앗** (docs/10 동일자 항목): 어떤 수치도 아래 없이
> "논문 재현"으로 인용 금지 — (1) A-Mem은 eq.(3) 메타데이터 임베딩(발표 수치를 낸 공식
> 코드의 content-only 임베딩과 다름), (2) empty-actions→양효과 폴백이 소형 모델에서
> evolution을 과다 유발, (3) read 링크확장이 전역 캡5(upstream per-hit와 상이),
> (4) MemoryOS recency τ=24h(논문 μ≈1e7s 아님), N_visit 되먹임은 배선돼 있으나
> ingest-후-eval 구조라 무효과, LPM은 append-only(upstream 프로필 문서 교체 미구현).

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
3. **비용 스펙트럼**: MemoryOS가 organizer 호출 91회로 가장 저렴 (A-Mem/Nemori의 1/9).
   정확도-비용 파레토에서 passthrough가 지배적, MemoryOS가 조직화 중 최선.
   > **⚠️ 2026-07-27 8차 정정 — 이 결론은 무효다.** "배치 설계 덕에"라고 적었던 그 배치는
   > **MemoryOS의 설계가 아니라 우리 구현의 이탈**이었다. 업스트림 STM은 1-page FIFO 롤링
   > 윈도우(`while is_full(): pop_oldest()`, `is_full()`이 `>=`이므로 pop 1회)이고 방출
   > **페이지마다** topic-summary 1콜 + continuity 1콜 + meta_info 1콜이 나간다. 419턴
   > ≈ 209페이지면 **600콜+** 규모이고, 우리는 `capacity`페이지당 1콜 + 체인 콜 0회로
   > 91콜이었다. 즉 **방법론 자체는 A-Mem/Nemori보다 비싸며**, 파레토에서 MemoryOS가
   > 차지한 위치는 재측정 전까지 근거가 없다. 상세:
   > `docs/research/fidelity-round8-remaining-pieces.md` §M1.
4. **방어층은 완성 단계**: 총 1,844 organizer 호출에서 drop 10회(0.5%). 0.6B로
   파이프라인 자체는 무너지지 않는다 — 품질이 병목이지 형식이 아님.
   → Phase 4(추출 태스크 SFT)의 가설을 지지: 형식은 해결됐으니 품질을 학습으로.

## 다음 검증 (전체 리뷰 시 논의)

- answer만 API 모델로 교체 → organizer 품질 효과 분리 (A-Mem 발표 후속)
- extract/distill만 4B-AWQ 티어링 → "organizer 품질 종속" 가설 직접 검증
- conv 1–9 확장 + 3-run 편차, Phase 4 SFT 후 재측정 (동일 하네스)
