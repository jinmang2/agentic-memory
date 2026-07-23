# 재현 실험 이식용 런북 (다른 PC에서 phase1b 돌리기)

> 목적: 이 저장소의 A-Mem×LoCoMo 재현 하네스를 **다른 PC**(OMC/환경 없음)에서
> 처음부터 돌려 rung 1b 헤드라인(K=3 mean±std)을 산출하고, 결과를 이 PC로 회수한다.
> 계약 문서는 `docs/14-amem-reproduction.md`. 이 런북은 그걸 self-contained 절차로 압축한 것.

## 0. 전제 (이 브랜치에 이미 다 있음)
- 브랜치: **`feat/locomo-eval-fidelity`** (origin push됨, HEAD `d38c96e`)
- 저장소: `github.com:jinmang2/agentic-memory`
- 하네스: `scripts/exp_amem_repro.py`, `scripts/repro/phase1b_headline.sh`, `scripts/repro/aggregate_headline.py`

## 1. 셋업 (uv 기반, ~5–10분 + 임베더 다운로드 90MB)
```bash
# (1) uv 없으면 설치
curl -LsSf https://astral.sh/uv/install.sh | sh    # 또는 pipx install uv

# (2) 브랜치 클론
git clone -b feat/locomo-eval-fidelity git@github.com:jinmang2/agentic-memory.git
cd agentic-memory       # 리포 루트 = 모든 스크립트의 기준

# (3) 의존성 동기화
#   - 로컬과 동일(권장, 확실): backends까지 포함해 무겁지만 import 에러 0
uv sync
#   - 또는 lean(embed만; 하네스는 profile="lite"=sqlite라 이걸로도 충분):
# uv sync --no-default-groups --group dev --group embed
```

## 2. 데이터셋 배치 (필수 — 하네스가 고정 경로를 읽음)
하네스 `DATA = ~/.agmem/datasets/locomo10.json` (`scripts/exp_amem_repro.py:43`). 그 경로에
`locomo10.json`(2.8MB, 1,986 QA)을 둔다. 파일은 upstream 리포 `data/locomo10.json`과 **바이트 동일**.
```bash
mkdir -p ~/.agmem/datasets
# 이 PC에서 scp로 옮기거나, upstream 리포에서:
#   git clone --depth 1 https://github.com/WujiangXu/AgenticMemory
#   cp AgenticMemory/data/locomo10.json ~/.agmem/datasets/locomo10.json
```
확인: `uv run python -c "import json,os;d=json.load(open(os.path.expanduser('~/.agmem/datasets/locomo10.json')));print(len(d),'convs', sum(len(s.get('qa',[])) for s in d),'QA')"` → `10 convs 1986 QA`.

## 3. API 키 (리포 루트 `.env.local`, gitignored)
```bash
printf 'OPENAI_API_KEY=sk-...\n' > .env.local      # 절대 커밋 금지 (git check-ignore .env.local 로 확인)
```
하네스가 dep 없이 `.env.local`을 로드한다(`agmem/_env.py:load_env_local`). 기존 env는 안 덮어씀.

## 4. 스모크 먼저 (배선 확인, conv0만, ~$0.35, ~40분)
```bash
bash scripts/repro/smoke.sh
```
기대: `wujiang overall F1 ~34–36`, `ours+J F1 ~31–35 / J ~51`. 여기까지 정상이면 full로.
(이 PC의 최근 스모크 실측: wujiang 35.39 / ours 34.91 / J 51.97, 총 $0.266.)

## 5. rung 1b 헤드라인 (본 실험, K=3 독립 ingest → mean±std)
```bash
K=3 WORKERS=8 INGEST_WORKERS=4 bash scripts/repro/phase1b_headline.sh
```
- **conv-병렬 ingest (기본 켜짐)**: `INGEST_WORKERS`개 대화를 **동시에** ingest한다. 대화 간은
  공유 상태가 없어 결과가 **순차와 바이트 동일**(대화 내부 turn은 여전히 순차 — evolution 순서
  의존). 벽시계가 seed당 ~8.4h → **가장 긴 대화(~690턴≈1h)** 수준으로 단축.
- **rate-limit 노브 = `INGEST_WORKERS`**: 각 대화 워커는 한 번에 **API 콜 1개**만 in-flight →
  동시 콜 ≈ `INGEST_WORKERS`. 계정 RPM/TPM 안에서 **4로 시작, 429 안 뜨면 올림**. 방어 이중화:
  OpenAI SDK가 429/5xx를 2회 백오프 재시도 + 오케스트레이터가 실패 대화를 `--retries`회 재시도
  (재시도 전 부분 store만 wipe). **RAM ≈ INGEST_WORKERS × ~1GB**(워커마다 torch+임베더 로드).
- **ETA (INGEST_WORKERS=4 기준)**: seed당 ingest **~2–2.5h** (10 대화를 4-병렬), K=3 ≈ **~6–8h**.
  (순차 대비 대략 3–4×; 429로 낮추면 그만큼 느려짐.) eval은 WORKERS=8로 수분.
- **비용: ~$4.8** (K=3) — 병렬은 **벽시계만 줄이고 콜 수·비용은 동일**. 실측은 `results/repro/*.json`.
- **중단해도 안전**: 대화별로 완료 판정(per-conv 요약 + 비어있지 않은 store). 재실행 시 완료 대화는
  skip, 부분/죽은 대화만 wipe 후 재-ingest. seed가 전부 끝나야 **통합 sentinel + `_all_ingest_seed<N>.json`**
  기록 → 그 전엔 `--eval-only`가 거부(loud). 데드라인이 오면 §6로 **끝난 seed만 집계**.
- 순차로 강제하려면 `INGEST_WORKERS=1`(대화당 1개씩; 결과 동일, 느림).
- 백그라운드: `nohup bash -c 'K=3 INGEST_WORKERS=4 bash scripts/repro/phase1b_headline.sh' &`

스크립트가 끝에서 자동으로 `aggregate_headline.py`를 호출해
`results/repro/gpt-4o-mini_all_wujiang_headline.json`(per-category F1 mean±std)을 만든다.

## 6. 부분 집계 (데드라인에 끝난 seed만으로)
```bash
uv run python scripts/repro/aggregate_headline.py \
  --out results/repro/gpt-4o-mini_all_wujiang_headline.json \
  --ingest-summaries results/repro/gpt-4o-mini_all_ingest_seed*.json \
  -- results/repro/gpt-4o-mini_all_k10_wujiang_expand-off_run1_seed*.json
```
글롭이 끝난 seed만 잡으므로 1개든 3개든 동작(1개면 std=0). config 불일치 seed는 평균 거부.

## 7. 결과 회수 (이 PC로)
산출물 5종 중 **작은 3종은 git-tracked**(커밋해서 push로 회수):
```bash
git add results/repro/*.json results/repro/*.records.jsonl results/repro/logs/
git commit -m "data(repro): phase1b K=3 headline from <other-pc> (#1)"
git push
```
- **무거운 2종**(`*.llm-trace.jsonl`·`*.memory.jsonl`)과 `stores/`는 gitignore(durable-on-disk).
  오프라인 re-score/분석에 필요하면 별도로 scp/rsync로 옮긴다(용량 큼 — full run 수백 MB~GB).
- 이 PC에서 `git pull` 하면 요약/records/logs로 헤드라인 표를 채울 수 있음.

## 참고 — 왜 K=3인가 / 무엇과 비교하나
- write 온도 0.7 → note 그래프가 랜덤 draw. 단일 수치는 지배 분산(±3–6 F1)을 못 보여줌.
  `--runs`는 answer 경로만 반복해서 이 write 분산을 못 잡음 → **seed별 독립 ingest가 정답**.
- 이 헤드라인(rung 1b) = **우리 재구현**을 wujiang 충실 메트릭으로 잰 것. 논문 Table 1(rung 0)과
  대조하는 행. upstream 원본 코드(rung 1a)는 별도 실험(`docs/14` §1, `phase1a_upstream.sh`).
