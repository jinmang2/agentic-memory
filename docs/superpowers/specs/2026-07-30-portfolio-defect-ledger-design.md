# Portfolio: Defect Ledger + Selective Proof + Claude Code Memory MCP — Design

Date: 2026-07-30
Status: approved-in-conversation; pending written-spec review

## Goal

Turn the agentic_memory repo into a portfolio artifact for the Korean job market (KO+EN)
built around its strongest differentiator: **defects found in the source papers and their
official code, what we did about them, and experimental proof** — plus a working
application (Claude Code memory via the existing `agmem-mcp` server).

Decisions made with the user:

- Experiment budget: **hard cap $50, API-only (gpt-4o-mini)**. No local inference.
- Audience/form: **domestic hiring, EN repo canon + KO companion articles**.
- Application: **Claude Code memory MCP**, including hook-based automation.
- Overall approach: **"defect ledger + selective proof"** — the ledger is the canonical
  document; proofs are two-tier (deterministic $0 / API ablations).
- All four API ablations confirmed (A-Mem, Nemori, A-MAC, MemoryOS — MemoryOS is not
  optional).

## Non-negotiable constraints (standing directives)

1. **Phase 0 gate**: round-12 re-audit findings (docs/research/fidelity-round12-fresh-eyes-reaudit.md)
   are currently unverified and unactioned, and the "no experiments until wiring verified"
   directive stands. No API spend until round 12 is verified and actioned.
2. **Pre-spend quote gate**: before each API experiment, produce a dry-run cost estimate
   from harness call-count instrumentation and get user approval.
3. **Full artifact capture**: every run persists LLM I/O traces, retrieval chunks, memory
   snapshots, timing, and cost (existing requirement — never pay for the same tokens twice).
4. No unattended local inference; no `git add -A`.

## §1 Defect ledger (canonical document)

New files: `docs/17-defect-ledger.md` (EN canon), `docs/17-defect-ledger.ko.md` (KO,
doubles as blog/interview material).

The twelve fidelity-round docs are process logs; the ledger reorganizes **outcomes only**
into three tiers:

- **Tier A — published numbers are artifacts**: A-MAC recall (substring-match bug),
  A-Mem published-numbers edition (`NameError` in the extractor → every note has empty
  keywords/context/tags after spending the LLM call), MemMachine (every eval entry point
  raises `TypeError` at the audited SHA).
- **Tier B — paper ≠ code**: MemoryOS (`evict_lfu`, no heat eviction anywhere),
  G-Memory (0.7 on 1−L2² ⇒ effective cosine 0.85), Nemori (0.85 merge threshold plumbed
  into a field no code reads), Zep/Graphiti (main has moved past its own paper),
  A-MAC (θ* weights fit while novelty pinned at 1.0, recency ≈ 0).
- **Tier C — evaluation-harness defects**: LoCoMo cat5 scoring bug, upstream stopword
  partial credit, MemoryOS cost distortion under batching deviation.

Entry schema: paper claim → what the official code actually does (code-line citation) →
our handling (fix, or disclosed deviation at the code site) → proof method (§2 tier) →
impact on published numbers. The 96 adversarial verdicts (94 confirmed / 2 refuted) are
the evidence base; the background-compiled defect catalog (this session) supplies
per-entry evidence pointers.

## §2 Two-tier proof + cost plan (cap $50)

**Tier-0 deterministic reproductions — $0.** Standalone scripts under
`scripts/repro/defects/`, one per defect where feasible, run in CI ("the CI proves it,
not the prose"):

- G-Memory threshold equivalence (hand-made normalized vectors).
- A-Mem `NameError` reproduction against the pinned upstream clone.
- MemMachine eval-entry `TypeError` reproduction.
- Nemori unread-config-field trace.
- LoCoMo re-scoring audits replayed over existing `results/` artifacts (no new spend).

Upstream clones live in `~/.agmem/upstream` (9 copies already present); CI jobs that
need them must skip cleanly when absent (capability-gating convention).

**Tier-1 API ablations — all four confirmed, gpt-4o-mini, LoCoMo.**
Measured anchors: A-Mem full campaign $8.95 (ingest $2.32/seed × 3 + eval $1.98).

| # | Experiment | Claim proven | Est. cost |
|---|---|---|---|
| 1 | A-Mem: empty-metadata edition vs fixed extractor | published numbers describe a metadata-less system; wasted-call cost quantified | ~$7–9 |
| 2 | Nemori: merge filter unread vs 0.85 enforced | missing filter ⇒ extra LLM merge calls (count + $) and quality impact | ~$5–7 |
| 3 | A-MAC: published θ* vs retuned gate | published weight vector is meaningless off its (degenerate) fitting distribution | ~$6–8 |
| 4 | MemoryOS: LFU vs heat eviction (conv subset) | the mislabeled eviction policy has measurable retention-quality impact | ~$10–14 |

Projected total ~$28–38 against the $50 cap; headroom absorbs re-runs. Discipline:
one seed for the main run, seed-replicate only headline deltas; quote-then-approve
before each spend; full artifact capture always.

## §3 Claude Code memory MCP

`src/agmem/mcp/server.py` already ships 7 tools + admin (stdio/HTTP). Remaining work
is integration, automation, demo, and reliability:

1. **Integration package**: example `.mcp.json`; CLAUDE.md instruction template
   (recall via `search_memory` at session start, capture via `add_memory`,
   `add_task_result` at task end).
2. **Hook automation (in scope per user)**: SessionStart hook injects recalled context;
   Stop/PostToolUse-style hook captures conversation turns automatically — instruction-only
   mode remains the fallback for users who don't install hooks.
3. **Methodology-switch demo**: same conversation ingested under `--organizers amem` vs
   `nemori,reasoning_bank`; show divergent memory organization via `memory_stats` /
   `admin_snapshot_log`. This is the framework's thesis made visible in the app.
4. **Reliability**: stdio round-trip e2e smoke test in CI (mock LLM); live demo runs
   budgeted at ~$2–5.
5. **Deliverables**: README quickstart section, recorded demo (asciinema/GIF),
   one KO write-up.

## §4 Phasing

- **Phase 0** — verify + action round-12 findings (lifts the experiment freeze).
- **Phase 1** — defect ledger EN + Tier-0 repro scripts wired into CI.
- **Phase 2** — per-experiment dry-run quote → approval → run all four ablations,
  results into ledger + README.
- **Phase 3** — MCP integration + hooks + demo + e2e smoke.
- **Phase 4** — KO ledger + KO MCP write-up; final README pass.

Phases 1 and 3 are independent; they may interleave, but Phase 2 strictly follows
Phase 0.

## Error handling / risk

- **Cost overrun**: quote gate per experiment; if a quote exceeds its table estimate by
  >50%, shrink the conversation subset or drop seeds, and re-quote.
- **Round-12 fixes change behavior**: expected — that is why Phase 0 precedes all
  measurement; ledger entries cite post-fix SHAs.
- **MemoryOS spend blowup** (upstream is 600+ calls/campaign): conv subset is chosen
  from the dry-run quote, not assumed.
- **Hook automation regressions**: hooks are additive and optional; instruction-only
  integration is the tested baseline.

## Testing

- Tier-0 repro scripts are themselves tests (CI).
- MCP e2e stdio smoke with mocked LLM in CI; one live smoke before recording the demo.
- Ablation harness reuses `agmem.bench` (LoCoMo) — no new measurement code paths
  without an upstream-diff check (audit technique already established).
