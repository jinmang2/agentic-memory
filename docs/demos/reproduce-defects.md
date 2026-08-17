# Demo D — check the ledger's claims on your own machine

[`docs/17-defect-ledger.md`](../17-defect-ledger.md) says things about other people's published
code: that a paid LLM call is thrown away, that a configured threshold lands in a field nobody
reads, that a released test-time-scaling loop cannot run at all. Those are not comfortable claims to
make from prose, so they are not made from prose.

Six scripts re-derive them on your machine. **No model call, no API key, $0, about 25 seconds.**

```bash
git clone https://github.com/jinmang2/agentic-memory && cd agentic-memory
uv sync --no-default-groups --group dev
cd scripts/repro/defects
uv run --no-default-groups --group dev python run_all.py --fetch
```

The group flags are repeated on the `run` line on purpose: a bare `uv run` re-syncs with the
*default* groups and would pull in the heavy vector and graph backends none of this needs.

`--fetch` pulls the five upstream projects at the exact commits the ledger cites — a depth-1 fetch
of one commit each, into `~/.agmem/upstream` (override with `$AGMEM_UPSTREAM`). Drop the flag if
you already have them; they are never modified and never deleted.

## What it prints

```
running 6 Tier-0 defect reproductions — no model call, no API key, $0

✓ PASS  repro_amem_nameerror.py  (18.2s)
      proved: static: memory_layer.py uses re.* without importing re -> NameError is inevitable
      proved: dynamic: valid LLM JSON still yields empty metadata after 1 spent call
✓ PASS  repro_gmemory_threshold.py  (0.1s)
      proved: upstream 0.7 on (1 - squared-L2) == cosine >= 0.85; our gate uses 0.85 directly
✓ PASS  repro_locomo_rescoring.py  (4.1s)
      proved: self-consistency: 12314 questions re-scored, max drift 5.00e-04
      proved: scorer lineage matters: 2804/12314 questions score differently
✓ PASS  repro_memmachine_typeerror.py  (0.4s)
      proved: static: the harness constructor-calls a type alias that is not a class
      proved: dynamic: calling the Annotated union raises TypeError ('types.UnionType' object is not callable)
✓ PASS  repro_nemori_dead_knob.py  (0.2s)
      proved: merger._similarity_threshold: 1 store, 0 loads — config plumbs into a dead field
✓ PASS  repro_reasoningbank_matts_inert.py  (0.5s)
      proved: induce_scaling calls one_step_chat unbound -> TypeError (missing 'text') before any LLM call
      proved: induce_scaling reads ONE result dir N times: **Trajectory 1..N :** are identical copies
      proved: scaled bank entry stores an unsplit, unpacked value where the consumer expects a list
      proved: MaTTS labels reward==0 as success; the sibling labels reward==1 — inverted
      proved: half the advertised --model surface dies before any model call (gpt-3.5 KeyError, gpt-4o TypeError) even in the sibling that runs, and the paper's Claude backbone cannot be selected at all
      proved: SEQUENTIAL_PROMPT: 2 definition(s), 0 references — sequential MaTTS is unwired

6 passed, 0 skipped, 0 failed — 14 claims re-proved in 23.4s, $0 spent.
The claims are the ledger's, restated by the scripts that prove them: docs/17-defect-ledger.md

VERDICT: all 6 reproductions held.
```

Every `proved:` line is printed by the script that proved it. The runner reports; it never
paraphrases a claim, because a paraphrase is exactly where a claim quietly becomes a weaker or
stronger one than what the code showed.

## What the six prove

| script | upstream project | the claim |
|---|---|---|
| `repro_amem_nameerror.py` | A-Mem (`WujiangXu/AgenticMemory`) | the published-numbers edition pays for a metadata call, hits a `NameError` from a missing `import re`, and returns empty metadata — for every note |
| `repro_gmemory_threshold.py` | G-Memory (`bingreeky/GMemory`) | the same constant `0.7` applied to a different distance: upstream's gate is cosine ≥ 0.85, not 0.7 |
| `repro_locomo_rescoring.py` | LoCoMo scoring | scorer lineage moves 2,804 of 12,314 question scores — which scorer produced a number is part of the number |
| `repro_memmachine_typeerror.py` | MemMachine | the harness constructor-calls a type alias; the call raises `TypeError` before anything runs |
| `repro_nemori_dead_knob.py` | Nemori (`nemori-ai/nemori`) | a configured similarity threshold is stored once and read zero times — the knob is dead |
| `repro_reasoningbank_matts_inert.py` | ReasoningBank / MaTTS (`google-research/reasoning-bank`) | the released MaTTS path dies before any model call, re-reads one trajectory N times, and inverts its own success label |

## Three properties this demo is built around

**A skip is not a pass.** When a script's evidence is missing it says so and proves nothing, and the
runner prints that as `SKIP` with the reason, keeps it out of the pass count, and ends on a
`VERDICT: partial` line naming how many scripts established nothing. A run in which everything
skipped exits non-zero — a broken setup must not be able to look like a clean bill of health.

**The pins are not restated anywhere.** The commit SHAs live in `_common.PINS`, the fetch procedure
lives in `fetch_upstream.py`, and [CI](../../.github/workflows/ci.yml) calls that same script rather
than carrying its own copy. So the commits CI proves things against and the commits you just fetched
are the same by construction. If a local clone drifts off its pin, the scripts print a warning
naming both SHAs instead of silently proving something about a different program.

**It already runs on every push.** This is not a demo path kept alive by hand. CI runs
`fetch_upstream.py` and then `run_all.py` on every push and pull request, so a reproduction that
stops reproducing breaks the build.

## When a claim stops holding

Upstream repositories move. If one of these projects fixes its defect, the right outcome is that the
script keeps proving what the pinned commit did — that is what a pinned SHA is for — and the ledger
row gains a note that it was fixed upstream. A reproduction that quietly starts passing against a
newer commit would be proving a different claim than the one written down.
