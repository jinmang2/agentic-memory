# Demos

Short, self-contained pages that show one thing each, built so a reader can check them rather than
take them on trust. Both of the ones here spend **$0** and make **no model call** — every number
comes out of an artifact already in the repository, or out of code running on your own machine.

| demo | what it shows | cost to run |
|---|---|---|
| [cost-is-tokens.md](cost-is-tokens.md) | turning one dedup flag off multiplied an ACE run's bill by 5.9× while the call count moved by 2 — the cost of a self-evolving playbook is carried in tokens, not requests | $0, no model call |
| [reproduce-defects.md](reproduce-defects.md) | six scripts re-derive the defect ledger's claims about upstream code on your machine, against pinned commits, in about 25 seconds | $0, no model call |
| [dogfooding.md](dogfooding.md) | the memory layer running inside Claude Code: the real hooks and the real MCP server, timed against the deadline that shaped their design | $0, no API key |

None is hand-written. The first is emitted by `scripts/repro/demo_cost_is_tokens.py` from
`results/repro/`; the second is the actual output of `scripts/repro/defects/run_all.py`, which CI
runs on every push; the third is written by `scripts/repro/demo_dogfooding.py` as it drives the
shipped hook binaries and server on a throwaway store.

**A number without its condition is not a number.** Every measurement quoted in these pages carries
the arm, model, benchmark and seed it came from. Where a demo shows a result that did not separate
from its control, it says so on the same page as the headline — that is the whole reason the control
arms exist.
