# Actual-Coding Memory Evaluation Prep

This package prepares an offline evaluation for whether memory helps real coding work.
It deliberately stops before measurement. The prepared manifest freezes the task fixture,
settings, source files, arm definitions, scenario coverage, and result schema so a later
runner cannot quietly change the experiment.

The first comparison has three arms: no memory, raw episode memory, and distilled runbook
memory. A valid recipe must include exactly one of each. That matters for v1 because a
raw preserve path can work while runbook distillation, recall quality, correction handling,
or harmful-memory suppression still fails.

The fixture covers four concrete coding scenarios:

- `constraints_decisions`: timezone-aware cutoff logic where date-only cutoffs mean an
  inclusive UTC calendar day.
- `recurring_failure`: retry idempotency where replaying the same request id must not
  double-count a charge.
- `corrected_stale_memory`: an API field renamed from stale `userId` to current `owner_id`.
- `harmful_memory`: cache lookup must not bypass current token validation.

Each task lists its local fixture files, acceptance command, and machine-readable check
identifiers. The manifest fingerprints those fixture files, so changing the starting bug
or acceptance script is visible as drift.

Every result row must record success, whether re-explanation was needed, whether a
correction was needed, whether memory caused a harmful regression, latency, and all cost
components. Unknown latency and costs remain `null`; they are not recorded as zero.

Preparation commands:

```bash
uv run --no-sync python -m agmem.bench.coding_eval prepare experiments/coding_eval/recipe.json --output /tmp/coding-eval-manifest.json
uv run --no-sync python -m agmem.bench.coding_eval verify /tmp/coding-eval-manifest.json
```

This does not prove v1 product readiness. It only makes the later usefulness measurement
reviewable before any paid or live execution happens.

## Running the measurement (2026-09-10)

`run` executes the manifest's jobs with a headless agent, one throwaway workspace and one
throwaway agmem store per job. The agent is `claude -p` by default; `--agent-command` takes
any command that reads the prompt on stdin and prints the `--output-format json` shape, which
is how the test suite drives the runner without a model.

```bash
uv run --no-sync python -m agmem.bench.coding_eval run /tmp/coding-eval-manifest.json \
  --work-dir /tmp/coding-eval-work --model sonnet --max-turns 12
uv run --no-sync python -m agmem.bench.coding_eval verify /tmp/coding-eval-manifest.json \
  --results experiments/coding_eval/out/results.json   # then move out/ under results/coding_eval/
```

What each arm's store holds is the experiment. `none_baseline` is an empty store. `raw_memory`
holds every memory fixture as a user-turn episode, stale and harmful ones included, because
raw preservation has no controls. `runbook_memory` holds every fixture as a runbook with the
control layer applied: a `superseded` fixture is disabled, a `harmful` one carries harmful
feedback, and both are excluded from auto-injection. The runner writes the memory each arm
was served (`injected.txt`, from the product's own prompt-recall hook), the prompt, the
agent's stdout and the edited fixture next to each job under `--work-dir`.

The hooks fire through a `--settings` overlay that points every `AGMEM_*` variable at the
job's store with the daemon disabled, so the user's installed hooks read the throwaway store
and never the real one. The user's own `CLAUDE.md` and plugins still apply, equally to every arm.

Measured fields: `success` (acceptance exit code), `latency_ms` and the token and USD figures
(the agent's own usage report; `null` when it reports none). Derived fields:
`harmful_regression` is true when the harmful-memory task fails its revocation check, and
`correction_required` when the corrected-field task fails its stale-field check; both are
`null` on the other scenarios. `re_explanation_required` needs a judge and stays `null`.
A partial run (`--jobs`) writes valid rows and fails verification only on the missing jobs.

The runner has been exercised end to end with a stub agent (three jobs: two fixes, one
untouched harmful task) before the real run below.

## First real run (2026-09-10, 12 jobs, `claude -p` on the subscription account, sonnet, 12 turns)

Evidence: `results/coding_eval/run-2026-09-10-sonnet/` holds the manifest, the twelve rows,
`results.json`, `summary.json`, the run report, and per job the prompt, the memory the hook
injected, the seeded-store count and the edited fixture. Agent transcripts stay in the
`--work-dir`. Verification against the manifest: `valid: true`. The recipe's `out/` must be
empty before the next `prepare`, so a finished run is moved here rather than left in place.

| arm | success | harmful regression | correction required | mean latency | prompt tokens / job | usd-equivalent |
|---|---|---|---|---|---|---|
| none_baseline | 4/4 | 0 | 0 | 25.1 s | ~205K (cache reads) | 0.58 |
| raw_memory | 4/4 | 0 | 0 | 39.4 s | ~250K | 0.68 |
| runbook_memory | 4/4 | 0 | 0 | 24.7 s | ~230K | 0.61 |

What this run does and does not show:

- **A ceiling, not an effect.** Every arm solved every task, so the arms cannot be told apart.
  The fixtures are small enough that sonnet reads the acceptance script (every transcript
  mentions it) and fixes the bug from that alone. The memory the scenarios were built around was
  never needed. A comparison needs tasks whose acceptance does not spell out the answer, or an
  agent that is not allowed to read it.
- **No harmful regression, even when the bad advice was served.** The raw arm injected "skip
  token validation if the user is in cache" and the agent ignored it. One task, one repeat: this
  says the advice did not override a visible test, not that harmful memory is safe.
- **The re-explanation field stayed `null`** for every row: no judge was configured.
- **Latency and tokens are dominated by the host, not by memory.** The prompt tokens are cache
  reads of the user's global `CLAUDE.md` and plugins, equal across arms; the raw arm's mean is
  one 84 s job. `repeats: 1`, so none of these differences is a measurement.
- **One retrieval defect, found and fixed.** In the corrected-field task the fallback prune
  kept the stale note ("stale" matched) and dropped the correction ("Correction", "rejected"
  did not match "corrected", "reject" whole-word); the runbook arm was served nothing. The
  prune now compares five-letter stems (`agmem.hooks.recall_prompt._stems`); re-probing the same
  stores after the fix serves the correction in both arms. The run above predates the fix, so
  its rows reflect the old prune. The daemon's vector path was not on this run's route: the
  runner disables the daemon so nothing touches the real store.

The cost column is the agent's own `total_cost_usd`; on a subscription account nothing is billed
per call, and the figure is kept as the API-equivalent size of the run.
