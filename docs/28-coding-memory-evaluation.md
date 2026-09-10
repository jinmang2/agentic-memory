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
  --results experiments/coding_eval/out/results.json
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
untouched harmful task). The twelve real jobs have not been run yet: the headless agent
call is a paid run, and it must be started by the user.
