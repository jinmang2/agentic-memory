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
