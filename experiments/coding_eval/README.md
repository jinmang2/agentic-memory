# Coding Memory Evaluation Prep

This fixture prepares actual-coding usefulness comparisons without running a model,
agent, or provider client.

Run:

```bash
uv run --no-sync python -m agmem.bench.coding_eval prepare experiments/coding_eval/recipe.json --output /tmp/coding-eval-manifest.json
uv run --no-sync python -m agmem.bench.coding_eval verify /tmp/coding-eval-manifest.json
```

The three required arms are:

- `none_baseline`: no injected memory.
- `raw_memory`: raw episode memory is available.
- `runbook_memory`: distilled experience/runbook memory is available.

Result rows are filled later by the measurement runner. Unknown latency and cost fields
must remain `null`; `0` means a measured zero and is invalid as a placeholder.

The four task fixtures are tiny local coding workspaces with deterministic acceptance
commands:

- timezone cutoff: `python -m fixtures.timezone_cutoff.acceptance`
- retry idempotency: `python -m fixtures.retry_idempotency.acceptance`
- corrected API field: `python -m fixtures.corrected_api_field.acceptance`
- cache validation bypass: `python -m fixtures.cache_validation.acceptance`
