# Offline preparation examples

These files describe proposed experiments and synthetic accounting examples. They do not run a benchmark.

- `analysis-contract.json`: contrasts, repeat units, metrics, interpretation, execution prerequisites.
- `execution-settings.json`: explicit baseline settings recovered from saved runs and adapter defaults; preparation hashes this file but does not apply it to the harness.
- `cost-ledger.fixture.json`: synthetic microUSD charges, including one retry. Prices and token counts are invented test values, not provider pricing or measured costs.

Actual datasets, settings containing credentials, saved databases, generated manifests and diagnosis outputs stay outside this directory. Local artifacts are kept under `.omx/artifacts/measurement-followup-20260908/`.

`fixed-store.recipe.json` reuses the two locally saved raw/experience snapshots across the vector/explorer arms, with three retrieval-and-reader repeats each. `fresh-write.recipe.json` describes three independent full builds per arm. Both require the local dataset and ignored TOML settings. Their short `source_files` list is an example, not a complete reproducibility inventory; the local generated recipes expand it to the current package and upstream source list.

`reader-only-design.json` is explicitly a design document, not a `prepare` recipe: it requires frozen prompt replay, which is not wired to the measurement harness in this batch.

The estimate recipe has six cost slots, including retries. These are disjoint estimate buckets: base component estimates exclude retry charges, and the retries slot contains those charges once. The actual ledger instead attributes every attempt to its component and reports retries as a non-additive subtotal. Do not add that subtotal to ledger totals or copy it into recipe retries while leaving the same charges in the component estimates.

`budget-reservation.fixture.json` is a reader-only synthetic scope: under a 100 microUSD limit, one 60 microUSD reservation remains pending while another reserved at 40 settles at 80. Its final exposure is 140, so the second settlement is breached and the final decision is blocked. This is not a whole-study cost estimate.
