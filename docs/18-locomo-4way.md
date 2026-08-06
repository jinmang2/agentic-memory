# The conversational four-way — LoCoMo, one harness, one judge

Four write paths measured against the same 10 LoCoMo conversations, through the same retrieval
harness, scored by the same judge at the same pin. Every row was produced by this repository's
runner (`scripts/exp_amem_repro.py` / `scripts/repro/ingest_parallel.py`) from a committed artifact
set; nothing here is transcribed from a paper.

**What this table is for, and what it is not.** It compares *mechanisms under one measurement
protocol*. **No row here is an attempt to reproduce a published number**, and every system in it
has one — A-Mem's Table 1, Nemori's 73.0 LoCoMo headline, Mem0's vendor figures. Where our row and
a published number disagree, the disagreement is documented in
[`17-defect-ledger.md`](17-defect-ledger.md) rather than explained away here.

## Protocol

| | |
|---|---|
| Benchmark | LoCoMo, all 10 conversations, 5,882 turns, 1,986 questions |
| Answer model | `gpt-4o-mini`, temperature per role from each arm's canned profile |
| Judge | Mem0-J binary judge @ `gpt-4o-mini`, 1,540 questions (categories 1-4; category 5 is answered but not judged — see ledger C-1) |
| Embedder | `text-embedding-3-small` for **all four arms** |
| Metrics | J = judge accuracy; F1 / BLEU-1 = our uniform lexical scorers (ledger C-2 on what those are and are not comparable to) |
| Seeds | **one** |

## Results

| arm | J | F1 | BLEU-1 | write calls | ingest $ | eval $ | total $ |
|---|---|---|---|---|---|---|---|
| **Nemori** arm A (upstream) | **67.60** | 46.79 | 41.74 | 3,579 | 0.87 | 1.37 | **2.24** |
| **Nemori** arm B (0.85 filter live) | 65.78 | 45.79 | 40.68 | 2,759 | 0.69 | 1.12 | **1.82** |
| **A-Mem** | 59.87 | 41.41 | 36.51 | 11,754 | 2.81 | 0.67 | **3.48** |
| **Mem0** `v0.1.94` | 31.82 | 24.71 | 21.57 | 5,890 | 1.87 | 0.30 | **2.17** |
| **Zep** | *not measured* | | | | | | |

### Read path — the arms are not given the same thing to read

Each arm retrieves through its own lineage's operating point, which is deliberate (fidelity over
symmetry, the same choice Track 1 made) and is the single most important thing to know before
citing any row against another.

| arm | retrieved types | k | link expansion | query | context handed to the answerer |
|---|---|---|---|---|---|
| A-Mem | `notes` | 10 | on | LLM-rewritten keywords | 1,913 tok/question |
| Nemori (both arms) | `episodes` + `semantic` | 10 + 20 | off | original question | 3,574–4,409 tok/question |
| Mem0 | `semantic` | 30 | off | original question | **837** tok/question |

Item *count* at read time is 30 for Nemori and Mem0 and 10 for A-Mem, so the spread in the last
column is not a k difference — it is what a "memory" is in each system. Mem0's unit is an atomic
fact averaging **46.0 characters** over all 5,427 of them (per-conversation means 43.2–48.9);
Nemori's is a narrative episode.

### What each write path did — the evolution log

Counts are operations actually applied (or judged and declined), summed over all 10 conversations,
from each run's `*.ops.jsonl`. A memory snapshot cannot produce this table: an UPDATE, a DELETE and
a NOOP all leave a store that looks the same afterwards.

| arm | operations | retrievable items at the end |
|---|---|---|
| A-Mem | ADD 5,882 · LINK 5,866 · UPDATE 16,342 | 5,882 notes |
| Nemori arm A | ADD 555 episodes + 1,926 semantic · MERGE 223 · INVALIDATE 223 | 2,704 |
| Nemori arm B | ADD 763 episodes + 1,745 semantic · MERGE 22 · INVALIDATE 22 | 2,530 |
| Mem0 | ADD 5,654 · **NOOP 26,209** · UPDATE 1,077 · DELETE 227 | 5,427 facts |

`ADD:episodic` is excluded from the table above: all four arms log exactly 5,882 of them, one per
turn, because that row is our harness's own raw-turn write rather than anything the methodology
decided. Where the arms differ is whether those turns *survive* as retrievable items — A-Mem and
Mem0's stores keep them (they are the `episodic` entry in each store's per-type counts), while
Nemori's keep none, its turns having been folded into episodes.

Mem0's 2,945 decision calls returned 33,167 semantic verdicts, **79.0% of them NOOP** — the model
was asked what to do about a candidate fact and answered "nothing" four times out of five. That
ratio is the reason this repo added `OpType.NOOP` to the core vocabulary for this track: a judged
non-mutation is a log row, and counting it as a discard or not counting it at all both lose the
measurement.

## Where Mem0's deficit comes from

The gap is not the answering model being wrong more often. It is the answering model **not being
given the answer**. Same prompt, same judge, same 1,540 questions:

| arm | abstained ("no information available") | correct when it did answer | J |
|---|---|---|---|
| Nemori arm A | 19.5% | 84.0% | 67.60 |
| Nemori arm B | 20.6% | 82.9% | 65.78 |
| A-Mem | 26.9% | 82.0% | 59.87 |
| Mem0 | **55.6%** | 71.7% | 31.82 |

`J ≈ (1 − abstention) × accuracy-when-answered` holds on every row (Mem0: 0.444 × 0.717 = 31.8).
Decomposing Mem0's 35.8-point deficit against Nemori arm A: roughly **26 points are coverage** and
**10 points are accuracy**. Retrieval itself never failed — every one of the 1,986 questions got
exactly 30 facts back, none empty — so this is a property of what the write path chose to keep and
how small the kept units are, not a retrieval bug.

The shape is counter-intuitive and worth stating plainly: **Mem0 stores twice as many retrievable
items as Nemori and retrieves the maximum k, and still hands the answerer a quarter of the
context.** More memories, less memory.

## Footnotes that must travel with any citation of this table

1. **Mem0's published LoCoMo numbers were not produced by this or any released code.** The paper's
   harness instantiates `MemoryClient` and posts to `https://api.mem0.ai`; the write path that made
   those numbers is server-side and closed. Our row measures the **paper-era OSS structure**
   (`v0.1.94`) under our harness and makes no claim to reproduce the published value. See ledger
   entry **M0-C1**.
2. **Do not pair our 31.82 with the "independent re-measurement of 32.4%" that circulates for
   Mem0.** That figure is a LongMemEval re-measurement of a vendor claim of 93.4%; ours is a LoCoMo
   judge accuracy. Different benchmark, different harness, different metric — the numerical
   proximity is a coincidence, and treating it as corroboration would be exactly the error this
   repository's ledger exists to catch.
3. **Embedder.** All four rows use `text-embedding-3-small`. Earlier `all-MiniLM-L6-v2` numbers for
   three of these arms exist in `results/` and are **not** interchangeable with these: re-basing
   moved A-Mem +9.87 J, Nemori arm A +4.22, arm B +0.26, and it reversed the ranking between A-Mem
   and the Nemori arms. No cross-embedder comparison may be cited from this repo without naming
   both embedders.
4. **Read-k and link expansion differ by arm** (table above), by design. A row's cost and its
   context budget are both downstream of that choice.
5. **Nemori arm A vs arm B** is the ledger's B-3 pair — the same code with the dead 0.85 merge
   filter revived. Arm A is the shipped (defective) behavior. On this embedder the *defect wins* on
   quality by 1.82 J while costing 29.7% more calls, which is the reverse of what the same
   comparison showed on MiniLM. Single seed, sign reversal — ledger B-3 carries the replication
   caveat.
6. **Mem0 store shape is our porting decision**: one store per conversation, not upstream's two
   per-speaker stores. This keeps the read path identical to the other three arms and roughly
   halves the per-session cost relative to the upstream harness; it also changes what each stored
   fact is "about".
7. **One seed.** No delta in this table has been seed-replicated except Nemori's arm A/B ordering
   on MiniLM (which replicated, and which the embedder change then reversed).

## Artifacts

Every number above resolves to a committed file under `results/repro/`:

| arm | ingest summary | eval summary |
|---|---|---|
| A-Mem | `gpt-4o-mini_all_ingest_e3s.json` | `gpt-4o-mini_all_k10_ours_expand-on_run1_e3s.json` |
| Nemori arm A | `gpt-4o-mini_nemori_upstream_all_ingest_e3sA.json` | `gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA.json` |
| Nemori arm B | `gpt-4o-mini_nemori_merge085_all_ingest_e3sB.json` | `gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB.json` |
| Mem0 | `gpt-4o-mini_mem0_v0194_all_ingest_e3sM.json` | `gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM.json` |

Each carries its per-conversation summaries, the full LLM I/O trace, the memory snapshot, the
evolution log (`*.ops.jsonl`) and the per-question records with retrieved chunks. The `k10` segment
in the eval filenames is a naming default, not the k that ran — the k that ran is recorded per
question inside the records file, and is the one in the read-path table above.
