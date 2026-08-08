# The conversational five-way — LoCoMo, one harness, one judge

Five write paths measured against the same 10 LoCoMo conversations, through the same retrieval
harness, scored by the same judge at the same pin. Every row was produced by this repository's
runner (`scripts/exp_amem_repro.py` / `scripts/repro/ingest_parallel.py`) from a committed artifact
set; nothing here is transcribed from a paper.

**What this table is for, and what it is not.** It compares *mechanisms under one measurement
protocol*. **No row here is an attempt to reproduce a published number**, and every system in it
has one — A-Mem's Table 1, Nemori's 73.0 LoCoMo headline, Mem0's vendor figures, Zep's 71.2 (which is LongMemEval, not LoCoMo — Zep reports no LoCoMo number at all). Where our row and
a published number disagree, the disagreement is documented in
[`17-defect-ledger.md`](17-defect-ledger.md) rather than explained away here.

## Protocol

| | |
|---|---|
| Benchmark | LoCoMo, all 10 conversations, 5,882 turns, 1,986 questions |
| Answer model | `gpt-4o-mini`, temperature per role from each arm's canned profile |
| Judge | Mem0-J binary judge @ `gpt-4o-mini`, 1,540 questions (categories 1-4; category 5 is answered but not judged — see ledger C-1) |
| Embedder | `text-embedding-3-small` for **all five arms** |
| Metrics | J = judge accuracy; F1 / BLEU-1 = our uniform lexical scorers (ledger C-2 on what those are and are not comparable to) |
| Seeds | **one** |

## Results

| arm | J | F1 | BLEU-1 | write calls | ingest $ | eval $ | total $ |
|---|---|---|---|---|---|---|---|
| **Nemori** arm A (upstream) | **67.60** | 46.79 | 41.74 | 3,579 | 0.87 | 1.37 | **2.24** |
| **Nemori** arm B (0.85 filter live) | 65.78 | 45.79 | 40.68 | 2,759 | 0.69 | 1.12 | **1.82** |
| **A-Mem** | 61.23 | 42.92 | 38.03 | 11,754 | 2.81 | 1.09 | **3.90** |
| **Zep** `cross_encoder` | 42.73 | 28.80 | 24.48 | 27,449 | 4.71 | 0.38 | **5.09** |
| **Mem0** `v0.1.94` | 31.82 | 24.71 | 21.57 | 5,890 | 1.87 | 0.30 | **2.17** |

Each row runs its own lineage's read path. For A-Mem that means two things its paper does not
describe but its evaluation harness does: an LLM keyword rewrite before every search, and a
link-expansion budget granted **per hit** rather than shared. Both were measured this campaign and
both are folded into the row above — the second one because it was **our** deviation and closing it
is a fidelity fix (+1.36 J), the first because it is A-Mem's own step and removing it would make the
row less faithful, not more (it is worth +5.26 J and is reported as a property of A-Mem below).

### Read path — the arms are not given the same thing to read

Each arm retrieves through its own lineage's operating point — deliberate, the same choice Track 1
made, and the single most important thing to know before citing any row against another.

| arm | retrieved types | k | link expansion | query | read-side LLM calls | context handed to the answerer |
|---|---|---|---|---|---|---|
| A-Mem | `notes` | 10 | on, per hit | LLM-generated keywords | **1,986** | 3,322 tok/question |
| Nemori (both arms) | `episodes` + `semantic` | 10 + 20 | off | original question | 0 | 3,574–4,409 tok/question |
| Zep | `facts` + `entities` + `communities` | 10 + 10 + 10 | off | original question | 0 | 1,086 tok/question |
| Mem0 | `semantic` | 30 | off | original question | 0 | **837** tok/question |

A-Mem is the only arm paying an LLM call to read, and that asymmetry is upstream's, not ours: its
harness rewrites each question into keywords before searching while Nemori's and Mem0's read the
question as written. Costing that step is the ablation below.

Zep's row is the one recipe in this table that is a *published configuration* rather than a set
of knobs we chose: `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` supplies its three subgraphs, their BM25
channels, a BFS channel for facts and entities, `rrf_k=1`, `dense_min_score=0.6` and a BGE
cross-encoder reranker as one object. It is the paper's §4.1 operating point, and the run's stamp
records `reranker: CrossEncoderReranker` with `degradations: []` — that reranker is the arm's whole
identity, and a silent fall back to no reranking would have turned this into a different upstream
recipe while still filing under this name.

Item *count* at read time is 30 for Nemori, Mem0 and Zep, and 10 for A-Mem, so the spread in the
last column is not a k difference — it is what a "memory" is in each system. Mem0's unit is an atomic
fact averaging **46.0 characters** over all 5,427 of them (per-conversation means 43.2–48.9);
Nemori's is a narrative episode.

### Closing our link-expansion deviation: +1.36 J, and a different mechanism

A-Mem's read path expands each retrieved note to its linked neighbours. Upstream gives **every hit
its own budget** — the loop appends and only then breaks on `if j >= k` (`memory_layer.py:895`), so
a hit may contribute up to k+1 neighbours — while ours spent a single budget of 5 across all hits.
That was our deviation, and it was the binding one:

| | global 5 (ours, before) | per hit (upstream's shape) | Δ |
|---|---|---|---|
| J | 59.87 | **61.23** | **+1.36** |
| F1 | 41.41 | 42.92 | +1.51 |
| BLEU-1 | 36.51 | 38.03 | +1.52 |
| notes served / question | 15.0 (13–15, saturated) | 26.7 (13–43) | +78% |
| context to the answerer | 1,913 tok | 3,322 tok | +74% |
| eval cost | $0.6704 | $1.0905 | +$0.42 |

**Whose cap actually binds is the surprise.** Upstream's per-hit cap can never fire at k=10, because
a note cannot hold more than 5 links — the write path retrieves only 5 neighbour candidates
(`memory_layer.py:755`; ours `AMemOrganizer(top_k=5)`). Measured on this store: 5,882 notes, 18,886
links, mean **3.21**, and the distribution stops dead at 5 with nothing above it. So upstream's
shape means "serve every link of every hit", and the real comparison is **~32 candidates versus 5**,
not 110 versus 5. Ours was the only cap doing any cutting, and it saturated on every question.

The mechanism is unlike the other two read-path findings in this document, and the category
breakdown shows it cleanly:

| category | Δ J |
|---|---|
| open-domain | **+5.21** |
| single-hop | +1.43 |
| multi-hop | +1.42 |
| temporal | **+0.00** |

The embedder re-base and the query-rewrite ablation both gained *least* on open-domain, because a
better-targeted search cannot find an answer that was never stored. This one gains *most* there,
and nothing at all on temporal. It is a **breadth** effect rather than a precision one: abstention
barely moves (26.9% → 25.1%) and accuracy-when-answered is flat (82.0% → 81.7%), so the extra 1,409
tokens per question mostly help questions answerable from background rather than from a pinpointed
fact. Paying 74% more context for +1.36 J is a poor deployment trade and a necessary fidelity one —
those are different questions, and this table answers the second.

One honest gap: the duplicate half of this deviation stays unpriced. Upstream serves a duplicate
link twice and burns a slot each time; our `RetrievalPipeline._assemble` dedups on
`(memory_type, id)` before anything reaches the bundle, an invariant every methodology here depends
on. Reproducing it would mean changing shared machinery for one arm, so it is disclosed rather than
measured.

### A-Mem's own query rewrite costs it 5.26 J points

A-Mem's paper describes retrieval as dense top-k over the notes plus a one-hop link expansion, with
no LLM call. Its evaluation harness does something else: `answer_question` begins with
`keywords = self.generate_query_llm(question)` and then searches with *those keywords instead of
the question* (`test_advanced.py:129,134`, and identically in the robust copy) — so
`"When did Caroline go to the LGBTQ support group?"` reaches the embedder as
`"Caroline, LGBTQ, support group, when"`. We reproduce that, which is why A-Mem is the only arm
here paying a read-side LLM call. This section prices the step.

Single variable, same ingested store, same judge, everything else byte-identical in the stamp:

| | keyword rewrite (upstream, headline row) | raw question (`amem_rawq`) | Δ |
|---|---|---|---|
| J | 59.87 | **65.13** | **+5.26** |
| F1 | 41.41 | 46.21 | +4.80 |
| BLEU-1 | 36.51 | 40.86 | +4.35 |
| read-side LLM calls | 1,986 | 0 | −1,986 |
| eval cost | $0.6704 | $0.6333 | −$0.037 |

**The published system pays an LLM call per question to score worse than the retrieval its own
paper describes.** The mechanism is coverage, read-side: abstention falls
26.9% → 21.6% while accuracy-when-answered barely moves (82.0% → 83.0%), and the retrieved sets
genuinely differ — the top-1 hit is the same for only 65.3% of questions, mean Jaccard 0.598. Gains
concentrate where retrieval decides the outcome (single-hop +6.30, multi-hop +6.03) and are
smallest on open-domain (+3.12), where the answer is not in memory to be found. That is the same
signature the embedder diagnostic produced, and for the same reason: replacing a sentence with its
keywords throws away the sentence-level semantics a 1,536-dimension embedder can use.

Both arms stay wired (`amem` and `amem_rawq` in `scripts/repro/configs.py`) — a knob whose effect is
measured is worth keeping addressable even if one setting is later retired.

Read it as an operating-point finding rather than a verdict on the design. It is measured at
`gpt-4o-mini` + `text-embedding-3-small`, and a keyword query may well have been the better trade
against the weaker embedders of the paper's era — the step is not obviously wrong, it is
**unmeasured upstream and no longer earning its cost here**. For anyone deploying A-Mem the
actionable form is short: the rewrite is a knob, and turning it off bought +5.26 J and −1,986 calls.

### The two read-path changes are not additive

All four cells were run, so the interaction is measured rather than assumed. Same store, same judge,
one seed each:

| | global-5 cap | per-hit cap | effect of the cap |
|---|---|---|---|
| **keyword rewrite** (upstream) | 59.87 | **61.23** ← headline | +1.36 |
| **raw question** | 65.13 | **65.58** | +0.45 |
| *effect of dropping the rewrite* | +5.26 | +4.35 | |

Adding the two deltas predicts 66.49; the measured value is **65.58**, an interaction of
**−0.91**. Both changes buy the same thing — coverage — so the second one to arrive finds less
left to buy. The abstention series shows it directly: 26.9% → 21.6% when the query is fixed, then
only → 21.0% when the budget is also opened, with accuracy-when-answered flat at 83.0% across both.

**A deployment reading, which is not the same as the fidelity reading.** The best cell is raw
question + per-hit at 65.58, but raw question + global-5 gets 65.13 for **60% of the context**
(1,937 vs 3,283 tokens/question, $0.6333 vs $1.0345 per eval). Paying 69% more context for +0.45 J
is a bad trade. The headline row above is chosen for lineage fidelity, not because it is the
configuration anyone should deploy — and the arm that best serves a deployment is the least
faithful of the four.

Read-side choices, stacked, move A-Mem across a 15-point range — worth knowing before reading any
single A-Mem number in this repository or elsewhere:

| A-Mem configuration | J |
|---|---|
| `all-MiniLM-L6-v2` + keyword rewrite + global-5 cap | 50.00 |
| `text-embedding-3-small` + keyword rewrite + global-5 cap | 59.87 |
| `text-embedding-3-small` + keyword rewrite + **per-hit cap** (**the row above**) | **61.23** |
| `text-embedding-3-small` + raw question + global-5 cap | 65.13 |
| `text-embedding-3-small` + raw question + per-hit cap | **65.58** |

### What each write path did — the evolution log

Counts are operations actually applied (or judged and declined), summed over all 10 conversations,
from each run's `*.ops.jsonl`. A memory snapshot cannot produce this table: an UPDATE, a DELETE and
a NOOP all leave a store that looks the same afterwards.

| arm | operations | retrievable items at the end |
|---|---|---|
| A-Mem | ADD 5,882 · LINK 5,866 · UPDATE 16,342 | 5,882 notes |
| Nemori arm A | ADD 555 episodes + 1,926 semantic · MERGE 223 · INVALIDATE 223 | 2,704 |
| Nemori arm B | ADD 763 episodes + 1,745 semantic · MERGE 22 · INVALIDATE 22 | 2,530 |
| Zep | ADD 8,778 facts + 2,599 entities + 1,243 communities · UPDATE 4,373 facts + 895 entities · **INVALIDATE 1,293** | 11,377 (8,778 facts + 2,599 entities) |
| Mem0 | ADD 5,654 · **NOOP 26,209** · UPDATE 1,077 · DELETE 227 | 5,427 facts |

`ADD:episodic` is excluded from the table above: all five arms log exactly 5,882 of them, one per
turn, because that row is our harness's own raw-turn write rather than anything the methodology
decided. Where the arms differ is whether those turns *survive* as retrievable items — A-Mem and
Mem0's stores keep them (they are the `episodic` entry in each store's per-type counts), while
Nemori's keep none, its turns having been folded into episodes. Zep keeps its raw episodes too
(verbatim-loss defense) but does not retrieve them: its recipe serves only the three subgraphs.

Zep's 1,293 INVALIDATEs are the paper's flagship bi-temporal mechanism firing — an edge whose
`valid_at` is strictly older than a contradicting new fact gets its `invalid_at` stamped rather than
being deleted. This is the first count of it we have. What it bought is the subject of the temporal
row below, and the answer is not what the mechanism's prominence predicts.

Mem0's 2,945 decision calls returned 33,167 semantic verdicts, **79.0% of them NOOP** — the model
was asked what to do about a candidate fact and answered "nothing" four times out of five. That
ratio is the reason this repo added `OpType.NOOP` to the core vocabulary for this track: a judged
non-mutation is a log row, and counting it as a discard or not counting it at all both lose the
measurement.

## Where the deficits come from

The gap is not the answering model being wrong more often. It is the answering model **not being
given the answer**. Same prompt, same judge, same 1,540 questions:

| arm | abstained ("no information available") | correct when it did answer | J |
|---|---|---|---|
| Nemori arm A | 19.5% | 84.0% | 67.60 |
| Nemori arm B | 20.6% | 82.9% | 65.78 |
| A-Mem | 26.9% | 82.0% | 59.87 |
| Zep | 32.2% | **62.5%** | 42.73 |
| Mem0 | **55.6%** | 71.7% | 31.82 |

`J ≈ (1 − abstention) × accuracy-when-answered` holds on every row (Mem0: 0.444 × 0.717 = 31.8).
Decomposing Mem0's 35.8-point deficit against Nemori arm A: roughly **26 points are coverage** and
**10 points are accuracy**. Retrieval itself never failed — every one of the 1,986 questions got
exactly 30 facts back, none empty — so this is a property of what the write path chose to keep and
how small the kept units are, not a retrieval bug.

The shape is counter-intuitive and worth stating plainly: **Mem0 stores twice as many retrievable
items as Nemori and retrieves the maximum k, and still hands the answerer a quarter of the
context.** More memories, less memory.

**Zep fails differently, and it is the only arm that does.** Its abstention (32.2%) sits between
A-Mem and Mem0, roughly where its context size predicts. Its accuracy *when it answered* — 62.5% —
is the **lowest of the five**, below even Mem0's 71.7%. Every other arm in this table answers
correctly 72-84% of the time once it has something to say; Zep does not. So its deficit is not only
coverage: it also gets the question wrong more often having decided to try.

The candidate explanation is what Zep hands over. The other four arms serve text the participants
actually produced — raw notes, narrative episodes, extracted sentences. Zep serves an *abstraction*:
SCREAMING_SNAKE_CASE triples and LLM-written community summaries, with the raw episodes deliberately
excluded from its recipe. Something is lost in that projection, and the paper says so itself about
its own weakest category (§ single-session-assistant, "verbatim detail is lost in the abstraction").
This table does not isolate that — doing so would mean serving Zep's episodes alongside its
subgraphs, which is a different recipe and a different run.

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
4b. **A-Mem's read-path deviations are now closed or disclosed, not outstanding.** The
   link-expansion cap shape was ours and is fixed in this row (+1.36 J, ledger C-7); the duplicate
   half of it is structurally unreproducible here and is disclosed instead. `13-amem-study.md` §6-2
   previously listed the keyword rewrite as a second deviation of ours; that was wrong and is
   corrected at the source — the rewrite is upstream's own step (ledger B-8).
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
   on MiniLM (which replicated, and which the embedder change then reversed). The A-Mem read-path
   ablation is single-seed too; its +5.26 J is large against the ±0.35 pp per-arm seed stability
   Track 1 measured, but that stability figure was measured on Nemori, not on A-Mem.
8. **Zep's row carries four conditions the others do not.** (a) Its write temperature is **0**,
   dated from pypi wheels to the paper's own release rather than read off the pinned clone, whose
   `DEFAULT_TEMPERATURE = 1` arrived with the release that made `gpt-5-mini` the default model —
   GPT-5 accepts only 1, so that constant tracks the model, not extraction. (b) It ran under a
   **0.1% drop tolerance** (`max_drop_rate`, stamped), not the zero-drop bar the other four met: at
   ~2,745 structured calls per conversation a clean conversation is a 37% event, and the binary gate
   made a healthy run wipe and re-pay. All ten conversations came in at **zero drops** anyway. (c)
   Its reranker is the only non-`Noop` one in the table, and it ran on GPU. (d) It is the only arm
   whose retrieved units are **abstractions rather than participant text** — see the accuracy
   discussion above.

9. **The ordering of the top three is not separated by the evidence.** A paired bootstrap over the
   per-question verdicts puts Nemori arm A over arm B at ΔJ +1.82 pp with a 95% CI of
   **[-0.32, +3.90]** (p = 0.097) — the interval includes zero. Arm B over A-Mem (+4.55) and A-Mem
   over the rest are separated. Re-scoring without the 99 questions an external audit flags as
   score-corrupting moves every arm up 1.07-2.68 pp and changes **no rank** (adjacent-pair flip
   probability ≤ 0.0001). Both analyses, their method, and the reruns are in
   [`research/locomo-gold-audit-replay.md`](research/locomo-gold-audit-replay.md).

10. **What the ranking is sensitive to.** Three of the five rows sit inside a 2.5-point band
   (67.60 / 65.78 / 65.13) and two harness choices — the embedder and the read query — moved a
   single arm by 15.13 points across this campaign. Treat the ordering among those three as
   unresolved at this precision; the gap that *is* robust is the one to Mem0.

## Per category — and the one result that runs against a paper's own claim

Same 1,540 judged questions, same judge, same category labels from LoCoMo's gold. The counts are
identical down every column (841 / 282 / 321 / 96), which is what makes the columns comparable at
all; it was checked rather than assumed before this table was written.

| arm | single-hop (841) | multi-hop (282) | temporal (321) | open-domain (96) |
|---|---|---|---|---|
| Nemori arm A | **75.51** | **69.86** | 56.39 | **29.17** |
| Nemori arm B | 73.60 | 65.25 | 57.32 | 27.08 |
| A-Mem | 65.64 | 57.45 | **62.62** | **29.17** |
| Zep | 47.32 | 47.16 | 33.64 | 19.79 |
| Mem0 | 34.96 | 32.27 | 26.48 | 20.83 |

**The temporal column is the finding.** Zep is the only arm here built around time: a bi-temporal
graph that separates when a fact became true from when the system learned it, and an invalidation
rule that fired 1,293 times during this ingest. It places **fourth of five** on temporal questions,
**29 points below A-Mem**, which has no temporal mechanism whatsoever — A-Mem's notes carry a
timestamp and nothing reads it.

Three things this does NOT say, each worth stating because the result invites all three:

- **It is not a failed reproduction of Zep's published number.** Zep reports 71.2 on LongMemEval and
  **no LoCoMo figure at all**. There is nothing here to disagree with.
- **It is not evidence the mechanism is broken.** The invalidations happened; `resolve_edge_
  contradictions` and `expire_new_edge` transcribe upstream's truth table and are tested against it.
  What the row shows is that firing 1,293 times did not convert into answering LoCoMo's temporal
  questions, which is a claim about this benchmark and this operating point.
- **It is not isolated from the arm's general deficit.** Zep is fourth on *every* category. Temporal
  is its worst relative showing, not its only one, and a mechanism-specific conclusion would need an
  ablation (Zep with invalidation off) that this campaign did not run.

What it does say is narrower and still worth having: **on LoCoMo, at the paper's own §4.1 operating
point, an explicit temporal knowledge graph did not beat a flat note store on temporal questions.**
Any claim that temporal structure is what buys temporal accuracy now has one measurement standing
against it, and the benchmark it was measured on is one its authors chose not to report.

## Artifacts

Every number above resolves to a committed file under `results/repro/`:

| arm | ingest summary | eval summary |
|---|---|---|
| A-Mem (headline: keyword rewrite + per-hit cap) | `gpt-4o-mini_all_ingest_e3s.json` | `gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH.json` |
| A-Mem (ablation: global-5 cap) | same ingest — one store, four read protocols | `gpt-4o-mini_all_k10_ours_expand-on_run1_e3s.json` |
| A-Mem (ablation: raw question, global-5 cap) | " | `gpt-4o-mini_amem_rawq_all_k10_ours_expand-on_run1_e3sRAWQ.json` |
| A-Mem (ablation: raw question, per-hit cap) | " | `gpt-4o-mini_amem_rawq_perhit_all_k10_ours_expand-on_run1_e3sRQPH.json` |
| Nemori arm A | `gpt-4o-mini_nemori_upstream_all_ingest_e3sA.json` | `gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA.json` |
| Nemori arm B | `gpt-4o-mini_nemori_merge085_all_ingest_e3sB.json` | `gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB.json` |
| Mem0 | `gpt-4o-mini_mem0_v0194_all_ingest_e3sM.json` | `gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM.json` |

Each carries its per-conversation summaries, the full LLM I/O trace, the memory snapshot, the
evolution log (`*.ops.jsonl`) and the per-question records with retrieved chunks. The `k10` segment
in the eval filenames is a naming default, not the k that ran — the k that ran is recorded per
question inside the records file, and is the one in the read-path table above.
