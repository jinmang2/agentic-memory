# Fidelity round 12 — 8-way fresh-eyes re-audit (2026-07-28)

8개 방법론을 방법론당 1개의 독립 감사 에이전트로 병렬 재감사한 원문 보고 모음.
각 에이전트는 우리 연구 문서를 증거로 쓰지 않고 ~/.agmem/upstream 클론을 직독했다.
지시: 결함 유형 5종 헌팅([[audit-defect-classes]] + 사문 knob + 동명 함수 다른 수식), 수정 금지.
주의: 아래는 단일 에이전트 보고 원문이며 아직 교차 검증 전이다. 조치 전 각 건의 재현 확인 필요.

> **상태 (2026-08-08 추가).** 이 문서는 **감사 시점의 스냅샷**이고, 이후 조치를 반영하지 않는다.
> 특히 각 섹션 말미의 "load-bearing items to act on later" 목록은 **당시** 열려 있던 것이며 지금은
> 상당수가 닫혔다 — 예: `[ace-rb]` #18(G-Memory→RB score 오염)은 커밋 `6fad7bb`에서, #14의 나머지
> 절반(judge 게이트)은 이번 라운드에서 닫혔다. **현재 상태의 정본은 `docs/17-defect-ledger.md`와
> `docs/research/upstream-defect-catalog.md`이며, 이 문서는 그 근거 로그다.**
> 이 배너 자체가 이번 라운드의 자기교정 항목이다: 외부 리뷰어가 이 문서의 열린-항목 목록을 현재
> 코드 상태로 읽고 이미 닫힌 결함을 미조치로 보고했고, 그 오독의 원인은 문서에 닫힘 표시가 없다는
> 것이었다. 닫힌 항목이 열린 것처럼 읽히는 감사 문서는 그 자체로 결함이다.

---

# [amem]

All comparisons are done. Here is the fresh-eyes audit report.

---

# A-Mem fidelity re-audit (organizer + LinkExpansion vs WujiangXu/AgenticMemory)

**Upstream state verified first:** `/home/jinmang2/.agmem/upstream/AgenticMemory` is `WujiangXu/AgenticMemory`, single squashed commit `0c8039f` (2026-03-05) containing **two variants**: the plain edition (`memory_layer.py` + `test_advanced.py`, the paper-repro path) and the robust edition (`memory_layer_robust.py` + `llm_text_parsers.py` + `test_advanced_robust.py`, added post-paper). The agiresearch library edition and A-mem-sys are **not** in `~/.agmem/upstream`, so docstring claims about those repos could not be re-verified here (noted below).

## Findings

**1. [behavior] First-note evolution call is skipped by us; the plain (published-numbers) edition spends it.**
Ours: `src/agmem/organizers/amem/organizer.py:215-216` — `if not neighbors: return ops` (no Ps3 call when the store is empty). Upstream plain: `memory_layer.py:753-758` — `find_related_memories` returns `("", [])` for the first note (`memory_layer.py:861-862`), but `process_memory` still formats the prompt with an empty neighbor block and **makes the evolution LLM call** (and `strengthen` can then extend `note.links` with dangling integer indices into an empty memory list, `memory_layer.py:833-835`). The robust edition skips like we do (`memory_layer_robust.py:473-474`). Why it matters: (a) per-conversation LLM call counts vs the plain edition differ by 1 (relevant to this project's cost-parity claims); (b) our behavior silently matches the *robust* edition here while the module docstring frames the batched call as matching the plain edition's single `process_memory` call — we don't SAY which variant this branch follows (defect class 1/5).

**2. [behavior] New-note tag refinement has an undocumented guard upstream doesn't have.**
Ours: `organizer.py:278-279` — `if new_tags and new_tags != note.tags` (no-op verdicts and empty-tag verdicts are dropped). Upstream plain: `memory_layer.py:834-836` — `note.tags = new_tags` **unconditionally**, including replacement by `[]` (strict schema requires the key but not non-emptiness); robust guards emptiness only (`memory_layer_robust.py:506-507`), not equality. Protective and probably right, but it is not in the module docstring's deviation list, and the equality guard also suppresses ops (op-count/evolution-log comparisons — the same concern the `organizer.py:304-310` comment applies to neighbor updates is not applied symmetrically in prose here).

**3. [behavior] LINK application dedups and lexicographically sorts links; upstream keeps append-order duplicates.**
Ours: `src/agmem/memory.py:661-663` — `merged = set(old)|set(new); data[key] = sorted(merged)`. Upstream: `note.links.extend(...)` (`memory_layer.py:835`, `memory_layer_robust.py:505`) — insertion order, duplicates accumulate across evolutions. This matters because the read path consumes links **in order under a cap**: `LinkExpansion` (`src/agmem/retrieval/steps.py:107-111`) iterates `data["links"]` and stops at `cap`, so when a note has more links than the cap, *which* neighbors get expanded differs from upstream (upstream = first-linked wins, `memory_layer.py:889-897`; ours = lowest UUID wins). The `LinkExpansion` docstring documents the per-hit-vs-global cap deviation but not the ordering/dedup facet.

**4. [doc] docs/16-abstraction-study.md session 1 undercounts residual deviations ("잔여 편차 1건").**
`docs/16-abstraction-study.md:23-24` claims the link-cap is the only residual deviation. But the project's own deep audit records another live behavioral deviation — immediate re-embedding of evolved neighbors vs upstream's stale-until-`consolidate_memories()` index (`docs/research/fidelity-deep-audit.md:40-41`, judged benign/superior but still a "변형" that changes neighbor selection during ingest) — and findings 1–3 above add more. The module docstring (`organizer.py:5-20`) likewise omits the re-embed-timing deviation from its deviation list. No code change implied; the summary claim is just too strong.

**5. [doc] fidelity-deep-audit.md §1.1 "strengthen" row describes code that no longer exists.**
`docs/research/fidelity-deep-audit.md:38` says our strengthen adds a **reverse LINK (bidirectional)** and **does not apply `tags_to_update`**. Current code is the opposite on both points: links are unidirectional ("unidirectional, as upstream", `organizer.py:264-272`) and `new_note_tags` **is** applied via a second UPDATE op (`organizer.py:278-297`, explicitly citing "audit P1-5"). The audit doc is a historical record that drove the fix, but nothing in it marks that row as superseded; a reader cross-checking claims (as I was instructed to) is initially misled. (Defect class 3.)

**6. [question] Embedded-document format follows the paper, not either upstream code — stated only in the deep audit, not at the code site.**
Ours: `src/agmem/core/types.py:91-100` — `content \n keywords \n tags \n context`, no labels (paper eq.(3) order). Upstream plain/robust: `"content:"+c+" context:"+X+" keywords: "+K+" tags: "+G` with labels, order c,X,K,G (`memory_layer.py:722`, `memory_layer_robust.py:387-392`), and after any `consolidate_memories()` the corpus silently switches to a *third* format (`content , context keywords tags`, `memory_layer.py:749-751`). The deep audit records this (`fidelity-deep-audit.md:34`) and judges it low-impact; the `Note.embedding_text` docstring says only "A-Mem finding". Decision needed only on whether the code-site comment should name the variant divergence; no behavior bug.

**7. [question] Docstring claims about repos not retained locally are unverifiable in this audit.**
`organizer.py:21-25` ("agiresearch add_note never calls analyze_content"; issues #23/#24/#32 characterizations) and `steps.py:96-97` ("agiresearch k per hit") reference `agiresearch/A-mem`, which is not under `~/.agmem/upstream` (only 9 upstreams kept, A-Mem represented solely by WujiangXu/AgenticMemory). Prior sessions verified these (fidelity-deep-audit §1.3); I could not independently re-verify. Everything checkable against the retained repo checked out (see below). If the audit-technique standard is "official function detachable for output diff," consider retaining the agiresearch snapshot too.

## Verified clean (explicit function-pair / constant comparisons)

- **`import re` absence / Ps1 death in plain edition** (`organizer.py:21-25` claim): confirmed. `memory_layer.py:1-19` has no `import re`; `analyze_content:380` raises NameError, the bare `except:` at 382 then hits undefined `e` at 383, the outer handler at 393 returns `{"keywords": [], "context": "General", "tags": []}` — every plain-edition note gets empty metadata after a wasted Ps1 LLM call. Robust edition parses properly (`memory_layer_robust.py:317-345`).
- **Write-path neighbor k=5** hardcoded in both editions (`memory_layer.py:755`, `memory_layer_robust.py:471`) vs our `top_k=5` default (`organizer.py:161-164`); the "paper's k=10 is the QA retrieval k" claim matches `test_advanced.py:417` (`--retrieve_k` default 10).
- **Evolution query text deviation is exactly as documented**: both upstream editions query with `note.content` only (`memory_layer.py:755`, `memory_layer_robust.py:471`); ours uses `embedding_text()` (`organizer.py:209`), declared at `organizer.py:16-18`.
- **Order of operations**: upstream runs evolution *before* the note enters memory/retriever (`add_note`, `memory_layer.py:714-722`); ours searches neighbors before ops are applied (ops applied post-return by `memory.py:_apply_ops`) — new note is never its own neighbor in either. ADD→LINK→UPDATE op order within the batch is safe (`memory.py:565-566` sequential).
- **Single batched evolution call** matches the plain edition's one `process_memory` call (`memory_layer.py:753-857`); robust's 3-call split correctly described in our docstring lineage notes.
- **Empty-`actions` semantics claim**: plain edition with `should_evolve=true, actions=[]` does nothing (loop over empty list, `memory_layer.py:830-831`) — our documented fallback-to-both (`organizer.py:245-247`) is a declared deviation, and the docstring's "upstream treats it as a no-op" is accurate.
- **`should_evolve` gating, action names** (`strengthen`/`update_neighbor`) — identical semantics (`organizer.py:243-247` vs `memory_layer.py:828-837`).
- **Unidirectional links**: only the new note gains links in both (`organizer.py:264-272` vs `memory_layer.py:835`).
- **Keywords immutable under `update_neighbor`** in both (`organizer.py:313-319` preserves `old["keywords"]`; upstream only touches tags/context, `memory_layer.py:844-856`).
- **LinkExpansion k+1 off-by-one claim** (`steps.py:96`): confirmed by trace of `memory_layer.py:889-897` and identically `memory_layer_robust.py:447-458` — the add-then-check loop emits k+1 neighbors per hit.
- **`is_servable` applied in LinkExpansion** (`steps.py:117`) — tombstone/invalidate ghost-hit fix present; invalidated notes also can't become evolution neighbors because INVALIDATE removes the vector (`memory.py:652-654`, notes not in `BITEMPORAL_TYPES`).
- **`get_items` preserves request order** (`sqlite_doc.py:250-260`) — neighbor prompt ordering equals similarity ordering, matching upstream presentation.
- **"talk start time" semantics**: our `talk_time = meta["date"] or ingest timestamp` (`organizer.py:179`) mirrors upstream's `timestamp=time` param fed from session `date_time` (`test_advanced.py:303-305`).
- **Bench keyword query rewrite** kept verbatim including the 'cosmos' quirk (`src/agmem/bench/locomo.py:52-59` == `test_advanced.py:96-103`).
- **Upstream dead code confirmed**: `HybridRetriever` (BM25, `alpha`) is never instantiated by either `AgenticMemorySystem`; `evo_threshold` is real (not dead) in upstream but obsolete in ours by design (immediate re-embed), as the deep audit records.
- **`_verdict_id` / dedup / bare-id-skip robustness paths** (`organizer.py:123-147`, 254-262, 299-310) — ours-only additions, each self-documenting with accurate rationale; no contradiction between docstring and code found in the current organizer module.

Nothing was edited; no tests were run.

---

# [nemori]

## Nemori fidelity re-audit — full report

Ours: `/home/jinmang2/agentic_memory/src/agmem/organizers/nemori/` (organizer.py, stages.py). Upstream: `/home/jinmang2/.agmem/upstream/nemori`. Read-only audit; nothing modified.

### Re-verification of the two recent docstring claims

- **Claim (b) — VERIFIED against upstream code.** `~/.agmem/upstream/nemori/nemori/llm/generators/semantic.py:59-71` constructs a fresh `SemanticMemory` per extracted statement; `nemori/domain/models.py:138` defaults `id` to a new `uuid.uuid4()`; `nemori/db/semantic_store.py:20-39` implements `save` as `INSERT ... ON CONFLICT (id) DO UPDATE` — an id-keyed upsert that, with always-fresh ids, is effectively append-only. `save_batch` just loops `save`. There is no read-modify, merge, or conflict path anywhere in the upstream repo, so the docstring's statement that `ThreeWayIntegrator` (ours, `stages.py:506`) is the paper text's only implementation is accurate with respect to this snapshot.
- **Claim (a) — CONSISTENT with upstream (paper itself not fetchable here).** Grep across upstream for boundary logic finds only the `boundary_reason` prompt slot (`nemori/llm/prompts.py:17-18`, fed the segment *topic* at `nemori/core/memory_system.py:121-123`); the repo contains exclusively `BatchSegmenter` (`nemori/llm/generators/segmenter.py`) — no per-message boundary detector exists anywhere in the deployed code. Nothing in upstream contradicts "the current text has no per-message mode; §3.2.1 is the batch form".

### Numbered findings

**1. [behavior] The "upstream" preset enforces a merge-similarity floor (0.85) that upstream's deployed code never applies — upstream's `similarity_threshold` is a dead knob.**
- Ours: `src/agmem/organizers/nemori/organizer.py:100` (`merge_similarity=0.85` in the `upstream` preset) → `src/agmem/organizers/nemori/stages.py:340-341` actively filters candidate hits to `s >= 0.85` before the merge-decision LLM call.
- Upstream: `nemori/llm/generators/merger.py:27,34` stores `_similarity_threshold=0.85` but `_find_similar` (merger.py:69-80) passes the top-5 qdrant hits to the LLM with **no** similarity filtering; grep confirms `_similarity_threshold` is never read anywhere. `nemori/config.py:86` (`merge_similarity_threshold`) is plumbed only into this dead field via `nemori/factory.py:55`.
- Why it matters: this is the exact defect class that caught the MemoryOS eviction mislabel — a preset labeled with a lineage whose code says otherwise. Under our "upstream" preset most merge-decision LLM calls are suppressed (top-5 cosine rarely clears 0.85), so merge rate and LLM call counts diverge from the deployed system the preset claims to replicate. The comment at `organizer.py:68-69` ("upstream: ... similarity 0.85, top-5, >1h gap ban") repeats config values, not code behavior.

**2. [behavior] Window semantics differ from the path that produced the published numbers; our `chunk_max=80` is unreachable (dead knob in ours).**
- Ours: `stages.py:174-183` — `BatchPartitioner.push` partitions as soon as `len(buffer) >= window` (20), so `_partition` (stages.py:196-199) always sees exactly 20 messages; the `range(0, len, chunk_max=80)` loop can never produce more than one chunk, and episodes can never span a 20-message boundary.
- Upstream: `nemori/core/memory_system.py:76-83` triggers `_process` at `count >= buffer_size_min` (eval config: **1**), and `_process` (memory_system.py:105-113) grabs **all** unprocessed messages, gating segmentation at `batch_threshold=20`. In `evaluation/locomo/add.py` (default `--batch-size 1`, fire-and-forget tasks + per-user lock, add.py:150-160, 227), adds race ahead of the first LLM call, so later `_process` calls grab most of the conversation and segment it in chunks of up to 80 (`nemori/llm/generators/segmenter.py:15,30-40`). Upstream's 20 is a *minimum gate on the accumulated backlog*, not a fixed window size.
- Why it matters: the segmentation LLM sees ~80-message contexts in the published runs vs exactly 20 in ours; episode boundaries and counts differ systematically. Additionally, upstream's `buffer_size_max` is validated but never read (`nemori/config.py:73,101`) — the eval config's `buffer_size_max: 20` is itself a dead knob (defect class 4, upstream side).

**3. [behavior] Missing `episode_min_messages` filter.**
- Upstream: `memory_system.py:118-119` silently *skips* groups smaller than `episode_min_messages` — library default 2 (`config.py:76`), meaning singleton groups are dropped and those messages lost.
- Ours: `stages.py:210-222` — `BatchPartitioner` never drops any group; 1-message segments become episodes.
- Why it matters: the eval configs set `episode_min_messages: 1` (`evaluation/locomo/config.json`, `evaluation/longmemeval/config.json`), so ours matches the published-numbers path but **not** the library-default lineage the preset comment ("code values") points at. One line of provenance is missing.

**4. [behavior] Calibration source after a merge.**
- Upstream: when a merge fires, semantics are regenerated from the **merged** episode, whose `source_messages` are target + new combined (`merger.py:133`, `memory_system.py:150-156`, `semantic.py:94-97`), so the prediction gap is computed over both episodes' raw messages.
- Ours: `organizer.py:480-485` builds `plain_text` from the **new segment only**, regardless of merge outcome (title/narrative are correctly the merged ones).
- Why it matters: on v4/upstream presets, whenever a merge happens our calibration compares the prediction against half the raw material upstream uses — different distilled facts.

**5. [behavior/question] Prediction-context K: 10 vs the published 20.**
- Ours: `organizer.py:344` — `semantic_top_k` defaults to 10 (annotated "repo search_top_k_semantic=10"); no preset overrides it.
- Upstream: library default is indeed 10 (`config.py:91`), but **both** eval configs set `search_top_k_semantic: 20`, and `memory_system.py:170-173` uses that same value for the predict-stage retrieval — the published numbers ran predict with top-20 knowledge.
- Why it matters: if the "upstream" preset means "reproduce published behavior" (defect class 2), it should carry `semantic_top_k=20`; if it means library defaults, the docstring should note the eval deviates.

**6. [doc/question] "PR#19 id-reuse dedup" is attributed to upstream code the archived snapshot does not contain.**
- Ours: `stages.py:12-13` ("the upstream *code* ships only append + PR#19 dedup"), `stages.py:524`, and `DedupIdReuseIntegrator` (`stages.py:632-639`, threshold 0.85).
- Upstream: the snapshot at `~/.agmem/upstream/nemori` has **no** dedup path anywhere; the only related artifact is the dead config knob `semantic_similarity_threshold=0.85` (`config.py:82`, never read). The 0.85 coincidence is suggestive, not evidence.
- Why it matters: the lineage claim rests on a GitHub PR not present locally and thus unverifiable in this audit. Either archive PR#19 alongside the snapshot or soften the docstring to "proposed in PR#19, not in the deployed snapshot".

**7. [behavior] (minor) Segment topic is discarded.**
- Upstream: the segmenter's per-group `topic` is threaded into the episode prompt as `boundary_reason` (`memory_system.py:121-123`, `prompts.py:17-18`).
- Ours: `BATCH_SEGMENT_SCHEMA` requests `topic` (`stages.py:129-131`) but `BatchPartitioner._partition` throws it away, and `EPISODE_PROMPT` (`organizer.py:132-151`) has no boundary-reason slot.
- Why it matters: the episode generator loses a conditioning signal it has upstream; the schema's `topic` field is a dead output in ours.

**8. [behavior] (minor) Merge-decision prompt drift.**
- Upstream: shows the new episode's **content only** (no title), candidates as ID + time + message counts + content **truncated to 200 chars** (`merger.py:87-94,173-183`); target selected by ID string.
- Ours: shows full title/narrative/timestamp for the new episode and **full** candidate narratives; target selected by index (`stages.py:349-371`).
- Why it matters: same decision framework, but the LLM sees materially more candidate text in ours — merge rates need not match. (The `>1h` gap ban itself is faithful: upstream hardcodes it in `MERGE_DECISION_PROMPT`, `prompts.py:403-406`; our conditional `TIME_GAP_RULE` at 1.0 (`stages.py:264-267`, `organizer.py:101`) renders the equivalent sentence for the upstream preset. Our v4 preset's `None` claims the paper has no time constraint — unverifiable here without the paper.)

**9. [doc] Dead knobs inside our own "upstream" preset.**
- `organizer.py:95-96` sets `buffer_min=2, buffer_max=25`, but with `segmenter="batch"`: `buffer_min` is documented-unused in `BatchPartitioner` (`stages.py:167-169`), and `buffer_max` is consumed only by `PerMessageBoundary` — `self.buffer_max` (`organizer.py:342`) has no reader on the batch path. Combined with finding 2's unreachable `chunk_max=80` (`organizer.py:97`), three of the preset's eight numbers are inert.
- Why it matters: they mirror upstream's own dead config (`buffer_size_max`, `episode_max_messages`, `semantic_similarity_threshold`, merger `similarity_threshold` — `config.py:73,77,82,86`, `merger.py:34` — all defined, never read), but the preset comment presents them as active "code values" without flagging the inertness on either side.

### Verified clean (function pairs / constants compared and found faithful)

- **Temperatures**: segmentation 0.2 (`segmenter.py:62`); everything else at request/client default 0.7 (`nemori/llm/orchestrator.py:38`, `nemori/llm/client.py:31`); eval answer generation 0.0 (`evaluation/locomo/search.py:168`, `evaluation/longmemeval/search.py:170`, `evaluation/longmemeval/evals.py:145`) — matches the organizer docstring's claim (`organizer.py:38-40`).
- **Upstream timestamp-parse fallback is `datetime.now()`** (`episode.py:64-69`) — our docstring's claim (`organizer.py:34-35`) is correct; our first-message-date fallback (`organizer.py:432`) is a disclosed deviation.
- **Cold-start switch**: our `if not known → direct extraction` (`organizer.py:506-515`) == upstream `if enable_pc and existing_semantics` (`semantic.py:51-54`); direct extraction reads the generated episode (title+content), not raw messages, in both.
- **Predict-stage retrieval**: query embedding = episode title+narrative in both (`organizer.py:501-505` vs `memory_system.py:169-176` using `episode.embedding` = `embed(f"{title} {content}")`, `episode.py:60`); only the separator differs (`"\n"` vs `" "` — also true of the episode `embedding_text`; negligible).
- **Calibration input format**: `role: content`, no timestamps, in both (`organizer.py:480` vs `semantic.py:94-97`).
- **Prompt substance**: `CALIBRATE_PROMPT` = 4 tests + 7 categories + time/date ban + present-tense atomic + quality-over-quantity == `EXTRACT_KNOWLEDGE_FROM_COMPARISON_PROMPT` (`prompts.py:93-156`); `DIRECT_EXTRACT_PROMPT` = 6 categories, no time ban == `SEMANTIC_GENERATION_PROMPT` (`prompts.py:160-238`, whose good examples indeed carry dates, lines 215-217); `EPISODE_PROMPT` keeps every mandatory element (third-person narrative, hour-precision time, relative→absolute parenthetical conversion + the hiking example, "not the current time") == `EPISODE_GENERATION_PROMPT` (`prompts.py:11-53`); `BATCH_SEGMENT_PROMPT` keeps the 5 signal classes, 30-min gap, <30% relevance, 2-15 messages, when-in-doubt-split == `BATCH_SEGMENTATION_PROMPT` (`prompts.py:241-308`).
- **Pipeline order**: narrate → merge-check → semantics-on-merged-episode (`organizer.py:444-485`) == `_process` (`memory_system.py:121-156`); semantic generation synchronous after merge in both.
- **Merge mechanics**: top-k 5 default (`config.py:87`); self-exclusion parity (upstream searches top_k+1 minus self because it saves before checking, `merger.py:74-77`; we check before ADD); every failure path → plain ADD (`stages.py:347-389`) == upstream's `(False, None, None)` (`merger.py:44-67`); earliest-timestamp default with LLM timestamp taking precedence when parseable (`merger.py:135-152` vs `stages.py:393-395`); merged provenance = union of source messages/ids.
- **Flush semantics**: below the 20 gate → single group, no segmentation LLM call (`memory_system.py:106-113` == `BatchPartitioner.flush`, `stages.py:185-194`).
- **Chunk stride 80** == `_SEGMENT_CHUNK_SIZE` (`segmenter.py:15`) — correct value, unreachable per finding 2.
- **Index hygiene**: ours 0-based vs upstream 1-based, each internally consistent; our leftover retention (`stages.py:220-222`) vs upstream silently dropping uncovered indices (`segmenter.py:76-83`) is a disclosed no-message-loss deviation, as is the within-batch supersession guard (no upstream counterpart needed — upstream hard-deletes).
- **Merger ON by default including the eval** (`factory.py:57`, `enable_episode_merging=True` default at `config.py:85`, both eval configs omit the key) — consistent with the organizer docstring's "ON by default in the repo's eval" (`organizer.py:27`).
- **Docstring self-consistency spot-checks (defect class 3)**: `organizer.py:36-37` buffer_max flush-includes-newest claim matches `stages.py:95-96`; v4 preset constants (merge_top_k=5, integrate_top_k=5, tau=0.70) match the `organizer.py:67` comment.

No file was modified; no tests were run.

---

# [memoryos]

# MemoryOS Fidelity Re-Audit — Report

Upstream clone at `/home/jinmang2/.agmem/upstream/MemoryOS` (HEAD 587ed77) contains both lineages: `memoryos-pypi/` and `eval/`. All citations below are verified against the files directly.

## Fresh fix re-verification (eviction relabel) — HOLDS

- pypi `mid_term.py:176-177`: `if len(self.sessions) > self.max_capacity: self.evict_lfu()` — line 177 is exactly the cited call. `evict_lfu` (`mid_term.py:71-75`) takes `min(self.access_frequency, key=self.access_frequency.get)` — an access-counter min, not a heat min.
- eval `mid_term_memory.py:119-120`: identical guard; line 120 is `self.evict_lfu()`. `evict_lfu` (`mid_term_memory.py:56-60`) is the same access-counter min.
- No heat-based eviction exists anywhere in either file; the heap is used only for the promotion peek. The paper's "lowest heat evicted" (§3.3) matches neither codebase — the relabel to `eviction="lfu"` in both presets (organizer.py:161,171) is correct, and keeping `"lowest_heat"` as an explicit non-lineage option is honest.
- DELETE reason: `organizer.py:925` emits `f"{self.eviction}_eviction"` — derived from the same attribute the eviction branch (organizer.py:912-916) dispatches on, so the reason names the policy that actually ran. Verified.
- Our "degrades to insertion-order FIFO at counter 0" comment is also faithful: upstream's `min` over a defaultdict resolves ties by insertion order too.

## Findings

**1. [behavior] The eval preset never forms pages — every message becomes its own page.** With `stm_capacity=1`, `on_message` (organizer.py:584-590) flushes after every single message: the user half is evicted alone, then the agent half opens a fresh page (`_pages`, organizer.py:462) and is evicted alone. Upstream eval receives *formed* `{user_input, agent_response}` pairs (`main_loco_parse.py:240-243`), so its pages are always full exchanges. Consequences: heat accrues ~2x (L=1 per *half*, i.e. 1.6 per exchange vs upstream 0.8), so the τ=5 promotion fires at roughly half the content — the exact wrong-unit defect class the 2026-07-27 audit fixed for capacity counting; topic-summary, continuity and meta_info calls run 2x per exchange; MTM pages are half-exchanges. The module docstring (organizer.py:39-41) frames flush-between-halves as an occasional boundary case; at capacity 1 it is systematic — 100% of exchanges.

**2. [behavior] Both lineages drop incomplete exchanges before MTM; we keep them.** pypi `updater.py:104` and eval `dynamic_update.py:126` both filter `if qa.get("user_input") and qa.get("agent_response")` — an exchange missing either side never reaches MTM. LoCoMo produces these routinely (consecutive same-speaker turns leave `agent_response == ""`, `main_loco_parse.py:169-183`). We evict every page regardless. Deliberate under the no-content-loss rule, but undocumented — the docstring documents the overwrite deviation (verified at `main_loco_parse.py:176-177`) and not this one. Combined with finding 1: upstream MTM contains *only* full exchanges; our eval preset contains *only* halves.

**3. [behavior] Merge candidate selection is a third scheme belonging to neither lineage.** pypi: argmax over ALL sessions of cos + Jaccard (`mid_term.py:206-226`). eval: top-1 by cosine only, then thresholds cos + containment-mean for that one candidate (`mid_term_memory.py:133-154`). Ours: top-3 by cosine, then argmax of the combined score (organizer.py:816-829). Under the eval preset we can merge into a session eval would never even consider; under pypi we can miss the true combined-score argmax when it sits outside cosine top-3. The "Consider top-3 candidates" comment presents this as fact, not as a deviation.

**4. [behavior] Our merge mutates the segment's matching key; upstream never does.** On merge we append the new summary to content, set `embedding_text = content[-2000:]` (re-embed) and union keywords (organizer.py:833-834, 852-853). Both lineages leave `summary`, `summary_embedding` and `summary_keywords` untouched on merge — only `details`/L change. So upstream's segment identity is frozen at creation while ours drifts, changing future merge comparisons and the read path's stage-1 gate. Undocumented.

**5. [behavior] `page_indexes` partition vs upstream's whole-batch-per-theme.** Both lineages call `insert_pages_into_session` once PER theme with the ENTIRE batch (`updater.py:180-185`, `dynamic_update.py:170-180`) — a multi-theme batch duplicates every page into each theme's session, and each target session gets `L += len(all pages)`. Our `TOPIC_SCHEMA` asks the model to partition pages into groups (organizer.py:804-809). Also, both upstream prompts cap at "a maximum of two themes" (`memoryos-pypi/prompts.py:73-74`, `eval/utils.py:128-133`); our `TOPIC_PROMPT` has no cap.

**6. [behavior] eval knowledge FIFOs don't exist upstream.** `eval/long_term_memory.py:9-10`: `knowledge_base` and `assistant_knowledge` are plain unbounded lists — no `deque(maxlen)`, no capacity anywhere in the eval lineage. Our `knowledge_capacity=100` FIFO with DELETE evictions (organizer.py:1000-1009) is a pypi trait (`memoryos-pypi/long_term.py:18-19`) imposed on the eval preset.

**7. [behavior] pypi's profile-write guard applied to eval; eval has none.** The `len >= 30` / `!= "none"` skip is pypi-only (`memoryos.py:186-188`); the eval driver writes `updated_profile` unconditionally (`main_loco_parse.py:53-57`). Ours applies the pypi guard under both presets (organizer.py:1101). Similarly the per-line "none"/"- none" filter is pypi's (`memoryos.py:196-204`); eval adds every split line unfiltered, including empties (`main_loco_parse.py:62-64`), and stores assistant knowledge as one un-split blob (`:67`).

**8. [doc] `UPDATE_PROFILE_PROMPT` provenance is wrong.** organizer.py:278 says it is "used ONLY by the eval lineage". The text is pypi's `UPDATE_PROFILE_{SYSTEM,USER}_PROMPT` (`memoryos-pypi/prompts.py:199-200`), which is DEAD in pypi — `gpt_update_profile` is defined (`utils.py:340`) but never called on any pypi live path. The eval lineage's merge call uses a completely different inline "Profile Merge Task" prompt (`eval/utils.py:301-350`: 4-category rules, conflict hierarchy, 1500-word cap). So our second call has eval's call *shape* with a prompt no lineage's run ever executed.

**9. [behavior] eval promotion call partition differs.** Upstream eval extracts profile AND private data in ONE call via section markers (`gpt_personality_analysis`, `eval/utils.py:238-299`), plus a separate `analyze_assistant_knowledge` call, plus the merge. Ours: analysis (profile only) + merge + one combined knowledge call for private+assistant (pypi's split). Same call count, different prompt routing for private facts.

**10. [behavior]+[doc] Read-side page scoring: the eval lineage never reads stored page embeddings.** Its search re-embeds `f"{user_input}{timestamp}{agent_response}"` fresh per page per query (`mid_term_memory.py:227-230`); only pypi dots the stored `"User: … Assistant: …"` vector (`mid_term.py:335-338` — and pypi's stored text has no "Assiant" typo; the typo is eval's, `:88`). Two consequences: the `MemoryOSPageRecall` docstring (steps.py:269-273) attributes the stored-vector scheme to "upstream" generically when it is pypi-only; and E3's claim (organizer.py:144-145) that "page-level retrieval therefore depends on which path stored the page" is false for the eval lineage — the inconsistently-stored embeddings are dead at read time there. E3 is real as a storage inconsistency but mischaracterized in its stated impact.

**11. [behavior] No first-stage session cut.** Both lineages cap stage one at `top_k_sessions=5` (pypi `search_sessions` default, `retriever.py` passes it through; eval `search_sessions_by_summary` top_k=5, driver doesn't override). `MemoryOSPageRecall` expands every segment hit that clears `segment_threshold`, and the configs search k=10 segments — there is no top-5-sessions knob at all.

**12. [behavior] Heat-feedback granularity.** Upstream bumps `N_visit`/LFU once per SESSION per query when any page matches, *before* the queue cut (`mid_term.py:344-351`; `mid_term_memory.py:234-240`). Our `on_retrieval` bumps once per SERVED page post-cut (organizer.py:533-548) — multi-page serves from one segment inflate the counters, and matched-but-cut sessions get nothing. Inert on ingest-then-eval runs (as the docstring notes for the loop generally), but the live-traffic shape differs and this specific granularity difference is undocumented.

**13. [behavior] pypi STM off-by-one.** pypi flushes at the START of the overflowing add (`memoryos.py:242-246`), so QA-time resident STM is `capacity` (10) pages and a page is evicted when the 11th arrives. We flush at the END of the add that reaches capacity, so resident is `capacity-1` (9) and eviction happens one exchange earlier. The docstring's "leaves capacity - 1 resident" (organizer.py:65) describes upstream's mid-add instant, not its observable state. (eval, capacity 1: both end empty — equal.)

**14. [question] Profile double-serving.** The harness prepends the profile document unconditionally (locomo.py:590-594, matching upstream), but `memoryos:user_profile` is also an embedded `semantic` item competing in the k=10 semantic recall — if it wins a slot it appears twice in one prompt. Upstream has only the unconditional channel. Also our single `semantic` channel (k=10) stands in for pypi's two separate top-20/threshold-0.01 knowledge searches and eval's top-10 user-knowledge search — acknowledged as approximation in locomo.py, but the k mismatch (10 vs 20+20) is not.

## Verified clean

- Heat formula per lineage: pypi α=β=γ=1, live recency via `compute_time_decay`, tau 24h (`mid_term.py:21-36`); eval 0.8/0.8/1e-4, stored `R_recency` init 1.0 (`mid_term_memory.py:24-28`) — presets exact.
- E1 (`mid_term_memory.py:177-180` outside the merge-success branch; pypi lacks it) — accurate, correctly NOT reproduced. E2 reproduced, including the quirk that the retrieval-hit refresh recomputes decay of a just-updated timestamp and thus always yields 1.0 — bit-faithful to `mid_term_memory.py:237-238`. E4 (`:163`) and E5 (`:31` default 7 vs driver 2000, `main_loco_parse.py:234`) exact.
- θ=0.6 both (memoryos.py:41 → updater; `main_loco_parse.py:236`; pypi Updater's own 0.5 default is dead upstream), τ=5.0 both, MTM 2000 both, STM 10/1 (`short_term.py:10`; `main_loco_parse.py:233`), page as counting unit (deque of `add_qa_pair` pairs; `L_interaction = len(processed_details)`).
- Keyword formulas exact: Jaccard `mid_term.py:215-218`; containment-mean `mid_term_memory.py:148-149` and `:216-218` (same formula write and read side in eval, as our knob-pairing rule assumes).
- Read-path constants: cap 7 pypi (`memoryos.py:38`, `retriever.py:24`) vs 10 eval (`main_loco_parse.py:237`) — config.py:110 default 7 and the `memoryos_eval` override to 10 are both right; thresholds 0.1/0.1 both drivers, eval class defaults (0.8/0.7) dead as config.py:113 claims; three keyword-term copies as steps.py describes (pypi `query_keywords = set()` `mid_term.py:292`; chromadb Jaccard via `extract_keywords_from_multi_summary` `mid_term.py:290,309-312`; eval `llm_extract_keywords` ≤3 `eval/utils.py:361-370`); recency term dead in every read path (eval `lambda_t=1` `:213`; pypi commented `:327`). Bench `MEMORYOS_KEYWORD_PROMPT` reproduces eval's extractor wording, gated to lineage="eval", empty set degrades to pypi.
- Two-call vs single profile shapes (`main_loco_parse.py:45-55` vs `memoryos.py:154-190`), skip-merge-when-no-old, merged-call fallback, hottest-only heap-top promotion, mark-ALL-pages-analyzed, heat reset fields, `update_user_profile(merge=False)`, no-unanalyzed skip — all match.
- Continuity/meta: pypi really does make the first-page LLM call with empty prev and discard it (`utils.py:359-374`, `updater.py:130-132`); eval short-circuits (`dynamic_update.py:13-14`); our skip and its call-count note are accurate. Newest-meta chain propagation matches both `_update_linked_pages_meta_info` and `_update_connected_pages`.
- STM never drained upstream; `recent_context` mirrors `get_all` history rendering (`memoryos.py:269-273`, `main_loco_parse.py:84-88`); harness injects profile unconditionally and full assistant-knowledge dump only under eval — matching the lineage split exactly.
- Pinning tests exist and assert the corrected facts (`tests/test_organizers_phase3.py:184-232, 1040`).

Overall: the fresh eviction fix and all five E-defect placements survive re-audit. The serious new items are findings 1-2 (the eval preset's degenerate half-exchange pages plus the unreproduced incomplete-page filter) and 4-6 (merge-key drift, whole-batch-per-theme, eval's unbounded knowledge) — everything else is prompt-provenance, granularity, or off-by-one level. Nothing was modified.

---

# [zep]

All load-bearing paths have been diffed. Here is the structured report.

---

# Fresh-eyes re-audit: Zep/Graphiti port vs upstream (graphiti @ 9140123, 2026-07-26)

## A. The fresh docstring's upstream claims (organizer.py:44-51) — symbol by symbol

All named symbols verified to exist as claimed:
- `SagaNode` — upstream `graphiti_core/nodes.py` (plus per-driver `saga_node_ops.py`); `_get_or_create_saga` at `graphiti_core/graphiti.py:346`, `summarize_saga` at `graphiti.py:438`. Not ported here — correct.
- `combined_extraction.extract_nodes_and_edges` — `graphiti_core/utils/maintenance/combined_extraction.py:41`, imported by `graphiti.py`. Not ported — correct.
- `temporal_operations.py` — no longer exists anywhere in the upstream tree; timestamps are integrated into the `extract_edges` prompt (`prompts/extract_edges.py:160-172`) with a fallback `_extract_edge_timestamps` in `edge_operations.py:576`, and `resolve_edge_contradictions` now lives in `edge_operations.py:538`. "Dissolved into extract_edges" — correct.

The claim "entity resolution follows current main (three-stage)" is only half-true — see findings 1–3.

## B. Findings

**1. [doc+behavior] "Three-stage as today's Graphiti" — stage 2 is half of upstream's stage 2.**
Ours: `organizer.py:327-354` — embedding candidates ≥0.6 → exact casefold name match → LLM. Upstream: `node_operations.py:627-690` → `dedup_helpers.py:220-280` — semantic candidates ≥0.6 → deterministic stage that is exact-normalized-name **plus fuzzy MinHash/LSH** (3-gram shingles, 32 permutations, Jaccard ≥0.9, entropy-gated at 1.5) → LLM. Our port has no fuzzy sub-stage at all, and the docstring (`organizer.py:4-6`) presents "deterministic exact-name match" as what "today's Graphiti does". Also: upstream escalates an *ambiguous* exact match (>1 candidates sharing the normalized name) to the LLM (`dedup_helpers.py:245-249`); ours returns the first exact hit in similarity order (`organizer.py:350-352`). Why it matters: "Alex Smith"/"alex smith " merge identically, but nickname/typo variants ("Katherine"/"Katharine") that upstream folds deterministically will burn an LLM call in ours or create duplicate nodes when the LLM misses — a graph-density and cost-shape difference under the very docstring that was committed to pin lineage.

**2. [behavior] Merge refresh of name/summary is the PAPER's behavior, presented as current main's.**
Ours: on LLM-confirmed duplicate, the canonical node's name AND summary are rewritten from the verdict (`organizer.py:373-394`, `RESOLVE_SCHEMA` asks for both). Upstream current main: `NodeDuplicate` is `{id, name, duplicate_candidate_id}` — **no summary field** (`prompts/dedupe_nodes.py:25-34`), and `_promote_resolved_node` (`dedup_helpers.py:174-196`) keeps the existing node, only promoting type labels; summaries are maintained by a separate attribute/summary batch path (`node_operations.py:833+`). The paper does say dedup "generates an updated name and summary", so the behavior itself is defensible — but the docstring sentence at `organizer.py:4-6` attributes the whole parenthetical, refresh included, to "as today's Graphiti does". That is exactly defect class 1/3: a piece of paper lineage hiding inside a "current main" label.

**3. [behavior] Resolution-call shape and candidate pool differ from main: per-entity LLM calls, k=5.**
Upstream batches ALL unresolved entities of a message into ONE dedupe call (`node_operations.py:552-556`) against up to `NODE_DEDUP_CANDIDATE_LIMIT = 15` candidates each (`node_operations.py:64`). Ours calls the LLM once per unresolved entity against k=5 candidates (`organizer.py:342-369`). Matters for any cost comparison (call counts per message differ by the entity count) and for recall of dedup candidates (5 vs 15). The docstring's `candidate_threshold` attribution to `NODE_DEDUP_COSINE_MIN_SCORE` (0.6) is correct.

**4. [behavior] Edge invalidation candidates: same-pair only, vs upstream's graph-wide semantic search.**
Ours: duplicate AND contradiction candidates are both `graph.edges_between(subj, obj)` (`organizer.py:704`). Upstream: duplicate candidates are same-pair (`edge_operations.py:365-370`, hybrid search filtered to the pair's edge uuids), but **invalidation candidates come from an unfiltered graph-wide `EDGE_HYBRID_SEARCH_RRF`** over the whole group (`edge_operations.py:408-419`), minus the duplicates. Our own research doc says the same ("의미적으로 관련된 기존 edge들", `docs/research/zep-graphiti.md:28`). Consequence: our port can never invalidate `Alice LIVES_IN Paris` when `Alice MOVED_TO London` arrives as an edge to a *different* node — the paper's flagship temporal mechanism only fires between the identical entity pair. The module docstring (`organizer.py:9-10`) honestly says "same-pair", but then claims "everything else the paper's shape" — this piece is neither the paper's nor main's shape.

**5. [behavior] `resolve_edge_contradictions` — cited as the source of our guard, but half its math is missing.**
Ours (`organizer.py:744-767`, comment cites "upstream resolve_edge_contradictions"): skip if `e.invalid_at <= valid_at` or `invalid_at <= e.valid_at`, otherwise invalidate. Upstream (`edge_operations.py:538-573`): same two skip conditions, **then invalidates only if `edge.valid_at < resolved_edge.valid_at` (strictly older, both non-None)**. Two divergences: (a) an existing fact with `valid_at` equal to or LATER than the new fact gets invalidated by us but never by upstream — we let an older-dated incoming fact kill a newer one; (b) upstream instead handles that case by expiring the NEW edge (`edge_operations.py:826-841`: if any candidate has a later `valid_at`, `resolved_edge.invalid_at = candidate.valid_at`), a mechanism our port lacks entirely — a new fact contradicted by already-known newer information enters our graph as active. Classic defect class 5. What matches: `t_invalid` = the invalidating fact's `valid_at` (both sides), and the store stamps `expired_at` (T′) separately (`stores/sqlite_graph.py:143-153`), so the bi-temporal axes themselves are faithful.

**6. [behavior] `valid_at` hard default to reference time when the model returns null.**
Ours: `valid_at = str(f.get("valid_at") or ref_time)` (`organizer.py:700`). Upstream: a null `valid_at` stays `None` after parse (`edge_operations.py:253-270`), with one retry via `_extract_edge_timestamps`; a `None`-valid_at edge can neither invalidate others nor be invalidated (all upstream conditions require non-None). The prompt instruction ("stated as current/ongoing → use reference time") matches upstream's prompt (`prompts/extract_edges.py:171`), but the code-level default makes every fact a dated, invalidation-capable fact — more aggressive temporal semantics than upstream. Interacts multiplicatively with finding 5.

**7. [behavior] No verbatim-duplicate fast path on edges.**
Upstream reuses an edge without any LLM call when normalized fact text + endpoints match exactly (`edge_operations.py:687-700`), and pre-dedups identical extractions within the batch (`edge_operations.py:344-358`). Ours always pays the `EDGE_RESOLVE` LLM call when any same-pair edge exists (`organizer.py:707-726`). Cost-shape difference for any call-count comparison; also our `duplicate_of` is single-valued where upstream's `duplicate_facts` is a list — minor.

**8. [behavior] RRF constant: ours k=60, upstream rank_const=1.**
Ours: `1.0 / (k + rank + 1)` with `k=60` (`src/agmem/retrieval/fusion.py:7,35`). Upstream: `1 / (i + rank_const)` with `rank_const=1` (`search_utils.py:1780-1786`) — i.e. rank-1 scores 1/1, not 1/61. This is NOT a monotone rescale across multiple channels: with two channels, items ranked (1,100) vs (3,3) order differently under the two constants. The fusion docstring discusses its channel-count divisor deviation (audit B2, acknowledged) at length but calls un-normalized RRF "the textbook form" and never mentions that upstream's constant is 1 — same-named function, different math, undocumented (defect class 5 + 3).

**9. [doc] NodeDistanceReranker docstring claims "same ordering" as upstream — it isn't.**
Ours ranks by true hop distance up to `max_hops=3` (`rerank.py:263-313`); the docstring says "Upstream reranks by shortest-path length over RELATES_TO, which is the same ordering." Upstream's actual query (`search_utils.py:1798-1856`) is a **1-hop adjacency test**: direct neighbors score 1, everything else `inf` (the "shortest path" comment there is stale). So upstream ties hop-2 with hop-10; ours orders them. Arguably ours matches the paper's description better, but the docstring asserts equivalence with upstream code that does not hold.

**10. [doc] EpisodeMentionsReranker: upstream sorts ASCENDING (least-mentioned first).**
Ours sorts mention count descending per the paper text (`rerank.py:241-260`). Upstream `episode_mentions_reranker` (`search_utils.py:1860-1896`) does `sorted_uuids.sort(key=scores)` ascending with missing nodes at `inf` — most-mentioned last, which contradicts the paper's "frequently referenced becomes more accessible" and is almost certainly an upstream bug. Our docstring cites the upstream function as the same computation without recording this divergence — for a repo whose method is "which piece follows which", the deliberate paper-over-upstream choice should be named (compare how `community.py` documents upstream's misdescribed tie-break honestly).

**11. [doc] "Transcribed field by field from search_config_recipes.py" — every combined recipe silently drops upstream's episode channel.**
Upstream `COMBINED_HYBRID_SEARCH_RRF/MMR/CROSS_ENCODER` all carry an `episode_config` (BM25, rrf/cross_encoder reranker) (`search_config_recipes.py:42-47,67-72,97-102`), and upstream's context string has an `<EPISODES>` section (`search_helpers.py`). Our table (`zep_graph/search.py:95-149`) omits episodes from all three, with a rationale at lines 48-51 ("Zep's context template has no raw-message section") — true of the **paper's** template, not of current upstream. The rationale is documented but attributes the omission to the wrong lineage, and "transcribed field by field" (line 94) overstates. Everything else in the table checks out exactly: cross_encoder = bm25+cosine+bfs on edges/nodes, communities never BFS (`CommunitySearchMethod` has no bfs member — `search_config.py:48-50`), MMR recipes pin `mmr_lambda=1`, plain `search()` defaults to `EDGE_HYBRID_SEARCH_RRF` (`graphiti.py:1570`), `BGERerankerClient` loads `BAAI/bge-reranker-v2-m3` (`cross_encoder/bge_reranker_client.py:36`), limits 10 / depth 3.

**12. [behavior] φ_bfs traverses undirected; upstream BFS is directed-outgoing and includes MENTIONS.**
Upstream `node_bfs_search`/`edge_bfs_search` walk `-[:RELATES_TO|MENTIONS*1..n]->` (outgoing only; `search_utils.py:448+, 790+`). Our `bfs.py:39-49` rings via `SqliteGraphStore.neighbors`, which matches `(e.src = w.id OR e.dst = w.id)` — undirected (`stores/sqlite_graph.py:165-167`) — so our frontier is strictly wider (a fact `B → A` makes B reachable from A here, not upstream). `bfs.py`'s docstring carefully documents its ordering deviation and origin derivation (both verified against `search.py:332-334, 540-541` — the source-node-only edge origins claim is correct) but says nothing about direction. Also unflagged: upstream's cosine channels apply `DEFAULT_MIN_SCORE = 0.6` (`search_utils.py:65`); our dense channel has no cutoff.

**13. [behavior, at-operating-point-equivalent] MMR algorithm differs: greedy-iterative vs upstream's one-shot batch.**
Ours (`rerank.py:91-104`): classic greedy MMR, redundancy vs already-selected set. Upstream (`search_utils.py:1901-1938`): one-shot score `λ·(q·c) + (λ−1)·max_sim(c, all other candidates)`, no iterative selection. At the shipped recipe's `λ=1` both reduce to pure relevance sort, so the recipe table's operating point is unaffected — but for any `λ<1` ("lower values penalize redundancy", our docstring) the two mechanisms rank differently. Undocumented class-5 divergence held dormant by the λ=1 default.

**14. [question] The recipe table is exported but nothing in production applies it.**
`zep_search_recipe()`/`config_kwargs()` are consumed only by `tests/test_organizers_phase3.py` and `__init__` exports — no harness/facade path turns a recipe into an `AgmemConfig`. Every individual knob it emits IS live (`config.py:68-96,199-213`; `memory.py:234-254`; `_build_reranker` `memory.py:304-332` correctly routes `model_name`/`lambda_` and injects graph_store/namespace for node-distance), so this is a stranded-adapter question, not dead knobs: is a measured run expected to hand-copy `config_kwargs()` output into TOML? If so, the "run stamped with settings nobody asked for" failure the table exists to prevent is one transcription away from returning.

**15. [doc, minor] Entity-extraction context window: paper n=4 vs upstream 3.**
`context_window=4` cites "paper n=4" (`organizer.py:284-296`) — consistent with the declared paper lineage, but upstream's `EPISODE_WINDOW_LEN = 3` (`graph_data_operations.py:29`); nowhere is the 4-vs-3 divergence from main recorded. Similarly, exact-match normalization: upstream lowercases AND collapses whitespace (`dedup_helpers.py:39-42`); ours casefolds only (`organizer.py:349`) — `"John  Smith"` and `"John Smith"` are distinct here.

## C. Verified clean

- **Write path is MemoryOp-only (B3 holds).** Zero direct `upsert_node`/`upsert_edge`/`invalidate_edge` calls in `zep_graph/` (mentions are docstrings only); all graph mutation flows through `AgenticMemory._apply_graph` (`memory.py:685-750`) and the INVALIDATE branch (`memory.py:643-656`), including communities (upsert + membership + DELETE→`remove_community`, `memory.py:677-683`). Endpoint ids ride the fact payload; missing ids warn loudly.
- **Bi-temporal store semantics**: `invalidate_edge` stamps `invalid_at` (T) and preserves-or-sets `expired_at` (T′) via COALESCE (`sqlite_graph.py:143-153`); re-upsert after invalidation is re-stamped (`memory.py:747-750`); invalidated facts stay servable and render "(Date range: valid – present)" matching upstream's context string (`steps.py:33-53, 776-782` vs `search_helpers.py` fact_json).
- **Label propagation** (`community.py:90-190`) is a faithful transcription of `community_operations.py:93-138` including both misdescribed behaviors (tie → larger *label*, not larger community; `candidate_rank > 1` gate), and its 2-cycle/max-rounds termination is a real, honestly-documented fix — upstream's synchronous `while True` does oscillate forever on a two-node, two-edge component.
- **Community build**: map-reduce pairing first-half-vs-second-half with odd carry matches `build_community` (`community_operations.py:174-196`); `MAX_SUMMARY_CHARS=1000` and `truncate_at_sentence` match `text_utils.py`; description-as-name matches `generate_summary_description`. `update_community`/`determine_entity_community` shape (keep-if-member → neighbor plurality → pair-summarize → rename, edge only when new) matches `community_operations.py:259-351`; the deterministic tie-break and membership-keyed community id are declared deviations with sound rationale. Cost-avoiding skip-unchanged-cluster deviation is documented at `_community_id`/`rebuild_communities`.
- **Search recipe table contents** (modulo finding 11): every field checked against `search_config_recipes.py` matches, including the two single-subgraph recipes being the only carriers of episode-mentions/node-distance, and the §4.1 cross-encoder = BGE-m3 operating-point argument, which upstream's `BGERerankerClient` corroborates.
- **BFS origin derivation** claims in `pipeline._bfs_ranking` and `GraphRecall` docstrings verified against `search.py:332-334/540-541` (explicit origins replace, not extend; edge origins = source nodes only). `graph_expansion_cap=0` default really disables GraphRecall (`steps.py:738-741`), and GraphRecall's docstring itself correctly discloses it is not an upstream mechanism.
- **Fact text construction**: statement as `content`, embedding on statement, SCREAMING_SNAKE_CASE `relation_type` enforcement matching both paper §6.1.3 and upstream's field spec; self-loop drop matches upstream's same-uuid drop (`edge_operations.py:232-241`); hallucinated-name drop matches upstream's name validation.
- **Entity item surfaces**: node fulltext = (name, summary) and community fulltext = name-only verified against `graph_queries.py:126-127,134-136`; name-only entity embedding matches `_semantic_candidate_search` querying `node.name`.
- **Config knobs all live** (no dead knobs found): `lexical_types`, `bfs_types`, `bfs_max_depth`, `reranker_params`, `graph_expansion_cap/hops`, `overrides["reranker"]` all reach the pipeline/reranker constructors.
- **`produces = ("facts", "entities", "communities")`** order + `ctx.bundle_ids` dedup — the earlier double-serving fix is intact.

Net assessment: the docstring's upstream-provenance claims (sagas, combined extraction, temporal_operations dissolution) are accurate, and the write-path/bi-temporal/community/recipe-table layers audit clean. The two claims that do not survive contact are "entity resolution follows current main" (findings 1–3: the stage *count* matches, the stage *contents* don't, and the merge refresh is paper-shape) and "everything else the paper's shape" for edge invalidation (findings 4–6: same-pair-only candidates, missing strictly-older guard, missing new-edge self-expiry, and a hard valid_at default — the temporal engine is measurably more trigger-happy and less far-reaching than either lineage).

---

# [gmemory]

All evidence is gathered. Here is the audit report.

---

# G-Memory fidelity re-audit (organizer + query graph + TaskGraphExpansion)

Ours: `/home/jinmang2/agentic_memory/src/agmem/organizers/gmemory/organizer.py` (O), `/home/jinmang2/agentic_memory/src/agmem/retrieval/steps.py` (S), `pipeline.py` (P), `config.py`, `tests/test_organizers_phase3.py:666`.
Upstream: `/home/jinmang2/.agmem/upstream/GMemory/mas/memory/mas_memory/GMemory.py` (U), `prompt.py` (UP), `tasks/run.py`, `tasks/mas_workflow/autogen/autogen.py` (and dylan/macnet — identical pattern).

## Findings

**1. [behavior] The 0.7 edge threshold is the same constant on different math (defect class 5).** O:222 applies 0.7 to true cosine (our `VectorStore` contract, `stores/base.py:60`; numpy/chroma stores confirmed). Upstream U:390-392 computes `similarity = 1 - distance` from `similarity_search_with_score` on a `Chroma` built with **no** `collection_metadata` (U:38-41) — Chroma's default space is `l2` (squared L2). The embedder is `sentence-transformers/all-MiniLM-L6-v2` (run.py:74), which normalizes its outputs, so distance = 2−2·cos and upstream's gate is `2cos−1 ≥ 0.7`, i.e. **effective cosine ≥ 0.85**. Our graph is materially denser than upstream's. (The organizer docstring's "TaskLayer.add_task_node's constants" is literally true — 0.7 and k=10 appear at U:360 and U:387 — but the quantity thresholded differs.)

**2. [behavior] The similarity gate compares different texts.** Upstream's Chroma docs embed `page_content = task_main` only (U:96-98): the query-graph gate and all task retrieval are task-vs-task. We store trajectories with `embedding_text = f"{task}\n{content}"[:2000]` (O:247), so O:206-211 measures new-task-vs-(old-task+condensed-trajectory). Systematically shifts similarity; compounds with finding 1 in an unknown direction.

**3. [behavior] Top-10 candidate pool is contaminated by insights.** Upstream's k=10 search runs over a collection containing only task docs. Ours searches `memory_type="strategies"` k=10 (O:207-212) and then filters `kind == "trajectory"` (O:214-218) — stored insights occupy candidate slots, so fewer than 10 trajectory candidates get considered. False negatives (missing edges) upstream cannot have.

**4. [behavior + doc] ADD suppression when full is not upstream code behavior.** Upstream `_update_rules` executes ADD unconditionally (U:843, 871-878); the cap is soft — a prompt suffix "stop ADD rule unless the new rule is VERY insightful" (UP:300-301) appended only when `len(insights_memory) > 10` (U:713-715), plus REMOVE strength 3. Upstream's rule list can and does exceed 10. Ours hard-drops ADD when full (O:308-309) and never varies the prompt with fullness. The claims at O:20-21 ("the rule cap is enforced upstream-style by suppressing ADD when full") and O:145-147 ("upstream full-list behavior") are contradicted by upstream code (defect class 3).

**5. [behavior] `list_full` is computed from a retrieval window, not the global list.** Upstream uses `len(self.insights_memory)` (U:843). Ours uses `len(insights)` where `insights = _fetch(..., k=insight_max)` (O:266, 299-300) — a similarity-fetch capped at 10 that can return fewer when the k*3 search window is crowded by trajectories, misreporting "not full" (REMOVE −1 instead of −3; ADD allowed). Also `n_insights += 1` (O:328) is a dead write — `n_insights` is never read after `list_full` is set once.

**6. [behavior + doc] Feedback reward leaks onto trajectories.** `on_retrieval` caches ALL served `strategies` ids including trajectories (O:161); `on_feedback` applies +1/−2 with no `kind` filter (O:424-439; only the DELETE is kind-gated, O:440). Upstream `backward` touches only `insights_cache`, which holds served insight rules exclusively (U:239, 292-297, 575-582). Our trajectory `score` fields (init 1.0/−2.0, itself an invented field — upstream trajectories carry a label, not a score) drift over time. The docstring "Reward shaping on served insights" (O:413) contradicts the code.

**7. [behavior] Finetune structure and correlation granularity deviate beyond the disclosed "regex → JSON" change.** Upstream `finetune_insights` runs `num_points=5` iterations per event, each anchored on a **random** stored task, retrieving 3 successes + 1 failure and issuing ~2 LLM calls (compare-pair + success-chunk, U:647-676, 719-748) — roughly 10 calls per event; the rule list shown is selected by correlation overlap ≥ len(tasks)/2 (U:672), and `relative_tasks` recorded per op batch are the 2–5 full task strings **in that prompt** (U:726-728, 740-744). Ours: one call, insights and 10 trajectories fetched by embedding similarity to the current task, label-blind (O:265-267), recording ~10 titles onto every touched insight (O:272). Consequences: (a) LLM call-count non-parity — load-bearing given this project's cost-measurement stance; (b) `positive/negative_correlation_tasks` grow faster and broader, which directly inflates Eq.(6) recall in `TaskGraphExpansion`. Only the parsing change is disclosed in the docstring.

**8. [behavior] Correlation keys are 80-char truncations, upstream uses full task text.** Write side O:241 (`title=task[:80]`), O:272; read side S:681-695 match truncated-to-truncated, so it is internally consistent — but distinct tasks sharing an 80-char prefix (plausible for templated benchmark tasks) collide: false-positive insight correlations, and false suppression of edge creation via the title-equality check at O:219. Upstream matches full `task_main` strings exactly (U:637, 856-875).

**9. [behavior] Repeated-task early-return is approximated two ways at once.** Upstream checks global graph membership (`if task_main in self.graph: return`, U:380-381) and unifies repeats into ONE node (node = task text). Ours checks only the current top-10 filtered neighbors' titles (O:219) — a repeat that misses the top-10 still gets edges — and always stores a second trajectory; when edges are skipped, the duplicate is an **isolated** node, so a later hit on it expands to nothing while upstream's single node keeps its whole neighborhood.

**10. [behavior] Eq.(5) neighbor scoring: parent×0.9 pseudo-score instead of re-similarity + threshold.** Upstream re-ranks the graph-expanded set by true cosine to the query and applies the caller's threshold before per-label topk cuts (U:122-169). Ours serves neighbors at `score*0.9`, no re-similarity, no threshold, global cap 5 (S:657-663). The global-cap deviation is disclosed ("capped globally like LinkExpansion", S:640); the pseudo-score and absent re-rank are not. Note also the harness runs `threshold=0.0` (run.py:131), so upstream's 0.3 signature default is not the operating point anyway.

**11. [behavior] Which memory the eval harness actually serves (defect class 2).** All three MAS workflows **discard the failed-trajectory list**: `successful_trajectories, _, insights = retrieve_memory(...)` (autogen.py:108, dylan.py:180, graph_mas.py:104), with `successful_topk=1, insights_topk=3, threshold=0.0` (run.py:128-131, autogen.py:27-30) — upstream never shows failure trajectories to the agent at read time; failures feed only the finetune. Our read path serves trajectories outcome-blind, "Mistakes: ..." content included, with no success/fail quota. Additionally, upstream serves insights **only** via correlation counting (`query_insights_with_score`, U:490-506) — never by rule-text embedding — while our dense channel retrieves insights by rule-text similarity (`embedding_text=rule`, O:324) as an extra channel, with `TaskGraphExpansion`'s count recall added on top at `insight_cap=10` (the `retrieve_memory` signature default, not the harness's 3; and the cap is not exposed in config — `default_read_steps` passes only the trajectory cap, S:744).

**12. [doc] docs/research/g-memory.md states signature defaults as the operating point.** Line 62: "successful_topk=2, failed_topk=1, insight_topk=10, retrieval threshold=0.3" are `retrieve_memory`'s Python defaults; the shipped harness uses 1 / 1-then-discarded / 3 / 0.0. Line 26: "cosine sim ≥ 0.7" — not cosine (finding 1).

**13. [question] Eq.(6)'s task set differs.** Upstream counts insight overlap over a fresh label-filtered retrieval (4 successes + 2 failures) **plus the query task string itself** (U:492-496); ours counts over served trajectory titles (hits + 1-hop, S:681-687). Same spirit; but ours can never match an insight correlated with the exact query text, and its set composition is hit-dependent. Whether this is close enough is a call for the maintainer, not a clean-verify.

**14. Minor.** (a) Our FINETUNE_PROMPT shows `(score=N)` per rule (O:277); upstream shows plain numbered rules. (b) Failed-task write is 1 combined LLM call in ours vs 2 upstream (`_extract_mas_message` + `_detect_mistakes`, U:265-290) — another call-parity nick. (c) Upstream AGREE resolves its target by rule **text** with a fall-through to index −1 that credits the *last* rule on mismatch (U:858-861 via `_retrieve_rule_index` returning −1) — an upstream bug our id-based path intentionally cannot reproduce; upstream also converts EDIT-of-existing-text to AGREE and drops duplicate-text ADDs (U:820-828), which we have no analog for. (d) Upstream's `hop` is configurable (default 1); our step hardcodes 1-hop. (e) The named test constructs `TaskGraphExpansion` directly, so pipeline wiring/cap behavior is untested by it (wiring verified by reading). (f) On the audit question "are the per-candidate LLM rerank and FINCH really the only gaps": no — findings 4, 6, 7, 10, 11 are additional undisclosed gaps.

## Verified clean

- Constants k=10 and 0.7 are literally upstream's (U:387, U:360); undirected rendering via forward `task_edges` + back-edge UPDATE is a faithful projection of `nx.Graph.add_edge`; per-call duplicate edges impossible (fresh `traj_id`); search-before-store ordering matches upstream (`add_task_node` runs before `add_documents`, U:93-101).
- Score semantics all verified against upstream code: ADD init 2 (U:874), EDIT +1 (U:868), AGREE +1 (U:861), REMOVE −1 / −3 when full (U:854-855), prune at score ≤ 0 (`clear_insights` keeps > 0, U:584-586), backward +1 success / −2 failure (U:292-295), applied only to served-since-last-feedback with the cache cleared after (U:297 / O:449).
- Correlation direction verified: positive extended on ADD/EDIT/AGREE, negative on REMOVE (U:856, 862, 869, 875) — our O:322, 343, 360, 376 match.
- Insight recall math: overlap-count with threshold 1, count-sorted descending (S:695-698) is arithmetically equivalent to upstream's per-task accumulation in `query_insights_with_score` + `_find_related_insights` default threshold (U:497-501, 631-641).
- "At most 4 operations, each existing rule at most 1 operation" is genuinely upstream's prompt text (UP:312); ours enforces it in code (`[:4]`, `touched`) where upstream trusts the LLM — stricter, prompt-faithful.
- `task_graph_expansion_cap` is live config, not dead: config.py:96 → memory.py:253 → pipeline.py:56/104 → steps.py:743-744; cap 0 removes the step.
- ReasoningBank inertness claim confirmed: RB `strategies` payloads carry neither `kind` nor `task_edges` (reasoning_bank/organizer.py:325-333 — no `kind` key anywhere in that file), so the trajectory loop skips everything and the empty title set early-returns before the insight scan (S:658, 687).
- No shared-state mutation in the step (`out = list(hits)`, local `seen`); `ctx.bundle_ids` honored for cross-type dedup; hallucinated/double-touched rule ids emit nothing (O:330-332); `on_feedback` UPDATE-on-missing is guarded by the facade (`memory.py:594-615`).
- `project_insights` role rewrite with raw-insights fallback matches upstream (U:304-350).
- Finetune cadence equals upstream at defaults (first event at task 5, every 5); the missing `start_insights_threshold` only diverges off-default.

The single highest-impact item is finding 1 (effective 0.85-vs-0.70 gate — every edge, and therefore everything Eq.(5)/(6) serves, depends on it), followed by 11 (our read path serves memory upstream's harness never serves) and 7 (call-count non-parity in the write path).

---

# [ace-rb]

All verification is complete. Final report follows.

---

# Fresh-eyes fidelity re-audit: ACE + ReasoningBank vs upstreams

## ACE findings

**1. [behavior] Our "always-on dedup" matches NEITHER upstream path — it is a third behavior, and the published operating point is dedup-OFF.**
- Upstream confirmed THIN as the study recorded: curator apply path is ADD-only with MERGE/DELETE as TODO comments (`~/.agmem/upstream/ace/playbook_utils.py:96-216`, validator at `ace/core/curator.py:212-215` only warns on other op types). The dedup component is opt-in (`ace/ace.py:41` default `use_bulletpoint_analyzer=False`; eval harnesses expose it as `store_true` default-off, `eval/finance/run.py:83`; README default `false`) and silently returns the playbook unchanged when sentence-transformers/faiss are missing (`ace/core/bulletpoint_analyzer.py:290-292`).
- However, when upstream's analyzer IS enabled it does something different from ours: it dedups the WHOLE playbook (old-old pairs included) and LLM-MERGES each similar group into one bullet with summed helpful/harmful counters (`bulletpoint_analyzer.py:193-336`, called with `merge=True` at `ace.py:600-606`). Ours (`src/agmem/organizers/ace/organizer.py:347-365`) is drop-the-new-bullet-only, never merges, never touches existing pairs.
- So we do not "reproduce" the trap, and we also do not reproduce the enabled behavior — the module docstring (organizer.py:9-12) documents the deviation honestly, but anyone comparing against upstream's published numbers should know those numbers come from the no-dedup path, and our refine axis is a distinct mechanism, not upstream's analyzer.

**2. [behavior] Curator prompt: one invented input, one dropped input, one invented field, one invented cap.**
Upstream `CURATOR_PROMPT` (`ace/prompts/curator.py:6-67`) vs our `CURATE_PROMPT` (organizer.py:91-113):
- (a) Our "Playbook now uses about {playbook_tokens} tokens" line has no upstream counterpart — upstream never tells the curator current playbook size (`count_tokens` is logging-only, `ace.py:491,626`). The `__init__` docstring (organizer.py:200-204) attributes "told the budget and the current size" to upstream; only the budget half is upstream's. Also [doc].
- (b) Upstream passes `question_context` to the curator; we do not.
- (c) Upstream feeds the FULL raw reflection (fields `reasoning`/`error_identification`/`root_cause_analysis`/`correct_approach`/`key_insight`, `ace/prompts/reflector.py:18-24`); we feed `key_insight` + `lessons` — and `lessons` is not an upstream reflector field at all (our schema invention, organizer.py:38,54).
- (d) Our `maxItems: 5` / `max_ops=5` cap (organizer.py:62,192) has no upstream counterpart — upstream accepts an unbounded operations list.

**3. [behavior, documented] Counter attribution is post-hoc evidence-based, not usage-based.** Upstream's reflector sees ONLY the bullets the generator self-cited (`bullet_ids` extracted from the generator's JSON, `ace/core/generator.py:74-102`, filtered via `extract_playbook_bullets`, used at `ace.py:505-521,549-564`); ours shows the full playbook and asks the reflector to tag from the trajectory (organizer.py:84-89,284-295). The module docstring (lines 18-21) states exactly this and points at `report_feedback()` as the usage-accurate path — claim verified accurate. Downstream consequence not stated anywhere: upstream tags on EVERY reflection round (up to 3 per incorrect task), so upstream counters accrue faster on failures.

**4. [behavior, verified correct] Multi-round reflection not ported — correctly.** Upstream's `max_num_rounds` loop interleaves reflection with `generator.generate` regeneration (`ace.py:499-545`); it is the generator loop, and our single reflection per `on_task_end` is the right boundary cut.

**5. [question] `get_playbook` exists and is the documented route, but it is NOT structurally the only injection route.** `AgenticMemory.get_playbook` (`src/agmem/memory.py:893-925`) does the full render, format-unified via `render_bullet_line` (`core/types.py:188-199`), exposed as an MCP tool (`mcp/server.py:111`), gated on an active playbook producer. But `default_memory_types` (memory.py:755-768) includes every active organizer's `produces` — with ACE active, "playbook" is in the default set, so a plain `search()` (including the MCP search tool) serves top-k playbook bullets through the generic pipeline, with no read step and no exclusion anywhere. That is exactly the partial-view read the organizer docstring forbids (organizer.py:15-17 "never top-k retrieval of bullets"). Nothing currently exploits it (no harness searches playbook), but the "only injection route" claim holds by convention, not structure.

**6. [doc] Stale line refs.** organizer.py:129 cites "ace.py:507/554" for the feedback-string branches; actual upstream lines are 515/558. Strings themselves are verbatim-correct.

**7. [question, minor] Both ACE roles ride one config knob.** Upstream allows distinct `reflector_model`/`curator_model` (`ace.py:36-63`); our reflect and curate calls both use role `"distill"` (organizer.py:289,328), so per-role model/temperature config cannot split the two roles.

**ACE verified clean:** environment-feedback strings verbatim (with graceful free-form pass-through matching `REFLECTOR_PROMPT_NO_GT`'s existence); `token_budget=80000` = `playbook_token_budget` (`ace.py:127`); curate-every-task = upstream `curator_frequency` default 1 (`ace.py:124`); `playbook_stats` bucket math equivalent to `get_playbook_stats` (`playbook_utils.py:218-254`; our `if` vs upstream's `elif` for "unused" is unreachable-different — a zero-counter bullet can never be high/problematic); ADD-only reproduced; token estimate chars/4 documented as substitution; both ACE and ACEBatch upstream variants checked — ACEBatch (`ace/ace_batch.py`) only parallelizes/chunks the same reflect→curate pipeline, no different math; eval harness path checked (defect class 2): `AceCls = ACEBatch if batch_size > 1 else ACE`, analyzer off by default; bullet-id prefix resolution (organizer.py:305-316) is a robustness addition beyond upstream's exact-match (`update_bullet_counts`), validated against real ids so it cannot mis-credit; neutral tags no-op both sides.

## ReasoningBank findings

**8. [behavior, verified] Success AND failure extraction faithful; persona abstraction validated by the second upstream variant.** WebArena `SUCCESSFUL_SI`/`FAILED_SI` (`WebArena/prompts/memory_instruction.py:15-61`) constraints all present in our condensed SIs (≤3 items, no literal task strings, think-first, when/when-not description, 1-3 sentence content). The SWE-Bench variant (`third_party/src/minisweagent/memory/instruction.py`) is the same structure with only the persona swapped ("expert in coding") — which is precisely what our `persona` parameter abstracts. Judge-reason appending matches upstream's autoeval-thoughts append verbatim in shape ("The task {succeeded/failed} because: ...", `induce_memory.py:159-162`). Upstream emits Markdown item blocks; our JSON schema is the framework's structured-output substitution — same title/description/content schema.

**9. [verified] Experiences/strategies split = upstream's retrieval unit.** Upstream retrieves top-1 EXPERIENCE by task-query embedding and injects all its `memory_items` (`WebArena/run.py:177-193` `select_memory(n=1)`; `memory_management.py:138-215`; SWE-Bench also `n=1`, `run/extra/swebench.py:182`). Our experience record (`embedding_text=task`, `item_ids`) + `ExpandExperiences` replace-with-members (`retrieval/steps.py:148-169`) reproduces this, including miss→no-injection.

**10. [question] The upstream operating point (top-1, experiences-only) is not wired as any default.** With RB active, `default_memory_types` = episodic + experiences + strategies, each at default k=10 — up to 10 experiences expanded plus 10 direct items. The k=1 experiences-only mode exists only as a docstring instruction to callers (organizer.py:8-11); no preset or config pins it. Latent divergence for any future bench harness.

**11. [verified] MaTTS parallel induction is real and called.** `on_scaled_task_end` exists in the hook contract (`organizers/base.py:98-110`), is dispatched by `AgenticMemory.add_scaled_task_result` (memory.py:382-413), and is exercised (`tests/test_organizers.py:95,123`). All three MaTTS deltas are upstream's: ≤5 items (`PARALLEL_SI`, memory_instruction.py:63-90), 1-5 sentence content, t=0.7 (`induce_scaling.py:196`) vs t=1.0 single-trajectory (`induce_memory.py:164,166`) — with temperature correctly stated-not-hardcoded per our role config design. User-message layout matches upstream byte patterns including the odd space in `**Trajectory {n} :**` (`induce_scaling.py:189-191`).

**12. [doc, enrichment] The "dead correctness label" claim is verified — and upstream's label is additionally INVERTED.** `induce_scaling.py:181-184` sets `status="success"` when `reward == 0` (opposite of `induce_memory.py:131-134`), stores it, and never renders it (`format_examples` uncalled). Our docstring (organizer.py:253-262) records "computed, passed, read by nothing" — accurate; the inversion is extra evidence the field is dead (it would be wrong if live), worth a one-line addition.

**13. [verified correct] Sequential scaling not ported — correctly.** `SEQUENTIAL_PROMPT` (memory_instruction.py:132-137) instructs the AGENT to rewrite its trajectory in `<think><action>` format: generator loop, exactly as our docstring argues.

**14. [question, minor] Outcome-string asymmetry.** With `self_judge=False`, any outcome not exactly `"success"` — including `"correct"` — gets the FAILURE SI (organizer.py:226); the single-trajectory fallback of `on_scaled_task_end` passes `outcome=""`, which with `self_judge=False` also lands on failure. ACE normalizes synonyms; RB does not.

**15. [question, dead-config] `max_items` is a partially dead knob.** Values >3 are inert (schema `maxItems: 3` and "at most 3" prompt text are hardcoded); values <3 only truncate post-hoc in `_emit` while the prompt still says 3.

**16. [note] Embedding substitution.** Upstream: gemini-embedding-001 3072-d, asymmetric (instruction-prefixed query vs RETRIEVAL_DOCUMENT cache, `memory_management.py:204-211`). Ours: framework embedder, symmetric. Consistent with project infra-substitution policy; the item-level `embedding_text = title\ndescription` is our convenience mode with no upstream counterpart (upstream never embeds items, only task queries).

## Cross-check (strategies type)

**17. [verified] TaskGraphExpansion inertness claim is true — and the premise about ACE is false.** ACE does not write to "strategies" at all (`produces = ("playbook",)`; all its ops target "playbook"). RB strategies payloads carry exactly `{id, title, description, content, outcome, embedding_text}` (organizer.py:325-332) — no `kind`, no `task_edges` — so `TaskGraphExpansion` (steps.py:654-709) skips them at every gate: edge expansion requires `kind=="trajectory"` (line 658), the titles seed set requires it (line 684), insight recall requires `kind=="insight"` (line 691). Only field the two methodologies share is `title`, and it is only read behind the kind gate. RB items pass through unmodified.

**18. [behavior] Residual G-Memory→RB feedback leak in a mixed-active config, contradicting the facade docstring's "fixed" claim.** `gmemory.on_retrieval` records EVERY served "strategies" id into `_served` (`organizers/gmemory/organizer.py:161`) — including ReasoningBank items when both organizers are active. `on_feedback` (lines 423-439) then applies +1/-2 score to any fed-back id found in "strategies", and the served-set gate is `if self._served and i["id"] not in self._served` (line 429) — bypassed entirely while `_served` is empty. So an RB item (append-only by design, no feedback semantics in the paper) can still pick up a G-Memory `score` field: always when `_served` is empty, and whenever it was served in a mixed bundle. It is protected from pruning only by the `kind=="insight"` gate (line 440). This is exactly the conflation `memory.py:873-891` says the hook fan-out eliminated — the claim holds for single-methodology configs only. Also [doc]: gmemory:158 says "served insight ids"; the code stores all strategies ids (trajectories, insights, and foreign RB items).

## Verified-clean summary

- **ACE:** two-role reflect→curate loop as two LLM calls; ADD-only reproduced; verbatim feedback strings; 80k budget; per-task curation = frequency 1; stats buckets; one shared playbook-line format across `Bullet.render`/`get_playbook` via `render_bullet_line`; full-playbook curator view; multi-round reflection correctly excluded; both upstream variants (ACE/ACEBatch) and the eval-harness path checked.
- **ReasoningBank:** success/failure SIs; judge-reason append; experiences-as-retrieval-unit with member expansion; MaTTS parallel hook real, wired, and tested, with all three upstream deltas (5 items / 1-5 sentences / t=0.7) correctly mapped; dead per-trajectory label correctly reproduced-as-dead; sequential scaling correctly excluded; item schema title/description/content; both upstream variants (WebArena + SWE-Bench minisweagent) checked and consistent.

The load-bearing items to act on later (no fixes made, per instructions): #1/#2 (ACE prompt/mechanism deltas vs the published operating point), #5 (playbook reachable via default top-k search), #18 (mixed-config feedback leak + gmemory docstring), #10 (RB operating point unpinned).

> **Status as of 2026-08-08** (see the banner at the top of this file — this list is a snapshot, not a live tracker):
> - **#18 CLOSED** by commit `6fad7bb`. `gmemory.on_retrieval` now records only `kind == "insight"` ids and the empty-`_served` bypass is gone, so a ReasoningBank item cannot acquire a `score` in a mixed config. Catalog GM-11.
> - **#12 CLOSED** as documentation (ledger B-7) and now carries a Tier-0 proof; **read with ledger A-5**, which shows the whole function it lives in raises before reaching a model.
> - **#14 CLOSED in full.** The `self_judge=False` half landed in round 12; the judge-GATE half (a recognized label such as `"correct"` being discarded and re-judged under the default `self_judge=True`) landed 2026-08-08 via `_is_labeled`.
> - **#13 UPGRADED from argument to fact**: sequential MaTTS is not merely "the generator's loop" — `SEQUENTIAL_PROMPT` has **zero references** in the pinned snapshot. Proof: `repro_reasoningbank_matts_inert.py`.
> - **#10 STILL OPEN**: `RB_READ_RECIPE` exists as a citable constant, but no preset applies it and `default_memory_types` still serves episodic + experiences + strategies at k=10. Blocked behind the in-flight campaign, not behind a decision — `scripts/repro/configs.py` is shared with running measurements.
> - **#1/#2/#5 (ACE)** unchanged by this round.

---

# [memmachine]

All verification is complete. Here is the fidelity re-audit report.

---

# MemMachine fresh-eyes re-audit — findings

## Numbered findings

**1. [question] At the audited SHA (18f1211), upstream's `init_memmachine_params` cannot execute at all — which undercuts our docs' "which harness runs at HEAD" story, though not claim 1's substance.**
`evaluation/utils/agent_utils.py:452-459` calls `LongTermMemoryParams(session_id=..., vector_graph_store=..., embedder=..., reranker=..., message_sentence_chunking=...)`, but `LongTermMemoryParams` is `Annotated[DeclarativeBackendParams | EventBackendParams, Field(discriminator="backend")]` (`packages/server/src/memmachine_server/episodic_memory/long_term_memory/long_term_memory.py:162-165`). I reproduced the exact construct with pydantic: calling the Annotated union raises `TypeError: 'types.UnionType' object is not callable`. So at this SHA **every** eval entry point through `init_memmachine_params` crashes at construction — not only the legacy `evaluation/episodic_memory/locomo_*.py` scripts via the 3-of-5 tuple unpack that `docs/research/memmachine.md` §4.3 records. §4.3's claim that `evaluation/retrieval_agent/` is the "권장 경로" that unpacks correctly (implying it runs) is false at HEAD; the published numbers necessarily predate the union refactor. Claim 1's substance survives: the kwargs match only `DeclarativeBackendParams` (declarative variant), and `EpisodicMemoryParams(..., short_term_memory=None)` is verbatim at `agent_utils.py:466`.

**2. [behavior] Our "verbatim" profile update prompt drops one word — "preferences," — from upstream.**
Byte-diff of the assembled prompts: upstream `semantic_memory/util/semantic_prompt_template.py` guideline reads "Names, basic demographics, **preferences,** and any personal details should ALWAYS be extracted."; ours (`src/agmem/organizers/memmachine/profile.py:304`) omits "preferences,". Everything else — all 37 tags and their order, `PROFILE_DESCRIPTION`, and the entire consolidation prompt including the missing comma in the no-op example — is byte-identical. A one-word diff in a claimed-verbatim extraction prompt is exactly the class of drift the porting notes forbid.

**3. [behavior] Profile `delete` is truncated by the 50-feature visibility page in ours; upstream deletes storage-wide.**
Upstream `semantic_ingestion.py:306-323` executes DELETE as a storage-level filter over `(set_id, category, tag, feature)` — unbounded, independent of the `max_features_per_update` page shown to the LLM. Ours (`profile.py:584-593`) iterates only `existing`, which is `self._features(...)[: self.max_features_per_update]` (`profile.py:524`) — so once a category exceeds 50 live features, a delete command leaves matching rows beyond the page alive. Reachable: the per-tag consolidation threshold is 20, but the page cap is per-category.

**4. [behavior] An unknown command verb becomes a DELETE in ours; upstream drops the whole message.**
Upstream `SemanticCommand.command` is an enum (`semantic_model.py:28-34`); a verb like `"update"` fails pydantic validation inside `llm_feature_update`, the exception escapes, and the caller logs and `continue`s (whole-message drop, `semantic_ingestion.py:221-240`). Ours (`profile.py:569-581`) checks only that all four keys are truthy, then treats anything not `"add"` as a delete. With `use_guided_json=False` (our experiment script's setting) the schema enum is not hard-enforced, so this path is live. Related edge in the same check: an empty-string `value` (accepted by upstream's `value: str`) is treated as malformed by our truthiness test and drops the whole batch.

**5. [behavior] Our SplitQuery caps sub-queries at 6; upstream enforces no cap in code.**
The 1-6 range is prompt-contract only; `split_query_agent.py:176-182` takes every non-blank line of the response. Ours slices `sub_queries[: self.max_queries]` with `max_queries=6` (`policies/retrieval.py:367, 383`). That's a silent hardening of upstream behavior — the same "fix vs reproduce" line the module elsewhere refuses to cross (it deliberately keeps `dedupe=False`).

**6. [behavior] ChainOfQuery presents the document pool to the sufficiency LLM in a different order than upstream.**
Upstream sorts the pool chronologically before assigning `[idx]` labels (`coq_agent.py:206-211`: `sorted(set(retrieved).union(evidence), key=created_at)`); ours builds it in dict-insertion order — round hits then evidence (`policies/retrieval.py:464`). Internally consistent (our `evidence_indices` index our own pool), but the model sees a differently ordered context, which can shift its sufficiency/evidence decisions. Not documented anywhere as a deviation.

**7. [doc→behavior] The module docstring claims "The task prompts are verbatim; only the envelope changes" — they are not.**
`policies/retrieval.py:70` vs the per-prompt comments that admit otherwise: `TOOL_SELECT_PROMPT` is "abridged to its mechanism" (upstream `tool_select_agent.py:21-81` includes 5 calibration examples and validation steps ours drops), `SPLIT_QUERY_PROMPT` drops upstream's 6 worked examples and the pronoun-resolution template (`split_query_agent.py:73-107`), and `SUFFICIENCY_PROMPT` is a heavy condensation of upstream's `COMBINED_SUFFICIENCY_AND_REWRITE_PROMPT` (input-normalization steps, edge cases, and the "EXACTLY these keys" constraint are cut), not merely re-enveloped. Prompt text is the routing/splitting distribution; the headline sentence contradicts the code and the local comments. Self-contradicting docstring — defect class 3.

**8. [behavior] The `event` preset reads through the declarative read path — cross-backend mixing inside one preset, the exact thing `MEMMACHINE_PRESETS` promises not to do.**
`retrieval/steps.py:735` registers `MemMachineContextualize` on `derivatives` unconditionally, and that step is a port of `declarative_memory.py::search_scored` (episode-level contexts, `_weighted_index_proximity` unify, required-reranker semantics, chronological-per-context). Upstream's event backend reads differently: segment-level context expansion (`event_memory.py:450-451`), embedding-score fallback when the reranker is None (`event_memory.py:474-478`), first-seen dedup by `_episode_uid` in score order with a 4x dedup over-fetch, and **no** weighted-proximity unification (`long_term_memory.py:307-378`, `_EVENT_BACKEND_DEDUP_OVERFETCH = 4` at line 106). `organizer.py:26-28` states "provenance never mixed inside one preset"; for the write path that holds, for the read path it does not. No published number uses the event backend, so this mislabels only our own `fidelity="event"` runs — but it mislabels them.

**9. [doc] `recent_context` docstring says "This follows the library" — the library's context includes long-term results too.**
Upstream `formalize_query_with_context` (`episodic_memory.py:478-545`) merges STM episodes **and** long-term search results under `<Episodes>`; ours (`organizer.py:496-515`) injects only the STM buffer. Defensible seam mapping (LTM is served by our search pipeline), and unreachable in both presets since STM is off — but the sentence overstates the match. The `<Summary>`/`<Episodes>` tags themselves are faithful.

**10. [doc] Minor docstring inaccuracies in `policies/retrieval.py`.**
(a) `SplitQuery.run` comment: "Upstream falls back to the original query when the split yields nothing ..., including when the call fails" — upstream only handles the empty-output case; an LLM exception propagates uncaught (`split_query_agent.py:172-182`), there is no failure fallback. (b) ChainOfQuery comment "retries once, then proceeds" — upstream's "retry" re-parses the *same* response string once (`coq_agent.py:226-243`), it never re-calls the LLM; our wording implies more than it is.

**11. [question] `consolidate_every=5` is an approximation of upstream's trigger, and the docstring's framing slightly overstates it.**
Upstream consolidation runs after each ingestion pass over a batch of **up to** 5 un-ingested messages (`semantic_ingestion.py:127-129, 274-277`); if ingestion cycles run per-message, the check effectively runs per message, not every 5th. `profile.py:475-483` presents 5 as "what actually sets how often the check runs." Low impact (the threshold gate dominates cost), but the mapping is an interpretation, not an equivalence.

## Verified clean

- **Claim 1 substance**: declarative-variant kwargs + `short_term_memory=None` (`agent_utils.py:452-469`); discriminator's "missing means declarative" rule (`common/configuration/episodic_config.py:13-46`); `DeclarativeMemoryParams.reranker` required (`declarative_memory.py:64-67`).
- **Claim 2, all four (declarative)**: derivative→source-episode nuclei with order-preserving dedup (`declarative_memory.py:369-396` / ours `steps.py:483-502`); `expand//3` backward, rest forward (`declarative_memory.py:398-400` / `steps.py:492-494`); reranker scores assembled context strings, never the derivative (`_score_episode_contexts`, `declarative_memory.py:499-514` / `steps.py:565-592`); `_weighted_index_proximity` exact math incl. nucleus at −0.25, and the `break`-not-`continue` unify (`declarative_memory.py:616-683` / `steps.py:518-534, 595-603`).
- **Served-order deviation still true and documented**: upstream sorts globally chronological (`_unify...` lines 663-669; `_do_rerank` sorts by `created_at` in **all three** branches, `agent_api.py:97-137`), our `MemoryBundle.render` sorts by score; documented in `steps.py` docstring, `policies/retrieval.py` adaptation 1, and reproduced via `_rescore`.
- **Claim 3**: `grep 'policy\.' retrieval_agent/` → zero hits; `MemMachineAgent.do_query` opens `_ = policy` (`memmachine_retriever.py:52`); `extra_params={}` at all three construction sites (`agent_utils.py:350-376`, `service_locator.py:28,38,54`) so `max_attempts=3`, `confidence_score=0.8` class defaults rule (`coq_agent.py:139-140`), while the harness's dead `QueryPolicy(confidence_score=10, max_attempts=3)` (`agent_utils.py:80-87`) never reaches them; `result.extend(res)` no dedup (`split_query_agent.py:197`), our `dedupe=False` default matches; unseparated rerank concatenation (`split_query_agent.py:209-212`, `coq_agent.py:354-355` — and yes, `used_query[0]` is the original query, so it is glued onto itself) reproduced at `retrieval.py:398, 496` including the `len>1` guard.
- **Chain-of-query union semantics**: evidence ∪ LAST-round hits only (`coq_agent.py:249-252`, evidence promotion via bounds-checked `evidence_indices`), parse-failure → `new_query` defaults to original → repeat → break; all mirrored in ours.
- **ToolSelect**: substring-over-ordered-children matching (`tool_select_agent.py:168-171`), child order [split, coq, memory], default tool = ChainOfQuery at every caller — ours identical.
- **Claim 4, all three defects verbatim upstream**: `consolidate_memories` (prompt, `semantic_prompt_template.py:225,232`) vs `consolidated_memories` (parser, `semantic_llm.py:122`); delete example missing schema-required `value` (`SemanticCommand`, all four required) with whole-message drop granularity (`semantic_ingestion.py:215-240`); fourth few-shot array closed with `}`. Our reproduction reads both spellings with a log (`profile.py:672-679`), counts `dropped_commands` (`profile.py:559`), keeps `keep_memories is None → no-op`, unions deleted features' citations, reverts renamed tags, embeds the value alone — all matching upstream (`semantic_ingestion.py:466-520`, `290-303`).
- **Claim 5**: `locomo_config.yaml:14-15` sets `semantic_memory.enabled: false`; the episodic write path (declarative and event) has zero LLM calls; the semantic tier loops messages inside a loop over categories (`process_semantic_type`, `semantic_ingestion.py:186-258`) = 1 call/message/category, exactly as `profile.py`'s header states.
- **Operating points**: `limit=30, expand_context=3` (`evaluation/episodic_memory/locomo_search.py:91-92`) vs library defaults 0/20 (`query_memory`, `declarative search_scored`) — split into config exactly as claimed; the derivative over-fetch `min(5*limit, 200)` (`declarative_memory.py:365`) is not a config knob but IS correctly encoded at the operating-point layer (`scripts/exp_locomo_conv0.py`: k=150 for `memmachine` 30/3, k=100 for `memmachine_library` 20/0, both with `memory_types=("derivatives",)`).
- **No dead knobs on our side**: `memmachine_expand_context`/`memmachine_context_limit` flow `config.py:127-128,225-230 → memory.py:251-252 → pipeline.py:102-103 → steps.py:735`; `query_strategy`/`query_strategy_limit` flow through `searcher_for` (`planned.py:104-118`) and it is called at all three entry points (`bench/locomo.py:550`, `bench/longmemeval.py:306`, `mcp/server.py:47`); the script's `agent_limit` maps to `query_strategy_limit` (`exp_locomo_conv0.py:207-209`).
- **Write-path details**: `f"{source}: {content}"` / per-sentence anchors (`_derive_derivatives`); LoCoMo ingest sets `producer_id = speaker` (`locomo_ingest.py:91`); all 30 `TextSegmenter` separators byte-identical in order, `chunk_size=500`, `chunk_overlap=0`, `keep_separator="end"`; `extract_sentences` regex identical (required class has no ASCII period — ours matches) and set semantics reproduced; event anchor `[{babel full date}] {producer}: {json.dumps(...)}` with `FormatOptions(time_style=None)` and `date_style="full"`/`en_US` defaults confirmed (`text_deriver.py:36-53`, `data_types.py:200-203`); the babel-vs-strftime zero-padding split between backends is real and correctly reproduced (`_format_date` at `declarative_memory.py:532-535` vs our `format_full_date`); `episodes_to_string` line format identical (`episode_model.py:68-99` / our `_memmachine_line`).
- **STM**: both prompts byte-identical to `common/configuration/default_episode_summary_{system,user}_prompt.txt` (user prompt differs only in trailing whitespace); `message_capacity` default 64000, `sample_configs` 500; eviction loop (`len > current_episode_count`, summarize whole remaining buffer, count reset) and `max_summary_length_words` = capacity/2/8 rounded **up** to 100 all identical (`short_term_memory.py:146-157, 222-240`); `set_summary`'s `if summary:` empty-rewrite guard reproduced; the inline-vs-background and structured-envelope deviations are documented.
- **Rerank branch structure** in the policy (`limit<=0` / within-limit / score) mirrors `_do_rerank`, with positional keys correctly preserving duplicates; upstream's chronological tails are the one documented drop.

No file was modified.

---

# [amac]

REPORT — Fresh-eyes re-audit of the A-MAC admission gate vs the release at `/home/jinmang2/.agmem/upstream/amac` (single-commit repo; all upstream paths relative to that root; ours absolute).

## Claim 1 — the four documented upstream defects: ALL CONFIRMED

1. **[behavior-upstream] (a) N pinned at 1.0, R at ~0.** Two independent causes, both present on *both* executable paths (the fitting path and the README-table eval path):
   - N: `optimize_weights_cv.py:78` and `run_all_baselines.py:98` call `extractors['novelty'].score(memory, [])` — existing memories hardcoded to `[]`; `features/novelty.py:47-49` returns `1.0` for an empty list, so SBERT is never consulted. N == 1.0 for every candidate.
   - R: `optimize_weights_cv.py:66,79` / `run_all_baselines.py:84,99` pass `current_time=time.time()`, while `data_loader.py:110-119` seeds LoCoMo timestamps from `"2023-05-01 10:00:00"` (fallback `datetime(2023,5,1)` + weekly offsets). With `features/recency.py:44-54` at lambda=0.01/h, age is tens of thousands of hours → `exp(-hundreds)` ≈ 0 for every candidate. Not literally clamped to 0.0 — a denormal-tiny positive — but constant to ~120 decimal places. Our docstring's "R == 0.0" (admission.py:29) is fair.
2. **[behavior-upstream] (b) bare-substring type keywords, fact prior 0.7 dominating.** `features/type_prior.py:129` — `if keyword in content` on `memory.content.lower()`; fact set `{'is','are','was','were',...}` at :69-72 with prior 0.7 at :38. `'is' in "sister"`/`"this"` is true, so nearly any substantive turn classifies fact. Under the release weights, T=0.7 contributes 0.42; with N=1.0 (+0.1), U fallback 0.5 (+0.05), R=0, base = 0.57 + 0.1·C ≥ theta 0.55 always — the fact-vs-unknown distinction alone *is* the admission decision. Confirmed arithmetically.
3. **[behavior-upstream] (c) 5-fold CV that never fits.** `optimize_weights_cv.py:180-185`: the fold loop assigns `X_train, X_val = ...` and `y_train, y_val = ...` (:181-182) but only calls `evaluate_weights_threshold(X_val, y_val, weights, threshold)` (:185). There is no fitting step anywhere in the loop — the procedure evaluates each *fixed* candidate (weights, threshold) pair on every val fold and averages (:188), so "CV" is just averaging the fixed grid's score over a partition of the same data; selection (:198-201) then reads the best off that, and `final_result` (:204) re-scores on the full data. `X_train`/`y_train` are dead locals — dead-config-knob confirmation requested in the task. (Note the separate `weight_optimizer.py` — read to avoid the "only one variant" defect class — also never fits: its Bayesian objective evaluates on `val_data`, and `val_data=train_data` when None, :69-70. And it passes `candidate.dialogue_turn` — which has `.text`, not `.content` — into extractors at :110-113, plus `existing_memories=[]` again at :113. It is not on the path that produced the operating point; `run_all_baselines.py:382-385` loads `results/optimized_weights_cv.json`, i.e. the CV script's output.)
4. **[behavior-upstream] (d) no POS/NER; regex-only in a non-default subclass with a spaCy TODO.** `features/type_prior.py:164-189`: `AdvancedTypePriorExtractor` adds 3 regex pattern groups; the TODO "Could integrate with spaCy NER" is at :168. The pipeline instantiates plain `TypePriorExtractor` (`optimize_weights_cv.py:246`, `run_all_baselines.py:74`), so even the regexes are dead. One phrasing nit on the task's claim wording: this is type *classification*, not entity extraction — there is no entity extraction anywhere upstream. Our docstring (admission.py:40-43) phrases it correctly.

## Claim 2 — the operating point: CONFIRMED, with a citation error in our comment

5. **[behavior] Values verified.** `[0.1, 0.1, 0.1, 0.1, 0.6]` / `threshold=0.55`, order `[U, C, N, R, T]`, live at upstream `README.md:79-80` (labelled "learned weights (optimized via cross-validation)"); the README Table row "Ours 0.417 / 0.972 / 0.583" is at `README.md:31`. The vector is exactly reachable as CV "candidate class 3" (`optimize_weights_cv.py:160-165`: `w = ones*0.1; w[i]=0.6`, sum already 1.0) and 0.55 is in the threshold grid (:168). Ours: `PAPER_WEIGHTS`/`PAPER_THRESHOLD` at `/home/jinmang2/agentic_memory/src/agmem/policies/admission.py:325-326`, wired as `AdmissionGate.__init__` defaults (:472-473), verbatim equal, same feature order (`decide` at :638-644 matches upstream `[u,c,n,r,t]` at `optimize_weights_cv.py:82`).
6. **[doc] Our source citation is wrong.** admission.py:322 says "(features/README + optimize_weights_cv candidate class 3)" — there is **no** `features/README` upstream (`features/` contains only `__init__.py` + the five extractors). The operating point lives in the repo-root `README.md`. Same comment also calls it the "Paper's cross-validated operating point", contradicting the module docstring's own careful correction 270 lines earlier (admission.py:47-49: "'Release's', not 'published'"). Two-line comment fix territory; the docstring itself is right.
7. **[question] The release never wires the operating point into runnable code.** `scorer.py`'s code defaults are equal weights `[0.2]*5` / threshold 0.5 (:60-65, :51); the README quick-start that shows the operating point is not runnable against the release's own classes (imports nonexistent `WeightOptimizerCV` — the class is `WeightOptimizer` at `weight_optimizer.py:16`; constructs `MemoryCandidate(turn=...)` vs the real field `dialogue_turn`; calls `scorer.score(candidate)` where `score` takes a `Features`). The operating point is real (CV-JSON → `run_all_baselines.py:382-385` consumes it) but "release's operating point" means the README + results artifact, not a code default. Worth one sentence wherever we cite it.

## Claim 3 — substring mode byte-faithfulness: CONFIRMED empirically

8. **[verified] `type_matching="substring"` reproduces the release classifier exactly.** Read-only comparison run (upstream `type_prior.py` imported standalone vs our `TypePriorClassifier(matching="substring")`): priors dict equal, all six keyword sets equal element-for-element (including `"i'm"` escaping), type-dict iteration order identical (`preference, identity, fact, plan, goal, temporary`); 5015/5015 classify+score matches on adversarial and random inputs, and crafted equal-count tie probes (`"want future"`, `"is will"`, `"age currently"`, …) all resolve identically. Tie-breaking is equivalent by construction: upstream `max(type_scores.items())` returns the first maximum in insertion order (`type_prior.py:136`); ours uses strict `>` over the same order (admission.py:289-291). Upstream's keyword containers are `set`s, but order only affects counting, not counts — no divergence possible there.
9. **[verified] The debugged mode is clearly labelled.** `"word"` mode changes exactly one thing: whole-token/phrase containment via padding (`_padded`, admission.py:243-247, test at :288) instead of raw substring. It is the default (:479), and the class docstring (:255-266) explicitly brands substring "a defect, not a preference". Config labels are unambiguous: `scripts/exp_locomo_conv0.py:474-481` `amem_amac` (word) vs :486-493 `amem_amac_upstream` (substring), each with an honest comment.
10. **[question] But the *default* pairs release-fitted weights with debugged features.** The module docstring itself states the weights "are *not* transferable to the debugged features" (admission.py:50-54), yet `AdmissionGate()` with no arguments is exactly that pairing (release weights+theta, word matching, live N, R≡1). The exp comment (:472-473) flags that numbers need re-tuning first, so it's disclosed at the config site — but anyone constructing a bare `AdmissionGate()` elsewhere gets a hybrid that is neither the release nor a tuned debugged gate, silently.
11. **[doc] "amem_amac_upstream" restores only defect (b), not the release's feature constants.** exp_locomo_conv0.py:482-485 says substring matching "is what its published recall 0.972 was produced under" — true but incomplete: 0.972 was also produced under N≡1.0, R≈0, U=Ollama-or-0.5, whereas that config runs N live, R≡1.0, U absent (=0). For a fact turn the release base is 0.57+0.1C vs ours 0.52+0.1N+0.1C — similar outcomes, different arithmetic. No config reproduces the full release operating conditions; if the pairwise contrast is ever written up, this should be stated.

## Claim 4 — attachment and scope

12. **[verified] The gate attaches only via the wrapper.** Functional importers of `agmem.policies.admission`: `organizers/gated.py:57`, the package re-export `policies/__init__.py:79`, the config site `scripts/exp_locomo_conv0.py:28,33` (builds the gate solely to pass into `AdmissionGated`), and tests. No organizer, no `memory.py`, no harness import; mentions elsewhere (`_porter.py`, `bench/locomo.py`, `retrieval/planned.py`, `policies/retrieval.py`) are docstrings only. No organizer has an `admission=` parameter.
13. **[behavior] The mechanism-agnostic claim has one real payload assumption: `novelty_types` defaults to A-Mem.** `AdmissionGate(novelty_types=("notes",))` (admission.py:480) is `AMemOrganizer.produces` verbatim (`organizers/amem/organizer.py:159`). `gated.py:26-27` lists `zep_graph` and `memoryos` as "Fits", but zep produces `("facts","entities","communities")` (`zep_graph/organizer.py:269`) and memoryos `("pages","semantic")` (`memoryos/organizer.py:325`) — and `AdmissionGated.__init__` (gated.py:67-76) mirrors `produces` for attribution but never feeds it into `gate.novelty_types`. So `AdmissionGated(ZepGraphOrganizer(), AdmissionGate())` searches type `"notes"`, finds nothing, and N is silently pinned at 1.0 (admission.py:554-562) — upstream defect (a)'s shape, reintroduced per-host. admission.py:488-489 does say novelty_types should be "the host organizer's `produces`", so it's caller responsibility by contract, but nothing warns or defaults from the wrapped organizer. Everything else the gate touches is universal: `Episode.content/timestamp/meta` (all with defaults, `core/types.py:60-68`), `ctx.embedder/vector_store/llm` — no host-specific payload fields.
14. **[behavior] `warm_start` path pins N=1.0 for the whole corpus.** `gated.py:96-102` decides *every* corpus episode before calling `wrapped.warm_start(admitted)` — necessary for the documented Nemori/MemoryOS bypass reason (:93-95), but it means on a fresh store no host notes exist during any decision, so N=1.0 throughout warm-start, whereas on the `on_message` path N is live turn-by-turn. The two ingest paths therefore implement *different* gates and the docstring doesn't say so. Mitigating fact (defect class 2 — which path the harness calls): the LoCoMo harness ingests via `mem.add_message` (`bench/locomo.py:456`), i.e. the on_message path, and raw episodes are indexed as `"episodic"` (memory.py:447-452), not `"notes"`, so measured runs are unaffected — this is a latent divergence, not a corrupted measurement.

## Claim 5 — feature-formula parity

15. **[verified] Confidence (C).** Pre-filter identical: raw lowercase whitespace-split word overlap > 0 (upstream `confidence.py:86-91`; ours admission.py:533-537, empty spans skipped equivalently). ROUGE-L argument order matches: upstream `scorer.score(evidence_span, memory_text)` (:108) = ours `rouge_l_fmeasure(span, content)` (:538) — target=span, prediction=candidate. Max over spans, 0.0 when no span, both sides. Window: upstream scans all prior turns; our default `history_window=None` does too (:481, :529-531). History semantics match: upstream `conversation_history = dialogue_turns[:i]` — strictly before the candidate (`data_loader.py:188`); ours appends the candidate to `_history` only *after* deciding (:679), so no self-match either. Caveat, documented at admission.py:83-87/122-127: ROUGE-L is a transcription of `rouge_score`'s tokenizer + vendored Porter stemmer with a known ~1.4%-of-vocabulary residual vs nltk — parity is functional, not byte-exact, and our tests pin reference values.
16. **[verified, documented deviation] Novelty (N).** Formula 1 − max cosine, clamped, empty→1.0 on both sides (`novelty.py:34-65` vs admission.py:541-563). Deviation (declared in the docstring): ANN top-1 per type via the run's own vector store/embedder instead of a full SBERT scan — value-equivalent given the store contract (chroma/lance both convert distance→similarity, `stores/chroma_vec.py:84-86`, `lance_vec.py:87-89`). One asymmetry nobody documents because upstream never exercises N: a negative top-1 cosine would give upstream N>1→clamp-to-1, ours keeps `best_similarity=0`→N=1 — same result.
17. **[verified] Recency (R).** Same `exp(-lambda·age_hours)`, same default lambda 0.01, equivalent clamping (upstream doesn't clamp negative age but clamps the result to ≤1, `recency.py:44-54`; ours clamps age to ≥0, admission.py:579-581 — identical outputs). The structural deadness (ours R≡1.0 streaming, release R≈0 replayed) is candidly documented at admission.py:565-578. Upstream's `AdaptiveRecencyExtractor` (type-specific decay) is dead code, correctly not ported.
18. **[behavior, under-documented deviation] Utility (U) unavailable-fallback differs: 0 vs 0.5.** Upstream returns 0.5 on any failure and substitutes 0.5 when the LLM is off (`utility.py:57-58`, `optimize_weights_cv.py:73-75`); ours returns `None` and decides on the U=0 lower bound (admission.py:659-662, "Identical to treating an unavailable U as 0"). At the paper weights this makes our LLM-free gate 0.05 stricter than the release's LLM-free path (admit iff base ≥ 0.55 vs effectively ≥ 0.50). The 0-vs-0.5 choice is stated in our code but never flagged as a *deviation from upstream's fallback* — worth one line. Other U deviations are declared (JSON via StructuredCaller vs bare-float+regex parse `utility.py:132-154`; dedicated temp-0 `admit` role). Prompt: context framing and rating sentence near-verbatim; final instruction necessarily differs (JSON vs "single floating-point number… Score:"); context window 5 matches (`utility.py:109` vs :503); turn labels use local enumerate index vs upstream's real `turn_id` (`utility.py:112-114` vs :607-608) — cosmetic. The short-circuit (:650-651) is our addition, correctly described as exact and cost-only.

## Dead-config knobs (defect class 4)

19. **[verified] Upstream dead knobs confirmed:** `X_train`/`y_train` (finding 3); `Memory.metadata` type branch (`type_prior.py:116-121`) dead — pipeline always passes `metadata={}` (`optimize_weights_cv.py:32`, `run_all_baselines.py:37`); `AdvancedTypePriorExtractor`, `AdaptiveRecencyExtractor`, `scorer.MemoryAdmissionScorer`'s 0.5-placeholder path, and `weight_optimizer.py` entirely — none on the operating-point path.
20. **[doc] Ours has two dead members of its own:** `AdmissionFeatures.to_tuple` (admission.py:342-345) has zero callers repo-wide; and the `declared_type` short-circuit reads `episode.meta.get("content_type")` (:636) but nothing anywhere sets `content_type` — our docstring (:277-279) notes *upstream's* equivalent branch is dead in its pipeline without noting ours is equally unreachable in ours. Both harmless, both worth a word.

## Verified clean

- Weight validation (non-negative, sum-to-1, len 5) matches the release's asserts (`scorer.py:67-69` vs admission.py:491-496).
- Feature order (U,C,N,R,T) consistent everywhere on both sides.
- Wrapper transparency: name/produces/consumes mirrored, `_cursor_scope` forwarded (safe — class-level default at `organizers/base.py:222`), on_task_end/retrieval/feedback/consolidate/retire/patch_unit all pass-through untouched; `on_task_end` correctly ungated with a correct rationale.
- Rejected turns cost 0 LLM calls (gate is LLM-free by default; wrapped organizer never invoked on reject, gated.py:84-86).
- Gate statefulness (one gate per conversation) is honored by the config's 0-arg factories (`exp_locomo_conv0.py:475,487` — fresh `AdmissionGate()` per lambda call).
- `admit iff S ≥ theta` (>= , not >) matches `scorer.py:184`.
- Module docstring's four-defect narrative (admission.py:23-49) is accurate against the release in every checked particular, including the "release's, not published" weights caveat and the recall-0.972 attribution (README.md:31).

Nothing was modified. The scratchpad comparison script is at `/tmp/claude-1000/-home-jinmang2-agentic-memory/cb97c89c-bbac-4fc7-95b2-8cc54bf70a14/scratchpad/cmp_typeprior.py`.


---

# 검증 라운드 (2026-07-28, 2차 8병렬 — 반증 지향, LLM/API/모델추론 0)

수단: 양측 코드 라인 직독 + 수제 벡터/스텁 결정적 재현. 각 검증자는 원 발견을 깨는 것이 임무.

## 판정 집계

| 방법론 | 판정 | 측정-무효화 확정 | 비고 |
|---|---|---|---|
| A-Mem | 6 CONFIRMED / 1 UNVERIFIABLE | #1(첫 노트 콜 ±1), #3(LINK 순서→cap 시 확장 대상 상이, 스텁 재현) | #7은 agiresearch 스냅샷 부재로 판정 불가 |
| Nemori | 7 CONFIRMED / #8 PARTLY / #6 절반 | #1(사문 0.85 실동작화, grep read 0회), #2(window=게이트, chunk_max 전 경로 도달불가 증명), #4, #5, #8 | #8 정정: upstream 후보에 Title: 포함(merger.py:180) |
| MemoryOS | 13 CONFIRMED / #7 PARTLY | #1(하네스가 반쪽 단위 전달 확인 — 반증 시도 실패), #2~6, #11, #14, #13(pypi), #8 실질 | #7 하위주장 반증: eval도 empties/"- None" 거름(long_term_memory.py:68-71) |
| Zep | 15/15 CONFIRMED | #1~8, #12 (#8 순서반전 수치 재현, #12 3-노드 재현) | |
| G-Memory | 14/14 CONFIRMED | #1,2,3,6,7,11 + 2차 4,5,9,10 | #1 실증: chromadb 1.5.9 기본 l2·제곱거리, cos0.85→1−d=0.700 정확히; 정규화는 캐시 modules.json의 2_Normalize로 무로드 확인. #6 실증: 스텁 훅 구동으로 traj score −1.0 변형 + _served 빈 상태 우회 |
| ACE+RB | 18/18 CONFIRMED | #1,2,3,5,10,14,18 (#5 playbook top-k 서빙, #18 RB score 변형 — 둘 다 스텁 재현) | #1의 논문 dedup 설정 자체는 UNVERIFIABLE(하네스/README 기본값에서 건전 추론) |
| MemMachine | 11/11 CONFIRMED | #3,4,5,6(profile/agent 런), #8(event 런), #2(verbatim 주장) | #1 pydantic 재현(TypeError, 버전 무관); #2 바이트-diff 1 hunk("preferences," 누락); #10(b)는 원 발견보다 강화 — upstream CoQ "retry"는 재시도 0회 |
| A-MAC | 7 CONFIRMED / #20 PARTLY | 현 측정 구성 기준 없음 (#13/#14는 잠재 — #13 스텁 재현: facts-type 0.99 중복에도 N=1.0) | #20 정정: to_tuple/content_type은 테스트가 사용 — "호출자 0"은 과장 |

전체: 판정 96건 중 REFUTED는 하위 주장 2건(MemoryOS #7 empties, A-MAC #20 표현)뿐. 원 감사의 정밀도가 매우 높았음.

## 확정 측정-무효화 대기열 (조치는 별도 승인 필요; "재현 vs 의도적 편차 문서화" 방향 결정 필요 건 다수)

1. **교차 오염(혼합 구성·즉시)**: G-Memory on_feedback의 kind 무필터 + _served 빈 상태 우회(RB 아이템 score 변형 재현됨); ACE playbook의 일반 top-k 도달 가능성.
2. **G-Memory query graph**: 실효 0.85 게이트, embed 텍스트(task-only여야), 후보 슬롯 insight 오염, read operating point(성공 top-1·insight 상관관계 전용·실패 미서빙), finetune 구조(~10콜·랜덤 앵커·relative_tasks 범위).
3. **Zep 시간 엔진+read**: strictly-older 가드/자기만료/valid_at 기본값, 무효화 후보 그래프 전역화, RRF 상수, entity resolution(fuzzy 단계·배치 콜·k15), BFS 방향+0.6 컷.
4. **MemoryOS eval 프리셋**: 반쪽 페이지(단위 결함), 불완전 exchange 드롭, merge 후보 제3방식·키 변형, 테마 배치, knowledge 무한 리스트, dead 프롬프트 배선.
5. **Nemori**: 0.85 필터 제거(프리셋에서), window 의미론(백로그 게이트화), merge 후 calibration 소스, semantic_top_k=20(발표 경로), merge 프롬프트 형상.
6. **MemMachine profile/agent**: delete 페이지 잘림, 미지 verb→delete·빈 value 드롭, "preferences," 복원, SplitQuery 캡, CoQ 풀 정렬, event read 경로 분리.
7. **A-Mem**: 첫 노트 evolution 콜(plain 계보 선택 시), LINK 순서 보존.
8. **문서 일괄**: 각 보고의 [doc] 건 전체 + A-MAC 인용 오류 + 검증 라운드가 추가한 정정 3건.

---

# 조치 진행 상태 (2026-07-28 사용자 일시정지 시점)

## 완료 (미커밋, working tree에 있음)
- **A-MAC 패키지**: admission.py(novelty_types=None sentinel + 공개 5건), gated.py(produces 자동 주입
  + warm_start 문서), exp_locomo_conv0.py(upstream config 불충분성), test_admission_gate.py(신규 1).
  완료 시점 전체 스위트 389 passed / 1 skipped.

## 실행 중이던 Wave 1 (중단 시 부분 편집 가능성 — 재개 시 git diff로 판별)
각 에이전트에 내린 확정 지시(방향 결정 포함):
- **G-Memory**: 게이트 0.85-on-true-cosine(l2 유도 주석, 0.7 코드에서 제거)·trajectory embedding_text=task-only·
  전역 정확 중복검사(full task 텍스트, 중복 시 ADD+엣지 모두 스킵)·correlation 키 full-text화·
  feedback insight-전용+빈 _served 우회 제거·insight vector store 제외(correlation 전용, insight_cap config화)·
  GMEMORY_READ_RECIPE 상수(1/0/3/0.0)·Eq.5 재스코어(가능하면 실구현)·finetune upstream 형상
  (랜덤앵커 5회·회당 2콜·seeded RNG finetune_seed=0·relative_tasks=프롬프트 내 태스크·ADD 무조건+
  프롬프트 접미사 soft cap·list_full 전역 카운트)·실패 태스크 2콜·docs/research/g-memory.md 교정.
- **Zep**: entity resolution main 형상(정규화 lowercase+공백붕괴·MinHash/LSH 3gram/32perm/0.9/entropy1.5·
  모호 exact→LLM 승격·배치 1콜 k≤15)·merge refresh는 동작 유지+paper-shape 재라벨·시간 진리표
  (strictly-older·신규엣지 자기만료·valid_at None 불활성, or ref_time 제거)·무효화 후보 그래프 전역
  dense 검색(하이브리드 스탠드인 공개)·verbatim fast path+배치 내 pre-dedup·rrf_k config(기본 60,
  zep 레시피 1)·BFS directed-outgoing 모드+dense_min_score 0.6 레시피·doc 5건(#9,10,11,13,14,15).
- **MemoryOS**: 페이지=완성 exchange 페어링(반쪽 금지, capacity는 완성 페이지 단위)·불완전 페이지
  드롭 기본+keep_incomplete_pages knob·merge 후보 계보별(pypi=전체 argmax, eval=top-1 cos→threshold)·
  merge 키 동결·테마 whole-batch+2테마 캡·eval knowledge 무한(단 empties/"- None" 거름은 유지—검증
  정정)·eval 프로필 무조건 write·eval Profile Merge 인라인 프롬프트 포트·eval 승격 1콜 섹션마커
  형상·pypi STM 오프바이원(resident=capacity)·PageRecall docstring pypi-전용 귀속+top_sessions=5 knob·
  heat 피드백 입도 공개·profile 임베딩 제외(이중 서빙 제거).
- **MemMachine**: "preferences," 복원(바이트 일치)·delete 저장소 전역·미지 verb=배치 드롭+빈 value
  허용·SplitQuery 캡 제거·CoQ 풀 시간순 정렬·docstring 정정("verbatim"→condensed, CoQ 재시도 0회)·
  event 프리셋용 event-read step(세그먼트 확장·reranker None 폴백·4x overfetch·proximity 없음)·
  docs/research/memmachine.md §4.3 교정(HEAD에서 전 eval 경로 TypeError).

## 미착수 Wave 2 (Wave 1 종료 후 실행 — 파일 충돌 회피용 순서)
- **Nemori**: "upstream" 프리셋 merge_similarity 0.85→None(사문 knob 실동작화 제거; knob 자체는 우리
  확장으로 문서화 유지)·BatchPartitioner를 백로그-grab 의미론으로(트리거 시 전체 버퍼, chunk_max=80
  이 벌크 ingest에서 도달 가능해짐)·merge 후 calibration 소스=병합 양쪽 원문·"upstream" 프리셋
  semantic_top_k=20(발표 경로; v1/v4는 10)·merge 프롬프트 upstream 형상(새 에피소드 time+content만,
  후보 200자 truncation, 후보에는 Title 포함—검증 정정 반영)·topic→boundary_reason 스레딩·
  episode_min_messages knob(기본 1=eval)·프리셋 사문 knob 정리·문서 교정.
- **A-Mem**: 첫 노트 evolution 스킵 유지+±1 콜수·robust-일치 공개(정밀 문서화)·LINK 삽입순+중복
  유지(sorted(set()) 제거)·태그는 plain 무조건 대입 재현(빈 태그 포함; robust 가드 변형 주석)·
  docs/16 "잔여 편차 1건" 교정·fidelity-deep-audit strengthen 행 supersession 표기·embedding 포맷
  3종 주석·agiresearch 스냅샷 부재 명기.
- **ACE/RB**: playbook을 default_memory_types에서 제외(get_playbook=유일 경로 구조화)·curator 프롬프트
  upstream 입력(question_context 추가·reflection 5필드 스키마로 재구성·playbook-size 줄 제거·lessons
  발명 제거·max_ops 캡 제거)·RB outcome 정규화("correct" 등)·max_items 배선(schema/프롬프트 동기화)·
  RB_READ_RECIPE 상수(top-1 experiences-only)+테스트·MaTTS 라벨 반전 주석 1줄·stale 라인 참조 교정.
  (G-Memory feedback 누출 fix는 G-Memory 패키지에 포함됨.)

## 재개 절차
1. `git status`/`git diff --stat`으로 Wave 1 각 패키지의 편집 상태 판별(A-MAC 4파일은 완성 확정).
2. 미완 패키지는 위 지시문으로 에이전트 재투입(완성 패키지와 파일 겹침 주의).
3. Wave 1 전체 확인 후 `uv run pytest tests/ -q` 전체 스위트 → 방법론별 커밋(서사형 본문, 커밋은
   docs/16·round12 참조 명기) → Wave 2 → 전체 스위트 → 커밋 → 본 문서·메모리 갱신.
4. 제약 불변: LLM/API/로컬모델 금지, 측정 금지, git add -A 금지, push는 지시 시에만.

---

# 조치 완결 기록 (2026-07-28)

위 "조치 진행 상태" 섹션은 일시정지 시점의 스냅샷이며, 이후 전 패키지가 실행 완료되었다.
8개 패키지 전부 방법론별 커밋으로 main에 반영:

| 커밋 | 패키지 |
|---|---|
| ed44610 | A-MAC — novelty_types 호스트 추종 + 공개 5건 |
| 6fad7bb | G-Memory — 0.85 게이트·콜수 패리티·insight correlation-전용·feedback 누출 제거 |
| a3bf258 | Zep — 시간 진리표·MinHash 결정 단계·배치 dedupe·rrf_k/dense_min_score |
| 1f4df42 | MemMachine — 프로필 파서 3건·event read step·문서 교정 |
| 7ced0eb | MemoryOS — 완성 exchange 페이지·계보별 프리셋 키 |
| 688c959 | Nemori — 사문 0.85 소생 제거·백로그 의미론·병합 calibration |
| 3038510 | A-Mem — LINK 순서/중복·태그 무조건 대입·문서 정밀화 |
| eba9e97 | ACE/RB — playbook 구조적 제외·curator 입력 정렬·RB 정규화/레시피 |

최종 스위트: **441 passed / 1 skipped** (라운드 시작 전 388/1, +53 테스트).
남긴 것(의도적, 코드 현장에 공개): A-Mem 첫 노트 스킵(robust 계보, −1콜)·LinkExpansion
중복 dedup·MemoryOS keep_incomplete knob·heat 피드백 입도·Zep MMR 그리디(λ=1 동치)·
rrf_k=1 점수 오프바이원(순서는 일치)·G-Memory seeded RNG·hop=1. GraphRecall 융합 방식은
여전히 측정 재개 시 결정 항목.
