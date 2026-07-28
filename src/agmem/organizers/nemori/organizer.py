"""Nemori organizer (arXiv:2508.03341, ACL'26) — compact implementation.

Two-step alignment: an LLM boundary detector watches the message buffer
and cuts a segment on a high-confidence topic shift (or when the buffer
hits ``buffer_max``); an episode generator then rewrites the segment as a
titled third-person narrative with every relative time expression anchored
to an absolute date (the paper's temporal-anchoring, its main Temporal
Reasoning win). Predict-calibrate: existing semantic knowledge predicts
the new episode's content, and only the prediction gap — checked against
the RAW segment, not the narrative — is distilled into atomic semantic
facts (the free-energy "surprise" principle).

Fidelity presets (``fidelity="v1"|"v4"|"upstream"``, see ``NEMORI_PRESETS``)
pick a segmenter (per-message vs batch), an episode-merge stance
(``EpisodeMerger`` on/off, and if on, whose thresholds — the paper's or
upstream's), and a semantic-integration strategy (plain append,
``DedupIdReuseIntegrator``, or ``ThreeWayIntegrator``). Any explicit kwarg
overrides the preset's value for that field; the no-arg default is "v1".

Deviations from the reference system (github.com/nemori-ai/nemori):
- boundary detection defaults to per-message (the paper v1 formalism,
  f_theta over the buffer) rather than the batched
  BATCH_SEGMENTATION_PROMPT the rewritten repo uses for backfill
  throughput — ``fidelity="v4"``/``"upstream"`` switch to the batch
  segmenter
- episode merging (paper-v4 §3.2.3 module, ON by default in the repo's
  eval) is implemented as ``EpisodeMerger``; LoCoMo's multi-day session
  gaps mean upstream's >1h-gap merge ban would block most merges anyway
- v4 semantic new/merge/conflict integration (§3.3.3) is implemented as
  ``ThreeWayIntegrator`` (plus ``DedupIdReuseIntegrator`` for the id-reuse
  dedup proposed in upstream PR#19 — not in the deployed snapshot)
- storage is our MemoryOp pipeline instead of PostgreSQL + Qdrant
- if episode generation fails we emit a mechanical episode instead of
  losing the segment (title = first words, narrative = raw messages);
  upstream's timestamp-parse fallback is datetime.now() (contradicting
  its own prompt) — ours is the first message's date
- on buffer_max the whole buffer INCLUDING the newest message is flushed;
  the v1 formalism flushes M only and keeps m_{t+1} for the next buffer
Upstream temperatures (segmentation 0.2; episode/predict/extract at the
client default 0.7; answers 0.0) are mirrored per-config in
scripts/exp_locomo_conv0.py.
"""

from __future__ import annotations

import logging

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, new_id
from agmem.organizers.base import Organizer, OrganizerContext
from agmem.organizers.nemori.stages import (
    BOUNDARY_PROMPT,  # noqa: F401 — re-exported for train/distill_data.py (nemori.BOUNDARY_PROMPT)
    BOUNDARY_SCHEMA,  # noqa: F401 — re-exported for train/distill_data.py (nemori.BOUNDARY_SCHEMA)
    AppendIntegrator,
    BatchPartitioner,
    DedupIdReuseIntegrator,
    EpisodeMerger,
    PerMessageBoundary,
    Segment,
    ThreeWayIntegrator,
    _fmt,
)

logger = logging.getLogger("agmem.organizers.nemori")

# Preset-value provenance is per-source, never mixed within a preset
# (docs/research/write-path-lifecycle-survey.md §5):
# - v1: paper v1 formalism (per-message boundary, no merge, plain append)
# - v4: paper values (window=20, K_e=5, K_m=5, tau=0.70, no time-gap ban)
# - upstream: github.com/nemori-ai/nemori DEPLOYED-BEHAVIOR values (batch
#   gate 20, chunk 80, merge top-5 with NO similarity floor, >1h gap ban,
#   episode_min 1, semantic top-20). "Deployed behavior", not "config
#   values": upstream defines several knobs its code never reads, and this
#   preset mirrors what the code DOES, not what the config file says
#   (round-12 findings 1, 5, 9).
NEMORI_PRESETS: dict[str, dict] = {
    "v1": dict(
        segmenter="per_message",
        episode_merge="off",
        semantic_integration="append",
        consolidation="off",
        boundary_confidence=0.7,
        buffer_min=2,
        buffer_max=25,
    ),
    "v4": dict(
        segmenter="batch",
        window=20,
        episode_merge="llm",
        merge_top_k=5,
        merge_similarity=None,
        merge_time_gap_hours=None,
        semantic_integration="llm3way",
        integrate_top_k=5,
        integrate_tau=0.70,
        consolidation="off",
    ),
    # buffer_min/buffer_max were REMOVED from this preset: with
    # segmenter="batch" nothing reads either (round-12 finding 9) — keeping
    # them presented inert numbers as active "code values". They mirrored
    # upstream's own four defined-never-read config knobs (buffer_size_max,
    # episode_max_messages, semantic_similarity_threshold, and the merger's
    # similarity_threshold — config.py:73,77,82,86, merger.py:34).
    "upstream": dict(
        segmenter="batch",
        # window == upstream batch_threshold=20: a MINIMUM GATE on the
        # accumulated backlog, not a fixed window (memory_system.py:105-113).
        window=20,
        chunk_max=80,  # _SEGMENT_CHUNK_SIZE (segmenter.py:15)
        # Both eval configs set episode_min_messages=1 (the library default
        # is 2, which silently DROPS singleton groups — messages lost).
        episode_min_messages=1,
        episode_merge="llm",
        merge_top_k=5,
        # None, NOT 0.85: upstream's merge_similarity_threshold=0.85
        # (config.py:86) is config-plumbed into merger._similarity_threshold
        # and NEVER read — _find_similar passes the top-5 hits to the merge
        # LLM unfiltered (merger.py:69-80; round-12 finding 1). The knob
        # itself stays available as our documented extension.
        merge_similarity=None,
        merge_time_gap_hours=1.0,
        semantic_integration="append",
        # Both eval configs run search_top_k_semantic=20, used by the
        # predict-stage retrieval (memory_system.py:170-173) — the published
        # path. The library signature default is 10 (config.py:91); v1/v4
        # keep that default (round-12 finding 5).
        semantic_top_k=20,
        consolidation="off",
    ),
}

EPISODE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "narrative": {"type": "string"},
        "timestamp": {"type": "string"},
    },
    "required": ["title", "narrative"],
}

PREDICT_SCHEMA = {
    "type": "object",
    "properties": {"prediction": {"type": "string"}},
    "required": ["prediction"],
}

CALIBRATE_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
}

# Paper §3.2.2 "Narrative Episode Generation".
# Condensed from EPISODE_GENERATION_PROMPT; temporal anchoring is mandatory,
# including upstream's parenthetical conversion style and its example.
# The "Boundary detection reason" slot carries the segmenter's per-group
# topic, exactly as upstream threads it (memory_system.py:121-123 →
# prompts.py:17-18); paths without a topic feed upstream's own filler
# "conversation" (round-12 finding 7 — the schema's topic field used to be
# a dead output here).
EPISODE_PROMPT = """You are an episodic memory generation expert. Convert this
conversation segment into one episodic memory.
1. title: a specific title for the episode (10-20 words)
2. narrative: a third-person past-tense narrative telling a coherent story:
   who took part and when, what was discussed, what decisions were made,
   what emotions were expressed, what plans or outcomes emerged. Include
   all important details; time should be precise to the hour when known.
   Time analysis: the timestamp in brackets before each message is the
   authoritative time. Convert every relative time expression in the text
   ("yesterday", "next week", "last month") into an absolute date, writing
   the converted time in parentheses right after the original expression —
   e.g. "the user expressed interest in going hiking on the upcoming
   weekend (March 16, 2024)".
3. timestamp: when the episode actually happened, analyzed from the message
   timestamps and content (ISO format; never the current time)

Segment:
{segment}

Boundary detection reason:
{boundary_reason}

Return JSON: {{"title": "...", "narrative": "...", "timestamp": "..."}}"""

# Paper §3.3.1 "Anticipatory Schema Synthesis".
# Condensed from PREDICTION_PROMPT: predict knowledge, not style.
PREDICT_PROMPT = """Given only an episode title and previously known knowledge,
predict what the episode's content says. Focus on: the core facts likely
discussed, key decisions or actions taken, knowledge exchanged, and the
logical flow of events. Predict ACTUAL CONTENT and KNOWLEDGE, not writing
style — ignore formatting, phrasing, timestamps, and tone.

Title: {title}

Known knowledge:
{knowledge}

Return JSON: {{"prediction": "..."}}"""

# The four tests shared by both semantic-extraction prompts (upstream
# EXTRACT_KNOWLEDGE_FROM_COMPARISON and SEMANTIC_GENERATION agree on these).
_FOUR_TESTS = """Each statement must pass all four tests:
- Persistence: still true 6 months from now
- Specificity: concrete and detailed, not vague
- Utility: helps predict future user needs or preferences
- Independence: understandable without the conversation context"""

# Paper §3.3.2 "Prediction Error Distillation".
# Condensed from EXTRACT_KNOWLEDGE_FROM_COMPARISON_PROMPT: seven high-value
# categories, the time/date ban, present-tense atomic style.
CALIBRATE_PROMPT = (
    """Compare the prediction against the actual conversation and
extract ONLY new or surprising knowledge the prediction missed or got wrong.
"""
    + _FOUR_TESTS
    + """
High-value categories: identity & background, preferences, technical
details (technologies, versions, methodologies), relationships, goals &
plans, beliefs & values, habits & patterns. Do NOT extract low-value
content: temporary emotional states, simple acknowledgments, vague
statements, or context-dependent remarks. Include ALL specific details
(names, versions, titles). Write each statement in present tense as one
atomic sentence. DO NOT include time/date information in the statement.
Quality over quantity.
Return an empty list if the prediction already covered everything.

Prediction:
{prediction}

Actual conversation:
{segment}

Return JSON: {{"facts": ["...", ...]}}"""
)

# Cold start (audit P1-7): with no prior semantic knowledge there is nothing
# to predict from — upstream switches to direct extraction over the generated
# episode (SEMANTIC_GENERATION_PROMPT). Its rules differ from the comparison
# path: six categories and NO time/date ban — upstream's own good examples
# carry dates ("joined Amazon in August 2020"), which matters for temporal
# questions answered from early semantic facts.
DIRECT_EXTRACT_PROMPT = (
    """Extract HIGH-VALUE, PERSISTENT knowledge from this
episode — long-term valuable knowledge, not temporary conversation details.
"""
    + _FOUR_TESTS
    + """
High-value categories: identity & professional (names, titles, companies,
education), persistent preferences (favorites, technology choices with
reasons), technical knowledge (technologies with versions, architectures,
decisions), relationships (family, colleagues, team structure), goals &
plans, patterns & habits (regular activities, workflows).
Do NOT extract low-value content: acknowledgments, confusion, temporary
emotions or reactions, or remarks about the conversation itself.
Include ALL specific details (names, versions, titles). Quality over
quantity.

Episode:
Title: {title}
Content: {narrative}

Return JSON: {{"facts": ["...", ...]}}"""
)


class NemoriOrganizer(Organizer):
    """Nemori (see module docstring for the segment/episode/semantic pipeline and
    upstream-deviation list)."""

    name = "nemori"

    produces = ("episodes", "semantic")

    def __init__(
        self,
        fidelity: str | None = None,
        segmenter: str | None = None,
        episode_merge: str | None = None,
        semantic_integration: str | None = None,
        consolidation: str | None = None,
        # Kept as named kwargs (rather than folded into **kw) for backward
        # compatibility with the pre-Task-10 positional/keyword surface.
        buffer_min: int | None = None,
        buffer_max: int | None = None,
        boundary_confidence: float | None = None,
        semantic_top_k: int | None = None,
        window: int | None = None,
        chunk_max: int | None = None,
        episode_min_messages: int | None = None,
        merge_top_k: int | None = None,
        merge_similarity: float | None = None,
        merge_time_gap_hours: float | None = None,
        dedup_threshold: float | None = None,
        integrate_top_k: int | None = None,
        integrate_tau: float | None = None,
    ) -> None:
        """`fidelity` selects a preset from `NEMORI_PRESETS` ("v1" if None/unknown,
        keeping the no-arg constructor stable); every other explicit (non-None)
        kwarg overrides that preset's value for the matching field. See the module
        docstring for what each preset combination implies (segmenter, merge
        stance, semantic-integration strategy)."""
        # Preset resolution: an explicit (non-None) kwarg always wins over
        # the preset's value for that field. fidelity=None (or unknown) ==
        # "v1" so the no-arg constructor stays config/test compatible.
        params = dict(NEMORI_PRESETS.get(fidelity, NEMORI_PRESETS["v1"]))
        overrides = dict(
            segmenter=segmenter,
            episode_merge=episode_merge,
            semantic_integration=semantic_integration,
            consolidation=consolidation,
            buffer_min=buffer_min,
            buffer_max=buffer_max,
            boundary_confidence=boundary_confidence,
            semantic_top_k=semantic_top_k,
            window=window,
            chunk_max=chunk_max,
            episode_min_messages=episode_min_messages,
            merge_top_k=merge_top_k,
            merge_similarity=merge_similarity,
            merge_time_gap_hours=merge_time_gap_hours,
            dedup_threshold=dedup_threshold,
            integrate_top_k=integrate_top_k,
            integrate_tau=integrate_tau,
        )
        params.update({k: v for k, v in overrides.items() if v is not None})

        self.buffer: list[Episode] = []
        if params["segmenter"] == "batch":
            self._segmenter = BatchPartitioner(
                window=params.get("window", 20),
                buffer_min=params.get("buffer_min", 2),
                chunk_max=params.get("chunk_max", 80),
            )
        else:
            self._segmenter = PerMessageBoundary(
                confidence=params.get("boundary_confidence", 0.7),
                buffer_min=params.get("buffer_min", 2),
                buffer_max=params.get("buffer_max", 25),
            )
        self._merger = (
            EpisodeMerger(
                top_k=params.get("merge_top_k", 5),
                similarity=params.get("merge_similarity"),
                time_gap_hours=params.get("merge_time_gap_hours"),
            )
            if params["episode_merge"] == "llm"
            else None
        )
        self._integrator = {
            "append": AppendIntegrator,
            "dedup": lambda: DedupIdReuseIntegrator(threshold=params.get("dedup_threshold", 0.85)),
            "llm3way": lambda: ThreeWayIntegrator(
                top_k=params.get("integrate_top_k", 5),
                tau=params.get("integrate_tau", 0.70),
            ),
        }[params["semantic_integration"]]()
        # Task 11: cursor-resumed deferred three-way consolidation (our
        # mixing — absent from both the paper and upstream, spec §2.3). The
        # consolidator lives in ``organizers.experimental``; import it lazily
        # so the faithful core (v1/v4/upstream) never drags the experimental
        # package into sys.modules unless this preset is selected (spec §61).
        self._consolidator = None
        if params["consolidation"] == "semantic_offline":
            from agmem.organizers.experimental.nemori_mixing import (
                SemanticOfflineConsolidator,
            )

            self._consolidator = SemanticOfflineConsolidator(
                top_k=params.get("integrate_top_k", 5),
                tau=params.get("integrate_tau", 0.70),
            )
        self.fidelity = fidelity
        self.params = params  # stats/stamping surface

        self.buffer_min = params.get("buffer_min", 2)  # repo config buffer_size_min=2
        self.buffer_max = params.get("buffer_max", 25)  # paper beta_max=25
        self.boundary_confidence = params.get("boundary_confidence", 0.7)  # paper sigma=0.7
        # Signature default 10 == upstream config.py:91; BOTH eval configs ran
        # 20 for the predict-stage retrieval (memory_system.py:170-173), which
        # the "upstream" preset carries (round-12 finding 5).
        self.semantic_top_k = params.get("semantic_top_k", 10)
        # Upstream drops segment groups smaller than episode_min_messages
        # (memory_system.py:118-119): with the LIBRARY default of 2
        # (config.py:76) singleton groups are silently skipped and those
        # messages LOST (they are still marked processed). Both eval configs
        # set 1, so the published path keeps everything — our default is 1
        # for the same reason (and our no-loss principle). Round-12 finding 3.
        self.episode_min_messages = int(params.get("episode_min_messages", 1))
        self._warned_no_llm = False

    def on_message(self, episode: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        """Returns `[]` (with a one-time warning) if `ctx.llm` is unset — messages
        bypass the buffer entirely rather than accumulating unprocessed, since
        there is no way to ever flush them without an LLM. Otherwise buffers the
        episode, lets the configured segmenter decide how much to cut, and flushes
        zero or more segments; ops from multiple segments in one call share a
        within-batch supersession guard so an earlier segment's INVALIDATE is
        respected by a later one before either is actually applied to the stores."""
        if ctx.llm is None:
            if not self._warned_no_llm:
                logger.warning(
                    "nemori: no LLM configured — boundary detection and "
                    "distillation disabled, messages bypass the buffer "
                    "(explicit degradation)"
                )
                self._warned_no_llm = True
            return []

        self.buffer.append(episode)
        segments, self.buffer = self._segmenter.push(self.buffer, ctx)
        return self._flush_segments(segments, ctx)

    def warm_start(self, corpus: list[Episode], ctx: OrganizerContext) -> list[MemoryOp]:
        """Bulk ingestion. With the batch segmenter this takes upstream's bulk
        shape: ``add_messages`` accepts the whole LIST (memory_system.py:76-83),
        so the entire corpus lands in the buffer before one ``_process``-style
        grab — the backlog can exceed ``window`` and ``chunk_max=80`` becomes
        genuinely reachable (round-12 finding 2; upstream's published runs got
        their large backlogs from async racing instead, see `BatchPartitioner`).
        The per-message boundary (v1 formalism) keeps the message-by-message
        replay — it is inherently online. Either way the tail is flushed, since
        warm start has no later message to trigger a boundary cut."""
        if not isinstance(self._segmenter, BatchPartitioner):
            ops = super().warm_start(corpus, ctx)
            ops.extend(self.flush_buffer(ctx))  # don't strand the tail segment
            return ops
        if ctx.llm is None:
            if not self._warned_no_llm:
                logger.warning(
                    "nemori: no LLM configured — boundary detection and "
                    "distillation disabled, messages bypass the buffer "
                    "(explicit degradation)"
                )
                self._warned_no_llm = True
            return []
        self.buffer.extend(corpus)
        segments, self.buffer = self._segmenter.push(self.buffer, ctx)
        ops = self._flush_segments(segments, ctx)
        # Below-gate tail -> single group, no segmentation LLM call (upstream's
        # flush()/short-batch path).
        ops.extend(self.flush_buffer(ctx))
        return ops

    def flush_buffer(self, ctx: OrganizerContext) -> list[MemoryOp]:
        """Flush whatever remains in the buffer (end-of-ingestion hook)."""
        if not self.buffer or ctx.llm is None:
            return []
        segments, self.buffer = self._segmenter.flush(self.buffer, ctx), []
        return self._flush_segments(segments, ctx)

    def consolidate(self, ctx: OrganizerContext) -> list[MemoryOp]:
        """No-op unless `consolidation="semantic_offline"` was selected at
        construction and `ctx.llm` is set — otherwise there is no deferred pass
        configured to run."""
        if self._consolidator is None or ctx.llm is None:
            return []
        return self._consolidator.run(self, ctx)

    # ---- internals ----------------------------------------------------------

    def _flush_segments(self, segments: list[Segment], ctx: OrganizerContext) -> list[MemoryOp]:
        """Common drain behind on_message / flush_buffer / warm_start.

        Call-local supersession guard (review I1): the whole batch of ops is
        applied to doc_store/vector_store only after the calling hook returns,
        so a fact/episode invalidated by an earlier segment still looks live to
        a later segment's candidate search. Threading one ``superseded`` set
        through every _flush_segment (into merger + ThreeWay exclude_ids)
        mirrors the consolidator's within-pass guard so the inline v4 path
        can't earn two merge heads for the same target within one batch.

        Groups smaller than ``episode_min_messages`` are skipped BEFORE any
        LLM call — upstream memory_system.py:118-119 (round-12 finding 3);
        with the upstream library default of 2 those messages are silently
        LOST, which is why both eval configs and our default use 1."""
        ops: list[MemoryOp] = []
        superseded: set[str] = set()
        for segment in segments:
            if len(segment.episodes) < self.episode_min_messages:
                continue
            ops.extend(self._flush_segment(segment, ctx, superseded))
        return ops

    def _flush_segment(
        self,
        segment: Segment,
        ctx: OrganizerContext,
        superseded: set[str] | None = None,
    ) -> list[MemoryOp]:
        # ``superseded`` is the call-local guard threaded from _flush_segments
        # (I1). None keeps _flush_segment independently callable (tests) — a
        # fresh set then guards only within this single segment.
        if superseded is None:
            superseded = set()
        episodes = segment.episodes
        segment_text = "\n".join(_fmt(e) for e in episodes)

        # 1. representation alignment: title + temporally-anchored narrative.
        # The segmenter's per-group topic rides in as the boundary reason
        # (upstream memory_system.py:121-123 — round-12 finding 7).
        generated = ctx.llm.call(
            "distill",
            EPISODE_PROMPT.format(segment=segment_text, boundary_reason=segment.topic),
            EPISODE_SCHEMA,
            required_keys=("title", "narrative"),
            phase="narrate",
        )
        fallback_title = " ".join(episodes[0].content.split()[:8])
        fallback_ts = episodes[0].meta.get("date") or episodes[0].timestamp.isoformat()
        if generated is None:
            logger.warning("nemori: episode generation failed — mechanical fallback episode")
            title, narrative, episode_timestamp = fallback_title, segment_text, fallback_ts
        else:
            title = str(generated.get("title", "")).strip() or fallback_title
            narrative = str(generated.get("narrative", "")).strip() or segment_text
            episode_timestamp = str(generated.get("timestamp", "")).strip() or fallback_ts

        source_ids = [e.id for e in episodes]
        episode_id = new_id()

        # 2. v4 §3.2.3 / upstream merger: is this the same event as a nearby
        # episode? Runs right after narration and before predict-calibrate
        # (v4 Alg.1 order). None (no merger, no candidates, LLM decline/fail)
        # keeps the plain ADD path so a segment is never lost to a merge
        # attempt.
        merged = (
            self._merger.merge_or_none(
                title, narrative, episode_timestamp, source_ids, ctx, exclude_ids=superseded
            )
            if self._merger
            else None
        )
        calib_episodes = episodes
        if merged is not None:
            merge_ops, episode_id, title, narrative = merged
            ops = list(merge_ops)  # ADD 대신 MERGE+INVALIDATE
            # the merged-away episode must not be re-offered as a live merge
            # candidate to a later segment in this same batch (I1)
            superseded.update(o.target_id for o in merge_ops if o.op is OpType.INVALIDATE)
            # Round-12 finding 4: after a MERGE, upstream regenerates
            # semantics from the MERGED episode, whose source_messages are
            # target + new combined (merger.py:133, memory_system.py:150-156)
            # and which feed the calibrate prompt as original_messages
            # (semantic.py:94-97) — so calibration must see BOTH episodes'
            # raw messages, target's first. The MERGE head already carries
            # the combined provenance (old's source_episode_ids + ours), and
            # the raw episodes are durable before on_message runs
            # (write-then-organize), so they hydrate from the doc store in
            # payload order. If nothing hydrates (e.g. a target seeded
            # without stored raws), fall back to the new segment alone.
            merged_source_ids = list(merge_ops[0].payload.get("source_episode_ids") or source_ids)
            hydrated = ctx.doc_store.get_episodes(merged_source_ids)
            if hydrated:
                calib_episodes = hydrated
                source_ids = merged_source_ids  # semantic provenance covers both episodes
        else:
            ops = [
                MemoryOp(
                    op=OpType.ADD,
                    target_type="episodes",
                    target_id=episode_id,
                    payload={
                        "id": episode_id,
                        "title": title,
                        "content": narrative,
                        "timestamp": episode_timestamp,
                        "source_episode_ids": source_ids,
                        "embedding_text": f"{title}\n{narrative}",
                    },
                )
            ]
        # upstream original_messages format for calibration: role: text,
        # no timestamps (time/date is banned from semantic statements anyway).
        # ``calib_episodes`` is the new segment, or after a merge the merged
        # episode's combined raw material (round-12 finding 4).
        plain_text = "\n".join(f"{e.role}: {e.content}" for e in calib_episodes)
        ops.extend(
            self._predict_calibrate(
                title, narrative, plain_text, episode_id, source_ids, ctx, superseded
            )
        )
        return ops

    def _predict_calibrate(
        self,
        title: str,
        narrative: str,
        plain_text: str,
        episode_id: str,
        source_ids: list[str],
        ctx: OrganizerContext,
        superseded: set[str] | None = None,
    ) -> list[MemoryOp]:
        if superseded is None:
            superseded = set()
        # Stage 1: predict the episode from title + retrieved knowledge only.
        query_embedding = ctx.embedder.embed([f"{title}\n{narrative}"])[0]
        hits = ctx.vector_store.search(
            query_embedding, k=self.semantic_top_k, memory_type="semantic", namespace=ctx.namespace
        )
        known = ctx.doc_store.get_items([h[0] for h in hits], "semantic")
        if not known:
            # cold start: nothing to predict from -> direct extraction
            # (upstream reads the generated episode, not the raw segment)
            calibration = ctx.llm.call(
                "distill",
                DIRECT_EXTRACT_PROMPT.format(title=title, narrative=narrative),
                CALIBRATE_SCHEMA,
                required_keys=("facts",),
                phase="predict_calibrate",
            )
        else:
            knowledge = "\n".join(f"- {k.get('content', '')}" for k in known)
            prediction = ctx.llm.call(
                "distill",
                PREDICT_PROMPT.format(title=title, knowledge=knowledge),
                PREDICT_SCHEMA,
                required_keys=("prediction",),
                phase="predict_calibrate",
            )
            if prediction is None:
                return []  # episode is stored; only distillation is skipped

            # Stage 2: calibrate against the RAW segment — extract the gap only.
            calibration = ctx.llm.call(
                "distill",
                CALIBRATE_PROMPT.format(
                    prediction=str(prediction.get("prediction", "")), segment=plain_text
                ),
                CALIBRATE_SCHEMA,
                required_keys=("facts",),
                phase="predict_calibrate",
            )
        if calibration is None:
            return []

        # Stage 3: integrate the prediction gap as atomic semantic facts
        # (v4 §3.3.3 P_con — delegated to self._integrator, default Append).
        ops: list[MemoryOp] = []
        for fact in calibration.get("facts", []):
            fact = str(fact).strip()
            if not fact:
                continue
            # exclude_ids drops anything already superseded earlier in this
            # batch from ThreeWay's candidate search, and each fact's own
            # INVALIDATEs feed back in so a later mutually-similar fact can't
            # re-merge an already-absorbed target (I1). Append/Dedup accept and
            # ignore the kwarg.
            out = self._integrator.integrate(
                fact, episode_id, source_ids, ctx, exclude_ids=superseded
            )
            superseded.update(o.target_id for o in out if o.op is OpType.INVALIDATE)
            ops.extend(out)
        return ops
