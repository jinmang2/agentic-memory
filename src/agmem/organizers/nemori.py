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

Deviations from the reference system (github.com/nemori-ai/nemori):
- boundary detection is per-message (the paper v1 formalism, f_theta over
  the buffer) rather than the batched BATCH_SEGMENTATION_PROMPT the
  rewritten repo uses for backfill throughput
- no episode merging (paper-v4 §3.2.3 module, ON by default in the repo's
  eval; deferred to the LongMemEval stage — LoCoMo's multi-day session
  gaps mean upstream's >1h-gap merge ban blocks most merges anyway)
- no v4 semantic new/merge/conflict integration (§3.3.3): semantic store
  is append-only, as in the repo's pre-v4 main path
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
from agmem.organizers.nemori_stages import (
    BOUNDARY_PROMPT,
    BOUNDARY_SCHEMA,
    PerMessageBoundary,
    _fmt,
)

logger = logging.getLogger("agmem.organizers.nemori")

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

# Condensed from EPISODE_GENERATION_PROMPT; temporal anchoring is mandatory,
# including upstream's parenthetical conversion style and its example.
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

Return JSON: {{"title": "...", "narrative": "...", "timestamp": "..."}}"""

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
    name = "nemori"

    def __init__(
        self,
        buffer_min: int = 2,
        buffer_max: int = 25,
        boundary_confidence: float = 0.7,
        semantic_top_k: int = 10,
    ) -> None:
        self.buffer_min = buffer_min  # repo config buffer_size_min=2 (not in paper v1)
        self.buffer_max = buffer_max  # paper beta_max=25
        self.boundary_confidence = boundary_confidence  # paper sigma_boundary=0.7
        self.semantic_top_k = semantic_top_k  # repo config search_top_k_semantic=10
        self.buffer: list[Episode] = []
        self._segmenter = PerMessageBoundary(boundary_confidence, buffer_min, buffer_max)
        self._warned_no_llm = False

    def on_message(self, ep: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        if ctx.llm is None:
            if not self._warned_no_llm:
                logger.warning(
                    "nemori: no LLM configured — boundary detection and "
                    "distillation disabled, messages bypass the buffer "
                    "(explicit degradation)"
                )
                self._warned_no_llm = True
            return []

        self.buffer.append(ep)
        segments, self.buffer = self._segmenter.push(self.buffer, ctx)
        ops: list[MemoryOp] = []
        for seg in segments:
            ops.extend(self._flush_segment(seg, ctx))
        return ops

    def warm_start(self, corpus: list[Episode], ctx: OrganizerContext) -> list[MemoryOp]:
        ops = super().warm_start(corpus, ctx)
        ops.extend(self.flush_buffer(ctx))  # don't strand the tail segment
        return ops

    def flush_buffer(self, ctx: OrganizerContext) -> list[MemoryOp]:
        """Flush whatever remains in the buffer (end-of-ingestion hook)."""
        if not self.buffer or ctx.llm is None:
            return []
        segments, self.buffer = self._segmenter.flush(self.buffer, ctx), []
        ops: list[MemoryOp] = []
        for seg in segments:
            ops.extend(self._flush_segment(seg, ctx))
        return ops

    # ---- internals ----------------------------------------------------------

    def _flush_segment(self, segment: list[Episode], ctx: OrganizerContext) -> list[MemoryOp]:
        seg_text = "\n".join(_fmt(e) for e in segment)

        # 1. representation alignment: title + temporally-anchored narrative
        gen = ctx.llm.call(
            "distill",
            EPISODE_PROMPT.format(segment=seg_text),
            EPISODE_SCHEMA,
            required_keys=("title", "narrative"),
        )
        fallback_title = " ".join(segment[0].content.split()[:8])
        fallback_ts = segment[0].meta.get("date") or segment[0].timestamp.isoformat()
        if gen is None:
            logger.warning("nemori: episode generation failed — mechanical fallback episode")
            title, narrative, ep_ts = fallback_title, seg_text, fallback_ts
        else:
            title = str(gen.get("title", "")).strip() or fallback_title
            narrative = str(gen.get("narrative", "")).strip() or seg_text
            ep_ts = str(gen.get("timestamp", "")).strip() or fallback_ts

        episode_id = new_id()
        source_ids = [e.id for e in segment]
        ops = [
            MemoryOp(
                op=OpType.ADD,
                target_type="episodes",
                target_id=episode_id,
                payload={
                    "id": episode_id,
                    "title": title,
                    "content": narrative,
                    "timestamp": ep_ts,
                    "source_episode_ids": source_ids,
                    "embedding_text": f"{title}\n{narrative}",
                },
            )
        ]
        # upstream original_messages format for calibration: role: text,
        # no timestamps (time/date is banned from semantic statements anyway)
        plain_text = "\n".join(f"{e.role}: {e.content}" for e in segment)
        ops.extend(
            self._predict_calibrate(title, narrative, plain_text, episode_id, source_ids, ctx)
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
    ) -> list[MemoryOp]:
        # Stage 1: predict the episode from title + retrieved knowledge only.
        emb = ctx.embedder.embed([f"{title}\n{narrative}"])[0]
        hits = ctx.vec.search(
            emb, k=self.semantic_top_k, memory_type="semantic", namespace=ctx.namespace
        )
        known = ctx.doc.get_items([h[0] for h in hits], "semantic")
        if not known:
            # cold start: nothing to predict from -> direct extraction
            # (upstream reads the generated episode, not the raw segment)
            cal = ctx.llm.call(
                "distill",
                DIRECT_EXTRACT_PROMPT.format(title=title, narrative=narrative),
                CALIBRATE_SCHEMA,
                required_keys=("facts",),
            )
        else:
            knowledge = "\n".join(f"- {k.get('content', '')}" for k in known)
            pred = ctx.llm.call(
                "distill",
                PREDICT_PROMPT.format(title=title, knowledge=knowledge),
                PREDICT_SCHEMA,
                required_keys=("prediction",),
            )
            if pred is None:
                return []  # episode is stored; only distillation is skipped

            # Stage 2: calibrate against the RAW segment — extract the gap only.
            cal = ctx.llm.call(
                "distill",
                CALIBRATE_PROMPT.format(
                    prediction=str(pred.get("prediction", "")), segment=plain_text
                ),
                CALIBRATE_SCHEMA,
                required_keys=("facts",),
            )
        if cal is None:
            return []

        # Stage 3: integrate the prediction gap as atomic semantic facts.
        ops: list[MemoryOp] = []
        for fact in cal.get("facts", []):
            fact = str(fact).strip()
            if not fact:
                continue
            fid = new_id()
            ops.append(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="semantic",
                    target_id=fid,
                    payload={
                        "id": fid,
                        "content": fact,
                        "episode_id": episode_id,
                        "source_episode_ids": source_ids,
                        "embedding_text": fact,
                    },
                )
            )
        return ops
