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
- no episode merging (a repo extra absent from the paper's formalism)
- storage is our MemoryOp pipeline instead of PostgreSQL + Qdrant
- if episode generation fails we emit a mechanical episode instead of
  losing the segment (title = first words, narrative = raw messages)
"""

from __future__ import annotations

import logging

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, new_id
from agmem.organizers.base import Organizer, OrganizerContext

logger = logging.getLogger("agmem.organizers.nemori")

BOUNDARY_SCHEMA = {
    "type": "object",
    "properties": {
        "boundary": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["boundary", "confidence"],
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

# Condensed from Nemori's segmentation criteria (topic shift, intent shift,
# temporal markers, content relatedness; "when in doubt, split").
BOUNDARY_PROMPT = """Decide whether the NEWEST message starts a new episode
(a different topic, intent, or time context) relative to the buffered
conversation. Signals: topic change, intent shift (e.g. information request
to decision), temporal markers ("by the way", or a gap of 30+ minutes
between message timestamps), content relatedness below ~30%.
Episodes work best with 2-15 messages. When in doubt, split.

Buffered conversation:
{buffer}

Newest message:
{message}

Return JSON: {{"boundary": true/false, "confidence": 0.0-1.0}}"""

# Condensed from EPISODE_GENERATION_PROMPT; temporal anchoring is mandatory.
EPISODE_PROMPT = """Convert this conversation segment into one episodic memory.
1. title: a short, specific title for the episode
2. narrative: a third-person past-tense narrative of what happened.
   IMPORTANT: convert every relative time expression ("yesterday",
   "next week", "last month") into an absolute date/time using the
   timestamp shown in brackets before each message.

3. timestamp: when the episode happened (copy the first message's timestamp)

Segment:
{segment}

Return JSON: {{"title": "...", "narrative": "...", "timestamp": "..."}}"""

# Condensed from PREDICTION_PROMPT: predict knowledge, not style.
PREDICT_PROMPT = """Given only an episode title and previously known knowledge,
predict what the episode's content says. Predict actual facts and knowledge,
not writing style.

Title: {title}

Known knowledge:
{knowledge}

Return JSON: {{"prediction": "..."}}"""

# Condensed from EXTRACT_KNOWLEDGE_FROM_COMPARISON_PROMPT (the four tests).
CALIBRATE_PROMPT = """Compare the prediction against the raw conversation and
extract ONLY new or surprising knowledge the prediction missed or got wrong.
Each statement must pass all four tests:
- Persistence: still true well after the conversation
- Specificity: concrete, not vague
- Utility: useful for future interactions
- Independence: self-contained and atomic
Write each statement in present tense with no relative time expressions.
Return an empty list if the prediction already covered everything.

Prediction:
{prediction}

Raw conversation:
{segment}

Return JSON: {{"facts": ["...", ...]}}"""

# Cold start (audit P1-7): with no prior semantic knowledge there is nothing
# to predict from — distill directly, under the same four tests.
DIRECT_EXTRACT_PROMPT = """Extract knowledge worth remembering from this conversation.
Each statement must pass all four tests:
- Persistence: still true well after the conversation
- Specificity: concrete, not vague
- Utility: useful for future interactions
- Independence: self-contained and atomic
Write each statement in present tense with no relative time expressions.

Raw conversation:
{segment}

Return JSON: {{"facts": ["...", ...]}}"""


def _fmt(ep: Episode) -> str:
    ts = ep.meta.get("date") or ep.timestamp.isoformat()
    return f"[{ts}] {ep.role}: {ep.content}"


class NemoriOrganizer(Organizer):
    name = "nemori"

    def __init__(self, buffer_min: int = 2, buffer_max: int = 25,
                 boundary_confidence: float = 0.7, semantic_top_k: int = 10) -> None:
        self.buffer_min = buffer_min          # paper buffer_size_min=2
        self.buffer_max = buffer_max          # paper beta_max=25
        self.boundary_confidence = boundary_confidence  # paper sigma_boundary=0.7
        self.semantic_top_k = semantic_top_k  # paper search_top_k_semantic=10
        self.buffer: list[Episode] = []
        self._warned_no_llm = False

    def on_message(self, ep: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        if ctx.llm is None:
            if not self._warned_no_llm:
                logger.warning("nemori: no LLM configured — boundary detection and "
                               "distillation disabled, messages bypass the buffer "
                               "(explicit degradation)")
                self._warned_no_llm = True
            return []

        self.buffer.append(ep)
        if len(self.buffer) < self.buffer_min:
            return []
        if len(self.buffer) >= self.buffer_max:
            segment, self.buffer = self.buffer, []
            return self._flush_segment(segment, ctx)

        verdict = ctx.llm.call(
            "extract",
            BOUNDARY_PROMPT.format(
                buffer="\n".join(_fmt(e) for e in self.buffer[:-1]),
                message=_fmt(self.buffer[-1]),
            ),
            BOUNDARY_SCHEMA, required_keys=("boundary", "confidence"),
        )
        if verdict is None:
            return []  # drop counted upstream; treat as no boundary
        if verdict.get("boundary") and float(verdict.get("confidence", 0.0)) >= self.boundary_confidence:
            # The newest message opened the next topic: it stays buffered.
            segment, self.buffer = self.buffer[:-1], [self.buffer[-1]]
            return self._flush_segment(segment, ctx)
        return []

    def warm_start(self, corpus: list[Episode], ctx: OrganizerContext) -> list[MemoryOp]:
        ops = super().warm_start(corpus, ctx)
        ops.extend(self.flush_buffer(ctx))  # don't strand the tail segment
        return ops

    def flush_buffer(self, ctx: OrganizerContext) -> list[MemoryOp]:
        """Flush whatever remains in the buffer (end-of-ingestion hook)."""
        if not self.buffer or ctx.llm is None:
            return []
        segment, self.buffer = self.buffer, []
        return self._flush_segment(segment, ctx)

    # ---- internals ----------------------------------------------------------

    def _flush_segment(self, segment: list[Episode], ctx: OrganizerContext) -> list[MemoryOp]:
        seg_text = "\n".join(_fmt(e) for e in segment)

        # 1. representation alignment: title + temporally-anchored narrative
        gen = ctx.llm.call("distill", EPISODE_PROMPT.format(segment=seg_text),
                           EPISODE_SCHEMA, required_keys=("title", "narrative"))
        fallback_title = " ".join(segment[0].content.split()[:8])
        fallback_ts = (segment[0].meta.get("date")
                       or segment[0].timestamp.isoformat())
        if gen is None:
            logger.warning("nemori: episode generation failed — mechanical fallback episode")
            title, narrative, ep_ts = fallback_title, seg_text, fallback_ts
        else:
            title = str(gen.get("title", "")).strip() or fallback_title
            narrative = str(gen.get("narrative", "")).strip() or seg_text
            ep_ts = str(gen.get("timestamp", "")).strip() or fallback_ts

        episode_id = new_id()
        source_ids = [e.id for e in segment]
        ops = [MemoryOp(
            op=OpType.ADD, target_type="episodes", target_id=episode_id,
            payload={"id": episode_id, "title": title, "content": narrative,
                     "timestamp": ep_ts, "source_episode_ids": source_ids,
                     "embedding_text": f"{title}\n{narrative}"},
        )]
        ops.extend(self._predict_calibrate(title, narrative, seg_text,
                                           episode_id, source_ids, ctx))
        return ops

    def _predict_calibrate(self, title: str, narrative: str, seg_text: str,
                           episode_id: str, source_ids: list[str],
                           ctx: OrganizerContext) -> list[MemoryOp]:
        # Stage 1: predict the episode from title + retrieved knowledge only.
        emb = ctx.embedder.embed([f"{title}\n{narrative}"])[0]
        hits = ctx.vec.search(emb, k=self.semantic_top_k,
                              memory_type="semantic", namespace=ctx.namespace)
        known = ctx.doc.get_items([h[0] for h in hits], "semantic")
        if not known:
            # cold start: nothing to predict from -> direct extraction
            cal = ctx.llm.call("distill",
                               DIRECT_EXTRACT_PROMPT.format(segment=seg_text),
                               CALIBRATE_SCHEMA, required_keys=("facts",))
        else:
            knowledge = "\n".join(f"- {k.get('content', '')}" for k in known)
            pred = ctx.llm.call("distill",
                                PREDICT_PROMPT.format(title=title, knowledge=knowledge),
                                PREDICT_SCHEMA, required_keys=("prediction",))
            if pred is None:
                return []  # episode is stored; only distillation is skipped

            # Stage 2: calibrate against the RAW segment — extract the gap only.
            cal = ctx.llm.call(
                "distill",
                CALIBRATE_PROMPT.format(prediction=str(pred.get("prediction", "")),
                                        segment=seg_text),
                CALIBRATE_SCHEMA, required_keys=("facts",),
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
            ops.append(MemoryOp(
                op=OpType.ADD, target_type="semantic", target_id=fid,
                payload={"id": fid, "content": fact, "episode_id": episode_id,
                         "source_episode_ids": source_ids, "embedding_text": fact},
            ))
        return ops
