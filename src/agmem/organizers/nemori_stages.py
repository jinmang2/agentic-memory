"""Assembled Nemori pipeline stages (spec §2.3).

``PerMessageBoundary`` is the first stage extracted out of ``NemoriOrganizer``:
the online, per-message boundary detector (the paper's v1 f_theta formalism)
that decides whether the newest buffered message opens a new episode. It is
v1-equivalent with the logic previously inlined in ``NemoriOrganizer.on_message``
— same prompt, same schema, same thresholds, same buffer-splitting semantics.
Later tasks will add the remaining stages (``BatchPartitioner``,
``EpisodeMerger``, and the semantic integrators) alongside this one so the
organizer becomes a thin composition of independently testable stages.
"""

from __future__ import annotations

from agmem.core.types import Episode
from agmem.organizers.base import OrganizerContext

BOUNDARY_SCHEMA = {
    "type": "object",
    "properties": {
        "boundary": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["boundary", "confidence"],
}

# Condensed from Nemori's segmentation criteria (BATCH_SEGMENTATION_PROMPT),
# recast for our online per-message mode (the paper v1 formalism).
BOUNDARY_PROMPT = """Decide whether the NEWEST message starts a new episode
(a different topic, intent, or time context) relative to the buffered
conversation. Be strict — high sensitivity to shifts. Signals:
- topic change (a different subject or activity)
- intent transition (e.g. information request to decision, discussion to
  casual chat)
- temporal markers ("earlier", "before", "by the way", "oh right", "also",
  or a gap of 30+ minutes between message timestamps)
- structural signals (explicit transition phrases like "changing topics",
  "speaking of which", "quick question", or a concluding statement
  indicating the current topic is finished)
- content relatedness below ~30%
Episodes work best with 2-15 messages. When in doubt, split.

Buffered conversation:
{buffer}

Newest message:
{message}

Return JSON: {{"boundary": true/false, "confidence": 0.0-1.0}}"""


def _fmt(ep: Episode) -> str:
    ts = ep.meta.get("date") or ep.timestamp.isoformat()
    return f"[{ts}] {ep.role}: {ep.content}"


class PerMessageBoundary:
    """Online per-message boundary detector (paper v1 f_theta formalism).

    ``push`` decides, for the current buffer, whether to cut one or more
    segments off the front and what remains buffered. The LLM is not injected
    at construction time — it comes from the shared ``ctx`` passed to each
    call, consistent with the rest of the pipeline's stages.
    """

    def __init__(
        self,
        confidence: float = 0.7,
        buffer_min: int = 2,
        buffer_max: int = 25,
    ) -> None:
        self.confidence = confidence  # paper sigma_boundary=0.7
        self.buffer_min = buffer_min  # repo config buffer_size_min=2 (not in paper v1)
        self.buffer_max = buffer_max  # paper beta_max=25

    def push(
        self, buffer: list[Episode], ctx: OrganizerContext
    ) -> tuple[list[list[Episode]], list[Episode]]:
        """Return (segments to flush, remaining buffer) for the given buffer."""
        if len(buffer) < self.buffer_min:
            return [], buffer
        if len(buffer) >= self.buffer_max:
            return [buffer], []

        verdict = ctx.llm.call(
            "extract",
            BOUNDARY_PROMPT.format(
                buffer="\n".join(_fmt(e) for e in buffer[:-1]),
                message=_fmt(buffer[-1]),
            ),
            BOUNDARY_SCHEMA,
            required_keys=("boundary", "confidence"),
        )
        if verdict is None:
            return [], buffer  # drop counted upstream; treat as no boundary
        if verdict.get("boundary") and float(verdict.get("confidence", 0.0)) >= self.confidence:
            # The newest message opened the next topic: it stays buffered.
            return [buffer[:-1]], [buffer[-1]]
        return [], buffer

    def flush(self, buffer: list[Episode], ctx: OrganizerContext) -> list[list[Episode]]:
        """Flush whatever remains in the buffer as the final segment(s)."""
        return [buffer] if buffer else []


BATCH_SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "episodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "indices": {"type": "array", "items": {"type": "integer"}},
                    "topic": {"type": "string"},
                },
                "required": ["indices"],
            },
        }
    },
    "required": ["episodes"],
}

# Condensed from upstream BATCH_SEGMENTATION_PROMPT (v4 Local Message
# Partitioning P_par): topic independence, 2-15 messages, when in doubt
# split, relevance < ~30% cut; indexed batch in, index groups out.
BATCH_SEGMENT_PROMPT = """Partition this conversation into topically coherent
episodes. Each episode centers on ONE core topic or event. Signals for a cut:
explicit topic shifts, intent transitions, temporal markers ("earlier",
"by the way", 30+ minute gaps), structural signals (transition phrases,
concluding statements), content relatedness below ~30%. Episodes work best
with 2-15 messages; when in doubt, split. Every message index must appear in
exactly one episode, in order.

Messages (indexed):
{messages}

Return JSON: {{"episodes": [{{"indices": [0, 1, ...], "topic": "..."}}]}}"""


class BatchPartitioner:
    """v4 §3.2.1 Local Message Partitioning / upstream batch segmentation.

    Buffers until ``window`` then LLM-partitions the whole buffer. flush()
    on a tail shorter than the window stores it as one segment without an
    LLM call — upstream's single-'conversation'-group path below
    batch_threshold."""

    def __init__(self, window: int = 20, buffer_min: int = 2, chunk_max: int = 80) -> None:
        self.window = window
        self.buffer_min = buffer_min
        self.chunk_max = chunk_max  # upstream: >80msg batches are chunked

    def push(
        self, buffer: list[Episode], ctx: OrganizerContext
    ) -> tuple[list[list[Episode]], list[Episode]]:
        if len(buffer) < self.window:
            return [], buffer
        return self._partition(buffer, ctx), []

    def flush(self, buffer: list[Episode], ctx: OrganizerContext) -> list[list[Episode]]:
        if not buffer:
            return []
        if len(buffer) < self.window:
            return [buffer]
        return self._partition(buffer, ctx)

    def _partition(
        self, buffer: list[Episode], ctx: OrganizerContext
    ) -> list[list[Episode]]:
        segments: list[list[Episode]] = []
        for start in range(0, len(buffer), self.chunk_max):
            chunk = buffer[start : start + self.chunk_max]
            indexed = "\n".join(f"[{i}] {_fmt(e)}" for i, e in enumerate(chunk))
            result = ctx.llm.call(
                "extract",
                BATCH_SEGMENT_PROMPT.format(messages=indexed),
                BATCH_SEGMENT_SCHEMA,
                required_keys=("episodes",),
            )
            groups = (result or {}).get("episodes") or []
            covered: set[int] = set()
            for g in groups:
                idxs = sorted(
                    i
                    for i in g.get("indices", [])
                    if isinstance(i, int) and 0 <= i < len(chunk) and i not in covered
                )
                if not idxs:
                    continue
                covered.update(idxs)
                segments.append([chunk[i] for i in idxs])
            leftover = [chunk[i] for i in range(len(chunk)) if i not in covered]
            if leftover:  # LLM 실패/누락 인덱스 — 세그먼트를 잃지 않는다 (프로젝트 원칙)
                segments.append(leftover)
        return segments
