"""MemoryOS organizer (arXiv:2506.06326, EMNLP'25 Oral) — compact port.

OS-style hierarchy: STM (fixed-size dialogue buffer) -> MTM (topic
segments with heat) -> LPM (profile facts). Heat = n_visit + length +
recency decay; segments crossing the threshold trigger profile/knowledge
extraction and reset (paper §MTM; constants match the pypi/chromadb core:
capacity 10, θ=0.6, τ=5, RECENCY_TAU_HOURS=24 — the eval core that
produced the paper's LoCoMo numbers differs: heat 0.8/0.8/1e-4, Dice
keyword term, STM cap=1; see docs/research/round5/memoryos).

Deviations: single vector index instead of FAISS-per-tier; eviction is
lowest-heat (paper-faithful) rather than the code's access-count LFU —
we have no read-path visit feedback, so N_visit stays 0 (round-5 N1);
STM flushes as a 10-message batch instead of upstream's 1-page FIFO
rolling window (round-5 N2); eviction emits DELETE ops through the
evolution log (auditable; upstream's old silent-loss issue #65 has since
been fixed upstream too).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, new_id
from agmem.organizers.base import Organizer, OrganizerContext

logger = logging.getLogger("agmem.organizers.memoryos")

TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "summary": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "message_indexes": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["topic", "summary"],
            },
        }
    },
    "required": ["groups"],
}

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {"profile_facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["profile_facts"],
}

TOPIC_PROMPT = """Split this dialogue batch into topical groups and summarize each.

Messages (indexed):
{messages}

Return JSON: {{"groups": [{{"topic": "short label", "summary": "2-3 sentence summary \
covering the concrete facts", "keywords": ["theme keyword", ...], \
"message_indexes": [0, 1, ...]}}]}}"""

PROFILE_PROMPT = """Extract durable user-profile facts and knowledge from this hot memory segment
(things worth remembering long-term about the user or the world; skip pleasantries).

Segment:
{content}

Return JSON: {{"profile_facts": ["self-contained fact", ...]}}"""


class MemoryOSOrganizer(Organizer):
    """MemoryOS STM/MTM/LPM organizer (arXiv:2506.06326; see module
    docstring for the paper-vs-eval-core constant discrepancy and other
    upstream deviations). Writes only via returned MemoryOps; STM/heat
    state lives in this instance, not the stores."""

    name = "memoryos"

    def __init__(
        self,
        stm_capacity: int = 10,
        mtm_capacity: int = 2000,
        heat_threshold: float = 5.0,
        similarity_threshold: float = 0.6,
        recency_tau_hours: float = 24.0,
    ) -> None:
        """``stm_capacity``: STM batch size that triggers a flush to MTM.
        ``mtm_capacity``: max MTM segments before lowest-heat eviction.
        ``heat_threshold``: heat (τ) at which a segment promotes to LPM.
        ``similarity_threshold``: F_score (θ) merge threshold — cosine +
        keyword Jaccard, paper eq.(3). ``recency_tau_hours``: heat's
        recency-decay time constant."""
        # mtm_capacity 2000 = upstream default (round-5 N6 fixed 200->2000)
        self.stm_capacity = stm_capacity
        self.mtm_capacity = mtm_capacity
        self.heat_threshold = heat_threshold
        self.similarity_threshold = similarity_threshold
        self.recency_tau_hours = recency_tau_hours
        self._stm: list[Episode] = []
        self._heat: dict[str, dict] = {}  # segment_id -> {n_visit, length, last_access}
        # Reverse indexes track which STM units back which MTM pages. Heat
        # eviction uses them to drop dead index entries; ``retire``/``patch_unit``
        # use them when this organizer is driven by ChainedConsumer over another
        # organizer's episodes (experimental) so a supersedes chain can retire
        # units and invalidate fully-absorbed pages. In-memory only — volatile
        # across restarts, same as ``_heat``.
        self._page_sources: dict[str, set[str]] = {}  # page_id -> {unit_id, ...}
        self._unit_pages: dict[str, set[str]] = {}  # unit_id -> {page_id, ...}

    # -- heat ------------------------------------------------------------------

    def _segment_heat(self, segment_id: str) -> float:
        h = self._heat.get(segment_id)
        if not h:
            return 0.0
        hours = (datetime.now(timezone.utc) - h["last_access"]).total_seconds() / 3600
        return h["n_visit"] + h["length"] + math.exp(-hours / self.recency_tau_hours)

    # -- hooks -------------------------------------------------------------------

    def on_retrieval(
        self, hits: list[tuple[str, str, float]], ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Bumps ``n_visit``/``last_access`` on served page hits (paper's
        heat-feedback loop, §3.4); always returns [] — heat lives in
        ``self._heat``, no store writes."""
        # upstream mid_term.py updates N_visit/last_visit_time on every
        # retrieval hit (paper §3.4) — the heat feedback loop round-5 N1
        # found missing. No ops needed: heat lives in organizer state.
        now = datetime.now(timezone.utc)
        for item_id, memory_type, _score in hits:
            if memory_type == "pages" and item_id in self._heat:
                h = self._heat[item_id]
                h["n_visit"] += 1
                h["last_access"] = now
        return []

    def on_message(self, episode: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        """Appends to the STM buffer; once it reaches ``stm_capacity``, flushes
        the whole batch to MTM via ``_evict_to_mtm`` and returns those ops
        (empty list otherwise).

        Chained use (driving this from another organizer's episodes) is an
        experimental composition handled by
        ``organizers.experimental.ChainedConsumer``, which calls this same
        entry point plus ``retire``/``patch_unit`` — keeping this organizer
        messages-only and paper-faithful."""
        self._stm.append(episode)
        if len(self._stm) < self.stm_capacity:
            return []
        batch, self._stm = self._stm, []
        return self._evict_to_mtm(batch, ctx)

    def patch_unit(self, unit: Episode) -> None:
        """In-place UPDATE of a unit still buffered in STM (used by
        ``ChainedConsumer`` when an upstream episode is revised before it has
        been paged). Once the unit has been paged, the update is ignored —
        documented staleness (spec §3)."""
        if any(e.id == unit.id for e in self._stm):
            self._stm = [unit if e.id == unit.id else e for e in self._stm]

    def _drop_page_index(self, page_id: str) -> None:
        """Remove page_id from the _page_sources/_unit_pages reverse
        indexes entirely — shared by heat-eviction (page gone, all its
        source links are dead) and retire (page invalidated, same
        thing). Without this, evicted/invalidated pages leak index
        entries and a later supersedes can make retire re-emit a stale
        INVALIDATE for a page that's already gone (review finding)."""
        for unit_id in self._page_sources.pop(page_id, ()):
            pages = self._unit_pages.get(unit_id)
            if pages is None:
                continue
            pages.discard(page_id)
            if not pages:
                self._unit_pages.pop(unit_id, None)

    def retire(self, superseded: set[str]) -> list[MemoryOp]:
        """Clean up derived state for absorbed units: drop them from STM;
        invalidate a page only once ALL of its sources are superseded
        (partial absorption leaves the page intact, spec §3). Called by
        ``ChainedConsumer`` when an upstream supersedes chain retires the
        episodes this organizer paged (experimental composition)."""
        ops: list[MemoryOp] = []
        self._stm = [e for e in self._stm if e.id not in superseded]
        for unit_id in superseded:
            for page_id in self._unit_pages.pop(unit_id, set()):
                source_ids = self._page_sources.get(page_id)
                if source_ids is None:
                    continue
                source_ids.discard(unit_id)
                if not source_ids:
                    self._drop_page_index(page_id)
                    self._heat.pop(page_id, None)
                    ops.append(
                        MemoryOp(
                            op=OpType.INVALIDATE,
                            target_type="pages",
                            target_id=page_id,
                            payload={"reason": "sources_superseded"},
                        )
                    )
        return ops

    def flush_buffer(self, ctx: OrganizerContext) -> list[MemoryOp]:
        """Forces any partial STM batch to MTM regardless of
        ``stm_capacity``; no-op (returns []) when STM is empty."""
        if not self._stm:
            return []
        batch, self._stm = self._stm, []
        return self._evict_to_mtm(batch, ctx)

    def warm_start(self, corpus, ctx: OrganizerContext) -> list[MemoryOp]:
        """Replays ``corpus`` through ``on_message`` (base behavior), then
        flushes any leftover partial STM batch so no episode is left
        un-paged after warm start."""
        ops = super().warm_start(corpus, ctx)
        ops.extend(self.flush_buffer(ctx))
        return ops

    # -- STM -> MTM ---------------------------------------------------------------

    def _evict_to_mtm(self, batch: list[Episode], ctx: OrganizerContext) -> list[MemoryOp]:
        if ctx.llm is None:
            logger.warning("memoryos: no LLM — storing mechanical segment (explicit degradation)")
            segment_id = new_id()
            content = "\n".join(e.content for e in batch)
            self._heat[segment_id] = {
                "n_visit": 0,
                "length": len(batch),
                "last_access": datetime.now(timezone.utc),
            }
            return [self._segment_add(segment_id, "batch", content, batch, ctx)]

        indexed = "\n".join(f"[{i}] {e.content}" for i, e in enumerate(batch))
        result = ctx.llm.call(
            "distill",
            TOPIC_PROMPT.format(messages=indexed),
            TOPIC_SCHEMA,
            required_keys=("groups",),
        )
        groups = (result or {}).get("groups") or [
            {
                "topic": "batch",
                "summary": "\n".join(e.content for e in batch),
                "message_indexes": list(range(len(batch))),
            }
        ]

        ops: list[MemoryOp] = []
        for g in groups:
            indexes = [
                i
                for i in g.get("message_indexes", [])
                if isinstance(i, int) and 0 <= i < len(batch)
            ] or list(range(len(batch)))
            members = [batch[i] for i in indexes]
            summary = str(g.get("summary", ""))
            keywords = [str(k).lower() for k in g.get("keywords") or []]
            embedding = ctx.embedder.embed([summary])[0]
            # F_score = cos + Jaccard(keywords), threshold 0.6 — paper eq.(3);
            # round-5 P0 restored the Jaccard term (cosine-only was stricter
            # and fragmented segments). Consider top-3 candidates.
            hits = ctx.vector_store.search(
                embedding, k=3, memory_type="pages", namespace=ctx.namespace
            )
            best_id, best_f = None, 0.0
            for hit_id, cos in hits:
                if hit_id not in self._heat:
                    continue
                candidate = ctx.doc_store.get_items([hit_id], "pages")
                candidate_keywords = set((candidate[0] if candidate else {}).get("keywords", []))
                union = candidate_keywords | set(keywords)
                jac = (len(candidate_keywords & set(keywords)) / len(union)) if union else 0.0
                f = cos + jac
                if f > best_f:
                    best_id, best_f = hit_id, f

            if best_id is not None and best_f >= self.similarity_threshold:
                segment_id = best_id  # merge into existing segment (F_score >= θ)
                existing = ctx.doc_store.get_items([segment_id], "pages")
                old = existing[0] if existing else {}
                content = (old.get("content", "") + "\n" + summary).strip()
                merged_kw = sorted(set(old.get("keywords", [])) | set(keywords))
                h = self._heat[segment_id]
                h["length"] += len(members)
                h["last_access"] = datetime.now(timezone.utc)
                for e in members:
                    self._unit_pages.setdefault(e.id, set()).add(segment_id)
                self._page_sources.setdefault(segment_id, set()).update(e.id for e in members)
                ops.append(
                    MemoryOp(
                        op=OpType.UPDATE,
                        target_type="pages",
                        target_id=segment_id,
                        payload={
                            "content": content,
                            "keywords": merged_kw,
                            "source_episode_ids": list(old.get("source_episode_ids", []))
                            + [e.id for e in members],
                            "embedding_text": content[-2000:],
                        },
                    )
                )
            else:
                segment_id = new_id()
                self._heat[segment_id] = {
                    "n_visit": 0,
                    "length": len(members),
                    "last_access": datetime.now(timezone.utc),
                }
                content = summary
                ops.append(
                    self._segment_add(
                        segment_id,
                        str(g.get("topic", "?")),
                        content,
                        members,
                        ctx,
                        keywords,
                    )
                )

            # heat >= τ -> promote to LPM (profile/knowledge), then reset
            if self._segment_heat(segment_id) >= self.heat_threshold:
                ops.extend(self._promote_to_lpm(segment_id, content, members, ctx))

        # Lowest-heat eviction when MTM over capacity (paper-faithful; the
        # code's access-count LFU needs read-path visit feedback we lack —
        # round-5 C5/N1)
        while len(self._heat) > self.mtm_capacity:
            coldest = min(self._heat, key=self._segment_heat)
            self._heat.pop(coldest)
            self._drop_page_index(coldest)
            ops.append(
                MemoryOp(
                    op=OpType.DELETE,
                    target_type="pages",
                    target_id=coldest,
                    payload={"reason": "lowest_heat_eviction"},
                )
            )
        return ops

    def _segment_add(
        self,
        segment_id: str,
        topic: str,
        content: str,
        members: list[Episode],
        ctx: OrganizerContext,
        keywords: list[str] | None = None,
    ) -> MemoryOp:
        for e in members:
            self._unit_pages.setdefault(e.id, set()).add(segment_id)
        self._page_sources.setdefault(segment_id, set()).update(e.id for e in members)
        return MemoryOp(
            op=OpType.ADD,
            target_type="pages",
            target_id=segment_id,
            payload={
                "id": segment_id,
                "topic": topic,
                "content": content,
                "keywords": sorted(set(keywords or [])),
                "source_episode_ids": [e.id for e in members],
                "embedding_text": content[:2000],
            },
        )

    def _promote_to_lpm(
        self, segment_id: str, content: str, members: list[Episode], ctx: OrganizerContext
    ) -> list[MemoryOp]:
        result = ctx.llm.call(
            "distill",
            PROFILE_PROMPT.format(content=content[:4000]),
            PROFILE_SCHEMA,
            required_keys=("profile_facts",),
        )
        if result is None:
            # upstream keeps heat intact on failure so the segment gets
            # another promotion attempt (round-5 N5: we used to reset first)
            return []
        h = self._heat[segment_id]
        h["n_visit"], h["length"] = 0, 0  # paper: reset after analysis
        ops = []
        for fact in result["profile_facts"]:
            fact = str(fact).strip()
            if not fact or fact.lower() in ("none", "n/a"):
                continue  # upstream long_term.py rejects empty/none knowledge
            fact_id = new_id()
            ops.append(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="semantic",
                    target_id=fact_id,
                    payload={
                        "id": fact_id,
                        "content": fact,
                        "kind": "profile",
                        "source_episode_ids": [e.id for e in members],
                        "embedding_text": fact,
                    },
                )
            )
        return ops
