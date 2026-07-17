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
    "properties": {"groups": {
        "type": "array",
        "items": {"type": "object",
                  "properties": {"topic": {"type": "string"},
                                 "summary": {"type": "string"},
                                 "keywords": {"type": "array",
                                              "items": {"type": "string"}},
                                 "message_indexes": {"type": "array",
                                                     "items": {"type": "integer"}}},
                  "required": ["topic", "summary"]}}},
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
    name = "memoryos"

    def __init__(self, stm_capacity: int = 10, mtm_capacity: int = 2000,
                 heat_threshold: float = 5.0, similarity_threshold: float = 0.6,
                 recency_tau_hours: float = 24.0) -> None:
        # mtm_capacity 2000 = upstream default (round-5 N6 fixed 200->2000)
        self.stm_capacity = stm_capacity
        self.mtm_capacity = mtm_capacity
        self.heat_threshold = heat_threshold
        self.similarity_threshold = similarity_threshold
        self.recency_tau_hours = recency_tau_hours
        self._stm: list[Episode] = []
        self._heat: dict[str, dict] = {}  # segment_id -> {n_visit, length, last_access}

    # -- heat ------------------------------------------------------------------

    def _segment_heat(self, seg_id: str) -> float:
        h = self._heat.get(seg_id)
        if not h:
            return 0.0
        hours = (datetime.now(timezone.utc)
                 - h["last_access"]).total_seconds() / 3600
        return h["n_visit"] + h["length"] + math.exp(-hours / self.recency_tau_hours)

    # -- hook --------------------------------------------------------------------

    def on_message(self, ep: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        self._stm.append(ep)
        if len(self._stm) < self.stm_capacity:
            return []
        batch, self._stm = self._stm, []
        return self._evict_to_mtm(batch, ctx)

    def flush_buffer(self, ctx: OrganizerContext) -> list[MemoryOp]:
        if not self._stm:
            return []
        batch, self._stm = self._stm, []
        return self._evict_to_mtm(batch, ctx)

    def warm_start(self, corpus, ctx: OrganizerContext) -> list[MemoryOp]:
        ops = super().warm_start(corpus, ctx)
        ops.extend(self.flush_buffer(ctx))
        return ops

    # -- STM -> MTM ---------------------------------------------------------------

    def _evict_to_mtm(self, batch: list[Episode], ctx: OrganizerContext) -> list[MemoryOp]:
        if ctx.llm is None:
            logger.warning("memoryos: no LLM — storing mechanical segment (explicit degradation)")
            seg_id = new_id()
            content = "\n".join(e.content for e in batch)
            self._heat[seg_id] = {"n_visit": 0, "length": len(batch),
                                  "last_access": datetime.now(timezone.utc)}
            return [self._segment_add(seg_id, "batch", content, batch, ctx)]

        indexed = "\n".join(f"[{i}] {e.content}" for i, e in enumerate(batch))
        result = ctx.llm.call("distill", TOPIC_PROMPT.format(messages=indexed),
                              TOPIC_SCHEMA, required_keys=("groups",))
        groups = (result or {}).get("groups") or [
            {"topic": "batch", "summary": "\n".join(e.content for e in batch),
             "message_indexes": list(range(len(batch)))}]

        ops: list[MemoryOp] = []
        for g in groups:
            idxs = [i for i in g.get("message_indexes", [])
                    if isinstance(i, int) and 0 <= i < len(batch)] or list(range(len(batch)))
            members = [batch[i] for i in idxs]
            summary = str(g.get("summary", ""))
            keywords = [str(k).lower() for k in g.get("keywords") or []]
            emb = ctx.embedder.embed([summary])[0]
            # F_score = cos + Jaccard(keywords), threshold 0.6 — paper eq.(3);
            # round-5 P0 restored the Jaccard term (cosine-only was stricter
            # and fragmented segments). Consider top-3 candidates.
            hits = ctx.vec.search(emb, k=3, memory_type="pages", namespace=ctx.namespace)
            best_id, best_f = None, 0.0
            for hid, cos in hits:
                if hid not in self._heat:
                    continue
                cand = ctx.doc.get_items([hid], "pages")
                cand_kw = set((cand[0] if cand else {}).get("keywords", []))
                union = cand_kw | set(keywords)
                jac = (len(cand_kw & set(keywords)) / len(union)) if union else 0.0
                f = cos + jac
                if f > best_f:
                    best_id, best_f = hid, f

            if best_id is not None and best_f >= self.similarity_threshold:
                seg_id = best_id  # merge into existing segment (F_score >= θ)
                existing = ctx.doc.get_items([seg_id], "pages")
                old = existing[0] if existing else {}
                content = (old.get("content", "") + "\n" + summary).strip()
                merged_kw = sorted(set(old.get("keywords", [])) | set(keywords))
                h = self._heat[seg_id]
                h["length"] += len(members)
                h["last_access"] = datetime.now(timezone.utc)
                ops.append(MemoryOp(
                    op=OpType.UPDATE, target_type="pages", target_id=seg_id,
                    payload={"content": content, "keywords": merged_kw,
                             "source_episode_ids": list(old.get("source_episode_ids", []))
                             + [e.id for e in members],
                             "embedding_text": content[-2000:]},
                ))
            else:
                seg_id = new_id()
                self._heat[seg_id] = {"n_visit": 0, "length": len(members),
                                      "last_access": datetime.now(timezone.utc)}
                content = summary
                ops.append(self._segment_add(seg_id, str(g.get("topic", "?")),
                                             content, members, ctx, keywords))

            # heat >= τ -> promote to LPM (profile/knowledge), then reset
            if self._segment_heat(seg_id) >= self.heat_threshold:
                ops.extend(self._promote_to_lpm(seg_id, content, members, ctx))

        # Lowest-heat eviction when MTM over capacity (paper-faithful; the
        # code's access-count LFU needs read-path visit feedback we lack —
        # round-5 C5/N1)
        while len(self._heat) > self.mtm_capacity:
            coldest = min(self._heat, key=self._segment_heat)
            self._heat.pop(coldest)
            ops.append(MemoryOp(op=OpType.DELETE, target_type="pages",
                                target_id=coldest,
                                payload={"reason": "lowest_heat_eviction"}))
        return ops

    def _segment_add(self, seg_id: str, topic: str, content: str,
                     members: list[Episode], ctx: OrganizerContext,
                     keywords: list[str] | None = None) -> MemoryOp:
        return MemoryOp(
            op=OpType.ADD, target_type="pages", target_id=seg_id,
            payload={"id": seg_id, "topic": topic, "content": content,
                     "keywords": sorted(set(keywords or [])),
                     "source_episode_ids": [e.id for e in members],
                     "embedding_text": content[:2000]},
        )

    def _promote_to_lpm(self, seg_id: str, content: str, members: list[Episode],
                        ctx: OrganizerContext) -> list[MemoryOp]:
        result = ctx.llm.call("distill", PROFILE_PROMPT.format(content=content[:4000]),
                              PROFILE_SCHEMA, required_keys=("profile_facts",))
        if result is None:
            # upstream keeps heat intact on failure so the segment gets
            # another promotion attempt (round-5 N5: we used to reset first)
            return []
        h = self._heat[seg_id]
        h["n_visit"], h["length"] = 0, 0  # paper: reset after analysis
        ops = []
        for fact in result["profile_facts"]:
            fact = str(fact).strip()
            if not fact or fact.lower() in ("none", "n/a"):
                continue  # upstream long_term.py rejects empty/none knowledge
            fid = new_id()
            ops.append(MemoryOp(
                op=OpType.ADD, target_type="semantic", target_id=fid,
                payload={"id": fid, "content": fact, "kind": "profile",
                         "source_episode_ids": [e.id for e in members],
                         "embedding_text": fact},
            ))
        return ops
