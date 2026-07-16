"""A-Mem organizer (arXiv:2502.12110, NeurIPS'25) — corrected implementation.

Pipeline per message: note construction (Ps1) -> top-k neighbor retrieval
-> single batched link/evolution call (Ps3) -> ADD note + LINK + UPDATE
neighbor ops.

Deviations from the reference code are deliberate bug fixes:
- neighbors are addressed by note ID, not result-list index (issue #32)
- similarity is true cosine via our vector stores (issue #24: reference
  ChromaDB collection used L2 while treating scores as similarity)
- evolution failure is an explicit drop, never a silent skip (issue #10)
Set ``fidelity="paper"`` only to mirror original hyperparameters (k=5);
the buggy behaviors themselves are not reproduced.
"""

from __future__ import annotations

import logging

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, Note
from agmem.organizers.base import Organizer, OrganizerContext

logger = logging.getLogger("agmem.organizers.amem")

NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {"type": "array", "items": {"type": "string"}},
        "context": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["keywords", "context", "tags"],
}

EVOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "should_evolve": {"type": "boolean"},
        "connections": {"type": "array", "items": {"type": "string"}},
        "neighbor_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "new_context": {"type": "string"},
                    "new_tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id"],
            },
        },
    },
    "required": ["should_evolve", "connections"],
}

# Condensed from A-Mem Ps1.
NOTE_PROMPT = """Generate a structured analysis of the content:
1. keywords: at least 3, most important first, exclude speaker names and timestamps
2. context: one sentence summarizing topic, key points, and who/what it concerns
3. tags: at least 3 (domain, format, type)

Content: "{content}"

Return JSON: {{"keywords": [...], "context": "...", "tags": [...]}}"""

# Condensed from A-Mem Ps2+Ps3 as one batched call (matches reference code's
# single process_memory call over the whole neighborhood).
EVOLVE_PROMPT = """A new memory note arrived. Decide how it relates to its nearest neighbors.

New note:
  content: "{content}"
  context: "{context}"
  tags: {tags}

Neighbors:
{neighbors}

Decide:
1. connections: neighbor IDs genuinely related to the new note (may be empty)
2. neighbor_updates: neighbors whose context/tags should be rewritten in light
   of the new note (only when it truly adds information; may be empty)

Return JSON: {{"should_evolve": true/false, "connections": ["<id>", ...],
"neighbor_updates": [{{"id": "<id>", "new_context": "...", "new_tags": [...]}}]}}"""


class AMemOrganizer(Organizer):
    name = "amem"

    def __init__(self, top_k: int = 5, fidelity: str = "fixed") -> None:
        self.top_k = top_k          # paper default k=5
        self.fidelity = fidelity

    def on_message(self, ep: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        if ctx.llm is None:
            logger.warning("amem: no LLM configured — storing bare note (explicit degradation)")
            note = Note(content=ep.content, namespace=ctx.namespace,
                        source_episode_ids=[ep.id], timestamp=ep.timestamp)
            return [self._add_op(note)]

        # 1. note construction (Ps1) — one LLM call
        meta = ctx.llm.call("extract", NOTE_PROMPT.format(content=ep.content),
                            NOTE_SCHEMA, required_keys=("keywords", "context", "tags"))
        note = Note(
            content=ep.content, namespace=ctx.namespace,
            keywords=[str(x) for x in (meta or {}).get("keywords", [])],
            tags=[str(x) for x in (meta or {}).get("tags", [])],
            context=str((meta or {}).get("context", "")),
            source_episode_ids=[ep.id], timestamp=ep.timestamp,
        )
        ops = [self._add_op(note)]

        # 2. neighbor retrieval — embedding includes metadata (A-Mem finding)
        note_emb = ctx.embedder.embed([note.embedding_text()])[0]
        hits = ctx.vec.search(note_emb, k=self.top_k,
                              memory_type="notes", namespace=ctx.namespace)
        neighbor_ids = [h[0] for h in hits]
        neighbors = ctx.doc.get_items(neighbor_ids, "notes")
        if not neighbors:
            return ops

        # 3. link + evolution (Ps3) — one batched LLM call over all neighbors
        neighbor_text = "\n".join(
            f'- id={n["id"]} content="{n.get("content", "")[:200]}" '
            f'context="{n.get("context", "")}" tags={n.get("tags", [])}'
            for n in neighbors
        )
        evo = ctx.llm.call(
            "distill",
            EVOLVE_PROMPT.format(content=note.content, context=note.context,
                                 tags=note.tags, neighbors=neighbor_text),
            EVOLVE_SCHEMA, required_keys=("should_evolve", "connections"),
        )
        if evo is None:
            return ops  # drop counted; note itself is still stored

        valid_ids = set(neighbor_ids)  # bug fix #32: only real note IDs
        connections = [c for c in evo.get("connections", []) if c in valid_ids]
        if connections:
            ops.append(MemoryOp(op=OpType.LINK, target_type="notes", target_id=note.id,
                                payload={"links": connections}))
            for cid in connections:  # bidirectional links
                ops.append(MemoryOp(op=OpType.LINK, target_type="notes",
                                    target_id=cid, payload={"links": [note.id]}))

        if evo.get("should_evolve"):
            by_id = {n["id"]: n for n in neighbors}
            for upd in evo.get("neighbor_updates", []):
                nid = upd.get("id")
                if nid not in valid_ids:
                    continue
                old = by_id[nid]
                new_context = str(upd.get("new_context") or old.get("context", ""))
                new_tags = [str(t) for t in (upd.get("new_tags") or old.get("tags", []))]
                refreshed = Note(
                    content=old.get("content", ""), id=nid, context=new_context,
                    tags=new_tags, keywords=old.get("keywords", []),
                )
                ops.append(MemoryOp(
                    op=OpType.UPDATE, target_type="notes", target_id=nid,
                    payload={"context": new_context, "tags": new_tags,
                             "embedding_text": refreshed.embedding_text()},
                ))
        return ops

    def _add_op(self, note: Note) -> MemoryOp:
        return MemoryOp(
            op=OpType.ADD, target_type="notes", target_id=note.id,
            payload={
                "id": note.id, "content": note.content, "keywords": note.keywords,
                "tags": note.tags, "context": note.context, "links": note.links,
                "source_episode_ids": note.source_episode_ids,
                "embedding_text": note.embedding_text(),
            },
        )
