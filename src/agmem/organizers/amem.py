"""A-Mem organizer (arXiv:2502.12110, NeurIPS'25) — corrected implementation.

Pipeline per message: note construction (Ps1) -> top-k neighbor retrieval
-> single batched link/evolution call (Ps3) -> ADD note + LINK + UPDATE
neighbor ops.

Deviations from the reference code are deliberate bug fixes (scope per
docs/research/fidelity-round3-paper-code-forensics.md §1.4 and the
round-4 verification, docs/research/fidelity-round4-verification.md):
- neighbors are addressed by note ID, not result-list index (issue #32:
  library edition updates the wrong notes and stores dangling link ids)
- similarity is true cosine via our vector stores (issue #23: library
  edition's score field has inverted meaning; issue #24: L2-vs-cosine)
- evolution failure is an explicit drop, never a silent skip (upstream
  wraps evolution in a broad try/except with no counter; no tracker issue)
- neighbor-retrieval query is the metadata-enriched embedding_text
  (paper eq.(3)-faithful); both upstream codes query with raw
  note.content only
- an empty ``actions`` array falls back to both effects (small models
  omit the field); upstream treats it as a no-op
Ps1 is effectively dead in BOTH official editions: agiresearch add_note
never calls analyze_content (metadata stays at constructor defaults),
and WujiangXu's plain memory_layer.py lacks ``import re`` so metadata
falls back to empty keywords/tags and context "General"; only
memory_layer_robust.py behaves as the paper describes.
Read-path counterpart (1-hop link expansion, upstream eval's
find_related_memories_raw) is implemented in retrieval/pipeline.py.
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
        "actions": {"type": "array",
                    "items": {"type": "string",
                              "enum": ["strengthen", "update_neighbor"]}},
        "connections": {"type": "array", "items": {"type": "string"}},
        "new_note_tags": {"type": "array", "items": {"type": "string"}},
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
# single process_memory call over the whole neighborhood, with its
# should_evolve + actions["strengthen","update_neighbor"] structure).
EVOLVE_PROMPT = """You are an AI memory evolution agent managing a knowledge base.
A new memory note arrived. Decide whether and how it should evolve the memory.

New note:
  content: "{content}"
  context: "{context}"
  keywords: {keywords}
  tags: {tags}

Nearest neighbors:
{neighbors}

Decide:
1. should_evolve: whether this note should trigger any memory evolution
2. actions: which evolutions to perform when should_evolve is true —
   "strengthen" (connect the new note to related neighbors and refine its
   tags) and/or "update_neighbor" (rewrite neighbors' context/tags)
3. connections (strengthen): neighbor IDs genuinely related to the new note
4. new_note_tags (strengthen): refined tags for the NEW note in light of its
   neighborhood (repeat current tags if no refinement needed)
5. neighbor_updates (update_neighbor): neighbors whose context/tags should be
   rewritten in light of the new note (only when it truly adds information)

Return JSON: {{"should_evolve": true/false,
"actions": ["strengthen", "update_neighbor"],
"connections": ["<id>", ...], "new_note_tags": [...],
"neighbor_updates": [{{"id": "<id>", "new_context": "...", "new_tags": [...]}}]}}"""


class AMemOrganizer(Organizer):
    name = "amem"

    def __init__(self, top_k: int = 5) -> None:
        # k=5 is the upstream CODE default (hardcoded in both editions'
        # find_related_memories); the paper's k=10 is the QA retrieval k.
        self.top_k = top_k

    def on_message(self, ep: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        # upstream "talk start time": the conversation date when known,
        # not the ingest wall clock
        talk_time = ep.meta.get("date") or ep.timestamp.isoformat()
        if ctx.llm is None:
            logger.warning("amem: no LLM configured — storing bare note (explicit degradation)")
            note = Note(content=ep.content, namespace=ctx.namespace,
                        source_episode_ids=[ep.id], timestamp=ep.timestamp)
            return [self._add_op(note, talk_time)]

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
        ops = [self._add_op(note, talk_time)]

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
            f'- id={n["id"]} time={n.get("timestamp", "")} '
            f'content="{n.get("content", "")}" context="{n.get("context", "")}" '
            f'keywords={n.get("keywords", [])} tags={n.get("tags", [])}'
            for n in neighbors
        )
        evo = ctx.llm.call(
            "distill",
            EVOLVE_PROMPT.format(content=note.content, context=note.context,
                                 keywords=note.keywords, tags=note.tags,
                                 neighbors=neighbor_text),
            EVOLVE_SCHEMA, required_keys=("should_evolve", "connections"),
        )
        if evo is None:
            return ops  # drop counted; note itself is still stored

        # Upstream gating: nothing happens unless should_evolve, and each
        # effect belongs to an action ("strengthen" -> links + new-note tags,
        # "update_neighbor" -> neighbor rewrites).
        if not evo.get("should_evolve"):
            return ops
        actions = {str(a).lower() for a in evo.get("actions") or []}
        if not actions:  # small models may omit the field; keep both effects
            actions = {"strengthen", "update_neighbor"}
        valid_ids = set(neighbor_ids)  # bug fix #32: only real note IDs

        if "strengthen" in actions:
            connections = [c for c in evo.get("connections", []) if c in valid_ids]
            if connections:
                # unidirectional, as upstream: only the new note gains links
                ops.append(MemoryOp(op=OpType.LINK, target_type="notes",
                                    target_id=note.id,
                                    payload={"links": connections}))
            # the evolution call may refine the NEW note's tags
            # (upstream tags_to_update — audit P1-5)
            new_tags = [str(t) for t in evo.get("new_note_tags") or []]
            if new_tags and new_tags != note.tags:
                refreshed_self = Note(content=note.content, id=note.id,
                                      keywords=note.keywords, tags=new_tags,
                                      context=note.context)
                ops.append(MemoryOp(
                    op=OpType.UPDATE, target_type="notes", target_id=note.id,
                    payload={"tags": new_tags,
                             "embedding_text": refreshed_self.embedding_text()},
                ))

        if "update_neighbor" in actions:
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

    def _add_op(self, note: Note, talk_time: str) -> MemoryOp:
        return MemoryOp(
            op=OpType.ADD, target_type="notes", target_id=note.id,
            payload={
                "id": note.id, "content": note.content, "keywords": note.keywords,
                "tags": note.tags, "context": note.context, "links": note.links,
                "source_episode_ids": note.source_episode_ids,
                "timestamp": talk_time,
                "embedding_text": note.embedding_text(),
            },
        )
