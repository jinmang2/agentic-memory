"""Zep-style temporal knowledge graph organizer (arXiv:2501.13956) — compact port.

Per message: entity extraction -> embedding-based entity resolution ->
fact (edge) extraction -> LLM contradiction check -> bi-temporal
INVALIDATE + ADD. Raw episodes stay untouched (verbatim-loss defense).

Deviations: the graph store is organizer-owned auxiliary structure written
directly (doc/vec still go through MemoryOps, which remain the audit
trail); community detection (label propagation) is TODO for a later phase.
"""

from __future__ import annotations

import logging

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, new_id
from agmem.organizers.base import Organizer, OrganizerContext
from agmem.stores.sqlite_graph import SqliteGraphStore

logger = logging.getLogger("agmem.organizers.zep_graph")

ENTITY_SCHEMA = {
    "type": "object",
    "properties": {"entities": {
        "type": "array", "maxItems": 5,
        "items": {"type": "object",
                  "properties": {"name": {"type": "string"},
                                 "type": {"type": "string"},
                                 "summary": {"type": "string"}},
                  "required": ["name"]}}},
    "required": ["entities"],
}

FACT_SCHEMA = {
    "type": "object",
    "properties": {"facts": {
        "type": "array", "maxItems": 5,
        "items": {"type": "object",
                  "properties": {"subject": {"type": "string"},
                                 "predicate": {"type": "string"},
                                 "object": {"type": "string"},
                                 "statement": {"type": "string"}},
                  "required": ["subject", "predicate", "object", "statement"]}}},
    "required": ["facts"],
}

CONTRA_SCHEMA = {
    "type": "object",
    "properties": {"contradicts": {"type": "array", "items": {"type": "string"}}},
    "required": ["contradicts"],
}

ENTITY_PROMPT = """Extract the distinct real-world entities mentioned in this message
(people, places, organizations, objects, concepts; include the speaker; at most 5).

Message: "{content}"

Return JSON: {{"entities": [{{"name": "...", "type": "Person|Place|Organization|Object|Topic",
"summary": "one clause"}}]}}"""

FACT_PROMPT = """Extract relationship facts between these entities from the message.
Use ONLY these entity names as subject/object: {names}

Message: "{content}"

Return JSON: {{"facts": [{{"subject": "<entity name>", "predicate": "snake_case_relation",
"object": "<entity name>", "statement": "the fact as one sentence"}}]}}"""

CONTRA_PROMPT = """A new fact arrived. Which of the existing facts does it contradict
(i.e. can no longer be true if the new fact is true)? Usually none.

Existing facts:
{existing}

New fact: "{statement}"

Return JSON: {{"contradicts": ["<edge id>", ...]}}"""


class ZepGraphOrganizer(Organizer):
    name = "zep_graph"

    def __init__(self, graph: SqliteGraphStore | None = None,
                 resolve_threshold: float = 0.85) -> None:
        self.graph = graph or SqliteGraphStore(":memory:")
        self.resolve_threshold = resolve_threshold

    def on_message(self, ep: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        if ctx.llm is None:
            logger.warning("zep_graph: no LLM — skipping graph construction (explicit skip)")
            return []

        extracted = ctx.llm.call("extract", ENTITY_PROMPT.format(content=ep.content),
                                 ENTITY_SCHEMA, required_keys=("entities",))
        if not extracted or not extracted.get("entities"):
            return []

        ops: list[MemoryOp] = []
        name_to_id: dict[str, str] = {}
        for ent in extracted["entities"][:5]:
            name = str(ent.get("name", "")).strip()
            if not name:
                continue
            summary = str(ent.get("summary", ""))
            emb = ctx.embedder.embed([f"{name}: {summary}" if summary else name])[0]
            hits = ctx.vec.search(emb, k=1, memory_type="entities",
                                  namespace=ctx.namespace)
            if hits and hits[0][1] >= self.resolve_threshold:
                name_to_id[name] = hits[0][0]  # resolved to existing node
                continue
            node_id = new_id()
            name_to_id[name] = node_id
            self.graph.upsert_node(node_id, ctx.namespace, name, summary,
                                   str(ent.get("type", "Entity")))
            ops.append(MemoryOp(
                op=OpType.ADD, target_type="entities", target_id=node_id,
                payload={"id": node_id, "name": name, "summary": summary,
                         "entity_type": str(ent.get("type", "Entity")),
                         "source_episode_ids": [ep.id],
                         "embedding_text": f"{name}: {summary}" if summary else name},
            ))

        if len(name_to_id) < 2:
            return ops

        facts = ctx.llm.call(
            "extract",
            FACT_PROMPT.format(names=list(name_to_id), content=ep.content),
            FACT_SCHEMA, required_keys=("facts",))
        if not facts:
            return ops

        ts = ep.timestamp.isoformat()
        for f in facts.get("facts", [])[:5]:
            subj, obj = name_to_id.get(f.get("subject")), name_to_id.get(f.get("object"))
            statement = str(f.get("statement", "")).strip()
            if not subj or not obj or not statement:
                continue  # entity name hallucinated by the model — drop the fact

            existing = self.graph.edges_between(subj, obj, ctx.namespace)
            if existing:
                verdict = ctx.llm.call(
                    "distill",
                    CONTRA_PROMPT.format(
                        existing="\n".join(f'- id={e["id"]} "{e["content"]}"'
                                           for e in existing),
                        statement=statement),
                    CONTRA_SCHEMA, required_keys=("contradicts",))
                valid_ids = {e["id"] for e in existing}
                for eid in (verdict or {}).get("contradicts", []):
                    if eid not in valid_ids:
                        continue
                    self.graph.invalidate_edge(eid, ts)
                    ops.append(MemoryOp(op=OpType.INVALIDATE, target_type="facts",
                                        target_id=eid, payload={"t_invalid": ts}))

            edge_id = new_id()
            self.graph.upsert_edge(edge_id, ctx.namespace, subj, obj,
                                   str(f.get("predicate", "related_to")),
                                   statement, valid_at=ts)
            ops.append(MemoryOp(
                op=OpType.ADD, target_type="facts", target_id=edge_id,
                payload={"id": edge_id, "content": statement,
                         "subject": f.get("subject"), "predicate": f.get("predicate"),
                         "object": f.get("object"), "valid_at": ts,
                         "source_episode_ids": [ep.id], "embedding_text": statement},
            ))
        return ops
