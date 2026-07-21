"""Zep-style temporal knowledge graph organizer (arXiv:2501.13956) — round-5 rebuild.

Per message: entity extraction with the last n previous messages as
context (paper §2.2.1, n=4) -> three-stage entity resolution as today's
Graphiti does (embedding candidates >= 0.6 -> deterministic exact-name
match -> LLM dedup judgment, refreshing the node's name/summary on merge)
-> fact extraction with INTEGRATED temporal fields (valid_at/invalid_at
resolved against the message timestamp, as upstream extract_edges now
does) -> same-pair duplicate/contradiction resolution in one LLM call
(duplicate -> provenance append, contradiction -> temporally-guarded
INVALIDATE, t_invalid = the invalidating fact's valid_at). Raw episodes
stay untouched (verbatim-loss defense).

Entity embeddings use the NAME only (upstream semantic candidate search);
the render layer shows "name: summary". The graph store comes from
ctx.graph_store (persistent under data_dir — audit X4) unless injected.
Community detection (label propagation) remains TODO (round-5 ①);
GraphRecall lives in retrieval/pipeline.py.
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
    "properties": {
        "entities": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["name"],
            },
        }
    },
    "required": ["entities"],
}

RESOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "duplicate_id": {"type": ["string", "null"]},
        "name": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["duplicate_id"],
}

FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "statement": {"type": "string"},
                    "valid_at": {"type": ["string", "null"]},
                    "invalid_at": {"type": ["string", "null"]},
                },
                "required": ["subject", "predicate", "object", "statement"],
            },
        }
    },
    "required": ["facts"],
}

EDGE_RESOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "duplicate_of": {"type": ["string", "null"]},
        "contradicts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["contradicts"],
}

ENTITY_PROMPT = """Extract the distinct real-world entities mentioned in the CURRENT
MESSAGE (people, places, organizations, objects, concepts). Always
include the speaker. Disambiguate pronouns ("he", "she", "they", "it")
into the named entity using the previous messages. Do not extract
relations, actions, dates, or bare pronouns as entities.

<PREVIOUS MESSAGES>
{previous}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{content}
</CURRENT MESSAGE>

Return JSON: {{"entities": [{{"name": "...", "type": "Person|Place|Organization|Object|Topic",
"summary": "one clause"}}]}}"""

RESOLVE_PROMPT = """Decide whether the NEW entity is the same real-world entity as one of
the CANDIDATES. Same thing under a different spelling or nickname counts
as a duplicate; a different thing with a similar name does NOT (e.g.
"Java" the language vs "Java" the island).

NEW entity: name="{name}", summary="{summary}"
Message context: "{content}"

Candidates:
{candidates}

Return JSON: {{"duplicate_id": "<candidate id or null>",
"name": "best canonical name", "summary": "one-clause merged summary"}}"""

FACT_PROMPT = """Extract relationship facts between these entities from the CURRENT
MESSAGE. Use ONLY these entity names as subject/object: {names}

REFERENCE TIME: {ref_time} (when the message was said). For each fact,
resolve when it became true:
- valid_at: ISO date the fact became true. Resolve relative expressions
  ("two weeks ago", "last year") against the reference time; if the fact
  is stated as current/ongoing, use the reference time; year-only dates
  become January 1 of that year.
- invalid_at: ISO date the fact stopped being true, ONLY when the
  message says it ended; otherwise null.

<PREVIOUS MESSAGES>
{previous}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{content}
</CURRENT MESSAGE>

Return JSON: {{"facts": [{{"subject": "<entity name>", "predicate": "snake_case_relation",
"object": "<entity name>", "statement": "the fact as one sentence",
"valid_at": "...", "invalid_at": null}}]}}"""

EDGE_RESOLVE_PROMPT = """A new fact arrived between the same two entities as some existing
facts. Decide two things:
1. duplicate_of: the id of an existing fact stating the SAME information
   (null if none).
2. contradicts: ids of existing facts that can no longer be true if the
   new fact is true (usually none).

Existing facts:
{existing}

New fact: "{statement}" (valid from {valid_at})

Return JSON: {{"duplicate_of": null, "contradicts": ["<edge id>", ...]}}"""


def _fmt(episode: Episode) -> str:
    return f"[{episode.timestamp.isoformat()}] {episode.role}: {episode.content}"


class ZepGraphOrganizer(Organizer):
    """Zep temporal-KG organizer (see module docstring for the extraction/resolution
    pipeline and paper mapping)."""

    name = "zep_graph"

    def __init__(
        self,
        graph: SqliteGraphStore | None = None,
        candidate_threshold: float = 0.6,
        context_window: int = 4,
    ) -> None:
        """`graph=None` defers to `ctx.graph_store` at hook time (facade-wired,
        persistent); pass an explicit `graph` to override that (e.g. standalone use).
        `candidate_threshold` is the min cosine similarity for entity-resolution
        embedding candidates (upstream `NODE_DEDUP_COSINE_MIN_SCORE`);
        `context_window` bounds how many recent messages are shown to the entity/fact
        extraction prompts (paper n=4)."""
        self._own_graph = graph
        self.candidate_threshold = candidate_threshold  # upstream NODE_DEDUP_COSINE_MIN_SCORE
        self.context_window = context_window  # paper n=4 previous messages
        self._recent: list[Episode] = []

    def _graph(self, ctx: OrganizerContext) -> SqliteGraphStore:
        g = self._own_graph or getattr(ctx, "graph_store", None)
        if g is None:  # standalone use without AgenticMemory wiring
            g = self._own_graph = SqliteGraphStore(":memory:")
        self._resolved_graph = g
        return g

    @property
    def graph(self) -> SqliteGraphStore | None:
        """The graph actually in use (own override, else the ctx-wired one)."""
        return self._own_graph or getattr(self, "_resolved_graph", None)

    # -- entity resolution ----------------------------------------------------

    def _resolve_entity(
        self, ent: dict, episode: Episode, ctx: OrganizerContext, ops: list[MemoryOp]
    ) -> str:
        """Three-stage resolution (Graphiti): embedding candidates ->
        exact normalized name -> LLM judgment. Returns the node id."""
        graph = self._graph(ctx)
        name = str(ent.get("name", "")).strip()
        summary = str(ent.get("summary", ""))
        etype = str(ent.get("type", "Entity"))

        query_embedding = ctx.embedder.embed([name])[0]  # name only, as upstream
        hits = [
            (i, s)
            for i, s in ctx.vector_store.search(
                query_embedding, k=5, memory_type="entities", namespace=ctx.namespace
            )
            if s >= self.candidate_threshold
        ]
        candidates = ctx.doc_store.get_items([i for i, _ in hits], "entities")

        norm = name.casefold()
        for c in candidates:  # deterministic exact-name match
            if str(c.get("name", "")).casefold() == norm:
                return c["id"]

        if candidates and ctx.llm is not None:  # LLM dedup judgment
            verdict = ctx.llm.call(
                "extract",
                RESOLVE_PROMPT.format(
                    name=name,
                    summary=summary,
                    content=episode.content,
                    candidates="\n".join(
                        f'- id={c["id"]} name="{c.get("name", "")}" '
                        f'summary="{c.get("summary", "")}"'
                        for c in candidates
                    ),
                ),
                RESOLVE_SCHEMA,
                required_keys=("duplicate_id",),
            )
            dup = (verdict or {}).get("duplicate_id")
            by_id = {c["id"]: c for c in candidates}
            if dup in by_id:
                # merge: refresh canonical name/summary (paper: "generates
                # an updated name and summary" — round-5 ⑦)
                new_name = str(verdict.get("name") or by_id[dup].get("name", name))
                new_summary = str(verdict.get("summary") or by_id[dup].get("summary", summary))
                graph.upsert_node(dup, ctx.namespace, new_name, new_summary, etype)
                ops.append(
                    MemoryOp(
                        op=OpType.UPDATE,
                        target_type="entities",
                        target_id=dup,
                        payload={
                            "name": new_name,
                            "summary": new_summary,
                            "embedding_text": new_name,
                        },
                    )
                )
                return dup

        node_id = new_id()
        graph.upsert_node(node_id, ctx.namespace, name, summary, etype)
        ops.append(
            MemoryOp(
                op=OpType.ADD,
                target_type="entities",
                target_id=node_id,
                payload={
                    "id": node_id,
                    "name": name,
                    "summary": summary,
                    "entity_type": etype,
                    "source_episode_ids": [episode.id],
                    "embedding_text": name,
                },
            )
        )
        return node_id

    # -- hook -----------------------------------------------------------------

    def on_message(self, episode: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        """Returns `[]` without calling the LLM if `ctx.llm` is unset (logged warning,
        explicit skip) or if entity extraction finds nothing. Entities are resolved
        and their ops appended before fact extraction runs, so a partial result (no
        facts, or fewer than 2 resolved entities) still keeps entity-resolution ops.
        Facts naming an unresolved/hallucinated entity are dropped individually."""
        previous = "\n".join(_fmt(e) for e in self._recent) or "(none)"
        self._recent = (self._recent + [episode])[-self.context_window :]

        if ctx.llm is None:
            logger.warning("zep_graph: no LLM — skipping graph construction (explicit skip)")
            return []
        graph = self._graph(ctx)

        extracted = ctx.llm.call(
            "extract",
            ENTITY_PROMPT.format(previous=previous, content=episode.content),
            ENTITY_SCHEMA,
            required_keys=("entities",),
        )
        if not extracted or not extracted.get("entities"):
            return []

        ops: list[MemoryOp] = []
        name_to_id: dict[str, str] = {}
        for ent in extracted["entities"][:10]:
            if str(ent.get("name", "")).strip():
                name_to_id[str(ent["name"]).strip()] = self._resolve_entity(ent, episode, ctx, ops)

        if len(name_to_id) < 2:
            return ops

        ref_time = episode.timestamp.isoformat()
        facts = ctx.llm.call(
            "extract",
            FACT_PROMPT.format(
                names=list(name_to_id),
                ref_time=ref_time,
                previous=previous,
                content=episode.content,
            ),
            FACT_SCHEMA,
            required_keys=("facts",),
        )
        if not facts:
            return ops

        for f in facts.get("facts", [])[:10]:
            subj, obj = name_to_id.get(f.get("subject")), name_to_id.get(f.get("object"))
            statement = str(f.get("statement", "")).strip()
            if not subj or not obj or not statement:
                continue  # entity name hallucinated by the model — drop the fact
            valid_at = str(f.get("valid_at") or ref_time)
            invalid_at = f.get("invalid_at") or None

            existing = graph.edges_between(subj, obj, ctx.namespace)
            if existing:
                by_id = {e["id"]: e for e in existing}
                verdict = (
                    ctx.llm.call(
                        "distill",
                        EDGE_RESOLVE_PROMPT.format(
                            existing="\n".join(
                                f'- id={e["id"]} "{e["content"]}" '
                                f"(valid {e.get('valid_at') or '?'} - "
                                f"{e.get('invalid_at') or 'present'})"
                                for e in existing
                            ),
                            statement=statement,
                            valid_at=valid_at,
                        ),
                        EDGE_RESOLVE_SCHEMA,
                        required_keys=("contradicts",),
                    )
                    or {}
                )

                dup = verdict.get("duplicate_of")
                if dup in by_id:
                    # duplicate: reuse the edge, append provenance (upstream
                    # episodes.append) — no new edge (round-5 ⑤)
                    items = ctx.doc_store.get_items([dup], "facts")
                    prov = list((items[0] if items else {}).get("source_episode_ids", []))
                    ops.append(
                        MemoryOp(
                            op=OpType.UPDATE,
                            target_type="facts",
                            target_id=dup,
                            payload={"source_episode_ids": prov + [episode.id]},
                        )
                    )
                    continue

                for contradicted_id in verdict.get("contradicts", []):
                    e = by_id.get(contradicted_id)
                    if e is None:
                        continue
                    # temporal-overlap guard (upstream
                    # resolve_edge_contradictions): don't retro-invalidate
                    # facts that had already ended, or that started after
                    # the new fact ended
                    if e.get("invalid_at") and str(e["invalid_at"]) <= valid_at:
                        continue
                    if invalid_at and e.get("valid_at") and str(invalid_at) <= str(e["valid_at"]):
                        continue
                    graph.invalidate_edge(contradicted_id, valid_at)
                    ops.append(
                        MemoryOp(
                            op=OpType.INVALIDATE,
                            target_type="facts",
                            target_id=contradicted_id,
                            payload={"t_invalid": valid_at},
                        )
                    )

            edge_id = new_id()
            graph.upsert_edge(
                edge_id,
                ctx.namespace,
                subj,
                obj,
                str(f.get("predicate", "related_to")),
                statement,
                valid_at=valid_at,
            )
            payload = {
                "id": edge_id,
                "content": statement,
                "subject": f.get("subject"),
                "predicate": f.get("predicate"),
                "object": f.get("object"),
                "valid_at": valid_at,
                "source_episode_ids": [episode.id],
                "embedding_text": statement,
            }
            if invalid_at:
                payload["invalid_at"] = str(invalid_at)
                graph.invalidate_edge(edge_id, str(invalid_at))
            ops.append(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="facts",
                    target_id=edge_id,
                    payload=payload,
                )
            )
        return ops
