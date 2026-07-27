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

Entity embeddings use the NAME only (upstream semantic candidate search)
while the item's ``content`` — what the render layer and the BM25 channel
see — is "name: summary", matching upstream's (name, summary) fulltext
index and its {entity_name, summary} context template. The graph store
comes from ctx.graph_store (persistent under data_dir — audit X4) unless
injected. GraphRecall lives in retrieval/steps.py.

Communities (the paper's third subgraph, §2.2.4) are built here too:
``flush_buffer`` runs a full label-propagation refresh when
``community_refresh`` is on (the default) and the graph changed since the
last one, and ``update_communities=True`` additionally extends communities
one entity at a time as messages arrive. The algorithm and its two summarization prompts
live in ``community.py``; this module owns the op emission, so community
nodes and their membership reach the graph the same way entities and facts
do — through the evolution log (audit B3), never by writing to the store.

Graph writes go through the op log like every other store. This organizer
used to call ``upsert_node``/``upsert_edge``/``invalidate_edge`` inline —
the only organizer that wrote to a store directly, so graph state was
invisible to the append-only log that base.py's contract rests on: a
replayed store came back with an empty graph (silently degrading
GraphRecall to plain vector RAG), and a hook that raised after a node
write left the graph ahead of the doc store. Now the entity/fact ops carry
what the graph needs (``entity_type`` on entities, ``subject_id``/
``object_id`` on facts) and ``AgenticMemory._apply_graph`` performs the
mutation (2026-07-27 audit B3). Reads still go straight to the store; the
one thing that changes for this code is that edges decided within a single
``on_message`` are not yet visible to ``edges_between``, which the local
``pending`` map covers.
"""

from __future__ import annotations

import hashlib
import logging
import re

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, new_id
from agmem.organizers.base import Organizer, OrganizerContext
from agmem.organizers.zep_graph.community import (
    DESCRIPTION_SCHEMA,
    MAX_SUMMARY_CHARS,
    SUMMARIZE_PAIR_PROMPT,
    SUMMARY_DESCRIPTION_PROMPT,
    SUMMARY_SCHEMA,
    label_propagation,
    truncate_at_sentence,
)
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

Guidelines (upstream extract_edges + paper §6.1.3):
- Extract facts ONLY between the entities listed above.
- Each fact must relate two DISTINCT entities.
- A fact involving MORE than two entities is emitted once per entity
  pair it holds between, repeating the same statement — that is how a
  multi-entity fact is represented (paper §2.2.2, "the same fact can be
  extracted multiple times between different entities … hyper-edges").
- predicate is a concise ALL-CAPS SCREAMING_SNAKE_CASE relation type
  (e.g. LOVES, LIVES_IN, IS_FRIENDS_WITH, WORKS_FOR).

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

Return JSON: {{"facts": [{{"subject": "<entity name>", "predicate": "SCREAMING_SNAKE_CASE",
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


def _node_text(name: str, summary: str) -> str:
    """`"name: summary"`, or the bare name when the model gave no summary —
    the entity form of upstream's context template ({entity_name, summary})
    and of its (name, summary) fulltext index."""
    return f"{name}: {summary}" if summary else name


def _relation_type(predicate: str) -> str:
    """Normalize a predicate to SCREAMING_SNAKE_CASE.

    Both the paper (§6.1.3: "a concise, all-caps description of the fact (e.g.,
    LOVES, IS_FRIENDS_WITH, WORKS_FOR)") and current upstream (``relation_type``
    field: "in SCREAMING_SNAKE_CASE") specify the casing, and it is not
    cosmetic: the predicate is part of the edge's identity in the graph, so
    ``lives_in`` and ``LIVES_IN`` from two different messages would read as two
    relation types. The prompt asks for it, and this makes it true regardless of
    what the model returns — a 0.6B model ignores casing instructions often
    enough that relying on the prompt alone would leave the graph mixed."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(predicate or "")).strip("_")
    return cleaned.upper() or "RELATED_TO"


def _community_id(member_ids: list[str]) -> str:
    """A community's id is derived from its membership, not freshly minted.

    Upstream mints a new uuid on every ``build_communities`` because it wipes
    the community subgraph first and rebuilds unconditionally. Here the rebuild
    emits ops into an append-only log and each community costs O(members) LLM
    calls, so re-minting would re-pay the whole summarization bill and append a
    DELETE + ADD pair for every community on every flush, even when the
    partition did not move. Keying on membership makes an unchanged cluster
    recognisable, so the refresh is idempotent."""
    digest = hashlib.sha1("\x1f".join(sorted(member_ids)).encode()).hexdigest()[:20]
    return f"zep:community:{digest}"


class ZepGraphOrganizer(Organizer):
    """Zep temporal-KG organizer (see module docstring for the extraction/resolution
    pipeline and paper mapping)."""

    name = "zep_graph"

    # facts BEFORE entities, matching the upstream-faithful config order: the
    # entities step (GraphRecall) pulls incident edge facts and dedups against
    # ids already in the bundle, so facts must be searched first or the same
    # fact is served twice. Communities last — they are the coarsest of the
    # paper's three subgraphs, so they lose ties for bundle space.
    produces = ("facts", "entities", "communities")

    def __init__(
        self,
        graph: SqliteGraphStore | None = None,
        candidate_threshold: float = 0.6,
        context_window: int = 4,
        community_refresh: bool = True,
        update_communities: bool = False,
    ) -> None:
        """`graph=None` defers to `ctx.graph_store` at hook time (facade-wired,
        persistent); pass an explicit `graph` to override that (e.g. standalone use).
        `candidate_threshold` is the min cosine similarity for entity-resolution
        embedding candidates (upstream `NODE_DEDUP_COSINE_MIN_SCORE`);
        `context_window` bounds how many recent messages are shown to the entity/fact
        extraction prompts (paper n=4).

        The two community knobs mirror upstream's two entry points, and their
        defaults are upstream's defaults. `community_refresh` runs the full
        label-propagation rebuild at `flush_buffer` — upstream's explicit
        `Graphiti.build_communities()`, which its own examples call once after
        ingestion. `update_communities` is the incremental single-step extension
        (`add_episode(update_communities=True)`, default False there too): each
        resolved entity joins the plurality community of its neighbours and that
        community is re-summarized. Turning it on costs 2 LLM calls per resolved
        entity per message, which is why it is not the default even though the
        paper presents incremental extension as label propagation's motivation
        (§2.2.4)."""
        self._own_graph = graph
        self.candidate_threshold = candidate_threshold  # upstream NODE_DEDUP_COSINE_MIN_SCORE
        self.context_window = context_window  # paper n=4 previous messages
        self.community_refresh = community_refresh
        self.update_communities = update_communities
        self._recent: list[Episode] = []
        # Entities resolved in the current on_message call, id -> summary, and
        # the endpoints of the edges it emitted, node id -> the other ends.
        self._touched: dict[str, str] = {}
        self._pending_neighbors: dict[str, list[str]] = {}
        # Whether any entity/fact op has been emitted since the last community
        # rebuild. The facade calls flush_buffer from both flush() and
        # consolidate(), and a rebuild is the most expensive thing this
        # organizer does, so an unchanged graph must not pay for it twice.
        self._graph_dirty = False

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
        exact normalized name -> LLM judgment. Returns the node id.

        Emits ops only — the node itself reaches the graph when the facade
        applies them (``AgenticMemory._apply_graph``)."""
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
                ops.append(
                    MemoryOp(
                        op=OpType.UPDATE,
                        target_type="entities",
                        target_id=dup,
                        payload={
                            "name": new_name,
                            "summary": new_summary,
                            # entity_type rides along so the graph upsert this op
                            # drives does not reset the node's type to the default
                            "entity_type": etype,
                            "content": _node_text(new_name, new_summary),
                            "embedding_text": new_name,
                        },
                    )
                )
                self._touched[dup] = new_summary
                return dup

        node_id = new_id()
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
                    # `content` is what the render layer and the BM25 channel see.
                    # Without it an entity item stored ONLY name/summary, so a
                    # retrieved entity rendered as an empty bullet and the lexical
                    # channel this config enables for `entities` indexed nothing
                    # at all (2026-07-27 round-7). "name: summary" matches both
                    # upstream surfaces: the node fulltext index is (name,
                    # summary) and the context template is {entity_name, summary}.
                    "content": _node_text(name, summary),
                    # ...while the dense channel embeds the NAME alone, which is
                    # what upstream's semantic candidate search queries with.
                    "embedding_text": name,
                },
            )
        )
        self._touched[node_id] = summary
        return node_id

    # -- communities (paper §2.2.4) -------------------------------------------

    def _summarize_pair(self, left: str, right: str, ctx: OrganizerContext) -> str | None:
        """One map-reduce step: fuse two summaries into one. `None` on a dropped
        call — the caller aborts that community rather than inventing a summary
        by concatenation, so the next refresh retries it."""
        out = ctx.llm.call(
            "distill",
            SUMMARIZE_PAIR_PROMPT.format(left=left, right=right),
            SUMMARY_SCHEMA,
            required_keys=("summary",),
        )
        text = str((out or {}).get("summary", "")).strip()
        return truncate_at_sentence(text, MAX_SUMMARY_CHARS) if text else None

    def _describe(self, summary: str, ctx: OrganizerContext) -> str | None:
        """The community NAME: a one-sentence description OF the summary
        (upstream `generate_summary_description`), not a short label. This is
        the field the community search channel embeds and indexes."""
        out = ctx.llm.call(
            "distill",
            SUMMARY_DESCRIPTION_PROMPT.format(summary=summary),
            DESCRIPTION_SCHEMA,
            required_keys=("description",),
        )
        text = str((out or {}).get("description", "")).strip()
        return text or None

    def _reduce_summaries(self, summaries: list[str], ctx: OrganizerContext) -> str | None:
        """Upstream `build_community`'s map-reduce: halve the list each round by
        summarizing the i-th against the (i + n/2)-th — FIRST half against
        SECOND half, not adjacent pairs — carrying an odd element forward
        untouched, until one summary remains."""
        summaries = list(summaries)
        length = len(summaries)
        while length > 1:
            odd_one_out: str | None = None
            if length % 2 == 1:
                odd_one_out = summaries.pop()
                length -= 1
            half = length // 2
            fused: list[str] = []
            for left, right in zip(summaries[:half], summaries[half:length], strict=True):
                merged = self._summarize_pair(left, right, ctx)
                if merged is None:
                    return None
                fused.append(merged)
            if odd_one_out is not None:
                fused.append(odd_one_out)
            summaries = fused
            length = len(summaries)
        return truncate_at_sentence(summaries[0], MAX_SUMMARY_CHARS) if summaries else None

    def _community_op(
        self, community_id: str, member_ids: list[str], summary: str, name: str
    ) -> MemoryOp:
        """ADD, not UPDATE, on purpose: a community's whole state is its name,
        summary and membership, all three of which the rebuild recomputes, so a
        full replace is the accurate op (same reasoning as MemoryOS's profile
        document) and it works whether or not the id already exists."""
        return MemoryOp(
            op=OpType.ADD,
            target_type="communities",
            target_id=community_id,
            payload={
                "id": community_id,
                "name": name,
                "summary": summary,
                # rendered form = upstream's {community_name, summary} template
                "content": _node_text(name, summary),
                # ...but the BM25 index upstream builds for communities covers
                # the NAME only (`community_name`), unlike entities' (name,
                # summary), and the embedding is the name as well.
                "lexical_text": name,
                "embedding_text": name,
                "member_ids": member_ids,
            },
        )

    def _build_community(self, member_ids: list[str], ctx: OrganizerContext) -> MemoryOp | None:
        """Summarize one cluster into a community node. Costs
        `len(members) - 1` pair calls plus one naming call."""
        by_id = {
            str(item.get("id")): str(item.get("summary") or item.get("name") or "")
            for item in ctx.doc_store.get_items(member_ids, "entities")
        }
        # Member order drives the map-reduce pairing and therefore the result,
        # so it is fixed here rather than left to store row order.
        summaries = [by_id[m] for m in member_ids if by_id.get(m)]
        if not summaries:
            return None
        summary = self._reduce_summaries(summaries, ctx)
        if summary is None:
            logger.warning("community build dropped: summarization returned nothing")
            return None
        name = self._describe(summary, ctx)
        if name is None:
            logger.warning("community build dropped: naming call returned nothing")
            return None
        return self._community_op(_community_id(member_ids), member_ids, summary, name)

    def rebuild_communities(self, ctx: OrganizerContext) -> list[MemoryOp]:
        """Full label-propagation refresh over the entity subgraph — upstream
        `Graphiti.build_communities()`, which is the periodic recomputation the
        paper pairs with incremental extension to bound drift (§2.2.4).

        Clusters whose membership is unchanged are skipped, so a second refresh
        with no new entities emits nothing and calls no LLM; clusters that no
        longer exist are DELETEd. That is where this departs from upstream,
        which wipes and rebuilds every community on every call — see
        `_community_id` for why."""
        graph = self._graph(ctx)
        namespace = ctx.namespace
        projection = graph.entity_projection(namespace)
        if not projection:
            return []
        wanted = {_community_id(c): sorted(c) for c in label_propagation(projection)}
        existing = {c["id"] for c in graph.communities(namespace)}

        ops: list[MemoryOp] = []
        for community_id, members in wanted.items():
            if community_id in existing and graph.community_members(community_id) == members:
                continue
            built = self._build_community(members, ctx)
            if built is not None:
                ops.append(built)
        for community_id in sorted(existing - set(wanted)):
            ops.append(
                MemoryOp(
                    op=OpType.DELETE,
                    target_type="communities",
                    target_id=community_id,
                    payload={"id": community_id},
                )
            )
        logger.info(
            "community refresh: %d entities -> %d communities (%d rebuilt, %d dropped)",
            len(projection),
            len(wanted),
            len(ops) - len(existing - set(wanted)),
            len(existing - set(wanted)),
        )
        return ops

    def _extend_community(
        self, node_id: str, summary: str, ctx: OrganizerContext
    ) -> MemoryOp | None:
        """Single-step incremental extension for one entity (upstream
        `update_community`): keep its community if it has one, else join the
        community holding the plurality of its neighbours, then re-summarize
        that community against this entity's summary.

        The plurality is resolved deterministically by (-votes, id); upstream
        iterates a dict and keeps the first strict maximum, so its tie-break is
        whatever order the driver returned.

        The edges this message just decided on are counted as votes even though
        the facade has not applied them yet — the same window the `pending` map
        covers for contradiction judgment. Without it the mechanism is dead
        rather than merely delayed: a newly extracted entity's ONLY edges are
        the ones in this message, so it would never have a visible neighbour
        and could never join anything. Upstream sees them because it saves
        nodes and edges before touching communities."""
        graph = self._graph(ctx)
        community = graph.community_of_node(node_id, ctx.namespace)
        is_new = False
        if community is None:
            votes: dict[str, int] = {}
            found: dict[str, dict] = {}
            for row in graph.neighbor_communities(node_id, ctx.namespace):
                votes[row["id"]] = votes.get(row["id"], 0) + 1
                found[row["id"]] = row
            for other in self._pending_neighbors.get(node_id, []):
                row = graph.community_of_node(other, ctx.namespace)
                if row is not None:
                    votes[row["id"]] = votes.get(row["id"], 0) + 1
                    found[row["id"]] = row
            if not votes:
                return None  # no neighbour is in a community yet
            winner = min(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            community, is_new = found[winner], True

        new_summary = self._summarize_pair(summary, str(community.get("summary", "")), ctx)
        if new_summary is None:
            return None
        new_name = self._describe(new_summary, ctx)
        if new_name is None:
            return None
        members = graph.community_members(community["id"])
        if is_new:
            members = sorted(set(members) | {node_id})
        # The id stays the community's own even though membership changed, so it
        # no longer matches `_community_id(members)`. That is what makes the next
        # full refresh rebuild it — which is exactly the drift correction the
        # paper's periodic recomputation exists for.
        return self._community_op(community["id"], members, new_summary, new_name)

    def flush_buffer(self, ctx: OrganizerContext) -> list[MemoryOp]:
        """End-of-ingest community refresh. No-op when nothing was written since
        the last one, or when `community_refresh=False`."""
        if not self.community_refresh or not self._graph_dirty or ctx.llm is None:
            return []
        self._graph_dirty = False
        return self.rebuild_communities(ctx)

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
        self._touched, self._pending_neighbors = {}, {}
        for ent in extracted["entities"][:10]:
            if str(ent.get("name", "")).strip():
                name_to_id[str(ent["name"]).strip()] = self._resolve_entity(ent, episode, ctx, ops)

        if len(name_to_id) < 2:
            return self._finish(ops, ctx)

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
            return self._finish(ops, ctx)

        # Edges this call has decided on but the facade has not applied yet.
        # Writes are deferred to `_apply_graph` now, so `edges_between` cannot
        # see them; without this, two facts about the same entity pair in ONE
        # message would each be judged against a graph missing the other.
        # Keyed by unordered pair, matching `edges_between`'s either-direction
        # match.
        pending: dict[frozenset[str], list[dict]] = {}

        for f in facts.get("facts", [])[:10]:
            subj, obj = name_to_id.get(f.get("subject")), name_to_id.get(f.get("object"))
            statement = str(f.get("statement", "")).strip()
            if not subj or not obj or not statement:
                continue  # entity name hallucinated by the model — drop the fact
            if subj == obj:
                # "Each fact should represent a clear relationship between two
                # DISTINCT nodes" (paper §6.1.3). A self-loop is also a graph
                # hazard here: `edges_between(x, x)` matches it twice through the
                # either-direction clause, so it would be its own duplicate
                # candidate on the next message.
                continue
            valid_at = str(f.get("valid_at") or ref_time)
            invalid_at = f.get("invalid_at") or None

            pair = frozenset((subj, obj))
            existing = graph.edges_between(subj, obj, ctx.namespace) + [
                e for e in pending.get(pair, []) if not e.get("invalid_at")
            ]
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
                    # the INVALIDATE op carries this to the graph; mark the
                    # local view so a later fact in this same message does not
                    # contradict an edge already on its way out
                    e["invalid_at"] = valid_at
                    ops.append(
                        MemoryOp(
                            op=OpType.INVALIDATE,
                            target_type="facts",
                            target_id=contradicted_id,
                            payload={"t_invalid": valid_at},
                        )
                    )

            edge_id = new_id()
            payload = {
                "id": edge_id,
                "content": statement,
                "subject": f.get("subject"),
                "predicate": _relation_type(f.get("predicate")),
                "object": f.get("object"),
                # Endpoint IDS, not just the names: the edge is rebuilt from
                # this payload by `_apply_graph`, and names are not resolvable
                # back to nodes at apply time (two entities can share a name,
                # and resolution already happened above).
                "subject_id": subj,
                "object_id": obj,
                "valid_at": valid_at,
                "source_episode_ids": [episode.id],
                "embedding_text": statement,
            }
            if invalid_at:
                payload["invalid_at"] = str(invalid_at)
            self._pending_neighbors.setdefault(subj, []).append(obj)
            self._pending_neighbors.setdefault(obj, []).append(subj)
            pending.setdefault(pair, []).append(
                {
                    "id": edge_id,
                    "content": statement,
                    "valid_at": valid_at,
                    "invalid_at": payload.get("invalid_at"),
                }
            )
            ops.append(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="facts",
                    target_id=edge_id,
                    payload=payload,
                )
            )
        return self._finish(ops, ctx)

    def _finish(self, ops: list[MemoryOp], ctx: OrganizerContext) -> list[MemoryOp]:
        """Mark the graph dirty for the next refresh and, when the incremental
        path is on, extend a community per entity this message resolved."""
        if ops:
            self._graph_dirty = True
        if not self.update_communities:
            return ops
        for node_id, summary in self._touched.items():
            extended = self._extend_community(node_id, summary, ctx)
            if extended is not None:
                ops.append(extended)
        return ops
