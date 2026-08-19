"""Zep-style temporal knowledge graph organizer (arXiv:2501.13956) — round-5 rebuild.

Per message: entity extraction with the last n previous messages as
context (paper §2.2.1, n=4) -> three-stage entity resolution as today's
Graphiti does (embedding candidates >= 0.6 with k=15 -> deterministic
stage: exact normalized-name match PLUS fuzzy MinHash/LSH, ``dedup.py``
-> ONE batched LLM dedup call for every entity the deterministic stage
left unresolved). A confirmed merge refreshes the canonical node's
name/summary — that refresh is the PAPER's dedup description ("generates
an updated name and summary"), not current main's: main's
``NodeDuplicate`` carries no summary field and ``_promote_resolved_node``
only promotes type labels (round-12 finding 2). Then fact extraction with
INTEGRATED temporal fields (valid_at/invalid_at resolved against the
message timestamp, as upstream extract_edges now does) -> duplicate/
contradiction resolution in one LLM call per new fact, with duplicate
candidates from the same entity pair and invalidation candidates from a
GRAPH-WIDE dense search minus the same-pair set (upstream
edge_operations.py:408-419; see ``on_message``). Duplicates append
provenance; contradictions apply upstream's actual truth table
(``resolve_edge_contradictions``/``expire_new_edge`` below): only a
strictly OLDER valid_at is invalidated, and a strictly NEWER candidate
expires the NEW edge at write time instead. Raw episodes stay untouched
(verbatim-loss defense).

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

Upstream snapshot provenance (docs/16 session 4): current graphiti main has
moved past the paper — ``SagaNode`` (``_get_or_create_saga``/``summarize_saga``),
single-call node+edge extraction (``combined_extraction.extract_nodes_and_
edges``), and ``temporal_operations.py`` dissolved into ``extract_edges``.
Sagas and combined extraction are NOT ported; entity resolution follows
current main (three-stage as above, with the merge refresh being the one
paper-lineage piece inside it — named where it happens), the temporal
truth table and the duplicate fast paths follow current main, and the
extraction prompts/context window follow the paper. This port is
therefore a dated mixed snapshot, deliberately: each piece's lineage is
named where it is used.
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
from agmem.organizers.zep_graph.dedup import (
    _normalize_string_exact,
    build_candidate_indexes,
    deterministic_resolve,
)
from agmem.stores.sqlite_graph import SqliteGraphStore

logger = logging.getLogger("agmem.organizers.zep_graph")

# Upstream node_operations.py:64 — the candidate pool per unresolved entity in
# the semantic search AND the LLM dedupe context. Ours used k=5 until round-12
# finding 3.
NODE_DEDUP_CANDIDATE_LIMIT = 15
# Upstream's invalidation-candidate search runs EDGE_HYBRID_SEARCH_RRF at its
# default limit (search_config.DEFAULT_SEARCH_LIMIT = 10) — the constructor
# knob `invalidation_candidate_limit` defaults to it.
INVALIDATION_CANDIDATE_LIMIT = 10

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

# One resolution entry per unresolved entity, integer-indexed like upstream's
# `NodeDuplicate {id, duplicate_candidate_id}` (prompts/dedupe_nodes.py) — the
# name/summary fields on top of it are the PAPER's merge refresh, which main
# does not ask the dedupe call for (round-12 finding 2, deliberate).
RESOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "resolutions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "duplicate_candidate_id": {"type": "integer"},
                    "name": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["id", "duplicate_candidate_id"],
            },
        }
    },
    "required": ["resolutions"],
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

RESOLVE_PROMPT = """Decide for each NEW entity whether it is the same real-world entity as
one of the CANDIDATES. Same thing under a different spelling or nickname
counts as a duplicate; a different thing with a similar name does NOT
(e.g. "Java" the language vs "Java" the island).

NEW entities:
{entities}

Candidates:
{candidates}

Message context: "{content}"

Return JSON with ONE entry per NEW entity:
{{"resolutions": [{{"id": <new entity id>,
"duplicate_candidate_id": <candidate_id of the duplicate, or -1 if none>,
"name": "best canonical name", "summary": "one-clause merged summary"}}]}}"""

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

# Upstream's resolve_edge context carries TWO candidate lists in one call
# (edge_operations.py:698-713): EXISTING FACTS (same entity pair — only these
# may be the duplicate) and FACT INVALIDATION CANDIDATES (graph-wide search
# hits — these plus the same-pair facts may be contradicted). Same split here;
# `duplicate_of` outside the same-pair section is ignored on parse.
EDGE_RESOLVE_PROMPT = """A new fact arrived. Decide two things:
1. duplicate_of: the id of an EXISTING FACT (same two entities) stating
   the SAME information (null if none). Only ids from the EXISTING FACTS
   list qualify.
2. contradicts: ids of facts from EITHER list that can no longer be true
   if the new fact is true (usually none).

EXISTING FACTS (between the same two entities):
{existing}

FACT INVALIDATION CANDIDATES (elsewhere in the graph):
{invalidation_candidates}

New fact: "{statement}" (valid from {valid_at})

Return JSON: {{"duplicate_of": null, "contradicts": ["<fact id>", ...]}}"""


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


def _ts(value) -> str | None:
    """ISO timestamp string or None. Comparisons below are string comparisons,
    which order correctly for uniformly-formatted ISO-8601 — the same
    convention the store's `_ACTIVE` filter and `edges_between` rely on."""
    return str(value) if value else None


def resolve_edge_contradictions(
    resolved_edge: dict, invalidation_candidates: list[dict]
) -> list[dict]:
    """Upstream ``resolve_edge_contradictions`` (edge_operations.py:538-573),
    the FULL truth table — dict-shaped (``valid_at``/``invalid_at`` ISO strings
    or None) instead of EntityEdge.

    Two skip conditions, then the strictly-older guard:
    - skip if the candidate had already ended before the new fact began
      (``edge.invalid_at <= resolved.valid_at``);
    - skip if the new fact ended before the candidate began
      (``resolved.invalid_at <= edge.valid_at``);
    - invalidate ONLY if ``edge.valid_at < resolved.valid_at`` — strictly
      older, both non-None. An equal or LATER candidate valid_at is never
      invalidated (that case expires the NEW edge instead — ``expire_new_edge``),
      and a None valid_at on either side is inert: every upstream condition
      requires non-None (round-12 findings 5/6).

    Mutates each invalidated candidate's ``invalid_at`` to the new fact's
    ``valid_at`` (t_invalid) and returns them; the T' axis (``expired_at``) is
    stamped by the store's ``invalidate_edge`` when the op is applied, which is
    where upstream's ``edge.expired_at = utc_now()`` lives here."""
    invalidated: list[dict] = []
    resolved_valid = _ts(resolved_edge.get("valid_at"))
    resolved_invalid = _ts(resolved_edge.get("invalid_at"))
    for edge in invalidation_candidates:
        edge_valid = _ts(edge.get("valid_at"))
        edge_invalid = _ts(edge.get("invalid_at"))
        if (
            edge_invalid is not None
            and resolved_valid is not None
            and edge_invalid <= resolved_valid
        ) or (
            edge_valid is not None
            and resolved_invalid is not None
            and resolved_invalid <= edge_valid
        ):
            continue
        elif edge_valid is not None and resolved_valid is not None and edge_valid < resolved_valid:
            edge["invalid_at"] = resolved_valid
            invalidated.append(edge)
    return invalidated


def expire_new_edge(resolved_edge: dict, invalidation_candidates: list[dict]) -> None:
    """Upstream's new-edge self-expiry (edge_operations.py:826-841): if any
    contradicting candidate has a strictly LATER valid_at than the new fact,
    the NEW edge enters the graph already expired — ``invalid_at`` = the
    EARLIEST such candidate's valid_at (upstream sorts candidates by valid_at,
    None last, and breaks on the first hit). Mutates ``resolved_edge`` in
    place; runs only when the new edge is not already expired (upstream keys on
    ``expired_at``, which it stamps whenever ``invalid_at`` is set — here that
    means: skip when the model already returned an invalid_at). Must run BEFORE
    ``resolve_edge_contradictions``, as upstream orders them: the self-expiry
    can set ``resolved.invalid_at``, which the skip conditions then read."""
    if resolved_edge.get("invalid_at"):
        return
    resolved_valid = _ts(resolved_edge.get("valid_at"))
    candidates = sorted(
        invalidation_candidates,
        key=lambda c: (c.get("valid_at") is None, _ts(c.get("valid_at")) or ""),
    )
    for candidate in candidates:
        candidate_valid = _ts(candidate.get("valid_at"))
        if (
            candidate_valid is not None
            and resolved_valid is not None
            and candidate_valid > resolved_valid
        ):
            resolved_edge["invalid_at"] = candidate_valid
            break


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
        invalidation_candidate_limit: int = INVALIDATION_CANDIDATE_LIMIT,
    ) -> None:
        """`graph=None` defers to `ctx.graph_store` at hook time (facade-wired,
        persistent); pass an explicit `graph` to override that (e.g. standalone use).
        `candidate_threshold` is the min cosine similarity for entity-resolution
        embedding candidates (upstream `NODE_DEDUP_COSINE_MIN_SCORE`);
        `context_window` bounds how many recent messages are shown to the entity/fact
        extraction prompts — 4 is the PAPER's n=4 (§2.2.1), and a recorded
        divergence from current main, whose `EPISODE_WINDOW_LEN = 3`
        (graph_data_operations.py:29; round-12 finding 15).
        `invalidation_candidate_limit` caps the graph-wide invalidation-candidate
        search per new fact — upstream runs EDGE_HYBRID_SEARCH_RRF at its
        default limit 10 there (edge_operations.py:408-419); see `on_message`
        for the dense-only stand-in.

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
        self.context_window = context_window  # paper n=4 (main uses 3 — see docstring)
        self.invalidation_candidate_limit = invalidation_candidate_limit
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

    def _new_entity_op(self, ent: dict, episode: Episode, ops: list[MemoryOp]) -> str:
        """ADD op for an entity that resolved to nothing existing; returns the
        new node id."""
        name = str(ent.get("name", "")).strip()
        summary = str(ent.get("summary", ""))
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
                    "entity_type": str(ent.get("type", "Entity")),
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

    def _merge_entity_op(
        self, ent: dict, canonical: dict, verdict: dict, ops: list[MemoryOp]
    ) -> str:
        """UPDATE op refreshing the canonical node on an LLM-confirmed merge.

        The name/summary refresh is the PAPER's dedup description ("generates
        an updated name and summary"), NOT current main's: main's
        ``NodeDuplicate`` is ``{id, name, duplicate_candidate_id}`` with no
        summary field, and ``_promote_resolved_node`` keeps the existing node,
        only promoting type labels — summaries there are maintained by a
        separate attribute/summary batch path (round-12 finding 2; behavior
        kept deliberately, lineage relabeled)."""
        dup = str(canonical.get("id"))
        name = str(ent.get("name", "")).strip()
        summary = str(ent.get("summary", ""))
        new_name = str(verdict.get("name") or canonical.get("name", name))
        new_summary = str(verdict.get("summary") or canonical.get("summary", summary))
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
                    "entity_type": str(ent.get("type", "Entity")),
                    "content": _node_text(new_name, new_summary),
                    "embedding_text": new_name,
                },
            )
        )
        self._touched[dup] = new_summary
        return dup

    def _resolve_entities(
        self, ents: list[dict], episode: Episode, ctx: OrganizerContext, ops: list[MemoryOp]
    ) -> list[str]:
        """Three-stage resolution as current main (``resolve_extracted_nodes``,
        node_operations.py:627-690): per entity, semantic candidates (name
        embedding, k=15, cosine >= 0.6) -> the deterministic stage from
        ``dedup.py`` (exact normalized name + fuzzy MinHash/LSH; an ambiguous
        exact match escalates instead of first-wins) -> ONE batched LLM dedupe
        call for ALL still-unresolved entities of the message against their
        merged candidate pool (node_operations.py:552-556). The batch shape is
        load-bearing for call-count parity: upstream pays at most one dedupe
        call per message, not one per entity (round-12 findings 1/3).

        Returns node ids aligned with ``ents``. Emits ops only — nodes reach
        the graph when the facade applies them (``AgenticMemory._apply_graph``)."""
        resolved: list[str | None] = [None] * len(ents)
        candidates_per_entity: list[list[dict]] = []
        unresolved: list[int] = []

        for idx, ent in enumerate(ents):
            name = str(ent.get("name", "")).strip()
            query_embedding = ctx.embedder.embed([name])[0]  # name only, as upstream
            hits = [
                (i, s)
                for i, s in ctx.vector_store.search(
                    query_embedding,
                    k=NODE_DEDUP_CANDIDATE_LIMIT,
                    memory_type="entities",
                    namespace=ctx.namespace,
                )
                if s >= self.candidate_threshold
            ]
            candidates = ctx.doc_store.get_items([i for i, _ in hits], "entities")
            candidates_per_entity.append(candidates)
            if not candidates:
                # no semantic candidates -> new node, no LLM (upstream leaves
                # the slot unresolved and never sends it to the dedupe call)
                continue
            match_id = deterministic_resolve(name, build_candidate_indexes(candidates))
            if match_id is not None:
                resolved[idx] = match_id
            else:
                unresolved.append(idx)

        if unresolved and ctx.llm is not None:
            # Merged candidate pool across the unresolved entities, deduped by
            # id, integer candidate ids as upstream's existing_nodes context.
            merged: list[dict] = []
            seen_ids: set[str] = set()
            for idx in unresolved:
                for c in candidates_per_entity[idx]:
                    cid = str(c.get("id"))
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        merged.append(c)
            verdict = ctx.llm.call(
                "extract",
                RESOLVE_PROMPT.format(
                    entities="\n".join(
                        f'- id={rel} name="{ents[idx].get("name", "")}" '
                        f'summary="{ents[idx].get("summary", "")}"'
                        for rel, idx in enumerate(unresolved)
                    ),
                    candidates="\n".join(
                        f'- candidate_id={i} name="{c.get("name", "")}" '
                        f'summary="{c.get("summary", "")}"'
                        for i, c in enumerate(merged)
                    ),
                    content=episode.content,
                ),
                RESOLVE_SCHEMA,
                required_keys=("resolutions",),
            )
            # Guardrails as upstream's _resolve_with_llm: invalid or repeated
            # ids are ignored, an invalid candidate id means "no duplicate",
            # entities the model skipped stay unresolved and become new nodes.
            processed: set[int] = set()
            for resolution in (verdict or {}).get("resolutions", []):
                rel = resolution.get("id")
                if not isinstance(rel, int) or not (0 <= rel < len(unresolved)):
                    continue
                if rel in processed:
                    continue
                processed.add(rel)
                idx = unresolved[rel]
                cand_id = resolution.get("duplicate_candidate_id")
                if isinstance(cand_id, int) and 0 <= cand_id < len(merged):
                    resolved[idx] = self._merge_entity_op(
                        ents[idx], merged[cand_id], resolution, ops
                    )

        return [
            node_id if node_id is not None else self._new_entity_op(ents[idx], episode, ops)
            for idx, node_id in enumerate(resolved)
        ]

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

    # This hook resolves each new episode against entities and facts ALREADY in
    # the stores (it searches them, and embeds to do it), so it sees a different
    # world if the corpus is indexed before hooks run. That is exactly what
    # ``bulk_ingest`` does, and this flag is what keeps it off that path.
    observes_store_on_message = True

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
        self._touched, self._pending_neighbors = {}, {}
        ents = [e for e in extracted["entities"][:10] if str(e.get("name", "")).strip()]
        node_ids = self._resolve_entities(ents, episode, ctx, ops)
        name_to_id = {
            str(ent["name"]).strip(): node_id for ent, node_id in zip(ents, node_ids, strict=True)
        }

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
        # Fast path (a): in-batch pre-dedup of identical extractions — same
        # endpoints (directional) + same normalized fact text collapse to one
        # edge before any resolution work (upstream edge_operations.py:344-358).
        seen_in_batch: set[tuple[str, str, str]] = set()

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
            normalized_fact = _normalize_string_exact(statement)
            batch_key = (subj, obj, normalized_fact)
            if batch_key in seen_in_batch:
                continue
            seen_in_batch.add(batch_key)
            # A null valid_at from the model STAYS None — upstream parses null
            # to None (edge_operations.py:253-270), and a None-valid_at edge can
            # neither invalidate nor be invalidated (every condition in the
            # truth table requires non-None). The old `or ref_time` default made
            # every fact a dated, invalidation-capable fact — more aggressive
            # temporal semantics than either lineage (round-12 finding 6).
            # Call-parity note: upstream retries a missing valid_at once via
            # `_extract_edge_timestamps` — an extra LLM call per new dateless
            # edge that is deliberately NOT reproduced here.
            valid_at = _ts(f.get("valid_at"))
            invalid_at = _ts(f.get("invalid_at"))

            pair = frozenset((subj, obj))
            existing = graph.edges_between(subj, obj, ctx.namespace) + [
                e for e in pending.get(pair, []) if not e.get("invalid_at")
            ]

            # Fast path (b): verbatim reuse — normalized fact text + directional
            # endpoints matching an existing edge exactly skip the LLM call and
            # just append provenance (upstream edge_operations.py:687-700).
            verbatim = next(
                (
                    e
                    for e in existing
                    if str(e.get("src")) == subj
                    and str(e.get("dst")) == obj
                    and _normalize_string_exact(str(e.get("content", ""))) == normalized_fact
                ),
                None,
            )
            if verbatim is not None:
                items = ctx.doc_store.get_items([verbatim["id"]], "facts")
                prov = list((items[0] if items else {}).get("source_episode_ids", []))
                ops.append(
                    MemoryOp(
                        op=OpType.UPDATE,
                        target_type="facts",
                        target_id=verbatim["id"],
                        payload={"source_episode_ids": prov + [episode.id]},
                    )
                )
                continue

            # Invalidation candidates come from a GRAPH-WIDE search, minus the
            # same-pair duplicates (upstream edge_operations.py:408-419). This
            # is a documented stand-in: upstream runs EDGE_HYBRID_SEARCH_RRF
            # (bm25 + cosine fused by RRF) over all edges of the group; here it
            # is the DENSE channel only, over the `facts` type in the
            # namespace, k = `invalidation_candidate_limit` (upstream's search
            # limit 10). Without this pool the paper's flagship temporal
            # mechanism only ever fired between the identical entity pair
            # (round-12 finding 4).
            same_pair_ids = {str(e["id"]) for e in existing}
            statement_embedding = ctx.embedder.embed([statement])[0]
            hit_ids = [
                i
                for i, _ in ctx.vector_store.search(
                    statement_embedding,
                    k=self.invalidation_candidate_limit,
                    memory_type="facts",
                    namespace=ctx.namespace,
                )
            ]
            inval_candidates = [
                item
                for item in ctx.doc_store.get_items(hit_ids, "facts")
                if str(item.get("id")) not in same_pair_ids
            ]

            if existing or inval_candidates:
                by_id = {str(e["id"]): e for e in existing}
                inval_by_id = {str(e["id"]): e for e in inval_candidates}

                def _fact_line(e: dict) -> str:
                    return (
                        f'- id={e["id"]} "{e["content"]}" '
                        f"(valid {e.get('valid_at') or '?'} - "
                        f"{e.get('invalid_at') or 'present'})"
                    )

                verdict = (
                    ctx.llm.call(
                        "distill",
                        EDGE_RESOLVE_PROMPT.format(
                            existing="\n".join(_fact_line(e) for e in existing) or "(none)",
                            invalidation_candidates="\n".join(
                                _fact_line(e) for e in inval_candidates
                            )
                            or "(none)",
                            statement=statement,
                            valid_at=valid_at or "unknown",
                        ),
                        EDGE_RESOLVE_SCHEMA,
                        required_keys=("contradicts",),
                    )
                    or {}
                )

                dup = verdict.get("duplicate_of")
                if dup in by_id:  # only a same-pair fact can be the duplicate
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

                # The LLM flags contradictions from either list (upstream's
                # continuous indexing across both); the temporal truth table
                # then decides what actually expires, in upstream's order:
                # self-expiry of the NEW edge first, then invalidation of the
                # strictly-older candidates.
                contradicted = [
                    by_id.get(cid) or inval_by_id.get(cid) for cid in verdict.get("contradicts", [])
                ]
                contradicted = [e for e in contradicted if e is not None]
                new_edge_times = {"valid_at": valid_at, "invalid_at": invalid_at}
                expire_new_edge(new_edge_times, contradicted)
                invalid_at = new_edge_times.get("invalid_at")
                for e in resolve_edge_contradictions(new_edge_times, contradicted):
                    # `resolve_edge_contradictions` already stamped the local
                    # view's invalid_at, so a later fact in this same message
                    # does not contradict an edge already on its way out; the
                    # INVALIDATE op carries it to the graph.
                    ops.append(
                        MemoryOp(
                            op=OpType.INVALIDATE,
                            target_type="facts",
                            target_id=str(e["id"]),
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
                # Either the model's own invalid_at or the self-expiry above.
                # `_apply_graph` re-stamps the graph edge after the upsert, so
                # the edge enters the graph already expired, as upstream's
                # resolved_edge does.
                payload["invalid_at"] = str(invalid_at)
            self._pending_neighbors.setdefault(subj, []).append(obj)
            self._pending_neighbors.setdefault(obj, []).append(subj)
            pending.setdefault(pair, []).append(
                {
                    "id": edge_id,
                    "content": statement,
                    # src/dst so the verbatim fast path can match pending edges
                    # the same way it matches store rows
                    "src": subj,
                    "dst": obj,
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
