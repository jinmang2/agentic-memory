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
- first-note skip: with an empty store we make no evolution call —
  this branch follows the ROBUST edition (memory_layer_robust.py:473-474).
  The plain (published-numbers) edition still spends the Ps3 call on an
  empty neighbor block (memory_layer.py:753-758, 861-862) and its
  strengthen can then extend ``note.links`` with dangling integer indices
  into an empty memory list. Vs the plain edition our per-conversation
  evolution call count is therefore exactly -1 (cost-parity caveat, same
  treatment as the MemoryOS first-page continuity call); reproducing the
  wasted call buys nothing, and the dangling-link half is structurally
  impossible under the facade (links are validated note IDs).
Tag refinement (``new_note_tags``) is applied UNCONDITIONALLY, including
empty lists, exactly like the plain edition's ``note.tags = new_tags``
(memory_layer.py:834-836; its strict schema requires the key but permits
``[]``). Variant: the robust edition guards emptiness only
(memory_layer_robust.py:506-507). A ``[]`` wipe is auditable in the
facade's op log, so fidelity wins over protectiveness here.
Ps1 is effectively dead in BOTH official editions: agiresearch add_note
never calls analyze_content (metadata stays at constructor defaults),
and WujiangXu's plain memory_layer.py lacks ``import re`` so metadata
falls back to empty keywords/tags and context "General"; only
memory_layer_robust.py behaves as the paper describes.
All agiresearch/A-mem claims in this module (the add_note/analyze_content
claim above, the issue #23/#24/#32 characterizations) refer to a repo not
retained in ~/.agmem/upstream; verified 2026-07-27, not re-verifiable
locally.
Read-path counterpart (1-hop link expansion, upstream eval's
find_related_memories_raw) is ``LinkExpansion`` in retrieval/steps.py,
configured by ``AgmemConfig.link_expansion_cap``. It used to live in
retrieval/pipeline.py and this line went stale when the read path was
plugin-ised; the same refactor also broke the ``--expand-links`` ablation
for a while (round-6 A1), so the pointer is worth keeping exact.
"""

from __future__ import annotations

import logging
from typing import Any

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
        "actions": {
            "type": "array",
            "items": {"type": "string", "enum": ["strengthen", "update_neighbor"]},
        },
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


def _verdict_id(entry: Any) -> str | None:
    """Neighbor id out of one evolution-verdict entry, whichever shape it has.

    ``EVOLVE_SCHEMA`` declares ``connections`` as id strings and
    ``neighbor_updates`` as objects, but nothing enforces it —
    ``use_guided_json`` is off for the local-model experiments, so the schema is
    advisory. Small models routinely swap the two shapes, and each swap used to
    raise inside ``_ingest`` (``dict`` is unhashable for the ``valid_ids``
    membership test; ``str`` has no ``.get``). Because the exception escaped
    after the note ADD op was built, and both the async worker and
    ``_propagate_events`` log-and-continue, the note was **silently lost** —
    corrupting the note counts that Nemori v4 Table 7 is measured on. Observed
    with Qwen3-0.6B on LoCoMo conv0.

    Recovering the id keeps the note and its link instead of discarding a verdict
    the model did express, in the same spirit as the ``str()`` coercions the
    surrounding code already applies to ``actions``/tags."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in ("id", "memory_id", "note_id"):
            value = entry.get(key)
            if isinstance(value, str):
                return value
    return None


class AMemOrganizer(Organizer):
    """A-Mem note organizer (arXiv:2502.12110 Ps1-Ps3; see module docstring
    for the upstream bug fixes this port applies). Per message: construct a
    note, retrieve ``top_k`` neighbors, then one batched evolution call
    decides links/tag refinements — all as returned ADD/LINK/UPDATE
    MemoryOps, never direct store writes."""

    name = "amem"

    produces = ("notes",)

    def __init__(self, top_k: int = 5) -> None:
        # k=5 is the upstream CODE default (hardcoded in both editions'
        # find_related_memories); the paper's k=10 is the QA retrieval k.
        self.top_k = top_k

    def on_message(self, episode: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        """Runs the full note pipeline (Ps1 construction -> neighbor retrieval
        -> one batched evolution call) for one message via ``_ingest``.

        Chained use (feeding this organizer another organizer's episodes) is an
        experimental composition and lives in
        ``organizers.experimental.ChainedConsumer``, not here — this organizer
        stays messages-only and paper-faithful."""
        return self._ingest(episode, ctx)

    def _ingest(self, episode: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        # upstream "talk start time": the conversation date when known,
        # not the ingest wall clock
        talk_time = episode.meta.get("date") or episode.timestamp.isoformat()
        if ctx.llm is None:
            logger.warning("amem: no LLM configured — storing bare note (explicit degradation)")
            note = Note(
                content=episode.content,
                namespace=ctx.namespace,
                source_episode_ids=[episode.id],
                timestamp=episode.timestamp,
            )
            return [self._add_op(note, talk_time)]

        # 1. note construction (Ps1) — one LLM call
        meta = ctx.llm.call(
            "extract",
            NOTE_PROMPT.format(content=episode.content),
            NOTE_SCHEMA,
            required_keys=("keywords", "context", "tags"),
        )
        note = Note(
            content=episode.content,
            namespace=ctx.namespace,
            keywords=[str(x) for x in (meta or {}).get("keywords", [])],
            tags=[str(x) for x in (meta or {}).get("tags", [])],
            context=str((meta or {}).get("context", "")),
            source_episode_ids=[episode.id],
            timestamp=episode.timestamp,
        )
        ops = [self._add_op(note, talk_time)]

        # 2. neighbor retrieval — embedding includes metadata (A-Mem finding)
        query_embedding = ctx.embedder.embed([note.embedding_text()])[0]
        hits = ctx.vector_store.search(
            query_embedding, k=self.top_k, memory_type="notes", namespace=ctx.namespace
        )
        neighbor_ids = [h[0] for h in hits]
        neighbors = ctx.doc_store.get_items(neighbor_ids, "notes")
        if not neighbors:
            return ops

        # 3. link + evolution (Ps3) — one batched LLM call over all neighbors
        neighbor_text = "\n".join(
            f"- id={n['id']} time={n.get('timestamp', '')} "
            f'content="{n.get("content", "")}" context="{n.get("context", "")}" '
            f"keywords={n.get('keywords', [])} tags={n.get('tags', [])}"
            for n in neighbors
        )
        evolution_verdict = ctx.llm.call(
            "distill",
            EVOLVE_PROMPT.format(
                content=note.content,
                context=note.context,
                keywords=note.keywords,
                tags=note.tags,
                neighbors=neighbor_text,
            ),
            EVOLVE_SCHEMA,
            required_keys=("should_evolve", "connections"),
        )
        if evolution_verdict is None:
            return ops  # explicit drop (verdict None) — the note ADD still stands

        # Upstream gating: nothing happens unless should_evolve, and each
        # effect belongs to an action ("strengthen" -> links + new-note tags,
        # "update_neighbor" -> neighbor rewrites).
        if not evolution_verdict.get("should_evolve"):
            return ops
        actions = {str(a).lower() for a in evolution_verdict.get("actions") or []}
        if not actions:  # small models may omit the field; keep both effects
            actions = {"strengthen", "update_neighbor"}
        # bug fix #32: only real note IDs — keyed off notes actually returned by
        # the doc store (not raw search hits), so a hit missing from the store
        # can't pass the gate and later KeyError on by_id lookup (2026-07-21 review A1)
        by_id = {n["id"]: n for n in neighbors}
        valid_ids = set(by_id)

        if "strengthen" in actions:
            seen_conn: set[str] = set()
            connections = []
            for entry in evolution_verdict.get("connections", []) or []:
                conn_id = _verdict_id(entry)
                # dedup: an object-shaped verdict can name the same neighbor twice
                if conn_id in valid_ids and conn_id not in seen_conn:
                    seen_conn.add(conn_id)
                    connections.append(conn_id)
            if connections:
                # unidirectional, as upstream: only the new note gains links
                ops.append(
                    MemoryOp(
                        op=OpType.LINK,
                        target_type="notes",
                        target_id=note.id,
                        payload={"links": connections},
                    )
                )
            # Evolution refines the NEW note's own tags in a SECOND op. This
            # deliberately mirrors upstream's post-add ``tags_to_update`` write
            # (audit P1-5) — it is not accidental redundancy, so keep it split
            # from the ADD above rather than folding the tags into it: the op
            # stream then matches upstream's add-then-evolve two phases.
            # Applied UNCONDITIONALLY, including [] — the plain edition's
            # ``note.tags = new_tags`` (memory_layer.py:834-836) has no guard,
            # and its strict schema requires the key but permits an empty
            # array; a [] wipe is auditable in the op log. Variant: the robust
            # edition guards emptiness only (memory_layer_robust.py:506-507).
            # A round-11 guard here (``if new_tags and new_tags != note.tags``)
            # also suppressed no-op verdicts, skewing evolution-log op counts
            # vs upstream (round-12 finding 2).
            new_tags = [str(t) for t in evolution_verdict.get("new_note_tags") or []]
            refreshed_self = Note(
                content=note.content,
                id=note.id,
                keywords=note.keywords,
                tags=new_tags,
                context=note.context,
            )
            ops.append(
                MemoryOp(
                    op=OpType.UPDATE,
                    target_type="notes",
                    target_id=note.id,
                    payload={
                        "tags": new_tags,
                        "embedding_text": refreshed_self.embedding_text(),
                    },
                )
            )

        if "update_neighbor" in actions:
            for upd in evolution_verdict.get("neighbor_updates", []) or []:
                note_id = _verdict_id(upd)
                if note_id not in valid_ids:
                    continue
                # A bare-id entry names a neighbor without saying how to rewrite
                # it, so there is nothing to apply. Skip rather than emit an
                # UPDATE that rewrites the neighbor's values to themselves — a
                # no-op op would still inflate the evolution-log op counts that
                # the repro experiments compare against upstream.
                if not isinstance(upd, dict):
                    continue
                old = by_id[note_id]
                new_context = str(upd.get("new_context") or old.get("context", ""))
                new_tags = [str(t) for t in (upd.get("new_tags") or old.get("tags", []))]
                refreshed = Note(
                    content=old.get("content", ""),
                    id=note_id,
                    context=new_context,
                    tags=new_tags,
                    keywords=old.get("keywords", []),
                )
                ops.append(
                    MemoryOp(
                        op=OpType.UPDATE,
                        target_type="notes",
                        target_id=note_id,
                        payload={
                            "context": new_context,
                            "tags": new_tags,
                            "embedding_text": refreshed.embedding_text(),
                        },
                    )
                )
        return ops

    def _add_op(self, note: Note, talk_time: str) -> MemoryOp:
        return MemoryOp(
            op=OpType.ADD,
            target_type="notes",
            target_id=note.id,
            payload={
                "id": note.id,
                "content": note.content,
                "keywords": note.keywords,
                "tags": note.tags,
                "context": note.context,
                "links": note.links,
                "source_episode_ids": note.source_episode_ids,
                "timestamp": talk_time,
                "embedding_text": note.embedding_text(),
            },
        )
