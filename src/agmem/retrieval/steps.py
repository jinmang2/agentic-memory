"""Read-path post-hydration steps, one plugin per methodology.

These restore upstream read-path semantics the deep fidelity audit found
missing (docs/research/fidelity-deep-audit.md). They used to be an
``if memory_type == ...`` chain inside ``RetrievalPipeline.search``, which meant
their knobs — the A-Mem link cap and Nemori's source-attachment ``r``, both
flagged as deliberate upstream deviations — were constructor defaults no caller
could reach. As plugins they are registered per memory type and configured from
``AgmemConfig``, so those deviations are finally ablatable.

Keyed on the MEMORY TYPE, never on which organizer is active: items written
straight to a store must get the same treatment as organizer-written ones
(tests/test_pipeline_p0.py relies on this, and so does any offline replay).

The uniform contract is ``run(hits, ctx) -> list[ScoredItem]`` returning the
final list for that type, which covers all three shapes the original branches
had: append (link expansion, graph recall), replace (experiences), and
mutate-then-return (source attachment).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agmem.core.types import ScoredItem
from agmem.stores.base import DocStore


@dataclass
class ReadContext:
    """Read-only handles a step needs. ``bundle_ids`` are the ids already
    selected for earlier memory types in the same search, so an expansion step
    can avoid re-serving them."""

    doc_store: DocStore
    namespace: str | None
    graph_store: Any | None = None
    bundle_ids: set[str] = field(default_factory=set)


class ReadStep:
    """Base: subclasses transform one memory type's hydrated hits."""

    def run(self, hits: list[ScoredItem], ctx: ReadContext) -> list[ScoredItem]:
        return hits


class LinkExpansion(ReadStep):
    """A-Mem 1-hop: pull linked neighbor notes of retrieved notes.

    Links are unidirectional as upstream. Cap semantics deviate: upstream caps
    PER HIT (agiresearch k per hit; WujiangXu k+1 via an off-by-one), so eval
    k=10 can pull ~100 link neighbors — WujiangXu #16/#21 show even upstream
    considers this ambiguous. We use one global cap (default 5); neighbors score
    just below their parent. Keep this deviation in result caveats when
    comparing multi-hop."""

    def __init__(self, cap: int = 5) -> None:
        self.cap = cap

    def run(self, hits: list[ScoredItem], ctx: ReadContext) -> list[ScoredItem]:
        seen = {s.item.data["id"] for s in hits}
        wanted: list[tuple[str, float]] = []
        for s in sorted(hits, key=lambda s: s.score, reverse=True):
            for linked_id in s.item.data.get("links", []):
                if linked_id not in seen and len(wanted) < self.cap:
                    seen.add(linked_id)
                    wanted.append((linked_id, s.score * 0.9))
        if not wanted:
            return hits
        by_id = dict(wanted)
        out = list(hits)
        for data in ctx.doc_store.get_items(list(by_id), "notes"):
            out.append(
                ScoredItem(
                    item=_DictItem(data),
                    memory_type="notes",
                    score=by_id.get(data.get("id"), 0.0),
                    provenance=data.get("source_episode_ids", []),
                )
            )
        return out


class AttachSources(ReadStep):
    """Nemori r=2: top-r episodes carry their raw source messages, rendered as
    ``role: content`` lines as upstream search.py does."""

    def __init__(self, top_r: int = 2) -> None:
        self.top_r = top_r

    def run(self, hits: list[ScoredItem], ctx: ReadContext) -> list[ScoredItem]:
        for s in sorted(hits, key=lambda s: s.score, reverse=True)[: self.top_r]:
            source_ids = s.item.data.get("source_episode_ids", [])
            if source_ids:
                episodes = ctx.doc_store.get_episodes(source_ids)
                s.item.data["_source_messages"] = [
                    f"{episode.role}: {episode.content}" for episode in episodes
                ]
        return hits


class ExpandExperiences(ReadStep):
    """ReasoningBank experience mode: an experience hit is REPLACED by its
    member strategy items (upstream injects the top-1 experience's items, never
    the record itself). An experience with no surviving items yields nothing —
    upstream's miss -> no-injection semantics."""

    def run(self, hits: list[ScoredItem], ctx: ReadContext) -> list[ScoredItem]:
        out: list[ScoredItem] = []
        for s in hits:
            ids = s.item.data.get("item_ids", [])
            for data in ctx.doc_store.get_items(ids, "strategies"):
                if data.get("deleted"):
                    continue
                out.append(
                    ScoredItem(
                        item=_DictItem(data),
                        memory_type="strategies",
                        score=s.score,
                        provenance=data.get("source_episode_ids", []),
                    )
                )
        return out


class GraphRecall(ReadStep):
    """Zep GraphRecall (round-5 ④, minimal form): retrieved entity nodes pull
    their incident ACTIVE edges; the edges' fact items join the bundle (deduped
    against already-selected facts), scored just below their entity.

    No-op when no graph store is wired.

    Every pulled fact gets the SAME ``base`` score, so the served order used to
    follow ``edges_for_nodes``'s row order, which the graph backends do not
    guarantee. The expanded facts are therefore emitted in a content-derived
    order (valid_at, content). Safe to change: docs/09 excludes zep_graph from
    measurement (skeleton grade), so there was no stable ordering to preserve.
    The A-Mem link expansion deliberately keeps its store-order emission — that
    one does back measured numbers."""

    def __init__(self, cap: int = 10) -> None:
        self.cap = cap

    def run(self, hits: list[ScoredItem], ctx: ReadContext) -> list[ScoredItem]:
        if ctx.graph_store is None:
            return hits
        seen = set(ctx.bundle_ids) | {s.item.data.get("id") for s in hits}
        node_ids = [s.item.data.get("id") for s in hits]
        edges = ctx.graph_store.edges_for_nodes([n for n in node_ids if n], ctx.namespace or "main")
        wanted: dict[str, float] = {}
        base = max((s.score for s in hits), default=0.0) * 0.9
        for e in edges:
            edge_id = e.get("id")
            if edge_id and edge_id not in seen and len(wanted) < self.cap:
                seen.add(edge_id)
                wanted[edge_id] = base
        out = list(hits)
        # Stable key must be content-derived, NOT the id: fact ids are random
        # uuids, so ordering by them would just move the nondeterminism instead
        # of removing it.
        pulled = sorted(
            ctx.doc_store.get_items(list(wanted), "facts"),
            key=lambda d: (str(d.get("valid_at") or ""), str(d.get("content") or "")),
        )
        for data in pulled:
            if data.get("deleted"):
                continue
            out.append(
                ScoredItem(
                    item=_DictItem(data),
                    memory_type="facts",
                    score=wanted.get(data.get("id"), 0.0),
                    provenance=data.get("source_episode_ids", []),
                )
            )
        return out


def default_read_steps(
    link_expansion_cap: int = 5,
    attach_sources_top_r: int = 2,
    graph_expansion_cap: int = 10,
) -> dict[str, ReadStep]:
    """The methodology-faithful default registry, memory type -> step.

    A cap of 0 drops that step entirely, preserving the falsy-cap disable the
    original ``if memory_type == "notes" and self.link_expansion_cap`` guards
    gave."""
    steps: dict[str, ReadStep] = {"experiences": ExpandExperiences()}
    if link_expansion_cap:
        steps["notes"] = LinkExpansion(link_expansion_cap)
    if attach_sources_top_r:
        steps["episodes"] = AttachSources(attach_sources_top_r)
    if graph_expansion_cap:
        steps["entities"] = GraphRecall(graph_expansion_cap)
    return steps


class _DictItem:
    """Lightweight wrapper so derived items render uniformly in a bundle.

    Render exposes methodology metadata (audit P0-4): note context/tags, item
    timestamps, and attached source messages."""

    def __init__(self, data: dict) -> None:
        """`data` is kept by reference, not copied — callers like `AttachSources`
        mutate it in place (e.g. to inject `_source_messages`)."""
        self.data = data
        self.content = data.get("content", "")

    def render(self) -> str:
        """Multi-line text injected verbatim into the LLM context — order and
        section labels here are part of the read-path prompt contract."""
        parts: list[str] = []
        title = self.data.get("title")
        head = f"{title}: " if title else ""
        # Bi-temporal facts render their validity range (Zep's context
        # template: "FACT (Date range: from - to)") so invalidated facts
        # are visibly historical instead of passing as current (round-5 X2).
        if self.data.get("valid_at") or self.data.get("invalid_at"):
            stamp = (
                f" (Date range: {self.data.get('valid_at') or 'unknown'}"
                f" - {self.data.get('invalid_at') or 'present'})"
            )
        else:
            ts = self.data.get("timestamp")
            stamp = f" ({ts})" if ts else ""
        parts.append(f"{head}{self.content}{stamp}")
        # ReasoningBank items carry when-to-apply guidance in description;
        # upstream injects the full item markdown (round-5 X3).
        if self.data.get("description"):
            parts.append(f"description: {self.data['description']}")
        if self.data.get("context"):
            parts.append(f"context: {self.data['context']}")
        if self.data.get("tags"):
            parts.append(f"tags: {', '.join(map(str, self.data['tags']))}")
        if self.data.get("_source_messages"):
            src = "\n".join(f"  - {m}" for m in self.data["_source_messages"])
            parts.append(f"Source Messages:\n{src}")
        return "\n".join(parts)
