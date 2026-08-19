"""Store protocols. All adapters for a slot implement the same interface."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agmem.core.ops import MemoryOp
from agmem.core.types import Episode


@runtime_checkable
class DocStore(Protocol):
    """Source of truth for raw episodes, derived items, and the op log."""

    def add_episode(self, episode: Episode) -> None: ...
    def get_episodes(self, ids: list[str]) -> list[Episode]: ...
    def count_episodes(self, namespace: str | None = None) -> int: ...

    def list_episodes(self, namespace: str | None = None) -> list[Episode]:
        """Every raw episode, OLDEST-FIRST, optionally scoped to one namespace.

        Ordering is part of the contract, not an implementation detail: callers take the tail as
        "most recent" (`hooks/recall.py`, `AgenticMemory`'s episode-mention frontier), so a store
        returning insertion order or an unordered scan silently serves the wrong slice.

        Declared here because it was missing and the omission cost something. Three call sites
        already depended on it while only `SqliteDocStore` implemented it, so a Postgres-backed run
        wrote a memory snapshot with no episodic rows at all — and because the snapshot writer
        guards the call with `getattr`, the loss presented as an artifact that simply had no
        transcript in it rather than as an error. `tests/test_store_contract.py` now pins the
        surface across every implementation.
        """
        ...

    def search_lexical(
        self, query: str, k: int = 10, namespace: str | None = None
    ) -> list[tuple[str, float]]:
        """(episode_id, score), highest-relevance-first. Engine-internal score
        conventions (e.g. bm25's lower-is-better) must be normalized so the
        caller always sees higher = more relevant, matching `VectorStore.search`."""
        ...

    def search_lexical_items(
        self, query: str, memory_type: str, k: int = 10, namespace: str | None = None
    ) -> list[tuple[str, float]]:
        """(item_id, score) over derived items of one ``memory_type``, highest-relevance-first,
        with the same normalized score convention as ``search_lexical``. The lexical channel of
        every Zep hybrid recipe: ``retrieval/pipeline.py`` calls it unguarded for each type in
        ``lexical_types``. Declared for the same reason as ``list_episodes`` — both backends
        implement it, and an undeclared method a pipeline calls is exactly the gap that cost a
        Postgres run its transcript."""
        ...

    def put_item(
        self, item_id: str, memory_type: str, namespace: str, data: dict[str, Any]
    ) -> None: ...
    def get_items(self, ids: list[str], memory_type: str) -> list[dict[str, Any]]: ...
    def list_items(
        self, memory_type: str, namespace: str | None = None
    ) -> list[dict[str, Any]]: ...

    # EvolutionLog
    def append(self, ops: list[MemoryOp]) -> None:
        """Append to the durable op log; callers must log before applying an
        op to derived state (docs/12 §3.2 log-first invariant)."""
        ...

    def tail(self, n: int = 20) -> list[MemoryOp]:
        """Most recent ``n`` ops, oldest-first within the returned slice."""
        ...

    def count(self) -> int: ...
    def ops_since(
        self, seq: int, target_type: str | None = None, limit: int = 10000
    ) -> list[tuple[int, MemoryOp]]:
        """(seq, op) pairs with seq strictly greater than the argument, seq
        ascending. A returned batch truncated at ``limit`` is not the full
        remainder — the caller must page by re-calling with the last seq seen."""
        ...

    def last_seq(self) -> int: ...

    def close(self) -> None: ...


@runtime_checkable
class VectorStore(Protocol):
    """Embedding index. ``score`` is cosine similarity (higher = closer).

    Rows are keyed by the BARE item id, in all five backends — while the doc stores key items
    on ``(id, memory_type)`` (the invariant `retrieval/pipeline.py`'s bundle dedup states: the
    same id under two types is two distinct items). An id shared across two memory types would
    therefore silently overwrite the other type's vector on ``add``, and the facade's
    INVALIDATE/DELETE paths (`memory.py`) delete by bare id and would take both. Deferred
    deliberately rather than rekeyed: every embedded id is a uuid4 (collision-free in
    practice), the only fixed non-uuid id (MemoryOS's ``memoryos:user_profile``) is
    doc-store-only via explicitly-null ``embedding_text`` and never gets a vector row, and
    rekeying to ``(id, memory_type)`` would orphan every vector store already persisted on
    disk."""

    dim: int

    def add(
        self,
        item_id: str,
        embedding: list[float],
        memory_type: str = "episodic",
        namespace: str = "main",
    ) -> None: ...
    def search(
        self,
        embedding: list[float],
        k: int = 10,
        memory_type: str | None = None,
        namespace: str | None = None,
    ) -> list[tuple[str, float]]:
        """(item_id, score) ranked by similarity, highest first, filtered to
        ``memory_type``/``namespace`` when given; unfiltered when omitted."""
        ...

    def get(self, ids: list[str]) -> dict[str, list[float]]: ...
    def delete(self, ids: list[str]) -> None: ...
    def count(self) -> int: ...
    def persist(self) -> None:
        """Flush to durable storage; a no-op for engines that write through
        on every call (implementations must document which they are)."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class GraphStore(Protocol):
    """Entity/edge/community store behind the graph read paths (Zep/Graphiti, G-Memory).

    **Declared late, and the reason belongs in the contract.** The three backends —
    `SqliteGraphStore`, `KuzuGraphStore`, `Neo4jGraphStore` — already carry an identical
    seventeen-method surface, so nothing was broken when this was written; what was missing was
    anything that *required* it. `DocStore` spent months without declaring `list_episodes` while
    three call sites used it and one backend lacked it, and the loss showed up as a memory snapshot
    that silently held no transcript. This protocol exists so a fourth graph backend cannot repeat
    that, and `tests/test_store_contract.py` is what makes the requirement bite.

    Two invariants that the method list alone does not convey:

    - **Edges are invalidated, never deleted.** `invalidate_edge` stamps `invalid_at` AND
      `expired_at` (bi-temporal: when the fact stopped holding / when the system learned it), and
      `counts` reports edges INCLUDING invalidated ones. Zep's temporal claim is this behaviour,
      and a backend that deleted instead would still pass every signature check while erasing the
      mechanism — so `active_only` defaults differ on purpose per method and are part of the
      contract, not per-engine taste. ACTIVE means `invalid_at IS NULL AND expired_at IS NULL`:
      the two are only ever stamped together, so a backend filtering on `invalid_at` alone read
      identically on every stored measurement — but the one-field filter and the one-field reset
      were still two copies of the semantics drifting from this contract, unified 2026-08-19.
    - **Communities are derived state, so `remove_community` is a hard delete** — membership
      included. That asymmetry with edges is deliberate: a community can be recomputed from the
      graph, an edge's history cannot be recovered once dropped.
    """

    def upsert_node(
        self,
        node_id: str,
        namespace: str,
        name: str,
        summary: str = "",
        entity_type: str = "Entity",
    ) -> None:
        """Upsert keyed on ``node_id``, NOT on ``name`` — reusing an id replaces that row."""
        ...

    def find_node_by_name(self, name: str, namespace: str) -> dict | None:
        """Case-insensitive exact match within ``namespace``; first match, or None."""
        ...

    def upsert_edge(
        self,
        edge_id: str,
        namespace: str,
        src: str,
        dst: str,
        predicate: str,
        content: str,
        valid_at: str | None = None,
    ) -> None:
        """Full-row replace by ``edge_id``; reusing an id RESETS ``invalid_at``/``expired_at``
        (revalidation) — call ``invalidate_edge`` afterward if that is not intended."""
        ...

    def edges_between(
        self, src: str, dst: str, namespace: str, active_only: bool = True
    ) -> list[dict]:
        """Either direction between the pair. ``active_only`` defaults True."""
        ...

    def invalidate_edge(self, edge_id: str, t_invalid: str) -> None:
        """Stamp an edge no-longer-true as of ``t_invalid``. Never deletes."""
        ...

    def edges_for_nodes(
        self, node_ids: list[str], namespace: str, active_only: bool = True
    ) -> list[dict]:
        """Edges incident to any of the nodes — the graph-recall expansion frontier."""
        ...

    def neighbors(
        self, node_id: str, namespace: str, hops: int = 1, direction: str = "both"
    ) -> list[dict]:
        """Nodes within ``hops`` steps over ACTIVE edges only."""
        ...

    def entity_projection(
        self, namespace: str, active_only: bool = False
    ) -> dict[str, dict[str, int]]:
        """node id -> {neighbour id: edge count}, the label-propagation input.

        Note the default: ``active_only=False`` here where the read paths use True, because
        community structure is built over the whole history rather than the currently-true slice.
        """
        ...

    def upsert_community(
        self, community_id: str, namespace: str, name: str, summary: str = ""
    ) -> None: ...
    def set_community_members(self, community_id: str, namespace: str, node_ids: list[str]) -> None:
        """Replace membership wholesale — not a merge."""
        ...

    def community_of_node(self, node_id: str, namespace: str) -> dict | None: ...
    def neighbor_communities(self, node_id: str, namespace: str) -> list[dict]:
        """One row per (relating edge, community) path, so a caller can weight by edge."""
        ...

    def communities(self, namespace: str) -> list[dict]: ...
    def community_members(self, community_id: str) -> list[str]: ...
    def remove_community(self, community_id: str) -> None:
        """Hard delete, membership included — communities are derived state."""
        ...

    def counts(self) -> dict[str, int]:
        """Row counts per table. The edge count INCLUDES invalidated edges."""
        ...

    def close(self) -> None: ...
