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
    def search_lexical(
        self, query: str, k: int = 10, namespace: str | None = None
    ) -> list[tuple[str, float]]:
        """(episode_id, score), highest-relevance-first. Engine-internal score
        conventions (e.g. bm25's lower-is-better) must be normalized so the
        caller always sees higher = more relevant, matching `VectorStore.search`."""
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
    """Embedding index. ``score`` is cosine similarity (higher = closer)."""

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
