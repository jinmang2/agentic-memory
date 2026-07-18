"""ChromaDB vector store (A-Mem's upstream engine, corrected).

The agiresearch A-Mem edition creates its collection without
``hnsw:space`` (Chroma defaults to L2 — issue #24) and keeps it
in-memory (notes lost on exit). This adapter is the faithful-but-fixed
counterpart: explicit cosine space and a PersistentClient when a path
is given (``None`` -> ephemeral, for tests/in-memory namespaces).
"""

from __future__ import annotations

import uuid
from pathlib import Path

from agmem.capabilities.requires import Requires

_COLLECTION = "agmem"


class ChromaVectorStore:
    """Cosine-space `VectorStore` over a Chroma collection (see module docstring
    for the upstream #24 space-metric fix this adapter carries).

    `path=None` gives an `EphemeralClient` with a private, uuid-suffixed
    collection (dropped in `close()`); a `path` gives a `PersistentClient`
    reusing one fixed collection name across instances.
    """

    requires = Requires(python_pkgs=("chromadb",))

    def __init__(self, path: str | Path | None = None, dim: int = 384) -> None:
        """Open/create the collection; raises `ValueError` if it was already
        built with a different `dim` (embedder swaps require a rebuild,
        docs/03 §1.2)."""
        import chromadb

        self.dim = dim
        if path is None:
            # EphemeralClient shares one in-process system: a fixed collection
            # name would leak state between store instances, so ephemeral
            # stores get a private collection (dropped in close()).
            self._client = chromadb.EphemeralClient()
            name = f"{_COLLECTION}-{uuid.uuid4().hex[:8]}"
            self._ephemeral_name = name
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path))
            name = _COLLECTION
            self._ephemeral_name = None
        # hnsw:space=cosine is the upstream-#24 fix; dim rides in metadata
        # because Chroma only discovers dimensionality on first insert.
        self._col = self._client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine", "agmem_dim": str(dim)}
        )
        stored = (self._col.metadata or {}).get("agmem_dim")
        if stored and int(stored) != dim:
            raise ValueError(
                f"vector index was built with dim={stored}, got dim={dim} — "
                "changing embedders requires rebuilding the collection (docs/03 §1.2)"
            )

    def add(
        self,
        item_id: str,
        embedding: list[float],
        memory_type: str = "episodic",
        namespace: str = "main",
    ) -> None:
        """Upsert by `item_id`; raises `ValueError` on an embedding/store dim mismatch."""
        if len(embedding) != self.dim:
            raise ValueError(f"embedding dim {len(embedding)} != store dim {self.dim}")
        self._col.upsert(
            ids=[item_id],
            embeddings=[embedding],
            metadatas=[{"namespace": namespace, "memory_type": memory_type}],
        )

    def search(
        self,
        embedding: list[float],
        k: int = 10,
        memory_type: str | None = None,
        namespace: str | None = None,
    ) -> list[tuple[str, float]]:
        """Converts Chroma's cosine *distance* to similarity (`1 - dist`) so results
        match the `VectorStore` contract of higher = closer."""
        clauses = []
        if namespace:
            clauses.append({"namespace": namespace})
        if memory_type:
            clauses.append({"memory_type": memory_type})
        where = clauses[0] if len(clauses) == 1 else ({"$and": clauses} if clauses else None)
        n = min(k, max(self._col.count(), 1))
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where,
            include=["distances"],
        )
        return [
            (item_id, 1.0 - float(dist))  # cosine distance -> similarity
            for item_id, dist in zip(res["ids"][0], res["distances"][0])
        ]

    def get(self, ids: list[str]) -> dict[str, list[float]]:
        """Ids not present in the collection are silently omitted from the result."""
        if not ids:
            return {}
        res = self._col.get(ids=ids, include=["embeddings"])
        return {
            item_id: [float(x) for x in embedding]
            for item_id, embedding in zip(res["ids"], res["embeddings"])
        }

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self._col.delete(ids=ids)

    def count(self) -> int:
        return int(self._col.count())

    def persist(self) -> None:
        """No-op: `PersistentClient` writes through on every call, unlike stores
        that batch to memory and need an explicit flush."""
        pass  # PersistentClient persists per write

    def close(self) -> None:
        """Drops the private ephemeral collection; persistent stores need no cleanup."""
        if self._ephemeral_name:
            try:
                self._client.delete_collection(self._ephemeral_name)
            except Exception:  # already gone — nothing to release
                pass
