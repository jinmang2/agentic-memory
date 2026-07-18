"""Qdrant vector store (full profile; Nemori's upstream engine).

Runs qdrant-client in LOCAL mode (embedded, no server/Docker): a path
gives a persistent on-disk instance, ``None`` gives ":memory:". Cosine
distance as the collection metric — scores come back as similarity
directly (higher = closer), matching the VectorStore protocol.

Our item ids are arbitrary strings; Qdrant point ids must be uuid/int,
so points are keyed by uuid5(item_id) and the original id rides in the
payload next to namespace/memory_type (both filterable).
"""

from __future__ import annotations

import uuid
from pathlib import Path

from agmem.capabilities.requires import Requires

_COLLECTION = "agmem"


def _point_id(item_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, item_id))


class QdrantVectorStore:
    """Cosine-metric `VectorStore` over an embedded (local-mode) Qdrant
    collection — see module docstring for the uuid5 point-id mapping."""

    requires = Requires(python_pkgs=("qdrant_client",))

    def __init__(self, path: str | Path | None = None, dim: int = 384) -> None:
        """`path=None` opens an in-memory client; opening an existing
        collection with a different `dim` raises `ValueError` (docs/03 §1.2)."""
        from qdrant_client import QdrantClient
        from qdrant_client import models as qm

        self._qm = qm
        self.dim = dim
        if path is None:
            self._client = QdrantClient(location=":memory:")
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))

        if self._client.collection_exists(_COLLECTION):
            stored = self._client.get_collection(_COLLECTION).config.params.vectors.size
            if stored != dim:
                raise ValueError(
                    f"vector index was built with dim={stored}, got dim={dim} — "
                    "changing embedders requires rebuilding the collection (docs/03 §1.2)"
                )
        else:
            self._client.create_collection(
                _COLLECTION,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )

    def add(
        self,
        item_id: str,
        embedding: list[float],
        memory_type: str = "episodic",
        namespace: str = "main",
    ) -> None:
        """Upsert by `item_id` (mapped to a uuid5 point id); raises `ValueError`
        on an embedding/store dim mismatch."""
        if len(embedding) != self.dim:
            raise ValueError(f"embedding dim {len(embedding)} != store dim {self.dim}")
        self._client.upsert(
            _COLLECTION,
            points=[
                self._qm.PointStruct(
                    id=_point_id(item_id),
                    vector=embedding,
                    payload={
                        "item_id": item_id,
                        "namespace": namespace,
                        "memory_type": memory_type,
                    },
                )
            ],
        )

    def search(
        self,
        embedding: list[float],
        k: int = 10,
        memory_type: str | None = None,
        namespace: str | None = None,
    ) -> list[tuple[str, float]]:
        """`namespace`/`memory_type` combine with AND only (no OR support);
        Qdrant's cosine score is returned as-is (already higher = closer)."""
        qm = self._qm
        must = []
        if namespace:
            must.append(qm.FieldCondition(key="namespace", match=qm.MatchValue(value=namespace)))
        if memory_type:
            must.append(
                qm.FieldCondition(key="memory_type", match=qm.MatchValue(value=memory_type))
            )
        hits = self._client.query_points(
            _COLLECTION,
            query=embedding,
            limit=k,
            query_filter=qm.Filter(must=must) if must else None,
        ).points
        return [(h.payload["item_id"], float(h.score)) for h in hits]

    def get(self, ids: list[str]) -> dict[str, list[float]]:
        """Ids not present in the collection are silently omitted from the result."""
        if not ids:
            return {}
        points = self._client.retrieve(
            _COLLECTION, ids=[_point_id(i) for i in ids], with_vectors=True
        )
        return {p.payload["item_id"]: list(p.vector) for p in points}

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self._client.delete(
            _COLLECTION,
            points_selector=self._qm.PointIdsList(points=[_point_id(i) for i in ids]),
        )

    def count(self) -> int:
        return int(self._client.count(_COLLECTION).count)

    def persist(self) -> None:
        """No-op: local mode writes through immediately when `path`-backed
        (and is inherently non-durable when `path=None`)."""
        pass  # local mode persists on write when path-backed

    def close(self) -> None:
        self._client.close()
