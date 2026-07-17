"""LanceDB vector store (standard profile default).

Embedded columnar engine (Lance format, mmap-friendly — docs/03 §5
constraint). A path gives a persistent dataset directory; ``None``
falls back to a private temp directory that is removed on close().
Cosine metric with prefiltered namespace/memory_type predicates.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from agmem.capabilities.requires import Requires

_TABLE = "vectors"


class LanceDBVectorStore:
    requires = Requires(python_pkgs=("lancedb",))

    def __init__(self, path: str | Path | None = None, dim: int = 384) -> None:
        import lancedb
        import pyarrow as pa

        self.dim = dim
        self._ephemeral_dir = tempfile.mkdtemp(prefix="agmem-lance-") if path is None else None
        root = self._ephemeral_dir or str(path)
        Path(root).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(root)

        if _TABLE in self._db.table_names():
            self._tbl = self._db.open_table(_TABLE)
            stored = self._tbl.schema.field("vector").type.list_size
            if stored != dim:
                raise ValueError(
                    f"vector index was built with dim={stored}, got dim={dim} — "
                    "changing embedders requires rebuilding the collection (docs/03 §1.2)"
                )
        else:
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("namespace", pa.string()),
                pa.field("memory_type", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ])
            self._tbl = self._db.create_table(_TABLE, schema=schema)

    def add(self, item_id: str, embedding: list[float],
            memory_type: str = "episodic", namespace: str = "main") -> None:
        if len(embedding) != self.dim:
            raise ValueError(f"embedding dim {len(embedding)} != store dim {self.dim}")
        # upsert = delete + add (ids are uuid hex, safe to inline)
        self._tbl.delete(f"id = '{item_id}'")
        self._tbl.add([{"id": item_id, "namespace": namespace,
                        "memory_type": memory_type, "vector": embedding}])

    def search(self, embedding: list[float], k: int = 10,
               memory_type: str | None = None,
               namespace: str | None = None) -> list[tuple[str, float]]:
        q = self._tbl.search(embedding).metric("cosine")
        preds = []
        if namespace:
            preds.append(f"namespace = '{namespace}'")
        if memory_type:
            preds.append(f"memory_type = '{memory_type}'")
        if preds:
            q = q.where(" AND ".join(preds), prefilter=True)
        rows = q.limit(k).to_list()
        return [(r["id"], 1.0 - float(r["_distance"])) for r in rows]

    def get(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        id_list = ", ".join(f"'{i}'" for i in ids)  # uuid hex, safe to inline
        rows = (self._tbl.search().where(f"id IN ({id_list})")
                .select(["id", "vector"]).to_list())
        return {r["id"]: [float(x) for x in r["vector"]] for r in rows}

    def count(self) -> int:
        return int(self._tbl.count_rows())

    def persist(self) -> None:
        pass  # Lance datasets are durable per write

    def close(self) -> None:
        if self._ephemeral_dir:
            shutil.rmtree(self._ephemeral_dir, ignore_errors=True)
