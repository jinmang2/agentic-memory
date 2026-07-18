"""sqlite-vec vector store (lite profile default when the extension loads).

Uses a vec0 virtual table with cosine distance plus a rowid<->item map
table carrying namespace/memory_type for post-filtering.
"""

from __future__ import annotations

import sqlite3
import struct
import threading
from pathlib import Path

from agmem.capabilities.requires import Requires


def _serialize(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class SqliteVecStore:
    requires = Requires(python_pkgs=("sqlite_vec",))

    def __init__(self, path: str | Path | None = None, dim: int = 384) -> None:
        import sqlite_vec  # gated by requires

        self.path = str(path) if path is not None else ":memory:"
        self.dim = dim
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._lock = threading.RLock()
        with self._lock, self._conn:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0("
                f"embedding float[{dim}] distance_metric=cosine)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS vec_map ("
                " rowid INTEGER PRIMARY KEY AUTOINCREMENT,"
                " item_id TEXT NOT NULL UNIQUE,"
                " namespace TEXT NOT NULL,"
                " memory_type TEXT NOT NULL)"
            )
            self._conn.execute("CREATE TABLE IF NOT EXISTS vec_meta (k TEXT PRIMARY KEY, v TEXT)")
            stored = self._conn.execute("SELECT v FROM vec_meta WHERE k='dim'").fetchone()
            if stored and int(stored[0]) != dim:
                raise ValueError(
                    f"vector index was built with dim={stored[0]}, got dim={dim} — "
                    "changing embedders requires rebuilding the collection (docs/03 §1.2)"
                )
            self._conn.execute("INSERT OR REPLACE INTO vec_meta VALUES ('dim', ?)", (str(dim),))

    def add(
        self,
        item_id: str,
        embedding: list[float],
        memory_type: str = "episodic",
        namespace: str = "main",
    ) -> None:
        if len(embedding) != self.dim:
            raise ValueError(f"embedding dim {len(embedding)} != store dim {self.dim}")
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO vec_map (item_id, namespace, memory_type) VALUES (?,?,?)"
                " ON CONFLICT(item_id) DO UPDATE SET namespace=excluded.namespace,"
                " memory_type=excluded.memory_type RETURNING rowid",
                (item_id, namespace, memory_type),
            )
            rowid = cur.fetchone()[0]
            self._conn.execute("DELETE FROM vectors WHERE rowid = ?", (rowid,))
            self._conn.execute(
                "INSERT INTO vectors (rowid, embedding) VALUES (?,?)",
                (rowid, _serialize(embedding)),
            )

    def search(
        self,
        embedding: list[float],
        k: int = 10,
        memory_type: str | None = None,
        namespace: str | None = None,
    ) -> list[tuple[str, float]]:
        # Over-fetch then post-filter on map attributes.
        fetch_k = k * 4 if (memory_type or namespace) else k
        with self._lock:
            rows = self._conn.execute(
                "SELECT v.rowid, v.distance, m.item_id, m.namespace, m.memory_type"
                " FROM vectors v JOIN vec_map m ON m.rowid = v.rowid"
                " WHERE v.embedding MATCH ? AND v.k = ?",
                (_serialize(embedding), fetch_k),
            ).fetchall()
        out: list[tuple[str, float]] = []
        for _rowid, dist, item_id, ns, mt in rows:
            if namespace and ns != namespace:
                continue
            if memory_type and mt != memory_type:
                continue
            out.append((item_id, 1.0 - float(dist)))  # cosine distance -> similarity
            if len(out) >= k:
                break
        return out

    def get(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        with self._lock:
            marks = ",".join("?" * len(ids))
            rows = self._conn.execute(
                f"SELECT m.item_id, v.embedding FROM vec_map m"
                f" JOIN vectors v ON v.rowid = m.rowid WHERE m.item_id IN ({marks})",
                ids,
            ).fetchall()
        return {item_id: list(struct.unpack(f"{self.dim}f", blob)) for item_id, blob in rows}

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        with self._lock, self._conn:
            rows = self._conn.execute(
                f"SELECT rowid FROM vec_map WHERE item_id IN ({marks})", ids
            ).fetchall()
            for (rowid,) in rows:
                self._conn.execute("DELETE FROM vectors WHERE rowid = ?", (rowid,))
            self._conn.execute(f"DELETE FROM vec_map WHERE item_id IN ({marks})", ids)

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM vec_map").fetchone()[0])

    def persist(self) -> None:
        pass  # SQLite persists on commit

    def close(self) -> None:
        with self._lock:
            self._conn.close()
