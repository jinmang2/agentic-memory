"""SQLite doc store: episodes + generic derived items + FTS5 + evolution log.

Single-file source of truth for the ``lite`` profile (docs/03 §5.2).
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from agmem.capabilities.requires import Requires
from agmem.core.ops import MemoryOp
from agmem.core.types import Episode

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_episodes_ns ON episodes(namespace);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    content, id UNINDEXED, namespace UNINDEXED
);

CREATE TABLE IF NOT EXISTS items (
    id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    namespace TEXT NOT NULL,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (id, memory_type)
);
CREATE INDEX IF NOT EXISTS idx_items_type_ns ON items(memory_type, namespace);

CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
    content, item_id UNINDEXED, memory_type UNINDEXED, namespace UNINDEXED
);

CREATE TABLE IF NOT EXISTS evolution_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    op TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    actor TEXT NOT NULL,
    t_transaction TEXT NOT NULL
);
"""

_FTS_TOKEN = re.compile(r"\w+", re.UNICODE)


def _fts_query(query: str) -> str:
    """Sanitize free text into a safe FTS5 OR-query."""
    tokens = _FTS_TOKEN.findall(query)
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else '""'


class SqliteDocStore:
    requires = Requires()  # stdlib only — always available

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = str(path) if path is not None else ":memory:"
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.RLock()
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    # -- episodes -----------------------------------------------------------

    def add_episode(self, ep: Episode) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO episodes (id, namespace, role, content, timestamp, meta)"
                " VALUES (?,?,?,?,?,?)",
                (
                    ep.id,
                    ep.namespace,
                    ep.role,
                    ep.content,
                    ep.timestamp.isoformat(),
                    json.dumps(ep.meta, ensure_ascii=False, default=str),
                ),
            )
            self._conn.execute("DELETE FROM episodes_fts WHERE id = ?", (ep.id,))
            self._conn.execute(
                "INSERT INTO episodes_fts (content, id, namespace) VALUES (?,?,?)",
                (ep.content, ep.id, ep.namespace),
            )

    def get_episodes(self, ids: list[str]) -> list[Episode]:
        if not ids:
            return []
        with self._lock:
            marks = ",".join("?" * len(ids))
            rows = self._conn.execute(
                f"SELECT id, namespace, role, content, timestamp, meta FROM episodes"
                f" WHERE id IN ({marks})",
                ids,
            ).fetchall()
        by_id = {
            r[0]: Episode(
                id=r[0],
                namespace=r[1],
                role=r[2],
                content=r[3],
                timestamp=datetime.fromisoformat(r[4]),
                meta=json.loads(r[5]),
            )
            for r in rows
        }
        return [by_id[i] for i in ids if i in by_id]  # preserve caller order

    def count_episodes(self, namespace: str | None = None) -> int:
        with self._lock:
            if namespace:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE namespace = ?", (namespace,)
                ).fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        return int(row[0])

    def search_lexical(
        self, query: str, k: int = 10, namespace: str | None = None
    ) -> list[tuple[str, float]]:
        """BM25 search; returns (episode_id, score) with higher = better."""
        sql = (
            "SELECT id, bm25(episodes_fts) AS rank FROM episodes_fts"
            " WHERE episodes_fts MATCH ?"
        )
        params: list[Any] = [_fts_query(query)]
        if namespace:
            sql += " AND namespace = ?"
            params.append(namespace)
        sql += " ORDER BY rank LIMIT ?"
        params.append(k)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        # bm25() is lower-is-better; negate so higher = better.
        return [(r[0], -float(r[1])) for r in rows]

    # -- generic derived items ----------------------------------------------

    def put_item(
        self, item_id: str, memory_type: str, namespace: str, data: dict[str, Any]
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO items (id, memory_type, namespace, data)"
                " VALUES (?,?,?,?)",
                (
                    item_id,
                    memory_type,
                    namespace,
                    json.dumps(data, ensure_ascii=False, default=str),
                ),
            )
            self._conn.execute(
                "DELETE FROM items_fts WHERE item_id = ? AND memory_type = ?",
                (item_id, memory_type),
            )
            content = str(data.get("content") or "")
            if content and not data.get("deleted"):
                self._conn.execute(
                    "INSERT INTO items_fts (content, item_id, memory_type,"
                    " namespace) VALUES (?,?,?,?)",
                    (content, item_id, memory_type, namespace),
                )

    def search_lexical_items(
        self, query: str, memory_type: str, k: int = 10, namespace: str | None = None
    ) -> list[tuple[str, float]]:
        """BM25 over derived items of one type (Zep hybrid search channel)."""
        sql = (
            "SELECT item_id, bm25(items_fts) AS rank FROM items_fts"
            " WHERE items_fts MATCH ? AND memory_type = ?"
        )
        params: list[Any] = [_fts_query(query), memory_type]
        if namespace:
            sql += " AND namespace = ?"
            params.append(namespace)
        sql += " ORDER BY rank LIMIT ?"
        params.append(k)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [(r[0], -float(r[1])) for r in rows]

    def list_items(
        self, memory_type: str, namespace: str | None = None
    ) -> list[dict[str, Any]]:
        """Full scan of one memory type (e.g. ACE's whole-playbook read)."""
        sql = "SELECT data FROM items WHERE memory_type = ?"
        args: list[Any] = [memory_type]
        if namespace:
            sql += " AND namespace = ?"
            args.append(namespace)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY updated_at", args).fetchall()
        out = [json.loads(r[0]) for r in rows]
        return [d for d in out if not d.get("deleted")]

    def get_items(self, ids: list[str], memory_type: str) -> list[dict[str, Any]]:
        if not ids:
            return []
        with self._lock:
            marks = ",".join("?" * len(ids))
            rows = self._conn.execute(
                f"SELECT id, data FROM items WHERE memory_type = ? AND id IN ({marks})",
                [memory_type, *ids],
            ).fetchall()
        by_id = {r[0]: json.loads(r[1]) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    # -- evolution log (append-only) -----------------------------------------

    def append(self, ops: list[MemoryOp]) -> None:
        if not ops:
            return
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO evolution_log (op, target_type, target_id, payload, actor, t_transaction)"
                " VALUES (?,?,?,?,?,?)",
                [
                    (
                        o.op.value,
                        o.target_type,
                        o.target_id,
                        json.dumps(o.payload, ensure_ascii=False, default=str),
                        o.actor,
                        o.t_transaction.isoformat(),
                    )
                    for o in ops
                ],
            )

    def tail(self, n: int = 20) -> list[MemoryOp]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT op, target_type, target_id, payload, actor, t_transaction"
                " FROM evolution_log ORDER BY seq DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [MemoryOp.from_row(*r) for r in reversed(rows)]

    def count(self) -> int:
        with self._lock:
            return int(
                self._conn.execute("SELECT COUNT(*) FROM evolution_log").fetchone()[0]
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def episode_to_dict(ep: Episode) -> dict[str, Any]:
    d = asdict(ep)
    d["timestamp"] = ep.timestamp.isoformat()
    return d
