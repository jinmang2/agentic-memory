"""PostgreSQL doc store — Nemori's upstream engine, embedded via pgserver.

``pgserver`` ships real PostgreSQL binaries as a pip wheel and
self-manages a per-datadir server instance (no system install, no root,
no Docker) — the same-family lightweight substitute mandated by the
real-stacks policy (docs/03 §5.2). Lexical search is genuine Postgres
tsvector/ts_rank (Nemori's tsvector channel), not an emulation.

Mirrors SqliteDocStore's contract: episodes + lexical, derived items with
per-type FTS, append-only evolution log. ``path`` is the pgserver data
directory; ``None`` uses a private temp dir removed on close().
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
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
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
);
CREATE INDEX IF NOT EXISTS idx_episodes_ns ON episodes(namespace);
CREATE INDEX IF NOT EXISTS idx_episodes_tsv ON episodes USING GIN (content_tsv);

CREATE TABLE IF NOT EXISTS items (
    id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    namespace TEXT NOT NULL,
    data TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    PRIMARY KEY (id, memory_type)
);
CREATE INDEX IF NOT EXISTS idx_items_type_ns ON items(memory_type, namespace);
CREATE INDEX IF NOT EXISTS idx_items_tsv ON items USING GIN (content_tsv);

CREATE TABLE IF NOT EXISTS evolution_log (
    seq BIGSERIAL PRIMARY KEY,
    op TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    actor TEXT NOT NULL,
    t_transaction TEXT NOT NULL
);
"""


class PostgresDocStore:
    """`DocStore` backed by a real, embedded PostgreSQL — see module
    docstring for the pgserver/tsvector rationale and the contract this
    mirrors from `SqliteDocStore`."""

    requires = Requires(python_pkgs=("pgserver", "psycopg"))

    def __init__(self, path: str | Path | None = None) -> None:
        """``path=None`` provisions a private temp datadir that `close()`
        deletes; a path reuses/creates a persistent datadir that survives
        close()."""
        import pgserver
        import psycopg

        self._ephemeral_dir = tempfile.mkdtemp(prefix="agmem-pg-") if path is None else None
        datadir = self._ephemeral_dir or str(path)
        Path(datadir).mkdir(parents=True, exist_ok=True)
        self._server = pgserver.get_server(datadir)
        self._conn = psycopg.connect(self._server.get_uri(), autocommit=True)
        self._lock = threading.RLock()
        with self._lock, self._conn.cursor() as cur:
            cur.execute(_SCHEMA)

    # -- episodes -------------------------------------------------------------

    def add_episode(self, episode: Episode) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO episodes (id, namespace, role, content, timestamp, meta)"
                " VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO UPDATE SET"
                " namespace=EXCLUDED.namespace, role=EXCLUDED.role,"
                " content=EXCLUDED.content, timestamp=EXCLUDED.timestamp,"
                " meta=EXCLUDED.meta",
                (
                    episode.id,
                    episode.namespace,
                    episode.role,
                    episode.content,
                    episode.timestamp.isoformat(),
                    json.dumps(episode.meta, ensure_ascii=False, default=str),
                ),
            )

    def get_episodes(self, ids: list[str]) -> list[Episode]:
        if not ids:
            return []
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, namespace, role, content, timestamp, meta"
                " FROM episodes WHERE id = ANY(%s)",
                (ids,),
            )
            rows = cur.fetchall()
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
        sql = "SELECT COUNT(*) FROM episodes"
        args: tuple = ()
        if namespace:
            sql += " WHERE namespace = %s"
            args = (namespace,)
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, args)
            return int(cur.fetchone()[0])

    def search_lexical(
        self, query: str, k: int = 10, namespace: str | None = None
    ) -> list[tuple[str, float]]:
        """tsvector search; returns (episode_id, score), higher = better."""
        sql = (
            "SELECT id, ts_rank(content_tsv, q) AS rank FROM episodes,"
            " websearch_to_tsquery('simple', %s) q WHERE content_tsv @@ q"
        )
        args: list[Any] = [query]
        if namespace:
            sql += " AND namespace = %s"
            args.append(namespace)
        sql += " ORDER BY rank DESC LIMIT %s"
        args.append(k)
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, args)
            return [(r[0], float(r[1])) for r in cur.fetchall()]

    # -- generic derived items ------------------------------------------------

    def put_item(
        self, item_id: str, memory_type: str, namespace: str, data: dict[str, Any]
    ) -> None:
        content = "" if data.get("deleted") else str(data.get("content") or "")
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO items (id, memory_type, namespace, data, content)"
                " VALUES (%s,%s,%s,%s,%s)"
                " ON CONFLICT (id, memory_type) DO UPDATE SET"
                " namespace=EXCLUDED.namespace, data=EXCLUDED.data,"
                " content=EXCLUDED.content, updated_at=now()",
                (
                    item_id,
                    memory_type,
                    namespace,
                    json.dumps(data, ensure_ascii=False, default=str),
                    content,
                ),
            )

    def get_items(self, ids: list[str], memory_type: str) -> list[dict[str, Any]]:
        if not ids:
            return []
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, data FROM items WHERE memory_type = %s AND id = ANY(%s)",
                (memory_type, ids),
            )
            rows = cur.fetchall()
        by_id = {r[0]: json.loads(r[1]) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def list_items(self, memory_type: str, namespace: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT data FROM items WHERE memory_type = %s"
        args: list[Any] = [memory_type]
        if namespace:
            sql += " AND namespace = %s"
            args.append(namespace)
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql + " ORDER BY updated_at", args)
            rows = cur.fetchall()
        out = [json.loads(r[0]) for r in rows]
        return [d for d in out if not d.get("deleted")]

    def search_lexical_items(
        self, query: str, memory_type: str, k: int = 10, namespace: str | None = None
    ) -> list[tuple[str, float]]:
        """(item_id, score), highest-relevance-first (ts_rank is already
        higher-is-better, unlike bm25 — no sign flip needed here)."""
        sql = (
            "SELECT id, ts_rank(content_tsv, q) AS rank FROM items,"
            " websearch_to_tsquery('simple', %s) q"
            " WHERE content_tsv @@ q AND memory_type = %s AND content <> ''"
        )
        args: list[Any] = [query, memory_type]
        if namespace:
            sql += " AND namespace = %s"
            args.append(namespace)
        sql += " ORDER BY rank DESC LIMIT %s"
        args.append(k)
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, args)
            return [(r[0], float(r[1])) for r in cur.fetchall()]

    # -- evolution log --------------------------------------------------------

    def append(self, ops: list[MemoryOp]) -> None:
        if not ops:
            return
        with self._lock, self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO evolution_log (op, target_type, target_id, payload,"
                " actor, t_transaction) VALUES (%s,%s,%s,%s,%s,%s)",
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
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "SELECT op, target_type, target_id, payload, actor, t_transaction"
                " FROM evolution_log ORDER BY seq DESC LIMIT %s",
                (n,),
            )
            rows = cur.fetchall()
        return [MemoryOp.from_row(*r) for r in reversed(rows)]

    def count(self) -> int:
        with self._lock, self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM evolution_log")
            return int(cur.fetchone()[0])

    def ops_since(
        self, seq: int, target_type: str | None = None, limit: int = 10000
    ) -> list[tuple[int, MemoryOp]]:
        """Read log entries after ``seq`` in order — the consolidate cursor surface."""
        sql = (
            "SELECT seq, op, target_type, target_id, payload, actor, t_transaction"
            " FROM evolution_log WHERE seq > %s"
        )
        args: list[Any] = [seq]
        if target_type is not None:
            sql += " AND target_type = %s"
            args.append(target_type)
        sql += " ORDER BY seq ASC LIMIT %s"
        args.append(limit)
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, args)
            rows = cur.fetchall()
        return [(int(r[0]), MemoryOp.from_row(*r[1:])) for r in rows]

    def last_seq(self) -> int:
        with self._lock, self._conn.cursor() as cur:
            cur.execute("SELECT MAX(seq) FROM evolution_log")
            row = cur.fetchone()
        return int(row[0] or 0)

    def close(self) -> None:
        """Shuts down the embedded pgserver and, for an ephemeral datadir,
        deletes it — irreversible; do not call while other stores share
        the same datadir."""
        with self._lock:
            self._conn.close()
        self._server.cleanup()
        if self._ephemeral_dir:
            shutil.rmtree(self._ephemeral_dir, ignore_errors=True)
