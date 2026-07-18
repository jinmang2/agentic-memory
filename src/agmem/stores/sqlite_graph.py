"""SQLite graph store: entity nodes + bi-temporal edges (Zep design).

Edges are never deleted — ``invalidate_edge`` records both temporal axes
(``invalid_at`` = when the fact stopped holding, T; ``expired_at`` = when
the system learned it, T', as upstream edge_operations does) so "what was
true then" and "what we believed then" stay queryable
(docs/research/zep-graphiti.md §A.2). k-hop via recursive CTE.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from agmem.capabilities.requires import Requires

_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    id TEXT PRIMARY KEY, namespace TEXT NOT NULL, name TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '', entity_type TEXT NOT NULL DEFAULT 'Entity',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_nodes_ns_name ON graph_nodes(namespace, name COLLATE NOCASE);
CREATE TABLE IF NOT EXISTS graph_edges (
    id TEXT PRIMARY KEY, namespace TEXT NOT NULL,
    src TEXT NOT NULL, dst TEXT NOT NULL,
    predicate TEXT NOT NULL, content TEXT NOT NULL,
    valid_at TEXT, invalid_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    expired_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_edges_pair ON graph_edges(namespace, src, dst);
"""

_ACTIVE = "invalid_at IS NULL AND expired_at IS NULL"


class SqliteGraphStore:
    """SQL recursive-CTE emulation of the shared graph-store contract used by

    Zep/G-Memory organizers (no formal `Protocol` collects it — Kuzu/Neo4j
    implement the same methods). Nodes are upserted by id; edges are never
    hard-deleted, only invalidated (see module docstring for the bi-temporal
    ``invalid_at``/``expired_at`` axes)."""

    requires = Requires()

    def __init__(self, path: str | Path | None = None) -> None:
        """``path=None`` opens an in-memory, non-persistent database."""
        self.path = str(path) if path is not None else ":memory:"
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def upsert_node(
        self,
        node_id: str,
        namespace: str,
        name: str,
        summary: str = "",
        entity_type: str = "Entity",
    ) -> None:
        """Upsert is keyed on `node_id`, not `name` — reusing an existing id
        overwrites name/summary/entity_type in place; a new name with a new id
        creates a second node even if it resolves to the same real-world entity
        (dedup is the organizer's job via `find_node_by_name`/embedding search)."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO graph_nodes (id, namespace, name, summary, entity_type)"
                " VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET"
                " name=excluded.name, summary=excluded.summary,"
                " entity_type=excluded.entity_type",
                (node_id, namespace, name, summary, entity_type),
            )

    def find_node_by_name(self, name: str, namespace: str) -> dict | None:
        """Case-insensitive exact match within `namespace`; returns the first
        match only — callers needing all same-name nodes must query differently."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM graph_nodes WHERE namespace=? AND name=? COLLATE NOCASE",
                (namespace, name),
            ).fetchone()
        return dict(row) if row else None

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
        """Full-row replace by `edge_id` — reusing an id resets `invalid_at`/
        `expired_at` to unset even if the prior row had been invalidated;
        call `invalidate_edge` afterward if that's not intended."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO graph_edges"
                " (id, namespace, src, dst, predicate, content, valid_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (edge_id, namespace, src, dst, predicate, content, valid_at),
            )

    def edges_between(
        self, src: str, dst: str, namespace: str, active_only: bool = True
    ) -> list[dict]:
        """Matches edges in either direction between `src`/`dst`; set
        `active_only=False` to also see invalidated edges."""
        sql = (
            "SELECT * FROM graph_edges WHERE namespace=? AND"
            " ((src=? AND dst=?) OR (src=? AND dst=?))"
        )
        if active_only:
            sql += f" AND {_ACTIVE}"
        with self._lock:
            rows = self._conn.execute(sql, (namespace, src, dst, dst, src)).fetchall()
        return [dict(r) for r in rows]

    def invalidate_edge(self, edge_id: str, t_invalid: str) -> None:
        """Marks an edge no-longer-true as of ``t_invalid`` without deleting
        it (bi-temporal: ``expired_at`` records when the system learned this,
        distinct from ``t_invalid``'s "was true until" axis)."""
        # round-5 ⑨: expired_at (T' axis) must be stamped too, preserving an
        # earlier value if the edge was already expired (upstream semantics).
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE graph_edges SET invalid_at=?, expired_at=COALESCE("
                "expired_at, strftime('%Y-%m-%dT%H:%M:%fZ','now')) WHERE id=?",
                (t_invalid, edge_id),
            )

    def neighbors(self, node_id: str, namespace: str, hops: int = 1) -> list[dict]:
        """Nodes reachable within ``hops`` steps over ACTIVE edges only
        (invalidated edges are not walked), excluding ``node_id`` itself."""
        with self._lock:
            rows = self._conn.execute(
                f"""
                WITH RECURSIVE walk(id, depth) AS (
                    SELECT ?, 0
                    UNION
                    SELECT CASE WHEN e.src = w.id THEN e.dst ELSE e.src END, w.depth + 1
                    FROM graph_edges e JOIN walk w
                      ON (e.src = w.id OR e.dst = w.id)
                    WHERE w.depth < ? AND e.namespace = ? AND {_ACTIVE}
                )
                SELECT DISTINCT n.* FROM walk w JOIN graph_nodes n ON n.id = w.id
                WHERE w.id != ?""",
                (node_id, hops, namespace, node_id),
            ).fetchall()
        return [dict(r) for r in rows]

    def edges_for_nodes(
        self, node_ids: list[str], namespace: str, active_only: bool = True
    ) -> list[dict]:
        """Edges incident to any of the nodes (GraphRecall expansion)."""
        if not node_ids:
            return []
        marks = ",".join("?" * len(node_ids))
        sql = (
            f"SELECT * FROM graph_edges WHERE namespace=? AND"
            f" (src IN ({marks}) OR dst IN ({marks}))"
        )
        if active_only:
            sql += f" AND {_ACTIVE}"
        with self._lock:
            rows = self._conn.execute(sql, (namespace, *node_ids, *node_ids)).fetchall()
        return [dict(r) for r in rows]

    def counts(self) -> dict[str, int]:
        """Edge count includes invalidated edges (never deleted, so `edges` is
        a lifetime total, not an active-only count)."""
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
            e = self._conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        return {"nodes": int(n), "edges": int(e)}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
