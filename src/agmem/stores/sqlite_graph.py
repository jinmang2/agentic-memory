"""SQLite graph store: entity nodes + bi-temporal edges + communities (Zep design).

Edges are never deleted — ``invalidate_edge`` records both temporal axes
(``invalid_at`` = when the fact stopped holding, T; ``expired_at`` = when
the system learned it, T', as upstream edge_operations does) so "what was
true then" and "what we believed then" stay queryable
(docs/research/zep-graphiti.md §A.2). k-hop via recursive CTE.

Communities are the third subgraph of the paper's three (§2.2.4): a
community node with a name and summary, plus ``HAS_MEMBER`` links to the
entity nodes it covers. They are derived state — label propagation over
the entity subgraph rebuilds them — so unlike edges they ARE deletable
(upstream ``remove_communities`` then rebuilds).
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
CREATE TABLE IF NOT EXISTS graph_communities (
    id TEXT PRIMARY KEY, namespace TEXT NOT NULL, name TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS graph_community_members (
    community_id TEXT NOT NULL, node_id TEXT NOT NULL, namespace TEXT NOT NULL,
    PRIMARY KEY (community_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_members_node ON graph_community_members(namespace, node_id);
"""

_ACTIVE = "invalid_at IS NULL AND expired_at IS NULL"


class SqliteGraphStore:
    """SQL recursive-CTE emulation of the shared graph-store contract used by
    Zep/G-Memory organizers (`stores.base.GraphStore`; Kuzu/Neo4j implement
    the same methods). Nodes are upserted by id; edges are never
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

    def neighbors(
        self, node_id: str, namespace: str, hops: int = 1, direction: str = "both"
    ) -> list[dict]:
        """Nodes reachable within ``hops`` steps over ACTIVE edges only
        (invalidated edges are not walked), excluding ``node_id`` itself.

        ``direction="both"`` (the generic-store default) treats edges as
        undirected; ``"out"`` follows edges src->dst only — the mode Zep's
        φ_bfs channel uses, because upstream's BFS queries walk
        ``-[:RELATES_TO|MENTIONS*1..n]->`` (outgoing only,
        search_utils.py:448+/790+; round-12 finding 12). Every other caller
        (GraphRecall, node-distance, communities) keeps ``"both"``."""
        if direction == "out":
            step = "SELECT e.dst, w.depth + 1 FROM graph_edges e JOIN walk w ON e.src = w.id"
        else:
            step = (
                "SELECT CASE WHEN e.src = w.id THEN e.dst ELSE e.src END, w.depth + 1"
                " FROM graph_edges e JOIN walk w ON (e.src = w.id OR e.dst = w.id)"
            )
        with self._lock:
            rows = self._conn.execute(
                f"""
                WITH RECURSIVE walk(id, depth) AS (
                    SELECT ?, 0
                    UNION
                    {step}
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

    # -- communities (paper §2.2.4) -------------------------------------------

    def entity_projection(
        self, namespace: str, active_only: bool = False
    ) -> dict[str, dict[str, int]]:
        """node id -> {neighbor id: edge count}, the input label propagation
        needs (upstream ``get_community_clusters`` builds exactly this).

        EVERY node in the namespace is a key, isolated ones included with an
        empty map — upstream seeds the projection from ``get_by_group_ids``, so
        an entity with no relations still gets its own singleton community
        rather than disappearing.

        ``active_only=False`` by default because upstream's projection query
        (``MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity)``) applies no temporal
        filter: an invalidated fact still counts as evidence that two entities
        belong together, which is the opposite of what recall wants but is what
        the published clustering does. Direction is collapsed — one undirected
        count per pair, as the upstream ``count(e)`` per neighbour is."""
        with self._lock:
            node_rows = self._conn.execute(
                "SELECT id FROM graph_nodes WHERE namespace=?", (namespace,)
            ).fetchall()
            sql = "SELECT src, dst, COUNT(*) AS c FROM graph_edges WHERE namespace=?"
            if active_only:
                sql += f" AND {_ACTIVE}"
            edge_rows = self._conn.execute(sql + " GROUP BY src, dst", (namespace,)).fetchall()
        projection: dict[str, dict[str, int]] = {str(r["id"]): {} for r in node_rows}
        for row in edge_rows:
            src, dst, count = str(row["src"]), str(row["dst"]), int(row["c"])
            if src in projection and dst in projection:
                projection[src][dst] = projection[src].get(dst, 0) + count
                projection[dst][src] = projection[dst].get(src, 0) + count
        return projection

    def upsert_community(
        self, community_id: str, namespace: str, name: str, summary: str = ""
    ) -> None:
        """Upsert by id; name/summary are replaced in place (an incremental
        member addition rewrites both, as upstream ``update_community`` does)."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO graph_communities (id, namespace, name, summary)"
                " VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET"
                " name=excluded.name, summary=excluded.summary",
                (community_id, namespace, name, summary),
            )

    def set_community_members(self, community_id: str, namespace: str, node_ids: list[str]) -> None:
        """Replace the community's membership wholesale — the applied op carries
        the full member list, so a replayed log converges instead of
        accumulating stale members."""
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM graph_community_members WHERE community_id=?", (community_id,)
            )
            self._conn.executemany(
                "INSERT OR IGNORE INTO graph_community_members"
                " (community_id, node_id, namespace) VALUES (?,?,?)",
                [(community_id, node_id, namespace) for node_id in node_ids],
            )

    def community_of_node(self, node_id: str, namespace: str) -> dict | None:
        """The community this entity already belongs to, or None. First match
        only — membership is 1:1 in this design, as upstream's
        ``determine_entity_community`` assumes when it reads ``records[0]``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT c.* FROM graph_communities c"
                " JOIN graph_community_members m ON m.community_id = c.id"
                " WHERE m.node_id=? AND c.namespace=?",
                (node_id, namespace),
            ).fetchone()
        return dict(row) if row else None

    def neighbor_communities(self, node_id: str, namespace: str) -> list[dict]:
        """One row per (relating edge, community) path from this node, so the
        caller can take a plurality over repeats — the shape upstream's
        ``(c:Community)-[:HAS_MEMBER]->(m:Entity)-[:RELATES_TO]-(n)`` match
        returns, where a neighbour joined by three edges votes three times.
        No temporal filter, matching that query."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.* FROM graph_edges e"
                " JOIN graph_community_members m"
                "   ON m.node_id = CASE WHEN e.src=? THEN e.dst ELSE e.src END"
                " JOIN graph_communities c ON c.id = m.community_id"
                " WHERE e.namespace=? AND (e.src=? OR e.dst=?) AND m.node_id != ?",
                (node_id, namespace, node_id, node_id, node_id),
            ).fetchall()
        return [dict(r) for r in rows]

    def communities(self, namespace: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM graph_communities WHERE namespace=? ORDER BY created_at, id",
                (namespace,),
            ).fetchall()
        return [dict(r) for r in rows]

    def community_members(self, community_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT node_id FROM graph_community_members WHERE community_id=? ORDER BY node_id",
                (community_id,),
            ).fetchall()
        return [str(r["node_id"]) for r in rows]

    def remove_community(self, community_id: str) -> None:
        """Hard delete, membership included. Communities are derived state, so
        unlike edges they carry no history worth keeping — upstream's
        ``build_communities`` opens with ``remove_communities`` for the same
        reason."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM graph_communities WHERE id=?", (community_id,))
            self._conn.execute(
                "DELETE FROM graph_community_members WHERE community_id=?", (community_id,)
            )

    def counts(self) -> dict[str, int]:
        """Edge count includes invalidated edges (never deleted, so `edges` is
        a lifetime total, not an active-only count)."""
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
            e = self._conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
            c = self._conn.execute("SELECT COUNT(*) FROM graph_communities").fetchone()[0]
        return {"nodes": int(n), "edges": int(e), "communities": int(c)}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
