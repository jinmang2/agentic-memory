"""Kuzu graph store — embedded real graph engine (Cypher, no server).

The same-family lightweight substitute for Neo4j under the real-stacks
policy (docs/03 §5.2): a genuine property-graph engine with Cypher,
runnable in-process (path-backed or in-memory). Mirrors SqliteGraphStore's
API and bi-temporal edge semantics (invalidate records both invalid_at and
expired_at; edges are never deleted).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from agmem.capabilities.requires import Requires

_NODE_COLS = ("id", "namespace", "name", "summary", "entity_type", "created_at")
_EDGE_COLS = (
    "id",
    "namespace",
    "src",
    "dst",
    "predicate",
    "content",
    "created_at",
    "expired_at",
    "valid_at",
    "invalid_at",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class KuzuGraphStore:
    requires = Requires(python_pkgs=("kuzu",))

    def __init__(self, path: str | Path | None = None) -> None:
        import kuzu

        if path is None:
            self._db = kuzu.Database(":memory:")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._db = kuzu.Database(str(path))
        self._conn = kuzu.Connection(self._db)
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS Entity("
                "id STRING PRIMARY KEY, namespace STRING, name STRING,"
                " summary STRING, entity_type STRING, created_at STRING)"
            )
            self._conn.execute(
                "CREATE REL TABLE IF NOT EXISTS RELATES(FROM Entity TO Entity,"
                " id STRING, namespace STRING, predicate STRING, content STRING,"
                " created_at STRING, expired_at STRING, valid_at STRING,"
                " invalid_at STRING)"
            )

    def _rows(self, result, cols: tuple[str, ...]) -> list[dict]:
        out = []
        while result.has_next():
            out.append(dict(zip(cols, result.get_next())))
        return out

    # -- nodes ----------------------------------------------------------------

    def upsert_node(
        self,
        node_id: str,
        namespace: str,
        name: str,
        summary: str = "",
        entity_type: str = "Entity",
    ) -> None:
        with self._lock:
            self._conn.execute(
                "MERGE (n:Entity {id: $id})"
                " ON CREATE SET n.namespace=$ns, n.name=$name, n.summary=$summary,"
                "  n.entity_type=$etype, n.created_at=$now"
                " ON MATCH SET n.name=$name, n.summary=$summary,"
                "  n.entity_type=$etype",
                {
                    "id": node_id,
                    "ns": namespace,
                    "name": name,
                    "summary": summary,
                    "etype": entity_type,
                    "now": _now(),
                },
            )

    def find_node_by_name(self, name: str, namespace: str) -> dict | None:
        with self._lock:
            res = self._conn.execute(
                "MATCH (n:Entity) WHERE n.namespace=$ns AND"
                " lower(n.name)=lower($name) RETURN n.id, n.namespace, n.name,"
                " n.summary, n.entity_type, n.created_at",
                {"ns": namespace, "name": name},
            )
            rows = self._rows(res, _NODE_COLS)
        return rows[0] if rows else None

    # -- edges ----------------------------------------------------------------

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
        with self._lock:
            self._conn.execute(
                "MATCH (a:Entity {id: $src}), (b:Entity {id: $dst})"
                " MERGE (a)-[e:RELATES {id: $eid}]->(b)"
                " ON CREATE SET e.namespace=$ns, e.predicate=$pred,"
                "  e.content=$content, e.valid_at=$valid, e.created_at=$now"
                " ON MATCH SET e.predicate=$pred, e.content=$content,"
                "  e.valid_at=$valid",
                {
                    "src": src,
                    "dst": dst,
                    "eid": edge_id,
                    "ns": namespace,
                    "pred": predicate,
                    "content": content,
                    "valid": valid_at,
                    "now": _now(),
                },
            )

    def edges_between(
        self, src: str, dst: str, namespace: str, active_only: bool = True
    ) -> list[dict]:
        active = " AND e.invalid_at IS NULL" if active_only else ""
        with self._lock:
            res = self._conn.execute(
                "MATCH (a:Entity)-[e:RELATES]-(b:Entity)"
                " WHERE e.namespace=$ns AND a.id=$src AND b.id=$dst"
                + active
                + " RETURN DISTINCT e.id, e.namespace, a.id, b.id, e.predicate,"
                " e.content, e.created_at, e.expired_at, e.valid_at, e.invalid_at",
                {"ns": namespace, "src": src, "dst": dst},
            )
            return self._rows(res, _EDGE_COLS)

    def invalidate_edge(self, edge_id: str, t_invalid: str) -> None:
        with self._lock:
            self._conn.execute(
                "MATCH ()-[e:RELATES]->() WHERE e.id=$eid"
                " SET e.invalid_at=$t, e.expired_at=coalesce(e.expired_at, $now)",
                {"eid": edge_id, "t": t_invalid, "now": _now()},
            )

    def edges_for_nodes(
        self, node_ids: list[str], namespace: str, active_only: bool = True
    ) -> list[dict]:
        if not node_ids:
            return []
        active = " AND e.invalid_at IS NULL" if active_only else ""
        with self._lock:
            res = self._conn.execute(
                "MATCH (a:Entity)-[e:RELATES]->(b:Entity)"
                " WHERE e.namespace=$ns AND (list_contains($ids, a.id)"
                " OR list_contains($ids, b.id))"
                + active
                + " RETURN DISTINCT e.id, e.namespace, a.id, b.id, e.predicate,"
                " e.content, e.created_at, e.expired_at, e.valid_at, e.invalid_at",
                {"ns": namespace, "ids": node_ids},
            )
            return self._rows(res, _EDGE_COLS)

    def neighbors(self, node_id: str, namespace: str, hops: int = 1) -> list[dict]:
        # hop-by-hop frontier walk over ACTIVE edges — version-stable across
        # Kuzu releases (recursive-rel predicate syntax varies)
        seen, frontier = {node_id}, [node_id]
        out: list[dict] = []
        for _ in range(max(hops, 0)):
            if not frontier:
                break
            edges = self.edges_for_nodes(frontier, namespace)
            nxt: list[str] = []
            for e in edges:
                for nid in (e["src"], e["dst"]):
                    if nid not in seen:
                        seen.add(nid)
                        nxt.append(nid)
            frontier = nxt
            if nxt:
                with self._lock:
                    res = self._conn.execute(
                        "MATCH (n:Entity) WHERE list_contains($ids, n.id)"
                        " RETURN n.id, n.namespace, n.name, n.summary,"
                        " n.entity_type, n.created_at",
                        {"ids": nxt},
                    )
                    out.extend(self._rows(res, _NODE_COLS))
        return out

    def counts(self) -> dict[str, int]:
        with self._lock:
            n = self._conn.execute("MATCH (n:Entity) RETURN count(n)").get_next()[0]
            e = self._conn.execute("MATCH ()-[e:RELATES]->() RETURN count(e)").get_next()[0]
        return {"nodes": int(n), "edges": int(e)}

    def close(self) -> None:
        self._conn.close()
        self._db.close()
