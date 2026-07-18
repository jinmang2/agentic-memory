"""Neo4j graph store — the full-profile server engine (Zep/Graphiti's own).

Activates only when the neo4j bolt service is detected (capability
Requires); connection settings come from AGMEM_NEO4J_URI /
AGMEM_NEO4J_USER / AGMEM_NEO4J_PASSWORD (defaults bolt://localhost:7687,
neo4j/neo4j). Mirrors SqliteGraphStore's API and bi-temporal semantics.
The ``path`` argument of the uniform store contract is ignored (server
engines have no data path).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from agmem.capabilities.requires import Requires


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Neo4jGraphStore:
    requires = Requires(python_pkgs=("neo4j",), services=("neo4j",))

    def __init__(self, path: str | Path | None = None) -> None:  # path unused
        from neo4j import GraphDatabase

        uri = os.environ.get("AGMEM_NEO4J_URI", "bolt://localhost:7687")
        auth = (
            os.environ.get("AGMEM_NEO4J_USER", "neo4j"),
            os.environ.get("AGMEM_NEO4J_PASSWORD", "neo4j"),
        )
        self._driver = GraphDatabase.driver(uri, auth=auth)
        with self._driver.session() as s:
            s.run(
                "CREATE CONSTRAINT agmem_entity_id IF NOT EXISTS"
                " FOR (n:Entity) REQUIRE n.id IS UNIQUE"
            )

    def _run(self, query: str, **params) -> list[dict]:
        with self._driver.session() as s:
            return [dict(r) for r in s.run(query, **params)]

    def upsert_node(
        self,
        node_id: str,
        namespace: str,
        name: str,
        summary: str = "",
        entity_type: str = "Entity",
    ) -> None:
        self._run(
            "MERGE (n:Entity {id: $id})"
            " ON CREATE SET n.namespace=$ns, n.created_at=$now"
            " SET n.name=$name, n.summary=$summary, n.entity_type=$etype",
            id=node_id,
            ns=namespace,
            name=name,
            summary=summary,
            etype=entity_type,
            now=_now(),
        )

    def find_node_by_name(self, name: str, namespace: str) -> dict | None:
        rows = self._run(
            "MATCH (n:Entity) WHERE n.namespace=$ns AND"
            " toLower(n.name)=toLower($name)"
            " RETURN n.id AS id, n.namespace AS namespace, n.name AS name,"
            " n.summary AS summary, n.entity_type AS entity_type,"
            " n.created_at AS created_at",
            ns=namespace,
            name=name,
        )
        return rows[0] if rows else None

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
        self._run(
            "MATCH (a:Entity {id: $src}), (b:Entity {id: $dst})"
            " MERGE (a)-[e:RELATES {id: $eid}]->(b)"
            " ON CREATE SET e.namespace=$ns, e.created_at=$now"
            " SET e.predicate=$pred, e.content=$content, e.valid_at=$valid",
            src=src,
            dst=dst,
            eid=edge_id,
            ns=namespace,
            pred=predicate,
            content=content,
            valid=valid_at,
            now=_now(),
        )

    _EDGE_RETURN = (
        " RETURN e.id AS id, e.namespace AS namespace,"
        " a.id AS src, b.id AS dst, e.predicate AS predicate,"
        " e.content AS content, e.created_at AS created_at,"
        " e.expired_at AS expired_at, e.valid_at AS valid_at,"
        " e.invalid_at AS invalid_at"
    )

    def edges_between(
        self, src: str, dst: str, namespace: str, active_only: bool = True
    ) -> list[dict]:
        active = " AND e.invalid_at IS NULL" if active_only else ""
        return self._run(
            "MATCH (a:Entity)-[e:RELATES]-(b:Entity)"
            " WHERE e.namespace=$ns AND a.id=$src AND b.id=$dst" + active + self._EDGE_RETURN,
            ns=namespace,
            src=src,
            dst=dst,
        )

    def invalidate_edge(self, edge_id: str, t_invalid: str) -> None:
        self._run(
            "MATCH ()-[e:RELATES {id: $eid}]->()"
            " SET e.invalid_at=$t, e.expired_at=coalesce(e.expired_at, $now)",
            eid=edge_id,
            t=t_invalid,
            now=_now(),
        )

    def edges_for_nodes(
        self, node_ids: list[str], namespace: str, active_only: bool = True
    ) -> list[dict]:
        if not node_ids:
            return []
        active = " AND e.invalid_at IS NULL" if active_only else ""
        return self._run(
            "MATCH (a:Entity)-[e:RELATES]->(b:Entity)"
            " WHERE e.namespace=$ns AND (a.id IN $ids OR b.id IN $ids)"
            + active
            + self._EDGE_RETURN,
            ns=namespace,
            ids=node_ids,
        )

    def neighbors(self, node_id: str, namespace: str, hops: int = 1) -> list[dict]:
        return self._run(
            f"MATCH p = (a:Entity {{id: $id}})-[:RELATES*1..{int(hops)}]-(n:Entity)"
            " WHERE all(r IN relationships(p) WHERE r.invalid_at IS NULL"
            " AND r.namespace=$ns)"
            " RETURN DISTINCT n.id AS id, n.namespace AS namespace,"
            " n.name AS name, n.summary AS summary,"
            " n.entity_type AS entity_type, n.created_at AS created_at",
            id=node_id,
            ns=namespace,
        )

    def counts(self) -> dict[str, int]:
        n = self._run("MATCH (n:Entity) RETURN count(n) AS c")[0]["c"]
        e = self._run("MATCH ()-[e:RELATES]->() RETURN count(e) AS c")[0]["c"]
        return {"nodes": int(n), "edges": int(e)}

    def close(self) -> None:
        self._driver.close()
