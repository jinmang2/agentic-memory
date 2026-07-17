from agmem.stores.base import DocStore, VectorStore
from agmem.stores.chroma_vec import ChromaVectorStore
from agmem.stores.kuzu_graph import KuzuGraphStore
from agmem.stores.lance_vec import LanceDBVectorStore
from agmem.stores.neo4j_graph import Neo4jGraphStore
from agmem.stores.postgres_doc import PostgresDocStore
from agmem.stores.qdrant_vec import QdrantVectorStore
from agmem.stores.sqlite_doc import SqliteDocStore
from agmem.stores.sqlite_graph import SqliteGraphStore
from agmem.stores.sqlite_vec import SqliteVecStore

# Preference order for the resolver: heaviest/most capable first (docs/01
# REGISTRY order). NumpyVectorStore is deliberately NOT a candidate: naive
# in-python fallbacks are banned as runtime defaults (docs/03 §5 정책) —
# it survives only as a protocol reference inside the test suite.
VECTOR_STORE_CANDIDATES: list[type] = [
    QdrantVectorStore,
    ChromaVectorStore,
    LanceDBVectorStore,
    SqliteVecStore,
]

# Neo4j (server, Zep/Graphiti's own engine) when detected; Kuzu is the
# embedded real graph engine (Neo4j's lightweight substitute per docs/03
# §5.2); SQLite recursive-CTE emulation is the last resort.
GRAPH_STORE_CANDIDATES: list[type] = [
    Neo4jGraphStore,
    KuzuGraphStore,
    SqliteGraphStore,
]

# PostgresDocStore = embedded real PostgreSQL via pgserver (Nemori's
# upstream engine, tsvector lexical included); SQLite is the single-file
# default.
DOC_STORE_CANDIDATES: list[type] = [
    PostgresDocStore,
    SqliteDocStore,
]

__all__ = [
    "DocStore",
    "VectorStore",
    "SqliteDocStore",
    "SqliteGraphStore",
    "SqliteVecStore",
    "LanceDBVectorStore",
    "QdrantVectorStore",
    "ChromaVectorStore",
    "KuzuGraphStore",
    "Neo4jGraphStore",
    "PostgresDocStore",
    "VECTOR_STORE_CANDIDATES",
    "GRAPH_STORE_CANDIDATES",
    "DOC_STORE_CANDIDATES",
]
