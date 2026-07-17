from agmem.stores.base import DocStore, VectorStore
from agmem.stores.chroma_vec import ChromaVectorStore
from agmem.stores.lance_vec import LanceDBVectorStore
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

__all__ = [
    "DocStore",
    "VectorStore",
    "SqliteDocStore",
    "SqliteGraphStore",
    "SqliteVecStore",
    "LanceDBVectorStore",
    "QdrantVectorStore",
    "ChromaVectorStore",
    "VECTOR_STORE_CANDIDATES",
]
