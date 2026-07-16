from agmem.stores.base import DocStore, VectorStore
from agmem.stores.numpy_vec import NumpyVectorStore
from agmem.stores.sqlite_doc import SqliteDocStore
from agmem.stores.sqlite_vec import SqliteVecStore

# Preference order for the resolver: heaviest/most capable first.
VECTOR_STORE_CANDIDATES: list[type] = [SqliteVecStore, NumpyVectorStore]

__all__ = [
    "DocStore",
    "VectorStore",
    "SqliteDocStore",
    "SqliteVecStore",
    "NumpyVectorStore",
    "VECTOR_STORE_CANDIDATES",
]
