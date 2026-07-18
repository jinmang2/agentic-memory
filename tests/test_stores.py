import os
from importlib.util import find_spec

import pytest

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode
from agmem.embed.fake import FakeEmbedder
from agmem.stores.chroma_vec import ChromaVectorStore
from agmem.stores.lance_vec import LanceDBVectorStore
from agmem.stores.numpy_vec import NumpyVectorStore
from agmem.stores.qdrant_vec import QdrantVectorStore
from agmem.stores.sqlite_doc import SqliteDocStore
from agmem.stores.sqlite_vec import SqliteVecStore


def _param(cls, pkg: str | None = None):
    marks = ([] if pkg is None or find_spec(pkg)
             else [pytest.mark.skip(reason=f"{pkg} not installed")])
    return pytest.param(cls, id=cls.__name__, marks=marks)


@pytest.fixture
def doc():
    store = SqliteDocStore(":memory:")
    yield store
    store.close()


def test_episode_roundtrip(doc):
    ep = Episode(content="파리 여행 계획을 세우고 있어요", role="user", namespace="t")
    doc.add_episode(ep)
    got = doc.get_episodes([ep.id])[0]
    assert got.content == ep.content
    assert got.timestamp == ep.timestamp
    assert doc.count_episodes("t") == 1


def test_lexical_search_ranks_match_first(doc):
    a = Episode(content="I love hiking in the mountains", namespace="t")
    b = Episode(content="My favorite food is sushi and ramen", namespace="t")
    doc.add_episode(a)
    doc.add_episode(b)
    hits = doc.search_lexical("sushi food", namespace="t")
    assert hits and hits[0][0] == b.id


def test_lexical_search_handles_special_chars(doc):
    doc.add_episode(Episode(content="hello world", namespace="t"))
    # must not raise FTS5 syntax errors
    assert doc.search_lexical('what "is" (hello) AND -world?', namespace="t")


def test_evolution_log_append_only(doc):
    ops = [MemoryOp(op=OpType.ADD, target_type="notes", target_id="n1",
                    payload={"content": "x"}, actor="amem")]
    doc.append(ops)
    doc.append([MemoryOp(op=OpType.INVALIDATE, target_type="facts", target_id="f1")])
    assert doc.count() == 2
    tail = doc.tail(10)
    assert tail[0].op is OpType.ADD and tail[0].payload == {"content": "x"}
    assert tail[1].op is OpType.INVALIDATE


def test_ops_since_and_last_seq(doc):
    assert doc.last_seq() == 0
    doc.append(
        [
            MemoryOp(op=OpType.ADD, target_type="semantic", target_id="s1"),
            MemoryOp(op=OpType.ADD, target_type="episodes", target_id="e1"),
        ]
    )
    doc.append([MemoryOp(op=OpType.UPDATE, target_type="semantic", target_id="s1")])
    end = doc.last_seq()
    assert end == 3
    all_ops = doc.ops_since(0)
    assert [o.target_id for _, o in all_ops] == ["s1", "e1", "s1"]
    assert [s for s, _ in all_ops] == [1, 2, 3]
    sem = doc.ops_since(1, target_type="semantic")
    assert [(s, o.op) for s, o in sem] == [(3, OpType.UPDATE)]


def test_items_roundtrip(doc):
    doc.put_item("s1", "strategies", "t", {"id": "s1", "title": "T", "content": "C"})
    items = doc.get_items(["s1"], "strategies")
    assert items[0]["title"] == "T"


# NumpyVectorStore is a protocol reference only (not a runtime candidate —
# docs/03 §5); the rest are the real engines, skipped if not installed.
VEC_CLASSES = [
    _param(NumpyVectorStore),
    _param(SqliteVecStore, "sqlite_vec"),
    _param(LanceDBVectorStore, "lancedb"),
    _param(QdrantVectorStore, "qdrant_client"),
    _param(ChromaVectorStore, "chromadb"),
]


@pytest.mark.parametrize("vec_cls", VEC_CLASSES)
def test_vector_similarity_ordering(vec_cls):
    emb = FakeEmbedder(dim=64)
    store = vec_cls(None, dim=64)
    texts = {
        "e1": "hiking mountains trail backpack",
        "e2": "sushi ramen tokyo restaurant",
        "e3": "mountains hiking gear boots",
    }
    for item_id, text in texts.items():
        store.add(item_id, emb.embed([text])[0], namespace="t")
    q = emb.embed(["hiking in the mountains"])[0]
    hits = store.search(q, k=2, namespace="t")
    assert {h[0] for h in hits} == {"e1", "e3"}
    assert hits[0][1] >= hits[1][1]
    store.close()


@pytest.mark.parametrize("vec_cls", VEC_CLASSES)
def test_vector_namespace_and_type_filter(vec_cls):
    emb = FakeEmbedder(dim=64)
    store = vec_cls(None, dim=64)
    v = emb.embed(["same text"])[0]
    store.add("a", v, memory_type="episodic", namespace="ns1")
    store.add("b", v, memory_type="strategies", namespace="ns1")
    store.add("c", v, memory_type="episodic", namespace="ns2")
    hits = store.search(v, k=10, memory_type="episodic", namespace="ns1")
    assert [h[0] for h in hits] == ["a"]
    store.close()


@pytest.mark.parametrize("vec_cls", VEC_CLASSES)
def test_vector_dim_mismatch_raises(vec_cls):
    store = vec_cls(None, dim=8)
    with pytest.raises(ValueError):
        store.add("x", [0.1] * 16)
    store.close()


@pytest.mark.parametrize("vec_cls", VEC_CLASSES)
def test_vector_upsert_replaces(vec_cls):
    emb = FakeEmbedder(dim=32)
    store = vec_cls(None, dim=32)
    store.add("a", emb.embed(["old text"])[0], namespace="t")
    new_vec = emb.embed(["completely different"])[0]
    store.add("a", new_vec, namespace="t")
    assert store.count() == 1
    hits = store.search(new_vec, k=1, namespace="t")
    assert hits[0][0] == "a" and hits[0][1] > 0.99
    store.close()


@pytest.mark.parametrize("vec_cls", VEC_CLASSES)
def test_vector_delete_removes_from_search(vec_cls):
    emb = FakeEmbedder(dim=32)
    store = vec_cls(None, dim=32)
    v = emb.embed(["hello world"])[0]
    store.add("a", v, namespace="t")
    store.add("b", emb.embed(["something else"])[0], namespace="t")
    store.delete(["a"])
    assert store.count() == 1
    assert [h[0] for h in store.search(v, k=5, namespace="t")] == ["b"]
    assert store.get(["a"]) == {}
    store.close()


@pytest.mark.parametrize("vec_cls", VEC_CLASSES)
def test_vector_delete_missing_is_noop(vec_cls):
    store = vec_cls(None, dim=8)
    store.delete(["no-such-id"])  # must not raise
    store.close()


ENGINE_CLASSES = [
    _param(SqliteVecStore, "sqlite_vec"),
    _param(LanceDBVectorStore, "lancedb"),
    _param(QdrantVectorStore, "qdrant_client"),
    _param(ChromaVectorStore, "chromadb"),
]


@pytest.mark.parametrize("vec_cls", ENGINE_CLASSES)
def test_engine_disk_persistence_and_dim_guard(tmp_path, vec_cls):
    emb = FakeEmbedder(dim=32)
    path = tmp_path / ("v.db" if vec_cls is SqliteVecStore else "store")
    store = vec_cls(path, dim=32)
    store.add("a", emb.embed(["hello world"])[0])
    store.persist()
    store.close()
    reloaded = vec_cls(path, dim=32)
    assert reloaded.count() == 1
    reloaded.close()
    # dim mismatch on reopen must be loud (docs/03 §1.2)
    with pytest.raises(ValueError):
        vec_cls(path, dim=16)


def test_numpy_store_persistence(tmp_path):
    emb = FakeEmbedder(dim=32)
    path = tmp_path / "v.npz"
    store = NumpyVectorStore(path, dim=32)
    store.add("a", emb.embed(["hello world"])[0])
    store.persist()
    reloaded = NumpyVectorStore(path, dim=32)
    assert reloaded.count() == 1
    # dim mismatch on reload must be loud (docs/03 §1.2)
    with pytest.raises(ValueError):
        NumpyVectorStore(path, dim=16)


@pytest.mark.skipif(not os.environ.get("AGMEM_TEST_PG"),
                    reason="embedded Postgres spins up a real server (~10s); "
                           "set AGMEM_TEST_PG=1 to include")
def test_postgres_doc_store_roundtrip():
    from agmem.stores.postgres_doc import PostgresDocStore

    s = PostgresDocStore(None)
    try:
        ep = Episode(content="hiking in the mountains", namespace="t")
        s.add_episode(ep)
        assert s.count_episodes("t") == 1
        assert s.search_lexical("hiking", namespace="t")[0][0] == ep.id
        s.put_item("i1", "facts", "t", {"id": "i1", "content": "Alice lives in Paris"})
        assert s.search_lexical_items("Paris", "facts", namespace="t")[0][0] == "i1"
        s.append([MemoryOp(op=OpType.ADD, target_type="facts", target_id="i1",
                           payload={}, actor="test")])
        assert s.count() == 1
    finally:
        s.close()
