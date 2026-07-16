import pytest

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode
from agmem.embed.fake import FakeEmbedder
from agmem.stores.numpy_vec import NumpyVectorStore
from agmem.stores.sqlite_doc import SqliteDocStore
from agmem.stores.sqlite_vec import SqliteVecStore


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


def test_items_roundtrip(doc):
    doc.put_item("s1", "strategies", "t", {"id": "s1", "title": "T", "content": "C"})
    items = doc.get_items(["s1"], "strategies")
    assert items[0]["title"] == "T"


VEC_CLASSES = [NumpyVectorStore, SqliteVecStore]


@pytest.mark.parametrize("vec_cls", VEC_CLASSES)
def test_vector_similarity_ordering(vec_cls):
    emb = FakeEmbedder(dim=64)
    store = vec_cls(dim=64) if vec_cls is SqliteVecStore else vec_cls(None, dim=64)
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
    store = vec_cls(dim=64) if vec_cls is SqliteVecStore else vec_cls(None, dim=64)
    v = emb.embed(["same text"])[0]
    store.add("a", v, memory_type="episodic", namespace="ns1")
    store.add("b", v, memory_type="strategies", namespace="ns1")
    store.add("c", v, memory_type="episodic", namespace="ns2")
    hits = store.search(v, k=10, memory_type="episodic", namespace="ns1")
    assert [h[0] for h in hits] == ["a"]
    store.close()


@pytest.mark.parametrize("vec_cls", VEC_CLASSES)
def test_vector_dim_mismatch_raises(vec_cls):
    store = vec_cls(dim=8) if vec_cls is SqliteVecStore else vec_cls(None, dim=8)
    with pytest.raises(ValueError):
        store.add("x", [0.1] * 16)
    store.close()


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
