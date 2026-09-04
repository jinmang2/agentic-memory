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
    marks = (
        [] if pkg is None or find_spec(pkg) else [pytest.mark.skip(reason=f"{pkg} not installed")]
    )
    return pytest.param(cls, id=cls.__name__, marks=marks)


@pytest.fixture
def doc():
    store = SqliteDocStore(":memory:")
    yield store
    store.close()


def test_episode_roundtrip(doc):
    episode = Episode(content="파리 여행 계획을 세우고 있어요", role="user", namespace="t")
    doc.add_episode(episode)
    got = doc.get_episodes([episode.id])[0]
    assert got.content == episode.content
    assert got.timestamp == episode.timestamp
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
    ops = [
        MemoryOp(
            op=OpType.ADD,
            target_type="notes",
            target_id="n1",
            payload={"content": "x"},
            actor="amem",
        )
    ]
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
    embedder = FakeEmbedder(dim=64)
    store = vec_cls(None, dim=64)
    texts = {
        "e1": "hiking mountains trail backpack",
        "e2": "sushi ramen tokyo restaurant",
        "e3": "mountains hiking gear boots",
    }
    for item_id, text in texts.items():
        store.add(item_id, embedder.embed([text])[0], namespace="t")
    q = embedder.embed(["hiking in the mountains"])[0]
    hits = store.search(q, k=2, namespace="t")
    assert {h[0] for h in hits} == {"e1", "e3"}
    assert hits[0][1] >= hits[1][1]
    store.close()


@pytest.mark.parametrize("vec_cls", VEC_CLASSES)
def test_vector_namespace_and_type_filter(vec_cls):
    embedder = FakeEmbedder(dim=64)
    store = vec_cls(None, dim=64)
    v = embedder.embed(["same text"])[0]
    store.add("a", v, memory_type="episodic", namespace="ns1")
    store.add("b", v, memory_type="strategies", namespace="ns1")
    store.add("c", v, memory_type="episodic", namespace="ns2")
    hits = store.search(v, k=10, memory_type="episodic", namespace="ns1")
    assert [h[0] for h in hits] == ["a"]
    store.close()


@pytest.mark.parametrize("vec_cls", VEC_CLASSES)
def test_a_rare_type_is_found_under_many_rows_of_another(vec_cls):
    """A filtered search returns fewer than k only when the pool holds fewer
    than k — never because rows of another type outranked them. The sqlite-vec
    store over-fetched a fixed 4x and post-filtered, so in a store of 3,574
    episodes and 107 runbooks a runbooks-only search returned nothing for a
    query that resembled a page (LongMemEval-V2, 2026-09-04)."""
    import random

    rng = random.Random(7)
    dim = 32
    store = vec_cls(None, dim=dim)
    query = [1.0] + [0.0] * (dim - 1)
    for i in range(400):  # the common type, all close to the query
        v = [1.0] + [rng.uniform(-0.05, 0.05) for _ in range(dim - 1)]
        store.add(f"ep-{i}", v, memory_type="episodic", namespace="t")
    for i in range(5):  # the rare type, far from it
        v = [0.0] * dim
        v[1 + i] = 1.0
        store.add(f"rb-{i}", v, memory_type="runbooks", namespace="t")
    hits = store.search(query, k=3, memory_type="runbooks", namespace="t")
    assert len(hits) == 3 and all(h[0].startswith("rb-") for h in hits)
    hits = store.search(query, k=10, memory_type="runbooks", namespace="t")
    assert len(hits) == 5  # the pool really holds five
    assert len(store.search(query, k=3, namespace="t")) == 3  # unfiltered: still one query's worth
    store.close()


@pytest.mark.parametrize("vec_cls", VEC_CLASSES)
def test_vector_dim_mismatch_raises(vec_cls):
    store = vec_cls(None, dim=8)
    with pytest.raises(ValueError):
        store.add("x", [0.1] * 16)
    store.close()


@pytest.mark.parametrize("vec_cls", VEC_CLASSES)
def test_vector_upsert_replaces(vec_cls):
    embedder = FakeEmbedder(dim=32)
    store = vec_cls(None, dim=32)
    store.add("a", embedder.embed(["old text"])[0], namespace="t")
    new_vec = embedder.embed(["completely different"])[0]
    store.add("a", new_vec, namespace="t")
    assert store.count() == 1
    hits = store.search(new_vec, k=1, namespace="t")
    assert hits[0][0] == "a" and hits[0][1] > 0.99
    store.close()


@pytest.mark.parametrize("vec_cls", VEC_CLASSES)
def test_vector_delete_removes_from_search(vec_cls):
    embedder = FakeEmbedder(dim=32)
    store = vec_cls(None, dim=32)
    v = embedder.embed(["hello world"])[0]
    store.add("a", v, namespace="t")
    store.add("b", embedder.embed(["something else"])[0], namespace="t")
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
    embedder = FakeEmbedder(dim=32)
    path = tmp_path / ("v.db" if vec_cls is SqliteVecStore else "store")
    store = vec_cls(path, dim=32)
    store.add("a", embedder.embed(["hello world"])[0])
    store.persist()
    store.close()
    reloaded = vec_cls(path, dim=32)
    assert reloaded.count() == 1
    reloaded.close()
    # dim mismatch on reopen must be loud (docs/03 §1.2)
    with pytest.raises(ValueError):
        vec_cls(path, dim=16)


def test_numpy_store_persistence(tmp_path):
    embedder = FakeEmbedder(dim=32)
    path = tmp_path / "v.npz"
    store = NumpyVectorStore(path, dim=32)
    store.add("a", embedder.embed(["hello world"])[0])
    store.persist()
    reloaded = NumpyVectorStore(path, dim=32)
    assert reloaded.count() == 1
    # dim mismatch on reload must be loud (docs/03 §1.2)
    with pytest.raises(ValueError):
        NumpyVectorStore(path, dim=16)


@pytest.mark.skipif(
    not os.environ.get("AGMEM_TEST_PG"),
    reason="embedded Postgres spins up a real server (~10s); set AGMEM_TEST_PG=1 to include",
)
def test_postgres_doc_store_roundtrip():
    from agmem.stores.postgres_doc import PostgresDocStore

    s = PostgresDocStore(None)
    try:
        episode = Episode(content="hiking in the mountains", namespace="t")
        s.add_episode(episode)
        assert s.count_episodes("t") == 1
        assert s.search_lexical("hiking", namespace="t")[0][0] == episode.id
        s.put_item("i1", "facts", "t", {"id": "i1", "content": "Alice lives in Paris"})
        assert s.search_lexical_items("Paris", "facts", namespace="t")[0][0] == "i1"
        # `lexical_text` override, same pin as the sqlite test below: index the
        # override, never the render-only `content` (this backend ignored it
        # until 2026-08-19)
        s.put_item(
            "c1",
            "communities",
            "t",
            {
                "id": "c1",
                "content": "Hiking Club: members share alpine trip reports",
                "lexical_text": "Hiking Club",
            },
        )
        assert s.search_lexical_items("Hiking", "communities", namespace="t")[0][0] == "c1"
        assert s.search_lexical_items("alpine", "communities", namespace="t") == []
        s.append(
            [MemoryOp(op=OpType.ADD, target_type="facts", target_id="i1", payload={}, actor="test")]
        )
        assert s.count() == 1
    finally:
        s.close()


def test_memory_types_covers_every_organizer_output():
    """`core/ops.py` declares `target_type` to be one of `MEMORY_TYPES`, so the
    tuple has to stay exhaustive — `experiences` was missing for as long as
    ReasoningBank had been emitting it, and nothing noticed because the
    constraint is documentation, not validation.

    `produces` is the declaration that goes stale, so check against it."""
    from agmem.core.types import MEMORY_TYPES
    from agmem.organizers import ORGANIZERS

    declared = {t for cls in ORGANIZERS.values() for t in cls.produces}
    assert declared - set(MEMORY_TYPES) == set()


def test_put_item_lexical_text_overrides_the_bm25_channel(doc):
    """`lexical_text` decouples what the lexical channel indexes from what
    `content` renders — Graphiti indexes community nodes on the name alone
    while rendering name + summary (see `SqliteDocStore.put_item`). Postgres
    ignored the override until 2026-08-19 and indexed `content`
    unconditionally, so the two backends BM25-matched different text for the
    same item; pinned here on sqlite and inside the gated Postgres roundtrip
    below."""
    doc.put_item(
        "c1",
        "communities",
        "t",
        {
            "id": "c1",
            "content": "Hiking Club: members share alpine trip reports",
            "lexical_text": "Hiking Club",
        },
    )
    assert doc.search_lexical_items("Hiking", "communities", namespace="t")[0][0] == "c1"
    # render-only text must NOT be lexically reachable once overridden
    assert doc.search_lexical_items("alpine", "communities", namespace="t") == []
