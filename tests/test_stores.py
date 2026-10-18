import json
import os
import sqlite3
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


KOREAN_QUERY = "데이터베이스를 마이그레이션하다가 버그"


def test_lexical_search_matches_korean_through_particles(doc):
    """`unicode61` keeps "데이터베이스를" as one token, so a Korean query with a
    particle on the word matched nothing stored without it, and the other way
    round. Both sides now carry syllable bigrams; unrelated Korean shares none."""
    hit = Episode(content="데이터베이스 마이그레이션 실패", namespace="t")
    bug = Episode(content="버그를 고쳤다", namespace="t")
    unrelated = Episode(content="오늘 점심은 김치찌개를 먹었다", namespace="t")
    for ep in (hit, bug, unrelated):
        doc.add_episode(ep)
    assert [i for i, _ in doc.search_lexical(KOREAN_QUERY, namespace="t")][:2] == [hit.id, bug.id]
    assert unrelated.id not in {i for i, _ in doc.search_lexical(KOREAN_QUERY, namespace="t")}
    assert [i for i, _ in doc.search_lexical("버그", namespace="t")] == [bug.id]
    assert [i for i, _ in doc.search_lexical("데이터베이스가", namespace="t")] == [hit.id]
    # a mixed-script token is split, so the English half is reachable alone
    doc.add_episode(mixed := Episode(content="API를 바꿨다", namespace="t"))
    assert [i for i, _ in doc.search_lexical("API", namespace="t")] == [mixed.id]

    doc.put_item("r1", "runbooks", "t", {"id": "r1", "content": "데이터베이스 마이그레이션 절차"})
    doc.put_item("r2", "runbooks", "t", {"id": "r2", "content": "점심 메뉴 고르기"})
    assert [i for i, _ in doc.search_lexical_items(KOREAN_QUERY, "runbooks", namespace="t")] == [
        "r1"
    ]


def test_lexical_terms_leave_english_unchanged():
    from agmem.stores.sqlite_doc import _fts_query, _lexical_text

    text = "Correction: API payloads now use owner_id"
    assert _lexical_text(text) == text
    assert _fts_query("sushi food") == '"sushi" OR "food"'


def _old_store(path, episodes: list[tuple[str, str]], items: list[tuple[str, str]]) -> None:
    """A store as the whole-token index wrote it: no `doc_meta`, raw FTS rows."""
    from agmem.stores import sqlite_doc

    conn = sqlite3.connect(path)
    conn.executescript(
        sqlite_doc._SCHEMA.replace(
            "CREATE TABLE IF NOT EXISTS doc_meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);", ""
        )
    )
    for ep_id, raw in episodes:
        conn.execute(
            "INSERT INTO episodes (id, namespace, role, content, timestamp) VALUES (?,?,?,?,?)",
            (ep_id, "t", "user", raw, "2026-10-01T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO episodes_fts (content, id, namespace) VALUES (?,?,?)", (raw, ep_id, "t")
        )
    for item_id, raw in items:
        conn.execute(
            "INSERT INTO items (id, memory_type, namespace, data) VALUES (?,?,?,?)",
            (item_id, "runbooks", "t", json.dumps({"id": item_id, "content": raw})),
        )
        conn.execute(
            "INSERT INTO items_fts (content, item_id, memory_type, namespace) VALUES (?,?,?,?)",
            (raw, item_id, "runbooks", "t"),
        )
    conn.commit()
    assert (
        conn.execute(
            "SELECT count(*) FROM episodes_fts WHERE episodes_fts MATCH '\"데이\"'"
        ).fetchone()[0]
        == 0
    )
    conn.close()


def _fts_ids(store, table: str = "episodes_fts", column: str = "id") -> list[str]:
    return sorted(r[0] for r in store._conn.execute(f"SELECT {column} FROM {table}"))


def test_a_store_indexed_before_hangul_terms_is_rebuilt_on_open(tmp_path):
    """A store written by the whole-token index has FTS rows with no bigrams,
    and no query would reach its Korean episodes until they were rewritten.
    Opening it rebuilds both FTS tables from the base tables, once."""
    from agmem.stores import sqlite_doc

    path = tmp_path / "memory.db"
    _old_store(path, [("e1", "데이터베이스 마이그레이션 실패")], [("r1", "버그 재현 절차")])

    store = SqliteDocStore(path)
    assert [i for i, _ in store.search_lexical(KOREAN_QUERY, namespace="t")] == ["e1"]
    assert [i for i, _ in store.search_lexical_items("버그를", "runbooks", namespace="t")] == ["r1"]
    assert store._lexical_index() == sqlite_doc.LEXICAL_INDEX
    store.close()

    store = SqliteDocStore(path)  # the second open finds the marker and leaves the rows be
    assert store._conn.execute("SELECT count(*) FROM episodes_fts").fetchone()[0] == 1
    assert store._conn.execute("SELECT count(*) FROM items_fts").fetchone()[0] == 1
    store.close()


def _korean_episodes(n: int) -> list[tuple[str, str]]:
    return [(f"e{i}", f"데이터베이스 마이그레이션 실패 {i}") for i in range(n)]


def test_the_lexical_rebuild_lets_other_writers_in_between_batches(tmp_path, monkeypatch):
    """The first rebuild ran in one transaction: 11.6 s on 45k Korean
    episodes, and every other opener — a hook's write, the daemon — failed on
    "database is locked" behind it. It now commits every `MIGRATE_BATCH` rows,
    so a writer with no patience at all (timeout 0) gets in between batches,
    and what it wrote is in the rebuilt index exactly once."""
    from agmem.stores import sqlite_doc

    path = tmp_path / "memory.db"
    _old_store(path, _korean_episodes(7), [])
    monkeypatch.setattr(sqlite_doc, "MIGRATE_BATCH", 2)
    written: list[str] = []

    def write_between_batches(_seconds: float) -> None:
        if len(written) == 3:
            return
        other = SqliteDocStore(path, migrate_lexical=False, timeout=0)
        try:
            ep = Episode(content=f"버그를 고쳤다 {len(written)}", namespace="t")
            other.add_episode(ep)
            written.append(ep.id)
        finally:
            other.close()

    monkeypatch.setattr(sqlite_doc.time, "sleep", write_between_batches)
    store = SqliteDocStore(path)
    try:
        assert len(written) == 3
        assert store._lexical_index() == sqlite_doc.LEXICAL_INDEX
        ids = _fts_ids(store)
        assert ids == sorted({*ids}) and len(ids) == 7 + len(written)  # no duplicates
        hits = {i for i, _ in store.search_lexical("데이터베이스를", k=50, namespace="t")}
        assert hits == {f"e{i}" for i in range(7)}
        assert {i for i, _ in store.search_lexical("버그", k=50, namespace="t")} == set(written)
    finally:
        store.close()


def test_an_interrupted_lexical_rebuild_resumes_without_duplicates(tmp_path, monkeypatch):
    """Killed between batches, a rebuild keeps its cursor and the old index
    keeps answering whole-token queries; the next open resumes, and a row
    rewritten in the meantime is indexed with its new content only."""
    from agmem.stores import sqlite_doc

    path = tmp_path / "memory.db"
    _old_store(path, _korean_episodes(5), [("r1", "버그 재현 절차"), ("r2", "점심 메뉴")])
    monkeypatch.setattr(sqlite_doc, "MIGRATE_BATCH", 2)
    monkeypatch.setattr(sqlite_doc.time, "sleep", lambda _s: None)

    def killed(self):
        raise KeyboardInterrupt  # the process dies before the swap

    monkeypatch.setattr(SqliteDocStore, "_swap_lexical_index", killed)
    with pytest.raises(KeyboardInterrupt):
        SqliteDocStore(path)
    monkeypatch.undo()
    monkeypatch.setattr(sqlite_doc.time, "sleep", lambda _s: None)

    hook = SqliteDocStore(path, migrate_lexical=False)
    try:
        assert hook._lexical_index() is None and hook._meta("lexical_cursor_episodes") == "5"
        assert {i for i, _ in hook.search_lexical("데이터베이스", k=50, namespace="t")} == {
            f"e{i}" for i in range(5)
        }
        hook.put_item("r1", "runbooks", "t", {"id": "r1", "content": "인덱스 재구성 절차"})
    finally:
        hook.close()

    store = SqliteDocStore(path)
    try:
        assert store._lexical_index() == sqlite_doc.LEXICAL_INDEX
        assert store._meta("lexical_cursor_episodes") is None
        assert _fts_ids(store) == sorted(f"e{i}" for i in range(5))
        assert _fts_ids(store, "items_fts", "item_id") == ["r1", "r2"]
        assert store.search_lexical_items("재구성을", "runbooks", namespace="t")[0][0] == "r1"
        assert store.search_lexical_items("버그", "runbooks", namespace="t") == []
        names = {r[0] for r in store._conn.execute("SELECT name FROM sqlite_master")}
        assert not {n for n in names if "_next" in n}
    finally:
        store.close()


def _leftovers(store) -> tuple[list[str], list[str]]:
    tables = [
        r[0]
        for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE '%_next'"
        )
    ]
    keys = [r[0] for r in store._conn.execute("SELECT k FROM doc_meta WHERE k != 'lexical_index'")]
    return tables, keys


def test_a_rebuild_that_loses_the_race_leaves_nothing_behind(tmp_path):
    """The marker was read outside the write lock, so a process that saw it
    missing while another finished the rebuild created fresh, empty `_next`
    tables and a `lexical_started` row that nothing would ever clear. The
    setup now runs under the lock and re-reads the marker; an open that finds
    the marker current clears anything an earlier race or kill left."""
    from agmem.stores import sqlite_doc

    path = tmp_path / "memory.db"
    _old_store(path, _korean_episodes(3), [])
    late = SqliteDocStore(path, migrate_lexical=False)
    SqliteDocStore(path).close()  # the winner migrates fully
    real, reads = late._lexical_index, []

    def stale_first_look() -> str | None:
        reads.append(1)
        return None if len(reads) == 1 else real()

    late._lexical_index = stale_first_look
    try:
        late._migrate_lexical_index()
        assert len(reads) >= 2  # it did look again, under the lock
        assert _leftovers(late) == ([], [])
        assert late._lexical_index() == sqlite_doc.LEXICAL_INDEX
    finally:
        late.close()

    raw = sqlite3.connect(path)  # what a pre-fix race or a kill left behind
    raw.execute("CREATE VIRTUAL TABLE episodes_fts_next USING fts5(content)")
    raw.execute("INSERT INTO doc_meta (k, v) VALUES ('lexical_started', 'x')")
    raw.commit()
    raw.close()
    store = SqliteDocStore(path)
    try:
        assert _leftovers(store) == ([], [])
        assert [i for i, _ in store.search_lexical("데이터베이스를", k=5, namespace="t")]
    finally:
        store.close()


def test_a_stop_bigram_that_is_a_words_stem_still_matches(doc):
    """하지 (summer solstice) is in the stoplist as the ending of "…하지 않는",
    and dropping it everywhere left "하지에" nothing to match "하지 행사" by.
    A run's first bigram is never dropped, nor a two-syllable word."""
    solstice = Episode(content="하지 행사 정리", namespace="t")
    name = Episode(content="이다 작가 인터뷰", namespace="t")
    doc.add_episode(solstice)
    doc.add_episode(name)
    assert [i for i, _ in doc.search_lexical("하지에 뭐 했지", namespace="t")] == [solstice.id]
    assert [i for i, _ in doc.search_lexical("이다의 신간", namespace="t")] == [name.id]


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


def test_widening_stops_at_the_knn_limit_instead_of_raising():
    """sqlite-vec's own cap, which the row-count bound did not cover.

    `search` widens the knn until `k` rows pass the filter or the table is
    exhausted, and bounded that widening by the row count alone. sqlite-vec
    refuses a `v.k` above 4096, so once a store passed that many vectors the
    widening raised `OperationalError` — measured on the dogfood store at 4,571
    vectors, where 3 of 10 real prompts died and reached the model as no memory
    at all (2026-09-06). Past the ceiling a filtered search now returns what
    the nearest `MAX_KNN_K` held, which is what this index can honestly say."""
    import random

    from agmem.stores.sqlite_vec import MAX_KNN_K, _serialize

    rng = random.Random(7)
    dim = 8
    store = SqliteVecStore(None, dim=dim)
    query = [1.0] + [0.0] * (dim - 1)

    # The cap belongs to the extension, not to us: assert it where it lives, so
    # a sqlite-vec that raises it later fails this test rather than the store.
    store.add("probe", query, memory_type="episodic", namespace="t")
    store._conn.execute(
        "SELECT v.rowid FROM vectors v WHERE v.embedding MATCH ? AND v.k = ?",
        (_serialize(query), MAX_KNN_K),
    ).fetchall()
    with pytest.raises(sqlite3.OperationalError, match="too large"):
        store._conn.execute(
            "SELECT v.rowid FROM vectors v WHERE v.embedding MATCH ? AND v.k = ?",
            (_serialize(query), MAX_KNN_K + 1),
        ).fetchall()

    for i in range(MAX_KNN_K + 4):  # past the ceiling, all close to the query
        v = [1.0] + [rng.uniform(-0.05, 0.05) for _ in range(dim - 1)]
        store.add(f"ep-{i}", v, memory_type="episodic", namespace="t")
    far = [0.0] * dim
    far[1] = 1.0
    store.add("rb-0", far, memory_type="runbooks", namespace="t")

    # The widening has to give up here, and giving up is returning, not raising.
    assert store.search(query, k=3, memory_type="runbooks", namespace="t") == []
    assert len(store.search(query, k=3, namespace="t")) == 3
    store.close()


def test_sqlite_vec_get_and_delete_take_more_ids_than_sqlite_will_bind():
    """`get` and `delete` bound one parameter per id in a single `IN (...)`, and
    SQLite refuses more than 32,766 — so the daemon's startup backfill, which
    asks `get` about every episode in a namespace, died with "too many SQL
    variables" on a store that size and the daemon never came up. Few stored
    vectors, many ids asked about: the limit is on the question, not the table."""
    store = SqliteVecStore(None, dim=2)
    for i in range(0, 40_000, 4_000):
        store.add(f"ep-{i}", [1.0, float(i)], namespace="t")
    ids = [f"ep-{i}" for i in range(40_000)]

    got = store.get(ids)
    assert sorted(got) == sorted(f"ep-{i}" for i in range(0, 40_000, 4_000))
    assert got["ep-36000"] == [1.0, 36000.0]

    store.delete(ids)
    assert store.count() == 0
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


def test_every_memory_type_declares_its_axis_and_update_policy():
    """`MEMORY_TYPE_INFO` is the functional-axis and update-policy declaration
    the flat tuple never carried (docs/research/agent-memory-axes-v1.md §2.1,
    §7.2). It must stay exhaustive with `MEMORY_TYPES`, and the values are a
    closed vocabulary — a new type gets a place on both axes or does not get
    in."""
    from agmem.core.types import MEMORY_TYPE_INFO, MEMORY_TYPES

    assert set(MEMORY_TYPE_INFO) == set(MEMORY_TYPES)
    axes = {"working", "episodic", "semantic", "procedural", "bookkeeping"}
    updates = {"append_only", "supersede", "abstract", "none"}
    for name, info in MEMORY_TYPE_INFO.items():
        assert info.axis in axes and info.update in updates, name
    # the pairings the research fixed: raw experience appends, facts supersede, procedures abstract
    assert MEMORY_TYPE_INFO["episodic"].update == "append_only"
    assert MEMORY_TYPE_INFO["facts"].update == "supersede"
    assert MEMORY_TYPE_INFO["runbooks"].axis == "procedural"
    assert MEMORY_TYPE_INFO["state"].axis == "bookkeeping"
    # nothing is working memory yet: the facade has no such type (the research's §2.2 row)
    assert not [n for n, i in MEMORY_TYPE_INFO.items() if i.axis == "working"]


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
