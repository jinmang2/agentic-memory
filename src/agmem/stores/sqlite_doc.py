"""SQLite doc store: episodes + generic derived items + FTS5 + evolution log.

Single-file source of truth for the ``lite`` profile (docs/03 §5.2).
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from agmem.capabilities.requires import Requires
from agmem.core.ops import MemoryOp
from agmem.core.types import Episode

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_episodes_ns ON episodes(namespace);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    content, id UNINDEXED, namespace UNINDEXED
);

CREATE TABLE IF NOT EXISTS items (
    id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    namespace TEXT NOT NULL,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (id, memory_type)
);
CREATE INDEX IF NOT EXISTS idx_items_type_ns ON items(memory_type, namespace);

CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
    content, item_id UNINDEXED, memory_type UNINDEXED, namespace UNINDEXED
);

CREATE TABLE IF NOT EXISTS evolution_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    op TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    actor TEXT NOT NULL,
    t_transaction TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS doc_meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""

_FTS_TOKEN = re.compile(r"\w+", re.UNICODE)

# Hangul syllables, U+AC00..U+D7A3. `unicode61` keeps a space-separated run of
# them as one token, and Korean writes its particles and endings onto the word:
# "버그를" is "버그" + object marker, so a whole-token index never matched one
# with the other, and "데이터베이스를 ..." found nothing stored as "데이터베이스".
# Syllable bigrams are the standard way around that without a morphological
# analyzer (the braincrew plugin's memory_search does the same): "버그를" ->
# 버그, 그를; "데이터베이스" -> 데이, 이터, 터베, 베이, 이스, so the shared
# stem shares terms whatever is attached to it. A one-syllable run stays whole.
HANGUL_RUN = re.compile(r"[\uac00-\ud7a3]+")
_SCRIPT_RUN = re.compile(r"[\uac00-\ud7a3]+|[^\W\uac00-\ud7a3]+")

# What the FTS rows hold, recorded in `doc_meta` under `lexical_index`. A store
# written before the Hangul terms were added has no record and gets its FTS
# tables rebuilt from the base tables on open (`_migrate_lexical_index`).
LEXICAL_INDEX = "hangul-bigram-1"


# Bigrams that are endings and particles rather than any word's stem, so two
# texts sharing one says nothing about their topic: "…확인했습니다" and "오늘 점심을
# 먹었습니다" share 습니 and 니다 and nothing else. Picked from document
# frequency over this repo's Korean docs (3,062 paragraphs: 에서 in 24%, 으로
# 22%, 한다 15%, 이다 13%, …) plus the polite forms a conversation uses and
# those docs do not. Dropped where a bigram decides a match — the FTS query and
# the prompt prune — and still indexed, so changing this list needs no rebuild.
HANGUL_STOP_BIGRAMS = frozenset(
    "에서 으로 한다 이다 는다 하지 하는 않는 이고 므로 니라 하고 이라 없다 아니 같은 "
    "니다 습니 합니 입니 하다 했다 했습 있다 었다 았다 였다 해서 지만 는데 있는 없는 "
    "어요 아요 에요 세요 해요 까지 부터 에게 하여 된다 되는".split()
)


def hangul_terms(text: str, *, drop_stop: bool = False) -> list[str]:
    """Syllable bigrams of every Hangul run in `text`; a one-syllable run is its own term.

    ``drop_stop`` leaves out `HANGUL_STOP_BIGRAMS`, but never a run's first
    bigram: an ending sits at the end of a word, and the first bigram is the
    word's stem — "하지에" (at the summer solstice) has 하지 and 지에, and
    dropping 하지 would leave it nothing to match "하지 행사" by. A two-syllable
    run's one bigram is its first, so "이다" as a word is kept too."""
    out: list[str] = []
    for run in HANGUL_RUN.findall(text):
        if len(run) == 1:
            out.append(run)
            continue
        out.extend(
            run[i : i + 2]
            for i in range(len(run) - 1)
            if not (drop_stop and i > 0 and run[i : i + 2] in HANGUL_STOP_BIGRAMS)
        )
    return out


def _extra_terms(text: str, *, drop_stop: bool = False) -> list[str]:
    """The terms `unicode61` does not produce on its own: for every token that
    holds Hangul, its Hangul bigrams, plus its other-script runs when it mixes
    the two ("API를" -> API, 를). Tokens without Hangul add nothing, so
    English text indexes and queries exactly as it did."""
    out: list[str] = []
    for token in _FTS_TOKEN.findall(text):
        if not HANGUL_RUN.search(token):
            continue
        for run in _SCRIPT_RUN.findall(token):
            terms = hangul_terms(run, drop_stop=drop_stop) if HANGUL_RUN.fullmatch(run) else [run]
            out.extend(t for t in terms if t != token)
    return out


def _lexical_text(text: str) -> str:
    """What an FTS row indexes for `text`: the text, then its `_extra_terms`."""
    extra = _extra_terms(text)
    return f"{text}\n{' '.join(extra)}" if extra else text


def _fts_query(query: str) -> str:
    """Sanitize free text into a safe FTS5 OR-query over its tokens and their
    `_extra_terms`, the same terms the index holds, less `HANGUL_STOP_BIGRAMS`."""
    tokens = list(dict.fromkeys([*_FTS_TOKEN.findall(query), *_extra_terms(query, drop_stop=True)]))
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else '""'


def _item_lexical_content(data: dict[str, Any]) -> str:
    # ``lexical_text`` overrides what the BM25 channel indexes, exactly as
    # ``embedding_text`` overrides what the dense channel embeds. The two
    # are not always the same field upstream: Graphiti indexes entity
    # nodes on (name, summary) but embeds the name alone, and indexes
    # community nodes on the name alone while rendering name + summary.
    # Without this the lexical channel is hardwired to whatever ``render``
    # wants, which is a different requirement.
    if data.get("deleted"):
        return ""
    return str(data.get("lexical_text") or data.get("content") or "")


# Base rows per transaction while an old FTS index is rebuilt, and how long an
# opener waits on another's write lock (sqlite3's default is 5 s).
MIGRATE_BATCH = 500
MIGRATE_PAUSE_S = 0.05
CONNECT_TIMEOUT_S = 15.0

# Invariant the rebuild's swap relies on: every rewrite of a base row is an
# `INSERT OR REPLACE` (`add_episode`, `put_item`), which deletes the row and
# inserts it with a fresh `created_at` / `updated_at`. An in-place `UPDATE` that
# kept the old stamp would be skipped by the swap's re-read, and a row the
# cursor had already passed would keep its pre-update FTS text.
# (base table, its FTS table, the FTS columns, the base rows' select with rowid
# first, the base column that changes whenever a row is rewritten).
_FTS_TABLES = (
    (
        "episodes",
        "episodes_fts",
        "content, id UNINDEXED, namespace UNINDEXED",
        "SELECT rowid, id, namespace, content FROM episodes",
        "created_at",
    ),
    (
        "items",
        "items_fts",
        "content, item_id UNINDEXED, memory_type UNINDEXED, namespace UNINDEXED",
        "SELECT rowid, id, memory_type, namespace, data FROM items",
        "updated_at",
    ),
)

_FTS_COLUMNS = {
    fts: [c.split()[0] for c in columns.split(",")] for _b, fts, columns, _s, _t in _FTS_TABLES
}


def _fts_fields(fts: str, row: tuple) -> tuple | None:
    """The FTS columns for one base row (`_FTS_TABLES` select minus rowid), or
    None when the row is not indexed — a deleted or empty item."""
    if fts == "episodes_fts":
        ep_id, ns, content = row
        return (_lexical_text(content), ep_id, ns)
    item_id, memory_type, ns, data = row
    content = _item_lexical_content(json.loads(data))
    return (_lexical_text(content), item_id, memory_type, ns) if content else None


class SqliteDocStore:
    """Default `DocStore`: one SQLite file backs episodes, derived items, and
    the evolution log, each with an FTS5 shadow table for lexical search.
    Safe to share across threads (serialized behind an internal lock)."""

    requires = Requires()  # stdlib only — always available

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        migrate_lexical: bool = True,
        timeout: float = CONNECT_TIMEOUT_S,
    ) -> None:
        """``path=None`` opens an in-memory, non-persistent database; a path
        creates parent dirs as needed and opens WAL mode for concurrent reads
        during writes. ``migrate_lexical=False`` opens a store whose FTS index
        predates `LEXICAL_INDEX` as it is (see `_migrate_lexical_index`)."""
        self.path = str(path) if path is not None else ":memory:"
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=timeout)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.RLock()
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
        if migrate_lexical:
            self._migrate_lexical_index()

    def _meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT v FROM doc_meta WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None

    def _lexical_index(self) -> str | None:
        return self._meta("lexical_index")

    def _migrate_lexical_index(self) -> None:
        """Rebuild both FTS tables when they were written under another
        `LEXICAL_INDEX` — in practice a store from before the Hangul terms,
        whose rows would otherwise never match a bigram query.

        In the background of every open, so it must not hold the write lock:
        the first version rebuilt in one transaction, which took 11.6 s on 45k
        Korean episodes while every other opener — hooks, the daemon — waited
        out its busy timeout and failed. Now the new index is built beside the
        old one (`<fts>_next`, keyed by the base table's rowid) in
        `MIGRATE_BATCH`-row transactions with the cursor saved in `doc_meta`,
        and swapped in by one short transaction at the end; the old index keeps
        serving until then (its rows still match whole tokens). Killed midway,
        the next open resumes at the cursor; two processes migrating at once
        take turns on the batches and the second finds the work done. Rows
        written while the rebuild runs are re-read at the swap: their
        `created_at` / `updated_at` is at or after the recorded start."""
        with self._lock:
            if self._lexical_index() == LEXICAL_INDEX:
                if self._migration_leftovers():
                    self._write_txn(lambda: False)  # the marker check clears them
                return
            if not self._write_txn(self._start_migration):
                return  # another process finished while this one looked
            while self._write_txn(self._migrate_batch):
                # Let a waiting writer in: SQLite's busy handler polls, and a
                # loop that takes the lock straight back starved it for seconds.
                time.sleep(MIGRATE_PAUSE_S)
            self._write_txn(self._swap_lexical_index)

    def _write_txn(self, step: Callable[[], bool]) -> bool:
        """Run `step` under the write lock if the index still needs migrating,
        and return what it returned. The marker is read again under the lock:
        a process that saw it missing may find another has finished since, and
        then only clears what an earlier race or kill left behind."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            if self._lexical_index() == LEXICAL_INDEX:
                self._clear_migration_leftovers()
                done = False
            else:
                done = step()
            self._conn.commit()
            return done
        except BaseException:
            self._conn.rollback()
            raise

    def _start_migration(self) -> bool:
        for _base, fts, columns, _sql, _stamp in _FTS_TABLES:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts}_next USING fts5({columns})"
            )
        self._conn.execute(
            "INSERT OR IGNORE INTO doc_meta (k, v)"
            " VALUES ('lexical_started', strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
        )
        return True

    def _migration_leftovers(self) -> bool:
        names = [f"{fts}_next" for _b, fts, _c, _s, _t in _FTS_TABLES]
        marks = ",".join("?" * len(names))
        tables = self._conn.execute(
            f"SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name IN ({marks})", names
        ).fetchone()[0]
        return bool(tables) or self._meta("lexical_started") is not None

    def _clear_migration_leftovers(self) -> None:
        for _base, fts, _columns, _sql, _stamp in _FTS_TABLES:
            self._conn.execute(f"DROP TABLE IF EXISTS {fts}_next")
        self._conn.execute(
            "DELETE FROM doc_meta WHERE k IN"
            " ('lexical_started', 'lexical_cursor_episodes', 'lexical_cursor_items')"
        )

    def _migrate_batch(self) -> bool:
        """One batch of each base table past its cursor; False once none is left."""
        moved = False
        for base, fts, _columns, select, _stamp in _FTS_TABLES:
            key = f"lexical_cursor_{base}"
            cursor = int(self._meta(key) or 0)
            rows = self._conn.execute(
                f"{select} WHERE rowid > ? ORDER BY rowid LIMIT ?", (cursor, MIGRATE_BATCH)
            ).fetchall()
            if not rows:
                continue
            self._index_next(fts, rows)
            self._conn.execute(
                "INSERT OR REPLACE INTO doc_meta (k, v) VALUES (?, ?)", (key, str(rows[-1][0]))
            )
            moved = True
        return moved

    def _swap_lexical_index(self) -> bool:
        started = self._meta("lexical_started") or ""
        while self._migrate_batch():  # whatever arrived since the last batch
            pass
        for base, fts, _columns, select, stamp in _FTS_TABLES:
            # Rewritten since the start: `INSERT OR REPLACE` may hand a row its
            # old rowid back, which no cursor would revisit.
            self._index_next(fts, self._conn.execute(f"{select} WHERE {stamp} >= ?", (started,)))
            self._conn.execute(
                f"DELETE FROM {fts}_next WHERE rowid NOT IN (SELECT rowid FROM {base})"
            )
            self._conn.execute(f"DROP TABLE {fts}")
            self._conn.execute(f"ALTER TABLE {fts}_next RENAME TO {fts}")
        self._conn.execute(
            "DELETE FROM doc_meta WHERE k IN"
            " ('lexical_started', 'lexical_cursor_episodes', 'lexical_cursor_items')"
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO doc_meta (k, v) VALUES ('lexical_index', ?)", (LEXICAL_INDEX,)
        )
        return True

    def _index_next(self, fts: str, rows) -> None:
        for row in rows:
            rowid, fields = row[0], _fts_fields(fts, row[1:])
            self._conn.execute(f"DELETE FROM {fts}_next WHERE rowid = ?", (rowid,))
            if fields is not None:
                names = ", ".join(_FTS_COLUMNS[fts])
                marks = ",".join("?" * (len(fields) + 1))
                self._conn.execute(
                    f"INSERT INTO {fts}_next (rowid, {names}) VALUES ({marks})", (rowid, *fields)
                )

    # -- episodes -----------------------------------------------------------

    def add_episode(self, episode: Episode) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO episodes (id, namespace, role, content, timestamp, meta)"
                " VALUES (?,?,?,?,?,?)",
                (
                    episode.id,
                    episode.namespace,
                    episode.role,
                    episode.content,
                    episode.timestamp.isoformat(),
                    json.dumps(episode.meta, ensure_ascii=False, default=str),
                ),
            )
            self._conn.execute("DELETE FROM episodes_fts WHERE id = ?", (episode.id,))
            self._conn.execute(
                "INSERT INTO episodes_fts (content, id, namespace) VALUES (?,?,?)",
                (_lexical_text(episode.content), episode.id, episode.namespace),
            )

    def get_episodes(self, ids: list[str]) -> list[Episode]:
        if not ids:
            return []
        with self._lock:
            marks = ",".join("?" * len(ids))
            rows = self._conn.execute(
                f"SELECT id, namespace, role, content, timestamp, meta FROM episodes"
                f" WHERE id IN ({marks})",
                ids,
            ).fetchall()
        by_id = {
            r[0]: Episode(
                id=r[0],
                namespace=r[1],
                role=r[2],
                content=r[3],
                timestamp=datetime.fromisoformat(r[4]),
                meta=json.loads(r[5]),
            )
            for r in rows
        }
        return [by_id[i] for i in ids if i in by_id]  # preserve caller order

    def list_episodes(self, namespace: str | None = None) -> list[Episode]:
        """Full scan of raw episodes (oldest-first), optionally namespace-scoped.
        Parallels `list_items` for derived types; used by the reproduction
        harness's post-ingest memory snapshot to dump every ingested turn."""
        sql = "SELECT id, namespace, role, content, timestamp, meta FROM episodes"
        args: list[Any] = []
        if namespace:
            sql += " WHERE namespace = ?"
            args.append(namespace)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY timestamp", args).fetchall()
        return [
            Episode(
                id=r[0],
                namespace=r[1],
                role=r[2],
                content=r[3],
                timestamp=datetime.fromisoformat(r[4]),
                meta=json.loads(r[5]),
            )
            for r in rows
        ]

    def count_episodes(self, namespace: str | None = None) -> int:
        with self._lock:
            if namespace:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE namespace = ?", (namespace,)
                ).fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        return int(row[0])

    def search_lexical(
        self, query: str, k: int = 10, namespace: str | None = None
    ) -> list[tuple[str, float]]:
        """BM25 search; returns (episode_id, score) with higher = better."""
        sql = "SELECT id, bm25(episodes_fts) AS rank FROM episodes_fts WHERE episodes_fts MATCH ?"
        params: list[Any] = [_fts_query(query)]
        if namespace:
            sql += " AND namespace = ?"
            params.append(namespace)
        sql += " ORDER BY rank LIMIT ?"
        params.append(k)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        # bm25() is lower-is-better; negate so higher = better.
        return [(r[0], -float(r[1])) for r in rows]

    # -- generic derived items ----------------------------------------------

    def put_item(
        self, item_id: str, memory_type: str, namespace: str, data: dict[str, Any]
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO items (id, memory_type, namespace, data) VALUES (?,?,?,?)",
                (
                    item_id,
                    memory_type,
                    namespace,
                    json.dumps(data, ensure_ascii=False, default=str),
                ),
            )
            self._conn.execute(
                "DELETE FROM items_fts WHERE item_id = ? AND memory_type = ?",
                (item_id, memory_type),
            )
            content = _item_lexical_content(data)
            if content:
                self._conn.execute(
                    "INSERT INTO items_fts (content, item_id, memory_type,"
                    " namespace) VALUES (?,?,?,?)",
                    (_lexical_text(content), item_id, memory_type, namespace),
                )

    def search_lexical_items(
        self, query: str, memory_type: str, k: int = 10, namespace: str | None = None
    ) -> list[tuple[str, float]]:
        """BM25 over derived items of one type (Zep hybrid search channel)."""
        sql = (
            "SELECT item_id, bm25(items_fts) AS rank FROM items_fts"
            " WHERE items_fts MATCH ? AND memory_type = ?"
        )
        params: list[Any] = [_fts_query(query), memory_type]
        if namespace:
            sql += " AND namespace = ?"
            params.append(namespace)
        sql += " ORDER BY rank LIMIT ?"
        params.append(k)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [(r[0], -float(r[1])) for r in rows]

    def list_items(self, memory_type: str, namespace: str | None = None) -> list[dict[str, Any]]:
        """Full scan of one memory type (e.g. ACE's whole-playbook read)."""
        sql = "SELECT data FROM items WHERE memory_type = ?"
        args: list[Any] = [memory_type]
        if namespace:
            sql += " AND namespace = ?"
            args.append(namespace)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY updated_at", args).fetchall()
        out = [json.loads(r[0]) for r in rows]
        return [d for d in out if not d.get("deleted")]

    def get_items(self, ids: list[str], memory_type: str) -> list[dict[str, Any]]:
        if not ids:
            return []
        with self._lock:
            marks = ",".join("?" * len(ids))
            rows = self._conn.execute(
                f"SELECT id, data FROM items WHERE memory_type = ? AND id IN ({marks})",
                [memory_type, *ids],
            ).fetchall()
        by_id = {r[0]: json.loads(r[1]) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    # -- evolution log (append-only) -----------------------------------------

    def append(self, ops: list[MemoryOp]) -> None:
        if not ops:
            return
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO evolution_log (op, target_type, target_id, payload, actor, t_transaction)"
                " VALUES (?,?,?,?,?,?)",
                [
                    (
                        o.op.value,
                        o.target_type,
                        o.target_id,
                        json.dumps(o.payload, ensure_ascii=False, default=str),
                        o.actor,
                        o.t_transaction.isoformat(),
                    )
                    for o in ops
                ],
            )

    def tail(self, n: int = 20) -> list[MemoryOp]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT op, target_type, target_id, payload, actor, t_transaction"
                " FROM evolution_log ORDER BY seq DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [MemoryOp.from_row(*r) for r in reversed(rows)]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM evolution_log").fetchone()[0])

    def ops_since(
        self, seq: int, target_type: str | None = None, limit: int = 10000
    ) -> list[tuple[int, MemoryOp]]:
        """Read log entries after ``seq`` in order — the consolidate cursor surface."""
        q = (
            "SELECT seq, op, target_type, target_id, payload, actor, t_transaction"
            " FROM evolution_log WHERE seq > ?"
        )
        args: list = [seq]
        if target_type is not None:
            q += " AND target_type = ?"
            args.append(target_type)
        q += " ORDER BY seq ASC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [(int(r[0]), MemoryOp.from_row(*r[1:])) for r in rows]

    def last_seq(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT MAX(seq) FROM evolution_log").fetchone()
        return int(row[0] or 0)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def episode_to_dict(episode: Episode) -> dict[str, Any]:
    """Serialize an `Episode` to a JSON-ready dict (timestamp as ISO 8601 string)."""
    d = asdict(episode)
    d["timestamp"] = episode.timestamp.isoformat()
    return d
