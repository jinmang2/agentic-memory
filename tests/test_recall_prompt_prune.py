"""The daemon-less prune keeps a hit whose words are inflected forms of the prompt's.

Found by the coding evaluation (docs/28, 2026-09-10): the prompt "honor the
corrected API field and reject the stale field" kept the stale note (shared
word: "stale") and dropped the correction ("Correction", "rejected") — the one
memory the task was built around. The prune now compares crude stems.
"""

from __future__ import annotations

from agmem.hooks.recall_prompt import _prune, _stems


def _item(text: str, score: float = 1.0) -> dict:
    return {"id": None, "memory_type": "runbooks", "score": score, "timestamp": "", "text": text}


def test_stems_fold_inflection_and_keep_short_words_whole():
    assert _stems("corrected Correction reject rejected") == {"corre", "rejec"}
    assert _stems("field fields") == {"field"}
    # tokens under MIN_TOKEN_CHARS stay out, exactly as before
    assert _stems("API the of") == set()


def test_prune_keeps_the_correction_and_the_stale_note_alike():
    prompt = "Update the parser to honor the corrected API field and reject the stale field."
    correction = _item("Correction: API payloads now use owner_id and userId must be rejected.")
    stale = _item("Stale note: API payloads use userId.", score=0.5)
    unrelated = _item("The build cache lives under .venv and warms in ten seconds.", score=0.9)
    kept = _prune(prompt, [stale, unrelated, correction])
    assert [it["text"][:10] for it in kept] == ["Correction", "Stale note"]


def test_prune_still_drops_hits_that_only_share_stopwords():
    kept = _prune("how do I use the pnpm filter", [_item("the turn that merely contains the")])
    assert kept == []
    # a prompt of short tokens only keeps every hit rather than none
    assert _prune("fix it", [_item("anything")]) == [_item("anything")]


KOREAN_PROMPT = "데이터베이스를 마이그레이션하다가 버그"


def test_korean_stems_are_syllable_bigrams_so_particles_do_not_hide_the_word():
    """Korean words are mostly two or three syllables, under `MIN_TOKEN_CHARS`,
    so the English rule gave a Korean prompt no stems and the prune kept
    everything. Bigrams make "데이터베이스를" share terms with "데이터베이스"."""
    assert _stems("데이터베이스") <= _stems("데이터베이스를")
    assert _stems("버그") == {"버그"} and "버그" in _stems("버그를")
    # one-syllable words are this rule's stopwords, like short English tokens
    assert _stems("이 좀 잘") == set()
    # a mixed-script token: the English half keeps the English rule
    assert _stems("Correction을 rejected") == {"corre", "rejec"}


def test_prune_keeps_the_korean_hit_and_drops_unrelated_korean():
    hit = _item("데이터베이스 마이그레이션 실패")
    unrelated = _item("오늘 점심은 김치찌개를 먹었다", score=2.0)
    assert _prune(KOREAN_PROMPT, [unrelated, hit]) == [hit]
    # short words only: the old rule had no stems here and kept both
    report = _item("버그 리포트 정리")
    assert _prune("버그를 고쳐줘", [unrelated, report]) == [report]


def test_the_daemon_less_fallback_answers_a_korean_prompt(tmp_path):
    """End to end over the doc store: the FTS query and the prune together.
    Before, the whole-token FTS index found nothing for this prompt."""
    from agmem.core.types import Episode
    from agmem.hooks.recall_prompt import fallback_items
    from agmem.stores.sqlite_doc import SqliteDocStore

    store = SqliteDocStore(tmp_path / "memory.db")
    try:
        store.add_episode(Episode(content="데이터베이스 마이그레이션 실패", namespace="t"))
        store.add_episode(Episode(content="오늘 점심은 김치찌개를 먹었다", namespace="t"))
        items = fallback_items(store, "t", KOREAN_PROMPT, 5, None)
    finally:
        store.close()
    assert [it["text"] for it in items] == ["데이터베이스 마이그레이션 실패"]


def test_polite_endings_alone_do_not_keep_an_unrelated_korean_hit():
    """Two polite sentences share 습니 and 니다 whatever they are about; the
    ending bigrams (`HANGUL_STOP_BIGRAMS`) do not count for overlap."""
    from agmem.stores.sqlite_doc import _fts_query

    prompt = "데이터베이스 마이그레이션을 확인했습니다"
    lunch = _item("오늘 점심을 먹었습니다")
    assert _stems(prompt) & _stems(lunch["text"]) == set()
    assert _prune(prompt, [lunch]) == []
    assert '"니다"' not in _fts_query(prompt) and '"데이"' in _fts_query(prompt)


def test_a_stop_bigram_that_begins_a_word_is_still_its_stem():
    """The stoplist drops endings, which sit at a word's end; a run's first
    bigram is its stem whatever the list says. "하지에" keeps 하지, "이다" (a
    name) keeps 이다, and the polite-ending pair above still shares nothing."""
    assert "하지" in _stems("하지에") and _stems("하지 행사 정리") == {"하지", "행사", "정리"}
    assert _stems("이다") == {"이다"} and "이다" in _stems("이다의 신간")
    assert _prune("하지에 뭐 했지", [_item("하지 행사 정리")]) == [_item("하지 행사 정리")]
    assert (
        _stems("데이터베이스 마이그레이션을 확인했습니다") & _stems("오늘 점심을 먹었습니다")
        == set()
    )
