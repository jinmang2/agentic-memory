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
