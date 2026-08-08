"""SentenceTransformerEmbedder — how it reaches for the model.

No network and no model file anywhere here: the tests inject a constructor that
records how it was called, because what is under test is the call, not the
weights.

Why this file exists: the lite profile's embedder is constructed on two surfaces
a user waits behind — the MCP server at startup and the capture hook on every
prompt — and measurement on 2026-08-08 showed each construction spending a hub
round trip to revision-check a model already on disk (15.5-16.0 s handshake
online vs 10.8-12.2 s with HF_HUB_OFFLINE=1). Loading cache-first removes that.
The behaviour is invisible from outside — a slower start looks like a slow
start — so it needs a test that watches the constructor arguments.
"""

from __future__ import annotations

import pytest

from agmem.embed.st_embedder import SentenceTransformerEmbedder


class RecordingCtor:
    """Stands in for `SentenceTransformer`, recording each construction.

    `fail_local` makes the cache-only attempt raise, which is what a cold
    machine (nothing downloaded yet) and an older sentence-transformers (no
    such keyword) both look like from here.
    """

    def __init__(self, *, fail_local: bool = False, fail_always: bool = False):
        self.fail_local = fail_local
        self.fail_always = fail_always
        self.calls: list[dict] = []

    def __call__(self, model_name, **kwargs):
        self.calls.append({"model_name": model_name, **kwargs})
        if self.fail_always:
            raise RuntimeError("model is broken")
        if self.fail_local and kwargs.get("local_files_only"):
            raise OSError("not in cache")
        return object()


def test_cached_model_is_loaded_without_touching_the_hub():
    """One construction, and it asks for local files only.

    The assertion that matters is `local_files_only`: without it the load
    succeeds just the same, only slower and only when the network is up, so
    nothing else here would notice its absence.
    """
    ctor = RecordingCtor()
    model = SentenceTransformerEmbedder._load(ctor, "intfloat/multilingual-e5-small", None)

    assert model is not None
    assert len(ctor.calls) == 1
    assert ctor.calls[0]["local_files_only"] is True
    assert ctor.calls[0]["model_name"] == "intfloat/multilingual-e5-small"


def test_uncached_model_falls_back_to_a_normal_load():
    """A cold machine must still get its model — cache-first is an optimization,
    not a requirement, and turning it into one would make a fresh install fail."""
    ctor = RecordingCtor(fail_local=True)
    model = SentenceTransformerEmbedder._load(ctor, "intfloat/multilingual-e5-small", "cpu")

    assert model is not None
    assert len(ctor.calls) == 2
    assert ctor.calls[0]["local_files_only"] is True
    assert "local_files_only" not in ctor.calls[1]
    assert ctor.calls[1]["device"] == "cpu"


def test_a_genuinely_broken_model_still_raises():
    """The fallback must not become a place where real failures go quiet.

    An embedder that returns without a model would produce a store whose
    writes have no vectors — searchable by nothing, and presenting as an empty
    memory rather than as an error.
    """
    ctor = RecordingCtor(fail_always=True)
    with pytest.raises(RuntimeError, match="model is broken"):
        SentenceTransformerEmbedder._load(ctor, "intfloat/multilingual-e5-small", None)

    assert len(ctor.calls) == 2
