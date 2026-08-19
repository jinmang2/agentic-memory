"""APIEmbedder — the hosted-embedding slot, and the cost accounting that makes
it quotable.

No network anywhere here: every test injects a fake OpenAI-shaped client.

Why this exists at all: `APIEmbedder` was named in the `full` profile
(`config.py`) since before Phase 2 but never implemented, so that profile
silently degraded to sentence-transformers. The embedder diagnostic — does the
9.6 pp gap between our Nemori J and the paper's 73.0 come from MiniLM standing
in for `gemini-embedding-001`? — cannot run without it.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from agmem.bench.registry import get_model, registry_cost_usd_split
from agmem.embed import EMBEDDER_CANDIDATES
from agmem.embed.api_embedder import APIEmbedder


class FakeEmbeddingsAPI:
    """Stands in for `OpenAI().embeddings`, recording what it was asked for."""

    def __init__(self, dim=4, total_tokens=7, vectors=None):
        self.dim = dim
        self.total_tokens = total_tokens
        self.vectors = vectors
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        n = len(kwargs["input"])
        vecs = self.vectors or [[float(i + 1)] * self.dim for i in range(n)]
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=v, index=i) for i, v in enumerate(vecs)],
            usage=SimpleNamespace(total_tokens=self.total_tokens),
        )


def _emb(**kw):
    api = FakeEmbeddingsAPI(
        **{k: v for k, v in kw.items() if k in {"dim", "total_tokens", "vectors"}}
    )
    e = APIEmbedder(
        model_name=kw.get("model_name", "text-embedding-3-small"),
        dim=kw.get("dim", 4),
        client=SimpleNamespace(embeddings=api),
    )
    return e, api


def test_embeds_and_returns_one_vector_per_text():
    e, api = _emb()
    out = e.embed(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == 4 for v in out)
    assert len(api.requests) == 1  # ONE request for the whole list, not one per text
    assert api.requests[0]["input"] == ["a", "b", "c"]
    assert api.requests[0]["model"] == "text-embedding-3-small"


def test_vectors_are_l2_normalized():
    """Cosine comparability is the whole point of this diagnostic.

    Arm B's 0.85 merge threshold is a cosine threshold, and MiniLM's outputs are
    normalized (`normalize_embeddings=True`). If hosted vectors arrived
    un-normalized, every similarity in the run would be on a different scale and
    the two embedders' filter rates would not be comparable — the exact quantity
    being measured. The API already returns unit vectors; normalizing again is
    a no-op there and a guard against a `dimensions`-truncated response.
    """
    e, _ = _emb(vectors=[[3.0, 4.0, 0.0, 0.0]])
    (v,) = e.embed(["a"])
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9)
    assert math.isclose(v[0], 0.6, rel_tol=1e-9)


def test_zero_vector_is_left_alone_not_nan():
    e, _ = _emb(vectors=[[0.0, 0.0, 0.0, 0.0]])
    (v,) = e.embed(["a"])
    assert v == [0.0, 0.0, 0.0, 0.0]


def test_results_are_reordered_by_index_not_trusted_in_arrival_order():
    """The API documents `data` as index-tagged, not order-guaranteed.

    A silent mis-order would attach every item's vector to a different item —
    a corruption that no test of counts or dims would catch, and that would look
    like "the hosted embedder retrieves worse" in exactly the measurement this
    class exists for.
    """
    api = FakeEmbeddingsAPI(dim=2)

    def create(**kwargs):
        api.requests.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.0, 1.0], index=1),
                SimpleNamespace(embedding=[1.0, 0.0], index=0),
            ],
            usage=SimpleNamespace(total_tokens=3),
        )

    api.create = create
    e = APIEmbedder(dim=2, client=SimpleNamespace(embeddings=api))
    assert e.embed(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_empty_input_makes_no_request():
    e, api = _emb()
    assert e.embed([]) == []
    assert api.requests == []
    assert e.calls == 0


def test_counts_calls_and_tokens_for_the_quote():
    e, _ = _emb(total_tokens=11)
    e.embed(["a", "b"])
    e.embed(["c"])
    assert e.calls == 2
    assert e.tokens == 22
    assert e.errors == 0
    assert e.latency_ms_total >= 0.0


def test_failed_request_is_counted_then_re_raised(monkeypatch):
    """Same contract as LLMClient.chat: a failure still costs an attempt, so it
    is recorded before the exception propagates — never swallowed.

    `calls` counts the work the caller asked for; `errors` counts ATTEMPTS that
    failed, so one dead request against the default retry budget reads as one
    call and three errors. The two are deliberately not the same number — a run
    that recovered a blip must not look pristine, and `transport_recoveries`
    says which of the errors were survived."""
    monkeypatch.setattr("time.sleep", lambda _s: None)
    api = FakeEmbeddingsAPI()

    def boom(**kwargs):
        raise RuntimeError("429")

    api.create = boom
    e = APIEmbedder(dim=4, client=SimpleNamespace(embeddings=api))
    with pytest.raises(RuntimeError):
        e.embed(["a"])
    assert e.calls == 1 and e.errors == 3 and e.tokens == 0
    assert e.transport_recoveries == 0, "retries were spent; none of them recovered anything"


def test_kind_is_a_noop_for_a_symmetric_model():
    # No query/passage prefixes: OpenAI embedding models are symmetric, unlike
    # the e5/bge families SentenceTransformerEmbedder prefixes for.
    e, api = _emb()
    e.embed(["a"], kind="query")
    e.embed(["a"], kind="passage")
    assert api.requests[0]["input"] == api.requests[1]["input"] == ["a"]


def test_dimensions_param_is_sent_only_when_shortening():
    # text-embedding-3-* support Matryoshka truncation; asking for the model's
    # native size must not send the parameter (older models reject it).
    e, api = _emb(dim=1536)
    e.embed(["a"])
    assert "dimensions" not in api.requests[0]

    api2 = FakeEmbeddingsAPI(dim=256)
    e2 = APIEmbedder(dim=256, client=SimpleNamespace(embeddings=api2))
    e2.embed(["a"])
    assert api2.requests[0]["dimensions"] == 256


def test_native_dim_is_known_per_model():
    assert APIEmbedder.NATIVE_DIMS["text-embedding-3-small"] == 1536
    e = APIEmbedder(client=SimpleNamespace(embeddings=FakeEmbeddingsAPI(dim=1536)))
    assert e.dim == 1536  # default, no explicit dim
    assert e.name == "text-embedding-3-small"


def test_unknown_model_fails_loud():
    with pytest.raises(KeyError, match="not a known embedding model"):
        APIEmbedder(model_name="text-embedding-9-imaginary", client=SimpleNamespace())


def test_registered_last_in_the_candidate_order():
    """The resolver walks EMBEDDER_CANDIDATES in order and takes the first
    satisfiable one. APIEmbedder must NOT outrank sentence-transformers: it
    costs money per call, so it is opt-in by explicit construction (the `full`
    profile / an explicit --embedder), never something a host happens to pick up
    because a package is installed."""
    assert APIEmbedder in EMBEDDER_CANDIDATES
    names = [c.__name__ for c in EMBEDDER_CANDIDATES]
    assert names.index("SentenceTransformerEmbedder") < names.index("APIEmbedder")


def test_embedding_model_is_priced_in_the_registry():
    spec = get_model("text-embedding-3-small")
    assert spec.usd_per_1m_in == 0.02
    assert spec.usd_per_1m_out == 0.0  # embeddings have no output tokens


def test_embed_role_prices_at_the_embedding_models_rates_not_the_chat_models():
    """Folded into the same budget shape as LLM roles, but priced separately via
    the existing role_models split (the mechanism --judge-model already uses).
    Pricing embed tokens at gpt-4o-mini's rates would overstate them 7.5x."""
    budget = {
        "generate": {"tokens_in": 1_000_000, "tokens_out": 0},
        "embed": {"tokens_in": 1_000_000, "tokens_out": 0},
    }
    cost = registry_cost_usd_split(budget, "gpt-4o-mini", {"embed": "text-embedding-3-small"})
    assert cost == pytest.approx(0.15 + 0.02)


# --------------------------------------------------------------------------
# Runner wiring: selecting the paid embedder, and accounting for what it spent.
# --------------------------------------------------------------------------

import importlib.util as _ilu
import sys
from pathlib import Path

_REPRO_PATH = Path(__file__).resolve().parent.parent / "scripts" / "exp_amem_repro.py"


def _load_repro():
    if str(_REPRO_PATH.parent) not in sys.path:
        sys.path.insert(0, str(_REPRO_PATH.parent))
    spec = _ilu.spec_from_file_location("exp_amem_repro", _REPRO_PATH)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_embedder_picks_sentence_transformers_by_default(monkeypatch):
    H = _load_repro()
    built = {}

    class FakeST:
        def __init__(self, name):
            built["st"] = name
            self.name, self.dim = name, 384

    monkeypatch.setattr(H, "SentenceTransformerEmbedder", FakeST)
    e = H.build_embedder("all-MiniLM-L6-v2")
    assert built["st"] == "all-MiniLM-L6-v2"
    assert not isinstance(e, APIEmbedder)


def test_build_embedder_picks_the_api_embedder_for_a_registered_embedding_model(monkeypatch):
    """`--embedder text-embedding-3-small` must not be handed to
    sentence-transformers, which would try to download it off HuggingFace and
    fail with an unrelated error."""
    H = _load_repro()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    e = H.build_embedder("text-embedding-3-small", client=SimpleNamespace(embeddings=None))
    assert isinstance(e, APIEmbedder)
    assert e.name == "text-embedding-3-small" and e.dim == 1536


def test_build_embedder_requires_the_key_before_spending(monkeypatch):
    H = _load_repro()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        H.build_embedder("text-embedding-3-small")


def test_embed_spend_is_folded_into_the_budget_and_priced_separately():
    """A paid embedder that did not report would leave an unaccounted line in
    every quote that used it — so its spend joins `llm_budget` under `embed` and
    is priced at its own registry rates, not the chat model's."""
    H = _load_repro()
    e = APIEmbedder(
        dim=4, client=SimpleNamespace(embeddings=FakeEmbeddingsAPI(total_tokens=1_000_000))
    )
    e.embed(["a"])
    budget = {"generate": {"calls": 1, "tokens_in": 1_000_000, "tokens_out": 0}}
    H.fold_embed_budget(budget, e)
    assert budget["embed"]["tokens_in"] == 1_000_000
    assert budget["embed"]["tokens_out"] == 0
    assert H.cost_usd(budget, "gpt-4o-mini", embed_model=e.name) == pytest.approx(0.15 + 0.02)


def test_folding_a_free_embedder_adds_nothing():
    H = _load_repro()
    budget = {"generate": {"calls": 1, "tokens_in": 10, "tokens_out": 0}}
    H.fold_embed_budget(budget, object())  # a SentenceTransformerEmbedder has no budget_row
    assert "embed" not in budget


def test_embed_model_name_is_none_for_a_free_embedder():
    """Guards the pricing path against a HuggingFace id reaching get_model,
    which would raise a loud KeyError mid-run — after the money was spent."""
    H = _load_repro()
    from agmem.embed.fake import FakeEmbedder

    assert H.embed_model_name(FakeEmbedder(dim=8)) is None
    e = APIEmbedder(dim=4, client=SimpleNamespace(embeddings=FakeEmbeddingsAPI()))
    assert H.embed_model_name(e) == "text-embedding-3-small"


def test_free_embedder_run_prices_exactly_as_before():
    """The whole wiring must be a no-op for every existing MiniLM run — those
    artifacts are already published and their cost_usd must not move."""
    H = _load_repro()
    from agmem.embed.fake import FakeEmbedder

    budget = {"generate": {"calls": 2, "tokens_in": 1_000_000, "tokens_out": 0}}
    free = FakeEmbedder(dim=8)
    H.fold_embed_budget(budget, free)
    assert H.cost_usd(budget, "gpt-4o-mini", embed_model=H.embed_model_name(free)) == pytest.approx(
        H.cost_usd(budget, "gpt-4o-mini")
    )


# ---------------------------------------------------------------------------
# Transport retries. Written after a 441-sample paid run died at 11:41 on
# 2026-08-17 with `openai.APIConnectionError` raised from a curator dedup
# embedding — the LLM path survives that class of blip and this one did not.
# ---------------------------------------------------------------------------


class FlakyEmbeddingsAPI(FakeEmbeddingsAPI):
    """Fails the first `fail_times` requests the way a dropped link does."""

    def __init__(self, fail_times: int, **kw):
        super().__init__(**kw)
        self.fail_times = fail_times
        self.attempts = 0

    def create(self, **kwargs):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise ConnectionError("Connection error.")
        return super().create(**kwargs)


def test_a_transport_blip_is_retried_rather_than_killing_the_caller(monkeypatch):
    """The failure this exists for: one dropped connection inside a curator's
    dedup embedding took down a run that had 291 samples left to answer.

    The LLM path already treats a connection failure as a re-send rather than a
    verdict (`StructuredCaller.transport_retries`), because no reply came back
    and there is nothing for the model to correct. An embedding request is the
    same shape of work and gets the same treatment."""
    monkeypatch.setattr("time.sleep", lambda _s: None)  # no real backoff in tests
    api = FlakyEmbeddingsAPI(fail_times=1, dim=4)
    e = APIEmbedder(dim=4, client=SimpleNamespace(embeddings=api))

    out = e.embed(["a"])

    assert len(out) == 1, "the caller gets its vector"
    assert api.attempts == 2, "the request was sent again, not abandoned"
    assert e.errors == 1, "the blip is still recorded — a recovered run is not a pristine one"
    assert e.transport_recoveries == 1


def test_retries_are_a_budget_and_the_error_still_propagates_when_it_runs_out(monkeypatch):
    """A link that is down, rather than blipping, must still surface. Silently
    returning a zero vector would poison the store with a neighbourless bullet
    and read as a retrieval defect months later."""
    monkeypatch.setattr("time.sleep", lambda _s: None)
    api = FlakyEmbeddingsAPI(fail_times=99, dim=4)
    e = APIEmbedder(dim=4, client=SimpleNamespace(embeddings=api), transport_retries=2)

    with pytest.raises(ConnectionError):
        e.embed(["a"])
    assert api.attempts == 3, "the original attempt plus its two retries"
    assert e.errors == 3, "every attempt cost latency and is counted"
    assert e.transport_recoveries == 0, "a recovery is a retry that worked, and none did"


def test_retries_are_off_by_configuration_for_a_replay(monkeypatch):
    """`transport_retries=0` restores the pre-fix behaviour exactly, so an arm
    measured before this parameter existed can be replayed without it."""
    monkeypatch.setattr("time.sleep", lambda _s: None)
    api = FlakyEmbeddingsAPI(fail_times=1, dim=4)
    e = APIEmbedder(dim=4, client=SimpleNamespace(embeddings=api), transport_retries=0)
    with pytest.raises(ConnectionError):
        e.embed(["a"])
    assert api.attempts == 1


def test_a_survived_blip_reaches_the_run_artifact(monkeypatch):
    """A recovered wobble has to land in `budget_row`, or it is a counter that
    only a log sees — which is what `StructuredCaller.transport_recoveries` has
    been since Track 2: written on every retry, read by nothing, and named for
    an outcome it does not check. A clean run keeps the row shape the other
    roles have; only a run that actually wobbled grows the key."""
    monkeypatch.setattr("time.sleep", lambda _s: None)
    clean, _ = _emb()
    clean.embed(["a"])
    assert "transport_recoveries" not in clean.budget_row()

    api = FlakyEmbeddingsAPI(fail_times=1, dim=4)
    wobbled = APIEmbedder(dim=4, client=SimpleNamespace(embeddings=api))
    wobbled.embed(["a"])
    row = wobbled.budget_row()
    assert row["transport_recoveries"] == 1
    assert row["calls"] == 1 and row["errors"] == 1


def test_an_oversized_input_is_truncated_rather_than_failing_the_batch():
    """The hosted models stop at 8192 tokens and answer a longer input with a
    400 — a request-shaped failure the transport retry cannot fix, which spends
    the attempt three times and then takes every other text in the batch down
    with it. On LongMemEval `_s` that was 5 turns out of 246,750 costing 5 of 500
    instances. Only the vector is computed from the prefix; the stored content is
    untouched, so a retrieved item still renders whole."""

    class _Recorder:
        def __init__(self):
            self.seen = None
            self.embeddings = SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            self.seen = kwargs["input"]
            return SimpleNamespace(
                data=[
                    SimpleNamespace(embedding=[1.0, 0.0], index=i)
                    for i, _ in enumerate(kwargs["input"])
                ],
                usage=SimpleNamespace(prompt_tokens=1),
            )

    rec = _Recorder()
    emb = APIEmbedder(model_name="text-embedding-3-small", client=rec, dim=2)
    huge = "x" * 76_591  # the longest single turn in longmemeval_s_cleaned
    emb.embed(["short one", huge])

    assert rec.seen[0] == "short one"  # untouched
    assert len(rec.seen[1]) == emb.max_input_chars < len(huge)
    assert emb.truncations == 1
    assert emb.budget_row()["input_truncations"] == 1


def test_nothing_is_truncated_when_nothing_is_oversized():
    class _Ok:
        def __init__(self):
            self.embeddings = SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    data=[
                        SimpleNamespace(embedding=[1.0, 0.0], index=i)
                        for i, _ in enumerate(kw["input"])
                    ],
                    usage=SimpleNamespace(prompt_tokens=1),
                )
            )

    emb = APIEmbedder(model_name="text-embedding-3-small", client=_Ok(), dim=2)
    emb.embed(["a", "b"])
    assert emb.truncations == 0
    assert "input_truncations" not in emb.budget_row()


class CeilingAPI(FakeEmbeddingsAPI):
    """Rejects any input longer than `limit` chars, the way the real endpoint
    rejects anything over 8192 TOKENS — a ceiling our character budget only
    estimates."""

    def __init__(self, limit, **kw):
        super().__init__(**kw)
        self.limit = limit
        self.attempts: list[int] = []

    def create(self, **kwargs):
        longest = max(len(t) for t in kwargs["input"])
        self.attempts.append(longest)
        if longest > self.limit:
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': \"Invalid 'input[0]': "
                'maximum input length is 8192 tokens."}}'
            )
        return super().create(**kwargs)


def _ceiling_emb(limit, **kw):
    api = CeilingAPI(limit, dim=4)
    e = APIEmbedder(
        model_name="text-embedding-3-small", dim=4, client=SimpleNamespace(embeddings=api)
    )
    return e, api


def test_a_length_400_shrinks_the_input_instead_of_losing_the_instance(monkeypatch):
    """The 3.0 chars-per-token FLOOR is an assumption about the text, and
    LongMemEval `_m` breaks it: a turn that fits 24,576 chars need not fit 8192
    tokens. Before this, the three transport retries re-sent identical bytes to a
    deterministic 400 and the whole instance died — 2 of the first 50 `_m` rows."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    e, api = _ceiling_emb(limit=5000)
    out = e.embed(["x" * 24_000])
    assert len(out) == 1 and len(out[0]) == 4
    # 24000 -> capped to 24576 (no-op) -> 12288 -> 6144 -> 3072, first fit
    assert api.attempts == [24_000, 12_288, 6_144, 3_072]


def test_shrinking_does_not_spend_the_transport_retries(monkeypatch):
    """Nothing about the transport failed, so a link that blips AFTER a shrink
    must still get its full retry budget."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    e, _api = _ceiling_emb(limit=5000)
    e.embed(["x" * 24_000])
    assert e.transport_retries == 2  # untouched
    assert e.errors == 3  # the three rejections are still on the record


def test_a_plain_400_is_not_mistaken_for_the_ceiling(monkeypatch):
    """An over-broad match would halve forever against an unrelated 400."""
    monkeypatch.setattr("time.sleep", lambda *_: None)

    class Always400(FakeEmbeddingsAPI):
        def create(self, **kwargs):
            self.requests.append(kwargs)
            raise RuntimeError("Error code: 400 - unknown parameter 'dimensions'")

    api = Always400()
    e = APIEmbedder(
        model_name="text-embedding-3-small", dim=4, client=SimpleNamespace(embeddings=api)
    )
    with pytest.raises(RuntimeError, match="unknown parameter"):
        e.embed(["hello"])
    assert len(api.requests) == 3  # one attempt plus the two transport retries, no halving


def test_an_input_that_fits_is_sent_byte_identically(monkeypatch):
    """`_s`'s 5 truncations succeeded on the first attempt and must still do so:
    this path may not change what an already-working run puts on the wire."""
    e, api = _ceiling_emb(limit=1_000_000)
    e.embed(["ordinary prose", "x" * 30_000])
    assert api.attempts == [24_576]  # capped once, sent once, never re-sent
    assert api.requests[0]["input"] == ["ordinary prose", "x" * 24_576]
    assert e.truncations == 1


def test_an_oversized_batch_is_split_not_truncated_wholesale(monkeypatch):
    """A batch cannot say WHICH input the API rejected. Halving the character
    budget for all of them would shorten innocent texts that were about to
    succeed, changing their vectors for no reason — so the batch splits until the
    offender is alone and only it pays."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    api = CeilingAPI(limit=5000, dim=4)
    e = APIEmbedder(
        model_name="text-embedding-3-small", dim=4, client=SimpleNamespace(embeddings=api)
    )
    innocent, offender = "a" * 4_000, "b" * 20_000
    out = e.embed([innocent, innocent, offender, innocent])
    assert len(out) == 4
    # every request that succeeded carried the innocents at full length
    sent = [r["input"] for r in api.requests if max(len(t) for t in r["input"]) <= 5000]
    assert all(t == innocent for req in sent for t in req if t.startswith("a"))
    # and the offender was cut alone, not alongside them
    cut = [
        r["input"] for r in api.requests if len(r["input"]) == 1 and r["input"][0].startswith("b")
    ]
    assert cut and len(cut[-1][0]) < 20_000


def test_a_single_oversized_text_still_halves_rather_than_splitting(monkeypatch):
    """Splitting a one-item batch would recurse forever; the char budget is the
    only lever left."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    e, api = _ceiling_emb(limit=5000)
    e.embed(["x" * 24_000])
    assert api.attempts == [24_000, 12_288, 6_144, 3_072]
