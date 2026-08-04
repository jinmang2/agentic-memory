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


def test_failed_request_is_counted_then_re_raised():
    """Same contract as LLMClient.chat: a failure still costs an attempt, so it
    is recorded before the exception propagates — never swallowed."""
    api = FakeEmbeddingsAPI()

    def boom(**kwargs):
        raise RuntimeError("429")

    api.create = boom
    e = APIEmbedder(dim=4, client=SimpleNamespace(embeddings=api))
    with pytest.raises(RuntimeError):
        e.embed(["a"])
    assert e.calls == 1 and e.errors == 1 and e.tokens == 0


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
