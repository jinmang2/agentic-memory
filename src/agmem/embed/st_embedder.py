"""sentence-transformers embedder (standard/lite-with-GPU profiles)."""

from __future__ import annotations

from agmem.capabilities.requires import Requires
from agmem.embed.base import EmbedKind

# model name -> output dim
KNOWN_MODELS = {
    "BAAI/bge-small-en-v1.5": 384,
    "intfloat/multilingual-e5-small": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-m3": 1024,
}

# Asymmetric models require role prefixes for proper retrieval quality.
PREFIXES: dict[str, dict[EmbedKind, str]] = {
    "intfloat/multilingual-e5-small": {"query": "query: ", "passage": "passage: "},
    "intfloat/multilingual-e5-base": {"query": "query: ", "passage": "passage: "},
    "intfloat/multilingual-e5-large": {"query": "query: ", "passage": "passage: "},
    # bge v1.5: query-side instruction only
    "BAAI/bge-small-en-v1.5": {
        "query": "Represent this sentence for searching relevant passages: ",
        "passage": "",
    },
}


class SentenceTransformerEmbedder:
    """`Embedder` backed by a local sentence-transformers model (the `lite`/
    `standard` profile default). Gated by `requires` — the import is deferred
    to `__init__` so the module loads even without the package installed."""

    requires = Requires(python_pkgs=("sentence_transformers",))

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str | None = None,
    ) -> None:
        """`device=None` lets sentence-transformers auto-select (GPU if
        available). `dim` comes from the model itself when it exposes one,
        else `KNOWN_MODELS`, else a hardcoded 384 guess."""
        from sentence_transformers import SentenceTransformer  # gated by requires

        self.name = model_name
        self._model = self._load(SentenceTransformer, model_name, device)
        get_dim = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        self.dim = get_dim() or KNOWN_MODELS.get(model_name, 384)
        self._prefixes = PREFIXES.get(model_name, {})

    @staticmethod
    def _load(ctor, model_name: str, device: str | None):
        """Construct the model from cache when possible, from the hub when not.

        Measured 2026-08-08 against the shipped `agmem-mcp` over stdio: the
        client handshake took 15.5-16.0 s with the network reachable and
        10.8-12.2 s with `HF_HUB_OFFLINE=1`, for a model already on disk in
        both runs. The difference is a revision check — sentence-transformers
        asks the hub whether the cached snapshot is current every time a model
        is constructed, and pays a round trip to be told it is.

        That cost lands on the two surfaces a user actually feels: the MCP
        server, which pays it before it can answer anything, and the capture
        hook, which pays it per prompt. Neither wants it. A local memory store
        that gets slower when the network is bad, and would stall behind a
        hanging DNS resolver, has the dependency backwards.

        So: ask for the cache first, and fall back to the ordinary path when
        that raises — which is what happens on a cold machine (nothing cached),
        and also on a sentence-transformers old enough not to accept the
        keyword. The fallback deliberately does not swallow anything: whatever
        the second attempt raises reaches the caller, so a genuinely broken
        model still fails loudly rather than as a mysteriously empty store.
        """
        try:
            return ctor(model_name, device=device, local_files_only=True)
        except Exception:
            return ctor(model_name, device=device)

    def embed(self, texts: list[str], kind: EmbedKind = "passage") -> list[list[float]]:
        """L2-normalized vectors; applies the model's role prefix from
        `PREFIXES` when `model_name` is asymmetric, else `kind` is a no-op."""
        prefix = self._prefixes.get(kind, "")
        inputs = [prefix + t for t in texts] if prefix else texts
        vecs = self._model.encode(inputs, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]
