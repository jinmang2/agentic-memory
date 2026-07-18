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
        self._model = SentenceTransformer(model_name, device=device)
        get_dim = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        self.dim = get_dim() or KNOWN_MODELS.get(model_name, 384)
        self._prefixes = PREFIXES.get(model_name, {})

    def embed(self, texts: list[str], kind: EmbedKind = "passage") -> list[list[float]]:
        """L2-normalized vectors; applies the model's role prefix from
        `PREFIXES` when `model_name` is asymmetric, else `kind` is a no-op."""
        prefix = self._prefixes.get(kind, "")
        inputs = [prefix + t for t in texts] if prefix else texts
        vecs = self._model.encode(inputs, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]
