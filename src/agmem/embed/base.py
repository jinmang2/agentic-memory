"""The `Embedder` slot Protocol every embedder implementation satisfies."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

EmbedKind = Literal["query", "passage"]


@runtime_checkable
class Embedder(Protocol):
    """`name` and `dim` must be set by `__init__` and stay constant for the
    instance's lifetime — callers size vector stores off `dim` once."""

    name: str
    dim: int

    def embed(self, texts: list[str], kind: EmbedKind = "passage") -> list[list[float]]:
        """Embed texts. ``kind`` matters for asymmetric models (e5 family
        needs 'query: '/'passage: ' prefixes); symmetric models ignore it."""
        ...
