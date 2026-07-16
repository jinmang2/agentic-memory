from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

EmbedKind = Literal["query", "passage"]


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str], kind: EmbedKind = "passage") -> list[list[float]]:
        """Embed texts. ``kind`` matters for asymmetric models (e5 family
        needs 'query: '/'passage: ' prefixes); symmetric models ignore it."""
        ...
