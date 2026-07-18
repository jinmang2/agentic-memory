"""Embedder implementations and the resolver's preference order for the slot."""

from agmem.embed.base import Embedder
from agmem.embed.fake import FakeEmbedder
from agmem.embed.st_embedder import SentenceTransformerEmbedder

# Preference order for the resolver.
EMBEDDER_CANDIDATES: list[type] = [SentenceTransformerEmbedder, FakeEmbedder]

__all__ = [
    "Embedder",
    "FakeEmbedder",
    "SentenceTransformerEmbedder",
    "EMBEDDER_CANDIDATES",
]
