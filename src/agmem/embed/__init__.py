"""Embedder implementations and the resolver's preference order for the slot."""

from agmem.embed.api_embedder import APIEmbedder
from agmem.embed.base import Embedder
from agmem.embed.fake import FakeEmbedder
from agmem.embed.st_embedder import SentenceTransformerEmbedder

# Preference order for the resolver: first satisfiable candidate wins.
# APIEmbedder sits behind FakeEmbedder deliberately — it is the only embedder
# here that costs money per call, so it must be chosen (the `full` profile, an
# explicit --embedder), never inherited because `openai` is installed.
EMBEDDER_CANDIDATES: list[type] = [SentenceTransformerEmbedder, FakeEmbedder, APIEmbedder]

__all__ = [
    "APIEmbedder",
    "Embedder",
    "FakeEmbedder",
    "SentenceTransformerEmbedder",
    "EMBEDDER_CANDIDATES",
]
