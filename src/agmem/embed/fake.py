"""Deterministic hashing embedder — dependency-free fallback and test double.

Hashed bag-of-words: shared tokens produce genuinely similar vectors, so
retrieval tests exercise real ranking behavior without a model download.
Not suitable for benchmark runs (resolver places it last).
"""

from __future__ import annotations

import hashlib
import math
import re

from agmem.capabilities.requires import Requires

_TOKEN = re.compile(r"\w+", re.UNICODE)


class FakeEmbedder:
    requires = Requires()

    def __init__(self, dim: int = 256) -> None:
        self.name = f"fake-hash-{dim}"
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.dim

    def embed(self, texts: list[str], kind: str = "passage") -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in _TOKEN.findall(text.lower()):
                vec[self._bucket(token)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out
