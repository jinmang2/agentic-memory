"""Brute-force numpy vector store — TEST FIXTURE ONLY.

Excluded from VECTOR_STORE_CANDIDATES: naive in-python engines are
banned as runtime defaults (docs/03 §5 정책 — the study's point is the
real backends). Kept as the simplest VectorStore protocol reference for
the test suite. Exact cosine; persists to a single .npz.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from agmem.capabilities.requires import Requires


class NumpyVectorStore:
    requires = Requires()  # numpy is a hard dependency of agmem itself

    def __init__(self, path: str | Path | None = None, dim: int = 384) -> None:
        self.path = Path(path) if path else None
        self.dim = dim
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._meta: list[tuple[str, str]] = []  # (namespace, memory_type)
        self._vecs = np.empty((0, dim), dtype=np.float32)
        self._pos: dict[str, int] = {}
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        data = np.load(self.path, allow_pickle=False)
        self._vecs = data["vecs"].astype(np.float32)
        if self._vecs.shape[1] != self.dim:
            raise ValueError(
                f"vector index was built with dim={self._vecs.shape[1]}, got dim={self.dim} — "
                "changing embedders requires rebuilding the collection (docs/03 §1.2)"
            )
        meta = json.loads(str(data["meta"]))
        self._ids = meta["ids"]
        self._meta = [tuple(m) for m in meta["meta"]]
        self._pos = {i: n for n, i in enumerate(self._ids)}

    def add(
        self,
        item_id: str,
        embedding: list[float],
        memory_type: str = "episodic",
        namespace: str = "main",
    ) -> None:
        if len(embedding) != self.dim:
            raise ValueError(f"embedding dim {len(embedding)} != store dim {self.dim}")
        vec = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        with self._lock:
            if item_id in self._pos:
                idx = self._pos[item_id]
                self._vecs[idx] = vec
                self._meta[idx] = (namespace, memory_type)
            else:
                self._pos[item_id] = len(self._ids)
                self._ids.append(item_id)
                self._meta.append((namespace, memory_type))
                self._vecs = np.vstack([self._vecs, vec[None, :]])

    def search(
        self,
        embedding: list[float],
        k: int = 10,
        memory_type: str | None = None,
        namespace: str | None = None,
    ) -> list[tuple[str, float]]:
        with self._lock:
            if not self._ids:
                return []
            q = np.asarray(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(q))
            if norm > 0:
                q = q / norm
            sims = self._vecs @ q  # rows are pre-normalized -> cosine
            order = np.argsort(-sims)
            out: list[tuple[str, float]] = []
            for idx in order:
                ns, mt = self._meta[idx]
                if namespace and ns != namespace:
                    continue
                if memory_type and mt != memory_type:
                    continue
                out.append((self._ids[idx], float(sims[idx])))
                if len(out) >= k:
                    break
            return out

    def get(self, ids: list[str]) -> dict[str, list[float]]:
        with self._lock:
            return {i: self._vecs[self._pos[i]].tolist() for i in ids if i in self._pos}

    def delete(self, ids: list[str]) -> None:
        with self._lock:
            drop = {i for i in ids if i in self._pos}
            if not drop:
                return
            keep = [n for n, i in enumerate(self._ids) if i not in drop]
            self._vecs = self._vecs[keep]
            self._ids = [self._ids[n] for n in keep]
            self._meta = [self._meta[n] for n in keep]
            self._pos = {i: n for n, i in enumerate(self._ids)}

    def count(self) -> int:
        with self._lock:
            return len(self._ids)

    def persist(self) -> None:
        if not self.path:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.path,
                vecs=self._vecs,
                meta=json.dumps({"ids": self._ids, "meta": self._meta}),
            )

    def close(self) -> None:
        self.persist()
