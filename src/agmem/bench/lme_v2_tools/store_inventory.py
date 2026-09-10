from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from agmem.bench.lme_v2_tools.manifest import Fingerprint, FixedStorePlan
from agmem.bench.lme_v2_tools.recipe import FixedStoreRecipe

SNAPSHOT_FILES: Final = (
    Path("memory_config.json"),
    Path("agmem_state.json"),
    Path("agmem/main/memory.db"),
    Path("agmem/main/vectors.db"),
    Path("agmem/main/graph.kuzu"),
)


class StoreInventoryError(ValueError):
    pass


def snapshot_fingerprints(root: Path) -> tuple[Fingerprint, ...]:
    expanded_root = root.expanduser()
    if expanded_root.is_symlink():
        raise StoreInventoryError(f"fixed store root must not be a symlink: {expanded_root}")
    snapshot_root = expanded_root.resolve()
    _check_root(snapshot_root)
    _reject_unallowlisted(snapshot_root)
    return tuple(_fingerprint(snapshot_root, relative) for relative in SNAPSHOT_FILES)


def snapshot_content_sha256(files: tuple[Fingerprint, ...]) -> str:
    hasher = hashlib.sha256()
    for item in files:
        hasher.update(item.path.encode())
        hasher.update(b"\0")
        hasher.update(str(item.size_bytes).encode())
        hasher.update(b"\0")
        hasher.update(item.sha256.encode())
        hasher.update(b"\0")
    return hasher.hexdigest()


def fixed_store_plans(stores: tuple[FixedStoreRecipe, ...]) -> tuple[FixedStorePlan, ...]:
    plans: list[FixedStorePlan] = []
    for store in stores:
        files = snapshot_fingerprints(store.path)
        plans.append(
            FixedStorePlan(
                id=store.id,
                write=store.write,
                path=str(store.path),
                files=files,
                content_sha256=snapshot_content_sha256(files),
            )
        )
    return tuple(plans)


def _check_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise StoreInventoryError(f"fixed store root must be a directory: {root}")


def _reject_unallowlisted(root: Path) -> None:
    allowed = {root / relative for relative in SNAPSHOT_FILES}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise StoreInventoryError(f"fixed store must not contain symlinks: {path}")
        if path.is_file() and path not in allowed:
            raise StoreInventoryError(f"unallowlisted fixed store file: {path}")
        if path.is_dir() and not _is_allowed_dir(root, path):
            raise StoreInventoryError(f"unallowlisted fixed store directory: {path}")


def _is_allowed_dir(root: Path, path: Path) -> bool:
    return any(path in (root / relative).parents for relative in SNAPSHOT_FILES)


def _fingerprint(root: Path, relative: Path) -> Fingerprint:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise StoreInventoryError(f"fixed store file missing: {path}")
    size = path.stat().st_size
    if size == 0:
        raise StoreInventoryError(f"fixed store file empty: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return Fingerprint(path=str(relative), sha256=hasher.hexdigest(), size_bytes=size)
