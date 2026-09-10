from __future__ import annotations

import math
from pathlib import Path

from .json_io import JsonObject


class RecipeError(ValueError):
    pass


def reject_unknown_keys(raw: JsonObject, allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise RecipeError(f"unknown {where} keys: {', '.join(unknown)}")


def required_str(raw: JsonObject, key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RecipeError(f"{key} must be a non-empty string")
    return value


def required_int(raw: JsonObject, key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RecipeError(f"{key} must be an integer")
    return value


def positive_int(raw: JsonObject, key: str) -> int:
    value = required_int(raw, key)
    if value <= 0:
        raise RecipeError(f"{key} must be positive")
    return value


def required_str_list(raw: JsonObject, key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise RecipeError(f"{key} must be a list of non-empty strings")
    if not value:
        raise RecipeError(f"{key} must not be empty")
    return tuple(item for item in value if isinstance(item, str))


def optional_str_list(raw: JsonObject, key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise RecipeError(f"{key} must be a list of non-empty strings")
    return tuple(item for item in value if isinstance(item, str))


def optional_str(raw: JsonObject, key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RecipeError(f"{key} must be null or a non-empty string")
    return value


def cost(raw: JsonObject, key: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or value < 0
        or not math.isfinite(value)
    ):
        raise RecipeError(f"cost {key} must be null or a finite non-negative number")
    return float(value)


def choice(value: str, allowed: tuple[str, ...], key: str) -> str:
    if value not in allowed:
        raise RecipeError(f"{key} must be one of: {', '.join(allowed)}")
    return value


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def resolve_snapshot_root(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.absolute()
    return (base_dir / path).absolute()
