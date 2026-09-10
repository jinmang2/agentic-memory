"""Strict parser and deterministic ledger for offline cost fixtures."""

from __future__ import annotations

from json import JSONDecodeError
from pathlib import Path
from typing import Final

from .budget import ReservationRequest
from .costs_types import (
    COMPONENT_ORDER,
    COST_SCHEMA_VERSION,
    CostComponent,
    CostInputError,
    UsageEvent,
)
from .json_io import JsonInputError, JsonObject, JsonValue, parse_json_object

ROOT_KEYS: Final = frozenset(
    {
        "schema_version",
        "budget_micro_usd",
        "expected_components",
        "zero_cost_components",
        "events",
        "reservations",
    }
)
EVENT_KEYS: Final = frozenset(
    {
        "event_id",
        "component",
        "base_event_id",
        "is_retry",
        "calls",
        "tokens_in",
        "tokens_out",
        "cost_micro_usd",
    }
)
RESERVATION_KEYS: Final = frozenset({"reservation_id", "estimate_micro_usd", "settlement_event_id"})


def read_cost_fixture(
    path: Path,
) -> tuple[
    tuple[CostComponent, ...],
    tuple[CostComponent, ...],
    tuple[UsageEvent, ...],
    tuple[ReservationRequest, ...],
    int | None,
]:
    """Parse the full cost fixture boundary into typed values."""
    fixture_path = path.expanduser().resolve()
    raw = _read_root(fixture_path)
    _reject_unknown(fixture_path, raw, ROOT_KEYS, "fixture")
    version = _required_int(fixture_path, raw, "schema_version")
    if version != COST_SCHEMA_VERSION:
        raise CostInputError(fixture_path, f"unsupported schema_version: {version}")
    expected = _component_list(fixture_path, raw, "expected_components", COMPONENT_ORDER)
    zero_components = _component_list(fixture_path, raw, "zero_cost_components", ())
    _check_zero_scope(fixture_path, expected, zero_components)
    return (
        expected,
        zero_components,
        _events(fixture_path, raw),
        _reservations(fixture_path, raw),
        _optional_int(fixture_path, raw, "budget_micro_usd"),
    )


def _read_root(path: Path) -> JsonObject:
    try:
        return parse_json_object(path.read_text(encoding="utf-8"), "cost fixture")
    except FileNotFoundError as error:
        raise CostInputError(path, "cost fixture not found") from error
    except JSONDecodeError as error:
        raise CostInputError(path, f"malformed JSON at line {error.lineno}") from error
    except JsonInputError as error:
        raise CostInputError(path, str(error)) from error


def _events(path: Path, raw: JsonObject) -> tuple[UsageEvent, ...]:
    value = raw.get("events")
    if not isinstance(value, list):
        raise CostInputError(path, "events must be a list")
    by_id: dict[str, UsageEvent] = {}
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise CostInputError(path, f"event {index} must be an object")
        event = _event(path, item, index)
        previous = by_id.get(event.event_id)
        if previous is None:
            by_id[event.event_id] = event
        elif previous != event:
            raise CostInputError(path, f"conflicting event_id {event.event_id!r}")
    return tuple(by_id.values())


def _event(path: Path, raw: JsonObject, index: int) -> UsageEvent:
    _reject_unknown(path, raw, EVENT_KEYS, f"event {index}")
    return UsageEvent(
        event_id=_required_str(path, raw, "event_id"),
        component=_component(path, _required_str(path, raw, "component")),
        base_event_id=_optional_str(path, raw, "base_event_id"),
        is_retry=_optional_bool(path, raw, "is_retry"),
        calls=_calls(path, raw),
        tokens_in=_optional_int(path, raw, "tokens_in"),
        tokens_out=_optional_int(path, raw, "tokens_out"),
        cost_micro_usd=_optional_int(path, raw, "cost_micro_usd"),
    )


def _reservations(path: Path, raw: JsonObject) -> tuple[ReservationRequest, ...]:
    value = raw.get("reservations", [])
    if not isinstance(value, list):
        raise CostInputError(path, "reservations must be a list")
    by_id: dict[str, ReservationRequest] = {}
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise CostInputError(path, f"reservation {index} must be an object")
        request = _reservation(path, item, index)
        previous = by_id.get(request.reservation_id)
        if previous is None:
            by_id[request.reservation_id] = request
        elif previous != request:
            raise CostInputError(path, f"conflicting reservation_id {request.reservation_id!r}")
    return tuple(by_id.values())


def _reservation(path: Path, raw: JsonObject, index: int) -> ReservationRequest:
    _reject_unknown(path, raw, RESERVATION_KEYS, f"reservation {index}")
    return ReservationRequest(
        reservation_id=_required_str(path, raw, "reservation_id"),
        estimate_micro_usd=_optional_int(path, raw, "estimate_micro_usd"),
        settlement_event_id=_optional_str(path, raw, "settlement_event_id"),
    )


def _component_list(
    path: Path, raw: JsonObject, key: str, default: tuple[CostComponent, ...]
) -> tuple[CostComponent, ...]:
    value = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, list):
        raise CostInputError(path, f"{key} must be a list")
    components = tuple(_component(path, item) for item in _str_items(path, value, key))
    duplicates = sorted({component for component in components if components.count(component) > 1})
    if duplicates:
        raise CostInputError(path, f"duplicate {key}: {duplicates[0].value}")
    return components


def _check_zero_scope(
    path: Path, expected: tuple[CostComponent, ...], zero_components: tuple[CostComponent, ...]
) -> None:
    extra = sorted(set(zero_components) - set(expected))
    if extra:
        raise CostInputError(path, f"zero_cost_components outside expected scope: {extra[0].value}")


def _reject_unknown(path: Path, raw: JsonObject, allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise CostInputError(path, f"unknown {where} keys: {', '.join(unknown)}")


def _component(path: Path, value: str) -> CostComponent:
    try:
        return CostComponent(value)
    except ValueError as error:
        raise CostInputError(path, f"unknown component: {value}") from error


def _str_items(path: Path, value: list[JsonValue], key: str) -> tuple[str, ...]:
    if not all(isinstance(item, str) for item in value):
        raise CostInputError(path, f"{key} must contain component strings")
    return tuple(item for item in value if isinstance(item, str))


def _calls(path: Path, raw: JsonObject) -> int:
    value = raw.get("calls", 1)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CostInputError(path, "calls must be a non-negative integer")
    return value


def _required_int(path: Path, raw: JsonObject, key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise CostInputError(path, f"{key} must be an integer")
    return value


def _optional_int(path: Path, raw: JsonObject, key: str) -> int | None:
    if key not in raw or raw[key] is None:
        return None
    value = raw[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CostInputError(path, f"{key} must be null or a non-negative integer")
    return value


def _required_str(path: Path, raw: JsonObject, key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CostInputError(path, f"{key} must be a non-empty string")
    return value


def _optional_str(path: Path, raw: JsonObject, key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CostInputError(path, f"{key} must be null or a non-empty string")
    return value


def _optional_bool(path: Path, raw: JsonObject, key: str) -> bool:
    value = raw.get(key, False)
    if not isinstance(value, bool):
        raise CostInputError(path, f"{key} must be a boolean")
    return value
