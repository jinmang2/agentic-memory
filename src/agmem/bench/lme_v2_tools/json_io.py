"""Shared JSON boundary that rejects ambiguous duplicate object keys."""

import json

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class JsonInputError(ValueError):
    """Raised when JSON violates the offline tools' input contract."""


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise JsonInputError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json(text: str) -> JsonValue:
    """Decode a JSON value, rejecting duplicate keys at every nesting level."""
    value: JsonValue = json.loads(text, object_pairs_hook=_unique_object)
    return value


def parse_json_object(text: str, label: str) -> JsonObject:
    """Decode an unambiguous JSON object or raise a labeled input error."""
    value = parse_json(text)
    if not isinstance(value, dict):
        raise JsonInputError(f"{label} root must be an object")
    return value
