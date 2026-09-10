from __future__ import annotations


class ApiPayloadError(RuntimeError):
    pass


def owner_from_payload(payload: dict[str, str]) -> str:
    try:
        return payload["userId"]
    except KeyError as error:
        raise ApiPayloadError("missing userId") from error
