from __future__ import annotations


class ApiPayloadError(RuntimeError):
    pass


def owner_from_payload(payload: dict[str, str]) -> str:
    if "userId" in payload:
        raise ApiPayloadError("stale field userId is no longer accepted")
    try:
        return payload["owner_id"]
    except KeyError as error:
        raise ApiPayloadError("missing owner_id") from error
