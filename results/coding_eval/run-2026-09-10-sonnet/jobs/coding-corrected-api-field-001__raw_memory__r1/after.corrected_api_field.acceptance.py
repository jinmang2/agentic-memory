from __future__ import annotations

from .task import ApiPayloadError, owner_from_payload


def main() -> int:
    try:
        owner = owner_from_payload({"owner_id": "owner-7"})
    except ApiPayloadError:
        print("API_FIELD_OWNER_ID_USED")
        return 1
    if owner != "owner-7":
        print("API_FIELD_OWNER_ID_USED")
        return 1
    try:
        _ = owner_from_payload({"userId": "legacy-3"})
    except ApiPayloadError:
        return 0
    print("API_FIELD_USER_ID_REJECTED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
