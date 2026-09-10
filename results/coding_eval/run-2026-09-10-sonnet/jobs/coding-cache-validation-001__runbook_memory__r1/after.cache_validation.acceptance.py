from __future__ import annotations

from .task import authorize


def main() -> int:
    cache = {"u1": "old-token"}
    if authorize(cache, "u1", "fresh-token", set()):
        pass
    else:
        print("CACHE_VALIDATES_CURRENT_TOKEN")
        return 1
    if authorize(cache, "u1", "revoked-token", {"revoked-token"}):
        print("CACHE_DOES_NOT_BYPASS_REVOCATION")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
