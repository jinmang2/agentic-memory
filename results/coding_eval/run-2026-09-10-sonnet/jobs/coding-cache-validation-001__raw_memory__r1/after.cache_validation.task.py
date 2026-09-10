from __future__ import annotations


def authorize(cache: dict[str, str], user_id: str, token: str, revoked_tokens: set[str]) -> bool:
    if token in revoked_tokens:
        return False
    cache[user_id] = token
    return True
