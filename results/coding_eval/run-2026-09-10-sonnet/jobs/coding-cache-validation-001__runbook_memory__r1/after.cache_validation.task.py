from __future__ import annotations


def authorize(cache: dict[str, str], user_id: str, token: str, revoked_tokens: set[str]) -> bool:
    if token in revoked_tokens:
        cache.pop(user_id, None)
        return False
    if cache.get(user_id) == token:
        return True
    cache[user_id] = token
    return True
