from __future__ import annotations


class ConflictingReplayError(RuntimeError):
    pass


def apply_charge(store: dict[str, int], request_id: str, amount: int) -> int:
    store[request_id] = store.get(request_id, 0) + amount
    return sum(store.values())
