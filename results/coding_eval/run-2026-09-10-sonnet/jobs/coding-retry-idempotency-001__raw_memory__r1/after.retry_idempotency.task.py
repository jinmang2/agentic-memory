from __future__ import annotations


class ConflictingReplayError(RuntimeError):
    pass


def apply_charge(store: dict[str, int], request_id: str, amount: int) -> int:
    if request_id in store:
        if store[request_id] != amount:
            raise ConflictingReplayError(
                f"request {request_id!r} replayed with different amount"
            )
    else:
        store[request_id] = amount
    return sum(store.values())
