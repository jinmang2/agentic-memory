from __future__ import annotations

from .task import ConflictingReplayError, apply_charge


def main() -> int:
    store: dict[str, int] = {}
    first_total = apply_charge(store, "req-1", 500)
    replay_total = apply_charge(store, "req-1", 500)
    if first_total != 500 or replay_total != 500:
        print("RETRY_IDEMPOTENCY_SINGLE_CHARGE")
        return 1
    try:
        _ = apply_charge(store, "req-1", 700)
    except ConflictingReplayError:
        return 0
    print("RETRY_IDEMPOTENCY_REJECTS_CONFLICT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
