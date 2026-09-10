from __future__ import annotations

from datetime import datetime


def is_on_or_before_cutoff(timestamp: str, cutoff_date: str) -> bool:
    event_time = datetime.fromisoformat(timestamp)
    cutoff = datetime.fromisoformat(f"{cutoff_date}T23:59:59.999999+00:00")
    return event_time <= cutoff
