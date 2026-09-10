from __future__ import annotations

from datetime import datetime, timedelta


def is_on_or_before_cutoff(timestamp: str, cutoff_date: str) -> bool:
    event_time = datetime.fromisoformat(timestamp)
    cutoff_start = datetime.fromisoformat(f"{cutoff_date}T00:00:00+00:00")
    next_day_start = cutoff_start + timedelta(days=1)
    return event_time < next_day_start
