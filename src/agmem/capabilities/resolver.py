"""Pick a concrete adapter for each slot.

Priority: explicit config override > profile default > first candidate
whose ``requires`` is satisfied. A forced-but-unsatisfiable choice
degrades with a logged CapabilityWarning (error only under strict).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Sequence

from agmem.capabilities.detect import HostCapabilities

logger = logging.getLogger("agmem.capabilities")


class CapabilityWarning(UserWarning):
    pass


class ResolutionError(RuntimeError):
    pass


def resolve(
    slot: str,
    candidates: Sequence[type],
    caps: HostCapabilities,
    override: str | None = None,
    profile_default: str | None = None,
    strict: bool = False,
) -> tuple[type, list[str]]:
    """Return (chosen_class, degradation_notes).

    ``candidates`` are ordered by preference (heaviest/most capable first).
    ``override``/``profile_default`` name a candidate class by __name__.
    """
    notes: list[str] = []
    by_name = {c.__name__: c for c in candidates}

    def satisfied(cls: type) -> tuple[bool, str]:
        req = getattr(cls, "requires", None)
        return req.check(caps) if req is not None else (True, "")

    # 1. explicit override
    for wanted, source in (
        (override, "config override"),
        (profile_default, "profile default"),
    ):
        if not wanted:
            continue
        if wanted not in by_name:
            msg = f"[{slot}] {source} '{wanted}' is not a known adapter"
            if strict:
                raise ResolutionError(msg)
            notes.append(msg)
            continue
        cls = by_name[wanted]
        ok, reason = satisfied(cls)
        if ok:
            return cls, notes
        msg = f"[{slot}] {source} '{wanted}' unavailable ({reason}); falling back"
        if strict:
            raise ResolutionError(msg)
        warnings.warn(msg, CapabilityWarning, stacklevel=2)
        logger.warning(msg)
        notes.append(msg)
        break  # override failed -> fall through to capability matching

    # 2. first satisfiable candidate in preference order
    for cls in candidates:
        ok, reason = satisfied(cls)
        if ok:
            return cls, notes
        notes.append(f"[{slot}] {cls.__name__} skipped ({reason})")

    raise ResolutionError(f"[{slot}] no candidate satisfiable: {notes}")
