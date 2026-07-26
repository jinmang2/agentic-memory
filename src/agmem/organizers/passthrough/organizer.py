"""No-op baseline: raw episodes only, zero LLM calls.

This is the control condition for every benchmark comparison — it
measures what plain hybrid retrieval over raw episodes achieves.
"""

from __future__ import annotations

from agmem.core.ops import MemoryOp
from agmem.core.types import Episode
from agmem.organizers.base import Organizer, OrganizerContext


class PassthroughOrganizer(Organizer):
    """Control-condition organizer: every hook is a no-op (see module docstring)."""

    name = "passthrough"

    def on_message(self, episode: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        """Always returns `[]` — the facade already stored/indexed the raw episode."""
        return []  # the facade already stored/indexed the raw episode
