"""Nemori methodology: the organizer plus the stage strategies it owns.

Split into a subpackage because ``organizers/`` holds one Organizer per module and
nothing else, while Nemori's fidelity switch needs its boundary/merge/integration
stages as separate swappable classes (docs/11 §4). Those stages are Nemori's own
paper mechanisms — the 2026-07-21 fidelity review's N1 fix turned specifically on
``ThreeWayIntegrator`` belonging to the faithful core rather than to
``experimental`` — so they are co-located with their owner instead of sitting
loose next to unrelated methodologies.

``NemoriOrganizer`` is re-exported here, so ``from agmem.organizers.nemori import
NemoriOrganizer`` resolves exactly as it did when this was a single module.
"""

from agmem.organizers.nemori.organizer import NemoriOrganizer

__all__ = ["NemoriOrganizer"]
