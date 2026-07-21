"""Experimental organizer compositions — NOT paper reproductions.

Everything here is a construction *we* added on top of the paper-faithful
organizers, with no counterpart in the A-Mem, MemoryOS, or Nemori papers or
their upstream code. It lives in its own package so the faithful organizers
stay pure and the fidelity boundary is explicit. Items graduate to the core
package only when an E2E measurement justifies promotion (docs/13 §5, spec
2026-07-21-organizer-experimental-split).

- ``ChainedConsumer``: feed one organizer another organizer's episodes
  (cross-organizer stacking, e.g. Nemori episodes -> A-Mem notes).
- ``SemanticOfflineConsolidator``: Nemori's ``consolidation="semantic_offline"``
  deferred/offline consolidation pass (absent from the Nemori paper and
  upstream — v4 §3.3.3 consolidation is inline). The three-way decision it
  reuses (``ThreeWayIntegrator``) is paper-faithful and lives in the core
  ``nemori_stages`` (corrected 2026-07-21 fidelity review).
"""

from __future__ import annotations

from agmem.organizers.experimental.chained import ChainedConsumer
from agmem.organizers.experimental.nemori_mixing import SemanticOfflineConsolidator

__all__ = ["ChainedConsumer", "SemanticOfflineConsolidator"]
