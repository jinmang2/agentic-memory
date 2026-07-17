"""Test harness knobs.

``AGMEM_TEST_VECTOR_STORE=<AdapterClassName>`` re-points every profile's
vector_store slot so the whole suite exercises a real engine — used by
scripts/test-engine-matrix.sh to run organizer/memory/pipeline tests on
Qdrant(local)/ChromaDB/LanceDB in addition to the sqlite-vec default.
"""

import os

_engine = os.environ.get("AGMEM_TEST_VECTOR_STORE")
if _engine:
    from agmem.config import PROFILES

    for _prof in PROFILES.values():
        _prof["vector_store"] = _engine
