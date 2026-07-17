#!/usr/bin/env bash
# Run the storage-consuming test files against every real vector engine.
# Usage: bash scripts/test-engine-matrix.sh
set -u
cd "$(dirname "$0")/.."
FILES="tests/test_organizers.py tests/test_organizers_phase3.py \
tests/test_memory.py tests/test_pipeline_p0.py tests/test_locomo.py tests/test_ace.py"
status=0
for engine in SqliteVecStore LanceDBVectorStore QdrantVectorStore ChromaVectorStore; do
    echo "=== engine: $engine ==="
    AGMEM_TEST_VECTOR_STORE=$engine .venv/bin/python -m pytest -q $FILES 2>&1 | tail -2
    [ "${PIPESTATUS[0]}" -ne 0 ] && status=1
done
exit $status
