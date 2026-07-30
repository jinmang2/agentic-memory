"""Every MemMachine eval entry point crashes before construction at the audited SHA.

At 18f1211, `LongTermMemoryParams` is an Annotated discriminated union — a type
expression, not a class — while `evaluation/utils/agent_utils.py` still calls it
like a constructor. Python raises TypeError before pydantic ever sees the kwargs,
so no eval harness at this SHA can have produced the published numbers.

Static half: both source facts asserted in the pinned clone. Dynamic half: the
exact construct replayed with pydantic (version-independent — the call fails at
the typing layer, not in pydantic).

Evidence: docs/research/upstream-defect-catalog.md §9; round-12 `# [memmachine]` #1.
"""

import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from _common import proven, upstream

LTM_PATH = (
    "packages/server/src/memmachine_server/episodic_memory/long_term_memory/long_term_memory.py"
)


def main() -> None:
    root = upstream("MemMachine")
    definition = (root / LTM_PATH).read_text()
    assert re.search(r"LongTermMemoryParams\s*=\s*Annotated\[", definition), (
        "LongTermMemoryParams is no longer the Annotated union"
    )
    harness = (root / "evaluation/utils/agent_utils.py").read_text()
    assert re.search(r"LongTermMemoryParams\s*\(", harness), (
        "the harness no longer calls LongTermMemoryParams(...)"
    )
    proven("static: the harness constructor-calls a type alias that is not a class")

    class DeclarativeBackendParams(BaseModel):
        backend: Literal["declarative"] = "declarative"

    class EventBackendParams(BaseModel):
        backend: Literal["event"] = "event"

    long_term_memory_params = Annotated[
        DeclarativeBackendParams | EventBackendParams, Field(discriminator="backend")
    ]
    try:
        long_term_memory_params(backend="declarative")
    except TypeError as exc:
        assert "not callable" in str(exc), f"unexpected TypeError text: {exc}"
        proven(f"dynamic: calling the Annotated union raises TypeError ({exc})")
        return
    raise AssertionError("the Annotated-union call unexpectedly succeeded")


if __name__ == "__main__":
    main()
