"""API-free counting LLM for dry-run quotes. Drop-in for LLMClient.chat: counts
calls per role and returns schema-valid canned JSON per organizer profile, so an
organizer's full write/read path executes with zero spend and the call counts are
REAL (branchy organizers — eviction, merge — cannot be quoted by per-turn
constants; see the 2026-07-31 readiness audit). Promoted from
tests/test_repro_artifacts.py::_FakeCountingLLM."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agmem import AgenticMemory
from agmem.config import AgmemConfig
from agmem.embed.fake import FakeEmbedder
from agmem.llm.client import RoleConfig
from agmem.llm.structured import StructuredCaller


def _amem_canned(role: str, prompt: str) -> str:
    if role == "extract":
        if "generate several keywords" in prompt:
            return '{"keywords": "alpha, beta"}'
        return '{"keywords": ["k1"], "context": "ctx", "tags": ["t1", "t2", "t3"]}'
    if role == "distill":
        return '{"should_evolve": false, "connections": []}'
    return "stub answer"


CANNED_RESPONSES: dict[str, Callable[[str, str], str]] = {
    "amem": _amem_canned,
    # "nemori" / "mem0" / "zep": registered by their track plans, with responses valid
    # against each organizer's structured-output schemas (else call branching is wrong).
}


class CountingLLM:
    """Counts ``chat()`` calls per role and returns canned, schema-valid JSON —
    no network, no API key, no spend."""

    def __init__(self, canned: str):
        if canned not in CANNED_RESPONSES:
            raise KeyError(
                f"no canned-response profile {canned!r} (known: {sorted(CANNED_RESPONSES)})"
            )
        self._canned = CANNED_RESPONSES[canned]
        self.calls: dict[str, int] = {}

    def chat(self, role, messages, budget_key=None, **overrides) -> str:
        self.calls[role] = self.calls.get(role, 0) + 1
        prompt = " ".join(m.get("content", "") for m in messages)
        return self._canned(role, prompt)


def build_counting_memory(
    canned: str,
    organizers_factory: Callable[[], list[Any]],
    data_dir: Any,
    namespace: str,
    memory_types: tuple[str, ...],
) -> tuple[AgenticMemory, CountingLLM]:
    """Build an ``AgenticMemory`` wired to a ``CountingLLM`` for ``canned`` — the
    full organizer write/read path runs for real, API-free, so the resulting
    call counts are real call counts, not per-turn constants."""
    cfg = AgmemConfig(
        profile="lite",
        data_dir=data_dir,
        llm_roles={"extract": RoleConfig(endpoint="x", model="m")},
        use_guided_json=False,
        sync_write=True,
        lexical_types=("episodic",),
    )
    mem = AgenticMemory(
        namespace=namespace,
        organizers=organizers_factory(),
        embedder=FakeEmbedder(dim=64),
        config=cfg,
    )
    fake = CountingLLM(canned)
    mem.llm = fake
    mem.structured = StructuredCaller(fake, use_guided_json=False)
    mem._ctx.llm = mem.structured  # organizer writes through the fake too
    return mem, fake
