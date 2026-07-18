"""Shared test doubles."""


def make_mem_multi(organizers, llm):
    """make_mem, generalized to a list of organizer instances (Task 12) —
    for scenarios where one organizer's output chains into another's
    on_memory_event (e.g. Nemori episodes -> MemoryOS pages)."""
    from agmem import AgenticMemory
    from agmem.embed.fake import FakeEmbedder

    mem = AgenticMemory(namespace="t", organizers=list(organizers), embedder=FakeEmbedder(dim=128))
    mem.structured = llm
    mem._ctx.llm = llm
    return mem


class StubLLM:
    """StructuredCaller stand-in: returns queued responses per role."""

    def __init__(self, responses: dict[str, list]):
        self.responses = {role: list(items) for role, items in responses.items()}
        self.calls: list[tuple[str, str]] = []
        self.systems: list[str] = []  # system message per call, "" if none
        self.drops: dict[str, int] = {}

    def call(self, role, prompt, schema, required_keys=(), **kwargs):
        self.calls.append((role, prompt))
        self.systems.append(str(kwargs.get("system", "")))
        items = self.responses.get(role)
        if not items:
            self.drops[role] = self.drops.get(role, 0) + 1
            return None
        return items.pop(0)
