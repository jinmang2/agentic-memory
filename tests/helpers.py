"""Shared test doubles."""


class StubLLM:
    """StructuredCaller stand-in: returns queued responses per role."""

    def __init__(self, responses: dict[str, list]):
        self.responses = {role: list(items) for role, items in responses.items()}
        self.calls: list[tuple[str, str]] = []
        self.systems: list[str] = []  # system message per call, "" if none
        self.drops: dict[str, int] = {}

    def call(self, role, prompt, schema, required_keys=(), **kw):
        self.calls.append((role, prompt))
        self.systems.append(str(kw.get("system", "")))
        items = self.responses.get(role)
        if not items:
            self.drops[role] = self.drops.get(role, 0) + 1
            return None
        return items.pop(0)
