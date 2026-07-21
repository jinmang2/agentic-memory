"""Role-routed LLM client over OpenAI-compatible endpoints.

Roles (extract/distill/judge/rerank/generate) map to independent
endpoint+model pairs so a 0.6B local model can handle extraction while
an API model judges (docs/03 §6 model tiering).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agmem.llm.budget import BudgetTracker

ROLES = ("extract", "distill", "judge", "rerank", "generate")


@dataclass
class RoleConfig:
    """Endpoint+model+sampling config for one role (docs/03 §6 tiering).
    `extra_body` is passed through to the OpenAI-compatible request as-is
    (e.g. vLLM's `guided_json`/`chat_template_kwargs`)."""

    endpoint: str
    model: str
    api_key: str = "not-needed"  # local servers ignore it
    temperature: float = 0.1
    max_tokens: int = 1024
    extra_body: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """Routes each role to its own `RoleConfig` and OpenAI-compatible
    endpoint, and records every call (success or failure) into `budget`."""

    def __init__(self, roles: dict[str, RoleConfig], budget: BudgetTracker | None = None) -> None:
        """`budget` defaults to a fresh `BudgetTracker` when omitted; pass a
        shared one to aggregate cost across multiple `LLMClient` instances."""
        self.roles = roles
        self.budget = budget or BudgetTracker()
        self._clients: dict[str, Any] = {}

    def _client_for(self, cfg: RoleConfig) -> Any:
        key = f"{cfg.endpoint}|{cfg.api_key}"
        if key not in self._clients:
            from openai import OpenAI

            self._clients[key] = OpenAI(base_url=cfg.endpoint, api_key=cfg.api_key)
        return self._clients[key]

    def has_role(self, role: str) -> bool:
        """Cheap membership check callers use to skip an optional role
        (e.g. `rerank`) instead of catching the `KeyError` from `chat`."""
        return role in self.roles

    def chat(
        self,
        role: str,
        messages: list[dict[str, str]],
        budget_key: str | None = None,
        **overrides: Any,
    ) -> str:
        """Raises `KeyError` if `role` has no `RoleConfig` — callers that
        want a role to be optional must check `has_role` first. On a
        transport/API exception, the attempt is still recorded in `budget`
        (as an error) before the exception is re-raised — never swallowed.
        Returns `""` if the response has no message content."""
        if role not in self.roles:
            raise KeyError(
                f"no LLM configured for role '{role}' (configured: {sorted(self.roles)})"
            )
        cfg = self.roles[role]
        client = self._client_for(cfg)
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if cfg.extra_body:
            kwargs["extra_body"] = cfg.extra_body
        kwargs.update(overrides)
        # budget_key is a named param, not part of **overrides, so it never
        # reaches the OpenAI API payload above.

        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception:
            self.budget.record(
                budget_key or role, 0, 0, (time.perf_counter() - start) * 1000, error=True
            )
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        usage = getattr(resp, "usage", None)
        self.budget.record(
            budget_key or role,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            latency_ms,
        )
        return resp.choices[0].message.content or ""
