"""Role-routed LLM client over OpenAI-compatible endpoints.

Roles (extract/distill/judge/rerank/generate) map to independent
endpoint+model pairs so a 0.6B local model can handle extraction while
an API model judges (docs/03 §6 model tiering).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agmem.llm.budget import BudgetTracker

logger = logging.getLogger("agmem.llm")

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

    def __init__(
        self,
        roles: dict[str, RoleConfig],
        budget: BudgetTracker | None = None,
        trace_path: Path | str | None = None,
    ) -> None:
        """`budget` defaults to a fresh `BudgetTracker` when omitted; pass a
        shared one to aggregate cost across multiple `LLMClient` instances.

        `trace_path` (optional) turns on the full-I/O trace sink: when set,
        every `chat()` appends ONE JSON line (the complete prompt + response,
        never truncated) to that file — the re-spend insurance the benchmark
        harness relies on. `None` (the default) keeps behavior unchanged and
        writes nothing, so existing callers are unaffected. It may also be set
        after construction (e.g. `client.trace_path = Path(...)`)."""
        self.roles = roles
        self.budget = budget or BudgetTracker()
        self._clients: dict[str, Any] = {}
        self.trace_path: Path | None = Path(trace_path) if trace_path else None
        self._trace_lock = threading.Lock()

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
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            self.budget.record(budget_key or role, 0, 0, latency_ms, error=True)
            self._trace(
                role, budget_key, cfg.model, messages, "", 0, 0, latency_ms, error=repr(exc)
            )
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        usage = getattr(resp, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0
        self.budget.record(budget_key or role, tokens_in, tokens_out, latency_ms)
        content = resp.choices[0].message.content or ""
        self._trace(
            role, budget_key, cfg.model, messages, content, tokens_in, tokens_out, latency_ms
        )
        return content

    def _trace(
        self,
        role: str,
        budget_key: str | None,
        model: str,
        messages: list[dict[str, str]],
        response_text: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        """Append one JSON line capturing the FULL prompt+response of a single
        `chat()` call (success or failure) to `self.trace_path`. No-op when the
        sink is unset. Prompt/response are never truncated — this file alone
        must let us replay/re-score offline with zero new API calls. Trace-write
        failures are logged, never raised, so they can't break a paid run."""
        if self.trace_path is None:
            return
        line = {
            "ts_iso": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "budget_key": budget_key or role,
            "model": model,
            "messages": messages,
            "response_text": response_text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": round(latency_ms, 3),
            "error": error,
        }
        try:
            payload = json.dumps(line, ensure_ascii=False, default=str) + "\n"
            with self._trace_lock, self.trace_path.open("a", encoding="utf-8") as f:
                f.write(payload)
        except Exception:  # durability best-effort: never break the LLM call
            logger.exception("failed to write LLM trace line (role=%s)", role)
