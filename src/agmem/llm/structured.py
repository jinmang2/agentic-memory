"""Structured (JSON) output with small-model defenses (docs/03 §6).

Defense layers:
1. Schemas stay flat and small (organizer responsibility).
2. ``guided_json`` is sent via extra_body when the endpoint supports it
   (vLLM); harmless elsewhere.
3. Parse failure -> one retry with the error appended.
4. Final failure -> None plus an explicit drop counter. Never a silent
   skip (the A-Mem lesson).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

from agmem.llm.client import LLMClient

logger = logging.getLogger("agmem.llm")

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object found in ``text`` (handles code fences)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


class StructuredCaller:
    def __init__(self, client: LLMClient, use_guided_json: bool = True) -> None:
        self.client = client
        self.use_guided_json = use_guided_json
        self.drops: dict[str, int] = {}
        self._lock = threading.Lock()

    def _drop(self, role: str, prompt: str, last_output: str) -> None:
        with self._lock:
            self.drops[role] = self.drops.get(role, 0) + 1
        logger.warning(
            "structured output dropped (role=%s, total_drops=%s): %.120s ...",
            role, self.drops.get(role), last_output,
        )

    def call(
        self,
        role: str,
        prompt: str,
        schema: dict[str, Any],
        required_keys: tuple[str, ...] = (),
        max_retries: int = 1,
        system: str = "You must respond with a single JSON object and nothing else.",
    ) -> dict[str, Any] | None:
        """Return the validated dict, or None after an explicit drop."""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": prompt}]
        overrides: dict[str, Any] = {}
        if self.use_guided_json:
            overrides["extra_body"] = {"guided_json": schema}

        last_output = ""
        for attempt in range(max_retries + 1):
            try:
                last_output = self.client.chat(role, messages, **overrides)
            except Exception as exc:  # endpoint/transport error
                logger.warning("LLM call failed (role=%s, attempt=%s): %s", role, attempt, exc)
                if self.use_guided_json and attempt == 0:
                    overrides = {}  # endpoint may reject guided_json — retry without
                    continue
                break
            parsed = extract_json(last_output)
            if parsed is not None and all(k in parsed for k in required_keys):
                return parsed
            messages.append({"role": "assistant", "content": last_output})
            messages.append({
                "role": "user",
                "content": ("Your previous reply was not valid JSON with keys "
                            f"{list(required_keys)}. Respond again with ONLY the JSON object."),
            })
        self._drop(role, prompt, last_output)
        return None
