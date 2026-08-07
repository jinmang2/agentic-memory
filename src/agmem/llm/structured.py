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
import time
from typing import Any

from agmem.llm.client import LLMClient

logger = logging.getLogger("agmem.llm")

_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | list | None:
    """Parse the first JSON value found in ``text`` (handles code fences)."""
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


def coerce_to_schema(parsed: Any, schema: dict[str, Any]) -> dict[str, Any] | None:
    """Schema-guided repair for common small-model deviations.

    A frequent 0.5B failure: returning the bare array when the schema is an
    object with a single array property (observed with Qwen3-0.6B). Wrap it.
    """
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        props = schema.get("properties", {})
        array_keys = [k for k, v in props.items() if v.get("type") == "array"]
        if len(array_keys) == 1:
            return {array_keys[0]: parsed}
    return None


class StructuredCaller:
    """Wraps `LLMClient` with the retry/repair/drop defense chain described
    in the module docstring; `call()` is the only public entry point."""

    def __init__(
        self,
        client: LLMClient,
        use_guided_json: bool = True,
        transport_retries: int = 2,
        reply_retries: int = 1,
    ) -> None:
        """`use_guided_json=False` skips the `guided_json` extra_body layer
        entirely (e.g. for endpoints that reject unknown fields outright).

        `transport_retries` is the budget for CONNECTION failures — timeouts,
        DNS blips, resets — and is deliberately separate from `call`'s
        `max_retries`, which exists for malformed model REPLIES. They are
        different failures with different fixes: a malformed reply earns a
        correction turn appended to the conversation, while a transport failure
        produced no reply at all and simply needs the same request sent again.

        Sized from a real incident (2026-08-04): on a flaky link a single blip
        anywhere in a ~400-call conversation dropped one structured call, which
        cost the entire conversation's ingest — the organizer lost that piece of
        write-path work, the conversation failed its clean-ingest check, and
        every call already paid for was re-spent. Two retries with backoff turn
        that into a pause. Set to 0 to restore the previous drop-immediately
        behavior.

        `reply_retries` is the same economics applied to the OTHER half of the
        chain, and it took a second incident to notice the asymmetry. The
        2026-08-07 Zep pilot issued 2,177 structured calls in one conversation;
        exactly one reply came back malformed, its single correction turn also
        failed, and the drop cost the whole conversation — 45 minutes and $0.31
        re-spent, on a gate that admits no drops. At that call volume a
        per-call malformed rate of 1/2177 makes a clean conversation a **37%**
        event, so the default of one correction turn is not sized for arms of
        this shape. Raising it is close to free: a correction turn is issued
        only when a reply already failed to parse, so an arm that never
        malforms never pays for it.

        It is a DEFAULT, not a policy: `call(max_retries=...)` still wins, and
        the default stays 1 so every arm measured before this parameter existed
        replays byte-identically.
        """
        self.client = client
        self.use_guided_json = use_guided_json
        self.transport_retries = transport_retries
        self.reply_retries = reply_retries
        self.drops: dict[str, int] = {}
        # Transport failures that were RECOVERED by a retry. Not drops — no work
        # was lost — but a run over a degrading link should be able to say so
        # rather than look pristine.
        self.transport_recoveries: dict[str, int] = {}
        self._lock = threading.Lock()

    def _drop(self, role: str, prompt: str, last_output: str) -> None:
        with self._lock:
            self.drops[role] = self.drops.get(role, 0) + 1
        logger.warning(
            "structured output dropped (role=%s, total_drops=%s): %.120s ...",
            role,
            self.drops.get(role),
            last_output,
        )

    def call(
        self,
        role: str,
        prompt: str,
        schema: dict[str, Any],
        required_keys: tuple[str, ...] = (),
        max_retries: int | None = None,
        system: str = "You must respond with a single JSON object and nothing else.",
        phase: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the validated dict, or None after an explicit drop.

        ``phase`` tags the budget entry as ``f"{role}/{phase}"`` (docs/04
        lifecycle phases — segment/narrate/merge/integrate/consolidate/
        predict_calibrate) so methodology cost comparisons can break down
        spend by write-path stage instead of just by role.

        ``max_retries`` is the number of CORRECTION turns for a malformed reply;
        ``None`` (the normal case — no organizer passes it) takes the instance
        default set at construction. See ``__init__`` for why that default is
        worth raising on high-call-count arms.
        """
        if max_retries is None:
            max_retries = self.reply_retries
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        overrides: dict[str, Any] = {}
        if self.use_guided_json:
            overrides["extra_body"] = {"guided_json": schema}
        budget_key = f"{role}/{phase}" if phase else None

        last_output = ""
        transport_left = self.transport_retries
        attempt = 0
        while attempt <= max_retries:
            try:
                last_output = self.client.chat(role, messages, budget_key=budget_key, **overrides)
            except Exception as exc:  # endpoint/transport error
                logger.warning("LLM call failed (role=%s, attempt=%s): %s", role, attempt, exc)
                if self.use_guided_json and overrides:
                    # The endpoint may be rejecting guided_json rather than being
                    # unreachable. Try once without it before spending a
                    # transport retry, as this has always done.
                    overrides = {}
                    continue
                if transport_left > 0:
                    # No reply came back, so there is nothing for the model to
                    # correct: re-send the SAME messages after a backoff, and do
                    # NOT consume `attempt` (the schema budget). Appending a
                    # correction turn here would blame the model for a network
                    # fault and change the prompt under test.
                    transport_left -= 1
                    time.sleep(2.0 ** (self.transport_retries - transport_left))
                    with self._lock:
                        self.transport_recoveries[role] = self.transport_recoveries.get(role, 0) + 1
                    continue
                break
            parsed = coerce_to_schema(extract_json(last_output), schema)
            if parsed is not None and all(k in parsed for k in required_keys):
                return parsed
            messages.append({"role": "assistant", "content": last_output})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON with keys "
                        f"{list(required_keys)}. Respond again with ONLY the JSON object."
                    ),
                }
            )
            attempt += 1  # a malformed REPLY is what the schema budget pays for
        self._drop(role, prompt, last_output)
        return None
