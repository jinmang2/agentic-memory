from agmem.llm.structured import coerce_to_schema, extract_json

ITEMS_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array"}},
    "required": ["items"],
}


def test_extract_json_from_code_fence():
    text = 'Here you go:\n```json\n{"a": 1}\n```'
    assert extract_json(text) == {"a": 1}


def test_extract_json_top_level_array():
    text = '```json\n[{"title": "x"}]\n```'
    assert extract_json(text) == [{"title": "x"}]


def test_coerce_bare_array_wrapped_into_single_array_field():
    # observed Qwen3-0.6B failure: bare array instead of {"items": [...]}
    parsed = extract_json('[{"title": "t", "description": "d", "content": "c"}]')
    coerced = coerce_to_schema(parsed, ITEMS_SCHEMA)
    assert coerced == {"items": [{"title": "t", "description": "d", "content": "c"}]}


def test_coerce_ambiguous_array_schema_refused():
    two_arrays = {"type": "object", "properties": {"a": {"type": "array"}, "b": {"type": "array"}}}
    assert coerce_to_schema([1, 2], two_arrays) is None


def test_coerce_dict_passthrough():
    assert coerce_to_schema({"x": 1}, ITEMS_SCHEMA) == {"x": 1}


# ---- transport-failure retry ------------------------------------------------
# Found live on 2026-08-04: on a flaky link a SINGLE transport blip anywhere in a
# ~400-call conversation destroyed the whole conversation's ingest. `max_retries`
# is a SCHEMA retry budget — the transport `except` broke straight out of the
# loop to `_drop`, so a timeout was never retried at this layer. The organizer
# then lost that piece of write-path work, the conversation failed its
# clean-ingest check, and every call already paid for was re-spent.

import threading

from agmem.llm.structured import StructuredCaller


class _ScriptedClient:
    """LLMClient stand-in: each entry is either an exception to raise or text."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.sent = []

    def chat(self, role, messages, budget_key=None, **overrides):
        self.calls += 1
        self.sent.append([dict(m) for m in messages])
        item = self.script.pop(0) if self.script else '{"ok": 1}'
        if isinstance(item, Exception):
            raise item
        return item


def _caller(script, **kw):
    c = StructuredCaller(_ScriptedClient(script), use_guided_json=False, **kw)
    return c, c.client


def test_transport_error_is_retried_not_dropped(monkeypatch):
    monkeypatch.setattr("agmem.llm.structured.time.sleep", lambda s: None)
    caller, client = _caller([TimeoutError("timed out"), '{"ok": 1}'])
    out = caller.call("extract", "p", {"type": "object"}, required_keys=("ok",))
    assert out == {"ok": 1}
    assert client.calls == 2
    assert caller.drops == {}  # nothing lost


def test_transport_retry_resends_the_original_prompt_unchanged(monkeypatch):
    """A transport failure produced no model output, so there is nothing to
    correct. Re-sending must NOT append the schema-correction turns that a
    malformed REPLY earns — that would blame the model for a network fault and
    change the prompt under test."""
    monkeypatch.setattr("agmem.llm.structured.time.sleep", lambda s: None)
    caller, client = _caller([TimeoutError("boom"), '{"ok": 1}'])
    caller.call("extract", "the original prompt", {"type": "object"}, required_keys=("ok",))
    assert client.sent[0] == client.sent[1]
    assert len(client.sent[1]) == 2  # system + user only


def test_transport_retries_are_bounded_then_dropped(monkeypatch):
    monkeypatch.setattr("agmem.llm.structured.time.sleep", lambda s: None)
    caller, client = _caller([TimeoutError("x")] * 10, transport_retries=2)
    assert caller.call("extract", "p", {"type": "object"}, required_keys=("ok",)) is None
    assert client.calls == 3  # 1 initial + 2 retries
    assert caller.drops == {"extract": 1}


def test_transport_retry_budget_is_separate_from_schema_retries(monkeypatch):
    """A network blip must not consume the budget that exists for malformed
    replies, and vice versa — they are different failures with different fixes."""
    monkeypatch.setattr("agmem.llm.structured.time.sleep", lambda s: None)
    # blip, then a malformed reply, then a good one: needs BOTH budgets
    caller, client = _caller([TimeoutError("x"), "not json at all", '{"ok": 1}'])
    out = caller.call("extract", "p", {"type": "object"}, required_keys=("ok",), max_retries=1)
    assert out == {"ok": 1}
    assert client.calls == 3
    assert caller.drops == {}


def test_transport_retry_backs_off(monkeypatch):
    slept = []
    monkeypatch.setattr("agmem.llm.structured.time.sleep", lambda s: slept.append(s))
    caller, _ = _caller([TimeoutError("x")] * 5, transport_retries=3)
    caller.call("extract", "p", {"type": "object"}, required_keys=("ok",))
    assert slept == sorted(slept) and len(slept) == 3  # monotonically increasing
    assert slept[0] > 0


def test_clean_run_behaviour_is_byte_identical():
    """The regression guard for the six conversations already ingested: they
    recorded zero drops, so no transport retry could ever have fired for them.
    A run with no transport failure must issue exactly the same calls as before.
    """
    caller, client = _caller(['{"ok": 1}'])
    assert caller.call("extract", "p", {"type": "object"}, required_keys=("ok",)) == {"ok": 1}
    assert client.calls == 1 and caller.drops == {}


def test_drops_counter_stays_threadsafe(monkeypatch):
    monkeypatch.setattr("agmem.llm.structured.time.sleep", lambda s: None)
    caller = StructuredCaller(_ScriptedClient([TimeoutError("x")] * 400), use_guided_json=False)
    threads = [
        threading.Thread(
            target=lambda: caller.call("r", "p", {"type": "object"}, required_keys=("ok",))
        )
        for _ in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert caller.drops["r"] == 20


def test_transport_recoveries_count_only_retries_that_worked(monkeypatch):
    """The field's own comment defines a recovery as a transport failure a retry
    RECOVERED. The counter used to increment in the except branch — before the
    retry's outcome existed — so it counted attempts (`APIEmbedder` got this
    right first; `StructuredCaller` now matches)."""
    monkeypatch.setattr("agmem.llm.structured.time.sleep", lambda s: None)
    caller, _ = _caller([TimeoutError("x"), TimeoutError("x"), '{"ok": 1}'], transport_retries=3)
    assert caller.call("extract", "p", {"type": "object"}, required_keys=("ok",)) == {"ok": 1}
    assert caller.transport_recoveries == {"extract": 2}


def test_spent_retries_that_never_recovered_are_not_recoveries(monkeypatch):
    monkeypatch.setattr("agmem.llm.structured.time.sleep", lambda s: None)
    caller, _ = _caller([TimeoutError("x")] * 10, transport_retries=2)
    assert caller.call("extract", "p", {"type": "object"}, required_keys=("ok",)) is None
    assert caller.drops == {"extract": 1}
    assert caller.transport_recoveries == {}, "retries were spent; none of them recovered anything"
