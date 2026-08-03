import logging

from agmem.llm.client import LLMClient, RoleConfig


class _StubResp:
    class _Choice:
        class _Msg:
            content = "ok"

        message = _Msg()

    def __init__(self):
        self.choices = [self._Choice()]
        self.usage = None  # provider omitted usage


class _StubClient:
    def __init__(self):
        self.last_kwargs = None
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _StubResp()


def _client_with_stub(cfg):
    c = LLMClient({"generate": cfg})
    stub = _StubClient()
    c._clients[f"{cfg.endpoint}|{cfg.api_key}"] = stub  # bypass real OpenAI construction
    return c, stub


def test_none_sampling_params_are_omitted():
    cfg = RoleConfig(endpoint="e", model="m", temperature=None, max_tokens=None)
    c, stub = _client_with_stub(cfg)
    c.chat("generate", [{"role": "user", "content": "hi"}])
    assert "temperature" not in stub.last_kwargs and "max_tokens" not in stub.last_kwargs


def test_missing_usage_warns_not_silent(caplog):
    cfg = RoleConfig(endpoint="e", model="m")
    c, _stub = _client_with_stub(cfg)
    with caplog.at_level(logging.WARNING, logger="agmem.llm"):
        c.chat("generate", [{"role": "user", "content": "hi"}])
    assert any("usage" in r.message for r in caplog.records)
    # tokens recorded as 0 — the warning is what stops this from becoming a silent $0 quote
    assert c.budget.summary()["generate"]["tokens_in"] == 0


def test_max_tokens_key_override_sends_max_completion_tokens():
    # gpt-5.6-luna (and any model requiring the newer Chat Completions param name)
    # rejects `max_tokens` outright (400 Unsupported parameter) — the RoleConfig
    # must send its value under whatever key the model requires.
    cfg = RoleConfig(
        endpoint="e", model="m", max_tokens=500, max_tokens_key="max_completion_tokens"
    )
    c, stub = _client_with_stub(cfg)
    c.chat("generate", [{"role": "user", "content": "hi"}])
    assert stub.last_kwargs["max_completion_tokens"] == 500
    assert "max_tokens" not in stub.last_kwargs


def test_max_tokens_key_default_is_unchanged():
    cfg = RoleConfig(endpoint="e", model="m", max_tokens=500)
    c, stub = _client_with_stub(cfg)
    c.chat("generate", [{"role": "user", "content": "hi"}])
    assert stub.last_kwargs["max_tokens"] == 500
    assert "max_completion_tokens" not in stub.last_kwargs
