"""`agmem.env`: the one place the deployment surfaces resolve which store to open."""

from __future__ import annotations

from pathlib import Path

import pytest

from agmem import env


def test_defaults_are_shared_by_facade_hooks_and_server():
    """The regression issue #2 reported was three modules each carrying a
    namespace default. There is one now, and the others import it."""
    import inspect

    from agmem import hooks
    from agmem.memory import AgenticMemory

    assert hooks.DEFAULT_NAMESPACE is env.DEFAULT_NAMESPACE
    sig = inspect.signature(AgenticMemory.__init__)
    assert sig.parameters["namespace"].default == env.DEFAULT_NAMESPACE


def test_precedence_explicit_env_config_default(monkeypatch, tmp_path):
    monkeypatch.delenv(env.ENV_NAMESPACE, raising=False)
    monkeypatch.delenv(env.ENV_DATA_DIR, raising=False)
    monkeypatch.delenv(env.ENV_CONFIG, raising=False)
    assert env.resolve_namespace() == env.DEFAULT_NAMESPACE
    assert env.resolve_data_dir() == env.DEFAULT_DATA_DIR
    assert env.resolve_config_path() is None

    assert env.resolve_data_dir(from_config=tmp_path / "toml") == tmp_path / "toml"

    monkeypatch.setenv(env.ENV_NAMESPACE, "from-env")
    monkeypatch.setenv(env.ENV_DATA_DIR, str(tmp_path / "env"))
    monkeypatch.setenv(env.ENV_CONFIG, str(tmp_path / "agmem.toml"))
    assert env.resolve_namespace() == "from-env"
    assert env.resolve_data_dir(from_config=tmp_path / "toml") == tmp_path / "env"
    assert env.resolve_config_path() == tmp_path / "agmem.toml"

    assert env.resolve_namespace("explicit") == "explicit"
    assert env.resolve_data_dir("~/x") == Path.home() / "x"
    assert env.resolve_config_path("~/c.toml") == Path.home() / "c.toml"


@pytest.mark.parametrize(
    "bad", ["", ".", "..", "a/b", "a\\b", "../up", ".hidden", "has space", "tab\there"]
)
def test_namespace_must_be_a_single_directory_name(bad):
    with pytest.raises(env.InvalidNamespace):
        env.validate_namespace(bad)


@pytest.mark.parametrize("good", ["main", "claude-code", "proj_1", "a.b", "한글"])
def test_ordinary_namespaces_pass(good):
    assert env.validate_namespace(good) == good
