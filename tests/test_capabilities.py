import pytest

from agmem.capabilities.detect import HostCapabilities, detect
from agmem.capabilities.requires import Requires
from agmem.capabilities.resolver import ResolutionError, resolve


def make_caps(**kwargs) -> HostCapabilities:
    base = dict(
        ram_gb=8.0,
        cpu_cores=4,
        vram_gb=6.0,
        gpu_name="RTX 2060",
        services={"neo4j": False, "ollama": True},
        python_pkgs={"sqlite_vec": True, "sentence_transformers": False},
    )
    base.update(kwargs)
    return HostCapabilities(**base)


def test_detect_real_host(tmp_path):
    caps = detect(cache_dir=tmp_path, force=True)
    assert caps.ram_gb > 0
    assert caps.cpu_cores >= 1
    # cache round-trip
    cached = detect(cache_dir=tmp_path)
    assert cached.ram_gb == caps.ram_gb


def test_requires_check():
    caps = make_caps()
    ok, _ = Requires().check(caps)
    assert ok
    ok, reason = Requires(ram_gb=16).check(caps)
    assert not ok and "RAM" in reason
    ok, reason = Requires(services=("neo4j",)).check(caps)
    assert not ok and "neo4j" in reason
    ok, _ = Requires(services=("ollama",), vram_gb=4).check(caps)
    assert ok
    ok, reason = Requires(python_pkgs=("sentence_transformers",)).check(caps)
    assert not ok and "sentence_transformers" in reason


class Heavy:
    requires = Requires(services=("neo4j",))


class Light:
    requires = Requires()


def test_resolver_prefers_first_satisfiable():
    cls, notes = resolve("graph_store", [Heavy, Light], make_caps())
    assert cls is Light
    assert any("Heavy" in n for n in notes)


def test_resolver_override_degrades_with_note():
    cls, notes = resolve("graph_store", [Heavy, Light], make_caps(), override="Heavy")
    assert cls is Light
    assert any("falling back" in n for n in notes)


def test_resolver_strict_raises():
    with pytest.raises(ResolutionError):
        resolve("graph_store", [Heavy, Light], make_caps(), override="Heavy", strict=True)


def test_resolver_no_candidate_raises():
    with pytest.raises(ResolutionError):
        resolve("graph_store", [Heavy], make_caps())


def test_load_config_reads_read_path_knobs(tmp_path):
    """`retrieval/steps.py` calls its upstream deviations "ablatable from
    AgmemConfig"; the repro runbook reaches config through TOML, where these
    keys used to be silently dropped."""
    from agmem.config import AgmemConfig, load_config

    path = tmp_path / "agmem.toml"
    path.write_text(
        '[profile]\nname = "lite"\n'
        "[retrieval]\n"
        'lexical_types = ["episodic", "facts"]\n'
        "link_expansion_cap = 0\n"
        "attach_sources_top_r = 4\n"
        "graph_expansion_cap = 3\n"
        "graph_expansion_hops = 3\n"
    )
    cfg = load_config(path)
    assert cfg.lexical_types == ("episodic", "facts")
    assert cfg.link_expansion_cap == 0  # 0 disables the step, not "unset"
    assert cfg.attach_sources_top_r == 4
    assert cfg.graph_expansion_cap == 3
    assert cfg.graph_expansion_hops == 3  # upstream MAX_SEARCH_DEPTH

    bare = tmp_path / "bare.toml"
    bare.write_text('[profile]\nname = "lite"\n')
    defaults = AgmemConfig()
    got = load_config(bare)
    assert (got.lexical_types, got.link_expansion_cap, got.attach_sources_top_r) == (
        defaults.lexical_types,
        defaults.link_expansion_cap,
        defaults.attach_sources_top_r,
    )
