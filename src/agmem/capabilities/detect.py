"""Host capability detection.

Detected once, cached to ``~/.agmem/capabilities.json`` (TTL 24h).
No third-party deps: /proc for RAM, nvidia-smi for VRAM, TCP probes for
services, HTTP probe for OpenAI-compatible LLM endpoints.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from importlib.util import find_spec
from pathlib import Path

CACHE_TTL_SECONDS = 24 * 3600

# service name -> default port to probe on localhost
SERVICE_PORTS = {
    "neo4j": 7687,
    "qdrant": 6333,
    "redis": 6379,
    "ollama": 11434,
    "postgres": 5432,
    "falkordb": 6379,  # redis protocol
}

# candidate OpenAI-compatible endpoints to health-check
DEFAULT_LLM_ENDPOINTS = [
    "http://localhost:8000/v1",  # vLLM / llama.cpp server convention
    "http://localhost:11434/v1",  # ollama
    "http://localhost:1234/v1",  # LM Studio
]


@dataclass
class EndpointInfo:
    """Health-check result for one candidate OpenAI-compatible endpoint."""

    base_url: str
    alive: bool
    models: list[str] = field(default_factory=list)


@dataclass
class HostCapabilities:
    """Point-in-time snapshot of what this host can run, cached to disk with
    a TTL by `detect()` and matched against `Requires` by the resolver."""

    ram_gb: float
    cpu_cores: int
    vram_gb: float | None = None
    gpu_name: str | None = None
    services: dict[str, bool] = field(default_factory=dict)
    llm_endpoints: list[EndpointInfo] = field(default_factory=list)
    python_pkgs: dict[str, bool] = field(default_factory=dict)
    detected_at: float = field(default_factory=time.time)

    def has_service(self, name: str) -> bool:
        """`False` for a service that was never probed (not just unavailable)."""
        return self.services.get(name, False)

    def has_pkg(self, name: str) -> bool:
        """Lazily probes and memoizes into `self.python_pkgs` on first call
        for a name outside `PROBE_PKGS`, so this mutates the instance."""
        if name not in self.python_pkgs:
            self.python_pkgs[name] = find_spec(name) is not None
        return self.python_pkgs[name]


def _ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 2)
    except OSError:
        pass
    return 0.0


def _gpu() -> tuple[float | None, str | None]:
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, mem_mib = out.stdout.strip().splitlines()[0].rsplit(",", 1)
            return round(float(mem_mib) / 1024, 2), name.strip()
    except (subprocess.SubprocessError, ValueError):
        pass
    return None, None


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _probe_endpoint(base_url: str, timeout: float = 1.0) -> EndpointInfo:
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            models = [m.get("id", "?") for m in data.get("data", [])]
            return EndpointInfo(base_url=base_url, alive=True, models=models)
    except Exception:
        return EndpointInfo(base_url=base_url, alive=False)


# packages whose presence gates optional adapters
PROBE_PKGS = [
    "sqlite_vec",
    "sentence_transformers",
    "lancedb",
    "qdrant_client",
    "kuzu",
    "neo4j",
    "chromadb",
    "torch",
]


def detect(
    cache_dir: Path | None = None,
    force: bool = False,
    extra_endpoints: list[str] | None = None,
) -> HostCapabilities:
    """Read `<cache_dir>/capabilities.json` if it exists and is younger than
    `CACHE_TTL_SECONDS`, otherwise re-probe the host and overwrite the cache
    (best-effort: a write failure is swallowed, not raised). `force=True`
    skips the cache entirely. `python_pkgs` is always re-probed fresh even
    on a cache hit — `find_spec` is cheap and a stale "not installed" would
    otherwise survive a `pip install` for the rest of the TTL window.
    `extra_endpoints` are probed in addition to `DEFAULT_LLM_ENDPOINTS`,
    de-duplicated, extras first."""
    cache_dir = cache_dir or Path.home() / ".agmem"
    cache_file = cache_dir / "capabilities.json"

    if not force and cache_file.exists():
        try:
            raw = json.loads(cache_file.read_text())
            if time.time() - raw.get("detected_at", 0) < CACHE_TTL_SECONDS:
                raw["llm_endpoints"] = [EndpointInfo(**e) for e in raw.get("llm_endpoints", [])]
                # never trust cached pkg probes: a pip install between runs
                # would otherwise stay invisible for the TTL. find_spec is
                # cheap, so re-probe fresh each process.
                raw["python_pkgs"] = {p: find_spec(p) is not None for p in PROBE_PKGS}
                return HostCapabilities(**raw)
        except (json.JSONDecodeError, TypeError):
            pass  # corrupt cache -> re-detect

    vram, gpu_name = _gpu()
    endpoints = [
        _probe_endpoint(url)
        for url in dict.fromkeys((extra_endpoints or []) + DEFAULT_LLM_ENDPOINTS)
    ]
    caps = HostCapabilities(
        ram_gb=_ram_gb(),
        cpu_cores=os.cpu_count() or 1,
        vram_gb=vram,
        gpu_name=gpu_name,
        services={name: _port_open(port) for name, port in SERVICE_PORTS.items()},
        llm_endpoints=endpoints,
        python_pkgs={p: find_spec(p) is not None for p in PROBE_PKGS},
    )

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(asdict(caps), indent=2))
    except OSError:
        pass  # cache is best-effort
    return caps
