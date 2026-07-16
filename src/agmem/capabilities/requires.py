"""Requirement declarations for backend adapters.

Every adapter class declares ``requires = Requires(...)``. The resolver
matches these against detected ``HostCapabilities`` — unsatisfied
adapters are skipped with an explicit warning, never silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agmem.capabilities.detect import HostCapabilities


@dataclass(frozen=True)
class Requires:
    ram_gb: float = 0.0
    vram_gb: float = 0.0
    services: tuple[str, ...] = ()
    python_pkgs: tuple[str, ...] = ()
    llm_endpoint: bool = False

    def check(self, caps: HostCapabilities) -> tuple[bool, str]:
        """Return (satisfied, reason-if-not)."""
        if self.ram_gb and caps.ram_gb < self.ram_gb:
            return False, f"needs {self.ram_gb}GB RAM, host has {caps.ram_gb}GB"
        if self.vram_gb and (caps.vram_gb or 0.0) < self.vram_gb:
            return False, f"needs {self.vram_gb}GB VRAM, host has {caps.vram_gb or 0}GB"
        for svc in self.services:
            if not caps.has_service(svc):
                return False, f"service '{svc}' not detected"
        for pkg in self.python_pkgs:
            if not caps.has_pkg(pkg):
                return False, f"python package '{pkg}' not installed"
        if self.llm_endpoint and not any(e.alive for e in caps.llm_endpoints):
            return False, "no live OpenAI-compatible LLM endpoint"
        return True, ""
