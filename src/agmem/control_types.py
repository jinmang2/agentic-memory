from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CONTROL_KEY = "control"
AUTO_INJECTION = "automatic_injection"
AUTO_ENABLED = "enabled"
AUTO_DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class ControlResult:
    changed: bool
    namespace: str
    memory_type: str
    memory_id: str
    item: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MemoryInspection:
    found: bool
    namespace: str
    memory_type: str
    memory_id: str
    item: dict[str, Any]
    eligible: bool
    eligibility_reasons: list[str]


@dataclass(frozen=True, slots=True)
class MemoryControlKeyError(KeyError):
    memory_type: str
    memory_id: str
    namespace: str | None = None

    def __str__(self) -> str:
        location = f" in namespace {self.namespace!r}" if self.namespace else ""
        return f"{self.memory_type}:{self.memory_id} not found{location}"


@dataclass(frozen=True, slots=True)
class UnsupportedMemoryTypeError(KeyError):
    memory_type: str

    def __str__(self) -> str:
        return f"unsupported memory type {self.memory_type!r}"


@dataclass(frozen=True, slots=True)
class UserControlOpError(ValueError):
    actor: str
    op: str
    target_type: str

    def __str__(self) -> str:
        return (
            "expected a user-control UPDATE op, got "
            f"actor={self.actor!r} op={self.op!r} target_type={self.target_type!r}"
        )
