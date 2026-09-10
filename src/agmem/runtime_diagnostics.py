from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from agmem import __version__
from agmem.hooks.daemon import health

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


class RuntimeFingerprint(TypedDict):
    source_sha256: dict[str, str | None]
    config_sha256: str | None


@dataclass(frozen=True, slots=True)
class RuntimeDiagnosis:
    status: Literal["ok", "mismatch", "unavailable"]
    mismatches: tuple[str, ...]
    warnings: tuple[str, ...]
    namespace: str
    data_dir: str
    config_path: str | None
    package_version: str
    interpreter: str
    runtime_fingerprint: RuntimeFingerprint


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def runtime_fingerprint(config_path: Path | None) -> RuntimeFingerprint:
    root = Path(__file__).parent
    paths = (
        "mcp/server.py",
        "mcp/management.py",
        "memory.py",
        "session_distill.py",
        "control.py",
        "control_data.py",
        "control_replay.py",
        "control_status.py",
        "control_types.py",
        "runtime_diagnostics.py",
        "env.py",
        "hooks/spool.py",
        "hooks/capture.py",
        "hooks/preserve.py",
        "hooks/distill.py",
        "hooks/recall.py",
        "hooks/recall_prompt.py",
        "organizers/experience/organizer.py",
        "organizers/experience/quality.py",
        "retrieval/pipeline.py",
    )
    return {
        "source_sha256": {name: _sha256_file(root / name) for name in paths},
        "config_sha256": _sha256_file(config_path) if config_path is not None else None,
    }


def compare_runtime(
    daemon: Mapping[str, JsonValue] | None,
    *,
    namespace: str,
    data_dir: Path,
    config_path: Path | None,
) -> RuntimeDiagnosis:
    fingerprint = runtime_fingerprint(config_path)
    mismatches: list[str] = []
    warnings: list[str] = []
    if daemon is not None:
        expected: dict[str, str | None] = {
            "data_dir": str(data_dir.resolve()),
            "config_path": str(config_path.resolve()) if config_path is not None else None,
            "module_path": str(Path(__file__).parent / "mcp/server.py"),
            "interpreter": sys.executable,
            "package_version": __version__,
        }
        for name, value in expected.items():
            actual = daemon.get(name)
            if name in ("data_dir", "config_path", "module_path", "interpreter") and isinstance(
                actual, str
            ):
                actual = str(Path(actual).resolve())
                value = str(Path(value).resolve()) if value is not None else None
            if actual != value:
                mismatches.append(name)
        if daemon.get("runtime_fingerprint") != fingerprint:
            mismatches.append("runtime_fingerprint")
        if daemon.get("default_namespace") != namespace:
            warnings.append("daemon_default_namespace_differs; explicit hook namespace is required")
    status: Literal["ok", "mismatch", "unavailable"] = "unavailable"
    if daemon is not None:
        status = "mismatch" if mismatches else "ok"
    return RuntimeDiagnosis(
        status=status,
        mismatches=tuple(mismatches),
        warnings=tuple(warnings),
        namespace=namespace,
        data_dir=str(data_dir.resolve()),
        config_path=str(config_path.resolve()) if config_path is not None else None,
        package_version=__version__,
        interpreter=sys.executable,
        runtime_fingerprint=fingerprint,
    )


def diagnose(
    *,
    namespace: str,
    data_dir: Path,
    config_path: Path | None,
    daemon_url: str | None = None,
) -> RuntimeDiagnosis:
    return compare_runtime(
        health(daemon_url, timeout=2),
        namespace=namespace,
        data_dir=data_dir,
        config_path=config_path,
    )
