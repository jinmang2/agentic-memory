"""POSIX JSONL spool with at-least-once recovery after process crashes, not power loss."""

from __future__ import annotations

import fcntl
import json
import logging
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

Action = Callable[[str], None]
type SpoolValue = None | bool | int | float | str | list[SpoolValue] | dict[str, SpoolValue]
SpoolPayload = Mapping[str, SpoolValue]
logger = logging.getLogger("agmem.hooks.spool")


@dataclass(frozen=True, slots=True)
class SpoolStatus:
    queued: int
    processing: int
    bad: int

    def as_dict(self) -> dict[str, int]:
        return {"queued": self.queued, "processing": self.processing, "bad": self.bad}


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _processing_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.processing")


def _quarantine_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.bad")


def _producer_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


def _drain_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.drain.lock")


def append_spool(body: SpoolPayload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(_producer_lock_path(path)), path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(body, ensure_ascii=False) + "\n")
        fp.flush()
        os.fsync(fp.fileno())
    _fsync_dir(path.parent)


def _claim(path: Path) -> Path | None:
    processing = _processing_path(path)
    with _exclusive_lock(_producer_lock_path(path)):
        if processing.exists():
            lines = _lines(processing) + _lines(path)
            _checkpoint(processing, lines)
            _write_lines(path, [])
            return processing if lines else None
        if not path.exists() or path.stat().st_size == 0:
            return None
        path.replace(processing)
        path.touch()
        _fsync_dir(path.parent)
    return processing


def _lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _checkpoint(processing: Path, lines: list[str]) -> None:
    if lines:
        _write_lines(processing, lines)
    elif processing.exists():
        processing.unlink()
        _fsync_dir(processing.parent)


def _write_lines(path: Path, lines: list[str]) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        if lines:
            fp.write("\n".join(lines) + "\n")
        fp.flush()
        os.fsync(fp.fileno())
    tmp.replace(path)
    _fsync_dir(path.parent)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _quarantine(queue: Path, reason: str, line: str) -> None:
    bad = _quarantine_path(queue)
    bad.parent.mkdir(parents=True, exist_ok=True)
    with bad.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps({"reason": reason, "line": line}, ensure_ascii=False) + "\n")
        fp.flush()
        os.fsync(fp.fileno())


def _transcript_path(line: str) -> str | None:
    try:
        body = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(body, dict):
        return None
    raw = body.get("transcript_path")
    if not isinstance(raw, str) or not raw:
        return None
    return raw


def _quarantine_reason(line: str, transcript: str | None) -> str | None:
    if transcript is None:
        try:
            body = json.loads(line)
        except json.JSONDecodeError:
            return "malformed-json"
        return "missing-path" if isinstance(body, dict) else "malformed-json"
    return None if Path(transcript).is_file() else "missing-file"


def drain_spool(path: Path, action: Action) -> int:
    with _exclusive_lock(_drain_lock_path(path)):
        processing = _claim(path)
        if processing is None:
            return 0
        done = 0
        retry: list[str] = []
        lines = _lines(processing)
        for index, line in enumerate(lines):
            remaining = lines[index + 1 :]
            transcript = _transcript_path(line)
            reason = _quarantine_reason(line, transcript)
            if reason is not None:
                _quarantine(path, reason, line)
                _checkpoint(processing, retry + remaining)
                continue
            if transcript is None:
                _checkpoint(processing, retry + remaining)
                continue
            try:
                action(transcript)
            except Exception as exc:  # noqa: BLE001 - arbitrary failures must remain retryable.
                logger.warning(
                    "agmem hook spool: retained failed row path=%s error=%s", transcript, exc
                )
                retry.append(line)
                _checkpoint(processing, retry + remaining)
                continue
            done += 1
            _checkpoint(processing, retry + remaining)
        return done


def spool_status(path: Path) -> SpoolStatus:
    return SpoolStatus(
        queued=len(_lines(path)),
        processing=len(_lines(_processing_path(path))),
        bad=len(_lines(_quarantine_path(path))),
    )
