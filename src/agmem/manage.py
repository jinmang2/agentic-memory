from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, assert_never

from agmem.config import load_config
from agmem.control import (
    correct_memory,
    disable_memory,
    inspect_memory,
    restore_memory,
    status,
)
from agmem.env import resolve_config_path, resolve_data_dir, resolve_namespace
from agmem.hooks import open_doc_store


class Command(StrEnum):
    STATUS = "status"
    DOCTOR = "doctor"
    INSPECT = "inspect"
    DISABLE = "disable"
    RESTORE = "restore"
    CORRECT = "correct"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agmem.manage")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status")
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--daemon-url", default=None)

    inspect = sub.add_parser("inspect")
    _add_ref(inspect)

    disable = sub.add_parser("disable")
    _add_ref(disable)
    disable.add_argument("--reason", default=None)

    restore = sub.add_parser("restore")
    _add_ref(restore)
    restore.add_argument("--reason", default=None)

    correct = sub.add_parser("correct")
    _add_ref(correct)
    correct.add_argument("--content", required=True)
    correct.add_argument("--reason", default=None)
    return parser


def _add_ref(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--type", required=True, dest="memory_type")
    parser.add_argument("--id", required=True, dest="memory_id")


def _open(args: argparse.Namespace):
    if args.config:
        import os

        os.environ["AGMEM_CONFIG"] = str(args.config)
    return open_doc_store(namespace=args.namespace, data_dir=args.data_dir)


def _meta(namespace: str, args: argparse.Namespace) -> dict[str, str | None]:
    config_path = resolve_config_path(args.config)
    config = load_config(config_path) if config_path is not None else None
    data_dir = resolve_data_dir(args.data_dir, from_config=config.data_dir if config else None)
    try:
        package_version = version("agmem")
    except PackageNotFoundError:
        package_version = None
    return {
        "namespace": namespace,
        "data_dir": str(data_dir),
        "config": str(config_path) if config_path is not None else None,
        "config_exists": str(config_path.exists()) if config_path is not None else None,
        "python": sys.executable,
        "package_version": package_version,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    command = Command(args.command)
    source = _meta(resolve_namespace(args.namespace), args)
    if command is Command.DOCTOR:
        from agmem.runtime_diagnostics import diagnose

        config_path = resolve_config_path(args.config)
        return asdict(
            diagnose(
                namespace=resolve_namespace(args.namespace),
                data_dir=Path(str(source["data_dir"])),
                config_path=config_path,
                daemon_url=args.daemon_url,
            )
        )
    ns, store = _open(args)
    try:
        match command:
            case Command.STATUS:
                from agmem.hooks.spool import spool_status

                result = status(
                    store,
                    namespace=ns,
                    store_path=str(source["data_dir"]) + f"/{ns}/memory.db",
                )
                result["queues"] = {
                    name: asdict(
                        spool_status(Path(str(source["data_dir"])) / ns / f"{name}-queue.jsonl")
                    )
                    for name in ("preserve", "distill")
                }
            case Command.INSPECT:
                result = asdict(
                    inspect_memory(
                        store,
                        namespace=ns,
                        memory_type=args.memory_type,
                        memory_id=args.memory_id,
                    )
                )
            case Command.DISABLE:
                result = asdict(
                    disable_memory(
                        store,
                        namespace=ns,
                        memory_type=args.memory_type,
                        memory_id=args.memory_id,
                        reason=args.reason,
                    )
                )
            case Command.RESTORE:
                result = asdict(
                    restore_memory(
                        store,
                        namespace=ns,
                        memory_type=args.memory_type,
                        memory_id=args.memory_id,
                        reason=args.reason,
                    )
                )
            case Command.CORRECT:
                result = asdict(
                    correct_memory(
                        store,
                        namespace=ns,
                        memory_type=args.memory_type,
                        memory_id=args.memory_id,
                        content=args.content,
                        reason=args.reason,
                    )
                )
            case unreachable:
                assert_never(unreachable)
        result["source"] = source
        return result
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = _run(args)
    except (KeyError, RuntimeError, OSError) as exc:
        json.dump({"error": str(exc)}, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 2
    json.dump(payload, sys.stdout, ensure_ascii=False, default=str)
    sys.stdout.write("\n")
    if args.command == "doctor" and payload.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
