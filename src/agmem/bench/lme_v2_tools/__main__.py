"""Expose offline preparation commands; output files are created exclusively."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import assert_never


class Command(StrEnum):
    """The command surface deliberately has no benchmark execution operation."""

    PREPARE = "prepare"
    VERIFY = "verify"
    AUDIT = "audit"
    DIAGNOSE = "diagnose"
    COSTS = "costs"


class Arguments(argparse.Namespace):
    """Mutable parser destination; argparse requires command and source before dispatch."""

    command: str = ""
    source: Path = Path()
    output: Path | None = None


def main(argv: list[str] | None = None) -> int:
    """Emit JSON, returning 2 on invalid inputs or drift without overwriting outputs."""
    parser = argparse.ArgumentParser(prog="python -m agmem.bench.lme_v2_tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in Command:
        subparser = subparsers.add_parser(command.value)
        subparser.add_argument("source", type=Path)
        subparser.add_argument("--output", type=Path)
    args = Arguments()
    parser.parse_args(argv, namespace=args)
    try:
        match Command(args.command):
            case Command.PREPARE:
                from .prepare import prepare

                payload = asdict(prepare(args.source))
                status = 0
            case Command.VERIFY:
                from .prepare import verify

                verification = verify(args.source)
                payload = asdict(verification)
                status = 0 if verification.valid else 2
            case Command.AUDIT:
                from .audit import audit_root

                payload = asdict(audit_root(args.source))
                status = 0
            case Command.DIAGNOSE:
                from .diagnostics import diagnose_root

                payload = asdict(diagnose_root(args.source))
                status = 0
            case Command.COSTS:
                from .costs import cost_report

                payload = asdict(cost_report(args.source))
                status = 0
            case unreachable:
                assert_never(unreachable)
        rendered = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output is None:
            print(rendered, end="")
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        return status
    except (OSError, ValueError) as exc:
        print(f"{args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
