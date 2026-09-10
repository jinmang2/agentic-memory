from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import assert_never


class Command(StrEnum):
    PREPARE = "prepare"
    VERIFY = "verify"


class Arguments(argparse.Namespace):
    command: str = ""
    source: Path = Path()
    output: Path | None = None
    results: Path | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agmem.bench.coding_eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser(Command.PREPARE.value)
    prepare_parser.add_argument("source", type=Path)
    prepare_parser.add_argument("--output", type=Path)
    verify_parser = subparsers.add_parser(Command.VERIFY.value)
    verify_parser.add_argument("source", type=Path)
    verify_parser.add_argument("--results", type=Path)
    verify_parser.add_argument("--output", type=Path)
    args = Arguments()
    parser.parse_args(argv, namespace=args)
    try:
        match Command(args.command):
            case Command.PREPARE:
                from agmem.bench.coding_eval.prepare import prepare

                payload = asdict(prepare(args.source))
                status = 0
            case Command.VERIFY:
                from agmem.bench.coding_eval.prepare import verify

                verification = verify(args.source, args.results)
                payload = asdict(verification)
                status = 0 if verification.valid else 2
            case unreachable:
                assert_never(unreachable)
        rendered = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output is None:
            print(rendered, end="")
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        return status
    except (OSError, ValueError) as error:
        print(f"{args.command}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
