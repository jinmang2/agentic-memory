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
    RUN = "run"


class Arguments(argparse.Namespace):
    command: str = ""
    source: Path = Path()
    output: Path | None = None
    results: Path | None = None
    work_dir: Path = Path()
    agent_command: str | None = None
    model: str | None = None
    max_turns: int | None = None
    timeout: int | None = None
    jobs: str = ""


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
    run_parser = subparsers.add_parser(Command.RUN.value)
    run_parser.add_argument("source", type=Path, help="prepared manifest")
    run_parser.add_argument("--work-dir", type=Path, required=True)
    run_parser.add_argument("--agent-command", default=None)
    run_parser.add_argument("--model", default=None)
    run_parser.add_argument("--max-turns", type=int, default=None)
    run_parser.add_argument("--timeout", type=int, default=None)
    run_parser.add_argument("--jobs", default="", help="comma-separated job ids; default all")
    run_parser.add_argument("--output", type=Path)
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
            case Command.RUN:
                from agmem.bench.coding_eval.run import RunOptions, run

                defaults = RunOptions(manifest_path=args.source, work_dir=args.work_dir)
                report = run(
                    RunOptions(
                        manifest_path=args.source,
                        work_dir=args.work_dir,
                        agent_command=args.agent_command or defaults.agent_command,
                        model=args.model or defaults.model,
                        max_turns=args.max_turns or defaults.max_turns,
                        timeout_s=args.timeout or defaults.timeout_s,
                        only_jobs=frozenset(j for j in args.jobs.split(",") if j),
                    )
                )
                payload = {
                    "results_path": str(report.results_path),
                    "verification": {
                        "valid": report.verification_valid,
                        "errors": list(report.verification_errors),
                    },
                    "jobs": [
                        {
                            "job_id": o.job_id,
                            "success": o.row["success"],
                            "failed_check": o.acceptance_failed_check,
                            "agent_error": o.agent_error,
                            "injected_chars": o.injected_chars,
                        }
                        for o in report.outcomes
                    ],
                }
                status = 0 if report.verification_valid else 2
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
