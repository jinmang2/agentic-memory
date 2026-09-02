"""`python -m agmem.sessions scan|render …` — look at what the adapter would ingest.

    python -m agmem.sessions scan                      # every session on this machine
    python -m agmem.sessions scan --project .          # this project's Claude Code sessions
    python -m agmem.sessions render <file.jsonl>       # the transcript a distiller would read

Prints to stdout and calls no model. `scan` is the inventory a Phase 3
dogfooding run starts from; `render` is how to check, by eye, that a session
came through with the harness noise gone and the secrets redacted.
"""

from __future__ import annotations

import argparse
import sys

from agmem.sessions import iter_claude_code_sessions, iter_codex_sessions, load


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m agmem.sessions")
    sub = ap.add_subparsers(dest="cmd", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("--project", default=None, help="cwd whose Claude Code sessions to list")
    scan.add_argument("--host", choices=["claude-code", "codex", "all"], default="all")
    render = sub.add_parser("render")
    render.add_argument("path")
    render.add_argument("--max-chars", type=int, default=None)
    args = ap.parse_args()

    if args.cmd == "render":
        traj = load(args.path)
        sys.stdout.write(traj.render(args.max_chars) + "\n")
        return 0

    paths = []
    if args.host in ("claude-code", "all"):
        paths += list(iter_claude_code_sessions(args.project))
    if args.host in ("codex", "all"):
        paths += list(iter_codex_sessions())
    total = 0
    for path in paths:
        try:
            traj = load(path)
        except OSError as exc:
            print(f"!! {path}: {exc}")
            continue
        total += len(traj.steps)
        print(
            f"{traj.host:11s} {traj.id[:12]:12s} steps={len(traj.steps):5d} "
            f"user={traj.user_turns:4d} cwd={traj.cwd or '?'}"
        )
    print(f"{len(paths)} session(s), {total} step(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
