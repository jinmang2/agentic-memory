"""`python -m agmem.sessions scan|render|ingest …` — look at, and take in, what
the adapter reads.

    python -m agmem.sessions scan                      # every session on this machine
    python -m agmem.sessions scan --project .          # this project's Claude Code sessions
    python -m agmem.sessions render <file.jsonl>       # the transcript a distiller would read
    python -m agmem.sessions ingest <file.jsonl> --dry-run     # what ingest would do, writes nothing
    python -m agmem.sessions ingest <file.jsonl> --no-distill   # raw steps only, no model call
    python -m agmem.sessions ingest --project . --limit 5       # 5 sessions, 5 model calls

`scan` is the inventory a Phase 3 dogfooding run starts from; `render` is how
to check, by eye, that a session came through with the harness noise gone and
the secrets redacted. `scan` and `render` call no model and open no store.

`ingest` is the only paid subcommand, and it is gated rather than trusted:
distillation is one model call per session, so it refuses to run without an
explicit `--limit N` (N at most 20), and it refuses to run at all when the
resolved config has no LLM role rather than skipping the distillation quietly.
`--dry-run` writes nothing and `--no-distill` makes no model call; the latter
is free only with a local embedder, so an API-backed embedder needs `--limit`
there too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agmem.sessions import iter_claude_code_sessions, iter_codex_sessions, load

# One model call per session, so an unbounded run is an unbounded bill. 20 is a
# dogfooding batch; a larger backfill is a decision to make deliberately, in a
# script that has an estimate attached.
MAX_DISTILL_LIMIT = 20


def _discover(args) -> list:
    """The session files this invocation covers: the positional paths, or the
    same discovery `scan` does."""
    if getattr(args, "paths", None):
        return [Path(p) for p in args.paths]
    paths = []
    if args.host in ("claude-code", "all"):
        paths += list(iter_claude_code_sessions(args.project))
    if args.host in ("codex", "all"):
        paths += list(iter_codex_sessions())
    return paths


def _load_lazily(paths):
    """Sessions one at a time, in discovery order, unreadable files reported and
    skipped. Lazy so a limited run does not parse the whole machine's history
    to take in five sessions."""
    for path in paths:
        try:
            yield load(path)
        except OSError as exc:
            print(f"!! {path}: {exc}")


def _load_all(paths) -> list:
    return list(_load_lazily(paths))


def _ingest(args) -> int:
    from agmem.hooks import _resolve, open_doc_store

    if args.limit is not None and args.limit < 0:
        print("--limit must be a non-negative number of sessions", file=sys.stderr)
        return 2
    distill = not args.no_distill and not args.dry_run
    if distill and (args.limit is None or args.limit > MAX_DISTILL_LIMIT):
        print(
            f"refusing to distil without a bounded batch: pass --limit N with "
            f"N <= {MAX_DISTILL_LIMIT} (or --no-distill / --dry-run to run without a model call)",
            file=sys.stderr,
        )
        return 2

    paths = _discover(args)

    if args.dry_run:
        # The doc store alone — through the same guard the recall hook uses, so a
        # config whose doc store is not SQLite is refused here too rather than
        # answered from a fresh, empty file that would call every session new.
        # And not even that when there is no store yet: opening one would create
        # the file, and a dry run that leaves a memory.db behind is not dry.
        ns, root, _config = _resolve(args.namespace, args.data_dir)
        if not (root / ns / "memory.db").exists():
            shown = 0
            for traj in _load_lazily(paths):
                if not traj.steps:
                    print(
                        f"{traj.host:11s} {traj.id[:24]:24s} steps=    0 user=   0 empty, skipped"
                    )
                    continue
                print(
                    f"{traj.host:11s} {traj.id[:24]:24s} steps={len(traj.steps):5d} "
                    f"user={traj.user_turns:4d} would ingest (no store yet)"
                )
                shown += 1
                if args.limit is not None and shown >= args.limit:
                    break
            print(f"dry run — nothing written; {shown} session(s) would be ingested")
            return 0
        ns, store = open_doc_store(args.namespace, args.data_dir)
        shown = 0
        try:
            for traj in _load_lazily(paths):
                if not traj.steps:
                    # `add_session` returns before storing or distilling anything
                    # for an empty session, so it costs nothing and counts for nothing.
                    print(
                        f"{traj.host:11s} {traj.id[:24]:24s} steps=    0 user=   0 empty, skipped"
                    )
                    continue
                bounds = [traj.episode_id(0), traj.episode_id(len(traj.steps) - 1)]
                seen = len(store.get_episodes(bounds)) == 2
                state = "already ingested" if seen else "would ingest"
                print(
                    f"{traj.host:11s} {traj.id[:24]:24s} steps={len(traj.steps):5d} "
                    f"user={traj.user_turns:4d} {state}"
                )
                shown += not seen
                if args.limit is not None and shown >= args.limit:
                    break
        finally:
            store.close()
        print(f"dry run — nothing written; {shown} session(s) would be ingested")
        return 0

    from dataclasses import replace

    from agmem.config import AgmemConfig
    from agmem.embed.api_embedder import APIEmbedder
    from agmem.llm.client import default_trace_path
    from agmem.memory import AgenticMemory

    ns, root, config = _resolve(args.namespace, args.data_dir)
    if distill and (config is None or "distill" not in config.llm_roles):
        # An explicit failure, not a silent skip: the whole point of the paid
        # path is the distillation, and a run that quietly stored raw steps
        # instead would look like it had succeeded. Decided from the config,
        # before the embedder is loaded, so the refusal costs nothing.
        print(
            "refusing to distil: the resolved config has no LLM role. Point "
            "AGMEM_CONFIG at a config with an [llm.distill] section, or pass "
            "--no-distill to persist the raw steps only.",
            file=sys.stderr,
        )
        return 2
    if config is None:
        config = AgmemConfig(profile="lite", data_dir=root, sync_write=True)
    else:
        config = replace(config, data_dir=root, sync_write=True)
    mem = AgenticMemory(namespace=ns, organizers=["experience"], config=config)
    trace = None
    try:
        if distill and mem.llm is not None:
            # Every paid run keeps its full prompt/response trace, by default
            # beside the store it fed. Not optional: a run whose model replies
            # are gone cannot be replayed, re-scored, or explained.
            trace = (
                Path(args.trace).expanduser()
                if args.trace
                else default_trace_path(root / ns, "ingest")
            )
            trace.parent.mkdir(parents=True, exist_ok=True)
            mem.llm.trace_path = trace
        if isinstance(mem.embedder, APIEmbedder) and args.limit is None:
            # `--no-distill` is free only with a local embedder. Every step of
            # every session goes through the embedder, so with an API-backed one
            # an unbounded backfill is an unbounded bill too.
            print(
                "refusing an unbounded run: the resolved embedder is API-backed, so "
                "persisting raw steps is a paid call per batch. Pass --limit N.",
                file=sys.stderr,
            )
            return 2
        processed = 0
        for traj in _load_lazily(paths):
            if not traj.steps:
                # Same line the dry run prints, and the same accounting: an
                # empty session stores nothing, calls nothing, and must not
                # consume the limit — the 2026-09-04 smoke lost its second
                # session to one.
                print(f"{traj.host:11s} {traj.id[:24]:24s} steps=    0 user=   0 empty, skipped")
                continue
            before = mem.log.count()
            ingest = mem.add_session(traj, outcome=args.outcome, distill=distill, force=args.force)
            mem.flush()
            state = (
                "already ingested"
                if ingest.already_ingested and not ingest.dispatched
                else f"persisted={len(ingest.episode_ids)}"
            )
            print(
                f"{traj.host:11s} {traj.id[:24]:24s} steps={len(traj.steps):5d} "
                f"{state} ops={mem.log.count() - before}"
            )
            # The limit counts sessions this run actually took in. Counting
            # discovered files instead would let the already-ingested head of the
            # listing consume the whole budget, and a backfill run in slices would
            # never advance past it.
            processed += not (ingest.already_ingested and not ingest.dispatched)
            if args.limit is not None and processed >= args.limit:
                break
        print(f"{processed} session(s) ingested this run")
        if distill:
            print(f"budget: {mem.budget.summary()}")
        if trace is not None:
            print(f"trace: {trace}")
    finally:
        mem.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m agmem.sessions")
    sub = ap.add_subparsers(dest="cmd", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("--project", default=None, help="cwd whose Claude Code sessions to list")
    scan.add_argument("--host", choices=["claude-code", "codex", "all"], default="all")
    render = sub.add_parser("render")
    render.add_argument("path")
    render.add_argument("--max-chars", type=int, default=None)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("paths", nargs="*", help="session files; omit to discover them")
    ingest.add_argument("--project", default=None, help="cwd whose Claude Code sessions to ingest")
    ingest.add_argument("--host", choices=["claude-code", "codex", "all"], default="all")
    ingest.add_argument(
        "--limit",
        type=int,
        default=None,
        help="most sessions to take in this run (already-ingested ones do not count)",
    )
    ingest.add_argument(
        "--no-distill",
        action="store_true",
        help="persist raw steps only; no model call (free with a local embedder)",
    )
    ingest.add_argument("--force", action="store_true", help="re-persist and re-distil")
    ingest.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ingest.add_argument("--outcome", default="unknown", help="caller's label for the session")
    ingest.add_argument(
        "--trace",
        default=None,
        help="where to append the full LLM I/O (default <data>/<ns>/traces/ingest-<stamp>.jsonl)",
    )
    ingest.add_argument("--namespace", default=None)
    ingest.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    if args.cmd == "render":
        traj = load(args.path)
        sys.stdout.write(traj.render(args.max_chars) + "\n")
        return 0

    if args.cmd == "ingest":
        return _ingest(args)

    paths = _discover(args)
    total = 0
    for traj in _load_all(paths):
        total += len(traj.steps)
        print(
            f"{traj.host:11s} {traj.id[:12]:12s} steps={len(traj.steps):5d} "
            f"user={traj.user_turns:4d} cwd={traj.cwd or '?'}"
        )
    print(f"{len(paths)} session(s), {total} step(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
