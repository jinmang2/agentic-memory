"""`python -m agmem.explore export|ask …` — the explorer read path from a shell.

    python -m agmem.explore export                      # write the workspace, no model
    python -m agmem.explore ask "how do I run the bench?"   # one exploration, N model calls

`export` costs nothing and is what `ask` does first anyway. `ask` is bounded
the way `agmem.sessions ingest` is: `--max-steps` is capped, and a config with
no `[llm.explore]` role is refused up front rather than answered from a vector
search the caller did not ask for.
"""

from __future__ import annotations

import argparse
import json
import sys

from agmem.explore.explorer import MAX_STEPS_CAP


def _open(args):
    from dataclasses import replace

    from agmem.config import AgmemConfig
    from agmem.hooks import _resolve
    from agmem.memory import AgenticMemory

    ns, root, config = _resolve(args.namespace, args.data_dir)
    if config is None:
        config = AgmemConfig(profile="lite", data_dir=root, sync_write=True)
    else:
        config = replace(config, data_dir=root, sync_write=True)
    return ns, root, config, AgenticMemory(namespace=ns, organizers=["experience"], config=config)


def _workspace_root(args, root, ns):
    from pathlib import Path

    return Path(args.root) if args.root else root / ns / "workspace"


def _export(args) -> int:
    from agmem.explore import export_workspace

    ns, root, _config, mem = _open(args)
    try:
        stats = export_workspace(mem, _workspace_root(args, root, ns))
    finally:
        mem.close()
    print(
        f"workspace {_workspace_root(args, root, ns)}: sessions={stats.sessions} "
        f"runbooks={stats.runbooks} messages={stats.messages} "
        f"written={stats.written} unchanged={stats.unchanged} removed={stats.removed}"
    )
    return 0


def _ask(args) -> int:
    from agmem.hooks import _resolve

    if args.max_steps < 1 or args.max_steps > MAX_STEPS_CAP:
        print(f"--max-steps must be between 1 and {MAX_STEPS_CAP}", file=sys.stderr)
        return 2
    _ns, _root, config = _resolve(args.namespace, args.data_dir)
    if config is None or "explore" not in config.llm_roles:
        print(
            "refusing to explore: the resolved config has no [llm.explore] role. "
            "Add one (see agmem.example.toml) or run `export` to only write the workspace.",
            file=sys.stderr,
        )
        return 2
    ns, root, _config, mem = _open(args)
    try:
        result = mem.research(
            args.query,
            root=_workspace_root(args, root, ns),
            max_steps=args.max_steps,
            budget_tokens=args.budget_tokens,
        )
        budget = mem.budget.summary()
    finally:
        mem.close()
    print(result.context or "(no context)")
    print()
    print("citations: " + json.dumps(result.citations))
    print(
        f"latency_s={result.latency_s:.2f} llm_calls={result.llm_calls} "
        f"steps={len(result.steps)} search_tool={result.search_tool} "
        f"degraded={result.degraded}"
    )
    print(f"budget: {budget}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m agmem.explore")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("export", "ask"):
        p = sub.add_parser(name)
        if name == "ask":
            p.add_argument("query")
            p.add_argument("--max-steps", type=int, default=8)
            p.add_argument("--budget-tokens", type=int, default=4000)
        p.add_argument("--root", default=None, help="workspace dir (default <data>/<ns>/workspace)")
        p.add_argument("--namespace", default=None)
        p.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    return _export(args) if args.cmd == "export" else _ask(args)


if __name__ == "__main__":
    raise SystemExit(main())
