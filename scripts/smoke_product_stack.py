"""End-to-end smoke for the product stack: hooks -> store -> MCP server.

Hermetic, no-download/no-model-call path:

    uv run python scripts/smoke_product_stack.py --hermetic --daemon

The default still honors the caller's `AGMEM_CONFIG`, which lets a maintainer
smoke a real configured backend deliberately. `--hermetic` writes a temp config
that forces `FakeEmbedder`; with `--daemon` it also points `[llm.distill]` at a
local OpenAI-compatible stub so the experience distiller can produce a runbook
without an external model or API key.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.smoke_product_stack_core import (
    NEEDLE,
    PROMPT,
    mcp_search,
    run_hook,
)
from scripts.smoke_product_stack_daemon import (
    daemon_path,
)
from scripts.smoke_product_stack_fixture import (
    configure_env,
    openai_stub,
    runbook_reply,
)


def parse_args() -> argparse.Namespace:
    description = (__doc__ or "End-to-end smoke for the product stack.").splitlines()[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--daemon", action="store_true", help="also exercise owned HTTP daemon hooks"
    )
    parser.add_argument(
        "--hermetic", action="store_true", help="force FakeEmbedder and local stub LLM"
    )
    parser.add_argument("--data-dir", default=None, help="throwaway store root (default: temp dir)")
    parser.add_argument("--namespace", default=None, help="export AGMEM_NAMESPACE for both layers")
    return parser.parse_args()


def run(args: argparse.Namespace, llm_url: str | None) -> int:
    data_dir = Path(args.data_dir) if args.data_dir else Path(tempfile.mkdtemp(prefix="agmem-"))
    env = configure_env(args.namespace, args.hermetic, data_dir, llm_url)
    print(f"store: {data_dir}")
    print(f"namespace: {args.namespace or '(defaults on both sides)'}")
    print(
        f"backend: {'hermetic FakeEmbedder/local stub' if args.hermetic else 'caller AGMEM_CONFIG'}"
    )
    ok = verify_capture_recall_mcp(env, data_dir)
    if args.daemon:
        ok = daemon_path(env, data_dir) and ok
    print("\nVERDICT:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def verify_capture_recall_mcp(env: dict[str, str], data_dir: Path) -> bool:
    capture_s, capture = run_hook(
        "agmem.hooks.capture",
        {"session_id": "smoke", "prompt": PROMPT, "hook_event_name": "UserPromptSubmit"},
        env,
    )
    print(f"capture hook      {capture_s:6.2f}s  exit={capture.returncode}")
    recall_s, recall = run_hook(
        "agmem.hooks.recall", {"session_id": "smoke", "hook_event_name": "SessionStart"}, env
    )
    recalled = NEEDLE in recall.stdout
    print(f"recall hook       {recall_s:6.2f}s  exit={recall.returncode}  found={recalled}")
    try:
        handshake_s, rendered, tools, server_ns = mcp_search(env)
    except (OSError, TimeoutError, subprocess.SubprocessError) as exc:
        print(f"mcp search failed: {exc}")
        rendered, tools, server_ns, handshake_s = "", set(), "", 0.0
    searched = NEEDLE in rendered
    print(f"mcp handshake     {handshake_s:6.2f}s  tools={len(tools)}")
    print(f"mcp search_memory         found={searched}  namespace={server_ns}")
    written = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    same_store = written == [server_ns]
    print(f"store dirs        {written}  server={server_ns}  same={same_store}")
    if not (
        recalled and searched and same_store and capture.returncode == 0 and recall.returncode == 0
    ):
        print(f"  persisted={recalled} embedded_and_searchable={searched} same_store={same_store}")
        print(f"  search returned: {rendered[:400]}")
        return False
    return True


def main() -> int:
    args = parse_args()
    if args.hermetic and args.daemon:
        with openai_stub([runbook_reply()]) as (llm_url, _requests):
            return run(args, llm_url)
    return run(args, None)


if __name__ == "__main__":
    raise SystemExit(main())
