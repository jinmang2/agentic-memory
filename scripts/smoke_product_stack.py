"""End-to-end smoke for the product stack: hooks -> store -> MCP server.

    uv run python scripts/smoke_product_stack.py

WHAT THIS COVERS THAT THE SUITE DOES NOT. `tests/test_hooks.py` drives the hooks
and `tests/test_mcp_server.py` drives the server, both properly, and neither
crosses the seam between them. The seam is where the product claim lives: the
capture hook pays for an embedder specifically so that the server's
`search_memory` can find what it wrote, and nothing checked that it does.

It is a script rather than a test because it cannot be hermetic. The hooks build
their own memory internally (`hooks.open_memory` pins the lite profile), so
there is no seam through which a test could inject `FakeEmbedder` on both sides
— which means the real model, ~10 s, and a machine that has it cached. The suite
must stay runnable on a box with none of that.

It also prints the timings the wiring documentation quotes (`docs/05` §2.3-2.4),
so those numbers can be re-derived on a new machine instead of trusted.

It tells neither side which namespace to use unless `--namespace` is given.
The hooks and the server each resolve it from the environment and their
defaults, and the verdict includes whether they landed on the same directory —
the failure issue #2 reported, which the previous version of this script could
not see because it passed one explicit namespace to both.

Exit code is the verdict: 0 if a prompt captured by the hook came back from the
server's search out of the same store, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

NEEDLE = "Reykjavik"
PROMPT = f"Remember that the {NEEDLE} deployment window is the second Tuesday of each month."
QUERY = "When can I deploy?"


def run_hook(module: str, payload: dict, env: dict) -> tuple[float, subprocess.CompletedProcess]:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", module],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
    )
    return time.perf_counter() - started, proc


def server_command() -> str:
    """The console script next to the running interpreter.

    Resolved from `sys.executable` rather than PATH on purpose: PATH would let
    the smoke pass against some other installation of agmem, which is the exact
    confusion `docs/05` §2.3 warns about for MCP client registration.
    """
    return str(Path(sys.executable).parent / "agmem-mcp")


def _text(result) -> str:
    return " ".join(getattr(c, "text", "") for c in (getattr(result, "content", None) or []))


async def search_over_mcp(env: dict) -> tuple[float, str, set, str]:
    """Start the server the way a registered MCP client would and search.

    Deliberately passes NO --namespace and NO --data-dir: the server has to
    find the store through the same environment the hooks used, or through
    its defaults. That is the seam issue #2 found open — each layer used to
    resolve the namespace on its own, with different defaults, and the old
    smoke handed both sides the same explicit value, so it could not fail
    the way a real registration did.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=server_command(), args=["--organizers", ""], env=env)
    started = time.perf_counter()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            handshake = time.perf_counter() - started
            tools = {t.name for t in (await session.list_tools()).tools}
            rendered = _text(await session.call_tool("search_memory", {"query": QUERY}))
            stats = json.loads(_text(await session.call_tool("memory_stats", {})))
    return handshake, rendered, tools, stats["stats"]["namespace"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--data-dir",
        default=None,
        help="where to build the throwaway store (default: a temp dir)",
    )
    ap.add_argument(
        "--namespace",
        default=None,
        help="export AGMEM_NAMESPACE for both layers (default: neither side is told, "
        "so the check is that their built-in defaults agree)",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(tempfile.mkdtemp(prefix="agmem-"))
    # AGMEM_CONFIG passes through from the caller's environment on purpose: a
    # machine that configured its embedder there should be smoked on it.
    env = {k: v for k, v in os.environ.items() if k != "AGMEM_NAMESPACE"}
    env["AGMEM_DATA_DIR"] = str(data_dir)
    if args.namespace:
        env["AGMEM_NAMESPACE"] = args.namespace
    print(f"store: {data_dir}")
    print(f"namespace: {args.namespace or '(defaults on both sides)'}")

    capture_s, capture = run_hook(
        "agmem.hooks.capture",
        {"session_id": "smoke", "prompt": PROMPT, "hook_event_name": "UserPromptSubmit"},
        env,
    )
    print(f"capture hook      {capture_s:6.2f}s  exit={capture.returncode}")
    if capture.returncode != 0:
        print(f"  stderr: {capture.stderr[-500:]}")

    recall_s, recall = run_hook(
        "agmem.hooks.recall",
        {"session_id": "smoke", "hook_event_name": "SessionStart"},
        env,
    )
    recalled = NEEDLE in recall.stdout
    print(f"recall hook       {recall_s:6.2f}s  exit={recall.returncode}  found={recalled}")

    handshake_s, rendered, tools, server_ns = asyncio.run(search_over_mcp(env))
    searched = NEEDLE in rendered
    print(f"mcp handshake     {handshake_s:6.2f}s  tools={len(tools)}")
    print(f"mcp search_memory         found={searched}  namespace={server_ns}")

    # The hooks wrote under `<data_dir>/<namespace>/`; the server reports the
    # namespace it resolved. One directory, and the same name, is the proof
    # that both layers resolved the store identically without being told.
    written = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    same_store = written == [server_ns]
    print(f"store dirs        {written}  server={server_ns}  same={same_store}")

    # The recall hook reads the doc store and the server searches vectors, so
    # the two answer different questions about the same write: one proves the
    # episode was persisted, the other that it was embedded. A capture that
    # skipped the embedder would still satisfy the first.
    ok = recalled and searched and same_store
    print("\nVERDICT:", "ok" if ok else "FAILED")
    if not ok:
        print(f"  persisted={recalled} embedded_and_searchable={searched} same_store={same_store}")
        print(f"  search returned: {rendered[:400]}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
