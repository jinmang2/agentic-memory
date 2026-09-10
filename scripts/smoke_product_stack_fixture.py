from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from scripts.smoke_product_stack_core import RUNBOOK_KEYWORD, RUNBOOK_NAME


@contextmanager
def openai_stub(
    replies: Sequence[str],
) -> Iterator[tuple[str, list[Mapping[str, str | int | bool | list[dict[str, str]]]]]]:
    queue = list(replies)
    requests: list[Mapping[str, str | int | bool | list[dict[str, str]]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            requests.append(body)
            content = queue.pop(0) if len(queue) > 1 else queue[0]
            payload = json.dumps(
                {
                    "id": "stub",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model", "stub"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()


def runbook_reply() -> str:
    return json.dumps(
        {
            "summary": "kept product smoke memory",
            "tasks": [
                {
                    "name": RUNBOOK_NAME,
                    "outcome": "success",
                    "procedure": ["preserve the transcript", "distill the rollout"],
                    "keywords": [RUNBOOK_KEYWORD],
                    "stage": "verify",
                    "steps": [0, 1],
                }
            ],
        }
    )


def write_rollout(path: Path, cwd: Path) -> None:
    ts = "2026-09-09T10:00:00.000Z"
    records = [
        {
            "timestamp": ts,
            "type": "session_meta",
            "payload": {
                "id": "product-smoke",
                "session_id": "product-smoke",
                "timestamp": ts,
                "cwd": str(cwd),
                "thread_source": "user",
                "source": "cli",
            },
        },
        {
            "timestamp": ts,
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"distill {RUNBOOK_KEYWORD}"}],
            },
        },
        {
            "timestamp": ts,
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "preserved and ready"}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def write_config(path: Path, llm_url: str | None = None) -> None:
    content = '[profile]\nname = "lite"\n\n[override]\nembedder = "FakeEmbedder"\n'
    if llm_url is not None:
        content += f'\n[llm.distill]\nendpoint = "{llm_url}"\nmodel = "stub"\napi_key = "stub"\n'
    path.write_text(content)


def configure_env(
    namespace: str | None, hermetic: bool, data_dir: Path, llm_url: str | None
) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "AGMEM_NAMESPACE"}
    env["AGMEM_DATA_DIR"] = str(data_dir)
    env["AGMEM_NO_DAEMON"] = "1"
    env["AGMEM_DAEMON_URL"] = "http://127.0.0.1:1"
    if namespace:
        env["AGMEM_NAMESPACE"] = namespace
    if hermetic:
        cfg = data_dir / "agmem-hermetic.toml"
        write_config(cfg, llm_url)
        env["AGMEM_CONFIG"] = str(cfg)
        env["PATH"] = os.environ.get("PATH", "")
    return env
