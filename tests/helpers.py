from contextlib import contextmanager

"""Shared test doubles."""


def make_mem_multi(organizers, llm):
    """make_mem, generalized to a list of organizer instances (Task 12) —
    for scenarios where one organizer's output chains into another's
    on_memory_event (e.g. Nemori episodes -> MemoryOS pages)."""
    from agmem import AgenticMemory
    from agmem.embed.fake import FakeEmbedder

    mem = AgenticMemory(namespace="t", organizers=list(organizers), embedder=FakeEmbedder(dim=128))
    mem.structured = llm
    mem._ctx.llm = llm
    return mem


class StubLLM:
    """StructuredCaller stand-in: returns queued responses per role."""

    def __init__(self, responses: dict[str, list]):
        self.responses = {role: list(items) for role, items in responses.items()}
        self.calls: list[tuple[str, str]] = []
        self.systems: list[str] = []  # system message per call, "" if none
        self.drops: dict[str, int] = {}

    def call(self, role, prompt, schema, required_keys=(), **kwargs):
        self.calls.append((role, prompt))
        self.systems.append(str(kwargs.get("system", "")))
        items = self.responses.get(role)
        if not items:
            self.drops[role] = self.drops.get(role, 0) + 1
            return None
        return items.pop(0)


@contextmanager
def openai_stub(replies: list[str]):
    """A local OpenAI-compatible `/chat/completions` endpoint answering with the
    queued `replies` in order (the last one repeats), so a CLI run in its own
    process makes a real HTTP call against no model and no key.

    Yields `(base_url, requests)`; `requests` collects each POST body so a test
    can assert what the CLI actually sent."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    queue = list(replies)
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
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

        def log_message(self, *_args):  # keep pytest output clean
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
