"""The explorer: a model that greps the workspace and reads the passages it
found, then hands back context it can cite.

WHY THIS AND NOT A BETTER INDEX. Every controlled result the v1 survey found
for episodic memory (docs/research/agent-memory-axes-v1.md §3.2) is the value
of *reaching the raw trajectory*, not of a cleverer summary of it:
LongMemEval-V2's AgentRunbook-C — no embedding index, a coding agent with
``rg`` and ``find`` over the trajectories — is the frontier of that benchmark
at 74.9 against 58.6 for the same notes behind a vector search. GAM makes the
same argument as design: keep the original, build the context just in time.
This module is that read path for this store, over the file view
``workspace.py`` writes.

THE LOOP IS JSON ACTIONS, NOT NATIVE TOOL CALLS. One structured call per
step, the model returning ``{"action": ..., ...}``; the tools run here, under
this process's rules, and the observation goes back as text. That keeps the
loop model-agnostic (a 9B model behind an OpenAI-compatible endpoint has no
tool-call contract worth relying on), and it keeps every side effect in code
that is ours to bound: argv lists, never a shell; every path resolved and
required to stay inside the workspace; output clipped; a timeout per tool.

WHAT IT REFUSES TO DO. It never falls back to a vector search. A dropped
call, a model that will not answer, a path outside the root — each becomes a
recorded observation or a ``degraded`` reason on the result, so a run that
produced nothing says so. The measurement this path exists for (§6: the
explorer against the vector read, at what latency) is meaningless if one arm
can quietly become the other.

LATENCY IS THE POINT, NOT A SIDE NOTE. ``ResearchResult.latency_s`` is the
wall clock of the whole call and each step carries its own seconds, because
LongMemEval-V2 scores accuracy *against* latency (LAFS) and the survey's one
firm number for this path is "400x slower than the index". A caller that
cannot see the seconds cannot place itself on that frontier.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger("agmem.explore")

ACTIONS = ("search", "list", "read", "final")
# The most exploration steps any entry point allows — the CLI and the MCP tool
# both clamp to this, because a step is a model call and the MCP surface is
# filled in by a model, not a person.
MAX_STEPS_CAP = 12
MAX_BUDGET_TOKENS = 16_000
MAX_READ_LINES = 200
MAX_LIST_ENTRIES = 200
MAX_HITS_PER_FILE = 50  # rg --max-count and grep -m are both per file; _clip bounds the total
TOOL_TIMEOUT_S = 10.0
CHARS_PER_TOKEN = 4  # the same rough figure MemoryBundle.render budgets with

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "reason": {"type": "string"},
        "pattern": {"type": "string"},
        "path": {"type": "string"},
        "start": {"type": "integer"},
        "end": {"type": "integer"},
        "context": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "lines": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "required": ["file", "lines"],
            },
        },
    },
    "required": ["action"],
}

# Modelled on AgentRunbook-C's instruction (docs/research/longmemeval.md §7.1):
# explore before answering, quote exact strings, and treat the derived notes
# as derived.
SYSTEM_PROMPT = """You are the memory of a coding agent, answering a question about past work \
by exploring a workspace of files with search and read tools. You must explore before you \
answer; never answer from the question alone.

The workspace:
- INDEX.md — one line per past session and per runbook. Read it first when you do not know \
where to look.
- sessions/<host>/<session-id>.md — the raw transcripts, ground truth. Steps are labelled \
[i] USER / ASSISTANT / TOOL_CALL(tool) / TOOL_RESULT(tool).
- runbooks/<id>.md — notes distilled from sessions. Derived, not authoritative: when a runbook \
and a transcript disagree, the transcript wins, and a runbook's `source:` line tells you \
which session and steps to check.
- messages/<YYYY-MM>.md — single messages and task lines outside any session, labelled by role.

Each turn, return ONE JSON object and nothing else:
- {"action": "search", "pattern": "<regex>", "path": "<dir or file, optional>", "reason": "..."}
- {"action": "list", "path": "<dir>", "reason": "..."}
- {"action": "read", "path": "<file>", "start": <line>, "end": <line>, "reason": "..."}
- {"action": "final", "context": "<what a future agent must know>", "citations": [{"file": \
"<path>", "lines": [<first>, <last>]}], "reason": "..."}

Rules: quote exact strings (flags, paths, error text, ids) from what you read; do not invent \
anything you did not see; cite the file and line range for each claim in `context`; keep \
`context` short enough to drop into an agent's prompt; say plainly in `context` when the \
workspace holds nothing relevant. Paths are relative to the workspace root."""

USER_TEMPLATE = """Question: {query}

Steps so far ({used} of {max_steps} used):
{transcript}

Return the next action as JSON."""

FORCED_SUFFIX = """

You have used all exploration steps. You must answer now: return action "final" with the \
best context you can cite from what you have already seen."""


@dataclass
class ResearchResult:
    """What one exploration produced, and what it cost.

    ``degraded`` is None on a normal finish; otherwise the reason nothing (or
    less than asked) came back — ``"llm_drop"`` when a structured call was
    dropped, ``"max_steps"`` when the model would not answer even when told to.
    ``context`` is then empty rather than made up."""

    query: str
    context: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    latency_s: float = 0.0
    # Seconds spent materializing the workspace before the loop, when the
    # caller (``AgenticMemory.research``) did that; included in ``latency_s``,
    # since the vector arm's latency covers everything it does per query too.
    export_s: float = 0.0
    llm_calls: int = 0
    search_tool: str = "grep"
    degraded: str | None = None


class Explorer:
    """The grep-and-read agent over one workspace root."""

    def __init__(
        self,
        root: Path | str,
        *,
        max_steps: int = 8,
        max_observation_chars: int = 4000,
        budget_tokens: int = 4000,
        role: str = "explore",
        search_tool: str | None = None,
    ) -> None:
        """``search_tool`` is ``"rg"`` or ``"grep"``; None picks ``rg`` when it
        is on PATH and ``grep`` otherwise. Both give ``file:line:text`` and the
        choice is reported on the result, because the two do not match
        identically (``rg`` respects ignore files, ``grep`` does not) and a
        measurement has to say which one ran."""
        self.root = Path(root).resolve()
        self.max_steps = max_steps
        self.max_observation_chars = max_observation_chars
        self.budget_tokens = budget_tokens
        self.role = role
        if search_tool is None:
            search_tool = "rg" if shutil.which("rg") else "grep"
        if search_tool not in ("rg", "grep"):
            raise ValueError(f"search_tool must be 'rg' or 'grep', not {search_tool!r}")
        self.search_tool = search_tool
        if max_steps < 1:
            # Zero steps would answer without exploring, which the system prompt
            # forbids; the entry points clamp, and a library caller is refused.
            raise ValueError("max_steps must be at least 1")

    # ---- the loop -----------------------------------------------------------

    def research(self, query: str, llm: Any) -> ResearchResult:
        """Run the loop with ``llm`` (a ``StructuredCaller`` or a stand-in with
        the same ``call``) and return the cited context."""
        started = perf_counter()
        result = ResearchResult(query=query, search_tool=self.search_tool)
        transcript: list[str] = []
        for _ in range(self.max_steps):
            reply = self._call(llm, query, transcript, len(result.steps), result)
            if reply is None:
                result.degraded = "llm_drop"
                break
            if reply.get("action") == "final":
                self._finish(reply, result)
                break
            self._act(reply, transcript, result)
        else:
            # Out of steps without an answer: one more call, told to answer.
            reply = self._call(llm, query, transcript, len(result.steps), result, forced=True)
            if reply is None:
                result.degraded = "llm_drop"
            elif reply.get("action") == "final":
                self._finish(reply, result)
            else:
                result.degraded = "max_steps"
        result.latency_s = perf_counter() - started
        return result

    def _call(
        self,
        llm: Any,
        query: str,
        transcript: list[str],
        used: int,
        result: ResearchResult,
        forced: bool = False,
    ) -> dict[str, Any] | None:
        prompt = USER_TEMPLATE.format(
            query=query,
            used=used,
            max_steps=self.max_steps,
            transcript="\n\n".join(transcript)
            if transcript
            else "(none yet — start with INDEX.md)",
        )
        if forced:
            prompt += FORCED_SUFFIX
        result.llm_calls += 1
        reply = llm.call(
            self.role,
            prompt,
            SCHEMA,
            required_keys=("action",),
            system=SYSTEM_PROMPT,
            phase="explore",
        )
        if reply is not None and reply.get("action") not in ACTIONS:
            # An unknown action is a wasted step, recorded as one rather than
            # treated as an answer.
            transcript.append(f"## step {used + 1}: unknown action {reply.get('action')!r}")
            result.steps.append(
                {
                    "action": str(reply.get("action")),
                    "observation": "unknown action",
                    "seconds": 0.0,
                }
            )
            return {"action": "noop"}
        return reply

    def _act(self, reply: dict[str, Any], transcript: list[str], result: ResearchResult) -> None:
        action = reply.get("action")
        if action == "noop":
            return
        t0 = perf_counter()
        if action == "search":
            argument = f"{reply.get('pattern', '')!s} in {reply.get('path') or '.'}"
            observation = self._search(str(reply.get("pattern") or ""), reply.get("path"))
        elif action == "list":
            argument = str(reply.get("path") or ".")
            observation = self._list(reply.get("path"))
        else:  # read
            argument = f"{reply.get('path')} {reply.get('start')}-{reply.get('end')}"
            observation = self._read(reply.get("path"), reply.get("start"), reply.get("end"))
        observation = self._clip(observation)
        seconds = perf_counter() - t0
        result.steps.append(
            {
                "action": action,
                "reason": str(reply.get("reason") or ""),
                "argument": argument,
                "observation": observation,
                "observation_chars": len(observation),
                "seconds": seconds,
            }
        )
        transcript.append(f"## step {len(result.steps)}: {action} {argument}\n{observation}")

    def _finish(self, reply: dict[str, Any], result: ResearchResult) -> None:
        t0 = perf_counter()
        context = str(reply.get("context") or "")
        limit = self.budget_tokens * CHARS_PER_TOKEN
        if len(context) > limit:
            context = context[:limit] + "\n…[context truncated to budget]"
        kept: list[dict[str, Any]] = []
        dropped = 0
        for citation in reply.get("citations") or []:
            valid = self._validate_citation(citation)
            if valid is None:
                dropped += 1
            else:
                kept.append(valid)
        result.context = context
        result.citations = kept
        result.steps.append(
            {
                "action": "final",
                "reason": str(reply.get("reason") or ""),
                "citations": len(kept),
                "dropped_citations": dropped,
                "seconds": perf_counter() - t0,
            }
        )

    # ---- tools --------------------------------------------------------------

    def _resolve(self, path: Any) -> tuple[Path | None, str | None]:
        """A path the model named, resolved inside the root — or the reason it
        is not. Absolute paths and ``..`` both land outside and are refused the
        same way; a symlink that escapes is caught by ``resolve()``."""
        raw = str(path or ".")
        candidate = (self.root / raw).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            return None, f"refused: {raw!r} is outside the workspace"
        return candidate, None

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _search(self, pattern: str, path: Any) -> str:
        if not pattern:
            return "refused: search needs a non-empty pattern"
        target, error = self._resolve(path)
        if error:
            return error
        if not target.exists():
            return f"no such path: {self._relative(target) or '.'}"
        rel = self._relative(target) or "."
        if self.search_tool == "rg":
            # --no-ignore: the workspace is a projection of the store, not a
            # checkout, so a .gitignore or .rgignore that happens to cover it
            # must not turn real matches into "(no matches)". The trailing "--"
            # keeps a file name that looks like a flag from being read as one.
            argv = [
                "rg",
                "-n",
                "--no-heading",
                "--no-ignore",
                "--color",
                "never",
                "--max-count",
                str(MAX_HITS_PER_FILE),
                "-e",
                pattern,
                "--",
                rel,
            ]
        else:
            argv = ["grep", "-rn", "-I", "-m", str(MAX_HITS_PER_FILE), "--", pattern, rel]
        try:
            proc = subprocess.run(
                argv,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=TOOL_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"search timed out after {TOOL_TIMEOUT_S:.0f}s"
        except OSError as exc:
            return f"search tool failed: {exc}"
        if proc.returncode not in (0, 1):
            return f"search failed: {proc.stderr.strip()[:500]}"
        lines = [line.removeprefix("./") for line in proc.stdout.splitlines()]
        return "\n".join(lines) if lines else "(no matches)"

    def _list(self, path: Any) -> str:
        target, error = self._resolve(path)
        if error:
            return error
        if not target.is_dir():
            return f"not a directory: {self._relative(target) or '.'}"
        entries = sorted(self._relative(p) for p in target.rglob("*") if p.is_file())
        if not entries:
            return "(empty)"
        shown = entries[:MAX_LIST_ENTRIES]
        more = len(entries) - len(shown)
        return "\n".join(shown) + (f"\n…[{more} more]" if more else "")

    def _read(self, path: Any, start: Any, end: Any) -> str:
        target, error = self._resolve(path)
        if error:
            return error
        if not target.is_file():
            return f"no such file: {self._relative(target)}"
        try:
            first = max(1, int(start or 1))
            last = int(end) if end is not None else first + MAX_READ_LINES - 1
        except (TypeError, ValueError):
            return "refused: start and end must be integers"
        if last < first:
            return "refused: end before start"
        last = min(last, first + MAX_READ_LINES - 1)
        lines = target.read_text(errors="replace").splitlines()
        if first > len(lines):
            return f"beyond end of file ({len(lines)} lines)"
        chunk = lines[first - 1 : last]
        return "\n".join(f"{first + i}: {line}" for i, line in enumerate(chunk))

    def _clip(self, text: str) -> str:
        if len(text) <= self.max_observation_chars:
            return text
        return text[: self.max_observation_chars] + "\n…[observation clipped]"

    def _validate_citation(self, citation: Any) -> dict[str, Any] | None:
        if not isinstance(citation, dict):
            return None
        target, error = self._resolve(citation.get("file"))
        if error or target is None or not target.is_file():
            return None
        lines = citation.get("lines")
        if not (isinstance(lines, list) and len(lines) == 2):
            return None
        try:
            first, last = int(lines[0]), int(lines[1])
        except (TypeError, ValueError):
            return None
        count = len(target.read_text(errors="replace").splitlines())
        if not 1 <= first <= last <= count:
            return None
        return {"file": self._relative(target), "lines": [first, last]}


__all__ = [
    "ACTIONS",
    "MAX_BUDGET_TOKENS",
    "MAX_STEPS_CAP",
    "SCHEMA",
    "SYSTEM_PROMPT",
    "Explorer",
    "ResearchResult",
]
