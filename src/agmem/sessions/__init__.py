"""Session logs of coding agents, read into one trajectory shape.

v1 moves the domain from conversational QA to agent trajectories, and the first
trajectories at hand are our own: every Claude Code session lands in
`~/.claude/projects/<project-key>/<session>.jsonl`, every Codex session in
`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`. This package reads both into a
`SessionTrajectory` — an ordered list of steps that are user text, assistant
text, tool calls and tool results — so that everything downstream (the
`experience` organizer, the daemon's backfill, the LongMemEval-V2 adapter) sees
one shape and never learns a host's record format.

WHAT IS KEPT AND WHAT IS NOT follows the filter Codex applies to its own
rollouts before memory extraction (`codex-rs/rollout/src/policy.rs` and
`memories/write/src/phase1.rs`, read at rust-v0.151.0 for
docs/research/product-memory-landscape.md §1.3):

- kept: user messages, assistant messages, tool calls, tool outputs;
- dropped: reasoning/thinking, developer and system messages, sub-agent
  side-chains, and the fragments the harness injects into user turns
  (system reminders, slash-command wrappers, task notifications, AGENTS.md
  and skill bodies) — text the user never typed and the next agent will get
  again anyway;
- tool outputs are cut to head and tail, because a memory that copies a
  build log is a build log;
- secrets are redacted before anything leaves this module. The patterns are
  the common key shapes, not a guarantee; Phase 3 of the v1 plan decides the
  fuller policy before any session data is captured for real.

Nothing here calls a model. Reading a session is a parse, and it has to stay
one so the daemon can backfill thousands of steps without a bill.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HOST_CLAUDE_CODE = "claude-code"
HOST_CODEX = "codex"

# Per-step caps. Tool outputs keep head and tail; text keeps the head.
TOOL_OUTPUT_MAX_CHARS = 2_000
TEXT_MAX_CHARS = 8_000
TOOL_INPUT_MAX_CHARS = 1_000

REDACTED = "[REDACTED_SECRET]"

# Shapes of credentials that show up in shell output and pasted config. Ordered
# longest-match first where prefixes overlap. Deliberately not exhaustive.
_SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"\bsk-(?:proj-|ant-|or-v1-)?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}"),
    re.compile(
        r"(?i)\b[A-Za-z0-9_]*(api[_-]?key|secret|token|password)\s*[=:]\s*['\"]?[A-Za-z0-9._\-]{16,}"
    ),
]

# Fragments the harness injects into the user turn. Each is removed wholesale.
_INJECTED_BLOCKS = [
    re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL),
    re.compile(r"<local-command-[a-z]+>.*?</local-command-[a-z]+>", re.DOTALL),
    re.compile(r"<command-[a-z]+>.*?</command-[a-z]+>", re.DOTALL),
    re.compile(r"<task-notification>.*?</task-notification>", re.DOTALL),
    re.compile(r"<teammate-message[^>]*>.*?</teammate-message>", re.DOTALL),
    re.compile(r"<recommended_plugins>.*?</recommended_plugins>", re.DOTALL),
    re.compile(r"<skill>.*?</skill>", re.DOTALL),
    re.compile(r"# AGENTS\.md instructions.*?</INSTRUCTIONS>", re.DOTALL),
    re.compile(r"<environment_context>.*?</environment_context>", re.DOTALL),
]


def redact_secrets(text: str) -> str:
    """Replace credential-shaped substrings with `REDACTED`."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def strip_injected(text: str) -> str:
    """Remove harness-injected fragments; returns "" when nothing the user wrote remains."""
    for pattern in _INJECTED_BLOCKS:
        text = pattern.sub("", text)
    return text.strip()


def clip_head_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n…[{omitted} chars omitted]…\n{text[-tail:]}"


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass
class Step:
    """One thing that happened, in order. `kind` is one of user / assistant /
    tool_call / tool_result; `text` is already stripped, clipped and redacted."""

    kind: str
    text: str
    timestamp: datetime | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionTrajectory:
    id: str
    host: str
    source_path: str
    cwd: str | None = None
    git_branch: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    steps: list[Step] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def user_turns(self) -> int:
        return sum(1 for s in self.steps if s.kind == "user")

    def episode_id(self, index: int) -> str:
        """The id the step at `index` gets as an `Episode`, derived rather than
        drawn.

        Ingest is idempotent because this is deterministic: reading the same
        session file again yields the same ids, and the doc store's
        INSERT OR REPLACE turns the second write into an overwrite instead of a
        duplicate. Host and session id are both in the hash, so two hosts that
        happen to name a session alike stay apart in one store. A step's
        position is its identity, which means an id survives the text changing
        (a wider clip, a new redaction pattern) and does NOT survive a step
        being inserted earlier — acceptable, because a session log is
        append-only once written.
        """
        digest = hashlib.sha1(f"{self.host}:{self.id}:{index}".encode()).hexdigest()
        return f"sess-{digest[:24]}"

    def to_episodes(self, namespace: str = "main"):
        """One `Episode` per step, carrying enough meta to rebuild the trajectory.

        `role` is the step kind so a reader can tell tool output from the user
        without opening meta; the daemon's backfill and the recency hook treat
        them all as episodes."""
        from agmem.core.types import Episode

        episodes = []
        for index, step in enumerate(self.steps):
            episodes.append(
                Episode(
                    id=self.episode_id(index),
                    content=step.text,
                    role=step.kind,
                    namespace=namespace,
                    timestamp=step.timestamp or self.started_at or datetime.now(UTC),
                    meta={
                        "source": self.host,
                        "session_id": self.id,
                        "step_index": index,
                        "kind": step.kind,
                        "cwd": self.cwd,
                        "tool_name": step.tool_name,
                        **step.meta,
                    },
                )
            )
        return episodes

    @property
    def task_text(self) -> str:
        """The session's opening user turn — what `add_task_result` gets as `task`."""
        for step in self.steps:
            if step.kind == "user":
                return step.text
        return ""

    def as_task_trajectory(self) -> list[dict[str, Any]]:
        """The step list the facade's `add_task_result` / `Organizer.on_task_end`
        take: one dict per step, with the session's host, cwd and id repeated on
        every step so an organizer can read them off the trajectory alone."""
        return [
            {
                "kind": step.kind,
                "text": step.text,
                "tool_name": step.tool_name,
                "timestamp": step.timestamp.isoformat() if step.timestamp else None,
                "host": self.host,
                "cwd": self.cwd,
                "session_id": self.id,
                "step_index": index,
                "episode_id": self.episode_id(index),
            }
            for index, step in enumerate(self.steps)
        ]

    def render(self, max_chars: int | None = None) -> str:
        """A plain transcript for a distiller to read: one line-block per step,
        labelled, in order. Head-clipped to `max_chars` when given."""
        header = [f"session: {self.id}", f"host: {self.host}"]
        if self.cwd:
            header.append(f"cwd: {self.cwd}")
        if self.git_branch:
            header.append(f"branch: {self.git_branch}")
        if self.started_at:
            header.append(f"started: {self.started_at.isoformat()}")
        blocks = ["\n".join(header)]
        for index, step in enumerate(self.steps):
            label = step.kind.upper()
            if step.tool_name:
                label += f"({step.tool_name})"
            blocks.append(f"[{index}] {label}\n{step.text}")
        text = "\n\n".join(blocks)
        return text if max_chars is None else clip_head_tail(text, max_chars)


def _finish(traj: SessionTrajectory) -> SessionTrajectory:
    stamps = [s.timestamp for s in traj.steps if s.timestamp]
    if stamps:
        traj.started_at = traj.started_at or min(stamps)
        traj.ended_at = max(stamps)
    return traj


def _text_step(kind: str, raw: str, ts: datetime | None, **meta: Any) -> Step | None:
    text = strip_injected(raw) if kind == "user" else raw.strip()
    if not text:
        return None
    text = redact_secrets(clip_head_tail(text, TEXT_MAX_CHARS))
    return Step(kind=kind, text=text, timestamp=ts, meta=meta)


def _tool_call_step(name: str, payload: Any, ts: datetime | None, call_id: str | None) -> Step:
    rendered = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return Step(
        kind="tool_call",
        text=redact_secrets(clip_head_tail(rendered, TOOL_INPUT_MAX_CHARS)),
        timestamp=ts,
        tool_name=name,
        tool_call_id=call_id,
    )


def _tool_result_step(
    output: Any, ts: datetime | None, call_id: str | None, tool_name: str | None, is_error: bool
) -> Step | None:
    text = _content_text(output)
    if not text.strip():
        return None
    return Step(
        kind="tool_result",
        text=redact_secrets(clip_head_tail(text.strip(), TOOL_OUTPUT_MAX_CHARS)),
        timestamp=ts,
        tool_name=tool_name,
        tool_call_id=call_id,
        meta={"is_error": is_error},
    )


def _content_text(content: Any) -> str:
    """Text out of the content shapes both hosts use: a string, or a list of parts
    with `text` / `input_text` / `output_text`."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("text", "input_text", "output_text"):
                    parts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


# --------------------------------------------------------------------------- Claude Code


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def load_claude_code(path: str | Path) -> SessionTrajectory:
    """One Claude Code session file into a trajectory.

    Record types other than `user` and `assistant` (mode, attachment, ai-title,
    file-history-*, queue-operation…) are harness bookkeeping. `isMeta`
    records are caveats the harness wrote in the user's voice; `isSidechain`
    records belong to a sub-agent's conversation — both excluded, as Codex
    excludes its own developer messages and sub-agent sessions."""
    path = Path(path)
    traj = SessionTrajectory(id=path.stem, host=HOST_CLAUDE_CODE, source_path=str(path))
    tool_names: dict[str, str] = {}
    for record in _iter_jsonl(path):
        kind = record.get("type")
        if kind not in ("user", "assistant") or record.get("isMeta") or record.get("isSidechain"):
            continue
        traj.cwd = traj.cwd or record.get("cwd")
        traj.git_branch = traj.git_branch or record.get("gitBranch")
        traj.meta.setdefault("session_id", record.get("sessionId"))
        traj.meta.setdefault("cli_version", record.get("version"))
        ts = _parse_ts(record.get("timestamp"))
        message = record.get("message") or {}
        content = message.get("content")
        if kind == "user":
            if isinstance(content, str):
                step = _text_step("user", content, ts)
                if step:
                    traj.steps.append(step)
                continue
            for part in content if isinstance(content, list) else []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    step = _text_step("user", str(part.get("text") or ""), ts)
                    if step:
                        traj.steps.append(step)
                elif part.get("type") == "tool_result":
                    call_id = part.get("tool_use_id")
                    step = _tool_result_step(
                        part.get("content"),
                        ts,
                        call_id,
                        tool_names.get(call_id or ""),
                        bool(part.get("is_error")),
                    )
                    if step:
                        traj.steps.append(step)
        else:
            traj.meta.setdefault("model", message.get("model"))
            for part in content if isinstance(content, list) else []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    step = _text_step("assistant", str(part.get("text") or ""), ts)
                    if step:
                        traj.steps.append(step)
                elif part.get("type") == "tool_use":
                    name = str(part.get("name") or "tool")
                    call_id = part.get("id")
                    if call_id:
                        tool_names[call_id] = name
                    traj.steps.append(_tool_call_step(name, part.get("input"), ts, call_id))
                # "thinking" parts are dropped on purpose.
    return _finish(traj)


# --------------------------------------------------------------------------- Codex


def load_codex(path: str | Path) -> SessionTrajectory:
    """One Codex rollout file into a trajectory.

    `session_meta` gives id, cwd and start; `response_item` payloads carry the
    conversation. Developer messages, reasoning items, `event_msg`,
    `turn_context` and `world_state` are dropped. `thread_source` is kept in
    meta so a caller can exclude sub-agent rollouts the way Codex's own
    memory pipeline does (it only claims interactive, root-thread sessions)."""
    path = Path(path)
    traj = SessionTrajectory(id=path.stem, host=HOST_CODEX, source_path=str(path))
    tool_names: dict[str, str] = {}
    for record in _iter_jsonl(path):
        kind = record.get("type")
        payload = record.get("payload") or {}
        ts = _parse_ts(record.get("timestamp"))
        if kind == "session_meta":
            traj.id = str(payload.get("id") or payload.get("session_id") or traj.id)
            traj.cwd = payload.get("cwd")
            traj.started_at = _parse_ts(payload.get("timestamp")) or ts
            traj.meta.update(
                {
                    "session_id": traj.id,
                    "cli_version": payload.get("cli_version"),
                    "thread_source": payload.get("thread_source"),
                    "source": payload.get("source"),
                    "model_provider": payload.get("model_provider"),
                }
            )
            continue
        if kind != "response_item":
            continue
        ptype = payload.get("type")
        if ptype == "message":
            role = payload.get("role")
            if role == "user":
                step = _text_step("user", _content_text(payload.get("content")), ts)
            elif role == "assistant":
                step = _text_step("assistant", _content_text(payload.get("content")), ts)
            else:
                step = None  # developer / system
            if step:
                traj.steps.append(step)
        elif ptype in ("custom_tool_call", "function_call"):
            name = str(payload.get("name") or "tool")
            call_id = payload.get("call_id")
            if call_id:
                tool_names[call_id] = name
            body = payload.get("input") if ptype == "custom_tool_call" else payload.get("arguments")
            traj.steps.append(_tool_call_step(name, body, ts, call_id))
        elif ptype in ("custom_tool_call_output", "function_call_output"):
            call_id = payload.get("call_id")
            step = _tool_result_step(
                payload.get("output"), ts, call_id, tool_names.get(call_id or ""), False
            )
            if step:
                traj.steps.append(step)
        # reasoning and everything else: dropped.
    return _finish(traj)


# --------------------------------------------------------------------------- discovery


def claude_code_project_key(cwd: str | Path) -> str:
    """Claude Code names a project directory by its absolute path with every
    character outside `[A-Za-z0-9-]` replaced by `-` — separators, dots and
    underscores alike: `/home/u/agentic_memory` becomes `-home-u-agentic-memory`
    (verified against this machine's `~/.claude/projects/` on 2026-09-02)."""
    return re.sub(r"[^A-Za-z0-9-]", "-", str(Path(cwd).resolve()))


def iter_claude_code_sessions(
    project_cwd: str | Path | None = None, home: str | Path | None = None
) -> Iterator[Path]:
    root = Path(home or Path.home()) / ".claude" / "projects"
    if not root.is_dir():
        return
    projects = (
        [root / claude_code_project_key(project_cwd)]
        if project_cwd
        else sorted(p for p in root.iterdir() if p.is_dir())
    )
    for project in projects:
        if project.is_dir():
            yield from sorted(project.glob("*.jsonl"))


def iter_codex_sessions(home: str | Path | None = None) -> Iterator[Path]:
    root = Path(home or Path.home()) / ".codex" / "sessions"
    if not root.is_dir():
        return
    yield from sorted(root.rglob("rollout-*.jsonl"))


def load(path: str | Path, host: str | None = None) -> SessionTrajectory:
    """Load by host, or guess it from the file name (`rollout-` is Codex)."""
    path = Path(path)
    host = host or (HOST_CODEX if path.name.startswith("rollout-") else HOST_CLAUDE_CODE)
    if host == HOST_CODEX:
        return load_codex(path)
    if host == HOST_CLAUDE_CODE:
        return load_claude_code(path)
    raise ValueError(f"unknown host {host!r}")


@dataclass(frozen=True)
class SessionAdmission:
    """Which sessions are worth taking in at all — the session-level admission
    the message-level gates (`organizers/gated.py`) have no place for.

    Deterministic and free: a session with fewer user turns or steps than the
    floor is refused before anything is persisted or distilled. The defaults
    refuse only what cannot teach a future agent anything — a session nobody
    typed into (hook noise, a summary-only file) or one with a single step.
    Callable so `add_session(admit=...)` takes any policy with the same shape:
    return the reason to refuse, or None to admit."""

    min_user_turns: int = 1
    min_steps: int = 2

    def __call__(self, traj: SessionTrajectory) -> str | None:
        if len(traj.steps) < self.min_steps:
            return f"{len(traj.steps)} step(s) < min_steps {self.min_steps}"
        if traj.user_turns < self.min_user_turns:
            return f"{traj.user_turns} user turn(s) < min_user_turns {self.min_user_turns}"
        return None


__all__ = [
    "HOST_CLAUDE_CODE",
    "HOST_CODEX",
    "SessionAdmission",
    "SessionTrajectory",
    "Step",
    "claude_code_project_key",
    "clip_head_tail",
    "iter_claude_code_sessions",
    "iter_codex_sessions",
    "load",
    "load_claude_code",
    "load_codex",
    "redact_secrets",
    "strip_injected",
]
