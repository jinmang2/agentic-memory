"""LongMemEval-V2 adapter: our memory behind the benchmark's `Memory` interface.

    python -m agmem.bench.lme_v2 estimate --upstream ~/.agmem/upstream/longmemeval-v2 \\
        --data-root DATA --domain web --tier small --write experience --read explorer
    python -m agmem.bench.lme_v2 run ... --max-usd 2 --output-dir runs/agmem_web_small

The benchmark (arXiv:2605.12493, harness at xiaowu0162/longmemeval-v2) hands a
memory system every trajectory of a haystack through `insert(trajectory)` and
then asks it, per question, `query(text, image) -> [{"type", "value"}]`; it
times `query()` alone, truncates what comes back to `--memory-context-max-tokens`
with the reader's tokenizer, and scores the reader's answer. The small tier
shares one haystack across every question of a domain, so the memory is built
once and `--save-memory` / `--load-memory-dir` round-trip it.

Two axes, four arms (docs/research/agent-memory-axes-v1.md §6, the v1 plan
Phase 2): what `insert` writes — `raw` (the trajectory as episodes, nothing
else) or `experience` (episodes plus one `experience` distillation per
trajectory) — and what `query` reads — `vector` (the fused dense+BM25 search,
rendered under a token budget) or `explorer` (the store exported as files and
a model that greps them). `raw`+`explorer` is the honest baseline the research
document names; `raw`+`vector` should land near the paper's RAG row, which is
how the harness wiring gets checked before anything is claimed.

What this module does NOT do: screenshots. A state's `screenshot` path is kept
in the step's metadata and never returned as an image item; the reader sees
text only. That is a stated limitation of every arm here, not a per-arm choice.

The upstream package is imported only by `register_with_upstream` and the
CLI, so the adapter class itself, and its tests, need no clone: `_MemoryBase`
mirrors the concrete surface of upstream `Memory` (params, thread-local query
context, `save_memory`) and `register_with_upstream` composes the two.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import threading
import types
from pathlib import Path
from typing import Any

from agmem.sessions import SessionTrajectory, Step

logger = logging.getLogger("agmem.bench.lme_v2")

MEMORY_TYPE = "agmem"
HOST = "lme-v2"
WRITES = ("raw", "experience")
READS = ("vector", "explorer")
# Per state, the AXTree text kept. A WebArena AXTree runs to tens of thousands
# of characters; the paper's RAG embeds three-state slices of it whole, our
# episodes hold one state each. The cap bounds the store and the embedder's
# work, and is a parameter because the right value is an experiment.
DEFAULT_MAX_STATE_CHARS = 12_000
# Keys of memory_params that name where THIS process keeps its files. They are
# left out of the persisted memory_config so a `--load-memory-dir` run, whose
# paths differ, still matches the saved config (upstream requires equality).
VOLATILE_PARAMS = ("data_dir", "workspace_dir")


# ----------------------------------------------------------------------------
# Trajectory -> session
# ----------------------------------------------------------------------------


def _goal_text(trajectory: dict[str, Any]) -> str:
    goal = trajectory.get("goal")
    if goal is None:
        goal = (trajectory.get("metadata") or {}).get("original_goal")
    if isinstance(goal, list):
        goal = " / ".join(str(g) for g in goal if str(g).strip())
    return str(goal or "").strip()


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n…[{len(text) - max_chars} chars of this state omitted]…"


def trajectory_to_session(
    trajectory: dict[str, Any], *, max_state_chars: int = DEFAULT_MAX_STATE_CHARS
) -> SessionTrajectory:
    """One benchmark trajectory as the session our facade ingests.

    Accepts both shapes the upstream `trajectory_store` does: the public one
    (`goal`, `start_url`, `states[{url, action, thought(s), accessibility_tree|text,
    screenshot}]`) and the internal one (`metadata.original_goal`,
    `content[{url, action, thoughts, observation{text, screenshot}}]`).

    Step layout: the goal as the opening `user` turn (so `task_text` and the
    distiller's task line are the goal), then per state an observation
    (`tool_result` from a tool named `observe`: the URL and the AXTree) followed
    by the agent's `assistant` turn (thoughts and the action taken from that
    state). Observation before action because that is the order the agent
    lived it, and the order a reader of the exported transcript expects."""
    trajectory_id = str(trajectory.get("id") or "").strip()
    if not trajectory_id:
        raise ValueError("trajectory has no id")
    states = trajectory.get("states")
    if states is None:
        states = trajectory.get("content") or []
    goal = _goal_text(trajectory)
    steps = [Step(kind="user", text=goal or f"trajectory {trajectory_id}", meta={"goal": True})]
    for index, state in enumerate(states):
        if not isinstance(state, dict):
            continue
        observation = state.get("observation") if isinstance(state.get("observation"), dict) else {}
        url = str(state.get("url") or "")
        tree = state.get("accessibility_tree") or state.get("text") or observation.get("text") or ""
        screenshot = state.get("screenshot") or observation.get("screenshot")
        step_no = state.get("step", index)
        meta = {
            "state_index": int(state.get("state_index", index)),
            "step": step_no,
            "url": url,
            "screenshot": screenshot,
        }
        body = (
            f"URL: {url}\n{_clip(str(tree), max_state_chars)}"
            if url
            else _clip(str(tree), max_state_chars)
        )
        steps.append(Step(kind="tool_result", text=body, tool_name="observe", meta=meta))
        thoughts = state.get("thoughts", state.get("thought"))
        action = state.get("action")
        parts = []
        if thoughts:
            parts.append(str(thoughts).strip())
        if action:
            parts.append(f"Action: {str(action).strip()}")
        if parts:
            steps.append(Step(kind="assistant", text="\n".join(parts), meta={"state_index": index}))
    return SessionTrajectory(
        id=trajectory_id,
        host=HOST,
        source_path=f"{HOST}:{trajectory_id}",
        cwd=str(trajectory.get("start_url") or "") or None,
        steps=steps,
        meta={
            "domain": trajectory.get("domain"),
            "goal": goal,
            "outcome": trajectory.get("outcome"),
            "n_states": len(states),
        },
    )


# ----------------------------------------------------------------------------
# The Memory implementation
# ----------------------------------------------------------------------------


class _MemoryBase:
    """The concrete surface of upstream `memory_modules.memory.Memory`, mirrored
    so the adapter runs (and is tested) without the clone. Attribute names are
    upstream's, because `register_with_upstream` puts upstream's class behind
    this one in the MRO and its methods must find the same fields."""

    memory_type: str = MEMORY_TYPE

    def __init__(self, memory_params: dict[str, object]) -> None:
        self.memory_params = dict(memory_params)
        self._query_context_local = threading.local()

    @property
    def memory_config(self) -> dict[str, Any]:
        params = {k: v for k, v in self.memory_params.items() if k not in VOLATILE_PARAMS}
        return {"memory_type": self.memory_type, "memory_params": params}

    def set_query_context(self, *, query_invocation_id: str) -> None:
        if not isinstance(query_invocation_id, str) or not query_invocation_id.strip():
            raise RuntimeError("query_invocation_id must be a non-empty string")
        self._query_context_local.context = {"query_invocation_id": query_invocation_id.strip()}

    def clear_query_context(self) -> None:
        if hasattr(self._query_context_local, "context"):
            delattr(self._query_context_local, "context")

    def get_query_context(self) -> dict[str, str]:
        context = getattr(self._query_context_local, "context", None)
        return dict(context) if isinstance(context, dict) else {}

    def save_memory(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "memory_config.json").write_text(
            json.dumps(self.memory_config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        self._save_backend(path)

    def _save_backend(self, output_dir: Path) -> None:
        return None

    def _load_backend(self, input_dir: Path) -> None:
        return None


class AgmemMemory(_MemoryBase):
    """`agmem` as a LongMemEval-V2 memory backend.

    memory_params (all optional but `write`/`read` are what an arm IS):
      write: "raw" | "experience"        what insert stores (see module docstring)
      read: "vector" | "explorer"         what query does
      config: path to an agmem TOML       LLM roles (`distill`, `explore`), embedder, profile
      namespace: str                      store namespace (default "main")
      data_dir: path                      the store; a fresh temp dir when absent
      workspace_dir: path                 the explorer's file view; default under data_dir
      top_k: int                          vector arm: items searched (default 10)
      budget_tokens: int                  vector arm: render budget (default 12000)
      max_steps: int                      explorer arm: exploration steps (default 8)
      explorer_budget_tokens: int         explorer arm: context length asked for (default 4000)
      max_state_chars: int                AXTree kept per state on insert

    Thread safety: `query` holds one lock. The harness times `query()` and may
    call it from several prompt-build workers; the wait is inside the timed
    region, so run with `--prompt-build-max-workers 1` when latency is the
    number being measured."""

    def __init__(self, memory_params: dict[str, object]) -> None:
        super().__init__(memory_params)
        p = self.memory_params
        self.write = str(p.get("write", "raw"))
        self.read = str(p.get("read", "vector"))
        if self.write not in WRITES:
            raise ValueError(f"write must be one of {WRITES}, got {self.write!r}")
        if self.read not in READS:
            raise ValueError(f"read must be one of {READS}, got {self.read!r}")
        self.config_path = Path(str(p["config"])).expanduser() if p.get("config") else None
        self.namespace = str(p.get("namespace", "main"))
        self.top_k = int(p.get("top_k", 10))
        self.budget_tokens = int(p.get("budget_tokens", 12_000))
        self.max_steps = int(p.get("max_steps", 8))
        self.explorer_budget_tokens = int(p.get("explorer_budget_tokens", 4_000))
        self.max_state_chars = int(p.get("max_state_chars", DEFAULT_MAX_STATE_CHARS))
        self.data_dir = (
            Path(str(p["data_dir"])).expanduser()
            if p.get("data_dir")
            else Path(tempfile.mkdtemp(prefix="agmem-lme-v2-"))
        )
        self.workspace_dir = (
            Path(str(p["workspace_dir"])).expanduser() if p.get("workspace_dir") else None
        )
        self._mem: Any = None
        self._lock = threading.Lock()
        self._exported = False
        self._inserted: list[str] = []
        self._query_trace_dir: Path | None = None
        self._metrics: dict[str, dict[str, Any]] = {}

    # -- store ---------------------------------------------------------------

    @property
    def mem(self) -> Any:
        """The facade, opened on first use so a memory built only to be
        loaded, or only to answer `memory_config`, costs nothing."""
        if self._mem is None:
            self._mem = self._open()
        return self._mem

    def _open(self) -> Any:
        from dataclasses import replace

        from agmem.config import AgmemConfig, load_config
        from agmem.memory import AgenticMemory

        if self.config_path is not None:
            config = replace(load_config(self.config_path), data_dir=self.data_dir, sync_write=True)
        else:
            config = AgmemConfig(profile="lite", data_dir=self.data_dir, sync_write=True)
        organizers = ["experience"] if self.write == "experience" else []
        mem = AgenticMemory(namespace=self.namespace, organizers=organizers, config=config)
        if self._query_trace_dir is not None and mem.llm is not None:
            mem.llm.trace_path = self._query_trace_dir / "agmem-llm-trace.jsonl"
        return mem

    def _explorer_root(self) -> Path:
        if self.workspace_dir is None:
            self.workspace_dir = self.data_dir / self.namespace / "workspace"
        return self.workspace_dir

    # -- upstream hooks -------------------------------------------------------

    def configure_runtime(self, **kwargs: object) -> None:
        """The harness passes `query_trace_dir`, generation settings and a
        cancel event. Only the trace dir matters here: our LLM I/O trace goes
        beside the harness's own query traces."""
        trace_dir = kwargs.get("query_trace_dir")
        if trace_dir:
            self._query_trace_dir = Path(str(trace_dir))
            self._query_trace_dir.mkdir(parents=True, exist_ok=True)
            if self._mem is not None and self._mem.llm is not None:
                self._mem.llm.trace_path = self._query_trace_dir / "agmem-llm-trace.jsonl"

    def insert(self, trajectory: dict[str, object]) -> None:
        session = trajectory_to_session(trajectory, max_state_chars=self.max_state_chars)
        with self._lock:
            ingest = self.mem.add_session(session, distill=self.write == "experience")
            self._inserted.append(session.id)
            self._exported = False
        if ingest.already_ingested and not ingest.dispatched:
            logger.info("lme_v2: trajectory %s already in the store", session.id)

    def query(self, query: str, query_image: str | None = None) -> list[dict[str, str]]:
        """Text items only (see module docstring on screenshots). The
        question's image, when there is one, is not used either: neither read
        path takes an image, and pretending otherwise would be a silent drop."""
        invocation = self.get_query_context().get("query_invocation_id", "")
        with self._lock:
            if self.read == "vector":
                metrics: dict[str, Any] = {}
                bundle = self.mem.search(query, k=self.top_k, metrics=metrics)
                text = bundle.render(self.budget_tokens)
                record = {
                    "read": "vector",
                    "items": len(bundle.items),
                    "latency_s": metrics.get("latency_s"),
                    "rendered_chars": len(text),
                }
            else:
                result = self.mem.research(
                    query,
                    root=self._explorer_root(),
                    refresh=not self._exported,
                    max_steps=self.max_steps,
                    budget_tokens=self.explorer_budget_tokens,
                )
                self._exported = True
                text = result.context or ""
                if result.citations:
                    cites = "\n".join(
                        f"- {c['file']} lines {c['lines'][0]}-{c['lines'][1]}"
                        for c in result.citations
                    )
                    text = f"{text}\n\nSources:\n{cites}"
                record = {
                    "read": "explorer",
                    "latency_s": result.latency_s,
                    "export_s": result.export_s,
                    "llm_calls": result.llm_calls,
                    "steps": len(result.steps),
                    "citations": len(result.citations),
                    "degraded": result.degraded,
                    "search_tool": result.search_tool,
                }
        if invocation:
            self._metrics[invocation] = record
        text = text.strip()
        return [{"type": "text", "value": text}] if text else []

    def post_query_hook(
        self,
        *,
        query: str,
        query_image: str | None,
        memory_context: list[dict[str, str]],
    ) -> dict[str, object] | None:
        """Our own accounting for the question just answered, which the
        harness records beside its own timing: what the read path did and
        what it cost, so a run's `per_question.jsonl` explains its latency."""
        invocation = self.get_query_context().get("query_invocation_id", "")
        return self._metrics.pop(invocation, None) if invocation else None

    # -- persistence -----------------------------------------------------------

    def _save_backend(self, output_dir: Path) -> None:
        """Copy the store into `<output_dir>/agmem`. The facade is closed for
        the copy (SQLite files must not be copied mid-write) and reopened on
        the next use, from the same place."""
        if self._mem is not None:
            self._mem.flush()
            self._mem.close()
            self._mem = None
        target = output_dir / "agmem"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(self.data_dir, target)
        (output_dir / "agmem_state.json").write_text(
            json.dumps(
                {
                    "memory_type": self.memory_type,
                    "write": self.write,
                    "read": self.read,
                    "namespace": self.namespace,
                    "inserted_trajectory_ids": list(self._inserted),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _load_backend(self, input_dir: Path) -> None:
        """Point the facade at the saved store. The explorer's file view goes
        to a fresh temp dir rather than into the artifact, so a loaded
        `memory_state/` is read and never written."""
        if self._mem is not None:
            self._mem.close()
            self._mem = None
        self.data_dir = input_dir / "agmem"
        if not self.data_dir.is_dir():
            raise RuntimeError(f"no agmem store under {input_dir}")
        state_path = input_dir / "agmem_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self._inserted = list(state.get("inserted_trajectory_ids", []))
        if self.workspace_dir is None:
            self.workspace_dir = Path(tempfile.mkdtemp(prefix="agmem-lme-v2-ws-"))
        self._exported = False

    def close(self) -> None:
        if self._mem is not None:
            self._mem.close()
            self._mem = None


# ----------------------------------------------------------------------------
# Upstream registration and the CLI
# ----------------------------------------------------------------------------


def register_with_upstream(upstream_root: str | Path) -> type:
    """Register `AgmemMemory` in the upstream registry under `memory_type`
    "agmem" and return the registered class.

    Upstream's registry is a module-level dict filled by decorators at import
    time; a type it does not know fails config validation. Rather than editing
    the clone, this composes a class that is both ours and upstream's
    `Memory` (ours first in the MRO, so our `__init__`, context handling and
    persistence are the ones that run) and registers that."""
    root = Path(upstream_root).expanduser().resolve()
    if not (root / "memory_modules" / "memory.py").is_file():
        raise FileNotFoundError(f"not a longmemeval-v2 checkout: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from memory_modules.memory import MEMORY_TYPES, Memory, register_memory

    if MEMORY_TYPE in MEMORY_TYPES:
        return MEMORY_TYPES[MEMORY_TYPE]
    composed = types.new_class(
        "AgmemMemory",
        (AgmemMemory, Memory),
        exec_body=lambda ns: ns.update({"__module__": __name__, "memory_type": MEMORY_TYPE}),
    )
    return register_memory(composed)


def build_memory_config(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {"write": args.write, "read": args.read}
    if args.config:
        params["config"] = str(Path(args.config).expanduser().resolve())
    for key in ("top_k", "budget_tokens", "max_steps", "explorer_budget_tokens", "max_state_chars"):
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value
    return {"memory_type": MEMORY_TYPE, "memory_params": params}


def _price(model: str) -> tuple[float, float]:
    from agmem.bench.registry import MODEL_REGISTRY

    spec = MODEL_REGISTRY.get(model)
    if spec is None:
        raise KeyError(f"{model!r} is not in agmem.bench.registry — add its price first")
    return spec.usd_per_1m_in, spec.usd_per_1m_out


def estimate(args: argparse.Namespace) -> dict[str, Any]:
    """What a run would cost before it is allowed to spend: token counts from
    the actual haystack (chars / 4), priced from the registry.

    Three paid parts, each named so the approval can be per item: the
    distillation on insert (experience arms only: one call per trajectory over
    a transcript clipped to the organizer's 60K characters), the explorer's
    calls on query (explorer arms only: up to max_steps + 2 calls per question,
    each carrying the prompt so far), and the reader (every arm: question plus
    the returned context, capped by the harness). The judge is a small extra
    the harness bills to OPENAI_API_KEY and is listed, not summed, because its
    price is not in our registry."""
    root = Path(args.upstream).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from data.public_data import load_haystack, load_questions, load_trajectories

    data_root = Path(args.data_root).expanduser().resolve()
    questions = load_questions(data_root, args.domain)
    if args.limit is not None:
        questions = questions[: args.limit]
    haystack = load_haystack(data_root, args.tier)
    trajectory_ids: dict[str, None] = {}
    for q in questions:
        for tid in haystack.get(q["id"], []):
            trajectory_ids[tid] = None
    trajectories = load_trajectories(data_root)
    chars = 0
    states = 0
    for tid in trajectory_ids:
        session = trajectory_to_session(trajectories[tid], max_state_chars=args.max_state_chars)
        chars += sum(len(s.text) for s in session.steps)
        states += session.meta["n_states"]
    tokens = chars / 4
    n_traj = len(trajectory_ids)
    n_q = len(questions)
    lines: dict[str, Any] = {
        "domain": args.domain,
        "tier": args.tier,
        "write": args.write,
        "read": args.read,
        "questions": n_q,
        "trajectories": n_traj,
        "states": states,
        "session_tokens_est": round(tokens),
    }
    total = 0.0
    if args.write == "experience":
        pin, pout = _price(args.distill_model)
        calls_in = n_traj * min(tokens / max(n_traj, 1), 15_000)
        cost = calls_in * pin / 1e6 + n_traj * 2_000 * pout / 1e6
        lines["distill"] = {"model": args.distill_model, "calls": n_traj, "usd": round(cost, 4)}
        total += cost
    if args.read == "explorer":
        pin, pout = _price(args.explore_model)
        calls = n_q * (args.max_steps + 2)
        cost = calls * 5_000 * pin / 1e6 + calls * 300 * pout / 1e6
        lines["explore"] = {"model": args.explore_model, "calls": calls, "usd": round(cost, 4)}
        total += cost
    pin, pout = _price(args.reader_model)
    context = args.budget_tokens if args.read == "vector" else args.explorer_budget_tokens
    context = min(context, args.memory_context_max_tokens)
    cost = n_q * (context + 500) * pin / 1e6 + n_q * 1_000 * pout / 1e6
    lines["reader"] = {"model": args.reader_model, "calls": n_q, "usd": round(cost, 4)}
    total += cost
    lines["judge"] = {
        "model": args.evaluator_model,
        "note": "billed by the harness to the evaluator key; not in our registry, not summed",
    }
    lines["total_usd_est"] = round(total, 4)
    return lines


def harness_argv(
    args: argparse.Namespace, runtime_dir: Path, memory_config_path: Path
) -> list[str]:
    """The upstream harness's argv, the way `evaluation/run_eval.py` builds it."""
    data_root = Path(args.data_root).expanduser().resolve()
    argv = [
        "evaluation.harness",
        "--domain", args.domain,
        "--questions-path", str(runtime_dir / "questions.json"),
        "--haystack-path", str(runtime_dir / "haystack.json"),
        "--trajectories-path", str(data_root / "trajectories.jsonl"),
        "--memory-config-path", str(memory_config_path),
        "--output-dir", str(Path(args.output_dir).expanduser().resolve()),
        "--model", args.reader_model,
        "--base-url", args.reader_base_url,
        "--api-key-env", args.reader_api_key_env,
        "--temperature", str(args.reader_temperature),
        "--top-p", str(args.reader_top_p),
        "--top-k", str(args.reader_top_k),
        "--max-completion-tokens", str(args.max_completion_tokens),
        "--memory-context-max-tokens", str(args.memory_context_max_tokens),
        "--reader-max-concurrent-requests", str(args.reader_max_concurrent_requests),
        "--prompt-build-max-workers", "1",
        "--evaluator-model", args.evaluator_model,
        "--evaluator-api-key-env", args.evaluator_api_key_env,
        "--evaluator-reasoning-effort", args.evaluator_reasoning_effort,
    ]  # fmt: skip
    if not args.reader_enable_thinking:
        argv.append("--reader-disable-thinking")
    if args.save_memory:
        argv.append("--save-memory")
    if args.skip_evaluation:
        argv.append("--skip-evaluation")
    if args.load_memory_dir:
        argv.extend(["--load-memory-dir", str(Path(args.load_memory_dir).expanduser().resolve())])
    return argv


def run(args: argparse.Namespace) -> int:
    """Materialize the runtime inputs the way upstream's `run_eval.py` does,
    register our memory type, and hand over to the upstream harness in
    process. Refuses to spend above `--max-usd` (the estimate is printed
    either way), and refuses an output dir that already holds a run."""
    est = estimate(args)
    print(json.dumps(est, indent=2))
    if args.max_usd is None:
        print("refusing to run without --max-usd: every paid run carries its cap", file=sys.stderr)
        return 2
    if est["total_usd_est"] > args.max_usd:
        print(
            f"refusing to run: estimate ${est['total_usd_est']} exceeds --max-usd {args.max_usd}",
            file=sys.stderr,
        )
        return 2
    output_dir = Path(args.output_dir).expanduser().resolve()
    if (output_dir / "per_question.jsonl").exists():
        print(f"refusing to overwrite a finished run in {output_dir}", file=sys.stderr)
        return 2
    register_with_upstream(args.upstream)
    from data.public_data import (
        materialize_runtime_haystack,
        materialize_runtime_questions,
        write_json,
    )

    data_root = Path(args.data_root).expanduser().resolve()
    runtime_dir = output_dir / "runtime_inputs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    selected = materialize_runtime_questions(
        data_root=data_root,
        domain=args.domain,
        question_ids=None,
        limit=args.limit,
        output_path=runtime_dir / "questions.json",
    )
    materialize_runtime_haystack(
        data_root=data_root,
        tier=args.tier,
        selected_questions=selected,
        output_path=runtime_dir / "haystack.json",
    )
    memory_config = build_memory_config(args)
    memory_config["memory_params"]["data_dir"] = str(output_dir / "agmem_store")
    memory_config_path = runtime_dir / "memory_config.json"
    write_json(memory_config_path, memory_config)
    write_json(output_dir / "agmem_estimate.json", est)
    argv = harness_argv(args, runtime_dir, memory_config_path)
    old_argv = sys.argv
    try:
        sys.argv = argv
        from evaluation.harness import main as harness_main

        harness_main()
    finally:
        sys.argv = old_argv
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--upstream", required=True, help="longmemeval-v2 checkout")
    p.add_argument("--data-root", required=True)
    p.add_argument("--domain", required=True, choices=["web", "enterprise"])
    p.add_argument("--tier", default="small", choices=["small", "medium"])
    p.add_argument("--limit", type=int, default=None, help="first N questions")
    p.add_argument("--write", default="raw", choices=WRITES)
    p.add_argument("--read", default="vector", choices=READS)
    p.add_argument("--config", default=None, help="agmem TOML with [llm.distill]/[llm.explore]")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--budget-tokens", type=int, default=12_000)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--explorer-budget-tokens", type=int, default=4_000)
    p.add_argument("--max-state-chars", type=int, default=DEFAULT_MAX_STATE_CHARS)
    p.add_argument("--distill-model", default="qwen/qwen3.5-9b")
    p.add_argument("--explore-model", default="qwen/qwen3.5-9b")
    p.add_argument("--reader-model", default="qwen/qwen3.5-9b")
    p.add_argument("--reader-base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--reader-api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--reader-temperature", type=float, default=0.6)
    p.add_argument("--reader-top-p", type=float, default=0.95)
    p.add_argument("--reader-top-k", type=int, default=20)
    p.add_argument("--reader-enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reader-max-concurrent-requests", type=int, default=4)
    p.add_argument("--max-completion-tokens", type=int, default=20_000)
    p.add_argument("--memory-context-max-tokens", type=int, default=200_000)
    p.add_argument("--evaluator-model", default="gpt-5.2")
    p.add_argument("--evaluator-api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--evaluator-reasoning-effort", default="medium")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agmem.bench.lme_v2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    est = sub.add_parser("estimate", help="price a run from the haystack, spend nothing")
    _add_common(est)
    runp = sub.add_parser("run", help="run the upstream harness with the agmem memory")
    _add_common(runp)
    runp.add_argument("--output-dir", required=True)
    runp.add_argument(
        "--max-usd", type=float, default=None, help="the cap this run was approved at"
    )
    runp.add_argument("--save-memory", action="store_true")
    runp.add_argument("--skip-evaluation", action="store_true")
    runp.add_argument("--load-memory-dir", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "estimate":
        print(json.dumps(estimate(args), indent=2))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
