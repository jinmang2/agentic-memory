"""ReasoningBank organizer (arXiv:2509.25140, ICLR'26).

on_task_end: (optional self-judge) -> distill up to 3 strategy items from
the trajectory — success AND failure both teach — then append-only ADD.
No pruning/merging by design (the paper's deliberate simplification).
The judge's reason is appended to the trajectory before extraction, as
upstream appends autoeval thoughts ("The task succeeded/failed because:").
Each task also stores an EXPERIENCE record (task query + its item ids):
retrieving memory_type "experiences" with k=1 reproduces upstream's
top-1-experience injection (the pipeline expands it to the member items);
retrieving "strategies" directly is the item-level convenience mode with
no upstream counterpart — upstream embeds only task queries, never items
(round-12 #16). ``RB_READ_RECIPE`` pins the upstream operating point.

on_scaled_task_end: MaTTS parallel induction (paper §3.3) — several
trajectories of ONE task distilled by self-contrast into up to 5 items
(``PARALLEL_SI``), a separate module upstream and a separate hook here.

This hook is PAPER-faithful, not RELEASE-faithful, and the distinction is
forced rather than chosen: the released ``induce_scaling.py`` does not
run. It calls ``one_step_chat`` on the CLASS from ``CLIENT_DICT`` without
instantiating it (the sibling ``induce_memory.py`` does instantiate), so
the unbound call takes the trajectory text as ``self`` and raises
TypeError before any model is reached; behind that crash sit three more
faults — the N "trajectories" it assembles are N reads of ONE directory,
the return tuple is stored unpacked, and the bank entry skips the
``split("\\n\\n")`` its own consumer relies on. So the elsewhere-standing
rule "reproduce it as shipped" (docs/17 §297, dead knobs stay dead) has
nothing to reproduce here — a crash is not an arm — and the only
available referent is the paper plus the ``PARALLEL_SI`` text itself.
Proof: ``scripts/repro/defects/repro_reasoningbank_matts_inert.py``;
catalog §8 RB-9.

MaTTS's OTHER half, sequential scaling, is absent for a reason of its
own, and it is also the half upstream never wired: ``SEQUENTIAL_PROMPT``
and ``SEQUENTIAL_FOLLOWING_PROMPT`` are defined in
``memory_instruction.py`` and referenced NOWHERE else in the snapshot
(same repro script). Independently of that, the prompt tells the AGENT to
re-examine and rewrite its own trajectory ("Output must stay in the same
<think>...</think><action> format"), so it is the generator's loop, not
the memory layer's — the same line ACE's multi-round reflection falls on
(docs/10, round-8 §A3). The memory layer then sees the refined trajectory
through the ordinary ``on_task_end``, and a harness doing sequential
scaling should hand it the ACCUMULATED rounds rather than only the final
one: the paper counts the intermediate refinement notes as memory signal
in their own right (§3.3; docs/research/nemori-reasoningbank.md §2A), so
passing only the last rewrite discards what that half contributes.

LLM roles used: ``judge`` (binary success call, t=0.0 in the paper),
``distill`` (extraction, t=1.0 in the paper AND in upstream
``induce_memory.py``; upstream's scaled induction uses t=0.7 instead —
``induce_scaling.py`` — so a MaTTS pass wanting the exact operating point
sets the role temperature for that pass). The SI rules ride in the system
message, as upstream's one_step_chat does; ``persona`` prepends a domain
persona ("You are an expert in web navigation.") when the benchmark has
one — upstream hardcodes that sentence into every SI.
"""

from __future__ import annotations

import json
import logging

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import StrategyItem, new_id
from agmem.organizers.base import Organizer, OrganizerContext

logger = logging.getLogger("agmem.organizers.reasoning_bank")

# Upstream's read operating point, pinned the way GMEMORY_READ_RECIPE and the
# Zep search recipes are: a named constant for runs to cite — it changes no
# behavior and NO preset auto-applies it. Both shipped harnesses retrieve the
# top-1 EXPERIENCE by task-query embedding (`select_memory(n=1, ...,
# cur_query=<task>)`, WebArena/run.py:177-193; minisweagent
# run/extra/swebench.py:182) and inject every member of its "memory_items"
# list joined by blank lines; a retrieval miss injects nothing. There is NO
# direct item-level channel upstream — hence strategies_topk 0. Reproduce
# with `search(task, memory_types=("experiences",), k=1)`: the pipeline's
# ExpandExperiences step replaces the experience hit with its member items,
# including the miss -> no-injection case.
RB_READ_RECIPE = {
    "experiences_topk": 1,
    "strategies_topk": 0,
}

# round-12 #14: outcome-string normalization. The synonym sets are ACE's
# ENVIRONMENT_FEEDBACK keys verbatim, so the two organizers cannot disagree
# about what a caller's label means.
#
# Two questions are asked of an outcome and they are NOT the same question:
# `_is_success` picks the SI, `_is_labeled` decides whether the judge is needed
# at all. Keeping the second as exact membership in ("success", "failure") was
# the other half of #14 — a caller passing "correct" had its label silently
# DISCARDED and paid for a judge call to re-derive it, which is the same
# inconsistency #14 fixed on the `self_judge=False` side. An unrecognized label
# (typo included) is not silently failure: it goes to the judge when one is
# available, and only falls back to the failure SI when it is not.
_SUCCESS_OUTCOMES = ("success", "correct")
_FAILURE_OUTCOMES = ("failure", "incorrect", "wrong")


def _norm(outcome: str) -> str:
    return str(outcome).strip().lower()


def _is_success(outcome: str) -> bool:
    return _norm(outcome) in _SUCCESS_OUTCOMES


def _is_labeled(outcome: str) -> bool:
    """Whether the caller's string names an outcome we recognize — the judge is
    for the ones it does not."""
    return _norm(outcome) in _SUCCESS_OUTCOMES + _FAILURE_OUTCOMES


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"success": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["success"],
}


def _extract_schema(max_items: int) -> dict:
    """Extraction schema whose ``maxItems`` matches the "at most N" the prompt
    advertises — the two derive from one knob so they cannot disagree
    (round-12 #15: maxItems used to be hardcoded at 3, making ``max_items>3``
    inert and ``<3`` a silent post-hoc truncation against a prompt still
    saying 3)."""
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["title", "description", "content"],
                },
            }
        },
        "required": ["items"],
    }


DEFAULT_MAX_ITEMS = 3  # upstream SUCCESSFUL_SI/FAILED_SI: "at most 3"

# MaTTS induction has its own budget: "You can extract *at most 5* memory items
# from all trajectories combined" (upstream PARALLEL_SI), against 3 for one.
# This one is an upstream constant, not the ``max_items`` knob.
SCALED_MAX_ITEMS = 5
SCALED_EXTRACT_SCHEMA = _extract_schema(SCALED_MAX_ITEMS)

JUDGE_PROMPT = """Judge whether the agent successfully completed the task.

Task: {task}

Trajectory:
{trajectory}

Return JSON: {{"success": true/false, "reason": "one sentence"}}"""

# Condensed from the paper's SUCCESSFUL_SI / FAILED_SI instructions.
# Injected as the SYSTEM message (upstream one_step_chat(system_msg=SI)).
# The "at most {max_items}" budget is rendered from the same knob that sets
# the schema's maxItems (round-12 #15); default 3 = upstream's budget.
EXTRACT_SUCCESS_SI = """The agent SUCCEEDED at this task. First think about WHY the approach \
worked, then distill at most {max_items} transferable strategy items.

Rules:
- Do NOT embed literal product names, queries, or task-specific strings.
- Prefer concrete executable procedures over abstract principles.
- description is ONE sentence stating WHEN to apply (and when not to).
- content is 1-3 sentences describing the insight. No duplicates.

Respond with a single JSON object:
{{"items": [{{"title": "...", "description": "...", "content": "..."}}]}}"""

EXTRACT_FAILURE_SI = """The agent FAILED at this task. Reflect on WHY it failed, then distill \
at most {max_items} preventative lessons or recovery procedures as strategy items.

Rules:
- Do NOT embed literal product names, queries, or task-specific strings.
- Prefer concrete executable procedures over abstract principles.
- description is ONE sentence stating WHEN to apply (and when not to).
- content is 1-3 sentences describing the insight. No duplicates.

Respond with a single JSON object:
{{"items": [{{"title": "...", "description": "...", "content": "..."}}]}}"""

# User message: query + trajectory, as upstream induce_memory formats it.
EXTRACT_USER_TEMPLATE = """**Query:** {task}

**Trajectory:**
{trajectory}"""

# MaTTS parallel induction, condensed from upstream PARALLEL_SI
# (WebArena/prompts/memory_instruction.py:63, used by induce_scaling.py). The
# domain persona is stripped for the same reason it is stripped from
# EXTRACT_*_SI — upstream hardcodes "You are an expert in web navigation." and we
# make it the `persona` argument so the organizer is not WebArena-shaped.
#
# Three numbers differ from the single-trajectory path and all three are
# upstream's: at most FIVE items (not 3), content 1-5 sentences (not 1-3), and
# t=0.7 (induce_scaling.py:196) against t=1.0 for one trajectory
# (induce_memory.py:164-166). The temperature is a role setting here, not an
# organizer constant, so it is stated rather than applied — a bench config that
# wants upstream's operating point sets `distill` accordingly for the scaled pass.
SCALED_EXTRACT_SI = """You will be given a user query and MULTIPLE trajectories showing how an \
agent attempted the same task. Some may have succeeded and others may have failed.

Compare and contrast the trajectories to identify the most useful and generalizable
strategies, using self-contrast reasoning:
- Identify patterns and strategies that consistently led to success.
- Identify mistakes or inefficiencies from failed trajectories and formulate
  preventative strategies.
- Prefer strategies that generalize beyond specific pages or exact wording.

Rules:
- Think first: why did some trajectories succeed while others failed?
- At most 5 memory items from all trajectories combined. No duplicates or overlaps.
- Do NOT embed literal product names, queries, or task-specific strings.
- description is ONE sentence stating WHEN to apply (and when not to).
- content is 1-5 sentences describing the insight.

Respond with a single JSON object:
{"items": [{"title": "...", "description": "...", "content": "..."}]}"""

# Upstream's own layout (induce_scaling.main): one query line, then each
# trajectory under a numbered header. The space before the colon is upstream's.
SCALED_USER_HEADER = "**Query:** {task}\n\n"
SCALED_USER_TRAJECTORY = "**Trajectory {n} :**\n{trajectory}\n\n"


def _format_trajectory(trajectory: list[dict], max_chars: int = 60000) -> str:
    # upstream feeds the full trajectory untruncated; the (raised) cap is a
    # context-overflow guard only — round-5 (f)
    text = "\n".join(json.dumps(step, ensure_ascii=False, default=str) for step in trajectory)
    if len(text) > max_chars:
        # keep head and tail — failures usually surface at the end
        half = max_chars // 2
        text = text[:half] + "\n...[truncated]...\n" + text[-half:]
    return text


class ReasoningBankOrganizer(Organizer):
    """ReasoningBank (see module docstring for the paper/upstream mapping)."""

    name = "reasoning_bank"

    produces = ("experiences", "strategies")

    def __init__(
        self,
        max_items: int = DEFAULT_MAX_ITEMS,
        self_judge: bool = True,
        persona: str | None = None,
    ) -> None:
        """`self_judge=True` runs the judge role when `outcome` names no outcome
        we recognize (`_is_labeled`); set False to always trust the caller-supplied
        `outcome` and skip that LLM call. Both paths normalize the same way, so
        "correct" counts as success and is trusted rather than re-judged
        (round-12 #14 and its follow-up). `persona` is prepended to the extraction
        system message verbatim when set (see module docstring).

        `max_items` drives the single-trajectory extraction budget end to end:
        the schema's ``maxItems`` AND the SI's "at most N" sentence (round-12
        #15). Default 3 is upstream's budget; the MaTTS scaled path keeps its
        own upstream constant of 5."""
        self.max_items = max_items
        self.self_judge = self_judge
        self.persona = persona  # e.g. "You are an expert in web navigation."
        self._schema = _extract_schema(max_items)
        self._success_si = EXTRACT_SUCCESS_SI.format(max_items=max_items)
        self._failure_si = EXTRACT_FAILURE_SI.format(max_items=max_items)

    def on_task_end(
        self, trajectory: list[dict], outcome: str, task: str, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Returns `[]` without calling the LLM if `ctx.llm` is unset, if the judge
        or extraction call drops (see `StructuredCaller`), or if extraction returns
        no items — never raises on LLM failure. Items missing a required field are
        skipped individually rather than failing the whole batch."""
        if ctx.llm is None:
            logger.warning(
                "reasoning_bank: no LLM configured — skipping distillation "
                "(explicit skip, task=%.60s)",
                task,
            )
            return []

        traj_text = _format_trajectory(trajectory)
        reason = ""

        if not _is_labeled(outcome) and self.self_judge:
            verdict = ctx.llm.call(
                "judge",
                JUDGE_PROMPT.format(task=task, trajectory=traj_text),
                JUDGE_SCHEMA,
                required_keys=("success",),
            )
            if verdict is None:
                return []  # drop already counted by StructuredCaller
            outcome = "success" if verdict["success"] else "failure"
            reason = str(verdict.get("reason", "")).strip()

        if reason:
            # upstream appends autoeval thoughts to the trajectory so the
            # extractor reflects on the actual cause (round-5 (c))
            status = "succeeded" if _is_success(outcome) else "failed"
            traj_text += f"\n\nThe task {status} because: {reason}"

        # SI selection rides the normalization (round-12 #14): with
        # self_judge=False, a caller-supplied "correct"/"Success" must not
        # silently land on the failure instructions.
        si = self._success_si if _is_success(outcome) else self._failure_si
        if self.persona:
            si = f"{self.persona}\n\n{si}"
        result = ctx.llm.call(
            "distill",
            EXTRACT_USER_TEMPLATE.format(task=task, trajectory=traj_text),
            self._schema,
            required_keys=("items",),
            system=si,
        )
        if result is None or not isinstance(result.get("items"), list):
            return []
        return self._emit(result["items"], outcome, task, self.max_items, ctx, [trajectory])

    def on_scaled_task_end(
        self, trajectories: list[list[dict]], task: str, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """MaTTS parallel induction (paper §3.3): distil from the CONTRAST across
        several trajectories of the same task.

        Not a variant of ``on_task_end`` with a bigger input — a different prompt
        (``SCALED_EXTRACT_SI``), a different budget (5 items, not 3) and a
        different temperature upstream. One trajectory falls through to the
        single-trajectory path, since contrasting a set of one is not the
        mechanism; an empty set is a no-op.

        **Written against the paper, because the released induction does not
        run** (module docstring; catalog §8 RB-9; repro
        ``repro_reasoningbank_matts_inert.py``). Two consequences are specific to
        this method:

        - The N trajectories here are genuinely DIFFERENT attempts. Upstream's
          are not: ``induce_scaling.main``'s ``res_dir``/``cur_task`` are
          loop-invariant and its caller passes one trial directory
          (``pipeline_scaling.py:73``, the spawn loop's leftover ``i``), so
          ``**Trajectory 1..N :**`` are byte-identical copies and the
          self-contrast the prompt asks for has nothing to contrast. Feeding
          distinct attempts is therefore a DEVIATION from the released code and
          a requirement of the mechanism — both, and it is recorded as such
          rather than presented as reproduction.
        - No self-judge and no outcome label, deliberately, and this one IS
          upstream's shape: it derives a per-trajectory correctness label from
          the harness reward and never puts it in the prompt — ``main`` computes
          ``status``, passes it to ``get_info``, and nothing reads the field;
          ``format_examples``, which would render the "## Correctness Signal"
          block, is never called. The label is additionally INVERTED
          (``induce_scaling.py:181-184`` sets ``status="success"`` when
          ``reward == 0``, the opposite of ``induce_memory.py``'s mapping) —
          extra evidence it is dead, since it would be wrong if live (round-12
          #12, ledger B-7). The signal the mechanism actually uses is the
          mixture itself ("Some may have succeeded and others may have failed"),
          so judging each trajectory here would invent a channel neither the
          paper nor the code has. Same family as the other dead upstream terms
          this project reproduces rather than repairs (MemoryOS's R_recency,
          A-MAC's N/R).

        The emitted items carry ``outcome="contrast"`` rather than
        success/failure: they are distilled from a mixed set, so neither label is
        true of them, and upstream's parallel bank has no per-item label at all.
        """
        trajectories = [t for t in trajectories if t]
        if not trajectories:
            return []
        if len(trajectories) == 1:
            # Not a contrast set — fall through to the ordinary single-trajectory
            # path. The empty outcome is deliberate and routed, not accidental
            # (round-12 #14): "" is not a label `_is_labeled` recognizes, so with
            # self_judge=True (default) the judge decides, and with
            # self_judge=False `_is_success("")` is False and the failure SI is
            # an explicit fallback rather than an artifact of string equality.
            return self.on_task_end(trajectories[0], "", task, ctx)
        if ctx.llm is None:
            logger.warning(
                "reasoning_bank: no LLM configured — skipping scaled distillation "
                "(explicit skip, task=%.60s)",
                task,
            )
            return []

        user = SCALED_USER_HEADER.format(task=task) + "".join(
            SCALED_USER_TRAJECTORY.format(n=i + 1, trajectory=_format_trajectory(trajectory))
            for i, trajectory in enumerate(trajectories)
        )
        si = SCALED_EXTRACT_SI
        if self.persona:
            si = f"{self.persona}\n\n{si}"
        result = ctx.llm.call(
            "distill",
            user,
            SCALED_EXTRACT_SCHEMA,
            required_keys=("items",),
            system=si,
        )
        if result is None or not isinstance(result.get("items"), list):
            return []
        return self._emit(result["items"], "contrast", task, SCALED_MAX_ITEMS, ctx, trajectories)

    def _emit(
        self,
        items: list,
        outcome: str,
        task: str,
        cap: int,
        ctx: OrganizerContext,
        trajectories: list[list[dict]],
    ) -> list[MemoryOp]:
        """Turn extracted items into ADD ops plus the experience record that ties
        them to the task query. Shared by both induction paths so they cannot
        drift apart in how a memory is stored — only in how it is produced."""
        ops: list[MemoryOp] = []
        item_ids: list[str] = []
        for raw in items[:cap]:
            if not all(
                isinstance(raw.get(f), str) and raw.get(f)
                for f in ("title", "description", "content")
            ):
                continue  # field-level fallback: keep valid items, skip broken ones
            item = StrategyItem(
                title=raw["title"],
                description=raw["description"],
                content=raw["content"],
                outcome=outcome,
                namespace=ctx.namespace,
            )
            ops.append(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="strategies",
                    target_id=item.id,
                    payload={
                        "id": item.id,
                        "title": item.title,
                        "description": item.description,
                        "content": item.content,
                        "outcome": outcome,
                        # Item-level embedding is our convenience mode with no
                        # upstream counterpart: upstream embeds only task
                        # queries (the experience record below), never items
                        # (round-12 #16).
                        "embedding_text": item.embedding_text(),
                    },
                )
            )
            item_ids.append(item.id)

        if item_ids:
            # experience record: the retrieval unit upstream actually uses —
            # task-query embedding, expanded to its member items at read time.
            #
            # ``trajectories`` keeps the attempts the items were distilled from,
            # as upstream's bank entry keeps ``think_list``/``action_list``
            # beside its ``memory_items`` (induce_memory.py:177-186) and the
            # paper's §A.2 schema says an entry is "task query, original
            # trajectory, memory items". It is never embedded and never served
            # (``ExpandExperiences`` replaces an experience with its items, so
            # nothing here reaches a prompt) — it exists so that the two things
            # upstream's file format supports stay possible for us: re-running
            # extraction under a different SI without re-running the agent, and
            # a Synapse-style arm, which reuses the raw trajectory and is the
            # baseline the paper compares against on the SAME store.
            #
            # A list of attempts, always — the single-trajectory path stores a
            # list of one — so a consumer never has to branch on which induction
            # path wrote the record.
            experience_id = new_id()
            ops.append(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="experiences",
                    target_id=experience_id,
                    payload={
                        "id": experience_id,
                        "task": task,
                        "outcome": outcome,
                        "item_ids": item_ids,
                        "trajectories": trajectories,
                        "embedding_text": task,
                    },
                )
            )
        return ops
