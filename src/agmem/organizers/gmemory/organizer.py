"""G-Memory organizer (arXiv:2506.07398, NeurIPS'25) — compact port.

Trajectory memory + reward-shaped insight rules with periodic LLM
finetune (ADD/EDIT/REMOVE/AGREE ops on the rule list — upstream parses
these from free text with a regex; we get them as structured JSON).

U = upstream ``mas/memory/mas_memory/GMemory.py`` (commit ``7b581c5``),
UP = its ``prompt.py``. Ported faithfully (round-12 reaudit): task-only
embedding (U:96-98), the effective-cosine 0.85 edge gate (derivation at
the call site; U:390-392), global exact-text repeat skip (U:380-381), the
finetune shape (``insights_point_num`` random-anchored iterations,
compare-pair + success-chunk calls, correlation-gated rule list,
per-prompt ``relative_tasks``; U:647-748), insights served ONLY via
correlation counting — never embedded (U:490-506), failure writes as two
LLM calls (U:265-290), and feedback touching served insights only
(U:239, 292-297).

Deviations (documented per docs/research/g-memory.md):
- The query graph (upstream: networkx + pickle sidecar) is item-payload
  adjacency: ``on_task_end`` writes ``task_edges``; the read side (paper
  Eq.(5) 1-hop + Eq.(6) correlation insight recall) is
  ``TaskGraphExpansion`` in retrieval/steps.py — everything rides the op
  log, no sidecar. The step's hop is hardcoded to 1; upstream's ``hop`` is
  configurable (U:43, default 1).
- Upstream ``retrieve_memory``'s per-successful-trajectory LLM importance
  rerank (``generative_task_user_prompt``, one call per candidate) is NOT
  ported — a read-side LLM cost the embedding ranking stands in for
  (docs/16 session 5).
- FINCH cluster-merge (``merge_insights``, every 20 tasks) is deferred.
- Rule ops arrive as structured JSON addressed by id, not regex-parsed
  numbered text — upstream's index/text resolution bugs (AGREE
  fall-through to -1 crediting the LAST rule, U:858-861 via U:887-892;
  EDIT-of-existing-text conversion to AGREE, U:825-828) are intentionally
  not reproduced.
- ``finetune_seed``: upstream leans on the harness's global
  ``random.seed(42)`` (tasks/run.py); we own a seeded ``random.Random``
  per organizer — our determinism deviation, seeded by default.

Score semantics follow the official code (round-5): ADD init 2, EDIT/AGREE
+1, REMOVE soft -1 (-3 when the global rule list is at/over the cap),
prune at <=0 (clear_insights); backward reward +1/-2 applies to insights
served since the last backward (on_retrieval cache). The rule cap is
SOFT: ADD always executes; fullness only appends upstream's prompt suffix
(count > cap, U:713-715) and hardens REMOVE (count >= cap, U:843/854).

Provenance, stated precisely because the previous wording was wrong: this
is NOT a clean-room reimplementation. ``github.com/bingreeky/GMemory`` was
cloned and read as the primary reference (round-5,
docs/research/round5/gmemory-verify-report.md cites commit ``7b581c5``),
and it had to be — the paper and the code disagree about the central
mechanism (§4.3 describes a summarisation function J plus a supporting
query set, the code implements Reflexion-style critique finetune with
ADD/EDIT/REMOVE/AGREE, backward reward and FINCH), and this port follows
the CODE. The score constants two lines up are read off that code.

That repository carries NO license file (checked again 2026-07-27), so
nothing there is granted for reuse by default. What is reproduced here is
behaviour and numeric constants rather than source text, which is the
same footing as every other port in this package — but the earlier
"clean-room ... from the paper" claim asserted an independence that the
audit trail contradicts, and an inaccurate provenance note is worse than
none.
"""

from __future__ import annotations

import json
import logging
import math
import random

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import new_id
from agmem.organizers.base import Organizer, OrganizerContext

logger = logging.getLogger("agmem.organizers.gmemory")

# The shipped eval harness's read operating point (tasks/run.py:128-131), NOT
# ``retrieve_memory``'s signature defaults (2 / 1 / 10 / 0.3, U:189-196). All
# three MAS workflows additionally DISCARD the failed-trajectory list at read
# time (``successful_trajectories, _, insights = retrieve_memory(...)``,
# autogen.py:108; dylan/macnet identical), so failed trajectories feed only the
# finetune. Exported like the ZEP search recipes: a named operating point for
# runs to cite — the constant itself changes no behavior.
GMEMORY_READ_RECIPE = {
    "successful_topk": 1,
    "failed_topk": 0,
    "insights_topk": 3,
    "threshold": 0.0,
}

SPARSIFY_SCHEMA = {
    "type": "object",
    "properties": {"key_steps": {"type": "array", "items": {"type": "string"}}},
    "required": ["key_steps"],
}

MISTAKES_SCHEMA = {
    "type": "object",
    "properties": {"mistakes": {"type": "array", "items": {"type": "string"}}},
    "required": ["mistakes"],
}

FINETUNE_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "maxItems": 4,  # upstream: at most 4 ops per prompt
            "items": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["ADD", "EDIT", "REMOVE", "AGREE"],
                    },
                    "id": {"type": "string"},
                    "rule": {"type": "string"},
                },
                "required": ["op"],
            },
        }
    },
    "required": ["operations"],
}

PROJECT_SCHEMA = {
    "type": "object",
    "properties": {"insights": {"type": "array", "items": {"type": "string"}}},
    "required": ["insights"],
}

SPARSIFY_PROMPT = """Condense this multi-agent task trajectory: keep only the decisive steps.

Task: {task}
Outcome: {outcome}
Trajectory:
{trajectory}

Return JSON: {{"key_steps": ["...", ...]}}"""

# Failure-only SECOND call: upstream ``_detect_mistakes`` is its own LLM call
# after key-step extraction (U:277-290). Call-count parity is load-bearing.
MISTAKES_PROMPT = """This multi-agent task FAILED. Identify the mistakes in the trajectory
that caused the failure.

Task: {task}
Trajectory:
{trajectory}

Return JSON: {{"mistakes": ["...", ...]}}"""

# Shared op instruction (UP:312 constraints; enforced in code below where
# upstream trusts the LLM). ``{suffix}`` carries the fullness suffix — upstream
# appends it to the SYSTEM message (U:721/736); ours is one prompt, part of the
# disclosed two-role-JSON deviation.
_FINETUNE_OPS = """
Propose operations on the rules: ADD a new rule, EDIT an existing rule (give its
id), AGREE with a rule the evidence supports (give its id), or REMOVE a rule the
evidence contradicts (give its id). Do at most 4 operations, and each existing
rule can get at most 1 operation. Write each rule in the form "XXX, because XXX".{suffix}

Return JSON: {{"operations": [{{"op": "ADD", "rule": "..."}},
{{"op": "EDIT", "id": "<rule id>", "rule": "..."}},
{{"op": "AGREE", "id": "<rule id>"}}, {{"op": "REMOVE", "id": "<rule id>"}}]}}"""

FINETUNE_COMPARE_PROMPT = (
    """You maintain a list of general insight rules for solving tasks.
Compare this successful trial against the similar failed trial and the existing rules.

## Trial 1 (success):
{success}

## Trial 2 (fail):
{failure}

## EXISTING RULES:
{rules}
"""
    + _FINETUNE_OPS
)

FINETUNE_SUCCESS_PROMPT = (
    """You maintain a list of general insight rules for solving tasks.
Examine these successful trials against the existing rules.

## Trials:
{trials}

## EXISTING RULES:
{rules}
"""
    + _FINETUNE_OPS
)

# UP:300-301 verbatim; appended only when the global rule count is strictly
# over the cap (U:713-715) — the ONLY write-side manifestation of the cap
# besides REMOVE hardening. ADD itself is never suppressed (U:871-878).
FINETUNE_SUFFIX = (
    "Focus on REMOVE or EDIT or AGREE rules first, and stop ADD rule unless "
    "the new rule is VERY insightful and different from EXISTING RULES."
)

PROJECT_PROMPT = """Rewrite these general insights so they are directly actionable for the
agent role "{role}" (drop insights irrelevant to that role).

Insights:
{insights}

Return JSON: {{"insights": ["role-tailored insight", ...]}}"""


def _task_of(item: dict) -> str:
    """The full task text — the exact-match key upstream uses for graph nodes
    and ``relative_tasks`` (U:380, 725-744). ``title`` fallback only for items
    written before the ``task`` field existed."""
    return str(item.get("task") or item.get("title") or "")


def _render_traj(item: dict) -> str:
    return f"[{item.get('outcome')}] {_task_of(item)}\n{str(item.get('content', ''))[:600]}"


def _render_rules(rules: list[dict]) -> str:
    """Plain numbered rules, as upstream renders them (U:760) — no score
    display (round-12 14a). The id rides along because our ops address rules
    by id, not by list index."""
    if not rules:
        return "(none)"
    return "\n".join(f"{n}. (id={r['id']}) {r['content']}" for n, r in enumerate(rules, 1))


def _cos(a: list[float], b: list[float]) -> float:
    numerator = sum(x * y for x, y in zip(a, b))
    denominator = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return numerator / denominator if denominator else 0.0


def _random_chunks(rng: random.Random, items: list[dict], k: int = 5) -> list[list[dict]]:
    """Upstream ``random_divide_list`` (mas/utils.py:29-49): shuffle, then
    split into ceil-balanced chunks of at most ``k``."""
    if not items:
        return []
    items = list(items)
    rng.shuffle(items)
    if len(items) <= k:
        return [items]
    num_chunks = math.ceil(len(items) / k)
    size = math.ceil(len(items) / num_chunks)
    return [items[i * size : (i + 1) * size] for i in range(num_chunks)]


class GMemoryOrganizer(Organizer):
    """G-Memory trajectory + insight-rule organizer (arXiv:2506.07398; see
    module docstring for what is ported faithfully and what deviates from the
    official code). Writes only via returned MemoryOps; reads go through
    ``ctx.doc_store`` / ``ctx.vector_store``."""

    name = "gmemory"

    produces = ("strategies",)

    def __init__(
        self,
        finetune_every: int = 5,
        insight_max: int = 10,
        finetune_points: int = 5,
        finetune_seed: int | None = 0,
    ) -> None:
        """``finetune_every``: upstream ``rounds_per_insights`` (U:45).
        ``insight_max``: upstream ``MAX_RULE_THRESHOLD`` (U:713) — a SOFT cap:
        ADD is never suppressed; a global rule count strictly over the cap
        appends the fullness prompt suffix (U:714) and at/over it hardens
        REMOVE to -3 (U:843/854). ``finetune_points``: upstream
        ``insights_point_num`` (U:46) — random-anchor iterations per finetune
        event. ``finetune_seed`` seeds the organizer-owned RNG (upstream
        relies on the harness's global ``random.seed(42)``, tasks/run.py);
        None for unseeded — seeded by default is our determinism deviation."""
        self.finetune_every = finetune_every
        self.insight_max = insight_max
        self.finetune_points = finetune_points
        self._rng = random.Random(finetune_seed)
        self._task_count = 0
        # upstream insights_cache: INSIGHT ids served since the last
        # on_feedback — reward applies only to rules the agent actually saw
        # (round-5 W-4); the cache never holds trajectories (U:239)
        self._served: set[str] = set()

    def on_retrieval(
        self, hits: list[tuple[str, str, float]], ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Records served INSIGHT ids into ``_served`` for ``on_feedback``
        reward attribution — upstream ``insights_cache`` holds served rules
        exclusively (U:239), so served trajectories (and foreign
        ``strategies`` items, e.g. ReasoningBank's) are not cached. Always
        returns [] — no store writes, no LLM calls."""
        ids = [i for i, mt, _ in hits if mt == "strategies"]
        if ids:
            self._served.update(
                d["id"]
                for d in ctx.doc_store.get_items(ids, "strategies")
                if d.get("kind") == "insight"
            )
        return []

    def on_task_end(
        self, trajectory: list[dict], outcome: str, task: str, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Sparsify (key steps; a failure gets a second mistake-detection
        call, U:265-290), store the trajectory unless the exact task text is
        already stored, link the query graph, and every ``finetune_every``
        tasks additionally run ``_finetune_insights``. Mechanical fallback to
        a truncated raw trajectory when no LLM is configured or sparsify
        fails."""
        traj_text = "\n".join(json.dumps(s, ensure_ascii=False, default=str) for s in trajectory)[
            :6000
        ]
        self._task_count += 1
        success = outcome == "success"

        if ctx.llm is None:
            logger.warning("gmemory: no LLM — storing mechanical trajectory (explicit degradation)")
            key_steps, mistakes = [traj_text[:1000]], []
        else:
            result = ctx.llm.call(
                "distill",
                SPARSIFY_PROMPT.format(task=task, outcome=outcome, trajectory=traj_text),
                SPARSIFY_SCHEMA,
                required_keys=("key_steps",),
            )
            key_steps = (
                [str(s) for s in result.get("key_steps", [])] if result else [traj_text[:1000]]
            )
            mistakes = []
            if not success:
                # second call on failure — upstream ``_detect_mistakes``; it
                # runs BEFORE storage, so a repeat still pays it (U:90, 277-279)
                mresult = ctx.llm.call(
                    "distill",
                    MISTAKES_PROMPT.format(task=task, trajectory=traj_text),
                    MISTAKES_SCHEMA,
                    required_keys=("mistakes",),
                )
                mistakes = [str(m) for m in mresult.get("mistakes", [])] if mresult else []

        content = "\n".join(key_steps) + ("\nMistakes: " + "; ".join(mistakes) if mistakes else "")
        ops: list[MemoryOp] = []

        # Repeat detection is GLOBAL and EXACT on the full task text —
        # upstream ``if task_main in self.graph: return`` (U:380-381) unifies
        # repeats into the one existing node. Mapping note: upstream's caller
        # still re-adds the Chroma doc after that early return (U:96-101),
        # harmless there because graph nodes are keyed by task text; here
        # node == item, so a second ADD would be an isolated duplicate — the
        # repeat write is dropped entirely, edges included. ``_task_count``
        # still advances, matching upstream ``memory_size``, which counts the
        # duplicate doc toward the finetune cadence (U:106).
        repeat = any(
            d.get("kind") == "trajectory" and not d.get("deleted") and d.get("task") == task
            for d in ctx.doc_store.list_items("strategies", ctx.namespace)
        )

        pending: dict | None = None
        pending_vec: list[float] | None = None
        if not repeat:
            # Query-graph edges (paper Eq.(9); upstream TaskLayer.add_task_node,
            # search-before-store as U:93-101): link the new task to existing
            # trajectories among the top-10 candidates, undirected — each
            # neighbor gains the back-edge via UPDATE.
            traj_id = new_id()
            task_edges: list[str] = []
            edge_updates: list[MemoryOp] = []
            pending_vec = ctx.embedder.embed([task[:2000]])[0]
            hit_scores = dict(
                ctx.vector_store.search(
                    pending_vec, k=10, memory_type="strategies", namespace=ctx.namespace
                )
            )
            if hit_scores:
                # Insights never enter the vector store (correlation is their
                # only read channel), so with G-Memory alone the k=10 pool is
                # all trajectories, like upstream's task-doc collection
                # (round-12 #3). The kind gate remains for mixed configs:
                # ReasoningBank shares "strategies" and does embed.
                neighbors = [
                    n
                    for n in ctx.doc_store.get_items(list(hit_scores), "strategies")
                    if n.get("kind") == "trajectory" and not n.get("deleted")
                ]
                for n in neighbors:
                    # 0.85, not upstream's literal 0.7: U:390-392 thresholds
                    # ``1 - distance`` at 0.7, where distance is Chroma's
                    # DEFAULT l2 space (squared L2) and the MiniLM embedder
                    # (tasks/run.py:74) normalizes its outputs, so
                    # distance = 2 - 2*cos and the gate is
                    # 1-(2-2cos) >= 0.7  <=>  cos >= 0.85. Our stores return
                    # true cosine (stores/base.py), so the honest constant
                    # here is 0.85.
                    if hit_scores.get(n["id"], 0.0) < 0.85:
                        continue
                    task_edges.append(n["id"])
                    edge_updates.append(
                        MemoryOp(
                            op=OpType.UPDATE,
                            target_type="strategies",
                            target_id=n["id"],
                            payload={"task_edges": [*n.get("task_edges", []), traj_id]},
                        )
                    )
            pending = {
                "id": traj_id,
                "title": task[:80],  # display only; keys below use the full text
                # full task text — the exact-match key for repeat detection and
                # correlation (upstream graph nodes / relative_tasks, U:380, 725-744)
                "task": task,
                "content": content,
                "outcome": outcome,
                # stored label stand-in (upstream ``label`` bool); NEVER moved
                # by feedback — backward touches insights only (U:292-297)
                "score": 1.0 if success else -2.0,
                "kind": "trajectory",
                "task_edges": task_edges,
                # upstream embeds page_content = task_main ONLY (U:96-98): the
                # edge gate and all task retrieval are task-vs-task
                "embedding_text": task[:2000],
            }
            ops.append(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="strategies",
                    target_id=traj_id,
                    payload=pending,
                )
            )
            ops.extend(edge_updates)

        if ctx.llm is not None and self._task_count % self.finetune_every == 0:
            ops.extend(self._finetune_insights(pending, pending_vec, ctx))
        return ops

    def _finetune_insights(
        self, pending: dict | None, pending_vec: list[float] | None, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Upstream ``finetune_insights`` (U:647-748): ``finetune_points``
        iterations, each anchored on a RANDOM stored task; fetch the 3 most
        similar successes and 1 failure (task-vs-task similarity), show only
        rules whose positive-correlation overlap with the fetched tasks is
        >= len(tasks)/2 (U:672), and issue one compare-pair call per
        (success, failure) pair plus one call per success chunk of <= 5
        (``random_divide_list``, U:704-748). ``relative_tasks`` recorded on
        touched rules are exactly the full task strings in that prompt
        (U:725-744).

        Upstream mutates ``insights_memory`` in place, and the just-finished
        task is already in Chroma when the finetune runs (U:101 -> 106); here
        the pending trajectory ADD has not been applied yet, so it joins the
        pool explicitly, and the rule list is a working copy diffed into ops
        at the end — the op log is coarser than upstream's per-iteration JSON
        dumps, the end state identical."""
        stored = [
            d for d in ctx.doc_store.list_items("strategies", ctx.namespace) if not d.get("deleted")
        ]
        trajs = [d for d in stored if d.get("kind") == "trajectory"]
        vectors = ctx.vector_store.get([t["id"] for t in trajs])
        if pending is not None and pending_vec is not None:
            trajs = [*trajs, pending]
            vectors[pending["id"]] = pending_vec
        if not trajs:
            return []

        originals = {d["id"]: d for d in stored if d.get("kind") == "insight"}
        working: list[dict] = [
            {
                "id": d["id"],
                "content": str(d.get("content", "")),
                "score": float(d.get("score", 0)),
                "pos": set(d.get("positive_correlation_tasks", [])),
                "neg": set(d.get("negative_correlation_tasks", [])),
            }
            for d in originals.values()
        ]

        for _ in range(self.finetune_points):
            anchor = self._rng.choice(trajs)
            anchor_vec = vectors.get(anchor["id"]) or []
            sims = {t["id"]: _cos(vectors.get(t["id"]) or [], anchor_vec) for t in trajs}
            ranked = sorted(trajs, key=lambda t: sims[t["id"]], reverse=True)
            successes = [t for t in ranked if t.get("outcome") == "success"][:3]
            failures = [t for t in ranked if t.get("outcome") != "success"][:1]
            # the anchor joins its side even when the similarity fetch already
            # returned it — upstream appends without dedup (U:666-669)
            (successes if anchor.get("outcome") == "success" else failures).append(anchor)

            task_mains = [_task_of(t) for t in successes + failures]
            # rules shown: correlation overlap >= half the fetched tasks
            # (U:672 via _find_related_insights), count-sorted desc (U:641)
            overlap = {w["id"]: sum(t in w["pos"] for t in task_mains) for w in working}
            shown = sorted(
                (w for w in working if overlap[w["id"]] >= len(task_mains) / 2),
                key=lambda w: overlap[w["id"]],
                reverse=True,
            )
            # fullness suffix at strictly > cap (U:713-715); the REMOVE
            # hardening in _apply_rule_ops is >= cap (U:843) — two different
            # comparisons upstream, both reproduced
            suffix = ("\n" + FINETUNE_SUFFIX) if len(working) > self.insight_max else ""

            # one compare call per (success, failure) pair, index-wise,
            # stopping at the shorter list (U:705-709, 719-729)
            for success_task, failed_task in zip(successes, failures):
                self._finetune_call(
                    FINETUNE_COMPARE_PROMPT.format(
                        success=_render_traj(success_task),
                        failure=_render_traj(failed_task),
                        rules=_render_rules(shown),
                        suffix=suffix,
                    ),
                    working,
                    shown,
                    [_task_of(success_task), _task_of(failed_task)],
                    ctx,
                )
            # one call per success chunk (U:711, 734-745)
            for chunk in _random_chunks(self._rng, successes):
                self._finetune_call(
                    FINETUNE_SUCCESS_PROMPT.format(
                        trials="\n\n".join(_render_traj(t) for t in chunk),
                        rules=_render_rules(shown),
                        suffix=suffix,
                    ),
                    working,
                    shown,
                    [_task_of(t) for t in chunk],
                    ctx,
                )
            # per-iteration prune — upstream clear_insights at U:750
            working[:] = [w for w in working if w["score"] > 0]

        return self._diff_ops(originals, working)

    def _finetune_call(
        self,
        prompt: str,
        working: list[dict],
        shown: list[dict],
        relative_tasks: list[str],
        ctx: OrganizerContext,
    ) -> None:
        """One rule-ops LLM call applied with upstream ``_update_rules``
        semantics (U:808-878): ADD unconditional (init 2); EDIT/AGREE +1;
        REMOVE -1, or -3 when the GLOBAL rule count is at/over the cap.
        Only rules SHOWN in this prompt are addressable (upstream maps prompt
        indices back through ``insight_ids``, U:684-700); each existing rule
        takes at most one operation, enforced in code where upstream trusts
        the prompt text (UP:312)."""
        result = ctx.llm.call("distill", prompt, FINETUNE_SCHEMA, required_keys=("operations",))
        if result is None:
            return
        by_id = {w["id"]: w for w in working}
        addressable = {w["id"] for w in shown}
        list_full = len(working) >= self.insight_max
        touched: set[str] = set()
        for raw in result.get("operations", [])[:4]:
            op, rule_id, rule = (
                raw.get("op"),
                raw.get("id"),
                str(raw.get("rule", "")).strip(),
            )
            if op == "ADD" and rule:
                # unconditional — the cap is soft, prompt-side only
                # (round-12 #4; upstream executes ADD even when full, U:871-878)
                working.append(
                    {
                        "id": new_id(),
                        "content": rule,
                        "score": 2.0,
                        "pos": set(relative_tasks),
                        "neg": set(),
                        "new": True,
                    }
                )
                continue
            if rule_id not in addressable or rule_id not in by_id or rule_id in touched:
                continue  # hallucinated, unshown, or double-touched ids emit nothing
            touched.add(rule_id)
            entry = by_id[rule_id]
            if op == "EDIT" and rule:
                entry["content"] = rule
                entry["score"] += 1.0
                entry["pos"] |= set(relative_tasks)
            elif op == "AGREE":
                entry["score"] += 1.0
                entry["pos"] |= set(relative_tasks)
            elif op == "REMOVE":
                entry["score"] -= 3.0 if list_full else 1.0
                entry["neg"] |= set(relative_tasks)

    def _diff_ops(self, originals: dict[str, dict], working: list[dict]) -> list[MemoryOp]:
        """Working rule list -> MemoryOps. A rule created and pruned within
        the same finetune event emits nothing (it never reached the store —
        upstream's transient rules likewise die between JSON dumps)."""
        ops: list[MemoryOp] = []
        surviving = {w["id"] for w in working}
        for w in working:
            pos, neg = sorted(w["pos"]), sorted(w["neg"])
            if w.get("new"):
                ops.append(
                    MemoryOp(
                        op=OpType.ADD,
                        target_type="strategies",
                        target_id=w["id"],
                        payload={
                            "id": w["id"],
                            "title": w["content"][:60],
                            "content": w["content"],
                            "kind": "insight",
                            "score": w["score"],
                            "positive_correlation_tasks": pos,
                            "negative_correlation_tasks": neg,
                            # Explicitly NO vector: upstream never embeds rules
                            # — insights are a JSON list reached only by
                            # correlation counting (U:490-506, 628-646). The
                            # facade treats a present-but-None embedding_text
                            # as "doc store only" (memory.py::_apply_one).
                            "embedding_text": None,
                        },
                    )
                )
                continue
            orig = originals[w["id"]]
            if (
                w["content"] != str(orig.get("content", ""))
                or w["score"] != float(orig.get("score", 0))
                or pos != sorted(orig.get("positive_correlation_tasks", []))
                or neg != sorted(orig.get("negative_correlation_tasks", []))
            ):
                ops.append(
                    MemoryOp(
                        op=OpType.UPDATE,
                        target_type="strategies",
                        target_id=w["id"],
                        payload={
                            "content": w["content"],
                            "score": w["score"],
                            "positive_correlation_tasks": pos,
                            "negative_correlation_tasks": neg,
                        },
                    )
                )
        for insight_id in originals:
            if insight_id not in surviving:
                ops.append(
                    MemoryOp(
                        op=OpType.DELETE,
                        target_type="strategies",
                        target_id=insight_id,
                        payload={"reason": "score_pruned"},
                    )
                )
        return ops

    def project_insights(self, role: str, insights: list[str], ctx: OrganizerContext) -> list[str]:
        """Role-specific insight rewriting (multi-agent injection path)."""
        if ctx.llm is None or not insights:
            return insights
        result = ctx.llm.call(
            "distill",
            PROJECT_PROMPT.format(role=role, insights="\n".join(f"- {i}" for i in insights)),
            PROJECT_SCHEMA,
            required_keys=("insights",),
        )
        return [str(i) for i in result["insights"]] if result else insights

    def on_feedback(
        self, memory_ids: list[str], helpful: bool, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Reward-shapes INSIGHTS served since the last feedback (+1 helpful /
        -2 not — upstream ``backward``, U:292-295), then prunes at score <= 0
        (clear_insights, U:584-586). Upstream's cache holds served rules
        exclusively (U:239), so trajectories never move: their stored score is
        a write-time label, not a feedback channel. Nothing served means
        nothing updates — an empty cache is an empty update set, not a bypass
        (the previous ``if self._served and ...`` guard let an empty cache
        reward EVERY fed-back strategies item, foreign ReasoningBank items
        included; round-12 #6/#18). The cache clears after each feedback
        (U:297)."""
        reward = 1.0 if helpful else -2.0
        served = [i for i in memory_ids if i in self._served]
        ops: list[MemoryOp] = []
        for item in ctx.doc_store.get_items(served, "strategies"):
            if item.get("deleted") or item.get("kind") != "insight":
                continue
            new_score = float(item.get("score", 0)) + reward
            ops.append(
                MemoryOp(
                    op=OpType.UPDATE,
                    target_type="strategies",
                    target_id=item["id"],
                    payload={"score": new_score},
                )
            )
            if new_score <= 0:
                ops.append(
                    MemoryOp(
                        op=OpType.DELETE,
                        target_type="strategies",
                        target_id=item["id"],
                        payload={"reason": "score_pruned"},
                    )
                )
        self._served.clear()
        return ops
