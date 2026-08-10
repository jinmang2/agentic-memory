"""ACE organizer — Agentic Context Engineering (arXiv:2510.04618, ICLR'26).

Playbook of itemized bullets with helpful/harmful counters, evolved by
delta operations instead of monolithic rewrites (avoids brevity bias /
context collapse). Roles: Reflector critiques the trajectory, Curator
emits ADD deltas; merge is deterministic (non-LLM).

Deviations from the reference repo, on purpose:
- The paper's MERGE/DELETE are unimplemented upstream (ADD-only curator);
  we keep ADD-only too but make embedding dedup (threshold 0.90) ALWAYS ON
  — upstream ships it opt-in and silently skips without deps (docs/research
  /ace-longmemeval.md §D), which is the reproduction trap we avoid.
- Counter updates go through the evolution log (UPDATE ops), so
  helpful/harmful history is auditable.
Read contract (round-5, structural since round-12 #5): ACE injects the
FULL playbook — use ``AgenticMemory.get_playbook()``, never top-k
retrieval of bullets. The facade enforces this structurally:
``default_memory_types`` excludes ``playbook``, so a plain ``search()``
cannot serve bullets; only an explicit ``memory_types=("playbook",)``
opts a caller into the partial view. The curator likewise sees the whole
playbook, and dedup also compares within the current batch. Reflector
tagging remains trajectory-evidence-based
(official attributes counters to bullets the Generator actually cited —
we lack that signal in a post-hoc organizer; report_feedback() is the
usage-accurate path).
"""

from __future__ import annotations

import logging

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Bullet
from agmem.organizers.base import Organizer, OrganizerContext

logger = logging.getLogger("agmem.organizers.ace")

# Upstream REFLECTOR_PROMPT's five output fields (prompts/reflector.py:18-24):
# reasoning / error_identification / root_cause_analysis / correct_approach /
# key_insight, plus bullet_tags. The curator is fed the FULL raw reflection —
# all five fields — as upstream passes the reflector's whole response string
# through as `recent_reflection` (ace.py:586). An earlier version here invented
# a `lessons` array instead; round-12 #2(c) removed it.
REFLECT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "error_identification": {"type": "string"},
        "root_cause_analysis": {"type": "string"},
        "correct_approach": {"type": "string"},
        "key_insight": {"type": "string"},
        "bullet_tags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "tag": {
                        "type": "string",
                        "enum": ["helpful", "harmful", "neutral"],
                    },
                },
                "required": ["id", "tag"],
            },
        },
    },
    "required": [
        "reasoning",
        "error_identification",
        "root_cause_analysis",
        "correct_approach",
        "key_insight",
    ],
}

# No maxItems: upstream's curator accepts an unbounded operations list (an
# earlier cap of 5 here was our invention, round-12 #2(d)).
CURATE_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["ADD"]},
                    "section": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["type", "section", "content"],
            },
        }
    },
    "required": ["operations"],
}

# The opening line is upstream's, not a paraphrase, and the difference is
# load-bearing. REFLECTOR_PROMPT (prompts/reflector.py:6-18) says the subject is
# "a MODEL's reasoning" and asks "what should THE MODEL have done instead". Our
# first version said "critique this task execution", and on a QA task the
# reflector duly personified a human operator — "the user may have overlooked
# ..." — which the curator then turned into advice for that operator (training
# programmes, mentorship pairings) instead of knowledge an answering model can
# apply. Measured on FiNER 2026-08-10: every bullet in a 30-sample run was
# process advice. Who the playbook is FOR is the whole mechanism.
REFLECT_PROMPT = """You are an expert analyst and educator. Diagnose why a MODEL's reasoning
went wrong by analyzing the gap between its predicted answer and the ground truth
(or, on success, why it worked). Be specific about what the model should have
done differently — the root cause, not the surface error.

Task: {task}
Outcome: {outcome}
Trajectory:
{trajectory}

Bullets from the playbook that were available (tag each as helpful/harmful/neutral
if you can tell from the trajectory; else omit):
{used_bullets}

Return JSON: {{"reasoning": "chain of thought / detailed analysis",
"error_identification": "what specifically went wrong (or 'nothing' on success)",
"root_cause_analysis": "why did this occur? what concept was misunderstood?",
"correct_approach": "what should have been done instead?",
"key_insight": "what strategy, formula, or principle should be remembered?",
"bullet_tags": [{{"id": "<bullet id>", "tag": "helpful"}}]}}"""

# Upstream's Context block (prompts/curator.py:7-9) is reproduced verbatim in
# substance because it is the sentence that decides what a bullet looks like:
# the playbook's reader is a model answering a SIMILAR question WITHOUT the gold
# answer the reflection was written against. Drop it and the curator writes
# advice for whoever it imagines is reading — the failure measured above.
CURATE_PROMPT = """You are a master curator of knowledge. Identify ONLY the NEW insights that are
MISSING from the current playbook.

The playbook you create will be used to help ANSWER SIMILAR QUESTIONS. The
reflection below was written with access to the ground truth, which will NOT be
available when the playbook is used — so every bullet must be content that helps
the playbook's reader produce a prediction that lands on the ground truth by
itself. Bullets must be actionable at answer time, not recommendations about
process, training or tooling.

Do NOT regenerate or rephrase existing bullets.
Avoid redundancy — if similar advice already exists, only add content that
complements it. Focus on quality over quantity. If there is nothing new to add,
return an empty operations list.

Training Context:
- Total token budget: {token_budget} tokens
- Progress: step {current_step}{progress_total}

Current Playbook Stats:
{playbook_stats}

Recent Reflection:
{recent_reflection}

Current playbook sections and bullets:
{playbook}

Question Context:
{question_context}

Return JSON: {{"operations": [{{"type": "ADD", "section": "<snake_case_section>",
"content": "one self-contained strategy/fact/pitfall"}}]}}"""

DEDUP_THRESHOLD = 0.90

# Upstream `playbook_token_budget` (ace.py:127). The curator is TOLD the budget
# rather than truncated at it: ACE's whole claim is that a comprehensive playbook
# is what prevents context collapse, so growth is steered by telling the model
# how much room is left, never by dropping bullets. Upstream never tells the
# curator the CURRENT playbook size — `count_tokens` is logging-only (ace.py:491,
# 626) — so neither do we (an invented "now uses about N tokens" line was
# removed, round-12 #2(a)).
PLAYBOOK_TOKEN_BUDGET = 80000

# Environment feedback, verbatim from upstream's two branches (ace.py:515/558).
# Our `outcome` is a free-form string, so this mapping only fires when it names
# one of the two states; anything else passes through as the no-ground-truth
# variant upstream also supports (`REFLECTOR_PROMPT_NO_GT`).
ENVIRONMENT_FEEDBACK = {
    "success": "Predicted answer matches ground truth",
    "correct": "Predicted answer matches ground truth",
    "failure": "Predicted answer does not match ground truth",
    "incorrect": "Predicted answer does not match ground truth",
    "wrong": "Predicted answer does not match ground truth",
}


def playbook_stats(bullets: list[dict]) -> str:
    """Upstream ``get_playbook_stats`` (playbook_utils.py:218), rendered for the
    curator prompt: totals plus the three health buckets it defines —
    high-performing (helpful > 5 and harmful < 2), problematic
    (harmful >= helpful, harmful > 0) and unused (no counter set) — and a
    per-section count."""
    total = high = problematic = unused = 0
    by_section: dict[str, int] = {}
    for bullet in bullets:
        total += 1
        helpful = int(bullet.get("helpful", 0))
        harmful = int(bullet.get("harmful", 0))
        if helpful > 5 and harmful < 2:
            high += 1
        elif harmful >= helpful and harmful > 0:
            problematic += 1
        if helpful + harmful == 0:
            unused += 1
        section = str(bullet.get("section", "general"))
        by_section[section] = by_section.get(section, 0) + 1
    sections = ", ".join(f"{name}={count}" for name, count in sorted(by_section.items()))
    return (
        f"- total bullets: {total}\n"
        f"- high-performing: {high}\n"
        f"- problematic: {problematic}\n"
        f"- unused: {unused}\n"
        f"- by section: {sections or '(none)'}"
    )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class ACEOrganizer(Organizer):
    """ACE playbook organizer (arXiv:2510.04618 §3; see module docstring for
    upstream deviations). Reflector+curator write only through returned
    MemoryOps — reads of the full playbook go through ``ctx.doc_store`` /
    ``ctx.vector_store`` (round-5 §3.2), never a partial top-k view."""

    name = "ace"

    produces = ("playbook",)

    def __init__(
        self,
        dedup_threshold: float = DEDUP_THRESHOLD,
        token_budget: int = PLAYBOOK_TOKEN_BUDGET,
        total_samples: int | None = None,
    ) -> None:
        """``dedup_threshold`` gates embedding-cosine dedup against both the
        existing playbook and the current curator batch (round-5 §3.4,
        always-on per module docstring). The curator's operations list is
        uncapped, as upstream's is (round-12 #2(d)).

        ``token_budget`` is upstream's ``playbook_token_budget`` (80,000): the
        curator is told the budget so it can slow the playbook's growth —
        only the budget, never the current size, which upstream keeps to its
        logs (round-12 #2(a)). It is a prompt input, never a truncation —
        dropping bullets to fit would be the context collapse the paper is
        about. ``total_samples`` is upstream's "Sample X out of Y" progress
        line; a memory layer does not know the dataset size, so it stays
        optional and the line degrades to the step alone."""
        self.dedup_threshold = dedup_threshold
        self.token_budget = token_budget
        self.total_samples = total_samples
        # Upstream's `step`, which drives both the progress line and its
        # `curator_frequency`. We curate on every task (frequency 1), so this
        # only feeds the prompt.
        self._step = 0

    # -- helpers -------------------------------------------------------------

    def _current_playbook(self, ctx: OrganizerContext) -> list[dict]:
        # The FULL playbook, as the official curator sees it (round-5 ACE
        # §3.2 — a task-similar top-k partial view let paraphrase duplicates
        # through, since "MISSING?" was judged against an incomplete list).
        return ctx.doc_store.list_items("playbook", namespace=ctx.namespace)

    def _render_playbook(self, bullets: list[dict]) -> str:
        # One display format everywhere: [section-id5], matching
        # Bullet.render() and memory.get_playbook() (round-5 ACE §3.6).
        if not bullets:
            return "(empty)"
        by_section: dict[str, list[str]] = {}
        for b in bullets:
            section = b.get("section", "general")
            by_section.setdefault(section, []).append(
                f"[{section}-{b['id'][:5]}] helpful={b.get('helpful', 0)} "
                f"harmful={b.get('harmful', 0)} :: {b.get('content', '')}"
            )
        return "\n".join(f"## {s}\n" + "\n".join(lines) for s, lines in sorted(by_section.items()))

    # -- hooks ---------------------------------------------------------------

    def on_feedback(
        self, memory_ids: list[str], helpful: bool, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Bump each named bullet's ``helpful``/``harmful`` counter by one.

        These counters are ACE's own (§3.3: bullets carry usage statistics that
        the curator reads back), so they belong to this organizer rather than to
        the facade. Ids that are not playbook bullets are ignored — another
        organizer owns them, or nothing does."""
        field = "helpful" if helpful else "harmful"
        ops: list[MemoryOp] = []
        for mid in memory_ids:
            bullets = ctx.doc_store.get_items([mid], "playbook")
            if not bullets:
                continue
            ops.append(
                MemoryOp(
                    op=OpType.UPDATE,
                    target_type="playbook",
                    target_id=mid,
                    payload={field: int(bullets[0].get(field, 0)) + 1},
                )
            )
        return ops

    def on_task_end(
        self, trajectory: list[dict], outcome: str, task: str, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Reflect on the trajectory, then curate new bullets. Returns []
        with no side effect when no LLM is configured or the reflection
        call fails (explicit skip, logged); otherwise returns UPDATE ops
        for helpful/harmful counters on tag-validated existing bullet ids
        plus ADD ops for curated bullets that survive dedup (see
        ``dedup_threshold``)."""
        if ctx.llm is None:
            logger.warning("ace: no LLM configured — skipping reflection (explicit skip)")
            return []

        import json as _json

        traj_text = "\n".join(_json.dumps(s, ensure_ascii=False, default=str) for s in trajectory)[
            :6000
        ]
        self._step += 1
        playbook = self._current_playbook(ctx)
        by_id = {b["id"]: b for b in playbook}
        rendered_playbook = self._render_playbook(playbook)

        # Boundary-cut note (round-12 #3): upstream reflects — and tags counters —
        # on EVERY reflection round of its generator loop (up to 3 per incorrect
        # task, ace.py:499-545); we reflect once per task, so upstream counters
        # accrue faster on failures. A consequence of cutting at on_task_end,
        # not a bug. Both this call and the curate call below ride the "distill"
        # role, so per-role model/temperature cannot split reflector from
        # curator — upstream allows distinct reflector/curator models
        # (ace.py:36-63); round-12 #7.
        reflection = ctx.llm.call(
            "distill",
            REFLECT_PROMPT.format(
                task=task,
                outcome=ENVIRONMENT_FEEDBACK.get(str(outcome).strip().lower(), outcome),
                trajectory=traj_text,
                used_bullets=rendered_playbook,
            ),
            REFLECT_SCHEMA,
            required_keys=("key_insight",),
        )
        if reflection is None:
            return []

        ops: list[MemoryOp] = []

        # counter updates from bullet tags (validated against real ids)
        for tag in reflection.get("bullet_tags", []) or []:
            # models echo the display id "[section-xxxxx]" or just "xxxxx" —
            # strip any section prefix, then resolve the 5-char prefix
            bullet_id_prefix = str(tag.get("id") or "").strip("[]").rsplit("-", 1)[-1]
            matches = [
                full
                for full in by_id
                if full == bullet_id_prefix or full.startswith(bullet_id_prefix)
            ]
            if len(matches) != 1 or tag.get("tag") not in ("helpful", "harmful"):
                continue
            full_id = matches[0]
            field = tag["tag"]
            ops.append(
                MemoryOp(
                    op=OpType.UPDATE,
                    target_type="playbook",
                    target_id=full_id,
                    payload={field: int(by_id[full_id].get(field, 0)) + 1},
                )
            )

        # The curator sees the FULL raw reflection (all five reflector fields +
        # tags), as upstream passes the reflector's whole response string through
        # (`recent_reflection`), and the question context, as upstream does
        # (`question_context`) — our closest counterpart is the task string.
        curated = ctx.llm.call(
            "distill",
            CURATE_PROMPT.format(
                playbook=rendered_playbook,
                recent_reflection=_json.dumps(reflection, ensure_ascii=False, indent=2),
                question_context=task,
                token_budget=self.token_budget,
                current_step=self._step,
                progress_total=(
                    f" of {self.total_samples}" if self.total_samples is not None else ""
                ),
                playbook_stats=playbook_stats(playbook),
            ),
            CURATE_SCHEMA,
            required_keys=("operations",),
        )
        if curated is None:
            return ops

        accepted_embeddings: list[list[float]] = []  # intra-batch dedup (round-5 §3.4)
        for raw in curated.get("operations") or []:
            content = str(raw.get("content", "")).strip()
            if not content:
                continue
            # deterministic grow-and-refine: embedding dedup, always on
            embedding = ctx.embedder.embed([content])[0]
            dup = ctx.vector_store.search(
                embedding, k=1, memory_type="playbook", namespace=ctx.namespace
            )
            if dup and dup[0][1] >= self.dedup_threshold:
                logger.info("ace: dedup skipped near-duplicate bullet (sim=%.2f)", dup[0][1])
                continue
            if any(
                _cosine(embedding, prev) >= self.dedup_threshold for prev in accepted_embeddings
            ):
                logger.info("ace: dedup skipped intra-batch near-duplicate")
                continue
            accepted_embeddings.append(embedding)
            bullet = Bullet(
                content=content,
                section=str(raw.get("section", "general")) or "general",
                namespace=ctx.namespace,
            )
            ops.append(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="playbook",
                    target_id=bullet.id,
                    payload={
                        "id": bullet.id,
                        "section": bullet.section,
                        "content": content,
                        "helpful": 0,
                        "harmful": 0,
                        "embedding_text": content,
                    },
                )
            )
        return ops
