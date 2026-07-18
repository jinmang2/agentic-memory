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
Read contract (round-5): ACE injects the FULL playbook — use
``AgenticMemory.get_playbook()``, never top-k retrieval of bullets. The
curator likewise sees the whole playbook, and dedup also compares within
the current batch. Reflector tagging remains trajectory-evidence-based
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

REFLECT_SCHEMA = {
    "type": "object",
    "properties": {
        "key_insight": {"type": "string"},
        "lessons": {"type": "array", "items": {"type": "string"}},
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
    "required": ["key_insight", "lessons"],
}

CURATE_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "maxItems": 5,
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

REFLECT_PROMPT = """You are a reflector. Critique this task execution and extract concrete insight.

Task: {task}
Outcome: {outcome}
Trajectory:
{trajectory}

Bullets from the playbook that were available (tag each as helpful/harmful/neutral
if you can tell from the trajectory; else omit):
{used_bullets}

Return JSON: {{"key_insight": "...", "lessons": ["specific, actionable", ...],
"bullet_tags": [{{"id": "<bullet id>", "tag": "helpful"}}]}}"""

CURATE_PROMPT = """You are a curator of a playbook. Identify ONLY the NEW insights that are
MISSING from the current playbook. Do NOT regenerate or rephrase existing bullets.

Current playbook sections and bullets:
{playbook}

New reflection:
key_insight: {key_insight}
lessons: {lessons}

Return JSON: {{"operations": [{{"type": "ADD", "section": "<snake_case_section>",
"content": "one self-contained strategy/fact/pitfall"}}]}}"""

DEDUP_THRESHOLD = 0.90


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

    def __init__(self, dedup_threshold: float = DEDUP_THRESHOLD, max_ops: int = 5) -> None:
        """``dedup_threshold`` gates embedding-cosine dedup against both the
        existing playbook and the current curator batch (round-5 §3.4,
        always-on per module docstring); ``max_ops`` caps ADD operations
        accepted from one curator call."""
        self.dedup_threshold = dedup_threshold
        self.max_ops = max_ops

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

    # -- hook ----------------------------------------------------------------

    def on_task_end(
        self, trajectory: list[dict], outcome: str, task: str, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Reflect on the trajectory, then curate new bullets. Returns []
        with no side effect when no LLM is configured or the reflection
        call fails (explicit skip, logged); otherwise returns UPDATE ops
        for helpful/harmful counters on tag-validated existing bullet ids
        plus ADD ops for curated bullets that survive dedup (see
        ``dedup_threshold``), up to ``max_ops``."""
        if ctx.llm is None:
            logger.warning("ace: no LLM configured — skipping reflection (explicit skip)")
            return []

        import json as _json

        traj_text = "\n".join(_json.dumps(s, ensure_ascii=False, default=str) for s in trajectory)[
            :6000
        ]
        playbook = self._current_playbook(ctx)
        by_id = {b["id"]: b for b in playbook}

        reflection = ctx.llm.call(
            "distill",
            REFLECT_PROMPT.format(
                task=task,
                outcome=outcome,
                trajectory=traj_text,
                used_bullets=self._render_playbook(playbook),
            ),
            REFLECT_SCHEMA,
            required_keys=("key_insight", "lessons"),
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

        curated = ctx.llm.call(
            "distill",
            CURATE_PROMPT.format(
                playbook=self._render_playbook(playbook),
                key_insight=reflection.get("key_insight", ""),
                lessons=reflection.get("lessons", []),
            ),
            CURATE_SCHEMA,
            required_keys=("operations",),
        )
        if curated is None:
            return ops

        accepted_embeddings: list[list[float]] = []  # intra-batch dedup (round-5 §3.4)
        for raw in (curated.get("operations") or [])[: self.max_ops]:
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
