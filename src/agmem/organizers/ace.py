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
                "properties": {"id": {"type": "string"},
                               "tag": {"type": "string", "enum": ["helpful", "harmful", "neutral"]}},
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
                "properties": {"type": {"type": "string", "enum": ["ADD"]},
                               "section": {"type": "string"},
                               "content": {"type": "string"}},
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


class ACEOrganizer(Organizer):
    name = "ace"

    def __init__(self, dedup_threshold: float = DEDUP_THRESHOLD,
                 max_ops: int = 5) -> None:
        self.dedup_threshold = dedup_threshold
        self.max_ops = max_ops

    # -- helpers -------------------------------------------------------------

    def _current_playbook(self, ctx: OrganizerContext, query_text: str,
                          k: int = 30) -> list[dict]:
        emb = ctx.embedder.embed([query_text])[0]
        hits = ctx.vec.search(emb, k=k, memory_type="playbook", namespace=ctx.namespace)
        return ctx.doc.get_items([h[0] for h in hits], "playbook")

    def _render_playbook(self, bullets: list[dict]) -> str:
        if not bullets:
            return "(empty)"
        by_section: dict[str, list[str]] = {}
        for b in bullets:
            by_section.setdefault(b.get("section", "general"), []).append(
                f"[{b['id'][:5]}] helpful={b.get('helpful', 0)} "
                f"harmful={b.get('harmful', 0)} :: {b.get('content', '')}")
        return "\n".join(f"## {s}\n" + "\n".join(lines)
                         for s, lines in sorted(by_section.items()))

    # -- hook ----------------------------------------------------------------

    def on_task_end(self, trajectory: list[dict], outcome: str,
                    task: str, ctx: OrganizerContext) -> list[MemoryOp]:
        if ctx.llm is None:
            logger.warning("ace: no LLM configured — skipping reflection (explicit skip)")
            return []

        import json as _json
        traj_text = "\n".join(_json.dumps(s, ensure_ascii=False, default=str)
                              for s in trajectory)[:6000]
        playbook = self._current_playbook(ctx, task)
        by_id = {b["id"]: b for b in playbook}

        reflection = ctx.llm.call(
            "distill",
            REFLECT_PROMPT.format(task=task, outcome=outcome, trajectory=traj_text,
                                  used_bullets=self._render_playbook(playbook)),
            REFLECT_SCHEMA, required_keys=("key_insight", "lessons"),
        )
        if reflection is None:
            return []

        ops: list[MemoryOp] = []

        # counter updates from bullet tags (validated against real ids)
        for tag in reflection.get("bullet_tags", []) or []:
            bid = tag.get("id")
            # models often echo the truncated 5-char display id — resolve it
            matches = [full for full in by_id if full == bid or full.startswith(str(bid))]
            if len(matches) != 1 or tag.get("tag") not in ("helpful", "harmful"):
                continue
            full_id = matches[0]
            field = tag["tag"]
            ops.append(MemoryOp(
                op=OpType.UPDATE, target_type="playbook", target_id=full_id,
                payload={field: int(by_id[full_id].get(field, 0)) + 1},
            ))

        curated = ctx.llm.call(
            "distill",
            CURATE_PROMPT.format(playbook=self._render_playbook(playbook),
                                 key_insight=reflection.get("key_insight", ""),
                                 lessons=reflection.get("lessons", [])),
            CURATE_SCHEMA, required_keys=("operations",),
        )
        if curated is None:
            return ops

        for raw in (curated.get("operations") or [])[: self.max_ops]:
            content = str(raw.get("content", "")).strip()
            if not content:
                continue
            # deterministic grow-and-refine: embedding dedup, always on
            emb = ctx.embedder.embed([content])[0]
            dup = ctx.vec.search(emb, k=1, memory_type="playbook",
                                 namespace=ctx.namespace)
            if dup and dup[0][1] >= self.dedup_threshold:
                logger.info("ace: dedup skipped near-duplicate bullet (sim=%.2f)", dup[0][1])
                continue
            bullet = Bullet(content=content,
                            section=str(raw.get("section", "general")) or "general",
                            namespace=ctx.namespace)
            ops.append(MemoryOp(
                op=OpType.ADD, target_type="playbook", target_id=bullet.id,
                payload={"id": bullet.id, "section": bullet.section,
                         "content": content, "helpful": 0, "harmful": 0,
                         "embedding_text": content},
            ))
        return ops
