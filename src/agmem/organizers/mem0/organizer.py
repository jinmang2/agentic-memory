"""Mem0's paper-era two-phase write path, ported from `v0.1.94`.

Port pin: mem0ai/mem0 tag `v0.1.94` (07ddd7cb…, 2025-04-26). Every citation
below resolves against `mem0/memory/main.py` at that tag unless said otherwise.
The tag is load-bearing: at HEAD this pipeline is gone, replaced by a
single-call ADD-only path, so an untagged citation to Mem0's "write path" names
a different system (docs/research/mem0.md §②-2).

Structure — **exactly two LLM calls per `add()`, always**:

1. `extract` — `FACT_RETRIEVAL_PROMPT` as the system message, `f"Input:\\n{parsed}"`
   as the user message (`:212-227` via `utils.get_fact_retrieval_messages`).
   Yields a list of standalone facts, possibly empty.
2. `distill` — one batched decision call over *all* facts at once (`:261-266`),
   not one per fact. It fires **unconditionally**: `:229-234` swallows a parse
   failure into `new_retrieved_facts = []` and falls through, so F=0 still costs
   the second call. The fact count changes the prompt, never the call count.

Between them, retrieval: each fact is embedded and searched with a hardcoded
`limit=5` and **no similarity floor** (`:241-246`), results unioned and deduped
by id, then renumbered `0..n-1` — upstream's own guard against the model
hallucinating UUIDs (`:255-259`, comment theirs).

The decision call returns four events, which map onto our evolution log 1:1 —
the reason this methodology was selected:

| upstream event | our op | side effect |
|---|---|---|
| `ADD` | `OpType.ADD` | new id; payload carries `content`/`hash`/`created_at` |
| `UPDATE` | `OpType.UPDATE` | merged into the existing item; `created_at` preserved |
| `DELETE` | `OpType.DELETE` | tombstone + vector drop; the op log IS upstream's history row |
| `NONE` | `OpType.NOOP` | logged, nothing changed |

Three upstream asymmetries reproduced rather than fixed (study §④), each
promoted from a silent loss to a counted one in `discarded`:

- `NONE` is not returned to the caller at all (`:326-327`), so upstream's own
  caller cannot tell "judged, unchanged" from "never judged". We log it.
- An entry with falsy `text` is dropped **regardless of event** — the check
  precedes the event branch (`:286-288`), so a text-less DELETE is a no-op too.
  Counted as `empty_text`.
- An `id` outside the integer mapping raises `KeyError` into the inner `except`
  (`:328-329`), which logs and continues; nothing counts the loss. Counted as
  `hallucinated_id`.

Three deliberate deviations from upstream, none of them silent:

- **One store per conversation.** The paper harness ingests each session twice
  into two per-speaker stores with roles swapped and searches both at QA time
  (`evaluation/src/memzero/{add,search}.py` @ `evaluation-archive`). Our harness
  has one namespace and one search, shared with every other arm of the 4-way
  table. Consequence, which the results write-up must state: upstream's harness
  cost is ~2x ours for the same session, and our per-`add()` call count is the
  library-level figure.
- **`NOOP` is a log row**, not a dropped event — see the table above and
  `core/ops.py`.
- **Embedder call parity is not attempted.** Upstream embeds each fact for
  retrieval and reuses that vector on ADD/UPDATE when the LLM echoed the fact
  verbatim, re-embedding otherwise (`:645-651`, `:729-733`); our facade owns
  embedding and embeds once per applied op. With a local MiniLM the difference
  is exactly zero dollars, and the study's "<=2F embedder calls" bound only
  bites on hosted embedders.
"""

from __future__ import annotations

import hashlib
import logging

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, new_id, utcnow
from agmem.organizers.base import Organizer, OrganizerContext
from agmem.organizers.mem0.prompts import (
    fact_retrieval_prompt,
    get_update_memory_messages,
    parse_messages,
)

logger = logging.getLogger(__name__)

FACTS_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
}

UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "memory": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "event": {"type": "string", "enum": ["ADD", "UPDATE", "DELETE", "NONE"]},
                    "old_memory": {"type": "string"},
                },
                "required": ["id", "text", "event"],
            },
        }
    },
    "required": ["memory"],
}


class Mem0Organizer(Organizer):
    """Mem0 v0.1.94 vector-variant write path. See the module docstring."""

    name = "mem0"
    produces = ("semantic",)

    def __init__(self, top_k: int = 5, batch_size: int = 1) -> None:
        # top_k=5 is upstream's hardcoded `limit=5` per fact (main.py:244 @
        # v0.1.94) — a literal, not a config knob, and there is NO similarity
        # floor beside it. batch_size is OURS: upstream's library batches
        # nothing, its harness passes 2 messages per add (add.py:46 @
        # evaluation-archive). Default 1 = library semantics; the paper-harness
        # shape is configs.py's job, so mechanism lives here and policy there.
        self.top_k = top_k
        self.batch_size = batch_size
        self._buffer: list[Episode] = []
        # Silent-discard promotion (organizer contract): every place upstream
        # loses work without counting it becomes a counter here.
        self.discarded: dict[str, int] = {}

    def on_message(self, episode: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        self._buffer.append(episode)
        if len(self._buffer) < self.batch_size:
            return []
        return self._drain(ctx)

    def flush_buffer(self, ctx: OrganizerContext) -> list[MemoryOp]:
        """Drain a short tail. `AgenticMemory.flush()` calls this, so a final
        odd message at `batch_size=2` still gets its `add()` instead of being
        stranded in the buffer (same contract as Nemori's segment buffer)."""
        return self._drain(ctx)

    def _drain(self, ctx: OrganizerContext) -> list[MemoryOp]:
        batch, self._buffer = self._buffer, []
        return self._add(batch, ctx) if batch else []

    def _discard(self, reason: str) -> None:
        self.discarded[reason] = self.discarded.get(reason, 0) + 1

    def _add(self, batch: list[Episode], ctx: OrganizerContext) -> list[MemoryOp]:
        # The batch's date. Upstream's harness attaches the session timestamp per
        # add() (add.py:83 @ evaluation-archive); the last message in the batch
        # carries it, mirroring A-Mem's talk_time (organizers/amem/organizer.py).
        talk_time = batch[-1].meta.get("date") or batch[-1].timestamp.isoformat()

        if ctx.llm is None:
            # Upstream's infer=False path (main.py:204-211): store each message
            # verbatim, no LLM, system messages excluded. Explicit degradation,
            # not a silent skip.
            logger.warning("mem0: no LLM configured — storing messages verbatim (infer=False path)")
            return [
                self._add_op(new_id(), ep.content, talk_time) for ep in batch if ep.role != "system"
            ]

        parsed = parse_messages([{"role": ep.role, "content": ep.content} for ep in batch])
        facts_resp = ctx.llm.call(
            "extract",
            f"Input:\n{parsed}",
            FACTS_SCHEMA,
            required_keys=("facts",),
            system=fact_retrieval_prompt(),
        )
        # A dropped extract is NOT a skipped add: upstream's json.loads failure
        # sets facts=[] and the flow continues to the decision call (:229-234).
        # Returning early here would make the call count depend on parse luck.
        facts = [str(f) for f in (facts_resp or {}).get("facts", [])]
        if facts_resp is None:
            self._discard("no_facts")

        # Per-fact top-5 with no floor, unioned and deduped by id. Insertion
        # order is first-seen (upstream builds a list then collapses it through a
        # dict, :247-252 — later duplicates overwrite the value but not the
        # position, and the value is the same text either way). Order is
        # load-bearing because it becomes the integer ids the model answers with.
        retrieved: dict[str, str] = {}
        for fact in facts:
            hits = ctx.vector_store.search(
                ctx.embedder.embed([fact])[0],
                k=self.top_k,
                memory_type="semantic",
                namespace=ctx.namespace,
            )
            for item in ctx.doc_store.get_items([h[0] for h in hits], "semantic"):
                retrieved.setdefault(str(item["id"]), str(item.get("content", "")))

        # UUID -> integer, upstream's own hallucination guard (:254-259).
        temp_uuid = {str(i): mid for i, mid in enumerate(retrieved)}
        shown = [{"id": str(i), "text": text} for i, text in enumerate(retrieved.values())]

        verdict = ctx.llm.call(
            "distill",
            get_update_memory_messages(shown, facts),
            UPDATE_SCHEMA,
            required_keys=("memory",),
        )
        if verdict is None:
            # Upstream's equivalent: a json parse failure leaves the variable as
            # [], whose .get raises AttributeError into the outer except
            # (:330-331), and nothing is applied. Same outcome, counted.
            self._discard("no_verdict")
            return []

        ops: list[MemoryOp] = []
        for resp in verdict.get("memory", []):
            text = str(resp.get("text") or "")
            if not text:
                # BEFORE the event branch, as upstream has it (:286-288) — a
                # DELETE with no text is dropped too, which reads like a bug and
                # is reproduced as one.
                self._discard("empty_text")
                continue
            event = str(resp.get("event") or "")
            if event == "ADD":
                ops.append(self._add_op(new_id(), text, talk_time))
                continue
            target = temp_uuid.get(str(resp.get("id")))
            if target is None:
                # Upstream indexes temp_uuid_mapping directly (:304/311/318), so
                # an out-of-range integer or a real UUID raises KeyError into the
                # inner except and the loss goes uncounted. Ours is counted.
                self._discard("hallucinated_id")
                continue
            if event == "UPDATE":
                ops.append(self._update_op(target, text, talk_time))
            elif event == "DELETE":
                ops.append(
                    MemoryOp(op=OpType.DELETE, target_type="semantic", target_id=target, payload={})
                )
            elif event == "NONE":
                ops.append(
                    MemoryOp(
                        op=OpType.NOOP,
                        target_type="semantic",
                        target_id=target,
                        payload={"text": text},
                    )
                )
            else:
                # Upstream's if/elif chain has no else: an unrecognized event
                # falls through silently. Reproduced as a no-op, counted.
                self._discard("unknown_event")
        return ops

    def _add_op(self, item_id: str, text: str, talk_time: str) -> MemoryOp:
        return MemoryOp(
            op=OpType.ADD,
            target_type="semantic",
            target_id=item_id,
            payload={
                "id": item_id,
                "content": text,  # metadata["data"] (:654)
                "hash": hashlib.md5(
                    text.encode()
                ).hexdigest(),  # metadata["hash"] (:655) — md5 is upstream's, kept for hash parity, not security
                "created_at": utcnow().isoformat(),  # (:656) — US/Pacific upstream, UTC here
                "timestamp": talk_time,  # harness metadata (add.py:83)
                "embedding_text": text,  # upstream embeds `data` itself
            },
        )

    def _update_op(self, target: str, text: str, talk_time: str) -> MemoryOp:
        return MemoryOp(
            op=OpType.UPDATE,
            target_type="semantic",
            target_id=target,
            payload={
                "content": text,
                "hash": hashlib.md5(
                    text.encode()
                ).hexdigest(),  # md5 is upstream's (:719), kept for hash parity, not security
                "updated_at": utcnow().isoformat(),  # (:721)
                "timestamp": talk_time,
                "embedding_text": text,
                # `created_at` is DELIBERATELY ABSENT. `_apply_one` merges an
                # UPDATE into the existing item, so omitting the key is what
                # preserves the original creation time — which is exactly what
                # upstream does by copying it off the existing payload (:720).
                # "Completing" this payload with a fresh created_at would erase
                # the very field the study checks.
            },
        )
