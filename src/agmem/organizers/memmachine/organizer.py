"""MemMachine organizer (arXiv:2604.04853) — the DEPLOYED-CODE lineage.

Which lineage this reproduces was decided before a line was written, the way
Nemori's and MemoryOS's were: the paper describes three tiers (short-term /
long-term episodic / **profile**) and the shipped server has no ``profile``
package at all — it has ``episodic_memory/{short_term,long_term,event,
declarative}_memory`` plus a separate top-level ``semantic_memory/`` clustering
subsystem. We reproduce the **deployed code**, because that is the only side
with a second implementation to diff against; the paper's tier names survive
here only as this note. Read at ``MemMachine/MemMachine`` commit ``18f1211``
(2026-07-20, Apache-2.0).

**The published LoCoMo number does not come from the segmenter/deriver path.**
``evaluation/utils/agent_utils.py::init_memmachine_params`` builds
``LongTermMemoryParams(vector_graph_store=..., embedder=..., reranker=...)``,
which is the ``declarative`` variant of the discriminated union, and
``EpisodicMemory(..., short_term_memory=None)``. So the measured configuration
is: declarative backend, no short-term memory, and therefore **zero LLM calls
on the whole write path** — not "one async summary call". ``TextSegmenter``
(``chunk_size=500``) and ``SentenceTextDeriver`` belong to the ``event``
backend, are BOTH non-default there (``EventLongTermMemoryConf`` defaults to
``segmenter=passthrough``, ``deriver=whole_text``), and no published number
runs through them. ``docs/research/memmachine.md`` §1.1 cited them as *the*
write path; that pointer is corrected there and encoded as presets here.

Two backends, and they are not interchangeable — same rule as
``MEMORYOS_PRESETS``/``NEMORI_PRESETS``, provenance never mixed inside one
preset:

- ``declarative`` (``declarative_memory.py``): one Derivative per episode,
  ``f"{source}: {content}"``, embedded and linked to its Episode node by a
  ``DERIVED_FROM`` edge. ``message_sentence_chunking=True`` switches it to one
  derivative per sentence. No segmentation stage exists at all. A reranker is
  REQUIRED (``DeclarativeMemoryParams.reranker`` is not optional) and it scores
  whole episode CONTEXTS, not single items — see ``MemMachineContextualize``.
- ``event`` (``event_memory/``): Segmenter -> Deriver -> vector store +
  SQL segment store. Config defaults are passthrough + whole-text; the anchor
  text is ``[{full date}] {producer}: {json.dumps(text)}``
  (``deriver/text_deriver.py::_format_for_embedding``, with
  ``FormatOptions(time_style=None)`` so the date survives and the time does
  not). Reranker optional.

Both are LLM-free per message, which is the reason this paper is in the
comparison table at all: A-Mem spends 2 calls per turn to extract, MemMachine
extracts nothing, and until now our table had no real point between
``passthrough`` and A-Mem.

Short-term memory is implemented here but **off in both presets**, because the
measured lineage has none. When enabled it is upstream's
``ShortTermMemory``: a character-budget deque (``message_capacity``, default
64000 upstream, 500 in ``sample_configs``) that evicts and then rewrites a
rolling summary with ONE LLM call (``short_term_memory.py::_do_evict`` ->
``ShortTermMemoryConsolidator._create_summary``). That call is the only
language model in MemMachine's entire write path.

Read-path counterpart: one step per backend on the ``derivatives`` type, and
the facade picks by THIS organizer's ``backend`` — ``MemMachineContextualize``
(declarative: episode mapping, asymmetric context expansion, context-level
rerank, weighted-proximity unification) or ``MemMachineEventContextualize``
(event: segment-level expansion, embedding-score fallback, first-seen episode
dedup, no unification). That dispatch is what makes the "provenance never
mixed inside one preset" rule above hold on the read side too. The eval
operating point (``evaluation/episodic_memory/locomo_search.py``: ``limit=30``,
``expand_context=3``) is a search recipe, not a property of this organizer —
it lives in ``AgmemConfig.memmachine_*``, the same way Zep's search recipes do.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, new_id
from agmem.organizers.base import Organizer, OrganizerContext

logger = logging.getLogger("agmem.organizers.memmachine")

# Upstream's two lineages of long-term memory. `backend` is upstream's own
# discriminator (`LongTermMemoryConf`, resolved by
# `_long_term_memory_backend_discriminator`, whose "missing means declarative"
# rule is why `declarative` is our default too).
#
# `stm_capacity` is 0 in BOTH presets on purpose: `init_memmachine_params`
# passes `short_term_memory=None`, so every published episodic-memory number
# was produced without one. Upstream's own default when a server DOES configure
# it is `message_capacity=64000`; `sample_configs/*.sample` sets 500.
MEMMACHINE_PRESETS: dict[str, dict] = {
    "declarative": {
        "backend": "declarative",
        # Declarative memory has no segmenter stage; the field exists so both
        # presets have the same shape and a mismatched value fails loudly
        # rather than being silently ignored.
        "segmenter": "passthrough",
        "deriver": "whole_text",  # = `message_sentence_chunking=False`
        "stm_capacity": 0,
    },
    "event": {
        "backend": "event",
        "segmenter": "passthrough",  # EventLongTermMemoryConf default
        "deriver": "whole_text",  # EventLongTermMemoryConf default
        "stm_capacity": 0,
    },
}

# `common/configuration/default_episode_summary_system_prompt.txt`, verbatim.
STM_SUMMARY_SYSTEM_PROMPT = """You are an AI agent that can make summary for a list of episode and previous summary. Please make a concise summary
for the giving episode. You must:
1. Make the summary as short as you can
2. Keep as much detail as you can
3. All the entities and relationships must be kept in the summary"""

# `common/configuration/default_episode_summary_user_prompt.txt`, verbatim
# (upstream validates that it carries exactly the three fields below).
STM_SUMMARY_USER_PROMPT = """You are responsible for maintaining a rolling summary that functions as the short-term memory for another LLM to understand the context later without access to any other information.
You are given an existing rolling summary and new episodes since that summary. Your task is to rewrite the summary from scratch, integrating new information from the episodes.

<instructions>
1. Maximize information density. Be meticulous in recording details instead of generalized statements.
2. Extract insights, constraints, preferences, conclusions, corrections, and decision-relevant facts.
3. Your summary will replace all past context including the previous summary and raw episodes, so include all names, events, places, and dates (including year, month, and day).
4. Older information that is no longer relevant may be compressed, merged, or removed.
</instructions>

<summary>
{summary}
</summary>

<episodes>
{episodes}
</episodes>

Your summary (under {max_length} words):"""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

# `episodic_memory/event_memory/segmenter/text_segmenter.py`, verbatim and in
# order — the priority list is the whole behavior of the splitter, so it is
# reproduced rather than approximated (fullwidth punctuation, the ideographic
# full stop and the zero-width space are all deliberate).
TEXT_SEGMENTER_SEPARATORS = [
    "\n\n",
    "],\n",
    "},\n",
    "),\n",
    "]\n",
    "}\n",
    ")\n",
    ",\n",
    "？\n",
    "?\n",
    "！\n",
    "!\n",
    "。\n",
    ".\n",
    "？",
    "? ",
    "！",
    "! ",
    "。",
    ". ",
    "; ",
    ": ",
    "—",
    "--",
    "，",
    "、",
    ", ",
    "​",
    " ",
    "",
]

# `common/utils.py::extract_sentences`, second stage. The first stage is
# nltk's `sent_tokenize`, which we call rather than reimplement.
_SENTENCE_TAIL = re.compile(r".*?(?:[?!\.？！。]*[?!？！。][?!\.？！。]*)+|.+$")


def extract_sentences(text: str) -> set[str]:
    """Port of upstream ``common/utils.py::extract_sentences``.

    Returns a SET, exactly as upstream does, and that is not a detail: sentence
    order is discarded and a repeated sentence yields ONE derivative. Both
    matter for any run with ``deriver="sentence_text"`` (or the declarative
    backend's ``message_sentence_chunking=True``), because the number of
    embedded anchors is then not the number of sentences.

    nltk is upstream's tokenizer (``from nltk import sent_tokenize``) and it is
    an optional dependency here, in the ``eval`` extra. A missing nltk raises
    rather than falling back to a hand-rolled splitter: a different sentence
    split is a different memory, and a silent substitute would make the run
    unfalsifiable (docs/03 §5)."""
    from nltk import sent_tokenize  # gated: `uv sync --extra eval`

    partitions = {
        partition for line in text.strip().splitlines() for partition in sent_tokenize(line.strip())
    }
    return {
        sentence
        for partition in partitions
        for sentence in _SENTENCE_TAIL.findall(partition)
        if any(c.isalnum() for c in sentence)
    }


def _speaker(episode: Episode) -> str:
    """Upstream's ``episode.source`` / ``ProducerContext.producer``.

    That is ``producer_id``, which the LoCoMo ingest sets to the speaker name
    (``locomo_ingest.py``). Our ingest carries the same value in
    ``meta["speaker"]``, so the mapping is speaker-then-role — the same
    resolution ``retrieval/steps.py::_speaker`` already uses for MemoryOS
    pages, kept identical so two methodologies cannot disagree about who said
    a message."""
    return str((episode.meta or {}).get("speaker") or episode.role)


def format_full_date(timestamp: datetime) -> str:
    """The event backend's anchor date: babel ``format_date(style="full",
    locale="en_US")``.

    Not ``strftime("%A, %B %d, %Y")``: CLDR's full date does not zero-pad the
    day, so babel renders "Thursday, May 5, 2022" where strftime renders
    "May 05". The declarative backend's own renderer DOES use the zero-padded
    strftime (``declarative_memory.py::_format_date``), so the two lineages
    print the same instant differently — reproduced rather than unified."""
    return f"{timestamp:%A, %B} {timestamp.day}, {timestamp.year}"


class MemMachineOrganizer(Organizer):
    """MemMachine derivative writer — mechanical, no LLM call per message.

    One episode becomes one or more ``derivatives``: embedding anchors that
    point back at the raw episode, which the facade has already stored. The
    read path never serves derivatives; it maps them back to their episodes
    (``MemMachineContextualize``), which is why this organizer looks so thin
    next to the extractive ones — the whole methodology is "index the messages
    themselves, spend the budget at read time".

    ``fidelity`` picks a lineage from ``MEMMACHINE_PRESETS``; any other
    explicit (non-None) argument overrides that preset's value, the same
    contract ``MemoryOSOrganizer`` uses.
    """

    name = "memmachine"

    produces = ("derivatives",)

    def __init__(
        self,
        fidelity: str = "declarative",
        *,
        segmenter: str | None = None,
        deriver: str | None = None,
        max_chunk_length: int | None = None,
        stm_capacity: int | None = None,
        stm_context_limit: int = 20,
    ) -> None:
        """``segmenter`` is ``passthrough``|``text`` and ``deriver`` is
        ``whole_text``|``sentence_text`` — upstream's ``SegmenterConf`` /
        ``DeriverConf`` tags. ``max_chunk_length`` is ``TextSegmenterConf``'s
        (upstream default 500).

        ``stm_capacity`` is upstream's ``message_capacity`` in CHARACTERS, and
        0 disables short-term memory entirely, which is what every published
        number ran with. ``stm_context_limit`` is how many recent turns
        ``recent_context`` injects (upstream ``query_memory``'s ``limit``
        default, 20).

        A ``declarative`` preset with ``segmenter="text"`` raises: declarative
        memory has no segmentation stage, so honouring it would invent a
        pipeline no upstream has."""
        params = dict(MEMMACHINE_PRESETS.get(fidelity, MEMMACHINE_PRESETS["declarative"]))
        if fidelity not in MEMMACHINE_PRESETS:
            logger.warning(
                "memmachine: unknown fidelity %r — falling back to 'declarative'", fidelity
            )
            fidelity = "declarative"
        self.fidelity = fidelity
        self.backend = params["backend"]
        self.segmenter = segmenter if segmenter is not None else params["segmenter"]
        self.deriver = deriver if deriver is not None else params["deriver"]
        self.max_chunk_length = 500 if max_chunk_length is None else max_chunk_length
        self.stm_capacity = params["stm_capacity"] if stm_capacity is None else max(0, stm_capacity)
        self.stm_context_limit = stm_context_limit

        if self.segmenter not in ("passthrough", "text"):
            raise ValueError(f"unknown segmenter {self.segmenter!r} (passthrough|text)")
        if self.deriver not in ("whole_text", "sentence_text"):
            raise ValueError(f"unknown deriver {self.deriver!r} (whole_text|sentence_text)")
        if self.backend == "declarative" and self.segmenter != "passthrough":
            raise ValueError(
                "the declarative backend has no segmentation stage — "
                "use fidelity='event' for a segmenter other than passthrough"
            )

        # Short-term memory state (upstream ShortTermMemory). `_since_summary`
        # is upstream's `_current_episode_count`: how many episodes arrived
        # after the last summarization, which is what its eviction loop
        # protects from being dropped before it has been summarized.
        self._buffer: list[Episode] = []
        self._since_summary = 0
        self._buffer_chars = 0
        self._summary = ""

    # ---- write path ---------------------------------------------------------

    def on_message(self, episode: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        """Segment -> derive -> ADD one item per derivative, then (only when
        short-term memory is enabled) run the eviction/summary step.

        No LLM call happens here in either preset. The raw episode is already
        durable and searchable by the time this runs (write-then-organize,
        docs/04 §2), which is exactly the Episode node upstream writes beside
        its derivatives."""
        anchors = self._anchors(episode)
        ops = [
            self._add_op(episode, text, offset, segment)
            for offset, (segment, text) in enumerate(anchors)
        ]
        ops.extend(self._short_term(episode, ctx))
        return ops

    def _anchors(self, episode: Episode) -> list[tuple[int | None, str]]:
        """``(segment index, anchor text)`` pairs, in emission order.

        The segment index is upstream's derivative -> segment link (each event
        derivative row carries ``_SEGMENT_UUID_FIELD_NAME``), which the event
        read path needs for its segment-level context expansion. Declarative
        derivatives carry ``None`` — that backend has no segment stage, and
        inventing an index for it would blur exactly the line the presets
        exist to keep."""
        source = _speaker(episode)
        if self.backend == "declarative":
            # `declarative_memory.py::_derive_derivatives`, ContentType.MESSAGE
            # branch: the whole message, or one derivative per sentence.
            #
            # `sorted` around `extract_sentences` is ours: upstream iterates the
            # SET, so its anchor order varies with string hashing between
            # processes. The set of anchors is identical either way — only the
            # emission (and therefore id) order is pinned, the same fix the zep
            # read path needed for its equal-scored facts.
            if self.deriver == "sentence_text":
                sentences = sorted(extract_sentences(episode.content))
                return [(None, f"{source}: {sentence}") for sentence in sentences]
            return [(None, f"{source}: {episode.content}")]

        # Event backend: segmenter first, then deriver, then the timestamped
        # anchor format (`_format_for_embedding`).
        stamp = format_full_date(episode.timestamp)
        anchors: list[tuple[int | None, str]] = []
        for segment_index, segment in enumerate(self._segments(episode.content)):
            texts = (
                sorted(extract_sentences(segment)) if self.deriver == "sentence_text" else [segment]
            )
            anchors.extend(
                (segment_index, f"[{stamp}] {source}: {json.dumps(text, ensure_ascii=False)}")
                for text in texts
            )
        return anchors

    def _segments(self, text: str) -> list[str]:
        """``PassthroughSegmenter`` (one segment per block) or
        ``TextSegmenter``.

        The text segmenter IS langchain's ``RecursiveCharacterTextSplitter``
        upstream, constructed with a 30-entry separator list and
        ``keep_separator="end"``; we call that same splitter rather than
        reimplement it, and raise when it is absent (``uv sync --extra
        memmachine``). A hand-rolled recursive splitter would be a different
        chunking, and "close enough chunking" is not something a later audit
        can distinguish from the real one."""
        if self.segmenter == "passthrough":
            return [text]
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # gated

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_length,
            chunk_overlap=0,
            separators=TEXT_SEGMENTER_SEPARATORS,
            keep_separator="end",
        )
        return splitter.split_text(text)

    def _add_op(
        self, episode: Episode, anchor: str, offset: int, segment: int | None = None
    ) -> MemoryOp:
        """One derivative. ``timestamp``/``source_episode_ids`` are the read
        path's whole index: ``MemMachineContextualize`` orders episodes by
        ``(timestamp, episode id)``, which is upstream's
        ``search_directional_nodes(by_properties=("timestamp", "uid"))``.
        Event derivatives additionally carry their ``segment`` index —
        upstream's per-derivative segment UUID — which is the unit
        ``MemMachineEventContextualize`` expands around."""
        payload = {
            "content": anchor,
            "embedding_text": anchor,
            "source_episode_ids": [episode.id],
            "timestamp": episode.timestamp.isoformat(),
            "offset": offset,
        }
        if segment is not None:
            payload["segment"] = segment
        return MemoryOp(
            op=OpType.ADD,
            target_type="derivatives",
            target_id=new_id(),
            payload=payload,
        )

    # ---- short-term memory (off in both presets) ----------------------------

    def _short_term(self, episode: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        """Upstream ``ShortTermMemory.add_episodes`` -> ``_do_evict``.

        The budget is CHARACTERS of message content plus the summary's own
        length, not a turn count (``_get_total_message_len``). Eviction drops
        already-summarized episodes from the front first, and only summarizes
        when the buffer is still over budget after that — so an episode is
        summarized before it can be dropped, and the summary rewrite sees the
        WHOLE remaining buffer rather than just the evicted prefix.

        Deviation, structural: upstream summarizes on a background worker
        (``ShortTermMemoryConsolidator``) and lets ingestion continue; our
        hooks are synchronous, so the call happens inline. The resulting
        summary is the same — the worker is a latency device, and it drains
        sequentially precisely so summaries are never applied out of order.

        Returns no ops: short-term memory is upstream's in-process deque and is
        never persisted as a memory item (it reaches the prompt through
        ``recent_context``, and its only durable trace upstream is an optional
        ``SessionDataManager`` checkpoint)."""
        if not self.stm_capacity:
            return []
        self._buffer.append(episode)
        self._since_summary += 1
        self._buffer_chars += len(episode.content)
        if self._total_len() <= self.stm_capacity:
            return []

        while len(self._buffer) > self._since_summary and self._total_len() > self.stm_capacity:
            self._buffer_chars -= len(self._buffer[0].content)
            self._buffer.pop(0)
        if not self._buffer or self._total_len() <= self.stm_capacity:
            return []

        batch = list(self._buffer)
        self._since_summary = 0
        self._summary = self._summarize(batch, ctx)
        return []

    def _total_len(self) -> int:
        return self._buffer_chars + len(self._summary)

    @property
    def max_summary_length_words(self) -> int:
        """Upstream ``ShortTermMemory.max_summary_length_words``: half the
        character budget at 8 characters per word, rounded UP to the next
        hundred."""
        words = int(self.stm_capacity / 2 / 8)
        return (words + 99) // 100 * 100

    def _summarize(self, episodes: list[Episode], ctx: OrganizerContext) -> str:
        """The one LLM call in MemMachine's write path.

        Upstream keeps the previous summary on any failure (``_create_summary``
        returns ``summary`` unchanged after logging), and so do we — a dropped
        call must not silently blank the rolling summary. Upstream additionally
        retries by halving the batch on a context-window error; that recovery is
        not reproduced, because our structured caller reports a drop rather than
        an oversize-input error we could branch on, and inventing the branch
        would be reproducing a code path we cannot trigger.

        Second deviation, unavoidable: upstream calls ``generate_response`` and
        takes the free text back, while every LLM call in this repo goes
        through the structured caller. The system prompt is therefore upstream's
        verbatim text plus the JSON-envelope instruction, and the summary is
        read out of ``{"summary": ...}``. The task prompt — the part that
        decides what the summary says — is untouched."""
        if ctx.llm is None:
            logger.warning("memmachine: no LLM configured — short-term summary skipped")
            return self._summary
        result = ctx.llm.call(
            "distill",
            STM_SUMMARY_USER_PROMPT.format(
                summary=self._summary,
                episodes=self.episodes_to_string(episodes),
                max_length=self.max_summary_length_words,
            ),
            SUMMARY_SCHEMA,
            required_keys=("summary",),
            system=(
                f"{STM_SUMMARY_SYSTEM_PROMPT}\n\n"
                'Respond with a single JSON object {"summary": "..."} and nothing else.'
            ),
        )
        summary = str((result or {}).get("summary") or "")
        # `set_summary` ignores an empty summary ("if summary:"), keeping the
        # previous one — an empty rewrite is treated as a failed one.
        return summary or self._summary

    @staticmethod
    def episodes_to_string(episodes: list[Episode]) -> str:
        """``common/episode_store/episode_model.py::episodes_to_string`` —
        the format upstream feeds both the summary prompt and the QA prompt."""
        return "".join(
            f"[{episode.timestamp:%A, %B %d, %Y} at {episode.timestamp:%I:%M %p}] "
            f"{_speaker(episode)}: {json.dumps(episode.content, ensure_ascii=False)}\n"
            for episode in episodes
        )

    def recent_context(self) -> str:
        """Upstream ``EpisodicMemory.formalize_query_with_context``: the rolling
        summary and the most recent turns, injected verbatim beside the
        retrieved memories.

        Empty in both presets, since neither configures short-term memory. Note
        that upstream's LoCoMo harness would not use this format anyway — it
        renders the summary under ``<WORKING MEMORY SUMMARY>`` and merges the
        recent turns into the retrieved-episode list — but that harness builds
        ``short_term_memory=None``, so the divergence is unreachable in the
        measured lineage.

        This follows the library's ENVELOPE, not its full content: upstream's
        ``formalize_query_with_context`` merges STM episodes AND long-term
        search results under ``<Episodes>``, while this method injects only
        the STM buffer — long-term results are served by the search pipeline
        here (the seam mapping), so the merged view never exists in one place.
        Unreachable in both presets while STM is off, but the split should be
        named, not papered over."""
        if not self.stm_capacity or (not self._summary and not self._buffer):
            return ""
        parts: list[str] = []
        if self._summary:
            parts.append(f"<Summary>\n{self._summary}\n</Summary>")
        recent = self._buffer[-self.stm_context_limit :] if self.stm_context_limit else []
        if recent:
            parts.append(f"<Episodes>\n{self.episodes_to_string(recent)}</Episodes>")
        return "\n".join(parts)

    def state(self) -> dict[str, Any]:
        """Short-term buffer snapshot, for tests and artifact capture."""
        return {
            "summary": self._summary,
            "buffered": len(self._buffer),
            "since_summary": self._since_summary,
            "chars": self._buffer_chars,
        }
