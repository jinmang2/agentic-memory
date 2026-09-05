"""Core domain types shared by every organizer and store.

Design rules (docs/03, docs/04):
- Raw ``Episode`` records are immutable — organizers derive from them,
  never rewrite them (verbatim-loss defense).
- Every derived item keeps ``source_episode_ids`` provenance.
- ``Fact`` carries the bi-temporal fields from the Zep design.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def new_id() -> str:
    """32-char hex UUID4, used as the default id factory for every dataclass here."""
    return uuid.uuid4().hex


def utcnow() -> datetime:
    """Timezone-aware UTC now — never mix with naive `datetime.utcnow()` results."""
    return datetime.now(UTC)


# Memory type tags used for namespacing collections and filtering search.
# Must stay exhaustive: `core/ops.py` declares `MemoryOp.target_type` to be one
# of these, and a type missing here is a type nobody can find by reading the
# vocabulary (`experiences` was absent for as long as ReasoningBank has emitted
# it). Not enforced at runtime — `test_stores.py` checks it against every
# organizer's `produces` instead, which is the declaration that can go stale.
MEMORY_TYPES = (
    "episodic",  # raw episodes (always present, written by the facade itself)
    "episodes",  # Nemori derived narrative episodes
    "notes",  # A-Mem zettelkasten notes
    "pages",  # MemoryOS dialogue pages / segments
    "derivatives",  # MemMachine embedding anchors; never served, they map back to episodes
    "semantic",  # Nemori distilled facts, MemoryOS LPM profile facts (kind="profile")
    "entities",  # Zep-graph entity nodes
    "facts",  # Zep-graph bi-temporal edges
    "communities",  # Zep-graph entity clusters (label propagation, paper §2.2.4)
    "strategies",  # ReasoningBank items, G-Memory trajectories/insights
    "experiences",  # ReasoningBank task records (expand to their member strategies)
    "playbook",  # ACE bullets
    "runbooks",  # experience organizer: one distilled task block per coding-agent session
    "state",  # internal bookkeeping, not a memory: consolidate cursors (base.cursor_key)
)


@dataclass(frozen=True)
class TypeInfo:
    """What a memory type IS, on the two axes the vocabulary never declared.

    ``axis`` is the functional kind (CoALA's working / episodic / semantic /
    procedural; docs/research/agent-memory-axes-v1.md §1.3 for why that frame),
    ``update`` the policy that kind implies: raw experience is appended and
    never rewritten, facts are superseded by newer facts, procedures are
    abstracted from several cases, and bookkeeping is neither. Declaration
    only — nothing reads it yet. It exists so the next reader of the vocabulary
    does not have to infer, from the paper each organizer cites, which of the
    fourteen names are the same kind of thing (2026-09-04 §2.1 of that document
    did exactly that, by hand)."""

    axis: str  # "working" | "episodic" | "semantic" | "procedural" | "bookkeeping"
    update: str  # "append_only" | "supersede" | "abstract" | "none"


MEMORY_TYPE_INFO: dict[str, TypeInfo] = {
    "episodic": TypeInfo("episodic", "append_only"),
    "episodes": TypeInfo("episodic", "append_only"),
    "pages": TypeInfo("episodic", "append_only"),
    "derivatives": TypeInfo("episodic", "none"),  # anchors back to episodes, never served
    "semantic": TypeInfo("semantic", "supersede"),
    "notes": TypeInfo("semantic", "supersede"),
    "entities": TypeInfo("semantic", "supersede"),
    "facts": TypeInfo("semantic", "supersede"),  # bi-temporal: INVALIDATE, never delete
    "communities": TypeInfo("semantic", "supersede"),
    "strategies": TypeInfo("procedural", "abstract"),
    "experiences": TypeInfo("procedural", "abstract"),
    "playbook": TypeInfo("procedural", "abstract"),
    "runbooks": TypeInfo("procedural", "abstract"),
    "state": TypeInfo("bookkeeping", "none"),
}

# Types that stay servable after INVALIDATE, rendered with their validity range
# instead of dropping out (Zep bi-temporal: facts are never deleted, only
# invalidated). Every other type disappears from retrieval the moment it is
# invalidated. Lives here rather than in memory.py because both the write side
# (`_apply_one` decides whether to drop the vector) and the read side
# (`retrieval.steps.is_servable`) need it, and retrieval cannot import the facade.
BITEMPORAL_TYPES = ("facts",)


@dataclass(frozen=True)
class Episode:
    """Immutable raw input: one message or one ingested chunk."""

    content: str
    role: str = "user"
    id: str = field(default_factory=new_id)
    namespace: str = "main"
    timestamp: datetime = field(default_factory=utcnow)
    meta: dict[str, Any] = field(default_factory=dict)

    def embedding_text(self) -> str:
        """Text handed to the embedder. Every memory-type dataclass implements
        this method so retrieval can embed heterogeneous items polymorphically
        without a shared base class; for `Episode` it is exactly `content`."""
        return self.content


@dataclass
class Note:
    """A-Mem style zettelkasten note."""

    content: str
    id: str = field(default_factory=new_id)
    namespace: str = "main"
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    context: str = ""
    links: list[str] = field(default_factory=list)
    source_episode_ids: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=utcnow)

    def embedding_text(self) -> str:
        """Content plus keywords/tags/context, unlike the plain-content default."""
        # A-Mem finding: embed content concatenated with metadata. Three
        # formats exist: ours = paper eq.(3) unlabeled order (content, K, G, X);
        # upstream both editions = labeled "content:.. context:.. keywords:..
        # tags:.." order c,X,K,G (memory_layer.py:722); and after any upstream
        # consolidate_memories() the corpus silently switches to a third format
        # ("content , context keywords tags", memory_layer.py:749-751).
        parts = [
            self.content,
            " ".join(self.keywords),
            " ".join(self.tags),
            self.context,
        ]
        return " \n".join(p for p in parts if p)


@dataclass
class SemanticFact:
    """Distilled knowledge statement (Nemori calibration output)."""

    content: str
    id: str = field(default_factory=new_id)
    namespace: str = "main"
    confidence: float = 1.0
    source_episode_ids: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=utcnow)

    def embedding_text(self) -> str:
        """Plain `content` — no metadata folded in (unlike `Note`)."""
        return self.content


@dataclass
class Entity:
    """Deduplicated entity node (Zep-graph)."""

    name: str
    id: str = field(default_factory=new_id)
    namespace: str = "main"
    summary: str = ""
    entity_type: str = "Entity"
    source_episode_ids: list[str] = field(default_factory=list)

    def embedding_text(self) -> str:
        """`"name: summary"` once an entity has an LLM-generated summary, else the bare name."""
        return f"{self.name}: {self.summary}" if self.summary else self.name


@dataclass
class Fact:
    """Bi-temporal edge between two entities (Zep design).

    ``valid_at``/``invalid_at`` describe the real world; ``created_at``/
    ``expired_at`` describe what the system believed and when. Facts are
    never deleted — they are invalidated (OpType.INVALIDATE).
    """

    subject_id: str
    predicate: str
    object_id: str
    content: str
    id: str = field(default_factory=new_id)
    namespace: str = "main"
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    created_at: datetime = field(default_factory=utcnow)
    expired_at: datetime | None = None
    source_episode_ids: list[str] = field(default_factory=list)

    def embedding_text(self) -> str:
        """Plain `content` (the fact sentence, not subject/predicate/object)."""
        return self.content


@dataclass
class StrategyItem:
    """ReasoningBank memory item (also used for G-Memory insights)."""

    title: str
    description: str
    content: str
    id: str = field(default_factory=new_id)
    namespace: str = "main"
    # success | failure | contrast. The third is ReasoningBank's MaTTS bank:
    # those items are distilled from a MIXED set of attempts, so neither of the
    # first two is true of them, and upstream's parallel induction carries no
    # per-item label at all (see ``ReasoningBankOrganizer.on_scaled_task_end``).
    outcome: str = "success"
    score: float = 0.0  # G-Memory reward shaping
    source_episode_ids: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=utcnow)

    def embedding_text(self) -> str:
        """Title + description only — `content` is excluded (render-only field)."""
        return f"{self.title}\n{self.description}"

    def render(self) -> str:
        """Markdown block injected into LLM context by `MemoryBundle.render`."""
        return f"## {self.title}\n{self.description}\n{self.content}"


def render_bullet_line(
    content: str, section: str, bullet_id: str, helpful: int, harmful: int
) -> str:
    """The one playbook-line format, shared by `Bullet.render` and
    `AgenticMemory.get_playbook`.

    The facade renders bullets straight from stored dicts rather than rebuilding
    `Bullet`s, so it had its own copy of this f-string. Two identical formats in
    two files is one silent divergence away from a playbook that reads
    differently depending on which entry point produced it — and the format is
    part of ACE's prompt contract (the Generator picks bullets out of it)."""
    return f"[{section}-{bullet_id[:5]}] helpful={helpful} harmful={harmful} :: {content}"


@dataclass
class Bullet:
    """ACE playbook bullet with helpful/harmful counters."""

    content: str
    section: str = "general"
    id: str = field(default_factory=new_id)
    namespace: str = "main"
    helpful: int = 0
    harmful: int = 0
    source_episode_ids: list[str] = field(default_factory=list)

    def embedding_text(self) -> str:
        """Plain `content` (section/helpful/harmful counters are render-only)."""
        return self.content

    def render(self) -> str:
        """Playbook-line form injected into LLM context, tagged with section
        and the helpful/harmful counters ACE's reflect step updates."""
        return render_bullet_line(self.content, self.section, self.id, self.helpful, self.harmful)


@dataclass
class ScoredItem:
    """A retrieval hit: the item plus where it came from and its rank score."""

    item: Any
    memory_type: str
    score: float
    provenance: list[str] = field(default_factory=list)


@dataclass
class MemoryBundle:
    """Search result across memory types, renderable under a token budget."""

    query: str
    items: list[ScoredItem] = field(default_factory=list)

    # Rough chars-per-token used by render(); good enough for budgeting.
    CHARS_PER_TOKEN = 4

    # Upstream-style section headers (Nemori search.py labels its context
    # "Episodic Memories:"/"Semantic Memories:"); other types get their name.
    SECTION_TITLES = {
        "episodes": "Episodic Memories",
        "semantic": "Semantic Memories",
        "episodic": "Messages",
        "notes": "Notes",
        "runbooks": "Runbooks",
    }

    def render(self, budget_tokens: int = 1600) -> str:
        """Select items by descending score, stopping at the FIRST item that does
        not fit, then render them grouped by memory type (in bundle insertion
        order) so each type forms one labeled section, as the upstream evals do.

        "Stopping at the first item that does not fit" is not "until the budget
        is spent", which is what this used to claim: one long item drops every
        lower-scored item after it, including ones that would have fit. Whether
        that is the right policy is an open decision, not a reproduction
        question — upstream A-Mem has no context budget at all (it injects the
        retrieved notes directly; ``max_tokens`` there is the generation cap), so
        the budget is our own addition and there is no upstream behavior to
        match. The alternatives each lose something: skipping the oversized item
        and continuing fills the budget but can drop the 3rd-most-relevant memory
        while keeping the 10th; truncating it preserves score order at the cost
        of serving a cut-off memory.

        It has never bound on a measured run: across the A-Mem reproduction's
        1986 questions x 2 configs, the largest bundle renders to 11,663 chars
        against a 24,000-char budget (49%), and no question exceeds it — so no
        published number here depends on which policy is chosen. Nemori/MemoryOS
        configs render narratives plus attached source messages and are much
        heavier, but their artifacts predate retrieval capture, so that is
        unmeasured rather than known-safe."""
        budget_chars = budget_tokens * self.CHARS_PER_TOKEN
        selected: list[ScoredItem] = []
        used = 0
        for scored in sorted(self.items, key=lambda s: s.score, reverse=True):
            item = scored.item
            text = item.render() if hasattr(item, "render") else getattr(item, "content", str(item))
            if used + len(text) > budget_chars and selected:
                break
            selected.append(scored)
            used += len(text)

        type_order = list(dict.fromkeys(s.memory_type for s in self.items))
        sections: list[str] = []
        for memory_type in type_order:
            picked = [s for s in selected if s.memory_type == memory_type]
            if not picked:
                continue
            title = self.SECTION_TITLES.get(memory_type, memory_type)
            lines = []
            for scored in sorted(picked, key=lambda s: s.score, reverse=True):
                item = scored.item
                text = (
                    item.render()
                    if hasattr(item, "render")
                    else getattr(item, "content", str(item))
                )
                lines.append(f"- {text}")
            sections.append(f"{title}:\n" + "\n".join(lines))
        return "\n\n".join(sections)
