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
from datetime import datetime, timezone
from typing import Any


def new_id() -> str:
    """32-char hex UUID4, used as the default id factory for every dataclass here."""
    return uuid.uuid4().hex


def utcnow() -> datetime:
    """Timezone-aware UTC now — never mix with naive `datetime.utcnow()` results."""
    return datetime.now(timezone.utc)


# Memory type tags used for namespacing collections and filtering search.
MEMORY_TYPES = (
    "episodic",  # raw episodes (always present)
    "episodes",  # Nemori derived narrative episodes
    "notes",  # A-Mem zettelkasten notes
    "pages",  # MemoryOS dialogue pages / segments
    "semantic",  # Nemori distilled facts, MemoryOS knowledge
    "entities",  # Zep-graph entity nodes
    "facts",  # Zep-graph bi-temporal edges
    "strategies",  # ReasoningBank items, G-Memory insights
    "playbook",  # ACE bullets
)


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
        # A-Mem finding: embed content concatenated with metadata.
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
    outcome: str = "success"  # success | failure
    score: float = 0.0  # G-Memory reward shaping
    source_episode_ids: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=utcnow)

    def embedding_text(self) -> str:
        """Title + description only — `content` is excluded (render-only field)."""
        return f"{self.title}\n{self.description}"

    def render(self) -> str:
        """Markdown block injected into LLM context by `MemoryBundle.render`."""
        return f"## {self.title}\n{self.description}\n{self.content}"


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
        return f"[{self.section}-{self.id[:5]}] helpful={self.helpful} harmful={self.harmful} :: {self.content}"


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
    }

    def render(self, budget_tokens: int = 1600) -> str:
        """Select items by descending score until the budget is spent, then
        render them grouped by memory type (in bundle insertion order) so
        each type forms one labeled section, as the upstream evals do."""
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
