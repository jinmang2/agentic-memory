"""MemMachine's semantic-memory tier — the paper's "profile", under another name.

``docs/research/memmachine.md`` §1.3 recorded that the deployed code has no
``profile`` package and "instead a clustering subsystem". Half right: the
package is ``semantic_memory/`` and it is a **profile extractor** — an LLM turns
each message into ``add``/``delete`` commands over a two-level key-value store
(*tag* -> *feature* -> *value*), with citations back to the episodes that
produced each feature and a periodic consolidation pass that merges a tag once
it grows past a threshold. The clustering (``cluster_manager``/
``cluster_splitter``) is a separate event-grouping state machine that the
ingestion path never touches; it is deliberately not ported here (see the end of
this docstring).

**This tier is where MemMachine's LLM budget actually goes, and it changes how
the headline may be quoted.** ``MemMachineOrganizer``'s claim — zero LLM calls
per message — is true of the *episodic* path only. With semantic memory enabled
the write path costs **one LLM call per message per category**
(``semantic_ingestion.py::process_semantic_type`` loops messages inside a loop
over ``resources.semantic_categories``), plus one per over-threshold tag at
consolidation. That is A-Mem's order of magnitude, not ``passthrough``'s. It is
off in the measured lineage — ``evaluation/episodic_memory/locomo_config.yaml``
sets ``semantic_memory.enabled: false`` — so no published number includes it,
and none of ours may either unless this organizer is explicitly active.

Faithful pieces (upstream file ``semantic_memory/`` at commit ``18f1211``):

- the update prompt (``util/semantic_prompt_template.build_update_prompt``) and
  the shipped default category ``profile`` with its 37 tags
  (``server/prompt/profile_prompt.py``), verbatim;
- the user-prompt envelope of ``semantic_llm.llm_feature_update``
  (``<OLD_PROFILE>`` as ``{tag: {feature: value}}`` JSON, then ``<HISTORY>``);
- ``max_features_per_update=50`` — the cap on how much of the existing profile
  the update call is shown;
- ``add``/``delete`` semantics: delete removes EVERY value under
  ``(category, tag, feature)``, and an update is expressed as delete-then-add;
- consolidation at ``consolidated_threshold=20`` features under one tag, where
  the model returns ``keep_memories`` (ids) plus merged features, everything not
  kept is deleted, and the merged feature inherits the union of the deleted
  features' citations.

Three defects found while porting, none smoothed over:

1. **The consolidation prompt documents a key the parser does not read.** The
   prompt's own schema and no-op example say ``consolidate_memories``; the
   parsed model field is ``consolidated_memories``
   (``semantic_llm.SemanticConsolidateMemoryRes``). Upstream decodes through
   ``instructor`` with the pydantic model as the response format, so a provider
   that enforces the schema hides this; one that merely reads the prompt does
   not, and then the group is **wiped** — every feature not in ``keep_memories``
   is deleted while ``consolidated_memories`` defaults to empty, so nothing is
   written back. We read both spellings, and log when the prompt's one arrives.
2. **The update prompt's own ``delete`` example is invalid against its schema.**
   ``SemanticCommand`` requires all four of ``command``/``tag``/``feature``/
   ``value``, but the documented delete form omits ``value``; a model that
   copies the example fails validation, and upstream's failure granularity is
   the whole message (the exception escapes ``llm_feature_update``, the caller
   logs and ``continue``s, and every command for that message is lost). We
   reproduce the drop granularity and count it instead of silently repairing.
3. **A JSON array closed with a brace.** The fourth few-shot example in the
   update prompt ends with ``}`` where the other three end with ``]``. Kept
   verbatim — it is what the model is actually shown, and "fixing" a few-shot
   example silently changes the extraction distribution.

Not ported, and why: the cluster subsystem (centroid state, reranker-guided
splitting) groups *events*, not features, and nothing in the ingestion path
calls it — it is a separate mechanism that would need its own study rather than
a footnote here. Multi-tenant set/category storage (``config_store/``) is
deployment infrastructure: this organizer takes its categories as a constructor
argument instead.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, new_id
from agmem.organizers.base import Organizer, OrganizerContext

logger = logging.getLogger("agmem.organizers.memmachine.profile")

# Marks the items this organizer owns inside the shared `semantic` type.
# MemoryOS already writes `kind="profile"` there for its single LPM document,
# which `bench/locomo.py` injects verbatim into every prompt; these are
# (tag, feature, value) triples and must NOT be swept into that section, so they
# carry their own kind (docs/04 §3.2: memory types are shared, `actor`/`kind`
# say who owns a row).
PROFILE_FEATURE_KIND = "profile_feature"

# `server/prompt/profile_prompt.py::meta_tags`, verbatim and in order — this is
# the shipped default category (`prompt.default_project_categories:
# [profile_prompt]`), and the tag list IS the extraction target.
PROFILE_TAGS: dict[str, str] = {
    "Assistant Response Preferences": "How the user prefers the assistant to communicate (style, tone, structure, data format).",
    "Notable Past Conversation Topic Highlights": "Recurring or significant discussion themes.",
    "Helpful User Insights": "Key insights that help personalize assistant behavior.",
    "User Interaction Metadata": "Behavioral/technical metadata about platform use.",
    "Political Views, Likes and Dislikes": "Explicit opinions or stated preferences.",
    "Psychological Profile": "Personality characteristics or traits.",
    "Decision-Making Style": "How the user tends to think, plan, and decide.",
    "Hard Skills": "Specialized technical or professional capabilities in a work domain.",
    "Soft Skills": "Professional competencies such as communication, teamwork, or leadership.",
    "Communication Style": "Describes the user's communication tone and pattern.",
    "Learning Preferences": "Preferred modes of receiving information.",
    "Cognitive Style": "How the user processes information or makes decisions.",
    "Emotional Drivers": "Motivators like fear of error or desire for clarity.",
    "Personal Values": "User's core values or principles.",
    "Career & Work Preferences": "Interests, titles, domains related to work.",
    "Productivity Style": "User's work rhythm, focus preference, or task habits.",
    "Working Habit Preferences": "Recurring, personally driven ways the user organizes, performs, communicates about, or manages work.",
    "Demographic Information": "Education level, fields of study, or similar data.",
    "Geographic & Cultural Context": "Physical location or cultural background.",
    "Financial Profile": "Any relevant information about financial behavior or context.",
    "Health & Wellness": "Physical/mental health indicators.",
    "Education & Knowledge Level": "Degrees, subjects, or demonstrated expertise.",
    "Platform Behavior": "Patterns in how the user interacts with the platform.",
    "Tech Proficiency": "Languages, tools, frameworks the user knows.",
    "Hobbies & Interests": "Non-work-related interests.",
    "Social Identity": "Group affiliations or demographics.",
    "Media Consumption Habits": "Types of media consumed (e.g., blogs, podcasts).",
    "Life Goals & Milestones": "Short- or long-term aspirations.",
    "Relationship & Family Context": "Any information about personal life.",
    "Risk Tolerance": "Comfort with uncertainty, experimentation, or failure.",
    "Assistant Trust Level": "Whether and when the user trusts assistant responses.",
    "Time Usage Patterns": "Frequency and habits of use.",
    "Preferred Content Format": "Formats preferred for answers (e.g., tables, bullet points).",
    "Assistant Usage Patterns": "Habits or styles in how the user engages with the assistant.",
    "Language Preferences": "Preferred tone and structure of assistant's language.",
    "Motivation Triggers": "Traits that drive engagement or satisfaction.",
    "Behavior Under Stress": "How the user reacts to failures or inaccurate responses.",
}

# `profile_prompt.py::description`, verbatim.
PROFILE_DESCRIPTION = """
    IMPORTANT: Extract ALL personal information, even basic facts like names, ages, locations, etc. Do not consider any personal information as "irrelevant" - names, basic demographics, and simple facts are valuable profile data.

    Category-specific rules:
    - Working Habit Preferences: A recurring, personally driven tendency in how the user organizes, performs, communicates about, or manages their work.
      * Explicit linguistic signals: "I prefer", "I like to", "I tend to", "I usually", "I'm used to", "I'd rather", "It's easier for me to", "I avoid", "I don't like", "I try to avoid", "Works best for me when".
      * Boundary test: If the external rule disappeared, would the user still choose to do it that way? If yes, it's a preference.
    - Psychological Profile (Personality traits): Only extract traits when the user clearly and directly indicates them. Avoid speculative inference.
    - Decision-Making Style: Only add when there is clear evidence for one of these dimensions: SystematicThinking, QualityFirstPrinciple, DataDrivenDecisionMaking, ForwardLookingPlanning, ClearResponsibilityBoundaries, ContinuousImprovementMindset.
    - Hard Skills: Extract ONLY when the user explicitly demonstrates or describes hands-on use of tools/technologies, or explains technical principles in depth.
      * Use a level descriptor in the value: Expert / Proficient / Familiar, based on evidence in the user's message.
      * Example value format: "Backend Development - Proficient; basis: built REST APIs in Python and FastAPI".
    - Soft Skills: Rate only when evidence exists in the user's message. Use the six dimensions: Communication, Teamwork, Emotional Intelligence, Time Management, Problem-Solving, Leadership.
      * Example value format: "Communication - Strong; basis: concise structured requests and explicit constraints".

    General guidance:
    - Only extract what is supported by the user's message; avoid fabricating evidence.
    - Use concise, factual value text and keep entries atomic.
"""


def build_update_prompt(tags: Mapping[str, str], description: str = "") -> str:
    """``util/semantic_prompt_template.build_update_prompt``, verbatim.

    Including the malformed fourth example (defect 3) and the ``<think>``
    instruction: our ``extract_json`` reads the JSON out of surrounding text, so
    the thinking block survives unchanged rather than being edited away."""
    return (
        """
        Your job is to handle memory extraction for a memory system, one which takes the form of a profile recording details relevant to the tags below.
        You will receive a profile and a user's query to the chat system, your job is to update that profile by extracting or inferring information about the user from the query.
        A profile is a two-level key-value store. We call the outer key the *tag*, and the inner key the *feature*. Together, a *tag* and a *feature* are associated with one or several *value*s.

        """
        + description
        + """

        How to construct profile entries:
        - Entries should be atomic. They should communicate a single discrete fact.
        - Entries should be as short as possible without corrupting meaning. Be careful when leaving out prepositions, qualifiers, negations, etc. Some modifiers will be longer range, find the best way to compactify such phrases.
        - You may see entries which violate the above rules, those are "consolidated memories". Don't rewrite those.
        - Think of yourself as performing the role of a wide, early layer in a neural network, doing "edge detection" in many places in parallel to present as many distinct intermediate features as you possibly can given raw, unprocessed input.

        The tags you are looking for include:
        """
        + "\n".join([f"\t- {key}: {value}" for key, value in tags.items()])
        + """

        To update the profile, you will output a JSON document containing a list of commands to be executed in sequence.

        CRITICAL: You MUST use the command format below. Do NOT create nested objects or use any other format.

        The following output will add a feature:
        [
            {
                "command": "add",
                "tag": "Preferred Content Format",
                "feature": "unicode_for_math",
                "value": true
            }
        ]
        The following will delete all values associated with the feature:
        [
            {
                "command": "delete",
                "tag" : "Language Preferences",
                "feature": "format"
            }
        ]
        The following will update a feature:
        [
            {
                "command": "delete",
                "tag": "Platform Behavior",
                "feature": "prefers_detailed_responses",
                "value": true
            },
            {
                "command": "add",
                "tag" : "Platform Behavior",
                "feature": "prefers_detailed_response",
                "value": false
            }
        ]

        Example Scenarios:
        Query: "Hi! My name is Katara"
        [
            {
                "command": "add",
                "tag": "Demographic Information",
                "feature": "name",
                "value": "Katara"
            }
        ]
        Query: "I'm planning a dinner party for 8 people next weekend and want to impress my guests with something special. Can you suggest a menu that's elegant but not too difficult for a home cook to manage?"
        [
            {
                "command": "add",
                "tag": "Hobbies & Interests",
                "feature": "home_cook",
                "value": "User cooks fancy food"
            },
            {
                "command": "add",
                "tag": "Financial Profile",
                "feature": "upper_class",
                "value": "User entertains guests at dinner parties, suggesting affluence."
            }
        ]
        Query: my boss (for the summer) is totally washed. he forgot how to all the basics but still thinks he does
        [
            {
                "command": "add",
                "tag": "Psychological Profile",
                "feature": "work_superior_frustration",
                "value": "User is frustrated with their boss for perceived incompetence"
            },
            {
                "command": "add",
                "tag": "Demographic Information",
                "feature": "summer_job",
                "value": "User is working a temporary job for the summer"
            },
            {
                "command": "add",
                "tag": "Communication Style",
                "feature": "informal_speech",
                "value": "User speaks with all lower case letters and contemporary slang terms."
            },
            {
                "command": "add",
                "tag": "Demographic Information",
                "feature": "young_adult",
                "value": "User is young, possibly still in college"
            }
        ]
        Query: Can you go through my inbox and flag any urgent emails from clients, then update the project status spreadsheet with the latest deliverable dates from those emails? Also send a quick message to my manager letting her know I'll have the budget report ready by end of day tomorrow.
        [
            {
                "command": "add",
                "tag": "Demographic Information",
                "feature": "traditional_office_job",
                "value": "User does clerical work, reporting to a manager"
            },
            {
                "command": "add",
                "tag": "Demographic Information",
                "feature": "client_facing_role",
                "value": "User handles communication of deadlines to and from clients"
            },
            {
                "command": "add",
                "tag": "Demographic Information",
                "feature": "autonomy_at_work",
                "value": "User sets their own deadlines and subtasks."
            }
        }
        Further Guidelines:
        - Not everything you ought to record will be explicitly stated. Make inferences.
        - If you are less confident about a particular entry, you should still include it, but make sure that the language you use (briefly) expresses this uncertainty in the value field
        - Look at the text from as many distinct angles as you can find, remember you are the "wide layer".
        - Keep only the key details (highest-entropy) in the feature name. The nuances go in the value field.
        - Do not couple together distinct details. Just because the user associates together certain details, doesn't mean you should
        - Do not create new tags which you don't see in the example profile. However, you can and should create new features.
        - If a user asks for a summary of a report, code, or other content, that content may not necessarily be written by the user, and might not be relevant to the user's profile.
        - Do not delete anything unless a user asks you to
        - Only return the empty list [] if the query contains absolutely no personal information about the user (e.g., asking about the weather, requesting code without personal context, etc.). Names, basic demographics, and any personal details should ALWAYS be extracted.
        - Listen to any additional instructions specific to the execution context provided underneath 'EXTRA EXTERNAL INSTRUCTIONS'
        - First, think about what should go in the profile inside <think> </think> tags. Then output only a valid JSON.
        - REMEMBER: Always use the command format with "command", "tag", "feature", and "value" keys. Never use nested objects or any other format.
    """
    )


def build_consolidation_prompt(tags: Mapping[str, str] | None = None) -> str:
    """``util/semantic_prompt_template.build_consolidation_prompt``, verbatim —
    including the ``consolidate_memories`` key that the parser does not read
    (defect 1) and the missing comma in its no-op example."""
    tag_section = ""
    if tags:
        tag_section = (
            "\n    The valid tags for this category are:\n"
            + "\n".join([f"\t- {key}: {value}" for key, value in tags.items()])
            + "\n    You MUST only use these tags. Do not create new tag names.\n"
        )
    return (
        """
    Your job is to perform memory consolidation for an llm long term memory system.
    Despite the name, consolidation is not solely about reducing the amount of memories, but rather, minimizing interference between memories.
    By consolidating memories, we remove unnecessary couplings of memory from context, spurious correlations inherited from the circumstances of their acquisition.

    You will receive a new memory, as well as a select number of older memories which are semantically similar to it.
    Produce a new list of memories to keep.

    A memory is a json object with 4 fields:
    - tag: broad category of memory
    - feature: executive summary of memory content
    - value: detailed contents of memory
    - metadata: object with 1 fields
    -- id: integer
    You will output consolidated memories, which are json objects with 4 fields:
    - tag: string
    - feature: string
    - value: string
    - metadata: object with 1 field
    -- citations: list of ids of old memories which influenced this one
    You will also output a list of old memories to keep (memories are deleted by default)
"""
        + tag_section
        + """
    Guidelines:
    Memories should not contain unrelated ideas. Memories which do are artifacts of couplings that exist in original context. Separate them. This minimizes interference.
    Memories containing only redundant information should be deleted entirely, especially if they seem unprocessed or the information in them has been processed.
    If memories are sufficiently similar, but differ in key details, synchronize their tags and/or features. This creates beneficial interference.
        - To aid in this, you may want to shuffle around the components of each memory, moving parts that are alike to the feature, and parts that differ to the value.
        - Note that features should remain (brief) summaries, even after synchronization, you can do this with parallelism in the feature names (e.g. likes_apples and likes_bananas).
        - Keep only the key details (highest-entropy) in the feature name. The nuances go in the value field.
        - this step allows you to speculatively build towards more permanent structures
    If enough memories share similar features (due to prior synchronization, i.e. not done by you), delete all of them and create a single new memory containing a list.
        - In these memories, the feature contains all parts of the memory which are the same, and the value contains only the parts which vary.
        - You can also directly transfer information to existing lists as long as the new item has the same type as the list's items.
        - Don't make lists too early. Have at least three examples in a non-gerrymandered category first. You need to find the natural groupings. Don't force it.

    Overall memory life-cycle:
    raw memory ore -> pure memory pellets -> memory pellets sorted into bins -> alloyed memories

    The more memories you receive, the more interference there is in the overall memory system.
    This causes cognitive load. cognitive load is bad.
    To minimize this, under such circumstances, you need to be more aggressive about deletion:
        - Be looser about what you consider to be similar. Some distinctions are not worth the energy to maintain.
        - Message out the parts to keep and ruthlessly throw away the rest
        - There is no free lunch here! at least some information must be deleted!

    Do not create new tag names.


    The proper noop syntax is:
    {
        "consolidate_memories": []
        "keep_memories": []
    }

    The final output schema is:
    <think> insert your chain of thought here. </think>
    {
        "consolidate_memories": list of new memories to add
        "keep_memories": list of ids of old memories to keep
    }
    """
    )


UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "commands": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "enum": ["add", "delete"]},
                    "tag": {"type": "string"},
                    "feature": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["command", "tag", "feature", "value"],
            },
        }
    },
    "required": ["commands"],
}

CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "consolidated_memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string"},
                    "feature": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["tag", "feature", "value"],
            },
        },
        "keep_memories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["keep_memories"],
}


@dataclass
class SemanticCategory:
    """One extraction target: a named tag vocabulary plus its rules.

    Upstream keeps these in a config store per tenant (``config_store/``); here
    they are constructor data, because a per-tenant registry is deployment
    infrastructure and not part of the mechanism."""

    name: str
    tags: Mapping[str, str] = field(default_factory=dict)
    description: str = ""

    @property
    def update_prompt(self) -> str:
        return build_update_prompt(self.tags, self.description)

    @property
    def consolidation_prompt(self) -> str:
        return build_consolidation_prompt(self.tags)


PROFILE_CATEGORY = SemanticCategory("profile", PROFILE_TAGS, PROFILE_DESCRIPTION)


class MemMachineProfileOrganizer(Organizer):
    """MemMachine semantic memory: one LLM call per message per category turns
    the message into add/delete commands over a tag -> feature -> value profile.

    Composes with ``MemMachineOrganizer`` rather than replacing it — upstream
    runs both subsystems side by side over the same episodes, and they share
    nothing but the episode stream. Running this one alone is also valid: it is
    the profile tier with no episodic index.
    """

    name = "memmachine_profile"

    produces = ("semantic",)

    def __init__(
        self,
        categories: list[SemanticCategory] | None = None,
        consolidation_threshold: int = 20,
        max_features_per_update: int = 50,
        consolidate_every: int = 5,
    ) -> None:
        """Defaults are upstream's: the shipped ``profile`` category,
        ``consolidated_threshold=20``, ``max_features_per_update=50``, and a
        consolidation check every 5 messages — the size of the un-ingested batch
        ``_process_single_set`` pulls (``get_history_messages(limit=5)``), which
        is what actually sets how often the check runs.

        ``consolidate_every=0`` leaves consolidation to the explicit
        ``consolidate()`` pass only."""
        self.categories = categories if categories is not None else [PROFILE_CATEGORY]
        self.consolidation_threshold = consolidation_threshold
        self.max_features_per_update = max_features_per_update
        self.consolidate_every = consolidate_every
        self._seen = 0
        self.dropped_commands = 0

    # ---- write path ---------------------------------------------------------

    def on_message(self, episode: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        if ctx.llm is None:
            logger.warning("memmachine_profile: no LLM configured — no features extracted")
            return []
        ops: list[MemoryOp] = []
        for category in self.categories:
            ops.extend(self._update_category(category, episode, ctx))
        self._seen += 1
        if self.consolidate_every and self._seen % self.consolidate_every == 0:
            ops.extend(self.consolidate(ctx))
        return ops

    def _features(self, category: str, ctx: OrganizerContext) -> list[dict]:
        """This category's live features, oldest first.

        Upstream pages the feature set at ``max_features_per_update`` and hands
        the LLM that first page, so a profile larger than the cap is only ever
        partially visible to the update call — the cap is a truncation, not a
        sample."""
        rows = [
            data
            for data in ctx.doc_store.list_items("semantic", ctx.namespace)
            if data.get("kind") == PROFILE_FEATURE_KIND
            and data.get("category") == category
            and not data.get("deleted")
        ]
        return rows

    def _update_category(
        self, category: SemanticCategory, episode: Episode, ctx: OrganizerContext
    ) -> list[MemoryOp]:
        existing = self._features(category.name, ctx)[: self.max_features_per_update]
        profile: dict[str, dict[str, str]] = {}
        for row in existing:
            profile.setdefault(str(row.get("tag", "")), {})[str(row.get("feature", ""))] = str(
                row.get("value", "")
            )
        verdict = ctx.llm.call(
            "distill",
            # `semantic_llm.llm_feature_update`'s user prompt, verbatim.
            "The old feature set is provided below:\n"
            "<OLD_PROFILE>\n"
            f"{json.dumps(profile, ensure_ascii=False)}\n"
            "</OLD_PROFILE>\n"
            "\n"
            "The history is provided below:\n"
            "<HISTORY>\n"
            f"{episode.content}\n"
            "</HISTORY>\n",
            UPDATE_SCHEMA,
            required_keys=("commands",),
            system=category.update_prompt,
        )
        if verdict is None:
            return []
        commands = verdict.get("commands") or []
        # Upstream's failure granularity is the whole message: a single command
        # that misses a required field raises inside `llm_feature_update`, the
        # caller logs and `continue`s, and every command for this message is
        # lost — including the well-formed ones. Its own delete example is one
        # of those malformed commands (defect 2), so this is reachable.
        if any(
            not isinstance(command, dict)
            or not all(command.get(key) for key in ("command", "tag", "feature", "value"))
            for command in commands
        ):
            self.dropped_commands += len(commands)
            logger.warning(
                "memmachine_profile: malformed command in batch of %d — dropping the batch "
                "(upstream's granularity)",
                len(commands),
            )
            return []

        ops: list[MemoryOp] = []
        pending: dict[tuple[str, str], MemoryOp] = {}
        for command in commands:
            tag, feature = str(command["tag"]), str(command["feature"])
            if str(command["command"]).lower() == "add":
                op = self._add_op(category.name, tag, feature, str(command["value"]), episode)
                pending[(tag, feature)] = op
                ops.append(op)
                continue
            # delete: every value under (category, tag, feature). Commands run
            # in sequence against storage upstream, so a delete after an add of
            # the same feature in the same batch removes it — here that means
            # retracting the op we just queued rather than deleting a row that
            # does not exist yet.
            queued = pending.pop((tag, feature), None)
            if queued is not None:
                ops.remove(queued)
            for row in existing:
                if row.get("tag") == tag and row.get("feature") == feature:
                    ops.append(
                        MemoryOp(
                            op=OpType.DELETE,
                            target_type="semantic",
                            target_id=str(row["id"]),
                            payload={},
                        )
                    )
        return ops

    def _add_op(
        self, category: str, tag: str, feature: str, value: str, episode: Episode
    ) -> MemoryOp:
        """One feature. ``embedding_text`` is the VALUE alone, as upstream
        embeds (``_apply_commands``: ``ingest_embed([command.value])``), while
        ``content`` carries the rendered triple so a retrieved feature reads as
        one."""
        return MemoryOp(
            op=OpType.ADD,
            target_type="semantic",
            target_id=new_id(),
            payload={
                "kind": PROFILE_FEATURE_KIND,
                "category": category,
                "tag": tag,
                "feature": feature,
                "value": value,
                "content": f"[{tag}] {feature}: {value}",
                "embedding_text": value,
                "source_episode_ids": [episode.id],
                "timestamp": episode.timestamp.isoformat(),
            },
        )

    # ---- consolidation ------------------------------------------------------

    def consolidate(self, ctx: OrganizerContext) -> list[MemoryOp]:
        """``_consolidate_set_memories_if_applicable``: one LLM call per TAG
        that has at least ``consolidation_threshold`` features.

        The threshold is the whole cost control — under it, this pass is a
        storage read and nothing else."""
        if ctx.llm is None:
            return []
        ops: list[MemoryOp] = []
        for category in self.categories:
            by_tag: dict[str, list[dict]] = {}
            for row in self._features(category.name, ctx):
                by_tag.setdefault(str(row.get("tag", "")), []).append(row)
            for tag, rows in by_tag.items():
                if self.consolidation_threshold > 0 and len(rows) < self.consolidation_threshold:
                    continue
                ops.extend(self._consolidate_tag(category, tag, rows, ctx))
        return ops

    def _consolidate_tag(
        self,
        category: SemanticCategory,
        tag: str,
        rows: list[dict],
        ctx: OrganizerContext,
    ) -> list[MemoryOp]:
        verdict = ctx.llm.call(
            "distill",
            json.dumps(
                [
                    {
                        "tag": row.get("tag"),
                        "feature": row.get("feature"),
                        "value": row.get("value"),
                        "metadata": {"id": row.get("id")},
                    }
                    for row in rows
                ],
                ensure_ascii=False,
            ),
            CONSOLIDATE_SCHEMA,
            required_keys=("keep_memories",),
            system=category.consolidation_prompt,
        )
        # `if consolidate_resp is None or consolidate_resp.keep_memories is None:
        # return` — a failed consolidation changes nothing, which is the one
        # place upstream is conservative and we keep it.
        if verdict is None or verdict.get("keep_memories") is None:
            logger.warning("memmachine_profile: consolidation dropped for tag %r", tag)
            return []
        merged = verdict.get("consolidated_memories")
        if merged is None and verdict.get("consolidate_memories") is not None:
            # Defect 1: the prompt's spelling. Upstream would read `None` here
            # and write nothing back while still deleting everything not kept.
            merged = verdict.get("consolidate_memories")
            logger.warning(
                "memmachine_profile: model used the prompt's `consolidate_memories` key, "
                "which upstream's parser ignores — reading it anyway"
            )
        keep = {str(item) for item in verdict.get("keep_memories") or []}

        ops: list[MemoryOp] = []
        citations: list[str] = []
        for row in rows:
            if str(row.get("id")) in keep:
                continue
            citations.extend(row.get("source_episode_ids") or [])
            ops.append(
                MemoryOp(
                    op=OpType.DELETE, target_type="semantic", target_id=str(row["id"]), payload={}
                )
            )
        # The merged feature inherits the union of the deleted features'
        # citations (`itertools.chain` over `memories_to_delete`), so provenance
        # survives the merge even though the rows do not.
        provenance = list(dict.fromkeys(citations))
        for entry in merged or []:
            if not isinstance(entry, dict) or not entry.get("value"):
                continue
            # `original_tag = memories[0].tag` — a consolidation that renames the
            # tag is reverted to the group's tag and logged.
            if entry.get("tag") and entry["tag"] != tag:
                logger.warning(
                    "memmachine_profile: consolidation changed tag %r -> %r; reverting",
                    tag,
                    entry["tag"],
                )
            op = MemoryOp(
                op=OpType.ADD,
                target_type="semantic",
                target_id=new_id(),
                payload={
                    "kind": PROFILE_FEATURE_KIND,
                    "category": category.name,
                    "tag": tag,
                    "feature": str(entry.get("feature", "")),
                    "value": str(entry["value"]),
                    "content": f"[{tag}] {entry.get('feature', '')}: {entry['value']}",
                    "embedding_text": str(entry["value"]),
                    "source_episode_ids": provenance,
                    "consolidated": True,
                },
            )
            ops.append(op)
        return ops
