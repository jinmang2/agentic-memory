"""MemoryOS organizer (arXiv:2506.06326, EMNLP'25 Oral) — compact port.

OS-style hierarchy: STM (fixed-size dialogue buffer) -> MTM (topic
segments with heat) -> LPM (user profile + knowledge). Heat =
alpha*N_visit + beta*L_interaction + gamma*R_recency; segments crossing
theta trigger profile/knowledge extraction and reset (paper §MTM).

``fidelity`` picks which upstream lineage the divergent constants come
from — ``"pypi"`` (default, the maintained library) or ``"eval"`` (the
harness that produced the paper's LoCoMo numbers). They differ in the heat
weights, whether recency is live or stored, the keyword-overlap formula, the
STM capacity and flush order, the merge-candidate scheme, the promotion call
partition and the knowledge-store shape; theta=0.6, tau=5.0, MTM capacity
2000 AND the eviction policy are shared — both upstream lineages evict by
LFU (``evict_lfu`` over an access counter: pypi ``mid_term.py:177``, eval
``mid_term_memory.py:120``). The paper's "segments with the lowest heat are
evicted" (§3.3) matches neither codebase; ``eviction="lowest_heat"`` keeps
that reading reachable as an explicit non-lineage option (docs/16 session
3). See ``MEMORYOS_PRESETS``, which also lists the five defects found in
the eval lineage and says which are reproduced.

The unit everything is counted in is the PAGE, not the message, because
that is upstream's unit and both constants depend on it: ``short_term.py``
holds a deque of ``add_qa_pair`` entries so ``short_term_capacity=10`` is
10 pages, and ``mid_term.py`` sets ``"L_interaction": len(processed_details)``
over pages. A page is one FULL ``{user_input, agent_response}`` exchange —
BOTH lineages receive formed pairs (pypi ``add_qa_pair``; eval builds them in
``main_loco_parse.process_conversation`` before any memory call). Because our
harness delivers the two halves as separate ``on_message`` calls, this
organizer PAIRS them itself (round-12 finding 1): an exchange stays open in
``_open`` until the other side's turn closes it (or the next opener turn /
an explicit drain does), and only CLOSED pages occupy STM slots — so at
``stm_capacity=1`` an exchange is never split across pages. Until the
2026-07-28 fix each HALF was its own page, which at the eval lineage's
capacity 1 made 100% of pages half-exchanges: heat accrued ~2x (L=1 per
half, ~1.6 per exchange vs upstream's beta*1 = 0.8 per full exchange), so
the tau=5 promotion fired at roughly half the content, and the per-page
chain/topic calls ran twice per exchange. Consecutive same-speaker turns
mirror upstream's pairing rule exactly (``main_loco_parse.py:169-183``): a
later non-opener turn OVERWRITES the exchange's response half — the earlier
reply drops out of the page (the raw episode stays stored by the facade) —
and each opener turn opens a fresh page, leaving the previous one without a
response. The pre-round-12 code appended instead of overwriting; that
deviation is gone.

Pages missing either half are DROPPED before MTM insertion — both lineages
filter ``if qa.get("user_input") and qa.get("agent_response")`` (pypi
``updater.py:104``, eval ``dynamic_update.py:126``), so upstream MTM holds
only full exchanges, and an all-incomplete batch makes no LLM calls at all
(the filter sits before page formation). ``keep_incomplete_pages=True`` is
this project's extension for the no-content-loss variant (and for chained
composition, whose single-key streams have no second half by construction);
the default reproduces the lineage drop.

Other deviations: single vector index instead of FAISS-per-tier. The
read-path N_visit feedback (paper §3.4) IS wired — ``on_retrieval`` bumps
n_visit/last_access on served ``pages`` hits, and the memoryos retrieval
config queries ``pages`` — so the heat loop is live, not missing (round-5
N1 restored it). It is merely *inert on ingest-then-eval benchmarks* such
as LoCoMo conv0: all writes (hence all promotion/eviction) complete during
ingest, so the retrieval-time bumps never feed back — N_visit effectively
stays 0 for the graded numbers, but by run *shape*, not by a missing
mechanism (2026-07-21 fidelity review, docs/10). Its granularity also
differs — see ``on_retrieval``. Eviction emits DELETE ops through the
evolution log (auditable; upstream's old silent-loss issue #65 has since
been fixed upstream too).

LPM (``_promote_to_lpm``) is upstream's three stores, not an append log
(docs/10 M1, closed 2026-07-27): one evolving user-profile DOCUMENT under a
fixed id, rewritten from its previous version and replaced wholesale
(``update_user_profile(merge=False)``), plus two knowledge stores —
user-private and assistant — whose shape is LINEAGE-SPLIT (round-12
findings 6-7): pypi keeps two ``deque(maxlen=knowledge_capacity)`` FIFOs
with a lowercase per-line "none" filter and a >=30-chars/"none" profile
guard, while the eval lineage keeps plain UNBOUNDED lists (no capacity
anywhere, ``eval/long_term_memory.py:9-10``), writes the profile
unconditionally (``main_loco_parse.py:53-57``), rejects only the exact
strings ""/"- None"/"- None." per entry (``long_term_memory.py:68-71``) and
stores assistant knowledge as ONE un-split blob (``main_loco_parse.py:67``).
Only pages not yet analysed feed the prompts, and the trigger inspects the
HOTTEST segment as upstream's heap-top peek does.

STM flush order is lineage-split too (round-12 finding 13): pypi flushes at
the START of the overflowing add (``memoryos.py:242-246`` — ``if is_full():
process(...)`` runs BEFORE ``add_qa_pair``), so resident STM sits at
``capacity`` pages and the oldest is evicted when the (capacity+1)-th
closes; the eval driver flushes AFTER its add (``main_loco_parse.py:241-243``),
which at its capacity 1 leaves STM empty after every exchange. Resident
pages are injected verbatim at QA time by ``recent_context()``, which is
what upstream's ``get_response`` does with ``short_term_memory.get_all()``.
Each evicted page also carries upstream's two dialogue-chain calls
(continuity + ``meta_info``, see ``_chain_meta``), and the eval lineage's
two-call profile update is ``profile_update="two_call"`` in
``MEMORYOS_PRESETS`` — pypi keeps the one-call shape that folds the old
profile into the analysis prompt.

Remaining gaps, tracked: upstream's ``Retriever`` runs its three channels
concurrently, which we do not (no behavioural difference, only wall-clock).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, new_id
from agmem.organizers.base import Organizer, OrganizerContext

logger = logging.getLogger("agmem.organizers.memoryos")

# No page_indexes: upstream never partitions the batch across themes — see
# TOPIC_PROMPT and the whole-batch insertion in _evict_to_mtm (round-12
# finding 5).
TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "summary": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["topic", "summary"],
            },
        }
    },
    "required": ["groups"],
}

# The user profile is a single evolving document, so it lives under one stable
# id and every LPM update replaces it (upstream update_user_profile(merge=False)).
PROFILE_ITEM_ID = "memoryos:user_profile"

# Two upstream lineages, and they are NOT interchangeable: `memoryos-pypi/` is
# the maintained library, while `eval/` is the harness that produced the paper's
# LoCoMo numbers. Provenance is per-source and never mixed within a preset
# (same rule as NEMORI_PRESETS).
#
# Defects in the eval lineage, reproduced deliberately where they are part of
# the operating point and flagged where they are not (A-Mem precedent — a
# reproduction must be able to reach the published configuration verbatim):
#
# E1. `insert_pages_into_session` credits the WRONG session. Its trailing
#     `session["L_interaction"] += len(pages)` sits OUTSIDE the merge branch, so
#     when the score misses the threshold and a NEW session is created instead,
#     the rejected candidate's heat still grows by the page count it never
#     received. `memoryos-pypi` does not have this. NOT reproduced: it credits
#     heat to a segment that holds none of the content, which would corrupt the
#     promotion order rather than reproduce it.
# E2. R_recency is dead — twice over. `compute_segment_heat` reads a STORED
#     `R_recency` (initialised 1.0, refreshed only on a retrieval hit) and
#     weights it gamma=1e-4 — four orders of magnitude under one interaction,
#     so it cannot change any comparison. And the refresh itself is vacuous:
#     `mid_term_memory.py:236-238` stamps `last_visit_time` to now BEFORE
#     computing the decay from it, so the stored value is re-set to 1.0 on
#     every hit and never once decays. Reproduced (see `on_retrieval`): it IS
#     the published operating point, and `recency="stored"` makes the deadness
#     visible instead of assumed.
# E3. Page embeddings differ by code path. `add_session` embeds
#     `f"User: {user_input} Assiant: {agent_response}"` (typo upstream's), while
#     the merge branch embeds `f"用户: {user_input}"` — different language prefix
#     and the assistant turn dropped entirely. A STORAGE inconsistency only:
#     the eval READ path never reads stored page embeddings — it re-embeds
#     `f"{user_input}{timestamp}{agent_response}"` fresh per page per query
#     (`mid_term_memory.py:227-230`) — so E3 is retrieval-inert in the eval
#     lineage; only pypi dots the stored vector (`mid_term.py:335-338`, and
#     pypi's stored text has no typo). NOT reproduced (we keep one embedding
#     text) — it is an inconsistency, not a setting. (Round-12 finding 10
#     corrected this entry's earlier claim that page retrieval "depends on
#     which path stored the page" — for the published eval numbers it cannot.)
# E4. The merge branch stamps every inserted page with the SEGMENT's keywords
#     instead of extracting per-page ones as `add_session` does. NOT reproduced.
# E5. `MidTermMemory.max_capacity` defaults to 7, but the eval driver passes
#     2000, so the class default is dead in the published run. We take 2000.
MEMORYOS_PRESETS: dict[str, dict] = {
    # memoryos-pypi/{memoryos,mid_term,short_term,long_term}.py
    "pypi": dict(
        stm_capacity=10,
        # pypi flushes BEFORE the overflowing add (memoryos.py:242-246):
        # resident STM = capacity, evict on the (capacity+1)-th page.
        stm_flush="before_add",
        heat_weights=(1.0, 1.0, 1.0),
        recency="live",
        keyword_similarity="jaccard",
        # Both lineages call evict_lfu at capacity; "lowest_heat" exists only
        # as the paper's sentence (§3.3) and belongs to no code lineage — pass
        # eviction="lowest_heat" explicitly to measure the paper's reading
        # (docs/16 session 3 fixed the earlier mislabel here).
        eviction="lfu",
        # one analysis call that already sees the previous profile
        profile_update="single",
        # merge candidate = argmax of cos+Jaccard over ALL sessions
        # (mid_term.py:206-226) — no first-stage cosine cut.
        merge_candidates="scan_all",
        # long_term.py:18-19: two deque(maxlen=knowledge_capacity) FIFOs
        knowledge_fifo=True,
        # memoryos.py:186-188: skip the profile write when the model returned
        # "none" or under 30 chars
        profile_guard=True,
        # memoryos.py:196-204: lowercase per-line none/- none/- none. filter
        knowledge_line_filter="pypi",
        # assistant knowledge split per line, one entry each
        assistant_blob=False,
    ),
    # eval/{main_loco_parse,dynamic_update,mid_term_memory,short_term_memory}.py
    "eval": dict(
        stm_capacity=1,
        # the eval driver flushes AFTER its add (main_loco_parse.py:241-243);
        # at its capacity 1 STM is empty after every exchange.
        stm_flush="after_add",
        heat_weights=(0.8, 0.8, 0.0001),
        recency="stored",
        keyword_similarity="containment_mean",
        eviction="lfu",
        # analysis WITHOUT the old profile, then a separate merge call
        profile_update="two_call",
        # merge candidate = top-1 by cosine alone, then threshold cos +
        # containment-mean for that ONE candidate (mid_term_memory.py:133-154);
        # an all-negative cosine field (`best_sim >= 0` guard) creates a new
        # session without any keyword check.
        merge_candidates="cosine_top1",
        # eval/long_term_memory.py:9-10: plain unbounded lists — no capacity
        # exists anywhere in the eval lineage (round-12 finding 6)
        knowledge_fifo=False,
        # main_loco_parse.py:53-57: profile written unconditionally
        profile_guard=False,
        # long_term_memory.py:68-71: exact, case-sensitive ""/"- None"/"- None."
        # rejection — a bare "None" or lowercase "- none" IS stored upstream
        knowledge_line_filter="eval",
        # main_loco_parse.py:67: assistant knowledge stored as one un-split blob
        assistant_blob=True,
    ),
}

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {"profile": {"type": "string"}},
    "required": ["profile"],
}

KNOWLEDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "private": {"type": "array", "items": {"type": "string"}},
        "assistant_knowledge": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["private"],
}

# Condensed from the lineages' multi-summary prompts. The "maximum of two
# themes" cap is upstream's own and IDENTICAL in both lineages (pypi
# `prompts.py:73-74` "with a maximum of two themes" / system "No more than two
# topics"; eval `utils.py:128-133` the same two phrases), so one wording serves
# both presets. No page indexes are requested: upstream never partitions the
# batch across themes (see _evict_to_mtm, round-12 finding 5).
TOPIC_PROMPT = """Analyze the following dialogue batch and generate subtopic summaries
(if applicable), with a maximum of two themes. Each summary needs the
subtopic name, keywords, and the summary text.

Exchanges:
{messages}

Return JSON: {{"groups": [{{"topic": "short label", "summary": "2-3 sentence summary \
covering the concrete facts", "keywords": ["theme keyword", ...]}}]}}"""

# Condensed from PERSONALITY_ANALYSIS_SYSTEM/USER_PROMPT (pypi). The output is
# the WHOLE updated profile, not a delta — upstream feeds the result straight to
# update_user_profile(merge=False). The dimension list is upstream's framing
# ("the 90 personality preference dimensions"); its psychological block is
# reproduced and the remaining dimensions are left to the model, since the
# release's own list is a long free-text enumeration rather than a fixed schema.
PROFILE_PROMPT = """You are a professional user preference analysis assistant. Analyze the
user's personality preferences from the conversation across personality
preference dimensions, including the psychological model: Extraversion
(preference for social activities), Openness (willingness to embrace new
ideas and experiences), Agreeableness (tendency to be friendly and
cooperative), Conscientiousness (responsibility and organizational ability),
Neuroticism (emotional stability and sensitivity).

For each dimension:
1. Read the conversation and determine whether the dimension is reflected.
2. If reflected, give the preference level (High / Medium / Low) and brief
   reasoning, including time, people, and context where possible.
3. If it is not reflected, do not list it.

Focus only on the user's preferences and traits.

Existing User Profile:
{existing_user_profile}

Latest User-AI Conversation:
{conversation}

Return the COMPREHENSIVE UPDATED profile — combining the existing profile with
what the new conversation adds, without redundancy — as JSON:
{{"profile": "the full updated user profile"}}"""

CONTINUITY_SCHEMA = {
    "type": "object",
    "properties": {"continuous": {"type": "boolean"}},
    "required": ["continuous"],
}

META_INFO_SCHEMA = {
    "type": "object",
    "properties": {"meta_info": {"type": "string"}},
    "required": ["meta_info"],
}

# Upstream CONTINUITY_CHECK_{SYSTEM,USER}_PROMPT, merged into one turn. Upstream
# asks for a bare "true"/"false" string; the one-field object is this project's
# structured-output adaptation, not a semantic change.
CONTINUITY_PROMPT = """You are a conversation continuity detector. Determine if these two
conversation pages are continuous (a true continuation without a topic
shift).

Previous Page:
{previous}

Current Page:
{current}

Continuous? Return JSON: {{"continuous": true}} or {{"continuous": false}}"""

# Upstream META_INFO_{SYSTEM,USER}_PROMPT.
META_INFO_PROMPT = """Update the conversation meta-summary by incorporating the new dialogue
while maintaining continuity.

Guidelines:
1. Start from the previous meta-summary (if it exists)
2. Add/update information based on the new dialogue
3. Keep it concise (1-2 sentences max)
4. Maintain context coherence

Previous Meta-summary: {last_meta}
New Dialogue:
{new_dialogue}

Return JSON: {{"meta_info": "the updated meta-summary"}}"""

# The eval lineage's live merge prompt, ported from its inline "Profile Merge
# Task" (`eval/utils.py:301-350`): consolidation rules, the conflict-resolution
# hierarchy, change timestamps, the 4-category structure and the 1500-word cap.
# Provenance note (round-12 finding 8): the text that used to sit here was
# pypi's UPDATE_PROFILE_{SYSTEM,USER}_PROMPT, which is DEAD upstream —
# `gpt_update_profile` is defined in pypi `utils.py:340` but never called on
# any pypi live path — so the eval preset's second call was running a prompt
# no lineage's run ever executed. This is the prompt the published LoCoMo
# numbers' merge call actually used.
PROFILE_MERGE_PROMPT = """You are a profile integration system. Your rules:
1. NEVER discard verified information
2. Conflict resolution hierarchy:
   Explicit statement > Implied trait > Assumption
3. Add timestamps when traits change:
   (Updated: [date]) for modified traits
4. Preserve the 4-category structure

# Profile Merge Task
Consolidate these profiles while:
- Preserving all valid observations
- Resolving conflicts
- Adding new dimensions

## Current Profile
{old_profile}

## New Data
{new_analysis}

## Rules
1. Keep ALL verified traits from both
2. Resolve conflicts by:
   a) New explicit evidence > old assumptions
   b) Mark as Neutral if contradictory
3. Add new dimensions from new data
4. Maintain EXACT original format

Output ONLY the merged profile (no commentary).
The generated content should not exceed 1500 words.

Return JSON: {{"profile": "the merged profile"}}"""

# pypi's knowledge-extraction call (condensed from
# KNOWLEDGE_EXTRACTION_SYSTEM/USER_PROMPT). Two separate outputs because pypi
# stores them in two different deques (user vs assistant LTM).
KNOWLEDGE_PROMPT = """You are a knowledge extraction assistant. Extract user private data
and assistant knowledge from the conversation below. Be extremely concise and
factual — use the shortest possible phrases.

1. private: personal information, preferences, or private facts about the
   user, each as "[brief fact]: [minimal context, including entities and time]".
2. assistant_knowledge: explicit statements about what the assistant did,
   provided, or demonstrated, each as "Assistant [brief action] at [time]".

Return an empty list for either when nothing qualifies.

Latest User-AI Conversation:
{conversation}

Return JSON: {{"private": ["..."], "assistant_knowledge": ["..."]}}"""

# The eval lineage's promotion analysis call, ported from
# `gpt_personality_analysis` (`eval/utils.py:238-299`): profile AND user data
# come out of ONE call — upstream splits the reply on the 【User Data】 section
# marker; the two JSON fields are this project's structured-output adaptation
# of that split (round-12 finding 9). Note the prompt never sees the old
# profile — revision happens only in the separate merge call.
EVAL_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "profile": {"type": "string"},
        "private": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["profile", "private"],
}

EVAL_ANALYSIS_PROMPT = """You are a personality and user data analysis engine. Rules:
1. Extract ONLY observable traits and data with direct evidence.
2. Include general user data such as events, dates, locations, and preferences.
3. Use concise and factual statements.
4. If no relevant information is found, output "None".

# Personality and User Data Analysis Task
Analyze the conversation and produce:

profile — the user profile in EXACTLY this format:
1. Core Psychological Traits:
   - [Trait]: [Positive/Negative/Neutral] (Evidence)
   - (Max 5 most prominent traits)
2. Content Preferences:
   - [Topic]: [Like/Dislike/Neutral] (Evidence)
   - (Max 5 strongest preferences)
3. Interaction Style:
   - [Style]: [Preference] (Evidence)
   - (e.g., Direct/Indirect, Detailed/Concise)
4. Value Alignment:
   - [Value]: [Strong/Weak] (Evidence)
   - (e.g., Honesty, Helpfulness)

private — user data facts, each as "[Fact]: [Details]" (e.g., "User mentioned
visiting a park on April 1st, 2025 in New York."). Include events, dates,
locations, preferences, or other general or private information explicitly
mentioned in the conversation. If none, write "None."

Conversation:
{conversation}

Return JSON: {{"profile": "...", "private": ["fact", ...]}}"""

# The eval lineage's separate assistant-knowledge call, ported from
# `analyze_assistant_knowledge` (`eval/utils.py:49-103`) with its few-shot
# examples. The output is ONE text blob — upstream stores the whole reply as a
# single entry (`main_loco_parse.py:67`), not one entry per line.
EVAL_ASSISTANT_SCHEMA = {
    "type": "object",
    "properties": {"assistant_knowledge": {"type": "string"}},
    "required": ["assistant_knowledge"],
}

EVAL_ASSISTANT_PROMPT = """You are an assistant knowledge extraction engine. Rules:
1. Extract ONLY explicit statements about the assistant's identity or knowledge.
2. Use concise and factual statements in the first person.
3. If no relevant information is found, output "None".

# Assistant Knowledge Extraction Task
Analyze the conversation and extract any fact or identity traits about the
assistant. If no traits can be extracted, reply with "None". The generated
content should be as concise as possible — the more concise, the better.
Format the extracted facts as "- [Fact]" lines.

Few-shot examples:
1. User: Can you recommend some movies.
   AI: Yes, I recommend Interstellar.
   Time: 2023-10-01
   -> "- I recommend Interstellar on 2023-10-01."

2. User: Can you help me with cooking recipes?
   AI: Yes, I have extensive knowledge of cooking recipes and techniques.
   Time: 2023-10-02
   -> "- I have cooking recipes and techniques on 2023-10-02."

3. User: That's interesting. I didn't know you could do that.
   AI: I'm glad you find it interesting!
   -> "None"

Conversation:
{conversation}

Return JSON: {{"assistant_knowledge": "- fact lines, or None"}}"""


class MemoryOSOrganizer(Organizer):
    """MemoryOS STM/MTM/LPM organizer (arXiv:2506.06326; see module
    docstring for the paper-vs-eval-core constant discrepancy and other
    upstream deviations). Writes only via returned MemoryOps; STM/heat
    state lives in this instance, not the stores."""

    name = "memoryos"

    produces = ("pages", "semantic")

    def __init__(
        self,
        fidelity: str | None = None,
        stm_capacity: int | None = None,
        mtm_capacity: int = 2000,
        heat_threshold: float = 5.0,
        similarity_threshold: float = 0.6,
        recency_tau_hours: float = 24.0,
        knowledge_capacity: int = 100,
        heat_weights: tuple[float, float, float] | None = None,
        recency: str | None = None,
        keyword_similarity: str | None = None,
        eviction: str | None = None,
        profile_update: str | None = None,
        merge_candidates: str | None = None,
        stm_flush: str | None = None,
        knowledge_fifo: bool | None = None,
        profile_guard: bool | None = None,
        knowledge_line_filter: str | None = None,
        assistant_blob: bool | None = None,
        keep_incomplete_pages: bool = False,
        dialogue_chain: bool = True,
        flush_stm_on_drain: bool = False,
    ) -> None:
        """``fidelity`` selects an upstream lineage from ``MEMORYOS_PRESETS``
        ("pypi" when None/unknown, keeping the no-arg constructor stable); every
        other explicit (non-None) argument overrides that preset's value, same
        rule as ``NemoriOrganizer``.

        ``stm_capacity``: STM size in PAGES (upstream's unit — one page is one
        full exchange, paired by ``on_message``); pypi's 10 is 10 exchanges
        (~20 messages of an alternating dialogue), and the eval lineage's 1
        flushes every completed exchange.
        ``mtm_capacity``: max MTM segments before eviction (2000 in both
        lineages — the eval class default of 7 is overridden by its own driver).
        ``heat_threshold``: heat (τ) at which a segment promotes to LPM (5.0 in
        both). ``similarity_threshold``: F_score (θ) merge threshold, cosine +
        keyword overlap, paper eq.(3) (0.6 in both).
        ``recency_tau_hours``: recency-decay time constant.
        ``knowledge_capacity``: max entries in EACH LPM knowledge FIFO
        (upstream pypi ``long_term_knowledge_capacity``); read only when
        ``knowledge_fifo`` — the eval lineage's stores are unbounded.
        ``heat_weights``/``recency``/``keyword_similarity``/``eviction``/
        ``profile_update``/``merge_candidates``/``stm_flush``/
        ``knowledge_fifo``/``profile_guard``/``knowledge_line_filter``/
        ``assistant_blob`` are the lineage-divergent knobs; see
        ``MEMORYOS_PRESETS``.

        ``keep_incomplete_pages`` is this project's extension, NOT a lineage
        knob: both lineages DROP pages missing either half before MTM
        insertion (pypi ``updater.py:104``, eval ``dynamic_update.py:126``),
        so the fidelity default is the drop; True keeps them for the
        framework's no-content-loss variant (and for chained composition,
        whose single-key streams never form a second half).

        ``dialogue_chain`` runs upstream's per-page continuity check and
        conversation-chain meta-summary (2 LLM calls per evicted page). It is on
        because upstream does it unconditionally, and it is a knob because it is
        the single largest term in this organizer's write cost — the numbers in
        docs/09 were produced without it, so switching it off is how those runs
        stay reproducible."""
        params = dict(MEMORYOS_PRESETS.get(fidelity, MEMORYOS_PRESETS["pypi"]))
        overrides = dict(
            stm_capacity=stm_capacity,
            heat_weights=heat_weights,
            recency=recency,
            keyword_similarity=keyword_similarity,
            eviction=eviction,
            profile_update=profile_update,
            merge_candidates=merge_candidates,
            stm_flush=stm_flush,
            knowledge_fifo=knowledge_fifo,
            profile_guard=profile_guard,
            knowledge_line_filter=knowledge_line_filter,
            assistant_blob=assistant_blob,
        )
        params.update({k: v for k, v in overrides.items() if v is not None})
        params.update(
            dialogue_chain=dialogue_chain,
            flush_stm_on_drain=flush_stm_on_drain,
            keep_incomplete_pages=keep_incomplete_pages,
            mtm_capacity=mtm_capacity,
            heat_threshold=heat_threshold,
            similarity_threshold=similarity_threshold,
            recency_tau_hours=recency_tau_hours,
            knowledge_capacity=knowledge_capacity,
        )
        self.fidelity = fidelity
        self.params = params  # stats/stamping surface, as NemoriOrganizer has

        # mtm_capacity 2000 = upstream default (round-5 N6 fixed 200->2000)
        self.stm_capacity = params["stm_capacity"]
        self.stm_flush = params["stm_flush"]
        self.mtm_capacity = mtm_capacity
        self.heat_threshold = heat_threshold
        self.similarity_threshold = similarity_threshold
        self.recency_tau_hours = recency_tau_hours
        self.knowledge_capacity = knowledge_capacity
        self.heat_weights = tuple(params["heat_weights"])
        self.recency = params["recency"]
        self.keyword_similarity = params["keyword_similarity"]
        self.eviction = params["eviction"]
        self.profile_update = params["profile_update"]
        self.merge_candidates = params["merge_candidates"]
        self.knowledge_fifo = params["knowledge_fifo"]
        self.profile_guard = params["profile_guard"]
        self.knowledge_line_filter = params["knowledge_line_filter"]
        self.assistant_blob = params["assistant_blob"]
        self.keep_incomplete_pages = keep_incomplete_pages
        self.dialogue_chain = params["dialogue_chain"]
        self.flush_stm_on_drain = params["flush_stm_on_drain"]
        # upstream eval's `access_frequency`: a retrieval counter that is NEVER
        # reset (unlike n_visit, which promotion zeroes), because LFU eviction
        # reads it. Kept separate for that reason.
        self._access: dict[str, int] = {}
        # STM state, in upstream's own unit: `_stm_pages` holds CLOSED pages
        # (full exchanges, or the half-exchanges upstream's offline pairing
        # also produces — consecutive-same-speaker leftovers), `_open` the
        # exchange still forming. Only closed pages occupy capacity slots, so
        # a lone user half never trips the flush (round-12 finding 1).
        self._stm_pages: list[list[Episode]] = []
        self._open: list[Episode] = []
        self._heat: dict[str, dict] = {}  # segment_id -> {n_visit, length, last_access}
        # Reverse indexes track which STM units back which MTM pages. Heat
        # eviction uses them to drop dead index entries; ``retire``/``patch_unit``
        # use them when this organizer is driven by ChainedConsumer over another
        # organizer's episodes (experimental) so a supersedes chain can retire
        # units and invalidate fully-absorbed pages. In-memory only — volatile
        # across restarts, same as ``_heat``.
        self._page_sources: dict[str, set[str]] = {}  # page_id -> {unit_id, ...}
        self._unit_pages: dict[str, set[str]] = {}  # unit_id -> {page_id, ...}
        # Whichever speaker/role opened the stream — upstream's `speaker_a`,
        # fixed for the life of the conversation (one organizer instance is one
        # conversation, as the experiment configs' 0-arg factories ensure).
        self._page_opener: str | None = None
        # LPM bookkeeping, in-memory like _heat: unit ids already folded into
        # the profile (upstream's per-page `analyzed` flag), and the insertion
        # order of each knowledge store so the pypi FIFO can evict the oldest.
        self._analyzed: set[str] = set()
        self._knowledge: dict[str, list[str]] = {}
        # Dialogue-chain state (upstream Updater.last_evicted_page_for_continuity
        # and the meta_info it carries forward): the previously evicted page and
        # the running chain summary, so the continuity judgment and the
        # meta-summary of the next page have something to continue FROM.
        self._last_page: list[Episode] = []
        self._last_meta: str = ""

    # -- pages -----------------------------------------------------------------

    def _page_key(self, episode: Episode) -> str:
        """Side of the exchange this message is on.

        ``meta["speaker"]`` when the ingest supplies one — LoCoMo is a two-named-
        speaker dialogue and its ``add_message`` calls all use ``role="user"``,
        so ``role`` alone cannot separate the sides — else ``role``, which is
        informative for LongMemEval and live traffic."""
        return str(episode.meta.get("speaker") or episode.role)

    def _page_complete(self, page: list[Episode]) -> bool:
        """Upstream's MTM admission predicate, in our representation: a page is
        complete when BOTH halves are present — pypi ``updater.py:104`` and
        eval ``dynamic_update.py:126`` filter ``if qa.get("user_input") and
        qa.get("agent_response")``, so a page missing either side reaches MTM
        in neither lineage."""
        sides = {self._page_key(e) == self._page_opener for e in page}
        return sides == {True, False}

    def _pages(self, episodes: list[Episode]) -> list[list[Episode]]:
        """Regroup stored episodes into their exchanges for prompt rendering
        (``_promote_to_lpm``'s conversation text): an opener turn starts a
        page, the other side's turn fills it — the same rule ``on_message``
        pairs the live stream with. Rendering-only; the live STM pairing is
        stateful and lives in ``on_message``/``_open``."""
        pages: list[list[Episode]] = []
        for episode in episodes:
            key = self._page_key(episode)
            if self._page_opener is None:
                self._page_opener = key
            if key == self._page_opener or not pages:
                pages.append([episode])
            else:
                pages[-1].append(episode)
        return pages

    # -- heat ------------------------------------------------------------------

    def _segment_heat(self, segment_id: str) -> float:
        """H = alpha*N_visit + beta*L_interaction + gamma*R_recency.

        ``recency="live"`` recomputes R from the elapsed time on every read, as
        ``memoryos-pypi``'s ``compute_segment_heat`` does via
        ``compute_time_decay``. ``recency="stored"`` reads the field the segment
        carries, which the eval lineage initialises to 1.0 and refreshes only on
        a retrieval hit — combined with its gamma=1e-4 that makes R unable to
        change any comparison (module docstring E2). The mode is explicit rather
        than implied by the weight so the deadness is visible in the config."""
        h = self._heat.get(segment_id)
        if not h:
            return 0.0
        alpha, beta, gamma = self.heat_weights
        if self.recency == "stored":
            recency = h.get("recency", 1.0)
        else:
            hours = (datetime.now(timezone.utc) - h["last_access"]).total_seconds() / 3600
            recency = math.exp(-hours / self.recency_tau_hours)
        return alpha * h["n_visit"] + beta * h["length"] + gamma * recency

    def _keyword_overlap(self, a: set[str], b: set[str]) -> float:
        """Keyword term of F_score — the two lineages compute it differently.

        pypi: Jaccard, ``intersection / union``.
        eval: the mean of the two containment ratios,
        ``0.5 * (|A&B|/|A| + |A&B|/|B|)`` — symmetric like Jaccard but strictly
        larger whenever the sets differ in size, so the same theta=0.6 merges
        more readily under the eval lineage."""
        if not a or not b:
            return 0.0
        overlap = len(a & b)
        if not overlap:
            return 0.0
        if self.keyword_similarity == "containment_mean":
            return 0.5 * (overlap / len(a) + overlap / len(b))
        return overlap / len(a | b)

    # -- hooks -------------------------------------------------------------------

    def on_retrieval(
        self, hits: list[tuple[str, str, float]], ctx: OrganizerContext
    ) -> list[MemoryOp]:
        """Bumps ``n_visit``/``last_access`` on served page hits (paper's
        heat-feedback loop, §3.4); always returns [] — heat lives in
        ``self._heat``, no store writes.

        Granularity divergence, disclosed (round-12 finding 12): upstream bumps
        each SESSION once per query when ANY page in it matches, and does so
        BEFORE the retrieval-queue cut (pypi ``mid_term.py:344-351``, eval
        ``mid_term_memory.py:234-240``). This hook bumps once per SERVED page
        AFTER the cut — a multi-page serve from one segment inflates its
        counters, and a session whose matches were all cut from the queue gets
        nothing. Behavior kept as-is: on ingest-then-eval runs the whole loop
        is inert (see the module docstring), so the divergence cannot reach a
        graded number there; live-traffic shapes would differ."""
        # upstream mid_term.py updates N_visit/last_visit_time on every
        # retrieval hit (paper §3.4) — the heat feedback loop round-5 N1
        # found missing. No ops needed: heat lives in organizer state.
        now = datetime.now(timezone.utc)
        for item_id, memory_type, _score in hits:
            if memory_type != "pages":
                continue
            # The served id is the SEGMENT's when the summary itself is served,
            # and the page's leading message id once `MemoryOSPageRecall` has
            # expanded the segment into its pages — which is the normal path
            # now. Upstream bumps the SESSION either way (`search_sessions`
            # updates N_visit for every session with a matched page), so a page
            # id is resolved back through the unit -> segment index. Without
            # this the two-stage read path would silently kill the feedback
            # loop round-5 N1 restored.
            for segment_id in {item_id} | self._unit_pages.get(item_id, set()):
                if segment_id not in self._heat:
                    continue
                h = self._heat[segment_id]
                h["n_visit"] += 1
                h["last_access"] = now
                # Pinned at 1.0 by upstream's own construction, mirrored
                # deliberately: eval `mid_term_memory.py:236-238` stamps
                # `last_visit_time = get_timestamp()` FIRST and then computes
                # `compute_time_decay(session["last_visit_time"], get_timestamp())`
                # — zero elapsed time, so the stored R never leaves its initial
                # 1.0. This line used to transcribe that order literally
                # (`exp(-(now - last_access))` with `last_access` just set to
                # `now`), which computed the same constant while looking like a
                # decay; written as the constant it is so the mirrored defect is
                # visible. Module docstring E2 already prices the field as dead
                # (gamma=1e-4); `recency="live"` is the variant that decays.
                h["recency"] = 1.0
                # `access_frequency`, the LFU counter — never reset by promotion.
                # Keyed on the SEGMENT, like the heat entry beside it: eviction
                # picks a segment, so a page id here would build a counter no
                # eviction ever reads.
                self._access[segment_id] = self._access.get(segment_id, 0) + 1
        return []

    def on_message(self, episode: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        """Pairs the message into upstream's page unit and, when a page CLOSES,
        runs the STM flush check in the active lineage's order (empty list
        otherwise).

        Pairing transcribed from upstream's own LoCoMo driver
        (``main_loco_parse.py:169-183``), which forms the pages BEFORE any
        memory call — both lineages' STMs only ever see formed
        ``{user_input, agent_response}`` pairs:

        - an opener turn always opens a fresh exchange, closing whatever was
          still forming (a second consecutive opener turn therefore leaves a
          page with no response — upstream's ``agent_response: ""`` entry);
        - the other side's turn completes the open exchange, closing it;
        - a non-opener turn with NO open exchange OVERWRITES the response half
          of the newest resident page (upstream ``processed[-1]
          ["agent_response"] = text`` — the earlier reply drops out of the
          page; the raw episode stays stored by the facade). When that page
          has already left STM (possible at ``stm_capacity=1``, where the
          exchange evicts as soon as it completes — a timing upstream's
          offline pairing cannot hit), the turn becomes upstream's
          ``{user_input: "", agent_response: text}`` orphan instead.

        Only CLOSED pages count toward ``stm_capacity``: the open exchange is
        not a page yet, so at capacity 1 an exchange is never split across
        pages (round-12 finding 1 — the pre-fix code paged each half
        separately). The flush itself moves one page per trigger, as
        upstream's ``while is_full(): pop_oldest()`` pops exactly one at
        capacity; its ORDER relative to the add is the lineage-split
        ``stm_flush`` (see ``_close_page``).

        Chained use (driving this from another organizer's episodes) is an
        experimental composition handled by
        ``organizers.experimental.ChainedConsumer``, which calls this same
        entry point plus ``retire``/``patch_unit`` — those single-key streams
        never form a response half, so they need
        ``keep_incomplete_pages=True`` to reach MTM at all."""
        key = self._page_key(episode)
        if self._page_opener is None:
            self._page_opener = key
        if key == self._page_opener:
            ops: list[MemoryOp] = []
            if self._open:
                ops = self._close_page(self._open, ctx)
            self._open = [episode]
            return ops
        if self._open:
            self._open.append(episode)
            page, self._open = self._open, []
            return self._close_page(page, ctx)
        if self._stm_pages:
            # upstream's overwrite (main_loco_parse.py:176-177): the newest
            # resident page's response half is REPLACED — code now aligned with
            # the rule; the earlier append-instead deviation is gone (round-12).
            last = self._stm_pages[-1]
            last[:] = [e for e in last if self._page_key(e) == self._page_opener] + [episode]
            return []
        return self._close_page([episode], ctx)

    def _close_page(self, page: list[Episode], ctx: OrganizerContext) -> list[MemoryOp]:
        """Admit one closed page to STM, flushing in the lineage's order.

        pypi flushes at the START of the overflowing add (``memoryos.py:
        242-246``: ``if is_full(): process_short_term_to_mid_term()`` runs
        BEFORE ``add_qa_pair``), so resident STM sits at ``capacity`` pages
        and the oldest is evicted when the (capacity+1)-th arrives — the
        pre-round-12 code evicted one page earlier and its docstring claimed
        "capacity - 1 resident" (finding 13). The eval driver flushes AFTER
        its add (``main_loco_parse.py:241-243``), which at its capacity 1
        drains the exchange immediately — its QA-time STM is empty."""
        ops: list[MemoryOp] = []
        if self.stm_flush == "after_add":
            self._stm_pages.append(page)
            while len(self._stm_pages) >= self.stm_capacity:
                ops.extend(self._evict_to_mtm([self._stm_pages.pop(0)], ctx))
        else:  # "before_add" (pypi)
            while len(self._stm_pages) >= self.stm_capacity:
                ops.extend(self._evict_to_mtm([self._stm_pages.pop(0)], ctx))
            self._stm_pages.append(page)
        return ops

    def patch_unit(self, unit: Episode) -> None:
        """In-place UPDATE of a unit still buffered in STM (used by
        ``ChainedConsumer`` when an upstream episode is revised before it has
        been paged). Once the unit has been paged, the update is ignored —
        documented staleness (spec §3)."""
        for page in self._stm_pages + ([self._open] if self._open else []):
            for i, e in enumerate(page):
                if e.id == unit.id:
                    page[i] = unit

    def _drop_page_index(self, page_id: str) -> None:
        """Remove page_id from the _page_sources/_unit_pages reverse
        indexes entirely — shared by heat-eviction (page gone, all its
        source links are dead) and retire (page invalidated, same
        thing). Without this, evicted/invalidated pages leak index
        entries and a later supersedes can make retire re-emit a stale
        INVALIDATE for a page that's already gone (review finding)."""
        for unit_id in self._page_sources.pop(page_id, ()):
            pages = self._unit_pages.get(unit_id)
            if pages is None:
                continue
            pages.discard(page_id)
            if not pages:
                self._unit_pages.pop(unit_id, None)

    def retire(self, superseded: set[str]) -> list[MemoryOp]:
        """Clean up derived state for absorbed units: drop them from STM;
        invalidate a page only once ALL of its sources are superseded
        (partial absorption leaves the page intact, spec §3). Called by
        ``ChainedConsumer`` when an upstream supersedes chain retires the
        episodes this organizer paged (experimental composition)."""
        ops: list[MemoryOp] = []
        self._stm_pages = [
            kept
            for page in self._stm_pages
            if (kept := [e for e in page if e.id not in superseded])
        ]
        self._open = [e for e in self._open if e.id not in superseded]
        for unit_id in superseded:
            for page_id in self._unit_pages.pop(unit_id, set()):
                source_ids = self._page_sources.get(page_id)
                if source_ids is None:
                    continue
                source_ids.discard(unit_id)
                if not source_ids:
                    self._drop_page_index(page_id)
                    self._heat.pop(page_id, None)
                    ops.append(
                        MemoryOp(
                            op=OpType.INVALIDATE,
                            target_type="pages",
                            target_id=page_id,
                            payload={"reason": "sources_superseded"},
                        )
                    )
        return ops

    def flush_buffer(self, ctx: OrganizerContext) -> list[MemoryOp]:
        """No-op by default: the STM tail is supposed to STAY in STM.

        Upstream never drains short-term memory — pages leave it one at a time
        as new ones arrive, and whatever is still resident is injected verbatim
        into the QA prompt (``get_response``: ``short_term_memory.get_all()`` ->
        ``history_text``). Draining it here would put the resident pages into
        MTM *and* the recent-context channel, which upstream does neither of.
        Nothing is lost by not draining: raw episodes are stored by the facade
        regardless, and ``recent_context`` serves them.

        ``flush_stm_on_drain=True`` restores the old force-drain, which is what
        the docs/09 runs did; keep it in mind when comparing against them. A
        drain also CLOSES the exchange still forming, so its half rides along
        (and is then subject to the incomplete-page drop unless
        ``keep_incomplete_pages``)."""
        if not self.flush_stm_on_drain or not (self._stm_pages or self._open):
            return []
        pages = self._stm_pages + ([self._open] if self._open else [])
        self._stm_pages, self._open = [], []
        return self._evict_to_mtm(pages, ctx)

    def recent_context(self) -> str:
        """The STM buffer as upstream renders it into the QA prompt.

        ``get_response`` builds ``history_text`` from ``short_term_memory
        .get_all()``, one line per page, and the LoCoMo driver
        (``main_loco_parse.generate_system_response_with_meta``) does the same
        with the two speakers' names substituted for User/Assistant. We keep the
        page's own speaker labels, which are those names when the ingest supplies
        them. The exchange still forming renders too — upstream's offline
        pairing would already have admitted it as a half-filled pair.

        Empty string when STM holds nothing, so a caller can inject it
        unconditionally."""
        lines = []
        for page in self._stm_pages + ([self._open] if self._open else []):
            body = "\n".join(f"{self._page_key(unit)}: {unit.content}" for unit in page)
            stamp = page[0].timestamp.isoformat() if page else ""
            lines.append(f"{body}\nTime: ({stamp})")
        return "\n".join(lines)

    def warm_start(self, corpus, ctx: OrganizerContext) -> list[MemoryOp]:
        """Replays ``corpus`` through ``on_message`` (base behavior), then
        flushes any leftover partial STM batch so no episode is left
        un-paged after warm start."""
        ops = super().warm_start(corpus, ctx)
        ops.extend(self.flush_buffer(ctx))
        return ops

    # -- dialogue chain (upstream Updater continuity + meta_info) -----------------

    def _page_text(self, page: list[Episode]) -> str:
        return "\n".join(f"{self._page_key(unit)}: {unit.content}" for unit in page)

    def _chain_meta(self, pages: list[list[Episode]], ctx: OrganizerContext) -> str:
        """Advance the conversation chain over the pages being evicted and return
        the newest chain summary.

        Upstream does two LLM calls per page (``updater.process_short_term_to_mid_term``):
        ``check_conversation_continuity`` against the previously evicted page,
        then ``generate_page_meta_info`` seeded with the previous summary when the
        pages are continuous and with nothing when they are not. A dropped call
        is treated as "not continuous" / "no new summary", which is also how
        upstream degrades — its continuity helper returns
        ``response.strip().lower() == "true"``, so anything unparseable is False.

        One deviation, in call count only: upstream asks the continuity question
        for the very first page too, against an empty previous page, and then
        throws the answer away (``if is_continuous and temp_last_page_in_batch``
        — the second operand is ``None``). We skip that call. Same behaviour,
        one fewer call per conversation, so a cost comparison against upstream
        should read this organizer's continuity count as pages - 1, not pages.

        Returns ONE summary rather than one per page because upstream propagates
        the newest summary over the whole chain
        (``_update_linked_pages_meta_info`` walks ``pre_page``/``next_page``
        setting ``page["meta_info"] = new_meta``), so every page in a chain
        carries the same value anyway. Our MTM item is the segment, not the page,
        so that value is stored on the segment the pages land in — the deviation
        is the granularity, not the content."""
        if not self.dialogue_chain or ctx.llm is None:
            return ""
        for page in pages:
            current = self._page_text(page)
            continuous = False
            if self._last_page:
                verdict = ctx.llm.call(
                    "judge",
                    CONTINUITY_PROMPT.format(
                        previous=self._page_text(self._last_page), current=current
                    ),
                    CONTINUITY_SCHEMA,
                    required_keys=("continuous",),
                )
                continuous = bool((verdict or {}).get("continuous"))
            result = ctx.llm.call(
                "distill",
                META_INFO_PROMPT.format(
                    last_meta=self._last_meta if continuous and self._last_meta else "None",
                    new_dialogue=current,
                ),
                META_INFO_SCHEMA,
                required_keys=("meta_info",),
            )
            new_meta = str((result or {}).get("meta_info", "")).strip()
            if new_meta:
                self._last_meta = new_meta
            self._last_page = page
        return self._last_meta

    # -- STM -> MTM ---------------------------------------------------------------

    def _merge_candidate(
        self,
        embedding: list[float],
        keywords: set[str],
        ctx: OrganizerContext,
        pending: dict[str, dict] | None = None,
    ) -> tuple[str | None, float]:
        """Pick the merge target for a new theme batch, per lineage
        (round-12 finding 3 removed the previous top-3-by-cosine hybrid,
        which belonged to neither lineage).

        ``merge_candidates="scan_all"`` (pypi ``mid_term.py:206-226``): argmax
        of ``cos + Jaccard`` over ALL live sessions — no first-stage cut.
        ``merge_candidates="cosine_top1"`` (eval ``mid_term_memory.py:133-154``):
        argmax by COSINE ALONE, then ``cos + containment-mean`` is computed for
        that single candidate and thresholded by the caller — a session outside
        the cosine top-1 can never merge no matter its keywords. Its
        ``best_sim >= 0`` guard is reproduced: an all-negative cosine field
        creates a new session without any keyword check.

        ``pending`` maps segment ids created EARLIER IN THIS FLUSH to their
        frozen {embedding, keywords}: upstream applies each theme's insert
        immediately, so theme two can merge into theme one's fresh session —
        our ops only reach the stores after the hook returns, so those
        candidates are scored from the local record instead."""
        pending = pending or {}
        cosines: dict[str, float] = {}
        stored_ids = [sid for sid in self._heat if sid not in pending]
        if stored_ids:
            vectors = ctx.vector_store.get(stored_ids)
            for sid in stored_ids:
                vec = vectors.get(sid)
                if vec:
                    cosines[sid] = _cosine(embedding, vec)
        for sid, entry in pending.items():
            cosines[sid] = _cosine(embedding, entry["embedding"])
        if not cosines:
            return None, 0.0

        def candidate_keywords(sid: str) -> set[str]:
            if sid in pending:
                return set(pending[sid]["keywords"])
            item = ctx.doc_store.get_items([sid], "pages")
            return set((item[0] if item else {}).get("keywords", []))

        if self.merge_candidates == "cosine_top1":
            best_sid = max(cosines, key=lambda sid: cosines[sid])
            if cosines[best_sid] < 0:
                return None, 0.0
            return best_sid, cosines[best_sid] + self._keyword_overlap(
                candidate_keywords(best_sid), keywords
            )
        best_sid, best_f = None, -math.inf
        for sid, cos in cosines.items():
            f = cos + self._keyword_overlap(candidate_keywords(sid), keywords)
            if f > best_f:
                best_sid, best_f = sid, f
        return best_sid, best_f

    def _evict_to_mtm(self, pages: list[list[Episode]], ctx: OrganizerContext) -> list[MemoryOp]:
        # The batch arrives as formed pages (on_message pairs the stream);
        # everything below — the prompt, the membership, and the heat's
        # L_interaction — is counted in pages (2026-07-27 audit B1).
        #
        # Incomplete pages drop FIRST, before any LLM call: both lineages
        # filter half-exchanges out ahead of page formation (pypi
        # updater.py:102-105, eval dynamic_update.py:124-127), and an
        # all-incomplete batch returns without calling anything (`if not
        # evicted: return`). `keep_incomplete_pages=True` is this project's
        # no-content-loss extension (round-12 finding 2).
        if not self.keep_incomplete_pages:
            pages = [p for p in pages if self._page_complete(p)]
        if not pages:
            return []
        batch = [e for page in pages for e in page]
        # Chain first: upstream forms pages (continuity + meta_info) BEFORE the
        # multi-topic summary, and the summary prompt then sees the same pages.
        meta_info = self._chain_meta(pages, ctx)
        page_unit_ids = [[e.id for e in page] for page in pages]
        if ctx.llm is None:
            logger.warning("memoryos: no LLM — storing mechanical segment (explicit degradation)")
            segment_id = new_id()
            content = "\n".join(e.content for e in batch)
            self._heat[segment_id] = {
                "n_visit": 0,
                "length": len(pages),
                "last_access": datetime.now(timezone.utc),
                "recency": 1.0,  # eval lineage stores R; 1.0 at creation
            }
            return [
                self._segment_add(
                    segment_id,
                    "batch",
                    content,
                    batch,
                    ctx,
                    page_units=page_unit_ids,
                )
            ]

        rendered = "\n".join(
            " | ".join(f"{self._page_key(e)}: {e.content}" for e in page) for page in pages
        )
        result = ctx.llm.call(
            "distill",
            TOPIC_PROMPT.format(messages=rendered),
            TOPIC_SCHEMA,
            required_keys=("groups",),
        )
        groups = (result or {}).get("groups") or [
            {"topic": "batch", "summary": "\n".join(e.content for e in batch), "keywords": []}
        ]

        ops: list[MemoryOp] = []
        # Segments created earlier in this same flush, so a later theme can
        # merge into them the way upstream (which applies each insert
        # immediately) can — see _merge_candidate.
        pending: dict[str, dict] = {}
        for g in groups:
            # Whole batch per theme (round-12 finding 5): BOTH lineages call
            # insert_pages_into_session once PER theme with the ENTIRE batch
            # (updater.py:180-185, dynamic_update.py:170-180) — a multi-theme
            # batch duplicates every page into each theme's target session, and
            # each target's L_interaction grows by len(all pages). The previous
            # page_indexes partition asked the model to split the batch, a
            # scheme neither lineage has.
            n_pages = len(pages)
            members = batch
            summary = str(g.get("summary", ""))
            keywords = [str(k).lower() for k in g.get("keywords") or []]
            embedding = ctx.embedder.embed([summary])[0]
            # F_score = cos + keyword overlap, threshold 0.6 — paper eq.(3);
            # candidate selection is lineage-split, see _merge_candidate.
            best_id, best_f = self._merge_candidate(embedding, set(keywords), ctx, pending)

            if best_id is not None and best_f >= self.similarity_threshold:
                segment_id = best_id  # merge into existing segment (F_score >= θ)
                h = self._heat[segment_id]
                # upstream: target_session["L_interaction"] += len(pages_to_insert)
                h["length"] += n_pages
                h["last_access"] = datetime.now(timezone.utc)
                for e in members:
                    self._unit_pages.setdefault(e.id, set()).add(segment_id)
                self._page_sources.setdefault(segment_id, set()).update(e.id for e in members)
                # Merge-key freeze (round-12 finding 4): BOTH lineages leave the
                # segment's summary, summary_embedding and summary_keywords
                # untouched on merge — only details/L change — so a segment's
                # matching identity is frozen at creation. Hence no content
                # append and no keyword union here, and ``embedding_text`` stays
                # out of the payload so the vector VALUE never moves. The
                # embedder CALL is still paid, though: ``_apply_one`` merges the
                # stored item back under the UPDATE, finds the original
                # ``embedding_text`` there, and re-embeds it to write back an
                # identical vector — one embedder call per merge, a real cost
                # (Zep's provenance-append UPDATEs pay the same way). The merged
                # theme's summary/keywords are simply discarded, as upstream
                # discards them.
                if segment_id in pending:
                    # target was created earlier in this flush — extend its ADD
                    # op in place rather than emitting an UPDATE the store
                    # could not resolve yet
                    payload = pending[segment_id]["op"].payload
                    payload["source_episode_ids"] = list(payload["source_episode_ids"]) + [
                        e.id for e in members
                    ]
                    payload["page_units"] = list(payload["page_units"]) + page_unit_ids
                    if meta_info:
                        payload["meta_info"] = meta_info
                else:
                    existing = ctx.doc_store.get_items([segment_id], "pages")
                    old = existing[0] if existing else {}
                    ops.append(
                        MemoryOp(
                            op=OpType.UPDATE,
                            target_type="pages",
                            target_id=segment_id,
                            payload={
                                "source_episode_ids": list(old.get("source_episode_ids", []))
                                + [e.id for e in members],
                                "page_units": list(old.get("page_units", [])) + page_unit_ids,
                                # newest chain summary wins, as upstream's
                                # _update_linked_pages_meta_info overwrites the
                                # chain (page-granularity stand-in, see
                                # _chain_meta)
                                **({"meta_info": meta_info} if meta_info else {}),
                            },
                        )
                    )
            else:
                segment_id = new_id()
                self._heat[segment_id] = {
                    "n_visit": 0,
                    # upstream: "L_interaction": len(processed_details), pages
                    "length": n_pages,
                    "last_access": datetime.now(timezone.utc),
                    "recency": 1.0,  # eval lineage stores R; 1.0 at creation
                }
                op = self._segment_add(
                    segment_id,
                    str(g.get("topic", "?")),
                    summary,
                    members,
                    ctx,
                    keywords,
                    meta_info,
                    page_unit_ids,
                )
                ops.append(op)
                # frozen identity for later themes in this flush: the recorded
                # embedding/keywords never change on merge (finding 4)
                pending[segment_id] = {
                    "embedding": embedding,
                    "keywords": set(keywords),
                    "op": op,
                }

        # heat >= τ -> promote to LPM (profile/knowledge), then reset.
        # The HOTTEST segment, once, not each segment this flush happened to
        # touch: upstream's `_trigger_profile_and_knowledge_update_if_needed`
        # peeks at the heap top (`self.mid_term_memory.heap[0]`) and analyses
        # only that one. Checking per-group promoted whichever segment was
        # written last rather than whichever is actually hot.
        # Cadence differs and is harmless: upstream checks after every page,
        # we check once per STM flush. Heat only moves when pages reach MTM
        # (recency decay and N_visit aside), so the extra upstream checks
        # between flushes cannot find a newly-hot segment — the one case that
        # does differ is several segments crossing tau in the same flush, where
        # upstream would work through them over the following adds.
        if self._heat:
            hottest = max(self._heat, key=self._segment_heat)
            if self._segment_heat(hottest) >= self.heat_threshold:
                ops.extend(self._promote_to_lpm(hottest, ctx))

        # Eviction when MTM is over capacity. Two policies, and the paper and
        # the code disagree about which one MemoryOS has:
        #   "lowest_heat" — the paper's description (§3.3); no code lineage
        #                   implements it, so only an explicit kwarg selects it.
        #   "lfu"         — what both upstream codebases actually do
        #                   (`evict_lfu`, min over an access counter); both
        #                   presets use this since the docs/16 label fix.
        # Worth knowing before reading an LFU run: the counter only moves on a
        # retrieval hit, so on an ingest-then-eval benchmark every segment sits
        # at 0 during ingest and `min` returns whichever key came first —
        # eviction degrades to insertion-order FIFO, not least-frequently-used.
        while len(self._heat) > self.mtm_capacity:
            if self.eviction == "lfu":
                coldest = min(self._heat, key=lambda sid: self._access.get(sid, 0))
            else:
                coldest = min(self._heat, key=self._segment_heat)
            self._heat.pop(coldest)
            self._access.pop(coldest, None)
            self._drop_page_index(coldest)
            ops.append(
                MemoryOp(
                    op=OpType.DELETE,
                    target_type="pages",
                    target_id=coldest,
                    payload={"reason": f"{self.eviction}_eviction"},
                )
            )
        return ops

    def _segment_add(
        self,
        segment_id: str,
        topic: str,
        content: str,
        members: list[Episode],
        ctx: OrganizerContext,
        keywords: list[str] | None = None,
        meta_info: str = "",
        page_units: list[list[str]] | None = None,
    ) -> MemoryOp:
        for e in members:
            self._unit_pages.setdefault(e.id, set()).add(segment_id)
        self._page_sources.setdefault(segment_id, set()).update(e.id for e in members)
        payload = {
            "id": segment_id,
            "topic": topic,
            "content": content,
            "keywords": sorted(set(keywords or [])),
            "source_episode_ids": [e.id for e in members],
            "embedding_text": content[:2000],
            # The segment's own page structure, persisted because retrieval needs
            # it: upstream searches sessions and then scores the PAGES inside the
            # matched ones (`search_sessions` -> `matched_pages`), and a flat
            # `source_episode_ids` cannot say where one exchange ends and the next
            # begins. Grouping is the organizer's (`on_message` pairing), so it is
            # recorded here rather than re-derived by the read path.
            "page_units": [list(page) for page in (page_units or [])],
        }
        if meta_info:
            # Rendered as "Conversation chain overview: ..." beside the segment,
            # which is where upstream puts it beside each retrieved page.
            payload["meta_info"] = meta_info
        return MemoryOp(
            op=OpType.ADD,
            target_type="pages",
            target_id=segment_id,
            payload=payload,
        )

    def _knowledge_ops(self, kind: str, lines: list, source_ids: list[str]) -> list[MemoryOp]:
        """Append knowledge entries to one LPM store, in the lineage's shape.

        pypi keeps two ``deque(maxlen=knowledge_capacity)`` and drops the
        oldest entry silently on overflow (we emit the eviction as a DELETE so
        the evolution log still records it); its per-line filter lowercases —
        blank lines and "none"/"- none"/"- none." in any case are dropped
        (``memoryos.py:196-204`` + ``long_term.add_knowledge_entry``). The
        eval lineage has NO capacity anywhere (plain lists,
        ``long_term_memory.py:9-10``, round-12 finding 6) and its
        ``add_knowledge`` rejects only the EXACT strings ""/"- None"/"- None."
        after strip (``long_term_memory.py:68-71``, case-sensitive) — a bare
        "None" or a lowercase "- none" is stored, faithfully."""
        ops: list[MemoryOp] = []
        ring = self._knowledge.setdefault(kind, [])
        for raw in lines or []:
            line = str(raw).strip()
            if self.knowledge_line_filter == "eval":
                if line in ("", "- None", "- None."):
                    continue
            elif not line or line.lower() in ("none", "- none", "- none."):
                continue
            entry_id = new_id()
            ops.append(
                MemoryOp(
                    op=OpType.ADD,
                    target_type="semantic",
                    target_id=entry_id,
                    payload={
                        "id": entry_id,
                        "content": line,
                        "kind": kind,
                        "source_episode_ids": list(source_ids),
                        "embedding_text": line,
                    },
                )
            )
            ring.append(entry_id)
            if self.knowledge_fifo:
                while len(ring) > self.knowledge_capacity:
                    ops.append(
                        MemoryOp(
                            op=OpType.DELETE,
                            target_type="semantic",
                            target_id=ring.pop(0),
                            payload={"reason": f"{kind}_fifo_capacity"},
                        )
                    )
        return ops

    def _promote_to_lpm(self, segment_id: str, ctx: OrganizerContext) -> list[MemoryOp]:
        """LPM update for one hot segment: rewrite the user-profile DOCUMENT and
        append to the two knowledge stores, then reset the segment's heat.

        This is upstream's ``_trigger_profile_and_knowledge_update_if_needed``,
        which is a document replacement and not a fact append (docs/10 M1, the
        gap this closes). Three things it does that appending never did:

        - The profile is ONE evolving document under a fixed id, rebuilt by an
          LLM that reads the previous version (``update_user_profile(...,
          merge=False)`` — a full replace). Appended facts could only accumulate
          and contradict; a rewrite can revise.
        - Knowledge is split into user-private and assistant knowledge, each in
          the lineage's store shape (see ``_knowledge_ops``), because upstream
          serves them through different channels.
        - Only pages not yet analysed feed the prompt, and the whole segment is
          then marked analysed. Without that a hot segment is re-analysed from
          scratch on every flush, paying for the same pages repeatedly.

        Call partition per lineage (round-12 finding 9 — the totals match, the
        ROUTING differs and is now faithful too):

        - pypi (``profile_update="single"``): one analysis call that folds the
          old profile in and returns the profile alone, plus one knowledge call
          returning the private AND assistant lists
          (``gpt_user_profile_analysis`` + ``gpt_knowledge_extraction``, run in
          parallel upstream).
        - eval (``"two_call"``): ONE ``gpt_personality_analysis`` call returns
          profile AND private data via section markers, a SEPARATE
          ``analyze_assistant_knowledge`` call returns the assistant blob, and
          the old profile is folded in by a third merge call
          (``gpt_update_profile`` with the inline "Profile Merge Task" prompt),
          skipped when there is no old profile to merge against. The previous
          code routed private facts through pypi's knowledge call under both
          presets.

        Returns [] and leaves heat intact when there is nothing new to analyse
        or a call drops, so the segment gets another attempt (round-5 N5)."""
        unit_ids = self._page_sources.get(segment_id, set())
        unanalyzed = [uid for uid in unit_ids if uid not in self._analyzed]
        if not unanalyzed:
            # upstream: "Hot session has no unanalyzed pages. Skipping."
            return []
        # `_page_sources` is a set, so conversation order comes from the
        # episodes themselves rather than from insertion order.
        episodes = sorted(ctx.doc_store.get_episodes(unanalyzed), key=lambda e: e.timestamp)
        conversation = "\n".join(
            " | ".join(f"{self._page_key(e)}: {e.content}" for e in page)
            for page in self._pages(episodes)
        )[:8000]

        existing = ctx.doc_store.get_items([PROFILE_ITEM_ID], "semantic")
        old_profile = str(existing[0].get("content", "")) if existing else ""

        if self.profile_update == "two_call":
            analysis = ctx.llm.call(
                "distill",
                EVAL_ANALYSIS_PROMPT.format(conversation=conversation),
                EVAL_ANALYSIS_SCHEMA,
                required_keys=("profile", "private"),
            )
            assistant = ctx.llm.call(
                "distill",
                EVAL_ASSISTANT_PROMPT.format(conversation=conversation),
                EVAL_ASSISTANT_SCHEMA,
                required_keys=("assistant_knowledge",),
            )
            if analysis is None or assistant is None:
                # a dropped call costs a retry, not the segment (round-5 N5)
                return []
            text = str(analysis.get("profile", "")).strip()
            if old_profile:
                merged = ctx.llm.call(
                    "distill",
                    PROFILE_MERGE_PROMPT.format(old_profile=old_profile, new_analysis=text),
                    PROFILE_SCHEMA,
                    required_keys=("profile",),
                )
                # A dropped merge keeps the analysis as the profile, which is
                # what upstream's `updated_profile = new_profile` fallback does
                # when there is nothing to merge against — losing the merge must
                # not lose the analysis.
                if merged is not None:
                    text = str(merged.get("profile", "")).strip()
            private_lines = analysis.get("private") or []
            # one un-split blob under the eval preset (main_loco_parse.py:66-67:
            # `if assistant_knowledge and != "None": add_assistant_knowledge(..)`)
            blob = str(assistant.get("assistant_knowledge", "")).strip()
            if not blob or blob == "None":
                assistant_lines = []
            elif self.assistant_blob:
                assistant_lines = [blob]
            else:
                # non-lineage override combination: per-line entries instead
                assistant_lines = blob.split("\n")
        else:
            profile = ctx.llm.call(
                "distill",
                PROFILE_PROMPT.format(
                    existing_user_profile=old_profile or "No existing profile data.",
                    conversation=conversation,
                ),
                PROFILE_SCHEMA,
                required_keys=("profile",),
            )
            knowledge = ctx.llm.call(
                "distill",
                KNOWLEDGE_PROMPT.format(conversation=conversation),
                KNOWLEDGE_SCHEMA,
                required_keys=("private",),
            )
            if profile is None or knowledge is None:
                # upstream wraps both futures in one try/except and returns
                # without resetting heat, so a dropped call costs a retry
                return []
            text = str(profile.get("profile", "")).strip()
            private_lines = knowledge.get("private") or []
            assistant_lines = knowledge.get("assistant_knowledge") or []

        ops: list[MemoryOp] = []
        source_ids = [e.id for e in episodes]
        # pypi guard, transcribed (memoryos.py:186-192): skip the replace when
        # the model returned "none" or something too short to be a profile, but
        # still take the knowledge and still reset heat. The eval driver has NO
        # guard — it writes whatever came back (main_loco_parse.py:53-57,
        # round-12 finding 7) — so `profile_guard=False` writes unconditionally.
        if not self.profile_guard or (text.lower() != "none" and len(text) >= 30):
            ops.append(
                MemoryOp(
                    # ADD, not UPDATE: the profile's whole state is this
                    # document, so a full replace under a stable id IS the
                    # semantics (same reasoning as base.cursor_op), and it works
                    # on the first write when no profile exists yet.
                    op=OpType.ADD,
                    target_type="semantic",
                    target_id=PROFILE_ITEM_ID,
                    payload={
                        "id": PROFILE_ITEM_ID,
                        "content": text,
                        "kind": "profile",
                        "source_episode_ids": source_ids,
                        # Explicitly EMPTY, meaning "do not index" (round-12
                        # finding 14): upstream serves the profile through ONE
                        # channel — the unconditional injection into every QA
                        # prompt (which the bench harness reproduces from the
                        # doc store) — never through embedded retrieval.
                        # Indexing this document let it compete in, and win
                        # slots of, the semantic k, appearing twice in one
                        # prompt; the empty embedding_text keeps it out of the
                        # vector store entirely.
                        "embedding_text": "",
                    },
                )
            )
        ops.extend(self._knowledge_ops("user_knowledge", private_lines, source_ids))
        ops.extend(self._knowledge_ops("assistant_knowledge", assistant_lines, source_ids))

        # upstream marks EVERY page of the session analysed, not just the ones
        # it fed the prompt (its own comment flags the choice); mirroring it
        # keeps the re-analysis cadence the same
        self._analyzed.update(unit_ids)
        h = self._heat[segment_id]
        h["n_visit"], h["length"] = 0, 0  # paper: reset after analysis
        h["last_access"] = datetime.now(timezone.utc)  # upstream: last_visit_time
        return ops


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain cosine — upstream normalizes both sides and dots, which is the
    same number; the explicit norms make this robust to unnormalized stores."""
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return num / (na * nb)
