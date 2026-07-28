"""MemoryOS organizer (arXiv:2506.06326, EMNLP'25 Oral) — compact port.

OS-style hierarchy: STM (fixed-size dialogue buffer) -> MTM (topic
segments with heat) -> LPM (user profile + knowledge). Heat =
alpha*N_visit + beta*L_interaction + gamma*R_recency; segments crossing
theta trigger profile/knowledge extraction and reset (paper §MTM).

``fidelity`` picks which upstream lineage the divergent constants come
from — ``"pypi"`` (default, the maintained library) or ``"eval"`` (the
harness that produced the paper's LoCoMo numbers). They differ in the heat
weights, whether recency is live or stored, the keyword-overlap formula and
the STM capacity; theta=0.6, tau=5.0, MTM capacity 2000 AND the eviction
policy are shared — both upstream lineages evict by LFU (``evict_lfu`` over
an access counter: pypi ``mid_term.py:177``, eval ``mid_term_memory.py:120``).
The paper's "segments with the lowest heat are evicted" (§3.3) matches
neither codebase; ``eviction="lowest_heat"`` keeps that reading reachable as
an explicit non-lineage option (docs/16 session 3). See ``MEMORYOS_PRESETS``,
which also lists the five defects found in the eval lineage and says which
are reproduced.

The unit everything is counted in is the PAGE, not the message, because
that is upstream's unit and both constants depend on it: ``short_term.py``
holds a deque of ``add_qa_pair`` entries so ``short_term_capacity=10`` is
10 pages, and ``mid_term.py`` sets ``"L_interaction": len(processed_details)``
over pages. A page is one (user_input, agent_response) exchange. Counting
messages instead — as this organizer did until the 2026-07-27 audit — put
BOTH constants in the wrong unit: heat ran ~2x high, so the
``heat_threshold`` promotion to LPM fired at about half the content, and
the STM batch was half the size upstream flushes.

Pages are rebuilt from the message stream by ``_pages``, following
upstream's own LoCoMo driver (``eval/main_loco_parse.py``): the first
speaker seen opens a page and any other speaker's turn attaches to the
page in progress. The pairing key is ``meta["speaker"]`` when the ingest
supplies one (LoCoMo) and ``role`` otherwise (LongMemEval, live traffic).
Two deviations, both deliberate: upstream OVERWRITES ``agent_response``
when two of the other speaker's turns land on one page, dropping the
first — we append instead, since losing verbatim content is barred here;
and because we receive the two halves as separate ``add_message`` calls
while upstream receives a formed pair, a flush can land between an
exchange's halves, splitting it across two segments.

Other deviations: single vector index instead of FAISS-per-tier. The
read-path N_visit feedback (paper §3.4) IS wired — ``on_retrieval`` bumps
n_visit/last_access on served ``pages`` hits, and the memoryos retrieval
config queries ``pages`` — so the heat loop is live, not missing (round-5
N1 restored it). It is merely *inert on ingest-then-eval benchmarks* such
as LoCoMo conv0: all writes (hence all promotion/eviction) complete during
ingest, so the retrieval-time bumps never feed back — N_visit effectively
stays 0 for the graded numbers, but by run *shape*, not by a missing
mechanism (2026-07-21 fidelity review, docs/10). Eviction emits DELETE ops
through the evolution log (auditable; upstream's old silent-loss issue #65
has since been fixed upstream too).

LPM (``_promote_to_lpm``) is upstream's three stores, not an append log
(docs/10 M1, closed 2026-07-27): one evolving user-profile DOCUMENT under a
fixed id, rewritten from its previous version and replaced wholesale
(``update_user_profile(merge=False)``), plus two bounded knowledge FIFOs —
user-private and assistant — at ``knowledge_capacity`` (upstream
``deque(maxlen=100)``). Only pages not yet analysed feed the prompts, and
the trigger inspects the HOTTEST segment as upstream's heap-top peek does.

STM is a 1-page FIFO rolling window, not a batch flush (``on_message``):
``is_full()`` is ``>=`` upstream, so its ``while is_full(): pop_oldest()``
emits exactly one page and leaves ``capacity - 1`` resident. Those resident
pages are then injected verbatim at QA time by ``recent_context()``, which
is what upstream's ``get_response`` does with
``short_term_memory.get_all()`` — retrieval cannot stand in for it, since
the point is recency rather than similarity. Each evicted page also carries
upstream's two dialogue-chain calls (continuity + ``meta_info``, see
``_chain_meta``), and the eval lineage's two-call profile update is
``profile_update="two_call"`` in ``MEMORYOS_PRESETS`` — pypi keeps the
one-call shape that folds the old profile into the analysis prompt.

Remaining gaps, tracked: the eval lineage's retrieval-constant lineage
(per-page embedding text and similar fine detail); upstream's
``Retriever`` runs its three channels concurrently, which we do not
(no behavioural difference, only wall-clock).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, new_id
from agmem.organizers.base import Organizer, OrganizerContext

logger = logging.getLogger("agmem.organizers.memoryos")

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
                    # PAGE indexes, not message indexes — the batch is presented
                    # to the model as numbered exchanges (see _pages).
                    "page_indexes": {"type": "array", "items": {"type": "integer"}},
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
# E2. R_recency is dead. `compute_segment_heat` reads a STORED `R_recency`
#     (initialised 1.0, refreshed only on a retrieval hit) and weights it
#     gamma=1e-4 — four orders of magnitude under one interaction, so it cannot
#     change any comparison. Reproduced: it IS the published operating point,
#     and `recency="stored"` makes the deadness visible instead of assumed.
# E3. Page embeddings differ by code path. `add_session` embeds
#     `f"User: {user_input} Assiant: {agent_response}"` (typo upstream's), while
#     the merge branch embeds `f"用户: {user_input}"` — different language prefix
#     and the assistant turn dropped entirely. Page-level retrieval therefore
#     depends on which path stored the page. NOT reproduced (we keep one
#     embedding text) — it is an inconsistency, not a setting.
# E4. The merge branch stamps every inserted page with the SEGMENT's keywords
#     instead of extracting per-page ones as `add_session` does. NOT reproduced.
# E5. `MidTermMemory.max_capacity` defaults to 7, but the eval driver passes
#     2000, so the class default is dead in the published run. We take 2000.
MEMORYOS_PRESETS: dict[str, dict] = {
    # memoryos-pypi/{memoryos,mid_term,short_term,long_term}.py
    "pypi": dict(
        stm_capacity=10,
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
    ),
    # eval/{main_loco_parse,dynamic_update,mid_term_memory,short_term_memory}.py
    "eval": dict(
        stm_capacity=1,
        heat_weights=(0.8, 0.8, 0.0001),
        recency="stored",
        keyword_similarity="containment_mean",
        eviction="lfu",
        # analysis WITHOUT the old profile, then a separate merge call
        profile_update="two_call",
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

TOPIC_PROMPT = """Split this dialogue batch into topical groups and summarize each.

Exchanges (indexed):
{messages}

Return JSON: {{"groups": [{{"topic": "short label", "summary": "2-3 sentence summary \
covering the concrete facts", "keywords": ["theme keyword", ...], \
"page_indexes": [0, 1, ...]}}]}}"""

# Condensed from PERSONALITY_ANALYSIS_SYSTEM/USER_PROMPT. The output is the
# WHOLE updated profile, not a delta — upstream feeds the result straight to
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

# Condensed from KNOWLEDGE_EXTRACTION_SYSTEM/USER_PROMPT. Two separate outputs
# because upstream stores them in two different deques (user vs assistant LTM).
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

# Upstream UPDATE_PROFILE_{SYSTEM,USER}_PROMPT, used ONLY by the eval lineage:
# its driver analyses the unanalysed pages WITHOUT the old profile
# (`gpt_personality_analysis`) and then merges in a SECOND call
# (`gpt_update_profile`), where the pypi library folds the old profile into the
# analysis prompt and does one call.
UPDATE_PROFILE_PROMPT = """You are an expert in merging and updating user profiles. Integrate
the new information into the old profile, maintaining consistency and
improving the overall understanding of the user. Avoid redundancy. The new
analysis is based on specific dimensions, try to incorporate these
insights meaningfully.

Please update the following user profile based on the new analysis. If the
old profile is empty or "None", create a new one based on the new analysis.

Old User Profile:
{old_profile}

New Analysis Data:
{new_analysis}

Return JSON: {{"profile": "the updated user profile"}}"""

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
        dialogue_chain: bool = True,
        flush_stm_on_drain: bool = False,
    ) -> None:
        """``fidelity`` selects an upstream lineage from ``MEMORYOS_PRESETS``
        ("pypi" when None/unknown, keeping the no-arg constructor stable); every
        other explicit (non-None) argument overrides that preset's value, same
        rule as ``NemoriOrganizer``.

        ``stm_capacity``: STM size in PAGES (upstream's unit — one page is one
        exchange, see ``_pages``) that triggers a flush to MTM; with an
        alternating two-sided dialogue pypi's 10 is ~20 messages, and the eval
        lineage's 1 flushes every exchange.
        ``mtm_capacity``: max MTM segments before eviction (2000 in both
        lineages — the eval class default of 7 is overridden by its own driver).
        ``heat_threshold``: heat (τ) at which a segment promotes to LPM (5.0 in
        both). ``similarity_threshold``: F_score (θ) merge threshold, cosine +
        keyword overlap, paper eq.(3) (0.6 in both).
        ``recency_tau_hours``: recency-decay time constant.
        ``knowledge_capacity``: max entries in EACH LPM knowledge FIFO
        (upstream ``long_term_knowledge_capacity``, one deque for user-private
        and one for assistant knowledge).
        ``heat_weights``/``recency``/``keyword_similarity``/``eviction``/
        ``profile_update`` are the lineage-divergent knobs; see
        ``MEMORYOS_PRESETS``.

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
        )
        params.update({k: v for k, v in overrides.items() if v is not None})
        params.update(
            dialogue_chain=dialogue_chain,
            flush_stm_on_drain=flush_stm_on_drain,
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
        self.dialogue_chain = params["dialogue_chain"]
        self.flush_stm_on_drain = params["flush_stm_on_drain"]
        # upstream eval's `access_frequency`: a retrieval counter that is NEVER
        # reset (unlike n_visit, which promotion zeroes), because LFU eviction
        # reads it. Kept separate for that reason.
        self._access: dict[str, int] = {}
        self._stm: list[Episode] = []
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
        # order of each knowledge FIFO so the oldest can be evicted at capacity.
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

    def _pages(self, episodes: list[Episode]) -> list[list[Episode]]:
        """Group a message run into upstream pages (one exchange each).

        Transcribed from upstream's own LoCoMo driver (``eval/main_loco_parse.py``):
        a turn by ``speaker_a`` opens a new page, any other speaker's turn fills
        the page in progress. Pure function of the input list apart from
        latching ``_page_opener`` on the very first message, so calling it
        repeatedly on a growing buffer is stable."""
        pages: list[list[Episode]] = []
        for episode in episodes:
            key = self._page_key(episode)
            if self._page_opener is None:
                self._page_opener = key
            if key == self._page_opener or not pages:
                pages.append([episode])
            else:
                # upstream assigns to `agent_response` (overwriting a previous
                # one); we append, so a second reply is kept rather than lost
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
        ``self._heat``, no store writes."""
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
                # eval lineage stores R rather than recomputing it, and this
                # hit is the only moment it ever moves (module docstring E2)
                h["recency"] = math.exp(
                    -(now - h["last_access"]).total_seconds() / 3600 / self.recency_tau_hours
                )
                # `access_frequency`, the LFU counter — never reset by promotion.
                # Keyed on the SEGMENT, like the heat entry beside it: eviction
                # picks a segment, so a page id here would build a counter no
                # eviction ever reads.
                self._access[segment_id] = self._access.get(segment_id, 0) + 1
        return []

    def on_message(self, episode: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        """Appends to the STM buffer; once it holds ``stm_capacity`` PAGES,
        evicts exactly ONE page — the oldest — to MTM and returns its ops
        (empty list otherwise).

        Pages, not messages — upstream's ``ShortTermMemory`` is a deque of
        ``add_qa_pair`` entries and its ``is_full()`` is ``len(memory) >=
        max_capacity``, so the same ``>=`` on the page count is the literal
        trigger. Because our two halves arrive as separate calls, the trigger
        can fire on a page whose reply has not arrived yet; that reply then
        opens the next buffer (see the module docstring's split-exchange
        deviation).

        ONE page, not the whole buffer: upstream's flush is
        ``while self.short_term_memory.is_full(): pop_oldest()``, and since
        ``is_full()`` is ``len >= capacity``, a single pop already clears the
        condition. So STM is a FIFO rolling window that stays at
        ``capacity - 1`` pages and hands one page to MTM per new page
        thereafter — the paper says so too (§3.3: "the **oldest dialogue page**
        is transferred from the STM to the MTM according to the FIFO
        principle"). This organizer used to flush the entire buffer and clear
        it, which changed three things at once: the topic-summary LLM ran once
        per ``capacity`` pages instead of once per page (the main reason
        docs/09 shows MemoryOS as the cheapest methodology), segments were cut
        along batch boundaries rather than page by page, and STM was left empty
        so the recent-context channel upstream injects at QA time had nothing
        in it (round-5 N2, closed 2026-07-27).

        Chained use (driving this from another organizer's episodes) is an
        experimental composition handled by
        ``organizers.experimental.ChainedConsumer``, which calls this same
        entry point plus ``retire``/``patch_unit`` — keeping this organizer
        messages-only and paper-faithful."""
        self._stm.append(episode)
        pages = self._pages(self._stm)
        if len(pages) < self.stm_capacity:
            return []
        oldest, rest = pages[0], pages[1:]
        self._stm = [unit for page in rest for unit in page]
        return self._evict_to_mtm(oldest, ctx)

    def patch_unit(self, unit: Episode) -> None:
        """In-place UPDATE of a unit still buffered in STM (used by
        ``ChainedConsumer`` when an upstream episode is revised before it has
        been paged). Once the unit has been paged, the update is ignored —
        documented staleness (spec §3)."""
        if any(e.id == unit.id for e in self._stm):
            self._stm = [unit if e.id == unit.id else e for e in self._stm]

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
        self._stm = [e for e in self._stm if e.id not in superseded]
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
        ``history_text``). Draining it here would put the most recent
        ``capacity - 1`` pages into MTM *and* the recent-context channel, which
        upstream does neither of. Nothing is lost by not draining: raw episodes
        are stored by the facade regardless, and ``recent_context`` serves them.

        With the eval lineage (``stm_capacity=1``) the question does not arise —
        STM is empty after every eviction.

        ``flush_stm_on_drain=True`` restores the old force-drain, which is what
        the docs/09 runs did; keep it in mind when comparing against them."""
        if not self.flush_stm_on_drain or not self._stm:
            return []
        batch, self._stm = self._stm, []
        return self._evict_to_mtm(batch, ctx)

    def recent_context(self) -> str:
        """The STM buffer as upstream renders it into the QA prompt.

        ``get_response`` builds ``history_text`` from ``short_term_memory
        .get_all()``, one line per page, and the LoCoMo driver
        (``main_loco_parse.generate_system_response_with_meta``) does the same
        with the two speakers' names substituted for User/Assistant. We keep the
        page's own speaker labels, which are those names when the ingest supplies
        them.

        Empty string when STM holds nothing, so a caller can inject it
        unconditionally."""
        lines = []
        for page in self._pages(self._stm):
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

    def _evict_to_mtm(self, batch: list[Episode], ctx: OrganizerContext) -> list[MemoryOp]:
        # The batch is a run of messages; MTM's unit is the page, so everything
        # below — the prompt's indexes, the group membership, and the heat's
        # L_interaction — is counted in pages (2026-07-27 audit B1).
        pages = self._pages(batch)
        # Chain first: upstream forms pages (continuity + meta_info) BEFORE the
        # multi-topic summary, and the summary prompt then sees the same pages.
        meta_info = self._chain_meta(pages, ctx)
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
                    page_units=[[e.id for e in page] for page in pages],
                )
            ]

        indexed = "\n".join(
            f"[{i}] " + " | ".join(f"{self._page_key(e)}: {e.content}" for e in page)
            for i, page in enumerate(pages)
        )
        result = ctx.llm.call(
            "distill",
            TOPIC_PROMPT.format(messages=indexed),
            TOPIC_SCHEMA,
            required_keys=("groups",),
        )
        groups = (result or {}).get("groups") or [
            {
                "topic": "batch",
                "summary": "\n".join(e.content for e in batch),
                "page_indexes": list(range(len(pages))),
            }
        ]

        ops: list[MemoryOp] = []
        for g in groups:
            # sorted(set(...)): a repeated index would otherwise inflate
            # L_interaction (and so heat) without adding any content
            indexes = sorted(
                {i for i in g.get("page_indexes", []) if isinstance(i, int) and 0 <= i < len(pages)}
            ) or list(range(len(pages)))
            member_pages = [pages[i] for i in indexes]
            members = [e for page in member_pages for e in page]
            n_pages = len(member_pages)
            summary = str(g.get("summary", ""))
            keywords = [str(k).lower() for k in g.get("keywords") or []]
            embedding = ctx.embedder.embed([summary])[0]
            # F_score = cos + Jaccard(keywords), threshold 0.6 — paper eq.(3);
            # round-5 P0 restored the Jaccard term (cosine-only was stricter
            # and fragmented segments). Consider top-3 candidates.
            hits = ctx.vector_store.search(
                embedding, k=3, memory_type="pages", namespace=ctx.namespace
            )
            best_id, best_f = None, 0.0
            for hit_id, cos in hits:
                if hit_id not in self._heat:
                    continue
                candidate = ctx.doc_store.get_items([hit_id], "pages")
                candidate_keywords = set((candidate[0] if candidate else {}).get("keywords", []))
                f = cos + self._keyword_overlap(candidate_keywords, set(keywords))
                if f > best_f:
                    best_id, best_f = hit_id, f

            if best_id is not None and best_f >= self.similarity_threshold:
                segment_id = best_id  # merge into existing segment (F_score >= θ)
                existing = ctx.doc_store.get_items([segment_id], "pages")
                old = existing[0] if existing else {}
                content = (old.get("content", "") + "\n" + summary).strip()
                merged_kw = sorted(set(old.get("keywords", [])) | set(keywords))
                h = self._heat[segment_id]
                # upstream: target_session["L_interaction"] += len(pages_to_insert)
                h["length"] += n_pages
                h["last_access"] = datetime.now(timezone.utc)
                for e in members:
                    self._unit_pages.setdefault(e.id, set()).add(segment_id)
                self._page_sources.setdefault(segment_id, set()).update(e.id for e in members)
                ops.append(
                    MemoryOp(
                        op=OpType.UPDATE,
                        target_type="pages",
                        target_id=segment_id,
                        payload={
                            "content": content,
                            "keywords": merged_kw,
                            "source_episode_ids": list(old.get("source_episode_ids", []))
                            + [e.id for e in members],
                            "embedding_text": content[-2000:],
                            "page_units": list(old.get("page_units", []))
                            + [[e.id for e in page] for page in member_pages],
                            # newest chain summary wins, as upstream's
                            # _update_linked_pages_meta_info overwrites the chain
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
                content = summary
                ops.append(
                    self._segment_add(
                        segment_id,
                        str(g.get("topic", "?")),
                        content,
                        members,
                        ctx,
                        keywords,
                        meta_info,
                        [[e.id for e in page] for page in member_pages],
                    )
                )

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
            # begins. Grouping is the organizer's (`_pages`), so it is recorded
            # here rather than re-derived by the read path.
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
        """Append knowledge lines to one FIFO deque, evicting past capacity.

        Upstream keeps two ``deque(maxlen=100)`` (``knowledge_base`` for the
        user, ``assistant_knowledge`` for the assistant) and drops the oldest
        entry silently on overflow. We emit the eviction as a DELETE so the
        evolution log still records it, and mirror upstream's line filter:
        blank lines and the literal "none"/"- none"/"- none." the extraction
        prompt is told to emit when it finds nothing."""
        ops: list[MemoryOp] = []
        ring = self._knowledge.setdefault(kind, [])
        for raw in lines or []:
            line = str(raw).strip()
            if not line or line.lstrip("- ").rstrip(".").strip().lower() in ("none", "n/a"):
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
        append to the two knowledge FIFOs, then reset the segment's heat.

        This is upstream's ``_trigger_profile_and_knowledge_update_if_needed``,
        which is a document replacement and not a fact append (docs/10 M1, the
        gap this closes). Three things it does that appending never did:

        - The profile is ONE evolving document under a fixed id, rebuilt by an
          LLM that reads the previous version (``update_user_profile(...,
          merge=False)`` — a full replace). Appended facts could only accumulate
          and contradict; a rewrite can revise.
        - Knowledge is split into user-private and assistant knowledge, each a
          bounded FIFO, because upstream serves them through different channels.
        - Only pages not yet analysed feed the prompt, and the whole segment is
          then marked analysed. Without that a hot segment is re-analysed from
          scratch on every flush, paying for the same pages repeatedly.

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
        # Two lineages, two shapes. pypi folds the previous profile into the
        # analysis prompt and does ONE call; the eval driver analyses the
        # unanalysed pages WITHOUT it (`gpt_personality_analysis`) and merges in a
        # SECOND call (`gpt_update_profile`), skipping that call when there is no
        # old profile to merge against. The difference is not cosmetic: in the
        # eval shape the analysis cannot revise a previous claim, only the merge
        # can, and the merge sees the analysis as "New Analysis Data" rather than
        # as a conversation.
        two_call = self.profile_update == "two_call"
        profile = ctx.llm.call(
            "distill",
            PROFILE_PROMPT.format(
                existing_user_profile=(
                    "No existing profile data."
                    if two_call
                    else (old_profile or "No existing profile data.")
                ),
                conversation=conversation,
            ),
            PROFILE_SCHEMA,
            required_keys=("profile",),
        )
        if two_call and profile is not None and old_profile:
            merged = ctx.llm.call(
                "distill",
                UPDATE_PROFILE_PROMPT.format(
                    old_profile=old_profile,
                    new_analysis=str(profile.get("profile", "")),
                ),
                PROFILE_SCHEMA,
                required_keys=("profile",),
            )
            # A dropped merge keeps the analysis as the profile, which is what
            # upstream's `updated_profile = new_profile` fallback does when there
            # is nothing to merge — losing the merge must not lose the analysis.
            if merged is not None:
                profile = merged
        knowledge = ctx.llm.call(
            "distill",
            KNOWLEDGE_PROMPT.format(conversation=conversation),
            KNOWLEDGE_SCHEMA,
            required_keys=("private",),
        )
        if profile is None or knowledge is None:
            # upstream wraps both futures in one try/except and returns without
            # resetting heat, so a dropped call costs a retry, not the segment
            return []

        ops: list[MemoryOp] = []
        source_ids = [e.id for e in episodes]
        text = str(profile.get("profile", "")).strip()
        # upstream guard, transcribed: skip the replace when the model returned
        # "none" or something too short to be a profile, but still take the
        # knowledge and still reset heat
        if text.lower() != "none" and len(text) >= 30:
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
                        "embedding_text": text[:2000],
                    },
                )
            )
        ops.extend(self._knowledge_ops("user_knowledge", knowledge.get("private"), source_ids))
        ops.extend(
            self._knowledge_ops(
                "assistant_knowledge", knowledge.get("assistant_knowledge"), source_ids
            )
        )

        # upstream marks EVERY page of the session analysed, not just the ones
        # it fed the prompt (its own comment flags the choice); mirroring it
        # keeps the re-analysis cadence the same
        self._analyzed.update(unit_ids)
        h = self._heat[segment_id]
        h["n_visit"], h["length"] = 0, 0  # paper: reset after analysis
        h["last_access"] = datetime.now(timezone.utc)  # upstream: last_visit_time
        return ops
