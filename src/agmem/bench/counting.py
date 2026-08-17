"""API-free counting LLM for dry-run quotes. Drop-in for LLMClient.chat: counts
calls per role and returns schema-valid canned JSON per organizer profile, so an
organizer's full write/read path executes with zero spend and the call counts are
REAL (branchy organizers — eviction, merge — cannot be quoted by per-turn
constants; see the 2026-07-31 readiness audit). Promoted from
tests/test_repro_artifacts.py::_FakeCountingLLM."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from agmem import AgenticMemory
from agmem.config import AgmemConfig
from agmem.embed.fake import FakeEmbedder
from agmem.llm.client import RoleConfig
from agmem.llm.structured import StructuredCaller


def _amem_canned(role: str, prompt: str) -> str:
    if role == "extract":
        if "generate several keywords" in prompt:
            return '{"keywords": "alpha, beta"}'
        return '{"keywords": ["k1"], "context": "ctx", "tags": ["t1", "t2", "t3"]}'
    if role == "distill":
        return '{"should_evolve": false, "connections": []}'
    return "stub answer"


def _nemori_segment_groups(prompt: str, group_size: int = 4) -> list[dict]:
    """Deterministic, schema-valid ``BATCH_SEGMENT_SCHEMA`` partition: fixed-size
    index groups covering every indexed message BATCH_SEGMENT_PROMPT embeds as
    ``[i] ...`` lines. A canned response has no other way to learn the chunk
    size, so it is read back out of the prompt text itself."""
    indices = [int(m) for m in re.findall(r"(?m)^\[(\d+)\]", prompt)]
    n = (max(indices) + 1) if indices else 1
    return [
        {"indices": list(range(start, min(start + group_size, n))), "topic": "conversation"}
        for start in range(0, n, group_size)
    ]


def _nemori_canned(role: str, prompt: str) -> str:
    """Schema-valid responses for every structured call in
    ``agmem.organizers.nemori.stages`` / ``organizer.py``. Merge-decision
    always returns "new" (no-merge) so write-path counting stays bounded and
    reproducible — it never triggers the merge-content synthesis call."""
    if role == "extract":
        if "Partition this conversation" in prompt:  # BATCH_SEGMENT_PROMPT
            return json.dumps({"episodes": _nemori_segment_groups(prompt)})
        if "NEWEST message starts a new episode" in prompt:  # BOUNDARY_PROMPT (v1 preset)
            return '{"boundary": true, "confidence": 0.9}'
        return "{}"
    if role == "distill":
        if "episodic memory generation expert" in prompt:  # EPISODE_PROMPT (narrate)
            return (
                '{"title": "Canned episode", '
                '"narrative": "A canned narrative of the segment.", '
                '"timestamp": "2024-01-01T00:00:00"}'
            )
        if "describes the SAME event" in prompt:  # MERGE_DECISION_PROMPT — no-merge
            return '{"decision": "new"}'
        if "Merge these two episodic memories" in prompt:  # MERGE_CONTENT_PROMPT
            return (
                '{"title": "Merged episode", '
                '"narrative": "A canned merged narrative.", '
                '"timestamp": "2024-01-01T00:00:00"}'
            )
        if "previously known knowledge" in prompt:  # PREDICT_PROMPT
            return '{"prediction": "A canned prediction of the episode content."}'
        if "Compare the prediction" in prompt:  # CALIBRATE_PROMPT
            return '{"facts": ["The user has a canned preference."]}'
        if "HIGH-VALUE, PERSISTENT knowledge" in prompt:  # DIRECT_EXTRACT_PROMPT (cold start)
            return '{"facts": ["The user has a canned preference."]}'
        if "existing similar statements" in prompt:  # INTEGRATE_PROMPT (ThreeWayIntegrator)
            return '{"decision": "new"}'
        return '{"facts": []}'
    return "stub answer"


def _mem0_canned(role: str, prompt: str) -> str:
    """Schema-valid responses for the two `v0.1.94` write-path calls.

    Always ADD, for the same reason nemori's profile always answers "new": it
    keeps counting deterministic while still GROWING the store, so per-fact
    retrieval and the decision prompt both scale the way a real run's do. Event
    mix cannot change the call count — the decision call is unconditional and
    batched — so this over-counts nothing and under-counts nothing at the call
    level, which is exactly the structural claim being quoted.

    Branching is on prompt substrings because ``CountingLLM`` joins every
    message including the system one, and Mem0 is the only organizer here whose
    first-phase prompt rides entirely in the system message (house pattern from
    ``_nemori_canned``).
    """
    if role == "extract":  # FACT_RETRIEVAL_PROMPT rides in the system message
        if "Personal Information Organizer" in prompt:
            return '{"facts": ["A canned fact about the user."]}'
        return '{"facts": []}'
    if role == "distill":
        if "smart memory manager" in prompt:
            return (
                '{"memory": [{"id": "0", "text": "A canned fact about the user.", "event": "ADD"}]}'
            )
        return '{"memory": []}'
    return "stub answer"


# -- zep -----------------------------------------------------------------------
#
# Zep is the first organizer here whose call count is NOT fixed by its control
# flow. The other three profiles can claim their counts are real because every
# branch they leave unexercised costs nothing: mem0's decision call is
# unconditional, nemori's merge-content call is the only data-dependent one and
# "new" suppresses it. Zep has three data-dependent sites and two of them are
# unbounded per message:
#
#   1. entity extraction   1/message, UNCONDITIONAL           -> exact
#   2. entity resolution    <=1/message (batched over all unresolved entities;
#                          fires when some entity has a semantic candidate the
#                          deterministic stage could not match)  -> bounded exact
#   3. fact extraction     1/message, but ONLY when >= 2 distinct entities
#                          resolved (`organizer.py`: `if len(name_to_id) < 2`)
#   4. edge resolution     1 per surviving NEW fact, once the graph holds any
#                          fact at all (the invalidation-candidate search is
#                          top-k with NO score floor, so it is non-empty from
#                          the second fact onward)      -> scales with fact yield
#   5/6. community summarize+describe  ~1 per entity per rebuild
#                                                     -> scales with entity yield
#
# Sites 4-6 are therefore a function of how many entities and facts a message
# yields, which is a property of the MODEL AND THE CORPUS, not of the code — no
# canned profile can measure it. So the profile takes those yields as explicit
# parameters and the quote reports a BAND across them, instead of one number
# that would read as measured. Sites 1-3 are exact under any setting.
#
# Entity names come from the message text (proper-noun-shaped tokens plus the
# speaker) rather than from a synthetic vocabulary: the real corpus is the one
# thing available at zero spend that carries realistic NAME RECURRENCE, and
# recurrence is what decides site 2 — a name already in the store resolves
# deterministically and costs nothing, a near-miss escalates to the LLM.

_ZEP_CURRENT_MESSAGE = re.compile(r"<CURRENT MESSAGE>\n(.*?)\n</CURRENT MESSAGE>", re.DOTALL)
_ZEP_SPEAKER = re.compile(r"^\([^)]*\)\s*([^:]+):")
_ZEP_PROPER = re.compile(r"\b[A-Z][a-z]{2,}\b")
_ZEP_FACT_NAMES = re.compile(r"as subject/object: \[(.*?)\]")
_ZEP_REF_TIME = re.compile(r"REFERENCE TIME: (\S+)")
# Sentence-initial function words match the proper-noun shape but are not
# entities; leaving them in would inflate the entity yield with tokens that
# recur in every message and so resolve deterministically forever, quietly
# biasing site 2 toward zero.
_ZEP_STOPWORDS = (
    "The That This These Those There Then They Their Yeah Yes Not And But For You Your Its "
    "Well Also What When Where Which While With Have Just Like Really Sure Wow Now Own "
    "How Why Who She Her His Him Was Were Are Did Does Been Being Because Maybe Might Must "
    "Some Such Thanks Thank Actually Absolutely Definitely Sorry Hey Hello Let Get Got "
    "Can Could Would Should Will Say Said See Look Feel Know Think Want Need Even Ever"
)
_ZEP_NOT_A_NAME = frozenset(_ZEP_STOPWORDS.split())


def _zep_message_names(prompt: str, cap: int) -> list[str]:
    """Entity names for the CURRENT MESSAGE: the speaker first (the prompt tells
    the model to "Always include the speaker"), then proper-noun-shaped tokens
    in order of appearance, deduped, capped at `cap`. A message with fewer than
    two of these yields fewer than two entities and so skips fact extraction —
    which is the organizer's real behaviour for entity-poor turns, not a
    shortcut of this profile's."""
    body = _ZEP_CURRENT_MESSAGE.search(prompt)
    text = body.group(1) if body else ""
    names: list[str] = []
    speaker = _ZEP_SPEAKER.match(text)
    if speaker:
        names.append(speaker.group(1).strip())
    for token in _ZEP_PROPER.findall(text):
        if token in _ZEP_NOT_A_NAME or token in names:
            continue
        names.append(token)
    return names[:cap]


def _zep_facts(prompt: str, per_message: int) -> list[dict]:
    """`per_message` facts over consecutive pairs of the entity names the FACT
    prompt was given, dated at the message's reference time.

    Each statement carries a slice of the CURRENT MESSAGE, so two messages about
    the SAME entity pair produce DIFFERENT statements. That is not decoration:
    the organizer's verbatim fast path (`edge_operations.py:687-700`) skips the
    edge-resolution call only when the normalized statement text matches an
    existing edge's exactly, and a pair-derived placeholder like "A is related
    to B" matches on every recurrence of the pair. Measured on conv0, that
    placeholder suppressed 79% of site 4 (160 calls where the corpus-derived
    statement yields 752) — a canned response that looked harmless was quietly
    quoting a fifth of the real bill. Real extraction says
    something different about a pair each time it comes up, so the fast path is
    the exception there; making it near-never here is the conservative side of
    that, and the band's high point is where the residual uncertainty lives."""
    listed = _ZEP_FACT_NAMES.search(prompt)
    names = re.findall(r"'([^']*)'", listed.group(1)) if listed else []
    body = _ZEP_CURRENT_MESSAGE.search(prompt)
    # Statement length feeds the prompt-size measurement downstream (existing
    # facts are rendered back into the edge-resolution prompt), so it is taken
    # from the corpus rather than invented: one sentence's worth of the message.
    said = re.sub(r"\s+", " ", (body.group(1) if body else "").split(":", 1)[-1]).strip()[:120]
    ref = _ZEP_REF_TIME.search(prompt)
    valid_at = ref.group(1) if ref else None
    facts = []
    for i in range(min(per_message, max(len(names) - 1, 0))):
        subject, obj = names[i], names[i + 1]
        facts.append(
            {
                "subject": subject,
                "predicate": "RELATED_TO",
                "object": obj,
                "statement": f"{subject} and {obj}: {said}",
                "valid_at": valid_at,
                "invalid_at": None,
            }
        )
    return facts


def zep_profile(
    entities_per_message: int = 4, facts_per_message: int = 2, merge_duplicates: bool = True
) -> Callable[[str, str], str]:
    """Build a schema-valid canned profile for `ZepGraphOrganizer` at a stated
    yield. Every response validates against the organizer's schemas
    (`ENTITY_SCHEMA`, `RESOLVE_SCHEMA`, `FACT_SCHEMA`, `EDGE_RESOLVE_SCHEMA`,
    `SUMMARY_SCHEMA`, `DESCRIPTION_SCHEMA`); a parse failure would silently
    change which branch runs and so which calls are counted.

    `merge_duplicates` decides the entity-resolution verdict. It does NOT change
    how often site 2 fires (that call is already made when the verdict is asked
    for), but it decides whether the graph accumulates one node per mention or
    one per real entity, which drives the community bill at sites 5/6 — the
    single widest term in the quote. True is the paper's intent (dedup is what
    the call is for); False is the degenerate upper bound. Both are quoted.

    Edge resolution always answers "not a duplicate, contradicts nothing", for
    the same reason `_mem0_canned` always answers ADD: it keeps the graph
    GROWING, so later messages face realistic candidate pools, and it cannot
    change this call's own count."""

    def _canned(role: str, prompt: str) -> str:
        if role == "extract":
            if "Extract the distinct real-world entities" in prompt:  # ENTITY_PROMPT
                names = _zep_message_names(prompt, entities_per_message)
                return json.dumps(
                    {
                        "entities": [
                            {"name": n, "type": "Person", "summary": f"a participant named {n}"}
                            for n in names
                        ]
                    }
                )
            if "Decide for each NEW entity" in prompt:  # RESOLVE_PROMPT
                ids = [int(m) for m in re.findall(r"(?m)^- id=(\d+)", prompt)]
                n_candidates = len(re.findall(r"(?m)^- candidate_id=(\d+)", prompt))
                # One candidate PER unresolved entity where the pool allows it.
                # Answering 0 for all of them would collapse every entity of a
                # message onto a single node — a graph so much smaller than any
                # real one that sites 5/6 would quote near zero.
                return json.dumps(
                    {
                        "resolutions": [
                            {
                                "id": i,
                                "duplicate_candidate_id": (
                                    min(i, n_candidates - 1) if merge_duplicates else -1
                                ),
                                "name": f"entity {i}",
                                "summary": "a canned merged summary",
                            }
                            for i in ids
                        ]
                    }
                )
            if "Extract relationship facts" in prompt:  # FACT_PROMPT
                return json.dumps({"facts": _zep_facts(prompt, facts_per_message)})
            return '{"entities": []}'
        if role == "distill":
            if "A new fact arrived" in prompt:  # EDGE_RESOLVE_PROMPT
                return '{"duplicate_of": null, "contradicts": []}'
            if "Synthesize the information" in prompt:  # SUMMARIZE_PAIR_PROMPT
                return '{"summary": "A canned community summary of the member entities."}'
            if "one sentence description" in prompt:  # SUMMARY_DESCRIPTION_PROMPT
                return '{"description": "Canned description of a community of entities."}'
            return "{}"
        return "stub answer"

    return _canned


def _rb_canned(role: str, prompt: str) -> str:
    """Schema-valid responses for ReasoningBank's write path.

    **The call count here is control-flow-fixed, unlike Zep's.** `on_task_end`
    makes exactly one `distill` call per task, and the `judge` call fires only
    when the caller's outcome is unlabeled (`_is_labeled`). Both benchmarks that
    drive this organizer supply an explicit `success`/`failure`, so the judge is
    off the path and no yield band is needed — which is why this profile can be
    a fixed reply where `zep_profile()` had to be parameterized.

    Three items, matching upstream's `DEFAULT_MAX_ITEMS` and the "at most 3" its
    SUCCESSFUL_SI/FAILED_SI advertise: the item COUNT is what scales the store,
    and therefore what scales a top-k read's retrieval work downstream, so
    answering with one would under-count the read side of a quote.

    The judge branch is answered anyway rather than left to the fallback. A
    profile that returns "stub answer" for a role the organizer may call would
    make `StructuredCaller` record a drop, and a drop silently subtracts a call
    from the very count being quoted.
    """
    if role == "judge":
        return '{"success": true, "reason": "canned verdict"}'
    if role == "distill":
        return json.dumps(
            {
                "items": [
                    {
                        "title": f"Canned strategy {i}",
                        "description": "A transferable step distilled from the trajectory.",
                        "content": "Check the candidate list before committing to a tag.",
                    }
                    for i in range(1, 4)
                ]
            }
        )
    return "stub answer"


CANNED_RESPONSES: dict[str, Callable[[str, str], str]] = {
    "amem": _amem_canned,
    "rb": _rb_canned,
    "nemori": _nemori_canned,
    "mem0": _mem0_canned,
    # Three points of the same Zep profile, not three profiles: the middle one
    # is what `zep_cross_encoder` maps to, the outer two exist so the quote can
    # state a band over the yields no canned response can measure (see above).
    "zep": zep_profile(),
    "zep_low": zep_profile(entities_per_message=3, facts_per_message=1),
    "zep_high": zep_profile(entities_per_message=6, facts_per_message=4, merge_duplicates=False),
}


class CountingLLM:
    """Counts ``chat()`` calls per role and returns canned, schema-valid JSON —
    no network, no API key, no spend."""

    def __init__(self, canned: str):
        if canned not in CANNED_RESPONSES:
            raise KeyError(
                f"no canned-response profile {canned!r} (known: {sorted(CANNED_RESPONSES)})"
            )
        self._canned = CANNED_RESPONSES[canned]
        self.calls: dict[str, int] = {}

    def chat(self, role, messages, budget_key=None, **overrides) -> str:
        self.calls[role] = self.calls.get(role, 0) + 1
        prompt = " ".join(m.get("content", "") for m in messages)
        return self._canned(role, prompt)


def build_counting_memory(
    canned: str,
    organizers_factory: Callable[[], list[Any]],
    data_dir: Any,
    namespace: str,
    memory_types: tuple[str, ...],
) -> tuple[AgenticMemory, CountingLLM]:
    """Build an ``AgenticMemory`` wired to a ``CountingLLM`` for ``canned`` — the
    full organizer write/read path runs for real, API-free, so the resulting
    call counts are real call counts, not per-turn constants."""
    cfg = AgmemConfig(
        profile="lite",
        data_dir=data_dir,
        llm_roles={"extract": RoleConfig(endpoint="x", model="m")},
        use_guided_json=False,
        sync_write=True,
        lexical_types=("episodic",),
    )
    mem = AgenticMemory(
        namespace=namespace,
        organizers=organizers_factory(),
        embedder=FakeEmbedder(dim=64),
        config=cfg,
    )
    fake = CountingLLM(canned)
    mem.llm = fake
    mem.structured = StructuredCaller(fake, use_guided_json=False)
    mem._ctx.llm = mem.structured  # organizer writes through the fake too
    return mem, fake
