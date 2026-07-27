"""MemMachine fidelity (arXiv:2604.04853, deployed-code lineage).

Every constant asserted here has a named upstream site in
``organizers/memmachine/organizer.py`` or ``MemMachineContextualize``; the point
of the file is that the two lineages stay apart and the read path's asymmetries
survive refactors.
"""

from datetime import datetime, timedelta, timezone

import pytest
from helpers import StubLLM, make_mem_multi

from agmem import AgenticMemory
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.memmachine import MEMMACHINE_PRESETS, MemMachineOrganizer
from agmem.retrieval.steps import (
    MemMachineContextualize,
    ReadContext,
    _weighted_index_proximity,
)

BASE = datetime(2022, 5, 20, 14, 30, tzinfo=timezone.utc)
TURNS = [
    ("Caroline", "I adopted a dog last week."),
    ("Melanie", "What is its name?"),
    ("Caroline", "Her name is Luna. She is a beagle."),
    ("Melanie", "Cute! I want to meet Luna."),
    ("Caroline", "She loves the park near my flat."),
    ("Melanie", "Which park is that?"),
    ("Caroline", "Riverside Park, ten minutes away."),
    ("Melanie", "I will bring my camera."),
]


def make_mem(organizer=None, llm=None):
    mem = AgenticMemory(
        namespace="t",
        organizers=[organizer or MemMachineOrganizer()],
        embedder=FakeEmbedder(dim=128),
    )
    if llm is not None:
        mem.structured = llm
        mem._ctx.llm = llm
    return mem


def ingest(mem, turns=TURNS):
    for index, (speaker, text) in enumerate(turns):
        mem.add_message(
            text, "user", timestamp=BASE + timedelta(minutes=index), meta={"speaker": speaker}
        )


def derivatives(mem):
    return sorted(mem.doc_store.list_items("derivatives", "t"), key=lambda d: d["timestamp"])


# ---- write path ------------------------------------------------------------


def test_declarative_preset_writes_one_derivative_per_message_and_calls_no_llm():
    """``declarative_memory.py::_derive_derivatives``, MESSAGE branch with
    ``message_sentence_chunking=False``: one derivative, ``f"{source}: {content}"``.

    The LLM assertion is the whole reason this methodology is in the comparison
    table — it is the first organizer here whose per-message write path has no
    language model in it at all."""
    llm = StubLLM({})
    mem = make_mem(llm=llm)
    try:
        ingest(mem)
        rows = derivatives(mem)
        assert len(rows) == len(TURNS)
        assert rows[0]["content"] == "Caroline: I adopted a dog last week."
        assert rows[0]["embedding_text"] == rows[0]["content"]
        assert llm.calls == []
        assert llm.drops == {}
    finally:
        mem.close()


def test_event_preset_anchor_carries_the_full_date_and_json_quoted_text():
    """``deriver/text_deriver.py::_format_for_embedding`` with
    ``FormatOptions(time_style=None)``: date survives, time does not, and the
    text is ``json.dumps``-ed while the producer prefix stays outside the
    quotes. The date is babel's CLDR ``full``, which does NOT zero-pad the day
    — unlike the declarative backend's own strftime renderer."""
    mem = make_mem(MemMachineOrganizer("event"))
    try:
        mem.add_message(
            "Her name is Luna.",
            "user",
            timestamp=datetime(2022, 5, 5, 9, 0, tzinfo=timezone.utc),
            meta={"speaker": "Caroline"},
        )
        (row,) = derivatives(mem)
        assert row["content"] == '[Thursday, May 5, 2022] Caroline: "Her name is Luna."'
    finally:
        mem.close()


def test_both_presets_default_to_no_segmentation_and_no_short_term_memory():
    """``EventLongTermMemoryConf`` defaults are passthrough + whole_text, and
    ``init_memmachine_params`` passes ``short_term_memory=None`` — so neither
    the chunker nor the summarizer is on any published number's path."""
    for name, preset in MEMMACHINE_PRESETS.items():
        organizer = MemMachineOrganizer(name)
        assert (organizer.segmenter, organizer.deriver) == ("passthrough", "whole_text")
        assert organizer.stm_capacity == 0 == preset["stm_capacity"]
        assert organizer.recent_context() == ""


def test_declarative_backend_rejects_a_segmenter_it_has_no_stage_for():
    with pytest.raises(ValueError, match="no segmentation stage"):
        MemMachineOrganizer("declarative", segmenter="text")
    with pytest.raises(ValueError, match="unknown deriver"):
        MemMachineOrganizer("event", deriver="llm")


def test_sentence_deriver_dedupes_because_upstream_returns_a_set():
    """``common/utils.py::extract_sentences`` returns a SET: a message that
    repeats a sentence produces ONE anchor for it, and sentence order is not
    preserved. Skipped rather than approximated when nltk is absent — a
    different sentence split is a different memory."""
    pytest.importorskip("nltk")
    mem = make_mem(MemMachineOrganizer("declarative", deriver="sentence_text"))
    try:
        mem.add_message(
            "Luna is a beagle. Luna is a beagle. She loves the park.",
            "user",
            timestamp=BASE,
            meta={"speaker": "Caroline"},
        )
        contents = {row["content"] for row in derivatives(mem)}
        assert contents == {
            "Caroline: Luna is a beagle.",
            "Caroline: She loves the park.",
        }
    finally:
        mem.close()


def test_text_segmenter_is_upstreams_splitter_or_nothing():
    """``TextSegmenter`` IS langchain's ``RecursiveCharacterTextSplitter``
    (chunk 500, overlap 0, a 30-entry separator list). We call it or raise;
    there is no hand-rolled stand-in to drift from."""
    pytest.importorskip("langchain_text_splitters")
    organizer = MemMachineOrganizer("event", segmenter="text", max_chunk_length=40)
    mem = make_mem(organizer)
    try:
        mem.add_message(
            "Luna loves the park. " * 6,
            "user",
            timestamp=BASE,
            meta={"speaker": "Caroline"},
        )
        assert len(derivatives(mem)) > 1
    finally:
        mem.close()


# ---- short-term memory (off by default, implemented for ablation) ----------


def test_short_term_memory_summarizes_once_over_budget_and_keeps_the_buffer():
    """``ShortTermMemory.add_episodes`` -> ``_do_evict``: the budget is
    CHARACTERS (content + the summary's own length), eviction drops only
    already-summarized episodes, and the rewrite sees the whole remaining
    buffer rather than the evicted prefix."""
    llm = StubLLM({"distill": [{"summary": "Caroline adopted a beagle named Luna."}]})
    organizer = MemMachineOrganizer("declarative", stm_capacity=80)
    mem = make_mem(organizer, llm=llm)
    try:
        ingest(mem, TURNS[:4])  # 104 characters of content, over the budget on the 4th
        assert len(llm.calls) == 1
        role, prompt = llm.calls[0]
        assert role == "distill"
        # the prompt renders episodes in upstream's `episodes_to_string` form
        assert '[Friday, May 20, 2022 at 02:30 PM] Caroline: "I adopted a dog last week."' in prompt
        assert "Your summary (under 100 words)" in prompt  # 80/2/8 rounded up to 100
        context = organizer.recent_context()
        assert "<Summary>\nCaroline adopted a beagle named Luna.\n</Summary>" in context
        assert "<Episodes>" in context
    finally:
        mem.close()


def test_a_failed_summary_call_keeps_the_previous_one():
    """``_create_summary`` returns the OLD summary on failure and
    ``set_summary`` ignores an empty rewrite — a dropped call must never blank
    the rolling summary."""
    llm = StubLLM({"distill": [{"summary": "first"}, {"summary": ""}]})
    organizer = MemMachineOrganizer("declarative", stm_capacity=60)
    mem = make_mem(organizer, llm=llm)
    try:
        ingest(mem)
        assert organizer.state()["summary"] == "first"
    finally:
        mem.close()


# ---- read path -------------------------------------------------------------


class _FakeDocStore:
    def __init__(self, mem):
        self._mem = mem

    def list_items(self, memory_type, namespace=None):
        return self._mem.doc_store.list_items(memory_type, namespace)

    def get_episodes(self, ids):
        return self._mem.doc_store.get_episodes(ids)


def _hits(mem, *indices):
    """Derivative hits for the given turn indices, best-first."""
    from agmem.core.types import ScoredItem
    from agmem.retrieval.steps import _DictItem

    rows = derivatives(mem)
    return [
        ScoredItem(item=_DictItem(rows[index]), memory_type="derivatives", score=1.0 - rank * 0.1)
        for rank, index in enumerate(indices)
    ]


def _run(mem, hits, **kwargs):
    step = MemMachineContextualize(**kwargs)
    return step.run(hits, ReadContext(doc_store=_FakeDocStore(mem), namespace="t"))


def test_context_expansion_is_one_third_backward_two_thirds_forward():
    """``expand_context // 3`` backward, the REST forward
    (``declarative_memory.py`` L399-400 / ``event_memory.py`` L450-451). A
    symmetric window would pull turns 4-6 around turn 5; upstream pulls 4-7."""
    mem = make_mem()
    try:
        ingest(mem)
        out = _run(mem, _hits(mem, 5), expand_context=3, limit=30)
        assert [s.item.data["content"].split(": ", 1)[1] for s in out] == [
            f'"{TURNS[index][1]}"' for index in (4, 5, 6, 7)
        ]
    finally:
        mem.close()


def test_expansion_is_clamped_to_the_limit_and_zero_means_the_nucleus_alone():
    """``expand_context = min(max(0, expand_context), max_num_episodes - 1)``,
    and upstream's own default is 0 — the mapping derivative -> episode is the
    read path even with no widening at all."""
    mem = make_mem()
    try:
        ingest(mem)
        assert len(_run(mem, _hits(mem, 5), expand_context=0, limit=30)) == 1
        # limit 2 clamps expand to 1 -> 0 backward, 1 forward
        out = _run(mem, _hits(mem, 5), expand_context=9, limit=2)
        assert [s.item.data["content"].split(": ", 1)[1] for s in out] == [
            f'"{TURNS[index][1]}"' for index in (5, 6)
        ]
    finally:
        mem.close()


def test_overflowing_context_keeps_the_episodes_nearest_its_nucleus():
    """``_unify_scored_anchored_episode_contexts``: a context that does not fit
    contributes by ``_weighted_index_proximity``, which is asymmetric — the
    nucleus first, then forward neighbours, then backward ones."""
    mem = make_mem()
    try:
        ingest(mem)
        out = _run(mem, _hits(mem, 1, 5), expand_context=3, limit=5)
        served = [s.item.data["content"] for s in out]
        # first context (turn 1) fits whole: turns 0,1,2,3
        assert len(served) == 5
        assert served[-1].endswith(f'"{TURNS[5][1]}"')  # only the nucleus of the second
    finally:
        mem.close()


def test_weighted_index_proximity_prefers_forward_recall():
    assert _weighted_index_proximity(3, 3) == -0.25  # the nucleus itself
    assert _weighted_index_proximity(4, 3) == 0.25  # one forward
    assert _weighted_index_proximity(2, 3) == 1.0  # one backward, ranked lower
    assert _weighted_index_proximity(5, 3) < _weighted_index_proximity(2, 3)


def test_contexts_are_ranked_by_the_reranker_not_by_the_seed_derivative():
    """Upstream scores the assembled CONTEXT string
    (``_score_episode_contexts`` -> ``reranker.score(query, contexts)``), so a
    weakly-matching seed inside a strong context still wins. Without a
    text-capable reranker (profile ``lite`` ships ``NoopReranker``) the nuclei
    keep their fused order — a weaker read path, visible in the config."""

    class ContextReranker:
        needs_text = True

        def rerank(self, query_emb, candidates, vectors, k, texts=None, query="", **kwargs):
            scored = [(cid, float("camera" in (texts or {}).get(cid, ""))) for cid, _ in candidates]
            return sorted(scored, key=lambda pair: pair[1], reverse=True)[:k]

    mem = make_mem()
    try:
        ingest(mem)
        kwargs = {"expand_context": 3, "limit": 4}
        step = MemMachineContextualize(**kwargs)
        ctx = ReadContext(
            doc_store=_FakeDocStore(mem),
            namespace="t",
            query="camera",
            reranker=ContextReranker(),
        )
        hits = _hits(mem, 1, 6)
        # Turn 6's context reaches turn 7 ("I will bring my camera"), so it is
        # merged FIRST and takes 3 of the 4 slots; the higher-scored seed's
        # context then only fits its own nucleus.
        assert [s.item.data["content"].split(": ", 1)[1] for s in step.run(hits, ctx)] == [
            f'"{TURNS[index][1]}"' for index in (1, 5, 6, 7)
        ]
        # Without a text-capable reranker the seed order decides instead, and
        # the first context fills the budget on its own.
        assert [s.item.data["content"].split(": ", 1)[1] for s in _run(mem, hits, **kwargs)] == [
            f'"{TURNS[index][1]}"' for index in (0, 1, 2, 3)
        ]
    finally:
        mem.close()


def test_end_to_end_search_serves_episodes_in_upstream_line_format():
    """The served item is ``episodic`` carrying ``[date at time] speaker:
    "content"`` — the prefix is part of upstream's QA prompt, and a bare
    episode loses it."""
    mem = make_mem()
    try:
        ingest(mem)
        bundle = mem.search("Luna", memory_types=["derivatives"], k=2)
        assert bundle.items
        assert all(s.memory_type == "episodic" for s in bundle.items)
        assert all(s.item.content.startswith("[Friday, May 20, 2022 at 02:") for s in bundle.items)
    finally:
        mem.close()


def test_a_deleted_derivative_leaves_the_episode_order_intact():
    """The ordered episode index is derived from the derivatives (DocStore has
    no time-ordered episode listing), so tombstones have to be filtered the way
    every other read path filters them."""
    mem = make_mem()
    try:
        ingest(mem)
        hits = _hits(mem, 5)  # resolved before the tombstone, which has no timestamp
        rows = derivatives(mem)
        mem.doc_store.put_item(
            rows[3]["id"], "derivatives", "t", {"id": rows[3]["id"], "deleted": True}
        )
        served = [s.item.data["content"] for s in _run(mem, hits, expand_context=3, limit=30)]
        assert not any(TURNS[3][1] in line for line in served)
        # turn 3 leaves the index, so the window slides onto turn 2 rather than
        # shrinking: one backward, two forward, still four episodes.
        assert [line.split(": ", 1)[1] for line in served] == [
            f'"{TURNS[index][1]}"' for index in (4, 5, 6, 7)
        ]
    finally:
        mem.close()


def test_chained_organizers_still_register_one_step_per_type():
    """The step is registered on the memory TYPE, so an item written straight
    to the store gets it too (module contract of ``retrieval/steps.py``)."""
    mem = make_mem_multi([MemMachineOrganizer()], StubLLM({}))
    try:
        assert isinstance(mem.pipeline.read_steps["derivatives"], MemMachineContextualize)
    finally:
        mem.close()


def test_episode_type_is_declared():
    """``produces`` drives ``default_memory_types``; ``derivatives`` must be in
    the shared vocabulary or nobody can find it by reading ``MEMORY_TYPES``."""
    from agmem.core.types import MEMORY_TYPES

    assert MemMachineOrganizer.produces == ("derivatives",)
    assert set(MemMachineOrganizer.produces) <= set(MEMORY_TYPES)


# ---- semantic memory: the paper's "profile" tier ---------------------------


def profile_features(mem):
    return [
        row
        for row in mem.doc_store.list_items("semantic", "t")
        if row.get("kind") == "profile_feature" and not row.get("deleted")
    ]


def test_the_profile_tier_costs_one_llm_call_per_message_per_category():
    """The scoping fact for the whole comparison table: MemMachine is LLM-free
    on the EPISODIC path only. ``semantic_ingestion.py`` loops messages inside a
    loop over categories, one ``llm_feature_update`` call each."""
    from agmem.organizers.memmachine import MemMachineProfileOrganizer, SemanticCategory

    categories = [
        SemanticCategory("profile", {"Demographic Information": "..."}),
        SemanticCategory("coding", {"Tech Proficiency": "..."}),
    ]
    llm = StubLLM({"distill": [{"commands": []} for _ in range(6)]})
    organizer = MemMachineProfileOrganizer(categories=categories, consolidate_every=0)
    mem = make_mem(organizer, llm=llm)
    try:
        ingest(mem, TURNS[:3])
        assert len(llm.calls) == 6  # 3 messages x 2 categories
    finally:
        mem.close()


def test_add_and_delete_commands_are_applied_in_sequence():
    """``_apply_commands``: delete removes EVERY value under
    ``(category, tag, feature)``, and an update is delete-then-add."""
    from agmem.organizers.memmachine import MemMachineProfileOrganizer

    llm = StubLLM(
        {
            "distill": [
                {
                    "commands": [
                        {
                            "command": "add",
                            "tag": "Hobbies & Interests",
                            "feature": "dog_owner",
                            "value": "User adopted a beagle named Luna",
                        }
                    ]
                },
                {
                    "commands": [
                        {
                            "command": "delete",
                            "tag": "Hobbies & Interests",
                            "feature": "dog_owner",
                            "value": "irrelevant",
                        },
                        {
                            "command": "add",
                            "tag": "Hobbies & Interests",
                            "feature": "dog_owner",
                            "value": "User owns two dogs",
                        },
                    ]
                },
            ]
        }
    )
    organizer = MemMachineProfileOrganizer(consolidate_every=0)
    mem = make_mem(organizer, llm=llm)
    try:
        ingest(mem, TURNS[:2])
        rows = profile_features(mem)
        assert [row["value"] for row in rows] == ["User owns two dogs"]
        assert rows[0]["content"] == "[Hobbies & Interests] dog_owner: User owns two dogs"
        assert rows[0]["embedding_text"] == "User owns two dogs"  # value alone, as upstream
    finally:
        mem.close()


def test_a_delete_retracts_an_add_queued_in_the_same_message():
    """Upstream applies commands sequentially against storage, so a delete after
    an add of the same feature removes it. Ours has to retract the queued op,
    since the row does not exist yet when the batch is built."""
    from agmem.organizers.memmachine import MemMachineProfileOrganizer

    llm = StubLLM(
        {
            "distill": [
                {
                    "commands": [
                        {"command": "add", "tag": "T", "feature": "f", "value": "v"},
                        {"command": "delete", "tag": "T", "feature": "f", "value": "v"},
                    ]
                }
            ]
        }
    )
    organizer = MemMachineProfileOrganizer(consolidate_every=0)
    mem = make_mem(organizer, llm=llm)
    try:
        ingest(mem, TURNS[:1])
        assert profile_features(mem) == []
    finally:
        mem.close()


def test_one_malformed_command_drops_the_whole_message():
    """Upstream's failure granularity, and its own delete EXAMPLE is malformed
    against its schema (no ``value``) — so this path is reachable by a model
    that copies the prompt."""
    from agmem.organizers.memmachine import MemMachineProfileOrganizer

    llm = StubLLM(
        {
            "distill": [
                {
                    "commands": [
                        {"command": "add", "tag": "T", "feature": "good", "value": "kept?"},
                        {"command": "delete", "tag": "T", "feature": "no_value"},
                    ]
                }
            ]
        }
    )
    organizer = MemMachineProfileOrganizer(consolidate_every=0)
    mem = make_mem(organizer, llm=llm)
    try:
        ingest(mem, TURNS[:1])
        assert profile_features(mem) == []  # the well-formed command dies with the batch
        assert organizer.dropped_commands == 2
    finally:
        mem.close()


def test_consolidation_fires_only_over_the_tag_threshold():
    """``consolidated_threshold=20`` features under ONE tag. Under it the pass
    is a storage read and costs nothing."""
    from agmem.organizers.memmachine import MemMachineProfileOrganizer

    organizer = MemMachineProfileOrganizer(consolidation_threshold=3, consolidate_every=0)
    llm = StubLLM({"distill": []})
    mem = make_mem(organizer, llm=llm)
    try:
        for index in range(2):
            mem.doc_store.put_item(
                f"f{index}",
                "semantic",
                "t",
                {
                    "id": f"f{index}",
                    "kind": "profile_feature",
                    "category": "profile",
                    "tag": "Hobbies & Interests",
                    "feature": f"f{index}",
                    "value": f"v{index}",
                    "source_episode_ids": [f"e{index}"],
                },
            )
        assert organizer.consolidate(mem._ctx) == []
        assert llm.calls == []
    finally:
        mem.close()


def test_consolidation_deletes_what_is_not_kept_and_inherits_its_citations():
    """``keep_memories`` is an allowlist — everything else is deleted — and the
    merged feature carries the union of the deleted features' citations."""
    from agmem.organizers.memmachine import MemMachineProfileOrganizer

    llm = StubLLM(
        {
            "distill": [
                {
                    "keep_memories": ["f0"],
                    "consolidated_memories": [
                        {"tag": "Hobbies & Interests", "feature": "pets", "value": "two dogs"}
                    ],
                }
            ]
        }
    )
    organizer = MemMachineProfileOrganizer(consolidation_threshold=3, consolidate_every=0)
    mem = make_mem(organizer, llm=llm)
    try:
        for index in range(3):
            mem.doc_store.put_item(
                f"f{index}",
                "semantic",
                "t",
                {
                    "id": f"f{index}",
                    "kind": "profile_feature",
                    "category": "profile",
                    "tag": "Hobbies & Interests",
                    "feature": f"f{index}",
                    "value": f"v{index}",
                    "source_episode_ids": [f"e{index}"],
                },
            )
        mem._apply_ops(organizer.consolidate(mem._ctx), organizer.name)
        rows = {row["id"]: row for row in profile_features(mem)}
        assert "f0" in rows and "f1" not in rows and "f2" not in rows
        merged = next(row for row in rows.values() if row.get("consolidated"))
        assert merged["value"] == "two dogs"
        assert sorted(merged["source_episode_ids"]) == ["e1", "e2"]
    finally:
        mem.close()


def test_the_prompts_key_for_merged_features_is_read_too():
    """Defect 1: the consolidation prompt documents ``consolidate_memories``
    while the parser reads ``consolidated_memories``. Upstream would delete
    everything not kept and write nothing back."""
    from agmem.organizers.memmachine import MemMachineProfileOrganizer

    llm = StubLLM(
        {
            "distill": [
                {
                    "keep_memories": [],
                    "consolidate_memories": [{"tag": "T", "feature": "merged", "value": "one"}],
                }
            ]
        }
    )
    organizer = MemMachineProfileOrganizer(consolidation_threshold=2, consolidate_every=0)
    mem = make_mem(organizer, llm=llm)
    try:
        for index in range(2):
            mem.doc_store.put_item(
                f"f{index}",
                "semantic",
                "t",
                {
                    "id": f"f{index}",
                    "kind": "profile_feature",
                    "category": "profile",
                    "tag": "T",
                    "feature": f"f{index}",
                    "value": f"v{index}",
                },
            )
        mem._apply_ops(organizer.consolidate(mem._ctx), organizer.name)
        values = [row["value"] for row in profile_features(mem)]
        assert values == ["one"]  # upstream would leave this empty
    finally:
        mem.close()


def test_profile_features_do_not_land_in_memoryos_profile_section():
    """``bench/locomo.py`` injects every ``semantic`` row with ``kind="profile"``
    verbatim as "User Profile:" — MemoryOS's single LPM document. These are
    (tag, feature, value) triples and carry their own kind so they reach the
    prompt through retrieval instead."""
    from agmem.organizers.memmachine.profile import PROFILE_FEATURE_KIND

    assert PROFILE_FEATURE_KIND != "profile"
