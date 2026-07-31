import logging

from helpers import StubLLM

from agmem import AgenticMemory
from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, new_id
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.nemori import NemoriOrganizer
from agmem.organizers.nemori.organizer import NEMORI_PRESETS
from agmem.organizers.nemori.stages import (
    AppendIntegrator,
    BatchPartitioner,
    DedupIdReuseIntegrator,
    PerMessageBoundary,
    Segment,
    ThreeWayIntegrator,
)


def test_presets_resolve_to_stages():
    v1 = NemoriOrganizer(fidelity="v1")
    assert isinstance(v1._segmenter, PerMessageBoundary)
    assert v1._merger is None and isinstance(v1._integrator, AppendIntegrator)

    v4 = NemoriOrganizer(fidelity="v4")
    assert isinstance(v4._segmenter, BatchPartitioner) and v4._segmenter.window == 20
    assert (
        v4._merger.top_k == 5
        and v4._merger.similarity is None
        and v4._merger.time_gap_hours is None
    )
    assert (
        isinstance(v4._integrator, ThreeWayIntegrator)
        and v4._integrator.tau == 0.70
        and v4._integrator.top_k == 5
    )

    up = NemoriOrganizer(fidelity="upstream")
    assert isinstance(up._segmenter, BatchPartitioner)
    # Round-12 finding 1: upstream's merge_similarity_threshold=0.85 is a dead
    # knob (never read) — the deployed merge path applies NO cosine floor.
    assert up._merger.similarity is None and up._merger.time_gap_hours == 1.0
    assert isinstance(up._integrator, AppendIntegrator)
    assert up.semantic_top_k == 20  # published predict-stage retrieval (finding 5)
    assert up.episode_min_messages == 1  # eval configs; library default 2 drops groups
    # Finding 9: knobs the batch path never reads were removed from the preset.
    assert "buffer_min" not in NEMORI_PRESETS["upstream"]
    assert "buffer_max" not in NEMORI_PRESETS["upstream"]

    # 명시 인자가 프리셋을 이긴다 (mixing)
    mix = NemoriOrganizer(
        fidelity="v4", semantic_integration="append", consolidation="semantic_offline"
    )
    assert isinstance(mix._integrator, AppendIntegrator)
    assert mix._consolidator is not None

    # 무인자 = v1 동치 (기존 config 호환)
    assert isinstance(NemoriOrganizer()._segmenter, PerMessageBoundary)


def make_mem(organizer, llm):
    mem = AgenticMemory(namespace="t", organizers=[organizer], embedder=FakeEmbedder(dim=128))
    mem.structured = llm
    mem._ctx.llm = llm
    return mem


def _eps(n):
    return [Episode(content=f"m{i}") for i in range(n)]


def test_batch_partitioner_waits_then_partitions():
    seg = BatchPartitioner(window=4)
    llm = StubLLM(
        {
            "extract": [
                {
                    "episodes": [
                        {"indices": [0, 1], "topic": "a"},
                        {"indices": [2, 3], "topic": "b"},
                    ]
                }
            ]
        }
    )
    ctx = type("C", (), {"llm": llm})()
    out, rest = seg.push(_eps(3), ctx)
    assert out == [] and len(rest) == 3  # window 미달 — LLM 콜 없음
    assert llm.calls == []
    out, rest = seg.push(_eps(4), ctx)
    assert [len(s.episodes) for s in out] == [2, 2] and rest == []
    assert [s.topic for s in out] == ["a", "b"]  # topic rides along (round-12 finding 7)


def test_batch_partitioner_flush_small_tail_single_group():
    seg = BatchPartitioner(window=20, buffer_min=2)
    llm = StubLLM({"extract": []})
    ctx = type("C", (), {"llm": llm})()
    out = seg.flush(_eps(3), ctx)
    assert [len(s.episodes) for s in out] == [3]  # <window: 단일 그룹, LLM 없음
    assert out[0].topic == "conversation"  # upstream's own filler topic
    assert llm.calls == []


def test_batch_partitioner_llm_failure_falls_back_to_one_segment():
    seg = BatchPartitioner(window=2)
    ctx = type("C", (), {"llm": StubLLM({"extract": []})})()  # 응답 소진 → None
    out, rest = seg.push(_eps(2), ctx)
    assert [len(s.episodes) for s in out] == [2] and rest == []


def test_batch_partitioner_backlog_grab_chunks_at_80():
    """Round-12 finding 2: ``window`` is a MINIMUM GATE on the backlog, not a
    fixed window — a buffer larger than window is grabbed whole and segmented
    in chunks of chunk_max=80 (upstream _SEGMENT_CHUNK_SIZE), so chunk_max is
    genuinely reachable under bulk ingestion."""
    seg = BatchPartitioner(window=20, chunk_max=80)
    llm = StubLLM(
        {
            "extract": [
                {"episodes": [{"indices": list(range(80)), "topic": "bulk"}]},
                {"episodes": [{"indices": list(range(20)), "topic": "tail"}]},
            ]
        }
    )
    ctx = type("C", (), {"llm": llm})()
    out, rest = seg.push(_eps(100), ctx)
    assert rest == []
    assert len(llm.calls) == 2  # one 80-message chunk + one 20-message chunk
    assert "[79]" in llm.calls[0][1] and "[79]" not in llm.calls[1][1]
    assert [len(s.episodes) for s in out] == [80, 20]
    assert [s.topic for s in out] == ["bulk", "tail"]


def test_warm_start_bulk_backlog_exceeds_window():
    """warm_start with the batch segmenter takes upstream's bulk shape
    (add_messages accepts the whole LIST, memory_system.py:76-83): the corpus
    lands in the buffer before one grab, so the segmentation LLM sees MORE
    than ``window`` messages in a single call — per-message feeding would have
    fired at exactly 20 (round-12 finding 2)."""
    llm = StubLLM(
        {
            "extract": [{"episodes": [{"indices": list(range(25)), "topic": "trip"}]}],
            "distill": [
                {"title": "t", "narrative": "n", "timestamp": "2026-01-01"},
                {"facts": []},  # cold-start direct extraction
            ],
        }
    )
    org = NemoriOrganizer(fidelity="v4")  # window=20
    mem = make_mem(org, llm)
    try:
        ops = org.warm_start(_eps(25), mem._ctx)
        seg_calls = [p for role, p in llm.calls if role == "extract"]
        assert len(seg_calls) == 1 and "[24]" in seg_calls[0]  # 25 > window, ONE grab
        assert any(o.target_type == "episodes" for o in ops)
    finally:
        mem.close()


def test_episode_min_messages_drops_small_groups_before_llm():
    """Round-12 finding 3: upstream skips groups below episode_min_messages
    (memory_system.py:118-119). The LIBRARY default (2) silently loses those
    messages; both eval configs — and our default — use 1, keeping singletons."""
    llm = StubLLM(
        {
            "distill": [
                {"title": "t", "narrative": "n", "timestamp": "2026-01-01"},
                {"facts": []},
            ]
        }
    )
    org2 = NemoriOrganizer(fidelity="upstream", episode_min_messages=2)
    mem = make_mem(org2, llm)
    try:
        assert org2._flush_segments([Segment(_eps(1))], mem._ctx) == []
        assert llm.calls == []  # dropped BEFORE any LLM call — the group is lost

        org1 = NemoriOrganizer(fidelity="upstream")  # default 1 = published path
        assert org1.episode_min_messages == 1
        ops = org1._flush_segments([Segment(_eps(1))], mem._ctx)
        assert any(o.target_type == "episodes" for o in ops)
        # Paths without a segmenter topic feed upstream's filler (finding 7).
        assert "Boundary detection reason:\nconversation" in llm.calls[0][1]
    finally:
        mem.close()


def test_segment_topic_threads_into_episode_prompt_as_boundary_reason():
    """Round-12 finding 7: the segmenter's per-group topic reaches the episode
    generator as the boundary reason (upstream memory_system.py:121-123 →
    prompts.py:17-18) instead of being discarded."""
    llm = StubLLM(
        {
            "extract": [{"episodes": [{"indices": [0, 1], "topic": "weekend hiking plans"}]}],
            "distill": [
                {"title": "t", "narrative": "n", "timestamp": "2026-01-01"},
                {"facts": []},
            ],
        }
    )
    org = NemoriOrganizer(fidelity="v4", window=2)
    mem = make_mem(org, llm)
    try:
        mem.add_message("let's hike")
        mem.add_message("this weekend")
        narrate = next(p for role, p in llm.calls if role == "distill")
        assert "Boundary detection reason:\nweekend hiking plans" in narrate
    finally:
        mem.close()


# ---------------- EpisodeMerger (v4 §3.2.3 / upstream merger) ----------------


def test_episode_merger_merges_and_supersedes():
    llm = StubLLM(
        {
            "extract": [  # PerMessageBoundary, buffer_min=1: a check on every message
                {"boundary": False, "confidence": 0.9},  # msg1: nothing to compare yet
                {"boundary": True, "confidence": 0.9},  # msg2: cut -> episode 1 = [msg1]
                {"boundary": False, "confidence": 0.9},  # msg3: stay buffered
                {"boundary": True, "confidence": 0.9},  # msg4: cut -> episode 2 = [msg2, msg3]
            ],
            "distill": [
                # 1st episode: narrate + direct-extract(cold start) — no merge
                # candidates exist yet, so the merger consumes no LLM call here.
                {
                    "title": "hiking plan",
                    "narrative": "User planned a hike.",
                    "timestamp": "2026-05-01",
                },
                {"facts": []},
                # 2nd episode: narrate -> merge decision(merge, target 0) -> merged content
                {
                    "title": "hiking plan 2",
                    "narrative": "More hiking talk.",
                    "timestamp": "2026-05-01",
                },
                {"decision": "merge", "target_index": 0},
                {
                    "title": "hiking plan (merged)",
                    "narrative": "Combined hike story.",
                    "timestamp": "2026-05-01",
                },
                {"facts": []},  # PC over the merged episode (cold start again)
            ],
        }
    )
    org = NemoriOrganizer(episode_merge="llm", buffer_min=1, boundary_confidence=0.7)
    mem = make_mem(org, llm)
    try:
        for text in (
            "planning a hike",
            "more hiking details",
            "still about hiking",
            "totally unrelated topic",
        ):
            mem.add_message(text)
        eps = [o for o in mem.log.tail(50) if o.target_type == "episodes"]
        merged = [o for o in eps if o.op == OpType.MERGE]
        inv = [o for o in eps if o.op == OpType.INVALIDATE]
        assert len(merged) == 1 and len(inv) == 1
        assert merged[0].payload["supersedes"] == [inv[0].target_id]
        assert inv[0].payload["superseded_by"] == merged[0].target_id
        assert merged[0].payload["title"] == "hiking plan (merged)"
    finally:
        mem.close()


def _seed_episode(mem, eid, title, content, source_ids=(), timestamp="2026-05-01"):
    """ADD one nemori episode via the facade so doc + vec stores are populated."""
    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.ADD,
                target_type="episodes",
                target_id=eid,
                payload={
                    "id": eid,
                    "title": title,
                    "content": content,
                    "timestamp": timestamp,
                    "source_episode_ids": list(source_ids),
                    "embedding_text": f"{title}\n{content}",
                },
            )
        ],
        actor="nemori",
    )


def test_upstream_preset_merge_has_no_similarity_floor():
    """Round-12 finding 1: upstream's merge similarity 0.85 is config-plumbed
    but NEVER read — _find_similar hands the top-5 hits to the LLM unfiltered.
    Under the "upstream" preset a low-cosine candidate must still reach the
    merge-decision LLM (the old preset value 0.85 suppressed the call)."""
    llm = StubLLM({"distill": [{"decision": "new"}]})
    org = NemoriOrganizer(fidelity="upstream")
    mem = make_mem(org, llm)
    try:
        # Disjoint vocabulary -> FakeEmbedder cosine ~0.0 against the query.
        _seed_episode(mem, new_id(), "banana zebra", "banana zebra quantum flux")
        out = org._merger.merge_or_none(
            "hiking plan", "User plans a hike", "2026-05-01", ["m1"], mem._ctx
        )
        assert out is None  # LLM said "new" -> caller takes the plain ADD path
        assert len(llm.calls) == 1  # the low-cosine candidate DID reach the LLM
    finally:
        mem.close()


def test_merge_candidate_scores_are_logged(caplog):
    """T1-2: stages.py:395-399 computes each candidate's cosine score and then
    discards it once the ``similarity`` floor is applied, so arm-B's 0.85
    filter rate is unexplainable on non-upstream embedders (round-12 gap).
    merge_or_none must log every candidate's raw score — BEFORE the floor is
    applied — on the organizer's existing "agmem.organizers.nemori" logger
    channel, at INFO (fix round 1: DEBUG would need the paid run's driver to
    both raise this channel's level and attach a persisting handler just to
    keep the records, defeating the point — ace/organizer.py's dedup-skip
    log is the repo's INFO precedent for this kind of score note; the
    unstructured %s-interpolated message itself still follows both that and
    zep_graph/community.py's debug log, no new sink). This is diagnostics
    only: the filtered-out candidate must still be logged even though it
    never reaches the LLM, and behavior (ops/return value/LLM calls) must be
    byte-identical to the pre-existing filter test."""
    llm = StubLLM({"distill": [{"decision": "new"}]})
    org = NemoriOrganizer(episode_merge="llm", merge_similarity=0.85)
    mem = make_mem(org, llm)
    try:
        low_id = new_id()
        # Disjoint vocabulary -> FakeEmbedder cosine ~0.0, well below 0.85 ->
        # filtered OUT of the candidate set (no LLM call) but must still be
        # logged, since the whole point is explaining the filter rate.
        _seed_episode(mem, low_id, "banana zebra", "banana zebra quantum flux")
        with caplog.at_level(logging.INFO, logger="agmem.organizers.nemori"):
            out = org._merger.merge_or_none(
                "hiking plan", "User plans a hike", "2026-05-01", ["m1"], mem._ctx
            )
        # Zero behavior change: same outcome as the pre-existing 0.85-filter path.
        assert out is None
        assert llm.calls == []

        score_records = [r for r in caplog.records if "merge candidate" in r.message]
        assert len(score_records) == 1
        record = score_records[0]
        # INFO, not DEBUG: a paid run at the library's default logging config
        # (no level override, no extra handler) must still capture this —
        # DEBUG would silently drop it, defeating the point of the log.
        assert record.levelno == logging.INFO
        assert record.name == "agmem.organizers.nemori"
        logged_ids = [hit_id for hit_id, _score in record.args[-1]]
        assert low_id in logged_ids  # logged even though the floor filtered it out
    finally:
        mem.close()


def test_merge_decision_prompt_upstream_exposure():
    """Round-12 finding 8 (verifier-corrected): the NEW episode is shown as
    time + content only — NO title (upstream merger.py:87-94); candidates DO
    carry ``Title:`` (merger.py:180 — the original audit had this backwards)
    and their content is truncated to 200 chars (merger.py:181). Target
    selection stays by index — upstream selects by ID string (documented
    envelope difference)."""
    long_content = " ".join(f"word{i}" for i in range(80))  # well over 200 chars
    llm = StubLLM({"distill": [{"decision": "new"}]})
    org = NemoriOrganizer(episode_merge="llm")
    mem = make_mem(org, llm)
    try:
        _seed_episode(mem, new_id(), "CANDTITLE hiking", long_content, source_ids=["r1", "r2"])
        org._merger.merge_or_none(
            "NEWTITLE secret", "User plans a hike", "2026-05-01", ["m1"], mem._ctx
        )
        prompt = llm.calls[0][1]
        assert "NEWTITLE" not in prompt  # new episode: time + content only
        assert "Time: 2026-05-01 (1 messages)" in prompt  # new episode exposure
        assert "Title: CANDTITLE hiking" in prompt  # candidates DO include Title
        assert f"Content: {long_content[:200]}..." in prompt  # 200-char truncation
        assert long_content not in prompt  # the full narrative is never shown
        assert "(2 messages)" in prompt  # candidate message count from provenance
    finally:
        mem.close()


def test_calibration_after_merge_sees_both_episodes_messages():
    """Round-12 finding 4: after a MERGE, calibration must be generated from
    the MERGED episode's raw material — target + new source messages combined
    (upstream merger.py:133, memory_system.py:150-156, semantic.py:94-97) —
    not from the new segment alone, and with the target's messages first."""
    llm = StubLLM(
        {
            "distill": [
                {"title": "hike 2", "narrative": "More hiking.", "timestamp": "2026-05-01"},
                {"decision": "merge", "target_index": 0},
                {"title": "hike merged", "narrative": "All hiking.", "timestamp": "2026-05-01"},
                {"prediction": "hiking happens"},
                {"facts": []},
            ]
        }
    )
    org = NemoriOrganizer(episode_merge="llm")
    mem = make_mem(org, llm)
    try:
        # The target episode's raw messages, durable in the doc store —
        # exactly as write-then-organize leaves them.
        old_raw = Episode(content="OLD-RAW packed boots yesterday", role="user", namespace="t")
        mem.doc_store.add_episode(old_raw)
        _seed_episode(mem, new_id(), "hike 1", "User plans a hike.", source_ids=[old_raw.id])
        # Existing semantic knowledge, so calibration takes the predict ->
        # calibrate path (cold start would read only the generated narrative).
        _seed_semantic(mem, [(new_id(), "User enjoys hiking trips")])
        new_raw = Episode(content="NEW-RAW booked the trailhead", role="user", namespace="t")
        mem.doc_store.add_episode(new_raw)

        ops = org._flush_segment(Segment([new_raw], topic="hiking"), mem._ctx)

        merge_op = next(o for o in ops if o.op is OpType.MERGE and o.target_type == "episodes")
        assert merge_op.payload["source_episode_ids"] == [old_raw.id, new_raw.id]
        calibrate_prompt = llm.calls[-1][1]
        assert "OLD-RAW" in calibrate_prompt and "NEW-RAW" in calibrate_prompt
        # Upstream order: target's source messages first, then the new ones.
        assert calibrate_prompt.index("OLD-RAW") < calibrate_prompt.index("NEW-RAW")
    finally:
        mem.close()


# ---------------- SemanticIntegrator: Dedup / ThreeWay (v4 §3.3.3) ----------------


def _seed_semantic(mem, entries, actor="nemori"):
    """entries: list of (id, content) -> ADD each via the facade so both
    doc + vec stores are populated, exactly as a real Stage-3 ADD would.

    The actor matters: the facade persists it onto the item, and Nemori's
    integrators only consider their own items as merge candidates (stages.py
    ``OWNER``). Seeding under any other name models a *foreign* organizer's
    fact, which is what ``test_integrators_ignore_another_organizers_semantic``
    does deliberately."""
    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.ADD,
                target_type="semantic",
                target_id=sid,
                payload={"id": sid, "content": content, "embedding_text": content},
            )
            for sid, content in entries
        ],
        actor=actor,
    )


def test_dedup_id_reuse_updates_existing():
    llm = StubLLM({})
    mem = make_mem(NemoriOrganizer(), llm)
    try:
        old_id = new_id()
        _seed_semantic(mem, [(old_id, "User likes hiking")])

        integrator = DedupIdReuseIntegrator(threshold=0.85)
        ops = integrator.integrate("User likes hiking", "ep-new", ["s-new"], mem._ctx)

        assert len(ops) == 1
        assert ops[0].op == OpType.UPDATE
        assert ops[0].target_id == old_id  # 기존 id 재사용 — 신규 ADD가 아님
        assert ops[0].payload["episode_id"] == "ep-new"  # PR#19 provenance refresh
        assert ops[0].payload["content"] == "User likes hiking"
    finally:
        mem.close()


def test_three_way_merge_branch():
    llm = StubLLM(
        {
            "distill": [
                {
                    "decision": "merge",
                    "target_indexes": [0, 1],
                    "statement": "unified statement",
                }
            ]
        }
    )
    mem = make_mem(NemoriOrganizer(), llm)
    try:
        id_a, id_b = new_id(), new_id()
        # Identical text -> FakeEmbedder gives similarity 1.0, safely above tau.
        _seed_semantic(
            mem,
            [
                (id_a, "User likes hiking on weekends"),
                (id_b, "User likes hiking on weekends"),
            ],
        )

        integrator = ThreeWayIntegrator()
        ops = integrator.integrate("User likes hiking on weekends", "ep-new", ["s-new"], mem._ctx)

        assert len(ops) == 3
        merge_op = next(o for o in ops if o.op == OpType.MERGE)
        inv_ops = [o for o in ops if o.op == OpType.INVALIDATE]
        assert len(inv_ops) == 2
        assert merge_op.payload["content"] == "unified statement"
        assert set(merge_op.payload["supersedes"]) == {id_a, id_b}
        assert {o.target_id for o in inv_ops} == {id_a, id_b}
        assert all(o.payload["superseded_by"] == merge_op.target_id for o in inv_ops)
        assert llm.calls  # LLM was consulted (candidates cleared tau)
    finally:
        mem.close()


def test_three_way_conflict_branch():
    llm = StubLLM(
        {
            "distill": [
                {
                    "decision": "conflict",
                    "target_indexes": [0],
                    "statement": "User now lives in Busan",
                }
            ]
        }
    )
    mem = make_mem(NemoriOrganizer(), llm)
    try:
        old_id = new_id()
        _seed_semantic(mem, [(old_id, "User lives in Seattle")])

        integrator = ThreeWayIntegrator()
        ops = integrator.integrate("User lives in Seattle", "ep-new", ["s-new"], mem._ctx)

        assert len(ops) == 2
        add_op = next(o for o in ops if o.op == OpType.ADD)
        inv_op = next(o for o in ops if o.op == OpType.INVALIDATE)
        assert add_op.payload["content"] == "User now lives in Busan"
        assert inv_op.target_id == old_id
        assert inv_op.payload["superseded_by"] == add_op.target_id
        assert inv_op.payload["reason"] == "conflict"
    finally:
        mem.close()


def test_three_way_tau_filters_candidates():
    llm = StubLLM({})  # no "distill" responses queued — a call would return None
    mem = make_mem(NemoriOrganizer(), llm)
    try:
        old_id = new_id()
        # Disjoint vocabulary -> FakeEmbedder cosine similarity ~0.0, well below tau.
        _seed_semantic(mem, [(old_id, "banana zebra quantum")])

        integrator = ThreeWayIntegrator()  # tau=0.70 default
        ops = integrator.integrate("User likes hiking", "ep-new", ["s-new"], mem._ctx)

        assert len(ops) == 1
        assert ops[0].op == OpType.ADD
        assert ops[0].payload["content"] == "User likes hiking"
        assert llm.calls == []  # tau filtered out the only candidate -> no LLM call
    finally:
        mem.close()


def test_inline_predict_calibrate_dedupes_mutual_facts_within_batch():
    """Review I1: the inline v4 path had the consolidator's within-pass
    supersession blindness — two mutually near-duplicate facts produced by a
    single predict-calibrate each earn their own ThreeWay merge head against
    the not-yet-applied store, leaving two live heads and a double-superseded
    target. A call-local ``superseded`` set threaded through _predict_calibrate
    (exclude_ids into ThreeWay + accumulate its INVALIDATEs) must absorb fact B
    into fact A's merge without ever reaching the LLM for B. A 2nd merge
    response is queued so a regression consumes it and yields a 2nd head."""
    llm = StubLLM(
        {
            "distill": [
                {"prediction": "the dog"},
                {"facts": ["The user's dog is named Max", "The user's dog is called Max"]},
                {"decision": "merge", "target_indexes": [0], "statement": "User's dog is Max"},
                {"decision": "merge", "target_indexes": [0], "statement": "duplicate head"},
            ]
        }
    )
    org = NemoriOrganizer(fidelity="v4")
    mem = make_mem(org, llm)
    try:
        xid = new_id()
        _seed_semantic(mem, [(xid, "The user's dog is named Max")])
        superseded: set[str] = set()
        ops = org._predict_calibrate(
            "unrelated topic", "quux blorp", "u: hi", "ep1", ["m1"], mem._ctx, superseded
        )
        merges = [o for o in ops if o.op is OpType.MERGE and o.target_type == "semantic"]
        # Exactly one merge head survives; fact B is excluded from candidates.
        assert len(merges) == 1
        # predict + calibrate + one integrate (fact B never reaches the LLM).
        assert len(llm.calls) == 3
        assert xid in superseded  # the target the merge absorbed
    finally:
        mem.close()


# ---------------- SemanticOfflineConsolidator (Task 11, our mixing) ----------------


def test_semantic_offline_consolidate_merges_and_advances_cursor():
    llm = StubLLM(
        {
            "distill": [
                {
                    "decision": "merge",
                    "target_indexes": [0],
                    "statement": "User's dog is named Max",
                }
            ]
        }
    )
    org = NemoriOrganizer(fidelity="v1", consolidation="semantic_offline")
    mem = make_mem(org, llm)
    try:
        # Inline-appended as the "nemori" actor so the consolidator's own-actor
        # filter picks them up, exactly as a real Stage-3 ADD would log them.
        # High token overlap (7/7 shared but one word) keeps FakeEmbedder's
        # hashed bag-of-words cosine above the default tau=0.70.
        for i, f in enumerate(["The user's dog is named Max", "The user's dog is called Max"]):
            mem._apply_ops(
                AppendIntegrator().integrate(f, f"ep{i}", [f"m{i}"], mem._ctx),
                actor="nemori",
            )
        n = mem.consolidate()
        assert n > 0
        ops = mem.log.tail(20)
        merges = [o for o in ops if o.op == OpType.MERGE and o.target_type == "semantic"]
        assert len(merges) == 1 and merges[0].payload["consolidated"] is True
        cursor = [o for o in ops if o.target_type == "state"]
        assert cursor and cursor[-1].payload["seq"] > 0

        # 2nd call: no new input -> own consolidated output (MERGE/INVALIDATE,
        # not ADD) is skipped, cursor advances with zero LLM calls.
        calls_before = len(llm.calls)
        mem.consolidate()
        assert len(llm.calls) == calls_before
    finally:
        mem.close()


def test_semantic_offline_consolidate_dedupes_mutual_duplicates_within_one_pass():
    """Review task-11-review.md Critical-1: within a single run() pass, two
    facts queued since the cursor that are mutual near-duplicates of each
    other must not each independently earn their own merge head. Because
    ops accumulated during the pass are only applied to doc/vec *after*
    run() returns, judging fact B against the still-live (not-yet-
    invalidated) store lets B "merge" against A a second time, producing
    two live consolidated heads instead of one. A second "merge" LLM
    response is queued specifically so a regression (no within-pass
    tracking) would consume it and produce a 2nd merge head; the fix must
    skip fact B once fact A's merge has already superseded it, without
    even reaching the LLM for B."""
    llm = StubLLM(
        {
            "distill": [
                {
                    "decision": "merge",
                    "target_indexes": [0],
                    "statement": "User's dog is named Max",
                },
                # Would be consumed by fact B's independent merge judgment if
                # the within-pass "already superseded" guard is missing.
                {
                    "decision": "merge",
                    "target_indexes": [0],
                    "statement": "User's dog is named Max (duplicate head)",
                },
            ]
        }
    )
    org = NemoriOrganizer(fidelity="v1", consolidation="semantic_offline")
    mem = make_mem(org, llm)
    try:
        for i, f in enumerate(["The user's dog is named Max", "The user's dog is called Max"]):
            mem._apply_ops(
                AppendIntegrator().integrate(f, f"ep{i}", [f"m{i}"], mem._ctx),
                actor="nemori",
            )
        n = mem.consolidate()
        assert n > 0
        ops = mem.log.tail(20)
        merges = [o for o in ops if o.op == OpType.MERGE and o.target_type == "semantic"]
        # Exactly one merge head must survive -- fact B is absorbed by fact A's
        # merge within this same pass, not independently re-merged.
        assert len(merges) == 1
        # Fact B must be skipped before ever reaching the LLM.
        assert len(llm.calls) == 1
    finally:
        mem.close()


def test_semantic_offline_consolidate_reprocesses_after_crash():
    """M3(d) / spec §1.4 crash semantics: if a pass's ops are produced but
    never applied (crash before the facade commits), the cursor never advances,
    so the next pass reprocesses the same still-live facts and converges to the
    same merge — no fact is silently orphaned by the missed commit."""
    llm = StubLLM(
        {
            "distill": [
                {"decision": "merge", "target_indexes": [0], "statement": "User's dog is Max"},
                {"decision": "merge", "target_indexes": [0], "statement": "User's dog is Max (2)"},
            ]
        }
    )
    org = NemoriOrganizer(fidelity="v1", consolidation="semantic_offline")
    mem = make_mem(org, llm)
    try:
        for i, f in enumerate(["The user's dog is named Max", "The user's dog is called Max"]):
            mem._apply_ops(
                AppendIntegrator().integrate(f, f"ep{i}", [f"m{i}"], mem._ctx),
                actor="nemori",
            )
        # 1st pass: ops are produced but deliberately NOT applied (crash).
        ops1 = org.consolidate(mem._ctx)
        merges1 = [o for o in ops1 if o.op is OpType.MERGE and o.target_type == "semantic"]
        assert len(merges1) == 1
        assert org.read_cursor(mem._ctx) == 0  # cursor never advanced (nothing committed)

        # 2nd pass reprocesses the same live facts -> same convergent merge.
        ops2 = org.consolidate(mem._ctx)
        merges2 = [o for o in ops2 if o.op is OpType.MERGE and o.target_type == "semantic"]
        assert len(merges2) == 1
    finally:
        mem.close()


def test_semantic_offline_consolidate_cursor_respects_scan_limit():
    """Review I2: when the semantic log since the cursor exceeds ops_since's
    limit, the cursor must advance only to the last scanned seq — not jump to
    last_seq() — or the truncated tail is dropped forever. With scan_limit=1
    and two independent facts, the first pass processes only fact 0 and the
    cursor stays at fact 0's seq; the second pass picks up fact 1."""
    llm = StubLLM({})  # dissimilar facts -> no candidates -> no LLM call
    org = NemoriOrganizer(fidelity="v1", consolidation="semantic_offline")
    org._consolidator.scan_limit = 1
    mem = make_mem(org, llm)
    try:
        for i, f in enumerate(["Alice works at Google", "Bob lives in Paris"]):
            mem._apply_ops(
                AppendIntegrator().integrate(f, f"ep{i}", [f"m{i}"], mem._ctx),
                actor="nemori",
            )
        sem = mem.doc_store.ops_since(0, target_type="semantic")
        seq0, seq1 = sem[0][0], sem[1][0]

        mem.consolidate()  # scan_limit=1 -> truncated after fact 0
        assert org.read_cursor(mem._ctx) == seq0  # cursor held at fact 0, not last_seq()
        assert seq0 < seq1

        mem.consolidate()  # resumes from fact 0's seq -> processes fact 1
        assert org.read_cursor(mem._ctx) == seq1

        # convergence: nothing left, no LLM calls ever made (facts dissimilar)
        calls_before = len(llm.calls)
        mem.consolidate()
        assert len(llm.calls) == calls_before == 0
    finally:
        mem.close()


def test_integrators_ignore_another_organizers_semantic():
    """ "semantic" is shared with MemoryOS, so a foreign fact must never be a
    merge/reuse target — see stages.py ``OWNER``.

    Both searching integrators are covered because they fail differently:
    ThreeWay would INVALIDATE the foreign fact, DedupIdReuse would overwrite
    its content in place under the ``nemori`` actor."""
    llm = StubLLM(
        {
            "distill": [
                {
                    "decision": "conflict",
                    "target_indexes": [0],
                    "statement": "User now lives in Busan",
                }
            ]
        }
    )
    mem = make_mem(NemoriOrganizer(), llm)
    try:
        foreign_id = new_id()
        _seed_semantic(mem, [(foreign_id, "User lives in Seattle")], actor="memoryos")

        # ThreeWay: identical text, but the only candidate is foreign -> plain
        # ADD, no LLM verdict consumed, foreign fact untouched.
        ops = ThreeWayIntegrator().integrate("User lives in Seattle", "ep", ["s"], mem._ctx)
        assert [o.op for o in ops] == [OpType.ADD]
        assert llm.calls == []

        # DedupIdReuse: top-1 is the foreign fact above threshold, yet its id
        # must not be reused.
        ops = DedupIdReuseIntegrator(threshold=0.85).integrate(
            "User lives in Seattle", "ep", ["s"], mem._ctx
        )
        assert [o.op for o in ops] == [OpType.ADD]
        assert ops[0].target_id != foreign_id

        assert mem.doc_store.get_items([foreign_id], "semantic")[0] == {
            **mem.doc_store.get_items([foreign_id], "semantic")[0],
            "content": "User lives in Seattle",
            "actor": "memoryos",
        }
    finally:
        mem.close()


def test_legacy_items_without_actor_stay_own():
    """Stores written before ``actor`` was persisted must keep resolving as
    Nemori's own, so no past run's behavior shifts under the ownership guard."""
    mem = make_mem(NemoriOrganizer(), StubLLM({}))
    try:
        old_id = new_id()
        _seed_semantic(mem, [(old_id, "User likes hiking")])
        data = mem.doc_store.get_items([old_id], "semantic")[0]
        del data["actor"]  # simulate a pre-field item
        mem.doc_store.put_item(old_id, "semantic", mem.namespace, data)

        ops = DedupIdReuseIntegrator(threshold=0.85).integrate(
            "User likes hiking", "ep-new", ["s-new"], mem._ctx
        )
        assert ops[0].op == OpType.UPDATE and ops[0].target_id == old_id
    finally:
        mem.close()
