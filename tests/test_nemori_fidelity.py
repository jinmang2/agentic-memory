from helpers import StubLLM

from agmem import AgenticMemory
from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, new_id
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.nemori import NemoriOrganizer
from agmem.organizers.nemori_stages import (
    AppendIntegrator,
    BatchPartitioner,
    DedupIdReuseIntegrator,
    EpisodeMerger,
    PerMessageBoundary,
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
    assert up._merger.similarity == 0.85 and up._merger.time_gap_hours == 1.0
    assert isinstance(up._integrator, AppendIntegrator)

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
    assert [len(s) for s in out] == [2, 2] and rest == []


def test_batch_partitioner_flush_small_tail_single_group():
    seg = BatchPartitioner(window=20, buffer_min=2)
    llm = StubLLM({"extract": []})
    ctx = type("C", (), {"llm": llm})()
    assert [len(s) for s in seg.flush(_eps(3), ctx)] == [3]  # <window: 단일 그룹, LLM 없음
    assert llm.calls == []


def test_batch_partitioner_llm_failure_falls_back_to_one_segment():
    seg = BatchPartitioner(window=2)
    ctx = type("C", (), {"llm": StubLLM({"extract": []})})()  # 응답 소진 → None
    out, rest = seg.push(_eps(2), ctx)
    assert [len(s) for s in out] == [2] and rest == []


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


# ---------------- SemanticIntegrator: Dedup / ThreeWay (v4 §3.3.3) ----------------


def _seed_semantic(mem, entries):
    """entries: list of (id, content) -> ADD each via the facade so both
    doc + vec stores are populated, exactly as a real Stage-3 ADD would."""
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
        actor="test",
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
