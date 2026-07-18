from helpers import StubLLM

from agmem import AgenticMemory
from agmem.core.ops import OpType
from agmem.core.types import Episode
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.nemori import NemoriOrganizer
from agmem.organizers.nemori_stages import BatchPartitioner


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
