from helpers import StubLLM

from agmem.core.types import Episode
from agmem.organizers.nemori_stages import BatchPartitioner


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
