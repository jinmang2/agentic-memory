"""Phase 0 exit criterion: add -> search works end-to-end with passthrough."""

import pytest

from agmem import AgenticMemory
from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.base import Organizer, OrganizerContext


@pytest.fixture
def mem():
    m = AgenticMemory(namespace="test", organizers=["passthrough"],
                      embedder=FakeEmbedder(dim=128))
    yield m
    m.close()


def test_add_and_search_end_to_end(mem):
    mem.add_message("I am planning a trip to Paris in October")
    mem.add_message("My cat's name is Mochi and she is 3 years old")
    mem.add_message("The Paris trip budget is around 3000 dollars")

    bundle = mem.search("trip to Paris", k=2)
    assert len(bundle.items) == 2
    contents = [s.item.content for s in bundle.items]
    assert all("Paris" in c for c in contents)

    rendered = bundle.render(budget_tokens=200)
    assert "Paris" in rendered and "Messages:" in rendered


def test_evolution_log_records_ingest(mem):
    mem.add_message("hello there")
    ops = mem.log.tail(5)
    assert len(ops) == 1
    assert ops[0].op is OpType.ADD
    assert ops[0].target_type == "episodic"
    assert ops[0].actor == "ingest"


def test_stats_shape(mem):
    mem.add_message("one")
    stats = mem.stats()
    assert stats["episodes"] == 1
    assert stats["vectors"] == 1
    assert stats["evolution_ops"] == 1
    assert stats["profile"] == "lite"


def test_capabilities_report(mem):
    report = mem.capabilities()
    assert report["active"]["organizers"] == ["passthrough"]
    assert "detected" in report and report["detected"]["cpu_cores"] >= 1


class StrategyStub(Organizer):
    """Emits one strategy op per task — exercises the derived-item path."""

    name = "stub"

    def on_task_end(self, trajectory, outcome, task, ctx: OrganizerContext):
        return [MemoryOp(
            op=OpType.ADD, target_type="strategies", target_id="strat-1",
            payload={"title": "Always check filters",
                     "content": f"When doing '{task}', verify filters first.",
                     "outcome": outcome},
        )]


def test_derived_items_are_searchable():
    mem = AgenticMemory(namespace="test", organizers=[StrategyStub()],
                        embedder=FakeEmbedder(dim=128))
    try:
        mem.add_task_result(trajectory=[{"step": 1}], outcome="success",
                            task="filter products by price")
        bundle = mem.search("how to filter products", memory_types=["strategies"], k=3)
        assert bundle.items
        assert bundle.items[0].item.data["title"] == "Always check filters"
        # op was logged with the organizer as actor
        actors = {op.actor for op in mem.log.tail(10)}
        assert "stub" in actors
    finally:
        mem.close()


def test_multi_type_search():
    mem = AgenticMemory(namespace="test", organizers=[StrategyStub()],
                        embedder=FakeEmbedder(dim=128))
    try:
        mem.add_message("we sell products with adjustable price filters")
        mem.add_task_result(trajectory=[], outcome="success",
                            task="filter products by price")
        bundle = mem.search("filter products", memory_types=["episodic", "strategies"], k=3)
        types = {s.memory_type for s in bundle.items}
        assert types == {"episodic", "strategies"}
    finally:
        mem.close()


def test_namespace_isolation():
    a = AgenticMemory(namespace="user-a", embedder=FakeEmbedder(dim=128))
    try:
        a.add_message("secret about user a")
        # same stores object is per-instance here; isolation is enforced by
        # namespace filters — simulate by querying a different namespace
        b_bundle = a.pipeline.search("secret", k=5, namespace="user-b")
        assert not b_bundle.items
    finally:
        a.close()


def test_unknown_organizer_raises():
    with pytest.raises(KeyError):
        AgenticMemory(organizers=["nope"], embedder=FakeEmbedder(dim=8))
