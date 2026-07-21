"""Phase 0 exit criterion: add -> search works end-to-end with passthrough."""

import pytest

from agmem import AgenticMemory
from agmem.core.ops import MemoryOp, OpType
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


def test_delete_op_leaves_no_ghost_hit(mem):
    """round-5 X1: DELETE must remove the vector too, not just tombstone."""
    add = MemoryOp(op=OpType.ADD, target_type="strategies", target_id="s1",
                   payload={"id": "s1", "title": "T", "content": "verify filters",
                            "embedding_text": "verify filters"})
    mem._apply_ops([add], actor="test")
    assert len(mem.search("verify filters", memory_types=["strategies"]).items) == 1

    mem._apply_ops([MemoryOp(op=OpType.DELETE, target_type="strategies",
                             target_id="s1")], actor="test")
    bundle = mem.search("verify filters", memory_types=["strategies"])
    assert bundle.items == []
    assert mem.vector_store.get(["s1"]) == {}


def test_strategy_description_rendered(mem):
    """round-5 X3: description must survive into the injected context."""
    add = MemoryOp(op=OpType.ADD, target_type="strategies", target_id="s2",
                   payload={"id": "s2", "title": "Re-read errors",
                            "description": "Use after any failed action",
                            "content": "Error text names the missing field.",
                            "embedding_text": "Re-read errors"})
    mem._apply_ops([add], actor="test")
    rendered = mem.search("errors", memory_types=["strategies"]).render()
    assert "Use after any failed action" in rendered


def test_invalidated_fact_renders_date_range(mem):
    """round-5 X2: bi-temporal facts must expose their validity range."""
    add = MemoryOp(op=OpType.ADD, target_type="facts", target_id="f1",
                   payload={"id": "f1", "content": "Alice lives in Paris",
                            "valid_at": "2024-01-01",
                            "embedding_text": "Alice lives in Paris"})
    mem._apply_ops([add], actor="test")
    mem._apply_ops([MemoryOp(op=OpType.INVALIDATE, target_type="facts",
                             target_id="f1",
                             payload={"t_invalid": "2025-06-01"})], actor="test")
    rendered = mem.search("Alice Paris", memory_types=["facts"]).render()
    assert "Date range: 2024-01-01 - 2025-06-01" in rendered
