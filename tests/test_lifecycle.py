"""Lifecycle contract: events, cursor, consolidate (spec §1)."""

from agmem.core.ops import MemoryOp, OpType
from agmem.organizers.base import MemoryEvent, Organizer, OrganizerContext


def _mk(organizers=("passthrough",)):
    from agmem import AgenticMemory
    from agmem.embed.fake import FakeEmbedder
    return AgenticMemory(namespace="t", organizers=list(organizers),
                         embedder=FakeEmbedder(dim=128))


def test_base_defaults_are_noop():
    org = Organizer()
    assert org.consumes == ()
    ev = MemoryEvent(source="x", op=OpType.ADD, target_type="episodes",
                     target_id="e1", payload={})
    assert org.on_memory_event(ev, None) == []
    assert org.consolidate(None) == []
    assert ev.supersedes == ()


def test_cursor_helpers_roundtrip():
    from agmem.stores.sqlite_doc import SqliteDocStore

    class C(Organizer):
        name = "curs"

    org, doc = C(), SqliteDocStore(None)
    ctx = OrganizerContext(doc=doc, vec=None, embedder=None, namespace="t")
    assert org.read_cursor(ctx) == 0
    op = org.cursor_op(7)
    assert (op.op, op.target_type, op.target_id) == (OpType.UPDATE, "state", "consolidate:curs")
    doc.put_item("consolidate:curs", "state", "t", {"id": "consolidate:curs", "seq": 7})
    assert org.read_cursor(ctx) == 7


def test_invalidate_preserves_first_and_removes_vector():
    mem = _mk()
    mem._apply_ops([MemoryOp(op=OpType.ADD, target_type="semantic", target_id="s1",
                             payload={"id": "s1", "content": "fact", "embedding_text": "fact"})],
                   actor="t")
    assert mem.vec.count() == 1
    mem._apply_ops([MemoryOp(op=OpType.INVALIDATE, target_type="semantic", target_id="s1",
                             payload={"t_invalid": "2026-01-01T00:00:00",
                                      "superseded_by": "s2"})], actor="t")
    item = mem.doc.get_items(["s1"], "semantic")[0]
    assert item["invalid_at"] == "2026-01-01T00:00:00"
    assert item["superseded_by"] == "s2"
    assert mem.vec.count() == 0  # semantic은 bi-temporal 렌더 타입이 아님 → 벡터 제거
    # 이중 무효화: 최초 시각 보존, 예외 없음
    mem._apply_ops([MemoryOp(op=OpType.INVALIDATE, target_type="semantic", target_id="s1",
                             payload={"t_invalid": "2027-01-01T00:00:00"})], actor="t")
    assert mem.doc.get_items(["s1"], "semantic")[0]["invalid_at"] == "2026-01-01T00:00:00"


def test_invalidate_facts_keeps_vector():
    mem = _mk()
    mem._apply_ops([MemoryOp(op=OpType.ADD, target_type="facts", target_id="f1",
                             payload={"id": "f1", "content": "A는 B다", "embedding_text": "A는 B다"})],
                   actor="t")
    mem._apply_ops([MemoryOp(op=OpType.INVALIDATE, target_type="facts", target_id="f1",
                             payload={})], actor="t")
    assert mem.vec.count() == 1  # Zep bi-temporal: 무효화돼도 validity 렌더 대상
