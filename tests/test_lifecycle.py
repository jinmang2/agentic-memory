"""Lifecycle contract: events, cursor, consolidate (spec §1)."""

from agmem.core.ops import MemoryOp, OpType
from agmem.organizers.base import MemoryEvent, Organizer, OrganizerContext


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
