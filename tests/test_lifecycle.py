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


class Emitter(Organizer):
    name = "emitter"
    def on_message(self, ep, ctx):
        return [
            MemoryOp(op=OpType.MERGE, target_type="episodes", target_id="new",
                     payload={"id": "new", "content": "merged", "supersedes": ["old"]}),
            MemoryOp(op=OpType.INVALIDATE, target_type="episodes", target_id="old",
                     payload={"superseded_by": "new"}),
        ]


class Consumer(Organizer):
    name = "consumer"
    consumes = ("episodes",)
    def __init__(self):
        self.seen: list[MemoryEvent] = []
    def on_memory_event(self, ev, ctx):
        self.seen.append(ev)
        # depth=1 검증용: consumer도 episodes를 반환하지만 재전파되면 안 됨
        return [MemoryOp(op=OpType.ADD, target_type="episodes", target_id=f"d1-{len(self.seen)}",
                         payload={"id": f"d1-{len(self.seen)}", "content": "derived"})]


def test_event_propagation_supersedes_and_depth1():
    consumer = Consumer()
    mem = _mk(organizers=[Emitter(), consumer])
    mem.add_message("hi")
    assert len(consumer.seen) == 1            # MERGE만 전파 (INVALIDATE 비전파)
    ev = consumer.seen[0]
    assert (ev.op, ev.target_id, ev.supersedes) == (OpType.MERGE, "new", ("old",))
    assert ev.source == "emitter"
    # consumer의 반환 op는 적용됐지만 (자기 자신에게도) 재전파되지 않음
    assert mem.doc.get_items(["d1-1"], "episodes")
    assert len(consumer.seen) == 1


def test_no_self_delivery_and_consumes_filter():
    class SelfSub(Emitter):
        name = "selfsub"
        consumes = ("episodes",)
        def __init__(self): self.seen = []
        def on_memory_event(self, ev, ctx):
            self.seen.append(ev); return []
    org = SelfSub()
    mem = _mk(organizers=[org])
    mem.add_message("hi")
    assert org.seen == []  # 자기 이벤트 제외
