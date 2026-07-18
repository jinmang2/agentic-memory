"""Lifecycle contract: events, cursor, consolidate (spec §1)."""

from agmem.core.ops import MemoryOp, OpType
from agmem.organizers.base import MemoryEvent, Organizer, OrganizerContext
from helpers import StubLLM, make_mem_multi


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


def test_consolidate_api_applies_ops_and_cursor():
    class Cons(Organizer):
        name = "cons"
        def consolidate(self, ctx):
            end = ctx.doc.last_seq()
            return [MemoryOp(op=OpType.ADD, target_type="semantic", target_id="c1",
                             payload={"id": "c1", "content": "merged fact",
                                      "consolidated": True, "embedding_text": "merged fact"}),
                    self.cursor_op(end)]
    org = Cons()
    mem = _mk(organizers=[org])
    mem.add_message("test")  # add an episode to ensure last_seq() > 0
    n = mem.consolidate()
    assert n == 2
    assert mem.doc.get_items(["c1"], "semantic")
    assert org.read_cursor(mem._ctx) > 0
    # state 항목은 벡터를 만들지 않는다
    assert mem.vec.count() == 2  # one from episode, one from consolidated item


def test_consolidate_drains_async_queue_first():
    """Review I3: consolidate() runs on the caller's thread and scans the log,
    so it must first drain any pending async organizer work — otherwise a
    just-queued (not-yet-applied) fact is invisible to the cursor scan."""
    import time

    from agmem import AgenticMemory
    from agmem.config import AgmemConfig
    from agmem.embed.fake import FakeEmbedder

    class Cons(Organizer):
        name = "cons"

        def consolidate(self, ctx):
            return []

    mem = AgenticMemory(
        namespace="t",
        organizers=[Cons()],
        embedder=FakeEmbedder(dim=128),
        config=AgmemConfig(sync_write=False),
    )
    try:
        seen: list[int] = []

        def slow_work():
            time.sleep(0.05)
            seen.append(1)

        mem._dispatch(slow_work)  # queued on the background worker
        mem.consolidate()  # must join() before returning
        assert seen == [1]  # queue was drained before consolidate finished
    finally:
        mem.close()


# ---------------- MemoryOS input="episodes" consumer (Task 12) --------------


def test_memoryos_consumes_nemori_episodes():
    from agmem.organizers.memoryos import MemoryOSOrganizer
    from agmem.organizers.nemori import NemoriOrganizer

    llm = StubLLM(
        {
            "extract": [
                {"boundary": False, "confidence": 0.9},  # msg1: nothing to compare yet
                {"boundary": True, "confidence": 0.9},  # msg2: cut -> episode = [msg1]
            ],
            "distill": [
                {"title": "t", "narrative": "n", "timestamp": "2026-01-01"},  # episode
                {"facts": []},  # cold-start direct extract
                {"groups": [{"topic": "g", "summary": "s", "message_indexes": [0]}]},  # MemoryOS segment
            ],
        }
    )
    mos = MemoryOSOrganizer(stm_capacity=1, input="episodes")
    mem = make_mem_multi([NemoriOrganizer(fidelity="v1", buffer_min=1), mos], llm)
    mem.add_message("hello", meta={"date": "2026-01-01"})
    mem.add_message("new topic", meta={"date": "2026-01-01"})  # boundary -> episode flush
    pages = [
        o for o in mem.log.tail(30) if o.target_type == "pages" and o.actor == "memoryos"
    ]
    assert pages  # Nemori 에피소드가 MemoryOS page로 흘러들어감
    # 에피소드 원문이 아니라 Nemori 서사가 STM에 들어갔는지: page의 source가 episode id
    ep_ids = [
        o.target_id for o in mem.log.tail(30) if o.target_type == "episodes" and o.op == OpType.ADD
    ]
    assert set(pages[0].payload["source_episode_ids"]) <= set(ep_ids)


def test_memoryos_retires_superseded_units():
    from agmem.organizers.memoryos import MemoryOSOrganizer

    mos = MemoryOSOrganizer(stm_capacity=1, input="episodes")
    mem = _mk(organizers=[mos])
    # page화 유도: LLM 없음 → mechanical segment (explicit degradation 경로)
    mem._propagate_events(
        [
            MemoryOp(
                op=OpType.ADD,
                target_type="episodes",
                target_id="e1",
                payload={"id": "e1", "content": "ep one"},
            )
        ],
        actor="src",
    )
    pages = [o for o in mem.log.tail(10) if o.target_type == "pages"]
    assert len(pages) == 1
    # e1을 흡수한 MERGE 도착 → page의 유일 소스가 superseded → page INVALIDATE
    mem._propagate_events(
        [
            MemoryOp(
                op=OpType.MERGE,
                target_type="episodes",
                target_id="e2",
                payload={"id": "e2", "content": "merged", "supersedes": ["e1"]},
            )
        ],
        actor="src",
    )
    inv = [o for o in mem.log.tail(10) if o.target_type == "pages" and o.op == OpType.INVALIDATE]
    assert len(inv) == 1 and inv[0].payload["reason"] == "sources_superseded"


def test_memoryos_heat_eviction_drops_reverse_index():
    """Review finding: lowest-heat eviction in _evict_to_mtm popped
    self._heat but left _page_sources/_unit_pages entries for the evicted
    page dangling -> permanent leak, and a later supersedes on the
    evicted page's source could make _retire emit a stale INVALIDATE for
    an already-DELETEd page."""
    from agmem.organizers.memoryos import MemoryOSOrganizer

    llm = StubLLM(
        {
            "distill": [
                {"groups": [{"topic": "g1", "summary": "alpha", "keywords": [], "message_indexes": [0]}]},
                {"groups": [{"topic": "g2", "summary": "beta", "keywords": [], "message_indexes": [0]}]},
            ]
        }
    )
    mos = MemoryOSOrganizer(stm_capacity=1, mtm_capacity=1, input="episodes")
    mem = make_mem_multi([mos], llm)

    mem._propagate_events(
        [
            MemoryOp(
                op=OpType.ADD,
                target_type="episodes",
                target_id="e1",
                payload={"id": "e1", "content": "ep one"},
            )
        ],
        actor="src",
    )
    mem._propagate_events(
        [
            MemoryOp(
                op=OpType.ADD,
                target_type="episodes",
                target_id="e2",
                payload={"id": "e2", "content": "ep two"},
            )
        ],
        actor="src",
    )

    pages_add = [o for o in mem.log.tail(20) if o.target_type == "pages" and o.op == OpType.ADD]
    deletes = [o for o in mem.log.tail(20) if o.target_type == "pages" and o.op == OpType.DELETE]
    assert len(pages_add) == 2  # 두 page 생성
    assert len(deletes) == 1  # mtm_capacity=1 → 하나는 즉시 축출
    evicted_id = deletes[0].target_id

    # 축출된 page의 역인덱스 엔트리가 남아있으면 안 됨 (누수 재현)
    assert evicted_id not in mos._page_sources
    assert all(evicted_id not in pages for pages in mos._unit_pages.values())

    evicted_add = next(o for o in pages_add if o.target_id == evicted_id)
    evicted_source = evicted_add.payload["source_episode_ids"][0]

    # 축출된 page의 소스 episode를 supersede하는 MERGE → 이미 DELETE된
    # page에 대해 stale INVALIDATE가 나오면 안 됨
    mem._propagate_events(
        [
            MemoryOp(
                op=OpType.MERGE,
                target_type="episodes",
                target_id="e3",
                payload={"id": "e3", "content": "merged", "supersedes": [evicted_source]},
            )
        ],
        actor="src",
    )
    inv = [o for o in mem.log.tail(20) if o.target_type == "pages" and o.op == OpType.INVALIDATE]
    assert inv == []


# ---------------- A-Mem input="episodes" consumer (Task 13) -----------------


def test_amem_consumes_episodes_and_retires_notes():
    from agmem.organizers.amem import AMemOrganizer

    org = AMemOrganizer(input="episodes")
    mem = _mk(organizers=[org])  # LLM 없음 → bare note 경로 (explicit degradation)
    mem._propagate_events(
        [
            MemoryOp(
                op=OpType.ADD,
                target_type="episodes",
                target_id="e1",
                payload={"id": "e1", "content": "ep narrative", "timestamp": "2026-01-01"},
            )
        ],
        actor="src",
    )
    notes = [o for o in mem.log.tail(10) if o.target_type == "notes"]
    assert len(notes) == 1 and notes[0].payload["source_episode_ids"] == ["e1"]
    mem._propagate_events(
        [
            MemoryOp(
                op=OpType.MERGE,
                target_type="episodes",
                target_id="e2",
                payload={"id": "e2", "content": "merged", "supersedes": ["e1"]},
            )
        ],
        actor="src",
    )
    inv = [
        o for o in mem.log.tail(10) if o.target_type == "notes" and o.op == OpType.INVALIDATE
    ]
    assert len(inv) == 1 and inv[0].target_id == notes[0].target_id


# ---------------- M3 spec §5 test-gap closures --------------------------------


def _ev(op, tid, **payload):
    return MemoryOp(op=op, target_type="episodes", target_id=tid, payload={"id": tid, **payload})


def test_memoryos_partial_supersede_keeps_page_until_all_sources_gone():
    """M3(a): a page backed by 2 sources survives when only 1 source is
    superseded (_retire's srcs.discard leaves a non-empty set); only once the
    last source is superseded does the page INVALIDATE fire."""
    from agmem.organizers.memoryos import MemoryOSOrganizer

    mos = MemoryOSOrganizer(stm_capacity=2, input="episodes")
    mem = _mk(organizers=[mos])  # no LLM -> one mechanical page over the batch
    mem._propagate_events(
        [_ev(OpType.ADD, "e1", content="one"), _ev(OpType.ADD, "e2", content="two")],
        actor="src",
    )
    pages = [o for o in mem.log.tail(20) if o.target_type == "pages" and o.op == OpType.ADD]
    assert len(pages) == 1 and set(pages[0].payload["source_episode_ids"]) == {"e1", "e2"}

    # supersede only e1 -> page still backed by e2 -> no INVALIDATE
    mem._propagate_events(
        [_ev(OpType.MERGE, "m1", content="merged", supersedes=["e1"])], actor="src"
    )
    still = [o for o in mem.log.tail(20) if o.target_type == "pages" and o.op == OpType.INVALIDATE]
    assert still == []

    # supersede e2 too -> all sources gone -> page INVALIDATE
    mem._propagate_events(
        [_ev(OpType.MERGE, "m2", content="merged2", supersedes=["e2"])], actor="src"
    )
    inv = [o for o in mem.log.tail(20) if o.target_type == "pages" and o.op == OpType.INVALIDATE]
    assert len(inv) == 1 and inv[0].payload["reason"] == "sources_superseded"


def test_memoryos_update_replaces_stm_unit_then_ignores_when_paged():
    """M3(b): MemoryOS UPDATE replaces the unit while still in STM, but is a
    no-op once the unit has been paged (documented staleness, spec §3)."""
    from agmem.organizers.memoryos import MemoryOSOrganizer

    mos = MemoryOSOrganizer(stm_capacity=2, input="episodes")
    mem = _mk(organizers=[mos])
    mem._propagate_events([_ev(OpType.ADD, "e1", content="v1")], actor="src")
    assert [e.content for e in mos._stm] == ["v1"]
    mem._propagate_events([_ev(OpType.UPDATE, "e1", content="v2")], actor="src")
    assert [e.content for e in mos._stm] == ["v2"]  # replaced in place

    # fill capacity -> e1(v2)+e2 evicted into a page; STM drains
    mem._propagate_events([_ev(OpType.ADD, "e2", content="w1")], actor="src")
    assert mos._stm == []
    pages_before = [o for o in mem.log.tail(30) if o.target_type == "pages"]
    # UPDATE for the now-paged e1 must be ignored -> no new page op
    mem._propagate_events([_ev(OpType.UPDATE, "e1", content="v3")], actor="src")
    pages_after = [o for o in mem.log.tail(30) if o.target_type == "pages"]
    assert len(pages_after) == len(pages_before)


def test_amem_update_does_not_rewrite_note():
    """M3(b): A-Mem consuming episodes does not re-distill a note on UPDATE."""
    from agmem.organizers.amem import AMemOrganizer

    org = AMemOrganizer(input="episodes")
    mem = _mk(organizers=[org])  # no LLM -> bare note
    mem._propagate_events(
        [_ev(OpType.ADD, "e1", content="narrative", timestamp="2026-01-01")], actor="src"
    )
    adds = [o for o in mem.log.tail(10) if o.target_type == "notes" and o.op == OpType.ADD]
    assert len(adds) == 1
    mem._propagate_events([_ev(OpType.UPDATE, "e1", content="revised narrative")], actor="src")
    adds_after = [o for o in mem.log.tail(10) if o.target_type == "notes" and o.op == OpType.ADD]
    assert len(adds_after) == 1  # no rewrite: still just the original ADD


def test_delete_op_is_not_propagated_as_event():
    """M3(c): DELETE (like INVALIDATE) is never delivered as a MemoryEvent —
    only ADD/UPDATE/MERGE propagate to subscribed consumers."""
    consumer = Consumer()
    mem = _mk(organizers=[consumer])
    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.DELETE,
                target_type="episodes",
                target_id="e1",
                payload={"reason": "evicted"},
            )
        ],
        actor="src",
    )
    assert consumer.seen == []
