"""Lifecycle contract: events, cursor, consolidate (spec §1)."""

from helpers import StubLLM, make_mem_multi

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode
from agmem.organizers.base import MemoryEvent, Organizer, OrganizerContext, overrides


def _mk(organizers=("passthrough",)):
    from agmem import AgenticMemory
    from agmem.embed.fake import FakeEmbedder

    return AgenticMemory(namespace="t", organizers=list(organizers), embedder=FakeEmbedder(dim=128))


def test_base_defaults_are_noop():
    org = Organizer()
    assert org.consumes == ()
    ev = MemoryEvent(source="x", op=OpType.ADD, target_type="episodes", target_id="e1", payload={})
    assert org.on_memory_event(ev, None) == []
    assert org.consolidate(None) == []
    assert ev.supersedes == ()


def test_cursor_helpers_roundtrip():
    from agmem.stores.sqlite_doc import SqliteDocStore

    class C(Organizer):
        name = "curs"

    org, doc = C(), SqliteDocStore(None)
    ctx = OrganizerContext(doc_store=doc, vector_store=None, embedder=None, namespace="t")
    assert org.read_cursor(ctx) == 0
    op = org.cursor_op(7)
    # ADD, not UPDATE: the cursor's entire state is `seq`, so a full replace is
    # right, and the first advance has no row to update. It was an UPDATE only
    # because `_apply_one` upserted, which is no longer true for UPDATE.
    assert (op.op, op.target_type, op.target_id) == (OpType.ADD, "state", "consolidate:curs")
    doc.put_item("consolidate:curs", "state", "t", {"id": "consolidate:curs", "seq": 7})
    assert org.read_cursor(ctx) == 7


def test_update_on_missing_item_is_not_an_upsert():
    """UPDATE used to upsert, so an UPDATE naming an absent id stored a fragment
    with no content and no provenance, which retrieval would then serve. The op
    is still logged — append happens before apply — so no history is lost."""
    mem = _mk()
    try:
        mem._apply_ops(
            [
                MemoryOp(
                    op=OpType.UPDATE, target_type="notes", target_id="ghost", payload={"helpful": 1}
                )
            ],
            actor="ace",
        )
        assert mem.doc_store.get_items(["ghost"], "notes") == []
        assert [(o.op, o.target_id) for o in mem.log.tail(5)] == [(OpType.UPDATE, "ghost")]
    finally:
        mem.close()


def test_merge_still_creates_its_target():
    """The narrowing above must not touch MERGE: a merge writes its result under
    a NEW id (Nemori emits MERGE(new) + INVALIDATE(absorbed)), so a missing
    target is the normal case for it."""
    mem = _mk()
    try:
        mem._apply_ops(
            [
                MemoryOp(
                    op=OpType.MERGE,
                    target_type="episodes",
                    target_id="merged1",
                    payload={"content": "two episodes combined", "supersedes": ["e1", "e2"]},
                )
            ],
            actor="nemori",
        )
        assert [d["id"] for d in mem.doc_store.get_items(["merged1"], "episodes")] == ["merged1"]
    finally:
        mem.close()


def test_cursor_advances_across_consolidate_cycles():
    """End-to-end guard for cursor_op switching to ADD: the cursor must still be
    created on the first advance and read back on the next cycle."""

    class Cur(Organizer):
        name = "cur"

        def consolidate(self, ctx):
            return [self.cursor_op(self.read_cursor(ctx) + 5)]

    mem = _mk([Cur()])
    try:
        seen = []
        for _ in range(3):
            mem.consolidate()
            seen.append(mem.doc_store.get_items(["consolidate:cur"], "state")[0]["seq"])
        assert seen == [5, 10, 15]
    finally:
        mem.close()


def test_on_retrieval_ops_do_not_wake_subscribers():
    """``on_retrieval`` must be cheap because it runs inline on the read path,
    but that contract binds only the hook: propagated ops become MemoryEvents,
    and a subscriber's handler may call an LLM (ChainedConsumer feeds the wrapped
    organizer's on_message). The ops are still applied — only the fan-out stops."""

    class Reader(Organizer):
        name = "reader"
        produces = ("pages",)

        def on_retrieval(self, hits, ctx):
            return [
                MemoryOp(
                    op=OpType.ADD,
                    target_type="pages",
                    target_id="p1",
                    payload={"content": "heat bump"},
                )
            ]

    class Sub(Organizer):
        name = "sub"
        consumes = ("pages",)

        def __init__(self):
            self.seen = 0

        def on_memory_event(self, ev, ctx):
            self.seen += 1
            return []

    sub = Sub()
    mem = _mk([Reader(), sub])
    try:
        mem.add_message("hello there")
        mem.search("hello", memory_types=["episodic"])
        assert sub.seen == 0
        assert [d["id"] for d in mem.doc_store.get_items(["p1"], "pages")] == ["p1"]
    finally:
        mem.close()


def test_invalidate_preserves_first_and_removes_vector():
    mem = _mk()
    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.ADD,
                target_type="semantic",
                target_id="s1",
                payload={"id": "s1", "content": "fact", "embedding_text": "fact"},
            )
        ],
        actor="t",
    )
    assert mem.vector_store.count() == 1
    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.INVALIDATE,
                target_type="semantic",
                target_id="s1",
                payload={"t_invalid": "2026-01-01T00:00:00", "superseded_by": "s2"},
            )
        ],
        actor="t",
    )
    item = mem.doc_store.get_items(["s1"], "semantic")[0]
    assert item["invalid_at"] == "2026-01-01T00:00:00"
    assert item["superseded_by"] == "s2"
    assert mem.vector_store.count() == 0  # semantic은 bi-temporal 렌더 타입이 아님 → 벡터 제거
    # 이중 무효화: 최초 시각 보존, 예외 없음
    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.INVALIDATE,
                target_type="semantic",
                target_id="s1",
                payload={"t_invalid": "2027-01-01T00:00:00"},
            )
        ],
        actor="t",
    )
    assert mem.doc_store.get_items(["s1"], "semantic")[0]["invalid_at"] == "2026-01-01T00:00:00"


def test_invalidate_facts_keeps_vector():
    mem = _mk()
    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.ADD,
                target_type="facts",
                target_id="f1",
                payload={"id": "f1", "content": "A는 B다", "embedding_text": "A는 B다"},
            )
        ],
        actor="t",
    )
    mem._apply_ops(
        [MemoryOp(op=OpType.INVALIDATE, target_type="facts", target_id="f1", payload={})], actor="t"
    )
    assert mem.vector_store.count() == 1  # Zep bi-temporal: 무효화돼도 validity 렌더 대상


class Emitter(Organizer):
    name = "emitter"

    def on_message(self, episode, ctx):
        return [
            MemoryOp(
                op=OpType.MERGE,
                target_type="episodes",
                target_id="new",
                payload={"id": "new", "content": "merged", "supersedes": ["old"]},
            ),
            MemoryOp(
                op=OpType.INVALIDATE,
                target_type="episodes",
                target_id="old",
                payload={"superseded_by": "new"},
            ),
        ]


class Consumer(Organizer):
    name = "consumer"
    consumes = ("episodes",)

    def __init__(self):
        self.seen: list[MemoryEvent] = []

    def on_memory_event(self, ev, ctx):
        self.seen.append(ev)
        # depth=1 검증용: consumer도 episodes를 반환하지만 재전파되면 안 됨
        return [
            MemoryOp(
                op=OpType.ADD,
                target_type="episodes",
                target_id=f"d1-{len(self.seen)}",
                payload={"id": f"d1-{len(self.seen)}", "content": "derived"},
            )
        ]


def test_event_propagation_supersedes_and_depth1():
    consumer = Consumer()
    mem = _mk(organizers=[Emitter(), consumer])
    mem.add_message("hi")
    assert len(consumer.seen) == 1  # MERGE만 전파 (INVALIDATE 비전파)
    ev = consumer.seen[0]
    assert (ev.op, ev.target_id, ev.supersedes) == (OpType.MERGE, "new", ("old",))
    assert ev.source == "emitter"
    # consumer의 반환 op는 적용됐지만 (자기 자신에게도) 재전파되지 않음
    assert mem.doc_store.get_items(["d1-1"], "episodes")
    assert len(consumer.seen) == 1


def test_no_self_delivery_and_consumes_filter():
    class SelfSub(Emitter):
        name = "selfsub"
        consumes = ("episodes",)

        def __init__(self):
            self.seen = []

        def on_memory_event(self, ev, ctx):
            self.seen.append(ev)
            return []

    org = SelfSub()
    mem = _mk(organizers=[org])
    mem.add_message("hi")
    assert org.seen == []  # 자기 이벤트 제외


def test_consolidate_api_applies_ops_and_cursor():
    class Cons(Organizer):
        name = "cons"

        def consolidate(self, ctx):
            end = ctx.doc_store.last_seq()
            return [
                MemoryOp(
                    op=OpType.ADD,
                    target_type="semantic",
                    target_id="c1",
                    payload={
                        "id": "c1",
                        "content": "merged fact",
                        "consolidated": True,
                        "embedding_text": "merged fact",
                    },
                ),
                self.cursor_op(end),
            ]

    org = Cons()
    mem = _mk(organizers=[org])
    mem.add_message("test")  # add an episode to ensure last_seq() > 0
    n = mem.consolidate()
    assert n == 2
    assert mem.doc_store.get_items(["c1"], "semantic")
    assert org.read_cursor(mem._ctx) > 0
    # state 항목은 벡터를 만들지 않는다
    assert mem.vector_store.count() == 2  # one from episode, one from consolidated item


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


# ---- hook contract: every hook lives on the base, probed via overrides() -------


def test_buffer_and_chaining_hooks_default_to_noop():
    """flush_buffer/retire/patch_unit are declared on Organizer, so callers
    invoke them unconditionally instead of getattr-probing."""
    org = Organizer()
    assert org.flush_buffer(None) == []
    assert org.retire({"e1"}) == []
    assert org.patch_unit(Episode(content="x")) is None


def test_overrides_separates_real_impl_from_inherited_noop():
    class Buffered(Organizer):
        name = "buffered"

        def flush_buffer(self, ctx):
            return []

    assert overrides(Buffered(), "flush_buffer")
    assert not overrides(Buffered(), "retire")  # inherited no-op
    assert not overrides(Organizer(), "flush_buffer")


def test_flush_drains_every_organizers_buffer():
    """flush() applies each organizer's flush_buffer ops under its own name."""

    class Buffered(Organizer):
        name = "buffered"

        def flush_buffer(self, ctx):
            return [
                MemoryOp(
                    op=OpType.ADD,
                    target_type="semantic",
                    target_id="tail",
                    payload={"id": "tail", "content": "tail fact", "embedding_text": "tail fact"},
                )
            ]

    mem = _mk(organizers=[Buffered(), Organizer()])  # base organizer contributes nothing
    mem.flush()
    stored = mem.doc_store.get_items(["tail"], "semantic")
    assert stored and stored[0]["content"] == "tail fact"
    assert mem.doc_store.tail(1)[0].actor == "buffered"


# ---- consolidate cursor scoping ------------------------------------------------


def test_single_instance_keeps_bare_cursor_key():
    """One instance per name -> unsuffixed key, so cursors persisted before
    instance scoping existed still resolve."""

    class Solo(Organizer):
        name = "solo"

    org = Solo()
    mem = _mk(organizers=[org])
    assert org.cursor_key == "consolidate:solo"
    assert mem.organizers[0].cursor_key == "consolidate:solo"


def test_duplicate_organizers_get_distinct_cursor_keys():
    """Two instances of one organizer would otherwise share a cursor id and
    clobber each other's consolidate progress (base.cursor_key)."""

    class Twin(Organizer):
        name = "twin"

    a, b, other = Twin(), Twin(), Organizer()
    mem = _mk(organizers=[a, b, other])
    assert (a.cursor_key, b.cursor_key) == ("consolidate:twin#0", "consolidate:twin#1")
    assert other.cursor_key == "consolidate:base"  # unique name stays bare
    # the keys must actually separate persisted progress, not just differ
    mem._apply_ops([a.cursor_op(5)], actor="twin")
    mem._apply_ops([b.cursor_op(9)], actor="twin")
    assert (a.read_cursor(mem._ctx), b.read_cursor(mem._ctx)) == (5, 9)


# ---- op attribution is non-destructive ----------------------------------------


def test_apply_ops_does_not_mutate_callers_ops():
    """Organizers keep references to the ops they return (Nemori's within-batch
    supersession guard), so attribution must be stamped on copies."""
    mem = _mk()
    op = MemoryOp(
        op=OpType.ADD,
        target_type="semantic",
        target_id="s1",
        payload={"id": "s1", "content": "fact", "embedding_text": "fact"},
    )
    mem._apply_ops([op], actor="stamped")
    assert op.actor == "system"  # caller's object untouched
    assert mem.doc_store.tail(1)[0].actor == "stamped"  # log carries the actor


# ---- produces -> default_memory_types -----------------------------------------


def test_default_memory_types_leads_with_episodic():
    """Raw episodes are written by the facade, not by any organizer, so they are
    always searchable and always first."""
    assert _mk(organizers=["passthrough"]).default_memory_types == ("episodic",)


def test_default_memory_types_follows_the_active_organizers():
    from agmem.organizers.amem import AMemOrganizer
    from agmem.organizers.nemori import NemoriOrganizer

    mem = _mk(organizers=[AMemOrganizer()])
    assert mem.default_memory_types == ("episodic", "notes")
    mem = _mk(organizers=[NemoriOrganizer(), AMemOrganizer()])
    assert mem.default_memory_types == ("episodic", "episodes", "semantic", "notes")


def test_default_memory_types_dedupes_shared_types():
    """Nemori and MemoryOS both write "semantic"; it must appear once."""
    from agmem.organizers.memoryos import MemoryOSOrganizer
    from agmem.organizers.nemori import NemoriOrganizer

    mem = _mk(organizers=[NemoriOrganizer(), MemoryOSOrganizer()])
    assert mem.default_memory_types == ("episodic", "episodes", "semantic", "pages")


def test_chained_consumer_forwards_produces():
    from agmem.organizers.amem import AMemOrganizer
    from agmem.organizers.experimental import ChainedConsumer
    from agmem.organizers.nemori import NemoriOrganizer

    mem = _mk(
        organizers=[NemoriOrganizer(), ChainedConsumer(AMemOrganizer(), "episodes")],
    )
    assert mem.default_memory_types == ("episodic", "episodes", "semantic", "notes")


def test_zep_declares_facts_before_entities():
    """Order is the read order: the entities step pulls incident edge facts and
    dedupes against ids already in the bundle, so facts must come first or the
    same fact is served twice."""
    from agmem.organizers.zep_graph import ZepGraphOrganizer

    assert ZepGraphOrganizer.produces == ("facts", "entities")


def test_explicit_memory_types_override_the_default():
    """Paper-faithful configs stay methodology-pure by naming types explicitly."""
    from agmem.organizers.amem import AMemOrganizer

    mem = _mk(organizers=[AMemOrganizer()])
    mem.add_message("paris museums")  # no LLM -> A-Mem stores a bare note
    assert mem.default_memory_types == ("episodic", "notes")
    # notes-only: the raw episode must not leak into an A-Mem paper run
    assert [s.memory_type for s in mem.search("paris", memory_types=("notes",)).items] == ["notes"]
    # the default now reaches the note the old ("episodic",) default hid
    assert [s.memory_type for s in mem.search("paris").items] == ["episodic", "notes"]


# ---------------- MemoryOS via ChainedConsumer (Task 12, experimental) ------


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
                {
                    "groups": [{"topic": "g", "summary": "s", "message_indexes": [0]}]
                },  # MemoryOS segment
            ],
        }
    )
    from agmem.organizers.experimental import ChainedConsumer

    mos = ChainedConsumer(MemoryOSOrganizer(stm_capacity=1), "episodes")
    mem = make_mem_multi([NemoriOrganizer(fidelity="v1", buffer_min=1), mos], llm)
    mem.add_message("hello", meta={"date": "2026-01-01"})
    mem.add_message("new topic", meta={"date": "2026-01-01"})  # boundary -> episode flush
    pages = [o for o in mem.log.tail(30) if o.target_type == "pages" and o.actor == "memoryos"]
    assert pages  # Nemori 에피소드가 MemoryOS page로 흘러들어감
    # 에피소드 원문이 아니라 Nemori 서사가 STM에 들어갔는지: page의 source가 episode id
    ep_ids = [
        o.target_id for o in mem.log.tail(30) if o.target_type == "episodes" and o.op == OpType.ADD
    ]
    assert set(pages[0].payload["source_episode_ids"]) <= set(ep_ids)


def test_memoryos_retires_superseded_units():
    from agmem.organizers.experimental import ChainedConsumer
    from agmem.organizers.memoryos import MemoryOSOrganizer

    mos = ChainedConsumer(MemoryOSOrganizer(stm_capacity=1), "episodes")
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
                {
                    "groups": [
                        {"topic": "g1", "summary": "alpha", "keywords": [], "message_indexes": [0]}
                    ]
                },
                {
                    "groups": [
                        {"topic": "g2", "summary": "beta", "keywords": [], "message_indexes": [0]}
                    ]
                },
            ]
        }
    )
    from agmem.organizers.experimental import ChainedConsumer

    inner = MemoryOSOrganizer(stm_capacity=1, mtm_capacity=1)
    mos = ChainedConsumer(inner, "episodes")
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
    assert evicted_id not in inner._page_sources
    assert all(evicted_id not in pages for pages in inner._unit_pages.values())

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


# ---------------- A-Mem via ChainedConsumer (Task 13, experimental) ---------


def test_amem_consumes_episodes_and_retires_notes():
    from agmem.organizers.amem import AMemOrganizer
    from agmem.organizers.experimental import ChainedConsumer

    org = ChainedConsumer(AMemOrganizer(), "episodes")
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
    inv = [o for o in mem.log.tail(10) if o.target_type == "notes" and o.op == OpType.INVALIDATE]
    assert len(inv) == 1 and inv[0].target_id == notes[0].target_id


# ---------------- M3 spec §5 test-gap closures --------------------------------


def _ev(op, tid, **payload):
    return MemoryOp(op=op, target_type="episodes", target_id=tid, payload={"id": tid, **payload})


def test_memoryos_partial_supersede_keeps_page_until_all_sources_gone():
    """M3(a): a page backed by 2 sources survives when only 1 source is
    superseded (_retire's source_ids.discard leaves a non-empty set); only
    once the last source is superseded does the page INVALIDATE fire."""
    from agmem.organizers.experimental import ChainedConsumer
    from agmem.organizers.memoryos import MemoryOSOrganizer

    mos = ChainedConsumer(MemoryOSOrganizer(stm_capacity=2), "episodes")
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
    from agmem.organizers.experimental import ChainedConsumer
    from agmem.organizers.memoryos import MemoryOSOrganizer

    inner = MemoryOSOrganizer(stm_capacity=2)
    mos = ChainedConsumer(inner, "episodes")
    mem = _mk(organizers=[mos])
    mem._propagate_events([_ev(OpType.ADD, "e1", content="v1")], actor="src")
    assert [e.content for e in inner._stm] == ["v1"]
    mem._propagate_events([_ev(OpType.UPDATE, "e1", content="v2")], actor="src")
    assert [e.content for e in inner._stm] == ["v2"]  # replaced in place

    # fill capacity -> e1(v2)+e2 evicted into a page; STM drains
    mem._propagate_events([_ev(OpType.ADD, "e2", content="w1")], actor="src")
    assert inner._stm == []
    pages_before = [o for o in mem.log.tail(30) if o.target_type == "pages"]
    # UPDATE for the now-paged e1 must be ignored -> no new page op
    mem._propagate_events([_ev(OpType.UPDATE, "e1", content="v3")], actor="src")
    pages_after = [o for o in mem.log.tail(30) if o.target_type == "pages"]
    assert len(pages_after) == len(pages_before)


def test_amem_update_does_not_rewrite_note():
    """M3(b): A-Mem consuming episodes does not re-distill a note on UPDATE."""
    from agmem.organizers.amem import AMemOrganizer
    from agmem.organizers.experimental import ChainedConsumer

    org = ChainedConsumer(AMemOrganizer(), "episodes")
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


# ---------------- Nemori K -> A-Mem (v4 Table 7, experimental) ---------------


def _k_to_amem(batch_key=None):
    """Nemori's distilled knowledge K fed to A-Mem in place of raw messages."""
    from agmem.organizers.amem import AMemOrganizer
    from agmem.organizers.experimental import ChainedConsumer

    consumer = ChainedConsumer(AMemOrganizer(), "semantic", batch_key=batch_key)
    mem = _mk(organizers=[consumer])
    return mem, consumer


def _emit_facts(mem, facts, episode_id):
    """Emit semantic ADDs the way Nemori's integrators do (payload carries
    episode_id, which is the batch key)."""
    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.ADD,
                target_type="semantic",
                target_id=fact_id,
                payload={
                    "id": fact_id,
                    "content": content,
                    "episode_id": episode_id,
                    "source_episode_ids": [episode_id],
                    "embedding_text": content,
                },
            )
            for fact_id, content in facts
        ],
        actor="nemori",
    )


def test_k_to_amem_per_fact_makes_one_note_each():
    mem, _ = _k_to_amem()
    _emit_facts(mem, [("f1", "Alice is a curator."), ("f2", "Alice prefers trains.")], "ep1")
    notes = mem.doc_store.list_items("notes", namespace="t")
    assert sorted(n["content"] for n in notes) == ["Alice is a curator.", "Alice prefers trains."]


def test_k_to_amem_batched_groups_one_episodes_facts():
    """A batch closes when the episode_id changes; the tail needs flush()."""
    mem, _ = _k_to_amem(batch_key="episode_id")
    _emit_facts(mem, [("f1", "Alice is a curator."), ("f2", "Alice prefers trains.")], "ep1")
    assert mem.doc_store.list_items("notes", namespace="t") == []  # still accumulating
    _emit_facts(mem, [("f3", "Bob lives in Seoul.")], "ep2")  # key change closes ep1
    notes = mem.doc_store.list_items("notes", namespace="t")
    assert [n["content"] for n in notes] == ["Alice is a curator.\nAlice prefers trains."]
    # provenance points at the upstream episode, not at an individual fact
    assert notes[0]["source_episode_ids"] == ["ep1"]
    mem.flush()  # ep2 has no following event to close it
    assert len(mem.doc_store.list_items("notes", namespace="t")) == 2


def test_k_to_amem_batched_retires_only_when_all_facts_superseded():
    mem, consumer = _k_to_amem(batch_key="episode_id")
    _emit_facts(mem, [("f1", "Alice is a curator."), ("f2", "Alice prefers trains.")], "ep1")
    mem.flush()
    note_id = mem.doc_store.list_items("notes", namespace="t")[0]["id"]
    assert consumer._retire({"f1"}) == []  # partially absorbed -> note stands
    retired = consumer._retire({"f2"})
    assert [(o.op, o.target_id) for o in retired] == [(OpType.INVALIDATE, note_id)]


# ---------------- previously-deferred bugs (roadmap #9 / #10) ----------------


def test_chained_flush_drains_the_wrapped_organizers_buffer():
    """``ChainedConsumer.flush_buffer`` used to stop at the adapter, so a
    chained MemoryOS's partial STM tail was stranded forever.

    ``stm_capacity=3`` with a single fed unit means no capacity trigger ever
    fires, which is exactly the tail case: only the flush can emit it."""
    from agmem.organizers.experimental import ChainedConsumer
    from agmem.organizers.memoryos import MemoryOSOrganizer

    mos = ChainedConsumer(MemoryOSOrganizer(stm_capacity=3), "episodes")
    mem = _mk(organizers=[mos])
    try:
        mem._apply_ops(
            [
                MemoryOp(
                    op=OpType.ADD,
                    target_type="episodes",
                    target_id="ep1",
                    payload={"id": "ep1", "content": "a narrated episode", "date": "2026-01-01"},
                )
            ],
            actor="nemori",
        )
        assert mos.wrapped._stm, "unit is buffered below capacity — nothing emitted yet"
        assert [o for o in mem.log.tail(30) if o.target_type == "pages"] == []

        mem.flush()

        pages = [o for o in mem.log.tail(30) if o.target_type == "pages"]
        assert pages, "flush must reach the wrapped organizer's buffer"
        assert not mos.wrapped._stm
    finally:
        mem.close()


def test_warm_start_logs_ingest_ops_like_add_message():
    """Backfilled episodes used to carry no ingest ADD op, which made the
    evolution log an incomplete record of the stores — replaying it could not
    rebuild them."""
    mem = _mk()
    try:
        mem.warm_start([Episode(content="backfilled one"), Episode(content="backfilled two")])
        ingest = [
            o
            for o in mem.log.tail(30)
            if o.actor == "ingest" and o.target_type == "episodic" and o.op is OpType.ADD
        ]
        assert len(ingest) == 2
        assert all(o.payload.get("warm_start") for o in ingest)  # distinguishable from live ingest

        mem.add_message("live traffic")
        live = [
            o for o in mem.log.tail(30) if o.actor == "ingest" and not o.payload.get("warm_start")
        ]
        assert len(live) == 1
    finally:
        mem.close()


def test_consolidate_cursor_is_the_only_deterministic_id_and_never_gets_a_vector():
    """Roadmap #11 (doc PK is ``(id, memory_type)``, every vector backend
    upserts by ``item_id`` alone) is deferred on the grounds that real ids are
    uuid4, so a cross-type collision cannot happen. The consolidate cursor is
    the one deterministic id in the codebase — pin that it stays out of the
    vector store, so the premise cannot rot silently."""
    from agmem.organizers.nemori import NemoriOrganizer

    org = NemoriOrganizer()
    mem = _mk(organizers=[org])
    try:
        before = mem.vector_store.count()
        mem._apply_ops([org.cursor_op(7)], actor="nemori")
        assert org.read_cursor(mem._ctx) == 7
        assert mem.vector_store.count() == before, "state items must never be indexed"
    finally:
        mem.close()
