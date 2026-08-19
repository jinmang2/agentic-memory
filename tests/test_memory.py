"""Phase 0 exit criterion: add -> search works end-to-end with passthrough."""

import gc
import threading
import weakref

import pytest

from agmem import AgenticMemory
from agmem.config import AgmemConfig
from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Bullet
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.base import Organizer, OrganizerContext


@pytest.fixture
def mem():
    m = AgenticMemory(namespace="test", organizers=["passthrough"], embedder=FakeEmbedder(dim=128))
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
        return [
            MemoryOp(
                op=OpType.ADD,
                target_type="strategies",
                target_id="strat-1",
                payload={
                    "title": "Always check filters",
                    "content": f"When doing '{task}', verify filters first.",
                    "outcome": outcome,
                },
            )
        ]


def test_derived_items_are_searchable():
    mem = AgenticMemory(
        namespace="test", organizers=[StrategyStub()], embedder=FakeEmbedder(dim=128)
    )
    try:
        mem.add_task_result(
            trajectory=[{"step": 1}], outcome="success", task="filter products by price"
        )
        bundle = mem.search("how to filter products", memory_types=["strategies"], k=3)
        assert bundle.items
        assert bundle.items[0].item.data["title"] == "Always check filters"
        # op was logged with the organizer as actor
        actors = {op.actor for op in mem.log.tail(10)}
        assert "stub" in actors
    finally:
        mem.close()


def test_multi_type_search():
    mem = AgenticMemory(
        namespace="test", organizers=[StrategyStub()], embedder=FakeEmbedder(dim=128)
    )
    try:
        mem.add_message("we sell products with adjustable price filters")
        mem.add_task_result(trajectory=[], outcome="success", task="filter products by price")
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


def test_profile_arg_conflicting_with_config_raises():
    """``profile=`` used to be dropped whenever ``config=`` was given, so every
    backend slot — and ``resolved_embed_model`` — resolved from the config's
    profile while the caller believed it had asked for another one, with no
    trace of the request left in ``stats()`` or ``capabilities()``."""
    with pytest.raises(ValueError, match="conflicts with config.profile"):
        AgenticMemory(
            profile="full", config=AgmemConfig(profile="lite"), embedder=FakeEmbedder(dim=8)
        )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, "lite"),
        ({"profile": "standard"}, "standard"),
        # agreement is allowed: docs/05 §1's example passes both, redundantly
        ({"profile": "lite", "config": AgmemConfig(profile="lite")}, "lite"),
        ({"config": AgmemConfig(profile="standard")}, "standard"),
    ],
)
def test_profile_resolution(kwargs, expected):
    mem = AgenticMemory(embedder=FakeEmbedder(dim=8), **kwargs)
    try:
        assert mem.config.profile == expected
        assert mem.stats()["profile"] == expected
    finally:
        mem.close()


def test_consolidate_drains_buffered_units_first():
    """An organizer's consolidate() resumes from the evolution log, so units
    still sitting in a segment/STM buffer are invisible to it — they have
    produced no ops yet. ``consolidate()`` alone used to scan an empty log and
    report success (0 ops, 0 items for three buffered messages)."""

    class Buffered(Organizer):
        name = "buffered"
        produces = ("semantic",)

        def __init__(self):
            self.buf = []

        def on_message(self, episode, ctx):
            self.buf.append(episode)
            return []

        def flush_buffer(self, ctx):
            ops = [
                MemoryOp(
                    op=OpType.ADD,
                    target_type="semantic",
                    target_id=f"b{i}",
                    payload={"content": e.content},
                )
                for i, e in enumerate(self.buf)
            ]
            self.buf = []
            return ops

    mem = AgenticMemory(namespace="t", organizers=[Buffered()], embedder=FakeEmbedder(dim=64))
    try:
        for text in ("one", "two", "three"):
            mem.add_message(text)
        assert mem.consolidate() == 3  # buffer drain is counted in the applied ops
        assert len(mem.doc_store.list_items("semantic", namespace="t")) == 3
    finally:
        mem.close()


def test_get_playbook_requires_an_active_producer():
    """Reading the playbook was type-owned while writing to it is producer-owned
    (docs/04 §3.4), so with ACE deconfigured one session would render a bullet
    and then return 0 from the feedback call meant to update it."""
    bullet = MemoryOp(
        op=OpType.ADD,
        target_type="playbook",
        target_id="p1",
        payload={"content": "check the log", "section": "ops"},
    )

    inactive = AgenticMemory(
        namespace="t", organizers=["passthrough"], embedder=FakeEmbedder(dim=64)
    )
    try:
        inactive._apply_ops([bullet], actor="ace")
        assert inactive.get_playbook() == ""
        assert inactive.report_feedback(["p1"], helpful=True) == 0
    finally:
        inactive.close()

    active = AgenticMemory(namespace="t2", organizers=["ace"], embedder=FakeEmbedder(dim=64))
    try:
        active._apply_ops([bullet], actor="ace")
        assert active.get_playbook() == "## ops\n[ops-p1] helpful=0 harmful=0 :: check the log"
        assert active.report_feedback(["p1"], helpful=True) == 1
    finally:
        active.close()


def test_playbook_line_format_is_shared_with_bullet_render():
    """The facade rendered bullets from stored dicts with its own copy of the
    f-string; the format is part of ACE's prompt contract, so one copy."""
    b = Bullet(content="check the log", section="ops", id="p1abcdef", helpful=2, harmful=1)
    mem = AgenticMemory(namespace="t", organizers=["ace"], embedder=FakeEmbedder(dim=64))
    try:
        mem._apply_ops(
            [
                MemoryOp(
                    op=OpType.ADD,
                    target_type="playbook",
                    target_id=b.id,
                    payload={
                        "content": b.content,
                        "section": b.section,
                        "helpful": b.helpful,
                        "harmful": b.harmful,
                    },
                )
            ],
            actor="ace",
        )
        assert b.render() in mem.get_playbook()
    finally:
        mem.close()


def test_close_stops_the_write_worker_and_releases_the_memory():
    """``close()`` left ``_drain`` looping forever on a daemon thread, and
    ``Thread(target=self._drain)`` held the bound method — so a closed memory
    (embedder, stores, organizers) stayed resident for the process lifetime.
    Latent only while ``sync_write`` defaults to True; a per-question or
    per-config benchmark loop would retain one memory per instance."""
    mem = AgenticMemory(
        namespace="t",
        organizers=["passthrough"],
        config=AgmemConfig(sync_write=False),
        embedder=FakeEmbedder(dim=64),
    )
    mem.add_message("hello")
    mem.flush()
    ref = weakref.ref(mem)

    mem.close()
    mem.close()  # idempotent

    assert not [t for t in threading.enumerate() if t.name == "agmem-worker"]
    del mem
    gc.collect()
    assert ref() is None


def test_context_manager_closes():
    with AgenticMemory(
        namespace="t",
        organizers=["passthrough"],
        config=AgmemConfig(sync_write=False),
        embedder=FakeEmbedder(dim=64),
    ) as mem:
        mem.add_message("hello")
    assert not [t for t in threading.enumerate() if t.name == "agmem-worker"]


def test_delete_op_leaves_no_ghost_hit(mem):
    """round-5 X1: DELETE must remove the vector too, not just tombstone."""
    add = MemoryOp(
        op=OpType.ADD,
        target_type="strategies",
        target_id="s1",
        payload={
            "id": "s1",
            "title": "T",
            "content": "verify filters",
            "embedding_text": "verify filters",
        },
    )
    mem._apply_ops([add], actor="test")
    assert len(mem.search("verify filters", memory_types=["strategies"]).items) == 1

    mem._apply_ops(
        [MemoryOp(op=OpType.DELETE, target_type="strategies", target_id="s1")], actor="test"
    )
    bundle = mem.search("verify filters", memory_types=["strategies"])
    assert bundle.items == []
    assert mem.vector_store.get(["s1"]) == {}


def test_strategy_description_rendered(mem):
    """round-5 X3: description must survive into the injected context."""
    add = MemoryOp(
        op=OpType.ADD,
        target_type="strategies",
        target_id="s2",
        payload={
            "id": "s2",
            "title": "Re-read errors",
            "description": "Use after any failed action",
            "content": "Error text names the missing field.",
            "embedding_text": "Re-read errors",
        },
    )
    mem._apply_ops([add], actor="test")
    rendered = mem.search("errors", memory_types=["strategies"]).render()
    assert "Use after any failed action" in rendered


def test_invalidated_fact_renders_date_range(mem):
    """round-5 X2: bi-temporal facts must expose their validity range."""
    add = MemoryOp(
        op=OpType.ADD,
        target_type="facts",
        target_id="f1",
        payload={
            "id": "f1",
            "content": "Alice lives in Paris",
            "valid_at": "2024-01-01",
            "embedding_text": "Alice lives in Paris",
        },
    )
    mem._apply_ops([add], actor="test")
    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.INVALIDATE,
                target_type="facts",
                target_id="f1",
                payload={"t_invalid": "2025-06-01"},
            )
        ],
        actor="test",
    )
    rendered = mem.search("Alice Paris", memory_types=["facts"]).render()
    assert "Date range: 2024-01-01 - 2025-06-01" in rendered


def test_noop_op_is_logged_but_changes_nothing(mem):
    """Mem0's fourth event (`NONE`) becomes a log row, not a side-channel counter.

    Upstream never returns it to the caller (`main.py:326-327` @ v0.1.94), so
    "judged and left alone" is indistinguishable there from "never judged". The
    whole point of routing it through the evolution log is that ours are
    distinguishable — which only holds if the op logs while touching nothing.
    """
    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.ADD,
                target_type="semantic",
                target_id="m1",
                payload={"id": "m1", "content": "user likes pizza"},
            )
        ],
        actor="t",
    )
    before = [dict(d) for d in mem.doc_store.list_items("semantic", namespace=mem.namespace)]
    log_before = mem.doc_store.count()

    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.NOOP,
                target_type="semantic",
                target_id="m1",
                payload={"text": "user likes pizza"},
            )
        ],
        actor="t",
    )
    after = [dict(d) for d in mem.doc_store.list_items("semantic", namespace=mem.namespace)]

    assert after == before  # no store effect at all
    assert mem.doc_store.count() == log_before + 1  # but the judgement is on the record
    assert mem.doc_store.tail(1)[0].op is OpType.NOOP


def test_noop_leaves_the_item_servable(mem):
    """A NOOP must not disturb the vector row either — the failure mode this
    guards is an implementation that "handles" NOOP by falling through to the
    DELETE branch's vector cleanup, which the store snapshot alone would not
    catch (the doc row would still be there)."""
    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.ADD,
                target_type="semantic",
                target_id="m1",
                payload={"id": "m1", "content": "user likes pizza"},
            )
        ],
        actor="t",
    )
    mem._apply_ops(
        [MemoryOp(op=OpType.NOOP, target_type="semantic", target_id="m1", payload={})],
        actor="t",
    )
    assert "user likes pizza" in mem.search("pizza", memory_types=["semantic"]).render()


# ---------------- bulk_ingest: the fast path, and where it must refuse ----------------


class _CountingEmbedder(FakeEmbedder):
    """FakeEmbedder that records how many REQUESTS it served, not just how many
    texts — the whole point of batching is the request count."""

    def __init__(self, dim=128):
        super().__init__(dim=dim)
        self.requests = 0
        self.batch_sizes: list[int] = []

    def embed(self, texts, kind="passage"):
        self.requests += 1
        self.batch_sizes.append(len(texts))
        return super().embed(texts, kind)


def _msgs(n):
    return [(f"turn {i} about Paris", "user" if i % 2 else "assistant", {"i": i}) for i in range(n)]


def test_bulk_ingest_stores_the_same_episodes_as_add_message_one_by_one():
    """The fast path may change how many requests it makes; it may not change what
    ends up in the stores."""
    out = {}
    for tag, bulk in (("bulk", True), ("loop", False)):
        emb = _CountingEmbedder()
        m = AgenticMemory(namespace=f"eq-{tag}", organizers=["passthrough"], embedder=emb)
        try:
            if bulk:
                m.bulk_add_messages(_msgs(10), batch_size=4)
            else:
                for c, r, meta in _msgs(10):
                    m.add_message(c, role=r, meta=meta)
            m.flush()
            eps = m.doc_store.list_episodes(namespace=m.namespace)
            out[tag] = (
                [(e.content, e.role, e.meta) for e in eps],  # namespace differs by construction
                len(m.doc_store.read_log()) if hasattr(m.doc_store, "read_log") else None,
                emb.requests,
            )
        finally:
            m.close()
    assert out["bulk"][0] == out["loop"][0]  # same episodes, same order, same meta
    assert out["bulk"][1] == out["loop"][1]  # same number of ops logged
    assert out["bulk"][2] < out["loop"][2]  # and it took fewer requests


def test_bulk_ingest_batches_the_embedding_requests():
    emb = _CountingEmbedder()
    m = AgenticMemory(namespace="batched", organizers=["passthrough"], embedder=emb)
    try:
        assert m.bulk_add_messages(_msgs(10), batch_size=4) == 10
        assert emb.batch_sizes == [4, 4, 2]  # 10 turns, 3 requests, not 10
    finally:
        m.close()


def test_bulk_ingest_still_runs_on_message_once_per_episode():
    seen = []

    class Watcher(Organizer):
        name = "watcher"

        def on_message(self, episode, ctx):
            seen.append(episode.content)
            return []

    m = AgenticMemory(
        namespace="hooks",
        organizers=[Watcher()],
        embedder=_CountingEmbedder(),
        config=AgmemConfig(sync_write=True),
    )
    try:
        m.bulk_add_messages(_msgs(5), batch_size=2)
        m.flush()
        assert seen == [c for c, _, _ in _msgs(5)]  # once each, in order
    finally:
        m.close()


def test_an_organizer_that_reads_the_stores_is_kept_off_the_batched_path():
    """`on_message` that queries the stores sees a different world when the corpus
    is indexed first. Such organizers declare `observes_store_on_message`, and
    bulk_ingest must route them to the per-message path rather than quietly hand
    them the whole corpus."""
    sizes = []

    class Nosy(Organizer):
        name = "nosy"
        observes_store_on_message = True

        def on_message(self, episode, ctx):
            sizes.append(len(ctx.doc_store.list_episodes(namespace=episode.namespace)))
            return []

    emb = _CountingEmbedder()
    m = AgenticMemory(
        namespace="nosy",
        organizers=[Nosy()],
        embedder=emb,
        config=AgmemConfig(sync_write=True),
    )
    try:
        m.bulk_add_messages(_msgs(5), batch_size=5)
        m.flush()
        # each hook saw only the prefix, exactly as per-message ingest gives it
        assert sizes == [1, 2, 3, 4, 5]
        assert emb.batch_sizes == [1, 1, 1, 1, 1]  # never batched
    finally:
        m.close()


def test_the_zep_organizer_is_the_one_that_declares_it():
    """It searches the vector store and reads the doc store inside `on_message`;
    if that ever stops being true the flag should go, and if another organizer
    starts doing it the flag has to arrive with it."""
    from agmem.organizers.base import Organizer as Base

    assert Base.observes_store_on_message is False
    from agmem.organizers.zep_graph.organizer import ZepGraphOrganizer

    assert ZepGraphOrganizer.observes_store_on_message is True


def test_bulk_ingest_of_nothing_makes_no_request():
    emb = _CountingEmbedder()
    m = AgenticMemory(namespace="empty", organizers=["passthrough"], embedder=emb)
    try:
        assert m.bulk_add_messages([]) == 0
        assert emb.requests == 0
    finally:
        m.close()


# ---------------- attach_llm: the wiring an organizer actually reads ----------------


class _RecordingClient:
    def __init__(self, reply="{}"):
        self.reply = reply
        self.calls = []

    def has_role(self, role):
        return True

    def chat(self, role, messages, budget_key=None, **overrides):
        self.calls.append({"role": role, "overrides": overrides})
        return self.reply


def test_attach_llm_reaches_the_organizer_context_not_just_the_attribute():
    """`OrganizerContext` captures `structured` when it is BUILT. Setting
    `mem.llm`, or even `mem.structured`, afterwards leaves `ctx.llm` holding the
    None from construction — and an organizer reads `ctx.llm`. Nemori answers a
    None by turning distillation off and logging it, so a paid arm comes out
    looking like a memory system that added nothing when it never ran."""
    m = AgenticMemory(namespace="attach", organizers=["passthrough"], embedder=FakeEmbedder(dim=32))
    try:
        assert m.organizers_have_llm() is False
        m.llm = _RecordingClient()
        assert m.organizers_have_llm() is False, "the attribute alone is not the wiring"
        m.attach_llm(_RecordingClient())
        assert m.organizers_have_llm() is True
        assert m._ctx.llm is m.structured
    finally:
        m.close()


def test_attach_llm_can_turn_off_guided_json_for_an_openai_endpoint():
    m = AgenticMemory(namespace="gj", organizers=["passthrough"], embedder=FakeEmbedder(dim=32))
    try:
        m.attach_llm(_RecordingClient(), use_guided_json=False)
        assert m.structured.use_guided_json is False
    finally:
        m.close()
