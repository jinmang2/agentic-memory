"""Phase 1 exit criterion: MemoryOp abstraction holds for RB + A-Mem."""

import importlib

from helpers import StubLLM

from agmem import AgenticMemory
from agmem.config import AgmemConfig
from agmem.core.ops import MemoryOp, OpType
from agmem.embed.fake import FakeEmbedder


def make_mem(organizer, llm) -> AgenticMemory:
    mem = AgenticMemory(namespace="t", organizers=[organizer], embedder=FakeEmbedder(dim=128))
    mem.structured = llm
    mem._ctx.llm = llm
    return mem


# ---------------- ReasoningBank ----------------


def rb_llm(success=True):
    return StubLLM(
        {
            "judge": [{"success": success, "reason": "checked final state"}],
            "distill": [
                {
                    "items": [
                        {
                            "title": "Verify filters before submit",
                            "description": "Use when a form has filter controls",
                            "content": "Check each filter state, then submit.",
                        },
                        {"title": "", "description": "broken item", "content": "x"},  # dropped
                        {
                            "title": "Re-read error banners",
                            "description": "Use after any failed action",
                            "content": "Error text usually names the missing field.",
                        },
                    ]
                }
            ],
        }
    )


def test_reasoning_bank_distills_and_indexes():
    from agmem.organizers.reasoning_bank import ReasoningBankOrganizer

    llm = rb_llm()
    mem = make_mem(ReasoningBankOrganizer(), llm)
    try:
        mem.add_task_result(
            trajectory=[{"a": 1}], outcome="unknown", task="filter products by price"
        )
        # judge was consulted for unknown outcome, then distill
        assert [r for r, _ in llm.calls] == ["judge", "distill"]
        ops = mem.log.tail(10)
        strategy_ops = [o for o in ops if o.target_type == "strategies"]
        assert len(strategy_ops) == 2  # broken item skipped (field fallback)
        assert all(o.actor == "reasoning_bank" for o in strategy_ops)

        bundle = mem.search("form filters", memory_types=["strategies"], k=2)
        titles = {s.item.data["title"] for s in bundle.items}
        assert "Verify filters before submit" in titles
    finally:
        mem.close()


def test_reasoning_bank_known_outcome_skips_judge():
    from agmem.organizers.reasoning_bank import ReasoningBankOrganizer

    llm = rb_llm()
    mem = make_mem(ReasoningBankOrganizer(), llm)
    try:
        mem.add_task_result(trajectory=[], outcome="failure", task="t")
        assert [r for r, _ in llm.calls] == ["distill"]
        # failure SI variant chosen — the rules ride in the system message
        assert "FAILED" in llm.systems[0]
    finally:
        mem.close()


def test_matts_contrasts_the_trajectory_set_instead_of_judging_each():
    """MaTTS parallel induction (paper §3.3, upstream `induce_scaling.py`): the
    signal is the MIXTURE of attempts, so there is no per-trajectory judge call
    and no success/failure label — upstream computes the labels and never puts
    them in the prompt. Budget is 5 items, not the single-trajectory 3."""
    from agmem.organizers.reasoning_bank import ReasoningBankOrganizer

    llm = rb_llm()
    mem = make_mem(ReasoningBankOrganizer(), llm)
    try:
        mem.add_scaled_task_result(
            trajectories=[[{"step": "a"}], [{"step": "b"}], [{"step": "c"}]],
            task="filter products by price",
        )
        assert [r for r, _ in llm.calls] == ["distill"]  # no judge, once per SET
        system, prompt = llm.systems[0], llm.calls[0][1]
        assert "compare and contrast" in system.lower() and "at most 5" in system.lower()
        # upstream's own layout, including the space before the colon
        assert "**Trajectory 1 :**" in prompt and "**Trajectory 3 :**" in prompt
        assert prompt.startswith("**Query:** filter products by price")

        strategy_ops = [o for o in mem.log.tail(20) if o.target_type == "strategies"]
        assert len(strategy_ops) == 2
        # neither "success" nor "failure" is true of a contrast-derived item
        assert {o.payload["outcome"] for o in strategy_ops} == {"contrast"}
    finally:
        mem.close()


def test_a_single_attempt_is_not_a_contrast_and_falls_back():
    """Contrasting a set of one is not the mechanism — it must take the ordinary
    single-trajectory path, judge included, rather than run the parallel prompt
    over one item."""
    from agmem.organizers.reasoning_bank import ReasoningBankOrganizer

    llm = rb_llm()
    mem = make_mem(ReasoningBankOrganizer(), llm)
    try:
        mem.add_scaled_task_result(trajectories=[[{"step": "a"}]], task="t")
        assert [r for r, _ in llm.calls] == ["judge", "distill"]
        assert "compare and contrast" not in llm.systems[0].lower()
    finally:
        mem.close()


def test_reasoning_bank_no_llm_explicit_skip():
    from agmem.organizers.reasoning_bank import ReasoningBankOrganizer

    mem = AgenticMemory(
        namespace="t", organizers=[ReasoningBankOrganizer()], embedder=FakeEmbedder(dim=128)
    )
    try:
        mem.add_task_result(trajectory=[], outcome="success", task="t")
        # raw episode logged, but no strategy ops
        assert all(o.target_type != "strategies" for o in mem.log.tail(10))
    finally:
        mem.close()


# ---------------- A-Mem ----------------


def test_amem_note_link_and_evolution():
    from agmem.organizers.amem import AMemOrganizer

    llm = StubLLM(
        {
            "extract": [
                {"keywords": ["파리", "여행"], "context": "파리 여행 계획", "tags": ["travel"]},
                {
                    "keywords": ["파리", "예산"],
                    "context": "파리 여행 예산",
                    "tags": ["travel", "budget"],
                },
            ],
            "distill": [
                # second note's evolution: link to first + update its context
                {
                    "should_evolve": True,
                    "connections": ["__FIRST__"],
                    "neighbor_updates": [
                        {
                            "id": "__FIRST__",
                            "new_context": "파리 여행 계획 (예산 300만원 확정)",
                            "new_tags": ["travel", "budget"],
                        }
                    ],
                },
            ],
        }
    )
    mem = make_mem(AMemOrganizer(top_k=5), llm)
    try:
        mem.add_message("다음 달에 파리로 여행 가려고 해")
        first_ops = [o for o in mem.log.tail(10) if o.target_type == "notes"]
        first_id = first_ops[0].target_id

        # patch the stub's placeholder with the real neighbor id
        resp = llm.responses["distill"][0]
        resp["connections"] = [first_id]
        resp["neighbor_updates"][0]["id"] = first_id

        mem.add_message("파리 여행 예산은 300만원이야")

        ops = mem.log.tail(20)
        kinds = [(o.op, o.target_type) for o in ops]
        assert (OpType.LINK, "notes") in kinds
        assert (OpType.UPDATE, "notes") in kinds

        # unidirectional link, as upstream: the NEW note links to the
        # neighbor; the neighbor itself gains no back-link
        second_id = next(
            o.target_id for o in ops if o.op is OpType.LINK and o.target_type == "notes"
        )
        second = mem.doc_store.get_items([second_id], "notes")[0]
        assert first_id in second["links"]
        first = mem.doc_store.get_items([first_id], "notes")[0]
        assert not first.get("links"), "upstream links are one-way"
        # UPDATE merged, not clobbered: content survived the context rewrite
        assert first["content"] == "다음 달에 파리로 여행 가려고 해"
        assert "300만원" in first["context"]
    finally:
        mem.close()


def test_amem_tag_refinement_is_unconditional_including_empty_wipe():
    """Round-12 finding 2: the plain (published-numbers) edition applies
    ``note.tags = new_tags`` with NO guard (memory_layer.py:834-836) — its
    strict schema requires the key but permits ``[]``. Our round-11
    emptiness+equality guard suppressed no-op verdicts (skewing evolution-log
    op counts) and blocked [] wipes; both now flow through as UPDATE ops, the
    wipe staying auditable in the op log. (Robust-edition variant: emptiness
    guard only, memory_layer_robust.py:506-507 — not what we reproduce.)"""
    from agmem.organizers.amem import AMemOrganizer

    llm = StubLLM(
        {
            "extract": [
                {"keywords": ["a"], "context": "c1", "tags": ["t1"]},
                {"keywords": ["a"], "context": "c2", "tags": ["travel", "budget"]},
            ],
            "distill": [
                {
                    "should_evolve": True,
                    "actions": ["strengthen"],
                    "connections": ["__FIRST__"],
                    "new_note_tags": [],  # the wipe the strict schema permits
                },
            ],
        }
    )
    mem = make_mem(AMemOrganizer(), llm)
    try:
        mem.add_message("first note about topic alpha")
        first_id = next(o.target_id for o in mem.log.tail(10) if o.target_type == "notes")
        llm.responses["distill"][0]["connections"] = [first_id]

        mem.add_message("second note about topic alpha")
        ops = mem.log.tail(20)
        second_id = [o.target_id for o in ops if o.op is OpType.ADD and o.target_type == "notes"][
            -1
        ]
        tag_updates = [
            o
            for o in ops
            if o.op is OpType.UPDATE and o.target_id == second_id and "tags" in o.payload
        ]
        assert len(tag_updates) == 1, "the empty verdict must still emit the UPDATE op"
        assert tag_updates[0].payload["tags"] == []
        second = mem.doc_store.get_items([second_id], "notes")[0]
        assert second["tags"] == [], "Ps1 tags are wiped, exactly as upstream would"
    finally:
        mem.close()


def test_amem_hallucinated_neighbor_ids_ignored():
    from agmem.organizers.amem import AMemOrganizer

    llm = StubLLM(
        {
            "extract": [
                {"keywords": ["a"], "context": "c1", "tags": ["t"]},
                {"keywords": ["a"], "context": "c2", "tags": ["t"]},
            ],
            "distill": [
                {
                    "should_evolve": True,
                    "connections": ["not-a-real-id"],
                    "neighbor_updates": [{"id": "also-fake", "new_context": "x"}],
                },
            ],
        }
    )
    mem = make_mem(AMemOrganizer(), llm)
    try:
        mem.add_message("first note about topic alpha")
        mem.add_message("second note about topic alpha")
        ops = mem.log.tail(20)
        # bug-fix #32 behavior: fake ids produce no LINK ops and no neighbor
        # UPDATE ops. The one UPDATE that MAY appear targets the new note
        # itself: strengthen's tag refinement is unconditional (round-12
        # finding 2, memory_layer.py:834-836), independent of connection
        # validity.
        assert all(o.op is not OpType.LINK for o in ops)
        note_ids = {o.target_id for o in ops if o.op is OpType.ADD and o.target_type == "notes"}
        second_id = [o.target_id for o in ops if o.op is OpType.ADD and o.target_type == "notes"][
            -1
        ]
        for o in ops:
            if o.op is OpType.UPDATE:
                assert o.target_id == second_id, "no neighbor may be updated via a fake id"
        assert note_ids, "both notes must still be stored"
    finally:
        mem.close()


def test_amem_object_shaped_connections_still_link():
    """Small models return `connections` as objects instead of id strings, which
    made `c in valid_ids` raise TypeError (dict is unhashable). The exception
    escaped `_ingest` after the note ADD was built, and both the background
    worker and `_propagate_events` swallow exceptions, so **the note vanished
    entirely** — silently corrupting note counts, which is exactly the Table 7
    storage metric. Observed with Qwen3-0.6B on locomo conv0."""
    from agmem.organizers.amem import AMemOrganizer

    llm = StubLLM(
        {
            "extract": [
                {"keywords": ["a"], "context": "c1", "tags": ["t"]},
                {"keywords": ["a"], "context": "c2", "tags": ["t"]},
            ],
            "distill": [
                {
                    "should_evolve": True,
                    # the malformation: objects, not id strings
                    "connections": [{"id": "__FIRST__", "reason": "same topic"}],
                },
            ],
        }
    )
    mem = make_mem(AMemOrganizer(), llm)
    try:
        mem.add_message("first note about topic alpha")
        first_id = next(o.target_id for o in mem.log.tail(10) if o.target_type == "notes")
        llm.responses["distill"][0]["connections"] = [{"id": first_id, "reason": "same topic"}]

        mem.add_message("second note about topic alpha")
        notes = mem.doc_store.list_items("notes", namespace=mem.namespace)
        assert len(notes) == 2, "the second note must survive a malformed verdict"
        linked = next(n for n in notes if n["id"] != first_id)
        assert first_id in linked["links"], "the id inside the object is still usable"
    finally:
        mem.close()


def test_amem_string_shaped_neighbor_updates_do_not_lose_the_note():
    """Mirror malformation: `neighbor_updates` as bare id strings, where
    `upd.get("id")` raised AttributeError and lost the note the same way."""
    from agmem.organizers.amem import AMemOrganizer

    llm = StubLLM(
        {
            "extract": [
                {"keywords": ["a"], "context": "c1", "tags": ["t"]},
                {"keywords": ["a"], "context": "c2", "tags": ["t"]},
            ],
            "distill": [
                {"should_evolve": True, "connections": [], "neighbor_updates": ["__FIRST__"]},
            ],
        }
    )
    mem = make_mem(AMemOrganizer(), llm)
    try:
        mem.add_message("first note about topic alpha")
        first_id = next(o.target_id for o in mem.log.tail(10) if o.target_type == "notes")
        llm.responses["distill"][0]["neighbor_updates"] = [first_id]

        mem.add_message("second note about topic alpha")
        notes = mem.doc_store.list_items("notes", namespace=mem.namespace)
        assert len(notes) == 2
        # a bare id carries no new context/tags, so the neighbor keeps its own
        first = mem.doc_store.get_items([first_id], "notes")[0]
        assert first["context"] == "c1"
    finally:
        mem.close()


def test_amem_degrades_without_llm():
    from agmem.organizers.amem import AMemOrganizer

    mem = AgenticMemory(namespace="t", organizers=[AMemOrganizer()], embedder=FakeEmbedder(dim=128))
    try:
        mem.add_message("bare note without llm")
        notes = [o for o in mem.log.tail(10) if o.target_type == "notes"]
        assert len(notes) == 1  # bare note stored, no crash
    finally:
        mem.close()


# ---------------- async worker ----------------


def test_async_write_flush():
    from agmem.organizers.reasoning_bank import ReasoningBankOrganizer

    llm = rb_llm()
    mem = AgenticMemory(
        namespace="t",
        organizers=[ReasoningBankOrganizer()],
        embedder=FakeEmbedder(dim=128),
        config=AgmemConfig(sync_write=False),
    )
    mem.structured = llm
    mem._ctx.llm = llm
    try:
        mem.add_task_result(trajectory=[], outcome="success", task="do a thing")
        mem.flush()  # block until worker applied the ops
        strategy_ops = [o for o in mem.log.tail(10) if o.target_type == "strategies"]
        assert len(strategy_ops) == 2
    finally:
        mem.close()


# ---------------- MMR ----------------


def test_mmr_prefers_diversity():
    from agmem.retrieval.rerank import MMRReranker

    # two near-duplicates + one distinct; query must not coincide with the
    # duplicates (when query == dup, relevance == redundancy and MMR ties)
    vectors = {
        "dup1": [0.95, 0.31, 0.0],
        "dup2": [0.95, 0.312, 0.0],
        "other": [0.5, -0.866, 0.0],
    }
    candidates = [("dup1", 0.9), ("dup2", 0.89), ("other", 0.7)]
    picked = MMRReranker(lambda_=0.5).rerank([1.0, 0.0, 0.0], candidates, vectors, k=2)
    ids = [c for c, _ in picked]
    assert ids[0] == "dup1"
    assert ids[1] == "other"  # diversity beats the near-duplicate


# ---------------- package layout invariant ----------------


def test_organizers_root_is_framework_only_and_methodologies_are_packages():
    """``organizers/`` root = the plugin framework; every methodology = a subpackage.

    The earlier version of this rule was "the root holds Organizer subclasses",
    with ``base.py``/``__init__.py`` carved out as exceptions — and an exception
    list is how the root got mixed in the first place. It also let ``gated.py``
    pass while re-mixing categories, since a composition adapter is framework,
    not a methodology. So the rule is now positional and exception-free: a
    plain module at the root is framework, and anything that implements a paper
    lives in its own package (which is also where its internal stages go, as
    Nemori's already do).
    """
    import pkgutil

    import agmem.organizers as pkg
    from agmem.organizers.base import Organizer

    FRAMEWORK = {"base", "gated"}
    stray_modules, packages = [], []
    for info in pkgutil.iter_modules(pkg.__path__):
        (packages if info.ispkg else stray_modules).append(info.name)

    assert sorted(stray_modules) == sorted(FRAMEWORK), (
        f"root modules must be exactly {sorted(FRAMEWORK)}, got {sorted(stray_modules)} — "
        "a methodology belongs in its own subpackage"
    )
    for name in packages:
        if name == "experimental":
            continue  # semantic quarantine, not a single methodology
        module = importlib.import_module(f"agmem.organizers.{name}")
        exported = [
            obj
            for obj in vars(module).values()
            if isinstance(obj, type) and issubclass(obj, Organizer) and obj is not Organizer
        ]
        assert exported, f"methodology package {name!r} must re-export its Organizer"


def test_methodology_packages_keep_their_single_module_import_path():
    """Promoting a module to a package must not move any call site."""
    from agmem.organizers.ace import ACEOrganizer
    from agmem.organizers.amem import AMemOrganizer
    from agmem.organizers.gmemory import GMemoryOrganizer
    from agmem.organizers.memoryos import MemoryOSOrganizer
    from agmem.organizers.nemori import NemoriOrganizer
    from agmem.organizers.passthrough import PassthroughOrganizer
    from agmem.organizers.reasoning_bank import ReasoningBankOrganizer
    from agmem.organizers.zep_graph import ZepGraphOrganizer

    for cls in (
        ACEOrganizer,
        AMemOrganizer,
        GMemoryOrganizer,
        MemoryOSOrganizer,
        NemoriOrganizer,
        PassthroughOrganizer,
        ReasoningBankOrganizer,
        ZepGraphOrganizer,
    ):
        assert cls.__module__.endswith(".organizer"), cls.__module__


def test_policies_declare_no_memory_type_and_emit_no_ops():
    """The operational test for belonging in ``policies/`` rather than ``organizers/``."""
    from agmem.organizers.base import Organizer
    from agmem.policies import AdmissionGate

    assert not issubclass(AdmissionGate, Organizer)
    assert not hasattr(AdmissionGate, "produces")
    assert not hasattr(AdmissionGate, "on_message")


def test_no_mechanism_imports_the_policies_package():
    """Orthogonality as an enforced fact, not a claim in a docstring.

    A mechanism must not know policies exist; only the adapter that applies one
    (``organizers/gated.py``) may import the package. If a future organizer grows
    an ``admission=`` constructor argument again, this fails.
    """
    import pathlib

    import agmem.organizers as pkg

    root = pathlib.Path(pkg.__path__[0])
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if path.name != "gated.py" and "agmem.policies" in path.read_text()
    ]
    assert offenders == [], f"mechanism modules importing policies: {offenders}"


def test_feedback_is_owned_by_the_producing_organizer():
    """`strategies` is written by both ReasoningBank and G-Memory, so a facade
    that branched on the memory type applied G-Memory's +1/-2 reward to
    ReasoningBank items — whose paper is deliberately append-only with no
    feedback loop at all. Feedback now fans out to organizers, so an item only
    moves when its own methodology is active."""
    from agmem.core.types import StrategyItem
    from agmem.organizers.gmemory import GMemoryOrganizer
    from agmem.organizers.reasoning_bank import ReasoningBankOrganizer

    item = StrategyItem(title="t", description="d", content="c")
    add = MemoryOp(
        op=OpType.ADD,
        target_type="strategies",
        target_id=item.id,
        payload={"id": item.id, "content": "c", "embedding_text": "c", "score": 1.0},
    )

    # ReasoningBank alone: no feedback semantics -> nothing happens.
    mem = AgenticMemory(
        namespace="t", organizers=[ReasoningBankOrganizer()], embedder=FakeEmbedder(dim=64)
    )
    try:
        mem._apply_ops([add], actor="reasoning_bank")
        assert mem.report_feedback([item.id], helpful=False) == 0
        assert mem.doc_store.get_items([item.id], "strategies")[0]["score"] == 1.0
    finally:
        mem.close()

    # G-Memory: reward applies only to INSIGHTS it actually served (upstream
    # insights_cache holds rules exclusively, U:239 — round-5 W-4 + round-12
    # #6/#18: a kind-less foreign item never moves even if its id leaks into
    # the cache, and an empty cache updates nothing instead of bypassing the
    # gate).
    org = GMemoryOrganizer()
    mem = AgenticMemory(namespace="t", organizers=[org], embedder=FakeEmbedder(dim=64))
    try:
        mem._apply_ops([add], actor="gmemory")
        assert mem.report_feedback([item.id], helpful=True) == 0  # nothing served
        org._served = {item.id}
        assert mem.report_feedback([item.id], helpful=True) == 0  # no kind -> not an insight
        assert mem.doc_store.get_items([item.id], "strategies")[0]["score"] == 1.0
        assert org._served == set()  # ...and the cache is cleared, not leaked

        rule = MemoryOp(
            op=OpType.ADD,
            target_type="strategies",
            target_id="rule-1",
            payload={
                "id": "rule-1",
                "content": "r, because r",
                "kind": "insight",
                "score": 1.0,
                "embedding_text": None,  # insights never enter the vector store
            },
        )
        mem._apply_ops([rule], actor="gmemory")
        org._served = {"rule-1"}
        assert mem.report_feedback(["rule-1"], helpful=True) == 1
        assert mem.doc_store.get_items(["rule-1"], "strategies")[0]["score"] == 2.0
    finally:
        mem.close()
