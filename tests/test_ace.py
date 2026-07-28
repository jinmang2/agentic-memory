from helpers import StubLLM

from agmem import AgenticMemory
from agmem.core.ops import OpType
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.ace import ACEOrganizer


def make_mem(llm):
    mem = AgenticMemory(namespace="t", organizers=[ACEOrganizer()], embedder=FakeEmbedder(dim=128))
    mem.structured = llm
    mem._ctx.llm = llm
    return mem


def reflection_stub(key_insight="Always validate filter state", **overrides):
    """A reflection in upstream's five-field shape (prompts/reflector.py:18-24):
    reasoning / error_identification / root_cause_analysis / correct_approach /
    key_insight (+ bullet_tags). The former `lessons` array was our invention
    and is gone (round-12 #2(c))."""
    reflection = {
        "reasoning": "The trace skipped the filter check.",
        "error_identification": "Submitted with a stale filter state.",
        "root_cause_analysis": "Assumed defaults persist across reloads.",
        "correct_approach": "Re-read every filter control before submitting.",
        "key_insight": key_insight,
        "bullet_tags": [],
    }
    reflection.update(overrides)
    return reflection


def ace_llm():
    return StubLLM(
        {
            "distill": [
                reflection_stub(),
                {
                    "operations": [
                        {
                            "type": "ADD",
                            "section": "web_forms",
                            "content": "Verify every filter control state before submit.",
                        },
                    ]
                },
            ],
        }
    )


def test_ace_adds_playbook_bullet_and_renders():
    mem = make_mem(ace_llm())
    try:
        mem.add_task_result(trajectory=[{"a": 1}], outcome="success", task="filter products")
        adds = [o for o in mem.log.tail(10) if o.target_type == "playbook" and o.op is OpType.ADD]
        assert len(adds) == 1 and adds[0].actor == "ace"
        rendered = mem.get_playbook()
        assert "web_forms" in rendered and "helpful=0" in rendered
    finally:
        mem.close()


def test_ace_dedup_skips_near_duplicate():
    llm = ace_llm()
    # second task curates an identical bullet -> must be deduped
    llm.responses["distill"] += [
        reflection_stub(key_insight="same"),
        {
            "operations": [
                {
                    "type": "ADD",
                    "section": "web_forms",
                    "content": "Verify every filter control state before submit.",
                }
            ]
        },
    ]
    mem = make_mem(llm)
    try:
        mem.add_task_result(trajectory=[], outcome="success", task="task one")
        mem.add_task_result(trajectory=[], outcome="success", task="task two")
        adds = [o for o in mem.log.tail(20) if o.target_type == "playbook" and o.op is OpType.ADD]
        assert len(adds) == 1  # exact duplicate embedding -> sim 1.0 >= 0.90
    finally:
        mem.close()


def test_ace_feedback_increments_counters():
    mem = make_mem(ace_llm())
    try:
        mem.add_task_result(trajectory=[], outcome="success", task="filter products")
        bullet_id = [o for o in mem.log.tail(10) if o.target_type == "playbook"][0].target_id
        assert mem.report_feedback([bullet_id], helpful=True) == 1
        assert mem.report_feedback([bullet_id], helpful=False) == 1
        data = mem.doc_store.get_items([bullet_id], "playbook")[0]
        assert data["helpful"] == 1 and data["harmful"] == 1
    finally:
        mem.close()


def test_llm_reranker_reorders_and_survives_failure():
    from agmem.retrieval.rerank import LLMReranker

    candidates = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    texts = {"a": "cat food", "b": "paris travel budget", "c": "gym schedule"}

    good = StubLLM({"rerank": [{"ranking": [1, 99, 0]}]})  # 99 invalid -> ignored
    out = LLMReranker(good).rerank(None, candidates, {}, 2, texts=texts, query="trip cost")
    assert [c for c, _ in out] == ["b", "a"]

    broken = StubLLM({})  # returns None -> drop -> fusion order preserved
    out = LLMReranker(broken).rerank(None, candidates, {}, 2, texts=texts, query="q")
    assert [c for c, _ in out] == ["a", "b"]


def test_ace_curator_gets_the_token_budget_stats_and_progress():
    """Upstream's curator prompt carries a "Training Context" block — token
    budget, progress, and playbook stats (`ace.py` passes `token_budget`,
    `current_step`, `total_samples`, `playbook_stats`).

    It matters more here than it looks: ACE injects the WHOLE playbook, so the
    only thing holding growth back is the curator knowing how much room is left.
    The budget is a prompt input, never a truncation — dropping bullets to fit
    would be the context collapse the paper is about."""
    from agmem.organizers.ace.organizer import PLAYBOOK_TOKEN_BUDGET, playbook_stats

    stats = playbook_stats(
        [
            {"section": "a", "helpful": 9, "harmful": 0},  # high-performing
            {"section": "a", "helpful": 0, "harmful": 3},  # problematic
            {"section": "b", "helpful": 0, "harmful": 0},  # unused
        ]
    )
    assert "total bullets: 3" in stats
    assert "high-performing: 1" in stats
    assert "problematic: 1" in stats
    assert "unused: 1" in stats
    assert "a=2, b=1" in stats

    llm = StubLLM(
        {
            "distill": [
                reflection_stub(key_insight="k"),
                {"operations": [{"type": "ADD", "section": "general", "content": "new bullet"}]},
            ]
        }
    )
    org = ACEOrganizer(total_samples=50)
    mem = AgenticMemory(namespace="t", organizers=[org], embedder=FakeEmbedder(dim=128))
    mem.structured = llm
    mem._ctx.llm = llm
    try:
        mem.add_task_result([{"step": 1}], "failure", "a task")
        curator_prompt = next(p for role, p in llm.calls if "curator" in p)
        assert f"Total token budget: {PLAYBOOK_TOKEN_BUDGET} tokens" in curator_prompt
        assert "step 1 of 50" in curator_prompt
        assert "Current Playbook Stats:" in curator_prompt
        # upstream's environment feedback replaces a recognised outcome word
        reflect_prompt = next(p for role, p in llm.calls if "reflector" in p)
        assert "Predicted answer does not match ground truth" in reflect_prompt
    finally:
        mem.close()


def test_ace_curator_inputs_match_upstream_shape():
    """Round-12 #2: the curator's inputs are upstream CURATOR_PROMPT's, no more
    and no less — (a) it is told the token budget but NEVER the current playbook
    size (upstream's `count_tokens` is logging-only); (b) it receives the
    question context; (c) it receives the FULL raw reflection, all five
    reflector fields, not a `key_insight`+invented-`lessons` digest."""
    llm = ace_llm()
    mem = make_mem(llm)
    try:
        mem.add_task_result(trajectory=[{"a": 1}], outcome="failure", task="filter products")
        curator_prompt = next(p for role, p in llm.calls if "curator" in p)
        # (a) budget yes, current-size line no
        assert "Total token budget:" in curator_prompt
        assert "now uses about" not in curator_prompt
        # (b) question context = the task
        assert "Question Context:" in curator_prompt
        assert "filter products" in curator_prompt
        # (c) the full reflection rides along, not just key_insight
        assert "Recent Reflection:" in curator_prompt
        for field_value in (
            "The trace skipped the filter check.",
            "Submitted with a stale filter state.",
            "Assumed defaults persist across reloads.",
            "Re-read every filter control before submitting.",
            "Always validate filter state",
        ):
            assert field_value in curator_prompt
        assert "lessons" not in curator_prompt
        # and the reflector was asked for upstream's five fields
        reflect_prompt = next(p for role, p in llm.calls if "reflector" in p)
        for field in (
            "reasoning",
            "error_identification",
            "root_cause_analysis",
            "correct_approach",
            "key_insight",
        ):
            assert field in reflect_prompt
    finally:
        mem.close()


def test_ace_curator_operations_are_uncapped():
    """Round-12 #2(d): upstream accepts an unbounded operations list — the old
    maxItems:5 / max_ops=5 cap was our invention. Seven distinct ops must yield
    seven bullets (token-disjoint contents, so dedup does not interfere)."""
    from agmem.organizers.ace.organizer import CURATE_SCHEMA

    assert "maxItems" not in CURATE_SCHEMA["properties"]["operations"]

    contents = [
        "alpha uno",
        "bravo dos",
        "charlie tres",
        "delta cuatro",
        "echo cinco",
        "foxtrot seis",
        "golf siete",
    ]
    llm = StubLLM(
        {
            "distill": [
                reflection_stub(),
                {
                    "operations": [
                        {"type": "ADD", "section": "general", "content": c} for c in contents
                    ]
                },
            ]
        }
    )
    mem = make_mem(llm)
    try:
        mem.add_task_result(trajectory=[], outcome="success", task="many insights")
        adds = [o for o in mem.log.tail(20) if o.target_type == "playbook" and o.op is OpType.ADD]
        assert len(adds) == 7
    finally:
        mem.close()


def test_playbook_is_not_a_default_search_type_but_explicit_opt_in_serves_it():
    """Round-12 #5: ACE's read contract (whole-playbook injection via
    `get_playbook`) is structural, not conventional — `default_memory_types`
    excludes `playbook`, so a plain `search()` cannot serve top-k bullets.
    An EXPLICIT `memory_types=("playbook",)` remains the caller's choice."""
    mem = make_mem(ace_llm())
    try:
        mem.add_task_result(trajectory=[{"a": 1}], outcome="success", task="filter products")
        assert "playbook" not in mem.default_memory_types

        default_bundle = mem.search("verify filter control state before submit")
        assert all(s.memory_type != "playbook" for s in default_bundle.items)

        explicit = mem.search(
            "verify filter control state before submit", memory_types=("playbook",), k=5
        )
        assert explicit.items and all(s.memory_type == "playbook" for s in explicit.items)
    finally:
        mem.close()
