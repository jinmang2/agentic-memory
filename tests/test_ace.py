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
        curator_prompt = next(p for role, p in llm.calls if "Recent Reflection:" in p)
        assert f"Total token budget: {PLAYBOOK_TOKEN_BUDGET} tokens" in curator_prompt
        assert "step 1 of 50" in curator_prompt
        assert "Current Playbook Stats:" in curator_prompt
        # upstream's environment feedback replaces a recognised outcome word
        reflect_prompt = next(p for role, p in llm.calls if "Trajectory:" in p)
        assert "Predicted answer does not match ground truth" in reflect_prompt
    finally:
        mem.close()


# The two prompts are located by a STRUCTURAL marker ("Trajectory:" for the
# reflector, "Recent Reflection:" for the curator) rather than by the words
# "reflector"/"curator". Those words are not in upstream's prompts either — its
# reflector opens "You are an expert analyst and educator" — so keying on them
# pinned a paraphrase of ours instead of the shape being asserted, and broke the
# moment the opening line was brought back into line with upstream.
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
        curator_prompt = next(p for role, p in llm.calls if "Recent Reflection:" in p)
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
        reflect_prompt = next(p for role, p in llm.calls if "Trajectory:" in p)
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


# ---------------------------------------------------------------------------
# Retry rounds (D4): upstream's `_train_step` does not reflect once — it
# reflects, regenerates, and reflects on the regenerated attempt, up to
# `max_num_rounds` (ace.py:498-543). Our default cuts at one reflection, which
# is deviation D4 in docs/19; these tests pin the shape of the arm that closes
# it, and that the default arm is untouched by its existence.
# ---------------------------------------------------------------------------


def retry_llm(rounds: int):
    """`rounds` reflections, each with a distinguishable key_insight, then the
    curator's single operations reply."""
    return StubLLM(
        {
            "distill": [reflection_stub(key_insight=f"insight-{i}") for i in range(rounds)]
            + [
                {
                    "operations": [
                        {"type": "ADD", "section": "web_forms", "content": "Check the filter."}
                    ]
                }
            ],
        }
    )


def make_retry_mem(llm, **kwargs):
    mem = AgenticMemory(
        namespace="t", organizers=[ACEOrganizer(**kwargs)], embedder=FakeEmbedder(dim=128)
    )
    mem.structured = llm
    mem._ctx.llm = llm
    return mem


def test_ace_retry_reflects_then_regenerates_and_never_reflects_on_the_last_attempt():
    """Upstream's round is (reflect, regenerate), three times, with no trailing
    reflection: `for round_num in range(max_num_rounds)` reflects on the current
    attempt and then regenerates from it (ace.py:501-543), so the attempt
    produced by the FINAL regeneration is never diagnosed. The curator is handed
    `recent_reflection` — the reflection written about the second-to-last
    attempt (ace.py:583). Getting this order wrong would spend the same money
    and feed the curator a different input."""
    llm = retry_llm(rounds=3)
    mem = make_retry_mem(llm, max_rounds=3)
    seen: list[str] = []

    def regenerate(reflection_text: str):
        seen.append(reflection_text)
        return {"step": len(seen) + 1, "prediction": "still wrong"}, "failure"

    try:
        mem._ctx  # noqa: B018 - context is built lazily by the facade
        mem.organizers[0].set_retry_generator(regenerate)
        mem.add_task_result(trajectory=[{"a": 1}], outcome="failure", task="filter products")
    finally:
        mem.close()

    reflect_calls = [p for role, p in llm.calls if role == "distill" and "playbook" not in p[:40]]
    assert len(seen) == 3, "three regenerations, one per round"
    assert len(llm.calls) == 4, "three reflections and one curation"
    del reflect_calls
    curate_prompt = llm.calls[-1][1]
    assert "insight-2" in curate_prompt, "the curator sees the LAST reflection taken"
    assert "insight-0" not in curate_prompt


def test_ace_retry_stops_as_soon_as_a_regeneration_lands():
    """`if data_processor.answer_is_correct(...): break` (ace.py:541-544). The
    rounds are a budget, not a schedule — a sample corrected on the first retry
    costs one reflection and one regeneration, and the curator is handed the
    reflection that produced the correction."""
    llm = retry_llm(rounds=3)
    mem = make_retry_mem(llm, max_rounds=3)
    calls: list[str] = []

    def regenerate(reflection_text: str):
        calls.append(reflection_text)
        return {"step": 2, "prediction": "right"}, "success"

    try:
        mem.organizers[0].set_retry_generator(regenerate)
        mem.add_task_result(trajectory=[{"a": 1}], outcome="failure", task="filter products")
    finally:
        mem.close()

    assert len(calls) == 1
    assert len(llm.calls) == 2, "one reflection, one curation"
    assert "insight-0" in llm.calls[-1][1]


def test_ace_never_regenerates_a_task_that_was_already_correct():
    """Upstream's else-branch reflects once to tag helpful bullets and does not
    regenerate (ace.py:548-570). A correct sample costs the same in the retry
    arm as in ours."""
    llm = retry_llm(rounds=1)
    mem = make_retry_mem(llm, max_rounds=3)
    calls: list[str] = []

    try:
        mem.organizers[0].set_retry_generator(lambda r: (calls.append(r), ({}, "failure"))[1])
        mem.add_task_result(trajectory=[{"a": 1}], outcome="success", task="filter products")
    finally:
        mem.close()

    assert calls == []
    assert len(llm.calls) == 2, "one reflection, one curation"


def test_ace_without_a_retry_generator_is_the_arm_we_already_measured():
    """The default organizer must be untouched by the arm's existence: one
    reflection, no regeneration, whatever `max_rounds` says. The measured
    online/nodedup arms are this path, and a change here would silently
    re-price them."""
    llm = retry_llm(rounds=1)
    mem = make_retry_mem(llm, max_rounds=3)  # rounds configured, no generator wired
    try:
        mem.add_task_result(trajectory=[{"a": 1}], outcome="failure", task="filter products")
        adds = [o for o in mem.log.tail(10) if o.target_type == "playbook" and o.op is OpType.ADD]
    finally:
        mem.close()
    assert len(llm.calls) == 2
    assert len(adds) == 1


def test_ace_retry_counters_accumulate_across_rounds_instead_of_overwriting():
    """Upstream applies each round's counter updates to the playbook before the
    next round runs (`self.playbook = update_bullet_counts(...)`, ace.py:521),
    so a bullet tagged helpful in two rounds ends at +2. Ours returns ops that
    the facade applies at the end, so the increments have to be tracked while
    the rounds run — computing every payload from the same stored value would
    write +1 twice and land on +1."""
    seed = StubLLM(
        {
            "distill": [
                reflection_stub(),
                {
                    "operations": [
                        {"type": "ADD", "section": "web_forms", "content": "Check the filter."}
                    ]
                },
            ]
        }
    )
    mem = make_retry_mem(seed, max_rounds=3)
    try:
        mem.add_task_result(trajectory=[{"a": 1}], outcome="success", task="seed the playbook")
        bullet_id = [
            o.target_id
            for o in mem.log.tail(10)
            if o.target_type == "playbook" and o.op is OpType.ADD
        ][0]

        tag = [{"id": bullet_id[:5], "tag": "helpful"}]
        llm = StubLLM(
            {
                "distill": [
                    reflection_stub(bullet_tags=tag),
                    reflection_stub(bullet_tags=tag),
                    reflection_stub(bullet_tags=tag),
                    {"operations": []},
                ]
            }
        )
        mem.structured = llm
        mem._ctx.llm = llm
        mem.organizers[0].set_retry_generator(lambda r: ({"step": 2}, "failure"))
        mem.add_task_result(trajectory=[{"a": 1}], outcome="failure", task="filter products")

        rendered = mem.get_playbook()
    finally:
        mem.close()
    assert "helpful=3" in rendered, rendered
