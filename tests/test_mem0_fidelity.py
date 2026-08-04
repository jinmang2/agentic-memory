"""Mem0 port fidelity — every assertion here names the upstream site it pins.

Port pin: mem0ai/mem0 tag v0.1.94 (07ddd7cb…, 2025-04-26). Tests that read the
pinned clone skip cleanly when it is absent (CI convention: the upstream copies
live under $AGMEM_UPSTREAM, default ~/.agmem/upstream, and CI fetches only some
of them).
"""

import os
import subprocess
from pathlib import Path

import pytest

from agmem.organizers.mem0.prompts import (
    DEFAULT_UPDATE_MEMORY_PROMPT,
    FACT_RETRIEVAL_PROMPT,
    fact_retrieval_prompt,
    get_update_memory_messages,
    parse_messages,
)

UPSTREAM = Path(os.environ.get("AGMEM_UPSTREAM", Path.home() / ".agmem" / "upstream")) / "mem0"
PIN = "07ddd7cb4bd67962cf9a988d7b5c3f3920fad2d4"


def _upstream_prompts_py() -> str:
    """The pinned `mem0/configs/prompts.py`, or a skip.

    The pin is re-checked on every read rather than once at import: a clone that
    silently moved would otherwise turn these byte-equality tests into
    equality-with-something-else, which passes just as green.
    """
    if not (UPSTREAM / ".git").exists():
        pytest.skip(f"pinned mem0 clone not present at {UPSTREAM}")
    # check=False: a missing tag should surface as the SHA-mismatch message below,
    # which names the expected pin, not as a bare CalledProcessError.
    sha = subprocess.run(
        ["git", "-C", str(UPSTREAM), "rev-parse", "v0.1.94"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    assert sha == PIN, f"clone moved: v0.1.94 is {sha}, expected {PIN}"
    # check=True here: past the pin assertion, a failing `git show` can only mean
    # the file moved within the tag, and an empty string would otherwise reach the
    # comparisons as a plain inequality with no cause attached.
    return subprocess.run(
        ["git", "-C", str(UPSTREAM), "show", "v0.1.94:mem0/configs/prompts.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _strip_date(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("- Today's date is "))


def test_fact_retrieval_prompt_is_byte_identical_to_upstream():
    src = _upstream_prompts_py()
    # Upstream declares it as an f-string, so every literal brace in the few-shot
    # examples is doubled at the source level and the date is an embedded
    # expression. Undo the doubling, ignore the date line (compared separately).
    upstream_block = src.split('FACT_RETRIEVAL_PROMPT = f"""', 1)[1].split('"""', 1)[0]
    upstream_value = upstream_block.replace("{{", "{").replace("}}", "}")
    assert _strip_date(FACT_RETRIEVAL_PROMPT) == _strip_date(upstream_value)


def test_fact_retrieval_date_line_matches_upstreams_format():
    # prompts.py:49 @ v0.1.94 -> `- Today's date is {datetime.now().strftime("%Y-%m-%d")}.`
    # Ours defers the interpolation to call time; the rendered line must still be
    # the line upstream renders, or the prompt differs on a token the model reads.
    lines = [ln for ln in fact_retrieval_prompt().splitlines() if ln.startswith("- Today's date")]
    assert len(lines) == 1
    import datetime as _dt

    # naive local time on purpose — see fact_retrieval_prompt's docstring
    today = _dt.datetime.now().strftime("%Y-%m-%d")  # noqa: DTZ005
    assert lines[0] == f"- Today's date is {today}."
    # and the unrendered constant keeps the placeholder, so nothing freezes a date
    assert "{TODAY}" in FACT_RETRIEVAL_PROMPT


def test_update_memory_prompt_is_byte_identical_to_upstream():
    src = _upstream_prompts_py()
    upstream_block = src.split('DEFAULT_UPDATE_MEMORY_PROMPT = """', 1)[1].split('"""', 1)[0]
    assert DEFAULT_UPDATE_MEMORY_PROMPT == upstream_block


def test_update_memory_prompt_keeps_upstreams_trailing_whitespace():
    """Two lines of `DEFAULT_UPDATE_MEMORY_PROMPT` end in whitespace upstream.

    This is the specific thing hand-transcription loses without a diff showing
    anything, which is why the constant is generated from the clone. Asserted
    independently of the byte-equality test above so that a whitespace-stripping
    tool run over this repo fails with a message naming what it broke.
    """
    stripped = [
        i for i, ln in enumerate(DEFAULT_UPDATE_MEMORY_PROMPT.splitlines()) if ln != ln.rstrip()
    ]
    assert stripped == [40, 41]


def test_update_envelope_interpolates_python_repr_not_json():
    old = [{"id": "0", "text": "Loves cheese pizza"}]
    prompt = get_update_memory_messages(old, ["Dislikes cheese pizza"])
    assert "[{'id': '0', 'text': 'Loves cheese pizza'}]" in prompt  # repr, not json.dumps
    assert "['Dislikes cheese pizza']" in prompt
    assert '"event" : "<Operation to be performed>"' in prompt
    assert DEFAULT_UPDATE_MEMORY_PROMPT in prompt  # default prompt is the head of the envelope


def test_update_envelope_keeps_the_instruction_block_after_the_json_schema():
    """The envelope does not end at the JSON structure.

    Upstream continues with a seven-line instruction block and a final "Do not
    return anything except the JSON format." (prompts.py:318-325 @ v0.1.94). The
    plan that specified this port reproduced the function only up to the schema,
    so an implementer following the plan rather than the clone would have
    shipped a truncated prompt that still looks complete.
    """
    prompt = get_update_memory_messages([], ["a fact"])
    assert "Follow the instruction mentioned below:" in prompt
    assert "If the current memory is empty, then you have to add the new retrieved facts" in prompt
    assert prompt.rstrip().endswith("Do not return anything except the JSON format.")


def test_update_envelope_is_byte_identical_to_upstream_for_the_same_inputs():
    """Strongest form: build the envelope from the clone's own source and compare.

    Covers the parts the substring assertions above cannot — the four-space
    continuation indentation, blank-line placement, and the custom-prompt
    default substitution — without re-typing any of it here.
    """
    src = _upstream_prompts_py()
    ns: dict = {}
    exec(compile(src, "upstream_prompts.py", "exec"), ns)  # noqa: S102 — pinned local file
    old = [{"id": "0", "text": "Loves cheese pizza"}]
    facts = ["Dislikes cheese pizza"]
    assert get_update_memory_messages(old, facts) == ns["get_update_memory_messages"](old, facts)


def test_custom_update_prompt_replaces_only_the_head():
    prompt = get_update_memory_messages([], [], custom_update_memory_prompt="CUSTOM HEAD")
    assert prompt.startswith("CUSTOM HEAD")
    assert DEFAULT_UPDATE_MEMORY_PROMPT not in prompt
    assert "You must return your response in the following JSON structure only:" in prompt


def test_parse_messages_matches_upstream_role_rendering():
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert parse_messages(msgs) == "user: a\nassistant: b\n"


def test_parse_messages_drops_unknown_roles_like_upstream():
    # memory/utils.py:9-18 @ v0.1.94 tests three roles with three independent
    # `if`s and has no else — a "tool" or "function" message vanishes. Reproduced,
    # not fixed: our facade only ever emits `user`, and repairing it here would
    # make the port diverge on a path upstream leaves broken.
    msgs = [{"role": "tool", "content": "x"}, {"role": "system", "content": "s"}]
    assert parse_messages(msgs) == "system: s\n"


# --------------------------------------------------------------------------
# Task 3: the two-phase write. Every assertion pins an upstream site by line.
# --------------------------------------------------------------------------

from helpers import StubLLM, make_mem_multi

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode
from agmem.organizers.mem0 import Mem0Organizer


def _msg(text, date="12 May 2023", speaker="Caroline"):
    return Episode(content=f"({date}) {speaker}: {text}", meta={"speaker": speaker, "date": date})


def _shown_order(decision_prompt, texts):
    """The candidate texts in the order the decision prompt numbered them.

    The prompt embeds the candidate list as a Python `repr`, so position in the
    string IS the integer id the model is told to answer with. Recovering it here
    lets a test assert which uuid an id maps back to, rather than only that the
    set of mapped uuids is right — the difference between "renumbering happened"
    and "renumbering is correct".
    """
    return sorted(texts, key=decision_prompt.index)


def _seed(mem, pairs):
    """Put semantic items in the store the way an earlier add() would have."""
    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.ADD,
                target_type="semantic",
                target_id=i,
                payload={"id": i, "content": t, "embedding_text": t},
            )
            for i, t in pairs
        ],
        actor="mem0",
    )


def test_add_is_exactly_two_llm_calls_even_with_zero_facts():
    org = Mem0Organizer()
    llm = StubLLM({"extract": [{"facts": []}], "distill": [{"memory": []}]})
    mem = make_mem_multi([org], llm)
    org.on_message(_msg("hi"), mem._ctx)
    # study §②-1: the decision call fires unconditionally — main.py:261 is reached
    # even when new_retrieved_facts is empty. F is irrelevant to the call count.
    assert [role for role, _ in llm.calls] == ["extract", "distill"]
    mem.close()


def test_dropped_extract_still_makes_the_decision_call():
    """A parse failure upstream sets facts=[] and falls THROUGH (main.py:229-234).

    If the port returned early here, the per-add call count would depend on
    whether the model happened to emit valid JSON — and the whole claim under
    test is that the count is exactly 2 regardless.
    """
    org = Mem0Organizer()
    llm = StubLLM({"distill": [{"memory": []}]})  # no extract response queued -> None
    mem = make_mem_multi([org], llm)
    org.on_message(_msg("hi"), mem._ctx)
    assert [role for role, _ in llm.calls] == ["extract", "distill"]
    assert org.discarded["no_facts"] == 1
    mem.close()


def test_fact_retrieval_prompt_is_the_system_message():
    # upstream sends FACT_RETRIEVAL_PROMPT as system and "Input:\n{parsed}" as
    # user (main.py:219-227 via utils.get_fact_retrieval_messages). Ours must
    # too, or the prompt is a different prompt.
    org = Mem0Organizer()
    llm = StubLLM({"extract": [{"facts": []}], "distill": [{"memory": []}]})
    mem = make_mem_multi([org], llm)
    org.on_message(_msg("hi"), mem._ctx)
    assert llm.systems[0].startswith("You are a Personal Information Organizer")
    assert llm.calls[0][1] == "Input:\nuser: (12 May 2023) Caroline: hi\n"
    mem.close()


def test_update_decision_is_one_batched_call_for_all_facts():
    org = Mem0Organizer()
    llm = StubLLM({"extract": [{"facts": ["f1", "f2", "f3"]}], "distill": [{"memory": []}]})
    mem = make_mem_multi([org], llm)
    org.on_message(_msg("hi"), mem._ctx)
    # main.py:261-266 builds ONE prompt over the whole fact list
    assert sum(1 for role, _ in llm.calls if role == "distill") == 1
    mem.close()


def test_retrieval_is_topk_per_fact_unioned_deduped_and_renumbered():
    """Per-fact `limit=5`, no similarity floor, union deduped by id, ids 0..n-1.

    main.py:241-259. The renumbering is upstream's own UUID-hallucination guard,
    and the absence of a score floor is why a weakly-related item still reaches
    the decision prompt.
    """
    org = Mem0Organizer(top_k=2)
    llm = StubLLM({"extract": [{"facts": ["f1", "f2"]}], "distill": [{"memory": []}]})
    mem = make_mem_multi([org], llm)
    _seed(mem, [(f"m{i}", f"seeded fact {i}") for i in range(6)])
    org.on_message(_msg("hi"), mem._ctx)

    decision_prompt = next(p for role, p in llm.calls if role == "distill")
    # 2 facts x top_k 2 = at most 4 candidates, at least 2 (dedup may collapse them)
    shown_ids = [f"'id': '{i}'" for i in range(4)]
    present = [s for s in shown_ids if s in decision_prompt]
    assert 2 <= len(present) <= 4
    assert present == shown_ids[: len(present)]  # contiguous from 0, no UUIDs shown
    assert "m0" not in decision_prompt and "m5" not in decision_prompt
    mem.close()


def test_events_map_to_ops_with_upstream_side_effects():
    org = Mem0Organizer()
    llm = StubLLM(
        {
            "extract": [{"facts": ["f1"]}],
            "distill": [
                {
                    "memory": [
                        {"id": "0", "text": "kept as is", "event": "NONE"},
                        {"id": "1", "text": "revised text", "event": "UPDATE"},
                        {"id": "2", "text": "gone", "event": "DELETE"},
                        {"id": "9", "text": "brand new", "event": "ADD"},
                    ]
                }
            ],
        }
    )
    mem = make_mem_multi([org], llm)
    _seed(mem, [("m0", "alpha"), ("m1", "beta"), ("m2", "gamma")])
    ops = org.on_message(_msg("hi"), mem._ctx)

    by_op = {op.op: op for op in ops}
    assert set(by_op) == {OpType.NOOP, OpType.UPDATE, OpType.DELETE, OpType.ADD}
    # ADD's id is fresh, not the "9" the model made up (main.py:296-302 mints a uuid)
    assert by_op[OpType.ADD].target_id not in {"m0", "m1", "m2", "9"}
    assert by_op[OpType.ADD].payload["content"] == "brand new"
    assert "hash" in by_op[OpType.ADD].payload and "created_at" in by_op[OpType.ADD].payload
    # UPDATE/DELETE/NOOP resolve integer -> real uuid via temp_uuid_mapping. Pin
    # the mapping element-wise, not as a set: id "0" must be the FIRST candidate
    # the prompt showed, "1" the second, "2" the third. A reversed or shuffled
    # mapping would satisfy a set comparison while sending every verdict to the
    # wrong item.
    decision_prompt = next(p for role, p in llm.calls if role == "distill")
    order = _shown_order(decision_prompt, ["alpha", "beta", "gamma"])
    text_of = {"alpha": "m0", "beta": "m1", "gamma": "m2"}
    assert by_op[OpType.NOOP].target_id == text_of[order[0]]
    assert by_op[OpType.UPDATE].target_id == text_of[order[1]]
    assert by_op[OpType.DELETE].target_id == text_of[order[2]]
    # created_at is ABSENT on UPDATE so the facade's merge preserves the original
    assert "created_at" not in by_op[OpType.UPDATE].payload
    assert by_op[OpType.DELETE].payload == {}
    mem.close()


def test_update_preserves_created_at_and_advances_timestamp():
    """main.py:717-721 rebuilds the payload but copies `created_at` off the
    existing item, while the caller's metadata carries the NEW session date."""
    org = Mem0Organizer()
    llm = StubLLM(
        {
            "extract": [{"facts": ["f1"]}],
            "distill": [{"memory": [{"id": "0", "text": "revised", "event": "UPDATE"}]}],
        }
    )
    mem = make_mem_multi([org], llm)
    _seed(mem, [("m0", "original")])
    mem._apply_ops(
        [
            MemoryOp(
                op=OpType.UPDATE,
                target_type="semantic",
                target_id="m0",
                payload={"created_at": "2020-01-01T00:00:00+00:00", "timestamp": "1 Jan 2020"},
            )
        ],
        actor="mem0",
    )
    ops = org.on_message(_msg("hi", date="14 Jun 2023"), mem._ctx)
    mem._apply_ops(ops, actor="mem0")

    item = mem.doc_store.get_items(["m0"], "semantic")[0]
    assert item["created_at"] == "2020-01-01T00:00:00+00:00"  # untouched
    assert item["content"] == "revised"
    assert item["timestamp"] == "14 Jun 2023"  # advanced to the new batch's date
    assert "updated_at" in item
    mem.close()


def test_empty_text_entry_is_discarded_and_counted_regardless_of_event():
    # main.py:286-288 drops any entry with falsy text BEFORE looking at the
    # event, DELETE included. Upstream logs and moves on; we also count.
    org = Mem0Organizer()
    llm = StubLLM(
        {
            "extract": [{"facts": ["f1"]}],
            "distill": [
                {
                    "memory": [
                        {"id": "0", "text": "", "event": "DELETE"},
                        {"id": "0", "text": "", "event": "ADD"},
                    ]
                }
            ],
        }
    )
    mem = make_mem_multi([org], llm)
    _seed(mem, [("m0", "alpha")])
    ops = org.on_message(_msg("hi"), mem._ctx)
    assert ops == []
    assert org.discarded["empty_text"] == 2
    mem.close()


def test_hallucinated_id_is_discarded_and_counted():
    # main.py:304/311/318 index temp_uuid_mapping directly; an out-of-range
    # integer or a real UUID raises KeyError into the inner except (:328-329)
    # and the failed op is not counted as a failure anywhere. Ours is counted.
    org = Mem0Organizer()
    llm = StubLLM(
        {
            "extract": [{"facts": ["f1"]}],
            "distill": [
                {
                    "memory": [
                        {"id": "77", "text": "x", "event": "UPDATE"},
                        {"id": "deadbeef-uuid", "text": "y", "event": "DELETE"},
                    ]
                }
            ],
        }
    )
    mem = make_mem_multi([org], llm)
    _seed(mem, [("m0", "alpha")])
    ops = org.on_message(_msg("hi"), mem._ctx)
    assert ops == []
    assert org.discarded["hallucinated_id"] == 2
    mem.close()


def test_dropped_decision_call_applies_nothing_and_is_counted():
    # StructuredCaller returns None after its retry budget; upstream's equivalent
    # (json parse failure -> [] -> .get on a list -> AttributeError caught at
    # :330-331) also applies nothing.
    org = Mem0Organizer()
    llm = StubLLM({"extract": [{"facts": ["f1"]}]})  # no distill response queued
    mem = make_mem_multi([org], llm)
    ops = org.on_message(_msg("hi"), mem._ctx)
    assert ops == []
    assert org.discarded["no_verdict"] == 1
    mem.close()


def test_batch_size_buffers_and_flush_drains_the_tail():
    org = Mem0Organizer(batch_size=2)
    llm = StubLLM(
        {
            "extract": [{"facts": []}] * 3,
            "distill": [{"memory": []}] * 3,
        }
    )
    mem = make_mem_multi([org], llm)
    assert org.on_message(_msg("a"), mem._ctx) == []  # buffered, no call yet
    assert [r for r, _ in llm.calls] == []
    org.on_message(_msg("b"), mem._ctx)
    assert [r for r, _ in llm.calls] == ["extract", "distill"]  # one add
    org.on_message(_msg("c"), mem._ctx)
    assert [r for r, _ in llm.calls] == ["extract", "distill"]  # tail still buffered
    org.flush_buffer(mem._ctx)
    assert [r for r, _ in llm.calls] == ["extract", "distill", "extract", "distill"]
    assert org.flush_buffer(mem._ctx) == []  # draining an empty buffer costs nothing
    mem.close()


def test_batched_add_renders_both_messages_into_one_prompt():
    org = Mem0Organizer(batch_size=2)
    llm = StubLLM({"extract": [{"facts": []}], "distill": [{"memory": []}]})
    mem = make_mem_multi([org], llm)
    org.on_message(_msg("first", speaker="Caroline"), mem._ctx)
    org.on_message(_msg("second", speaker="Melanie"), mem._ctx)
    assert llm.calls[0][1] == (
        "Input:\nuser: (12 May 2023) Caroline: first\nuser: (12 May 2023) Melanie: second\n"
    )
    mem.close()


def test_no_llm_stores_messages_verbatim():
    # upstream's infer=False path (main.py:204-211): one memory per non-system
    # message, no LLM call at all.
    org = Mem0Organizer(batch_size=2)
    mem = make_mem_multi([org], None)
    mem._ctx.llm = None
    org.on_message(_msg("a"), mem._ctx)
    ops = org.on_message(_msg("b"), mem._ctx)
    assert [op.op for op in ops] == [OpType.ADD, OpType.ADD]
    assert [op.payload["content"] for op in ops] == [
        "(12 May 2023) Caroline: a",
        "(12 May 2023) Caroline: b",
    ]
    mem.close()


def test_registered_in_the_organizer_registry():
    # tests/test_stores.py walks ORGANIZERS.values() to check `produces` against
    # MEMORY_TYPES — an unregistered organizer is silently exempt from that check.
    from agmem.organizers import ORGANIZERS

    assert ORGANIZERS["mem0"] is Mem0Organizer
    assert Mem0Organizer.produces == ("semantic",)
