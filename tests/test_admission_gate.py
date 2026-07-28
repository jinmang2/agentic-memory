"""A-MAC admission gate (policies/admission.py).

The ROUGE-L reference values are not hand-derived: they were produced by the
real ``rouge_score`` package (``RougeScorer(["rougeL"], use_stemmer=True)``,
which is what the A-MAC release constructs) in a throwaway venv and pinned here,
so this file is the standing cross-check against upstream's actual metric
without taking the dependency.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from helpers import StubLLM, make_mem_multi

from agmem._porter import PorterStemmer
from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode
from agmem.embed.fake import FakeEmbedder
from agmem.organizers.amem import AMemOrganizer
from agmem.organizers.base import Organizer, OrganizerContext
from agmem.organizers.gated import AdmissionGated
from agmem.policies.admission import (
    PAPER_THRESHOLD,
    PAPER_WEIGHTS,
    AdmissionGate,
    TypePriorClassifier,
    rouge_l_fmeasure,
    rouge_tokenize,
)
from agmem.stores.numpy_vec import NumpyVectorStore
from agmem.stores.sqlite_doc import SqliteDocStore

# (target, prediction, rouge_score's rougeL fmeasure)
ROUGE_REFERENCE = [
    (
        "I prefer Python because it's very readable.",
        "User prefers Python for readability",
        0.4615384615384615,
    ),
    (
        "What programming language do you prefer?",
        "User prefers Python for readability",
        0.1818181818181818,
    ),
    ("It's sunny and warm.", "User lives in Tokyo and has three cats", 0.15384615384615385),
    (
        "I went to the beach with my sister on Saturday.",
        "I went to the beach with my sister on Saturday.",
        1.0,
    ),
    ("the cat sat on the mat", "a cat was sitting on a mat", 0.4615384615384615),
    ("Hello!!! world???", "hello world", 1.0),
    ("running runner runs", "run runs running", 0.6666666666666666),
    ("", "nonempty text", 0.0),
    ("a b c", "c b a", 0.3333333333333333),
    ("My birthday is on March 15th.", "birthday March 15", 0.4444444444444444),
    ("Caroline's dog is named Max", "caroline dog named max", 0.8),
    ("ABC abc AbC", "abc", 0.5),
]


@pytest.mark.parametrize("target,prediction,expected", ROUGE_REFERENCE)
def test_rouge_l_matches_rouge_score_package(target, prediction, expected):
    assert rouge_l_fmeasure(target, prediction) == pytest.approx(expected, abs=1e-12)


def test_rouge_l_is_symmetric_in_its_arguments():
    # Precision/recall swap but their harmonic mean does not.
    a, b = "the cat sat on the mat", "a cat was sitting on a mat"
    assert rouge_l_fmeasure(a, b) == pytest.approx(rouge_l_fmeasure(b, a))


def test_rouge_tokenize_only_stems_tokens_longer_than_three_chars():
    # rouge_score's `len(x) > 3` guard: "runs" is stemmed, "run" and "sat" are not.
    assert rouge_tokenize("runs running sat run") == ["run", "run", "sat", "run"]
    # non-alphanumerics become separators, so a possessive splits in two
    assert rouge_tokenize("Caroline's dog") == ["carolin", "s", "dog"]


def test_rouge_tokenize_can_disable_stemming():
    assert rouge_tokenize("running", stemmer=False) == ["running"]


# ---------------------------------------------------------------------------
# vendored stemmer: the step-1c mode switch
# ---------------------------------------------------------------------------


def test_porter_mode_switches_step1c_condition():
    original, nltk_mode = PorterStemmer(mode="original"), PorterStemmer(mode="nltk")
    # (*v*) Y->I fires on a vowel-preceded y; nltk's (*c and not c) does not.
    assert original.stem("saturday") == "saturdai"
    assert nltk_mode.stem("saturday") == "saturday"
    # ...and the reverse: nltk conflates cry/cried, the 1980 rule does not.
    assert nltk_mode.stem("cry") == nltk_mode.stem("cried") == "cri"
    assert original.stem("cry") == "cry" and original.stem("cried") == "cri"
    # both agree where the letter before y is a consonant
    assert original.stem("happy") == nltk_mode.stem("happy") == "happi"


def test_porter_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        PorterStemmer(mode="martin")


def test_locomo_normalize_keeps_the_scored_mode():
    """The metric stored runs were scored with must not drift silently."""
    from agmem.bench import locomo

    assert locomo._STEMMER.mode == "original"


# ---------------------------------------------------------------------------
# Type Prior
# ---------------------------------------------------------------------------


def test_substring_mode_reproduces_the_release_defect():
    """`'is' in content` hits "sister", so a plain narrative turn scores as `fact`."""
    release = TypePriorClassifier(matching="substring")
    turn = "I went to the beach with my sister on Saturday."
    assert release.classify(turn) == "fact"
    assert release.score(turn) == 0.7


def test_word_mode_does_not_match_keywords_inside_words():
    fixed = TypePriorClassifier(matching="word")
    turn = "I went to the beach with my sister on Saturday."
    assert fixed.classify(turn) == "unknown"
    assert fixed.score(turn) == 0.5
    # a real whole-word hit still classifies
    assert fixed.classify("I prefer tea over coffee.") == "preference"
    assert fixed.score("I prefer tea over coffee.") == 0.9


def test_word_mode_matches_multiword_keys_and_contractions():
    fixed = TypePriorClassifier(matching="word")
    assert fixed.classify("My name is Caroline and I live in Boston.") == "identity"
    assert fixed.classify("I'm a nurse") == "identity"


def test_declared_type_short_circuits_the_rules():
    fixed = TypePriorClassifier()
    assert fixed.classify("anything at all", declared_type="plan") == "plan"
    assert fixed.score("anything at all", declared_type="plan") == 0.5
    # an unmapped declared type falls back to the `unknown` prior, as upstream
    assert fixed.score("x", declared_type="not-a-category") == 0.5


def test_type_tie_breaks_on_keyword_dict_order():
    """The release resolves ties with `max`, which keeps the first maximum, so
    the winner is whichever type comes first in TYPE_KEYWORDS."""
    fixed = TypePriorClassifier(matching="word")
    # one `preference` hit ("like") and one `plan` hit ("tomorrow"): preference
    # is declared first, so it wins the 1-1 tie.
    assert fixed.classify("i like tomorrow") == "preference"


def test_unknown_matching_mode_rejected():
    with pytest.raises(ValueError, match="matching must be"):
        TypePriorClassifier(matching="regex")


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def _ctx(llm=None):
    return OrganizerContext(
        doc_store=SqliteDocStore(None),
        vector_store=NumpyVectorStore(dim=128),
        embedder=FakeEmbedder(dim=128),
        namespace="t",
        llm=llm,
    )


def _ep(content, **meta):
    return Episode(content=content, namespace="t", meta=meta)


@pytest.mark.parametrize(
    "weights,match",
    [
        ((0.5, 0.5), "5 entries"),
        ((0.2, 0.2, 0.2, 0.2, 0.3), "sum to 1"),
        ((-0.1, 0.2, 0.3, 0.3, 0.3), "non-negative"),
    ],
)
def test_weight_constraints_are_enforced(weights, match):
    with pytest.raises(ValueError, match=match):
        AdmissionGate(weights=weights)


def test_paper_operating_point_is_the_default():
    gate = AdmissionGate()
    assert gate.weights == PAPER_WEIGHTS
    assert gate.threshold == PAPER_THRESHOLD
    assert gate.use_utility is False  # never spend an LLM call unasked


def test_novelty_is_one_against_an_empty_index():
    gate = AdmissionGate()
    assert gate.novelty("brand new content", _ctx()) == 1.0


def test_novelty_falls_with_a_near_duplicate_in_the_index():
    ctx = _ctx()
    text = "Caroline went to the LGBTQ support group"
    ctx.vector_store.add("n1", ctx.embedder.embed([text])[0], "notes", "t")
    gate = AdmissionGate()
    assert gate.novelty(text, ctx) == pytest.approx(0.0, abs=1e-6)
    # a different type is not consulted, so N stays 1.0
    other = AdmissionGate(novelty_types=("pages",))
    assert other.novelty(text, ctx) == 1.0


def test_recency_is_degenerate_for_a_freshly_arrived_turn():
    """Documented finding: age ~ 0 at admission time, so R is pinned at 1.0."""
    gate = AdmissionGate()
    assert gate.recency(_ep("hi")) == 1.0


def test_recency_follows_the_paper_decay_when_a_reference_time_is_given():
    gate = AdmissionGate(decay_rate=0.01)
    ts = datetime(2023, 5, 1, 10, 0, 0, tzinfo=timezone.utc)
    ep = Episode(content="hi", namespace="t", timestamp=ts)
    assert gate.recency(ep, now=ts + timedelta(hours=69.3)) == pytest.approx(0.5, abs=1e-3)
    # the release's wall-clock reference underflows R to ~0 over LoCoMo spans
    assert gate.recency(ep, now=ts + timedelta(days=1000)) == pytest.approx(0.0, abs=1e-9)


def test_confidence_needs_lexical_overlap_with_an_earlier_turn():
    gate, ctx = AdmissionGate(), _ctx()
    assert gate.confidence("I prefer Python") == 0.0  # empty transcript
    gate.decide(_ep("I prefer Python because it is readable"), ctx)
    assert gate.confidence("I prefer Python") > 0.5
    # no shared token -> the span filter drops every candidate span
    assert gate.confidence("quantum chromodynamics") == 0.0


def test_confidence_window_bounds_the_transcript_scanned():
    ctx = _ctx()
    gate = AdmissionGate(history_window=1)
    gate.decide(_ep("apples and oranges"), ctx)
    gate.decide(_ep("completely unrelated line"), ctx)
    # only the most recent turn is visible, and it shares nothing
    assert gate.confidence("apples and oranges") == 0.0


def test_transcript_records_rejected_turns_too():
    """C asks what the conversation said, not what memory kept."""
    ctx = _ctx()
    gate = AdmissionGate()
    rejected = gate.decide(_ep("Wow."), ctx)
    assert not rejected.admit
    assert gate._history == ["Wow."]


def test_paper_weights_admit_a_preference_and_reject_an_interjection():
    ctx = _ctx()
    gate = AdmissionGate()
    assert gate.decide(_ep("I prefer tea over coffee."), ctx).admit
    assert not gate.decide(_ep("Wow."), ctx).admit


def test_utility_short_circuit_skips_the_llm_when_the_lower_bound_admits():
    llm = StubLLM({"admit": [{"score": 0.0}]})
    ctx = _ctx(llm)
    gate = AdmissionGate(use_utility=True)
    # T=0.9 -> base = 0.1*N + 0.1*R + 0.6*0.9 = 0.74 >= theta already
    decision = gate.decide(_ep("I prefer tea over coffee."), ctx)
    assert decision.admit
    assert decision.utility_evaluated is False
    assert decision.features.utility is None
    assert decision.score == pytest.approx(0.74)  # the U=0 lower bound that proved it
    assert llm.calls == []
    assert gate.stats.as_dict()["utility_skipped"] == 1


def test_utility_short_circuit_skips_the_llm_when_the_upper_bound_rejects():
    llm = StubLLM({"admit": [{"score": 1.0}]})
    ctx = _ctx(llm)
    gate = AdmissionGate(use_utility=True)
    # T=0.2 -> base = 0.32, and even U=1.0 only reaches 0.42 < 0.55
    decision = gate.decide(_ep("nothing here", content_type="temporary"), ctx)
    assert not decision.admit
    assert decision.utility_evaluated is False
    assert decision.score == pytest.approx(0.42)  # the U=1 upper bound that ruled it out
    assert llm.calls == []


def test_paper_weights_still_call_the_llm_inside_the_straddling_band():
    """The short-circuit is exact, not a blanket skip: a turn scoring in
    [theta - w_U, theta) genuinely needs U, and under the paper's weights the
    `unknown` category with low confidence lands exactly there. So enabling
    `use_utility` is not free for mid-scoring turns."""
    llm = StubLLM({"admit": [{"score": 1.0}]})
    ctx = _ctx(llm)
    gate = AdmissionGate(use_utility=True)
    # T=0.5 (unknown), C=0.0 -> base = 0.50; base < 0.55 <= base + 0.1
    decision = gate.decide(_ep("Wow."), ctx)
    assert decision.utility_evaluated is True
    assert [role for role, _ in llm.calls] == ["admit"]
    assert decision.score == pytest.approx(0.60)
    assert decision.admit


def test_utility_is_called_when_it_can_flip_the_verdict():
    # w_U large enough that U decides: base sits below theta, base + w_U above.
    llm = StubLLM({"admit": [{"score": 1.0}]})
    ctx = _ctx(llm)
    gate = AdmissionGate(weights=(0.5, 0.0, 0.0, 0.0, 0.5), threshold=0.55, use_utility=True)
    decision = gate.decide(_ep("Wow."), ctx)  # T=0.5 -> base=0.25, +w_U -> 0.75
    assert decision.utility_evaluated is True
    assert decision.features.utility == 1.0
    assert decision.score == pytest.approx(0.75)
    assert decision.admit
    assert [role for role, _ in llm.calls] == ["admit"]


def test_utility_drop_degrades_to_the_lower_bound():
    """A dropped verdict must not be read as a high utility score."""
    llm = StubLLM({})  # no queued response -> StructuredCaller-style drop
    ctx = _ctx(llm)
    gate = AdmissionGate(weights=(0.5, 0.0, 0.0, 0.0, 0.5), threshold=0.55, use_utility=True)
    decision = gate.decide(_ep("Wow."), ctx)
    assert decision.utility_evaluated is False
    assert not decision.admit  # base=0.25 alone cannot clear 0.55
    assert decision.score == pytest.approx(0.75)  # reported as the ruling-out bound


def test_utility_disabled_makes_the_gate_llm_free():
    llm = StubLLM({"admit": [{"score": 1.0}]})
    ctx = _ctx(llm)
    gate = AdmissionGate(use_utility=False)
    for text in ("I prefer tea.", "Wow.", "My name is Caroline."):
        gate.decide(_ep(text), ctx)
    assert llm.calls == []
    assert gate.stats.as_dict()["utility_calls"] == 0


def test_score_equals_the_weighted_sum_when_utility_is_evaluated():
    llm = StubLLM({"admit": [{"score": 0.4}]})
    ctx = _ctx(llm)
    gate = AdmissionGate(weights=(0.5, 0.0, 0.0, 0.0, 0.5), threshold=0.55, use_utility=True)
    decision = gate.decide(_ep("Wow."), ctx)
    f = decision.features
    expected = 0.5 * f.utility + 0.5 * f.type_prior
    assert decision.score == pytest.approx(expected)


def test_stats_track_admit_rate_and_per_feature_means():
    ctx = _ctx()
    gate = AdmissionGate()
    for text in ("I prefer tea over coffee.", "Wow.", "Ugh."):
        gate.decide(_ep(text), ctx)
    stats = gate.stats.as_dict()
    assert stats["seen"] == 3
    assert stats["admitted"] == 1
    assert stats["rejected"] == 2
    assert stats["admit_rate"] == pytest.approx(1 / 3)
    assert stats["type_counts"] == {"preference": 1, "unknown": 2}
    # U was never observed, so its mean is 0.0 rather than a division by zero
    assert stats["utility_observations"] == 0
    assert stats["feature_means"]["utility"] == 0.0
    assert stats["feature_means"]["recency"] == pytest.approx(1.0)


def test_decision_as_dict_is_json_safe_and_complete():
    decision = AdmissionGate().decide(_ep("I prefer tea."), _ctx())
    row = decision.as_dict()
    assert set(row) == {
        "admit",
        "score",
        "threshold",
        "content_type",
        "utility_evaluated",
        "utility",
        "confidence",
        "novelty",
        "recency",
        "type_prior",
    }
    assert row["utility"] is None
    assert isinstance(row["admit"], bool)


def test_features_to_tuple_substitutes_an_unevaluated_utility():
    decision = AdmissionGate().decide(_ep("Wow."), _ctx())
    assert decision.features.to_tuple()[0] == 0.0
    assert decision.features.to_tuple(utility_default=0.5)[0] == 0.5


def test_content_type_can_be_declared_through_episode_meta():
    decision = AdmissionGate().decide(_ep("Wow.", content_type="preference"), _ctx())
    assert decision.content_type == "preference"
    assert decision.features.type_prior == 0.9
    assert decision.admit  # 0.6*0.9 + 0.1*1.0(N) = 0.64 >= 0.55


# ---------------------------------------------------------------------------
# AMemOrganizer integration
# ---------------------------------------------------------------------------


def _amem_llm():
    return StubLLM(
        {
            "extract": [{"keywords": ["tea"], "context": "beverage", "tags": ["pref"]}],
            "distill": [{"should_evolve": False, "connections": []}],
        }
    )


def test_amem_stores_every_message_when_ungated():
    """A-Mem's paper behaviour is the baseline the gate is measured against, and
    the mechanism must not know policies exist."""
    llm = _amem_llm()
    ctx = _ctx(llm)
    org = AMemOrganizer()
    assert not hasattr(org, "admission")  # no policy coupling on the mechanism
    ops = org.on_message(_ep("Wow."), ctx)
    assert len(ops) == 1
    assert [role for role, _ in llm.calls] == ["extract"]


def test_gated_amem_spends_no_llm_calls_on_a_rejected_message():
    llm = _amem_llm()
    ctx = _ctx(llm)
    org = AdmissionGated(AMemOrganizer(), AdmissionGate())
    assert org.on_message(_ep("Wow."), ctx) == []
    assert llm.calls == []  # both Ps1 and Ps2/Ps3 skipped
    stats = org.gate.stats.as_dict()
    assert (stats["seen"], stats["admitted"]) == (1, 0)


def test_gated_amem_still_ingests_an_admitted_message():
    llm = _amem_llm()
    ctx = _ctx(llm)
    org = AdmissionGated(AMemOrganizer(), AdmissionGate())
    ops = org.on_message(_ep("I prefer tea over coffee."), ctx)
    assert len(ops) == 1
    assert [role for role, _ in llm.calls] == ["extract"]
    assert org.gate.stats.as_dict()["admitted"] == 1


def test_gate_rejection_leaves_no_trace_in_the_op_log():
    """A rejected turn must produce zero ops, not an op the facade then filters."""
    ctx = _ctx(_amem_llm())
    org = AdmissionGated(AMemOrganizer(), AdmissionGate())
    produced = [op for text in ("Wow.", "Ugh.", "Hmm.") for op in org.on_message(_ep(text), ctx)]
    assert produced == []


_E2E_TURNS = (
    "I prefer tea over coffee.",  # preference 0.9 -> admit
    "Wow.",  # unknown 0.5 -> reject
    "My name is Caroline and I live in Boston.",  # identity 0.9 -> admit
    "Ugh.",  # unknown -> reject
    "I went to the beach with my sister on Saturday.",  # the `is`-in-"sister" turn
)


def _run_e2e(gate):
    """Drive the assembled facade, not the organizer in isolation."""
    n = len(_E2E_TURNS)
    llm = StubLLM(
        {
            "extract": [{"keywords": ["k"], "context": "c", "tags": ["t"]}] * n,
            "distill": [{"should_evolve": False, "connections": []}] * n,
        }
    )
    org = AMemOrganizer() if gate is None else AdmissionGated(AMemOrganizer(), gate)
    mem = make_mem_multi([org], llm)
    for turn in _E2E_TURNS:
        mem.add_message(turn)
    return len(mem.doc_store.list_items("notes", "t")), len(llm.calls), org


def test_end_to_end_gate_cuts_notes_and_write_calls_through_the_facade():
    notes_ungated, calls_ungated, _ = _run_e2e(None)
    # 5 extract + 4 distill: the first note has no neighbours, so its Ps2/Ps3 is skipped
    assert (notes_ungated, calls_ungated) == (5, 9)

    notes_gated, calls_gated, org = _run_e2e(AdmissionGate())
    assert notes_gated == 2  # preference + identity
    assert calls_gated == 3  # 2 extract + 1 distill
    stats = org.gate.stats.as_dict()
    assert stats["seen"] == 5 and stats["admitted"] == 2
    assert stats["admit_rate"] == pytest.approx(0.4)
    assert calls_gated < calls_ungated  # the whole point: fewer write-path calls


def test_substring_matching_admits_more_which_is_defect_2_in_our_own_pipeline():
    """The release's substring bug is not abstract: restoring it admits the
    "…with my sister…" turn as `fact`, raising the admit rate on our data."""
    _, calls_word, word_org = _run_e2e(AdmissionGate(type_matching="word"))
    notes_sub, calls_sub, sub_org = _run_e2e(AdmissionGate(type_matching="substring"))
    assert notes_sub == 3  # the `sister` turn is now admitted
    assert calls_sub > calls_word
    assert sub_org.gate.stats.as_dict()["type_counts"]["fact"] == 1
    assert "fact" not in word_org.gate.stats.as_dict()["type_counts"]


def test_gate_math_matches_a_hand_computed_score():
    """Pin the formula end to end: S = w.f with the paper's weights."""
    ctx = _ctx()
    gate = AdmissionGate()
    decision = gate.decide(_ep("I prefer tea over coffee."), ctx)
    f = decision.features
    assert f.type_prior == 0.9  # preference
    assert f.novelty == 1.0  # empty index
    assert f.recency == 1.0  # age ~ 0
    assert f.confidence == 0.0  # empty transcript
    expected = 0.1 * 0.0 + 0.1 * 1.0 + 0.1 * 1.0 + 0.6 * 0.9
    assert decision.score == pytest.approx(expected)
    assert decision.score == pytest.approx(0.74)
    assert decision.admit and not math.isnan(decision.score)


# ---------------------------------------------------------------------------
# AdmissionGated: the claim that policies/ is cross-cutting
# ---------------------------------------------------------------------------


class _Recording(Organizer):
    """Minimal mechanism double: records what reached it, produces one type."""

    name = "recording"
    produces = ("notes",)
    consumes = ("episodes",)

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.warm: list[str] = []
        self.tasks: list[str] = []

    def on_message(self, episode, ctx):
        self.seen.append(episode.content)
        return [MemoryOp(op=OpType.ADD, target_type="notes", target_id=episode.id, payload={})]

    def warm_start(self, corpus, ctx):
        self.warm = [e.content for e in corpus]
        return []

    def on_task_end(self, trajectory, outcome, task, ctx):
        self.tasks.append(task)
        return []


def test_gate_applies_to_an_arbitrary_organizer_not_just_amem():
    """The point of putting the gate in policies/: it is not A-Mem-specific."""
    inner = _Recording()
    org = AdmissionGated(inner, AdmissionGate())
    ctx = _ctx()
    for text in ("I prefer tea over coffee.", "Wow."):
        org.on_message(_ep(text), ctx)
    assert inner.seen == ["I prefer tea over coffee."]  # the interjection never arrived


def test_wrapper_mirrors_identity_so_the_gate_is_invisible_downstream():
    inner = _Recording()
    org = AdmissionGated(inner, AdmissionGate())
    assert (org.name, org.produces, org.consumes) == ("recording", ("notes",), ("episodes",))


def test_wrapper_forwards_the_consolidate_cursor_scope():
    """AgenticMemory stamps _cursor_scope on the instance it holds — the wrapper —
    so it must reach the wrapped organizer or persisted progress would fork."""
    inner = _Recording()
    org = AdmissionGated(inner, AdmissionGate())
    org._cursor_scope = "recording#1"
    assert inner._cursor_scope == "recording#1"
    assert inner.cursor_key == "consolidate:recording#1"


def test_unset_novelty_types_default_from_the_wrapped_organizers_produces():
    """A gate left at novelty_types=None must compare N against the host's own
    output types; otherwise wrapping any non-A-Mem organizer searches "notes",
    finds nothing, and reintroduces upstream defect (a)'s N == 1.0."""

    class _Facts(_Recording):
        produces = ("facts",)

    ctx = _ctx()
    text = "Caroline went to the LGBTQ support group"
    ctx.vector_store.add("f1", ctx.embedder.embed([text])[0], "facts", "t")

    org = AdmissionGated(_Facts(), AdmissionGate())
    assert org.gate.novelty_types == ("facts",)
    assert org.gate.novelty(text, ctx) == pytest.approx(0.0, abs=1e-6)

    # an explicit novelty_types from the caller — even ("notes",) — is honored
    explicit = AdmissionGated(_Facts(), AdmissionGate(novelty_types=("notes",)))
    assert explicit.gate.novelty_types == ("notes",)
    assert explicit.gate.novelty(text, ctx) == 1.0

    # standalone (unwrapped) use keeps the A-Mem fallback at decide time
    assert AdmissionGate().novelty_types is None
    assert AdmissionGate().novelty(text, ctx) == 1.0  # nothing under "notes"


def test_warm_start_is_filtered_because_hosts_override_it():
    """A gate living only in on_message would be bypassed by Nemori/MemoryOS,
    which replace warm_start with their own replay."""
    inner = _Recording()
    org = AdmissionGated(inner, AdmissionGate())
    corpus = [_ep("I prefer tea over coffee."), _ep("Wow."), _ep("My name is Caroline.")]
    org.warm_start(corpus, _ctx())
    assert inner.warm == ["I prefer tea over coffee.", "My name is Caroline."]
    assert inner.seen == []  # warm_start must not double-dispatch through on_message


def test_task_end_passes_through_ungated():
    """Admission scores a candidate against a transcript; a trajectory is neither."""
    inner = _Recording()
    org = AdmissionGated(inner, AdmissionGate())
    org.on_task_end([{"step": 1}], "success", "some task", _ctx())
    assert inner.tasks == ["some task"]
    assert org.gate.stats.as_dict()["seen"] == 0  # the gate was never consulted


def test_decisions_are_retained_for_artifact_capture():
    org = AdmissionGated(_Recording(), AdmissionGate())
    ctx = _ctx()
    for text in ("I prefer tea over coffee.", "Wow."):
        org.on_message(_ep(text), ctx)
    assert [d.admit for d in org.decisions] == [True, False]
    assert all(set(d.as_dict()) for d in org.decisions)


def test_gating_passthrough_is_a_no_op_by_construction():
    """Documented applicability limit: passthrough emits no ops and the facade
    already stored the raw episode, so a veto cannot change anything."""
    from agmem.organizers.passthrough import PassthroughOrganizer

    ctx = _ctx()
    plain, gated = PassthroughOrganizer(), AdmissionGated(PassthroughOrganizer(), AdmissionGate())
    assert plain.on_message(_ep("Wow."), ctx) == gated.on_message(_ep("Wow."), ctx) == []


def test_task_driven_organizers_have_no_message_hook_to_gate():
    """Pins the applicability matrix in gated.py: ACE / G-Memory / ReasoningBank
    are on_task_end-driven, so an episode-keyed gate has nothing to decide."""
    from agmem.organizers.ace import ACEOrganizer
    from agmem.organizers.base import overrides
    from agmem.organizers.gmemory import GMemoryOrganizer
    from agmem.organizers.reasoning_bank import ReasoningBankOrganizer

    for cls in (ACEOrganizer, GMemoryOrganizer, ReasoningBankOrganizer):
        org = cls()
        assert not overrides(org, "on_message"), cls.__name__
        assert overrides(org, "on_task_end"), cls.__name__
