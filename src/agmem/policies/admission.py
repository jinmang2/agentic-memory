"""A-MAC admission control — a write-path gate in front of an organizer (arXiv:2603.04549).

A-MAC ("Adaptive Memory Admission Control for LLM Agents Using Weighted Feature
Scoring", ICLR'26 submission) scores each candidate turn on five interpretable
features and admits it only when the weighted sum clears a threshold::

    S(m) = w_U*U(m) + w_C*C(m) + w_N*N(m) + w_R*R(m) + w_T*T(m),  admit iff S(m) >= theta

with ``w_i >= 0`` and ``sum(w_i) == 1``. The paper's motivation is A-Mem
specifically: its Table 1 reports A-Mem at recall 1.000 / precision 0.371,
i.e. A-Mem stores every turn and ~63% of the notes it writes are never cited by
any question. That makes ``AMemOrganizer`` the natural host, and the gate sits
in front of ``_ingest`` so a rejected turn costs **zero** LLM calls instead of
A-Mem's two (Ps1 construction + Ps2/Ps3 evolution).

Why this module re-derives the features instead of porting the release
---------------------------------------------------------------------
The official code (github.com/GuilinDev/Adaptive_Memory_Admission_Control_LLM_Agents,
single commit 40407ae, MIT) was read end-to-end alongside the paper. Four
defects make a literal port reproduce artifacts rather than the mechanism;
each is verified in ``docs/research/amac-admission-gate.md`` and handled here:

1. **N and R are constants in the released feature pipeline.**
   ``optimize_weights_cv.extract_features_for_candidates`` calls
   ``novelty.score(memory, [])`` — existing memories hardcoded to the empty
   list, and ``NoveltyExtractor`` returns 1.0 for an empty set, so N == 1.0 for
   every candidate and SBERT is never consulted. It also passes
   ``current_time=time.time()`` while LoCoMo timestamps are seeded from
   ``2023-05-01``; at lambda=0.01/h that is ``exp(-284) ~ 6e-124``, so R == 0.0
   for every candidate. Three of five features carry signal, not five.
2. **Type Prior matches keywords as bare substrings.** ``'is' in content`` is
   true for "sister", "this", "island"; the ``fact`` keyword set is
   ``{'is','are','was','were',...}``, so almost any substantive English turn
   classifies as ``fact`` (prior 0.7). Under the release's weights that alone
   decides admission — see ``TypePriorClassifier``.
3. **The 5-fold CV never fits on the training folds.** ``optimize_weights_cv``
   splits into ``(train_idx, val_idx)`` but only ever evaluates
   ``(X_val, y_val)``; ``X_train``/``y_train`` are dead locals. Weight/threshold
   selection is therefore performed on the same data the numbers are read off.
4. **The paper's "part-of-speech cues" for Type Prior are absent from the
   release** — the released classifier is keyword sets plus, in a non-default
   subclass, a few regexes, with a ``TODO: Could integrate with spaCy NER``.
   We implement what shipped and do not invent the POS stage.

Consequence for anyone about to measure: the release's weights
``[0.1, 0.1, 0.1, 0.1, 0.6]`` / ``theta=0.55`` were fit **on top of defects 1
and 2**. ("Release's", not "published": the paper discloses theta* = 0.55 —
Table 3 — but never the weight vector; its Table 2 gives only relative feature
importance, so the vector exists ONLY in the released code. docs/16 session 8.)
With N pinned at 1.0, R at 0.0 and Type Prior saturating at ``fact``,
that operating point reduces to "admit iff the turn is not a bare
interjection". The values are kept as this module's defaults so the paper's
configuration is reachable verbatim, but they are *not* transferable to the
debugged features; re-tuning needs a labeled pass, which is a measurement.
``type_matching="substring"`` reproduces defect 2 exactly for that comparison.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agmem._porter import PorterStemmer
from agmem.core.types import Episode
from agmem.organizers.base import OrganizerContext

logger = logging.getLogger("agmem.policies.admission")

# ---------------------------------------------------------------------------
# ROUGE-L (Confidence feature)
# ---------------------------------------------------------------------------

# Tokenizer transcribed from google-research/rouge/tokenize.py, which is what
# `rouge_score.rouge_scorer.RougeScorer([...], use_stemmer=True)` runs and what
# the A-MAC release's ConfidenceExtractor constructs. The `len(x) > 3` guard is
# load-bearing and easy to miss: short tokens are left unstemmed.
_NON_ALPHANUM = re.compile(r"[^a-z0-9]+")
_VALID_TOKEN = re.compile(r"^[a-z0-9]+$")
# mode="nltk" because rouge_score stems with nltk's default. Not bit-exact with
# it — see agmem._porter on the ~1.4%-of-vocabulary residual — but C compares a
# candidate against transcript spans through this same tokenizer on both sides,
# so a consistent relabeling of a stem class leaves the LCS unchanged.
_STEMMER = PorterStemmer(mode="nltk")


def rouge_tokenize(text: str, stemmer: bool = True) -> list[str]:
    """``rouge_score``'s tokenizer: lowercase, non-alphanumerics to spaces,
    whitespace split, Porter-stem tokens longer than 3 chars, drop the rest."""
    lowered = _NON_ALPHANUM.sub(" ", str(text).lower())
    tokens = re.split(r"\s+", lowered)
    if stemmer:
        tokens = [_STEMMER.stem(tok) if len(tok) > 3 else tok for tok in tokens]
    return [tok for tok in tokens if _VALID_TOKEN.match(tok)]


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence, rolling one row of the DP."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[j] + 1)
            else:
                current.append(max(current[j], previous[j + 1]))
        previous = current
    return previous[-1]


def rouge_l_fmeasure(target: str, prediction: str, stemmer: bool = True) -> float:
    """ROUGE-L F-measure, argument order matching ``RougeScorer.score(target, prediction)``.

    Precision is over the prediction's tokens, recall over the target's, and the
    F-measure is their harmonic mean — so the value is symmetric in the two
    arguments even though the call is not. Reimplemented rather than taking the
    ``rouge_score`` dependency (which pulls ``absl-py``/``nltk``/``six`` for one
    feature) on the same terms as ``agmem._porter``: a transcription of a fully
    specified algorithm, not a naive stand-in. Cross-checked against
    ``rouge_score`` itself on 12 pairs — tokenization and F-measure both match
    except where the vendored stemmer's residual nltk gap bites
    (tests/test_admission_gate.py pins the reference values)."""
    target_tokens = rouge_tokenize(target, stemmer)
    prediction_tokens = rouge_tokenize(prediction, stemmer)
    if not target_tokens or not prediction_tokens:
        return 0.0
    lcs = _lcs_length(target_tokens, prediction_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Type Prior (T)
# ---------------------------------------------------------------------------

# Priors and keyword sets transcribed verbatim from features/type_prior.py.
# KEYWORD ORDER IS LOAD-BEARING: the release resolves ties with
# `max(type_scores.items(), key=...)`, and `max` returns the first maximum, so
# the winner of an equal-count tie is whichever type comes first in this dict.
TYPE_PRIORS: dict[str, float] = {
    "preference": 0.9,
    "identity": 0.9,
    "belief": 0.9,
    "value": 0.9,
    "fact": 0.7,
    "knowledge": 0.7,
    "information": 0.7,
    "plan": 0.5,
    "goal": 0.5,
    "intention": 0.5,
    "task": 0.5,
    "temporary": 0.2,
    "ephemeral": 0.2,
    "transient": 0.2,
    "state": 0.3,
    "unknown": 0.5,
}

TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "preference": (
        "prefer",
        "like",
        "dislike",
        "hate",
        "love",
        "favorite",
        "enjoy",
        "appreciate",
        "avoid",
        "want",
        "wish",
    ),
    "identity": (
        "i am",
        "i'm",
        "my name is",
        "called",
        "live in",
        "from",
        "work as",
        "job",
        "profession",
        "nationality",
        "age",
    ),
    "fact": (
        "is",
        "are",
        "was",
        "were",
        "always",
        "never",
        "true",
        "false",
        "known",
        "established",
        "defined",
    ),
    "plan": (
        "will",
        "going to",
        "plan to",
        "intend to",
        "scheduled",
        "planning",
        "upcoming",
        "future",
        "tomorrow",
        "next",
    ),
    "goal": (
        "want to",
        "hope to",
        "aim to",
        "goal",
        "objective",
        "target",
        "aspire",
        "strive",
    ),
    "temporary": (
        "currently",
        "right now",
        "at the moment",
        "today",
        "this morning",
        "this afternoon",
        "temporarily",
    ),
}

_WORDISH = re.compile(r"[^a-z0-9']+")


def _padded(text: str) -> str:
    """Lowercase, collapse everything but letters/digits/apostrophes to single
    spaces, and pad — so ``f" {kw} " in padded`` is a whole-token test that also
    spans multi-word keys ("my name is") and keeps contractions ("i'm")."""
    return f" {_WORDISH.sub(' ', str(text).lower()).strip()} "


class TypePriorClassifier:
    """Rule-based content-category classifier behind the T feature.

    ``matching`` selects how a keyword is tested against the content:

    - ``"word"`` (default) — whole-token/phrase containment. What the release
      evidently meant.
    - ``"substring"`` — the release's literal ``keyword in content``, kept so
      the paper's operating point is reproducible. It is a defect, not a
      preference: ``'is'`` hits "sister" and "this", ``'was'`` hits "wasn't"'s
      stem and any past-tense line, so nearly every substantive turn scores as
      ``fact`` (0.7) while only bare interjections fall through to ``unknown``
      (0.5). Under the release's weights that difference *is* the admission
      decision, which is how the release reaches recall 0.972.

    Neither mode implements the paper's "part-of-speech cues" — no POS tagger
    ships in the release (see module docstring, defect 4)."""

    def __init__(self, matching: str = "word", priors: dict[str, float] | None = None) -> None:
        if matching not in ("word", "substring"):
            raise ValueError(f"matching must be 'word' or 'substring', got {matching!r}")
        self.matching = matching
        self.priors = dict(TYPE_PRIORS)
        if priors:
            self.priors.update(priors)

    def classify(self, content: str, declared_type: str | None = None) -> str:
        """Content category. ``declared_type`` short-circuits the rules, mirroring
        the release's ``memory.metadata['type']`` branch (dead in its own
        pipeline, which always passes ``metadata={}``)."""
        if declared_type:
            return declared_type
        haystack = str(content).lower() if self.matching == "substring" else _padded(content)
        best_type, best_count = None, 0
        for mem_type, keywords in TYPE_KEYWORDS.items():
            if self.matching == "substring":
                count = sum(1 for kw in keywords if kw in haystack)
            else:
                count = sum(1 for kw in keywords if f" {kw} " in haystack)
            # Strict > keeps the first maximum, matching the release's `max`.
            if count > best_count:
                best_type, best_count = mem_type, count
        return best_type or "unknown"

    def score(self, content: str, declared_type: str | None = None) -> float:
        """T in [0, 1] — the prior for the classified type."""
        return self.priors.get(self.classify(content, declared_type), self.priors["unknown"])


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

UTILITY_PROMPT = """Given the following conversation context:

{context}

Candidate memory to evaluate:
"{content}"

Rate the likelihood that this memory will be useful for continuing the \
conversation or completing future tasks, from 0.0 (completely useless) to 1.0 \
(extremely useful).

Respond with JSON: {{"score": <number between 0.0 and 1.0>}}"""

UTILITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"score": {"type": "number"}},
    "required": ["score"],
}

# Paper's cross-validated operating point (features/README + optimize_weights_cv
# candidate class 3), ordered (U, C, N, R, T). See the module docstring on why
# this pair does not transfer to the debugged features.
PAPER_WEIGHTS = (0.1, 0.1, 0.1, 0.1, 0.6)
PAPER_THRESHOLD = 0.55


@dataclass(frozen=True)
class AdmissionFeatures:
    """The five feature values for one candidate, ordered as in ``S(m)``.

    ``utility`` is ``None`` when the LLM call was provably unable to change the
    verdict and was skipped (see ``AdmissionGate.decide``)."""

    utility: float | None
    confidence: float
    novelty: float
    recency: float
    type_prior: float

    def to_tuple(self, utility_default: float = 0.0) -> tuple[float, ...]:
        """(U, C, N, R, T), substituting ``utility_default`` for an unevaluated U."""
        u = self.utility if self.utility is not None else utility_default
        return (u, self.confidence, self.novelty, self.recency, self.type_prior)


@dataclass(frozen=True)
class AdmissionDecision:
    """One admission verdict, carrying everything needed to audit it later.

    ``score`` is the exact ``S(m)`` when ``utility_evaluated`` is True. When the
    utility call was skipped it is instead the bound that *proved* the verdict —
    the U=0 lower bound for an admit, the U=1 upper bound for a reject — so the
    comparison ``score >= threshold`` still reads correctly either way."""

    admit: bool
    score: float
    threshold: float
    features: AdmissionFeatures
    content_type: str
    utility_evaluated: bool

    def as_dict(self) -> dict[str, Any]:
        """Flat JSON-safe row, for per-decision artifact capture."""
        return {
            "admit": self.admit,
            "score": self.score,
            "threshold": self.threshold,
            "content_type": self.content_type,
            "utility_evaluated": self.utility_evaluated,
            "utility": self.features.utility,
            "confidence": self.features.confidence,
            "novelty": self.features.novelty,
            "recency": self.features.recency,
            "type_prior": self.features.type_prior,
        }


@dataclass
class AdmissionStats:
    """Running gate counters — the observability the write-path-critics survey
    wanted: admit rate and per-feature distributions without a re-spend."""

    seen: int = 0
    admitted: int = 0
    utility_calls: int = 0
    utility_skipped: int = 0
    feature_sums: dict[str, float] = field(
        default_factory=lambda: {
            "utility": 0.0,
            "confidence": 0.0,
            "novelty": 0.0,
            "recency": 0.0,
            "type_prior": 0.0,
        }
    )
    utility_observations: int = 0
    type_counts: dict[str, int] = field(default_factory=dict)

    def record(self, decision: AdmissionDecision) -> None:
        self.seen += 1
        if decision.admit:
            self.admitted += 1
        if decision.utility_evaluated:
            self.utility_calls += 1
        else:
            self.utility_skipped += 1
        f = decision.features
        self.feature_sums["confidence"] += f.confidence
        self.feature_sums["novelty"] += f.novelty
        self.feature_sums["recency"] += f.recency
        self.feature_sums["type_prior"] += f.type_prior
        if f.utility is not None:
            self.feature_sums["utility"] += f.utility
            self.utility_observations += 1
        self.type_counts[decision.content_type] = self.type_counts.get(decision.content_type, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        """Counters plus derived rates/means. Feature means divide by the number
        of candidates that actually produced the feature — U's denominator is
        ``utility_observations``, not ``seen``, so short-circuited candidates do
        not drag the mean toward zero."""
        means = {
            name: (total / self.seen if self.seen else 0.0)
            for name, total in self.feature_sums.items()
            if name != "utility"
        }
        means["utility"] = (
            self.feature_sums["utility"] / self.utility_observations
            if self.utility_observations
            else 0.0
        )
        return {
            "seen": self.seen,
            "admitted": self.admitted,
            "rejected": self.seen - self.admitted,
            "admit_rate": self.admitted / self.seen if self.seen else 0.0,
            "utility_calls": self.utility_calls,
            "utility_skipped": self.utility_skipped,
            "feature_means": means,
            "utility_observations": self.utility_observations,
            "type_counts": dict(self.type_counts),
        }


class AdmissionGate:
    """Weighted-feature admission gate; stateful across a conversation.

    The gate keeps its own rolling transcript because the Confidence feature
    needs the turns preceding the candidate and no store API exposes "recent
    episodes" (``DocStore`` fetches episodes by id). One gate instance therefore
    belongs to one conversation — the experiment configs already build
    organizers through 0-arg factories per run for exactly this kind of
    per-conversation state.

    Cost arithmetic, which decides whether ``use_utility`` is ever worth it:
    A-Mem spends 2 LLM calls per admitted turn and the gate's U costs 1 more, so
    with U enabled and a rejection rate ``r`` the call count goes from ``2`` to
    ``1 + 2(1-r)`` per turn — a saving only when ``r > 0.5``. ``use_utility`` is
    off by default, which makes the gate entirely LLM-free and its saving
    exactly ``2r``. When it is on, ``utility_short_circuit`` skips the call
    whenever the interval ``[base, base + w_U]`` falls entirely on one side of
    the threshold, since U cannot then change the verdict. That test is exact,
    never an approximation — but it is not a blanket skip: under the paper's
    weights a turn whose LLM-free score lands in ``[theta - w_U, theta)`` really
    does need U, and the ``unknown`` type prior (0.5) with low confidence sits
    precisely in that band, so mid-scoring turns still pay for the call."""

    def __init__(
        self,
        weights: tuple[float, float, float, float, float] = PAPER_WEIGHTS,
        threshold: float = PAPER_THRESHOLD,
        decay_rate: float = 0.01,
        use_utility: bool = False,
        utility_short_circuit: bool = True,
        utility_role: str = "admit",
        utility_context_turns: int = 5,
        type_matching: str = "word",
        novelty_types: tuple[str, ...] = ("notes",),
        history_window: int | None = None,
    ) -> None:
        """``weights`` is (w_U, w_C, w_N, w_R, w_T) and must be non-negative and
        sum to 1, as the paper constrains and the release asserts.

        ``decay_rate`` is lambda per hour (paper: 0.01, a ~69h half-life).
        ``novelty_types`` are the memory types N compares against — the host
        organizer's ``produces``. ``history_window`` caps the transcript the
        Confidence feature scans; ``None`` keeps the release's behaviour of
        scanning every prior turn."""
        if len(weights) != 5:
            raise ValueError(f"weights must have 5 entries (U, C, N, R, T), got {len(weights)}")
        if any(w < 0 for w in weights):
            raise ValueError(f"weights must be non-negative, got {weights}")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1, got {weights} summing to {sum(weights)}")
        self.weights = tuple(float(w) for w in weights)
        self.threshold = float(threshold)
        self.decay_rate = float(decay_rate)
        self.use_utility = use_utility
        self.utility_short_circuit = utility_short_circuit
        self.utility_role = utility_role
        self.utility_context_turns = utility_context_turns
        self.novelty_types = novelty_types
        self.history_window = history_window
        self.type_prior = TypePriorClassifier(matching=type_matching)
        self.stats = AdmissionStats()
        self._history: list[str] = []
        self._warned_missing_role = False

    # -- features -----------------------------------------------------------

    def confidence(self, content: str) -> float:
        """C = max ROUGE-L over prior turns sharing at least one whitespace token.

        The release pre-filters spans by raw word overlap before scoring, so a
        candidate with no lexical anchor in the transcript gets 0.0 rather than a
        weak best-of-nothing; that filter is kept.

        Construct-validity caveat worth carrying into any write-up: the paper
        frames C as hallucination defense ("well-supported by conversation
        evidence"), but the candidate the release scores *is* a verbatim
        transcript turn (``candidate_to_memory`` sets ``content`` to
        ``dialogue_turn.text``), and so is ours at gate time. A verbatim turn
        cannot be unsupported by the transcript, so what C actually measures
        here is lexical echo of earlier turns — closer to redundancy than to
        confidence."""
        history = self._history
        if self.history_window is not None:
            history = history[-self.history_window :]
        if not history:
            return 0.0
        candidate_words = set(str(content).lower().split())
        best = 0.0
        for span in history:
            if not candidate_words & set(span.lower().split()):
                continue
            best = max(best, rouge_l_fmeasure(span, content))
        return best

    def novelty(self, content: str, ctx: OrganizerContext) -> float:
        """N = 1 - max cosine similarity against already-stored memories.

        Deviation from the release, which loops SBERT over a Python list of
        every existing memory: we ask the run's own vector index for its top
        hit per type in ``novelty_types``. ``VectorStore.search`` is contracted
        to return cosine similarity, highest first, so the top hit *is* the
        ``max`` the formula asks for — same value, one ANN query instead of a
        full scan, and the embedder stays the one retrieval uses. An empty index
        yields 1.0, as in the release."""
        embedding = ctx.embedder.embed([content])[0]
        best_similarity = 0.0
        found = False
        for memory_type in self.novelty_types:
            hits = ctx.vector_store.search(
                embedding, k=1, memory_type=memory_type, namespace=ctx.namespace
            )
            if hits:
                found = True
                best_similarity = max(best_similarity, hits[0][1])
        if not found:
            return 1.0
        return max(0.0, min(1.0, 1.0 - best_similarity))

    def recency(self, episode: Episode, now: datetime | None = None) -> float:
        """R = exp(-lambda * age_hours), age measured from ``now`` (default: the
        candidate's own timestamp).

        **R carries no discriminative signal in a streaming gate** and this is
        structural, not a wiring bug we can fix. A turn is admitted the moment it
        arrives, so its age is ~0 and R is ~1.0 for every candidate. The release
        gets *variation* only because it scores a whole replayed corpus against
        one ``time.time()``, turning R into "which session was this from" — and
        then destroys even that, because lambda=0.01/h against LoCoMo's
        month-scale spans underflows R to ~1e-124 for every candidate alike.
        Both settings leave R constant; only its constant differs. It is
        implemented faithfully, exposed through ``decay_rate``, and reported in
        the stats so the deadness is visible rather than assumed."""
        reference = now or episode.timestamp
        age_hours = (reference - episode.timestamp).total_seconds() / 3600.0
        return max(0.0, min(1.0, math.exp(-self.decay_rate * max(0.0, age_hours))))

    def utility(self, content: str, ctx: OrganizerContext) -> float | None:
        """U from one LLM call at temperature 0, or ``None`` if unavailable.

        Two deviations from the release, both to route through machinery this
        project already trusts: the response is requested as JSON through
        ``StructuredCaller`` (which owns guided-json, one retry, and explicit
        drops) rather than as a bare float parsed by regex; and the call uses its
        own ``admit`` role so the paper's temperature-0 requirement is a
        ``RoleConfig`` rather than a per-call override ``call()`` does not
        accept. A missing ``admit`` role degrades explicitly — warn once, return
        ``None`` — instead of silently borrowing A-Mem's 0.7-temperature
        ``extract`` role."""
        if ctx.llm is None:
            return None
        client = getattr(ctx.llm, "client", None)
        if client is not None and not client.has_role(self.utility_role):
            if not self._warned_missing_role:
                logger.warning(
                    "admission: no LLM role %r configured — utility feature disabled "
                    "(explicit degradation)",
                    self.utility_role,
                )
                self._warned_missing_role = True
            return None
        history = self._history[-self.utility_context_turns :] if self.utility_context_turns else []
        context = "\n".join(f"Turn {i}: {turn}" for i, turn in enumerate(history))
        verdict = ctx.llm.call(
            self.utility_role,
            UTILITY_PROMPT.format(context=context, content=content),
            UTILITY_SCHEMA,
            required_keys=("score",),
            phase="admit",
        )
        if not verdict:
            return None
        try:
            return max(0.0, min(1.0, float(verdict["score"])))
        except (TypeError, ValueError):
            logger.warning("admission: unparseable utility score %r", verdict.get("score"))
            return None

    # -- decision -----------------------------------------------------------

    def decide(
        self, episode: Episode, ctx: OrganizerContext, now: datetime | None = None
    ) -> AdmissionDecision:
        """Score ``episode`` and decide admission, then record it in the
        transcript and stats.

        The candidate joins the transcript whether or not it was admitted:
        Confidence asks what the *conversation* already said, which does not
        depend on what the memory kept."""
        content = episode.content
        declared = episode.meta.get("content_type")
        content_type = self.type_prior.classify(content, declared)
        w_u, w_c, w_n, w_r, w_t = self.weights

        confidence = self.confidence(content)
        novelty = self.novelty(content, ctx)
        recency = self.recency(episode, now)
        type_prior = self.type_prior.priors.get(content_type, self.type_prior.priors["unknown"])
        base = w_c * confidence + w_n * novelty + w_r * recency + w_t * type_prior

        # U lands in [base, base + w_u]; evaluate it only when that interval
        # straddles the threshold, i.e. when it can still change the answer.
        utility: float | None = None
        evaluated = False
        decided_by_bounds = base >= self.threshold or base + w_u < self.threshold
        if self.use_utility and not (self.utility_short_circuit and decided_by_bounds):
            utility = self.utility(content, ctx)
            evaluated = utility is not None

        if evaluated:
            score = base + w_u * float(utility)
            admit = score >= self.threshold
        else:
            # No U value: admit on the lower bound, else report the upper bound
            # that ruled it out. Identical to treating an unavailable U as 0.
            admit = base >= self.threshold
            score = base if admit else base + w_u

        decision = AdmissionDecision(
            admit=admit,
            score=score,
            threshold=self.threshold,
            features=AdmissionFeatures(
                utility=utility,
                confidence=confidence,
                novelty=novelty,
                recency=recency,
                type_prior=type_prior,
            ),
            content_type=content_type,
            utility_evaluated=evaluated,
        )
        self.stats.record(decision)
        self._history.append(content)
        return decision
