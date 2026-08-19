"""Hosted embedding models over an OpenAI-compatible `/embeddings` endpoint.

The `full` profile has named `APIEmbedder` since before Phase 2 (`config.py`)
while no such class existed, so that profile silently resolved to
sentence-transformers instead — recorded, but not fixed, at
`docs/01-capability-system.md`. This is the class.

What it is for right now: the embedder diagnostic. Our Nemori reproduction lands
9.6 pp under the paper's LoCoMo headline, and the embedder is the named confound
— upstream ran `gemini-embedding-001`, we ran `all-MiniLM-L6-v2` for
cross-methodology comparability with the A-Mem campaign. Deciding how much of
that gap is the embedder needs a run on a hosted model, which needs this.

Cost is first-class here for the same reason it is in `LLMClient`: this is the
first embedder whose calls cost money, so `calls`/`tokens`/`errors` accumulate on
the instance and a benchmark run folds them into its budget under an `embed`
role, priced at the embedding model's own registry rates. An embedder that spent
silently would put an unaccounted line in every quote that used it.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from agmem.capabilities.requires import Requires
from agmem.embed.base import EmbedKind

logger = logging.getLogger(__name__)

# Native output width per model. Used both as the default `dim` and to decide
# whether the request needs a `dimensions` parameter at all.
NATIVE_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Input ceiling per model, in tokens. All three OpenAI embedding models stop at
# 8192 and answer a longer input with a 400, not a truncation — which is a
# request-shaped failure, so the transport retry above cannot help and simply
# spends the attempt twice more before raising.
MAX_INPUT_TOKENS = {
    "text-embedding-3-small": 8192,
    "text-embedding-3-large": 8192,
    "text-embedding-ada-002": 8192,
}
# Conservative chars-per-token floor used to convert that ceiling into a
# character budget without a tokenizer dependency. The LongMemEval corpus
# measures 4.610 with o200k_base, so 3.0 leaves ~35% of headroom for text that
# tokenizes worse than prose. It is a FLOOR on purpose: over-truncating one
# oversized turn costs a little of its tail, while under-truncating costs the
# whole request.
CHARS_PER_TOKEN_FLOOR = 3.0
# The floor above is an assumption about the text, and LongMemEval `_m` breaks it:
# 2 of the first 50 instances died on a 400 because a turn that fits 24,576 chars
# did not fit 8192 tokens. Lowering the floor globally is the wrong fix — it would
# re-truncate ordinary prose and silently change what every other run indexes — so
# the ceiling is discovered per request instead, by halving only the input the API
# actually rejected. `_s`'s 5 truncations are unaffected: they succeeded on the
# first attempt and still do.
MIN_INPUT_CHARS = 1024


def _is_input_too_long(exc: Exception) -> bool:
    """Whether `exc` is the ceiling 400 rather than a transport blip.

    Matched on the message because the SDK raises one `BadRequestError` for every
    request-shaped problem; an over-broad match would turn a genuine 400 into an
    endless halving loop, so both halves of OpenAI\'s wording are required.
    """
    text = str(exc).lower()
    return "maximum input length" in text or ("8192" in text and "token" in text)


class APIEmbedder:
    """`Embedder` backed by a hosted `/embeddings` endpoint.

    Placed LAST in `EMBEDDER_CANDIDATES` on purpose: every other embedder in
    this repo is free, and the resolver picks the first satisfiable candidate.
    A paid slot must be chosen deliberately — by the `full` profile or an
    explicit `--embedder` — never inherited because `openai` happens to be
    installed.
    """

    requires = Requires(python_pkgs=("openai",))
    NATIVE_DIMS = NATIVE_DIMS

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        endpoint: str = "https://api.openai.com/v1",
        dim: int | None = None,
        client: Any | None = None,
        transport_retries: int = 2,
    ) -> None:
        """`dim` below the model's native width uses the API's Matryoshka
        truncation; `None` keeps the native width. `client` is an injection
        point for tests — production passes `api_key`/`endpoint` and lets the
        constructor build the real one.

        ``transport_retries`` is the budget for CONNECTION failures — timeouts,
        DNS blips, resets — and mirrors `StructuredCaller.transport_retries`
        deliberately, because it is the same failure with the same fix: no
        response came back, so the request is simply sent again. The LLM path
        has had this since Track 2; the embedder did not, and on 2026-08-17 a
        single `openai.APIConnectionError` raised from a curator's dedup
        embedding killed a paid run with 291 of 441 samples still to answer.
        Nothing was lost — the runner resumes — but the failure was recoverable
        two layers down and was not recovered.

        Set it to 0 to replay an arm measured before this parameter existed."""
        if model_name not in NATIVE_DIMS:
            raise KeyError(
                f"{model_name!r} is not a known embedding model "
                f"(known: {sorted(NATIVE_DIMS)}). Add it to NATIVE_DIMS with its "
                f"native width, and to bench/registry.py with its price."
            )
        self.name = model_name
        self.native_dim = NATIVE_DIMS[model_name]
        self.dim = dim or self.native_dim
        if self.dim > self.native_dim:
            raise ValueError(
                f"{model_name} emits {self.native_dim} dims; {self.dim} was requested "
                f"(the API can truncate, never widen)"
            )
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI  # gated by `requires`

            self._client = OpenAI(base_url=endpoint, api_key=api_key)

        # Cost accounting, mirroring BudgetTracker's fields. A failed request
        # still counts a call (it cost an attempt and latency), same contract as
        # LLMClient.chat.
        self.calls = 0
        self.tokens = 0
        self.errors = 0
        self.latency_ms_total = 0.0
        self.transport_retries = transport_retries
        # Blips that a retry RECOVERED. Not failures — no work was lost — but a
        # run over a degrading link must be able to say so instead of looking
        # pristine, which is the same reason `StructuredCaller` counts them.
        self.transport_recoveries = 0
        # Inputs whose tail was dropped to fit the model's ceiling. Counted
        # because it is a silent change to what got indexed, and a run that did
        # it must be able to say how often rather than look clean.
        self.truncations = 0
        self.max_input_chars = int(MAX_INPUT_TOKENS[model_name] * CHARS_PER_TOKEN_FLOOR)

    def _cap(self, texts: list[str], max_chars: int) -> list[str]:
        """`texts` with any oversized entry cut to `max_chars`, counting each cut."""
        out = []
        for text in texts:
            if len(text) > max_chars:
                self.truncations += 1
                logger.warning(
                    "embedding input of %d chars exceeds %s's ceiling — indexing the "
                    "first %d chars (%d truncated so far)",
                    len(text),
                    self.name,
                    max_chars,
                    self.truncations,
                )
                text = text[:max_chars]
            out.append(text)
        return out

    def embed(self, texts: list[str], kind: EmbedKind = "passage") -> list[list[float]]:
        """L2-normalized vectors, one per input, in input order.

        `kind` is accepted and ignored: OpenAI's embedding models are symmetric,
        unlike the e5/bge families `SentenceTransformerEmbedder` prefixes for.

        The whole list goes in ONE request. Callers here mostly pass single-item
        lists, but the ones that don't (a batch of facts) should not pay N round
        trips for it.
        """
        if not texts:
            return []
        # Truncate BEFORE sending. Left un-truncated, one oversized input fails
        # the whole request with a 400 that no retry can fix, and the caller
        # loses every other text in the batch with it — on LongMemEval `_s` that
        # was 5 turns out of 246,750 taking 5 of 500 instances down with them,
        # each one a >24K-character turn in an otherwise ordinary haystack.
        # The stored CONTENT is untouched; only the vector is computed from the
        # prefix, so a retrieved item still renders whole.
        budget = self.max_input_chars
        capped = self._cap(texts, budget)
        kwargs: dict[str, Any] = {"model": self.name, "input": capped}
        if self.dim != self.native_dim:
            # Older models reject the parameter outright, so send it only when
            # it would actually do something.
            kwargs["dimensions"] = self.dim

        self.calls += 1
        transport_left = self.transport_retries
        spent = 0
        while True:
            start = time.perf_counter()
            try:
                resp = self._client.embeddings.create(**kwargs)
                # Counted here and not at the moment of the retry, because a
                # recovery is a retry that WORKED. `StructuredCaller` increments
                # its own counter in the except branch, so a call that retried
                # twice and then dropped still reports two "recoveries" there —
                # this does not copy that.
                self.transport_recoveries += spent
                break
            except Exception as exc:
                # Every attempt cost latency and is counted, failed or not —
                # the same contract as `LLMClient.chat`, and the reason a
                # recovered blip still shows up in `errors`.
                self.errors += 1
                self.latency_ms_total += (time.perf_counter() - start) * 1000
                if _is_input_too_long(exc) and len(texts) > 1:
                    # A BATCH cannot say which of its inputs was too long, and
                    # halving the budget for all of them would truncate innocent
                    # texts that were about to succeed — silently changing their
                    # vectors. Split instead: the offender ends up alone and takes
                    # the shrink below, everything else is sent unchanged. At most
                    # log2(batch) extra requests, on a batch that was going to fail
                    # outright before this existed.
                    mid = len(texts) // 2
                    logger.warning(
                        "%s rejected a batch of %d as over 8192 tokens — splitting "
                        "rather than truncating the whole batch",
                        self.name,
                        len(texts),
                    )
                    return self.embed(texts[:mid], kind) + self.embed(texts[mid:], kind)
                if _is_input_too_long(exc) and budget > MIN_INPUT_CHARS:
                    # Not a blip: the same bytes will 400 forever, and the class
                    # docstring has always said so. Retrying them costs three
                    # attempts and then loses the WHOLE instance — on `_m` that
                    # was a 4% row-failure rate, enough for the completeness
                    # check to refuse the run a score. Halve and send again; this
                    # does not consume a transport retry, because nothing about
                    # the transport failed.
                    budget //= 2
                    logger.warning(
                        "%s rejected an input as over 8192 tokens — halving the "
                        "character budget to %d and retrying",
                        self.name,
                        budget,
                    )
                    kwargs["input"] = self._cap(texts, budget)
                    continue
                if transport_left <= 0:
                    # A link that is down, rather than blipping, must surface.
                    # Returning a zero vector here would poison the store with
                    # a neighbourless item and read as a retrieval defect
                    # months later.
                    raise
                logger.warning(
                    "embedding request failed (%s retries left): %s", transport_left, exc
                )
                transport_left -= 1
                spent += 1
                time.sleep(2.0 ** (self.transport_retries - transport_left))
        self.latency_ms_total += (time.perf_counter() - start) * 1000

        usage = getattr(resp, "usage", None)
        total = getattr(usage, "total_tokens", 0) or 0
        if usage is None:
            logger.warning(
                "%s returned no usage for embeddings — token counts recorded as 0, "
                "cost figures for this run will UNDERCOUNT.",
                self.name,
            )
        self.tokens += total

        # Order by the response's own `index`, never by arrival order: the API
        # documents `data` as index-tagged, and a silent mis-order would attach
        # each item's vector to a different item — invisible to any check of
        # counts or widths, and indistinguishable from "this embedder retrieves
        # worse" in the very measurement this class exists for.
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [_l2(list(d.embedding)) for d in ordered]

    def budget_row(self) -> dict[str, float]:
        """This embedder's spend in `BudgetTracker.summary()`'s row shape, so a
        run can fold it into `llm_budget` under an `embed` key and price it with
        the existing per-role split (`registry_cost_usd_split`). Embeddings have
        no output tokens, hence the hard zero.

        `transport_recoveries` rides along ONLY when a blip was actually
        survived. A run that never wobbled keeps the row shape every other role
        has, and a run that did says so in the artifact rather than in a log
        nobody reads — which is the difference between this counter and
        `StructuredCaller`'s, written since Track 2 and read by nothing."""
        row = {
            "calls": self.calls,
            "tokens_in": self.tokens,
            "tokens_out": 0,
            "latency_ms_avg": round(self.latency_ms_total / self.calls, 1) if self.calls else 0.0,
            "errors": self.errors,
        }
        if self.transport_recoveries:
            row["transport_recoveries"] = self.transport_recoveries
        if self.truncations:
            # Same rule as recoveries: absent when it never happened, present in
            # the artifact when it did, because "we indexed a prefix of N inputs"
            # is a property of the measurement and not of the log.
            row["input_truncations"] = self.truncations
        return row


def _l2(vec: list[float]) -> list[float]:
    """Unit-normalize, leaving an all-zero vector alone rather than emitting NaNs.

    The hosted API already returns unit vectors, so this is a no-op on the happy
    path. It is here because cosine comparability is the point of the diagnostic
    this class was written for: Nemori's 0.85 merge threshold is a cosine
    threshold, and two embedders whose outputs live on different scales produce
    filter rates that cannot be compared.
    """
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec
