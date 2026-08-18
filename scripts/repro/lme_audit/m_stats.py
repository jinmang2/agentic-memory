"""$0 audit of `longmemeval_m_cleaned.json` — the variant we had never acquired.

`_m` is the regime `docs/research/longmemeval.md` §9.4 names as the biggest hole
in our claim: ~1.1M tokens per instance, so it does not fit any context window,
which is *why* write paths exist. Everything here is measured through
`iter_longmemeval`, because `_m` cannot be loaded any other way — `json.loads` on
this file projects past 24 GB against a 7.9 GB + 8 GB swap machine (§10.2).

Four questions:
  1. scale: sessions / turns / chars / tokens per instance, vs `_s`
  2. **boundaries**: are the streamed instances the real elements of the array?
     `_m` has no published per-instance counts to check against, but it carries
     THE SAME 500 questions as `_s` — same ids, questions, answers, types. If the
     streamed instances reproduce that set exactly, the element boundaries are
     right; a scanner that mis-split would not land on 500 matching ids.
  3. does upstream's 126,200-token cap bind here, and by how much? (`_s`: 0/500)
  4. where does the evidence sit once the haystack is 10x longer?

Run: `uv run python scripts/repro/lme_audit/m_stats.py`
     `LME_EXACT_TOKENS=1 uv run --with tiktoken python .../m_stats.py` for o200k
     counts instead of the corpus-measured chars-per-token estimate (much slower).
"""

import os
import resource
import statistics as st
import time
from collections import Counter

from agmem.bench.longmemeval import (
    CHARS_PER_TOKEN,
    QUESTION_TYPES,
    evidence_session_ids,
    is_abstention,
    iter_longmemeval,
    render_sessions,
    upstream_max_history_tokens,
)

M_PATH = os.path.expanduser("~/.agmem/datasets/longmemeval_m_cleaned.json")
S_PATH = os.path.expanduser("~/.agmem/datasets/longmemeval_s_cleaned.json")
UPSTREAM_CAP = upstream_max_history_tokens(128_000, "con")  # 126,200

_enc = None
if os.environ.get("LME_EXACT_TOKENS"):
    import tiktoken

    _enc = tiktoken.get_encoding("o200k_base")


def tokens_of(text: str) -> float:
    return len(_enc.encode(text)) if _enc is not None else len(text) / CHARS_PER_TOKEN


def question_key(inst):
    """The fields `_s` and `_m` must agree on: they are the same 500 questions
    sampled against different haystack sizes (sample_haystack_and_timestamp.py)."""
    return (
        str(inst["question_id"]),
        str(inst["question"]),
        str(inst["answer"]),
        str(inst["question_type"]),
    )


print(f"file: {M_PATH}")
print(f"size: {os.path.getsize(M_PATH):,} bytes")
print(f"token counting: {'o200k_base (exact)' if _enc else f'chars/{CHARS_PER_TOKEN} (estimate)'}")

# ---- `_s` question set, haystacks discarded as we go (a few hundred small dicts)
s_keys = {question_key(inst) for inst in iter_longmemeval(S_PATH)}
print(f"\n`_s` questions loaded for the boundary check: {len(s_keys)}")

t0 = time.time()
m_keys = set()
types: Counter = Counter()
n_sess, n_turns, n_chars, n_tokens, n_ev = [], [], [], [], []
pos = []  # evidence position, normalised: 0.0 = oldest session, 1.0 = newest
ragged = []  # instances whose parallel haystack arrays disagree in length
over_cap = 0
n_abs = 0

for inst in iter_longmemeval(M_PATH):
    m_keys.add(question_key(inst))
    types[str(inst["question_type"])] += 1
    n_abs += is_abstention(str(inst["question_id"]))

    sessions = inst.get("haystack_sessions", [])
    ids = inst.get("haystack_session_ids", [])
    dates = inst.get("haystack_dates", [])
    if not (len(sessions) == len(ids) == len(dates)):
        ragged.append((str(inst["question_id"]), len(sessions), len(ids), len(dates)))

    n_sess.append(len(sessions))
    n_turns.append(sum(len(s) for s in sessions))

    # The rendered prompt, not the raw chars: this is what actually has to fit.
    rendered = render_sessions(inst)
    n_chars.append(len(rendered))
    toks = tokens_of(rendered)
    n_tokens.append(toks)
    over_cap += toks > UPSTREAM_CAP

    ev = set(evidence_session_ids(inst))
    n_ev.append(len(ev))
    for i, sid in enumerate(ids):
        if sid in ev and len(ids) > 1:
            pos.append(i / (len(ids) - 1))
    del inst  # the whole point: nothing accumulates but these counters

dt = time.time() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def line(label, values, fmt=",.0f"):
    print(
        f"{label:<26} mean {format(st.mean(values), fmt)}  p50 {format(st.median(values), fmt)}"
        f"  min {format(min(values), fmt)}  max {format(max(values), fmt)}"
    )


print(f"\ninstances: {len(n_sess)}")
print("types:", dict(types))
print("unknown types:", {t for t in types} - set(QUESTION_TYPES))
print(f"abstention: {n_abs}")
line("sessions/instance", n_sess)
line("turns/instance", n_turns)
line("rendered chars/instance", n_chars)
line("rendered tokens/instance", n_tokens)
line("evidence sessions/inst", n_ev)
print(f"\ntotal sessions {sum(n_sess):,}  turns {sum(n_turns):,}  tokens {sum(n_tokens):,.0f}")

print("\n--- boundary check (question 2) ---")
print(f"`_m` distinct questions: {len(m_keys)}")
print(f"identical to `_s`'s set: {m_keys == s_keys}")
if m_keys != s_keys:
    print(f"  only in _m: {len(m_keys - s_keys)}   only in _s: {len(s_keys - m_keys)}")
print(f"ragged haystack arrays : {len(ragged)}" + (f" {ragged[:5]}" if ragged else ""))

print("\n--- upstream's cap (question 3) ---")
print(f"cap = {UPSTREAM_CAP:,} tokens (128k - 800 gen - 1000 reserve)")
print(f"instances over it: {over_cap}/{len(n_tokens)}   (`_s`: 0/500)")
if n_tokens:
    print(f"median instance is {st.median(n_tokens) / UPSTREAM_CAP:.1f}x the cap")

print("\n--- evidence position (question 4) ---")
if pos:
    qs = st.quantiles(pos, n=10)
    print(f"p10 {qs[0]:.2f}  p50 {st.median(pos):.2f}  p90 {qs[-1]:.2f}  mean {st.mean(pos):.2f}")

print("\n--- this machine ---")
print(f"streaming peak RSS: {peak:,.0f} MB   wall clock: {dt:.0f}s")
