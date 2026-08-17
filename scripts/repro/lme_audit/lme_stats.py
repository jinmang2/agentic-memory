"""$0 audit of the cleaned LongMemEval_S release, using our own port's loaders.

Three questions:
  1. do the port's invariants still hold on this file (500 / 6 types / 30 abs)?
  2. scale: sessions, turns, chars -> tokens per instance
  3. the head-vs-tail truncation question: upstream's `orig-session` keeps the
     LAST topk sessions (run_generation.py:172); our `_capped` keeps the FIRST.
     At which cap does each direction start dropping evidence sessions?
"""

import os
import statistics as st
from collections import Counter

from agmem.bench.longmemeval import (
    QUESTION_TYPES,
    evidence_session_ids,
    is_abstention,
    load_longmemeval,
)

PATH = os.path.expanduser("~/.agmem/datasets/longmemeval_s_cleaned.json")
CHARS_PER_TOKEN = 4.045  # the corpus-specific ratio recorded in docs/research/ace-longmemeval.md §D

data = load_longmemeval(PATH)
print(f"instances: {len(data)}")

types = Counter(str(x["question_type"]) for x in data)
print("types:", dict(types))
print("unknown types:", {t for t in types} - set(QUESTION_TYPES))
print("abstention (substring):", sum(is_abstention(str(x["question_id"])) for x in data))
print("abstention (endswith) :", sum(str(x["question_id"]).endswith("_abs") for x in data))

n_sess, n_turns, n_chars, n_ev = [], [], [], []
# evidence position, normalised: 0.0 = oldest session, 1.0 = newest
pos = []
head_drop = Counter()  # cap -> n instances losing >=1 evidence session under head truncation
tail_drop = Counter()
CAPS = (5, 10, 20, 30, 40, 50, 60)

for inst in data:
    sess = inst.get("haystack_sessions", [])
    ids = [str(s) for s in inst.get("haystack_session_ids", [])]
    ev = evidence_session_ids(inst)
    n_sess.append(len(sess))
    n_turns.append(sum(len(s) for s in sess))
    n_chars.append(sum(len(t.get("content", "")) for s in sess for t in s))
    n_ev.append(len(ev))
    idx = [i for i, sid in enumerate(ids) if sid in ev]
    if ev and len(sess) > 1:
        pos += [i / (len(sess) - 1) for i in idx]
    for cap in CAPS:
        if not ev:
            continue
        head = set(ids[:cap])
        tail = set(ids[-cap:])
        if not ev <= head:
            head_drop[cap] += 1
        if not ev <= tail:
            tail_drop[cap] += 1


def line(name, xs):
    print(
        f"{name:22s} min {min(xs):>9,.0f}  p50 {st.median(xs):>9,.0f}  "
        f"mean {st.mean(xs):>9,.0f}  max {max(xs):>9,.0f}  sum {sum(xs):>13,.0f}"
    )


print()
line("sessions/instance", n_sess)
line("turns/instance", n_turns)
line("chars/instance", n_chars)
line("est tokens/instance", [c / CHARS_PER_TOKEN for c in n_chars])
line("evidence sessions", n_ev)
print(
    f"\ninstances over 50 sessions (run_generation.sh default topk): "
    f"{sum(1 for n in n_sess if n > 50)} / {len(n_sess)}"
)

print("\nevidence session position (0=oldest, 1=newest):")
print(
    f"  n={len(pos)}  p10 {st.quantiles(pos, n=10)[0]:.2f}  p50 {st.median(pos):.2f}  "
    f"p90 {st.quantiles(pos, n=10)[8]:.2f}  mean {st.mean(pos):.2f}"
)

n_with_ev = sum(1 for x in data if evidence_session_ids(x))
print(f"\nevidence loss by truncation direction ({n_with_ev} instances have evidence):")
print(f"{'cap':>5} {'head[:cap] loses':>18} {'tail[-cap:] loses':>19}")
for cap in CAPS:
    print(
        f"{cap:>5} {head_drop[cap]:>13} ({100 * head_drop[cap] / n_with_ev:4.1f}%)"
        f" {tail_drop[cap]:>13} ({100 * tail_drop[cap] / n_with_ev:4.1f}%)"
    )
