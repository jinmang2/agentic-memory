"""Independent check of upstream issue #54 (2026-08-16): duplicate haystack_session_id."""

import os
from collections import Counter

from agmem.bench.longmemeval import load_longmemeval

data = load_longmemeval(os.path.expanduser("~/.agmem/datasets/longmemeval_s_cleaned.json"))
total = dups = 0
rows = []
for d in data:
    ids = [str(s) for s in d["haystack_session_ids"]]
    total += len(ids)
    c = Counter(ids)
    rep = {k: v for k, v in c.items() if v > 1}
    if rep:
        # is the content identical, and are the dates different, as the issue claims?
        for sid in rep:
            idx = [i for i, x in enumerate(ids) if x == sid]
            bodies = {repr(d["haystack_sessions"][i]) for i in idx}
            dates = {d["haystack_dates"][i] for i in idx}
            rows.append(
                (str(d["question_id"]), sid, len(idx), len(bodies) == 1, len(dates) == len(idx))
            )
        dups += 1
print(f"session occurrences total: {total:,}")
print(f"instances with a repeated session id: {dups}")
print(f"unique after collapsing pairs: {total - sum(r[2] - 1 for r in rows):,}")
print(
    f"\n{'question_id':<20} {'session_id':<24} {'n':>2} {'content identical':>17} {'dates differ':>13}"
)
for qid, sid, n, same_body, diff_dates in rows:
    print(f"{qid:<20} {sid:<24} {n:>2} {same_body!s:>17} {diff_dates!s:>13}")
