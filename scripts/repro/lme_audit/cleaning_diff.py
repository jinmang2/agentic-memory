"""What did the 2025-09-19 cleaning actually change in LongMemEval_S?

The author deprecated `xiaowu0162/longmemeval` the same day `longmemeval-cleaned`
appeared, saying the new release "removes noisy history sessions that interfere
with the answer correctness". Nobody has published what that means numerically,
and it decides whether any pre-2025-09 published number can be compared to a run
on today's data.

The oracle file is already settled: sha256-identical across the two releases.
This script settles `_s`, per instance:
  - are the question / answer / type / date / evidence-id fields untouched?
  - how many sessions were removed, from how many instances?
  - was any EVIDENCE session ever removed or edited?
"""

import os
import statistics as st
from collections import Counter

from agmem.bench.longmemeval import evidence_session_ids, load_longmemeval

# The withdrawn release is deliberately NOT kept under ~/.agmem/datasets: nothing in this repo
# should ever measure on it by accident. Point LME_S_ORIG at a scratch copy -- see this
# directory's README for the download line.
ORIG = os.environ.get("LME_S_ORIG", "/tmp/longmemeval_s_orig.json")
CLEAN = os.path.expanduser("~/.agmem/datasets/longmemeval_s_cleaned.json")


def sess_text(s):
    return "".join(f"{t.get('role')}\x00{t.get('content')}\x01" for t in s)


a = {str(x["question_id"]): x for x in load_longmemeval(ORIG)}
b = {str(x["question_id"]): x for x in load_longmemeval(CLEAN)}
print(f"instances: original {len(a)}, cleaned {len(b)}")
print(
    f"question_ids only in original: {len(set(a) - set(b))}, only in cleaned: {len(set(b) - set(a))}"
)

field_changed = Counter()
removed_counts, added_counts, edited_counts = [], [], []
n_changed = 0
evidence_removed = evidence_edited = 0

for qid in sorted(set(a) & set(b)):
    x, y = a[qid], b[qid]
    for f in ("question", "answer", "question_type", "question_date"):
        if str(x.get(f)) != str(y.get(f)):
            field_changed[f] += 1
    if evidence_session_ids(x) != evidence_session_ids(y):
        field_changed["answer_session_ids"] += 1

    ax = {str(sid): s for sid, s in zip(x["haystack_session_ids"], x["haystack_sessions"])}
    by = {str(sid): s for sid, s in zip(y["haystack_session_ids"], y["haystack_sessions"])}
    removed, added = set(ax) - set(by), set(by) - set(ax)
    edited = {sid for sid in set(ax) & set(by) if sess_text(ax[sid]) != sess_text(by[sid])}
    if removed or added or edited:
        n_changed += 1
    removed_counts.append(len(removed))
    added_counts.append(len(added))
    edited_counts.append(len(edited))

    ev = evidence_session_ids(x) | evidence_session_ids(y)
    if removed & ev:
        evidence_removed += 1
    if edited & ev:
        evidence_edited += 1

print(f"\nfields changed anywhere: {dict(field_changed) or 'NONE'}")
print(f"instances whose haystack changed: {n_changed}/{len(a)}")
print(
    f"sessions removed: total {sum(removed_counts)}, "
    f"instances affected {sum(1 for n in removed_counts if n)}, "
    f"max per instance {max(removed_counts)}, mean over affected "
    f"{st.mean([n for n in removed_counts if n]) if any(removed_counts) else 0:.2f}"
)
print(f"sessions added:  total {sum(added_counts)}  (affected {sum(1 for n in added_counts if n)})")
print(
    f"sessions edited in place: total {sum(edited_counts)} (affected {sum(1 for n in edited_counts if n)})"
)
print(f"\nEVIDENCE sessions removed in any instance: {evidence_removed}")
print(f"EVIDENCE sessions edited  in any instance: {evidence_edited}")

sa = [len(x["haystack_sessions"]) for x in a.values()]
sb = [len(y["haystack_sessions"]) for y in b.values()]
print(
    f"\nsessions/instance: original p50 {st.median(sa)} sum {sum(sa)} -> cleaned p50 {st.median(sb)} sum {sum(sb)}"
)
