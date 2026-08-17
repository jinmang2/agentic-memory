"""Independently check two reported gold defects against the file we hold."""

import os

from agmem.bench.longmemeval import evidence_session_ids, load_longmemeval

data = {
    str(d["question_id"]): d
    for d in load_longmemeval(os.path.expanduser("~/.agmem/datasets/longmemeval_s_cleaned.json"))
}

print("═══ issue #41: 370a8ff4 (gold = '15 weeks'?) ═══")
d = data["370a8ff4"]
print("question:", d["question"])
print("gold answer:", repr(d["answer"]))
print("question_date:", d["question_date"], "| evidence sessions:", evidence_session_ids(d))
for sid, date, sess in zip(d["haystack_session_ids"], d["haystack_dates"], d["haystack_sessions"]):
    if str(sid) in evidence_session_ids(d):
        print(f"  {sid}  date={date}")
        for t in sess:
            if t.get("has_answer"):
                print(f"    [has_answer {t['role']}] {t['content'][:220]}")

print("\n═══ issue #22: 51a45a95 (has_answer 턴이 1개뿐?) ═══")
d = data["51a45a95"]
print("question:", d["question"], "| gold:", repr(d["answer"]))
n_ha = 0
for sid, sess in zip(d["haystack_session_ids"], d["haystack_sessions"]):
    if str(sid) in evidence_session_ids(d):
        for i, t in enumerate(sess):
            if t.get("has_answer"):
                n_ha += 1
                print(f"  [has_answer turn {i} / {t['role']}] {t['content'][:200]}")
        print(f"  (evidence session {sid}: {len(sess)} turns, {n_ha} labelled)")
