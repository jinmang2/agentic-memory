"""Do abstention questions carry evidence sessions? The retrieval script assumes not."""

import os

from agmem.bench.longmemeval import evidence_session_ids, is_abstention, load_longmemeval

data = load_longmemeval(os.path.expanduser("~/.agmem/datasets/longmemeval_s_cleaned.json"))
abs_q = [d for d in data if is_abstention(str(d["question_id"]))]
with_ev = [d for d in abs_q if evidence_session_ids(d)]
print(
    f"abstention questions: {len(abs_q)}, of which with non-empty answer_session_ids: {len(with_ev)}"
)
for d in with_ev[:5]:
    n_user_ev = sum(
        1
        for sid, s in zip(d["haystack_session_ids"], d["haystack_sessions"])
        if str(sid) in evidence_session_ids(d)
        for t in s
        if t.get("role") == "user" and t.get("has_answer")
    )
    print(
        f"  {d['question_id']:<26} type={d['question_type']:<26} ev_sessions={len(evidence_session_ids(d))} user_ev_turns={n_user_ev}"
    )
# disambiguate the two 51s
no_user = [
    d
    for d in data
    if not any(
        t.get("has_answer") and t.get("role") == "user"
        for sid, s in zip(d["haystack_session_ids"], d["haystack_sessions"])
        if str(sid) in evidence_session_ids(d)
        for t in s
    )
]
print(f"\nno user-side evidence turn: {len(no_user)}")
print(f"  of which abstention: {sum(1 for d in no_user if is_abstention(str(d['question_id'])))}")
print(
    f"  of which NON-abstention: {sum(1 for d in no_user if not is_abstention(str(d['question_id'])))}"
)
asst_only = [
    d
    for d in no_user
    if any(
        t.get("has_answer") and t.get("role") != "user"
        for sid, s in zip(d["haystack_session_ids"], d["haystack_sessions"])
        if str(sid) in evidence_session_ids(d)
        for t in s
    )
]
print(f"  of which evidence exists ONLY in assistant turns: {len(asst_only)}")
print(
    f"  ...and among those, abstention: {sum(1 for d in asst_only if is_abstention(str(d['question_id'])))}"
)
