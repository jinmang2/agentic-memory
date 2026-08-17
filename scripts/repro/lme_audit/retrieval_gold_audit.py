"""What does run_retrieval.py's indexing actually cover, and what does it silently drop?

Three facts to establish on the real cleaned _s release:

R2. The retrieval gold is the SUBSTRING "answer" in a session id
    (`correct_docs = [d for d in corpus_ids if "answer" in d]`, run_retrieval.py:272),
    not `answer_session_ids`. How are ids actually shaped, and do the two agree?

R1. Only USER turns are indexed (`process_item_flat_index`:206,214). An evidence
    session whose evidence lives only in ASSISTANT turns is relabelled
    answer_->noans_ (:209), i.e. treated as a non-answer document.

R6. main() then drops from every reported retrieval metric both the abstention
    instances (documented) and any instance with no user-side has_answer turn
    (:400, undocumented in the README). How many questions is that, and which
    types do they come from?
"""

import os
from collections import Counter

from agmem.bench.longmemeval import evidence_session_ids, is_abstention, load_longmemeval

data = load_longmemeval(os.path.expanduser("~/.agmem/datasets/longmemeval_s_cleaned.json"))

# --- R2: id shapes -------------------------------------------------------
prefixes = Counter()
for inst in data:
    for sid in inst["haystack_session_ids"]:
        prefixes[str(sid).split("_")[0]] += 1
print("session-id first token, over all 23,867 sessions:")
for p, n in prefixes.most_common(10):
    print(f"    {p:<12} {n:>7,}")

gold_by_substring_ok = gold_by_substring_bad = 0
for inst in data:
    ids = [str(s) for s in inst["haystack_session_ids"]]
    by_sub = {i for i in ids if "answer" in i}
    by_field = evidence_session_ids(inst)
    (gold_by_substring_ok := gold_by_substring_ok + 1) if by_sub == by_field else (
        gold_by_substring_bad := gold_by_substring_bad + 1
    )
print(f"\nR2: substring-gold == answer_session_ids on {gold_by_substring_ok}/{len(data)} instances")

# --- R1 / R6: user-side evidence ----------------------------------------
no_user_evidence, only_assistant, by_type = [], [], Counter()
ev_sessions_total = ev_sessions_user_only = 0
for inst in data:
    ev = evidence_session_ids(inst)
    user_ev_turns = 0
    assistant_ev_turns = 0
    for sid, sess in zip(inst["haystack_session_ids"], inst["haystack_sessions"]):
        if str(sid) not in ev:
            continue
        ev_sessions_total += 1
        u = sum(1 for t in sess if t.get("role") == "user" and t.get("has_answer"))
        a = sum(1 for t in sess if t.get("role") != "user" and t.get("has_answer"))
        user_ev_turns += u
        assistant_ev_turns += a
        if u:
            ev_sessions_user_only += 1
    if user_ev_turns == 0:
        no_user_evidence.append(str(inst["question_id"]))
        by_type[str(inst["question_type"])] += 1
        if assistant_ev_turns:
            only_assistant.append(str(inst["question_id"]))

n_abs = sum(1 for x in data if is_abstention(str(x["question_id"])))
print(
    f"\nevidence sessions: {ev_sessions_total}, of which reachable via a user turn: {ev_sessions_user_only}"
)
print(f"\nR1/R6: instances with NO user-side has_answer turn: {len(no_user_evidence)}")
print(f"       of those, evidence exists only in ASSISTANT turns: {len(only_assistant)}")
print(f"       by question_type: {dict(by_type)}")
print(
    f"\nretrieval metrics denominator: 500 - {n_abs} (abstention) - "
    f"{len([q for q in no_user_evidence if not is_abstention(q)])} (no user target) = "
    f"{500 - n_abs - len([q for q in no_user_evidence if not is_abstention(q)])}"
)
