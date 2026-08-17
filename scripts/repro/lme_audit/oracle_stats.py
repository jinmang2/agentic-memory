"""Oracle variant: exact size, and whether its sessions really are unsorted."""

import os
import statistics as st

import tiktoken

from agmem.bench.longmemeval import evidence_session_ids, load_longmemeval, render_sessions

enc = tiktoken.get_encoding("o200k_base")
data = load_longmemeval(os.path.expanduser("~/.agmem/datasets/longmemeval_oracle.json"))
print("instances:", len(data))
sess = [len(d["haystack_sessions"]) for d in data]
turns = [sum(len(s) for s in d["haystack_sessions"]) for d in data]
print(
    f"sessions/inst  p50 {st.median(sess)} mean {st.mean(sess):.2f} max {max(sess)} sum {sum(sess)}"
)
print(
    f"turns/inst     p50 {st.median(turns)} mean {st.mean(turns):.1f} max {max(turns)} sum {sum(turns)}"
)
unsorted = sum(1 for d in data if d["haystack_dates"] != sorted(d["haystack_dates"]))
print(f"instances whose haystack_dates are NOT ascending: {unsorted}/{len(data)}")
# all instances = evidence only?
ev_only = sum(
    1 for d in data if set(map(str, d["haystack_session_ids"])) == evidence_session_ids(d)
)
print(f"instances whose haystack == exactly the evidence sessions: {ev_only}/{len(data)}")
toks = [len(enc.encode(render_sessions(d), allowed_special={"<|endoftext|>"})) for d in data]
print(
    f"render tokens/inst p50 {st.median(toks):,} mean {st.mean(toks):,.0f} max {max(toks):,} TOTAL {sum(toks):,}"
)
