"""Does upstream's own context cap bind on the cleaned _s release?

run_generation.py:343 computes `max_retrieval_length = model_max_length - gen_length - 1000`,
= 128000 - 800 - 1000 = 126,200 for gpt-4o with CoN, and truncates the assembled
history string with tiktoken o200k_base (:266-271) -- keeping the HEAD, i.e.
dropping the newest sessions, since chunks were date-sorted first.

We rebuild the identical history string with our port's `render_sessions`
(orig-session + history_format=json) and count with the same tokenizer.
Sampled: the 50 largest instances by chars plus 50 evenly spaced others.
"""

import json
import os

import tiktoken

from agmem.bench.longmemeval import load_longmemeval, render_sessions

PATH = os.path.expanduser("~/.agmem/datasets/longmemeval_s_cleaned.json")
CAP_CON = 128_000 - 800 - 1_000  # 126,200
CAP_DIRECT = 128_000 - 500 - 1_000  # 126,500

enc = tiktoken.get_encoding("o200k_base")
data = load_longmemeval(PATH)

sized = sorted(
    range(len(data)),
    key=lambda i: sum(len(t.get("content", "")) for s in data[i]["haystack_sessions"] for t in s),
    reverse=True,
)
sample = sorted(set(sized[:50]) | set(sized[::10]))
print(f"sampling {len(sample)} of {len(data)} instances (50 largest + every 10th)")

rows = []
for i in sample:
    hist = render_sessions(data[i])
    n = len(enc.encode(hist, allowed_special={"<|endoftext|>"}))
    rows.append((data[i]["question_id"], len(hist), n))

rows.sort(key=lambda r: -r[2])
print(f"\n{'question_id':<28} {'chars':>10} {'tokens':>9} {'chars/tok':>10}")
for qid, c, n in rows[:10]:
    print(f"{qid:<28} {c:>10,} {n:>9,} {c / n:>10.3f}")
print("  ...")
for qid, c, n in rows[-3:]:
    print(f"{qid:<28} {c:>10,} {n:>9,} {c / n:>10.3f}")

toks = [n for _, _, n in rows]
ratio = sum(c for _, c, _ in rows) / sum(toks)
over_con = sum(1 for n in toks if n > CAP_CON)
print(f"\nsampled max {max(toks):,} tokens, median {sorted(toks)[len(toks) // 2]:,}")
print(f"measured chars/token on this corpus: {ratio:.3f}")
print(f"over the CoN cap ({CAP_CON:,}): {over_con} / {len(toks)} sampled")
print(
    f"over the direct cap ({CAP_DIRECT:,}): {sum(1 for n in toks if n > CAP_DIRECT)} / {len(toks)}"
)

# extrapolate to all 500 with the measured ratio
allc = [sum(len(t.get("content", "")) for s in d["haystack_sessions"] for t in s) for d in data]
# render adds per-session framing; scale by the sampled render/content overhead
overhead = sum(c for _, c, _ in rows) / sum(allc[i] for i in sample)
est = [c * overhead / ratio for c in allc]
print(
    f"\nrender overhead over raw content: x{overhead:.4f}; "
    f"estimated full-corpus tokens {sum(est):,.0f}"
)
print(f"estimated instances over the CoN cap: {sum(1 for e in est if e > CAP_CON)} / 500")
print(json.dumps({"max_tokens_sampled": max(toks), "chars_per_token": round(ratio, 3)}))
