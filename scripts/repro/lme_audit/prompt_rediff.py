"""Byte-diff our full-context prompt against a literal transcription of upstream's.

`upstream_prompt` below is `prepare_prompt` from run_generation.py @ 9e0b455,
reduced to the branches a full-history run actually takes (retriever_type=
orig-session, history_format=json, useronly=false, cot=true, no key expansion)
and with the tiktoken truncation kept. Nothing is "improved": the head-keeping
topk slice (:172), the in-place has_answer pop (:177-191), the date sort (:225),
the block layout (:252) and the template (:55) are copied as they stand.

Ours is `render_sessions` + `ANSWER_PROMPT_CON`, i.e. exactly what
`run_instance(full_context=True, reading_method="con")` sends.

Four configurations are compared, because they are the ways this can be run and
only two of them are faithful:
  A  topk >= n_sessions on _s        -> byte-identical
  B  topk = 50 (run_generation.sh's default) on _s
  C  oracle, rendered as it ships    -> NOT sorted, so 34/500 differ
  D  oracle, `sort_haystack_by_date` -> byte-identical, and this is what the
     driver must do (docs/research/longmemeval.md §8.5 guard 1)

C and D are the same 500 instances and the same code path; the only difference
is the one call the driver makes before rendering. That is the point of keeping
C: it prices the guard.
"""

import copy
import json
import os

import tiktoken

from agmem.bench.longmemeval import (
    ANSWER_PROMPT_CON,
    load_longmemeval,
    render_sessions,
    sort_haystack_by_date,
)

S = os.path.expanduser("~/.agmem/datasets/longmemeval_s_cleaned.json")
ORACLE = os.path.expanduser("~/.agmem/datasets/longmemeval_oracle.json")
enc = tiktoken.get_encoding("o200k_base")


def upstream_prompt(entry, topk_context, max_retrieval_length=126_200):
    """run_generation.py:46-282, orig-session / json / useronly=false / cot=true."""
    answer_prompt_template = "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):"
    question_date_string = entry["question_date"]
    question_string = entry["question"]

    retrieved_chunks = []
    for session_date, session_entry in zip(entry["haystack_dates"], entry["haystack_sessions"]):
        retrieved_chunks.append((session_date, session_entry))
    retrieved_chunks = retrieved_chunks[-topk_context:]  # :172 -- keep LATEST

    retrieved_chunks_cleaned = []  # :177-191
    for date, session_entry in retrieved_chunks:
        for turn_entry in session_entry:
            if type(turn_entry) is dict and "has_answer" in turn_entry:
                turn_entry.pop("has_answer")
        retrieved_chunks_cleaned.append((date, session_entry))
    retrieved_chunks = retrieved_chunks_cleaned

    retrieved_chunks.sort(key=lambda x: x[0])  # :225

    history_string = ""
    for i, (chunk_date, chunk_entry) in enumerate(retrieved_chunks):
        sess_string = "\n" + json.dumps(chunk_entry)  # :238
        history_string += f"\n### Session {i + 1}:\nSession Date: {chunk_date}\nSession Content:\n{sess_string}\n"  # :252

    tokens = enc.encode(history_string, allowed_special={"<|endoftext|>"})  # :266-271
    if len(tokens) > max_retrieval_length:
        history_string = enc.decode(tokens[:max_retrieval_length])
    return answer_prompt_template.format(history_string, question_date_string, question_string)


def our_prompt(entry, max_sessions=None):
    return ANSWER_PROMPT_CON.format(
        history=render_sessions(entry, max_sessions),
        question_date=str(entry.get("question_date", "")),
        question=entry["question"],
    )


def compare(name, path, n, topk, max_sessions, sort=False):
    data = load_longmemeval(path)[:n]
    same = 0
    diffs = []
    for inst in data:
        # upstream pops in place; give each side its own copy so neither is
        # scored against an instance the other already mutated.
        up = upstream_prompt(copy.deepcopy(inst), topk)
        ours = our_prompt(
            sort_haystack_by_date(copy.deepcopy(inst)) if sort else copy.deepcopy(inst),
            max_sessions,
        )
        if up == ours:
            same += 1
        else:
            diffs.append((str(inst["question_id"]), len(up), len(ours)))
    print(f"\n[{name}]  n={len(data)}  identical: {same}/{len(data)}")
    for qid, a, b in diffs[:5]:
        print(f"    differs {qid:<24} upstream {a:>9,} chars   ours {b:>9,} chars")
    if len(diffs) > 5:
        print(f"    ... {len(diffs) - 5} more")
    return same, len(data)


compare("A  _s, topk=1000 (README's recommendation)", S, 60, 1000, None)
compare("B  _s, topk=50 (run_generation.sh default)", S, 60, 50, 50)
compare("C  oracle, topk=1000 (dates not sorted)", ORACLE, 500, 1000, None)
compare("D  oracle, sort_haystack_by_date (the driver's path)", ORACLE, 500, 1000, None, sort=True)
