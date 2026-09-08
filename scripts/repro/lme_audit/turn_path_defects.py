"""Two defects in upstream's `prepare_prompt` that only the turn-level paths reach.

Both are transcribed from `run_generation.py` @ 9e0b455 rather than imported, for the
same reason `prompt_rediff.py` transcribes: the module imports `transformers` at the top
level, which this environment does not carry. Every transcribed line keeps its upstream
line number in a comment, so the transcription can be diffed against the clone by eye.

D1 (`--useronly true` + `flat-turn` crashes)
    :135 appends `(date, cur_round_data)` where `cur_round_data` is a LIST of turns (:125-128),
    and :144 then filters with `x[-1]['role']`. Indexing a list with a string raises TypeError.
    Reachable from the shipped driver: run_generation.sh takes USERONLY as $6 (:14) and README
    documents it (README:180), though it recommends `false`.

D2 (`orig-turn` / `oracle-turn` leak the turn-level gold label into the prompt)
    :102 appends `(date, x)` where `x` is a single turn DICT, not a list of them. The cleanup
    loop at :181 then iterates that dict, so `turn_entry` is a KEY ('role', 'content',
    'has_answer'), `type(turn_entry) == dict` at :182 is always False, and :183 never runs.
    :238 dumps the dict whole, `has_answer` included. `orig-session` is unaffected (its chunk
    really is a list of turns) and so is `flat-turn` (:135 also passes a list).
    `orig-turn` is reachable as the `full-history-turn` alias (run_generation.sh:50);
    `oracle-turn` is not reachable from the shell at all (§2.8), only by calling the module.

Run: uv run python scripts/repro/lme_audit/turn_path_defects.py
Needs: ~/.agmem/datasets/longmemeval_oracle.json  (D2 is also shown on a real instance)
Spends: $0.
"""

import json
import os

from agmem.bench.longmemeval import load_longmemeval

ORACLE = os.path.expanduser("~/.agmem/datasets/longmemeval_oracle.json")


def collect_chunks(
    retriever_type, haystack_dates, haystack_sessions, ranked_items, corpusid2entry, useronly=False
):
    """run_generation.py:91-144, verbatim for the branches this file is about.

    `useronly` is threaded in because upstream handles it in TWO different places: the
    orig-* branches filter at collection time (:93-102), while flat-turn post-filters the
    assembled chunks (:143-144). That asymmetry is the whole of D1.
    """
    retrieved_chunks = []
    if retriever_type == "orig-session":  # :91
        for session_date, session_entry in zip(haystack_dates, haystack_sessions):
            retrieved_chunks.append((session_date, session_entry))  # :96
    elif retriever_type == "orig-turn":  # :97
        for session_date, session_entry in zip(haystack_dates, haystack_sessions):
            if useronly:
                # :100 -- orig-turn applies useronly HERE, at collection time, and is fine
                retrieved_chunks += [
                    (session_date, x) for x in session_entry if x["role"] == "user"
                ]
            else:
                retrieved_chunks += [(session_date, x) for x in session_entry]  # :102
    elif retriever_type == "flat-turn":  # :119
        for ret_result_entry in ranked_items:
            cid = ret_result_entry["corpus_id"].replace("noans_", "answer_")
            converted_corpus_id = "_".join(cid.split("_")[:-1])  # :121
            converted_turn_id = int(cid.split("_")[-1]) - 1  # :122
            try:
                cur_round_data = [corpusid2entry[converted_corpus_id][converted_turn_id]]  # :125
                converted_next_turn_id = converted_turn_id + 1
                if converted_next_turn_id < len(corpusid2entry[converted_corpus_id]):
                    cur_round_data.append(
                        corpusid2entry[converted_corpus_id][converted_next_turn_id]
                    )  # :128
            except Exception:  # noqa: BLE001, S112 — upstream's bare except/continue, kept as-is
                continue  # :131
            retrieved_chunks.append((ret_result_entry["date"], cur_round_data))  # :135
    else:
        raise NotImplementedError
    return retrieved_chunks


def apply_useronly(retrieved_chunks, useronly, merge_key_expansion_into_value="none"):
    """run_generation.py:143-144, the flat-turn branch's user-side filter."""
    if useronly and not merge_key_expansion_into_value == "replace":  # noqa: SIM201 — :143 verbatim
        retrieved_chunks = [x for x in retrieved_chunks if x[-1]["role"] == "user"]  # :144
    return retrieved_chunks


def clean_and_render(retrieved_chunks):
    """run_generation.py:176-191 (has_answer removal) then :234-238 (json format)."""
    cleaned = []
    for retrieved_item in retrieved_chunks:
        try:
            date, session_entry = retrieved_item  # :180
            for turn_entry in session_entry:  # :181
                if type(turn_entry) == dict and "has_answer" in turn_entry:  # :182
                    turn_entry.pop("has_answer")  # :183
            cleaned.append((date, session_entry))
        except Exception:  # noqa: BLE001 — upstream tells 2-tuples from 3-tuples by exception
            date, expansion_entry, session_entry = retrieved_item  # :186
            for turn_entry in session_entry:
                if type(turn_entry) == dict and "has_answer" in turn_entry:
                    turn_entry.pop("has_answer")
            cleaned.append((date, expansion_entry, session_entry))
    return "".join("\n" + json.dumps(c[-1]) for c in cleaned)  # :238


TOY_DATES = ["2023/01/01 (Sun) 00:00"]
TOY_SESSIONS = [
    [
        {"role": "user", "content": "my car GPS broke", "has_answer": True},
        {"role": "assistant", "content": "sorry to hear", "has_answer": False},
    ]
]
TOY_RANKED = [{"corpus_id": "noans_x_1", "date": TOY_DATES[0]}]
TOY_ENTRY_MAP = {"answer_x": TOY_SESSIONS[0]}


def d1_useronly_flat_turn():
    print("D1  --useronly true: flat-turn post-filters (:144), orig-turn filters inline (:100)")
    for retriever_type in ("flat-turn", "orig-turn"):
        chunks = collect_chunks(
            retriever_type,
            TOY_DATES,
            [[dict(t) for t in s] for s in TOY_SESSIONS],
            TOY_RANKED,
            {k: [dict(t) for t in v] for k, v in TOY_ENTRY_MAP.items()},
            useronly=True,
        )
        try:
            # only flat-turn reaches :143-144; the orig-* branches already filtered above
            kept = (
                apply_useronly(chunks, useronly=True) if retriever_type == "flat-turn" else chunks
            )
            print(f"    {retriever_type:<14} survives, {len(kept)} chunk(s) kept")
        except TypeError as e:
            print(f"    {retriever_type:<14} raises TypeError: {e}   <-- upstream :144")


def d2_gold_leak():
    print("\nD2  has_answer reaches the prompt")
    for retriever_type in ("orig-turn", "orig-session"):
        rendered = clean_and_render(
            collect_chunks(retriever_type, TOY_DATES, [list(s) for s in TOY_SESSIONS], [], {})
        )
        leaked = "has_answer" in rendered
        print(f"    {retriever_type:<14} has_answer in prompt: {leaked}")
        for line in rendered.strip().splitlines():
            print(f"        {line}")

    # the same thing on a real instance, so the finding does not rest on a toy input
    data = load_longmemeval(ORACLE)
    entry = next(
        d for d in data if any(t.get("has_answer") for s in d["haystack_sessions"] for t in s)
    )
    n_leaked = 0
    for retriever_type in ("orig-turn", "orig-session"):
        rendered = clean_and_render(
            collect_chunks(
                retriever_type,
                list(entry["haystack_dates"]),
                [[dict(t) for t in s] for s in entry["haystack_sessions"]],
                [],
                {},
            )
        )
        count = rendered.count('"has_answer"')
        n_leaked += count
        print(f"    {retriever_type:<14} {entry['question_id']}: {count} has_answer keys rendered")
    assert n_leaked > 0, "expected orig-turn to leak on real data"


if __name__ == "__main__":
    d1_useronly_flat_turn()
    d2_gold_leak()
