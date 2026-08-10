"""FiNER benchmark pipeline for ACE (arXiv:2510.04618, Apache-2.0).

FiNER is the benchmark behind ACE's *online* adaptation headline — the README
claims "-91.5% latency and -83.6% token cost vs Dynamic Cheatsheet" for
`--mode online --task_name finer`, and 69.1% -> 81.9% accuracy. Unlike AppWorld,
whose experiment code is absent from the public repo (ledger B-6), FiNER ships
with its data, so it is the one ACE claim this campaign can actually put a
number against.

The task: given a filing excerpt and a list of candidate US GAAP tags, emit the
tags for the highlighted values. Every row of the shipped test split
(`finer_test_subset_006_seed42.jsonl`, 441 rows) asks for exactly **4** tags, so
one sample is four tagging decisions, 1,764 in total.

Everything in the SCORING section is transcribed from the official clone, and
every file:line below refers to it:

    eval/finance/data_processor.py -- parsing, the two correctness rules
    eval/finance/run.py            -- data config, mode dispatch
    utils.py                       -- extract_answer, evaluate_test_set
    ace/ace.py                     -- the online window loop, the train step
    ace/prompts/generator.py       -- GENERATOR_PROMPT

**Five traps, all found by reading that code rather than the paper.** They are
why this module returns what it returns:

1. **"Accuracy" names two different quantities, and they are printed together.**
   `evaluate_test_set` builds one dict holding `accuracy` — the TAG-level micro
   rate `correct_tags / total_tags` from `_evaluate_finer_accuracy`
   (data_processor.py:183-200) — beside `correct` / `total`, which are
   SAMPLE-level exact match, incremented as `1 if is_correct else 0`
   (utils.py:253-254) where `is_correct` demands all four tags
   (`score == 1`, data_processor.py:152). The summary line then prints them as
   one fact: `f"Final Accuracy: {accuracy:.3f} ({correct}/{total})"`
   (utils.py:286) — the parenthetical does not equal the number in front of it.
   The saved online result carries the same pairing (`"accuracy":
   final_test_accuracy` with `"correct": correct_count_sample_based`,
   ace.py:1097-1098), so a reader who recomputes correct/total from the artifact
   gets a different number than the artifact's own headline. A third quantity,
   also labelled "Accuracy", is printed every 50 samples (utils.py:269).
   `aggregate` therefore returns BOTH, always, under names that say which is
   which — the same rule LongMemEval's port follows for the same reason.
2. **A sample that raises leaves the denominator instead of scoring zero.**
   `evaluate_single_test_sample` catches every exception and returns
   `(None, msg)` (utils.py:198-199); the caller prints it and `continue`s
   (utils.py:248-250), so the row never reaches `total`, `answers` or
   `targets`. Failures shrink the population rather than costing accuracy. We
   count an unanswerable sample as wrong and report `n_failed` separately, so
   the denominator is the split.
3. **`eval()` runs on model output inside the scorer.** `prediction =
   eval(prediction.replace(",", "").replace("$", ""))` and `ground_truth =
   eval(ground_truth)`, wrapped in a bare `except: pass`
   (data_processor.py:142-146). On FiNER the values are GAAP tag names, so
   `eval` raises `NameError` and the `pass` restores the string compare — the
   call is inert here and dangerous in general. (Its `.replace(",", "")` is also
   dead: the string was already split on commas.) We compare strings and
   numbers explicitly; see `_coerce`.
4. **Over-prediction is free.** When the model emits more tags than the gold
   has, `pred` is truncated to `len(label)` and the extras are never seen
   (data_processor.py:134-139); under-prediction is padded with `""` and each
   pad counts wrong. Guessing wide is costless, guessing short is not. We keep
   upstream's arithmetic — this is the metric ACE's number is stated in — but
   `tag_counts` also reports `n_over`, so the leniency is visible instead of
   silent.
5. **The training signal and the reported metric disagree.** The reflection loop
   fires on `answer_is_correct`, all-or-nothing over the four tags
   (ace.py:477), while the headline is tag-level. A sample with 3 of 4 tags
   right is a failure to learn from and a 0.75 to report.

**Call structure, which is the point of the cost comparison (rescope §3, track
5 deliverable 2).** Upstream's per-sample training cost is outcome-dependent:
1 generate, then either 1 reflect (correct, ace.py:546-560) or up to
`max_num_rounds`=3 rounds of reflect+regenerate (incorrect, ace.py:498-543),
plus a curator call every `curator_frequency`=1 steps — so 3 to 8 calls per
sample, and a *worse* model costs strictly more to adapt. Ours cuts at
`on_task_end`: 1 generate (here), 1 reflect, 1 curate, fixed at 3 regardless of
outcome, with no regeneration loop. That is a boundary difference, not a
saving — upstream's extra calls buy retries that can flip a sample to correct
within the training step, which our shape cannot do. Any latency or token
claim compared across the two must say so.

Deviations from upstream, deliberate:

- **D1 (structured output)**: upstream's `--json_mode` is a `store_true`
  defaulting OFF (run.py:77-78), so the shipped path parses free text with a
  regex ladder (`extract_answer`, utils.py:100-130). We call through
  `mem.structured`, which is the JSON-mode shape. `extract_answer` is
  transcribed anyway and applied to the structured reply's text so a degraded
  run scores by the same rule.
- **D2 (dedup)**: our curator dedups at 0.90 always; upstream's analyzer is
  opt-in, off in `ace.py`, off in this harness's flag, and off again by silent
  fallback when `sentence-transformers`/`faiss` are missing (ledger B-6). Ours
  is a third behavior, not upstream's switched on — it drops the incoming
  bullet where upstream LLM-merges groups.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agmem.memory import AgenticMemory

logger = logging.getLogger("agmem.bench.finer")

# The shipped splits (eval/finance/data/sample_config.json). Sizes are the line
# counts of the files at the pinned snapshot, asserted by the loader so a
# swapped file cannot masquerade as the split a published number used.
SPLITS = {
    "train": ("finer_train_batched_1000_samples.jsonl", 1000),
    "val": ("finer_val_batched_500_samples.jsonl", 500),
    "test": ("finer_test_subset_006_seed42.jsonl", 441),
}

# Every row of the test split asks for exactly this many tags. Trap 1's
# arithmetic in ace.py:971 (`window_accuracy * window_total`, a tag-level rate
# times a SAMPLE count) is only numerically defensible because this constant
# holds; `load_finer` checks it rather than trusting it.
TAGS_PER_SAMPLE = 4

# ace/prompts/generator.py, transcribed. Upstream formats it positionally with
# (playbook, reflection, question, context); named fields here so a caller
# cannot silently swap two of them.
GENERATOR_PROMPT = """You are an analysis expert tasked with answering questions using your knowledge, a curated playbook of strategies and insights and a reflection that goes over the diagnosis of all previous mistakes made while answering the question.

**Instructions:**
- Read the playbook carefully and apply relevant strategies, formulas, and insights
- Pay attention to common mistakes listed in the playbook and avoid them
- Show your reasoning step-by-step
- Be concise but thorough in your analysis
- If the playbook contains relevant code snippets or formulas, use them appropriately
- Double-check your calculations and logic before providing the final answer

Your output should be a json object, which contains the following fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- bullet_ids: each line in the playbook has a bullet_id. all bulletpoints in the playbook that's relevant, helpful for you to answer this question, you should include their bullet_id in this list
- final_answer: your concise final answer


**Playbook:**
{playbook}

**Reflection:**
{reflection}

**Question:**
{question}

**Context:**
{context}
"""

GENERATE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "bullet_ids": {"type": "array", "items": {"type": "string"}},
        "final_answer": {"type": "string"},
    },
    "required": ["final_answer"],
}

# utils.py:110 — the first fallback in `extract_answer`'s ladder.
_FINISH_RE = re.compile(r"Finish\[(.*?)\]")
_JSON_DQ_RE = re.compile(r'"final_answer"\s*:\s*"([^"]*)"')
_JSON_SQ_RE = re.compile(r"'final_answer'\s*:\s*'([^']*)'")
NO_ANSWER = "No final answer found"


# ---------------- loading ----------------


def load_finer(path: str | Path, split: str = "test") -> list[dict[str, Any]]:
    """Read one shipped split and check it is the split it claims to be.

    `path` is the upstream `eval/finance/data` directory. The row count is
    asserted against `SPLITS` and the per-sample tag count against
    `TAGS_PER_SAMPLE`: both are load-bearing for the metric (trap 1), and a
    quietly different file would move a published-looking number without
    moving anything a stamp records."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; have {sorted(SPLITS)}")
    filename, expected = SPLITS[split]
    data_path = Path(path) / filename
    rows = [json.loads(line) for line in data_path.open(encoding="utf-8") if line.strip()]
    if len(rows) != expected:
        raise ValueError(
            f"{data_path}: expected {expected} rows for split {split!r}, got {len(rows)}"
        )
    odd = [i for i, r in enumerate(rows) if len(split_tags(r.get("target", ""))) != TAGS_PER_SAMPLE]
    if odd:
        raise ValueError(
            f"{data_path}: {len(odd)} rows do not carry {TAGS_PER_SAMPLE} tags "
            f"(first at index {odd[0]}) — the online loop's accuracy arithmetic assumes they do"
        )
    return rows


def parse_instruction_and_input(all_context: str) -> tuple[str, str]:
    """data_processor.py:31-46, transcribed including its fall-through.

    The shipped format is ``Instruction: ...\\nInput: ...\\nAnswer: ``. When
    either marker is missing upstream returns ``("", all_context)`` — the whole
    blob becomes the *question* and the context is empty — so a format change
    degrades to a different prompt rather than an error. Kept, because a run
    that silently reshapes the prompt is exactly what a fidelity port has to be
    able to reproduce; `process_task_data` counts how often it happens."""
    if "Input: " in all_context and "Instruction: " in all_context:
        instruction_part = all_context.split("Input: ")[0].strip()
        instruction_part = instruction_part.split("Instruction: ")[1].strip()
        remaining = all_context.split("Input: ")[1]
        input_text = remaining.split("Answer: ")[0].strip()
        return input_text, instruction_part
    return "", all_context


def process_task_data(raw: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """data_processor.py:85-124 for `task_name="finer"`. Returns the processed
    samples and the number that hit the fall-through above."""
    out: list[dict[str, Any]] = []
    fell_through = 0
    for item in raw:
        context = item.get("context", "")
        input_text, question = parse_instruction_and_input(context)
        if not input_text:
            fell_through += 1
        out.append(
            {
                "context": input_text,
                "question": question,
                "target": item.get("target", ""),
            }
        )
    return out, fell_through


def windows(samples: list[Any], size: int) -> Iterator[tuple[int, list[Any]]]:
    """The online loop's windowing (ace.py:934-942): consecutive slices of
    `online_eval_frequency` samples, the last one short. Each window is tested
    with the playbook as it stands and only then trained on, which is what makes
    the online number a measure of adaptation rather than of a final artifact."""
    if size < 1:
        raise ValueError(f"window size must be >= 1, got {size}")
    for start in range(0, len(samples), size):
        yield start, samples[start : start + size]


# ---------------- answer extraction ----------------


def extract_answer(response: str) -> str:
    """utils.py:100-130, transcribed: JSON first, then `Finish[...]`, then a
    `final_answer` regex in double and single quotes, else `NO_ANSWER`.

    The ladder's order matters and its last rung is a sentinel, not an
    exception — a model that answers in prose scores as wrong rather than
    crashing the run, and `aggregate` reports how many landed there."""
    try:
        parsed = json.loads(response)
        return str(parsed.get("final_answer", NO_ANSWER))
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    for pattern in (_FINISH_RE, _JSON_DQ_RE, _JSON_SQ_RE):
        matches = pattern.findall(response or "")
        if matches:
            return matches[-1]
    return NO_ANSWER


# ---------------- scoring ----------------


def split_tags(value: str) -> list[str]:
    """data_processor.py:128-131: split on commas, lowercase, strip."""
    return [v.lower().strip() for v in str(value).split(",")]


def _coerce(value: str) -> Any:
    """What upstream's `eval()` (trap 3) achieves on data that is not a tag
    name, without running the model's output as code.

    Upstream calls `eval` on both sides so that `"5.0"` and `"5"` compare
    equal. Numbers are the whole of what that buys on this task family;
    anything `eval` would do beyond parsing a literal is behaviour we do not
    want. Non-numeric input returns unchanged, which is the branch FiNER always
    takes.

    The comma strip is DEAD and kept dead, exactly as upstream's is: the caller
    split on commas before reaching here, so `"1,000"` arrived as two tags and
    there is no comma left to remove. Reproducing it as live would change the
    metric ACE's number is stated in."""
    text = value.replace(",", "").replace("$", "").strip()
    try:
        return float(text)
    except ValueError:
        return value


def tag_counts(predicted: str, ground_truth: str) -> tuple[int, int, int]:
    """Upstream's per-sample tag arithmetic (data_processor.py:126-152),
    returning `(correct, total, n_over)`.

    `total` is the gold's tag count after upstream's length reconciliation:
    a longer prediction is truncated and a shorter one padded with `""`, so the
    denominator is always the gold's. `n_over` is how many predicted tags the
    truncation discarded unscored — upstream's fourth trap, surfaced rather than
    fixed, because the metric ACE publishes is the one with it in."""
    pred = split_tags(predicted)
    label = split_tags(ground_truth)
    n_over = max(0, len(pred) - len(label))
    if len(pred) != len(label):
        if len(pred) > len(label):
            pred = pred[: len(label)]
        else:
            pred += [""] * (len(label) - len(pred))
    correct = sum(1 for p, g in zip(pred, label) if _coerce(p) == _coerce(g))
    return correct, len(pred), n_over


def answer_is_correct(predicted: str, ground_truth: str) -> bool:
    """Sample-level exact match — every tag right (data_processor.py:152,
    `score == 1`). This is the signal the training loop branches on, and it is
    NOT the quantity upstream reports as accuracy (trap 1, trap 5)."""
    correct, total, _ = tag_counts(predicted, ground_truth)
    return total > 0 and correct == total


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Both accuracies, always, plus the counts that make them readable.

    - `tag_accuracy` — upstream's reported metric: correct tags / gold tags,
      micro-averaged over the split (`_evaluate_finer_accuracy`).
    - `sample_accuracy` — upstream's `correct/total`: the fraction of samples
      with every tag right.

    They are different numbers on the same run and upstream prints them as one
    (utils.py:286). A row with no answer counts as wrong in both, rather than
    leaving the split (trap 2); `n_failed` and `n_no_answer` say how many rows
    that was."""
    if not rows:
        return {"tag_accuracy": 0.0, "sample_accuracy": 0.0, "n": 0}
    correct_tags = sum(int(r["correct_tags"]) for r in rows)
    total_tags = sum(int(r["total_tags"]) for r in rows)
    exact = sum(1 for r in rows if r.get("is_correct"))
    return {
        "tag_accuracy": round(100 * correct_tags / total_tags, 2) if total_tags else 0.0,
        "sample_accuracy": round(100 * exact / len(rows), 2),
        "n": len(rows),
        "n_tags": total_tags,
        "n_failed": sum(1 for r in rows if r.get("failed")),
        "n_no_answer": sum(1 for r in rows if r.get("pred") == NO_ANSWER),
        "n_over_predicted": sum(int(r.get("n_over", 0)) for r in rows),
    }


# ---------------- pipeline ----------------


def answer(
    mem: AgenticMemory,
    sample: dict[str, Any],
    reflection: str = "(empty)",
    capture: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """One generator turn: inject the FULL playbook, generate, extract.

    The playbook comes from `mem.get_playbook()` — ACE's read contract is whole-
    playbook injection, never top-k (organizers/ace §read contract), and the
    facade enforces it by keeping `playbook` out of `default_memory_types`.
    Returns `(final_answer, bullet_ids)`; `bullet_ids` is what the generator
    claims it used, which is the signal upstream's reflector attributes counters
    with. Raises `RuntimeError` when no structured client is configured, rather
    than scoring a run that never called a model."""
    if mem.structured is None:
        raise RuntimeError("finer.answer requires a structured LLM client")
    playbook = mem.get_playbook() or "(empty)"
    prompt = GENERATOR_PROMPT.format(
        playbook=playbook,
        reflection=reflection,
        question=sample["question"],
        context=sample["context"],
    )
    reply = mem.structured.call(
        "generate", prompt, GENERATE_SCHEMA, required_keys=("final_answer",)
    )
    if not reply:
        return NO_ANSWER, []
    raw = reply.get("final_answer", "")
    # A healthy structured reply needs no extraction. A degraded one — the field
    # missing, or the model having stuffed its whole free-text answer into it —
    # goes through upstream's own ladder rather than being scored as prose (D1).
    final = str(raw).strip()
    if not final:
        final = extract_answer(json.dumps(reply, ensure_ascii=False))
    bullet_ids = [str(b) for b in (reply.get("bullet_ids") or [])]
    if capture is not None:
        capture.update(
            {
                "playbook_chars": len(playbook),
                "bullet_ids": bullet_ids,
                "reasoning": reply.get("reasoning", ""),
            }
        )
    return final, bullet_ids


def score_sample(sample: dict[str, Any], pred: str, failed: bool = False) -> dict[str, Any]:
    """One scored row, carrying both correctness views so `aggregate` never has
    to re-derive either."""
    correct, total, n_over = tag_counts(pred, sample["target"])
    return {
        "question": sample["question"][:200],
        "target": sample["target"],
        "pred": pred,
        "correct_tags": correct,
        "total_tags": total,
        "n_over": n_over,
        "is_correct": total > 0 and correct == total,
        "failed": failed,
    }


def adapt(mem: AgenticMemory, sample: dict[str, Any], row: dict[str, Any]) -> None:
    """The training half of one online step: hand the graded attempt to the
    organizers as a finished task.

    Upstream reflects on the generator's *reasoning trace* and, when wrong,
    regenerates up to three times (ace.py:498-560). We cut at `on_task_end`, so
    the trajectory is the single attempt and the outcome is upstream's
    all-or-nothing `is_correct` — the same signal its loop branches on, which
    keeps the two comparable on what is learned from even though they differ on
    how many calls it takes."""
    trajectory = [
        {
            "step": 1,
            "action": "answer_finer_sample",
            "question": sample["question"][:2000],
            "prediction": row["pred"],
            "target": sample["target"],
            "tags_correct": f"{row['correct_tags']}/{row['total_tags']}",
        }
    ]
    mem.add_task_result(
        trajectory,
        outcome="success" if row["is_correct"] else "failure",
        task=sample["question"][:2000],
    )
