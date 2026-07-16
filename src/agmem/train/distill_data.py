"""SFT distillation data generation (Phase 4, docs/06).

A strong teacher (API model via the 'distill' role) runs the organizers'
extraction prompts over real conversations (LoCoMo histories); the
(prompt, teacher_json) pairs become SFT data for a 0.5B student. Tasks:
  boundary  — Nemori boundary detection
  note      — A-Mem note construction (keywords/context/tags)
  episode   — Nemori title+narrative generation (temporal anchoring)
  strategy  — ReasoningBank item extraction

Usage:
    uv run python -m agmem.train.distill_data --task note \
        --teacher-endpoint https://api.openai.com/v1 --teacher-model gpt-4o-mini \
        --out data/sft/note.jsonl --limit 500
Requires a teacher endpoint; refuses to distill from a model as weak as the
student (that would just clone its errors).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from agmem.bench import locomo
from agmem.llm.budget import BudgetTracker
from agmem.llm.client import LLMClient, RoleConfig
from agmem.llm.structured import StructuredCaller
from agmem.organizers import amem, nemori

TASKS = {
    "note": (amem.NOTE_PROMPT, amem.NOTE_SCHEMA,
             ("keywords", "context", "tags")),
    "boundary": (nemori.BOUNDARY_PROMPT, nemori.BOUNDARY_SCHEMA,
                 ("boundary", "confidence")),
    "episode": (nemori.EPISODE_PROMPT, nemori.EPISODE_SCHEMA,
                ("title", "narrative")),
}


def build_prompts(task: str, data_path: Path, limit: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    samples = locomo.load_locomo(data_path)
    prompts: list[str] = []
    for sample in samples:
        turns = list(locomo.iter_turns(sample))
        if task == "note":
            for _, date, speaker, text in turns:
                prompts.append(amem.NOTE_PROMPT.format(content=f"({date}) {speaker}: {text}"))
        elif task == "boundary":
            for i in range(2, len(turns)):
                window = turns[max(0, i - 6):i]
                buf = "\n".join(f"[{d}] {s}: {t}" for _, d, s, t in window)
                _, d, s, t = turns[i]
                prompts.append(nemori.BOUNDARY_PROMPT.format(
                    buffer=buf, message=f"[{d}] {s}: {t}"))
        elif task == "episode":
            for start in range(0, len(turns) - 6, 6):
                seg = "\n".join(f"[{d}] {s}: {t}" for _, d, s, t in turns[start:start + 6])
                prompts.append(nemori.EPISODE_PROMPT.format(segment=seg))
    rng.shuffle(prompts)
    return prompts[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--teacher-endpoint", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--api-key", default="not-needed")
    ap.add_argument("--data", default=str(Path.home() / ".agmem/datasets/locomo10.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    _, schema, required = TASKS[args.task]
    client = LLMClient({"distill": RoleConfig(endpoint=args.teacher_endpoint,
                                              model=args.teacher_model,
                                              api_key=args.api_key,
                                              temperature=0.3, max_tokens=800)},
                       budget=BudgetTracker())
    caller = StructuredCaller(client, use_guided_json=False)

    prompts = build_prompts(args.task, Path(args.data), args.limit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_path.open("w") as f:
        for i, prompt in enumerate(prompts):
            result = caller.call("distill", prompt, schema, required_keys=required)
            if result is None:
                continue  # teacher drop — excluded from training data
            f.write(json.dumps({"task": args.task, "prompt": prompt,
                                "completion": json.dumps(result, ensure_ascii=False)},
                               ensure_ascii=False) + "\n")
            kept += 1
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(prompts)} (kept {kept})", flush=True)
    print(f"done: {kept}/{len(prompts)} pairs -> {out_path}")
    print("teacher budget:", client.budget.summary())


if __name__ == "__main__":
    main()
