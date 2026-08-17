"""Re-diff our aggregate() against the official print_qa_metrics.py at SHA 9e0b455.

Not a reimplementation comparison: we SHELL OUT to the official script and parse
what it prints, so the reference is the released artifact, not our reading of it.
Unequal type counts and a 25% abstention share on purpose -- that is where the
two accuracies separate.
"""

import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
from agmem.bench.longmemeval import QUESTION_TYPES, aggregate

OFFICIAL = Path.home() / ".agmem/upstream/longmemeval/src/evaluation/print_qa_metrics.py"
rng = random.Random(20260817)
worst = {"task_averaged": 0.0, "overall": 0.0, "abstention": 0.0}
trials = 50
for t in range(trials):
    records, refs = [], []
    for qt in QUESTION_TYPES:  # deliberately unequal counts
        for i in range(rng.randint(3, 40)):
            abs_ = rng.random() < 0.25
            qid = f"t{t}_{qt}_{i}" + ("_abs" if abs_ else "")
            label = rng.random() < 0.6
            records.append({"question_id": qid, "question_type": qt, "label": label})
            refs.append({"question_id": qid, "question_type": qt})
    with tempfile.TemporaryDirectory() as d:
        hyp = Path(d) / "h.log"
        ref = Path(d) / "r.json"
        hyp.write_text(
            "\n".join(
                json.dumps(
                    {
                        "question_id": r["question_id"],
                        "autoeval_label": {"model": "gpt-4o-2024-08-06", "label": r["label"]},
                    }
                )
                for r in records
            )
        )
        ref.write_text(json.dumps(refs))
        out = subprocess.run(
            ["uv", "run", "--with", "numpy", "python", str(OFFICIAL), str(hyp), str(ref)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    off = {
        "task_averaged": float(re.search(r"Task-averaged Accuracy: ([\d.]+)", out).group(1)) * 100,
        "overall": float(re.search(r"Overall Accuracy: ([\d.]+)", out).group(1)) * 100,
        "abstention": float(re.search(r"Abstention Accuracy: ([\d.]+)", out).group(1)) * 100,
    }
    ours = aggregate(records)
    mine = {
        "task_averaged": ours["task_averaged"],
        "overall": ours["overall"],
        "abstention": ours["abstention"]["acc"],
    }
    for k, prev in worst.items():
        # official rounds to 4dp on the 0-1 scale = 2dp in pp, which is our rounding
        worst[k] = max(prev, abs(off[k] - mine[k]))
print(f"{trials} randomized record sets, unequal type counts, ~25% abstention")
for k, v in worst.items():
    print(f"  max |ours - official| on {k:<14} = {v:.6f} pp")
