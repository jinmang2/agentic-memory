"""The two places the LongMemEval C4 driver can lose a question quietly.

Both are about the same failure the benchmark's own code has (LME-A18): a run
that did not answer everything reporting a score anyway. The driver's guard is a
completeness check, and these tests cover the parts of it that a row COUNT
cannot see — plus the cost fold, which is what keeps a 500-call judge from being
priced at the reader's rate.

No dataset and no network: the driver's helpers are pure functions over rows.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import logging
import sys
from pathlib import Path

_DRIVER = Path(__file__).resolve().parent.parent / "scripts" / "repro" / "exp_lme_reading.py"


def _load_driver():
    if str(_DRIVER.parent) not in sys.path:
        sys.path.insert(0, str(_DRIVER.parent))
    spec = _ilu.spec_from_file_location("exp_lme_reading", _DRIVER)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_an_unjudged_row_is_not_resumable(tmp_path):
    """A question whose API call failed writes a row so the failure cannot vanish.
    Treating that row as done would leave a hole no row count can see: 500 rows,
    499 judged, aggregated over 499 while the stamp says complete."""
    driver = _load_driver()
    path = tmp_path / "arm.records.jsonl"
    _write(
        path,
        [
            {"question_id": "q1", "label": True},
            {"question_id": "q2", "label": None, "error": "APIError: 429"},
            {"question_id": "q3", "label": False},
        ],
    )
    kept, done = driver.resume_rows(path, logging.getLogger("t"))
    assert [r["question_id"] for r in kept] == ["q1", "q3"]
    assert done == {"q1", "q3"}, "q2 must be re-answered, not skipped"


def test_a_truncated_final_line_ends_the_resume_rather_than_raising(tmp_path):
    driver = _load_driver()
    path = tmp_path / "arm.records.jsonl"
    path.write_text(
        json.dumps({"question_id": "q1", "label": True})
        + "\n"
        + '{"question_id": "q2", "lab'  # the shape a kill leaves
    )
    kept, done = driver.resume_rows(path, logging.getLogger("t"))
    assert done == {"q1"}
    assert [r["question_id"] for r in kept] == ["q1"]


def test_row_keys_fold_back_into_roles_so_the_judge_prices_as_the_judge(tmp_path):
    """Per-row budget keys are what make a concurrent run's cost attributable at
    all, and they are also what would misprice it: `cost_usd` prices the key
    named `judge` at the judge model's rates, so 500 keys named `judge|<qid>`
    would bill the pinned gpt-4o judge at gpt-4o-mini's rates."""
    driver = _load_driver()
    raw = {
        "generate|q1": {
            "calls": 1,
            "tokens_in": 1_000_000,
            "tokens_out": 0,
            "latency_ms_avg": 10.0,
        },
        "generate|q2": {
            "calls": 1,
            "tokens_in": 1_000_000,
            "tokens_out": 0,
            "latency_ms_avg": 30.0,
        },
        "judge|q1": {"calls": 1, "tokens_in": 1_000_000, "tokens_out": 0, "latency_ms_avg": 5.0},
    }
    folded = driver.fold_row_keys(raw)
    assert folded["generate"]["calls"] == 2
    assert folded["generate"]["tokens_in"] == 2_000_000
    assert folded["generate"]["latency_ms_avg"] == 20.0  # call-weighted, not key-averaged
    assert folded["judge"]["calls"] == 1

    from agmem.bench.registry import registry_cost_usd_split as cost_usd_split

    priced = cost_usd_split(folded, "gpt-4o-mini", {"judge": "gpt-4o-2024-08-06"})
    # 2M reader tokens at $0.15/M + 1M judge tokens at $2.50/M
    assert round(priced, 4) == round(2 * 0.15 + 2.50, 4)
    # The same call ledger priced WITHOUT folding cannot see a judge at all.
    unfolded = cost_usd_split(raw, "gpt-4o-mini", {"judge": "gpt-4o-2024-08-06"})
    assert round(unfolded, 4) == round(3 * 0.15, 4)
