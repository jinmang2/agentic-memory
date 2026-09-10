"""Exercise new offline commands through the public CLI with sockets denied."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest_plugins = ("test_lme_v2_tools_cli",)

MODULE = "agmem.bench.lme_v2_tools"


def test_diagnose_preserves_unknown_evidence(
    scored_root: Path, offline_env: dict[str, str]
) -> None:
    result = subprocess.run(
        [sys.executable, "-m", MODULE, "diagnose", str(scored_root)],
        env=offline_env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report
    assert "q1" in result.stdout
    assert "unknown" in result.stdout or "null" in result.stdout


def test_costs_reports_synthetic_retry_without_double_counting(
    offline_env: dict[str, str],
) -> None:
    fixture = (
        Path(__file__).resolve().parents[1] / "experiments/lme_v2_followup/cost-ledger.fixture.json"
    )
    result = subprocess.run(
        [sys.executable, "-m", MODULE, "costs", str(fixture)],
        env=offline_env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["total_micro_usd"] == 70
    assert report["retry_cost_micro_usd"] == 10


@pytest.mark.parametrize("action", ["diagnose", "costs"])
def test_followup_cli_preserves_existing_output(
    tmp_path: Path, scored_root: Path, offline_env: dict[str, str], action: str
) -> None:
    fixture = (
        Path(__file__).resolve().parents[1] / "experiments/lme_v2_followup/cost-ledger.fixture.json"
    )
    source = scored_root if action == "diagnose" else fixture
    output = tmp_path / "report.json"
    output.write_text("keep existing report", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", MODULE, action, str(source), "--output", str(output)],
        env=offline_env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2
    assert "File exists" in result.stderr
    assert output.read_text(encoding="utf-8") == "keep existing report"
    assert "Traceback" not in result.stderr
