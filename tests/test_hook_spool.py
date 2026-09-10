from __future__ import annotations

import json
import multiprocessing
import threading
import time
from pathlib import Path

from agmem.hooks.preserve import spool
from agmem.hooks.spool import drain_spool


class RetryableTestError(Exception):
    pass


def _drain_spool_process(
    queue_text: str,
    seen_text: str,
    started_text: str,
    release_text: str,
    results_text: str,
) -> None:
    queue = Path(queue_text)
    seen = Path(seen_text)
    started = Path(started_text)
    release = Path(release_text)
    results = Path(results_text)

    def action(path: str) -> None:
        with seen.open("a", encoding="utf-8") as fp:
            fp.write(path + "\n")
        started.write_text("1", encoding="utf-8")
        deadline = time.monotonic() + 5
        while not release.exists():
            if time.monotonic() >= deadline:
                raise RetryableTestError("release marker was not written")
            time.sleep(0.02)

    result = drain_spool(queue, action)
    with results.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps({"result": result}) + "\n")


def _wait_exists(path: Path) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"{path} was not created")


def _row(transcript: Path) -> dict[str, str]:
    return {"transcript_path": str(transcript), "session_id": transcript.stem}


def test_drain_spool_leaves_appends_made_while_action_runs(tmp_path):
    transcript = tmp_path / "one.jsonl"
    transcript.write_text("{}\n")
    late = tmp_path / "late.jsonl"
    late.write_text("{}\n")
    queue = tmp_path / "preserve-queue.jsonl"
    spool(_row(transcript), queue)
    seen: list[str] = []

    def action(path: str) -> None:
        seen.append(path)
        spool(_row(late), queue)

    assert drain_spool(queue, action) == 1

    assert seen == [str(transcript)]
    (remaining,) = queue.read_text(encoding="utf-8").splitlines()
    assert json.loads(remaining)["transcript_path"] == str(late)
    assert not queue.with_name(f"{queue.name}.processing").exists()


def test_drain_spool_keeps_raised_actions_retryable(tmp_path, caplog):
    transcript = tmp_path / "retry.jsonl"
    transcript.write_text("{}\n")
    queue = tmp_path / "distill-queue.jsonl"
    spool(_row(transcript), queue)
    calls = 0

    def failing_action(_path: str) -> None:
        nonlocal calls
        calls += 1
        raise RetryableTestError("temporary failure")

    with caplog.at_level("WARNING", logger="agmem.hooks.spool"):
        assert drain_spool(queue, failing_action) == 0
    assert calls == 1
    assert "temporary failure" in caplog.text
    assert queue.read_text(encoding="utf-8") == ""
    processing = queue.with_name(f"{queue.name}.processing")
    (retry,) = processing.read_text(encoding="utf-8").splitlines()
    assert json.loads(retry)["transcript_path"] == str(transcript)

    seen: list[str] = []
    assert drain_spool(queue, seen.append) == 1
    assert seen == [str(transcript)]
    assert not processing.exists()


def test_drain_spool_attempts_new_appends_while_a_failed_row_is_retryable(tmp_path):
    failing = tmp_path / "failing.jsonl"
    failing.write_text("{}\n")
    later = tmp_path / "later.jsonl"
    later.write_text("{}\n")
    queue = tmp_path / "preserve-queue.jsonl"
    spool(_row(failing), queue)

    def action(path: str) -> None:
        if path == str(failing):
            raise RetryableTestError("still failing")
        seen.append(path)

    seen: list[str] = []
    assert drain_spool(queue, action) == 0
    spool(_row(later), queue)

    assert drain_spool(queue, action) == 1

    assert seen == [str(later)]
    processing = queue.with_name(f"{queue.name}.processing")
    (retry,) = processing.read_text(encoding="utf-8").splitlines()
    assert json.loads(retry)["transcript_path"] == str(failing)


def test_drain_spool_recovers_processing_file_left_by_restart(tmp_path):
    transcript = tmp_path / "restart.jsonl"
    transcript.write_text("{}\n")
    queue = tmp_path / "preserve-queue.jsonl"
    processing = queue.with_name(f"{queue.name}.processing")
    processing.write_text(json.dumps(_row(transcript)) + "\n", encoding="utf-8")
    seen: list[str] = []

    assert drain_spool(queue, seen.append) == 1

    assert seen == [str(transcript)]
    assert not processing.exists()


def test_drain_spool_serializes_concurrent_consumers(tmp_path):
    transcript = tmp_path / "one-active.jsonl"
    transcript.write_text("{}\n")
    queue = tmp_path / "distill-queue.jsonl"
    spool(_row(transcript), queue)
    started = threading.Event()
    release = threading.Event()
    seen: list[str] = []
    results: list[int] = []

    def action(path: str) -> None:
        seen.append(path)
        started.set()
        release.wait(timeout=5)

    def drain() -> None:
        results.append(drain_spool(queue, action))

    first = threading.Thread(target=drain)
    second = threading.Thread(target=drain)

    first.start()
    assert started.wait(timeout=5)
    second.start()
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert seen == [str(transcript)]
    assert sorted(results) == [0, 1]
    assert not queue.with_name(f"{queue.name}.processing").exists()


def test_drain_spool_serializes_concurrent_processes(tmp_path):
    transcript = tmp_path / "one-process-active.jsonl"
    transcript.write_text("{}\n")
    queue = tmp_path / "distill-queue.jsonl"
    seen = tmp_path / "seen.txt"
    started = tmp_path / "started.txt"
    release = tmp_path / "release.txt"
    results = tmp_path / "results.jsonl"
    spool(_row(transcript), queue)

    first = multiprocessing.Process(
        target=_drain_spool_process,
        args=(str(queue), str(seen), str(started), str(release), str(results)),
    )
    second = multiprocessing.Process(
        target=_drain_spool_process,
        args=(str(queue), str(seen), str(started), str(release), str(results)),
    )
    first.start()
    _wait_exists(started)
    second.start()
    release.write_text("1", encoding="utf-8")
    first.join(timeout=5)
    second.join(timeout=5)

    assert first.exitcode == 0
    assert second.exitcode == 0
    assert seen.read_text(encoding="utf-8").splitlines() == [str(transcript)]
    assert sorted(json.loads(line)["result"] for line in results.read_text().splitlines()) == [
        0,
        1,
    ]
    assert not queue.with_name(f"{queue.name}.processing").exists()


def test_drain_spool_quarantines_malformed_and_missing_rows(tmp_path):
    transcript = tmp_path / "ok.jsonl"
    transcript.write_text("{}\n")
    missing = tmp_path / "missing.jsonl"
    queue = tmp_path / "preserve-queue.jsonl"
    queue.write_text(
        "not json\n"
        + json.dumps({"session_id": "empty"})
        + "\n"
        + json.dumps({"transcript_path": str(missing)})
        + "\n"
        + json.dumps(_row(transcript))
        + "\n",
        encoding="utf-8",
    )
    seen: list[str] = []

    assert drain_spool(queue, seen.append) == 1

    assert seen == [str(transcript)]
    assert queue.read_text(encoding="utf-8") == ""
    assert not queue.with_name(f"{queue.name}.processing").exists()
    bad = [
        json.loads(line)
        for line in queue.with_name(f"{queue.name}.bad").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["reason"] for row in bad] == ["malformed-json", "missing-path", "missing-file"]
    assert bad[0]["line"] == "not json"
