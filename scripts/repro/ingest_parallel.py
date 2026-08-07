"""Conversation-parallel ingest orchestrator (rate-limit controllable).

WHY: ingest is the paid, hours-long phase — per turn it issues ~2 sequential LLM
calls (note construction Ps1 + evolution Ps3). Within ONE conversation those
calls MUST stay sequential: evolution is a read-modify-write on the growing note
graph, so turn N links against the state left by turns 1..N-1. Reorder or
parallelize them and the graph — hence retrieval, hence answers — changes.

But ACROSS conversations there is zero shared state: ``exp_amem_repro`` already
builds each conversation in a fresh ``AgenticMemory`` (namespace ``repro-conv{i}``,
its own persisted store), and the sequential harness closes one before opening the
next. Conversation order never affected any result. So ingesting the 10
conversations CONCURRENTLY yields byte-identical per-conversation graphs — it only
overlaps the network waits, cutting wall-clock by up to ~#convs.

This orchestrator is a DROP-IN for ``exp_amem_repro.py --conv all --ingest-only``:
it fans out one ``--conv i --ingest-only --no-sentinel`` subprocess per
conversation (each fully isolated — own store namespace, own five artifacts via a
per-conv ``--tag-suffix``), capped at ``--workers`` concurrent workers, then writes
the SAME combined ``.ingest_complete.json`` sentinel + ``<model>_all_ingest<sfx>.json``
summary the sequential path would, so ``--eval-only`` and the headline aggregator
work unchanged.

RATE-LIMIT CONTROL (the point of ``--workers``): each conversation worker has at
most ONE LLM call in flight at a time (its current turn), so in-flight API calls
≈ ``--workers``. That is the single knob to stay under an account's RPM/TPM:
start at 4, raise if no 429s. Defense in depth: the OpenAI SDK already retries
429/5xx twice with exponential backoff (``OpenAI(...)`` default ``max_retries=2``),
and this orchestrator additionally retries a whole failed conversation up to
``--retries`` times (wiping its partial store first, since re-ingesting a
populated store would duplicate notes). ``--stagger`` spreads worker startup so
they do not all hit the API on the same tick.

RAM — and why ``--workers`` is not purely a rate-limit knob: a local
sentence-transformers embedder loads torch per worker (~1 GB RSS), and a config
whose store overrides bring up embedded engines (Nemori's Postgres + Qdrant)
adds a server process per conversation on top. On 2026-08-04 three concurrent
Nemori workers on a 4-core/8 GB WSL2 host drove swap to 5 GB, and the host
started failing DNS: one conversation died outright and another finished having
silently dropped 15 structured-output calls. Size ``--workers`` against the
HOST, not just the rate limit, and treat the clean-ingest check in
``conv_is_done`` as the backstop rather than the plan. A hosted ``--embedder``
(APIEmbedder) removes the torch share but not the store-engine share.

Usage (mirrors the sequential ingest, add --workers):
    uv run python scripts/repro/ingest_parallel.py \\
        --data-dir results/repro/stores/full_all_seed1 \\
        --convs all --workers 4 --tag-suffix _seed1

    # a non-amem methodology (artifact names gain the config segment):
    uv run python scripts/repro/ingest_parallel.py \\
        --config nemori_upstream --embedder text-embedding-3-small \\
        --data-dir results/repro/stores/nemori-armA-e3s \\
        --convs all --workers 1 --tag-suffix _e3sA
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Reuse the harness's sentinel writer, cost model, and per-run budget merge so the
# combined summary + sentinel are byte-for-byte what the sequential path emits.
# exp_amem_repro lives in scripts/ (one level up); import it as a module (its
# __main__ guard means importing does not run anything).
_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import exp_amem_repro as H  # noqa: E402

logger = logging.getLogger(__name__)


def parse_convs(spec: str) -> list[int]:
    """Turn a --convs spec into a sorted, de-duplicated conversation-index list.
    Accepts ``all`` (0..9), a range ``a-b`` (inclusive), or a comma list
    ``0,2,5`` (ranges allowed inside, e.g. ``0-3,7``). Out-of-range indices raise
    so a typo cannot silently under-ingest."""
    if spec.strip() == "all":
        return list(range(10))
    out: set[int] = set()
    try:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:  # inclusive range "a-b" (indices are non-negative)
                a, b = part.split("-", 1)
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
    except ValueError as exc:
        raise SystemExit(f"--convs {spec!r} is malformed ({exc}); use 'all', '0-9', or '0,2,5'")
    bad = [i for i in out if i < 0 or i > 9]
    if bad:
        raise SystemExit(f"conv indices out of range 0-9: {sorted(bad)}")
    if not out:
        raise SystemExit(f"--convs {spec!r} selected no conversations")
    return sorted(out)


def _model_safe(model: str, config: str = "amem") -> str:
    """Match the harness's output-tag sanitization AND its config segment
    (exp_amem_repro.main): a non-default --config is appended to the model part
    of every artifact name, while `amem` leaves names untouched.

    Missing the config half is not cosmetic. Every path this orchestrator
    computes — the completion check, the per-conv summaries it merges, the
    combined summary it writes — would point at a filename the harness never
    wrote. `conv_is_done` would answer False for conversations that WERE
    ingested, so a resumed or retried run would silently re-ingest and re-pay
    for all of them, and the merge step would then fail on files that do not
    exist."""
    safe = model.replace("/", "-").replace(":", "-")
    return safe if config == "amem" else f"{safe}_{config}"


def store_dir_for(data_dir: str, conv: int) -> Path:
    """Persisted store dir for one conversation: ``<data_dir>/repro-conv{i}``
    (AgmemConfig data_dir + the harness namespace ``repro-conv{i}``)."""
    return Path(data_dir).expanduser() / f"repro-conv{conv}"


def per_conv_summary_path(model: str, conv: int, tag_suffix: str, config: str = "amem") -> Path:
    """Per-conv ingest summary the worker writes (tag ``<model>_conv{i}_ingest<sfx>_c{i}``).
    Its EXISTENCE is the completion signal: the harness writes it only after the
    conv's ingest + snapshot fully finish, so a present summary ⟹ that conv's
    store is complete."""
    return H.OUT / f"{_model_safe(model, config)}_conv{conv}_ingest{tag_suffix}_c{conv}.json"


def drop_budget(summary: dict, max_drop_rate: float) -> int:
    """How many role-keyed drops this conversation may carry and still count as
    clean: ``floor(max_drop_rate * structured calls)``.

    PROPORTIONAL, because the thing being defended against is proportional. The
    zero-drop bar was set against a host-failure incident that lost 15 calls out
    of ~1,180 (1.3%) — a corrupted measurement. It cannot tell that apart from
    one malformed reply in 2,177 (0.05%), which is ordinary model
    non-determinism, and at Zep's call volume the second is the common case: a
    per-call malformed rate of 1/2177 makes a clean conversation a 37% event, so
    a binary gate turns a healthy run into an endless wipe-and-re-pay loop.

    Sized so the incident that motivated the gate still fails it: at the default
    0.1%, 1,180 calls buy a budget of 1 and the DNS run's 15 drops are still
    rejected, while the pilot's single drop passes. `embed` is excluded — it is
    not a structured call and cannot drop this way.
    """
    if max_drop_rate <= 0:
        return 0
    budget = summary.get("llm_budget") or {}
    structured = sum(s.get("calls", 0) for role, s in budget.items() if role != "embed")
    return int(structured * max_drop_rate)


def conv_is_done(
    model: str,
    data_dir: str,
    conv: int,
    tag_suffix: str,
    config: str = "amem",
    max_drop_rate: float = 0.0,
) -> bool:
    """A conversation counts as already-ingested iff its per-conv summary exists,
    its store dir is non-empty, AND that summary reports a CLEAN ingest.

    The first two rule out a crashed worker (summary missing) and a summary
    orphaned from its store. The third rules out the failure mode that motivated
    it, found live on 2026-08-04: under memory pressure the host began failing
    DNS, and while one conversation crashed outright — caught by the existence
    check — another RAN TO COMPLETION while losing 15 structured-output calls to
    connection errors. It wrote every artifact, so it looked finished, and its
    memory capacity then differed from a clean ingest of the same conversation
    for a reason that had nothing to do with the variable under study.

    An ingest that silently dropped work is a wrong measurement, not a cheap
    one, so it is treated as not-done and retried on a clean store. The bar is
    the one every conversation of the comparison baseline already meets: zero
    LLM errors and zero drops OF THE KIND THAT MEAN LOST WORK.

    That qualifier is load-bearing, because ``_merge_budget`` folds two unlike
    things into one ``drops`` block. Structured-output failures are keyed by
    ROLE (``extract``, ``distill``) and mean the harness paid for a response it
    could not parse — the 2026-08-04 signal. Organizer discards are namespaced
    ``"{organizer}/{reason}"`` and mean the response arrived intact and the
    organizer judged it inapplicable. Pooling them is right for cost accounting
    and wrong for a health gate, so only the role-keyed half gates here.

    Mem0 is what forces the distinction: ``mem0/hallucinated_id`` fires on
    every conversation by construction — upstream maps memory ids onto small
    integers precisely because the model invents UUIDs when shown real ones —
    so gating on it would mark every Mem0 conversation not-done, and
    ``_run_one`` would wipe the store and re-ingest to the retry cap before
    reporting FAILED. Measured on the conv0 pilot: six such discards in a run
    that was otherwise perfectly clean (420 calls, zero LLM errors).
    """
    sp = per_conv_summary_path(model, conv, tag_suffix, config)
    sd = store_dir_for(data_dir, conv)
    if not (sp.exists() and sd.exists() and any(sd.rglob("*"))):
        return False
    try:
        summary = json.loads(sp.read_text())
    except (OSError, json.JSONDecodeError):
        return False  # unreadable/truncated summary — not done, re-ingest cleanly
    lost = sum(n for key, n in (summary.get("drops") or {}).items() if "/" not in key)
    if lost > drop_budget(summary, max_drop_rate):
        return False
    # LLM errors stay at ZERO regardless of the drop budget. A drop is a reply
    # that arrived and would not parse; an error is a call that did not complete,
    # which is the host/transport signal the tolerance is not meant to cover.
    return not any(s.get("errors", 0) for s in (summary.get("llm_budget") or {}).values())


def worker_log_path(model: str, conv: int, tag_suffix: str, config: str = "amem") -> Path:
    """Where one worker's stdout+stderr is persisted.

    Under ``results/repro/logs/``, which docs/14 deliberately keeps git-tracked
    (the .gitignore un-ignores it despite the global ``*.log``), because a run's
    driver output is part of the durable record rather than console noise."""
    return (
        H.OUT / "logs" / f"{_model_safe(model, config)}_conv{conv}_ingest{tag_suffix}_c{conv}.log"
    )


def worker_cmd(args, conv: int) -> list[str]:
    """The subprocess command for one conversation's ingest.

    Split out of ``_run_one`` so the flag list is testable without spawning
    anything — ``--config`` in particular, whose absence made this orchestrator
    silently A-Mem-only: every conversation would ingest with the default `amem`
    organizer no matter which methodology the caller asked for, and the run would
    look perfectly successful."""
    return [
        sys.executable,
        str(_SCRIPTS / "exp_amem_repro.py"),
        "--conv",
        str(conv),
        "--ingest-only",
        "--no-sentinel",
        "--config",
        args.config,
        "--data-dir",
        args.data_dir,
        "--tag-suffix",
        f"{args.tag_suffix}_c{conv}",
        "--model",
        args.model,
        "--endpoint",
        args.endpoint,
        "--embedder",
        args.embedder,
        "--expand-links",
        args.expand_links,
    ] + (
        # getattr, not attribute access: this must fail CLOSED. An args object
        # without the field (a caller's namespace, an older invocation) then
        # omits the flag and the worker's run_ready gate applies — the safe
        # direction. Reading it directly would turn a missing field into a
        # crash mid-fan-out instead.
        ["--allow-unverified-config"] if getattr(args, "allow_unverified_config", False) else []
    )


def merge_ingest_summaries(summaries: list[dict], model: str) -> dict:
    """Fold per-conv ingest summaries into the combined blocks the sequential
    ``--conv all --ingest-only`` summary carries: summed per-role LLM budget
    (via the harness's run-budget merge, which re-averages latency from the
    call-weighted total), recomputed cost (at `model`'s registered rates),
    summed drops, summed ingest seconds, and merged memory_capacity (per-type
    counts + totals + bytes)."""
    budget = H._merge_run_budgets([s.get("llm_budget", {}) for s in summaries])
    drops: dict = {}
    for s in summaries:
        for role, n in (s.get("drops") or {}).items():
            drops[role] = drops.get(role, 0) + n
    ingest_s = round(sum((s.get("timing") or {}).get("ingest_s", 0.0) or 0.0 for s in summaries), 1)

    # Per-op counts keyed "OP:target_type", summed like every other block. The
    # snapshot in memory_capacity says what the memory ENDED as; only this says
    # what the write path DID — an UPDATE, a DELETE and a NOOP all leave a
    # snapshot that looks the same. `or None` mirrors the sequential path's
    # `op_counts or None` so an arm whose organizers log nothing reads as absent
    # rather than as a measured zero.
    op_counts: dict[str, int] = {}
    for s in summaries:
        for op, n in (s.get("op_counts") or {}).items():
            op_counts[op] = op_counts.get(op, 0) + n

    per_type: dict[str, int] = {}
    total_items = 0
    mem_bytes = 0
    for s in summaries:
        cap = s.get("memory_capacity") or {}
        for t, c in (cap.get("per_type") or {}).items():
            per_type[t] = per_type.get(t, 0) + c
        total_items += cap.get("total_items", 0) or 0
        mem_bytes += cap.get("memory_jsonl_bytes", 0) or 0
    memory_capacity = {
        "per_type": per_type,
        "total_items": total_items,
        "memory_jsonl_bytes": mem_bytes,
    }
    return {
        "llm_budget": budget,
        # judge_model omitted (defaults to None): ingest only issues extract/distill
        # calls, never a "judge" role, so there is nothing to split-price here.
        "cost_usd": H.cost_usd(budget, model),
        "drops": drops,
        "ingest_s": ingest_s,
        "memory_capacity": memory_capacity,
        "op_counts": op_counts or None,
    }


def _run_one(args, conv: int) -> tuple[int, bool, str]:
    """Ingest ONE conversation in an isolated subprocess, with retries. Returns
    (conv, ok, note). Skips instantly if already done; wipes a partial store
    before each (re)attempt so re-ingest is clean (locomo.ingest is not
    idempotent — re-ingesting a populated store would duplicate notes)."""
    rate = getattr(args, "max_drop_rate", 0.0)
    if conv_is_done(args.model, args.data_dir, conv, args.tag_suffix, args.config, rate):
        return conv, True, "skipped (already complete)"
    sd = store_dir_for(args.data_dir, conv)
    cmd = worker_cmd(args, conv)
    last = ""
    for attempt in range(1, args.retries + 2):  # 1 initial + args.retries retries
        if sd.exists():
            shutil.rmtree(sd)  # partial/crashed store -> clean slate
        proc = subprocess.run(cmd, capture_output=True, text=True)
        # Persist the worker's output BEFORE deciding success/failure. It is not
        # console noise: Nemori logs its merge-candidate similarity scores on an
        # INFO channel (organizers/nemori/stages.py), and that distribution is
        # the only evidence for whether the 0.85 merge threshold is
        # embedder-relative — the design risk the Track 1 precheck refused to
        # ship without a mitigation. This used to keep the last line of a
        # FAILURE and drop everything else, so a SUCCESSFUL conversation threw
        # the whole channel away. Appended per attempt so a retry does not erase
        # what the previous attempt recorded.
        log_path = worker_log_path(args.model, conv, args.tag_suffix, args.config)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"===== attempt {attempt} =====\n")
                fh.write(proc.stdout or "")
                fh.write(proc.stderr or "")
        except OSError:
            logger.exception("failed to persist worker log for conv%s", conv)
        if proc.returncode == 0 and conv_is_done(
            args.model, args.data_dir, conv, args.tag_suffix, args.config, rate
        ):
            note = "ok" if attempt == 1 else f"ok (attempt {attempt})"
            return conv, True, note
        last = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
        last = last[0][:300]
        if attempt <= args.retries:
            time.sleep(min(30.0, 2.0**attempt))  # backoff before conv-level retry
    return conv, False, f"FAILED after {args.retries + 1} attempts: {last}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Conversation-parallel drop-in for exp_amem_repro --conv all --ingest-only"
    )
    ap.add_argument("--data-dir", required=True, help="persist stores here (shared across convs)")
    ap.add_argument("--convs", default="all", help="'all', range '0-9', or list '0,2,5'")
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="max concurrent conversation ingests ≈ max in-flight API calls "
        "(the rate-limit knob; each conv keeps at most one call in flight). RAM ≈ "
        "workers × ~1 GB. Start at 4; raise if no 429s.",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=2,
        help="conv-level retries on failure (on top of the OpenAI SDK's own 429 "
        "backoff); the partial store is wiped before each retry",
    )
    ap.add_argument(
        "--stagger",
        type=float,
        default=1.0,
        help="seconds between launching workers, to avoid a startup thundering herd",
    )
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--endpoint", default="https://api.openai.com/v1")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument(
        "--config",
        default="amem",
        help="organizer config from scripts/repro/configs.py (default amem)",
    )
    ap.add_argument("--expand-links", choices=["off", "on"], default="off")
    ap.add_argument("--tag-suffix", default="", help="e.g. _seed1 (matches the sequential path)")
    ap.add_argument(
        "--max-drop-rate",
        type=float,
        default=0.0,
        help="fraction of a conversation's STRUCTURED calls that may be dropped and still "
        "count as a clean ingest (default 0.0 = the historical zero-drop bar). Use on arms "
        "whose per-conversation call count makes zero drops improbable; 0.001 still rejects "
        "the 2026-08-04 host-failure run. Recorded in the combined summary.",
    )
    ap.add_argument(
        "--allow-unverified-config",
        action="store_true",
        help="pilot a config whose run_ready is False. Refuses more than one conversation "
        "here for the same reason the worker does: the gate exists to stop a full "
        "campaign's spend on an arm that has never survived a real run.",
    )
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be >= 1")

    convs = parse_convs(args.convs)
    if args.allow_unverified_config and len(convs) > 1:
        raise SystemExit(
            f"--allow-unverified-config covers a single-conversation pilot only, "
            f"got {len(convs)} ({args.convs}). The worker refuses this too; it is "
            f"checked here as well so the refusal costs nothing instead of arriving "
            f"one subprocess spawn at a time."
        )
    print(
        f"[parallel-ingest] {len(convs)} convs {convs} | workers={args.workers} "
        f"(≈{args.workers} in-flight API calls) | data-dir={args.data_dir}",
        flush=True,
    )
    t0 = time.perf_counter()
    results: dict[int, tuple[bool, str]] = {}
    # ThreadPool of subprocess.run workers: threads only wait on child processes,
    # so max_workers caps concurrent conv ingests (hence API concurrency) exactly.
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for i, conv in enumerate(convs):
            if i and args.stagger:
                time.sleep(args.stagger)
            futs[ex.submit(_run_one, args, conv)] = conv
        for fut in as_completed(futs):
            conv, ok, note = fut.result()
            results[conv] = (ok, note)
            print(f"[parallel-ingest] conv{conv}: {note}", flush=True)
    wall_s = round(time.perf_counter() - t0, 1)

    failed = sorted(c for c, (ok, _) in results.items() if not ok)
    if failed:
        for c in failed:
            print(f"[parallel-ingest] conv{c} {results[c][1]}", flush=True)
        raise SystemExit(
            f"{len(failed)} conv(s) failed to ingest: {failed}. NOT writing the "
            f"combined sentinel — rerun this command (completed convs are skipped) "
            f"before --eval-only."
        )

    out_path, sentinel, combined = finalize_combined(args, convs, wall_s)
    merged_ingest_s = combined["timing"]["ingest_s"]
    speedup = f"{merged_ingest_s / wall_s:.1f}x" if wall_s else "n/a"
    print(
        f"[parallel-ingest] DONE {len(convs)} convs in {wall_s}s wall "
        f"(vs {merged_ingest_s}s sequential compute ≈ {speedup} faster); "
        f"cost ${combined['cost_usd']}",
        flush=True,
    )
    print(f"[done] wrote {out_path} (combined ingest summary)", flush=True)
    print(f"[done] wrote {sentinel} (ingest-completion sentinel)", flush=True)


def finalize_combined(args, convs: list[int], wall_s: float) -> tuple[Path, Path, dict]:
    """Once every conv is ingested, aggregate the per-conv summaries into the
    combined ``<model>_all_ingest<sfx>.json`` + write the single authoritative
    ``.ingest_complete.json`` sentinel — byte-for-byte the pair the sequential
    ``--conv all --ingest-only`` emits, so ``--eval-only`` and the headline
    aggregator are unchanged. Returns (summary_path, sentinel_path, combined_dict)."""
    summaries = [
        json.loads(per_conv_summary_path(args.model, c, args.tag_suffix, args.config).read_text())
        for c in convs
    ]
    merged = merge_ingest_summaries(summaries, args.model)
    sha = H.git_sha()
    per_conv = [
        {"conv": c, "n_turns": (s.get("per_conv") or [{}])[0].get("n_turns")}
        for c, s in zip(convs, summaries)
    ]
    # store on-disk footprint of the shared data-dir (all conv namespaces).
    merged["memory_capacity"]["store_dir_bytes"] = H.dir_size_bytes(
        Path(args.data_dir).expanduser()
    )

    combined = {
        "stamp": {
            **{k: summaries[0].get("stamp", {}).get(k) for k in summaries[0].get("stamp", {})},
            "conv": "all",
            "commit": sha,  # canonical name; git_sha kept for existing readers
            "git_sha": sha,
            "parallel_ingest": {
                "workers": args.workers,
                "convs": convs,
                "wall_s": wall_s,
                # The clean-ingest bar this run was accepted under. Stamped
                # because it is not a constant across arms any more: an arm run
                # at a non-zero tolerance may carry dropped write-path calls
                # that the zero-drop arms could not, and a cross-arm comparison
                # has to be able to see that from the artifact alone.
                "max_drop_rate": getattr(args, "max_drop_rate", 0.0),
            },
        },
        "ingest_only": True,
        "per_conv": per_conv,
        "llm_budget": merged["llm_budget"],
        "cost_usd": merged["cost_usd"],
        "drops": merged["drops"],
        "timing": {
            "ingest_s": merged["ingest_s"],  # summed compute time across convs
            "wall_s": wall_s,  # actual wall-clock (overlapped)
            "total_s": merged["ingest_s"],
        },
        "memory_capacity": merged["memory_capacity"],
        "op_counts": merged["op_counts"],
        "per_conv_summaries": [
            per_conv_summary_path(args.model, c, args.tag_suffix, args.config).name for c in convs
        ],
    }
    out_path = H.OUT / f"{_model_safe(args.model, args.config)}_all_ingest{args.tag_suffix}.json"
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    sentinel = H.write_ingest_sentinel(args.data_dir, convs, per_conv, sha)
    return out_path, sentinel, combined


if __name__ == "__main__":
    main()
