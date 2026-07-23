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
they do not all hit the API on the same tick. RAM: each worker loads torch +
the embedder (~1 GB RSS), so peak ≈ ``workers × 1 GB``.

Usage (mirrors the sequential ingest, add --workers):
    uv run python scripts/repro/ingest_parallel.py \\
        --data-dir results/repro/stores/full_all_seed1 \\
        --convs all --workers 4 --tag-suffix _seed1
"""

from __future__ import annotations

import argparse
import json
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


def _model_safe(model: str) -> str:
    """Match the harness's output-tag sanitization (exp_amem_repro.main)."""
    return model.replace("/", "-").replace(":", "-")


def store_dir_for(data_dir: str, conv: int) -> Path:
    """Persisted store dir for one conversation: ``<data_dir>/repro-conv{i}``
    (AgmemConfig data_dir + the harness namespace ``repro-conv{i}``)."""
    return Path(data_dir).expanduser() / f"repro-conv{conv}"


def per_conv_summary_path(model: str, conv: int, tag_suffix: str) -> Path:
    """Per-conv ingest summary the worker writes (tag ``<model>_conv{i}_ingest<sfx>_c{i}``).
    Its EXISTENCE is the completion signal: the harness writes it only after the
    conv's ingest + snapshot fully finish, so a present summary ⟹ that conv's
    store is complete."""
    return H.OUT / f"{_model_safe(model)}_conv{conv}_ingest{tag_suffix}_c{conv}.json"


def conv_is_done(model: str, data_dir: str, conv: int, tag_suffix: str) -> bool:
    """A conversation counts as already-ingested iff its per-conv summary exists
    AND its store dir is non-empty — the pair rules out both a crashed worker
    (summary missing) and a summary orphaned from its store."""
    sp = per_conv_summary_path(model, conv, tag_suffix)
    sd = store_dir_for(data_dir, conv)
    return sp.exists() and sd.exists() and any(sd.rglob("*"))


def merge_ingest_summaries(summaries: list[dict]) -> dict:
    """Fold per-conv ingest summaries into the combined blocks the sequential
    ``--conv all --ingest-only`` summary carries: summed per-role LLM budget
    (via the harness's run-budget merge, which re-averages latency from the
    call-weighted total), recomputed cost, summed drops, summed ingest seconds,
    and merged memory_capacity (per-type counts + totals + bytes)."""
    budget = H._merge_run_budgets([s.get("llm_budget", {}) for s in summaries])
    drops: dict = {}
    for s in summaries:
        for role, n in (s.get("drops") or {}).items():
            drops[role] = drops.get(role, 0) + n
    ingest_s = round(sum((s.get("timing") or {}).get("ingest_s", 0.0) or 0.0 for s in summaries), 1)

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
        "cost_usd": H.cost_usd(budget),
        "drops": drops,
        "ingest_s": ingest_s,
        "memory_capacity": memory_capacity,
    }


def _run_one(args, conv: int) -> tuple[int, bool, str]:
    """Ingest ONE conversation in an isolated subprocess, with retries. Returns
    (conv, ok, note). Skips instantly if already done; wipes a partial store
    before each (re)attempt so re-ingest is clean (locomo.ingest is not
    idempotent — re-ingesting a populated store would duplicate notes)."""
    if conv_is_done(args.model, args.data_dir, conv, args.tag_suffix):
        return conv, True, "skipped (already complete)"
    sd = store_dir_for(args.data_dir, conv)
    cmd = [
        sys.executable,
        str(_SCRIPTS / "exp_amem_repro.py"),
        "--conv",
        str(conv),
        "--ingest-only",
        "--no-sentinel",
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
    ]
    last = ""
    for attempt in range(1, args.retries + 2):  # 1 initial + args.retries retries
        if sd.exists():
            shutil.rmtree(sd)  # partial/crashed store -> clean slate
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and conv_is_done(args.model, args.data_dir, conv, args.tag_suffix):
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
    ap.add_argument("--expand-links", choices=["off", "on"], default="off")
    ap.add_argument("--tag-suffix", default="", help="e.g. _seed1 (matches the sequential path)")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be >= 1")

    convs = parse_convs(args.convs)
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
        json.loads(per_conv_summary_path(args.model, c, args.tag_suffix).read_text()) for c in convs
    ]
    merged = merge_ingest_summaries(summaries)
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
            "git_sha": sha,
            "parallel_ingest": {"workers": args.workers, "convs": convs, "wall_s": wall_s},
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
        "per_conv_summaries": [
            per_conv_summary_path(args.model, c, args.tag_suffix).name for c in convs
        ],
    }
    out_path = H.OUT / f"{_model_safe(args.model)}_all_ingest{args.tag_suffix}.json"
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    sentinel = H.write_ingest_sentinel(args.data_dir, convs, per_conv, sha)
    return out_path, sentinel, combined


if __name__ == "__main__":
    main()
