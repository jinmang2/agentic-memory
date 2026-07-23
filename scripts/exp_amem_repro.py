"""A-Mem x LoCoMo REPRODUCTION harness (docs/14).

Drives the reproduction ladder for our A-Mem re-implementation against the
LoCoMo dataset on an OpenAI-compatible endpoint (default gpt-4o-mini). Mirrors
``scripts/exp_locomo_conv0.py``'s ``run()`` structure but:
  - loads OPENAI_API_KEY from repo-root ``.env.local`` (stdlib, no dotenv dep),
  - defaults to the WujiangXu/A-Mem faithful eval (``--eval-mode wujiang``),
  - uses the upstream A-Mem write/answer temperatures (0.7 write, 0.7 generate,
    0.5 cat5) and the all-MiniLM-L6-v2 embedder,
  - supports ``--conv all`` (micro-average over all 10 conversations),
  - supports write-once/read-sweep via ``--data-dir`` + ``--ingest-only`` /
    ``--eval-only`` for cheap k-sweeps.

This file is separate from exp_locomo_conv0.py, which is UNTOUCHED (it answers
at t=0.0 on a local Qwen model). Do NOT merge the two.

Examples (see scripts/repro/*.sh):
    uv run python scripts/exp_amem_repro.py --conv 0 --eval-mode wujiang
    uv run python scripts/exp_amem_repro.py --conv all --eval-mode ours --judge \\
        --expand-links on
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from agmem import AgenticMemory
from agmem._env import load_env_local
from agmem.bench import locomo
from agmem.config import AgmemConfig
from agmem.embed.st_embedder import SentenceTransformerEmbedder
from agmem.llm.client import RoleConfig
from agmem.organizers.amem import AMemOrganizer
from agmem.stores.sqlite_doc import episode_to_dict

DATA = Path.home() / ".agmem/datasets/locomo10.json"
OUT = Path(__file__).resolve().parent.parent / "results" / "repro"

# A-Mem answer channel: notes-only dense retrieval with LLM keyword queries
# (WujiangXu test_advanced.py generate_query_llm), matching exp_locomo_conv0's
# `amem` config. cat5 answers at 0.5 (upstream temperature_c5).
MEMORY_TYPES = ("notes",)
CAT5_TEMPERATURE = 0.5

# gpt-4o-mini published rates (USD per 1M tokens) used to turn the token budget
# into a self-describing cost_usd. Stored into the summary stamp so the number
# is reproducible even if list prices change later.
COST_RATES = {"model": "gpt-4o-mini", "usd_per_1m_in": 0.15, "usd_per_1m_out": 0.60}

# Memory types the snapshot enumerates. `episodic` is dumped from the episodes
# table (list_episodes); the rest are derived item types across all methodologies
# — empty types are skipped, so listing extras is harmless and future-proof.
SNAPSHOT_ITEM_TYPES = (
    "notes",
    "semantic",
    "facts",
    "entities",
    "episodes",
    "pages",
    "playbook",
    "strategies",
    "experiences",
    "state",
)


def git_sha() -> str | None:
    """Current HEAD sha (short) for the run stamp, or None outside a git tree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


SENTINEL_NAME = ".ingest_complete.json"


def write_ingest_sentinel(
    data_dir: str, conv_indices: list[int], per_conv: list[dict], sha
) -> Path:
    """Stamp a completion marker at ``<data_dir>/.ingest_complete.json`` AFTER a
    full ingest finishes. Its presence (and the conv list inside) is how
    ``--eval-only`` and the phase scripts distinguish a COMPLETE persisted store
    from a directory left behind by a crashed/partial ingest — a bare directory
    check would silently evaluate an incomplete store and micro-average over
    fewer conversations than intended."""
    path = Path(data_dir).expanduser() / SENTINEL_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "conv_indices": conv_indices,
                "n_convs": len(conv_indices),
                "per_conv": per_conv,
                "git_sha": sha,
                "utc_finished": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return path


def verify_ingest_sentinel(data_dir: str, conv_indices: list[int]) -> None:
    """Refuse to eval a store whose ingest did not provably COMPLETE for the
    requested conversations. Raises ``SystemExit`` (loud, actionable) when the
    sentinel is missing (partial/crashed ingest, or a store from before this
    guard existed) or does not cover every requested conv — never silently
    scores a truncated store."""
    path = Path(data_dir).expanduser() / SENTINEL_NAME
    if not path.exists():
        raise SystemExit(
            f"No ingest-completion sentinel at {path}. The store is missing or its "
            f"ingest did not complete. Re-run --ingest-only (delete the dir first "
            f"to avoid duplicate notes) before --eval-only."
        )
    try:
        done = set(json.loads(path.read_text()).get("conv_indices", []))
    except Exception as exc:
        raise SystemExit(f"Unreadable ingest sentinel {path}: {exc!r}") from exc
    missing = sorted(set(conv_indices) - done)
    if missing:
        raise SystemExit(
            f"Store {data_dir} was ingested for convs {sorted(done)} but eval "
            f"requested {conv_indices} (missing {missing}). Re-ingest the missing "
            f"conversations before --eval-only."
        )


def dir_size_bytes(path: Path) -> int:
    """Total bytes of every file under `path` (0 if missing) — for the persisted
    store's on-disk footprint in memory_capacity."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def cost_usd(merged_budget: dict) -> float:
    """USD cost from summed tokens × COST_RATES, across all roles."""
    tin = sum(s.get("tokens_in", 0) for s in merged_budget.values())
    tout = sum(s.get("tokens_out", 0) for s in merged_budget.values())
    return round(
        tin / 1_000_000 * COST_RATES["usd_per_1m_in"]
        + tout / 1_000_000 * COST_RATES["usd_per_1m_out"],
        6,
    )


def dump_memory_snapshot(mem: AgenticMemory, conv_idx: int, out) -> dict[str, int]:
    """Append the full memory state of one conversation to the open `out` handle
    — one JSON line per stored item across episodic + every derived type. Each
    line carries `{conv, memory_type, ...item fields}` (whatever the item dict
    holds: content/tags/links/keywords/context/metadata/kind). Returns a per-type
    count dict for the run's memory_capacity block."""
    counts: dict[str, int] = {}
    list_eps = getattr(mem.doc_store, "list_episodes", None)
    if callable(list_eps):
        eps = list_eps(mem.namespace)
        for ep in eps:
            out.write(
                json.dumps(
                    {"conv": conv_idx, "memory_type": "episodic", **episode_to_dict(ep)},
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
        if eps:
            counts["episodic"] = len(eps)
    for mtype in SNAPSHOT_ITEM_TYPES:
        items = mem.doc_store.list_items(mtype, namespace=mem.namespace)
        if not items:
            continue
        counts[mtype] = len(items)
        for item in items:
            out.write(
                json.dumps(
                    {"conv": conv_idx, "memory_type": mtype, **item},
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
    return counts


def make_roles(
    endpoint: str,
    model: str,
    api_key: str,
    write_temp: float = 0.7,
    generate_temp: float = 0.7,
    max_tokens: int = 1000,
) -> dict[str, RoleConfig]:
    """A-Mem upstream temperatures (test_advanced.py / memory_layer get_completion
    default 0.7): write-path roles (extract=Ps1 note, distill=Ps3 evolution) at
    ``write_temp`` 0.7; ``generate`` (cat1-4 answers) at ``generate_temp`` 0.7;
    cat5 answers override to 0.5 per-call in ``locomo.answer``. ``judge`` stays
    0.0 (deterministic grading, only used in ``ours`` mode)."""
    return {
        "extract": RoleConfig(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            temperature=write_temp,
            max_tokens=max_tokens,
        ),
        "distill": RoleConfig(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            temperature=write_temp,
            max_tokens=max_tokens,
        ),
        "judge": RoleConfig(
            endpoint=endpoint, model=model, api_key=api_key, temperature=0.0, max_tokens=max_tokens
        ),
        "generate": RoleConfig(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            temperature=generate_temp,
            max_tokens=max_tokens,
        ),
    }


def build_memory(
    args, embedder, conv_idx: int, roles, trace_path: Path | None = None
) -> AgenticMemory:
    """Fresh A-Mem-organized memory for one conversation. When ``--data-dir`` is
    set it is threaded into ``AgmemConfig.data_dir`` so the stores persist under
    ``<data_dir>/<namespace>/`` (config.py:65-123) — enabling ``--ingest-only``
    then ``--eval-only`` reload. Link expansion is toggled on the pipeline
    (``link_expansion_cap`` 5=on / 0=off). ``trace_path`` (when given) turns on
    the client's full-I/O trace sink so every LLM call of this conversation is
    appended to the shared run trace."""
    cfg = AgmemConfig(
        profile="lite",
        data_dir=Path(args.data_dir).expanduser() if args.data_dir else None,
        llm_roles=roles,
        use_guided_json=False,
        lexical_types=("episodic",),
        # Pin synchronous writes: the ingest-completion sentinel attests that the
        # ingest LOOP finished, and that only implies every note was actually
        # built+persisted if organizer writes run inline (a hard failure then
        # propagates out of locomo.ingest BEFORE the sentinel is stamped). With
        # async writes, organizer exceptions are log-only and a silently-failed
        # note would still be marked complete. Kept explicit so a default change
        # can't quietly weaken the sentinel guarantee.
        sync_write=True,
    )
    mem = AgenticMemory(
        namespace=f"repro-conv{conv_idx}",
        organizers=[AMemOrganizer()],
        embedder=embedder,
        config=cfg,
    )
    mem.pipeline.link_expansion_cap = 5 if args.expand_links == "on" else 0
    if trace_path is not None and mem.llm is not None:
        mem.llm.trace_path = trace_path
    return mem


def combine_aggs(aggs: list[dict]) -> dict:
    """Exact micro-average of per-conversation agg dicts (each carries mean
    f1/bleu1 as 0-100 percentages plus row count ``n``). Weighting each conv's
    mean by its ``n`` recovers the total F1 sum, so this equals scoring every
    QA in one pool. Also folds ``j_score``/``j_n`` when present."""
    n = sum(a["n"] for a in aggs)
    if n == 0:
        return {}
    out = {
        "f1": round(sum(a["f1"] * a["n"] for a in aggs) / n, 2),
        "bleu1": round(sum(a["bleu1"] * a["n"] for a in aggs) / n, 2),
        "n": n,
    }
    jn = sum(a.get("j_n", 0) for a in aggs)
    if jn:
        out["j_score"] = round(sum(a.get("j_score", 0) * a.get("j_n", 0) for a in aggs) / jn, 2)
        out["j_n"] = jn
    return out


def micro_average(results: list[dict]) -> dict:
    """Combine per-conversation evaluate() outputs into one micro-averaged
    overall + by_category block."""
    overall = combine_aggs([r["overall"] for r in results if r["overall"]])
    cats: dict[str, list[dict]] = {}
    for r in results:
        for cat, agg in r["by_category"].items():
            cats.setdefault(cat, []).append(agg)
    by_category = {cat: combine_aggs(aggs) for cat, aggs in sorted(cats.items())}
    return {"overall": overall, "by_category": by_category}


def write_records_sidecar(path: Path, runs_out: list[dict]) -> int:
    """Persist the per-question audit trail to a JSONL sidecar — one JSON object
    per line, each tagged with ``run`` (run number) and ``conv`` (conversation
    index) plus the ``locomo.evaluate()`` record fields (q/gold/pred/cat/f1, and
    ``j`` when judged). JSONL keeps it appendable/streamable/greppable. Every
    question of every conversation and every run is written. Returns the number
    of lines written."""
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in runs_out:
            run_no = r.get("run")
            for rec in r.get("records", []):
                f.write(json.dumps({"run": run_no, **rec}, ensure_ascii=False) + "\n")
                n += 1
    return n


def eval_conversations(
    args,
    embedder,
    roles,
    conv_indices: list[int],
    trace_path: Path | None = None,
    memory_path: Path | None = None,
) -> dict:
    """Ingest (unless --eval-only) + QA over the selected conversations, then
    micro-average. Returns the combined result plus a merged LLM budget, merged
    structured-output drops, split ingest/eval timing, and (when `memory_path`
    is given) a full post-ingest memory snapshot + memory_capacity block. Every
    LLM call is appended to `trace_path` when set."""
    samples = locomo.load_locomo(DATA)
    per_conv = []
    all_records: list[dict] = []
    merged_budget: dict = {}
    merged_drops: dict = {}
    total_questions = 0
    ingest_s = 0.0
    eval_s = 0.0
    snapshot_counts: dict[str, int] = {}
    # Snapshot the memory of every conversation to one JSONL (truncate per call
    # so a re-run of the same tag overwrites rather than appends stale state).
    mem_file = memory_path.open("w", encoding="utf-8") if memory_path else None
    try:
        for idx in conv_indices:
            sample = samples[idx]
            mem = build_memory(args, embedder, idx, roles, trace_path=trace_path)
            try:
                if not args.eval_only:
                    t_ing = time.perf_counter()
                    n_turns = locomo.ingest(mem, sample)
                    mem.consolidate()
                    ingest_s += time.perf_counter() - t_ing
                if mem_file is not None:
                    for mtype, c in dump_memory_snapshot(mem, idx, mem_file).items():
                        snapshot_counts[mtype] = snapshot_counts.get(mtype, 0) + c
                if args.ingest_only:
                    print(
                        f"[conv{idx}] ingested {n_turns} turns, "
                        f"notes persisted under {args.data_dir}",
                        flush=True,
                    )
                    per_conv.append({"conv": idx, "n_turns": n_turns})
                    _merge_budget(merged_budget, merged_drops, mem)
                    continue
                questions = locomo.select_questions(sample)
                total_questions += len(questions)
                t_eval = time.perf_counter()
                res = locomo.evaluate(
                    mem,
                    questions,
                    k=args.k,
                    memory_types=MEMORY_TYPES,
                    keyword_queries=True,
                    judge=args.judge and args.eval_mode == "ours",
                    eval_mode=args.eval_mode,
                    cat5_temperature=CAT5_TEMPERATURE,
                    capture_retrieval=True,
                    workers=args.workers,
                    progress=lambda i, nq, _i=idx: (
                        print(f"[conv{_i}] {i}/{nq}", flush=True) if i % 40 == 0 else None
                    ),
                )
                eval_s += time.perf_counter() - t_eval
                res["conv"] = idx
                per_conv.append(res)
                # keep the per-question audit trail (records) for the durable
                # sidecar, tagged with the conversation index; the summary JSON
                # stays lean.
                for rec in res.get("records", []):
                    all_records.append({"conv": idx, **rec})
                _merge_budget(merged_budget, merged_drops, mem)
            finally:
                mem.close()
    finally:
        if mem_file is not None:
            mem_file.close()

    memory_capacity = None
    if memory_path is not None:
        memory_capacity = {
            "per_type": snapshot_counts,
            "total_items": sum(snapshot_counts.values()),
            "memory_jsonl_bytes": memory_path.stat().st_size if memory_path.exists() else 0,
            "store_dir_bytes": (
                dir_size_bytes(Path(args.data_dir).expanduser()) if args.data_dir else None
            ),
        }

    if args.ingest_only:
        return {
            "ingest_only": True,
            "per_conv": per_conv,
            "llm_budget": merged_budget,
            "drops": merged_drops,
            "ingest_s": round(ingest_s, 1),
            "memory_capacity": memory_capacity,
        }

    combined = micro_average([r for r in per_conv if "overall" in r])
    combined["per_conv"] = [
        {"conv": r["conv"], "overall": r["overall"], "by_category": r["by_category"]}
        for r in per_conv
        if "overall" in r
    ]
    combined["llm_budget"] = merged_budget
    combined["drops"] = merged_drops
    combined["n_questions"] = total_questions
    combined["ingest_s"] = round(ingest_s, 1)
    combined["eval_s"] = round(eval_s, 1)
    combined["memory_capacity"] = memory_capacity
    # carried internally to the sidecar writer in main(); NOT inlined into the
    # summary JSON (which selects a lean set of keys).
    combined["records"] = all_records
    return combined


def _merge_budget(merged_budget: dict, merged_drops: dict, mem: AgenticMemory) -> None:
    """Fold one conversation's per-role LLM budget and structured-output drops
    into the run-level accumulators (summed across all conversations). Latency is
    accumulated as a true call-weighted total (``latency_ms_avg * calls`` per
    conv, since ``BudgetTracker.summary`` exposes only the average) so the merged
    ``latency_ms_avg`` is the correct mean over ALL calls of the run — not the
    last conversation's average (the prior bug, which made the summary's latency
    reflect only conv 9 under ``--conv all``)."""
    for role, stats in mem.budget.summary().items():
        agg = merged_budget.setdefault(
            role,
            {
                "calls": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "errors": 0,
                "latency_ms_total": 0.0,
                "latency_ms_avg": 0.0,
            },
        )
        calls = stats.get("calls", 0)
        agg["calls"] += calls
        agg["tokens_in"] += stats.get("tokens_in", 0)
        agg["tokens_out"] += stats.get("tokens_out", 0)
        agg["errors"] += stats.get("errors", 0)
        agg["latency_ms_total"] += stats.get("latency_ms_avg", 0.0) * calls
        agg["latency_ms_avg"] = (
            round(agg["latency_ms_total"] / agg["calls"], 1) if agg["calls"] else 0.0
        )
    if mem.structured is not None:
        for role, n in mem.structured.drops.items():
            merged_drops[role] = merged_drops.get(role, 0) + n


def _merge_run_budgets(budgets: list[dict]) -> dict:
    """Sum a list of per-run (already per-conv-merged) budgets into one campaign
    budget — calls/tokens/errors added, latency re-averaged from the summed
    ``latency_ms_total``. Used so the top-level ``llm_budget``/``cost_usd``
    reflect what was ACTUALLY paid across all ``--runs`` (each run re-issues the
    answer calls), not just run 1 (the prior bug — runs 2..N were dropped from
    the summary)."""
    merged: dict = {}
    for b in budgets:
        for role, stats in b.items():
            agg = merged.setdefault(
                role,
                {
                    "calls": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "errors": 0,
                    "latency_ms_total": 0.0,
                    "latency_ms_avg": 0.0,
                },
            )
            agg["calls"] += stats.get("calls", 0)
            agg["tokens_in"] += stats.get("tokens_in", 0)
            agg["tokens_out"] += stats.get("tokens_out", 0)
            agg["errors"] += stats.get("errors", 0)
            # per-run budgets come from _merge_budget, which carries the exact
            # call-weighted latency_ms_total — re-average from that sum. .get with
            # a 0.0 default so a budget from any other source can't KeyError here.
            agg["latency_ms_total"] += stats.get("latency_ms_total", 0.0)
            agg["latency_ms_avg"] = (
                round(agg["latency_ms_total"] / agg["calls"], 1) if agg["calls"] else 0.0
            )
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="A-Mem x LoCoMo reproduction harness")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--endpoint", default="https://api.openai.com/v1")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--conv", default="0", help="conversation index (0-9) or 'all'")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--eval-mode", choices=["wujiang", "ours"], default="wujiang")
    ap.add_argument("--expand-links", choices=["off", "on"], default="off")
    ap.add_argument("--judge", action="store_true", help="J-score (ours mode only)")
    ap.add_argument("--runs", type=int, default=1, help="repeat QA for mean±std")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent QA workers (read-only store; results identical to 1)",
    )
    ap.add_argument("--data-dir", default=None, help="persist stores here")
    ap.add_argument(
        "--tag-suffix",
        default="",
        help="append to the output tag (e.g. _seed1) so repeated full runs into "
        "distinct stores don't overwrite each other's artifacts",
    )
    ap.add_argument("--ingest-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be >= 1")

    if args.ingest_only and not args.data_dir:
        ap.error("--ingest-only requires --data-dir (ephemeral stores would be lost)")
    if args.eval_only and not args.data_dir:
        ap.error("--eval-only requires --data-dir (nothing to reload otherwise)")
    if args.ingest_only and args.eval_only:
        ap.error("--ingest-only and --eval-only are mutually exclusive")

    load_env_local()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to repo-root .env.local "
            "(KEY=VALUE, gitignored) or export it before running."
        )

    conv_indices = list(range(10)) if args.conv == "all" else [int(args.conv)]
    # Fail loud if asked to score a store whose ingest didn't provably finish for
    # these convs (partial/crashed ingest) — before spending any answer calls.
    if args.eval_only:
        verify_ingest_sentinel(args.data_dir, conv_indices)
    roles = make_roles(args.endpoint, args.model, api_key)
    embedder = SentenceTransformerEmbedder(args.embedder)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stores").mkdir(parents=True, exist_ok=True)
    model_safe = args.model.replace("/", "-").replace(":", "-")
    conv_tag = "all" if args.conv == "all" else f"conv{args.conv}"

    utc_started = datetime.now(timezone.utc).isoformat()
    sha = git_sha()

    if args.ingest_only:
        tag = f"{model_safe}_{conv_tag}_ingest{args.tag_suffix}"
        # five-artifact capture even for ingest: trace + memory snapshot land on
        # durable disk so the (paid) write path is never re-run to inspect it.
        trace_path = OUT / f"{tag}.llm-trace.jsonl"
        memory_path = OUT / f"{tag}.memory.jsonl"
        combined = eval_conversations(
            args, embedder, roles, conv_indices, trace_path=trace_path, memory_path=memory_path
        )
        summary = {
            "stamp": _stamp(args, sha, utc_started, datetime.now(timezone.utc).isoformat(), None),
            "ingest_only": True,
            "per_conv": combined.get("per_conv"),
            "llm_budget": combined.get("llm_budget"),
            "cost_usd": cost_usd(combined.get("llm_budget", {})),
            "drops": combined.get("drops"),
            "timing": {"ingest_s": combined.get("ingest_s"), "total_s": combined.get("ingest_s")},
            "memory_capacity": combined.get("memory_capacity"),
            "llm_trace_file": trace_path.name,
            "memory_file": memory_path.name,
        }
        # Mark the store COMPLETE for exactly the convs just ingested so a later
        # --eval-only (or the phase scripts) can trust it — a crashed ingest never
        # reaches this line, so a bare store dir without the sentinel is rejected.
        sentinel = write_ingest_sentinel(
            args.data_dir, conv_indices, combined.get("per_conv") or [], sha
        )
        out_path = OUT / f"{tag}.json"
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[ingest-only] done ({conv_tag}); reload with --eval-only", flush=True)
        print(f"[done] wrote {sentinel} (ingest-completion sentinel)", flush=True)
        print(f"[done] wrote {out_path}", flush=True)
        print(f"[done] wrote {trace_path} (full LLM I/O trace)", flush=True)
        print(f"[done] wrote {memory_path} (memory snapshot)", flush=True)
        return

    tag = (
        f"{model_safe}_{conv_tag}_k{args.k}_{args.eval_mode}"
        f"_expand-{args.expand_links}_run{args.runs}{args.tag_suffix}"
    )
    trace_path = OUT / f"{tag}.llm-trace.jsonl"
    memory_path = OUT / f"{tag}.memory.jsonl"

    total_t0 = time.perf_counter()
    runs_out = []
    for run in range(1, args.runs + 1):
        t0 = time.perf_counter()
        # trace/memory snapshots capture the full state; multiple --runs append to
        # the shared trace (each line is timestamped) and overwrite the snapshot.
        combined = eval_conversations(
            args, embedder, roles, conv_indices, trace_path=trace_path, memory_path=memory_path
        )
        combined["run_seconds"] = round(time.perf_counter() - t0, 1)
        combined["run"] = run
        runs_out.append(combined)
        print(
            f"[run {run}/{args.runs}] {conv_tag} overall={combined['overall']}",
            flush=True,
        )
    total_s = round(time.perf_counter() - total_t0, 1)
    utc_finished = datetime.now(timezone.utc).isoformat()

    # mean±std across runs (overall F1/BLEU-1)
    run_summary = {}
    if args.runs > 1:
        f1s = [r["overall"].get("f1", 0.0) for r in runs_out]
        b1s = [r["overall"].get("bleu1", 0.0) for r in runs_out]
        run_summary = {
            "f1_mean": round(statistics.mean(f1s), 2),
            "f1_std": round(statistics.stdev(f1s), 2) if len(f1s) > 1 else 0.0,
            "bleu1_mean": round(statistics.mean(b1s), 2),
            "bleu1_std": round(statistics.stdev(b1s), 2) if len(b1s) > 1 else 0.0,
        }

    first = runs_out[0]
    # Top-level budget/cost reflect what was ACTUALLY paid across ALL runs (each
    # run re-issues the answer calls), not just run 1 — every credit is recorded.
    # Per-run budget + cost also land in each `runs[]` entry so nothing is lost.
    budget = _merge_run_budgets([r.get("llm_budget", {}) for r in runs_out])
    merged_drops: dict = {}
    for r in runs_out:
        for role, n in (r.get("drops") or {}).items():
            merged_drops[role] = merged_drops.get(role, 0) + n
    # durable per-question audit trail (every question, every conv, every run),
    # written next to the summary as an appendable/greppable JSONL sidecar. The
    # summary only POINTS to it via records_file — records are not inlined here.
    records_name = f"{tag}.records.jsonl"
    n_records = write_records_sidecar(OUT / records_name, runs_out)

    result = {
        "stamp": _stamp(args, sha, utc_started, utc_finished, first.get("n_questions")),
        "overall": first["overall"],
        "by_category": first["by_category"],
        "per_conv": first.get("per_conv"),
        "llm_budget": budget,
        "cost_usd": cost_usd(budget),
        "drops": merged_drops,
        "timing": {
            "ingest_s": first.get("ingest_s"),
            "eval_s": first.get("eval_s"),
            "total_s": total_s,
        },
        "memory_capacity": first.get("memory_capacity"),
        "run_summary": run_summary,
        "runs": [
            {
                "run": r["run"],
                "overall": r["overall"],
                "run_seconds": r["run_seconds"],
                "llm_budget": r.get("llm_budget", {}),
                "cost_usd": cost_usd(r.get("llm_budget", {})),
            }
            for r in runs_out
        ],
        # five-artifact pointers (docs/14 §Artifacts): summary + records are
        # git-committed; the heavy trace + memory snapshot stay durable on disk.
        "records_file": records_name,
        "llm_trace_file": trace_path.name,
        "memory_file": memory_path.name,
    }

    out_path = OUT / f"{tag}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[done] wrote {out_path}", flush=True)
    print(f"[done] wrote {OUT / records_name} ({n_records} question records)", flush=True)
    print(f"[done] wrote {trace_path} (full LLM I/O trace)", flush=True)
    print(f"[done] wrote {memory_path} (memory snapshot)", flush=True)
    print(f"[cost] ~${result['cost_usd']} (gpt-4o-mini rates in stamp)", flush=True)
    if run_summary:
        print(f"[mean±std] {run_summary}", flush=True)


def _stamp(args, sha: str | None, utc_started: str, utc_finished: str, n_questions) -> dict:
    """Self-describing config stamp for the summary JSON: model/embedder/k/
    eval-mode/temps/expand + provenance (git_sha, dataset_path, timestamps) +
    the cost rates so cost_usd is reproducible, and the cat5 seed note."""
    return {
        "model": args.model,
        "endpoint": args.endpoint,
        "embedder": args.embedder,
        "k": args.k,
        "eval_mode": args.eval_mode,
        "temps": {"write": 0.7, "generate": 0.7, "cat5": CAT5_TEMPERATURE},
        "expand_links": args.expand_links,
        "conv": args.conv,
        "runs": args.runs,
        "workers": args.workers,
        "n_questions": n_questions,
        "memory_types": list(MEMORY_TYPES),
        "keyword_queries": True,
        "eval_only": args.eval_only,
        "dataset": "locomo10",
        "dataset_path": str(DATA),
        "git_sha": sha,
        "utc_started": utc_started,
        "utc_finished": utc_finished,
        "cost_rates": COST_RATES,
        "cat5_seed": "md5(question)&1 — deterministic MCQ option order (locomo.cat5_options)",
    }


if __name__ == "__main__":
    main()
