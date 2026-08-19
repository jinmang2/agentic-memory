"""X2 prep — the write-policy oracle's $0 half, extracted from artifacts that already exist.

X2 asks what a learned write policy could have bought: if the store had kept only the items that
ever mattered, how much of each arm's deficit closes? The paid half of that question re-runs
retrieval + answer + judge against a filtered store (arm당 $0.3–1.4, the plan's anchor). This
script is the half that costs nothing: from the committed `*.records.jsonl` (per-question served
context + verdict) and `*.memory.ops.jsonl` / `*.memory.jsonl` (every write decision, and the
store it left behind) it computes, per docs/18 arm,

  * the **contributed set** — items that appeared at least once in the served context of a
    correctly-judged answer, under the arm's own headline read path (and, where the same store was
    read through several measured configs, the union across all of them);
  * the **proxy oracle store** — contributed-only, written as a keep-id list the paid step can
    consume, with the deletion fraction the proxy implies;
  * **lineage presence** — whether ops carry produced-item-id → source-turn provenance
    (`source_episode_ids`), recomputed from the ops rows rather than asserted, because the
    counterfactual upper bound needs it and one arm turns out not to have it;
  * **Mem0 NOOP/discard quality** — the 79%-NOOP question: did the items the write policy kept
    re-confirming turn out to be the useful ones?
  * the **planned paid step** per arm — store, questions, expected call counts, quoted cost —
    emitted as data so the coordinator can fire it without re-deriving anything.

"Contributed" is a *proxy* for "contributed", and the direction of its error is arm-dependent:
for pure-dense read paths (Nemori, Mem0) a correct question's served top-k survives the filtering
exactly (every item it served is contributed by definition, and dense scores are pool-independent),
so the paid re-run reads as a lower bound on the oracle; for pool-dependent read paths (Zep's
BM25/BFS/reranker fusion, A-Mem's link expansion) even that guarantee lapses. The true oracle is
counterfactual and is bracketed, not measured — the bracket is recorded in the plan's §X2 append
(2026-08-19) and in `docs/research/x2-write-oracle.md`.

Fail-closed: every records file's J is recomputed and checked against the docs/18 anchor, every
ops file's op counts against the evolution-log table, every snapshot against its summary's
`memory_capacity`, and every served item id against the snapshot; any mismatch aborts before a
byte is written.

Run:  uv run python scripts/repro/x2_oracle_prep.py
Writes `results/ext/x2/prep.json` and `results/ext/x2/keep_ids/<arm>.json`, nothing else.
Deterministic: same artifacts in, same bytes out. No network, no LLM calls.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPRO = REPO_ROOT / "results" / "repro"
OUT_DIR = REPO_ROOT / "results" / "ext" / "x2"

# The plan's cost anchor for the paid step (docs/_internal/plans/2026-08-07-expansion-layer-design.md
# §1.3): re-running retrieval + answer + judge against a virtual store is arm당 $0.3–1.4, calibrated
# by each arm's own measured eval cost — which is what `paid_step.quoted_cost_usd` carries below.
PLAN_COST_ANCHOR = (
    "arm당 $0.3–1.4 (expansion-layer plan §1.3; per-arm quote = the arm's measured eval cost_usd)"
)

# docs/18 anchors. J is the published headline; ops are the evolution-log table (docs/18 "What each
# write path did"); retrievable is the store column of the same table. The script recomputes all
# three and aborts on mismatch.
ARMS = [
    {
        "key": "nemori_a",
        "label": "Nemori arm A (upstream)",
        "records_stem": "gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA",
        "store_stem": "gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA",
        "ops_stem": "gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA",
        "anchor_j": 67.60,
        "anchor_ops": {
            "ADD:episodic": 5882,
            "ADD:episodes": 555,
            "ADD:semantic": 1926,
            "MERGE:episodes": 223,
            "INVALIDATE:episodes": 223,
        },
        "anchor_retrievable": 2704,
        "union_records": [],
        "read_path_pool_dependence": "none — dense top-k per type, no expansion, no fusion; correct-context preservation argument holds",
    },
    {
        "key": "nemori_b",
        "label": "Nemori arm B (0.85 filter live)",
        "records_stem": "gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB",
        "store_stem": "gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB",
        "ops_stem": "gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB",
        "anchor_j": 65.78,
        "anchor_ops": {
            "ADD:episodic": 5882,
            "ADD:episodes": 763,
            "ADD:semantic": 1745,
            "MERGE:episodes": 22,
            "INVALIDATE:episodes": 22,
        },
        "anchor_retrievable": 2530,
        "union_records": [],
        "read_path_pool_dependence": "none — dense top-k per type; preservation argument holds",
    },
    {
        "key": "amem",
        "label": "A-Mem (kw+perhit headline)",
        "records_stem": "gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH",
        "store_stem": "gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH",
        "ops_stem": "gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH",
        "anchor_j": 61.23,
        "anchor_ops": {
            "ADD:episodic": 5882,
            "ADD:notes": 5882,
            "LINK:notes": 5866,
            "UPDATE:notes": 16342,
        },
        "anchor_retrievable": 5882,
        # The same ingested store read through the other three measured cells of the 2x2
        # (docs/18 "The two read-path changes are not additive").
        "union_records": [
            ("gpt-4o-mini_all_k10_ours_expand-on_run1_e3s", 59.87),
            ("gpt-4o-mini_amem_rawq_all_k10_ours_expand-on_run1_e3sRAWQ", 65.13),
            ("gpt-4o-mini_amem_rawq_perhit_all_k10_ours_expand-on_run1_e3sRQPH", 65.58),
        ],
        "read_path_pool_dependence": "link expansion — expansion set shrinks when linked notes are deleted; base dense hits preserved, expansions not guaranteed",
    },
    {
        "key": "zep_cross_encoder",
        "label": "Zep cross_encoder (§4.1)",
        "records_stem": "gpt-4o-mini_zep_cross_encoder_all_k10_ours_expand-off_run1_e3sZ",
        # The headline eval's own snapshot predates the communities-roster fix (docs/18 ‡); the rrf
        # sweep summary's snapshot is the same shared ingest, complete — the x6 convention.
        "store_stem": "gpt-4o-mini_zep_rrf_all_k10_ours_expand-off_run1_e3sZrrf",
        "ops_stem": "gpt-4o-mini_zep_cross_encoder_all_k10_ours_expand-off_run1_e3sZ",
        "anchor_j": 42.73,
        "anchor_ops": {
            "ADD:episodic": 5882,
            "ADD:facts": 8778,
            "ADD:entities": 2599,
            "ADD:communities": 1243,
            "UPDATE:facts": 4373,
            "UPDATE:entities": 895,
            "INVALIDATE:facts": 1293,
        },
        "anchor_retrievable": 12620,
        # The measured recipes over the identical store (docs/18 recipe sweep) whose records carry
        # served contexts. mmr is NOT here and not forgotten: its process died before writing a
        # summary and its records.jsonl was reconstructed from the LLM trace, which preserves
        # verdicts and token usage but no retrieval blocks — its served item ids are unrecoverable.
        "union_records": [
            ("gpt-4o-mini_zep_rrf_all_k10_ours_expand-off_run1_e3sZrrf", 41.62),
            ("gpt-4o-mini_zep_edge_rrf_all_k10_ours_expand-off_run1_e3sZerrf", 34.87),
            (
                "gpt-4o-mini_zep_edge_episode_mentions_all_k10_ours_expand-off_run1_e3sZmentions",
                33.05,
            ),
        ],
        "read_path_pool_dependence": "BM25 IDF + BFS + RRF fusion + cross-encoder over the fused candidate set — every channel is pool-dependent; no preservation guarantee",
    },
    {
        "key": "mem0",
        "label": "Mem0 v0.1.94",
        "records_stem": "gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM",
        "store_stem": "gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM",
        "ops_stem": "gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM",
        "anchor_j": 31.82,
        "anchor_ops": {
            "ADD:episodic": 5882,
            "ADD:semantic": 5654,
            "NOOP:semantic": 26209,
            "UPDATE:semantic": 1077,
            "DELETE:semantic": 227,
        },
        "anchor_retrievable": 5427,
        "union_records": [],
        "read_path_pool_dependence": "none — dense top-30 over semantic; preservation argument holds",
    },
]

# MemoryOS has no LoCoMo ops/records artifact set: the only file on disk is
# results/locomo-conv0-memoryos.json, a conv-0 F1/BLEU summary from before the artifact contract
# existed (docs/14 §9b names that era). Nothing to filter, nothing to re-run — enumerated so the
# absence is a recorded fact rather than an omission.
UNAVAILABLE = {
    "memoryos": "no *.memory.ops.jsonl / *.records.jsonl on disk; only results/locomo-conv0-memoryos.json (conv-0 F1/BLEU summary, pre-artifact-contract run)"
}

MEM0_INGEST_SUMMARIES = [f"gpt-4o-mini_mem0_v0194_conv{i}_ingest_e3sM_c{i}.json" for i in range(10)]

# docs/18: 2,945 decision calls returned 33,167 semantic verdicts, 79.0% NOOP.
MEM0_VERDICT_ANCHOR = 33167
MEM0_NOOP_PCT_ANCHOR = 79.0


def die(msg: str) -> None:
    raise SystemExit(f"x2: REFUSING TO WRITE — {msg}")


def load_records(stem: str, anchor_j: float) -> list[dict]:
    """All rows, J recomputed over judged rows and checked against the docs/18 anchor."""
    rows = []
    with (REPRO / f"{stem}.records.jsonl").open() as fh:
        for line in fh:
            rows.append(json.loads(line))
    judged = [r for r in rows if "j" in r]
    n = len(judged)
    j_score = round(sum(bool(r["j"]) for r in judged) / n * 100, 2)
    if n != 1540 or j_score != anchor_j:
        die(
            f"{stem}: recomputed J {j_score} over {n} judged rows vs published {anchor_j} over 1540"
        )
    return rows


def served_ids(rows: list[dict], only_correct: bool) -> Counter:
    """Item-id -> number of served appearances, over judged rows (category 5 has no verdict and
    cannot define contribution)."""
    counts: Counter = Counter()
    for r in rows:
        if "j" not in r:
            continue
        if only_correct and not r["j"]:
            continue
        for hit in r["retrieval"]["retrieved"]:
            counts[hit["id"]] += 1
    return counts


def load_ops(stem: str, anchor: dict[str, int]) -> list[dict]:
    ops = []
    with (REPRO / f"{stem}.memory.ops.jsonl").open() as fh:
        for line in fh:
            ops.append(json.loads(line))
    counts = Counter(f"{o['op']}:{o['target_type']}" for o in ops)
    if dict(counts) != anchor:
        die(f"{stem}: op counts {dict(sorted(counts.items()))} vs docs/18 evolution table {anchor}")
    return ops


def load_store(store_stem: str, anchor_retrievable: int) -> tuple[dict[str, dict], list[str], dict]:
    """Servable rows keyed by id, the served types, and the summary."""
    summary = json.loads((REPRO / f"{store_stem}.json").read_text())
    served_types = summary["stamp"]["memory_types"]
    per_type = summary["memory_capacity"]["per_type"]
    retrievable = sum(per_type[t] for t in served_types)
    if retrievable != anchor_retrievable:
        die(f"{store_stem}: capacity says {retrievable} servable vs docs/18 {anchor_retrievable}")
    items: dict[str, dict] = {}
    with (REPRO / f"{store_stem}.memory.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("memory_type") in served_types:
                items[row["id"]] = row
    if len(items) != retrievable:
        die(f"{store_stem}: snapshot has {len(items)} servable rows vs capacity {retrievable}")
    return items, served_types, summary


def lineage_presence(ops: list[dict], served_types: list[str]) -> dict:
    """Recomputed, not asserted: of the ops that create servable items (ADD/MERGE), how many carry
    source_episode_ids? Communities are born from member entities, so member_ids counts as
    indirect lineage there."""
    per_type: dict[str, dict] = {}
    for o in ops:
        t = o["target_type"]
        if t not in served_types or o["op"] not in ("ADD", "MERGE"):
            continue
        payload = o.get("payload") or {}
        slot = per_type.setdefault(
            t, {"creating_ops": 0, "with_source_episode_ids": 0, "with_member_ids": 0}
        )
        slot["creating_ops"] += 1
        if payload.get("source_episode_ids"):
            slot["with_source_episode_ids"] += 1
        if payload.get("member_ids"):
            slot["with_member_ids"] += 1
    direct = sum(s["with_source_episode_ids"] for s in per_type.values())
    indirect = sum(s["with_member_ids"] for s in per_type.values())
    total = sum(s["creating_ops"] for s in per_type.values())
    if total > 0 and direct + indirect >= total:
        verdict = "present"
    elif direct + indirect == 0:
        verdict = "absent"
    else:
        verdict = "partial"
    return {"verdict": verdict, "per_type": per_type}


def mem0_noop_quality(ops: list[dict], store: dict[str, dict], contributed: set[str]) -> dict:
    """The 79%-NOOP question, answered as far as $0 allows: a NOOP is the policy re-confirming an
    existing item — did the items it kept re-confirming turn out to be the ones that helped?"""
    noop_targets = Counter(o["target_id"] for o in ops if o["op"] == "NOOP")
    added = {o["target_id"] for o in ops if o["op"] == "ADD" and o["target_type"] == "semantic"}
    deleted = {o["target_id"] for o in ops if o["op"] == "DELETE"}
    verdicts = sum(1 for o in ops if o["target_type"] == "semantic")
    noops = sum(noop_targets.values())
    if verdicts != MEM0_VERDICT_ANCHOR:
        die(f"mem0: {verdicts} semantic verdicts vs docs/18 {MEM0_VERDICT_ANCHOR}")
    noop_pct = round(noops / verdicts * 100, 1)
    if noop_pct != MEM0_NOOP_PCT_ANCHOR:
        die(f"mem0: NOOP {noop_pct}% of verdicts vs docs/18 {MEM0_NOOP_PCT_ANCHOR}%")

    in_store = set(store)
    reconfirmed_in_store = {t for t in noop_targets if t in in_store}
    never_noop_in_store = in_store - set(noop_targets)

    def contributed_rate(ids: set[str]) -> float | None:
        return round(len(ids & contributed) / len(ids) * 100, 1) if ids else None

    drops: Counter = Counter()
    for name in MEM0_INGEST_SUMMARIES:
        for reason, n in (json.loads((REPRO / name).read_text()).get("drops") or {}).items():
            drops[reason] += n

    return {
        "semantic_verdicts": verdicts,
        "noop_ops": noops,
        "noop_pct_of_verdicts": noop_pct,
        "distinct_noop_targets": len(noop_targets),
        "noop_targets_surviving_to_final_store": len(reconfirmed_in_store),
        "noop_targets_later_deleted": len(set(noop_targets) & deleted),
        "adds_later_deleted": len(added & deleted),
        # The quality signal: items the policy re-confirmed vs items it added once and never
        # touched again, scored by whether they ever served a correct answer.
        "contributed_rate_pct": {
            "noop_reconfirmed_items": contributed_rate(reconfirmed_in_store),
            "never_reconfirmed_items": contributed_rate(never_noop_in_store),
            "whole_store": contributed_rate(in_store),
        },
        "max_noops_on_one_item": max(noop_targets.values()),
        "discards": dict(sorted(drops.items())),
        "discards_source": "per-conv ingest summaries' `drops` (counted declines that upstream loses uncounted — organizer docstring)",
    }


def build_arm(arm: dict) -> dict:
    rows = load_records(arm["records_stem"], arm["anchor_j"])
    ops = load_ops(arm["ops_stem"], arm["anchor_ops"])
    store, served_types, summary = load_store(arm["store_stem"], arm["anchor_retrievable"])

    served_all = served_ids(rows, only_correct=False)
    contributed = served_ids(rows, only_correct=True)
    missing = [i for i in served_all if i not in store]
    if missing:
        die(f"{arm['key']}: {len(missing)} served ids absent from snapshot (e.g. {missing[0]})")

    union = set(contributed)
    union_sources = [f"{arm['records_stem']}.records.jsonl"]
    for stem, anchor in arm["union_records"]:
        union |= set(served_ids(load_records(stem, anchor), only_correct=True))
        union_sources.append(f"{stem}.records.jsonl")

    keep = sorted(set(contributed))
    keep_path = OUT_DIR / "keep_ids" / f"{arm['key']}.json"
    keep_path.parent.mkdir(parents=True, exist_ok=True)
    keep_path.write_text(
        json.dumps(
            {
                "arm": arm["key"],
                "definition": "items appearing >=1 time in the served context of a judged-correct answer, headline read path",
                "records_source": f"results/repro/{arm['records_stem']}.records.jsonl",
                "store_source": f"results/repro/{arm['store_stem']}.memory.jsonl",
                "keep_ids": keep,
                "keep_ids_union_all_read_configs": sorted(union),
            },
            indent=1,
        )
        + "\n"
    )

    n_store = len(store)
    per_type_kept = Counter(store[i]["memory_type"] for i in keep)
    per_type_all = Counter(v["memory_type"] for v in store.values())

    result = {
        "label": arm["label"],
        "sources": {
            "records": f"results/repro/{arm['records_stem']}.records.jsonl",
            "ops": f"results/repro/{arm['ops_stem']}.memory.ops.jsonl",
            "store": f"results/repro/{arm['store_stem']}.memory.jsonl + .json (memory_capacity, stamp.memory_types)",
            "union_records": union_sources,
        },
        "published_j": arm["anchor_j"],
        "store_items_servable": n_store,
        "op_counts": dict(sorted(Counter(f"{o['op']}:{o['target_type']}" for o in ops).items())),
        "served": {
            "distinct_items_ever_served": len(served_all),
            "distinct_items_in_correct_contexts": len(contributed),
            "served_slots_in_correct_contexts": sum(contributed.values()),
        },
        "proxy_oracle": {
            "keep_items": len(keep),
            "delete_items": n_store - len(keep),
            "deletion_fraction": round(1 - len(keep) / n_store, 4),
            "deletion_fraction_by_type": {
                t: round(1 - per_type_kept.get(t, 0) / n, 4)
                for t, n in sorted(per_type_all.items())
            },
            "keep_ids_file": str(keep_path.relative_to(REPO_ROOT)),
            "union_across_measured_read_configs": {
                "keep_items": len(union),
                "deletion_fraction": round(1 - len(union & set(store)) / n_store, 4),
                "n_read_configs": 1 + len(arm["union_records"]),
            },
        },
        "lineage": lineage_presence(ops, served_types),
        "read_path_pool_dependence": arm["read_path_pool_dependence"],
        "paid_step": {
            "what": "rebuild the store from keep_ids (embeddings reused from the committed snapshot's "
            "embedding_text — no embedding spend for stored items), re-run the arm's own headline "
            "read path + answer + judge over the same 1,986 questions",
            "store": str(keep_path.relative_to(REPO_ROOT)),
            "questions": {"answered": 1986, "judged": 1540},
            "expected_llm_calls": {
                "generate": 1986,
                "judge": 1540,
                "read_side": 1986 if arm["key"] == "amem" else 0,
            },
            "quoted_cost_usd": summary.get("cost_usd"),
            "quote_basis": PLAN_COST_ANCHOR,
            "notes": (
                "cross-encoder reranker needs the GPU and 8,528 s of eval wall clock (docs/18 recipe table)"
                if arm["key"] == "zep_cross_encoder"
                else "read-side keyword-rewrite calls are the arm's own lineage (docs/18) and stay in"
                if arm["key"] == "amem"
                else ""
            ),
        },
    }
    if arm["key"] == "mem0":
        result["noop_quality"] = mem0_noop_quality(ops, store, set(contributed))
    return result


def main() -> None:
    arms = {arm["key"]: build_arm(arm) for arm in ARMS}
    out = {
        "generated_by": "scripts/repro/x2_oracle_prep.py",
        "date": "2026-08-19",
        "spec": "docs/_internal/plans/2026-08-07-expansion-layer-design.md §X2 (+ 2026-08-19 design append)",
        "contributed_definition": (
            "proxy: an item contributed iff it appeared >=1 time in retrieval.retrieved of a row with "
            "j==True (1,540 judged rows; category 5 is answered but unjudged and cannot define "
            "contribution). The true oracle is counterfactual; this proxy brackets it — see the plan append."
        ),
        "arms": arms,
        "unavailable": UNAVAILABLE,
        "paid_round_total_quote_usd": round(
            sum(a["paid_step"]["quoted_cost_usd"] for a in arms.values()), 2
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "prep.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT_DIR / 'prep.json'}")
    for key, a in arms.items():
        po = a["proxy_oracle"]
        print(
            f"  {key:18s} store {a['store_items_servable']:6d}  keep {po['keep_items']:6d}  "
            f"delete {po['deletion_fraction'] * 100:5.1f}%  (union keeps {po['union_across_measured_read_configs']['keep_items']})  "
            f"lineage {a['lineage']['verdict']}"
        )


if __name__ == "__main__":
    main()
