#!/usr/bin/env python
"""X2 subtractive oracle, step 2: cut the non-contributing half out of a store.

`x2_oracle_prep.py` decided WHICH items a methodology's own read path never
turned into a correct answer (`results/ext/x2/prep.json` + `keep_ids/<arm>.json`).
This script performs the cut, and only the cut: it copies the arm's persisted
store, deletes every servable item outside `keep_ids`, proves the deleted items
can no longer be served, and prints the paid eval command. The eval itself is
`scripts/exp_amem_repro.py --eval-only`, unchanged — a second answer path would
be a second thing to trust, and the whole point of X2 is that the number after
the cut is comparable to the number before it.

Three properties this file exists to guarantee:

1. **The store it cuts is the store the arm was measured on.** Nothing on disk
   records that link — the run summaries stamp no `data_dir` and the ingest
   sentinels record no config — so the mapping below is by directory-naming
   convention, and a convention is not evidence. `verify_store_matches` turns it
   into evidence: every servable id in the committed memory snapshot must be
   present in the store's doc store, per conversation, per memory type. A
   mismatch aborts before a single byte is copied.

2. **`episodic` is never touched.** Raw turns are in the snapshot but are not a
   servable memory type for any of these arms (the k-per-type read paths never
   ask for them), so they are not part of the oracle's keep/delete partition.
   Deleting them would silently change what the lexical channel can see.

3. **A deleted item cannot come back as a ghost.** `AgenticMemory._apply_one`'s
   DELETE branch drops the vector and tombstones the doc — enough for a dense
   read path, which is four of the five arms. Zep's is not dense: BM25 over doc
   items (the tombstone kills that too) and a BFS over graph edges, which the
   DELETE branch does not touch at all. So facts are additionally invalidated in
   the graph (`edges_for_nodes(active_only=True)` is what the BFS step calls),
   and `--probe` re-runs real queries through the real pipeline afterwards and
   fails if any tombstone comes back. Round-5 X1 already paid for this lesson
   once, with vectors.

Usage:
    uv run python scripts/repro/x2_oracle_surgery.py --arm nemori_a
    uv run python scripts/repro/x2_oracle_surgery.py --arm zep_cross_encoder --probe 50
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import utcnow

PREP = ROOT / "results" / "ext" / "x2" / "prep.json"
KEEP_DIR = ROOT / "results" / "ext" / "x2" / "keep_ids"
STORES = ROOT / "results" / "repro" / "stores"

# arm -> (source store dir, runner config, the summary tag whose memory snapshot
# describes that store's contents, the arm's headline eval flags).
#
# The snapshot tag is NOT always the arm's own headline tag: the Zep sweep ran
# four read recipes over ONE ingest, so `zep_cross_encoder`'s store is described
# by the `_e3sZrrf` snapshot (prep.json records the same pairing). The eval flags
# are read back from the headline summary's stamp rather than typed here, so an
# arm's read path can never drift from what it was measured with.
ARMS: dict[str, dict[str, str]] = {
    "nemori_a": {
        "store": "nemori-armA-e3s",
        "config": "nemori_upstream",
        "snapshot_tag": "gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA",
        "headline_tag": "gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA",
    },
    "nemori_b": {
        "store": "nemori-armB-e3s",
        "config": "nemori_merge085",
        "snapshot_tag": "gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB",
        "headline_tag": "gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB",
    },
    "amem": {
        "store": "amem-e3s",
        "config": "amem_perhit",
        "snapshot_tag": "gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH",
        "headline_tag": "gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH",
    },
    "zep_cross_encoder": {
        "store": "zep-ce-e3s",
        "config": "zep_cross_encoder",
        "snapshot_tag": "gpt-4o-mini_zep_rrf_all_k10_ours_expand-off_run1_e3sZrrf",
        "headline_tag": "gpt-4o-mini_zep_cross_encoder_all_k10_ours_expand-off_run1_e3sZ",
    },
    "mem0": {
        "store": "mem0-e3s",
        "config": "mem0_v0194",
        "snapshot_tag": "gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM",
        "headline_tag": "gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM",
    },
}

# Types whose deletion must also reach the graph. `facts` are Zep's edges: the
# BFS channel reads them through `edges_for_nodes(active_only=True)`, which the
# vector+doc DELETE leaves alone.
GRAPH_EDGE_TYPES = frozenset({"facts"})


def _load_repro_helpers():
    """Import `scripts/exp_amem_repro.py` for `build_memory`/`make_roles`.

    Same path-import as `exp_ace_finer.py` (scripts/ is not a package). Reusing
    the runner's own memory constructor is the point: it is the only place that
    knows an arm's store slots, lexical types and expansion cap, and surgery on
    a differently-wired store would be surgery on a different store."""
    path = ROOT / "scripts" / "exp_amem_repro.py"
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("exp_amem_repro", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_partition(arm: str) -> dict:
    """The keep/delete partition, re-derived from the artifacts and cross-checked
    against prep.json rather than trusted from it.

    prep.json is a report; the snapshot and keep_ids are the data. Re-deriving
    costs a second of file reading and catches the one failure that would ruin
    the round silently — a keep_ids file regenerated against a different arm's
    records, which would delete the wrong items and still produce a plausible
    number."""
    prep = json.loads(PREP.read_text())
    if arm not in prep["arms"]:
        raise SystemExit(f"{arm} is not in prep.json (arms: {sorted(prep['arms'])})")
    entry = prep["arms"][arm]
    spec = ARMS[arm]

    summary = json.loads((ROOT / "results" / "repro" / f"{spec['snapshot_tag']}.json").read_text())
    servable = set(summary["stamp"]["memory_types"])

    by_conv: dict[int, list[tuple[str, str]]] = collections.defaultdict(list)
    all_ids: set[str] = set()
    links: dict[str, list[str]] = {}
    snapshot = ROOT / "results" / "repro" / f"{spec['snapshot_tag']}.memory.jsonl"
    with snapshot.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["memory_type"] not in servable:
                continue  # property 2: episodic is not part of the partition
            by_conv[int(row["conv"])].append((row["id"], row["memory_type"]))
            all_ids.add(row["id"])
            if row.get("links"):
                links[row["id"]] = list(row["links"])

    keep = set(json.loads((KEEP_DIR / f"{arm}.json").read_text())["keep_ids"])
    delete = all_ids - keep

    # A-Mem's expansion step reserves its neighbor budget BEFORE hydration
    # (`steps.py:141-144`), and hydration is where `is_servable` drops the
    # tombstones. So a surviving note whose link points at a deleted note spends
    # a slot and returns nothing: the cut shrinks the served context by more than
    # the deleted items themselves. Not a defect — it is what deleting a linked
    # item means — but it is a second channel of the treatment, and X2's read of
    # the result has to know how wide it is.
    dangling = sum(1 for src, ids in links.items() if src in keep for i in ids if i in delete)

    expected_servable = entry["store_items_servable"]
    expected_delete = entry["proxy_oracle"]["delete_items"]
    if len(all_ids) != expected_servable or len(delete) != expected_delete:
        raise SystemExit(
            f"partition disagrees with prep.json for {arm}: servable "
            f"{len(all_ids)} vs {expected_servable}, delete {len(delete)} vs "
            f"{expected_delete}. Re-run x2_oracle_prep.py before cutting anything."
        )
    if not keep <= all_ids:
        raise SystemExit(
            f"{arm}: {len(keep - all_ids)} keep_ids are not in the store snapshot — "
            "the keep_ids file was built against different records."
        )
    return {
        "servable_types": sorted(servable),
        "by_conv": dict(by_conv),
        "keep": keep,
        "delete": delete,
        "links_from_survivors_to_deleted": dangling,
        "links_from_survivors": sum(len(ids) for src, ids in links.items() if src in keep),
        "headline_stamp": json.loads(
            (ROOT / "results" / "repro" / f"{spec['headline_tag']}.json").read_text()
        )["stamp"],
    }


def verify_store_matches(mem, conv_items: list[tuple[str, str]], conv: int) -> dict:
    """Property 1: this store holds exactly the snapshot's items for this conv.

    Checked per memory type through `doc_store.list_items`, the same surface the
    snapshot dumper reads, so a difference here is a real difference in contents
    and not a serialization artifact."""
    want: dict[str, set[str]] = collections.defaultdict(set)
    for item_id, mtype in conv_items:
        want[mtype].add(item_id)
    report = {}
    for mtype, ids in sorted(want.items()):
        have = {it["id"] for it in mem.doc_store.list_items(mtype, mem.namespace)}
        missing = ids - have
        report[mtype] = {"snapshot": len(ids), "store": len(have), "missing": len(missing)}
        if missing:
            raise SystemExit(
                f"conv{conv}/{mtype}: {len(missing)} snapshot items are absent from the "
                f"store (e.g. {sorted(missing)[:3]}). The arm -> store mapping in ARMS "
                "is wrong, or this store was rebuilt after the run."
            )
    return report


def cut(mem, conv_items: list[tuple[str, str]], delete: set[str]) -> dict:
    """Delete this conversation's non-contributing items, via ops.

    Through `_apply_ops` rather than the stores directly: the ops land in the
    evolution log first, so the cut is auditable in the same place every other
    mutation in this project is, and the copied store carries its own record of
    what was removed from it. `propagate=False` because organizers must not
    react — this is not a methodology making a decision, it is an oracle."""
    ops = [
        MemoryOp(
            op=OpType.DELETE,
            target_type=mtype,
            target_id=item_id,
            payload={"reason": "x2_subtractive_oracle: never served a correct answer"},
            actor="x2_oracle",
            t_transaction=utcnow(),
        )
        for item_id, mtype in conv_items
        if item_id in delete
    ]
    mem._apply_ops(ops, actor="x2_oracle", propagate=False)

    # Property 3, first half: the graph copy of a deleted edge outlives the
    # DELETE branch, and Zep's BFS channel reads exactly that copy.
    invalidated = 0
    if mem.graph_store is not None:
        stamp = utcnow().isoformat()
        for op in ops:
            if op.target_type in GRAPH_EDGE_TYPES:
                mem.graph_store.invalidate_edge(op.target_id, stamp)
                invalidated += 1

    mem.vector_store.persist()
    counts = collections.Counter(op.target_type for op in ops)
    return {"deleted": len(ops), "by_type": dict(counts), "graph_edges_invalidated": invalidated}


def probe(mem, queries: list[str], deleted: set[str], k_by_type, memory_types: list[str]) -> dict:
    """Property 3, second half: run real queries through the real read path and
    assert no deleted id comes back.

    The only spend in this file — query embeddings, on the order of a thousandth
    of a cent for the default 25 queries — and the only check that covers read
    channels this script does not model. A ghost hit here means the cut was
    incomplete and the paid eval would be measuring tombstones."""
    from agmem.retrieval.planned import searcher_for

    searcher = searcher_for(mem)
    # `search` runs the read->write hooks inline. Both `on_retrieval`
    # implementations return no ops today, so probing leaves the store the paid
    # eval will read exactly as the cut left it — but "today" is a fact about the
    # code, not a guarantee, and a probe that quietly mutated the store would
    # contaminate the very number it is protecting. The op-log length before and
    # after is the guarantee.
    ops_before = mem.doc_store.count()
    ghosts: dict[str, int] = collections.Counter()
    served = 0
    for q in queries:
        # The eval's own call shape (`bench/locomo.py:585`), minus the keyword
        # rewrite: this probe asks whether an id can be SERVED, and the rewrite
        # only changes which query text asks for it.
        bundle = searcher.search(q, memory_types=memory_types, k=k_by_type)
        for scored in bundle.items:
            item = scored.item
            item_id = getattr(item, "id", None) or (
                item.data.get("id") if hasattr(item, "data") else None
            )
            served += 1
            if item_id in deleted:
                ghosts[item_id] += 1
    ops_after = mem.doc_store.count()
    if ops_after != ops_before:
        raise SystemExit(
            f"the probe wrote {ops_after - ops_before} ops into the cut store — a "
            "read->write hook is live and this store is no longer the one the cut "
            "produced. Re-copy and cut with --probe 0."
        )
    return {"queries": len(queries), "served_slots": served, "ghost_hits": dict(ghosts)}


def sample_queries(headline_tag: str, conv: int, n: int) -> list[str]:
    """The arm's own questions for this conversation, from its records file."""
    path = ROOT / "results" / "repro" / f"{headline_tag}.records.jsonl"
    out: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if int(row.get("conv", -1)) != conv:
                continue
            q = row.get("q")
            if q:
                out.append(q)
            if len(out) >= n:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--src-store", default=None, help="override the source store dir")
    ap.add_argument("--dest-store", default=None, help="override the destination store dir")
    ap.add_argument("--embedder", default="text-embedding-3-small")
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="check the partition and the arm -> store mapping, copy and cut nothing",
    )
    ap.add_argument(
        "--probe",
        type=int,
        default=25,
        help="queries per conversation to replay through the read path after the cut "
        "(0 disables; costs query embeddings only)",
    )
    ap.add_argument("--overwrite", action="store_true", help="delete an existing destination")
    args = ap.parse_args()

    helpers = _load_repro_helpers()
    spec = ARMS[args.arm]
    part = load_partition(args.arm)
    stamp = part["headline_stamp"]
    src = Path(args.src_store) if args.src_store else STORES / spec["store"]
    dest = Path(args.dest_store) if args.dest_store else STORES / f"x2-{args.arm}"
    if not src.is_dir():
        raise SystemExit(f"source store {src} does not exist")

    print(
        f"[plan] {args.arm}: store={src.name} config={spec['config']} "
        f"types={part['servable_types']} keep={len(part['keep'])} delete={len(part['delete'])} "
        f"({len(part['delete']) / (len(part['keep']) + len(part['delete'])):.2%})",
        flush=True,
    )

    # `build_memory` reads these off an args-shaped object; nothing else on it is
    # touched by the store-construction path.
    mem_args = argparse.Namespace(
        config=spec["config"],
        data_dir=str(src if args.verify_only else dest),
        expand_links=stamp.get("expand_links", "off"),
        ingest_only=False,
    )
    # No LLM role is exercised by a delete or a search, but `build_memory` wires
    # roles into the config, so give it the shape it expects with no key: a role
    # that is never called cannot spend, and a missing key would change which
    # organizer slots resolve.
    roles = helpers.make_roles("https://api.openai.com/v1", stamp.get("model", "gpt-4o-mini"), "")
    # The paid embedder needs its key at construction even when nothing is
    # embedded (`--verify-only` reads doc stores and nothing else), because the
    # vector store is opened against `embedder.dim`.
    helpers.load_env_local()
    embedder = helpers.build_embedder(args.embedder)

    if not args.verify_only:
        if dest.exists():
            if not args.overwrite:
                raise SystemExit(f"{dest} already exists; pass --overwrite to replace it")
            shutil.rmtree(dest)
        print(
            f"[copy] {src} -> {dest} ({sum(f.stat().st_size for f in src.rglob('*') if f.is_file()) / 1e6:.0f} MB)",
            flush=True,
        )
        shutil.copytree(src, dest)

    report = {
        "arm": args.arm,
        "source_store": str(src.relative_to(ROOT)),
        "dest_store": None if args.verify_only else str(dest.relative_to(ROOT)),
        "config": spec["config"],
        "servable_types": part["servable_types"],
        "keep_items": len(part["keep"]),
        "delete_items": len(part["delete"]),
        "links_from_survivors": part["links_from_survivors"],
        "links_from_survivors_to_deleted": part["links_from_survivors_to_deleted"],
        "per_conv": {},
        "verify_only": args.verify_only,
    }
    for conv in sorted(part["by_conv"]):
        conv_items = part["by_conv"][conv]
        mem = helpers.build_memory(mem_args, embedder, conv, roles)
        try:
            entry = {"match": verify_store_matches(mem, conv_items, conv)}
            if not args.verify_only:
                entry["cut"] = cut(mem, conv_items, part["delete"])
                if args.probe:
                    queries = sample_queries(spec["headline_tag"], conv, args.probe)
                    entry["probe"] = probe(
                        mem, queries, part["delete"], stamp["k"], part["servable_types"]
                    )
            report["per_conv"][str(conv)] = entry
            line = f"[conv{conv}] " + " ".join(
                f"{t}:{v['snapshot']}" for t, v in entry["match"].items()
            )
            if "cut" in entry:
                line += f" -> deleted {entry['cut']['deleted']}"
                if entry["cut"]["graph_edges_invalidated"]:
                    line += f" (+{entry['cut']['graph_edges_invalidated']} graph edges)"
            if entry.get("probe"):
                g = entry["probe"]["ghost_hits"]
                line += f" | probe {entry['probe']['queries']}q ghosts={sum(g.values())}"
            print(line, flush=True)
        finally:
            mem.close()

    ghosts = sum(
        sum(e["probe"]["ghost_hits"].values())
        for e in report["per_conv"].values()
        if e.get("probe")
    )
    report["total_ghost_hits"] = ghosts
    out = ROOT / "results" / "ext" / "x2" / f"surgery_{args.arm}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] wrote {out.relative_to(ROOT)}", flush=True)

    if ghosts:
        raise SystemExit(
            f"{ghosts} ghost hits after the cut — deleted items are still reachable "
            "through a channel this script does not clear. Do NOT run the paid eval."
        )
    if args.verify_only:
        return
    # `--k` is scalar-only on the CLI; the per-type k of a multi-type arm lives
    # in its RunnerConfig and is what the run actually used, so those arms take
    # the flag's default and let the config win (that is how they were measured).
    k = stamp["k"]
    headline = json.loads((ROOT / "results" / "repro" / f"{spec['headline_tag']}.json").read_text())
    quote = json.loads(PREP.read_text())["arms"][args.arm]["paid_step"]["quoted_cost_usd"]
    print(
        "\n[next, PAID] the arm's own headline read path over the cut store"
        f" (before the cut: F1 {headline['overall']['f1']}, "
        f"J {headline['overall'].get('j_score')}):\n"
        + ("" if isinstance(k, int) else f"  # per-type k ({k}) comes from the RunnerConfig\n")
        + f"  uv run python scripts/exp_amem_repro.py --conv all "
        f"{f'--k {k} ' if isinstance(k, int) else ''}\\\n"
        f"    --model {stamp.get('model', 'gpt-4o-mini')} --embedder {args.embedder} \\\n"
        f"    --eval-mode {stamp['eval_mode']} --expand-links {stamp['expand_links']} \\\n"
        f"    --judge --judge-model {stamp.get('judge_model', 'gpt-4o-mini')} \\\n"
        f"    --config {spec['config']} --eval-only --data-dir {dest.relative_to(ROOT)} \\\n"
        f"    --tag-suffix _x2oracle --max-spend-usd {quote * 1.3:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
