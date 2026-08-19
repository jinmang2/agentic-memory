"""X6 — the compression-distortion curve, measured from artifacts that already exist.

The rate-distortion framing (arXiv:2607.08032) treats a memory system as a lossy code: the write
path compresses a conversation corpus into a store, the read path serves a slice of it, and the
score is what survives. This script turns that framing into measured points using **nothing that
costs money**: every coordinate below is recomputed from a run summary, a records file, a memory
snapshot or an LLM trace that is already on disk, and the script refuses to write anything if any
recomputed headline disagrees with the number `docs/18-locomo-4way.md` / `docs/20-lme-reading.md`
published for that arm.

Axes, and where each comes from:

  compression  — generate-prompt tokens per question: `llm_budget.generate.tokens_in / calls` from
                 the committed eval summary (LoCoMo), or the mean of per-row
                 `usage.generate.tokens_in` from the records file (LongMemEval, whose summaries are
                 per-process and undercount resumed arms). The Zep `mmr` arm died before writing a
                 summary; its tokens come from its own LLM trace, the same source its published
                 cost was summed from. Store-side compression (item count, mean item length,
                 snapshot bytes) is reported per point but is the secondary axis: the served slice
                 is what the reader ever sees.
  distortion   — J (LoCoMo, judged rows only) or overall accuracy (LongMemEval), recomputed from
                 the per-question verdicts, with a percentile-bootstrap 95% CI (10,000 resamples,
                 seed 0 — the convention `scripts/ext/x1_power.py` fixed). The LoCoMo points also
                 carry the coverage decomposition J = (1 - abstention) x accuracy-when-answered,
                 because the campaign's data says compression spends coverage, not accuracy.

Two honesty rules are enforced in code rather than prose:

  * The LoCoMo curve has **no measured uncompressed endpoint**. The 72.90 full-context figure the
    campaign quotes is Mem0's published Table 2 (arXiv:2504.19413), a different harness — it is
    drawn as an external dashed anchor, never as a point with a compression coordinate.
  * A point whose artifact cannot supply a coordinate is emitted with that field null and a named
    reason, not interpolated.

Run:  uv run python scripts/repro/x6_compression_curve.py
Writes `results/ext/x6/curve.json` and `docs/research/assets/x6-compression-curve.svg`, nothing
else. Deterministic: same artifacts in, same bytes out.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_svg import ACCENT, BACKGROUND, GRID, INK, MUTED, text_element, wrap  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
REPRO = REPO_ROOT / "results" / "repro"
OUT_JSON = REPO_ROOT / "results" / "ext" / "x6" / "curve.json"
OUT_SVG = REPO_ROOT / "docs" / "research" / "assets" / "x6-compression-curve.svg"

N_BOOT = 10_000
SEED = 0

# The abstention sentence is the harness's own contract (src/agmem/bench/locomo.py ANSWER_PROMPT):
# the model is told to reply exactly this when the memories do not contain the answer.
ABSTAIN_MARKER = "no information available"

# LoCoMo arms. anchor = the J that docs/18 publishes for the arm; the script aborts on mismatch.
LOCOMO_ARMS = [
    # stem, short label, family, anchor_J
    ("gpt-4o-mini_nemori_upstream_all_k10_ours_expand-off_run1_e3sA", "arm A", "nemori", 67.60),
    ("gpt-4o-mini_nemori_merge085_all_k10_ours_expand-off_run1_e3sB", "arm B", "nemori", 65.78),
    ("gpt-4o-mini_amem_perhit_all_k10_ours_expand-on_run1_e3sPH", "kw+perhit", "amem", 61.23),
    ("gpt-4o-mini_all_k10_ours_expand-on_run1_e3s", "kw+global5", "amem", 59.87),
    ("gpt-4o-mini_amem_rawq_all_k10_ours_expand-on_run1_e3sRAWQ", "rawq+global5", "amem", 65.13),
    (
        "gpt-4o-mini_amem_rawq_perhit_all_k10_ours_expand-on_run1_e3sRQPH",
        "rawq+perhit",
        "amem",
        65.58,
    ),
    (
        "gpt-4o-mini_zep_cross_encoder_all_k10_ours_expand-off_run1_e3sZ",
        "cross_encoder",
        "zep",
        42.73,
    ),
    ("gpt-4o-mini_zep_rrf_all_k10_ours_expand-off_run1_e3sZrrf", "rrf", "zep", 41.62),
    ("gpt-4o-mini_zep_mmr_all_k10_ours_expand-off_run1_e3sZmmr", "mmr", "zep", 40.78),
    ("gpt-4o-mini_zep_edge_rrf_all_k10_ours_expand-off_run1_e3sZerrf", "edge_rrf", "zep", 34.87),
    (
        "gpt-4o-mini_zep_edge_episode_mentions_all_k10_ours_expand-off_run1_e3sZmentions",
        "edge_mentions",
        "zep",
        33.05,
    ),
    ("gpt-4o-mini_mem0_v0194_all_k10_ours_expand-off_run1_e3sM", "Mem0 v0.1.94", "mem0", 31.82),
]
# The three-subgraph Zep evals share one ingest, and the headline eval's snapshot predates the
# communities-roster fix (docs/18 ‡) while mmr never wrote a summary at all — both cite the rrf
# sweep summary's complete snapshot instead, disclosed per point in store_source.
ZEP_SHARED_STORE_STEM = "gpt-4o-mini_zep_rrf_all_k10_ours_expand-off_run1_e3sZrrf"
MMR_STEM = "gpt-4o-mini_zep_mmr_all_k10_ours_expand-off_run1_e3sZmmr"
ZEP_HEADLINE_STEM = "gpt-4o-mini_zep_cross_encoder_all_k10_ours_expand-off_run1_e3sZ"

# The external anchor for the LoCoMo panel. Not ours, not a point: Mem0's paper measured LoCoMo
# full-context at 72.90 overall J (arXiv:2504.19413 Table 2, verified against the original
# 2026-07-23). Our harness has never run a LoCoMo full-context arm — that absence is the X5 gap.
LOCOMO_EXTERNAL_FULL_CONTEXT = {
    "j": 72.90,
    "source": "Mem0 paper, arXiv:2504.19413 Table 2 (Full-context row), their harness and judge",
    "measured_here": False,
}

# LongMemEval arms — all gpt-4o-mini x chain-of-note, so the only things moving along this curve
# are the corpus the retriever faced and whether anything was retrieved at all.
LME_ARMS = [
    # tag, label, corpus, read, anchor_overall, embed regime
    ("gpt-4o-mini_lme_oracle_con", "oracle full", "oracle", "full", 83.60, None),
    ("gpt-4o-mini_lme_oracle_con_k10", "oracle top-10", "oracle", "k10", 80.60, "per-turn"),
    ("gpt-4o-mini_lme_s_con", "_s full", "s", "full", 60.40, None),
    ("gpt-4o-mini_lme_s_con_k50", "_s top-50", "s", "k50", 81.60, "per-turn"),
    ("gpt-4o-mini_lme_s_con_k50_batched", "_s top-50 batched", "s", "k50", 81.40, "batched-128"),
    ("gpt-4o-mini_lme_m_con_k50_batched", "_m top-50 batched", "m", "k50", 72.80, "batched-128"),
]


def bootstrap_ci(verdicts: list[bool], rng: np.random.Generator) -> tuple[float, float]:
    arr = np.array(verdicts, dtype=np.float64)
    idx = rng.integers(0, len(arr), size=(N_BOOT, len(arr)))
    means = arr[idx].mean(axis=1) * 100.0
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def die(msg: str) -> None:
    raise SystemExit(f"x6: REFUSING TO WRITE — {msg}")


def locomo_point(
    stem: str, label: str, family: str, anchor: float, rng: np.random.Generator
) -> dict:
    records_path = REPRO / f"{stem}.records.jsonl"
    verdicts: list[bool] = []
    abstained = 0
    with records_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if "j" not in row:  # adversarial rows are unjudged by protocol (docs/18, ledger C-1)
                continue
            verdicts.append(bool(row["j"]))
            if ABSTAIN_MARKER in (row.get("pred") or "").strip().lower():
                abstained += 1
    n = len(verdicts)
    j_score = round(sum(verdicts) / n * 100, 2)
    if n != 1540 or j_score != anchor:
        die(f"{stem}: recomputed J {j_score} over {n} rows vs published {anchor} over 1540")

    # Served-prompt tokens. Every arm's own summary carries its generate budget except mmr, whose
    # process died before writing one — its trace is the artifact its published cost came from.
    if stem == MMR_STEM:
        tokens_in = calls = 0
        with (REPRO / f"{stem}.llm-trace.jsonl").open() as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("budget_key") == "generate" and not row.get("error"):
                    tokens_in += int(row["tokens_in"])
                    calls += 1
        tokens_source = f"{stem}.llm-trace.jsonl (generate rows)"
        summary = json.loads((REPRO / f"{stem}.reconstructed.json").read_text())
    else:
        summary_name = f"{stem}.json"
        summary = json.loads((REPRO / summary_name).read_text())
        gen = summary["llm_budget"]["generate"]
        tokens_in, calls = gen["tokens_in"], gen["calls"]
        tokens_source = f"{summary_name} llm_budget.generate"
    if calls != 1986:
        die(f"{stem}: expected 1986 generate calls, saw {calls}")
    tok_per_q = round(tokens_in / calls, 1)

    # Store shape, from the snapshot of the store this arm actually read.
    store_stem = ZEP_SHARED_STORE_STEM if stem in (MMR_STEM, ZEP_HEADLINE_STEM) else stem
    store_summary_name = f"{store_stem}.json"
    store_summary = json.loads((REPRO / store_summary_name).read_text())
    per_type = store_summary["memory_capacity"]["per_type"]
    served_types = store_summary["stamp"]["memory_types"]
    retrievable = sum(per_type[t] for t in served_types)
    unit_chars_total = unit_count = 0
    with (REPRO / f"{store_stem}.memory.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("memory_type") in served_types:
                unit_chars_total += len(row.get("content") or "")
                unit_count += 1
    if unit_count != retrievable:
        die(f"{store_stem}: snapshot has {unit_count} servable rows vs capacity {retrievable}")

    lo, hi = bootstrap_ci(verdicts, rng)
    abstention = abstained / n
    return {
        "bench": "locomo",
        "arm": label,
        "family": family,
        "stem": stem,
        "n_judged": n,
        "J": j_score,
        "J_ci95": [round(lo, 2), round(hi, 2)],
        "abstention_pct": round(abstention * 100, 1),
        # docs/18's identity: J = (1 - abstention) x accuracy-when-answered, so the accuracy term
        # is J/(1 - abstention) — the handful of verdicts that are correct *while* abstaining stay
        # in the numerator, exactly as the published decomposition counted them.
        "acc_when_answered_pct": round(j_score / (1 - abstention), 1),
        "prompt_tokens_per_q": tok_per_q,
        "prompt_tokens_source": tokens_source,
        "store_items_retrievable": retrievable,
        "store_unit_mean_chars": round(unit_chars_total / unit_count, 1),
        "store_snapshot_bytes": store_summary["memory_capacity"]["memory_jsonl_bytes"],
        "store_source": f"{store_stem}.memory.jsonl + {store_summary_name} (memory_capacity, stamp.memory_types)",
        "records": f"results/repro/{stem}.records.jsonl",
        "eval_cost_usd": summary.get("cost_usd"),
    }


def lme_point(
    tag: str,
    label: str,
    corpus: str,
    read: str,
    anchor: float,
    embed: str | None,
    rng: np.random.Generator,
) -> dict:
    records_path = REPRO / f"{tag}.records.jsonl"
    verdicts: list[bool] = []
    tokens = []
    with records_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            verdicts.append(bool(row["label"]))
            tokens.append(row["usage"]["generate"]["tokens_in"])
    n = len(verdicts)
    overall = round(sum(verdicts) / n * 100, 2)
    if n != 500 or overall != anchor:
        die(f"{tag}: recomputed overall {overall} over {n} rows vs published {anchor} over 500")
    lo, hi = bootstrap_ci(verdicts, rng)
    summary = json.loads((REPRO / f"{tag}.json").read_text())
    return {
        "bench": "lme",
        "arm": label,
        "corpus": corpus,
        "read": read,
        "embed_regime": embed,
        "tag": tag,
        "n": n,
        "overall": overall,
        "overall_ci95": [round(lo, 2), round(hi, 2)],
        "task_averaged": summary["aggregate"]["task_averaged"],
        "prompt_tokens_per_q": round(sum(tokens) / n, 1),
        "prompt_tokens_source": f"{tag}.records.jsonl (mean usage.generate.tokens_in; summaries are per-process and undercount resumed arms)",
        "records": f"results/repro/{tag}.records.jsonl",
        "eval_cost_usd": summary.get("cost_usd"),
    }


# ---------------------------------------------------------------------------- figure


def log_x(tok: float, lo: float, hi: float, x0: float, x1: float) -> float:
    return x0 + (math.log10(tok) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)


FAMILY_COLORS = {"nemori": "#15803d", "amem": "#1d4ed8", "zep": "#7c3aed", "mem0": "#b45309"}
FAMILY_NAMES = {"nemori": "Nemori", "amem": "A-Mem", "zep": "Zep", "mem0": "Mem0"}


def panel_locomo(points: list[dict]) -> list[str]:
    X0, X1, Y0, Y1 = 70.0, 560.0, 360.0, 60.0
    T_LO, T_HI, J_LO, J_HI = 380.0, 5200.0, 25.0, 77.0

    def xf(tok: float) -> float:
        return log_x(tok, T_LO, T_HI, X0, X1)

    def yf(j: float) -> float:
        return Y0 + (j - J_LO) / (J_HI - J_LO) * (Y1 - Y0)

    el: list[str] = []
    el.append(
        text_element(
            X0,
            34,
            "LoCoMo — five write paths, twelve read operating points",
            size=14,
            weight="bold",
        )
    )
    el.append(
        text_element(
            X0,
            50,
            "J (judged 1,540) vs generate-prompt tokens per question, log x",
            size=11,
            fill=MUTED,
        )
    )
    for j in (30, 40, 50, 60, 70):
        y = yf(j)
        el.append(
            f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        el.append(text_element(X0 - 8, y + 4, str(j), size=10, fill=MUTED, anchor="end"))
    for tok in (500, 1000, 2000, 4000):
        x = xf(tok)
        el.append(
            f'<line x1="{x:.1f}" y1="{Y0}" x2="{x:.1f}" y2="{Y0 + 5}" stroke="{MUTED}" stroke-width="1"/>'
        )
        el.append(text_element(x, Y0 + 18, f"{tok:,}", size=10, fill=MUTED, anchor="middle"))
    el.append(
        text_element(
            (X0 + X1) / 2,
            Y0 + 34,
            "prompt tokens / question (log)",
            size=11,
            fill=MUTED,
            anchor="middle",
        )
    )

    # External anchor: published, not measured here, so it has a J and no compression coordinate.
    y_ext = yf(LOCOMO_EXTERNAL_FULL_CONTEXT["j"])
    el.append(
        f'<line x1="{X0}" y1="{y_ext:.1f}" x2="{X1}" y2="{y_ext:.1f}" stroke="{ACCENT}" '
        'stroke-width="1.2" stroke-dasharray="6 4"/>'
    )
    el.append(
        text_element(
            X1,
            y_ext - 5,
            "full-context 72.90 — Mem0 paper Table 2 (external; never measured in this harness)",
            size=9.5,
            fill=ACCENT,
            anchor="end",
        )
    )

    # Family legend, in the empty upper-left quadrant.
    for i, fam in enumerate(("nemori", "amem", "zep", "mem0")):
        y = 100 + 13 * i
        el.append(f'<circle cx="{X0 + 14}" cy="{y - 3.5:.1f}" r="4" fill="{FAMILY_COLORS[fam]}"/>')
        el.append(text_element(X0 + 24, y, FAMILY_NAMES[fam], size=10, fill=INK))

    # Within-store polylines: the first-class evidence, one store with only the read path moved.
    def poly(labels: list[str], color: str) -> None:
        pts = [p for lbl in labels for p in points if p["arm"] == lbl]
        d = " ".join(f"{xf(p['prompt_tokens_per_q']):.1f},{yf(p['J']):.1f}" for p in pts)
        el.append(
            f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="1" stroke-dasharray="2 3" opacity="0.7"/>'
        )

    poly(["kw+global5", "kw+perhit"], FAMILY_COLORS["amem"])
    poly(["rawq+global5", "rawq+perhit"], FAMILY_COLORS["amem"])
    poly(["edge_mentions", "edge_rrf", "rrf", "mmr", "cross_encoder"], FAMILY_COLORS["zep"])

    # (dx, dy, anchor) per point, chosen against the measured coordinates so nothing collides.
    offsets = {
        "arm A": (6, -6, "start"),
        "arm B": (6, 12, "start"),
        "kw+perhit": (6, 12, "start"),
        "kw+global5": (-6, 12, "end"),
        "rawq+global5": (-6, -4, "end"),
        "rawq+perhit": (-6, 14, "end"),
        "cross_encoder": (6, -6, "start"),
        "rrf": (-6, 8, "end"),
        "mmr": (6, 12, "start"),
        "edge_rrf": (6, -6, "start"),
        "edge_mentions": (6, 14, "start"),
        "Mem0 v0.1.94": (6, -2, "start"),
    }
    for p in points:
        x, y = xf(p["prompt_tokens_per_q"]), yf(p["J"])
        lo, hi = p["J_ci95"]
        color = FAMILY_COLORS[p["family"]]
        el.append(
            f'<line x1="{x:.1f}" y1="{yf(lo):.1f}" x2="{x:.1f}" y2="{yf(hi):.1f}" stroke="{color}" stroke-width="1.2" opacity="0.55"/>'
        )
        el.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        dx, dy, anchor = offsets[p["arm"]]
        label = f"{p['arm']} {p['J']:.1f} · ab {p['abstention_pct']:.0f}%"
        el.append(text_element(x + dx, y + dy, label, size=9.5, fill=INK, anchor=anchor))
    return el


def panel_lme(points: list[dict]) -> list[str]:
    X0, X1, Y0, Y1 = 660.0, 1150.0, 360.0, 60.0
    T_LO, T_HI, A_LO, A_HI = 2000.0, 300000.0, 55.0, 90.0

    def xf(tok: float) -> float:
        return log_x(tok, T_LO, T_HI, X0, X1)

    def yf(a: float) -> float:
        return Y0 + (a - A_LO) / (A_HI - A_LO) * (Y1 - Y0)

    el: list[str] = []
    el.append(
        text_element(X0, 34, "LongMemEval — one reader, three corpus sizes", size=14, weight="bold")
    )
    el.append(
        text_element(
            X0,
            50,
            "overall (n=500), gpt-4o-mini x chain-of-note; squares full context, circles retrieval",
            size=11,
            fill=MUTED,
        )
    )
    for a in (60, 70, 80, 90):
        y = yf(a)
        el.append(
            f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        el.append(text_element(X0 - 8, y + 4, str(a), size=10, fill=MUTED, anchor="end"))
    for tok, lbl in (
        (3000, "3K"),
        (10000, "10K"),
        (30000, "30K"),
        (100000, "100K"),
        (300000, "300K"),
    ):
        x = xf(tok)
        el.append(
            f'<line x1="{x:.1f}" y1="{Y0}" x2="{x:.1f}" y2="{Y0 + 5}" stroke="{MUTED}" stroke-width="1"/>'
        )
        el.append(text_element(x, Y0 + 18, lbl, size=10, fill=MUTED, anchor="middle"))
    el.append(
        text_element(
            (X0 + X1) / 2,
            Y0 + 34,
            "prompt tokens / question (log)",
            size=11,
            fill=MUTED,
            anchor="middle",
        )
    )

    by_arm = {p["arm"]: p for p in points}
    corpus_color = {"oracle": "#0f766e", "s": "#1d4ed8", "m": "#b45309"}

    def arrow(a: str, b: str, color: str) -> None:
        pa, pb = by_arm[a], by_arm[b]
        x1, y1 = xf(pa["prompt_tokens_per_q"]), yf(pa["overall"])
        x2, y2 = xf(pb["prompt_tokens_per_q"]), yf(pb["overall"])
        el.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="1" stroke-dasharray="2 3" opacity="0.8"/>'
        )

    arrow("oracle full", "oracle top-10", corpus_color["oracle"])
    arrow("_s full", "_s top-50", corpus_color["s"])

    offsets = {
        "oracle full": (6, -8, "start"),
        "oracle top-10": (6, 16, "start"),
        "_s full": (6, -8, "start"),
        "_s top-50": (6, -8, "start"),
        "_s top-50 batched": (6, 18, "start"),
        "_m top-50 batched": (6, 4, "start"),
    }
    for p in points:
        x, y = xf(p["prompt_tokens_per_q"]), yf(p["overall"])
        color = corpus_color[p["corpus"]]
        lo, hi = p["overall_ci95"]
        el.append(
            f'<line x1="{x:.1f}" y1="{yf(lo):.1f}" x2="{x:.1f}" y2="{yf(hi):.1f}" stroke="{color}" stroke-width="1.2" opacity="0.55"/>'
        )
        if p["read"] == "full":
            el.append(
                f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" fill="{color}"/>'
            )
        elif p["embed_regime"] == "batched-128" and p["corpus"] == "s":
            # The batched twin of a per-turn arm: same point twice over, drawn hollow so the pair
            # reads as one measurement's numerical-regime jitter rather than as two findings.
            el.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{BACKGROUND}" stroke="{color}" stroke-width="1.5"/>'
            )
        else:
            el.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        dx, dy, anchor = offsets[p["arm"]]
        el.append(
            text_element(
                x + dx, y + dy, f"{p['arm']} {p['overall']:.1f}", size=9.5, fill=INK, anchor=anchor
            )
        )

    # The _m corpus has no uncompressed endpoint to draw: 1.11M tokens is 8.8x a 128K window.
    el.append(
        text_element(
            X1 - 4,
            yf(66.5),
            "_m full context: no such point exists —",
            size=9.5,
            fill=MUTED,
            anchor="end",
        )
    )
    el.append(
        text_element(
            X1 - 4,
            yf(64.5),
            "1.11M tokens fits no window (docs/20)",
            size=9.5,
            fill=MUTED,
            anchor="end",
        )
    )
    return el


def render_svg(locomo: list[dict], lme: list[dict]) -> str:
    caption = (
        "Left: on LoCoMo no measured uncompressed endpoint exists in this harness (the dashed 72.90 is Mem0's published "
        "full-context, their harness); across arms, more served tokens track more J, and the ab% beside each point says "
        "where compression cuts — coverage, not accuracy. Right: on LongMemEval the sign of distortion against the "
        "uncompressed endpoint flips with corpus size — retrieval costs 3.0 points where the haystack fits (oracle) and "
        "buys 21.2 where it fits badly (_s); at _m the endpoint itself is impossible. Error bars: percentile-bootstrap "
        "95% CI, 10,000 resamples, seed 0. Generated by scripts/repro/x6_compression_curve.py from committed summaries "
        "and disk-durable records; every coordinate is in results/ext/x6/curve.json with its named source."
    )
    width, height = 1200, 500
    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
    )
    lines.append(f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>')
    lines.extend(panel_locomo(locomo))
    lines.extend(panel_lme(lme))
    y = 424
    for row in wrap(caption, 175):
        lines.append(text_element(70, y, row, size=10.5, fill=MUTED))
        y += 14
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    rng = np.random.default_rng(SEED)
    locomo = [locomo_point(*arm, rng) for arm in LOCOMO_ARMS]
    lme = [lme_point(*arm, rng) for arm in LME_ARMS]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "script": "scripts/repro/x6_compression_curve.py",
            "n_boot": N_BOOT,
            "seed": SEED,
            "abstain_marker": ABSTAIN_MARKER,
            "locomo_external_full_context": LOCOMO_EXTERNAL_FULL_CONTEXT,
            "note": "all points recomputed from named artifacts; script aborts if any recomputed headline disagrees with docs/18 or docs/20",
        },
        "locomo": locomo,
        "lme": lme,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    OUT_SVG.write_text(render_svg(locomo, lme) + "\n")
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)} ({len(locomo)} locomo + {len(lme)} lme points)")
    print(f"wrote {OUT_SVG.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
