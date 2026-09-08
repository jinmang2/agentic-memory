"""Pre-registration III executed on what is on disk. $0 — reads records only.

The organizer arm never reached 500 rows. Pre-reg III's rule is "the largest
type-complete prefix available at analysis time", so the registered cut is the
set of question types the arm finished, and everything else is reported apart
from it as an unregistered observation.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lme_c4_analysis import TYPES, mcnemar, paired_ci

OUT = Path(__file__).resolve().parent.parent.parent / "results" / "repro"
ORG = "gpt-4o-mini_lme_s_nemori_k50"
PASS = "gpt-4o-mini_lme_s_con_k50_batched"


def load(tag):
    rows = {}
    for line in (OUT / f"{tag}.records.jsonl").open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[str(r["question_id"])] = r
    return rows


org, pas = load(ORG), load(PASS)
unjudged = [q for q, r in org.items() if r.get("label") is None]
common = sorted((set(org) & set(pas)) - set(unjudged))
print(
    f"organizer rows {len(org)} ({len(unjudged)} unjudged) · comparator {len(pas)} · common {len(common)}"
)

# type completeness against the release's own counts
FULL = {
    "single-session-user": 70,
    "single-session-preference": 30,
    "single-session-assistant": 56,
    "multi-session": 133,
    "temporal-reasoning": 133,
    "knowledge-update": 78,
}
have = {}
for q in common:
    have[org[q]["question_type"]] = have.get(org[q]["question_type"], 0) + 1
complete = [t for t, n in have.items() if n == FULL[t]]
print("per type on the common subset:", {t: f"{n}/{FULL[t]}" for t, n in sorted(have.items())})
print("type-complete:", complete)


def block(ids, title):
    types = np.array([org[q]["question_type"] for q in ids])
    a = np.array([bool(org[q]["label"]) for q in ids])
    b = np.array([bool(pas[q]["label"]) for q in ids])

    def task_avg(lab, typ):
        return 100 * float(np.mean([lab[typ == t].mean() for t in TYPES if (typ == t).any()]))

    def overall(lab, typ):
        del typ
        return 100 * float(lab.mean())

    print(f"\n=== {title}  (n={len(ids)}) ===")
    for name, lab in (("organizer(nemori)", a), ("passthrough", b)):
        print(
            f"  {name:>18}  task_avg {task_avg(lab, types):6.2f}  overall {overall(lab, types):6.2f}  "
            + "  ".join(
                f"{t.replace('single-session-', 'ss-')} {100 * lab[types == t].mean():.2f}"
                for t in TYPES
                if (types == t).any()
            )
        )
    ta = paired_ci(task_avg, a, b, types, 10_000, 0)
    ov = paired_ci(overall, a, b, types, 10_000, 0)
    mc = mcnemar(a, b)
    print(
        f"  paired Δ (organizer − passthrough)  task_avg {ta['delta_pp']:+6.2f} [{ta['lo']:+.2f},{ta['hi']:+.2f}]"
        f"   overall {ov['delta_pp']:+6.2f} [{ov['lo']:+.2f},{ov['hi']:+.2f}]"
    )
    print(
        f"  McNemar {mc['a_only']}/{mc['b_only']} discordant {mc['discordant']} p={mc['p_exact']}"
    )

    # read budget: the -12.2% handicap the addendum registered
    pc_o = np.array([org[q]["prompt_chars"] for q in ids], dtype=float)
    pc_p = np.array([pas[q]["prompt_chars"] for q in ids], dtype=float)
    print(
        f"  prompt chars  organizer {pc_o.mean():,.0f}  passthrough {pc_p.mean():,.0f}  "
        f"Δ {100 * (pc_o.mean() / pc_p.mean() - 1):+.1f}%"
    )
    er_o = [org[q].get("evidence_recall_prompt") for q in ids]
    er_ob = [org[q].get("evidence_recall_bundle") for q in ids]
    er_p = [pas[q].get("evidence_recall_bundle") for q in ids]
    for name, vals in (
        ("organizer prompt", er_o),
        ("organizer bundle", er_ob),
        ("passthrough bundle", er_p),
    ):
        v = [x for x in vals if x is not None]
        print(
            f"  evidence_recall {name:>20}: n={len(v):3d} mean={np.mean(v):.4f}"
            if v
            else f"  evidence_recall {name}: none"
        )
    return {"n": len(ids), "ta": ta, "ov": ov, "mcnemar": mc}


reg_ids = [q for q in common if org[q]["question_type"] in complete]
res = {
    "registered_type_complete": block(
        reg_ids, "REGISTERED cut — type-complete only: " + ", ".join(sorted(complete))
    ),
    "exploratory_all_common": block(
        common, "UNREGISTERED — every judged common row, incomplete types included"
    ),
}
(OUT / "lme_organizer_prefix_paired.json").write_text(
    json.dumps(
        {
            "arms": [ORG, PASS],
            "note": "prefix analysis of an arm that never reached 500 rows; see docs/20 Pre-registration III",
            "organizer_rows_on_disk": len(org),
            "unjudged": unjudged,
            "type_counts_common": have,
            "type_complete": sorted(complete),
            **res,
        },
        indent=2,
    )
)
print("\nwrote results/repro/lme_organizer_prefix_paired.json")
