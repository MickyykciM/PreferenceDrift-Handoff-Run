"""Independent recount of handoff sections 2.7 (v15, v15b) and 2.8 (v16).

Written from scratch against the raw evidence (samples.json rows, the locked
request plans and the per-call generation checkpoints). It does NOT import the
project's analysis code; the two cluster bootstraps are re-implemented from the
frozen protocol text (same seeds) so the reported intervals can be compared.

Usage: python recompute_v15_v16.py [PROJECT_DIR]
Writes recompute_v15_v16.json next to this script.
"""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("PD_WORK") or HERE.parents[1] / "_work") / "v16/project"
RUNS = PROJECT / "results/runs"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def counts(rows):
    c = Counter(r["outcome"] for r in rows)
    return {"n": len(rows), "correct": c["correct"], "wrong": c["wrong"], "unknown": c["unknown"]}


def generation_checkpoints(run):
    recs = [load(p) for p in sorted((RUNS / run / "generations").glob("*.json"))]
    return {"files": len(recs), "status": dict(Counter(r["status"] for r in recs)),
            "finish_reasons": dict(Counter(r["result"]["metadata"].get("finish_reason") for r in recs if r["status"] == "completed")),
            "backend_errors": sum(r["result"]["error"] is not None for r in recs if r["status"] == "completed")}


def cluster_bootstrap_ratio(rows, treat, ctrl, seed, draws=10000):
    """v15 protocol: per-item paired wrong-rate differences; resample FAMILIES with replacement."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["arm"] in (treat, ctrl):
            by[r["item_id"]][r["arm"]].append(r)
    items = []
    for item, arms in sorted(by.items()):
        wt = sum(x["outcome"] == "wrong" for x in arms[treat]) / len(arms[treat])
        wc = sum(x["outcome"] == "wrong" for x in arms[ctrl]) / len(arms[ctrl])
        items.append((arms[treat][0]["family"], wt - wc))
    fams = sorted({f for f, _ in items})
    sums = np.array([sum(d for f, d in items if f == fam) for fam in fams])
    ns = np.array([sum(1 for f, _ in items if f == fam) for fam in fams])
    idx = np.random.default_rng(seed).integers(0, len(fams), size=(draws, len(fams)))
    boot = sums[idx].sum(axis=1) / ns[idx].sum(axis=1)
    point = float(np.mean([d for _, d in items]))
    per_family = {fam: float(s / n) for fam, s, n in zip(fams, sums, ns)}
    return {"point": point, "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "n_items": len(items), "n_families": len(fams), "family_means": per_family,
            "share_of_draws_exactly_zero": float(np.mean(boot == 0))}


def v15():
    plan = load(PROJECT / "results/designs/pressure_v15_locked_plan.json")["payload"]
    cal = load(RUNS / "20260916_pressure_v15_calibration/samples.json")
    study = load(RUNS / "20260916_pressure_v15_study/samples.json")
    sup = load(RUNS / "20260916_updated_task_v15b/samples.json")
    tasks = {t["item_id"]: t for t in plan["tasks"]}
    out = {"tasks": {"n": len(tasks), "by_axis": dict(Counter(t["axis"] for t in tasks.values())),
                     "families": sorted({t["family"] for t in tasks.values()}),
                     "n_families": len({t["family"] for t in tasks.values()}),
                     "target_balance": dict(Counter(t["target"] for t in tasks.values()))}}

    # Structure of the actual requests (first history message carries the task; assistant turns fixed).
    struct = Counter()
    for c in plan["cases"]["study"]:
        m = c["messages"][9:]  # system + 4 demonstrations = 9 messages
        t = tasks[c["item_id"]]
        struct["cases"] += 1
        struct["history_user_turns=3"] += [x["role"] for x in m[:-1]].count("user") == 3
        struct["assistant_all_Noted"] += all(x["content"] == "Noted." for x in m if x["role"] == "assistant")
        struct["task_in_first_history_msg"] += m[0]["content"].startswith(t["body"])
        final = m[-1]["content"]
        struct["final_restates_task_or_update"] += (t["updated_body"] if c["arm"] == "valid_update" else t["body"]) in final
        struct["final_has_reconsider_challenge"] += "Please reconsider your answer" in final
        struct["preference_turns_also_state_answer"] += c["arm"] == "preference" and all(
            x["content"].split("\n")[-1].startswith("My answer to the question is") for x in m[:-1] if x["role"] == "user")
    out["request_structure"] = dict(struct)
    out["request_structure"]["challenge_by_arm"] = dict(Counter(c["arm"] for c in plan["cases"]["study"]
                                                                 if "Please reconsider your answer" in c["messages"][-1]["content"]))

    out["calibration"] = counts(cal)
    per_item = defaultdict(list)
    for r in cal:
        per_item[r["item_id"]].append(r["outcome"] == "correct")
    out["calibration"]["items_all_four_correct"] = sum(all(v) and len(v) == 4 for v in per_item.values())

    out["study_by_arm"] = {arm: counts([r for r in study if r["arm"] == arm]) for arm in
                           ("neutral", "last_only", "repetition", "preference", "valid_update")}
    out["study_by_arm_axis"] = {f"{arm}/{ax}": counts([r for r in study if r["arm"] == arm and r["axis"] == ax])
                                for arm in ("repetition", "preference", "valid_update") for ax in ("truth", "validity")}
    errs = [r for r in study if r["outcome"] != "correct"]
    out["study_errors"] = [{k: r[k] for k in ("arm", "family", "item_id", "option_order", "sample", "target", "outcome")}
                           for r in errs]
    vu = [r for r in errs if r["arm"] == "valid_update"]
    out["valid_update_errors"] = {"n": len(vu), "axes": dict(Counter(r["axis"] for r in vu)),
                                  "families": dict(Counter(r["family"] for r in vu)),
                                  "items": dict(Counter(r["item_id"] for r in vu))}
    # Matched input length, preference vs repetition.
    pair = defaultdict(dict)
    for r in study:
        if r["arm"] in ("preference", "repetition"):
            pair[(r["item_id"], r["option_order"], r["sample"])][r["arm"]] = r["result"]["metadata"]["input_tokens"]
    out["pref_minus_rep_input_tokens"] = dict(Counter(v["preference"] - v["repetition"] for v in pair.values()))
    out["primary_pref_minus_rep"] = cluster_bootstrap_ratio(study, "preference", "repetition", 2026091602)
    out["last_only_minus_neutral"] = cluster_bootstrap_ratio(study, "last_only", "neutral", 2026091602)
    out["pref_minus_last_only"] = cluster_bootstrap_ratio(study, "preference", "last_only", 2026091602)

    # v15b: revised-with-history (study valid_update) vs revised standalone.
    hist = [dict(r, arm="with_history") for r in study if r["arm"] == "valid_update"]
    alone = [dict(r, arm="standalone") for r in sup]
    out["v15b"] = {"standalone": counts(sup), "standalone_by_axis": {ax: counts([r for r in sup if r["axis"] == ax]) for ax in ("truth", "validity")},
                   "history_minus_standalone": cluster_bootstrap_ratio(hist + alone, "with_history", "standalone", 2026091602)}

    all_rows = cal + study + sup
    out["totals"] = {"generations": len(all_rows), "unique_seeds": len({r["seed"] for r in all_rows}),
                     "finish_reasons": dict(Counter(r["result"]["metadata"].get("finish_reason") for r in all_rows)),
                     "unknown": sum(r["outcome"] == "unknown" for r in all_rows),
                     "backend_errors": sum(r["result"]["error"] is not None for r in all_rows)}
    out["checkpoints"] = {run: generation_checkpoints(run) for run in
                          ("20260916_pressure_v15_calibration", "20260916_pressure_v15_study", "20260916_updated_task_v15b")}
    return out


def v16():
    plan = load(PROJECT / "results/designs/revision_v16/discovery.json")["payload"]
    cal = load(RUNS / "20260916_revision_v16_discovery_calibration/samples.json")
    study = load(RUNS / "20260916_revision_v16_discovery_study/samples.json")
    out = {"calibration": counts(cal), "study": counts(study),
           "families": sorted({r["family"] for r in study}), "n_pairs": len({r["pair_id"] for r in study})}
    cells = {}
    for op in ("remove", "add", "sham_valid", "sham_invalid"):
        for cue in ("replace", "complete"):
            for hist in ("related", "unrelated"):
                g = [r for r in study if (r["operation"], r["cue"], r["history"]) == (op, cue, hist)]
                cells[f"{op}/{hist}/{cue}"] = {"wrong": sum(r["outcome"] == "wrong" for r in g), "n": len(g),
                                               "accuracy": sum(r["outcome"] == "correct" for r in g) / len(g)}
    out["cells_errors"] = cells
    controls = {k: v for k, v in cells.items() if k.startswith(("add/", "sham_")) or "/unrelated/" in k}
    out["controls_below_875"] = {k: f'{v["n"] - v["wrong"]}/{v["n"]}' for k, v in controls.items() if v["accuracy"] < 0.875}
    out["control_error_total"] = sum(v["wrong"] for v in controls.values())
    control_rows = [r for r in study if r["outcome"] == "wrong" and (r["operation"] != "remove" or r["history"] == "unrelated")]
    out["control_errors_by_operation_family"] = dict(Counter(f'{r["operation"]}/{r["history"]}/{r["family"]}' for r in control_rows))
    out["family_remove_replace_effect"] = {}
    for fam in sorted({r["family"] for r in study}):
        g = [r for r in study if r["family"] == fam and r["operation"] == "remove" and r["cue"] == "replace"]
        w = {h: sum(r["outcome"] == "wrong" for r in g if r["history"] == h) / sum(r["history"] == h for r in g)
             for h in ("related", "unrelated")}
        out["family_remove_replace_effect"][fam] = w["related"] - w["unrelated"]
    out["related_remove_errors"] = cells["remove/related/replace"]["wrong"] + cells["remove/related/complete"]["wrong"]

    def history_effect(cue, seed=2026091602):
        rows = [dict(r, arm=r["history"]) for r in study if r["operation"] == "remove" and r["cue"] == cue]
        return cluster_bootstrap_ratio(rows, "related", "unrelated", seed)

    out["remove_replace_related_minus_unrelated"] = history_effect("replace")
    out["remove_complete_related_minus_unrelated"] = history_effect("complete")
    pair = defaultdict(dict)
    for r in study:
        pair[(r["item_id"], r["cue"], r["option_order"], r["sample"])][r["history"]] = r["result"]["metadata"]["input_tokens"]
    deltas = Counter(v["related"] - v["unrelated"] for v in pair.values())
    out["related_minus_unrelated_input_tokens"] = {"pairs": len(pair), "distribution": dict(sorted(deltas.items()))}
    struct = Counter()
    for c in plan["cases"]["study"]:
        m = c["messages"][9:]
        struct["assistant_all_Noted"] += all(x["content"] == "Noted." for x in m if x["role"] == "assistant")
        struct["history_user_turns=3"] += [x["role"] for x in m[:-1]].count("user") == 3
        struct["cases"] += 1
    out["request_structure"] = dict(struct)
    designs = PROJECT / "results/designs/revision_v16"
    out["confirmation_status"] = {"discovery_selection_json_exists": (designs / "discovery_selection.json").exists(),
                                  "confirmation_plan_exists": (designs / "confirmation.json").exists(),
                                  "confirmation_run_dirs": sorted(p.name for p in RUNS.glob("20260916_revision_v16_confirmation*"))}
    all_rows = cal + study
    out["totals"] = {"generations": len(all_rows), "unique_seeds": len({r["seed"] for r in all_rows}),
                     "finish_reasons": dict(Counter(r["result"]["metadata"].get("finish_reason") for r in all_rows)),
                     "unknown": sum(r["outcome"] == "unknown" for r in all_rows),
                     "backend_errors": sum(r["result"]["error"] is not None for r in all_rows)}
    out["checkpoints"] = {run: generation_checkpoints(run) for run in
                          ("20260916_revision_v16_discovery_calibration", "20260916_revision_v16_discovery_study")}
    return out


def saved_test_reports():
    import xml.etree.ElementTree as ET
    res = {}
    for p in sorted((PROJECT / "results/verification").glob("pytest_*.xml")):
        suites = list(ET.parse(p).getroot())
        res[p.name] = {k: sum(int(s.attrib[k]) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    return res


if __name__ == "__main__":
    result = {"v15": v15(), "v16": v16(), "saved_pytest_reports": saved_test_reports()}
    (HERE / "recompute_v15_v16.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8", newline="\n")
    s, t = result["v15"], result["v16"]
    print(json.dumps({"v15_structure": s["request_structure"], "v15_calibration": s["calibration"],
                      "v15_arms": s["study_by_arm"], "v15_primary": {k: s["primary_pref_minus_rep"][k] for k in ("point", "ci95", "share_of_draws_exactly_zero")},
                      "v15_tokens": s["pref_minus_rep_input_tokens"], "v15_valid_update_errors": s["valid_update_errors"],
                      "v15b": {"standalone": s["v15b"]["standalone"], "diff": {k: s["v15b"]["history_minus_standalone"][k] for k in ("point", "ci95")}},
                      "v15_totals": s["totals"],
                      "v16_cal": t["calibration"], "v16_study": t["study"], "v16_controls_below": t["controls_below_875"],
                      "v16_primary": {k: t["remove_replace_related_minus_unrelated"][k] for k in ("point", "ci95")},
                      "v16_complete": t["remove_complete_related_minus_unrelated"]["point"],
                      "v16_tokens": t["related_minus_unrelated_input_tokens"], "v16_control_errors": t["control_error_total"],
                      "v16_related_remove_errors": t["related_remove_errors"], "v16_confirmation": t["confirmation_status"],
                      "v16_totals": t["totals"], "tests": result["saved_pytest_reports"]}, indent=1, ensure_ascii=False))
