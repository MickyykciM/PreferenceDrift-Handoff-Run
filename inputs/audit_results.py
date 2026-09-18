"""Offline audit of the supplied archive. No generation, judging, or source edits.

Run with Python 3.10+; outputs computed results to stdout.
"""
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "PreferenceDrift_ClaudeCode_Handoff_2026-09-04" / "project"
RUN = PROJECT / "results/runs/20260704T182542Z_full100"
sys.path.insert(0, str(PROJECT))
from preference_drift.build_dialogues import build_dialogues
from preference_drift.metrics import compute_metrics


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def exact_p(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def kappa(pairs):
    n = len(pairs)
    a, b = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    observed = sum(x == y for x, y in pairs) / n
    expected = sum(a[x] * b[x] for x in a.keys() | b.keys()) / n**2
    return {"n": n, "agreement": observed, "kappa": (observed - expected) / (1 - expected)}


def compare(index, ref, comp, bad, bootstrap=False):
    pairs = [(index[q][ref][-1] in bad, index[q][comp][-1] in bad) for q in sorted(index)]
    ds = [int(b) - int(a) for a, b in pairs]
    b, c = ds.count(-1), ds.count(1)
    result = {"n": len(ds), "ref_error": sum(a for a, _ in pairs),
              "comp_error": sum(b for _, b in pairs), "excess": sum(ds) / len(ds),
              "ref_only": b, "comp_only": c, "exact_mcnemar_p": exact_p(b, c)}
    if bootstrap:
        rng = random.Random(20260906)
        means = sorted(sum(rng.choices(ds, k=len(ds))) / len(ds) for _ in range(20000))
        result["paired_question_bootstrap_percentile_ci95"] = [means[499], means[19499]]
    return result


def main():
    raw_path = RUN / "raw_outputs_gpt-4o-mini.jsonl"
    raw = rows(raw_path)
    raw_by_id = {r["dialogue_id"]: r for r in raw}
    questions = read(PROJECT / "data/questions.json")
    templates = read(PROJECT / "data/dialogue_templates.json")
    built = build_dialogues(questions, templates)
    existing = rows(RUN / "dialogues.jsonl")
    conditions = list(templates["conditions"])
    source_groups = defaultdict(list)
    for q in questions:
        source_groups[q.get("source_row")].append(q["id"])
    report = {
        "scope": "Offline reanalysis of archived labels, not independent factual relabeling.",
        "bootstrap": {"unit": "question; all condition pairs kept together", "draws": 20000,
                      "seed": 20260906, "interval": "percentile; exploratory, fixed curated question set"},
        "raw_integrity": {"rows": len(raw), "unique_dialogue_ids": len(raw_by_id),
                          "turns": sum(len(r["turns"]) for r in raw),
                          "model_errors": sum(bool(r.get("error")) for r in raw),
                          "all_four_turns": all(len(r["turns"]) == 4 for r in raw),
                          "built_dialogues_equal_archived": built == existing,
                          "user_schedule_matches": all([t["user_message"] for t in raw_by_id[d["dialogue_id"]]["turns"]] == d["turns"] for d in built),
                          "duplicate_question_text": [x for x, n in Counter(q["base_question"].strip().lower() for q in questions).items() if n > 1],
                          "duplicate_source_rows": {str(k): v for k, v in source_groups.items() if k is not None and len(v) > 1},
                          "category_counts": dict(Counter(q["category"] for q in questions)),
                          "sha256_raw": hashlib.sha256(raw_path.read_bytes()).hexdigest()},
        "initial_response_identity": {}, "judges": {}}
    for a, b in [(conditions[0], conditions[1]), (conditions[1], conditions[2])]:
        unequal = [q["id"] for q in questions if raw_by_id[q["id"] + "::" + a]["turns"][0]["assistant_response"] != raw_by_id[q["id"] + "::" + b]["turns"][0]["assistant_response"]]
        report["initial_response_identity"][b + "_vs_" + a] = {"different_texts": len(unequal), "n": len(questions)}

    paths = {"gpt-4.1-mini": ("scored_outputs_gpt-4o-mini.jsonl", "summary_metrics_gpt-4o-mini.json"),
             "gpt-4o": ("scored_outputs_gpt-4o-mini_judge-gpt-4o.jsonl", "summary_metrics_gpt-4o-mini_judge-gpt-4o.json")}
    turn_labels = {}
    for judge, (scored_file, metrics_file) in paths.items():
        scored = rows(RUN / scored_file)
        index = defaultdict(dict)
        labels = {}
        table = defaultdict(lambda: defaultdict(Counter))
        for row in scored:
            assert row["dialogue_id"] in raw_by_id
            assert len(row["turns"]) == 4
            assert all(t["assistant_response"] == rt["assistant_response"] and t["user_message"] == rt["user_message"]
                       for t, rt in zip(row["turns"], raw_by_id[row["dialogue_id"]]["turns"]))
            ll = [t["judgment"]["label"] for t in row["turns"]]
            index[row["question_id"]][row["condition"]] = ll
            for i, label in enumerate(ll, 1):
                labels[(row["dialogue_id"], i)] = label
                table[row["condition"]][i][label] += 1
        turn_labels[judge] = labels
        metrics = compute_metrics(scored)
        result = {"rows": len(scored), "unique_ids": len({r["dialogue_id"] for r in scored}),
                  "source_responses_match_raw": True,
                  "computed_metrics_equal_archived": metrics == read(RUN / metrics_file),
                  "turn_label_counts": table, "condition_summary": metrics["condition_summary"],
                  "archived_comparison_keys": list(metrics["paired_comparisons"]),
                  "comparisons": {}, "strict_T1_correct_to_T4_incorrect": {}, "category_effects": {}}
        for bad_name, bad in [("original", {"incorrect", "unclear"}),
                              ("incorrect_only", {"incorrect"}),
                              ("all_noncorrect", {"incorrect", "unclear", "partial"})]:
            result["comparisons"][bad_name] = {
                b + "_vs_" + a: compare(index, a, b, bad, bootstrap=(bad_name == "original"))
                for a, b in [(conditions[1], conditions[2]), (conditions[0], conditions[1]), (conditions[0], conditions[2])]}
        for cond in conditions:
            ll = [v[cond] for v in index.values()]
            denominator = sum(x[0] == "correct" for x in ll)
            numerator = sum(x[0] == "correct" and x[-1] == "incorrect" for x in ll)
            result["strict_T1_correct_to_T4_incorrect"][cond] = {
                "numerator": numerator, "denominator_T1_correct": denominator,
                "conditional_rate": numerator / denominator if denominator else None}
        for cat in sorted({q["category"] for q in questions}):
            sub = {q["id"]: index[q["id"]] for q in questions if q["category"] == cat}
            result["category_effects"][cat] = compare(sub, conditions[1], conditions[2], {"incorrect", "unclear"})
        report["judges"][judge] = result

    mini, primary = turn_labels["gpt-4.1-mini"], turn_labels["gpt-4o"]
    bad = {"incorrect", "unclear"}
    pairs = [(mini[k], primary[k]) for k in mini]
    binary = [(a in bad, b in bad) for a, b in pairs]
    report["inter_judge"] = {"four_label": kappa(pairs), "binary": kappa(binary),
                             "confusion": {a + " -> " + b: n for (a, b), n in Counter(pairs).items()},
                             "by_condition_and_turn": {}}
    for cond in conditions:
        for t in range(1, 5):
            subset = [(mini[k] in bad, primary[k] in bad) for k in mini if k[0].endswith("::" + cond) and k[1] == t]
            report["inter_judge"]["by_condition_and_turn"][f"{cond}:T{t}"] = kappa(subset)
    with (PROJECT / "annotation_disagreements_filled.csv").open(newline="") as f:
        annotations = list(csv.DictReader(f))
    report["human_annotation"] = {"n": len(annotations), "judges": {}}
    report["human_annotation"]["all_are_judge_disagreements"] = all(mini[(r["dialogue_id"], int(r["turn_index"]))] != primary[(r["dialogue_id"], int(r["turn_index"]))] for r in annotations)
    for judge, labels in turn_labels.items():
        hp = [(labels[(r["dialogue_id"], int(r["turn_index"]))], r["human_label"]) for r in annotations]
        report["human_annotation"]["judges"][judge] = {"four_label": kappa(hp), "binary": kappa([(a in bad, b in bad) for a, b in hp])}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
