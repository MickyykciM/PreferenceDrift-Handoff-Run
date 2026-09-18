"""Step 2: compare re-generated audits with the audits saved in the snapshot.

Every leaf that differs is listed and classified. Only two kinds of difference
are expected: wall-clock timestamps of the audit itself, and absolute workspace
paths (original Mac workspace vs this machine). Anything else is reported as
SUBSTANTIVE.

Usage: python adapter/compare_audits.py
"""
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("PD_WORK") or REPO / "_work") / "v16/project"
ORIGINAL = "/Users/henrywang/Documents/ChatGPT/L3 lab/PreferenceDrift_ClaudeCode_Handoff_2026-09-04/project"
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def diff(a, b, path="$"):
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return [(path, a, b)]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", a.get(k, "<missing>"), b.get(k, "<missing>")))
            else:
                out.extend(diff(a[k], b[k], f"{path}.{k}"))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [(path + ".length", len(a), len(b))]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(diff(x, y, f"{path}[{i}]"))
        return out
    return [] if a == b else [(path, a, b)]


def classify(path, saved, rerun):
    # The original v14 audit ran with --verify-local-weights (re-hashing the 4.27 GB MLX weights on the
    # original Mac). The weights are not in the handoff and downloading them is not authorised here.
    if path.endswith(".local_weight_hash_rechecked") and saved is True and rerun is False:
        return "weights_not_rehashed_here"
    if isinstance(saved, str) and isinstance(rerun, str):
        if saved.startswith(ORIGINAL):
            local = str(PROJECT.resolve())
            tail = saved[len(ORIGINAL):]
            if rerun.replace("\\", "/") == local.replace("\\", "/") + tail:
                return "workspace_path"
        if TIMESTAMP.match(saved) and TIMESTAMP.match(rerun):
            return "audit_timestamp"
    return "SUBSTANTIVE"


def main():
    official = REPO / "outputs/official"
    v16 = load(official / "revision_v16_final_audit.rerun.json")
    pairs = {
        "v14 final audit": (load(PROJECT / "results/analyses/measurement_repair_v14_final_audit.json"),
                            load(official / "measurement_repair_v14_final_audit.rerun.json")),
        "v15 final audit": (load(PROJECT / "results/analyses/pressure_v15_final_audit.json"),
                            load(official / "pressure_v15_final_audit.rerun.json")),
        "v16 final audit": (load(PROJECT / "results/analyses/revision_v16_final_audit.json"), v16[0]),
        "v16 failure examples": (load(PROJECT / "results/analyses/revision_v16_failure_examples.json"), v16[1]),
    }
    report = {}
    for name, (saved, rerun) in pairs.items():
        items = diff(saved, rerun)
        classes = {}
        for p, s, r in items:
            classes.setdefault(classify(p, s, r), []).append({"path": p, "saved": s, "rerun": r})
        report[name] = {"differing_leaves": len(items), "by_class": {k: len(v) for k, v in classes.items()},
                        "substantive": classes.get("SUBSTANTIVE", []),
                        "examples": {k: v[:3] for k, v in classes.items() if k != "SUBSTANTIVE"},
                        "identical_apart_from_paths_and_timestamps": not classes.get("SUBSTANTIVE")}
    out = REPO / "outputs/02_audit_comparison.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    for name, r in report.items():
        print(f"{name}: {r['differing_leaves']} differing leaves {r['by_class']} -> "
              f"{'OK' if r['identical_apart_from_paths_and_timestamps'] else 'SUBSTANTIVE DIFFERENCES'}")
        for s in r["substantive"][:10]:
            print("   ", s["path"], "| saved:", str(s["saved"])[:120], "| rerun:", str(s["rerun"])[:120])


if __name__ == "__main__":
    main()
