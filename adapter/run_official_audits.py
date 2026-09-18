"""Step 1: run the project's OWN read-only verification entry points (no inference).

Each step runs in a fresh subprocess through adapter/audit_env.py with the
snapshot project as cwd. Nothing is written inside the snapshot: derived audits
are written to outputs/official/, logs to outputs/logs/. The project's --write
flags (which would overwrite results/analyses/*.json) are never used.

Usage: python adapter/run_official_audits.py [--project _work/v16/project]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V14 = ("results/runs/20260914_repetition_qwen3_8bit_dev_v14", "results/runs/20260914_repetition_qwen3_8bit_holdout_v14",
       "results/designs/repetition_qwen3_8bit_locked_selection_v14.json")
V15_PLAN = "results/designs/pressure_v15_locked_plan.json"


def steps(out):
    o = lambda name: str(out / name)  # noqa: E731
    return [
        ("v14_verify_development", ["script", "experiments/run_measurement_v14.py", "--run-dir", V14[0], "--verify-only"]),
        ("v14_verify_holdout", ["script", "experiments/run_measurement_v14.py", "--run-dir", V14[1], "--verify-only"]),
        ("v14_full_audit", ["script", "experiments/audit_measurement_v14.py", "--development", V14[0], "--holdout", V14[1],
                            "--selection", V14[2], "--write-to", o("measurement_repair_v14_final_audit.rerun.json")]),
        ("v15_verify_calibration", ["script", "experiments/run_pressure_v15.py", "--plan", V15_PLAN,
                                    "--out", "results/runs/20260916_pressure_v15_calibration", "--verify-only"]),
        ("v15_verify_study", ["script", "experiments/run_pressure_v15.py", "--plan", V15_PLAN,
                              "--out", "results/runs/20260916_pressure_v15_study", "--verify-only"]),
        ("v15b_verify", ["script", "diagnostics/updated_task_v15b.py", "--plan", "results/designs/updated_task_v15b_locked_plan.json",
                         "--out", "results/runs/20260916_updated_task_v15b", "--verify-only"]),
        ("v15_full_audit", ["call", "diagnostics.audit_pressure_v15:audit", "--json-out", o("pressure_v15_final_audit.rerun.json")]),
        ("v16_verify_calibration", ["script", "studies/revision_v16/run.py", "--verify-only",
                                    "--out", "results/runs/20260916_revision_v16_discovery_calibration"]),
        ("v16_verify_study", ["script", "studies/revision_v16/run.py", "--verify-only",
                              "--out", "results/runs/20260916_revision_v16_discovery_study"]),
        ("v16_full_audit", ["call", "diagnostics.audit_revision_v16:audit", "--json-out", o("revision_v16_final_audit.rerun.json"),
                            "--args-json", json.dumps(["path:results/runs/20260916_revision_v16"])]),
        ("pytest_full_suite", ["pytest", "-q", "-p", "no:cacheprovider", "--junitxml=" + o("pytest_windows_adapter.xml"),
                               "--basetemp=" + str(REPO / "_work/pytest_tmp")]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=REPO / "_work/v16/project")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out, logs = REPO / "outputs/official", REPO / "outputs/logs"
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "PYTHONHASHSEED": "0"}
    summary_path = REPO / "outputs/01_official_audits_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() and a.only else {}
    for name, command in steps(out):
        if a.only and name not in a.only:
            continue
        cmd = [sys.executable, str(REPO / "adapter/audit_env.py"), "--project", str(a.project.resolve()), *command]
        start = time.time()
        proc = subprocess.run(cmd, cwd=a.project, env=env, capture_output=True, text=True, encoding="utf-8")
        elapsed = round(time.time() - start, 1)
        (logs / f"{name}.stdout.txt").write_text(proc.stdout, encoding="utf-8", newline="\n")
        (logs / f"{name}.stderr.txt").write_text(proc.stderr, encoding="utf-8", newline="\n")
        summary[name] = {"exit_code": proc.returncode, "seconds": elapsed,
                         "command": " ".join(command), "stdout_tail": proc.stdout.strip().splitlines()[-3:],
                         "stderr_tail": proc.stderr.strip().splitlines()[-3:]}
        print(f"{name}: exit {proc.returncode} in {elapsed}s", flush=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
