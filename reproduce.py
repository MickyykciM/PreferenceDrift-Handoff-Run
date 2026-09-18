"""Re-run the whole read-only verification from the files in inputs/.

No model inference, no API calls, no downloads beyond the Python packages below.

Setup (once; Python 3.12, packages pinned to the project's own lock file):
    uv venv --python 3.12 _work/.venv
    uv pip install --python _work/.venv/Scripts/python.exe numpy==2.5.3 jsonschema==4.19.2 \
        pytest==9.1.1 openai==2.44.0 python-dotenv==1.2.2 tqdm==4.70.1 requests==2.32.5
    (on macOS/Linux the interpreter is _work/.venv/bin/python)

Run:
    _work/.venv/Scripts/python.exe reproduce.py

Steps: unpack the v16 snapshot -> integrity check against HANDOFF_MANIFEST.json ->
the project's own verify-only/audit entry points and the full pytest suite through
adapter/audit_env.py -> diff against the audits saved in the snapshot -> the three
independent recount scripts and an unmodified re-run of the outer audit_results.py ->
integrity check again (proves nothing was written inside the snapshot).
"""
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
WORK = REPO / "_work/v16"
PROJECT = WORK / "project"


def step(title, *args):
    print(f"\n=== {title}", flush=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, *map(str, args)], cwd=REPO, env=env, check=True)


def main():
    if not (WORK / "HANDOFF_MANIFEST.json").exists():
        print("Unpacking inputs/PreferenceDrift_Revision_v16_2026-09-16.zip ...", flush=True)
        with zipfile.ZipFile(REPO / "inputs/PreferenceDrift_Revision_v16_2026-09-16.zip") as z:
            z.extractall(WORK)
    step("0. integrity before", "adapter/check_integrity.py", "--out", "outputs/00_integrity.json")
    step("1. official read-only audits + pytest", "adapter/run_official_audits.py")
    step("2. compare with saved audits", "adapter/compare_audits.py")
    step("3a. recount sections 2.7-2.8 (v15/v16)", "recompute/v15_v16/recompute_v15_v16.py", PROJECT)
    step("3b. recount sections 2.1-2.4 (API, v1-v3)", "recompute/api_v1_v3/recompute_api_v1_v3.py", PROJECT)
    step("3b'. re-run the outer audit_results.py unmodified", "adapter/rerun_audit_results.py")
    step("3c. recount sections 2.5-2.6 (v4-v14)", "recompute/v4_v14/recompute_v4_v14.py", "--project", PROJECT)
    step("4. integrity after", "adapter/check_integrity.py", "--out", "outputs/03_integrity_after_runs.json")
    print("\nDone. See outputs/ and recompute/*/FINDINGS.md.")


if __name__ == "__main__":
    main()
