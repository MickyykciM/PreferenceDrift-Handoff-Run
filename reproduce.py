"""Re-run the whole read-only verification from the files in inputs/.

No model inference, no API calls, no downloads beyond the Python packages below.

Setup (once; Python 3.12, packages pinned to the project's own lock file):
    uv venv --python 3.12 _work/.venv
    uv pip install --python _work/.venv/Scripts/python.exe numpy==2.5.3 jsonschema==4.19.2 \
        pytest==9.1.1 openai==2.44.0 python-dotenv==1.2.2 tqdm==4.70.1 requests==2.32.5
    (on macOS/Linux the interpreter is _work/.venv/bin/python)

Run:
    _work/.venv/Scripts/python.exe reproduce.py [--work-dir DIR]

--work-dir (or the PD_WORK environment variable) chooses where the snapshot is
unpacked; the default is _work/ inside the repository. The deepest path inside the
snapshot is 146 characters, so on Windows without long-path support the work
directory must be short (for example C:\\pdw); the script checks this up front.

Steps: unpack the v16 snapshot -> integrity check against HANDOFF_MANIFEST.json ->
the project's own verify-only/audit entry points and the full pytest suite through
adapter/audit_env.py -> diff against the audits saved in the snapshot -> the three
independent recount scripts and an unmodified re-run of the outer audit_results.py ->
integrity check again (proves nothing was written inside the snapshot).
"""
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
SNAPSHOT_ZIP = REPO / "inputs/PreferenceDrift_Revision_v16_2026-09-16.zip"
WIN_MAX_PATH = 259


def long_paths_enabled():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            return winreg.QueryValueEx(key, "LongPathsEnabled")[0] == 1
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=Path(os.environ.get("PD_WORK") or REPO / "_work"))
    a = ap.parse_args()
    work = a.work_dir.resolve()
    snapshot, project = work / "v16", work / "v16/project"
    if os.name == "nt" and not long_paths_enabled():
        with zipfile.ZipFile(SNAPSHOT_ZIP) as z:
            deepest = max(len(n) for n in z.namelist())
        if len(str(snapshot)) + 1 + deepest > WIN_MAX_PATH:
            sys.exit(f"Work directory {snapshot} is too long: the snapshot's deepest path would exceed Windows' "
                     f"{WIN_MAX_PATH}-character limit. Re-run with a short --work-dir, e.g. "
                     f"'reproduce.py --work-dir C:\\pdw'.")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "PD_WORK": str(work)}

    def step(title, *args):
        print(f"\n=== {title}", flush=True)
        subprocess.run([sys.executable, *map(str, args)], cwd=REPO, env=env, check=True)

    if not (snapshot / "HANDOFF_MANIFEST.json").exists():
        print(f"Unpacking {SNAPSHOT_ZIP.name} into {snapshot} ...", flush=True)
        with zipfile.ZipFile(SNAPSHOT_ZIP) as z:
            z.extractall(snapshot)
    step("0. integrity before", "adapter/check_integrity.py", "--work", snapshot, "--out", "outputs/00_integrity.json")
    step("1. official read-only audits + pytest", "adapter/run_official_audits.py", "--project", project)
    step("2. compare with saved audits", "adapter/compare_audits.py")
    step("3a. recount sections 2.7-2.8 (v15/v16)", "recompute/v15_v16/recompute_v15_v16.py", project)
    step("3b. recount sections 2.1-2.4 (API, v1-v3)", "recompute/api_v1_v3/recompute_api_v1_v3.py", project)
    step("3b'. re-run the outer audit_results.py unmodified", "adapter/rerun_audit_results.py")
    step("3c. recount sections 2.5-2.6 (v4-v14)", "recompute/v4_v14/recompute_v4_v14.py", "--project", project)
    step("4. integrity after", "adapter/check_integrity.py", "--work", snapshot, "--out", "outputs/03_integrity_after_runs.json")
    print("\nDone. See outputs/ and recompute/*/FINDINGS.md.")


if __name__ == "__main__":
    main()
