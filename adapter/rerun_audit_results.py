"""Re-run the handoff's outer `audit_results.py` UNMODIFIED against the v16 snapshot.

The script expects the 2026-09-04 package layout beside it:
    <dir>/audit_results.py
    <dir>/PreferenceDrift_ClaudeCode_Handoff_2026-09-04/project/...
This builds that layout under _work/api_audit_layout/ with a directory junction
(Windows) or symlink (POSIX) that points at the extracted v16 project, runs a
byte-identical copy of the script, and compares its JSON output with
inputs/audit_results.json field by field and by line-ending-normalised hash.

Never delete the layout recursively: the junction points INTO the snapshot.
Remove only the link itself (Windows: `cmd /c rmdir <link>`).

Usage: python adapter/rerun_audit_results.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO / "_work/v16/project"
LAYOUT = REPO / "_work/api_audit_layout"
LINK = LAYOUT / "PreferenceDrift_ClaudeCode_Handoff_2026-09-04" / "project"
OUT = REPO / "recompute/api_v1_v3/audit_results_rerun.json"


def leaves(value, prefix="$"):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from leaves(v, f"{prefix}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, value


def main():
    LINK.parent.mkdir(parents=True, exist_ok=True)
    if not LINK.exists():
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(LINK), str(PROJECT)], check=True, capture_output=True)
        else:
            LINK.symlink_to(PROJECT, target_is_directory=True)
    script = LAYOUT / "audit_results.py"
    shutil.copyfile(REPO / "inputs/audit_results.py", script)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, str(script)], cwd=LAYOUT, env=env, capture_output=True, text=True,
                          encoding="utf-8", check=True)
    OUT.write_text(proc.stdout, encoding="utf-8", newline="\n")
    saved_text = (REPO / "inputs/audit_results.json").read_text(encoding="utf-8")
    saved, rerun = json.loads(saved_text), json.loads(proc.stdout)
    a, b = dict(leaves(saved)), dict(leaves(rerun))
    differing = sorted(k for k in set(a) | set(b) if a.get(k, "<missing>") != b.get(k, "<missing>"))
    norm = lambda s: hashlib.sha256(s.replace("\r\n", "\n").encode()).hexdigest()  # noqa: E731
    print(json.dumps({"script_identical_to_input": script.read_bytes() == (REPO / "inputs/audit_results.py").read_bytes(),
                      "fields_saved": len(a), "fields_rerun": len(b), "differing_fields": differing[:20],
                      "n_differing": len(differing), "same_key_order": list(saved) == list(rerun),
                      "sha256_normalised_saved": norm(saved_text), "sha256_normalised_rerun": norm(proc.stdout)}, indent=2))


if __name__ == "__main__":
    main()
