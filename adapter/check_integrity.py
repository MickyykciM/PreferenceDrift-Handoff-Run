"""Step 0: byte-level integrity of the handoff package (read-only).

Checks the outer ZIP entries, the inner v16 ZIP hash/size/entry count/CRCs, every
file in HANDOFF_MANIFEST.json against the extracted working copy, and the four
key SHA-256 values quoted in the handoff prompt. Writes a JSON report.

Usage: python adapter/check_integrity.py [--work _work/v16] [--out outputs/00_integrity.json]
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INNER_SHA256 = "358d723519b6bf28b7d3223fe568a93be61bb6257c7af664f67e9ebd6b7f10f8"
INNER_BYTES = 31_008_540
KEY_FILES = {  # handoff prompt, section 6
    "v14 development samples.json": ("project/results/runs/20260914_repetition_qwen3_8bit_dev_v14/samples.json",
                                     "b2171414a11a338e259d710472c73d0712dab5e766bc63c3b0874f829b0a5e68"),
    "v14 holdout samples.json": ("project/results/runs/20260914_repetition_qwen3_8bit_holdout_v14/samples.json",
                                 "72774ed35c684695cf5c728497bf37752d1b3744927e4f35ef43986c6ac8644b"),
    "v16 discovery calibration samples.json": ("project/results/runs/20260916_revision_v16_discovery_calibration/samples.json",
                                               "a6e1ce97788b30ae494f87e1408c4579182354a3bbc0223b2ea75defc08ba06e"),
    "v16 discovery study samples.json": ("project/results/runs/20260916_revision_v16_discovery_study/samples.json",
                                         "5568a68840e45acbf05502a37e8fa470a670b72126e4b45d933aca91fd70c834"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=REPO / "_work/v16")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/00_integrity.json")
    a = ap.parse_args()
    inner = REPO / "inputs/PreferenceDrift_Revision_v16_2026-09-16.zip"
    report = {"inputs": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                         for p in sorted((REPO / "inputs").iterdir()) if p.is_file()}}

    with zipfile.ZipFile(inner) as z:
        bad_crc = z.testzip()
        names = z.namelist()
        files = [n for n in names if not n.endswith("/")]
    report["inner_zip"] = {"sha256": sha256(inner), "sha256_expected": INNER_SHA256,
                           "bytes": inner.stat().st_size, "bytes_expected": INNER_BYTES,
                           "entries_total": len(names), "file_entries": len(files),
                           "crc_first_bad_entry": bad_crc}
    report["inner_zip"]["ok"] = (report["inner_zip"]["sha256"] == INNER_SHA256 and
                                 report["inner_zip"]["bytes"] == INNER_BYTES and bad_crc is None)

    manifest = json.loads((a.work / "HANDOFF_MANIFEST.json").read_text(encoding="utf-8"))
    listed = manifest["files"]
    missing, mismatched, size_mismatch = [], [], []
    for rel, meta in listed.items():
        p = a.work / rel
        if not p.is_file():
            missing.append(rel)
            continue
        if p.stat().st_size != meta["bytes"]:
            size_mismatch.append(rel)
        if sha256(p) != meta["sha256"]:
            mismatched.append(rel)
    on_disk = {p.relative_to(a.work).as_posix() for p in a.work.rglob("*") if p.is_file()}
    unlisted = sorted(on_disk - set(listed) - {"HANDOFF_MANIFEST.json"})
    report["handoff_manifest"] = {"format": manifest.get("format"), "weights_included": manifest.get("weights_included"),
                                  "virtual_environment_included": manifest.get("virtual_environment_included"),
                                  "files_listed": len(listed), "files_on_disk": len(on_disk),
                                  "missing": missing, "sha256_mismatch": mismatched, "size_mismatch": size_mismatch,
                                  "unlisted_on_disk": unlisted[:50], "unlisted_count": len(unlisted)}
    report["handoff_manifest"]["ok"] = not (missing or mismatched or size_mismatch or unlisted)

    report["key_files"] = {}
    for label, (rel, expected) in KEY_FILES.items():
        got = sha256(a.work / rel)
        report["key_files"][label] = {"path": rel, "sha256": got, "expected": expected,
                                      "manifest_sha256": listed[rel]["sha256"], "ok": got == expected == listed[rel]["sha256"]}
    report["ok"] = (report["inner_zip"]["ok"] and report["handoff_manifest"]["ok"] and
                    all(v["ok"] for v in report["key_files"].values()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "inner_zip": report["inner_zip"],
                      "manifest": {k: v for k, v in report["handoff_manifest"].items() if k not in ("unlisted_on_disk",)},
                      "key_files_ok": {k: v["ok"] for k, v in report["key_files"].items()}}, indent=2))


if __name__ == "__main__":
    main()
