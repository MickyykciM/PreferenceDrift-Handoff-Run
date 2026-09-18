"""Read-only Windows audit adapter for the PreferenceDrift v16 snapshot.

The handoff asks for an independent, audit-only path-mapping layer instead of
editing the original manifests or code. This launcher installs four in-process
adaptations and then runs one project entry point. No file in the snapshot is
modified; scripts/check_integrity.py re-hashes every file afterwards.

1. fcntl stub. The project imports POSIX `fcntl` at module level for run locks.
   The stub's flock() RAISES, so any path that would start or continue an
   inference run fails loudly. Verification paths never take the lock.
2. Inference blocked. mlx, mlx_lm, torch, transformers and huggingface_hub are
   made unimportable (none is installed in the audit venv either).
3. POSIX relative-path strings. Frozen source-hash maps were recorded on macOS
   with keys such as "experiments/run_pressure_v15.py"; str(WindowsPath) would
   give backslashes. Only RELATIVE Windows paths are rendered with "/" (which
   Win32 accepts); absolute paths are untouched.
4. Original-root mapping. Five run manifests store absolute paths under the
   original Mac workspace. When the project's read_json() loads a run manifest
   or selection receipt, string values beginning with ORIGINAL_ROOT are mapped
   to the local snapshot root in native form (so the v15b check
   `old["study_path"] == str(study_path.resolve())` compares like with like).
   As a fallback, the same prefix is mapped when a pathlib.Path is constructed.
   This is the in-process equivalent of a symlink at the original location.

In pytest mode only, flock() is a no-op instead of raising: the test suite runs
fake/fixture backends in temporary directories and needs a working lock call.
Inference stays impossible because the model libraries are blocked.

Usage (cwd = snapshot project root):
  python audit_env.py --project . script <script.py> [args...]
  python audit_env.py --project . call <module>:<function> --json-out F [--args-json '[...]']
  python audit_env.py --project . pytest [pytest args...]
"""
import argparse
import importlib
import json
import os
import pathlib
import runpy
import sys
import types

ORIGINAL_ROOT = "/Users/henrywang/Documents/ChatGPT/L3 lab/PreferenceDrift_ClaudeCode_Handoff_2026-09-04/project"
BLOCKED = ("mlx", "mlx.core", "mlx_lm", "torch", "transformers", "huggingface_hub")
LOCATION_FILES = ("manifest.json", "selection_receipt.json", "discovery_selection.json")
MAPPED = []


def install(project_root, lock_mode="deny"):
    root = pathlib.Path(project_root).resolve()
    local_root = str(root).replace("\\", "/")
    native_root = str(root)

    if "fcntl" not in sys.modules:
        try:
            import fcntl  # noqa: F401  (real module on POSIX)
        except ImportError:
            stub = types.ModuleType("fcntl")
            stub.LOCK_SH, stub.LOCK_EX, stub.LOCK_NB, stub.LOCK_UN = 1, 2, 4, 8

            def flock(*_args, **_kwargs):
                if lock_mode != "noop":
                    raise RuntimeError("audit adapter: run locks are disabled; only read-only verification is permitted")

            stub.flock = flock
            sys.modules["fcntl"] = stub

    for name in BLOCKED:
        sys.modules[name] = None

    if os.name == "nt":
        original_str = pathlib.PurePath.__str__

        def posix_relative_str(self):
            text = original_str(self)
            if not self.drive and not self.root:
                text = text.replace("\\", "/")
            return text

        pathlib.PureWindowsPath.__str__ = posix_relative_str

    original_init = pathlib.PurePath.__init__

    def mapped_init(self, *args):
        converted = []
        for arg in args:
            if isinstance(arg, str) and (arg == ORIGINAL_ROOT or arg.startswith(ORIGINAL_ROOT + "/")):
                MAPPED.append(arg)
                arg = local_root + arg[len(ORIGINAL_ROOT):]
            converted.append(arg)
        original_init(self, *converted)

    pathlib.PurePath.__init__ = mapped_init
    sys.path.insert(0, str(root))

    # Map stored workspace locations when a run manifest / selection receipt is read.
    from preference_drift import pilot_common
    original_read_json = pilot_common.read_json

    def relocate(value):
        if isinstance(value, str) and (value == ORIGINAL_ROOT or value.startswith(ORIGINAL_ROOT + "/")):
            MAPPED.append(value)
            return native_root + value[len(ORIGINAL_ROOT):].replace("/", os.sep)
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        return value

    def read_json(path):
        value = original_read_json(path)
        return relocate(value) if pathlib.Path(path).name in LOCATION_FILES else value

    pilot_common.read_json = read_json
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return root


def jsonable(value):
    if isinstance(value, dict):
        return {str(k) if not isinstance(k, (str, int, float, bool)) or k is None else k: jsonable(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, pathlib.PurePath):
        return str(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("mode", choices=("script", "call", "pytest"))
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    a = parser.parse_args()
    root = install(a.project, lock_mode="noop" if a.mode == "pytest" else "deny")
    os.chdir(root)
    if a.mode == "script":
        script, *args = a.rest
        sys.argv = [script, *args]
        runpy.run_path(script, run_name="__main__")
    elif a.mode == "pytest":
        import pytest
        code = pytest.main(a.rest)
        print(f"[audit_env] original-root paths mapped: {len(MAPPED)}", file=sys.stderr)
        raise SystemExit(code)
    else:
        sub = argparse.ArgumentParser()
        sub.add_argument("target")
        sub.add_argument("--json-out", required=True)
        sub.add_argument("--args-json", default="[]")
        c = sub.parse_args(a.rest)
        module_name, function_name = c.target.split(":")
        function = getattr(importlib.import_module(module_name), function_name)
        # Arguments written as "path:<p>" become the absolute path ROOT / <p>, like the project's own defaults.
        args = [root / x[5:] if isinstance(x, str) and x.startswith("path:") else x
                for x in json.loads(c.args_json)]
        result = function(*args)
        out = pathlib.Path(c.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(jsonable(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                       encoding="utf-8", newline="\n")
        print(json.dumps({"written": str(out), "original_root_paths_mapped": len(MAPPED)}))
    print(f"[audit_env] original-root paths mapped: {len(MAPPED)}", file=sys.stderr)


if __name__ == "__main__":
    main()
