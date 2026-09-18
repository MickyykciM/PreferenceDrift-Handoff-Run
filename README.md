# PreferenceDrift handoff: verification run and next-step proposal (2026-09-17)

This repository holds the **output of running the PreferenceDrift cloud handoff** (`PreferenceDrift_Cloud_Handoff_2026-09-16.zip`).

The handoff itself fixes the scope (its §0): *read-only evidence verification, related-work verification, method
design and a written handoff; no new experiment code, no new model runs, no weight downloads, no paid compute;
saved data may be recomputed but never overwritten.* So "running the project" here means:

1. checking the package byte for byte,
2. re-running the project's **own** verification and audit entry points and its full test suite on a new machine
   (Windows) through an audit-only adapter, without touching the evidence,
3. recounting every checkable number in the handoff from the raw per-generation data with independent code,
4. checking the related literature, and
5. writing the phase-1 report the handoff asks for (in Chinese), including a minimal next experiment for approval.

No language model was run, no weights were downloaded, and no API or paid service was used. The experiments
themselves ran on Apple-silicon MLX on the owner's Mac. Regenerating them would need those local weights
and the owner's approval, and the handoff explicitly defers that.

**Main deliverable: [`REPORT_zh.md`](REPORT_zh.md)** (Chinese, as the handoff requests).

## Results at a glance

| Check | Result |
|---|---|
| Package integrity | Inner ZIP SHA-256 `358d7235…7f10f8` matches the handoff; all 7,855 files match `HANDOFF_MANIFEST.json`; the four key `samples.json` hashes match |
| Project's own read-only verification (v14, v15, v15b, v16) | **10/10 entry points pass on Windows**; every zero-call resume makes 0 new calls |
| Re-generated vs saved audits | v15 final audit and v16 failure export **identical**; v16 audit differs only in 2 workspace-path fields; v14 audit only in 7 audit timestamps and the local-weight re-hash flag (weights not in the package) |
| Full test suite | 282/283 pass; the 1 failure needs symlink privilege on Windows (WinError 1314), not a project defect |
| Outer `audit_results.py` (unmodified) | Reproduces `audit_results.json`: 500/500 fields, same normalised SHA-256 |
| Handoff numbers, sections 2.1-2.8 | **0 conflicts** across ~150 claims (2.1-2.4: 54 match, 3 partial; 2.5-2.6: 64 match, 2 unverifiable; 2.7-2.8: all match). Only wording corrections, listed in the report section 1.6 |
| Evidence untouched | All 7,855 files re-hashed after every run: unchanged, no stray files |
| Design issues found | v15's "preference" turns also assert the answer (`My answer to the question is X.`); 288/480 v15 final messages ask the model to "reconsider your answer" although it never gave one |
| Literature | "Opinion before the first answer shifts it" is well covered (Sharma et al. ICLR 2024 and others). Not found in prior work: separate want / believe / instruct conditions, an exposure-matched control, early-vs-last position with an identical final turn, a label-vs-meaning control inside a sycophancy design |
| Proposed next step | A minimal fixed-prefix **v17** (report sections 4-6): want / believe / instruct / exposure sentences that differ only in the attitude verb, a common neutral final question, an exact first-label log-odds read-out validated first on already-opened data, 3 paraphrase sets plus a wording placebo, at least 20 families per bank, and a rule-generated confirmation bank that always runs. Revised after an independent methodological review (`review/DESIGN_REVIEW.md`). Awaiting the owner's decisions; nothing was implemented or run |

## Repository layout

| Path | What it is |
|---|---|
| `REPORT_zh.md` | Phase-1 report: verification summary, research ledger, research question, literature, proposed v17 design, decision rules, open decisions |
| `inputs/` | The four files of the handoff exactly as received (prompt, v16 snapshot ZIP, `audit_results.py/.json`); hashes in `outputs/00_integrity.json` |
| `adapter/` | Audit-only Windows adapter and runners (`check_integrity.py`, `audit_env.py`, `run_official_audits.py`, `compare_audits.py`, `rerun_audit_results.py`) |
| `recompute/` | Independent recount scripts, their JSON output and claim-by-claim `FINDINGS.md` per handoff section |
| `literature/` | Verified related-work notes |
| `review/` | Independent methodological review of the v17 draft and how each point was addressed |
| `outputs/` | Integrity reports, official audit re-runs (`official/`), comparison with saved audits, pytest XML, logs |
| `reproduce.py` | Re-runs everything from `inputs/` |

## Reproduce

```bash
uv venv --python 3.12 _work/.venv
uv pip install --python _work/.venv/Scripts/python.exe numpy==2.5.3 jsonschema==4.19.2 pytest==9.1.1 openai==2.44.0 python-dotenv==1.2.2 tqdm==4.70.1 requests==2.32.5
_work/.venv/Scripts/python.exe reproduce.py
```

On macOS or Linux, use `_work/.venv/bin/python`. The whole run takes about one minute. The packages are the
project's own locked versions (`requirements-mlx-v5.lock.txt`) minus the MLX, torch and transformers inference
stack, which is deliberately absent so that inference is impossible.

### What the adapter changes, and what it does not

`adapter/audit_env.py` makes four in-process adaptations. It never edits a file in the snapshot, and the
integrity check re-hashes all 7,855 files after the run to prove it:

1. it stubs the POSIX-only `fcntl` module; its `flock()` raises during audits, so no run can be started;
2. it makes `mlx`, `torch`, `transformers` and `huggingface_hub` unimportable;
3. it renders relative Windows paths with `/`, so the frozen macOS source-hash maps compare like with like;
4. it maps the original workspace root recorded in five run manifests
   (`/Users/.../PreferenceDrift_ClaudeCode_Handoff_2026-09-04/project`) to the local snapshot. This is the
   in-process equivalent of a symlink.

These are the handoff's "audit-only path mapping"; the original manifests stay byte-identical.
