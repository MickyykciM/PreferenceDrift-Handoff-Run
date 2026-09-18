#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent recount of PreferenceDrift handoff sections 2.5 and 2.6 from raw data.

Scope
-----
* 2.5: v4 measurement calibration on Qwen2.5-1.5B and Qwen2.5-3B.
* 2.6: the v5-v14 measurement-repair ledger (per-attempt table, the 2,879 + 2 total,
  the v13/v14 paired comparisons and the v14 details).

Method
------
* Standalone: Python stdlib + numpy only. It imports NO project module and runs NO
  project script. No model inference, no network access, no downloads.
* Read-only on the project snapshot (files are only opened for reading).
* Rawest input: results/runs/<run>/generations/*.json, one checkpoint per model call
  (request messages/seed/decoding, status, raw output text, finish_reason metadata).
* Gold labels: the frozen design saved inside each run (cases.json for v4, the
  manifest's inputs.cases for v5+). They are joined to the generations by the exact
  request messages (not by project hashes), and they are re-derived independently from
  the stored formal certificates and/or the prompt text wherever that is possible.
* Output parsers and scoring gates are re-implemented here from each version's frozen
  rule; saved samples.json / labels / summaries / audits are read ONLY to cross-check.

usage: python recompute_v4_v14.py [--project PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import json
import platform
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

DEFAULT_PROJECT = str(Path(os.environ.get("PD_WORK") or Path(__file__).resolve().parents[2] / "_work") / "v16/project")
ORDERS = ("positive_first", "negative_first")
ENUMS = {"truth": ("TRUE", "FALSE"), "validity": ("VALID", "INVALID")}
YESNO = {"truth": {"YES": "TRUE", "NO": "FALSE"}, "validity": {"YES": "VALID", "NO": "INVALID"}}

RUN_V4 = {"qwen2.5-1.5b": "20260914_qwen15b_measurement_calibration_v4",
          "qwen2.5-3b": "20260914_qwen3b_measurement_replication_v4"}
RUNTIME_PROFILES = ("mps_fp16_sdpa", "mps_fp16_eager", "mps_bf16_eager", "cpu_fp32_eager")

# (attempt id, handoff row label, run dirs, parser kind, gate kind, handoff "completed" value)
ATTEMPTS = [
    ("runtime_v5", "1.5B four runtime/numeric configurations",
     [f"20260914_runtime_v5_{p}" for p in RUNTIME_PROFILES], "v4json", "runtime", 80),
    ("format_3b_v5", "3B BF16 four formats (interrupted)",
     ["20260914_format_development_qwen3b_v5"], "v5format_hf", "partial", 34),
    ("mlx_load_failure_v5", "7B MLX first load",
     ["20260914_mlx_qwen7b_screen_v5_load_failure"], None, "none", 0),
    ("mlx_screen_v5", "7B format screen", ["20260914_mlx_qwen7b_screen_v5"], "v5format", "screen", 64),
    ("mlx_regression_v5", "7B old-task regression", ["20260914_mlx_qwen7b_regression_v5"], "v5format", "v4gate", 232),
    ("v6_dev", "v6 independent single-axis direct, development",
     ["20260914_clean_direct_qwen7b_dev_v6"], "clean_direct", "clean", 96),
    ("v6_holdout", "v6 first sealed validation", ["20260914_clean_direct_qwen7b_holdout_v6"], "clean_direct", "clean", 96),
    ("v7_dev", "v7 short reasoning, development", ["20260914_reasoning_qwen7b_dev_v7"], "clean_reasoning", "clean", 192),
    ("v8_dev", "v8 rules + balanced demos, 7B", ["20260914_guided_qwen7b_dev_v8"], "guided", "clean", 192),
    ("v9_dev", "v9 Qwen3-4B 4-bit development", ["20260914_guided_qwen3_4b_dev_v9"], "guided", "clean", 192),
    ("v9_holdout", "v9 second validation bank", ["20260914_guided_qwen3_4b_holdout_v9"], "guided", "clean", 96),
    ("v10_dev_partial", "v10 constrained binary choice, first version (interrupted)",
     ["20260914_constrained_qwen3_4b_dev_v10"], "constrained_v10", "partial", 165),
    ("v11_dev", "v11 corrected callback accounting", ["20260914_constrained_qwen3_4b_dev_v11"], "constrained_v11", "clean", 288),
    ("v12_dev", "v12 axis-specific demos, free generation", ["20260914_axis_guided_qwen3_4b_dev_v12"], "guided", "clean", 288),
    ("v13_dev", "v13 8-bit same-request development", ["20260914_guided_qwen3_4b_8bit_dev_v13"], "guided", "clean", 288),
    ("v13_holdout", "v13 third validation bank", ["20260914_guided_qwen3_4b_8bit_holdout_v13"], "guided", "clean", 96),
    ("v14_dev", "v14 generated-text repetition penalty, development",
     ["20260914_repetition_qwen3_8bit_dev_v14"], "guided", "clean", 384),
    ("v14_holdout", "v14 fourth bank, fresh instances", ["20260914_repetition_qwen3_8bit_holdout_v14"], "guided", "clean", 96),
]

# Hashes quoted in the named source documents / saved audits (checked against the files).
DOC_HASHES = {
    "20260914_qwen15b_measurement_calibration_v4": ("1301bace640039b827c99afad81e127f900fd0635f319d7f647b390c6ad86ee9", "MEASUREMENT_CALIBRATION_REPORT_V4"),
    "20260914_qwen3b_measurement_replication_v4": ("d5c4ac1c2875f62c04bddce903d23c33778679b4306b845cee2bbf92cb4a80ed", "MODEL_REPLICATION_REPORT_V4"),
    "20260914_runtime_v5_mps_fp16_sdpa": ("32b616b0f306d80981e7411b1e5e18f1a7708b3239c7035c0f779793d9dedbc3", "runtime_audit_v5.json"),
    "20260914_runtime_v5_mps_fp16_eager": ("e3f7d9e507223f6f0d16285e311e1a0a8a2c46ac143f5463ab8324f999ec5e19", "runtime_audit_v5.json"),
    "20260914_runtime_v5_mps_bf16_eager": ("cd507256e075d875bed74066c673181c7be7bbd06fb79e8d6e953e14e235290b", "runtime_audit_v5.json"),
    "20260914_runtime_v5_cpu_fp32_eager": ("4e437358e5d29c6c6ffba4897f7796f56bb3647910091c5b52e1471d277d1510", "runtime_audit_v5.json"),
    "20260914_mlx_qwen7b_screen_v5": ("a5431cfa84f3ad090131d094182abde97f2392316faf721aa3c408c7332731bd", "v5_attempt_ledger.json"),
    "20260914_clean_direct_qwen7b_holdout_v6": ("2dae1aa74cabd97665fdcf8d8bd7fca72fa445f9ae8e997863ad4c834e743aed", "CLEAN_V6_SERIALIZATION_INCIDENT (pre-fix raw hash)"),
    "20260914_reasoning_qwen7b_dev_v7": ("2b388c0386da6cb07ac6fee814a74cf73975e49361092b039ce646aea6d61621", "audit_measurement_v7.py RAW_HASH"),
    "20260914_guided_qwen7b_dev_v8": ("3be8e9893fccd19899cfe255ff73afdff4b5649dfb0ad6baf0ea1c15da91ad78", "audit_measurement_v8.py RAW_HASH"),
    "20260914_guided_qwen3_4b_dev_v9": ("8a4acb2d06340d9135e9f21fb004648056038515fe151e30ed50e7e290d7a54f", "quantization_paired_v13.json"),
    "20260914_guided_qwen3_4b_holdout_v9": ("b4cd2ae4d98d5b9abcb74588fee64ce09acaa4aa086ef6f89cf471074a03b7d8", "quantization_paired_v13.json"),
    "20260914_guided_qwen3_4b_8bit_dev_v13": ("2da773483d986082001cd12b273b42b1f5cddf22c5a84f3daf657679e05a5eb8", "quantization_paired_v13.json"),
    "20260914_repetition_qwen3_8bit_dev_v14": ("b2171414a11a338e259d710472c73d0712dab5e766bc63c3b0874f829b0a5e68", "MEASUREMENT_REPAIR_REPORT / v14 final audit"),
    "20260914_repetition_qwen3_8bit_holdout_v14": ("72774ed35c684695cf5c728497bf37752d1b3744927e4f35ef43986c6ac8644b", "MEASUREMENT_REPAIR_REPORT / v14 final audit"),
}


# --------------------------------------------------------------------------- utilities
def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mkey(messages):
    """Canonical key of a request's message list (exact role/content match)."""
    return json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def frac_str(a, b):
    return f"{a}/{b}"


# --------------------------------------------------------------------------- parsers
def _unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key")
        out[key] = value
    return out


def parse_json_object(text, fields):
    """v4 rule: optional ```json/``` fence; strict JSON object with unique keys, exactly the
    requested fields, each value one of that field's two labels."""
    t = text.strip()
    if t.startswith("```json\n") and t.endswith("\n```"):
        t = t[len("```json\n"):-len("\n```")].strip()
    elif t.startswith("```\n") and t.endswith("\n```"):
        t = t[len("```\n"):-len("\n```")].strip()
    try:
        obj = json.loads(t, object_pairs_hook=_unique_pairs)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != set(fields):
        return None
    for key, value in obj.items():
        if key not in ENUMS or not isinstance(value, str) or value not in ENUMS[key]:
            return None
    return obj


FINAL_RE = re.compile(r"(?m)^FINAL: (\{[^\n]+\})\s*$")


def parse_final_json(text, fields):
    """v5 reasoning formats: exactly one final 'FINAL: {json}' line that ends the text."""
    found = FINAL_RE.findall(text)
    if len(found) != 1 or not text.rstrip().endswith(found[0]):
        return None
    return parse_json_object(found[0], fields)


def label_exact(text):
    """v6 clean_direct and v10/v11 constrained: the whole stripped reply is YES or NO."""
    t = text.strip()
    return t if t in ("YES", "NO") else None


ANSWER_LINE_RE = re.compile(r"(?m)^ANSWER: (YES|NO)\s*$")


def label_answer_line(text):
    """v7 clean_reasoning: exactly one 'ANSWER: YES|NO' line, and the text ends with it."""
    t = text.strip()
    found = ANSWER_LINE_RE.findall(t)
    if len(found) == 1 and t.endswith("ANSWER: " + found[0]):
        return found[0]
    return None


ANSWER_END_RE = re.compile(r"\bANSWER: (YES|NO)\s*\Z")


def label_answer_end(text):
    """v8/v9/v12/v13/v14 guided: text ends with 'ANSWER: YES|NO' and contains 'ANSWER:' once."""
    t = text.strip()
    match = ANSWER_END_RE.search(t)
    if match is None or t.count("ANSWER:") != 1:
        return None
    return match.group(1)


def parse_output(kind, case, text):
    gold = case["gold"]
    if "literal" in gold:
        t = text.strip()
        return {"literal": t} if t else None
    fields = list(gold)
    if kind == "v4json":
        return parse_json_object(text, fields)
    if kind in ("v5format", "v5format_hf"):
        if "reasoning" in (case.get("format") or ""):
            return parse_final_json(text, fields)
        return parse_json_object(text, fields)
    axis = case["axis"]
    if kind == "clean_direct":
        label = label_exact(text)
    elif kind == "clean_reasoning":
        label = label_answer_line(text)
    elif kind == "guided":
        label = label_answer_end(text)
    elif kind in ("constrained_v10", "constrained_v11"):
        label = label_exact(text)
    else:
        raise ValueError(kind)
    return None if label is None else {axis: YESNO[axis][label]}


def is_known(kind, parsed, result):
    """Whether a parsed reply is scoreable under that version's frozen rule."""
    if parsed is None:
        return False
    meta = result.get("metadata") or {}
    if kind in ("v4json", "v5format_hf"):  # HF runners: excluded only when length-capped
        return meta.get("finish_reason") != "length"
    ok = result.get("error") is None and meta.get("finish_reason") == "eos"
    if kind == "constrained_v10":  # original (buggy) accounting: exactly 2 callbacks
        ok = ok and meta.get("constraint_steps") == 2
    if kind == "constrained_v11":  # corrected accounting: 3 callbacks, 2 emitted tokens
        ok = ok and meta.get("constraint_steps") == 3 and meta.get("finite_logit_checks") == 3 and meta.get("output_tokens") == 2
    return ok


def evaluate(kind, case, gen):
    result = gen["result"]
    text = result.get("text") or ""
    meta = result.get("metadata") or {}
    parsed = parse_output(kind, case, text)
    known = is_known(kind, parsed, result)
    correct = {k: (parsed[k] == v) if known else None for k, v in case["gold"].items()}
    exact = all(correct.values()) if known else None
    finish = meta.get("finish_reason")
    if finish == "length":
        outcome = "truncated"
    elif not known:
        outcome = "unknown"
    else:
        outcome = "correct" if exact else "wrong"
    return {"parsed": parsed, "known": known, "correct": correct, "exact": exact, "outcome": outcome,
            "finish": finish, "output_tokens": meta.get("output_tokens"), "text": text}


# --------------------------------------------------------------------------- gold re-derivation
def ev(expr, world):
    if isinstance(expr, str):
        return bool(world[expr])
    op = expr[0]
    if op == "not":
        return not ev(expr[1], world)
    a, b = ev(expr[1], world), ev(expr[2], world)
    if op == "implies":
        return (not a) or b
    if op == "or":
        return a or b
    if op == "and":
        return a and b
    raise ValueError(op)


def atoms_of(expr, acc):
    if isinstance(expr, str):
        acc.add(expr)
    else:
        for sub in expr[1:]:
            atoms_of(sub, acc)
    return acc


def entails(premises, conclusion):
    names = set()
    for p in premises:
        atoms_of(p, names)
    atoms_of(conclusion, names)
    names = sorted(names)
    satisfiable = False
    for bits in itertools.product((False, True), repeat=len(names)):
        world = dict(zip(names, bits))
        if all(ev(p, world) for p in premises):
            satisfiable = True
            if not ev(conclusion, world):
                return False
    if not satisfiable:
        raise ValueError("inconsistent premises")
    return True


def parse_sentence(s):
    """Tiny grammar for the legacy renderings: atom | 'If X, then Y' | 'It is not the case that X'."""
    s = s.strip()
    for prefix in ("It is not the case that ", "it is not the case that "):
        if s.startswith(prefix):
            return ["not", parse_sentence(s[len(prefix):])]
    m = re.fullmatch(r"[Ii]f (.+?), then (.+)", s)
    if m:
        return ["implies", parse_sentence(m.group(1)), parse_sentence(m.group(2))]
    return s


V4_FACT = re.compile(r'^- "([^"]+)" is (TRUE|FALSE)\.$', re.M)
CLEAN_FACT = re.compile(r'^- The statement "([^"]+)" is (true|false)\.$', re.M)
PREMISE_LINE = re.compile(r"^\d+\. (.+)\.$", re.M)
V4_CONCL = re.compile(r"^CONCLUSION: (.+)\.$", re.M)
CLEAN_STMT = re.compile(r"^Statement to check: (.+)\.$", re.M)
CLEAN_CONCL = re.compile(r"^Proposed conclusion: (.+)\.$", re.M)
ARITH = re.compile(r"Claim: (\d+) ([+-]) (\d+) = (\d+)\.")
GREATER = re.compile(r"Claim: (\d+) is greater than (\d+)\.")
LESS_EQ = re.compile(r"Claim: (\d+) is less than or equal to (\d+)\.")
EVEN = re.compile(r"Claim: (\d+) is an even integer\.")
SUM_Q = re.compile(r"What is (\d+) \+ (\d+)\?")
COPY_Q = re.compile(r"Reply with exactly the word (\w+) and nothing else\.")


def text_legacy_gold(text, fields, clean):
    """Re-derive legacy truth/validity from the prompt text itself."""
    out = {}
    fact_re, concl_re = (CLEAN_FACT, CLEAN_CONCL) if clean else (V4_FACT, V4_CONCL)
    world = {m.group(1): m.group(2).upper() == "TRUE" for m in fact_re.finditer(text)}
    if "truth" in fields:
        m = (CLEAN_STMT if clean else V4_CONCL).search(text)
        if m is None or not world:
            return None
        out["truth"] = "TRUE" if ev(parse_sentence(m.group(1)), world) else "FALSE"
    if "validity" in fields:
        m = concl_re.search(text)
        premises = [parse_sentence(x.group(1)) for x in PREMISE_LINE.finditer(text)]
        if m is None or not premises:
            return None
        out["validity"] = "VALID" if entails(premises, parse_sentence(m.group(1))) else "INVALID"
    return out


def text_arith_truth(text):
    m = ARITH.search(text)
    if m:
        a, op, b, t = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        return (a + b if op == "+" else a - b) == t
    m = GREATER.search(text)
    if m:
        return int(m.group(1)) > int(m.group(2))
    m = LESS_EQ.search(text)
    if m:
        return int(m.group(1)) <= int(m.group(2))
    m = EVEN.search(text)
    if m:
        return int(m.group(1)) % 2 == 0
    return None


def cert_truth(cert, family):
    if not isinstance(cert, dict):
        return None
    if {"world", "premises", "conclusion", "lexicon"} <= set(cert):  # full legacy item
        return ev(cert["conclusion"], cert["world"])
    if "expr" in cert and "world" in cert:
        return ev(cert["expr"], cert["world"])
    if "atom_value" in cert:
        return (not cert["atom_value"]) if cert["negate"] else bool(cert["atom_value"])
    if "members" in cert:
        return cert["target"] in cert["members"]
    if "left" in cert and "right" in cert and cert.get("operator") == ">":
        return cert["left"] > cert["right"]
    if "values" in cert and cert.get("operator") in ("and", "or"):
        values = [bool(v) for v in cert["values"]]
        return all(values) if cert["operator"] == "and" else any(values)
    if "n" in cert:
        return cert["n"] % 2 == 0
    if "a" in cert and "b" in cert:
        if "target" in cert:
            op = cert.get("operator") or {"integer_equality": "+", "integer_addition": "+",
                                          "integer_subtraction": "-"}.get(family)
            if op == "+":
                return cert["a"] + cert["b"] == cert["target"]
            if op == "-":
                return cert["a"] - cert["b"] == cert["target"]
            return None
        if family == "integer_nonstrict_order":
            return cert["a"] <= cert["b"]
    return None


def cert_validity(cert):
    if not isinstance(cert, dict) or "premises" not in cert or "conclusion" not in cert:
        return None
    return entails(cert["premises"], cert["conclusion"])


def derive_gold(case):
    """Return {method: derived gold dict} from independent routes (certificate / text)."""
    gold = case["gold"]
    fields = sorted(gold)
    routes = {}
    text = case.get("prompt") or case.get("body") or ""
    if "literal" in gold:
        m = SUM_Q.search(text)
        if m:
            routes["text"] = {"literal": str(int(m.group(1)) + int(m.group(2)))}
        m = COPY_Q.search(text)
        if m:
            routes["text"] = {"literal": m.group(1)}
        return routes
    section = case.get("section")
    if section in ("legacy_single", "legacy_joint"):
        derived = text_legacy_gold(case["prompt"], fields, clean=False)
        if derived is not None:
            routes["text"] = derived
    elif case.get("source") == "legacy" and case.get("body"):
        derived = text_legacy_gold(case["body"], fields, clean=True)
        if derived is not None:
            routes["text"] = derived
    if fields == ["truth"]:
        value = cert_truth(case.get("certificate"), case.get("family"))
        if value is not None:
            routes["certificate"] = {"truth": "TRUE" if value else "FALSE"}
        value = text_arith_truth(text)
        if value is not None:
            routes["text_arithmetic"] = {"truth": "TRUE" if value else "FALSE"}
    elif fields == ["validity"]:
        value = cert_validity(case.get("certificate"))
        if value is not None:
            routes["certificate"] = {"validity": "VALID" if value else "INVALID"}
    return routes


# --------------------------------------------------------------------------- loading
def norm_case(raw):
    if "source_case" in raw:
        sc = raw["source_case"]
        return {"case_id": raw["case_id"], "messages": raw["messages"], "key": mkey(raw["messages"]),
                "gold": raw["gold"], "axis": raw["axis"], "option_order": raw["option_order"],
                "item_id": raw["item_id"], "format": raw.get("format"),
                "section": sc.get("section"), "context": sc.get("context"), "question_order": sc.get("question_order"),
                "source": sc.get("source", sc.get("section")), "family": sc.get("family"),
                "certificate": sc.get("certificate"), "prompt": sc.get("prompt"), "body": sc.get("body")}
    messages = [{"role": "user", "content": raw["prompt"]}]
    return {"case_id": raw["case_id"], "messages": messages, "key": mkey(messages), "gold": raw["gold"],
            "axis": raw["axis"], "option_order": raw["option_order"], "item_id": raw["item_id"], "format": "direct",
            "section": raw["section"], "context": raw["context"], "question_order": raw["question_order"],
            "source": raw["section"], "family": raw.get("family"), "certificate": raw.get("certificate"),
            "prompt": raw["prompt"], "body": None}


class Run:
    def __init__(self, project, name):
        self.name = name
        self.dir = project / "results" / "runs" / name
        self.exists = self.dir.is_dir()
        self.listing = sorted(p.name for p in self.dir.iterdir()) if self.exists else []
        self.manifest = read_json(self.dir / "manifest.json") if (self.dir / "manifest.json").exists() else None
        self.generations = []
        gdir = self.dir / "generations"
        if gdir.is_dir():
            for path in sorted(gdir.glob("*.json")):
                record = read_json(path)
                self.generations.append({"file": path.name, "request_id": record.get("request_id"),
                                         "status": record.get("status"), "attempts": record.get("attempts"),
                                         "request": record.get("request"), "result": record.get("result"),
                                         "model": record.get("model")})
        if (self.dir / "cases.json").exists():
            self.cases_source = "cases.json"
            self.cases = [norm_case(c) for c in read_json(self.dir / "cases.json")]
        elif self.manifest and "inputs" in self.manifest and "cases" in self.manifest["inputs"]:
            self.cases_source = "manifest.json:inputs.cases"
            self.cases = [norm_case(c) for c in self.manifest["inputs"]["cases"]]
        else:
            self.cases_source = None
            self.cases = []
        self.samples = read_json(self.dir / "samples.json") if (self.dir / "samples.json").exists() else None
        self.summary = read_json(self.dir / "summary.mechanical.json") if (self.dir / "summary.mechanical.json").exists() else None
        self.by_key = {}
        for case in self.cases:
            self.by_key.setdefault(case["key"], []).append(case)

    @property
    def completed(self):
        return [g for g in self.generations if g["status"] == "completed"]


def join_and_score(run, kind):
    """Join each completed generation to its design case(s) by exact messages and score it."""
    problems = []
    evals = {}
    for gen in run.completed:
        key = mkey(gen["request"]["messages"])
        aliases = run.by_key.get(key)
        if not aliases:
            problems.append(f"generation {gen['file']} matches no design case")
            continue
        if any(a["gold"] != aliases[0]["gold"] for a in aliases):
            problems.append(f"alias group with inconsistent gold at {gen['file']}")
        if key in evals:
            problems.append(f"duplicate completed generation for one prompt: {gen['file']}")
        e = evaluate(kind, aliases[0], gen)
        e.update(key=key, file=gen["file"], seed=gen["request"]["seed"], decoding=gen["request"]["decoding"],
                 item_id=aliases[0]["item_id"], axis=aliases[0]["axis"], option_order=aliases[0]["option_order"],
                 source=aliases[0]["source"], family=aliases[0]["family"], gold=aliases[0]["gold"],
                 aliases=len(aliases), metadata=gen["result"].get("metadata") or {})
        evals[key] = e
    views = []
    for case in run.cases:
        if case["key"] in evals:
            views.append({**case, "ev": evals[case["key"]]})
    return evals, views, problems


# --------------------------------------------------------------------------- statistics / gates
def cell_stats(views):
    unique = {}
    for v in views:
        unique.setdefault(v["key"], v)
    rows = list(unique.values())
    by_gold = {}
    for r in rows:
        g = json.dumps(r["gold"], sort_keys=True)
        entry = by_gold.setdefault(g, {"n": 0, "correct": 0})
        entry["n"] += 1
        entry["correct"] += r["ev"]["exact"] is True
    return {"case_records": len(views), "n": len(rows),
            "correct": sum(r["ev"]["exact"] is True for r in rows),
            "known": sum(bool(r["ev"]["known"]) for r in rows),
            "unknown_or_excluded": sum(not r["ev"]["known"] for r in rows),
            "parsed": sum(r["ev"]["parsed"] is not None for r in rows),
            "outcomes": dict(Counter(r["ev"]["outcome"] for r in rows)),
            "labels": dict(Counter(json.dumps(r["ev"]["parsed"], sort_keys=True) if r["ev"]["parsed"] is not None
                                   else "UNPARSED" for r in rows)),
            "by_gold": by_gold}


def contrast(left, right):
    lmap, rmap = {}, {}
    for v in left:
        if v["item_id"] in lmap:
            raise ValueError("two cases for one item in one arm")
        lmap[v["item_id"]] = v
    for v in right:
        if v["item_id"] in rmap:
            raise ValueError("two cases for one item in one arm")
        rmap[v["item_id"]] = v
    if set(lmap) != set(rmap):
        raise ValueError("unpaired items")
    pairs = {}
    for item in sorted(lmap):
        a, b = lmap[item], rmap[item]
        if a["gold"] != b["gold"]:
            raise ValueError("paired gold mismatch")
        pairs.setdefault((a["key"], b["key"]), (a, b, []))[2].append(item)
    known = [(a, b, items) for a, b, items in pairs.values() if a["ev"]["known"] and b["ev"]["known"]]
    changed = [(a, b, items) for a, b, items in known if a["ev"]["parsed"] != b["ev"]["parsed"]]
    field_changes = Counter()
    for a, b, _ in known:
        for field in a["gold"]:
            if a["ev"]["parsed"].get(field) != b["ev"]["parsed"].get(field):
                field_changes[field] += 1
    return {"pairs": len(pairs), "known_pairs": len(known), "unknown_pairs": len(pairs) - len(known),
            "changed": len(changed), "changed_items": sorted(items[0] for _, _, items in changed),
            "changed_by_field": dict(field_changes),
            "wrong_to_right": sum(1 for a, b, _ in known if a["ev"]["exact"] is False and b["ev"]["exact"] is True),
            "right_to_wrong": sum(1 for a, b, _ in known if a["ev"]["exact"] is True and b["ev"]["exact"] is False),
            "unknown_items": sorted(items[0] for a, b, items in pairs.values()
                                    if not (a["ev"]["known"] and b["ev"]["known"]))}


def cell_ok(stats, minimum=Fraction(7, 8), per_gold=False):
    if stats["n"] == 0:
        return False
    ok = Fraction(stats["correct"], stats["n"]) >= minimum and Fraction(stats["parsed"], stats["n"]) >= Fraction(19, 20)
    if per_gold:
        ok = ok and all(Fraction(g["correct"], g["n"]) >= Fraction(3, 4) for g in stats["by_gold"].values())
    return ok


def contrast_ok(c, max_flip=Fraction(1, 8)):
    return c["unknown_pairs"] == 0 and c["known_pairs"] > 0 and Fraction(c["changed"], c["known_pairs"]) <= max_flip


def select(views, **factors):
    return [v for v in views if all(v.get(k) == val for k, val in factors.items())]


def v4_analysis(views):
    """Reimplementation of the frozen v4 descriptive cells, contrasts and readiness gate."""
    cells, contrasts = {}, {}
    for axis in ("truth", "validity"):
        for ctx in ("full", "relevant"):
            for o in ORDERS:
                cells[f"legacy_single/{axis}/{ctx}/{o}"] = cell_stats(select(views, section="legacy_single", axis=axis, context=ctx, option_order=o))
            contrasts[f"enum/legacy/{axis}/{ctx}"] = contrast(
                select(views, section="legacy_single", axis=axis, context=ctx, option_order=ORDERS[0]),
                select(views, section="legacy_single", axis=axis, context=ctx, option_order=ORDERS[1]))
        for o in ORDERS:
            contrasts[f"context/{axis}/{o}"] = contrast(
                select(views, section="legacy_single", axis=axis, context="full", option_order=o),
                select(views, section="legacy_single", axis=axis, context="relevant", option_order=o))
            cells[f"extension/{axis}/{o}"] = cell_stats(select(views, section="extension", axis=axis, option_order=o))
        contrasts[f"enum/extension/{axis}"] = contrast(select(views, section="extension", axis=axis, option_order=ORDERS[0]),
                                                       select(views, section="extension", axis=axis, option_order=ORDERS[1]))
    for qo in ("truth_first", "validity_first"):
        for o in ORDERS:
            cells[f"legacy_joint/{qo}/{o}"] = cell_stats(select(views, section="legacy_joint", question_order=qo, option_order=o))
        contrasts[f"enum/joint/{qo}"] = contrast(select(views, section="legacy_joint", question_order=qo, option_order=ORDERS[0]),
                                                 select(views, section="legacy_joint", question_order=qo, option_order=ORDERS[1]))
    for o in ORDERS:
        contrasts[f"question_order/{o}"] = contrast(select(views, section="legacy_joint", question_order="truth_first", option_order=o),
                                                    select(views, section="legacy_joint", question_order="validity_first", option_order=o))
    sanity = cell_stats(select(views, section="sanity"))
    gates = {}
    for axis in ("truth", "validity"):
        cell_names = [f"legacy_single/{axis}/relevant/{o}" for o in ORDERS] + [f"extension/{axis}/{o}" for o in ORDERS]
        contrast_names = [f"enum/legacy/{axis}/relevant", f"enum/extension/{axis}"]
        cc = {k: cell_ok(cells[k]) for k in cell_names}
        sc = {k: contrast_ok(contrasts[k]) for k in contrast_names}
        gates[axis] = {"passed": sanity["correct"] == 8 and all(cc.values()) and all(sc.values()),
                       "cell_checks": cc, "stability_checks": sc}
    return {"cells": cells, "contrasts": contrasts, "sanity": sanity, "gates": gates,
            "joint_task_ready": all(g["passed"] for g in gates.values())}


SOURCE_ORDER = ("legacy", "extension", "opened_v5", "opened_v7", "opened_v10", "heldout", "heldout_v7", "heldout_v10", "heldout_v14")


def clean_analysis(views, per_gold=True):
    """Reimplementation of the v6+ per-source/axis/order gate (87.5% accuracy, 95% parse,
    75% per gold label, zero unknown pairs, <=12.5% enumeration flips)."""
    sources = [s for s in SOURCE_ORDER if any(v["source"] == s for v in views)]
    cells, contrasts, gates = {}, {}, {}
    for axis in ("truth", "validity"):
        ac, ct = {}, {}
        for s in sources:
            arms = []
            for o in ORDERS:
                arm = [v for v in views if v["axis"] == axis and v["source"] == s and v["option_order"] == o]
                arms.append(arm)
                ac[f"{s}/{axis}/{o}"] = cell_stats(arm)
            ct[f"{s}/{axis}"] = contrast(*arms)
        cc = {k: cell_ok(v, per_gold=per_gold) for k, v in ac.items()}
        sc = {k: contrast_ok(v) for k, v in ct.items()}
        gates[axis] = {"passed": bool(ac) and bool(ct) and all(cc.values()) and all(sc.values()),
                       "failed_cell_checks": [k for k, ok in cc.items() if not ok],
                       "failed_stability_checks": [k for k, ok in sc.items() if not ok]}
        cells.update(ac)
        contrasts.update(ct)
    return {"sources": sources, "cells": cells, "contrasts": contrasts, "gates": gates,
            "passed": all(g["passed"] for g in gates.values())}


def screen_analysis(views):
    formats = ("direct", "reasoning", "fewshot_direct", "fewshot_reasoning")
    out = {}
    for fmt in formats:
        cells, contrasts = {}, {}
        for axis in ("truth", "validity"):
            for o in ORDERS:
                cells[f"{fmt}/{axis}/{o}"] = cell_stats(select(views, format=fmt, axis=axis, option_order=o))
            contrasts[f"{fmt}/{axis}"] = contrast(select(views, format=fmt, axis=axis, option_order=ORDERS[0]),
                                                  select(views, format=fmt, axis=axis, option_order=ORDERS[1]))
        passed = all(cell_ok(c, minimum=Fraction(1)) for c in cells.values()) and all(contrast_ok(c, max_flip=Fraction(0)) for c in contrasts.values())
        out[fmt] = {"passed": passed, "correct": sum(c["correct"] for c in cells.values()),
                    "n": sum(c["n"] for c in cells.values()), "cells": cells, "contrasts": contrasts}
    selected = next((f for f in formats if out[f]["passed"]), None)
    return {"formats": out, "selected_first_passing": selected}


# --------------------------------------------------------------------------- per-run summary
def outcome_table(evals):
    table = {"all": Counter(), "truth": Counter(), "validity": Counter(), "joint": Counter(), "literal": Counter()}
    for e in evals.values():
        table["all"][e["outcome"]] += 1
        table.setdefault(e["axis"], Counter())[e["outcome"]] += 1
    return {k: dict(v) for k, v in table.items() if v}


def noncorrect_listing(evals, include_unknown=True, limit=40):
    rows = []
    for e in sorted(evals.values(), key=lambda x: (x["axis"], x["item_id"], x["option_order"])):
        if e["outcome"] == "correct" or (e["outcome"] == "unknown" and not include_unknown):
            continue
        rows.append({"item_id": e["item_id"], "axis": e["axis"], "option_order": e["option_order"], "source": e["source"],
                     "family": e["family"], "gold": e["gold"], "parsed": e["parsed"], "outcome": e["outcome"],
                     "finish_reason": e["finish"], "output_tokens": e["output_tokens"], "file": e["file"]})
    return rows[:limit], len(rows)


def stored_score_check(run, evals, kind):
    """Compare my parse/score with the project's stored per-sample score (samples.json)."""
    if run.samples is None:
        return None
    compared = mismatch_parsed = mismatch_correct = result_differs = 0
    examples = []
    for row in run.samples:
        key = mkey(row["request"]["messages"])
        e = evals.get(key)
        if e is None:
            continue
        compared += 1
        if "score" in row:
            stored_parsed, stored_correct = row["score"]["parsed"], row["score"]["correct"]
        elif "parsed" in row:
            stored_parsed, stored_correct = row["parsed"], row["correct"]
        else:  # v4 rows have no score; handled through labels.mechanical.json
            stored_parsed, stored_correct = None, None
            continue
        if stored_parsed != e["parsed"]:
            mismatch_parsed += 1
            if len(examples) < 5:
                examples.append({"file": e["file"], "stored": stored_parsed, "recount": e["parsed"]})
        if stored_correct != e["correct"]:
            mismatch_correct += 1
            if len(examples) < 5:
                examples.append({"file": e["file"], "stored_correct": stored_correct, "recount_correct": e["correct"]})
    gen_results = {mkey(g["request"]["messages"]): g["result"] for g in run.completed}
    for row in run.samples:
        key = mkey(row["request"]["messages"])
        if key in gen_results and gen_results[key] != row["result"]:
            result_differs += 1
    return {"rows_compared": compared, "parsed_mismatches": mismatch_parsed, "correct_mismatches": mismatch_correct,
            "samples_result_differs_from_generation_checkpoint": result_differs, "examples": examples}


def v4_labels_check(run, views):
    path = run.dir / "labels.mechanical.json"
    if not path.exists():
        return None
    stored = {c["case_id"]: c["score"] for c in read_json(path)}
    compared = mism = 0
    for v in views:
        s = stored.get(v["case_id"])
        if s is None:
            continue
        compared += 1
        if s["parsed"] != v["ev"]["parsed"] or s["correct"] != v["ev"]["correct"]:
            mism += 1
    return {"case_records_compared": compared, "mismatches": mism}


def saved_summary_check(run, cells, contrasts):
    """Compare recount cells/contrasts with the saved summary.mechanical.json (derived file)."""
    if run.summary is None:
        return None
    saved_cells = dict(run.summary.get("cells", {}))
    if "sanity" in run.summary and isinstance(run.summary["sanity"], dict) and "exact_correct" in run.summary["sanity"]:
        saved_cells.setdefault("sanity", run.summary["sanity"])
    out = {"cells_compared": 0, "cell_mismatches": [], "contrasts_compared": 0, "contrast_mismatches": []}
    for name, s in saved_cells.items():
        mine = cells.get(name)
        if mine is None:
            continue
        out["cells_compared"] += 1
        if (s["exact_correct"], s["unique_prompts"], s["unknown_or_excluded"]) != (mine["correct"], mine["n"], mine["unknown_or_excluded"]):
            out["cell_mismatches"].append({"cell": name, "saved": [s["exact_correct"], s["unique_prompts"], s["unknown_or_excluded"]],
                                           "recount": [mine["correct"], mine["n"], mine["unknown_or_excluded"]]})
    for name, s in run.summary.get("contrasts", {}).items():
        mine = contrasts.get(name)
        if mine is None:
            continue
        out["contrasts_compared"] += 1
        if (s["changed_answers"], s["unique_prompt_pairs"], s["unknown_pairs"]) != (mine["changed"], mine["pairs"], mine["unknown_pairs"]):
            out["contrast_mismatches"].append({"contrast": name, "saved": [s["changed_answers"], s["unique_prompt_pairs"], s["unknown_pairs"]],
                                               "recount": [mine["changed"], mine["pairs"], mine["unknown_pairs"]]})
    return out


def gold_check(run):
    checked = derivable = mismatches = 0
    routes = Counter()
    examples = []
    for case in run.cases:
        checked += 1
        derived = derive_gold(case)
        if derived:
            derivable += 1
        for route, value in derived.items():
            routes[route] += 1
            if value != case["gold"]:
                mismatches += 1
                if len(examples) < 5:
                    examples.append({"case_id": case["case_id"], "item_id": case["item_id"], "route": route,
                                     "stored_gold": case["gold"], "derived": value})
    return {"case_records": checked, "independently_rederived": derivable, "by_route": dict(routes),
            "mismatches": mismatches, "examples": examples}


def run_block(run, kind):
    evals, views, problems = join_and_score(run, kind) if kind else ({}, [], [])
    statuses = Counter(g["status"] for g in run.generations)
    completed = run.completed
    block = {
        "run": run.name, "exists": run.exists, "listing": run.listing,
        "manifest_status": (run.manifest or {}).get("status"),
        "manifest_new_calls_last_invocation": (run.manifest or {}).get("new_calls", (run.manifest or {}).get("generation_calls_this_invocation")),
        "model": ({k: (completed[0]["model"] or {}).get(k) for k in ("model_id", "revision", "device", "dtype", "quantization", "backend")}
                  if completed else None),
        "design_case_records": len(run.cases), "design_unique_prompts": len(run.by_key), "design_source": run.cases_source,
        "generation_files": len(run.generations), "status_counts": dict(statuses),
        "attempts_field_sum": sum(g["attempts"] or 0 for g in run.generations),
        "completed": len(completed), "unfinished": len(run.generations) - len(completed),
        "finish_reasons": dict(Counter(((g["result"] or {}).get("metadata") or {}).get("finish_reason") for g in completed)),
        "errors": sum((g["result"] or {}).get("error") is not None for g in completed),
        "join_problems": problems,
        "completed_matched_to_design": len(evals),
        "design_prompts_without_completed_generation": len(run.by_key) - len(evals),
        "outcomes": outcome_table(evals) if kind else None,
        "gold_rederivation": gold_check(run) if run.cases else None,
        "samples_json_sha256": sha256_file(run.dir / "samples.json") if (run.dir / "samples.json").exists() else None,
    }
    return block, evals, views


# --------------------------------------------------------------------------- claims
CLAIMS = []


def claim(cid, section, text, expected, recomputed, basis, sources, note="", status=None):
    if status is None:
        status = "MATCH" if expected == recomputed else "CONFLICT"
    CLAIMS.append({"id": cid, "section": section, "claim": text, "expected": expected, "recomputed": recomputed,
                   "status": status, "basis": basis, "sources": sources, "note": note})


def cellfrac(cells, name):
    c = cells[name]
    return frac_str(c["correct"], c["n"])


# --------------------------------------------------------------------------- main analysis
def analyse(project):
    out = {"meta": {"script": Path(__file__).name, "project": str(project), "python": platform.python_version(),
                    "numpy": np.__version__, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "read_only": True, "project_modules_imported": False, "model_inference": False}}
    runs = {}

    def get_run(name):
        if name not in runs:
            runs[name] = Run(project, name)
        return runs[name]

    # ---------------- Section 2.5: v4 calibration --------------------------------------
    s25 = {}
    v4_evals = {}
    for model, name in RUN_V4.items():
        run = get_run(name)
        block, evals, views = run_block(run, "v4json")
        v4_evals[model] = evals
        analysis = v4_analysis(views)
        cells = analysis["cells"]
        validity_single = {}
        for v in views:
            if v["section"] in ("legacy_single", "extension") and v["axis"] == "validity":
                validity_single.setdefault(v["key"], v)
        vlabels = Counter((v["ev"]["parsed"] or {}).get("validity", "UNPARSED") for v in validity_single.values())
        vlabels_by_order = {o: dict(Counter((v["ev"]["parsed"] or {}).get("validity", "UNPARSED")
                                            for v in validity_single.values() if v["option_order"] == o)) for o in ORDERS}
        joint = {}
        for qo in ("truth_first", "validity_first"):
            for o in ORDERS:
                rows = {v["key"]: v for v in select(views, section="legacy_joint", question_order=qo, option_order=o)}
                joint[f"{qo}/{o}"] = {"n": len(rows),
                                      "both_correct": sum(r["ev"]["exact"] is True for r in rows.values()),
                                      "validity_labels": dict(Counter((r["ev"]["parsed"] or {}).get("validity", "UNPARSED") for r in rows.values())),
                                      "truth_labels": dict(Counter((r["ev"]["parsed"] or {}).get("truth", "UNPARSED") for r in rows.values()))}
        alias_groups = Counter(len(v) for v in run.by_key.values())
        s25[model] = {
            **block,
            "aliases": len(run.cases) - len(run.by_key), "alias_group_sizes": dict(alias_groups),
            "labels_mechanical_check": v4_labels_check(run, views),
            "saved_summary_check": saved_summary_check(run, {**cells, "sanity": analysis["sanity"]}, analysis["contrasts"]),
            "saved_gate": (run.summary or {}).get("engineering_gate", {}).get("joint_task_ready"),
            "cells": {k: {"correct": c["correct"], "n": c["n"], "unknown": c["unknown_or_excluded"], "labels": c["labels"]} for k, c in cells.items()},
            "sanity": {"correct": analysis["sanity"]["correct"], "n": analysis["sanity"]["n"]},
            "contrasts": {k: {"changed": c["changed"], "pairs": c["pairs"], "unknown_pairs": c["unknown_pairs"],
                              "changed_by_field": c["changed_by_field"], "wrong_to_right": c["wrong_to_right"],
                              "right_to_wrong": c["right_to_wrong"]} for k, c in analysis["contrasts"].items()},
            "gates": analysis["gates"], "joint_task_ready": analysis["joint_task_ready"],
            "single_validity_unique_prompts": len(validity_single), "single_validity_labels": dict(vlabels),
            "single_validity_labels_by_order": vlabels_by_order, "joint": joint,
        }
    # identical requests across the two models
    reqs = {}
    for model, name in RUN_V4.items():
        reqs[model] = {json.dumps({"m": g["request"]["messages"], "s": g["request"]["seed"], "d": g["request"]["decoding"]},
                                  sort_keys=True, ensure_ascii=False) for g in get_run(name).completed}
    s25["same_requests_across_models"] = {"qwen2.5-1.5b": len(reqs["qwen2.5-1.5b"]), "qwen2.5-3b": len(reqs["qwen2.5-3b"]),
                                          "identical_sets": reqs["qwen2.5-1.5b"] == reqs["qwen2.5-3b"]}
    out["section_2_5"] = s25

    src25 = ["results/runs/20260914_qwen15b_measurement_calibration_v4/{cases.json,generations/}",
             "results/runs/20260914_qwen3b_measurement_replication_v4/{cases.json,generations/}"]
    for model in RUN_V4:
        m = s25[model]
        claim(f"2.5-{model}-records", "2.5", "264 condition records -> 232 unique real generations; 32 aliases (" + model + ")",
              {"case_records": 264, "unique_prompts": 232, "completed_generations": 232, "aliases": 32},
              {"case_records": m["design_case_records"], "unique_prompts": m["design_unique_prompts"],
               "completed_generations": m["completed"], "aliases": m["aliases"]}, "RAW", src25[0 if model == "qwen2.5-1.5b" else 1],
              note=f"alias groups: {m['alias_group_sizes']}; every case joined to exactly one generation by exact prompt text")
        claim(f"2.5-{model}-eos", "2.5", "all generations ended with EOS (" + model + ")", {"eos": 232},
              m["finish_reasons"], "RAW", src25[0 if model == "qwen2.5-1.5b" else 1])
    m = s25["qwen2.5-1.5b"]
    c = m["cells"]
    claim("2.5-1.5b-legacy-truth-relevant", "2.5", "1.5B old truth items, irrelevant argument removed: default 8/8, reversed 6/8",
          ["8/8", "6/8"], [frac_str(c["legacy_single/truth/relevant/positive_first"]["correct"], c["legacy_single/truth/relevant/positive_first"]["n"]),
                           frac_str(c["legacy_single/truth/relevant/negative_first"]["correct"], c["legacy_single/truth/relevant/negative_first"]["n"])],
          "RAW", src25[0])
    claim("2.5-1.5b-ext-truth", "2.5", "1.5B new truth items 14/16 vs 8/16", ["14/16", "8/16"],
          [frac_str(c["extension/truth/positive_first"]["correct"], c["extension/truth/positive_first"]["n"]),
           frac_str(c["extension/truth/negative_first"]["correct"], c["extension/truth/negative_first"]["n"])], "RAW", src25[0])
    claim("2.5-1.5b-validity-all-invalid", "2.5", "1.5B: 80 distinct single-item validity prompts, all INVALID",
          {"unique_prompts": 80, "INVALID": 80}, {"unique_prompts": m["single_validity_unique_prompts"], **m["single_validity_labels"]}, "RAW", src25[0])
    claim("2.5-1.5b-sanity", "2.5", "1.5B basic copy/addition 8/8", "8/8", frac_str(m["sanity"]["correct"], m["sanity"]["n"]), "RAW", src25[0])
    claim("2.5-1.5b-gate", "2.5", "1.5B two-axis screen failed",
          {"truth_passed": False, "validity_passed": False, "joint_task_ready": False},
          {"truth_passed": m["gates"]["truth"]["passed"], "validity_passed": m["gates"]["validity"]["passed"], "joint_task_ready": m["joint_task_ready"]},
          "RAW", src25[0], note="gate re-implemented from analyze_calibration_v4 rules; saved summary joint_task_ready=" + str(m["saved_gate"]))
    claim("2.5-1.5b-tests", "2.5", "1.5B: 168 tests passed", 168, None, "NOT VERIFIABLE", "docs/MEASUREMENT_CALIBRATION_REPORT_V4_2026-09-14.md",
          note="no saved test record (JUnit/receipt) for v4 in the snapshot; only the report/README text states it; re-running tests is out of scope and the v16 suite has changed since",
          status="NOT VERIFIABLE")
    m3 = s25["qwen2.5-3b"]
    c3 = m3["cells"]
    claim("2.5-3b-same-requests", "2.5", "3B: same-request re-check (same 232 requests as 1.5B)",
          {"identical_sets": True, "n": 232}, {"identical_sets": s25["same_requests_across_models"]["identical_sets"],
                                               "n": s25["same_requests_across_models"]["qwen2.5-3b"]}, "RAW", "; ".join(src25),
          note="messages + seed + decoding compared request by request")
    claim("2.5-3b-ext-truth", "2.5", "3B new truth items 15/16 vs 12/16", ["15/16", "12/16"],
          [frac_str(c3["extension/truth/positive_first"]["correct"], c3["extension/truth/positive_first"]["n"]),
           frac_str(c3["extension/truth/negative_first"]["correct"], c3["extension/truth/negative_first"]["n"])], "RAW", src25[1])
    claim("2.5-3b-validity-labels", "2.5", "3B: of 80 single-item validity outputs 73 INVALID, 7 VALID",
          {"unique_prompts": 80, "INVALID": 73, "VALID": 7}, {"unique_prompts": m3["single_validity_unique_prompts"], **m3["single_validity_labels"]},
          "RAW", src25[1], note=f"by order: {m3['single_validity_labels_by_order']}")
    j = m3["joint"]
    claim("2.5-3b-joint-swap", "2.5", "3B default enum: swapping joint rule/field order moves validity 16/16 VALID -> 16/16 INVALID; both-correct 5/16 each",
          {"truth_first": {"VALID": 16}, "validity_first": {"INVALID": 16}, "both_correct": ["5/16", "5/16"]},
          {"truth_first": j["truth_first/positive_first"]["validity_labels"], "validity_first": j["validity_first/positive_first"]["validity_labels"],
           "both_correct": [frac_str(j["truth_first/positive_first"]["both_correct"], j["truth_first/positive_first"]["n"]),
                            frac_str(j["validity_first/positive_first"]["both_correct"], j["validity_first/positive_first"]["n"])]},
          "RAW", src25[1])
    claim("2.5-3b-gate", "2.5", "3B: both axes still fail",
          {"truth_passed": False, "validity_passed": False, "joint_task_ready": False},
          {"truth_passed": m3["gates"]["truth"]["passed"], "validity_passed": m3["gates"]["validity"]["passed"], "joint_task_ready": m3["joint_task_ready"]},
          "RAW", src25[1], note="saved summary joint_task_ready=" + str(m3["saved_gate"]))
    claim("2.5-3b-tests", "2.5", "3B: 187 tests passed", 187, None, "NOT VERIFIABLE", "docs/MODEL_REPLICATION_REPORT_V4_2026-09-14.md",
          note="no saved test record for v4; stated only in report/README", status="NOT VERIFIABLE")
    claim("2.5-same-family", "2.5", "only two checkpoints of the same series (Qwen2.5), not a cross-family validation",
          ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"], [s25["qwen2.5-1.5b"]["model"]["model_id"], s25["qwen2.5-3b"]["model"]["model_id"]],
          "RAW", "generation metadata")

    # ---- supplementary: numbers in the two named 2.5 source reports that the handoff summarizes
    cases15 = get_run(RUN_V4["qwen2.5-1.5b"]).cases
    xm = {}
    for axis in ("truth", "validity"):
        ks = sorted({c["key"] for c in cases15 if c["section"] == "extension" and c["axis"] == axis and c["option_order"] == "positive_first"})
        e15, e3 = v4_evals["qwen2.5-1.5b"], v4_evals["qwen2.5-3b"]
        xm[axis] = {"n": len(ks), "wrong_to_right": sum(e15[k]["exact"] is False and e3[k]["exact"] is True for k in ks),
                    "right_to_wrong": sum(e15[k]["exact"] is True and e3[k]["exact"] is False for k in ks)}
    s25["cross_model_extension_default_order"] = xm
    rep15, rep3 = "docs/MEASUREMENT_CALIBRATION_REPORT_V4_2026-09-14.md", "docs/MODEL_REPLICATION_REPORT_V4_2026-09-14.md"

    def cf(model, name):
        cc = s25[model]["cells"][name]
        return frac_str(cc["correct"], cc["n"])

    def ch(model, name):
        cc = s25[model]["contrasts"][name]
        return frac_str(cc["changed"], cc["pairs"])

    for model, rep in (("qwen2.5-1.5b", rep15), ("qwen2.5-3b", rep3)):
        tag = "1.5b" if model == "qwen2.5-1.5b" else "3b"
        exp_cells = ({"truth_full": ["9/16", "9/16"], "truth_relevant": ["8/8", "6/8"], "truth_ext": ["14/16", "8/16"],
                      "validity_full": ["8/16", "8/16"], "validity_relevant": ["4/8", "4/8"], "validity_ext": ["8/16", "8/16"],
                      "joint": ["5/16", "4/16", "4/16", "4/16"]} if tag == "1.5b" else
                     {"truth_full": ["11/16", "10/16"], "truth_relevant": ["5/8", "4/8"], "truth_ext": ["15/16", "12/16"],
                      "validity_full": ["9/16", "8/16"], "validity_relevant": ["5/8", "4/8"], "validity_ext": ["9/16", "8/16"],
                      "joint": ["5/16", "5/16", "5/16", "4/16"]})
        got_cells = {"truth_full": [cf(model, f"legacy_single/truth/full/{o}") for o in ORDERS],
                     "truth_relevant": [cf(model, f"legacy_single/truth/relevant/{o}") for o in ORDERS],
                     "truth_ext": [cf(model, f"extension/truth/{o}") for o in ORDERS],
                     "validity_full": [cf(model, f"legacy_single/validity/full/{o}") for o in ORDERS],
                     "validity_relevant": [cf(model, f"legacy_single/validity/relevant/{o}") for o in ORDERS],
                     "validity_ext": [cf(model, f"extension/validity/{o}") for o in ORDERS],
                     "joint": [cf(model, f"legacy_joint/{qo}/{o}") for qo in ("truth_first", "validity_first") for o in ORDERS]}
        claim(f"2.5src-{tag}-tables", "2.5 source", f"{tag} report tables (single-item cells default/reversed; joint both-correct TF-def, TF-rev, VF-def, VF-rev)",
              exp_cells, got_cells, "RAW", rep)
    claim("2.5src-1.5b-flips", "2.5 source", "1.5B enumeration changes: full-context truth 2/16, relevant truth 2/8, new truth 6/16",
          ["2/16", "2/8", "6/16"], [ch("qwen2.5-1.5b", "enum/legacy/truth/full"), ch("qwen2.5-1.5b", "enum/legacy/truth/relevant"),
                                    ch("qwen2.5-1.5b", "enum/extension/truth")], "RAW", rep15)
    ctx15 = s25["qwen2.5-1.5b"]["contrasts"]
    claim("2.5src-1.5b-context", "2.5 source", "1.5B context pairs (full -> relevant, 16 item pairs): 7 wrong->right default, 3 reversed",
          [7, 3], [ctx15["context/truth/positive_first"]["wrong_to_right"], ctx15["context/truth/negative_first"]["wrong_to_right"]], "RAW", rep15)
    claim("2.5src-3b-flips", "2.5 source", "3B enumeration changes: new truth 3/16, new validity 5/16, relevant truth 1/8, relevant validity 1/8",
          ["3/16", "5/16", "1/8", "1/8"], [ch("qwen2.5-3b", "enum/extension/truth"), ch("qwen2.5-3b", "enum/extension/validity"),
                                           ch("qwen2.5-3b", "enum/legacy/truth/relevant"), ch("qwen2.5-3b", "enum/legacy/validity/relevant")], "RAW", rep3)
    ctx3 = s25["qwen2.5-3b"]["contrasts"]
    claim("2.5src-3b-context", "2.5 source", "3B context pairs: accuracy falls 1/16 (default) and 2/16 (reversed) when the argument is removed",
          ["-1/16", "-2/16"], [f"{ctx3[f'context/truth/{o}']['wrong_to_right'] - ctx3[f'context/truth/{o}']['right_to_wrong']}/16" for o in ORDERS],
          "RAW", rep3)
    j3 = s25["qwen2.5-3b"]["joint"]
    claim("2.5src-3b-joint-labels", "2.5 source", "3B joint validity outputs: TF-def 16 VALID; TF-rev 11 VALID/5 INVALID; VF-def 16 INVALID; VF-rev 16 INVALID; field-order swap changes validity 16/16 (default), 11/16 (reversed)",
          {"labels": [{"VALID": 16}, {"VALID": 11, "INVALID": 5}, {"INVALID": 16}, {"INVALID": 16}], "validity_changes": ["16/16", "11/16"]},
          {"labels": [j3["truth_first/positive_first"]["validity_labels"], j3["truth_first/negative_first"]["validity_labels"],
                      j3["validity_first/positive_first"]["validity_labels"], j3["validity_first/negative_first"]["validity_labels"]],
           "validity_changes": [f"{ctx3[f'question_order/{o}']['changed_by_field'].get('validity', 0)}/{ctx3[f'question_order/{o}']['pairs']}" for o in ORDERS]},
          "RAW", rep3)
    claim("2.5src-cross-model", "2.5 source", "1.5B -> 3B, default order: new truth 2 wrong->right, 1 right->wrong; new validity 3 wrong->right, 2 right->wrong",
          {"truth": [2, 1], "validity": [3, 2]}, {ax: [xm[ax]["wrong_to_right"], xm[ax]["right_to_wrong"]] for ax in ("truth", "validity")},
          "RAW", rep3)

    # ---------------- Section 2.6: per-attempt ledger ------------------------------------
    attempts = []
    all_evals = {}
    for aid, label, names, kind, gate_kind, handoff_completed in ATTEMPTS:
        entry = {"attempt": aid, "row": label, "runs": names, "handoff_completed": handoff_completed, "parser": kind}
        blocks, evals_all, views_all = [], {}, []
        for name in names:
            run = get_run(name)
            block, evals, views = run_block(run, kind)
            block["stored_score_check"] = stored_score_check(run, evals, kind) if kind else None
            blocks.append(block)
            evals_all.update({(name, k): v for k, v in evals.items()})
            views_all.extend(views)
            all_evals[name] = evals
        entry["completed"] = sum(b["completed"] for b in blocks)
        entry["unfinished"] = sum(b["unfinished"] for b in blocks)
        entry["planned_unique_prompts"] = sum(b["design_unique_prompts"] for b in blocks)
        entry["run_blocks"] = blocks
        entry["outcomes"] = outcome_table({i: e for i, e in enumerate(evals_all.values())}) if kind else None
        listing, total = noncorrect_listing({i: e for i, e in enumerate(evals_all.values())},
                                            include_unknown=gate_kind != "partial" and aid != "v7_dev")
        entry["noncorrect"] = listing
        entry["noncorrect_listed_total"] = total
        if gate_kind == "clean":
            analysis = clean_analysis(views_all)
            entry["gate"] = {"passed": analysis["passed"], "axes": analysis["gates"]}
            entry["cells"] = {k: {"correct": c["correct"], "n": c["n"], "unknown": c["unknown_or_excluded"], "by_gold": c["by_gold"]}
                              for k, c in analysis["cells"].items()}
            entry["contrasts"] = {k: {"changed": c["changed"], "pairs": c["pairs"], "unknown_pairs": c["unknown_pairs"],
                                      "changed_items": c["changed_items"], "unknown_items": c["unknown_items"]}
                                  for k, c in analysis["contrasts"].items()}
            entry["saved_summary_check"] = saved_summary_check(get_run(names[0]), analysis["cells"], analysis["contrasts"])
            entry["saved_decision_passed"] = ((get_run(names[0]).summary or {}).get("decision") or {}).get("passed")
        elif gate_kind == "v4gate":
            analysis = v4_analysis(views_all)
            entry["gate"] = {"original_v4_gate_passed": analysis["joint_task_ready"], "axes": analysis["gates"]}
            entry["cells"] = {k: {"correct": c["correct"], "n": c["n"], "unknown": c["unknown_or_excluded"]} for k, c in analysis["cells"].items()}
            entry["sanity"] = {"correct": analysis["sanity"]["correct"], "n": analysis["sanity"]["n"]}
            entry["contrasts"] = {k: {"changed": c["changed"], "pairs": c["pairs"], "unknown_pairs": c["unknown_pairs"]} for k, c in analysis["contrasts"].items()}
            entry["saved_summary_check"] = saved_summary_check(get_run(names[0]), {**analysis["cells"], "sanity": analysis["sanity"]}, analysis["contrasts"])
            entry["saved_decision_passed"] = ((get_run(names[0]).summary or {}).get("decision") or {}).get("original_v4_gate_passed")
            entry["same_prompts_as_v4"] = ({c["key"] for c in get_run(names[0]).cases} == set(get_run(RUN_V4["qwen2.5-1.5b"]).by_key))
        elif gate_kind == "screen":
            analysis = screen_analysis(views_all)
            entry["screen"] = {f: {"passed": a["passed"], "correct": a["correct"], "n": a["n"]} for f, a in analysis["formats"].items()}
            entry["selected_first_passing"] = analysis["selected_first_passing"]
            cells = {}
            contrasts = {}
            for a in analysis["formats"].values():
                cells.update(a["cells"])
                contrasts.update(a["contrasts"])
            entry["saved_summary_check"] = saved_summary_check(get_run(names[0]), cells, contrasts)
            sel_path = project / "results/designs/mlx_qwen7b_selected_format_v5.json"
            entry["saved_selection_record"] = read_json(sel_path).get("selected_format") if sel_path.exists() else None
        attempts.append(entry)
    out["attempts"] = attempts
    att = {a["attempt"]: a for a in attempts}

    # ---- runtime details (row 1)
    rt = {}
    texts = {}
    v4_texts = {mkey(g["request"]["messages"]): g["result"]["text"] for g in get_run(RUN_V4["qwen2.5-1.5b"]).completed}
    for p in RUNTIME_PROFILES:
        run = get_run(f"20260914_runtime_v5_{p}")
        ev_ = all_evals[run.name]
        texts[p] = {k: e["text"] for k, e in ev_.items()}
        rt[p] = {"n": len(ev_), "finish": dict(Counter(e["finish"] for e in ev_.values())),
                 "output_tokens": dict(Counter(e["output_tokens"] for e in ev_.values())),
                 "exact_correct": sum(e["exact"] is True for e in ev_.values()),
                 "same_text_as_v4_original": sum(v4_texts.get(k) == e["text"] for k, e in ev_.items())}
    keys = set(texts["mps_fp16_sdpa"])
    identical3 = sum(1 for k in keys if texts["mps_fp16_sdpa"][k] == texts["mps_bf16_eager"].get(k) == texts["cpu_fp32_eager"].get(k))
    rt["identical_text_fp16_sdpa_bf16_fp32"] = identical3
    rt["prompts"] = len(keys)
    att["runtime_v5"]["runtime_details"] = rt

    # ---- v11 rounding amendment (diagnostic probabilities)
    v11 = get_run("20260914_constrained_qwen3_4b_dev_v11")
    probs = []
    trace_ok = 0
    argmax_ok = 0
    for g in v11.completed:
        meta = g["result"]["metadata"]
        p = meta["choice_probabilities"]
        probs.append([p["YES"], p["NO"]])
        label = g["result"]["text"].strip()
        trace_ok += meta["constraint_steps"] == 3 and meta["finite_logit_checks"] == 3 and meta["output_tokens"] == 2 and meta["finish_reason"] == "eos"
        argmax_ok += p[label] + 0.001 >= max(p.values())
    arr = np.array(probs, dtype=np.float64)
    dev = np.abs(arr.sum(axis=1) - 1.0)
    att["v11_dev"]["probability_amendment"] = {
        "records": int(arr.shape[0]), "pairs_off_by_more_than_0.001": int((dev > 0.001).sum()),
        "max_abs_sum_deviation": float(dev.max()), "all_within_2^-7": bool((dev <= 2 ** -7).all()),
        "trace_ok_(3 steps, 3 finite checks, 2 tokens, eos)": int(trace_ok), "emitted_label_is_argmax_within_0.001": int(argmax_ok)}
    v10 = get_run("20260914_constrained_qwen3_4b_dev_v10")
    v10_ev = {}
    for g in v10.completed:
        key = mkey(g["request"]["messages"])
        v10_ev[key] = evaluate("constrained_v11", v10.by_key[key][0], g)
    att["v10_dev_partial"]["informational_not_a_claim"] = {
        "constraint_steps": dict(Counter(g["result"]["metadata"].get("constraint_steps") for g in v10.completed)),
        "outcomes_under_original_v10_rule": att["v10_dev_partial"]["outcomes"],
        "outcomes_if_scored_with_v11_accounting": dict(Counter(e["outcome"] for e in v10_ev.values())),
        "note": "v10 is an interrupted attempt; the handoff (correctly) reports no accuracy for it."}
    fmt3b = get_run("20260914_format_development_qwen3b_v5")
    fmt_counts = Counter()
    for g in fmt3b.completed:
        fmt_counts[fmt3b.by_key[mkey(g["request"]["messages"])][0]["format"]] += 1
    att["format_3b_v5"]["completed_by_format"] = dict(fmt_counts)
    att["format_3b_v5"]["planned_by_format"] = dict(Counter(c["format"] for c in fmt3b.cases))
    lf = get_run("20260914_mlx_qwen7b_screen_v5_load_failure")
    att["mlx_load_failure_v5"]["load_failure_record"] = read_json(lf.dir / "load_failure.json") if (lf.dir / "load_failure.json").exists() else None
    att["mlx_load_failure_v5"]["generations_dir_present"] = (lf.dir / "generations").is_dir()

    # ---- paired comparisons (v13 vs v9 union; v14 vs v13 union)
    def state3(e):
        return e["outcome"] if e["outcome"] in ("correct", "wrong") else "unknown"

    def paired(old_names, new_name):
        old = {}
        for n in old_names:
            old.update(all_evals[n])
        new = all_evals[new_name]
        same = set(old) == set(new)
        common = set(old) & set(new)
        trans = Counter(f"{state3(old[k])} -> {state3(new[k])}" for k in common)
        trans4 = Counter(f"{old[k]['outcome']} -> {new[k]['outcome']}" for k in common)
        changed = [{"item_id": new[k]["item_id"], "option_order": new[k]["option_order"], "axis": new[k]["axis"],
                    "old": old[k]["outcome"], "new": new[k]["outcome"]} for k in sorted(common) if state3(old[k]) != state3(new[k])]
        return {"old_prompts": len(old), "new_prompts": len(new), "same_prompt_set": same,
                "same_seed": all(old[k]["seed"] == new[k]["seed"] for k in common),
                "same_decoding": all(old[k]["decoding"] == new[k]["decoding"] for k in common),
                "old_correct": sum(state3(e) == "correct" for e in old.values()),
                "new_correct": sum(state3(e) == "correct" for e in new.values()),
                "transitions": dict(trans), "transitions_with_truncation": dict(trans4), "changed": changed}

    out["paired_v13_vs_v9"] = paired(["20260914_guided_qwen3_4b_dev_v9", "20260914_guided_qwen3_4b_holdout_v9"], "20260914_guided_qwen3_4b_8bit_dev_v13")
    out["paired_v14_vs_v13"] = paired(["20260914_guided_qwen3_4b_8bit_dev_v13", "20260914_guided_qwen3_4b_8bit_holdout_v13"], "20260914_repetition_qwen3_8bit_dev_v14")

    # ---- totals and counting conventions
    completed_total = sum(a["completed"] for a in attempts)
    unfinished_total = sum(a["unfinished"] for a in attempts)
    reg = get_run("20260914_mlx_qwen7b_regression_v5")
    running_sum = []
    acc = 0
    for a in attempts:
        acc += a["completed"]
        running_sum.append({"after": a["attempt"], "cumulative_completed": acc})
    all_2026_0914 = sorted(p.name for p in (project / "results/runs").iterdir() if p.name.startswith("20260914_"))
    v5_v14_dirs = [n for n in all_2026_0914 if not re.search(r"_v[34]$", n)]
    listed = [n for a in ATTEMPTS for n in a[2]]
    out["totals"] = {
        "convention_A_completed_checkpoints": completed_total,
        "unfinished_checkpoints": unfinished_total,
        "convention_A_definition": "count generation checkpoint files with status=='completed' (one per real model call; aliases share one call; truncated calls count; interrupted attempts count only their completed calls; load failure = 0)",
        "alt_B_count_7B_regression_case_records_instead_of_unique_calls": completed_total - att["mlx_regression_v5"]["completed"] + len(reg.cases),
        "alt_C_include_unfinished_checkpoints": completed_total + unfinished_total,
        "alt_D_exclude_interrupted_attempts": completed_total - att["format_3b_v5"]["completed"] - att["v10_dev_partial"]["completed"],
        "alt_E_planned_requests_of_interrupted_attempts": completed_total - att["format_3b_v5"]["completed"] - att["v10_dev_partial"]["completed"]
                                                           + att["format_3b_v5"]["planned_unique_prompts"] + att["v10_dev_partial"]["planned_unique_prompts"],
        "cumulative": running_sum,
        "run_dirs_20260914_not_v3_v4": v5_v14_dirs,
        "all_v5_v14_dirs_covered_by_ledger": sorted(v5_v14_dirs) == sorted(listed),
    }
    # saved audit counters (derived files) vs recount partial sums
    cum = {r["after"]: r["cumulative_completed"] for r in running_sum}
    an = project / "results/analyses"
    saved_counters = {}
    for fname, field, at in (("measurement_repair_v6_final_audit.json", "completed_generations_this_repair_round", "v6_holdout"),
                             ("measurement_repair_v7_failure_audit.json", "completed_generations_through_v7", "v7_dev"),
                             ("measurement_repair_v8_failure_audit.json", "completed_generations_through_v8", "v8_dev"),
                             ("measurement_repair_v9_failure_audit.json", "completed_generations_this_repair_round", "v9_holdout"),
                             ("measurement_repair_through_v11_audit.json", "completed_generations_through_v11", "v11_dev"),
                             ("measurement_repair_v12_failure_audit.json", "completed_generations_this_repair_round", "v12_dev"),
                             ("measurement_repair_v13_failure_audit.json", "completed_generations_this_repair_round", "v13_holdout"),
                             ("measurement_repair_v14_final_audit.json", "completed_generations_this_repair_round", "v14_holdout")):
        data = read_json(an / fname)
        saved_counters[fname] = {"saved": data.get(field), "saved_unfinished": data.get("unfinished_generations"),
                                 "recount_cumulative": cum[at], "match": data.get(field) == cum[at]}
    out["totals"]["saved_audit_counters"] = saved_counters

    # ---- banks opened / timeline / tests / hashes
    banks = []
    for name in v5_v14_dirs:
        run = get_run(name)
        inputs = (run.manifest or {}).get("inputs", {})
        if inputs.get("phase") == "holdout":
            banks.append({"run": name, "created_utc": run.manifest.get("created_utc"),
                          "sources": sorted({c["source"] for c in run.cases}), "format": inputs.get("format"),
                          "selection_locked_utc": (inputs.get("selection") or {}).get("created_utc")})
    banks.sort(key=lambda b: b["created_utc"])
    dev_sources = {name: sorted({c["source"] for c in get_run(name).cases})
                   for name in v5_v14_dirs
                   if ((get_run(name).manifest or {}).get("inputs", {}).get("phase") in ("development", "screen"))}
    out["banks"] = {"holdout_runs": banks, "development_sources": dev_sources}
    designs = project / "results/designs"
    out["v14_timeline"] = {
        "bank_committed": read_json(designs / "heldout_v14_commitment.json")["created_utc"],
        "development_started": get_run("20260914_repetition_qwen3_8bit_dev_v14").manifest["created_utc"],
        "selection_locked": read_json(designs / "repetition_qwen3_8bit_locked_selection_v14.json")["created_utc"],
        "validation_started": get_run("20260914_repetition_qwen3_8bit_holdout_v14").manifest["created_utc"],
    }
    tests = {}
    receipt_path = project / "results/verification/test_receipt_v14.json"
    receipt = read_json(receipt_path) if receipt_path.exists() else {}
    for tag in ("v12", "v13", "v14"):
        for env in ("mlx", "original"):
            path = project / f"results/verification/pytest_{env}_{tag}.xml"
            if not path.exists():
                continue
            root = ET.parse(path).getroot()
            suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
            testcases = list(root.iter("testcase"))
            bad = sum(1 for t in testcases if t.find("failure") is not None or t.find("error") is not None)
            skipped = sum(1 for t in testcases if t.find("skipped") is not None)
            entry = {"testcase_elements": len(testcases), "declared_tests": sum(int(s.get("tests", 0)) for s in suites),
                     "failures_or_errors": bad, "skipped": skipped, "sha256": sha256_file(path)}
            if tag == "v14" and path.name in receipt:
                entry["receipt_sha256_match"] = receipt[path.name]["sha256"] == entry["sha256"]
            tests[f"{env}_{tag}"] = entry
    out["junit"] = tests
    hashes = {}
    for name, (expected, source) in DOC_HASHES.items():
        path = project / "results/runs" / name / "samples.json"
        actual = sha256_file(path) if path.exists() else None
        hashes[name] = {"doc_or_audit_value": expected, "source": source, "actual": actual, "match": actual == expected}
    out["samples_hash_checks"] = hashes
    v14audit = read_json(an / "measurement_repair_v14_final_audit.json")
    out["v14_final_audit_decision"] = v14audit.get("decision")

    # ---- v6 serialization incident support: certificate containers now JSON lists
    v6h = get_run("20260914_clean_direct_qwen7b_holdout_v6")
    out["v6_incident"] = {"cases_with_values_list_certificate": sum(1 for c in v6h.cases if isinstance(c["certificate"], dict)
                                                                    and isinstance(c["certificate"].get("values"), list)),
                          "holdout_samples_sha256_equals_prefix_hash": hashes["20260914_clean_direct_qwen7b_holdout_v6"]["match"]}

    # ---------------- 2.6 claims --------------------------------------------------------
    S = "docs/MEASUREMENT_REPAIR_REPORT_2026-09-14.md"

    def oc(aid):
        return att[aid]["outcomes"]["all"]

    def fr(aid):
        o = oc(aid)
        return f"{o.get('correct', 0)}/{att[aid]['completed']}"

    claim("2.6-total", "2.6", "v5-v14: 2,879 completed real generations plus 2 unfinished checkpoints",
          {"completed": 2879, "unfinished": 2}, {"completed": completed_total, "unfinished": unfinished_total}, "RAW",
          "results/runs/20260914_* (v5-v14 attempt dirs)/generations/*.json",
          note="convention A: completed checkpoint files; equals the plain sum of the table's 'completed calls' column")
    r = att["runtime_v5"]["runtime_details"]
    claim("2.6-row1", "2.6", "1.5B four runtime/numeric configs: 80; CPU FP32, MPS BF16, orig FP16 SDPA identical text on 20 prompts; FP16 eager all truncated",
          {"completed": 80, "identical_3_profiles": 20, "fp16_eager_truncated": 20},
          {"completed": att["runtime_v5"]["completed"], "identical_3_profiles": r["identical_text_fp16_sdpa_bf16_fp32"],
           "fp16_eager_truncated": r["mps_fp16_eager"]["finish"].get("length", 0)}, "RAW",
          "results/runs/20260914_runtime_v5_*/generations/", note=f"same text as the original v4 run: { {p: r[p]['same_text_as_v4_original'] for p in RUNTIME_PROFILES} }; exact-correct per profile: { {p: r[p]['exact_correct'] for p in RUNTIME_PROFILES} }")
    a = att["format_3b_v5"]
    claim("2.6-row2", "2.6", "3B BF16 four formats: 34/64, interrupted, plus 1 unfinished checkpoint",
          {"completed": 34, "planned": 64, "unfinished": 1}, {"completed": a["completed"], "planned": a["planned_unique_prompts"], "unfinished": a["unfinished"]},
          "RAW", "results/runs/20260914_format_development_qwen3b_v5/generations/ + manifest", note=f"manifest status={a['run_blocks'][0]['manifest_status']}; model {a['run_blocks'][0]['model']}")
    a = att["mlx_load_failure_v5"]
    claim("2.6-row3", "2.6", "7B MLX first load: 0 calls, failed before generation", 0,
          a["completed"], "RAW", "results/runs/20260914_mlx_qwen7b_screen_v5_load_failure/",
          note=f"no generations/ directory ({a['generations_dir_present']}); load_failure.json generation_calls={(a['load_failure_record'] or {}).get('generation_calls')}")
    a = att["mlx_screen_v5"]
    claim("2.6-row4", "2.6", "7B format screen: 64; direct and reasoning 16/16 each; direct chosen by fixed order",
          {"completed": 64, "direct": "16/16", "reasoning": "16/16", "selected": "direct"},
          {"completed": a["completed"], "direct": frac_str(a["screen"]["direct"]["correct"], a["screen"]["direct"]["n"]),
           "reasoning": frac_str(a["screen"]["reasoning"]["correct"], a["screen"]["reasoning"]["n"]), "selected": a["selected_first_passing"]},
          "RAW", "results/runs/20260914_mlx_qwen7b_screen_v5/", note=f"all formats: { {f: (v['correct'], v['n'], v['passed']) for f, v in a['screen'].items()} }; saved selection record={a['saved_selection_record']}")
    a = att["mlx_regression_v5"]
    claim("2.6-row5", "2.6", "7B old-task regression: 232; two-axis gate failed",
          {"completed": 232, "gate_passed": False}, {"completed": a["completed"], "gate_passed": a["gate"]["original_v4_gate_passed"]},
          "RAW", "results/runs/20260914_mlx_qwen7b_regression_v5/", note=f"264 case records share 232 calls; truth passed={a['gate']['axes']['truth']['passed']}, validity passed={a['gate']['axes']['validity']['passed']}")
    for aid, row, exp_completed, exp_correct, exp_pass, extra in (
            ("v6_dev", "v6 dev direct: 96 = 94/96, passes development", 96, 94, True, None),
            ("v6_holdout", "v6 first sealed validation: 96 = 83/96, fails", 96, 83, False, None),
            ("v8_dev", "v8 rules+balanced demos 7B: 192 = 180/192; group accuracy / order gate fails", 192, 180, False, None),
            ("v9_dev", "v9 Qwen3-4B 4-bit dev: 192 = 190/192, passes development", 192, 190, True, None),
            ("v9_holdout", "v9 second validation: 96 = 94/96, other 2 = same subtraction item looping to truncation; unknown gate fails", 96, 94, False, None),
            ("v11_dev", "v11 corrected callback count: 288 = 266/288, no unknown; group gate fails", 288, 266, False, None),
            ("v13_dev", "v13 8-bit same-request dev: 288 = 284/288, passes development", 288, 284, True, None),
            ("v13_holdout", "v13 third validation: 96 = 94/96, 1 validity error, 1 subtraction loop truncation; unknown gate fails", 96, 94, False, None),
            ("v14_dev", "v14 repetition penalty dev: 384 = 378/384, zero unknown, all groups pass", 384, 378, True, None),
            ("v14_holdout", "v14 fourth bank validation: 96 = 94/96, zero unknown, all groups pass", 96, 94, True, None)):
        a = att[aid]
        o = oc(aid)
        claim(f"2.6-{aid}", "2.6", row, {"completed": exp_completed, "correct": exp_correct, "gate_passed": exp_pass},
              {"completed": a["completed"], "correct": o.get("correct", 0), "gate_passed": a["gate"]["passed"]}, "RAW",
              f"results/runs/{a['runs'][0]}/generations/ + manifest cases",
              note=f"outcomes={o}; failed checks: " + json.dumps({ax: g["failed_cell_checks"] + g["failed_stability_checks"] for ax, g in a["gate"]["axes"].items()})
              + f"; saved decision passed={a.get('saved_decision_passed')}")
    a = att["v7_dev"]
    o = oc("v7_dev")
    claim("2.6-v7_dev", "2.6", "v7 short reasoning dev: 192 = 22 correct, 4 wrong, 166 format-unknown; fails",
          {"completed": 192, "correct": 22, "wrong": 4, "unknown": 166, "gate_passed": False},
          {"completed": a["completed"], "correct": o.get("correct", 0), "wrong": o.get("wrong", 0),
           "unknown": o.get("unknown", 0) + o.get("truncated", 0), "gate_passed": a["gate"]["passed"]}, "RAW",
          "results/runs/20260914_reasoning_qwen7b_dev_v7/", note=f"truncated={o.get('truncated', 0)}")
    a = att["v10_dev_partial"]
    claim("2.6-v10", "2.6", "v10 constrained first version: 165/288 plus 1 unfinished checkpoint; no full accuracy",
          {"completed": 165, "planned": 288, "unfinished": 1}, {"completed": a["completed"], "planned": a["planned_unique_prompts"], "unfinished": a["unfinished"]},
          "RAW", "results/runs/20260914_constrained_qwen3_4b_dev_v10/", note=f"manifest status={a['run_blocks'][0]['manifest_status']}; every completed record has constraint_steps={a['informational_not_a_claim']['constraint_steps']} (the v10 scorer required 2)")
    a = att["v12_dev"]
    o = a["outcomes"]
    claim("2.6-v12", "2.6", "v12 axis demos free generation: 288 = 272/288, 2 truncated, 3 other truth errors, 11 validity errors; fails",
          {"completed": 288, "correct": 272, "truncated": 2, "truth_wrong": 3, "validity_wrong": 11, "gate_passed": False},
          {"completed": a["completed"], "correct": o["all"].get("correct", 0), "truncated": o["all"].get("truncated", 0),
           "truth_wrong": o.get("truth", {}).get("wrong", 0), "validity_wrong": o.get("validity", {}).get("wrong", 0), "gate_passed": a["gate"]["passed"]},
          "RAW", "results/runs/20260914_axis_guided_qwen3_4b_dev_v12/", note=f"truncated by axis: truth={o.get('truth', {}).get('truncated', 0)}, validity={o.get('validity', {}).get('truncated', 0)}; unknown={o['all'].get('unknown', 0)}")
    # supplementary bullets
    p13 = out["paired_v13_vs_v9"]
    claim("2.6-v13-vs-v9", "2.6", "v13 4-bit/8-bit both 284 correct on the same 288 requests; unknown->correct and correct->wrong both occur",
          {"same_requests": True, "old_correct": 284, "new_correct": 284, "has_unknown_to_correct": True, "has_correct_to_wrong": True},
          {"same_requests": p13["same_prompt_set"] and p13["same_seed"] and p13["same_decoding"], "old_correct": p13["old_correct"],
           "new_correct": p13["new_correct"], "has_unknown_to_correct": p13["transitions"].get("unknown -> correct", 0) > 0,
           "has_correct_to_wrong": p13["transitions"].get("correct -> wrong", 0) > 0}, "RAW",
          "v9 dev+holdout and v13 dev generations/", note=f"transitions={p13['transitions']}")
    p14 = out["paired_v14_vs_v13"]
    claim("2.6-v14-vs-v13", "2.6", "v14 vs the matching 384 v13 requests: total correct still 378; one truncation -> correct, one correct -> wrong",
          {"same_requests": True, "old_correct": 378, "new_correct": 378, "truncated_to_correct": 1, "correct_to_wrong": 1},
          {"same_requests": p14["same_prompt_set"] and p14["same_seed"] and p14["same_decoding"], "old_correct": p14["old_correct"],
           "new_correct": p14["new_correct"], "truncated_to_correct": p14["transitions_with_truncation"].get("truncated -> correct", 0),
           "correct_to_wrong": p14["transitions_with_truncation"].get("correct -> wrong", 0)}, "RAW",
          "v13 dev+holdout and v14 dev generations/", note=f"transitions={p14['transitions_with_truncation']}")
    d14 = att["v14_dev"]["outcomes"]
    h14 = att["v14_holdout"]
    hc = h14["cells"]
    claim("2.6-v14-dev-axes", "2.6", "v14 development: truth 192/192, validity 186/192",
          ["192/192", "186/192"], [f"{d14['truth'].get('correct', 0)}/{sum(d14['truth'].values())}",
                                   f"{d14['validity'].get('correct', 0)}/{sum(d14['validity'].values())}"], "RAW",
          "results/runs/20260914_repetition_qwen3_8bit_dev_v14/")
    claim("2.6-v14-holdout-axes", "2.6", "v14 fourth bank: truth 24/24 in each order; validity 22/24 and 24/24; validity enumeration flips 2/24",
          {"truth": ["24/24", "24/24"], "validity": ["22/24", "24/24"], "validity_flips": "2/24"},
          {"truth": [frac_str(hc["heldout_v14/truth/positive_first"]["correct"], hc["heldout_v14/truth/positive_first"]["n"]),
                     frac_str(hc["heldout_v14/truth/negative_first"]["correct"], hc["heldout_v14/truth/negative_first"]["n"])],
           "validity": [frac_str(hc["heldout_v14/validity/positive_first"]["correct"], hc["heldout_v14/validity/positive_first"]["n"]),
                        frac_str(hc["heldout_v14/validity/negative_first"]["correct"], hc["heldout_v14/validity/negative_first"]["n"])],
           "validity_flips": frac_str(h14["contrasts"]["heldout_v14/validity"]["changed"], h14["contrasts"]["heldout_v14/validity"]["pairs"])},
          "RAW", "results/runs/20260914_repetition_qwen3_8bit_holdout_v14/", note=f"flipped items {h14['contrasts']['heldout_v14/validity']['changed_items']}; weakest gold subgroup by_gold={hc['heldout_v14/validity/positive_first']['by_gold']}")
    pa = att["v11_dev"]["probability_amendment"]
    claim("2.6-v11-amendment", "2.6", "v11 post-hoc tolerance amendment for low-precision auxiliary probabilities; labels, termination, accuracy, failure unchanged",
          {"pairs_off_by_more_than_0.001": 13, "max_abs_sum_deviation": 0.0031890869140625, "labels_as_stored": True, "gate_passed": False},
          {"pairs_off_by_more_than_0.001": pa["pairs_off_by_more_than_0.001"], "max_abs_sum_deviation": pa["max_abs_sum_deviation"],
           "labels_as_stored": (att["v11_dev"]["run_blocks"][0]["stored_score_check"] or {}).get("parsed_mismatches") == 0
           and (att["v11_dev"]["run_blocks"][0]["stored_score_check"] or {}).get("correct_mismatches") == 0,
           "gate_passed": att["v11_dev"]["gate"]["passed"]}, "RAW", "docs/CONSTRAINED_V11_ROUNDING_AMENDMENT.md; v11 generations/ metadata",
          note=f"all within 2^-7: {pa['all_within_2^-7']}; trace ok {pa['trace_ok_(3 steps, 3 finite checks, 2 tokens, eos)']}/288; argmax ok {pa['emitted_label_is_argmax_within_0.001']}/288")
    v7s = att["v7_dev"]["run_blocks"][0]["stored_score_check"]
    v6s = [att[x]["run_blocks"][0]["stored_score_check"] for x in ("v6_dev", "v6_holdout")]
    claim("2.6-no-rescoring", "2.6", "v6 JSON tuple/list compatibility and v7 format problem retained; no post-hoc re-scoring of old failures",
          {"v6_stored_scores_equal_strict_parse": True, "v7_stored_scores_equal_strict_parse": True, "v7_unknown_retained": 166, "v6_holdout_raw_unchanged": True},
          {"v6_stored_scores_equal_strict_parse": all(s["parsed_mismatches"] == 0 and s["correct_mismatches"] == 0 for s in v6s),
           "v7_stored_scores_equal_strict_parse": v7s["parsed_mismatches"] == 0 and v7s["correct_mismatches"] == 0,
           "v7_unknown_retained": oc("v7_dev").get("unknown", 0), "v6_holdout_raw_unchanged": out["v6_incident"]["holdout_samples_sha256_equals_prefix_hash"]},
          "RAW", "v6/v7 generations + samples.json; docs/CLEAN_V6_SERIALIZATION_INCIDENT.md",
          note=f"v6 holdout cases whose certificate 'values' is a JSON list: {out['v6_incident']['cases_with_values_list_certificate']} (incident doc says 24)")
    hold_srcs = [b["sources"] for b in banks]
    claim("2.6-banks-opened", "2.6", "all four earlier validation banks have been opened (cannot be called unseen)",
          {"holdout_runs": 4, "banks": [["heldout"], ["heldout_v7"], ["heldout_v10"], ["heldout_v14"]]},
          {"holdout_runs": len(banks), "banks": hold_srcs}, "RAW", "manifests of the four holdout runs",
          note="v5 bank opened by v6 holdout, v7 bank by v9 holdout, v10 bank by v13 holdout, v14 bank by v14 holdout; development sources: "
               + json.dumps({k.replace("20260914_", ""): v for k, v in dev_sources.items()}))
    claim("2.6-v14-flags", "2.6", "v14: joint_task_validated=false, preference_effect_established=false, publication_ready=false",
          {"joint_task_validated": False, "preference_effect_established": False, "publication_ready": False},
          {k: (out["v14_final_audit_decision"] or {}).get(k) for k in ("joint_task_validated", "preference_effect_established", "publication_ready")},
          "DERIVED", "results/analyses/measurement_repair_v14_final_audit.json",
          note=f"same audit: scoped_measurement_ready={(out['v14_final_audit_decision'] or {}).get('scoped_measurement_ready')}; recount agrees both v14 gates pass")
    claim("2.6-v14-tests", "2.6", "v14: 269 tests passed in each of two environments",
          {"mlx": 269, "original": 269, "failures": 0},
          {"mlx": tests.get("mlx_v14", {}).get("testcase_elements"), "original": tests.get("original_v14", {}).get("testcase_elements"),
           "failures": tests.get("mlx_v14", {}).get("failures_or_errors", 0) + tests.get("original_v14", {}).get("failures_or_errors", 0)},
          "DERIVED", "results/verification/pytest_{mlx,original}_v14.xml + test_receipt_v14.json",
          note="saved JUnit records only (tests not re-run); receipt sha256 match: " + str({k: v.get("receipt_sha256_match") for k, v in tests.items() if k.endswith("v14")}))
    nz = {a["attempt"]: a["run_blocks"][0]["manifest_new_calls_last_invocation"] for a in attempts if a["run_blocks"][0]["manifest_status"] == "completed"}
    claim("2.6-v14-hashes-resume", "2.6", "v14 dev/validation raw hashes as recorded and zero-call resume on both",
          {"dev_hash_match": True, "holdout_hash_match": True, "dev_new_calls_last_invocation": 0, "holdout_new_calls_last_invocation": 0},
          {"dev_hash_match": hashes["20260914_repetition_qwen3_8bit_dev_v14"]["match"], "holdout_hash_match": hashes["20260914_repetition_qwen3_8bit_holdout_v14"]["match"],
           "dev_new_calls_last_invocation": nz.get("v14_dev"), "holdout_new_calls_last_invocation": nz.get("v14_holdout")},
          "DERIVED", "samples.json sha256 (computed) vs report; manifest new_calls",
          note="hash is computed from the file (RAW); zero-call resume only as recorded in the manifest")

    # ---- every counted call came from a real backend; the model named in each row
    model_rows = {}
    backends = Counter()
    for a in attempts:
        seen = Counter()
        for name in a["runs"]:
            for g in get_run(name).completed:
                mdl = g["model"] or {}
                backends[mdl.get("backend")] += 1
                q = mdl.get("quantization")
                seen[(mdl.get("backend"), mdl.get("model_id"), mdl.get("dtype"), json.dumps(q, sort_keys=True),
                      json.dumps({k: v for k, v in (mdl.get("repetition_control") or {}).items() if k != "source_sha256"}, sort_keys=True))] += 1
        model_rows[a["attempt"]] = [{"backend": k[0], "model_id": k[1], "dtype": k[2], "quantization": json.loads(k[3]),
                                     "repetition_control": json.loads(k[4]) or None, "calls": n} for k, n in seen.items()]
    out["models_by_attempt"] = model_rows
    out["totals"]["backends_across_completed"] = dict(backends)
    expected_models = {"runtime_v5": "Qwen/Qwen2.5-1.5B-Instruct", "format_3b_v5": "Qwen/Qwen2.5-3B-Instruct",
                       "mlx_screen_v5": "mlx-community/Qwen2.5-7B-Instruct-4bit", "mlx_regression_v5": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                       "v6_dev": "mlx-community/Qwen2.5-7B-Instruct-4bit", "v6_holdout": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                       "v7_dev": "mlx-community/Qwen2.5-7B-Instruct-4bit", "v8_dev": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                       "v9_dev": "mlx-community/Qwen3-4B-Instruct-2507-4bit", "v9_holdout": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
                       "v10_dev_partial": "mlx-community/Qwen3-4B-Instruct-2507-4bit", "v11_dev": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
                       "v12_dev": "mlx-community/Qwen3-4B-Instruct-2507-4bit", "v13_dev": "mlx-community/Qwen3-4B-Instruct-2507-8bit",
                       "v13_holdout": "mlx-community/Qwen3-4B-Instruct-2507-8bit", "v14_dev": "mlx-community/Qwen3-4B-Instruct-2507-8bit",
                       "v14_holdout": "mlx-community/Qwen3-4B-Instruct-2507-8bit"}
    claim("2.6-models", "2.6", "row models: 1.5B runtime, 3B BF16, 7B MLX (v5-v8), Qwen3-4B 4-bit (v9-v12), Qwen3-4B 8-bit (v13-v14); v14 adds repetition penalty 1.10 over 64 generated tokens; all calls from real backends",
          {"models": expected_models, "3b_dtype": "bfloat16", "v14_repetition": {"factor": 1.1, "window_generated_tokens": 64, "prompt_tokens_penalized": False, "gold_access": False},
           "fake_backend_calls": 0},
          {"models": {aid: rows[0]["model_id"] for aid, rows in model_rows.items() if rows and len({r["model_id"] for r in rows}) == 1 and aid in expected_models},
           "3b_dtype": model_rows["format_3b_v5"][0]["dtype"], "v14_repetition": model_rows["v14_dev"][0]["repetition_control"],
           "fake_backend_calls": sum(n for b, n in backends.items() if b not in ("huggingface", "mlx"))},
          "RAW", "generation checkpoint 'model' blocks",
          note="backends: " + json.dumps(dict(backends)) + "; quantization bits: " + json.dumps({aid: [(r["quantization"] or {}).get("bits") for r in rows] for aid, rows in model_rows.items()}))

    # ---- observation outside the claim scope: negation-scope ambiguity in the rendered English
    # 'it is not the case that (A) or (B)' can be read as NOT(A or B) (the gold) or (NOT A) or B.
    ambiguous = re.compile(r"it is not the case that \([^()]*\) (and|or) \(")
    amb = {}
    for a in attempts:
        c = Counter()
        for name in a["runs"]:
            run = get_run(name)
            for key, e in all_evals.get(name, {}).items():
                case = run.by_key[key][0]
                if ambiguous.search(case.get("body") or case.get("prompt") or ""):
                    c[f"{case['family']}:{e['outcome']}"] += 1
        if c:
            amb[a["attempt"]] = dict(c)
    v11run = get_run("20260914_constrained_qwen3_4b_dev_v11")
    _, v11views, _ = join_and_score(v11run, "constrained_v11")
    cf_views = []
    for v in v11views:
        if v["family"] == "demorgan_disjunction" and v["ev"]["outcome"] == "wrong":
            v = {**v, "ev": {**v["ev"], "parsed": dict(v["gold"]), "exact": True, "outcome": "correct",
                             "correct": {k: True for k in v["gold"]}}}
        cf_views.append(v)
    cf = clean_analysis(cf_views)
    out["observations"] = {
        "negation_scope_ambiguity": {
            "pattern": "it is not the case that (X) and|or (Y)  -- intended NOT(X op Y); a narrow reading (NOT X) op Y is possible",
            "outcomes_by_attempt": amb,
            "v11_counterfactual_NOT_A_RESCORE": {
                "assumption": "the 6 v11 demorgan_disjunction answers (INVALID) are treated as correct",
                "gate_passed": cf["passed"],
                "failed_checks": {ax: g["failed_cell_checks"] + g["failed_stability_checks"] for ax, g in cf["gates"].items()},
                "opened_v7_validity_cells": {o: frac_str(cf["cells"][f"opened_v7/validity/{o}"]["correct"], cf["cells"][f"opened_v7/validity/{o}"]["n"]) for o in ORDERS}}}}

    # ---- evidence: truncations are repetition loops; v7 unknowns are final-line format failures
    loops = []
    for a in attempts:
        for name in a["runs"]:
            for e in all_evals.get(name, {}).values():
                if e["outcome"] == "truncated":
                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", e["text"]) if s.strip()]
                    rep = Counter(sentences).most_common(1)[0][1] if sentences else 0
                    loops.append({"attempt": a["attempt"], "item_id": e["item_id"], "option_order": e["option_order"],
                                  "output_tokens": e["output_tokens"], "sentences": len(sentences),
                                  "distinct_sentences": len(set(sentences)), "max_sentence_repeats": rep,
                                  "distinct_characters": len(set(e["text"])), "head": e["text"][:90]})
    out["truncation_evidence"] = loops
    v7_unknown = [e for e in all_evals["20260914_reasoning_qwen7b_dev_v7"].values() if e["outcome"] == "unknown"]
    why = Counter()
    later_rule = Counter()
    for e in v7_unknown:
        t = e["text"].strip()
        n = t.count("ANSWER:")
        if n == 0:
            why["no ANSWER: marker"] += 1
        elif n > 1:
            why["several ANSWER: markers"] += 1
        elif ANSWER_END_RE.search(t):
            why["single final marker on the explanation line (v7 requires its own line)"] += 1
        else:
            why["marker not at the end"] += 1
        label = label_answer_end(t)
        if label is not None:
            later_rule["correct" if YESNO[e["axis"]][label] == e["gold"][e["axis"]] else "wrong"] += 1
    att["v7_dev"]["unknown_breakdown"] = {"why": dict(why),
                                          "informational_if_the_later_v8_rule_were_applied_(NOT_a_rescore)": dict(later_rule)}

    # ---- totals breakdown: what the 2,879 consists of
    overall = Counter()
    keys_all = set()
    for a in attempts:
        for name in a["runs"]:
            for key, e in all_evals.get(name, {}).items():
                overall[e["outcome"]] += 1
                keys_all.add(key)
    out["totals"]["outcomes_across_completed"] = dict(overall)
    out["totals"]["distinct_request_message_lists_across_completed"] = len(keys_all)
    out["totals"]["note_on_outcomes"] = ("v10's 165 are 'unknown' under its own (buggy) scorer; the 3B format run's 34 were scored "
                                         "with the v5 format parser but no accuracy is claimed for either interrupted attempt")

    # ---- supplementary: detail numbers in the named 2.6 source documents
    def cells(aid, name):
        cc = att[aid]["cells"][name]
        return frac_str(cc["correct"], cc["n"])

    def flips(aid, name):
        cc = att[aid]["contrasts"][name]
        return frac_str(cc["changed"], cc["pairs"])

    claim("2.6src-v6-holdout", "2.6 source", "v6 first sealed validation: truth 20/24 in both orders (< 21/24)",
          ["20/24", "20/24"], [cells("v6_holdout", f"heldout/truth/{o}") for o in ORDERS], "RAW", S)
    claim("2.6src-v8", "2.6 source", "v8: no format unknown; one validity group 20/24; enumeration flips 4/24",
          {"unknown": 0, "group": "20/24", "flips": "4/24"},
          {"unknown": oc("v8_dev").get("unknown", 0) + oc("v8_dev").get("truncated", 0),
           "group": cells("v8_dev", "opened_v5/validity/negative_first"), "flips": flips("v8_dev", "opened_v5/validity")}, "RAW", S,
          note=f"opened_v5/validity/positive_first = {cells('v8_dev', 'opened_v5/validity/positive_first')}")
    v9names = [("legacy", "truth"), ("extension", "truth"), ("opened_v5", "truth"), ("legacy", "validity"), ("extension", "validity"), ("opened_v5", "validity")]
    claim("2.6src-v9-dev-table", "2.6 source", "v9 dev groups (YES-first, NO-first): 8/8 8/8; 16/16 16/16; 24/24 24/24; 8/8 8/8; 16/16 16/16; 23/24 23/24; both errors = holdout_reverse_chain_0; zero unknown, zero flips",
          {"cells": [["8/8", "8/8"], ["16/16", "16/16"], ["24/24", "24/24"], ["8/8", "8/8"], ["16/16", "16/16"], ["23/24", "23/24"]],
           "errors": ["holdout_reverse_chain_0", "holdout_reverse_chain_0"], "unknown": 0, "flips": 0},
          {"cells": [[cells("v9_dev", f"{s}/{ax}/{o}") for o in ORDERS] for s, ax in v9names],
           "errors": [n["item_id"] for n in att["v9_dev"]["noncorrect"]], "unknown": oc("v9_dev").get("unknown", 0) + oc("v9_dev").get("truncated", 0),
           "flips": sum(c["changed"] for c in att["v9_dev"]["contrasts"].values())}, "RAW", S)
    claim("2.6src-v11", "2.6 source", "v11: opened-v7 validity groups both 18/24", ["18/24", "18/24"],
          [cells("v11_dev", f"opened_v7/validity/{o}") for o in ORDERS], "RAW", S)
    d13 = att["v13_dev"]["outcomes"]
    claim("2.6src-v13-dev", "2.6 source", "v13 dev: truth 144/144, validity 140/144; errors = holdout_reverse_chain_0 and _2 in both orders",
          {"truth": "144/144", "validity": "140/144", "errors": sorted(["holdout_reverse_chain_0"] * 2 + ["holdout_reverse_chain_2"] * 2)},
          {"truth": f"{d13['truth'].get('correct', 0)}/{sum(d13['truth'].values())}", "validity": f"{d13['validity'].get('correct', 0)}/{sum(d13['validity'].values())}",
           "errors": sorted(n["item_id"] for n in att["v13_dev"]["noncorrect"])}, "RAW", S)
    h13 = att["v13_holdout"]
    trunc13 = [n for n in h13["noncorrect"] if n["outcome"] == "truncated"]
    claim("2.6src-v13-holdout", "2.6 source", "v13 third bank: truth 24/24 and 23/24, validity 23/24 and 24/24; the truncation is '47 - 18 = 29' in NO-first order",
          {"truth": ["24/24", "23/24"], "validity": ["23/24", "24/24"], "truncated": ["v10_integer_subtraction_0_1", "negative_first"]},
          {"truth": [cells("v13_holdout", f"heldout_v10/truth/{o}") for o in ORDERS], "validity": [cells("v13_holdout", f"heldout_v10/validity/{o}") for o in ORDERS],
           "truncated": [trunc13[0]["item_id"], trunc13[0]["option_order"]] if trunc13 else None}, "RAW", S,
          note="item v10_integer_subtraction_0_1 is the claim '47 - 18 = 29' (TRUE) in the v10 bank")
    dev14 = att["v14_dev"]
    claim("2.6src-v14-dev", "2.6 source", "v14 dev: all six errors are reverse-chain (one-way implication reversed); opened-v10 validity flips 2/24, all other source/axis flips 0",
          {"error_families": ["reverse_chain"] * 6, "opened_v10_validity_flips": "2/24", "other_flips": 0},
          {"error_families": [n["family"] for n in dev14["noncorrect"]], "opened_v10_validity_flips": flips("v14_dev", "opened_v10/validity"),
           "other_flips": sum(c["changed"] for k, c in dev14["contrasts"].items() if k != "opened_v10/validity")}, "RAW", S)
    claim("2.6src-v14-holdout", "2.6 source", "v14 fourth bank: weakest gold subgroup YES-first INVALID 10/12 (>= 9/12); errors v14_reverse_chain_1 and _2 in YES-first",
          {"weakest": "10/12", "errors": [["v14_reverse_chain_1", "positive_first"], ["v14_reverse_chain_2", "positive_first"]]},
          {"weakest": frac_str(hc["heldout_v14/validity/positive_first"]["by_gold"]['{"validity": "INVALID"}']["correct"],
                               hc["heldout_v14/validity/positive_first"]["by_gold"]['{"validity": "INVALID"}']["n"]),
           "errors": [[n["item_id"], n["option_order"]] for n in h14["noncorrect"]]}, "RAW", S)
    tl = out["v14_timeline"]
    claim("2.6src-v14-timeline", "2.6 source", "v14 bank committed 03:07:00, dev started 03:08:59, lock 05:03:28, validation 05:03:31 (UTC, 2026-09-15)",
          ["03:07:00", "03:08:59", "05:03:28", "05:03:31"], [tl[k][11:19] for k in ("bank_committed", "development_started", "selection_locked", "validation_started")],
          "DERIVED", "results/designs/heldout_v14_commitment.json, repetition_qwen3_8bit_locked_selection_v14.json, v14 manifests",
          note="timestamps are self-recorded by the project, not independently attested")
    claim("2.6src-cumulative", "2.6 source", "running totals: through v11 1,727 (+2 unfinished); with v12 2,015; with v13 2,399; final 2,879",
          [1727, 2015, 2399, 2879], [cum["v11_dev"], cum["v12_dev"], cum["v13_holdout"], cum["v14_holdout"]], "RAW", S,
          note="saved audit counters agree at every stage: " + str({k: v["match"] for k, v in saved_counters.items()}))
    claim("2.6src-v7-no-new-bank", "2.6 source", "v7 and v8 failed in development without opening a bank; the v7 bank was first opened by v9's validation",
          {"first_opening_of_heldout_v7": "20260914_guided_qwen3_4b_holdout_v9", "holdout_runs_with_v7_or_v8_format": 0},
          {"first_opening_of_heldout_v7": next((b["run"] for b in banks if "heldout_v7" in b["sources"]), None),
           "holdout_runs_with_v7_or_v8_format": sum(1 for b in banks if b["format"] == "clean_reasoning" or "qwen7b" in b["run"] and "v8" in b["run"])},
          "RAW", "holdout run manifests")
    out["claims"] = CLAIMS
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=Path(DEFAULT_PROJECT))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().with_name("recompute_v4_v14.json"))
    args = parser.parse_args()
    project = args.project.resolve()
    if args.out.resolve().is_relative_to(project):
        raise SystemExit("Refusing to write inside the read-only project snapshot")
    result = analyse(project)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1, default=str)
    counts = Counter(c["status"] for c in result["claims"])
    print(f"claims: {dict(counts)}")
    for c in result["claims"]:
        if c["status"] != "MATCH":
            print(f"  {c['status']}: {c['id']}: expected={json.dumps(c['expected'], ensure_ascii=False)} recomputed={json.dumps(c['recomputed'], ensure_ascii=False)}")
    t = result["totals"]
    print(f"completed={t['convention_A_completed_checkpoints']} unfinished={t['unfinished_checkpoints']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
