#!/usr/bin/env python3
"""Report Flash Next R2 on-host progress without running or mutating anything.

ChatGPT Work sessions can stop mid-build/benchmark. This reporter reconstructs
state from manifests, llama-swap aliases and the newest per-phase summary.env,
then prints exactly one conservative NEXT_ACTION. It never loads/unloads a model,
changes config, or promotes production.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import glob
import json
import re

PROD = "qwen3.8-flash-next:256k"
FOUNDATION = "qwen3.8-flash-next-r2-modern-foundation:256k"
MTP = "qwen3.8-flash-next-r2-modern-mtp:256k"
FINAL = "qwen3.8-flash-next-r2-final-pre-sweep:256k"
LAYER = "qwen3.8-flash-next-r2-split-layer:256k"
TENSOR = "qwen3.8-flash-next-r2-split-tensor-1x1:256k"


def env_file(path: Path | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path or not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw or raw.lstrip().startswith("#"):
            continue
        k, v = raw.split("=", 1)
        if k.strip():
            out[k.strip()] = v.strip()
    return out


def latest(pattern: str) -> Path | None:
    paths = [Path(p) for p in glob.glob(pattern)]
    paths = [p for p in paths if p.is_file()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def json_gate(path: Path | None, key="gate") -> str | None:
    if not path or not path.is_file():
        return None
    try:
        x = json.loads(path.read_text(encoding="utf-8"))
        if key:
            x = x.get(key, {})
        v = x.get("result") if isinstance(x, dict) else None
        return str(v) if v is not None else None
    except Exception:
        return "INVALID"


def alias_names(config: Path) -> set[str]:
    if not config.is_file():
        return set()
    names = set()
    for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\s*([^#\s][^:]*:[^:]+):\s*(?:#.*)?$", line)
        if m:
            names.add(m.group(1).strip())
    return names


def show_summary(label: str, path: Path | None, data: dict[str, str], keys: tuple[str, ...]):
    print(f"[{label}] {path or 'NONE'}")
    for k in keys:
        if k in data:
            print(f"  {k}={data[k]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/app/share/llama_box/config/config-rocm714.yaml")
    ap.add_argument("--logs", default="/app/share/openclaw_tools/logs")
    ap.add_argument("--root", default="/app/share/llama_box/src")
    args = ap.parse_args()

    config = Path(args.config)
    logs = Path(args.logs)
    root = Path(args.root)
    aliases = alias_names(config)

    foundation_manifest = root / "llama.cpp-flashnext-modern-foundation-20260918/r2-meta/modern-foundation-manifest.txt"
    mtp_manifest = root / "llama.cpp-flashnext-r2-modern-mtp-20260918/r2-meta/stage10-modern-mtp-base.txt"
    phase3_manifest = root / "llama.cpp-flashnext-r2-final-pre-sweep-20260919/r2-meta/phase3-compose-manifest.txt"
    phase6_manifest = root / "llama.cpp-flashnext-r2-tensor-split-20260919/r2-meta/phase6-tensor-split-manifest.txt"
    ratio_manifest = root / "llama.cpp-flashnext-r2-tensor-split-20260919/r2-meta/phase6b-ratios.env"

    foundation_analysis = logs / "flashnext-r2-modern-foundation-ab.foundation-analysis.json"
    p1 = latest(str(logs / "flashnext-r2-phase1-*/summary.env")); p1d = env_file(p1)
    p2 = latest(str(logs / "flashnext-r2-phase2-*/summary.env")); p2d = env_file(p2)
    p3 = latest(str(logs / "flashnext-r2-phase3-validation-*/summary.env")); p3d = env_file(p3)
    p4 = latest(str(logs / "flashnext-r2-phase4-*/summary.env")); p4d = env_file(p4)
    p5 = latest(str(logs / "flashnext-r2-phase5-gdn-*/summary.env")); p5d = env_file(p5)
    p6 = latest(str(logs / "flashnext-r2-phase6-tensor-split-*/summary.env")); p6d = env_file(p6)
    p6b = latest(str(logs / "flashnext-r2-phase6b-ratio-*/summary.env")); p6bd = env_file(p6b)

    print("=== Flash Next R2 state report ===")
    print(f"config_exists={config.is_file()}")
    print(f"production_alias_present={PROD in aliases}")
    print(f"production_alias={PROD}")
    print(f"foundation_manifest={foundation_manifest.is_file()} alias={FOUNDATION in aliases}")
    print(f"foundation_gate={json_gate(foundation_analysis)}")
    print(f"mtp_manifest={mtp_manifest.is_file()} alias={MTP in aliases}")
    print(f"phase3_manifest={phase3_manifest.is_file()} alias={FINAL in aliases}")
    print(f"phase6_manifest={phase6_manifest.is_file()} layer_alias={LAYER in aliases} tensor_alias={TENSOR in aliases}")
    print(f"phase6b_ratio_manifest={ratio_manifest.is_file()}")
    print()

    show_summary("phase1", p1, p1d, ("FOUNDATION_RESULT","MTP_RESULT","TOPK_SMOKE_RESULT","TOPK_FULL_RESULT","PRODUCTION_PROMOTED","FINISHED"))
    show_summary("phase2", p2, p2d, ("TOPK_FINAL","LONG_WINNER_ALIAS","PLE_PASS","FRSPEC_PASS","PRODUCTION_PROMOTED","FINISHED"))
    show_summary("phase3", p3, p3d, ("RUNTIME_BUNDLE","NATIVE_RS_ROLLBACK","SHORT_GATE","LONG_GATE","CACHED_GATE","ROLLBACK_GATE","VALIDATION","PRODUCTION_PROMOTED","FINISHED"))
    show_summary("phase4", p4, p4d, ("MTP_THROUGHPUT_WINNER","MTP_CACHED_GATE","MTP_ACCEPTED_ALIAS","GRAPH_WINNER","PARAM_WINNER_ALIAS","PRODUCTION_PROMOTED","FINISHED"))
    show_summary("phase5", p5, p5d, ("THROUGHPUT_MODE","CACHED_GATE","ROLLBACK_GATE","PHASE5_WINNER_MODE","PHASE5_WINNER_ALIAS","PRODUCTION_PROMOTED","FINISHED"))
    show_summary("phase6", p6, p6d, ("SHORT_GATE_RC","LONG_GATE_RC","CACHED_GATE","ROLLBACK_GATE","PHASE6_WINNER_MODE","PHASE6_WINNER_ALIAS","PRODUCTION_PROMOTED","FINISHED"))
    show_summary("phase6b", p6b, p6bd, ("PHASE6B_WINNER_RATIO","PHASE6B_WINNER_ALIAS","CONFIRM_SHORT","CONFIRM_LONG","CACHED_GATE","ROLLBACK_GATE","PRODUCTION_PROMOTED","FINISHED"))

    warnings = []
    if PROD not in aliases:
        warnings.append("production alias is missing")
    for name, d in (("phase1",p1d),("phase2",p2d),("phase3",p3d),("phase4",p4d),("phase5",p5d),("phase6",p6d),("phase6b",p6bd)):
        if d.get("PRODUCTION_PROMOTED") not in (None, "NO"):
            warnings.append(f"{name} says PRODUCTION_PROMOTED={d.get('PRODUCTION_PROMOTED')}")
    if warnings:
        print("\nWARNINGS")
        for w in warnings:
            print(f"  - {w}")
        print("NEXT_ACTION=STOP_AND_INSPECT_PRODUCTION")
        return

    topk_decided = p1d.get("TOPK_SMOKE_RESULT") in {"PASS", "REJECT", "FAIL", "HIP_GRAPH_INTERACTION"}

    if not foundation_manifest.is_file() or FOUNDATION not in aliases:
        nxt = "bash flashnext-r2/scripts/run_phase1_real_ab.sh"
        why = "Modern Foundation is not fully prepared/registered."
    elif json_gate(foundation_analysis) != "PASS":
        nxt = "bash flashnext-r2/scripts/resume_phase1_after_work.sh"
        why = "Foundation exists but no current-schema PASS is recorded."
    elif p1d.get("MTP_RESULT") != "PASS" or not topk_decided:
        nxt = "bash flashnext-r2/scripts/resume_phase1_after_work.sh"
        why = "Phase-1 MTP/TOP_K decision is incomplete."
    elif not p2d.get("FINISHED"):
        nxt = "bash flashnext-r2/scripts/run_phase2_real_ab.sh"
        why = "Phase-1 is decided; Phase-2 has not completed."
    elif not phase3_manifest.is_file() or FINAL not in aliases:
        nxt = "bash flashnext-r2/scripts/prepare_phase3_final_candidate.sh"
        why = "Phase-2 completed; winner composition has not been prepared."
    elif p3d.get("VALIDATION") != "PASS":
        nxt = "bash flashnext-r2/scripts/run_phase3_final_validation.sh"
        why = "Final pre-sweep candidate has not passed all Phase-3 hard gates."
    elif not p4d.get("PARAM_WINNER_ALIAS"):
        nxt = "bash flashnext-r2/scripts/run_phase4_mtp_graph_sweep.sh"
        why = "Phase-3 passed; MTP-depth/HIP-Graph sweep is incomplete."
    elif not p5d.get("PHASE5_WINNER_ALIAS"):
        nxt = "bash flashnext-r2/scripts/prepare_phase5_gdn_microfusion.sh && bash flashnext-r2/scripts/run_phase5_verified.sh"
        why = "Phase-4 winner exists; GDN microfusion decision is incomplete."
    elif not phase6_manifest.is_file() or LAYER not in aliases or TENSOR not in aliases:
        nxt = "bash flashnext-r2/scripts/prepare_phase6_tensor_split.sh"
        why = "Phase-5 winner exists; layer-vs-tensor candidate has not been prepared."
    elif not p6d.get("PHASE6_WINNER_MODE"):
        nxt = "bash flashnext-r2/scripts/run_phase6_verified.sh"
        why = "Phase-6 runtimes exist but verified layer-vs-tensor A/B is incomplete."
    elif p6d.get("PHASE6_WINNER_MODE") == "TENSOR_1x1" and not ratio_manifest.is_file():
        nxt = "bash flashnext-r2/scripts/prepare_phase6b_tensor_ratio_sweep.sh"
        why = "Tensor 1:1 won; heterogeneous ratio arms are not prepared."
    elif p6d.get("PHASE6_WINNER_MODE") == "TENSOR_1x1" and not p6bd.get("PHASE6B_WINNER_ALIAS"):
        nxt = "bash flashnext-r2/scripts/run_phase6b_verified.sh"
        why = "Ratio arms exist but verified Phase-6b sweep is incomplete."
    else:
        winner = p6bd.get("PHASE6B_WINNER_ALIAS") or p6d.get("PHASE6_WINNER_ALIAS") or p5d.get("PHASE5_WINNER_ALIAS")
        nxt = f"FINAL_OPENCLAW_REGRESSION winner={winner or 'UNKNOWN'}"
        why = "All automated R2 experiment phases with recorded state are complete; do not promote until real OpenClaw session regression passes."

    print("\nDECISION")
    print(f"WHY={why}")
    print(f"NEXT_ACTION={nxt}")
    print("PRODUCTION_PROMOTION=NO")


if __name__ == "__main__":
    main()
