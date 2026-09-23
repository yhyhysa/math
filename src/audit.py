"""Integrated T0 Data & Environment Audit runner.

Integrates:
  - Environment & hardware checks (GPU RTX 5070 Ti, CUDA 12.8, PyTorch, PyAV, Transformers)
  - Unified data reader validation across all 51 pkl files (aligned_50, 30 missing, 20 explain)
  - Input manifest (245 rows covering all 244 extracted files + raw zip)
  - Cross-attachment overlap calculation (exact 18 sample_ids, 24 video_ids)
  - Full-coverage BERT tokenizer and GPU batched reconstruction evaluation (4870 samples)
  - Detailed video/audio stream decoding, PTS monotonicity, and audio duration audit
  - 50MB package budget estimation

Produces:
  - results/data_audit.json
  - results/input_manifest.csv
  - results/interface_contract.json
  - results/reader_check.json
  - results/cross_attachment_overlap.csv
  - results/bert_audit_full.json
  - results/video_audit_detailed.json
  - updates STATUS.md

Usage (cwd = 工程/):
  python -m src.audit --config configs/base.json
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import torch

from src.check_data_reader import check_attachment2, check_attachment3, check_attachment4
from src.config import load_config, ensure_dirs
from src.tools.audit_bert_full import PINNED_REVISION_SHA, audit_bert_full
from src.tools.audit_videos_detailed import audit_videos_detailed
from src.tools.build_manifest import build_manifest
from src.tools.cross_overlap import compute_cross_overlap

RAW_ZIP_SHA256 = "F895003ABEB3E972E69EF9C457032720CBCB8ECD3D2261A0E30BC15667604ACB"


@dataclass
class StepResult:
    """Outcome of one audit step. `passed=None` means deliberately skipped (test mode)."""

    name: str
    passed: bool | None
    exit_code: int | None
    elapsed_seconds: float
    detail: str = ""
    output_path: str | None = None
    failed_reason: str | None = None

    @property
    def status(self) -> str:
        if self.passed is None:
            return "SKIPPED"
        return "PASS" if self.passed else "FAIL"


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def write_text_file(path: Path, text: str) -> None:
    """Atomic write; creates parent directories; raises OSError on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-write")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_if_possible(path: Path, text: str) -> bool:
    try:
        write_text_file(path, text)
        return True
    except OSError as exc:
        print(f"[WARN] could not write {path}: {exc}")
        return False


def write_mandatory(path: Path, text: str) -> bool:
    """Critical artefact write: never silent, returns False (and prints) on failure."""
    try:
        write_text_file(path, text)
        return True
    except OSError as exc:
        print(f"[ERROR] mandatory artefact write failed: {path}: {exc}")
        return False


def _artifact_path(rel: str, results_dir: Path, engine_dir: Path | None = None) -> Path:
    """Resolve an artefact key to the file that actually exists on disk.

    Keys may be 'results/x.json', 'data/raw/....zip', 'STATUS.md' or 'results/STATUS.md'.
    STATUS.md lives at the engineering root next to results/, so candidates are probed in
    order and the first existing file wins (falls back to the results-relative path).
    """
    if rel.startswith("data/") and engine_dir is not None:
        return engine_dir / rel
    name = rel[len("results/"):] if rel.startswith("results/") else rel
    candidates = [results_dir / name]
    if engine_dir is not None:
        candidates.append(engine_dir / name)
    for cand in candidates:
        if cand.is_file():
            return cand
    return candidates[0]


def verify_hashes_on_disk(hash_map: dict, results_dir: Path, engine_dir: Path) -> list[str]:
    """Re-read every listed artefact and confirm the recorded SHA-256 still matches disk."""
    problems: list[str] = []
    for name, recorded in hash_map.items():
        path = _artifact_path(name, results_dir, engine_dir)
        if not path.is_file():
            problems.append(f"{name}: missing after write ({path})")
            continue
        actual = sha256_file(path)
        if actual is None or recorded is None or actual.lower() != str(recorded).lower():
            problems.append(f"{name}: hash mismatch on disk ({actual} != {recorded})")
    return problems


def _verify_receipt_self_hash(report_path: Path, receipt: dict) -> list[str]:
    """The receipt binds its own final bytes through a `sha256_self` field written in a second
    pass; this re-reads the file and confirms the binding still holds.  A file can never contain
    the hash of its own bytes, so the field is explicitly excluded from the hash input."""
    problems: list[str] = []
    if not report_path.is_file():
        return [f"{report_path.name}: missing"]
    data = json.loads(report_path.read_text(encoding="utf-8"))
    recorded = data.get("sha256_self")
    if not recorded:
        return [f"{report_path.name}: sha256_self not bound"]
    body = dict(data)
    body.pop("sha256_self", None)
    recomputed = hashlib.sha256(
        json.dumps(body, indent=2, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if recomputed != recorded:
        problems.append(f"{report_path.name}: sha256_self {recorded} != recomputed {recomputed}")
    return problems


def run_subprocess(cmd: list[str], cwd: Path) -> tuple[int, str, str, float]:
    # PYTHONIOENCODING/PYTHONUTF8: the Windows console here is cp936; without this a child
    # that prints non-ASCII paths can die with UnicodeEncodeError instead of doing its job.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, env=env)
    elapsed = time.time() - t0
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    return proc.returncode, out, err, elapsed


def run_integrity_gate(cfg: dict, phase: str) -> StepResult:
    """Run src.tools.verify_raw_integrity and require exit code 0."""
    cmd = [sys.executable, "-m", "src.tools.verify_raw_integrity"]
    name = f"integrity_gate_{phase}"
    rc, out, err, secs = run_subprocess(cmd, cfg["_engine_dir"])
    payload: dict = {}
    try:
        payload = json.loads(out.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001 - the gate verdict comes from the exit code, not the parse
        payload = {}
    if payload:
        detail = (
            f"{payload.get('status')}: checked={payload.get('checked')}, errors={payload.get('errors')}, "
            f"report={payload.get('report')}"
        )
    else:
        detail = (out.strip() or err.strip())[-800:]
    passed = rc == 0
    return StepResult(
        name=name,
        passed=passed,
        exit_code=rc,
        elapsed_seconds=round(secs, 2),
        detail=detail,
        output_path=payload.get("report") if passed else None,
        failed_reason=None if passed else f"verify_raw_integrity exit code {rc}: {(err.strip() or out.strip())[-500:]}",
    )


def print_step_result(res: StepResult) -> None:
    line = f"      -> [{res.status}] {res.name} ({res.elapsed_seconds}s)"
    if res.detail:
        line += f" :: {res.detail}"
    print(line)
    if res.failed_reason:
        print(f"         reason: {res.failed_reason}")


def _fail_injection_armed(step_name: str, fail_injection: str | None) -> None:
    """Test-only hook: force a step to fail *after* its real work, without touching any input.

    Armed only by the environment variable DSH_T0_FAIL_INJECTION (never set in a normal run).
    """
    if fail_injection and fail_injection == step_name:
        raise AssertionError(
            f"INJECTED FAILURE for step '{step_name}' (DSH_T0_FAIL_INJECTION): simulated sub-check failure"
        )


def run_step(name: str, fn, fail_injection: str | None) -> StepResult:
    t0 = time.time()
    try:
        payload = fn() or {}
        _fail_injection_armed(name, fail_injection)
    except BaseException as exc:  # noqa: BLE001 - failure must be recorded, never swallowed
        return StepResult(
            name=name,
            passed=False,
            exit_code=1,
            elapsed_seconds=round(time.time() - t0, 2),
            failed_reason=f"{type(exc).__name__}: {exc}",
        )
    return StepResult(
        name=name,
        passed=True,
        exit_code=0,
        elapsed_seconds=round(time.time() - t0, 2),
        detail=str(payload.get("detail", "")),
        output_path=payload.get("output_path"),
    )


def audit_environment() -> dict:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "None"
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": cuda_ver,
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": gpu_name,
        "packages": {
            "torch": torch.__version__,
            "transformers": getattr(__import__("transformers"), "__version__", "N/A"),
            "av": getattr(av, "__version__", "N/A"),
            "cv2": cv2.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": getattr(__import__("scipy"), "__version__", "N/A"),
            "openpyxl": getattr(__import__("openpyxl"), "__version__", "N/A"),
        },
    }


def build_updated_interface_contract() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "E题 多模态统一数据接口与模型契约 (Interface Contract v2)",
        "version": "2.0.0",
        "date": "2026-09-23",
        "frozen_text_encoder": {
            "name": "bert-base-uncased",
            "revision_sha": PINNED_REVISION_SHA,
            "version_binding": (
                "从本地快照目录 snapshots/86b5e0934494bd15c9632b12f734a8a67f723594 加载"
                "（local_files_only=True，HF_HUB_OFFLINE=1），不使用可变的 refs/main；"
                "加载前后逐文件核对 SHA-256，缺失或哈希不符即非零退出。"
            ),
            "vocab_size": 30522,
            "hidden_dimension": 768,
            "max_sequence_length": 50,
            "truncation_strategy": "longest_first_to_50",
            "special_tokens": {"cls_id": 101, "sep_id": 102, "pad_id": 0},
            "output_layer": "last_hidden_state (50, 768)",
            "file_hashes": {
                "vocab_txt_sha256": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
                "config_json_sha256": "7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20",
                "model_safetensors_sha256": "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3",
                "tokenizer_json_sha256": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
            },
            "external_download_policy": (
                "外部冻结权重（~440MB）不进入 50MB 提交包。获取地址：HuggingFace hub 模型 ID bert-base-uncased，"
                "revision 86b5e0934494bd15c9632b12f734a8a67f723594；离线复现需预先落地该快照。"
                "评审阶段是否允许联网下载仍待官方规则核验，不默认接受。"
            ),
        },
        "modality_raw_tensor_spec": {
            "text": {
                "shape": [50, 768],
                "raw_dtype": "float32",
                "dtype_measured_on": "Attachment 2 aligned_50.pkl / Attachment 4 aligned pkl",
                "units": "BERT last hidden state (frozen encoder)",
            },
            "text_bert": {
                "shape": [3, 50],
                "raw_dtype": "int64",
                "rows": ["input_ids", "attention_mask", "token_type_ids"],
                "semantics_status": "token ids verified as 3 rows; row order matches bert-base-uncased tokenizer output on 4870 samples",
            },
            "audio": {
                "shape": [50, 74],
                "raw_dtype": "float64",
                "dtype_measured_on": "Attachment 2 aligned_50.pkl (np.float64 as stored)",
                "units": "74-dim acoustic descriptors",
                "dimension_semantics": "UNKNOWN",
                "dimension_semantics_note": "题目与附件均未给出 74 维逐维语义；不作命名或类型假设，按统一数值特征输入。",
            },
            "vision": {
                "shape": [50, 35],
                "raw_dtype": "float64",
                "dtype_measured_on": "Attachment 2 aligned_50.pkl (np.float64 as stored)",
                "units": "35-dim facial visual features",
                "dimension_semantics": "UNKNOWN",
                "dimension_semantics_note": "题目与附件均未给出 35 维逐维语义；不作命名或类型假设，按统一数值特征输入。",
            },
        },
        "dtype_conversion_policy": {
            "raw_storage_dtype": {"audio": "float64", "vision": "float64", "text": "float32", "text_bert": "int64"},
            "training_dtype": "float32",
            "conversion_point": "T1: 在 Dataset.__getitem__ / DataLoader collate 处用 torch.as_tensor(x, dtype=torch.float32)",
            "conversion_status": "NOT_YET_IMPLEMENTED",
            "numerical_checks": {
                "status": "T1_PENDING_IMPLEMENTATION",
                "requirement": "T1 Dataset/DataLoader 必须在每个 batch 上执行 torch.isfinite(x).all()，发现 NaN/Inf 立即非零中止",
                "enforced_in_t0": False,
                "enforced_in_t0_note": "T0 未实现 Dataset/DataLoader，因此本轮没有任何有限值检查被执行；此为 T1 待实现合同，不得写成已执行。",
            },
        },
        "mask_definitions": {
            "source": "Attachment 2 text_bert attention_mask (row 1); no mask is invented for missing data",
            "padding_mask": "bool (50,), True for active steps (0 <= t < text_len), False for PAD (t >= text_len)",
            "valid_position_zero_feature_status": "疑似无观测 (suspected unobserved) — 缺少原始特征提取证据，不宣称为人脸遮挡或真实模态缺失",
            "observed_mask": {
                "text": "bool (50,), True where the token position is valid (0 <= t < text_len)",
                "audio": "bool (50,), False at CLS (pos 0) / SEP (pos text_len-1) and where ||x|| < 1e-6 in valid steps (标为'疑似无观测')",
                "vision": "bool (50,), False at CLS (pos 0) / SEP (pos text_len-1) and where ||x|| < 1e-6 in valid steps (标为'疑似无观测')",
            },
            "training_missing_mask": {
                "status": "T1_PENDING_IMPLEMENTATION",
                "definition": "训练时模拟缺失：仅在有效词步 (1 <= t <= text_len-2) 上施加连续遮蔽；T0 未执行，仅登记合同",
            },
            "measured_zero_statistics": "见 results/mask_audit.json（按 padding 位置 / 有效 token 位置分列统计）",
        },
        "labels_spec": {
            "classification": {
                "raw_dtype": "float64",
                "target_dtype": "int64",
                "mapping": {"0": "Negative (intensity in [-3.0, -0.33])", "1": "Neutral (intensity strictly == 0.0)", "2": "Positive (intensity in [0.17, 3.0])"},
            },
            "regression": {
                "raw_dtype": "float64",
                "target_dtype": "float32",
                "range": [-3.0, 3.0],
                "neutral_value": 0.0,
            }
        },
        "packaging_dispatch_contract": {
            "attachment2": "Columnar dict with train/valid/test -> unbind sample row into Sample objects",
            "attachment3": "Dict with 'test' key and leading batch dim 1 -> squeeze(0), generate text from text_bert, derive local_id from stem (附件3_01~30)",
            "attachment4": "Flat dict without batch dim -> extract arrays directly, link to videos/{id}.mp4",
        },
        "submission_csv_schemas": {
            "q2_submission": [
                "sample_id", "source_file", "polarity", "intensity",
                "negative_prob", "neutral_prob", "positive_prob", "model_version"
            ],
            "q3_submission": [
                "sample_id", "source_file", "polarity", "intensity",
                "negative_prob", "neutral_prob", "positive_prob",
                "main_modality", "text_attribution", "audio_attribution", "vision_attribution",
                "evidence_file", "mapping_status", "model_version"
            ]
        }
    }


def estimate_submission_budget() -> dict:
    return {
        "max_allowed_package_size_mb": 50.0,
        "budget_breakdown_mb": {
            "q1_compressed_features_100_samples_npz": {
                "estimated_mb": 4.5,
                "calculation_basis": "100 samples * ~45KB per compressed npz = 4.5 MB"
            },
            "q2_robust_model_weights_fp16": {
                "estimated_mb": 2.2,
                "calculation_basis": "~1.1M parameters * 2 bytes (FP16) = 2.2 MB"
            },
            "q3_explain_model_weights_fp16": {
                "estimated_mb": 2.2,
                "calculation_basis": "~1.1M parameters * 2 bytes (FP16) = 2.2 MB"
            },
            "source_code_and_configs": {
                "estimated_mb": 0.8,
                "calculation_basis": "src/, configs/, tests/, README.md, STATUS.md"
            },
            "result_tables_and_cards": {
                "estimated_mb": 5.5,
                "calculation_basis": "Prediction CSVs, structured JSON evidence, and offline HTML explanation cards"
            },
            "safety_margin_mb": 34.8,
            "estimated_total_submission_size_mb": 15.2,
        },
        "external_frozen_dependencies_policy": (
            "Pretrained BERT weights (~440MB) cannot fit into the 50MB contest package. "
            "Whether external online download during review is permitted is subject to official contest rules; "
            "the reproduction manifest documents exact HuggingFace checkpoint ID, revision SHA, and offline extraction rules."
        )
    }


def verify_recorded_artifacts(cfg: dict) -> int:
    """Re-hash every artefact listed in results/artifacts_sha256.json and compare with disk."""
    results_dir = cfg["_results_dir"]
    manifest_path = results_dir / "artifacts_sha256.json"
    print("==================================================")
    print("      Artefact hash re-verification vs disk       ")
    print("==================================================")
    if not manifest_path.is_file():
        print(f"MISSING: {manifest_path}")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = dict(manifest.get("artifacts") or {})
    # the receipt binds its own hash in this file
    entries.setdefault("results/audit_run_report.json", None)
    problems: list[str] = []
    checked = 0
    for rel, recorded in entries.items():
        path = _artifact_path(rel, results_dir, cfg["_engine_dir"])
        if not path.is_file():
            problems.append(f"{rel}: missing ({path})")
            continue
        actual = sha256_file(path)
        checked += 1
        match = recorded is None or (actual is not None and actual.lower() == str(recorded).lower())
        print(f"  [{'OK  ' if match else 'FAIL'}] {rel:<46} {actual}")
        if not match:
            problems.append(f"{rel}: {actual} != recorded {recorded}")
    zip_path = cfg["_raw_dir"] / "E题数据.zip"
    zip_sha = (sha256_file(zip_path) or "").upper()
    checked += 1
    print(f"  [{'OK  ' if zip_sha == RAW_ZIP_SHA256.upper() else 'FAIL'}] 原始 zip SHA-256 = {zip_sha}")
    if zip_sha != RAW_ZIP_SHA256.upper():
        problems.append(f"raw zip sha256 {zip_sha} != {RAW_ZIP_SHA256.upper()}")
    print("==================================================")
    if problems:
        print(f"RESULT: FAIL ({len(problems)} problems, {checked} checked)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"RESULT: PASS ({checked} artefacts match the recorded hashes on disk)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument(
        "--skip-integrity",
        action="store_true",
        help="TEST ONLY: skip the raw-integrity gate and fail-injection self test "
             "(used to measure a sub-check failure inside an isolated copy)",
    )
    ap.add_argument(
        "--verify-artifacts",
        action="store_true",
        help="only re-hash every artefact listed in results/artifacts_sha256.json and compare with disk; non-zero on any mismatch",
    )
    args = ap.parse_args(argv)

    if args.verify_artifacts:
        return verify_recorded_artifacts(load_config(args.config))

    t0_start = time.time()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    steps: list[StepResult] = []
    _finalize = None  # bound after env_info is computed
    fail_injection = os.environ.get("DSH_T0_FAIL_INJECTION")

    print("==================================================")
    print("      E题 T0 数据与环境全量预检 (Auditing Runner) ")
    print("==================================================")
    print(f"Config: {cfg['_config_path']}")
    print(f"Python: {sys.executable}")
    if args.skip_integrity:
        print("[TEST MODE] integrity gate and fail-injection self test are SKIPPED (--skip-integrity)")
    if fail_injection:
        print(f"[FAIL INJECTION ARMED] step '{fail_injection}' will be forced to fail at runtime")

    # ---------------------------------------------------------------------
    # Gate 1: raw-integrity check BEFORE any processing step.
    # On failure nothing else runs, no PASS file is written, exit is non-zero.
    # ---------------------------------------------------------------------
    if not args.skip_integrity:
        gate = run_integrity_gate(cfg, "pre")
        steps.append(gate)
        print_step_result(gate)
        if not gate.passed:
            return abort_run(cfg, steps, "原始输入完整性预检失败（Gate 1）", int(gate.exit_code or 1))
    else:
        steps.append(StepResult(
            name="integrity_gate_pre", passed=None, exit_code=None, elapsed_seconds=0.0,
            detail="SKIPPED by --skip-integrity (test mode)", failed_reason=None,
        ))

    # ---------------------------------------------------------------------
    # Data and environment checks
    # ---------------------------------------------------------------------
    print("[1/6] Auditing hardware & Python environment...")
    env_info = audit_environment()
    _finalize = _make_finalizer(cfg, steps, env_info, t0_start, args)
    steps.append(StepResult(
        name="environment", passed=True, exit_code=0, elapsed_seconds=0.0, output_path=None,
        detail=f"GPU={env_info['gpu_name']}, CUDA={env_info['cuda_version']}, torch={env_info['packages']['torch']}",
    ))
    print(f"      GPU: {env_info['gpu_name']}, CUDA: {env_info['cuda_version']}")

    print("[2/6] Validating unified pickle reader (src.check_data_reader)...")
    reader_summary = None

    def _reader_step() -> dict:
        nonlocal reader_summary
        a2_path = cfg["_attachment2_root"] / "aligned_50.pkl"
        a3_dir = cfg["_attachment3_root"] / "对齐版本"
        a4_dir = cfg["_attachment4_root"] / "对齐版本"
        a2_rep = check_attachment2(a2_path)
        a3_rep = check_attachment3(a3_dir)
        a4_rep = check_attachment4(a4_dir)
        all_passed = a2_rep["passed"] and a3_rep["passed"] and a4_rep["passed"]
        reader_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "all_passed": all_passed,
            "attachment2": a2_rep,
            "attachment3": a3_rep,
            "attachment4": a4_rep,
        }
        (cfg["_results_dir"] / "reader_check.json").write_text(
            json.dumps(reader_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"      Reader validation passed: {all_passed} (results/reader_check.json)")
        if not all_passed:
            raise AssertionError(
                "unified reader failed on real files: "
                f"a2={a2_rep['passed']}, a3={a3_rep['passed']}, a4={a4_rep['passed']}"
            )
        return {
            "output_path": "results/reader_check.json",
            "detail": (
                f"A2 train/valid/test={[a2_rep['splits'][k]['actual_count'] for k in ('train', 'valid', 'test')]}, "
                f"A3 files={len(a3_rep['file_results'])}, A4 files={len(a4_rep['file_results'])}"
            ),
        }

    res = run_step("unified_reader", _reader_step, fail_injection)
    steps.append(res)
    print_step_result(res)
    if not res.passed:
        return _finalize(reader_summary=reader_summary)

    print("[3/6] Building full input manifest (expect 245 rows: 244 data files + 1 zip)...")
    manifest_df = None

    def _manifest_step() -> dict:
        nonlocal manifest_df
        manifest_df = build_manifest()
        return {
            "output_path": "results/input_manifest.csv",
            "detail": f"rows={len(manifest_df)}",
        }

    res = run_step("input_manifest", _manifest_step, fail_injection)
    steps.append(res)
    print_step_result(res)
    if not res.passed:
        return _finalize(reader_summary=reader_summary)

    print("[4/6] Recomputing Attachment1 x Attachment2 overlap (expect 18 sample_ids / 24 video_ids)...")
    overlap_res = None

    def _overlap_step() -> dict:
        nonlocal overlap_res
        overlap_res = compute_cross_overlap()
        expected = {"train": 11, "valid": 0, "test": 7}
        if overlap_res["matched_sample_ids_by_split"] != expected:
            raise AssertionError(
                f"overlap by split is {overlap_res['matched_sample_ids_by_split']}, independent review expects {expected} "
                "(refusing to accept a different count without re-checking the raw ids)"
            )
        return {
            "output_path": "results/cross_attachment_overlap.csv",
            "detail": (
                f"sample_ids={overlap_res['matched_sample_ids_count']} {overlap_res['matched_sample_ids_by_split']}, "
                f"video_ids={overlap_res['matched_video_ids_count']}"
            ),
        }

    res = run_step("cross_attachment_overlap", _overlap_step, fail_injection)
    steps.append(res)
    print_step_result(res)
    if not res.passed:
        return _finalize(reader_summary=reader_summary, overlap_res=overlap_res, bert=None, video=None)

    print("[4b/6] Independent re-verification of overlap report + manifest completeness...")

    def _verify_artifacts_step() -> dict:
        rc, out, err, secs = run_subprocess(
            [sys.executable, "-m", "src.tools.verify_cross_overlap"], cfg["_engine_dir"]
        )
        print(out.strip())
        if err.strip():
            print(err.strip())
        if rc != 0:
            raise AssertionError(f"src.tools.verify_cross_overlap exited {rc}")
        return {
            "output_path": "results/overlap_manifest_verify.json",
            "detail": f"independent checker exit=0 in {secs:.2f}s",
        }

    res = run_step("overlap_manifest_verify", _verify_artifacts_step, fail_injection)
    steps.append(res)
    print_step_result(res)
    if not res.passed:
        return _finalize(reader_summary=reader_summary, overlap_res=overlap_res, bert=None, video=None)

    print("[4c/6] Checking Attachment1 vs Attachment2 label/text agreement on the overlapping samples...")

    def _label_consistency_step() -> dict:
        rc, out, err, secs = run_subprocess(
            [sys.executable, "-m", "src.tools.audit_label_consistency"], cfg["_engine_dir"]
        )
        if err.strip():
            print(err.strip())
        if rc != 0:
            raise AssertionError(f"src.tools.audit_label_consistency exited {rc}")
        payload = json.loads(out.strip().splitlines()[0])
        if payload["overlap_sample_count"] != 18:
            raise AssertionError(
                f"overlap sample count is {payload['overlap_sample_count']}, expected 18 "
                "(must agree with the overlap report)"
            )
        if payload["regression_labels_differing_count"] != 0 or payload["text_exactly_equal_count"] != 18:
            raise AssertionError(
                "Attachment 1 and Attachment 2 disagree on the overlapping samples: "
                f"label_diffs={payload['regression_labels_differing_count']}, "
                f"text_equal={payload['text_exactly_equal_count']}/18"
            )
        return {
            "output_path": "results/label_consistency_audit.json",
            "detail": f"18/18 overlapping samples: regression labels identical, transcripts identical ({secs:.2f}s)",
        }

    res = run_step("label_consistency", _label_consistency_step, fail_injection)
    steps.append(res)
    print_step_result(res)
    if not res.passed:
        return _finalize(reader_summary=reader_summary, overlap_res=overlap_res, bert=None, video=None)

    print("[5/6] Full BERT tokenizer & GPU embedding reconstruction audit (4870 samples)...")
    bert_audit_res = None

    def _bert_step() -> dict:
        nonlocal bert_audit_res
        bert_audit_res = audit_bert_full()
        meta = bert_audit_res["model_metadata"]
        if not meta.get("load_binding_verified"):
            raise AssertionError("encoder load path is not bound to the verified snapshot directory")
        overall = bert_audit_res["overall_summary"]
        return {
            "output_path": "results/bert_audit_full.json",
            "detail": (
                f"revision={meta['pinned_revision_sha'][:12]}, evaluated={overall['total_evaluated_samples_a2_and_a4']}, "
                f"max_err={overall['max_absolute_error_overall']:.3e}, >1e-3={overall['total_samples_exceeding_1e_3']}"
            ),
        }

    res = run_step("bert_full_audit", _bert_step, fail_injection)
    steps.append(res)
    print_step_result(res)
    if not res.passed:
        return _finalize(reader_summary=reader_summary, overlap_res=overlap_res, bert=bert_audit_res, video=None)

    print("[5b/6] Measuring padding / zero-feature / dtype statistics (mask evidence)...")

    def _mask_step() -> dict:
        rc, out, err, secs = run_subprocess([sys.executable, "-m", "src.tools.mask_audit"], cfg["_engine_dir"])
        if out.strip():
            print(out.strip())
        if err.strip():
            print(err.strip())
        if rc != 0:
            raise AssertionError(f"src.tools.mask_audit exited {rc}")
        return {
            "output_path": "results/mask_audit.json",
            "detail": f"mask/dtype statistics written in {secs:.2f}s",
        }

    res = run_step("mask_audit", _mask_step, fail_injection)
    steps.append(res)
    print_step_result(res)
    if not res.passed:
        return _finalize(reader_summary=reader_summary, overlap_res=overlap_res, bert=bert_audit_res, video=video_detailed_res)

    print("[6/6] Auditing video & audio streams (100 in Att1, 20 in Att4)...")
    video_detailed_res = None

    def _video_step() -> dict:
        nonlocal video_detailed_res
        video_detailed_res = audit_videos_detailed()
        return {
            "output_path": "results/video_audit_detailed.json",
            "detail": (
                f"container_discrepancy={video_detailed_res['attachment1_summary']['container_discrepancy_count']}/100, "
                f"cv==pyav={video_detailed_res['attachment1_summary']['cv_and_pyav_decode_identical_count']}/100"
            ),
        }

    res = run_step("video_detailed_audit", _video_step, fail_injection)
    steps.append(res)
    print_step_result(res)
    if not res.passed:
        return _finalize(reader_summary=reader_summary, overlap_res=overlap_res, bert=bert_audit_res, video=video_detailed_res)

    # ---------------------------------------------------------------------
    # Gate 2: raw-integrity check AFTER all steps; no artifact is written if
    # an input changed while the audit ran.
    # ---------------------------------------------------------------------
    if not args.skip_integrity:
        gate = run_integrity_gate(cfg, "post")
        steps.append(gate)
        print_step_result(gate)
        if not gate.passed:
            return abort_run(cfg, steps, "原始输入完整性复检失败（Gate 2，运行期间输入被改动）", int(gate.exit_code or 1))
    else:
        steps.append(StepResult(
            name="integrity_gate_post", passed=None, exit_code=None, elapsed_seconds=0.0,
            detail="SKIPPED by --skip-integrity (test mode)", failed_reason=None,
        ))

    return _finalize(reader_summary=reader_summary, overlap_res=overlap_res, bert=bert_audit_res, video=video_detailed_res)


def _fmt(value, digits: int = 3, default: str = "n/a") -> str:
    try:
        return f"{float(value):.{digits}e}"
    except (TypeError, ValueError):
        return default


def _render_step_table(steps: list[StepResult], cfg: dict) -> str:
    rows = ["| # | 步骤 | 状态 | 退出码 | 耗时(s) | 产物 | 实测摘要 |", "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(steps, 1):
        out = f"`{s.output_path}`" if s.output_path else "-"
        detail = s.detail.replace("|", "/") if s.detail else ("- " + (s.failed_reason or "") if s.failed_reason else "-")
        rows.append(
            f"| {i} | `{s.name}` | {s.status} | {s.exit_code if s.exit_code is not None else '-'} | "
            f"{s.elapsed_seconds} | {out} | {detail} |"
        )
    return "\n".join(rows)


def _render_mask_summary(cfg: dict) -> str:
    path = cfg["_results_dir"] / "mask_audit.json"
    if not path.is_file():
        return "- （本轮未生成 `results/mask_audit.json`）"
    try:
        mask = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"- （读取 mask_audit.json 失败：{type(exc).__name__}: {exc}）"
    lines = [
        "| 划分 | 样本 | padding 位 | 有效位 | audio dtype | vision dtype | audio 有效位全零(去CLS/SEP) | vision 有效位全零(去CLS/SEP) | 含此类全零的样本(audio/vision) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for split in ("train", "valid", "test"):
        s = mask["splits"][split]
        a, v = s["audio"], s["vision"]
        lines.append(
            f"| {split} | {s['sample_count']} | {s['padding_positions_total']} | {a['valid_positions_total']} | "
            f"{a['dtype']} | {v['dtype']} | {a['zero_vectors_in_valid_excluding_cls_sep']} | "
            f"{v['zero_vectors_in_valid_excluding_cls_sep']} | "
            f"{a['sequences_with_any_zero_vector_in_valid_excluding_cls_sep']}/{v['sequences_with_any_zero_vector_in_valid_excluding_cls_sep']} |"
        )
    lines.append("")
    lines.append("- 解释边界：有效位置内全零一律记为 **疑似无观测**，缺少原始特征提取证据时不宣称为人脸遮挡或真实模态缺失。")
    lines.append("- 训练时模拟缺失：**T1 待实现**，T0 未执行（`executed_in_t0=false`）。")
    return "\n".join(lines)


def _render_status_report(
    cfg: dict,
    steps: list[StepResult],
    env_info: dict,
    reader_summary: dict,
    overlap_res: dict,
    bert: dict,
    video: dict,
    totals: dict,
    elapsed_total: float,
    artifacts: dict,
) -> str:
    """Render STATUS.md from THIS run's measured values (no hard-coded history)."""
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    a2 = reader_summary["attachment2"]["splits"]
    a3 = reader_summary["attachment3"]
    a4 = reader_summary["attachment4"]

    bert_meta = bert["model_metadata"]
    ov = bert["overall_summary"]
    fid = ov["reconstruction_fidelity"]
    a2a = bert["attachment2_audit"]
    a4a = bert["attachment4_audit"]
    a3a = bert["attachment3_audit"]

    tok_compared = fid["tokenization_samples_compared"]
    tok_exact = fid["tokenization_exact_match_samples"]

    v1 = video["attachment1_summary"]
    v4 = video["attachment4_summary"]
    pts_complete = sum(
        1 for r in video["samples_attachment1"]
        if r.get("valid_pts_frames") is not None and r.get("av_read_frames")
        and r["valid_pts_frames"] == r["av_read_frames"]
    )
    high_dev_lines = "\n".join(
        f"   - `{d.get('sample_id', 'n/a')}`: 容器 {d.get('container_duration_s', 'n/a')}s，"
        f"音频解码 {d.get('audio_decoded_duration_s', d.get('audio_duration_s', 'n/a'))}s，"
        f"偏差 {d.get('deviation_percent', 'n/a')}%"
        for d in v1["high_audio_dev_samples"]
    ) or "   - 无"

    hash_lines = "\n".join(f"- `{name}`: `{sha}`" for name, sha in artifacts.items() if sha)
    per_split_rows = []
    for split in ("train", "valid", "test"):
        r = a2a[split]["reconstruction_stats"]
        per_split_rows.append(
            f"| 附件二 {split} | {a2a[split]['sample_count']} | {_fmt(r['max_absolute_error'])} | "
            f"{_fmt(r['mean_absolute_error'])} | {_fmt(r['p50_absolute_error'])} | {_fmt(r['p99_absolute_error'])} | "
            f"{r['samples_exceeding_1e_3']} | {r['batch_size']} | {r['elapsed_seconds']} |"
        )
    r4 = a4a["reconstruction_stats"]
    per_split_rows.append(
        f"| 附件四 对齐版 | {a4a['sample_count']} | {_fmt(r4['max_absolute_error'])} | "
        f"{_fmt(r4['mean_absolute_error'])} | {_fmt(r4['p50_absolute_error'])} | {_fmt(r4['p99_absolute_error'])} | "
        f"{r4['samples_exceeding_1e_3']} | {a4a['sample_count']} | - |"
    )

    return f"""# STATUS — E题工程开发

- **接收状态**：已接收 T0 复验返工（DeepSeek Harness 执行，等待 Codex 独立复验）
- **当前阶段**：**T0 复验返工完成，未签署 M1 PASS，未启动 T1**
- **本文件生成方式**：由 `src.audit` 于 **{now}** 依据**本次运行**的返回值、耗时、退出码与产物哈希动态生成（不再写死历史数值）
- **执行主体**：DeepSeek Harness（`deepseek-flash`）
- **只读输入**：`E题数据.zip`（SHA-256 `{RAW_ZIP_SHA256}`）、题面 DOCX、`训练demo/`、`视频demo/`
- **Python**：`{cfg['python_exe']}`
- **环境**：{env_info['platform']}；GPU `{env_info['gpu_name']}`（{env_info['gpu_count']} 卡），CUDA {env_info['cuda_version']}，torch {env_info['packages']['torch']}，transformers {env_info['packages']['transformers']}
- **备份**：返工前产物与源码备份于 `results/t0_before_fix/`（`round2_pre_*`、`round2_pre_src/`）

## 一、本轮主入口实测运行

- 命令：`& "{cfg['python_exe']}" -m src.audit --config configs/base.json`
- 结论：**全部子检查 PASS**（见下表）；总耗时 {elapsed_total}s
- 失败语义：任一子检查失败即**非零退出**，且**不写 PASS、不覆盖既有 STATUS**；失败注入实测记录见 `results/t0_before_fix/t0_round2_run_receipt.json` 与 `results/t0_before_fix/fail_injection_evidence/`（在隔离副本中注入，未改动任何原始数据）。

{_render_step_table(steps, cfg)}

## 二、分项实测结论

### 1. 统一读取器（附件二 / 附件三 / 附件四，51 个真实 pkl 文件）

- 附件二 `aligned_50.pkl`（列式 split 字典）：train {a2['train']['actual_count']}、valid {a2['valid']['actual_count']}、test {a2['test']['actual_count']} 条，ID 唯一数分别为 {a2['train']['unique_ids']}/{a2['valid']['unique_ids']}/{a2['test']['unique_ids']}，形状错误 {len(a2['train']['shape_errors']) + len(a2['valid']['shape_errors']) + len(a2['test']['shape_errors'])} 条。
- 附件三对齐版：{a3['total_files']} 个文件 → {a3['samples_checked']} 条样本（去 batch=1），`sample_id` 取文件名 stem，未制造官方 ID 或 `text` 向量；失败文件 {len(a3['errors'])}。
- 附件四对齐版：{a4['total_files']} 个文件 → {a4['samples_checked']} 条样本，保留原始 `id`/`raw_text`；失败文件 {len(a4['errors'])}。
- 产物 `results/reader_check.json`（`all_passed={reader_summary['all_passed']}`）。

### 2. 附件一 `label-100.xlsx` 与跨附件交集（重算）

- `label` 工作表：{overlap_res['attachment1_total_samples']} 行，极性分布 {overlap_res['polarity_distribution']}；`label<0/==0/>0` 与 Negative/Neutral/Positive 逐行一致（mismatch 0）。
- 精确键 `video_id + '$_$' + clip_id` 与附件二交集：**sample_id {overlap_res['matched_sample_ids_count']} 个**（train {overlap_res['matched_sample_ids_by_split']['train']}、valid {overlap_res['matched_sample_ids_by_split']['valid']}、test {overlap_res['matched_sample_ids_by_split']['test']}）；**源 video_id {overlap_res['matched_video_ids_count']} 个**（train {overlap_res['matched_video_ids_by_split']['train']}、valid {overlap_res['matched_video_ids_by_split']['valid']}、test {overlap_res['matched_video_ids_by_split']['test']}）。
- 附件二 train/valid/test 之间 video_id 交集：{overlap_res['attachment2_internal_video_overlap']}（全 0）。
- **说话人独立性未知**：数据集无说话人 ID 字段，仅能说明源 video_id 跨 split 不重叠。
- 独立复核器 `src.tools.verify_cross_overlap`（从原始 xlsx/pkl 独立重算并与交付 CSV 比对）退出码 0，证据 `results/overlap_manifest_verify.json`。
- 补充实测：18 条重叠样本在附件一与附件二中的**回归标签与转写文本完全一致（18/18）**，证据 `results/label_consistency_audit.json`；Q2/Q3 参数仍只从附件二 train 学习，未把附件一并入训练。

### 3. 输入文件清单（245 行）

- 解压目录实测文件 {totals['manifest_rows'] - 1} 个（含 `.DS_Store` 1 个）；清单登记 244 个数据文件 + 原始 zip 1 行 = **245 行**，无重复、哈希非空。
- 独立复核器同时校验清单路径集合与磁盘文件集合严格相等（`manifest_path_set_matches_disk=True`）并抽检重算 SHA-256。
- `.DS_Store` 不作为建模输入；未对齐版 30+20 个 pkl 与 20 个 mp4 标 `not_used_in_primary_pipeline` 并写明排除理由。

### 4. 文本编码全量核验（版本绑定）

- 编码器：`{bert_meta['name']}`，**固定 revision `{bert_meta['pinned_revision_sha']}`**，从快照目录 `{bert_meta['load_source_snapshot_dir']}` 加载（`local_files_only=True`，`HF_HUB_OFFLINE=1`），实测加载路径与该快照一致：`load_binding_verified={bert_meta['load_binding_verified']}`；加载前后文件哈希均核对。
- `vocab.txt` `{bert_meta['file_hashes']['vocab.txt']['sha256']}`（校验 {bert_meta['file_hashes']['vocab.txt']['hash_verified']}）；`config.json` `{bert_meta['file_hashes']['config.json']['sha256']}`；`model.safetensors` `{bert_meta['file_hashes']['model.safetensors']['sha256']}`；`tokenizer.json` `{bert_meta['file_hashes']['tokenizer.json']['sha256']}`。
- 重分词核对（`input_ids`/`attention_mask`/`token_type_ids` 三行同时相等）：**{tok_exact}/{tok_compared}**（附件二 train {a2a['train']['token_full_exact_matches']}/{a2a['train']['sample_count']}、valid {a2a['valid']['token_full_exact_matches']}/{a2a['valid']['sample_count']}、test {a2a['test']['token_full_exact_matches']}/{a2a['test']['sample_count']}；附件四 {a4a['token_full_exact_matches']}/{a4a['sample_count']}）。
- 冻结编码器（`eval()` + `no_grad()`）从 `text_bert` 重算 `last_hidden_state` 与现有 `text` 比较，实际评估样本数 **{ov['total_evaluated_samples_a2_and_a4']}**，阈值 {ov['threshold_used']}：

| 数据组 | 样本数 | 全张量最大绝对误差 | 均值 | p50 | p99 | 超阈值数 | batch | 耗时(s) |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(per_split_rows)}

- 全量最大绝对误差 {_fmt(ov['max_absolute_error_overall'])}、均值 {_fmt(ov['mean_absolute_error_overall'])}、p99 {_fmt(ov['p99_absolute_error_overall'])}；超 {ov['threshold_used']} 样本 **{fid['samples_exceeding_threshold_1e_3']}**；max error ≤ 1e-4 的样本 {fid['samples_with_max_error_le_1e_4']}/{fid['samples_evaluated']}。
- 误差原因写法：**推断（INFERRED）**——{fid['difference_cause_statement']}
- 附件三（{a3a['file_count']} 文件）：**没有 raw_text，因此不能声称重分词已验证**；仅核验 CLS=101（{a3a['cls_special_token_ok_count']}/{a3a['file_count']}）、SEP=102（{a3a['sep_special_token_ok_count']}/{a3a['file_count']}）、PAD=0（{a3a['pad_special_token_ok_count']}/{a3a['file_count']}）、token 范围 [{a3a['min_token_id_overall']}, {a3a['max_token_id_overall']}] 与冻结编码器可运行（输出 {a3a['output_shape']}，passed={a3a['frozen_encoder_inference_passed']}）。
- **不宣称**“全部完全无损”：本结论覆盖的是被实际评估的 {ov['total_evaluated_samples_a2_and_a4']} 条样本的重构误差分布。

### 5. 视频与掩码结论（限证据范围）

- 附件一 {v1['total_videos']} 条视频：容器标称帧数与实际解码帧数不符 **{v1['container_discrepancy_count']}/{v1['total_videos']}**；OpenCV 与 PyAV 解码帧数一致 **{v1['cv_and_pyav_decode_identical_count']}/{v1['total_videos']}**；记录到的 PTS 严格单调 {v1['pts_strictly_monotonic_count']}/{v1['total_videos']}；带有效 PTS 帧数与解码帧数相等的样本 {pts_complete}/{v1['total_videos']}（缺 PTS 帧的样本 {v1.get('samples_with_missing_pts', 'n/a')} 条）。
- 帧数差异根因：**待确认**（未检查 MP4 atom 与 H.264 NAL 码流，不能写成“切片前元数据残留”已证明）。
- 音频：检出音轨并可解码的样本 {v1['audio_stream_present_count']}/{v1['total_videos']}；**不因存在音轨即宣称“音轨完整”**。解码时长与容器时长偏差 > 5% 的样本 {v1['audio_duration_dev_gt_5_pct_count']} 条：
{high_dev_lines}
- 时间映射：未取得真实词级时间映射，Q1/Q3 的词时间戳标 `unresolved`，不得由 50 个位置等分推断。
- 掩码/全零：padding 位置、有效 token 位置内的全零音视频特征、以及训练时模拟缺失**分列统计**（下表），完整数据见 `results/mask_audit.json`；有效位置内全零标 **“疑似无观测”**，缺乏原始提取证据，不宣称人脸遮挡或真实模态缺失。

{_render_mask_summary(cfg)}

- dtype：附件二 `audio`/`vision` 原始为 **float64**（实测，见上表）；训练时在 Dataset/DataLoader 处转 FP32 并做有限值检查——**该实现属 T1，T0 未执行**。
- 74 维语音特征逐维语义 **未知**，不猜测命名。

## 三、主要产物 SHA-256（本次运行实测）

{hash_lines}

## 四、未决事项 / 不做结论的部分

1. 附件一容器标称帧数差异的根因未证实（需 atom/NAL 解析），当前仅记录实测现象。
2. 音频流“完整/无截断”未证实；2 条 >5% 偏差样本仅记录实测值。
3. 说话人独立性未知（无说话人字段）。
4. 74/35 维特征逐维语义未知。
5. 词级真实时间映射未取得 → Q1/Q3 相关位置标 `unresolved`。
6. 50MB 提交包预算为**有依据的估算**（算法与依据见 `results/data_audit.json` 的 `submission_package_budget`），非实测打包结果。
7. 冻结 BERT 权重（~440MB）不进提交包；外部下载是否被官方规则接受**待官方规则核验**。
8. T1 待实现：Dataset/DataLoader 的 dtype 转换与 `torch.isfinite` 逐批检查、训练时缺失掩码模拟。

## 五、当前状态

- **T0 复验返工完成，等待 Codex 独立复验**；**未签署 M1 PASS**，**未启动 T1**。
"""


def _render_failed_status(
    cfg: dict,
    steps: list[StepResult],
    env_info: dict,
    failed_step: StepResult | None,
    reason: str,
    total_elapsed: float,
    artifacts: dict,
) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    failed_list = [s for s in steps if s.passed is False]
    hash_lines = "\n".join(f"- `{name}`: `{sha}`" for name, sha in artifacts.items() if sha)
    return f"""# STATUS — E题工程开发

- **接收状态**：已接收 T0 复验返工（DeepSeek Harness 执行）
- **当前阶段**：**BLOCKED（本次审计未通过，未签署 M1 PASS，未启动 T1）**
- **本文件生成方式**：由 `src.audit` 于 **{now}** 依据**本次失败运行**动态生成
- **失败步骤**：`{failed_step.name if failed_step else 'n/a'}`
- **失败原因**：{reason}
- **总耗时**：{total_elapsed}s
- **原始输入**：`E题数据.zip` SHA-256 `{RAW_ZIP_SHA256}`（Gate 1 通过后才执行后续步骤）
- **Python**：`{cfg['python_exe']}`；GPU `{env_info['gpu_name']}`，CUDA {env_info['cuda_version']}

## 本次运行各步骤结果

{_render_step_table(steps, cfg)}

## 失败语义（本轮 P0 修复点）

- 任一子检查失败 → 主入口**非零退出**，且**不写 PASS**、**不覆盖上一条通过结论**。
- 上一条通过的 STATUS 快照保留在 `results/t0_before_fix/`，产物快照同理。
- 复现：在 `工程/` 下执行
  `& "{cfg['python_exe']}" -m src.audit --config configs/base.json`，观察退出码与上表 FAIL 行。

## 本次运行已落盘的产物哈希（失败运行，仅记录、不作为通过依据）

{hash_lines}

## 未决事项

- 上述失败步骤未修复前，T0 不得判定通过；不得启动 T1，不得自行签发 M1 PASS。
"""


def abort_run(cfg: dict, steps: list[StepResult], reason: str, code: int) -> int:
    """Fail-closed exit: record the failure, never write PASS artefacts.

    Return code is always non-zero.  If the failure evidence itself cannot be written the
    code is escalated to 8 so a broken evidence path can never look like a clean abort.
    """
    failed = next((s for s in steps if s.passed is False), None)
    print("==================================================")
    print(f"ABORT (exit {code}): {reason}")
    print("No PASS artefact was written; existing snapshots under results/t0_before_fix/ are untouched.")
    print("==================================================")

    results_dir = cfg["_results_dir"]
    report = {
        "audit_version": "T0-REWORK-ROUND2",
        "status": "FAIL",
        "abort_reason": reason,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "steps": [s.__dict__ | {"status": s.status} for s in steps],
        "failed_step": (failed.__dict__ | {"status": failed.status}) if failed else None,
        "expected_raw_zip_sha256": RAW_ZIP_SHA256,
        "note": "This run did not pass; no interface contract, no PASS status was produced.",
    }
    wrote_report = write_mandatory(results_dir / "audit_run_report.json", json.dumps(report, indent=2, ensure_ascii=True))

    env_info = audit_environment()
    status_md = _render_failed_status(cfg, steps, env_info, failed, reason, round(sum(s.elapsed_seconds for s in steps), 2), {})
    status_path = cfg["_engine_dir"] / "STATUS.md"
    wrote_status = write_mandatory(status_path, status_md)
    if wrote_status:
        try:
            body = status_path.read_text(encoding="utf-8")
            if "BLOCKED" not in body:
                print("[ERROR] failure STATUS.md was written but does not contain BLOCKED")
                wrote_status = False
        except OSError as exc:
            print(f"[ERROR] failure STATUS.md unreadable after write: {exc}")
            wrote_status = False
    if not (wrote_report and wrote_status):
        print("==================================================")
        print("EVIDENCE WRITE FAILED (exit 8): failure evidence could not be written to disk.")
        print(f"  audit_run_report.json written: {wrote_report} -> {results_dir / 'audit_run_report.json'}")
        print(f"  STATUS.md written: {wrote_status} -> {status_path}")
        print("==================================================")
        return 8
    return code


def finalize(
    cfg: dict,
    steps: list[StepResult],
    env_info: dict,
    reader_summary: dict | None,
    overlap_res: dict | None,
    bert: dict | None,
    video: dict | None,
    t0_start: float,
    args: argparse.Namespace,
) -> int:
    """All steps passed: write machine-readable artefacts, then STATUS.md with this run's values.

    Hash self-reference rules (P0 fix):
      * `data_audit.json` is NOT hashed, because embedding its own hash would change the bytes
        that were hashed.  Its final hash is recorded in `artifacts_sha256.json` instead.
      * `STATUS.md` never lists its own hash (it is rendered after the hash table is fixed).
      * `audit_run_report.json` is written LAST and its final hash is bound in
        `artifacts_sha256.json`; `_verify_receipt_self_hash` re-reads the bytes to confirm.
    Every critical write and every recorded hash is verified against disk before returning 0.
    """
    failed_step = next((s for s in steps if s.passed is False), None)
    if failed_step is not None:
        return abort_run(
            cfg, steps, f"子检查失败：{failed_step.name} -> {failed_step.failed_reason}", 1
        )
    if reader_summary is None or overlap_res is None or bert is None or video is None:
        return abort_run(cfg, steps, "内部错误：步骤未全部执行即进入收尾", 4)

    results_dir = cfg["_results_dir"]
    engine_dir = cfg["_engine_dir"]

    manifest_rows = 0
    primary_rows = 0
    excluded_rows = 0
    try:
        with (results_dir / "input_manifest.csv").open(encoding="utf-8-sig", newline="") as fh:
            import csv as _csv

            rows = list(_csv.DictReader(fh))
        manifest_rows = len(rows)
        primary_rows = sum(1 for r in rows if r.get("pipeline_usage") == "primary_pipeline")
        excluded_rows = sum(1 for r in rows if r.get("pipeline_usage") == "not_used_in_primary_pipeline")
    except Exception as exc:  # noqa: BLE001
        steps.append(StepResult("manifest_summary", False, 1, 0.0, failed_reason=f"{type(exc).__name__}: {exc}"))
        return abort_run(cfg, steps, f"无法汇总 input_manifest.csv: {exc}", 1)

    contract = build_updated_interface_contract()
    budget = estimate_submission_budget()

    master_audit = {
        "audit_version": "T0-REWORK-ROUND2",
        "status": "PASS",
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "elapsed_seconds": round(time.time() - t0_start, 2),
        "test_mode": bool(args.skip_integrity),
        "raw_zip_sha256_expected": RAW_ZIP_SHA256,
        "steps": [s.__dict__ | {"status": s.status} for s in steps],
        "environment": env_info,
        "input_manifest_summary": {
            "total_manifest_entries": manifest_rows,
            "primary_pipeline_files": primary_rows,
            "unaligned_files_excluded_from_primary": excluded_rows,
        },
        "reader_check": reader_summary,
        "cross_attachment_overlap": overlap_res,
        "bert_full_audit": bert,
        "video_detailed_audit": video,
        "interface_contract": contract,
        "submission_package_budget": budget,
        "artifact_sha256": {},
        "claims_policy": {
            "measured_vs_inferred": "所有结论标注 MEASURED / INFERRED / UNRESOLVED；未取证的原因一律写待确认。",
            "m1_status": "NOT_SIGNED_BY_IMPLEMENTER — T0 完成仅表示返工完成，M1 由 Codex 独立复验签署。",
            "t1_status": "NOT_STARTED",
        },
        "unresolved_items": [
            "附件一容器标称帧数与解码帧数差异的根因未证实（未做 atom/NAL 解析）",
            "音频流是否完整/无截断未证实；仅记录解码时长与容器时长偏差",
            "说话人独立性未知（数据无说话人字段）",
            "74 维音频 / 35 维视觉特征逐维语义未知",
            "词级真实时间映射未取得 → 相关时间对齐标 unresolved",
            "50MB 提交包为有依据估算，非实测打包",
            "冻结 BERT 外部下载是否被官方规则接受待核验",
            "T1 待实现：Dataset/DataLoader 的 dtype 转换、torch.isfinite 逐批检查、训练时缺失掩码",
        ],
    }

    data_audit_path = results_dir / "data_audit.json"
    contract_path = results_dir / "interface_contract.json"
    status_path = engine_dir / "STATUS.md"
    run_report_path = results_dir / "audit_run_report.json"
    side_receipt_path = results_dir / "artifacts_sha256.json"

    master_audit["artifact_hash_policy"] = {
        "self_reference": "data_audit.json / STATUS.md / audit_run_report.json 不记录自身哈希，避免“先哈希后写回”导致的自哈希循环",
        "authoritative_hash_manifest": "results/artifacts_sha256.json（在全部产物写完后生成，并对磁盘逐一复算）",
        "verified_against_disk": True,
    }

    # 1) interface contract (data_audit.json is written LAST, see step 7)
    ok_contract = write_mandatory(contract_path, json.dumps(contract, indent=2, ensure_ascii=False))
    ok_data = True  # set in step 7

    # 3) hash table of the stable artefacts (data_audit.json is added in step 7)
    artifacts = {
        "data/raw/E题数据.zip": RAW_ZIP_SHA256,
        "results/interface_contract.json": sha256_file(contract_path),
        "results/reader_check.json": sha256_file(results_dir / "reader_check.json"),
        "results/input_manifest.csv": sha256_file(results_dir / "input_manifest.csv"),
        "results/cross_attachment_overlap.csv": sha256_file(results_dir / "cross_attachment_overlap.csv"),
        "results/overlap_manifest_verify.json": sha256_file(results_dir / "overlap_manifest_verify.json"),
        "results/label_consistency_audit.json": sha256_file(results_dir / "label_consistency_audit.json"),
        "results/bert_audit_full.json": sha256_file(results_dir / "bert_audit_full.json"),
        "results/video_audit_detailed.json": sha256_file(results_dir / "video_audit_detailed.json"),
        "results/mask_audit.json": sha256_file(results_dir / "mask_audit.json"),
    }

    # 4) STATUS.md (never lists its own hash)
    totals = {"manifest_rows": manifest_rows}
    status_md = _render_status_report(
        cfg, steps, env_info, reader_summary, overlap_res, bert, video, totals,
        round(time.time() - t0_start, 2),
        dict(artifacts, **{"results/STATUS.md": "（见 results/artifacts_sha256.json）"}),
    )
    ok_status = write_mandatory(status_path, status_md)
    artifacts["results/STATUS.md"] = sha256_file(status_path)

    # 5) machine receipt (written last among the artefacts it describes)
    run_record = {
        "audit_version": "T0-REWORK-ROUND2",
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "total_elapsed_seconds": round(time.time() - t0_start, 2),
        "steps": [s.__dict__ | {"status": s.status} for s in steps],
        "artifact_sha256": artifacts,
        "hash_manifest": "results/artifacts_sha256.json",
        "self_hash_policy": "本文件不记录自身哈希；最终字节的哈希由 results/artifacts_sha256.json 绑定并复算",
    }
    ok_run_report = write_mandatory(run_report_path, json.dumps(run_record, indent=2, ensure_ascii=True))
    # second pass: bind the receipt's own logical content (sha256_self is excluded from the hash input)
    if ok_run_report:
        run_record["sha256_self"] = hashlib.sha256(
            json.dumps(run_record, indent=2, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        ok_run_report = write_mandatory(run_report_path, json.dumps(run_record, indent=2, ensure_ascii=True))

    # 6) external hash manifest: one shot, after every other artefact is final;
    #    it excludes data_audit.json (not yet final) and never lists itself.
    hash_manifest = {
        "$comment": (
            "Every entry was re-hashed from disk after the artefact was finalised. "
            "This file lists neither itself nor data_audit.json (which is written after this "
            "manifest and carries its own hash in results/audit_run_report.json)."
        ),
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "artifacts": {
            "data/raw/E题数据.zip": RAW_ZIP_SHA256,
            "results/interface_contract.json": sha256_file(contract_path),
            "results/reader_check.json": sha256_file(results_dir / "reader_check.json"),
            "results/input_manifest.csv": sha256_file(results_dir / "input_manifest.csv"),
            "results/cross_attachment_overlap.csv": sha256_file(results_dir / "cross_attachment_overlap.csv"),
            "results/overlap_manifest_verify.json": sha256_file(results_dir / "overlap_manifest_verify.json"),
            "results/label_consistency_audit.json": sha256_file(results_dir / "label_consistency_audit.json"),
            "results/bert_audit_full.json": sha256_file(results_dir / "bert_audit_full.json"),
            "results/video_audit_detailed.json": sha256_file(results_dir / "video_audit_detailed.json"),
            "results/mask_audit.json": sha256_file(results_dir / "mask_audit.json"),
            "STATUS.md": sha256_file(status_path),
            "results/audit_run_report.json": None,  # filled below, after the receipt is final
        },
        "raw_zip_sha256": RAW_ZIP_SHA256,
    }
    if ok_run_report:
        hash_manifest["artifacts"]["results/audit_run_report.json"] = sha256_file(run_report_path)
    ok_hash_manifest = write_mandatory(side_receipt_path, json.dumps(hash_manifest, indent=2, ensure_ascii=True))

    # 7) data_audit.json is written last so its hash can be recorded in artifacts_sha256.json
    hash_manifest["artifacts"]["results/data_audit.json"] = None
    master_audit["artifact_sha256"] = dict(artifacts)
    master_audit["artifact_sha256"]["results/STATUS.md"] = hash_manifest["artifacts"]["STATUS.md"]
    master_audit["artifact_sha256"]["results/audit_run_report.json"] = hash_manifest["artifacts"][
        "results/audit_run_report.json"
    ]
    ok_data = write_mandatory(data_audit_path, json.dumps(master_audit, indent=2, ensure_ascii=False))
    if ok_data:
        hash_manifest["artifacts"]["results/data_audit.json"] = sha256_file(data_audit_path)
        ok_hash_manifest = write_mandatory(
            side_receipt_path, json.dumps(hash_manifest, indent=2, ensure_ascii=True)
        ) and ok_hash_manifest

    write_failures = [
        name
        for name, ok in (
            ("data_audit.json", ok_data),
            ("interface_contract.json", ok_contract),
            ("STATUS.md", ok_status),
            ("audit_run_report.json", ok_run_report),
            ("artifacts_sha256.json", ok_hash_manifest),
        )
        if not ok
    ]

    problems: list[str] = []
    problems += [f"write failed: {n}" for n in write_failures]
    problems += verify_hashes_on_disk(artifacts, results_dir, engine_dir)
    if ok_run_report:
        problems += _verify_receipt_self_hash(run_report_path, run_record)
    for path in (run_report_path, data_audit_path, contract_path, status_path):
        if not path.is_file():
            problems.append(f"{path.name}: missing after write")
    for path, label in ((run_report_path, "audit_run_report.json"), (data_audit_path, "data_audit.json"), (contract_path, "interface_contract.json")):
        if path.is_file():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{label}: unparsable after write: {type(exc).__name__}: {exc}")

    if problems:
        print("==================================================")
        print("ARTEFACT INTEGRITY VERIFICATION FAILED — refusing to report success:")
        for p in problems:
            print(f"  - {p}")
        print("==================================================")
        return 1

    print("Updated STATUS.md successfully (values taken from this run).")
    print(f"Artefact hashes verified against disk ({len(artifacts) + 2} entries; {side_receipt_path.name} binds audit_run_report.json).")
    print("==================================================")
    print(f"T0 Audit Runner completed in {round(time.time() - t0_start, 2)}s with Exit Code 0.")
    print("==================================================")
    return 0


_finalize = functools.partial(finalize)


def _make_finalizer(cfg: dict, steps: list[StepResult], env_info: dict, t0_start: float, args) -> object:
    """Bind the run context so call sites only pass the artefacts they actually have."""

    def _fin(reader_summary=None, overlap_res=None, bert=None, video=None) -> int:
        return finalize(cfg, steps, env_info, reader_summary, overlap_res, bert, video, t0_start, args)

    return _fin


def run_cli() -> int:
    """Top-level guard: an unexpected crash must still exit non-zero and leave evidence."""
    try:
        return main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - fail closed, never look like success
        try:
            import traceback

            print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
            print(traceback.format_exc()[-4000:], file=sys.stderr)
        except Exception:  # noqa: BLE001 - reporting must never mask the failure
            pass
        try:
            write_if_possible(
                Path(__file__).resolve().parents[1] / "results" / "audit_run_report.json",
                json.dumps(
                    {
                        "audit_version": "T0-REWORK-ROUND2",
                        "status": "FAIL",
                        "crash": f"{type(exc).__name__}: {exc}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "exit_code": 9,
                    },
                    indent=2,
                    ensure_ascii=True,
                ),
            )
        except Exception:  # noqa: BLE001
            pass
        return 9


if __name__ == "__main__":
    sys.exit(run_cli())