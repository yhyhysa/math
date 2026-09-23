"""Measured padding / zero-feature statistics for Attachment 2 (mask & dtype evidence).

Reports, per split and per modality (audio 74-d, vision 35-d):
  * padding positions (t >= text_len) and valid positions (0 <= t < text_len)
  * all-zero vectors by position class (whole sequence / padding / valid)
  * all-zero vectors in valid positions *excluding* CLS (t=0) and SEP (t=text_len-1)
  * exact-zero fraction of individual entries, and the measured numpy dtype

Interpretation boundary (do NOT over-claim):
  all-zero vectors inside valid positions are recorded as "疑似无观测"
  (suspected unobserved).  Without the original feature-extraction evidence this
  cannot be attributed to face occlusion or genuine modality absence.

Also checks label/annotation agreement inside Attachment 2 itself, since no
separate Attachment 2 annotation column is trusted without measurement.

Outputs:
  - results/mask_audit.json

Usage (cwd = 工程/):
  python -m src.tools.mask_audit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from src.config import load_config

DELIM = "$_$"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def modality_stats(arr: np.ndarray, valid: np.ndarray) -> dict:
    """arr: (N, T, D); valid: (N, T) boolean."""
    norms = np.linalg.norm(arr, axis=2)                 # (N, T)
    zero_vec = norms < 1e-6                             # exact-zero vectors (float safe)
    n, t, d = arr.shape
    pad = ~valid
    inner = valid.copy()
    inner[:, 0] = False                                 # CLS position
    len_each = valid.sum(axis=1)
    for i in range(n):
        if len_each[i] >= 1:
            inner[i, len_each[i] - 1] = False           # SEP / last valid position
    return {
        "shape": [int(n), int(t), int(d)],
        "dtype": str(arr.dtype),
        "padding_positions_total": int(pad.sum()),
        "valid_positions_total": int(valid.sum()),
        "zero_vectors_total": int(zero_vec.sum()),
        "zero_vectors_in_padding": int((zero_vec & pad).sum()),
        "zero_vectors_in_valid_positions": int((zero_vec & valid).sum()),
        "zero_vectors_in_valid_excluding_cls_sep": int((zero_vec & inner).sum()),
        "sequences_with_any_zero_vector_in_valid_positions": int(((zero_vec & valid).any(axis=1)).sum()),
        "sequences_with_any_zero_vector_in_valid_excluding_cls_sep": int(((zero_vec & inner).any(axis=1)).sum()),
        "exact_zero_entry_fraction_overall": float((arr == 0).mean()),
        "exact_zero_entry_fraction_in_valid_positions": float((arr == 0)[valid].mean()) if valid.any() else None,
        "all_zero_interpretation": "疑似无观测 (suspected unobserved); 缺少原始特征提取证据，不宣称为人脸遮挡或真实模态缺失",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--out", default="results/mask_audit.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    cfg = load_config(args.config)
    a2_path = cfg["_attachment2_root"] / "aligned_50.pkl"

    with open(a2_path, "rb") as fh:
        d2 = pickle.load(fh)

    report: dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "input": {"aligned_50_pkl": str(a2_path), "aligned_50_pkl_sha256": sha256_file(a2_path)},
        "mask_source": "Attachment 2 text_bert attention_mask (row 1); no mask is invented",
        "splits": {},
        "scope": ("Padding/zero-feature counting and dtype measurement only. "
                  "No claim is made about the cause of zero features."),
    }

    label_checks = {}
    for split in ("train", "valid", "test"):
        sp = d2[split]
        tb = np.asarray(sp["text_bert"])          # (N, 3, 50)
        valid = tb[:, 1].astype(bool)
        audio = np.asarray(sp["audio"])
        vision = np.asarray(sp["vision"])
        cls = np.asarray(sp["classification_labels"]).reshape(-1)
        reg = np.asarray(sp["regression_labels"]).reshape(-1)

        expected = np.where(reg < 0, 0, np.where(reg == 0, 1, 2)).astype(float)
        label_checks[split] = {
            "classification_label_matches_regression_sign": int((cls == expected).sum()),
            "sample_count": int(cls.size),
            "regression_min": float(reg.min()),
            "regression_max": float(reg.max()),
            "classification_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(cls, return_counts=True))},
        }

        report["splits"][split] = {
            "sample_count": int(audio.shape[0]),
            "sequence_length": int(audio.shape[1]),
            "text_len_min": int(valid.sum(axis=1).min()),
            "text_len_max": int(valid.sum(axis=1).max()),
            "text_len_mean": float(valid.sum(axis=1).mean()),
            "padding_positions_total": int((~valid).sum()),
            "text_padding_mask": {
                "definition": "True for active token steps (0 <= t < text_len), False for PAD",
                "cls_always_valid": bool(valid[:, 0].all()),
                "pad_entries_nonzero_id_count": int((tb[:, 0][~valid] != 0).sum()),
            },
            "audio": modality_stats(audio, valid),
            "vision": modality_stats(vision, valid),
        }

    report["label_consistency_within_attachment2"] = label_checks
    report["training_time_simulated_missing_mask"] = {
        "status": "T1_PENDING_IMPLEMENTATION",
        "definition": "训练时模拟缺失：仅在有效词步 (1 <= t <= text_len-2) 上施加连续遮蔽",
        "executed_in_t0": False,
        "note": "T0 未实现 Dataset/DataLoader，因此本轮没有任何训练时缺失模拟被执行。",
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = cfg["_engine_dir"] / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "status": "PASS",
        "report": str(out_path),
        "elapsed_seconds": round(time.time() - t0, 2),
        "summary": {
            s: {
                "audio_zero_valid_excl_cls_sep": report["splits"][s]["audio"]["zero_vectors_in_valid_excluding_cls_sep"],
                "vision_zero_valid_excl_cls_sep": report["splits"][s]["vision"]["zero_vectors_in_valid_excluding_cls_sep"],
                "audio_dtype": report["splits"][s]["audio"]["dtype"],
                "vision_dtype": report["splits"][s]["vision"]["dtype"],
            }
            for s in ("train", "valid", "test")
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
