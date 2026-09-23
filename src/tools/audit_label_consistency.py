"""Read-only consistency audit between Attachment 1 labels and Attachment 2 labels.

Context: the 18 (video_id, clip_id) pairs that occur in both Attachment 1
(`label-100.xlsx`) and Attachment 2 (`aligned_50.pkl`) are the only place where the
two label sources can be compared.  This tool reports the raw comparison and does
NOT decide the training policy; it only records measured agreement.

Usage (cwd = 工程/):
  python -m src.tools.audit_label_consistency
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
import pandas as pd

from src.config import load_config

DELIM = "$_$"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--out", default="results/label_consistency_audit.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    cfg = load_config(args.config)
    excel_path = cfg["_video_attachment1_root"] / "label-100.xlsx"
    a2_path = cfg["_attachment2_root"] / "aligned_50.pkl"

    df = pd.read_excel(excel_path, sheet_name="label")
    a1 = {}
    for i in range(len(df)):
        vid = str(df["video_id"].iloc[i]).strip()
        cid = str(df["clip_id"].iloc[i]).strip()
        a1[f"{vid}{DELIM}{cid}"] = {
            "label": float(df["label"].iloc[i]),
            "annotation": str(df["annotation"].iloc[i]).strip(),
            "text": str(df["text"].iloc[i]) if "text" in df.columns else None,
        }

    with open(a2_path, "rb") as fh:
        d2 = pickle.load(fh)

    rows = []
    for split in ("train", "valid", "test"):
        sp = d2[split]
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in sp["id"]]
        cls = np.asarray(sp["classification_labels"]).reshape(-1)
        reg = np.asarray(sp["regression_labels"]).reshape(-1)
        raw = sp["raw_text"] if "raw_text" in sp else None
        for i, sid in enumerate(ids):
            if sid not in a1:
                continue
            a2_text = str(raw[i]) if raw is not None else None
            rows.append({
                "sample_id": sid,
                "split": split,
                "a1_regression_label": a1[sid]["label"],
                "a1_annotation": a1[sid]["annotation"],
                "a2_regression_label": float(reg[i]),
                "a2_classification_label": float(cls[i]),
                "regression_abs_diff": abs(float(reg[i]) - a1[sid]["label"]),
                "regression_exact_equal": bool(float(reg[i]) == a1[sid]["label"]),
                "text_equal": (a1[sid]["text"] == a2_text) if (a1[sid]["text"] is not None and a2_text is not None) else None,
                "a1_text_excerpt": (a1[sid]["text"][:120] if a1[sid]["text"] else None),
                "a2_text_excerpt": (a2_text[:120] if a2_text else None),
            })

    n = len(rows)
    exact = sum(1 for r in rows if r["regression_exact_equal"])
    text_equal = sum(1 for r in rows if r["text_equal"])
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "overlap_sample_count": n,
        "regression_labels_exactly_equal_count": exact,
        "regression_labels_differing_count": n - exact,
        "max_regression_abs_diff": max((r["regression_abs_diff"] for r in rows), default=None),
        "text_exactly_equal_count": text_equal,
        "text_compared_count": sum(1 for r in rows if r["text_equal"] is not None),
        "rows": rows,
        "inputs": {
            "label_100_xlsx": str(excel_path),
            "label_100_xlsx_sha256": sha256_file(excel_path),
            "aligned_50_pkl": str(a2_path),
            "aligned_50_pkl_sha256": sha256_file(a2_path),
        },
        "scope": ("Comparison of Attachment 1 labels/transcripts with the 18 overlapping Attachment 2 samples. "
                  "Reported as measurement only; no training policy is asserted here."),
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = cfg["_engine_dir"] / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))
    for r in rows:
        print(f"  {r['split']:5s} {r['sample_id']:24s} a1_reg={r['a1_regression_label']:+.4f} "
              f"a2_reg={r['a2_regression_label']:+.4f} a1_ann={r['a1_annotation']:8s} text_equal={r['text_equal']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
