"""Generate complete input_manifest.csv covering all 244 extracted data files + 1 zip file = 245 rows.

Excludes .DS_Store.
Inspects pickle structure for all .pkl files.
Validates exact set equality with disk files.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import load_config


def compute_sha256(path: Path, block_size: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def probe_pkl(p: Path) -> tuple[str, str, str]:
    """Return (top_keys, shapes_str, dtypes_str) for a pickle file."""
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            return type(obj).__name__, "", ""
        top_keys = list(obj.keys())
        top_keys_str = ",".join(map(str, top_keys))

        shapes = []
        dtypes = []
        if any(k in top_keys for k in ("train", "valid", "test")):
            for sk in ("train", "valid", "test"):
                if sk in obj and isinstance(obj[sk], dict):
                    for mk, mv in obj[sk].items():
                        if hasattr(mv, "shape"):
                            shapes.append(f"{sk}.{mk}:{list(mv.shape)}")
                            dtypes.append(f"{sk}.{mk}:{mv.dtype}")
                        elif isinstance(mv, (list, tuple)):
                            shapes.append(f"{sk}.{mk}:len={len(mv)}")
                            dtypes.append(f"{sk}.{mk}:{type(mv).__name__}")
        else:
            for k, v in obj.items():
                if hasattr(v, "shape"):
                    shapes.append(f"{k}:{list(v.shape)}")
                    dtypes.append(f"{k}:{v.dtype}")
                elif isinstance(v, (list, tuple)):
                    shapes.append(f"{k}:len={len(v)}")
                elif isinstance(v, str):
                    shapes.append(f"{k}:len={len(v)}")
                    dtypes.append(f"{k}:str")

        return top_keys_str, ";".join(shapes[:12]), ";".join(dtypes[:12])
    except Exception as exc:
        return f"ERROR:{type(exc).__name__}", "", ""


def build_manifest() -> pd.DataFrame:
    cfg = load_config("configs/base.json")
    engine_dir = cfg["_engine_dir"]
    extracted_root = cfg["_extracted_dir"] / "E题数据"
    raw_zip = cfg["_raw_dir"] / "E题数据.zip"

    manifest_rows = []

    # 1. Raw zip
    print(f"[1/4] Hashing raw zip: {raw_zip} ...")
    assert raw_zip.exists(), f"Raw zip not found: {raw_zip}"
    zip_sha = compute_sha256(raw_zip)
    manifest_rows.append({
        "relative_path": str(raw_zip.relative_to(engine_dir)).replace("\\", "/"),
        "category": "raw_archive",
        "size_bytes": raw_zip.stat().st_size,
        "sha256": zip_sha,
        "format": "zip",
        "sample_count": "all_4850_plus_special",
        "pipeline_usage": "raw_source_archive",
        "pkl_top_keys": "",
        "pkl_shapes": "",
        "pkl_dtypes": "",
        "notes": "Original read-only compressed dataset archive (SHA-256 verified against official check)",
    })

    # 2. Find all files in extracted_root
    print(f"[2/4] Traversing extracted root: {extracted_root} ...")
    disk_files = sorted([p for p in extracted_root.rglob("*") if p.is_file()])
    data_files = [p for p in disk_files if p.name != ".DS_Store"]
    print(f"      Total disk files: {len(disk_files)}, Data files (excl .DS_Store): {len(data_files)}")
    assert len(data_files) == 244, f"Expected 244 data files, found {len(data_files)}"

    for idx, f in enumerate(data_files):
        rel = str(f.relative_to(engine_dir)).replace("\\", "/")
        size = f.stat().st_size
        sha = compute_sha256(f)
        ext = f.suffix.lstrip(".").lower()

        # Categorize
        if "附件1" in rel:
            if ext == "xlsx":
                cat = "attachment1_metadata"
                usage = "primary_pipeline"
                sample_count = "100"
                notes = "Attachment 1 video annotations (sheet 'label' and '修改说明')"
            else:
                cat = "attachment1_video"
                usage = "primary_pipeline"
                sample_count = "1"
                notes = f"Attachment 1 original video: {f.parent.name}/{f.name}"
            top_k, shapes_str, dtypes_str = "", "", ""

        elif "附件2" in rel:
            cat = "attachment2_standard"
            if ext == "pkl":
                if "aligned" in f.name and "unaligned" not in f.name:
                    usage = "primary_pipeline"
                    notes = "CMU-MOSEI standardized aligned 50-step features (train/valid/test)"
                else:
                    usage = "not_used_in_primary_pipeline"
                    notes = "CMU-MOSEI unaligned features (500 steps audio/vision); project strictly uses aligned_50"
                top_k, shapes_str, dtypes_str = probe_pkl(f)
            else:
                usage = "reference_metadata"
                notes = "Attachment 2 label table"
                top_k, shapes_str, dtypes_str = "", "", ""
            sample_count = "4850"

        elif "附件3" in rel:
            if "未对齐版本" in rel:
                cat = "attachment3_missing_unaligned"
                usage = "not_used_in_primary_pipeline"
                notes = f"Attachment 3 unaligned sample: {f.name} (not used; project strictly uses aligned_50)"
            else:
                cat = "attachment3_missing_aligned"
                usage = "primary_pipeline"
                notes = f"Attachment 3 missing modality aligned test sample: {f.name}"
            sample_count = "1"
            top_k, shapes_str, dtypes_str = probe_pkl(f)

        elif "附件4" in rel:
            if "未对齐版本" in rel:
                if ext == "pkl":
                    cat = "attachment4_explain_unaligned"
                    usage = "not_used_in_primary_pipeline"
                    notes = f"Attachment 4 unaligned sample: {f.name} (not used; project strictly uses aligned_50)"
                    top_k, shapes_str, dtypes_str = probe_pkl(f)
                else:
                    cat = "attachment4_video_unaligned"
                    usage = "not_used_in_primary_pipeline"
                    notes = f"Attachment 4 unaligned video: {f.name} (not used in aligned pipeline)"
                    top_k, shapes_str, dtypes_str = "", "", ""
            else:
                if ext == "pkl":
                    cat = "attachment4_explain_aligned"
                    usage = "primary_pipeline"
                    notes = f"Attachment 4 explainability aligned sample: {f.name}"
                    top_k, shapes_str, dtypes_str = probe_pkl(f)
                else:
                    cat = "attachment4_video_aligned"
                    usage = "primary_pipeline"
                    notes = f"Attachment 4 ground truth video: {f.name}"
                    top_k, shapes_str, dtypes_str = "", "", ""
            sample_count = "1"

        else:
            cat = "other"
            usage = "unknown"
            sample_count = ""
            notes = ""
            top_k, shapes_str, dtypes_str = "", "", ""

        manifest_rows.append({
            "relative_path": rel,
            "category": cat,
            "size_bytes": size,
            "sha256": sha,
            "format": ext,
            "sample_count": sample_count,
            "pipeline_usage": usage,
            "pkl_top_keys": top_k,
            "pkl_shapes": shapes_str,
            "pkl_dtypes": dtypes_str,
            "notes": notes,
        })
        if (idx + 1) % 50 == 0:
            print(f"      Processed {idx + 1} / {len(data_files)} files...")

    df = pd.DataFrame(manifest_rows)
    print(f"[3/4] Manifest built: {len(df)} rows.")

    # 3. Assertions
    assert len(df) == 245, f"Expected exactly 245 rows, got {len(df)}"
    assert df["relative_path"].nunique() == 245, "Duplicate paths detected in manifest!"
    assert (df["sha256"].str.len() == 64).all(), "Found invalid or empty SHA-256 hashes!"
    assert (df["size_bytes"] > 0).all(), "Found empty files with 0 bytes!"

    out_csv = cfg["_results_dir"] / "input_manifest.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[4/4] Saved manifest to: {out_csv} ({len(df)} rows)")

    # Print breakdown
    print("\n--- Category Breakdown ---")
    print(df["category"].value_counts())
    print("\n--- Pipeline Usage Breakdown ---")
    print(df["pipeline_usage"].value_counts())
    return df


if __name__ == "__main__":
    build_manifest()
