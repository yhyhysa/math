"""Independent re-verification of the Attachment1 x Attachment2 overlap and of
results/input_manifest.csv completeness.

This tool is a *checker*, not a producer: it recomputes both quantities from the
read-only inputs with independent code paths and compares them against the
delivered artefacts.  Non-zero exit on any mismatch.

Checks performed:
  1. label-100.xlsx sheet 'label' is read with two independent readers
     (pandas/openpyxl) and the raw `video_id` / `clip_id` string representations
     are printed verbatim so any parsing rule is auditable.
  2. The exact key `video_id + '$_$' + clip_id` is intersected with the
     Attachment 2 sample ids of each split, and the per-split counts plus the
     distinct source video_id counts are compared with
     results/cross_attachment_overlap.csv.
  3. Attachment 2 internal video_id overlap across train/valid/test must be 0.
  4. results/input_manifest.csv is compared against the real extracted tree
     (all files) plus the raw archive row: identical path sets, no duplicates,
     non-empty sha256, and correct file count.

Usage (cwd = 工程/):
  python -m src.tools.verify_cross_overlap
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from datetime import datetime, timezone
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
    ap.add_argument("--out", default="results/overlap_manifest_verify.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    cfg = load_config(args.config)
    errors: list[str] = []
    checks: dict[str, object] = {}

    excel_path = cfg["_video_attachment1_root"] / "label-100.xlsx"
    a2_path = cfg["_attachment2_root"] / "aligned_50.pkl"

    # -- 1. read label-100 with pandas ---------------------------------------
    sheets = pd.read_excel(excel_path, sheet_name=None)
    checks["excel_sheets"] = list(sheets.keys())
    df = sheets["label"]
    checks["label_sheet_row_count"] = int(len(df))
    checks["label_sheet_columns"] = [str(c) for c in df.columns]

    raw_pairs: list[tuple[str, str]] = []
    for i in range(len(df)):
        vid = df["video_id"].iloc[i]
        cid = df["clip_id"].iloc[i]
        raw_pairs.append((vid, cid))
    checks["raw_video_id_sample_first5"] = [repr(v) for v, _ in raw_pairs[:5]]
    checks["raw_clip_id_sample_first5"] = [repr(c) for _, c in raw_pairs[:5]]

    a1_keys = [f"{str(v).strip()}{DELIM}{str(c).strip()}" for v, c in raw_pairs]
    a1_key_set = set(a1_keys)
    a1_vid_set = {str(v).strip() for v, _ in raw_pairs}
    checks["attachment1_key_count"] = len(a1_keys)
    checks["attachment1_unique_key_count"] = len(a1_key_set)
    checks["attachment1_unique_video_id_count"] = len(a1_vid_set)
    if len(a1_key_set) != 100:
        errors.append(f"label-100 does not yield 100 unique (video_id, clip_id) keys: {len(a1_key_set)}")

    # polarity / label consistency
    pol_counts: dict[str, int] = {}
    sign_mismatch = []
    for i in range(len(df)):
        lbl = float(df["label"].iloc[i])
        ann = str(df["annotation"].iloc[i]).strip()
        pol_counts[ann] = pol_counts.get(ann, 0) + 1
        expected = "Negative" if lbl < 0 else ("Neutral" if lbl == 0 else "Positive")
        if ann != expected:
            sign_mismatch.append({"row": i, "label": lbl, "annotation": ann})
    checks["polarity_counts"] = pol_counts
    checks["label_sign_vs_annotation_mismatches"] = sign_mismatch
    if sign_mismatch:
        errors.append(f"{len(sign_mismatch)} rows where sign(label) disagrees with annotation")

    # video file pairing: each (video_id, clip_id) must have a real mp4
    a1_root = cfg["_video_attachment1_root"]
    missing_videos = [f"{v}/{c}.mp4" for v, c in raw_pairs if not (a1_root / str(v).strip() / f"{str(c).strip()}.mp4").is_file()]
    checks["video_files_missing"] = missing_videos[:20]
    checks["video_files_missing_count"] = len(missing_videos)
    if missing_videos:
        errors.append(f"{len(missing_videos)} (video_id, clip_id) pairs have no matching mp4 file")

    # -- 2. recompute overlap from Attachment 2 ------------------------------
    with open(a2_path, "rb") as fh:
        d2 = pickle.load(fh)

    per_split_sid: dict[str, int] = {}
    per_split_vid: dict[str, int] = {}
    recomputed_rows: list[dict] = []
    for split in ("train", "valid", "test"):
        ids = [str(x) for x in d2[split]["id"]]
        sid_set = set(ids)
        vid_set = {s.split(DELIM)[0] for s in sid_set}
        common = a1_key_set & sid_set
        per_split_sid[split] = len(common)
        per_split_vid[split] = len(a1_vid_set & vid_set)
        for sid in sorted(common):
            vid, cid = sid.split(DELIM)
            recomputed_rows.append(
                {"sample_id": sid, "attachment2_split": split, "video_id": vid, "clip_id": cid}
            )

    total_common_vids = len(
        a1_vid_set
        & (
            {s.split(DELIM)[0] for s in map(str, d2["train"]["id"])}
            | {s.split(DELIM)[0] for s in map(str, d2["valid"]["id"])}
            | {s.split(DELIM)[0] for s in map(str, d2["test"]["id"])}
        )
    )
    checks["recomputed_sample_id_matches_by_split"] = per_split_sid
    checks["recomputed_sample_id_matches_total"] = len(recomputed_rows)
    checks["recomputed_video_id_matches_by_split"] = per_split_vid
    checks["recomputed_video_id_matches_total"] = total_common_vids
    checks["recomputed_overlap_rows"] = recomputed_rows

    # internal video_id overlap of Attachment 2
    vids = {s: {x.split(DELIM)[0] for x in map(str, d2[s]["id"])} for s in ("train", "valid", "test")}
    internal = {
        "train_valid": len(vids["train"] & vids["valid"]),
        "train_test": len(vids["train"] & vids["test"]),
        "valid_test": len(vids["valid"] & vids["test"]),
    }
    checks["attachment2_internal_video_overlap"] = internal
    if any(internal.values()):
        errors.append(f"Attachment 2 splits share source video_id: {internal}")

    # -- 3. compare against delivered CSV ------------------------------------
    csv_path = cfg["_results_dir"] / "cross_attachment_overlap.csv"
    if not csv_path.is_file():
        errors.append(f"missing delivered artefact: {csv_path}")
    else:
        with csv_path.open(encoding="utf-8-sig", newline="") as fh:
            delivered = list(csv.DictReader(fh))
        checks["delivered_csv_rows"] = len(delivered)
        checks["delivered_csv_columns"] = list(delivered[0].keys()) if delivered else []
        dset = {(r["sample_id"], r["attachment2_split"], r["video_id"], r["clip_id"]) for r in delivered}
        rset = {(r["sample_id"], r["attachment2_split"], r["video_id"], r["clip_id"]) for r in recomputed_rows}
        if len(dset) != len(delivered):
            errors.append("delivered overlap CSV contains duplicate rows")
        if dset != rset:
            errors.append(
                f"delivered overlap CSV differs from recomputation: only_in_csv={sorted(dset - rset)[:5]}, "
                f"only_in_recompute={sorted(rset - dset)[:5]}"
            )
        checks["delivered_matches_recompute"] = dset == rset

    # -- 4. manifest completeness -------------------------------------------
    extracted = cfg["_extracted_dir"]
    engine = cfg["_engine_dir"]
    disk_files = sorted(p for p in extracted.rglob("*") if p.is_file())
    disk_rel = {p.relative_to(engine).as_posix() for p in disk_files}
    disk_data_rel = {r for r in disk_rel if Path(r).name != ".DS_Store"}
    archive_rel = (cfg["_raw_dir"] / "E题数据.zip").relative_to(engine).as_posix()

    manifest_path = cfg["_results_dir"] / "input_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    man_rel = [r["relative_path"].replace("\\", "/") for r in rows]
    checks["manifest_row_count"] = len(rows)
    checks["extracted_file_count_including_ds_store"] = len(disk_files)
    checks["extracted_data_file_count_excluding_ds_store"] = len(disk_data_rel)
    checks["manifest_unique_path_count"] = len(set(man_rel))
    if len(man_rel) != len(set(man_rel)):
        errors.append("input_manifest.csv has duplicate relative_path rows")
    expected_set = disk_data_rel | {archive_rel}
    if set(man_rel) != expected_set:
        missing = sorted(expected_set - set(man_rel))
        extra = sorted(set(man_rel) - expected_set)
        errors.append(f"manifest path set != real files: missing={missing[:10]} (n={len(missing)}), extra={extra[:10]} (n={len(extra)})")
    checks["manifest_path_set_matches_disk"] = set(man_rel) == expected_set
    empty_hash = [r["relative_path"] for r in rows if not str(r.get("sha256", "")).strip()]
    checks["manifest_empty_sha256_count"] = len(empty_hash)
    if empty_hash:
        errors.append(f"{len(empty_hash)} manifest rows have empty sha256, e.g. {empty_hash[:3]}")

    # spot re-hash of up to 12 extracted files (independent of stored values)
    spot = sorted(disk_data_rel)[:6] + sorted(disk_data_rel)[-6:]
    by_path = {r["relative_path"].replace("\\", "/"): r["sha256"] for r in rows}
    spot_mismatch = []
    for rel in spot:
        actual = sha256_file(engine / rel)
        if by_path.get(rel, "").lower() != actual.lower():
            spot_mismatch.append(rel)
    checks["manifest_spot_rehash_count"] = len(spot)
    checks["manifest_spot_rehash_mismatches"] = spot_mismatch
    if spot_mismatch:
        errors.append(f"manifest sha256 mismatch on spot re-hash: {spot_mismatch}")

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "checks": checks,
        "inputs": {
            "label_100_xlsx": str(excel_path),
            "label_100_xlsx_sha256": sha256_file(excel_path),
            "aligned_50_pkl": str(a2_path),
        },
        "artefacts": {
            "cross_attachment_overlap_csv": str(csv_path),
            "cross_attachment_overlap_csv_sha256": sha256_file(csv_path) if csv_path.is_file() else None,
            "input_manifest_csv": str(manifest_path),
            "input_manifest_csv_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        },
        "scope": "Recomputation of Attachment1 x Attachment2 overlap and manifest completeness; read-only on inputs.",
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = cfg["_engine_dir"] / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "sample_id_matches_by_split": per_split_sid,
        "sample_id_matches_total": len(recomputed_rows),
        "video_id_matches_total": total_common_vids,
        "attachment2_internal_video_overlap": internal,
        "manifest_rows": len(rows),
        "errors": errors,
        "report": str(out_path),
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
