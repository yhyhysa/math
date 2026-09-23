"""Independent validation entry for data reader (load_pickle) against ALL real pickle files.

Verifies:
  1. Attachment 2 (aligned_50.pkl): 3 splits -> 3395 (train), 728 (valid), 727 (test) samples.
     Shapes: text(50,768), text_bert(3,50), audio(50,74), vision(50,35).
  2. Attachment 3 (30 files): 30 files -> 30 samples.
     Shapes: audio(50,74), vision(50,35), text_bert(3,50). No fabricated text or ID.
  3. Attachment 4 (20 files): 20 files -> 20 samples.
     Shapes: text(50,768), text_bert(3,50), audio(50,74), vision(50,35), raw_text, id.

Writes report to: results/reader_check.json
Exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.config import load_config
from src.data import load_pickle


def check_attachment2(a2_path: Path) -> dict:
    payload = load_pickle(a2_path, strict=True)
    report = {
        "file": str(a2_path.name),
        "packaging": payload.packaging,
        "splits": {},
        "errors": [],
    }

    expected_counts = {"train": 3395, "valid": 728, "test": 727}
    for split_name, expected_n in expected_counts.items():
        if split_name not in payload.splits:
            report["errors"].append(f"Missing split '{split_name}'")
            continue
        samples = payload.splits[split_name]
        actual_n = len(samples)
        ids = [s.sample_id for s in samples]
        unique_ids = len(set(ids))

        shape_errors = []
        for idx, s in enumerate(samples):
            # check shapes
            t = s.get_array("text")
            tb = s.get_array("text_bert")
            a = s.get_array("audio")
            v = s.get_array("vision")
            if t is None or t.shape != (50, 768):
                shape_errors.append(f"Sample {idx} invalid text shape: {getattr(t, 'shape', None)}")
            if tb is None or tb.shape != (3, 50):
                shape_errors.append(f"Sample {idx} invalid text_bert shape: {getattr(tb, 'shape', None)}")
            if a is None or a.shape != (50, 74):
                shape_errors.append(f"Sample {idx} invalid audio shape: {getattr(a, 'shape', None)}")
            if v is None or v.shape != (50, 35):
                shape_errors.append(f"Sample {idx} invalid vision shape: {getattr(v, 'shape', None)}")
            if s.classification_label is None or s.classification_label not in (0.0, 1.0, 2.0):
                shape_errors.append(f"Sample {idx} invalid classification label: {s.classification_label}")
            if s.regression_label is None or s.regression_label < -3.0 or s.regression_label > 3.0:
                shape_errors.append(f"Sample {idx} invalid regression label: {s.regression_label}")
            if shape_errors and len(shape_errors) >= 5:
                break

        split_passed = (actual_n == expected_n) and (unique_ids == expected_n) and not shape_errors
        report["splits"][split_name] = {
            "expected_count": expected_n,
            "actual_count": actual_n,
            "unique_ids": unique_ids,
            "shape_errors": shape_errors,
            "passed": split_passed,
        }
        if not split_passed:
            report["errors"].append(f"Split '{split_name}' validation failed: n={actual_n}/{expected_n}, uniq_ids={unique_ids}/{expected_n}, shape_errs={len(shape_errors)}")

    report["passed"] = (len(report["errors"]) == 0)
    return report


def check_attachment3(a3_dir: Path) -> dict:
    files = sorted(a3_dir.glob("*.pkl"))
    report = {
        "directory": str(a3_dir.name),
        "total_files": len(files),
        "samples_checked": 0,
        "unique_sample_ids": 0,
        "file_results": [],
        "errors": [],
    }

    if len(files) != 30:
        report["errors"].append(f"Expected 30 files in Attachment 3 aligned, found {len(files)}")

    all_ids = []
    for f in files:
        try:
            payload = load_pickle(f, strict=True)
            if "test" not in payload.splits:
                report["errors"].append(f"{f.name}: missing 'test' split")
                continue
            samples = payload.splits["test"]
            if len(samples) != 1:
                report["errors"].append(f"{f.name}: expected exactly 1 sample, got {len(samples)}")
                continue

            s = samples[0]
            all_ids.append(s.sample_id)
            errs = []
            if s.sample_id != f.stem:
                errs.append(f"Expected sample_id '{f.stem}', got '{s.sample_id}'")
            if s.has("text"):
                errs.append("Manufactured 'text' array unexpectedly present")
            if s.raw_text is not None:
                errs.append("Manufactured 'raw_text' unexpectedly present")

            tb = s.get_array("text_bert")
            a = s.get_array("audio")
            v = s.get_array("vision")
            if tb is None or tb.shape != (3, 50):
                errs.append(f"Invalid text_bert shape: {getattr(tb, 'shape', None)}")
            if a is None or a.shape != (50, 74):
                errs.append(f"Invalid audio shape: {getattr(a, 'shape', None)}")
            if v is None or v.shape != (50, 35):
                errs.append(f"Invalid vision shape: {getattr(v, 'shape', None)}")

            passed = len(errs) == 0
            report["file_results"].append({
                "file": f.name,
                "sample_id": s.sample_id,
                "passed": passed,
                "errors": errs,
            })
            if not passed:
                report["errors"].extend([f"{f.name}: {e}" for e in errs])
        except Exception as exc:
            report["errors"].append(f"{f.name}: load_pickle exception: {exc}")

    report["samples_checked"] = len(all_ids)
    report["unique_sample_ids"] = len(set(all_ids))
    if len(all_ids) != 30 or len(set(all_ids)) != 30:
        report["errors"].append(f"ID uniqueness check failed: count={len(all_ids)}, unique={len(set(all_ids))}")

    report["passed"] = (len(report["errors"]) == 0)
    return report


def check_attachment4(a4_dir: Path) -> dict:
    files = sorted(a4_dir.glob("*.pkl"))
    report = {
        "directory": str(a4_dir.name),
        "total_files": len(files),
        "samples_checked": 0,
        "unique_sample_ids": 0,
        "file_results": [],
        "errors": [],
    }

    if len(files) != 20:
        report["errors"].append(f"Expected 20 files in Attachment 4 aligned, found {len(files)}")

    all_ids = []
    for f in files:
        try:
            payload = load_pickle(f, strict=True)
            samples = payload.splits.get("single", [])
            if len(samples) != 1:
                report["errors"].append(f"{f.name}: expected 1 sample in 'single', got {len(samples)}")
                continue

            s = samples[0]
            all_ids.append(s.sample_id)
            errs = []

            t = s.get_array("text")
            tb = s.get_array("text_bert")
            a = s.get_array("audio")
            v = s.get_array("vision")

            if t is None or t.shape != (50, 768):
                errs.append(f"Invalid text shape: {getattr(t, 'shape', None)}")
            if tb is None or tb.shape != (3, 50):
                errs.append(f"Invalid text_bert shape: {getattr(tb, 'shape', None)}")
            if a is None or a.shape != (50, 74):
                errs.append(f"Invalid audio shape: {getattr(a, 'shape', None)}")
            if v is None or v.shape != (50, 35):
                errs.append(f"Invalid vision shape: {getattr(v, 'shape', None)}")
            if not s.raw_text:
                errs.append("Missing raw_text string")

            passed = len(errs) == 0
            report["file_results"].append({
                "file": f.name,
                "sample_id": s.sample_id,
                "passed": passed,
                "errors": errs,
            })
            if not passed:
                report["errors"].extend([f"{f.name}: {e}" for e in errs])
        except Exception as exc:
            report["errors"].append(f"{f.name}: load_pickle exception: {exc}")

    report["samples_checked"] = len(all_ids)
    report["unique_sample_ids"] = len(set(all_ids))
    if len(all_ids) != 20 or len(set(all_ids)) != 20:
        report["errors"].append(f"ID uniqueness check failed: count={len(all_ids)}, unique={len(set(all_ids))}")

    report["passed"] = (len(report["errors"]) == 0)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--out", default="results/reader_check.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    cfg = load_config(args.config)

    print("==================================================")
    print("      Unified Pickle Reader Verification Check    ")
    print("==================================================")

    # 1. Attachment 2
    a2_path = cfg["_attachment2_root"] / "aligned_50.pkl"
    print(f"[1/3] Checking Attachment 2: {a2_path} ...")
    a2_rep = check_attachment2(a2_path)
    print(f"      Passed: {a2_rep['passed']}, Errors: {len(a2_rep['errors'])}")

    # 2. Attachment 3
    a3_dir = cfg["_attachment3_root"] / "对齐版本"
    print(f"[2/3] Checking Attachment 3 (30 files): {a3_dir} ...")
    a3_rep = check_attachment3(a3_dir)
    print(f"      Passed: {a3_rep['passed']}, Errors: {len(a3_rep['errors'])}")

    # 3. Attachment 4
    a4_dir = cfg["_attachment4_root"] / "对齐版本"
    print(f"[3/3] Checking Attachment 4 (20 files): {a4_dir} ...")
    a4_rep = check_attachment4(a4_dir)
    print(f"      Passed: {a4_rep['passed']}, Errors: {len(a4_rep['errors'])}")

    all_passed = a2_rep["passed"] and a3_rep["passed"] and a4_rep["passed"]

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "all_passed": all_passed,
        "attachment2": a2_rep,
        "attachment3": a3_rep,
        "attachment4": a4_rep,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary written to: {out_path}")
    print("==================================================")
    print(f"OVERALL RESULT: {'PASS (code 0)' if all_passed else 'FAIL (code 1)'}")
    print("==================================================")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
