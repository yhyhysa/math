"""Compute cross-attachment overlap between Attachment 1 and Attachment 2.

Reads:
  - Attachment 1: MOSEI数据集部分原始视频-100条/label-100.xlsx (sheets 'label' and '修改说明')
  - Attachment 2: aligned_50.pkl

Outputs:
  - results/cross_attachment_overlap.csv
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pickle

from src.config import load_config


def compute_cross_overlap() -> dict:
    cfg = load_config("configs/base.json")
    excel_path = cfg["_video_attachment1_root"] / "label-100.xlsx"
    a2_path = cfg["_attachment2_root"] / "aligned_50.pkl"

    # 1. Read label-100.xlsx sheet 'label'
    df_label = pd.read_excel(excel_path, sheet_name="label")
    print(f"Read {len(df_label)} rows from sheet 'label'.")

    # Verify counts
    counts = df_label["annotation"].value_counts().to_dict()
    print(f"Polarity counts in label-100: {counts}")
    assert counts == {"Positive": 57, "Neutral": 25, "Negative": 18}, f"Unexpected counts: {counts}"

    # Verify label < 0, == 0, > 0 matches annotation
    for idx, row in df_label.iterrows():
        lbl = float(row["label"])
        ann = str(row["annotation"]).strip()
        if lbl < 0:
            assert ann == "Negative", f"Row {idx}: lbl={lbl} but ann={ann}"
        elif lbl == 0:
            assert ann == "Neutral", f"Row {idx}: lbl={lbl} but ann={ann}"
        else:
            assert ann == "Positive", f"Row {idx}: lbl={lbl} but ann={ann}"
    print("Label values strictly match annotation categories for all 100 rows.")

    # 2. Read sheet '修改说明'
    df_fix = pd.read_excel(excel_path, sheet_name=1)  # 2nd sheet: 修改说明
    fix_summary = {}
    for idx, row in df_fix.iterrows():
        k = str(row.iloc[0]).strip()
        v = str(row.iloc[1]).strip()
        fix_summary[k] = v
    print(f"Read sheet '修改说明': {fix_summary}")

    # 3. Read Attachment 2
    with open(a2_path, "rb") as f:
        d2 = pickle.load(f)

    # 4. Construct exact string sample_id: video_id + '$_$' + clip_id
    delim = "$_$"
    a1_entries = []
    for idx, row in df_label.iterrows():
        vid = str(row["video_id"]).strip()
        cid = str(row["clip_id"]).strip()
        sid = vid + delim + cid
        a1_entries.append({"sample_id": sid, "video_id": vid, "clip_id": cid, "label": float(row["label"]), "annotation": str(row["annotation"])})

    a1_sid_set = {e["sample_id"] for e in a1_entries}
    a1_vid_set = {e["video_id"] for e in a1_entries}
    print(f"Attachment 1 unique sample_ids: {len(a1_sid_set)}, unique video_ids: {len(a1_vid_set)}")

    # 5. Intersect with Attachment 2 train/valid/test
    matches = []
    split_counts = {}
    split_vid_counts = {}

    for split in ["train", "valid", "test"]:
        s_ids = set(d2[split]["id"])
        s_vids = {sid.split(delim)[0] for sid in s_ids}

        common_sids = a1_sid_set & s_ids
        common_vids = a1_vid_set & s_vids

        split_counts[split] = len(common_sids)
        split_vid_counts[split] = len(common_vids)
        print(f"Attachment 2 split '{split}': {len(common_sids)} matching sample_ids, {len(common_vids)} matching video_ids")

        for msid in sorted(common_sids):
            vid, cid = msid.split(delim)
            matches.append({
                "sample_id": msid,
                "attachment2_split": split,
                "video_id": vid,
                "clip_id": cid,
            })

    # Total unique video_ids in Attachment 2
    all_a2_vids = (
        {sid.split(delim)[0] for sid in d2["train"]["id"]}
        | {sid.split(delim)[0] for sid in d2["valid"]["id"]}
        | {sid.split(delim)[0] for sid in d2["test"]["id"]}
    )
    total_common_vids = len(a1_vid_set & all_a2_vids)
    print(f"Total cross-attachment sample_id matches: {len(matches)} (train: {split_counts['train']}, valid: {split_counts['valid']}, test: {split_counts['test']})")
    print(f"Total cross-attachment video_id matches: {total_common_vids}")

    # Check Attachment 2 internal video_id overlap across train, valid, test
    train_vids = {sid.split(delim)[0] for sid in d2["train"]["id"]}
    valid_vids = {sid.split(delim)[0] for sid in d2["valid"]["id"]}
    test_vids = {sid.split(delim)[0] for sid in d2["test"]["id"]}
    int_tv = len(train_vids & valid_vids)
    int_tt = len(train_vids & test_vids)
    int_vt = len(valid_vids & test_vids)
    print(f"Attachment 2 internal video_id overlap: train-valid={int_tv}, train-test={int_tt}, valid-test={int_vt}")

    # 6. Save results/cross_attachment_overlap.csv
    df_matches = pd.DataFrame(matches)
    out_csv = Path("results/cross_attachment_overlap.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_matches.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df_matches)} matching rows to {out_csv}")

    result = {
        "attachment1_total_samples": len(df_label),
        "polarity_distribution": counts,
        "fix_explanation_sheet": fix_summary,
        "matched_sample_ids_count": len(matches),
        "matched_sample_ids_by_split": split_counts,
        "matched_video_ids_count": total_common_vids,
        "matched_video_ids_by_split": split_vid_counts,
        "attachment2_internal_video_overlap": {
            "train_valid": int_tv,
            "train_test": int_tt,
            "valid_test": int_vt,
        },
        "speaker_independence_status": "未知（数据集中无独立说话人ID字段，仅能保证源video_id跨split无重叠）",
        "matches": matches,
    }
    return result


if __name__ == "__main__":
    compute_cross_overlap()
