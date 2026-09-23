"""Compare original Q1 and cropped-face candidate without changing either."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/processed/q1"
NEW = ROOT / "data/processed/q1_v2"
OUT = ROOT / "results/q1_v2"
IMMUTABLE_KEYS = ("text", "audio", "text_bert", "text_valid", "audio_valid",
                  "audio_imputed", "vision_imputed", "token_word", "token_start_s",
                  "token_end_s", "audio_coverage", "video_pts_s", "audio_frame_start_s",
                  "audio_frame_end_s", "text_windows", "audio_windows", "text_bert_windows",
                  "text_valid_windows", "audio_valid_windows", "audio_imputed_windows",
                  "vision_imputed_windows", "token_word_windows")


def main() -> int:
    old_paths = sorted(OLD.glob("*.json"))
    new_paths = sorted(NEW.glob("*.json"))
    if len(old_paths) != 100 or {p.stem for p in old_paths} != {p.stem for p in new_paths}:
        raise ValueError("Both versions must contain the same 100 samples")
    rows, errors = [], []
    for path in old_paths:
        sid = path.stem
        a = json.loads(path.read_text(encoding="utf-8"))
        b = json.loads((NEW / path.name).read_text(encoding="utf-8"))
        if a["video_sha256"] != b["video_sha256"] or a["words"] != b["words"]:
            errors.append(f"{sid}: video or word alignment changed")
        with np.load(OLD / f"{sid}.npz", allow_pickle=False) as x, np.load(
                NEW / f"{sid}.npz", allow_pickle=False) as y:
            for key in IMMUTABLE_KEYS:
                if not np.array_equal(x[key], y[key], equal_nan=True):
                    errors.append(f"{sid}: unchanged modality differs at {key}")
            old_mask, new_mask = x["video_frame_valid"], y["video_frame_valid"]
            gained = int((~old_mask & new_mask).sum())
            lost = int((old_mask & ~new_mask).sum())
            old_tokens, new_tokens = int(x["vision_valid"].sum()), int(y["vision_valid"].sum())
            if old_tokens > new_tokens or int(new_mask.sum()) < int(old_mask.sum()):
                errors.append(f"{sid}: coverage declined")
        mode = b["visual_refinement"]["mode"]
        rows.append({"sample_id": sid, "visual_mode": mode,
                     "old_face_frames": int(a["video_face_frames"]),
                     "new_face_frames": int(b["video_face_frames"]),
                     "sampled_frames": int(b["video_processed_frames"]),
                     "gained_frames": gained, "lost_frames": lost,
                     "old_vision_valid_tokens": old_tokens,
                     "new_vision_valid_tokens": new_tokens,
                     "unresolved_words": int(b["alignment"]["unresolved_words"]),
                     "manual_face_review": mode != "original",
                     "video_path": b["video_path"]})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "q1_version_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    priority = [r for r in rows if r["manual_face_review"] or r["new_face_frames"] == 0]
    priority.sort(key=lambda r: (0 if r["old_face_frames"] == 0 and r["manual_face_review"] else
                                 1 if r["manual_face_review"] else 2,
                                 r["new_face_frames"] / r["sampled_frames"], r["sample_id"]))
    with (OUT / "q1_face_review_priority.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("priority",) + tuple(rows[0]))
        writer.writeheader()
        writer.writerows([{"priority": i, **r} for i, r in enumerate(priority, 1)])
    summary = {
        "status": "PASS" if not errors else "FAIL", "samples": len(rows),
        "cropped_clips": sum(r["manual_face_review"] for r in rows),
        "zero_face_clips_before": sum(r["old_face_frames"] == 0 for r in rows),
        "zero_face_clips_after": sum(r["new_face_frames"] == 0 for r in rows),
        "face_frames_before": sum(r["old_face_frames"] for r in rows),
        "face_frames_after": sum(r["new_face_frames"] for r in rows),
        "face_frames_gained": sum(r["gained_frames"] for r in rows),
        "face_frames_lost": sum(r["lost_frames"] for r in rows),
        "vision_valid_tokens_before": sum(r["old_vision_valid_tokens"] for r in rows),
        "vision_valid_tokens_after": sum(r["new_vision_valid_tokens"] for r in rows),
        "priority_review_clips": len(priority), "errors": errors,
    }
    (OUT / "q1_version_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
