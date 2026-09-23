r"""Build a separate Q1 candidate with cropped-face recovery; never overwrite q1.

Run from 工程 with the full Q1 environment:
  ..\训练demo\.venv\Scripts\python.exe -m src.q1_visual_refine
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

import av
import mediapipe as mp
import numpy as np

from .q1_extract import face_vector, intervals_from_pts, make_windows, pool_interval, sha256

SOURCE = ROOT / "data/processed/q1"
TARGET = ROOT / "data/processed/q1_v2"
RESULTS = ROOT / "results/q1_v2"
FACE_MODEL = ROOT / "models/face_landmarker.task"
FPS = 10.0
FACE_THRESHOLD = 0.5
CROPS = (0.7, 0.5)


def extract_crop(path: Path, fps: float, crop_ratio: float):
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="models/face_landmarker.task"),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True,
    )
    times, vectors, valid, switches = [], [], [], []
    first_pts = last_selected = tracked_face = None
    last_ms = -1
    decoded = 0
    blend_names = None
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        with av.open(str(path)) as container:
            for frame in container.decode(container.streams.video[0]):
                decoded += 1
                if frame.time is None:
                    continue
                pts = float(frame.time)
                if first_pts is None:
                    first_pts = pts
                if last_selected is not None and pts - last_selected < 1 / fps - 1e-5:
                    continue
                last_selected = pts
                last_ms = max(last_ms + 1, int(round((pts - first_pts) * 1000)))
                rgb = frame.to_ndarray(format="rgb24")
                h, w = rgb.shape[:2]
                ch, cw = round(h * crop_ratio), round(w * crop_ratio)
                top, left = (h - ch) // 2, (w - cw) // 2
                rgb = np.ascontiguousarray(rgb[top:top + ch, left:left + cw])
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_ms)
                vector, blend_names, face_position = face_vector(result, blend_names)
                if vector is not None:
                    if tracked_face is not None:
                        center, scale = face_position
                        old_center, old_scale = tracked_face
                        jump = np.linalg.norm(center - old_center) > 0.6 * old_scale
                        size_jump = max(scale / old_scale, old_scale / scale) > 1.6
                        if jump or size_jump:
                            switches.append(round(pts, 4))
                            vector = None
                    tracked_face = face_position
                times.append(pts)
                vectors.append(vector if vector is not None else np.zeros(61, dtype=np.float32))
                valid.append(vector is not None)
    if not times:
        raise ValueError("No sampled frames")
    return (np.stack(vectors), np.asarray(times), np.asarray(valid, dtype=bool),
            blend_names, decoded, switches)


def process_one(meta_path: Path, version_hash: str, face_model_sha: str) -> dict:
    started = time.time()
    sid = meta_path.stem
    old = json.loads(meta_path.read_text(encoding="utf-8"))
    if float(old["fps_limit"]) != FPS:
        raise ValueError("Baseline FPS differs from Q1 v2 sampling rate")
    video = Path(old["video_path"])
    if not video.is_file() or sha256(video) != old["video_sha256"]:
        raise ValueError("Original video missing or SHA differs from baseline")
    with np.load(SOURCE / f"{sid}.npz", allow_pickle=False) as src:
        arrays = {key: src[key] for key in src.files}
    if sha256(SOURCE / f"{sid}.npz") != old["feature_sha256"]:
        raise ValueError("Baseline feature SHA mismatch")
    original_frames = int(old["video_face_frames"])
    count = int(old["video_processed_frames"])
    if count != len(arrays["video_pts_s"]) or original_frames != int(arrays["video_frame_valid"].sum()):
        raise ValueError("Baseline frame counters disagree")

    options = []
    chosen = None
    if original_frames / count < FACE_THRESHOLD:
        for ratio in CROPS:
            vectors, pts, valid, blend_names, decoded, switches = extract_crop(video, FPS, ratio)
            if len(pts) != count or not np.allclose(pts, arrays["video_pts_s"], atol=1e-4, rtol=0):
                raise ValueError(f"Crop {ratio} changed sampled frame times")
            detected = int(valid.sum())
            options.append({"crop_ratio": ratio, "face_frames": detected,
                            "possible_face_switches": len(switches)})
            if detected > original_frames and (chosen is None or detected > chosen[2].sum()):
                chosen = (ratio, vectors, valid, blend_names, decoded, switches)

    mode = "original"
    new = dict(old)
    if chosen is not None:
        ratio, vectors, valid, blend_names, decoded, switches = chosen
        mode = f"center_crop_{ratio}"
        starts, ends = intervals_from_pts(arrays["video_pts_s"], float(old["video_duration_s"]))
        vision = np.zeros_like(arrays["vision"])
        vision_valid = np.zeros_like(arrays["vision_valid"])
        vision_coverage = np.zeros_like(arrays["vision_coverage"])
        reasons = arrays["token_failure_reason"].copy()
        for ti, wi in enumerate(arrays["token_word"]):
            if wi < 0 or not old["words"][int(wi)]["valid"]:
                continue
            a, b = float(arrays["token_start_s"][ti]), float(arrays["token_end_s"][ti])
            vision[ti], vision_valid[ti], vision_coverage[ti] = pool_interval(
                vectors, starts, ends, valid, a, b)
            if arrays["audio_valid"][ti]:
                reasons[ti] = "" if vision_valid[ti] else "no_valid_face_frame"
        arrays.update({"vision": vision, "vision_valid": vision_valid,
                       "vision_coverage": vision_coverage, "video_frame_valid": valid,
                       "token_failure_reason": reasons})
        windows = make_windows(arrays, len(vision))
        for key in ("vision_windows", "vision_valid_windows", "token_failure_reason_windows"):
            arrays[key] = windows[key]
        new["visual_feature_names"] = (
            blend_names if blend_names is not None else [f"blendshape_{i}" for i in range(52)])
        new["visual_feature_names"] += ["head_x_rad", "head_y_rad", "head_z_rad",
                                        "eye_open_left", "eye_open_right", "mouth_open",
                                        "mouth_width", "brow_eye_left", "brow_eye_right"]
        new["possible_face_switch_times_s"] = switches
        new["video_decoded_frames"] = decoded
        new["video_face_frames"] = int(valid.sum())
        new["quality_status"] = "REVIEW"  # Crop requires speaker/face identity review.
        new["token_failure_reasons"] = reasons.tolist()
    new["visual_refinement"] = {
        "mode": mode, "baseline_face_frames": original_frames,
        "sampled_frames": count, "candidates": options,
        "manual_face_identity_review_required": mode != "original",
        "text_audio_from_baseline_sha256": old["feature_sha256"],
        "baseline_pipeline_sha256": old["pipeline_sha256"],
        "face_model_sha256": face_model_sha,
    }
    new["pipeline_sha256"] = version_hash
    new["baseline_elapsed_s"] = old.get("elapsed_s")
    new["elapsed_s"] = round(time.time() - started, 2)
    TARGET.mkdir(parents=True, exist_ok=True)
    out_npz = TARGET / f"{sid}.npz"
    np.savez_compressed(out_npz, **arrays)
    new["feature_file"] = str(out_npz)
    new["feature_sha256"] = sha256(out_npz)
    (TARGET / f"{sid}.json").write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"sample_id": sid, "status": new["quality_status"], "token_count": new["token_count"],
            "window_count": new["window_count"], "unresolved_words": new["alignment"]["unresolved_words"],
            "audio_valid_tokens": int(arrays["audio_valid"].sum()),
            "vision_valid_tokens": int(arrays["vision_valid"].sum()),
            "video_decoded_frames": new["video_decoded_frames"], "video_processed_frames": count,
            "face_frames": new["video_face_frames"], "elapsed_s": new["elapsed_s"], "error": "",
            "visual_mode": mode, "baseline_face_frames": original_frames}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Only first N baseline samples")
    parser.add_argument("--ids", default="", help="Comma-separated exact sample IDs")
    args = parser.parse_args()
    paths = sorted(SOURCE.glob("*.json"))
    if args.ids:
        wanted = set(args.ids.split(","))
        paths = [p for p in paths if p.stem in wanted]
        if len(paths) != len(wanted):
            raise ValueError("Some IDs do not exist in baseline")
    if args.limit:
        paths = paths[:args.limit]
    face_model_sha = sha256(FACE_MODEL)
    version_hash = hashlib.sha256(
        (sha256(Path(__file__)) + sha256(ROOT / "src/q1_extract.py")
         + face_model_sha + str(FPS) + str(CROPS)).encode()
    ).hexdigest()
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = []
    for i, path in enumerate(paths, 1):
        try:
            record = process_one(path, version_hash, face_model_sha)
        except Exception as exc:
            record = {"sample_id": path.stem, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        records.append(record)
        print(f"[{i}/{len(paths)}] {record['status']} {path.stem} "
              f"{record.get('visual_mode', '')} {record.get('face_frames', '')} "
              f"{record.get('error', '')}", flush=True)
    with (RESULTS / "q1_feature_inventory.csv").open("w", encoding="utf-8-sig", newline="") as f:
        cols = ("sample_id", "status", "token_count", "window_count", "unresolved_words",
                "audio_valid_tokens", "vision_valid_tokens", "video_decoded_frames",
                "video_processed_frames", "face_frames", "elapsed_s", "error",
                "visual_mode", "baseline_face_frames")
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(records)
    failures = [r for r in records if r["status"] == "FAIL"]
    print(json.dumps({"processed": len(records), "failures": len(failures),
                      "output": str(TARGET)}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
