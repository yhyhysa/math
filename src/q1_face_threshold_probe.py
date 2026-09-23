r"""Compare MediaPipe face thresholds on selected original videos without rewriting Q1 features.

Run from 工程:
  ..\训练demo\.venv\Scripts\python.exe -m src.q1_face_threshold_probe
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

import av
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "q1-face-probe"
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

DATA = ROOT / "data/processed/q1"
OUT = ROOT / "results/q1_face_threshold_probe"
SETTINGS = {
    "original_0.5": (.5, .5, .5, 1.0),
    "relaxed_0.3": (.3, .3, .5, 1.0),
    "relaxed_0.2": (.2, .2, .3, 1.0),
    "center_crop_0.7": (.5, .5, .5, .7),
    "center_crop_0.5": (.5, .5, .5, .5),
}
DEFAULT_IDS = (
    "-571d8cVauQ$_$0", "-HwX2H8Z4hY$_$9", "-iRBcNs9oI8$_$3",
    "-MeTTeMJBNc$_$0", "-NFrJFQijFE$_$1", "-UUCSKoHeMA$_$0",
    "-egA8-b7-3M$_$26",  # face-positive control
)


def run_one(meta: dict, name: str, setting: tuple[float, float, float, float],
            fps: float, contact_sheets: bool) -> dict:
    detect_conf, presence_conf, track_conf, crop_ratio = setting
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="models/face_landmarker.task"),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1, min_face_detection_confidence=detect_conf,
        min_face_presence_confidence=presence_conf, min_tracking_confidence=track_conf,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True,
    )
    selected = detected = 0
    first_pts = last_selected = None
    last_ms = -1
    successful = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as model:
        with av.open(meta["video_path"]) as container:
            for frame in container.decode(container.streams.video[0]):
                if frame.time is None:
                    continue
                t = float(frame.time)
                if first_pts is None:
                    first_pts = t
                if last_selected is not None and t - last_selected < 1 / fps - 1e-5:
                    continue
                last_selected = t
                selected += 1
                timestamp_ms = max(last_ms + 1, int(round((t - first_pts) * 1000)))
                last_ms = timestamp_ms
                rgb = frame.to_ndarray(format="rgb24")
                if crop_ratio < 1:
                    height, width = rgb.shape[:2]
                    crop_h, crop_w = round(height * crop_ratio), round(width * crop_ratio)
                    top, left = (height - crop_h) // 2, (width - crop_w) // 2
                    rgb = np.ascontiguousarray(rgb[top:top + crop_h, left:left + crop_w])
                result = model.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                valid = bool(result.face_landmarks and result.face_blendshapes
                             and result.facial_transformation_matrixes)
                if valid:
                    detected += 1
                    if contact_sheets and name != "original_0.5":
                        points = np.array([(p.x, p.y) for p in result.face_landmarks[0]])
                        successful.append((t, rgb, points.min(axis=0), points.max(axis=0)))
    if successful and name != "original_0.5" and contact_sheets:
        positions = np.unique(np.linspace(0, len(successful) - 1, min(4, len(successful))).astype(int))
        fig, axes = plt.subplots(1, len(positions), figsize=(3.2 * len(positions), 2.8), constrained_layout=True)
        for ax, i in zip(np.atleast_1d(axes), positions):
            t, rgb, lo, hi = successful[i]
            ax.imshow(rgb)
            h, w = rgb.shape[:2]
            ax.add_patch(plt.Rectangle((lo[0] * w, lo[1] * h), (hi[0] - lo[0]) * w,
                                       (hi[1] - lo[1]) * h, fill=False, edgecolor="lime", linewidth=1.5))
            ax.set_title(f"{t:.2f} s")
            ax.axis("off")
        fig.suptitle(f"Detected faces for {meta['sample_id'].replace('$', '·')} · {name}")
        fig.savefig(OUT / f"{meta['sample_id']}_{name}.png", dpi=160, facecolor="white")
        plt.close(fig)
    return {"sample_id": meta["sample_id"], "setting": name,
            "detection_threshold": detect_conf, "presence_threshold": presence_conf,
            "tracking_threshold": track_conf, "fps": fps, "sampled_frames": selected,
            "center_crop_ratio": crop_ratio,
            "detected_frames": detected, "detected_rate": round(detected / selected, 4)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default=",".join(DEFAULT_IDS), help="Comma-separated exact sample IDs")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--settings", default=",".join(SETTINGS),
                        help="Comma-separated setting names; defaults to all")
    parser.add_argument("--output", default="threshold_probe.csv", help="CSV filename inside results directory")
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    names = args.settings.split(",")
    if not names or any(name not in SETTINGS for name in names):
        raise ValueError(f"--settings must use names from {list(SETTINGS)}")
    if Path(args.output).name != args.output or not args.output.endswith(".csv"):
        raise ValueError("--output must be a CSV filename")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in args.ids.split(","):
        meta = json.loads((DATA / f"{sid}.json").read_text(encoding="utf-8"))
        for name in names:
            row = run_one(meta, name, SETTINGS[name], args.fps, not args.no_contact_sheets)
            rows.append(row)
            print(f"{sid} {name}: {row['detected_frames']}/{row['sampled_frames']}", flush=True)
    with (OUT / args.output).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"samples": len(set(r["sample_id"] for r in rows)),
                      "settings": names, "rows": len(rows),
                      "csv": str(OUT / args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
