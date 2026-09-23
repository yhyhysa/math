r"""Visual audit card: words, audio energy, stored face mask and original frames.

Run from 工程 (the original MP4 must be present):
  ..\训练demo\.venv\Scripts\python.exe -m src.q1_review_figure --sample-id='-egA8-b7-3M$_$26'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

import av
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "q1-review-figure"
import matplotlib.pyplot as plt
import numpy as np


def read_audio(path: Path) -> tuple[np.ndarray, float]:
    blocks = []
    start = None
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(stream):
            if start is None and frame.time is not None:
                start = float(frame.time)
            blocks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(frame))
        blocks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(None))
    if not blocks:
        raise ValueError("No audio decoded")
    return np.concatenate(blocks).astype(np.float32) / 32768.0, float(start or 0)


def choose_frames(path: Path, target_times: list[float]) -> list[tuple[float, np.ndarray]]:
    chosen = [None] * len(target_times)
    distances = np.full(len(target_times), np.inf)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            t = float(frame.time)
            for i, target in enumerate(target_times):
                distance = abs(t - target)
                if distance < distances[i]:
                    distances[i] = distance
                    chosen[i] = (t, frame.to_ndarray(format="rgb24"))
    if any(item is None for item in chosen):
        raise ValueError("No video frames decoded")
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default="-egA8-b7-3M$_$26", help="Exact Q1 sample ID")
    parser.add_argument("--times", default="", help="Four comma-separated original-video seconds")
    parser.add_argument("--variant", choices=("q1", "q1_v2"), default="q1")
    args = parser.parse_args()
    path = ROOT / "data/processed" / args.variant / f"{args.sample_id}.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    video = Path(meta["video_path"])
    if not video.is_file():
        raise FileNotFoundError(f"Original video required: {video}")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arr:
        face_time = arr["video_pts_s"].astype(float)
        face_valid = arr["video_frame_valid"].astype(bool)
    duration = max(float(meta["video_duration_s"]), float(face_time[-1]))
    targets = ([float(x) for x in args.times.split(",")] if args.times else
               [duration * q for q in (.16, .38, .62, .84)])
    if len(targets) != 4 or any(t < 0 or t > duration for t in targets):
        raise ValueError("--times must contain four seconds inside the clip duration")
    frames = choose_frames(video, targets)
    wave, audio_start = read_audio(video)
    hop = 160  # 10 ms at 16 kHz; plotted RMS is for review, not the saved 70-D feature.
    n = len(wave) // hop
    rms = np.sqrt(np.mean(wave[:n * hop].reshape(n, hop) ** 2, axis=1))
    rms_time = audio_start + (np.arange(n) + .5) * hop / 16000

    fig = plt.figure(figsize=(14, 8.8), constrained_layout=True)
    grid = fig.add_gridspec(4, 4, height_ratios=(1.45, 1.1, .58, 2.25))
    ax_words = fig.add_subplot(grid[0, :])
    ax_audio = fig.add_subplot(grid[1, :], sharex=ax_words)
    ax_face = fig.add_subplot(grid[2, :], sharex=ax_words)
    frame_axes = [fig.add_subplot(grid[3, i]) for i in range(4)]

    for word in meta["words"]:
        if word["start"] is None or word["end"] is None:
            continue
        a, b = float(word["start"]), float(word["end"])
        color = "#297C91" if word["valid"] else "#D9774D"
        ax_words.broken_barh([(a, max(b - a, .005))], (.05, .47), facecolors=color, alpha=.9)
        ax_words.text((a + b) / 2, .69, word["text"], rotation=55, ha="left", va="bottom", fontsize=7.5)
    ax_words.set(ylim=(0, 1.52), yticks=[], title="Transcript words at estimated speech times")
    ax_words.text(.99, .96, "Blue: accepted   Orange: low confidence", transform=ax_words.transAxes,
                  ha="right", va="top", fontsize=8)

    ax_audio.fill_between(rms_time, rms, color="#7776A8", alpha=.30)
    ax_audio.plot(rms_time, rms, color="#7776A8", linewidth=.8)
    ax_audio.set(ylabel="RMS", title="Audio energy (10 ms review windows)")
    ax_audio.set_ylim(bottom=0)

    ax_face.scatter(face_time, np.where(face_valid, 1., 0.),
                    c=np.where(face_valid, "#488B58", "#CE654D"), s=12, linewidths=0)
    ax_face.set(ylim=(-.2, 1.2), yticks=[0, 1], yticklabels=["No", "Yes"],
                ylabel="Face", xlabel="Time from video start (s)")
    ax_face.grid(axis="x", alpha=.2)
    for t, _ in frames:
        for ax in (ax_words, ax_audio, ax_face):
            ax.axvline(t, color="#707070", linewidth=.7, linestyle="--", alpha=.55)
    for ax, (t, img) in zip(frame_axes, frames):
        ax.imshow(img)
        ax.axis("off")
        nearest = int(np.argmin(abs(face_time - t)))
        status = "face detected" if face_valid[nearest] else "no face detected"
        ax.set_title(f"Original frame {t:.2f} s\n{status}", fontsize=9)
    ax_words.set_xlim(0, duration)
    for ax in (ax_words, ax_audio):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{args.variant} alignment review · {meta['sample_id'].replace('$', '·')} · {meta['quality_status']}", fontsize=13)
    invalid = [w["text"] for w in meta["words"] if not w["valid"]]
    note = ("Estimated word times require listening to the original audio; face status does not prove speaker identity."
            + (f"  Unresolved words: {', '.join(invalid[:8])}" if invalid else ""))
    fig.text(.5, -.012, note, ha="center", fontsize=8)
    out = ROOT / "results" / ("q1_v2" if args.variant == "q1_v2" else "") / "q1_review_figures"
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{meta['sample_id']}_review"
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white", metadata={"Date": None})
    plt.close(fig)
    print(json.dumps({"sample_id": meta["sample_id"], "status": meta["quality_status"],
                      "decoded_frames": meta["video_decoded_frames"],
                      "sampled_frames": meta["video_processed_frames"],
                      "face_frames": meta["video_face_frames"],
                      "frame_times_s": [round(t, 3) for t, _ in frames],
                      "png": str(base.with_suffix(".png")), "svg": str(base.with_suffix(".svg"))},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
