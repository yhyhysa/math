r"""Draw a random Q1 clip's three word-aligned feature matrices on real seconds.

Run from 工程 with the existing Python environment:
  ..\训练demo\.venv\Scripts\python.exe -m src.q1_timeline_demo --seed 23
  ..\训练demo\.venv\Scripts\python.exe -m src.q1_timeline_demo --sample-id=<ID>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "q1-timeline"
import matplotlib.pyplot as plt
import numpy as np

DATA = ROOT / "data/processed/q1"
OUT = ROOT / "results/q1_demo"


def pick_sample(data: Path, seed: int, include_review: bool) -> Path:
    options = []
    for p in data.glob("*.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        if (include_review or m["quality_status"] == "PASS") and m["video_face_frames"] > 0:
            options.append(p)
    if not options:
        raise ValueError("No eligible feature files")
    return sorted(options)[int(np.random.default_rng(seed).integers(len(options)))]


def make_heatmap(x: np.ndarray, valid: np.ndarray, token_word: np.ndarray,
                 words: list[dict], duration: float, step: float = 0.05) -> np.ndarray:
    """Average subwords to words, then hold word features over their timed span."""
    bins = max(1, int(np.ceil(duration / step)))
    image = np.full((x.shape[1], bins), np.nan, dtype=np.float32)
    for wi, word in enumerate(words):
        if not word["valid"] or word["start"] is None:
            continue
        selected = (token_word == wi) & valid
        if not selected.any():
            continue
        a = max(0, int(np.floor(word["start"] / step)))
        b = min(bins, max(a + 1, int(np.ceil(word["end"] / step))))
        image[:, a:b] = x[selected].mean(axis=0)[:, None]
    # Within this clip, standardize each channel across time. This shows change,
    # not absolute magnitude or a cross-sample comparison.
    mean = np.nanmean(image, axis=1, keepdims=True)
    std = np.nanstd(image, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        image = (image - mean) / np.maximum(std, 1e-6)
    return np.clip(image, -2.5, 2.5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=23, help="Reproducible random selection")
    parser.add_argument("--sample-id", help="Plot one exact clip ID instead of random selection")
    parser.add_argument("--include-review", action="store_true", help="Allow clips needing manual review")
    parser.add_argument("--variant", choices=("q1", "q1_v2"), default="q1")
    args = parser.parse_args()
    data = ROOT / "data/processed" / args.variant
    out = ROOT / "results" / ("q1_v2" if args.variant == "q1_v2" else "") / "q1_demo"
    meta_path = data / f"{args.sample_id}.json" if args.sample_id else pick_sample(data, args.seed, args.include_review)
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    duration = max(float(m["video_duration_s"]),
                   float(m["audio_start_s"]) + float(m["audio_decoded_duration_s"]))
    with np.load(meta_path.with_suffix(".npz"), allow_pickle=False) as arr:
        token_word = arr["token_word"]
        specifications = (
            ("Text · BERT 768", arr["text"], token_word >= 0),
            ("Audio · 70", arr["audio"], arr["audio_valid"]),
            ("Vision · 61", arr["vision"], arr["vision_valid"]),
        )
        images = [make_heatmap(values, valid, token_word, m["words"], duration)
                  for _, values, valid in specifications]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.2, 1.2]})
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#E8E8E8")
    for ax, (name, _, _), matrix in zip(axes, specifications, images):
        im = ax.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest",
                       extent=(0, duration, 0, matrix.shape[0]), cmap=cmap,
                       vmin=-2.5, vmax=2.5)
        ax.set_ylabel(f"{name}\nfeature index")
        ax.set_xlim(0, duration)
        ax.grid(False)
        for word in m["words"]:
            if word["valid"] and word["start"] is not None:
                ax.axvline(float(word["start"]), lw=0.35, color="#303030", alpha=0.28)
    axes[-1].set_xlabel("")
    label_words = [w for w in m["words"] if w["valid"] and w["start"] is not None]
    stride = max(1, int(np.ceil(len(label_words) / 20)))
    for word in label_words[::stride]:
        center = (float(word["start"]) + float(word["end"])) / 2
        axes[-1].annotate(word["text"], (center, 0), xytext=(0, -23),
                          textcoords="offset points", rotation=60, ha="right", va="top", fontsize=7)
    fig.subplots_adjust(right=0.89, bottom=0.23, hspace=0.10)
    colorbar_ax = fig.add_axes((0.91, 0.20, 0.014, 0.55))
    fig.colorbar(im, cax=colorbar_ax, label="Within-clip z-score (clipped ±2.5)")
    display_id = m["sample_id"].replace("$", "·")
    fig.suptitle(f"Q1 word-aligned multimodal features · {display_id} · {m['quality_status']}", fontsize=13)
    fig.text(0.5, 0.075, "Time from clip start (seconds)", ha="center", fontsize=10)
    fig.text(0.5, 0.025,
             "Gray = no valid timed feature. BERT token vectors are averaged by word; all panels use the same word time spans.",
             ha="center", fontsize=9)
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{m['sample_id']}_timeline"
    fig.savefig(base.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(base.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
    plt.close(fig)
    summary = {"sample_id": m["sample_id"], "seed": args.seed if args.sample_id is None else None,
               "status": m["quality_status"], "words": len(m["words"]),
               "duration_s": round(duration, 3), "png": str(base.with_suffix(".png")),
               "svg": str(base.with_suffix(".svg"))}
    (out / "last_demo.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
