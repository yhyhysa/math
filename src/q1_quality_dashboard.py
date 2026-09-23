"""Recompute Q1 coverage proxies and a four-panel dashboard from actual outputs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "q1-quality-dashboard"
import matplotlib.pyplot as plt
import numpy as np

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("q1", "q1_v2"), default="q1")
    args = parser.parse_args()
    data = ROOT / "data/processed" / args.variant
    results = ROOT / "results" / ("q1_v2" if args.variant == "q1_v2" else "")
    results.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(data.glob("*.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        n_words = len(m["words"])
        valid_words = sum(bool(w["valid"]) for w in m["words"])
        n_frames = int(m["video_processed_frames"])
        face_frames = int(m["video_face_frames"])
        rows.append({"sample_id": m["sample_id"], "status": m["quality_status"],
                     "words": n_words, "valid_words": valid_words,
                     "word_alignment_coverage": valid_words / n_words,
                     "processed_video_frames": n_frames, "face_frames": face_frames,
                     "face_detection_rate": face_frames / n_frames,
                     "video_duration_s": m["video_duration_s"]})
    if len(rows) != 100 or len({r["sample_id"] for r in rows}) != 100:
        raise ValueError("Expected 100 unique Q1 samples")
    target_csv = results / "q1_quality_proxy_metrics.csv"
    with target_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    word = np.array([r["word_alignment_coverage"] for r in rows])
    face = np.array([r["face_detection_rate"] for r in rows])
    categories = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    counts = np.histogram(word, bins=[0, .2, .4, .6, .8, 1.000001])[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    bins = np.linspace(0, 1, 11)
    axes[0, 0].hist(word, bins=bins, color="#287E94", edgecolor="white")
    axes[0, 0].set(title="Word alignment coverage", xlabel="Valid aligned words / transcript words", ylabel="Clips")
    axes[0, 1].hist(face, bins=bins, color="#639447", edgecolor="white")
    axes[0, 1].set(title="Face detection rate", xlabel="Face-detected / sampled video frames", ylabel="Clips")
    for status, color in (("PASS", "#287E94"), ("REVIEW", "#D16F4F")):
        selected = [r for r in rows if r["status"] == status]
        axes[1, 0].scatter([r["word_alignment_coverage"] for r in selected],
                           [r["face_detection_rate"] for r in selected],
                           label=f"{status} ({len(selected)})", color=color, alpha=.78, s=28)
    axes[1, 0].set(title="Clip-level coverage", xlabel="Word alignment coverage",
                   ylabel="Face detection rate", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    axes[1, 0].legend(frameon=False, fontsize=8)
    bars = axes[1, 1].bar(categories, counts, color=["#D16F4F", "#DA9A60", "#CEAF62", "#81A680", "#287E94"])
    axes[1, 1].set(title="Word coverage bands", xlabel="Word alignment coverage", ylabel="Clips")
    axes[1, 1].bar_label(bars, padding=2, fontsize=9)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Q1 automatic coverage proxies (100 clips)", fontsize=14)
    fig.text(.5, -.015, "These are coverage measures, not independently verified timing or face-identity accuracy.",
             ha="center", fontsize=9)
    base = results / "q1_quality_proxy_dashboard"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white", metadata={"Date": None})
    plt.close(fig)
    report = {"samples": len(rows), "status": dict(Counter(r["status"] for r in rows)),
              "word_alignment_coverage_mean": round(float(word.mean()), 4),
              "word_alignment_coverage_median": round(float(np.median(word)), 4),
              "word_alignment_coverage_micro": round(sum(r["valid_words"] for r in rows) / sum(r["words"] for r in rows), 4),
              "face_detection_rate_mean": round(float(face.mean()), 4),
              "face_detection_rate_median": round(float(np.median(face)), 4),
              "face_detection_rate_micro": round(sum(r["face_frames"] for r in rows) / sum(r["processed_video_frames"] for r in rows), 4),
              "zero_face_clips": int((face == 0).sum()),
              "word_coverage_band_counts": dict(zip(categories, counts.tolist())),
              "definitions": {"word_alignment_coverage": "Count(valid CTC words) / count(transcript whitespace words)",
                              "face_detection_rate": "Count(face-valid sampled frames) / count(sampled video frames)"},
              "limits": "Neither proxy measures manual word-boundary error or correct-speaker face identity."}
    (results / "q1_quality_proxy_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
