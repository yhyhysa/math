"""Independent structural validation of all Q1 outputs; run from 工程."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/q1"
INVENTORY = ROOT / "results/q1_feature_inventory.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    with INVENTORY.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    errors = []
    statuses = Counter(r["status"] for r in rows)
    reasons = Counter()
    review_rows = []
    total_words = total_unresolved = total_audio = total_vision = 0
    if len(rows) != 100 or len({r["sample_id"] for r in rows}) != 100:
        errors.append(f"Inventory must contain 100 unique IDs; got {len(rows)}")
    for row in rows:
        sid = row["sample_id"]
        meta_path, npz_path = DATA / f"{sid}.json", DATA / f"{sid}.npz"
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            with np.load(npz_path, allow_pickle=False) as x:
                n = m["token_count"]
                if m["sample_id"] != sid or m["quality_status"] != row["status"]:
                    raise ValueError("Metadata and inventory disagree")
                if m["feature_sha256"] != sha256(npz_path):
                    raise ValueError("Feature SHA mismatch")
                if x["text"].shape != (n, 768) or x["audio"].shape != (n, 70) or x["vision"].shape != (n, 61):
                    raise ValueError("Feature shape mismatch")
                if x["text_bert"].shape != (3, n):
                    raise ValueError("BERT input shape mismatch")
                if x["text_windows"].shape != (m["window_count"], 50, 768):
                    raise ValueError("Window shape mismatch")
                if any(not np.isfinite(x[k]).all() for k in ("text", "audio", "vision")):
                    raise ValueError("Nonfinite feature")
                wi = x["token_word"]
                av, vv = x["audio_valid"], x["vision_valid"]
                times = x["token_start_s"]
                bad = [i for i, w in enumerate(m["words"]) if not w["valid"]]
                for i in bad:
                    positions = np.flatnonzero(wi == i)
                    if np.any(av[positions]) or np.any(vv[positions]) or np.any(np.isfinite(times[positions])):
                        raise ValueError(f"Invalid word {i} has valid token modality or time")
                if np.any(x["audio"][~av]) or np.any(x["vision"][~vv]):
                    raise ValueError("Invalid token has nonzero modality features")
                if np.any(np.diff(x["video_pts_s"]) < 0):
                    raise ValueError("Video PTS not monotonic")
                if np.any((x["audio_coverage"] < 0) | (x["audio_coverage"] > 1)) or np.any((x["vision_coverage"] < 0) | (x["vision_coverage"] > 1)):
                    raise ValueError("Coverage outside [0,1]")
                positions = set()
                for win in m["windows"]:
                    first, last = win["first_content_token"], win["last_content_token"]
                    if last is not None:
                        positions.update(range(first, last + 1))
                if positions != set(range(1, n - 1)):
                    raise ValueError("Windows do not cover all content tokens")
                if int(av.sum()) != int(row["audio_valid_tokens"]) or int(vv.sum()) != int(row["vision_valid_tokens"]):
                    raise ValueError("Valid token counts disagree")
                if len(bad) != int(row["unresolved_words"]):
                    raise ValueError("Invalid word count disagrees")
                total_words += len(m["words"])
                total_unresolved += len(bad)
                total_audio += int(av.sum())
                total_vision += int(vv.sum())
                reasons.update(w["failure_reason"] for w in m["words"] if not w["valid"])
                face_fraction = m["video_face_frames"] / m["video_processed_frames"]
                for w in m["words"]:
                    if not w["valid"]:
                        review_rows.append({"sample_id": sid, "reason": w["failure_reason"],
                                            "word": w["text"], "start_s": w["start"],
                                            "end_s": w["end"], "score": w["score"],
                                            "face_frame_fraction": round(face_fraction, 3)})
                if face_fraction < 0.5:
                    review_rows.append({"sample_id": sid, "reason": "low_face_coverage",
                                        "word": "", "start_s": "", "end_s": "", "score": "",
                                        "face_frame_fraction": round(face_fraction, 3)})
                for t in m["possible_face_switch_times_s"]:
                    review_rows.append({"sample_id": sid, "reason": "possible_face_switch",
                                        "word": "", "start_s": t, "end_s": "", "score": "",
                                        "face_frame_fraction": round(face_fraction, 3)})
        except Exception as exc:
            errors.append(f"{sid}: {type(exc).__name__}: {exc}")
    report = {"status": "PASS" if not errors else "FAIL", "samples": len(rows),
              "statuses": dict(statuses), "total_words": total_words,
              "unresolved_words": total_unresolved, "unresolved_reasons": dict(reasons),
              "audio_valid_tokens": total_audio, "vision_valid_tokens": total_vision,
              "errors": errors}
    target = ROOT / "results/q1_validation.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (ROOT / "results/q1_review_queue.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("sample_id", "reason", "word", "start_s", "end_s", "score", "face_frame_fraction"))
        writer.writeheader()
        writer.writerows(review_rows)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
