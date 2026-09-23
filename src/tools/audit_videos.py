"""Audit 100 original videos in Attachment 1 and 20 videos in Attachment 4."""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import av
import pandas as pd
import numpy as np

from src.config import load_config


def audit_videos():
    cfg = load_config("configs/base.json")
    root = cfg["_video_attachment1_root"]
    excel_path = root / "label-100.xlsx"
    df = pd.read_excel(excel_path)

    results = []
    discrepancy_count = 0
    audio_missing_count = 0

    for idx, row in df.iterrows():
        vid = str(row["video_id"])
        cid = str(row["clip_id"])
        sample_id = f"{vid}$_{cid}"
        vpath = root / vid / f"{cid}.mp4"

        # 1. OpenCV probe
        cap = cv2.VideoCapture(str(vpath))
        cv_prop_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv_fps = float(cap.get(cv2.CAP_PROP_FPS))
        cv_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cv_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv_read_frames = 0
        cv_pts = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv_pts.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
            cv_read_frames += 1
        cap.release()

        # 2. PyAV probe
        container = av.open(str(vpath))
        v_stream = container.streams.video[0] if container.streams.video else None
        a_stream = container.streams.audio[0] if container.streams.audio else None

        av_prop_frames = v_stream.frames if v_stream else 0
        av_v_codec = v_stream.codec_context.name if v_stream else None
        av_time_base = str(v_stream.time_base) if v_stream else None
        av_duration_s = float(v_stream.duration * v_stream.time_base) if (v_stream and v_stream.duration and v_stream.time_base) else (float(container.duration / 1e6) if container.duration else 0.0)

        av_read_frames = 0
        av_pts = []
        if v_stream:
            for packet in container.demux(v_stream):
                for frame in packet.decode():
                    av_read_frames += 1
                    if frame.time is not None:
                        av_pts.append(float(frame.time))

        has_audio = a_stream is not None
        audio_codec = a_stream.codec_context.name if a_stream else None
        audio_rate = a_stream.rate if a_stream else None
        audio_channels = a_stream.channels if a_stream else None
        audio_frames = 0
        if a_stream:
            container.seek(0)
            for packet in container.demux(a_stream):
                for frame in packet.decode():
                    audio_frames += frame.samples

        container.close()

        is_discrepant = (cv_prop_frames != cv_read_frames)
        if is_discrepant:
            discrepancy_count += 1
        if not has_audio:
            audio_missing_count += 1

        results.append({
            "sample_id": sample_id,
            "video_id": vid,
            "clip_id": cid,
            "file_size": vpath.stat().st_size,
            "cv_prop_frames": cv_prop_frames,
            "cv_read_frames": cv_read_frames,
            "cv_fps": cv_fps,
            "cv_width": cv_width,
            "cv_height": cv_height,
            "av_prop_frames": av_prop_frames,
            "av_read_frames": av_read_frames,
            "av_duration_s": av_duration_s,
            "av_v_codec": av_v_codec,
            "av_time_base": av_time_base,
            "has_audio": has_audio,
            "audio_codec": audio_codec,
            "audio_rate": audio_rate,
            "audio_channels": audio_channels,
            "audio_total_samples": audio_frames,
            "is_discrepant": is_discrepant,
            "pts_monotonic": bool(np.all(np.diff(av_pts) > 0)) if len(av_pts) > 1 else True,
            "first_pts": av_pts[0] if av_pts else None,
            "last_pts": av_pts[-1] if av_pts else None,
        })

    summary = {
        "total_videos": len(results),
        "discrepancy_count": discrepancy_count,
        "discrepancy_ratio": discrepancy_count / len(results),
        "audio_missing_count": audio_missing_count,
        "samples": results
    }
    return summary


if __name__ == "__main__":
    res = audit_videos()
    print(f"Audited {res['total_videos']} videos.")
    print(f"OpenCV prop vs read discrepancy: {res['discrepancy_count']} / {res['total_videos']} ({res['discrepancy_ratio']*100:.1f}%)")
    print(f"Audio missing count: {res['audio_missing_count']}")
    out_file = Path("results/video_audit_100.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Saved audit to {out_file}")
