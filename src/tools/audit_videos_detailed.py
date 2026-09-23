"""Detailed audit of 100 original videos in Attachment 1 and 20 videos in Attachment 4.

Records:
  - cv_read_frames, av_read_frames, valid_pts_frames
  - first_pts, last_pts, pts_monotonic
  - container_duration_s
  - audio_sample_rate, audio_channels, audio_samples, audio_duration_s
  - audio_to_container_ratio = audio_duration_s / container_duration_s
  - Identifies samples with abs(audio_to_container_ratio - 1.0) > 0.05
  - Root cause labeled as "待确认" (to be confirmed).

Outputs:
  - results/video_audit_detailed.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

from src.config import load_config


def audit_videos_detailed() -> dict:
    t0 = time.time()
    cfg = load_config("configs/base.json")
    a1_root = cfg["_video_attachment1_root"]
    excel_path = a1_root / "label-100.xlsx"
    df = pd.read_excel(excel_path, sheet_name="label")

    print("==================================================")
    print("      Detailed Video & Audio Stream Audit         ")
    print("==================================================")

    a1_results = []
    discrepancy_count = 0
    audio_present_count = 0
    pts_monotonic_count = 0
    high_audio_dev_samples = []

    for idx, row in df.iterrows():
        vid = str(row["video_id"]).strip()
        cid = str(row["clip_id"]).strip()
        sample_id = f"{vid}$_{cid}"
        vpath = a1_root / vid / f"{cid}.mp4"

        # 1. OpenCV
        cap = cv2.VideoCapture(str(vpath))
        cv_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv_fps = float(cap.get(cv2.CAP_PROP_FPS))
        cv_read = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            cv_read += 1
        cap.release()

        # 2. PyAV
        container = av.open(str(vpath))
        vs = container.streams.video[0] if container.streams.video else None
        as_ = container.streams.audio[0] if container.streams.audio else None

        container_dur_s = float(container.duration / 1e6) if container.duration else 0.0

        av_read = 0
        pts_list = []
        if vs:
            for packet in container.demux(vs):
                for frame in packet.decode():
                    av_read += 1
                    if frame.time is not None:
                        pts_list.append(float(frame.time))

        has_audio = (as_ is not None)
        audio_rate = as_.rate if as_ else None
        audio_channels = as_.channels if as_ else None
        audio_samples = 0
        if as_:
            container.seek(0)
            for packet in container.demux(as_):
                for frame in packet.decode():
                    audio_samples += frame.samples

        container.close()

        audio_dur_s = (audio_samples / audio_rate) if (audio_rate and audio_samples > 0) else 0.0
        audio_dev_ratio = (audio_dur_s / container_dur_s) if container_dur_s > 0 else 0.0
        audio_dev_abs = abs(audio_dev_ratio - 1.0)

        is_disc = (cv_prop != cv_read)
        if is_disc:
            discrepancy_count += 1
        if has_audio:
            audio_present_count += 1

        is_mono = bool(np.all(np.diff(pts_list) > 0)) if len(pts_list) > 1 else True
        if is_mono:
            pts_monotonic_count += 1

        if audio_dev_abs > 0.05:
            high_audio_dev_samples.append({
                "sample_id": sample_id,
                "container_duration_s": round(container_dur_s, 4),
                "audio_decoded_duration_s": round(audio_dur_s, 4),
                "audio_duration_s": round(audio_dur_s, 4),
                "audio_to_container_ratio": round(audio_dev_ratio, 4),
                "audio_coverage_ratio": round(audio_dev_ratio, 6),
                "deviation_percent": round(audio_dev_abs * 100, 2),
                "audio_integrity_status": (
                    "无法判定：未检查 MP4 atom 结构与 AAC/H.264 码流，不能仅凭可解码即宣称音轨完整"
                ),
                "root_cause_status": "待确认（未做 MP4 atom / NAL 解析）",
            })

        a1_results.append({
            "sample_id": sample_id,
            "video_id": vid,
            "clip_id": cid,
            "cv_prop_frames": cv_prop,
            "cv_read_frames": cv_read,
            "av_read_frames": av_read,
            "cv_av_identical": (cv_read == av_read),
            "valid_pts_frames": len(pts_list),
            "frames_without_valid_pts": int(av_read - len(pts_list)),
            "pts_coverage_ratio": round(len(pts_list) / av_read, 6) if av_read else None,
            "first_pts_s": pts_list[0] if pts_list else None,
            "last_pts_s": pts_list[-1] if pts_list else None,
            "pts_span_s": round(pts_list[-1] - pts_list[0], 6) if len(pts_list) > 1 else None,
            "pts_monotonic": is_mono,
            "container_duration_s": round(container_dur_s, 4),
            "video_pts_span_to_container_ratio": (
                round((pts_list[-1] - pts_list[0]) / container_dur_s, 6)
                if (len(pts_list) > 1 and container_dur_s > 0) else None
            ),
            "has_audio_stream": has_audio,
            "audio_rate_hz": audio_rate,
            "audio_channels": audio_channels,
            "audio_total_samples": audio_samples,
            "audio_decoded_duration_s": round(audio_dur_s, 4),
            "audio_to_container_ratio": round(audio_dev_ratio, 4),
            "audio_coverage_ratio": round(audio_dev_ratio, 6),
            "audio_deviation_gt_5_pct": (audio_dev_abs > 0.05),
            "audio_integrity_status": (
                "无法判定：未检查 MP4 atom 结构与 H.264/AAC 码流，"
                "不能仅凭存在音轨即宣称音轨完整或无截断"
            ),
            "time_mapping_status": "unresolved（未取得词级真实时间映射；不得由 50 个位置等分推断）",
            "container_discrepancy": is_disc,
            "root_cause_status": "待确认（未做 MP4 atom / H.264 NAL 解析；不得写成已证明的切片前元数据残留）",
        })

    # Attachment 4 videos (20)
    a4_vdir = cfg["_attachment4_root"] / "对齐版本" / "videos"
    a4_vids = sorted(a4_vdir.glob("*.mp4"))
    a4_results = []
    a4_disc_count = 0

    for vp in a4_vids:
        cap = cv2.VideoCapture(str(vp))
        cv_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv_read = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            cv_read += 1
        cap.release()

        container = av.open(str(vp))
        vs = container.streams.video[0] if container.streams.video else None
        as_ = container.streams.audio[0] if container.streams.audio else None
        container_dur_s = float(container.duration / 1e6) if container.duration else 0.0

        av_read = 0
        pts_list = []
        if vs:
            for packet in container.demux(vs):
                for frame in packet.decode():
                    av_read += 1
                    if frame.time is not None:
                        pts_list.append(float(frame.time))

        has_audio = (as_ is not None)
        audio_rate = as_.rate if as_ else None
        audio_channels = as_.channels if as_ else None
        audio_samples = 0
        if as_:
            container.seek(0)
            for packet in container.demux(as_):
                for frame in packet.decode():
                    audio_samples += frame.samples

        container.close()

        audio_dur_s = (audio_samples / audio_rate) if (audio_rate and audio_samples > 0) else 0.0
        audio_dev_ratio = (audio_dur_s / container_dur_s) if container_dur_s > 0 else 0.0
        is_disc = (cv_prop != cv_read)
        if is_disc:
            a4_disc_count += 1

        a4_results.append({
            "video_file": vp.name,
            "cv_prop_frames": cv_prop,
            "cv_read_frames": cv_read,
            "av_read_frames": av_read,
            "cv_av_identical": (cv_read == av_read),
            "valid_pts_frames": len(pts_list),
            "frames_without_valid_pts": int(av_read - len(pts_list)),
            "pts_coverage_ratio": round(len(pts_list) / av_read, 6) if av_read else None,
            "first_pts_s": pts_list[0] if pts_list else None,
            "last_pts_s": pts_list[-1] if pts_list else None,
            "container_duration_s": round(container_dur_s, 4),
            "has_audio_stream": has_audio,
            "audio_decoded_duration_s": round(audio_dur_s, 4),
            "audio_to_container_ratio": round(audio_dev_ratio, 4),
            "audio_coverage_ratio": round(audio_dev_ratio, 6),
            "container_discrepancy": is_disc,
            "time_mapping_status": "unresolved（未取得词级真实时间映射）",
        })

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "attachment1_summary": {
            "total_videos": len(a1_results),
            "container_discrepancy_count": discrepancy_count,
            "container_discrepancy_ratio": discrepancy_count / len(a1_results),
            "cv_and_pyav_decode_identical_count": sum(1 for r in a1_results if r["cv_av_identical"]),
            "audio_stream_present_count": audio_present_count,
            "pts_strictly_monotonic_count": pts_monotonic_count,
            "frames_with_valid_pts_count": sum(r["valid_pts_frames"] for r in a1_results),
            "frames_decoded_count": sum(r["av_read_frames"] for r in a1_results),
            "samples_with_full_pts_coverage": sum(
                1 for r in a1_results if r["pts_coverage_ratio"] == 1.0
            ),
            "samples_with_missing_pts": sum(1 for r in a1_results if r["frames_without_valid_pts"] > 0),
            "audio_duration_dev_gt_5_pct_count": len(high_audio_dev_samples),
            "high_audio_dev_samples": high_audio_dev_samples,
            "container_discrepancy_root_cause": (
                "待确认：实测仅支持“容器标称帧数 CAP_PROP_FRAME_COUNT 与实际解码帧数不符，"
                "而 OpenCV 与 PyAV 解码帧数一致、记录到的 PTS 单调”。"
                "本轮未解析 MP4 atom 结构与 H.264 NAL 码流，因此不得写成已证明的‘切片前元数据残留’。"
            ),
            "audio_track_integrity_statement": (
                "不可宣称‘音轨完整’：仅实测到音轨存在且音频帧可解码，并记录了逐条实际解码时长、"
                "容器时长与覆盖比。未检查 MP4 atom 与 AAC/H.264 码流，因此静音、截断或丢帧均未排除。"
            ),
            "time_mapping_statement": (
                "无法直接由视频位置等分获得词时间戳；必须通过 Forced Alignment 建立真实 PTS 对应。"
                "本轮未取得真实时间映射，相关位置标记为 unresolved。"
            ),
        },
        "attachment4_summary": {
            "total_videos": len(a4_results),
            "container_discrepancy_count": a4_disc_count,
            "cv_and_pyav_decode_identical_count": sum(1 for r in a4_results if r["cv_av_identical"]),
            "audio_stream_present_count": sum(1 for r in a4_results if r["has_audio_stream"]),
        },
        "samples_attachment1": a1_results,
        "samples_attachment4": a4_results,
    }

    out_file = cfg["_results_dir"] / "video_audit_detailed.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Detailed video audit saved to: {out_file}")
    print(f"Attachment 1 Discrepancy: {discrepancy_count}/100, CV==PyAV: {report['attachment1_summary']['cv_and_pyav_decode_identical_count']}/100")
    print(f"Audio dev > 5% samples: {len(high_audio_dev_samples)}: {high_audio_dev_samples}")
    return report


if __name__ == "__main__":
    audit_videos_detailed()
