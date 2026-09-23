"""Q1: extract text/audio/face features on real word times from Attachment 1.

Run from 工程 with the existing read-only demo Python:
  python -m src.q1_extract --limit 3 --fps 10
  python -m src.q1_extract --fps 10 --resume
Only derived data under 工程 is written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE / "vendor"))

import av
import librosa
import mediapipe as mp
import numpy as np
import openpyxl
import opensmile
import torch
from scipy.spatial.transform import Rotation
from transformers import AutoModelForCTC, AutoTokenizer, BertModel

RAW_ROOT = ENGINE / "data/raw/extracted/E题数据"
VIDEO_ROOT = RAW_ROOT / "附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条"
OUT_ROOT = ENGINE / "data/processed/q1"
RESULTS = ENGINE / "results"
MODELS = ENGINE / "models"
BERT_DIR = MODELS / "bert-base-uncased/86b5e0934494bd15c9632b12f734a8a67f723594"
CTC_DIR = MODELS / "wav2vec2-base-960h"
FACE_MODEL = MODELS / "face_landmarker.task"
SAMPLE_RATE = 16000
HOP = 160
N_FFT = 400
MIN_WORD_SCORE = 0.10  # Mean CTC posterior on forced word letters; conservative review gate.


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows() -> list[dict]:
    book = VIDEO_ROOT / "label-100.xlsx"
    ws = openpyxl.load_workbook(book, read_only=True, data_only=True)["label"]
    it = iter(ws.values)
    keys = next(it)
    rows = [dict(zip(keys, r)) for r in it if r[0] is not None]
    if len(rows) != 100:
        raise ValueError(f"Expected 100 label rows, found {len(rows)}")
    for r in rows:
        r["sample_id"] = f"{r['video_id']}$_${r['clip_id']}"
        r["video_path"] = VIDEO_ROOT / str(r["video_id"]) / f"{r['clip_id']}.mp4"
        if not r["video_path"].is_file():
            raise FileNotFoundError(r["video_path"])
    if len({r["sample_id"] for r in rows}) != 100:
        raise ValueError("Duplicate attachment-1 sample IDs")
    return rows


def decode_audio(path: Path) -> tuple[np.ndarray, float]:
    blocks: list[np.ndarray] = []
    start = None
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("No audio stream")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(stream):
            if start is None and frame.time is not None:
                start = float(frame.time)
            for out in resampler.resample(frame):
                blocks.append(out.to_ndarray().reshape(-1).astype(np.float32) / 32768.0)
        for out in resampler.resample(None):
            blocks.append(out.to_ndarray().reshape(-1).astype(np.float32) / 32768.0)
    if not blocks:
        raise ValueError("Audio stream produced no samples")
    wave = np.concatenate(blocks)
    if not np.isfinite(wave).all():
        raise ValueError("Decoded audio contains NaN or Inf")
    return wave, float(start or 0.0)


def _ctc_words(text: str, vocab: dict[str, int]) -> tuple[list[dict], list[int], list[int]]:
    words = []
    targets = []
    target_to_word = []
    for match in re.finditer(r"\S+", text):
        raw = match.group()
        letters = "".join(c for c in raw.replace("’", "'").upper() if c in vocab and c != "|")
        w = {"text": raw, "char_start": match.start(), "char_end": match.end(),
             "ctc_text": letters, "start": None, "end": None, "score": None,
             "valid": False, "failure_reason": "unsupported_ctc_text" if not letters else "no_ctc_path"}
        wi = len(words)
        words.append(w)
        if not letters:
            continue
        if targets:
            targets.append(vocab["|"])
            target_to_word.append(-1)
        for c in letters:
            targets.append(vocab[c])
            target_to_word.append(wi)
    if not targets:
        raise ValueError("No transcript characters supported by CTC vocabulary")
    return words, targets, target_to_word


def viterbi_ctc(logp: np.ndarray, targets: list[int], blank: int = 0) -> np.ndarray:
    """Maximum-probability CTC state path, including repeat-letter blank rule."""
    t_count, _ = logp.shape
    s_count = 2 * len(targets) + 1
    if t_count < len(targets):
        raise ValueError(f"CTC has {t_count} frames for {len(targets)} labels")
    labels = np.full(s_count, blank, dtype=np.int32)
    labels[1::2] = targets
    dp = np.full(s_count, -np.inf, dtype=np.float64)
    dp[0] = logp[0, blank]
    dp[1] = logp[0, targets[0]]
    back = np.zeros((t_count, s_count), dtype=np.int8)
    valid_skip = np.zeros(s_count, dtype=bool)
    valid_skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
    for t in range(1, t_count):
        advance = np.concatenate(([-np.inf], dp[:-1]))
        skip = np.concatenate(([-np.inf, -np.inf], dp[:-2]))
        skip[~valid_skip] = -np.inf
        choices = np.stack((dp, advance, skip))
        move = choices.argmax(axis=0)
        dp = choices[move, np.arange(s_count)] + logp[t, labels]
        back[t] = move.astype(np.int8)
    state = s_count - 1 if dp[-1] >= dp[-2] else s_count - 2
    if not np.isfinite(dp[state]):
        raise ValueError("CTC alignment has no finite path")
    path = np.empty(t_count, dtype=np.int32)
    for t in range(t_count - 1, -1, -1):
        path[t] = state
        if t:
            state -= int(back[t, state])
    return path


def align_words(text: str, wave: np.ndarray, audio_start: float,
                ctc_model, vocab: dict[str, int], device: str) -> tuple[list[dict], dict]:
    words, targets, target_to_word = _ctc_words(text, vocab)
    normalized = (wave - wave.mean()) / max(float(wave.std()), 1e-7)
    x = torch.from_numpy(normalized).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = ctc_model(x).logits[0]
        logp = torch.log_softmax(logits, dim=-1).cpu().numpy()
    path = viterbi_ctc(logp, targets, blank=vocab["<pad>"])
    frame_period = len(wave) / SAMPLE_RATE / len(path)
    frame_lists = [[] for _ in words]
    scores = [[] for _ in words]
    for t, state in enumerate(path):
        if state % 2 == 0:
            continue
        target_i = (state - 1) // 2
        wi = target_to_word[target_i]
        if wi >= 0:
            frame_lists[wi].append(t)
            scores[wi].append(float(math.exp(float(logp[t, targets[target_i]]))))
    for wi, frames in enumerate(frame_lists):
        if frames:
            words[wi]["start"] = round(audio_start + min(frames) * frame_period, 4)
            words[wi]["end"] = round(audio_start + (max(frames) + 1) * frame_period, 4)
            words[wi]["score"] = round(float(np.mean(scores[wi])), 4)
            if words[wi]["score"] >= MIN_WORD_SCORE:
                words[wi]["valid"] = True
                words[wi]["failure_reason"] = None
            else:
                words[wi]["failure_reason"] = "low_ctc_confidence"
    unresolved = sum(not w["valid"] for w in words)
    info = {"ctc_frames": len(path), "ctc_targets": len(targets),
            "word_count": len(words), "unresolved_words": unresolved,
            "min_word_score": MIN_WORD_SCORE,
            "mean_word_score": round(float(np.mean([w["score"] for w in words if w["score"] is not None])), 4)}
    return words, info


def extract_audio_features(wave: np.ndarray, audio_start: float, smile) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    # openSMILE uses ASCII for config paths; force an ASCII relative path from 工程.
    frame = smile.process_signal(wave, sampling_rate=SAMPLE_RATE)
    smile_x = frame.to_numpy(dtype=np.float32)
    smile_centers = np.array([float((a + (b - a) / 2).total_seconds()) for a, b in frame.index], dtype=np.float64)
    mfcc = librosa.feature.mfcc(y=wave, sr=SAMPLE_RATE, n_mfcc=20, n_fft=N_FFT,
                                hop_length=HOP, center=False).T.astype(np.float32)
    delta = librosa.feature.delta(mfcc.T, mode="nearest").T.astype(np.float32)
    stft = np.abs(librosa.stft(wave, n_fft=N_FFT, hop_length=HOP, center=False))
    extras = np.stack((
        librosa.feature.rms(S=stft, frame_length=N_FFT)[0],
        librosa.feature.zero_crossing_rate(wave, frame_length=N_FFT, hop_length=HOP, center=False)[0],
        librosa.feature.spectral_centroid(S=stft, sr=SAMPLE_RATE)[0],
        librosa.feature.spectral_bandwidth(S=stft, sr=SAMPLE_RATE)[0],
        librosa.feature.spectral_rolloff(S=stft, sr=SAMPLE_RATE)[0],
    ), axis=1).astype(np.float32)
    n = min(len(mfcc), len(delta), len(extras))
    if n == 0 or len(smile_x) == 0:
        raise ValueError("No audio feature frames")
    centers = (np.arange(n) * HOP + N_FFT / 2) / SAMPLE_RATE
    smile_interp = np.stack([np.interp(centers, smile_centers, smile_x[:, j])
                             for j in range(smile_x.shape[1])], axis=1).astype(np.float32)
    values = np.concatenate((smile_interp, mfcc[:n], delta[:n], extras[:n]), axis=1)
    if values.shape[1] != 70 or not np.isfinite(values).all():
        raise ValueError(f"Audio features invalid: {values.shape}")
    starts = audio_start + np.arange(n) * HOP / SAMPLE_RATE
    ends = starts + N_FFT / SAMPLE_RATE
    names = list(smile.feature_names) + [f"mfcc_{i}" for i in range(20)] + [f"mfcc_delta_{i}" for i in range(20)] + ["rms", "zcr", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff"]
    return values, starts, ends, names


GEOMETRY = ((159, 145), (386, 374), (13, 14), (61, 291), (105, 159), (334, 386))
GEOM_NAMES = ("eye_open_left", "eye_open_right", "mouth_open", "mouth_width", "brow_eye_left", "brow_eye_right")


def face_vector(result, blend_names: list[str] | None) -> tuple[np.ndarray | None, list[str] | None, tuple | None]:
    if not result.face_landmarks or not result.face_blendshapes or not result.facial_transformation_matrixes:
        return None, blend_names, None
    categories = result.face_blendshapes[0]
    if blend_names is None:
        blend_names = [c.category_name for c in categories]
    scores = {c.category_name: float(c.score) for c in categories}
    if len(blend_names) != 52 or any(name not in scores for name in blend_names):
        raise ValueError("Face blendshape set is not 52 stable names")
    matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
    pose = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=False).astype(np.float32)
    pts = np.array([(p.x, p.y) for p in result.face_landmarks[0]], dtype=np.float32)
    scale = float(np.linalg.norm(pts[33] - pts[263]))
    if scale < 1e-5:
        return None, blend_names, None
    geom = np.array([np.linalg.norm(pts[a] - pts[b]) / scale for a, b in GEOMETRY], dtype=np.float32)
    vector = np.concatenate((np.array([scores[n] for n in blend_names], dtype=np.float32), pose, geom))
    if vector.shape != (61,) or not np.isfinite(vector).all():
        return None, blend_names, None
    return vector, blend_names, (pts.mean(axis=0), scale)


def extract_visual(path: Path, landmarker, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int, list[float]]:
    selected_times = []
    selected_vectors = []
    valid = []
    blend_names = None
    decoded = 0
    last_selected = -np.inf
    first_pts = None
    last_ms = -1
    tracked_face = None
    possible_face_switches = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            decoded += 1
            if frame.time is None:
                continue
            pts = float(frame.time)
            if first_pts is None:
                first_pts = pts
            if fps > 0 and pts - last_selected < 1 / fps - 1e-5:
                continue
            last_selected = pts
            timestamp_ms = max(last_ms + 1, int(round((pts - first_pts) * 1000)))
            last_ms = timestamp_ms
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.to_ndarray(format="rgb24"))
            result = landmarker.detect_for_video(image, timestamp_ms)
            vector, blend_names, face_position = face_vector(result, blend_names)
            if vector is not None:
                if tracked_face is not None:
                    center, scale = face_position
                    old_center, old_scale = tracked_face
                    jump = np.linalg.norm(center - old_center) > 0.6 * old_scale
                    size_jump = max(scale / old_scale, old_scale / scale) > 1.6
                    if jump or size_jump:
                        possible_face_switches.append(round(pts, 4))
                        vector = None  # Boundary frame is uncertain; start a new shot next frame.
                tracked_face = face_position
            selected_times.append(pts)
            selected_vectors.append(vector if vector is not None else np.zeros(61, dtype=np.float32))
            valid.append(vector is not None)
    if not selected_times:
        raise ValueError("No video frames with PTS")
    names = (blend_names if blend_names is not None else [f"blendshape_{i}" for i in range(52)]) + ["head_x_rad", "head_y_rad", "head_z_rad"] + list(GEOM_NAMES)
    return np.stack(selected_vectors), np.array(selected_times), np.array(valid, dtype=bool), names, decoded, possible_face_switches


def intervals_from_pts(pts: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
    if len(pts) == 1:
        return np.array([pts[0]]), np.array([max(duration, pts[0] + 0.1)])
    mid = (pts[1:] + pts[:-1]) / 2
    starts = np.concatenate(([max(0, pts[0] - (mid[0] - pts[0]))], mid))
    ends = np.concatenate((mid, [min(duration, pts[-1] + (pts[-1] - mid[-1]))]))
    return starts, ends


def pool_interval(values: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                  valid: np.ndarray, a: float, b: float) -> tuple[np.ndarray, bool, float]:
    weights = np.maximum(0, np.minimum(ends, b) - np.maximum(starts, a)) * valid
    total = float(weights.sum())
    if total <= 0 or b <= a:
        return np.zeros(values.shape[1], dtype=np.float32), False, 0.0
    # Frame windows overlap. Their summed weights are valid for a weighted mean,
    # but coverage is the length of their union within the word interval.
    intervals = sorted((max(float(s), a), min(float(e), b)) for s, e, ok in zip(starts, ends, valid)
                       if ok and e > a and s < b)
    covered = 0.0
    left, right = intervals[0]
    for s, e in intervals[1:]:
        if s > right:
            covered += right - left
            left, right = s, e
        else:
            right = max(right, e)
    covered += right - left
    return (weights @ values / total).astype(np.float32), True, min(1.0, covered / (b - a))


def bert_text(text: str, tokenizer, bert, device: str):
    encoded = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=False)
    offsets = encoded.pop("offset_mapping")[0].cpu().numpy()
    with torch.inference_mode():
        x = bert(**{k: v.to(device) for k, v in encoded.items()}).last_hidden_state[0].cpu().numpy().astype(np.float32)
    bert_rows = np.stack([encoded["input_ids"][0].numpy(), encoded["attention_mask"][0].numpy(), encoded["token_type_ids"][0].numpy()])
    return x, bert_rows, offsets


def make_windows(arrays: dict, token_count: int) -> dict:
    content = list(range(1, token_count - 1))
    starts = [0]
    while starts[-1] + 48 < len(content):
        starts.append(starts[-1] + 38)
    out = {name + "_windows": [] for name in ("text", "audio", "vision", "text_bert", "text_valid", "audio_valid", "vision_valid", "audio_imputed", "vision_imputed", "token_word", "token_failure_reason")}
    windows = []
    for begin in starts:
        positions = [0] + content[begin:begin + 48] + [token_count - 1]
        length = len(positions)
        windows.append({"first_content_token": begin + 1, "last_content_token": positions[-2] if length > 2 else None, "length": length})
        for name in ("text", "audio", "vision"):
            x = arrays[name][positions]
            out[name + "_windows"].append(np.pad(x, ((0, 50 - length), (0, 0))))
        for name in ("text_valid", "audio_valid", "vision_valid", "audio_imputed", "vision_imputed", "token_word", "token_failure_reason"):
            x = arrays[name][positions]
            pad = -1 if name == "token_word" else ("padding" if name == "token_failure_reason" else False)
            out[name + "_windows"].append(np.pad(x, (0, 50 - length), constant_values=pad))
        ids = arrays["text_bert"][:, positions]
        out["text_bert_windows"].append(np.pad(ids, ((0, 0), (0, 50 - length))))
    return {k: np.stack(v) for k, v in out.items()} | {"windows": windows}


def process_one(row: dict, tokenizer, bert, ctc_model, vocab: dict,
                landmarker, smile, device: str, fps: float, pipeline_sha: str) -> dict:
    started = time.time()
    video_path = row["video_path"]
    text = str(row["text"])
    wave, audio_start = decode_audio(video_path)
    words, align_info = align_words(text, wave, audio_start, ctc_model, vocab, device)
    audio_x, audio_s, audio_e, audio_names = extract_audio_features(wave, audio_start, smile)
    visual_x, visual_pts, visual_valid, visual_names, decoded, possible_face_switches = extract_visual(video_path, landmarker, fps)
    with av.open(str(video_path)) as container:
        duration = float(container.duration / av.time_base) if container.duration else max(float(visual_pts[-1]), audio_start + len(wave) / SAMPLE_RATE)
    visual_s, visual_e = intervals_from_pts(visual_pts, duration)
    text_x, text_bert, offsets = bert_text(text, tokenizer, bert, device)
    length = len(text_x)
    audio_aligned = np.zeros((length, 70), dtype=np.float32)
    vision_aligned = np.zeros((length, 61), dtype=np.float32)
    audio_valid = np.zeros(length, dtype=bool)
    vision_valid_token = np.zeros(length, dtype=bool)
    text_valid = np.ones(length, dtype=bool)
    audio_imputed = np.zeros(length, dtype=bool)
    vision_imputed = np.zeros(length, dtype=bool)
    token_failure_reason = np.full(length, "special_token", dtype="<U32")
    token_word = np.full(length, -1, dtype=np.int32)
    token_start = np.full(length, np.nan, dtype=np.float32)
    token_end = np.full(length, np.nan, dtype=np.float32)
    audio_cover = np.zeros(length, dtype=np.float32)
    vision_cover = np.zeros(length, dtype=np.float32)
    for ti, (a, b) in enumerate(offsets):
        if a == b:
            continue
        overlap = [max(0, min(int(b), w["char_end"]) - max(int(a), w["char_start"])) for w in words]
        if not overlap or max(overlap) == 0:
            continue
        wi = int(np.argmax(overlap))
        token_word[ti] = wi
        w = words[wi]
        token_failure_reason[ti] = "" if w["valid"] else w["failure_reason"]
        if not w["valid"]:
            continue
        s, e = float(w["start"]), float(w["end"])
        token_start[ti], token_end[ti] = s, e
        audio_aligned[ti], audio_valid[ti], audio_cover[ti] = pool_interval(
            audio_x, audio_s, audio_e, np.ones(len(audio_x), dtype=bool), s, e)
        vision_aligned[ti], vision_valid_token[ti], vision_cover[ti] = pool_interval(
            visual_x, visual_s, visual_e, visual_valid, s, e)
        if not audio_valid[ti]:
            token_failure_reason[ti] = "no_audio_frame"
        elif not vision_valid_token[ti]:
            token_failure_reason[ti] = "no_valid_face_frame"
    arrays = {"text": text_x, "audio": audio_aligned, "vision": vision_aligned,
              "text_bert": text_bert, "text_valid": text_valid,
              "audio_valid": audio_valid, "vision_valid": vision_valid_token,
              "audio_imputed": audio_imputed, "vision_imputed": vision_imputed,
              "token_failure_reason": token_failure_reason,
              "token_word": token_word, "token_start_s": token_start, "token_end_s": token_end,
              "audio_coverage": audio_cover, "vision_coverage": vision_cover,
              "video_pts_s": visual_pts.astype(np.float32), "video_frame_valid": visual_valid,
              "audio_frame_start_s": audio_s.astype(np.float32), "audio_frame_end_s": audio_e.astype(np.float32)}
    windows = make_windows(arrays, length)
    window_meta = windows.pop("windows")
    arrays.update(windows)
    if not np.isfinite(text_x).all() or not np.isfinite(audio_aligned).all() or not np.isfinite(vision_aligned).all():
        raise ValueError("Output has NaN or Inf in feature arrays")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_npz = OUT_ROOT / f"{row['sample_id']}.npz"
    np.savez_compressed(out_npz, **arrays)
    metadata = {
        "sample_id": row["sample_id"], "video_path": str(video_path), "video_sha256": sha256(video_path),
        "transcript": text, "words": words, "alignment": align_info,
        "quality_status": "REVIEW" if (align_info["unresolved_words"] or possible_face_switches
                                        or int(visual_valid.sum()) < 0.5 * len(visual_valid)) else "PASS",
        "token_failure_reasons": token_failure_reason.tolist(),
        "possible_face_switch_times_s": possible_face_switches,
        "audio_start_s": audio_start, "audio_decoded_duration_s": round(len(wave) / SAMPLE_RATE, 4),
        "video_duration_s": duration, "video_decoded_frames": decoded, "video_processed_frames": len(visual_pts),
        "video_face_frames": int(visual_valid.sum()), "token_count": length, "window_count": len(window_meta),
        "windows": window_meta, "audio_feature_names": audio_names, "visual_feature_names": visual_names,
        "model_paths": {"bert": str(BERT_DIR), "ctc": str(CTC_DIR), "face": str(FACE_MODEL)},
        "pipeline_sha256": pipeline_sha,
        "fps_limit": fps, "feature_file": str(out_npz), "feature_sha256": sha256(out_npz),
        "elapsed_s": round(time.time() - started, 2),
    }
    (OUT_ROOT / f"{row['sample_id']}.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"sample_id": row["sample_id"], "status": metadata["quality_status"], "token_count": length,
            "window_count": len(window_meta), "unresolved_words": align_info["unresolved_words"],
            "audio_valid_tokens": int(audio_valid.sum()), "vision_valid_tokens": int(vision_valid_token.sum()),
            "video_decoded_frames": decoded, "video_processed_frames": len(visual_pts),
            "face_frames": int(visual_valid.sum()), "elapsed_s": metadata["elapsed_s"], "error": ""}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Process first N rows only; 0 means all 100")
    parser.add_argument("--ids", default="", help="Comma-separated exact sample IDs")
    parser.add_argument("--fps", type=float, default=10.0, help="Max processed video frames per second; 0 means all frames")
    parser.add_argument("--resume", action="store_true", help="Skip samples with both outputs present")
    args = parser.parse_args()
    rows = load_rows()
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["sample_id"] in wanted]
        if len(rows) != len(wanted):
            raise ValueError("Some --ids do not exist in label-100.xlsx")
    if args.limit:
        rows = rows[:args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR, local_files_only=True, use_fast=True)
    bert = BertModel.from_pretrained(BERT_DIR, local_files_only=True).to(device).eval()
    ctc_model = AutoModelForCTC.from_pretrained(CTC_DIR, local_files_only=True).to(device).eval()
    vocab = json.loads((CTC_DIR / "vocab.json").read_text(encoding="utf-8"))
    opensmile.Smile.default_config_root = property(lambda self: "vendor/opensmile/core/config")
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    if len(smile.feature_names) != 25:
        raise ValueError(f"eGeMAPSv02 LLD yielded {len(smile.feature_names)} features, expected 25")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        # MediaPipe's native loader cannot open this workspace's Chinese absolute path.
        base_options=mp.tasks.BaseOptions(model_asset_path="models/face_landmarker.task"),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True,
    )
    pipeline_hash = hashlib.sha256()
    for artifact in (Path(__file__), BERT_DIR / "model.safetensors", CTC_DIR / "model.safetensors", FACE_MODEL):
        pipeline_hash.update(sha256(artifact).encode("ascii"))
    pipeline_sha = pipeline_hash.hexdigest()
    records = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows, 1):
        sample_id = row["sample_id"]
        if args.resume:
            npz_path, meta_path = (OUT_ROOT / f"{sample_id}.npz", OUT_ROOT / f"{sample_id}.json")
            if npz_path.is_file() and meta_path.is_file():
                try:
                    old = json.loads(meta_path.read_text(encoding="utf-8"))
                    current = (old.get("video_sha256") == sha256(row["video_path"])
                               and old.get("transcript") == str(row["text"])
                               and old.get("fps_limit") == args.fps
                               and old.get("pipeline_sha256") == pipeline_sha
                               and old.get("alignment", {}).get("min_word_score") == MIN_WORD_SCORE
                               and old.get("feature_sha256") == sha256(npz_path))
                    if current:
                        print(f"[{i}/{len(rows)}] SKIP {sample_id}", flush=True)
                        continue
                except (OSError, ValueError, KeyError):
                    pass
        print(f"[{i}/{len(rows)}] START {sample_id}", flush=True)
        try:
            # VIDEO-mode timestamps restart at each clip, so the tracker must too.
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
                record = process_one(row, tokenizer, bert, ctc_model, vocab, landmarker, smile, device, args.fps, pipeline_sha)
        except Exception as exc:
            record = {"sample_id": sample_id, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        records.append(record)
        print(f"[{i}/{len(rows)}] {record['status']} {sample_id}: {record.get('error') or record.get('elapsed_s')}", flush=True)
    inventory = RESULTS / "q1_feature_inventory.csv"
    if inventory.exists() and (args.limit or args.ids or args.resume):
        with inventory.open(encoding="utf-8-sig", newline="") as f:
            prior = {r["sample_id"]: r for r in csv.DictReader(f)}
        prior.update({r["sample_id"]: r for r in records})
        records = list(prior.values())
    cols = ("sample_id", "status", "token_count", "window_count", "unresolved_words", "audio_valid_tokens",
            "vision_valid_tokens", "video_decoded_frames", "video_processed_frames", "face_frames", "elapsed_s", "error")
    with inventory.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(records)
    failed = [r for r in records if r["status"] == "FAIL"]
    print(json.dumps({"processed_this_run": len(rows), "inventory_rows": len(records),
                      "failures": len(failed), "inventory": str(inventory)}, ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
