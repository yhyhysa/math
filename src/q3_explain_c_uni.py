"""Q3: reuse the frozen Q2 C_uni ensemble; predict A4 and inspect evidence.

Run from 工程: python -m src.q3_explain_c_uni --tag my_run
No fitting, labels from A4, or model selection occurs here.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoTokenizer, BertModel

from src.config import load_config
from src.data import load_pickle
from src.q1_extract import CTC_DIR, MIN_WORD_SCORE, align_words, decode_audio
from src.q2_locked_test_eval import ensemble
from src.q2_train_baseline import BERT_REV, BERT_SHA256, content_mask, sha256
from src.q2_train_transformer import SeparateTemporalTransformer, predict, recode_text
from src.q2_valid_missing_audit import SEEDS, load_checkpoint

MODS = ("text", "audio", "vision")
CLASSES = ("negative", "neutral", "positive")
RUN_PREFIX = "q2_c_aug_1layer_concat_eff_tonight_uni_20260924"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_models(root: Path, device: torch.device) -> tuple[list, dict]:
    checkpoints = [load_checkpoint(root, f"{RUN_PREFIX}_seed_{seed}", seed)[1] for seed in SEEDS]
    ref = checkpoints[0]
    for ckpt in checkpoints:
        if ckpt["architecture"] != "separate" or ckpt["fusion"] != "concat" or ckpt["layers"] != 1 or ckpt.get("auxiliary_weight") != .1:
            raise ValueError("Saved C_uni checkpoint is not the selected Q2 architecture")
        for mod in ("audio", "vision"):
            if not np.array_equal(ckpt[f"{mod}_mean"], ref[f"{mod}_mean"]) or not np.array_equal(ckpt[f"{mod}_std"], ref[f"{mod}_std"]):
                raise ValueError("C_uni checkpoints use different normalization")
    models = []
    for ckpt in checkpoints:
        model = SeparateTemporalTransformer(layers=1, fusion="concat", auxiliary=True).to(device).eval()
        model.load_state_dict(ckpt["state_dict"])
        model.requires_grad_(False)
        models.append(model)
    return models, ref


def make_view(samples: list, reference: dict) -> dict[str, np.ndarray]:
    tb = np.stack([s.arrays["text_bert"] for s in samples]).astype(np.int64)
    content = np.stack([content_mask(row) for row in tb])
    arrays = {name: np.stack([s.arrays[name] for s in samples]).astype(np.float32)
              for name in MODS}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite A4 input")
    available = np.stack((content,
                          content & np.any(np.abs(arrays["audio"]) > 1e-6, axis=2),
                          content & np.any(np.abs(arrays["vision"]) > 1e-6, axis=2)), axis=-1)
    for i, name in enumerate(("audio", "vision"), 1):
        arrays[name] = (arrays[name] - reference[f"{name}_mean"]) / reference[f"{name}_std"]
        arrays[name] *= available[:, :, i, None]
    return {**arrays, "available": available, "content": content}


def one_view(view: dict, i: int) -> dict:
    return {key: value[i].copy() for key, value in view.items()}


def stack_views(views: list[dict], device: torch.device) -> dict:
    return {key: torch.as_tensor(np.stack([v[key] for v in views]), device=device)
            for key in ("text", "audio", "vision", "available", "content")}


def model_scores(models: list, views: list[dict], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    tensor_view = stack_views(views, device)
    return ensemble([predict(model, tensor_view, batch_size=96) for model in models])


def text_reference(root: Path) -> np.ndarray:
    # A2 train only. The same official text vectors are already the C_uni input.
    path = next((root / "data" / "raw" / "extracted" / "E题数据" / "附件2-数据集特征文件").rglob("aligned_50.pkl"))
    payload = load_pickle(path)
    rows = payload.splits["train"]
    text = np.stack([s.arrays["text"] for s in rows]).astype(np.float32)
    content = np.stack([content_mask(s.arrays["text_bert"]) for s in rows])
    return text[content].mean(axis=0, dtype=np.float64).astype(np.float32)


def replace(view: dict, mod: int, positions: np.ndarray, ref: np.ndarray,
            mode: str = "mean") -> dict:
    out = {key: value.copy() for key, value in view.items()}
    name = MODS[mod]
    selected = positions & out["available"][:, mod]
    if selected.any():
        out[name][selected] = ref if mode == "mean" else 0
        if mode == "missing":
            out["available"][selected, mod] = False
    return out


def shapley(models: list, original: dict, text_ref: np.ndarray,
            device: torch.device, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    refs = (text_ref, np.zeros(74, np.float32), np.zeros(35, np.float32))
    views = []
    for subset in range(8):
        current = original
        for mod in range(3):
            if not subset & (1 << mod):
                current = replace(current, mod, original["content"], refs[mod], mode)
        views.append(current)
    probs, values = model_scores(models, views, device)
    pred_class = int(probs[7].argmax())
    cls = np.zeros(3, dtype=np.float64)
    reg = np.zeros(3, dtype=np.float64)
    for mod in range(3):
        for subset in range(8):
            if subset & (1 << mod):
                continue
            k = subset.bit_count()
            weight = math.factorial(k) * math.factorial(2-k) / 6
            cls[mod] += weight * (probs[subset | (1 << mod), pred_class] - probs[subset, pred_class])
            reg[mod] += weight * (values[subset | (1 << mod)] - values[subset])
    if not np.isclose(cls.sum(), probs[7, pred_class]-probs[0, pred_class], atol=1e-5) or not np.isclose(reg.sum(), values[7]-values[0], atol=1e-5):
        raise AssertionError("Shapley efficiency failed")
    return cls, reg, probs, values


def words_and_tokens(raw: str, tb: np.ndarray, tokenizer) -> list[dict]:
    encoded = tokenizer(raw, padding="max_length", truncation=True, max_length=50,
                        return_offsets_mapping=True)
    if encoded["input_ids"] != tb[0].tolist() or encoded["attention_mask"] != tb[1].tolist():
        raise ValueError("A4 raw text does not match supplied BERT token IDs")
    words = []
    for match in re.finditer(r"\S+", raw):
        positions = [i for i, (start, end) in enumerate(encoded["offset_mapping"])
                     if end > start and min(match.end(), end) > max(match.start(), start)]
        if positions:
            words.append({"word": match.group(), "char_start": match.start(),
                          "char_end": match.end(), "positions": positions})
    if not words:
        raise ValueError("No raw words mapped to BERT content")
    return words


def windows(words: list[dict], width: int = 3) -> list[dict]:
    result = []
    for start in range(len(words)):
        selected = words[start:min(start + width, len(words))]
        indices = sorted({p for w in selected for p in w["positions"]})
        if indices:
            result.append({"word_start": start, "word_end": start + len(selected),
                           "token_start": min(indices), "token_end": max(indices)+1,
                           "indices": indices, "char_start": selected[0]["char_start"],
                           "char_end": selected[-1]["char_end"]})
    return result


def local_evidence(models: list, original: dict, win: list[dict], text_ref: np.ndarray,
                   device: torch.device, baseline_prob: np.ndarray, baseline_value: float,
                   pred_class: int) -> list[dict]:
    refs = (text_ref, np.zeros(74, np.float32), np.zeros(35, np.float32))
    variants, descriptors = [], []
    for mod in range(3):
        for window in win:
            positions = np.zeros(50, bool)
            positions[window["indices"]] = True
            if not np.any(original["available"][positions, mod]):
                continue
            variants.append(replace(original, mod, positions, refs[mod]))
            descriptors.append((mod, window))
    if not variants:
        return []
    probs, values = model_scores(models, variants, device)
    rows = []
    for (mod, window), probability, value in zip(descriptors, probs, values):
        rows.append({**window, "modality": MODS[mod],
                     "class_drop": float(baseline_prob[pred_class]-probability[pred_class]),
                     "strength_change": float(baseline_value-value)})
    return rows


def top_nonoverlap(rows: list[dict], mod: str, count: int = 2) -> list[dict]:
    selected = []
    for row in sorted((x for x in rows if x["modality"] == mod and x["class_drop"] > 0),
                      key=lambda x: x["class_drop"], reverse=True):
        if all(row["word_end"] <= other["word_start"] or row["word_start"] >= other["word_end"]
               for other in selected):
            selected.append(row)
        if len(selected) == count:
            break
    return selected


def recoded_text_check(models: list, original: dict, tb: np.ndarray, indices: list[int],
                       bert: BertModel, device: torch.device, pred_class: int,
                       base_prob: float) -> float:
    missing = np.zeros((1, 50), dtype=bool)
    missing[0, indices] = True
    recoded = recode_text(tb[None], missing, bert, device, batch_size=1)[0]
    changed = {key: value.copy() for key, value in original.items()}
    changed["text"] = recoded
    probability, _ = model_scores(models, [changed], device)
    return float(base_prob - probability[0, pred_class])


def media_alignment(video: Path, raw: str, ctc, vocab: dict, device: torch.device):
    wave, start = decode_audio(video)
    words, info = align_words(raw, wave, start, ctc, vocab, str(device))
    return words, info


def evidence_time(row: dict, aligned: list[dict] | None) -> tuple[float | None, float | None, str]:
    if aligned is None:
        return None, None, "alignment_failed"
    matches = [w for w in aligned if w["char_end"] > row["char_start"] and w["char_start"] < row["char_end"]]
    if not matches or not all(w["valid"] and w["start"] is not None and w["end"] is not None for w in matches):
        return None, None, "low_confidence_or_unmapped"
    return float(min(w["start"] for w in matches)), float(max(w["end"] for w in matches)), "ctc_reconstructed"


def save_frame(video: Path, timestamp: float, path: Path) -> float | None:
    import av
    best = None
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            delta = abs(float(frame.time)-timestamp)
            if best is None or delta < best[0]:
                best = (delta, float(frame.time), frame.to_image())
            if float(frame.time) > timestamp + .25:
                break
    if best is None:
        return None
    best[2].save(path, quality=80)
    return best[1]


def make_html(out: Path, predictions: list[dict], evidence: list[dict], modality: list[dict],
              video_dir: Path) -> None:
    by_id = {row["sample_id"]: [] for row in predictions}
    for row in evidence:
        by_id[row["sample_id"]].append(row)
    main = {row["sample_id"]: row for row in modality if row["reference"] == "mean"}
    cards = []
    for pred in predictions:
        sid = pred["sample_id"]
        video = quote(os.path.relpath(video_dir / f"{sid}.mp4", out).replace(os.sep, "/"), safe="/.")
        group = by_id[sid]
        lines = []
        for row in group:
            time_text = "时间待核" if row["start_s"] == "" else f'{float(row["start_s"]):.2f}–{float(row["end_s"]):.2f}s'
            seek = "" if row["start_s"] == "" else f'onclick="seek(\'{sid}\',{float(row["start_s"]):.3f})"'
            frame = f'<img src="{html.escape(row["frame_file"])}" alt="视频关键帧">' if row["frame_file"] else ""
            lines.append(f'<div class="ev" {seek}><b>{html.escape(row["modality"])}</b> '
                         f'{html.escape(row["text_span"])} · {time_text} · 概率下降 {float(row["feature_class_drop"]):+.3f}'
                         f' · {html.escape(row["mapping_status"])}{frame}</div>')
        scores = main[sid]
        bars = ''.join(f'<div>{label} <meter min="-1" max="1" low="-.1" high=".1" optimum="1" value="{max(-1,min(1,float(scores[f"class_{mod}"])))}"></meter> {float(scores[f"class_{mod}"]):+.3f}</div>'
                       for label, mod in (("文本", "text"), ("音频", "audio"), ("视觉", "vision")))
        cards.append(f'<article><h2>{sid} · {pred["polarity"]} · 强度 {float(pred["strength"]):+.2f}</h2>'
                     f'<p>三类概率：负 {float(pred["p_negative"]):.3f} / 中 {float(pred["p_neutral"]):.3f} / 正 {float(pred["p_positive"]):.3f}；主要参考：{html.escape(pred["main_modality"])}</p>'
                     f'<p>有效位置：文本 {pred["text_observed"]}，音频 {pred["audio_observed"]}，视觉 {pred["vision_observed"]}（上限 50，填充不计）</p>'
                     f'<p class="raw">{html.escape(pred["raw_text"])}</p><div class="layout"><video id="video{sid}" controls preload="metadata" src="{html.escape(video)}"></video>'
                     f'<div><h3>模态贡献（对当前预测类别，带符号）</h3>{bars}<h3>候选证据，点击跳到原视频</h3>{"".join(lines) or "没有正向支持的可观测候选"}</div></div></article>')
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>问题三人工审查</title><style>body{font:16px system-ui,sans-serif;margin:2em auto;max-width:1150px;background:#f5f6f8;color:#15202b}article{background:white;padding:1.2em;margin:1em 0;border-radius:12px}video{width:100%;max-width:520px}.layout{display:grid;grid-template-columns:1fr 1fr;gap:1em}.raw{background:#eef3fa;padding:.8em}.ev{padding:.5em;border-bottom:1px solid #ddd;cursor:pointer}.ev:hover{background:#eef8ff}.ev img{display:block;width:180px;margin:.4em 0}meter{width:180px}@media(max-width:750px){.layout{display:block}}</style><h1>附件四：逐条看第三问</h1><p>点击证据跳到原视频的估计时间。贡献是模型扰动结果，不是因果权重；音视频特征行对应的秒数需人工核对。附件四无标签，无法计算准确率。</p><script>function seek(id,s){const v=document.getElementById("video"+id);v.currentTime=s;v.play()}</script>' + ''.join(cards) + '</html>'
    (out / "人工审查.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="New run name: letters, numbers, _ or -")
    parser.add_argument("--limit", type=int, default=20, help="Smoke test 1..19; full submission requires 20")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag) or not 1 <= args.limit <= 20:
        parser.error("Invalid tag or limit")
    cfg = load_config()
    root = cfg["_engine_dir"]
    out = root / "results" / f"q3_c_uni_{args.tag}"
    if out.exists():
        raise FileExistsError(f"Preserving existing run: {out}")
    out.mkdir(parents=True)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models, ckpt = load_models(root, device)
        bert_dir = root / "models" / "bert-base-uncased" / BERT_REV
        if sha256(bert_dir / "model.safetensors") != BERT_SHA256:
            raise ValueError("Pinned BERT hash mismatch")
        source_a2 = next((root / "data/raw/extracted/E题数据/附件2-数据集特征文件").rglob("aligned_50.pkl"))
        model_files = [root / "checkpoints" / f"{RUN_PREFIX}_seed_{seed}" / "best.pt" for seed in SEEDS]
        model_files += [bert_dir / "model.safetensors", CTC_DIR / "model.safetensors", source_a2]
        (out / "model_manifest.json").write_text(json.dumps(
            [{"path": str(path), "sha256": sha256(path)} for path in model_files],
            ensure_ascii=False, indent=2), encoding="utf-8")
        tokenizer = AutoTokenizer.from_pretrained(bert_dir, local_files_only=True, use_fast=True)
        bert = BertModel.from_pretrained(bert_dir, local_files_only=True).to(device).eval()
        bert.requires_grad_(False)
        text_ref = text_reference(root)
        source = cfg["_attachment4_root"] / "对齐版本"
        expected = [f"{i:02d}.pkl" for i in range(1, 21)]
        if sorted(p.name for p in source.glob("*.pkl")) != expected:
            raise ValueError("A4 aligned packaging differs from expected 20 files")
        samples, manifest = [], []
        for name in expected[:args.limit]:
            path = source / name
            video = source / "videos" / (path.stem + ".mp4")
            payload = load_pickle(path)
            if list(payload.splits) != ["single"] or len(payload.splits["single"]) != 1:
                raise ValueError(f"Unexpected A4 sample packaging: {name}")
            sample = payload.splits["single"][0]
            if sample.sample_id != path.stem or sample.classification_label is not None or sample.regression_label is not None or not video.is_file():
                raise ValueError(f"Unexpected A4 ID, labels or video: {name}")
            if {k: sample.arrays[k].shape for k in sample.arrays} != {"text": (50,768), "text_bert": (3,50), "audio": (50,74), "vision": (50,35)}:
                raise ValueError(f"Wrong A4 feature dimensions: {name}")
            words_and_tokens(sample.raw_text, sample.arrays["text_bert"], tokenizer)
            samples.append(sample)
            manifest.append({"sample_id": sample.sample_id, "feature_file": str(path), "feature_sha256": sha256(path),
                             "video_file": str(video), "video_sha256": sha256(video)})
        write_csv(out / "input_manifest.csv", manifest)
        view = make_view(samples, ckpt)
        base, strengths = model_scores(models, [one_view(view, i) for i in range(len(samples))], device)
        ctc = AutoModelForCTC.from_pretrained(CTC_DIR, local_files_only=True).to(device).eval()
        ctc.requires_grad_(False)
        vocab = json.loads((CTC_DIR / "vocab.json").read_text(encoding="utf-8"))
        frame_dir = out / "frames"
        frame_dir.mkdir()
        predictions, contributions, evidence, alignment = [], [], [], []
        for i, sample in enumerate(samples):
            sid = sample.sample_id
            original = one_view(view, i)
            pred_class = int(base[i].argmax())
            sh = {}
            for mode in ("mean", "missing"):
                cls, reg, probs, values = shapley(models, original, text_ref, device, mode)
                if not np.allclose(probs[7], base[i], atol=1e-5) or not np.isclose(values[7], strengths[i], atol=1e-5):
                    raise AssertionError("Original prediction changed during explanation")
                sh[mode] = cls
                contributions.append({"sample_id": sid, "reference": mode, "fixed_class": CLASSES[pred_class],
                                      **{f"class_{mod}": float(cls[j]) for j,mod in enumerate(MODS)},
                                      **{f"strength_{mod}": float(reg[j]) for j,mod in enumerate(MODS)},
                                      "base_class_probability": float(probs[0,pred_class]),
                                      "full_class_probability": float(probs[7,pred_class]),
                                      "shapley_sum_error": float(abs(cls.sum()-(probs[7,pred_class]-probs[0,pred_class])))})
            best = int(sh["mean"].argmax())
            main_mod = MODS[best] if sh["mean"][best] > 0 and original["available"][:, best].any() else "no_positive_support"
            predictions.append({"sample_id": sid, "polarity_code": pred_class, "polarity": CLASSES[pred_class],
                                "strength": float(strengths[i]), "p_negative": float(base[i,0]),
                                "p_neutral": float(base[i,1]), "p_positive": float(base[i,2]),
                                "main_modality": main_mod, "raw_text": sample.raw_text,
                                "text_observed": int(original["available"][:,0].sum()),
                                "audio_observed": int(original["available"][:,1].sum()),
                                "vision_observed": int(original["available"][:,2].sum()),
                                "model": "C_uni_three_seed_ensemble"})
            words = words_and_tokens(sample.raw_text, sample.arrays["text_bert"], tokenizer)
            candidates = local_evidence(models, original, windows(words), text_ref, device,
                                        base[i], strengths[i], pred_class)
            try:
                aligned, info = media_alignment(Path(manifest[i]["video_file"]), sample.raw_text, ctc, vocab, device)
                error = ""
            except (ValueError, RuntimeError) as exc:
                aligned, info, error = None, {"word_count": len(words), "unresolved_words": len(words)}, repr(exc)
            alignment.append({"sample_id": sid, "word_count": info["word_count"],
                              "unresolved_words": info["unresolved_words"],
                              "mean_word_score": info.get("mean_word_score", ""), "error": error,
                              "method": "forced CTC alignment of supplied transcript to original video audio",
                              "av_feature_time_status": "estimated from aligned token position; official AV row timestamps unavailable"})
            for mod in MODS:
                for rank, row in enumerate(top_nonoverlap(candidates, mod), 1):
                    start, end, status = evidence_time(row, aligned)
                    text_span = sample.raw_text[row["char_start"]:row["char_end"]]
                    recode_drop = ""
                    if mod == "text":
                        recode_drop = recoded_text_check(models, original, sample.arrays["text_bert"],
                                                         row["indices"], bert, device, pred_class, base[i,pred_class])
                    frame_file, frame_pts = "", ""
                    if mod == "vision" and start is not None:
                        name = f"{sid}_vision_{rank}.jpg"
                        frame_pts = save_frame(Path(manifest[i]["video_file"]), (start+end)/2, frame_dir / name)
                        frame_file = f"frames/{name}" if frame_pts is not None else ""
                    evidence.append({"sample_id": sid, "modality": mod, "rank": rank,
                                     "fixed_class": CLASSES[pred_class], "text_span": text_span,
                                     "char_start": row["char_start"], "char_end": row["char_end"],
                                     "token_start": row["token_start"], "token_end_exclusive": row["token_end"],
                                     "start_s": start if start is not None else "", "end_s": end if end is not None else "",
                                     "frame_pts_s": frame_pts, "frame_file": frame_file,
                                     "feature_class_drop": row["class_drop"],
                                     "feature_strength_change": row["strength_change"],
                                     "original_text_recode_class_drop": recode_drop,
                                     "mapping_status": status,
                                     "human_review": "pending"})
            print(f"A4 {sid}: {CLASSES[pred_class]}, main={main_mod}, evidence={sum(x['sample_id']==sid for x in evidence)}, CTC unresolved={info['unresolved_words']}", flush=True)
        write_csv(out / "predictions.csv", predictions)
        write_csv(out / "modality_contributions.csv", contributions)
        write_csv(out / "key_evidence.csv", evidence)
        write_csv(out / "alignment_quality.csv", alignment)
        write_csv(out / "人工审查记录.csv", [
            {"sample_id": row["sample_id"], "model_polarity": row["polarity"],
             "model_strength": row["strength"], "human_polarity": "", "human_strength_comment": "",
             "evidence_text_matches": "", "evidence_audio_time_matches": "",
             "evidence_visual_frame_matches": "", "review_note": ""}
            for row in predictions])
        make_html(out, predictions, evidence, contributions, source / "videos")
        summary = {"status": "COMPLETE_AWAITING_HUMAN_REVIEW" if args.limit == 20 else "SMOKE_ONLY",
                   "samples": len(samples), "model": RUN_PREFIX, "seeds": list(SEEDS),
                   "training_performed": False, "attachment4_labels_used": False,
                   "mean_reference": "A2 train content-token text mean; zero after Q2 train normalization for audio and vision; original availability retained",
                   "missing_reference": "modality availability set false; sensitivity analysis only",
                   "reference_sensitive_main_count": sum(int(np.argmax(np.array([x[f'class_{m}'] for m in MODS])) != np.argmax(np.array([y[f'class_{m}'] for m in MODS]))) for x,y in zip(contributions[::2], contributions[1::2])),
                   "vision_empty_clips": [p["sample_id"] for p in predictions if p["vision_observed"] == 0],
                   "ctc_unresolved_words": sum(x["unresolved_words"] for x in alignment),
                   "mapping_limit": "A4 does not include official audio/video feature row timestamps. CTC seconds are reconstructed from original media; AV row-to-second links require human review.",
                   "accuracy_available": False}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(out), **summary}, ensure_ascii=False), flush=True)
    except Exception as exc:
        (out / "STATUS.json").write_text(json.dumps({"status": "BLOCKED", "error": repr(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
