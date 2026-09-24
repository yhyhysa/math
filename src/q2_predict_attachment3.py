"""Run a saved Q2 ensemble on the 30 unlabeled aligned A3 samples."""
from __future__ import annotations

import argparse
import csv
import json
import re

import numpy as np
import torch
from transformers import BertModel

from src.config import load_config
from src.data import load_pickle
from src.q2_locked_test_eval import ensemble
from src.q2_train_baseline import (
    BERT_REV, BERT_SHA256, Baseline, content_mask, pooled,
    predict as predict_mlp, sha256,
)
from src.q2_train_transformer import (
    SeparateTemporalTransformer, as_view, predict as predict_temporal, recode_text,
)
from src.q2_valid_missing_audit import SEEDS, load_checkpoint


def longest_run(flags: np.ndarray) -> int:
    best = now = 0
    for flag in flags:
        now = now + 1 if flag else 0
        best = max(best, now)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("TV", "C", "C_uni"), default="C_uni",
                        help="C_uni: final auxiliary-head model; C/TV: historical models")
    parser.add_argument("--candidate-prefix", default=None,
                        help="finished three-modal run prefix without _seed_N; writes a separate A3 result")
    args = parser.parse_args()
    if args.model == "C_uni":
        if args.candidate_prefix:
            parser.error("C_uni fixes its training prefix; use --model C for other candidates")
        args.model = "C"
        args.candidate_prefix = "q2_c_aug_1layer_concat_eff_tonight_uni_20260924"
    if args.candidate_prefix and (args.model != "C" or
                                  not re.fullmatch(r"[A-Za-z0-9_-]+", args.candidate_prefix)):
        parser.error("candidate-prefix requires --model C and a safe run prefix")
    cfg = load_config()
    root = cfg["_engine_dir"]
    out_name = (f"q2_attachment3_candidate_{args.candidate_prefix}" if args.candidate_prefix else
                "q2_attachment3_locked_20260924" if args.model == "TV" else
                "q2_attachment3_three_modal_20260924")
    out = root / "results" / out_name
    out.mkdir(parents=True, exist_ok=True)
    for name in ("predictions.csv", "summary.json", "input_manifest.csv"):
        if (out / name).exists():
            raise FileExistsError(f"Preserve the first A3 prediction: {out / name}")
    lock = root / "results" / "q2_locked_test_20260924" / "模型锁定与测试协议.md"
    test = root / "results" / "q2_locked_test_20260924" / "test_summary.json"
    if not lock.exists() or json.loads(test.read_text(encoding="utf-8"))["status"] != "LOCKED_TEST_EVALUATED":
        raise ValueError("Locked model and one-off test record are required")
    bert_dir = root / "models" / "bert-base-uncased" / BERT_REV
    if sha256(bert_dir / "model.safetensors") != BERT_SHA256:
        raise ValueError("Pinned BERT weight hash mismatch")
    run_name = ((lambda seed: f"q2_aug_TV_step4_20260924_seed_{seed}") if args.model == "TV" else
                (lambda seed: f"{args.candidate_prefix}_seed_{seed}") if args.candidate_prefix else
                (lambda seed: f"q2_c_aug_1layer_concat_eff_step9_lr3e4_20260924_seed_{seed}"))
    saved = [load_checkpoint(root, run_name(seed), seed)[1] for seed in SEEDS]
    ref = saved[0]
    for checkpoint in saved:
        if args.model == "TV":
            if checkpoint["modalities"] != "TV" or checkpoint.get("fusion", "concat") != "concat":
                raise ValueError("A3 checkpoint differs from the saved TV model")
        elif args.candidate_prefix:
            if checkpoint["architecture"] != "separate" or checkpoint["fusion"] not in ("concat", "reliability_gate") or \
                    checkpoint["learning_rate"] != ref["learning_rate"] or \
                    checkpoint.get("auxiliary_weight", 0.0) != ref.get("auxiliary_weight", 0.0) or \
                    checkpoint.get("mask_pattern", "single") != ref.get("mask_pattern", "single"):
                raise ValueError("Candidate checkpoint configurations differ")
        elif checkpoint["architecture"] != "separate" or checkpoint["fusion"] != "concat" or \
                checkpoint["learning_rate"] != 3e-4:
            raise ValueError("A3 checkpoint differs from the saved three-modal comparator")
        for name in ("audio", "vision"):
            if not np.array_equal(checkpoint[f"{name}_mean"], ref[f"{name}_mean"]) or \
                    not np.array_equal(checkpoint[f"{name}_std"], ref[f"{name}_std"]):
                raise ValueError("Locked training normalization differs across seeds")
    source = cfg["_attachment3_root"] / "对齐版本"
    expected = [f"附件3_{i:02d}.pkl" for i in range(1, 31)]
    actual = sorted(p.name for p in source.glob("*.pkl"))
    if actual != expected:
        raise ValueError(f"Expected exactly 30 aligned A3 files; found {len(actual)}")
    samples, manifest = [], []
    for name in expected:
        path = source / name
        payload = load_pickle(path)
        if list(payload.splits) != ["test"] or len(payload.splits["test"]) != 1:
            raise ValueError(f"Unexpected A3 packaging: {name}")
        sample = payload.splits["test"][0]
        if sample.sample_id != path.stem or sample.classification_label is not None or \
                sample.regression_label is not None or sample.raw_text is not None or \
                set(sample.arrays) != {"text_bert", "audio", "vision"}:
            raise ValueError(f"Unexpected A3 metadata or labels: {name}")
        shapes = {key: sample.arrays[key].shape for key in sample.arrays}
        if shapes != {"text_bert": (3, 50), "audio": (50, 74), "vision": (50, 35)}:
            raise ValueError(f"Unexpected A3 feature dimensions: {name}: {shapes}")
        if not all(np.isfinite(sample.arrays[key]).all() for key in ("audio", "vision")):
            raise ValueError(f"Nonfinite A3 feature: {name}")
        content_mask(sample.arrays["text_bert"])
        samples.append(sample)
        manifest.append({"sample_id": sample.sample_id, "source_file": name,
                         "sha256": sha256(path)})
    tb = np.stack([s.arrays["text_bert"] for s in samples]).astype(np.int64)
    content = np.stack([content_mask(row) for row in tb])
    audio = np.stack([s.arrays["audio"] for s in samples]).astype(np.float32)
    vision = np.stack([s.arrays["vision"] for s in samples]).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Rebuilding the 30 A3 text features with locked BERT...", flush=True)
    bert = BertModel.from_pretrained(bert_dir, local_files_only=True).to(device).eval()
    bert.requires_grad_(False)
    text = recode_text(tb, np.zeros_like(content), bert, device)
    del bert
    observed = {
        "text": content.copy(),
        "audio": content & np.any(np.abs(audio) > 1e-6, axis=2),
        "vision": content & np.any(np.abs(vision) > 1e-6, axis=2),
    }
    for name, array in (("audio", audio), ("vision", vision)):
        array -= ref[f"{name}_mean"]
        array /= ref[f"{name}_std"]
        array *= observed[name][..., None]
    data = {"array": {"text": text, "audio": audio, "vision": vision},
            "content": content, "observed": observed}
    x = (torch.as_tensor(pooled(data, keep_modalities="TV"), device=device)
         if args.model == "TV" else as_view(data, device))
    predictions = []
    for seed, checkpoint in zip(SEEDS, saved):
        model = (Baseline(checkpoint["input_dim"]) if args.model == "TV" else
                 SeparateTemporalTransformer(layers=checkpoint["layers"], fusion=checkpoint["fusion"],
                                             auxiliary=checkpoint.get("auxiliary_weight", 0.0) > 0)).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        predictions.append(predict_mlp(model, x) if args.model == "TV" else
                           predict_temporal(model, x))
        print(f"Saved {args.model} seed {seed} predicted 30 A3 samples", flush=True)
        del model
    probabilities, strength = ensemble(predictions)
    if probabilities.shape != (30, 3) or strength.shape != (30,) or \
            not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6) or \
            not np.isfinite(strength).all() or np.max(np.abs(strength)) > 3:
        raise ValueError("Invalid three-class probability or strength output")
    classes = ("negative", "neutral", "positive")
    rows = []
    for i, sample in enumerate(samples):
        prediction = int(probabilities[i].argmax())
        rows.append({
            "sample_id": sample.sample_id,
            "source_file": sample.source_file,
            "polarity_code": prediction,
            "polarity": classes[prediction],
            "strength": float(strength[i]),
            "p_negative": float(probabilities[i, 0]),
            "p_neutral": float(probabilities[i, 1]),
            "p_positive": float(probabilities[i, 2]),
            "content_tokens": int(content[i].sum()),
            "audio_zero_rows": int((content[i] & ~observed["audio"][i]).sum()),
            "vision_zero_rows": int((content[i] & ~observed["vision"][i]).sum()),
            "audio_longest_zero_block": longest_run((content[i] & ~observed["audio"][i])[content[i]]),
            "vision_longest_zero_block": longest_run((content[i] & ~observed["vision"][i])[content[i]]),
            "model_version": ("q2_aug_TV_step4_20260924_three_seed_equal_ensemble" if args.model == "TV" else
                              f"{args.candidate_prefix}_three_seed_equal_ensemble" if args.candidate_prefix else
                              "q2_c_aug_1layer_concat_eff_step9_lr3e4_20260924_three_seed_equal_ensemble"),
        })
    with (out / "input_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    with (out / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": ("LOCKED_A3_PREDICTED" if args.model == "TV" else
                   "SAVED_CANDIDATE_A3_PREDICTED" if args.candidate_prefix else
                   "THREE_MODAL_COMPARATOR_A3_PREDICTED"),
        "sample_count": len(rows),
        "bert_sha256": BERT_SHA256,
        "model": f"{run_name(SEEDS[0]).rsplit('_seed_', 1)[0]}, seeds 20260923/20260924/20260925",
        "modalities_used": ("text,vision" if args.model == "TV" else "text,audio,vision"),
        "labels_used": False, "a3_used_for_model_selection": False,
        "prediction_counts": {label: sum(row["polarity"] == label for row in rows) for label in classes},
        "audio_zero_clips": sum(row["audio_zero_rows"] > 0 for row in rows),
        "vision_zero_clips": sum(row["vision_zero_rows"] > 0 for row in rows),
        "note": "A3 has no labels. Probabilities are model scores, not calibrated certainty."
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
