"""Read-only A2 valid audit: natural zero rows, effective masks, and saved-model scores."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel

from src.config import load_config
from src.data import load_pickle
from src.q2_compare_step3 import write_csv
from src.q2_train_baseline import (
    BERT_REV, BERT_SHA256, DATA_SHA256, MODALITY_SETS, POSITIONS, RATIOS,
    Baseline, create_panel, interval_mask, metrics, pooled, predict as predict_mlp,
    prepare, sha256,
)
from src.q2_train_transformer import (
    SeparateTemporalTransformer, as_view, make_panel, predict as predict_temporal,
)


SEEDS = (20260923, 20260924, 20260925)
SHORT = {"T": "text", "A": "audio", "V": "vision"}


def load_checkpoint(root: Path, name: str, seed: int) -> tuple[dict, dict]:
    result = root / "results" / name
    if (result / "exit_code.txt").read_text(encoding="utf-8-sig").strip() != "0":
        raise ValueError(f"Training did not finish: {name}")
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    if summary["seed"] != seed or summary["source_sha256"] != DATA_SHA256 or summary["bert_sha256"] != BERT_SHA256:
        raise ValueError(f"Wrong seed or input hash: {name}")
    checkpoint = torch.load(root / "checkpoints" / name / "best.pt", map_location="cpu", weights_only=False)
    if checkpoint["seed"] != seed or checkpoint["data_sha256"] != DATA_SHA256 or checkpoint["bert_sha256"] != BERT_SHA256:
        raise ValueError(f"Checkpoint metadata differs: {name}")
    return summary, checkpoint


def scene_loss_rows(valid: dict, cohorts: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for position in POSITIONS:
        for ratio in RATIOS:
            missing = np.stack([interval_mask(c, position, ratio) for c in valid["content"]])
            nominal = missing.sum(axis=1) / valid["content"].sum(axis=1)
            for group in MODALITY_SETS:
                observed = [valid["observed"][SHORT[short]] for short in group]
                available = sum(x.sum(axis=1) for x in observed)
                lost = sum((x & missing).sum(axis=1) for x in observed)
                targeted = missing.sum(axis=1) * len(group)
                tv_selected = [short for short in group if short in "TV"]
                tv_lost = sum(((valid["observed"][SHORT[short]] & missing).sum(axis=1)
                               for short in tv_selected), np.zeros(len(valid["content"]), dtype=np.int64))
                for cohort, selected in cohorts.items():
                    rows.append({
                        "modalities": group, "position": position, "ratio": ratio,
                        "cohort": cohort, "n": int(selected.sum()),
                        "nominal_token_ratio": float(nominal[selected].mean()),
                        "nominal_weighted_ratio": float(missing[selected].sum() / valid["content"][selected].sum()),
                        "actual_newly_lost_content_ratio": float(lost[selected].sum() /
                            (valid["content"][selected].sum() * len(group))),
                        "actual_newly_lost_ratio": float(lost[selected].sum() / available[selected].sum()),
                        "effective_target_fraction": float(lost[selected].sum() / targeted[selected].sum()),
                        "targeted_rows_already_zero": int((targeted[selected] - lost[selected]).sum()),
                        "no_new_loss_clips": int((lost[selected] == 0).sum()),
                        "tv_no_new_loss_clips": int((tv_lost[selected] == 0).sum()),
                        "mean_newly_lost_rows": float(lost[selected].mean()),
                    })
    return rows


def cohort_scores(valid: dict, cohorts: dict[str, np.ndarray], clean: tuple[np.ndarray, np.ndarray],
                  scenes: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    rows = []
    for cohort, selected in cohorts.items():
        labels, values = valid["class"][selected], valid["value"][selected]
        base = metrics(labels, values, clean[0][selected], clean[1][selected])
        masked = [metrics(labels, values, logits[selected], estimate[selected])
                  for logits, estimate in scenes]
        masked_f1 = float(np.mean([m["macro_f1"] for m in masked]))
        masked_mae = float(np.mean([m["mae"] for m in masked]))
        j = .25 * (base["mae"] / 3 + 1 - base["macro_f1"]) + \
            .75 * (masked_mae / 3 + 1 - masked_f1)
        rows.append({
            "cohort": cohort, "n": len(labels), "negative": int((labels == 0).sum()),
            "neutral": int((labels == 1).sum()), "positive": int((labels == 2).sum()),
            "selection_J": j, "clean_accuracy": base["accuracy"],
            "clean_macro_f1": base["macro_f1"], "clean_mae": base["mae"],
            "masked_macro_f1": masked_f1, "masked_mae": masked_mae,
        })
    return rows


def verify_previous(summary: dict, result_dir: Path, valid: dict,
                    clean: tuple[np.ndarray, np.ndarray], scenes: list[tuple[np.ndarray, np.ndarray]]) -> float:
    all_rows = cohort_scores(valid, {"all": np.ones(len(valid["class"]), dtype=bool)}, clean, scenes)[0]
    prior = summary["best"]
    checks = (
        (all_rows["selection_J"], prior["selection_J"]),
        (all_rows["clean_macro_f1"], prior["clean"]["macro_f1"]),
        (all_rows["clean_mae"], prior["clean"]["mae"]),
        (all_rows["masked_macro_f1"], prior["masked_mean"]["macro_f1"]),
        (all_rows["masked_mae"], prior["masked_mean"]["mae"]),
    )
    with (result_dir / "scenario_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        recorded = list(csv.DictReader(stream))
    if len(recorded) != len(scenes):
        raise ValueError(f"Scenario count differs: {result_dir}")
    labels, values = valid["class"], valid["value"]
    for (logits, estimate), row in zip(scenes, recorded):
        m = metrics(labels, values, logits, estimate)
        checks += tuple((m[key], float(row[key])) for key in ("accuracy", "macro_f1", "mae"))
    error = max(abs(float(a) - float(b)) for a, b in checks)
    if error > 1e-4:
        raise ValueError(f"Saved-model reproduction differs by {error:.6g}: {result_dir}")
    return error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20260924")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("tag must contain only letters, digits, underscores, or hyphens")
    cfg = load_config()
    root = cfg["_engine_dir"]
    output = root / "results" / f"q2_valid_missing_audit_{args.tag}"
    if output.exists():
        raise FileExistsError(f"Use a new --tag, preserving earlier results: {output}")
    source = cfg["_attachment2_root"] / "aligned_50.pkl"
    bert_dir = root / "models" / "bert-base-uncased" / BERT_REV
    if sha256(source) != DATA_SHA256 or sha256(bert_dir / "model.safetensors") != BERT_SHA256:
        raise ValueError("Pinned attachment or BERT hash differs")
    payload = load_pickle(source)
    valid = prepare(payload.splits["valid"])
    content = valid["content"]
    natural_audio = (content & ~valid["observed"]["audio"]).any(axis=1)
    natural_vision = (content & ~valid["observed"]["vision"]).any(axis=1)
    natural_any = natural_audio | natural_vision
    cohorts = {"all": np.ones(len(content), dtype=bool),
               "no_natural_zero": ~natural_any, "natural_zero": natural_any}
    if not all(mask.any() for mask in cohorts.values()):
        raise ValueError("An audit cohort is empty")
    loss_rows = scene_loss_rows(valid, cohorts)
    for name in ("audio", "vision"):
        valid["array"][name] = valid["array"][name].copy()
    names = {
        "TV_old": lambda seed: f"q2_aug_TV_step4_20260924_seed_{seed}",
        "C_low_lr": lambda seed: f"q2_c_aug_1layer_concat_eff_step9_lr3e4_20260924_seed_{seed}",
    }
    saved = {(candidate, seed): load_checkpoint(root, make_name(seed), seed)
             for candidate, make_name in names.items() for seed in SEEDS}
    reference = saved[("TV_old", SEEDS[0])][1]
    for _, checkpoint in saved.values():
        for name in ("audio", "vision"):
            if not np.array_equal(checkpoint[f"{name}_mean"], reference[f"{name}_mean"]) or \
                    not np.array_equal(checkpoint[f"{name}_std"], reference[f"{name}_std"]):
                raise ValueError("Different train normalization; separate normalization is required")
            if not np.isfinite(checkpoint[f"{name}_mean"]).all():
                raise ValueError("Nonfinite normalization statistics")
    for name in ("audio", "vision"):
        valid["array"][name] = ((valid["array"][name] - reference[f"{name}_mean"]) /
                                reference[f"{name}_std"])
        valid["array"][name] *= valid["observed"][name][..., None]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, errors = [], {}
    print("Preparing 54 TV pooled validation views...", flush=True)
    tv_clean_array = pooled(valid, keep_modalities="TV")
    tv_clean = torch.as_tensor(tv_clean_array, device=device)
    tv_panel_array, tv_descriptors = create_panel(valid, bert_dir, device, keep_modalities="TV")
    tv_panel = torch.as_tensor(tv_panel_array, device=device)
    if [(r["modalities"], r["position"], r["ratio"]) for r in tv_descriptors] != \
            [(r["modalities"], r["position"], r["ratio"]) for r in loss_rows if r["cohort"] == "all"]:
        raise ValueError("TV panel order differs from loss audit")
    n = len(content)
    for i, descriptor in enumerate(tv_descriptors):
        if descriptor["modalities"] == "A" and not np.array_equal(
                tv_panel_array[i*n:(i+1)*n], tv_clean_array):
            raise ValueError("A-only masking changed a TV model input")
    audio_only_prediction_error = {}
    for seed in SEEDS:
        name = names["TV_old"](seed)
        summary, checkpoint = saved[("TV_old", seed)]
        if checkpoint["modalities"] != "TV" or checkpoint.get("fusion", "concat") != "concat":
            raise ValueError(f"Wrong TV checkpoint: {name}")
        model = Baseline(checkpoint["input_dim"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        clean = predict_mlp(model, tv_clean)
        logits, estimate = predict_mlp(model, tv_panel)
        scenes = [(logits[i*n:(i+1)*n], estimate[i*n:(i+1)*n]) for i in range(54)]
        max_error = 0.0
        for (descriptor, (scene_logits, scene_values)) in zip(tv_descriptors, scenes):
            if descriptor["modalities"] == "A":
                max_error = max(max_error, float(np.max(np.abs(scene_logits - clean[0]))),
                                float(np.max(np.abs(scene_values - clean[1]))))
        if max_error > 1e-5:
            raise ValueError(f"A-only masking changed TV predictions beyond numerical tolerance: {max_error}")
        audio_only_prediction_error[name] = max_error
        errors[name] = verify_previous(summary, root / "results" / name, valid, clean, scenes)
        rows += [{"candidate": "TV_old", "seed": seed, **row}
                 for row in cohort_scores(valid, cohorts, clean, scenes)]
        print(f"Reproduced {name}; max metric error={errors[name]:.2g}", flush=True)
        del model
    del tv_panel, tv_panel_array, tv_clean
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Preparing 54 temporal validation views...", flush=True)
    temporal_clean = as_view(valid, device)
    bert = BertModel.from_pretrained(bert_dir, local_files_only=True).to(device).eval()
    bert.requires_grad_(False)
    temporal_panel = make_panel(valid, temporal_clean, bert, device)
    del bert
    if [(r["modalities"], r["position"], r["ratio"]) for r, _, _ in temporal_panel] != \
            [(r["modalities"], r["position"], r["ratio"]) for r in loss_rows if r["cohort"] == "all"]:
        raise ValueError("Temporal panel order differs from loss audit")
    for seed in SEEDS:
        name = names["C_low_lr"](seed)
        summary, checkpoint = saved[("C_low_lr", seed)]
        if checkpoint["architecture"] != "separate" or checkpoint["fusion"] != "concat" or \
                checkpoint["learning_rate"] != 3e-4:
            raise ValueError(f"Wrong temporal checkpoint: {name}")
        model = SeparateTemporalTransformer(layers=1, fusion="concat").to(device)
        model.load_state_dict(checkpoint["state_dict"])
        clean = predict_temporal(model, temporal_clean)
        scenes = [predict_temporal(model, {**temporal_clean, "available": available, "text": text})
                  for _, available, text in temporal_panel]
        errors[name] = verify_previous(summary, root / "results" / name, valid, clean, scenes)
        rows += [{"candidate": "C_low_lr", "seed": seed, **row}
                 for row in cohort_scores(valid, cohorts, clean, scenes)]
        print(f"Reproduced {name}; max metric error={errors[name]:.2g}", flush=True)
        del model
    means = []
    fields = ("selection_J", "clean_accuracy", "clean_macro_f1", "clean_mae",
              "masked_macro_f1", "masked_mae")
    for candidate in names:
        for cohort in cohorts:
            subset = [row for row in rows if row["candidate"] == candidate and row["cohort"] == cohort]
            means.append({"candidate": candidate, "cohort": cohort, "n": subset[0]["n"],
                          **{field: statistics.mean(float(row[field]) for row in subset) for field in fields}})
    group_means = []
    for group in MODALITY_SETS:
        subset = [row for row in loss_rows if row["modalities"] == group and row["cohort"] == "all"]
        group_means.append({"modalities": group,
                            "nominal_token_ratio": statistics.mean(row["nominal_token_ratio"] for row in subset),
                            "nominal_weighted_ratio": statistics.mean(row["nominal_weighted_ratio"] for row in subset),
                            "actual_newly_lost_content_ratio": statistics.mean(
                                row["actual_newly_lost_content_ratio"] for row in subset),
                            "actual_newly_lost_ratio": statistics.mean(row["actual_newly_lost_ratio"] for row in subset),
                            "effective_target_fraction": statistics.mean(row["effective_target_fraction"] for row in subset),
                            "no_new_loss_clips_per_scene": statistics.mean(row["no_new_loss_clips"] for row in subset),
                            "max_no_new_loss_clips": max(row["no_new_loss_clips"] for row in subset),
                            "tv_no_new_loss_clips_per_scene": statistics.mean(row["tv_no_new_loss_clips"] for row in subset)})
    output.mkdir()
    write_csv(output / "scene_actual_loss.csv", loss_rows)
    write_csv(output / "loss_group_means.csv", group_means)
    write_csv(output / "cohort_seed_metrics.csv", rows)
    write_csv(output / "cohort_means.csv", means)
    metadata = {"status": "MODEL_REPRODUCTION_PASS", "source_sha256": DATA_SHA256, "bert_sha256": BERT_SHA256,
                "samples": len(content), "content_positions": int(content.sum()),
                "natural_audio_clips": int(natural_audio.sum()),
                "natural_vision_clips": int(natural_vision.sum()),
                "natural_zero_clips": int(natural_any.sum()),
                "natural_audio_zero_rows": int((content & ~valid["observed"]["audio"]).sum()),
                "natural_vision_zero_rows": int((content & ~valid["observed"]["vision"]).sum()),
                "tv_audio_only_prediction_max_abs_error": audio_only_prediction_error,
                "reproduction_max_metric_error": errors,
                "note": "A2 valid only. Zero row means suspected unobserved, not proven deliberate missing. A2 test/A3 unused."}
    (output / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **metadata}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
