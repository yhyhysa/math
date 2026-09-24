"""One-off locked A2 test evaluation; never trains or selects a model on test."""
from __future__ import annotations

import json
import statistics

import numpy as np
import torch
from transformers import BertModel

from src.config import load_config
from src.data import load_pickle
from src.q2_compare_step3 import write_csv
from src.q2_train_baseline import (
    BERT_REV, BERT_SHA256, DATA_SHA256, Baseline, create_panel,
    metrics, pooled, predict as predict_mlp, prepare, sha256,
)
from src.q2_train_transformer import (
    SeparateTemporalTransformer, as_view, make_panel, predict as predict_temporal,
)
from src.q2_valid_missing_audit import SEEDS, load_checkpoint


def probabilities(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def ensemble(outputs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    return (np.mean([probabilities(logits) for logits, _ in outputs], axis=0),
            np.mean([values for _, values in outputs], axis=0))


def score(data: dict, group: np.ndarray, clean: tuple[np.ndarray, np.ndarray],
          scenes: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    y_class, y_value = data["class"][group], data["value"][group]
    base = metrics(y_class, y_value, clean[0][group], clean[1][group])
    masked = [metrics(y_class, y_value, logits[group], estimate[group]) for logits, estimate in scenes]
    mean = {key: statistics.mean(row[key] for row in masked)
            for key in ("accuracy", "macro_f1", "mae")}
    correlations = [row["pearson"] for row in masked if row["pearson"] is not None]
    mean["pearson"] = statistics.mean(correlations) if correlations else None
    j = .25 * (base["mae"] / 3 + 1 - base["macro_f1"]) + \
        .75 * (mean["mae"] / 3 + 1 - mean["macro_f1"])
    return {"n": int(group.sum()), "J_descriptive": j,
            **{f"clean_{key}": base[key] for key in
               ("accuracy", "macro_f1", "f1_0", "f1_1", "f1_2", "mae", "pearson")},
            **{f"masked_{key}": mean[key] for key in
               ("accuracy", "macro_f1", "mae", "pearson")}}


def main() -> None:
    cfg = load_config()
    root = cfg["_engine_dir"]
    out = root / "results" / "q2_locked_test_20260924"
    if not (out / "模型锁定与测试协议.md").exists():
        raise FileNotFoundError("The pre-test model lock is missing")
    for name in ("test_model_metrics.csv", "test_scenario_metrics.csv", "test_summary.json"):
        if (out / name).exists():
            raise FileExistsError(f"Test was already evaluated; preserve the first result: {out / name}")
    audit = json.loads((root / "results" / "q2_valid_missing_audit_20260924_v2" /
                        "summary.json").read_text(encoding="utf-8"))
    if audit["status"] != "MODEL_REPRODUCTION_PASS":
        raise ValueError("The valid reproduction check did not pass")
    source = cfg["_attachment2_root"] / "aligned_50.pkl"
    bert_dir = root / "models" / "bert-base-uncased" / BERT_REV
    if sha256(source) != DATA_SHA256 or sha256(bert_dir / "model.safetensors") != BERT_SHA256:
        raise ValueError("Pinned A2 or BERT hash mismatch")
    names = {
        "TV_primary": lambda seed: f"q2_aug_TV_step4_20260924_seed_{seed}",
        "C_fixed_comparator": lambda seed: f"q2_c_aug_1layer_concat_eff_step9_lr3e4_20260924_seed_{seed}",
    }
    saved = {(candidate, seed): load_checkpoint(root, make_name(seed), seed)
             for candidate, make_name in names.items() for seed in SEEDS}
    reference = saved[("TV_primary", SEEDS[0])][1]
    for _, checkpoint in saved.values():
        for name in ("audio", "vision"):
            if not np.array_equal(checkpoint[f"{name}_mean"], reference[f"{name}_mean"]) or \
                    not np.array_equal(checkpoint[f"{name}_std"], reference[f"{name}_std"]):
                raise ValueError("Training normalization differs among locked checkpoints")
    print("Loading locked A2 test once...", flush=True)
    payload = load_pickle(source)
    test = prepare(payload.splits["test"])
    if len(test["class"]) != 727:
        raise ValueError("Unexpected A2 test sample count")
    natural_audio = (test["content"] & ~test["observed"]["audio"]).any(axis=1)
    natural_vision = (test["content"] & ~test["observed"]["vision"]).any(axis=1)
    natural_any = natural_audio | natural_vision
    groups = {"all": np.ones(len(test["class"]), dtype=bool),
              "no_natural_zero": ~natural_any, "natural_zero": natural_any}
    for name in ("audio", "vision"):
        test["array"][name] = ((test["array"][name] - reference[f"{name}_mean"]) /
                               reference[f"{name}_std"])
        test["array"][name] *= test["observed"][name][..., None]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, scenario_rows = [], []
    print("Preparing locked TV test views...", flush=True)
    tv_clean = torch.as_tensor(pooled(test, keep_modalities="TV"), device=device)
    tv_panel_array, descriptors = create_panel(test, bert_dir, device, keep_modalities="TV")
    tv_panel = torch.as_tensor(tv_panel_array, device=device)
    predictions = []
    n = len(test["class"])
    for seed in SEEDS:
        checkpoint = saved[("TV_primary", seed)][1]
        if checkpoint["modalities"] != "TV" or checkpoint.get("fusion", "concat") != "concat":
            raise ValueError("Locked TV checkpoint configuration differs")
        model = Baseline(checkpoint["input_dim"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        clean = predict_mlp(model, tv_clean)
        logits, values = predict_mlp(model, tv_panel)
        scenes = [(logits[i*n:(i+1)*n], values[i*n:(i+1)*n]) for i in range(54)]
        predictions.append((clean, scenes))
        for cohort, mask in groups.items():
            if mask.any():
                rows.append({"candidate": "TV_primary", "seed": seed, "cohort": cohort,
                             **score(test, mask, clean, scenes)})
        print(f"TV seed {seed} evaluated", flush=True)
        del model
    tv_ensemble = (ensemble([p[0] for p in predictions]),
                   [ensemble([p[1][i] for p in predictions]) for i in range(54)])
    for cohort, mask in groups.items():
        if mask.any():
            rows.append({"candidate": "TV_primary", "seed": "ensemble", "cohort": cohort,
                         **score(test, mask, *tv_ensemble)})
    shared_fields = ("modalities", "position", "ratio", "n", "masked_positions",
                     "mean_actual_ratio", "newly_lost_observed_ratio")
    for descriptor, (logits, values) in zip(descriptors, tv_ensemble[1]):
        scenario_rows.append({"candidate": "TV_primary",
                              **{key: descriptor[key] for key in shared_fields},
                              **metrics(test["class"], test["value"], logits, values)})
    del predictions, tv_panel, tv_panel_array, tv_clean
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Preparing locked three-modal test views...", flush=True)
    temporal_clean = as_view(test, device)
    bert = BertModel.from_pretrained(bert_dir, local_files_only=True).to(device).eval()
    bert.requires_grad_(False)
    temporal_panel = make_panel(test, temporal_clean, bert, device)
    del bert
    if [(d["modalities"], d["position"], d["ratio"]) for d, _, _ in temporal_panel] != \
            [(d["modalities"], d["position"], d["ratio"]) for d in descriptors]:
        raise ValueError("Locked models have different 54-scenario ordering")
    predictions = []
    for seed in SEEDS:
        checkpoint = saved[("C_fixed_comparator", seed)][1]
        if checkpoint["architecture"] != "separate" or checkpoint["fusion"] != "concat" or \
                checkpoint["learning_rate"] != 3e-4:
            raise ValueError("Locked comparator checkpoint configuration differs")
        model = SeparateTemporalTransformer(layers=1, fusion="concat").to(device)
        model.load_state_dict(checkpoint["state_dict"])
        clean = predict_temporal(model, temporal_clean)
        scenes = [predict_temporal(model, {**temporal_clean, "available": available, "text": text})
                  for _, available, text in temporal_panel]
        predictions.append((clean, scenes))
        for cohort, mask in groups.items():
            if mask.any():
                rows.append({"candidate": "C_fixed_comparator", "seed": seed, "cohort": cohort,
                             **score(test, mask, clean, scenes)})
        print(f"Comparator seed {seed} evaluated", flush=True)
        del model
    c_ensemble = (ensemble([p[0] for p in predictions]),
                  [ensemble([p[1][i] for p in predictions]) for i in range(54)])
    for cohort, mask in groups.items():
        if mask.any():
            rows.append({"candidate": "C_fixed_comparator", "seed": "ensemble", "cohort": cohort,
                         **score(test, mask, *c_ensemble)})
    for (descriptor, _, _), (logits, values) in zip(temporal_panel, c_ensemble[1]):
        scenario_rows.append({"candidate": "C_fixed_comparator",
                              **{key: descriptor[key] for key in shared_fields},
                              **metrics(test["class"], test["value"], logits, values)})
    write_csv(out / "test_model_metrics.csv", rows)
    write_csv(out / "test_scenario_metrics.csv", scenario_rows)
    summary = {"status": "LOCKED_TEST_EVALUATED", "source_sha256": DATA_SHA256,
               "bert_sha256": BERT_SHA256, "test_n": n,
               "natural_audio_clips": int(natural_audio.sum()),
               "natural_vision_clips": int(natural_vision.sum()),
               "natural_zero_clips": int(natural_any.sum()),
               "primary": "TV_primary three-seed equal-probability/strength ensemble",
               "comparator": "C_fixed_comparator three-seed equal-probability/strength ensemble",
               "model_switch_or_test_tuning": False,
               "note": "Test labels used once for locked evaluation only. A3/A4/Q1 unused."}
    (out / "test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "test_n": n,
                      "ensemble_all": [row for row in rows if row["seed"] == "ensemble" and row["cohort"] == "all"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
