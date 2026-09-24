"""Show where neutral A2-valid clips are misclassified by saved Q2 ensembles."""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import torch

from src.config import load_config
from src.data import load_pickle
from src.q2_compare_step3 import SEEDS, write_csv
from src.q2_compare_tonight import CONFIGS
from src.q2_locked_test_eval import ensemble
from src.q2_train_baseline import DATA_SHA256, metrics, prepare, sha256
from src.q2_train_transformer import SeparateTemporalTransformer, as_view, predict
from src.q2_valid_missing_audit import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CONFIGS))
    parser.add_argument("--output-name", default="q2_tonight_20260924")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_name):
        parser.error("output-name must be a simple directory name")
    cfg = load_config()
    root = cfg["_engine_dir"]
    out = root / "results" / args.output_name
    if (out / "neutral_confusion.csv").exists() or (out / "neutral_summary.json").exists():
        raise FileExistsError("Preserve the first neutral diagnostic")
    source = cfg["_attachment2_root"] / "aligned_50.pkl"
    if sha256(source) != DATA_SHA256:
        raise ValueError("Pinned A2 source hash differs")
    selected = args.candidate or json.loads((root / "results/q2_tonight_20260924/selection.json").read_text(encoding="utf-8"))["selected"]
    candidates = (selected,) if args.candidate or selected == "C_current" else ("C_current", selected)
    valid = prepare(load_pickle(source).splits["valid"])
    if len(valid["class"]) != 728:
        raise ValueError("Expected 728 fixed valid samples")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, summaries = [], {}
    for candidate in candidates:
        short, tag, lr, aux, pattern = CONFIGS[candidate]
        saved = [load_checkpoint(root, f"q2_c_aug_1layer_{short}_eff_{tag}_seed_{seed}", seed)[1]
                 for seed in SEEDS]
        ref = saved[0]
        arrays = {name: arr.copy() for name, arr in valid["array"].items()}
        for name in ("audio", "vision"):
            for checkpoint in saved[1:]:
                if not np.array_equal(checkpoint[f"{name}_mean"], ref[f"{name}_mean"]) or \
                        not np.array_equal(checkpoint[f"{name}_std"], ref[f"{name}_std"]):
                    raise ValueError("Seed normalization differs")
            arrays[name] = ((arrays[name] - ref[f"{name}_mean"]) /
                            ref[f"{name}_std"]) * valid["observed"][name][..., None]
        view = as_view({**valid, "array": arrays}, device)
        outputs = []
        for checkpoint in saved:
            model = SeparateTemporalTransformer(layers=checkpoint["layers"],
                                                fusion=checkpoint["fusion"],
                                                auxiliary=checkpoint.get("auxiliary_weight", 0.0) > 0).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            outputs.append(predict(model, view))
            del model
        probabilities, strengths = ensemble(outputs)
        predictions = probabilities.argmax(axis=1)
        classes = ("negative", "neutral", "positive")
        for truth in range(3):
            for predicted in range(3):
                rows.append({"candidate": candidate, "true_class": classes[truth],
                             "predicted_class": classes[predicted],
                             "count": int(((valid["class"] == truth) & (predictions == predicted)).sum())})
        summaries[candidate] = metrics(valid["class"], valid["value"], probabilities, strengths)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "neutral_confusion.csv", rows)
    (out / "neutral_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "neutral_f1": {key: value["f1_1"] for key, value in summaries.items()}},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
