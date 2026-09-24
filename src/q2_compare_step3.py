"""Compare the fixed three-seed Q2 model-A and model-B validation experiments."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260923, 20260924, 20260925)
METRICS = ("selection_J", "clean_accuracy", "clean_macro_f1", "clean_f1_1",
           "clean_mae", "clean_pearson", "masked_accuracy", "masked_macro_f1",
           "masked_mae", "masked_pearson")


def run_path(candidate: str, seed: int, tag: str) -> Path:
    if candidate == "A_clean":
        name = f"q2_baseline_30ep_seed_{seed}"
    elif candidate == "A_aug":
        name = f"q2_aug_20260924_seed_{seed}"
    else:
        family, mode = candidate.split("_")
        name = f"q2_b_{mode}_{family[1:]}layer_{tag}_seed_{seed}"
    return ROOT / "results" / name


def read_run(path: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    with (path / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
        history = list(csv.DictReader(stream))
    with (path / "scenario_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        scenarios = list(csv.DictReader(stream))
    if len(history) != 30 or len(scenarios) != 54 or summary["epochs_ran"] != 30:
        raise ValueError(f"Incomplete 30-epoch / 54-scenario run: {path}")
    if len({(row["modalities"], row["position"], row["ratio"]) for row in scenarios}) != 54:
        raise ValueError(f"Repeated validation scenario: {path}")
    if summary["train_n"] != 3395 or summary["valid_n"] != 728 or summary["scenarios"] != 54:
        raise ValueError(f"Wrong train/valid size: {path}")
    if summary["max_epochs"] != 30 or summary["patience"] != 30:
        raise ValueError(f"Unequal training budget: {path}")
    if not Path(summary["checkpoint"]).exists():
        raise ValueError(f"Missing best checkpoint: {path}")
    status = path / "exit_code.txt"
    if status.exists() and status.read_text(encoding="utf-8-sig").strip() != "0":
        raise ValueError(f"Failed run: {path}")
    return summary, scenarios


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20260924", help="Tag used by run_q2_transformer.ps1")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("tag must contain only letters, digits, underscores, or hyphens")
    candidates = ("A_clean", "A_aug", "B1_clean", "B1_aug", "B2_clean", "B2_aug")
    reference = None
    per_seed = []
    scene_records: dict[tuple[str, str, str], dict[str, list[dict]]] = {}
    for candidate in candidates:
        for seed in SEEDS:
            path = run_path(candidate, seed, args.tag)
            summary, scenarios = read_run(path)
            if summary["seed"] != seed:
                raise ValueError(f"Wrong seed: {path}")
            if candidate.startswith("B"):
                expected_layers = int(candidate[1])
                expected_aug = candidate.endswith("_aug")
                if (summary["model"] != "B_gated_temporal_transformer" or
                        summary["layers"] != expected_layers or
                        summary["augment"] != expected_aug):
                    raise ValueError(f"Wrong model configuration: {path}")
            else:
                if summary["model"] != "A_clean_masked_mean_MLP":
                    raise ValueError(f"Wrong model A: {path}")
            fingerprint = (
                summary["source_sha256"], summary["bert_sha256"],
                tuple((r["modalities"], r["position"], r["ratio"], r["n"],
                       r["masked_positions"], r["mean_actual_ratio"],
                       r["newly_lost_observed_ratio"]) for r in scenarios),
            )
            if reference is None:
                reference = fingerprint
            elif fingerprint != reference:
                raise ValueError(f"Input or validation panel differs: {path}")
            for case in scenarios:
                key = (case["modalities"], case["position"], case["ratio"])
                scene_records.setdefault(key, {}).setdefault(candidate, []).append(case)
            best = summary["best"]
            row = {"candidate": candidate, "seed": seed, "best_epoch": summary["best_epoch"],
                   "parameter_count": summary["parameter_count"],
                   "seconds": summary["seconds"], "selection_J": best["selection_J"]}
            for prefix, key in (("clean", "clean"), ("masked", "masked_mean")):
                for metric in ("accuracy", "macro_f1", "mae", "pearson"):
                    row[f"{prefix}_{metric}"] = (
                        statistics.mean(float(case["pearson"]) for case in scenarios)
                        if prefix == "masked" and metric == "pearson" else best[key][metric]
                    )
            row["clean_f1_0"] = best["clean"]["f1_0"]
            row["clean_f1_1"] = best["clean"]["f1_1"]
            row["clean_f1_2"] = best["clean"]["f1_2"]
            row["clean_sign_conflict"] = best["clean"]["sign_conflict"]
            per_seed.append(row)

    means = []
    for candidate in candidates:
        records = [row for row in per_seed if row["candidate"] == candidate]
        line = {"candidate": candidate, "seeds": len(records),
                "parameter_count": records[0]["parameter_count"]}
        for metric in METRICS:
            values = [row[metric] for row in records]
            line[f"mean_{metric}"] = statistics.mean(values)
            line[f"sd_{metric}"] = statistics.stdev(values)
        means.append(line)
    ranked = sorted(means, key=lambda row: (
        row["mean_selection_J"], row["mean_masked_mae"],
        row["parameter_count"], row["candidate"],
    ))
    best = min(
        (row for row in ranked if row["mean_selection_J"] <= ranked[0]["mean_selection_J"] + 1e-4),
        key=lambda row: (row["mean_masked_mae"], row["parameter_count"], row["candidate"]),
    )
    output = ROOT / "results" / f"q2_step3_{args.tag}"
    output.mkdir(exist_ok=True)
    write_csv(output / "all_seed_metrics.csv", per_seed)
    write_csv(output / "candidate_means.csv", ranked)
    scene_means = []
    for key, by_candidate in sorted(scene_records.items()):
        if any(len(by_candidate.get(candidate, [])) != len(SEEDS) for candidate in candidates):
            raise ValueError(f"Incomplete scenario seed group: {key}")
        reference_case = by_candidate["A_aug"][0]
        line = {"modalities": key[0], "position": key[1], "ratio": key[2],
                "n": reference_case["n"], "masked_positions": reference_case["masked_positions"],
                "mean_actual_ratio": reference_case["mean_actual_ratio"],
                "newly_lost_observed_ratio": reference_case["newly_lost_observed_ratio"]}
        for candidate in candidates:
            for metric in ("accuracy", "macro_f1", "mae", "pearson"):
                line[f"{candidate}_{metric}"] = statistics.mean(
                    float(case[metric]) for case in by_candidate[candidate]
                )
        for metric in ("accuracy", "macro_f1", "mae", "pearson"):
            line[f"delta_B2_aug_minus_A_aug_{metric}"] = (
                line[f"B2_aug_{metric}"] - line[f"A_aug_{metric}"]
            )
        scene_means.append(line)
    write_csv(output / "scenario_candidate_means.csv", scene_means)
    result = {"status": "MODEL_A_B_VALIDATION_COMPARISON", "seeds": SEEDS,
              "model_B_tag": args.tag,
              "ranked_by_J": [row["candidate"] for row in ranked],
              "best_candidate": best["candidate"],
              "scenario_fingerprint_verified": True,
              "note": "Validation-only selection. No A2 test, A3, A4 or Q1 features used."}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "best_candidate": best["candidate"],
                      "best_mean_J": best["mean_selection_J"]}), flush=True)


if __name__ == "__main__":
    main()
