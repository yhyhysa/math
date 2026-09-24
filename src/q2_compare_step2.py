"""Compare paired 30-epoch Q2 model-A baseline and augmented runs."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260923, 20260924, 20260925)
METRICS = ("accuracy", "macro_f1", "mae", "pearson")


def read_run(directory: Path) -> tuple[dict, dict[tuple[str, str, str], dict]]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "scenario_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    scenarios = {(row["modalities"], row["position"], row["ratio"]): row for row in rows}
    if len(rows) != 54 or len(scenarios) != 54:
        raise ValueError(f"Expected 54 unique scenarios: {directory}")
    if summary["train_n"] != 3395 or summary["valid_n"] != 728 or summary["epochs_ran"] != 30:
        raise ValueError(f"Not a complete paired 30-epoch run: {directory}")
    return summary, scenarios


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def average(values: list[float]) -> float:
    return float(statistics.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20260924", help="Tag used by run_q2_augmented.ps1")
    args = parser.parse_args()
    if not args.tag.replace("_", "").replace("-", "").isalnum():
        parser.error("tag must contain only letters, digits, underscores, or hyphens")
    paired = []
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for seed in SEEDS:
        baseline_dir = ROOT / "results" / f"q2_baseline_30ep_seed_{seed}"
        augmented_dir = ROOT / "results" / f"q2_aug_{args.tag}_seed_{seed}"
        base, base_scenarios = read_run(baseline_dir)
        aug, aug_scenarios = read_run(augmented_dir)
        for field in ("seed", "source_sha256", "bert_sha256", "model", "parameter_count", "train_n", "valid_n"):
            if base[field] != aug[field]:
                raise ValueError(f"Paired input mismatch seed={seed} field={field}")
        if base["seed"] != seed or base.get("augment") is True or aug.get("augment") is not True:
            raise ValueError(f"Invalid run mode or seed={seed}")
        if base_scenarios.keys() != aug_scenarios.keys():
            raise ValueError(f"Scenario panel mismatch seed={seed}")
        line = {"seed": seed, "baseline_best_epoch": base["best_epoch"],
                "augmented_best_epoch": aug["best_epoch"]}
        for key in ("selection_J",):
            line[f"baseline_{key}"] = base["best"][key]
            line[f"augmented_{key}"] = aug["best"][key]
            line[f"delta_{key}"] = aug["best"][key] - base["best"][key]
        for view in ("clean", "masked_mean"):
            for metric in METRICS:
                if metric == "pearson" and view == "masked_mean":
                    baseline_value = average([float(row["pearson"]) for row in base_scenarios.values()])
                    augmented_value = average([float(row["pearson"]) for row in aug_scenarios.values()])
                else:
                    baseline_value = base["best"][view][metric]
                    augmented_value = aug["best"][view][metric]
                line[f"baseline_{view}_{metric}"] = baseline_value
                line[f"augmented_{view}_{metric}"] = augmented_value
                line[f"delta_{view}_{metric}"] = augmented_value - baseline_value
        paired.append(line)
        for key in base_scenarios:
            b, a = base_scenarios[key], aug_scenarios[key]
            for field in ("n", "masked_positions", "mean_actual_ratio", "newly_lost_observed_ratio"):
                if b[field] != a[field]:
                    raise ValueError(f"Scenario definition mismatch seed={seed} {key} {field}")
            grouped.setdefault(key, []).append({"seed": seed, "baseline": b, "augmented": a})
    scenario_rows = []
    for key, records in sorted(grouped.items()):
        row = {"modalities": key[0], "position": key[1], "ratio": key[2], "seeds": len(records)}
        for metric in METRICS:
            baseline_mean = average([float(entry["baseline"][metric]) for entry in records])
            augmented_mean = average([float(entry["augmented"][metric]) for entry in records])
            row[f"baseline_{metric}"] = baseline_mean
            row[f"augmented_{metric}"] = augmented_mean
            row[f"delta_{metric}"] = augmented_mean - baseline_mean
        scenario_rows.append(row)
    means = {}
    for field in paired[0]:
        if field.startswith("delta_"):
            values = [float(row[field]) for row in paired]
            means[field] = {"mean": average(values), "sample_sd": statistics.stdev(values)}
    output = ROOT / "results" / f"q2_step2_{args.tag}"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "paired_seed_metrics.csv", paired)
    write_csv(output / "scenario_delta.csv", scenario_rows)
    summary = {"status": "PAIRED_MODEL_A_COMPARISON", "seeds": list(SEEDS),
               "baseline": "q2_baseline_30ep_seed_<seed>",
               "augmented": f"q2_aug_{args.tag}_seed_<seed>",
               "delta_direction": "augmented minus baseline",
               "mean_delta": means,
               "note": "Validation-only comparison; no A2 test or A3 used."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "mean_delta": means}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
