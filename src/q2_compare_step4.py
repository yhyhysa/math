"""Compare Q2 modality ablations and the fixed empty-position mask on valid only."""
from __future__ import annotations

import argparse
import json
import re
import statistics

from src.q2_compare_step3 import ROOT, SEEDS, read_run, write_csv

CANDIDATES = ("A_T", "A_TA", "A_TV", "A_TAV", "B_old", "B_mask")
METRICS = ("selection_J", "clean_accuracy", "clean_macro_f1", "clean_neutral_f1",
           "clean_mae", "clean_pearson", "masked_accuracy", "masked_macro_f1",
           "masked_mae", "masked_pearson")


def run_name(candidate: str, seed: int, tag: str,
             gate_tag: str | None = None, clean_tag: str | None = None) -> str:
    if candidate == "A_TAV_clean":
        return f"q2_baseline_30ep_seed_{seed}"
    if candidate.endswith("_clean"):
        return f"q2_clean_{candidate[2:-6]}_{clean_tag}_seed_{seed}"
    if candidate == "A_gate":
        return f"q2_aug_gate_{gate_tag}_seed_{seed}"
    if candidate == "A_TAV":
        return f"q2_aug_20260924_seed_{seed}"
    if candidate.startswith("A_"):
        return f"q2_aug_{candidate[2:]}_{tag}_seed_{seed}"
    if candidate == "B_old":
        return f"q2_b_aug_2layer_20260924_seed_{seed}"
    return f"q2_b_aug_2layer_{tag}_seed_{seed}"


def flatten(candidate: str, seed: int, summary: dict, scenes: list[dict]) -> dict:
    best = summary["best"]
    row = {"candidate": candidate, "seed": seed, "best_epoch": summary["best_epoch"],
           "parameter_count": summary["parameter_count"], "selection_J": best["selection_J"],
           "clean_neutral_f1": best["clean"]["f1_1"]}
    for source, prefix in (("clean", "clean"), ("masked_mean", "masked")):
        for metric in ("accuracy", "macro_f1", "mae", "pearson"):
            row[f"{prefix}_{metric}"] = (
                statistics.mean(float(scene["pearson"]) for scene in scenes)
                if source == "masked_mean" and metric == "pearson"
                and metric not in best[source] else best[source][metric]
            )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="step4_20260924")
    parser.add_argument("--gate-tag", help="also compare a coverage-gate run with this tag")
    parser.add_argument("--clean-tag", help="also compare clean-training modality ablations")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", value)
           for value in (args.tag, args.gate_tag, args.clean_tag) if value is not None):
        parser.error("tags must contain only letters, digits, underscores, or hyphens")
    candidates = CANDIDATES + (("A_gate",) if args.gate_tag else ())
    if args.clean_tag:
        candidates += ("A_T_clean", "A_TA_clean", "A_TV_clean", "A_TAV_clean")
    per_seed, groups = [], {}
    reference = None
    for candidate in candidates:
        for seed in SEEDS:
            path = ROOT / "results" / run_name(candidate, seed, args.tag, args.gate_tag, args.clean_tag)
            summary, scenes = read_run(path)
            status = path / "exit_code.txt"
            if candidate != "A_TAV_clean" and (not status.exists() or status.read_text(encoding="utf-8-sig").strip() != "0"):
                raise ValueError(f"Missing or failed exit status: {path}")
            # The original clean TAV baseline predates the explicit augment field.
            if candidate == "A_TAV_clean" and "augment" not in summary:
                command = summary.get("command", "")
                if not command.startswith("python -m src.q2_train_baseline ") or "--augment" in command:
                    raise ValueError(f"Legacy baseline command unexpectedly augmented: {path}")
            augment = summary.get("augment", False) if candidate == "A_TAV_clean" else summary["augment"]
            if summary["seed"] != seed or augment != (not candidate.endswith("_clean")):
                raise ValueError(f"Wrong seed or augmentation: {path}")
            if candidate == "A_gate":
                if (summary["model"] != "A_coverage_gate" or
                        summary.get("modalities") != "TAV" or
                        summary.get("fusion") != "coverage_gate"):
                    raise ValueError(f"Wrong coverage-gate model: {path}")
            elif candidate.startswith("A_"):
                if (summary["model"] != "A_clean_masked_mean_MLP" or
                        summary.get("modalities", "TAV") != candidate[2:].removesuffix("_clean")):
                    raise ValueError(f"Wrong ablation modalities: {path}")
            elif (summary["model"] != "B_gated_temporal_transformer" or
                  summary["layers"] != 2 or
                  summary.get("mask_empty", False) != (candidate == "B_mask")):
                raise ValueError(f"Wrong Transformer mask setting: {path}")
            fingerprint = (summary["source_sha256"], summary["bert_sha256"],
                           tuple((r["modalities"], r["position"], r["ratio"],
                                  r["n"], r["masked_positions"], r["mean_actual_ratio"])
                                 for r in scenes))
            if reference is None:
                reference = fingerprint
            elif fingerprint != reference:
                raise ValueError(f"Different input or validation panel: {path}")
            per_seed.append(flatten(candidate, seed, summary, scenes))
            for scene in scenes:
                group = "text_missing" if "T" in scene["modalities"] else "text_available"
                for key in (scene["modalities"], group):
                    groups.setdefault((candidate, key), []).append(scene)

    means = []
    for candidate in candidates:
        rows = [row for row in per_seed if row["candidate"] == candidate]
        line = {"candidate": candidate, "seeds": len(rows),
                "parameter_count": rows[0]["parameter_count"]}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            line[f"mean_{metric}"] = statistics.mean(values)
            line[f"sd_{metric}"] = statistics.stdev(values)
        means.append(line)

    contrasts = []
    comparisons = (("A_TA", "A_T"), ("A_TV", "A_T"),
                   ("A_TAV", "A_T"), ("B_mask", "B_old"))
    if args.gate_tag:
        comparisons += (("A_gate", "A_TAV"), ("A_gate", "A_T"))
    if args.clean_tag:
        comparisons += (("A_TA_clean", "A_T_clean"), ("A_TV_clean", "A_T_clean"),
                        ("A_TAV_clean", "A_T_clean"))
    for treatment, control in comparisons:
        for seed in SEEDS:
            a = next(row for row in per_seed if row["candidate"] == treatment and row["seed"] == seed)
            b = next(row for row in per_seed if row["candidate"] == control and row["seed"] == seed)
            contrasts.append({"treatment": treatment, "control": control, "seed": seed,
                              **{f"delta_{metric}": float(a[metric]) - float(b[metric])
                                 for metric in METRICS}})

    group_means = []
    for (candidate, group), rows in sorted(groups.items()):
        group_means.append({"candidate": candidate, "missing_group": group,
                            "scenario_seed_rows": len(rows),
                            **{f"mean_{metric}": statistics.mean(float(row[metric]) for row in rows)
                               for metric in ("accuracy", "macro_f1", "mae", "pearson")}})

    output = ROOT / "results" / f"q2_{args.clean_tag or args.gate_tag or args.tag}"
    if output.exists():
        raise FileExistsError(f"Comparison output already exists: {output}")
    output.mkdir()
    write_csv(output / "all_seed_metrics.csv", per_seed)
    write_csv(output / "candidate_means.csv", means)
    write_csv(output / "paired_deltas.csv", contrasts)
    write_csv(output / "missing_group_means.csv", group_means)
    report = {"status": "VALIDATION_ABLATION_ONLY", "seeds": SEEDS,
              "candidates": candidates, "source_sha256": reference[0],
              "bert_sha256": reference[1], "scenario_fingerprint_verified": True,
              "note": "A2 test, A3, A4, Q1 features and outside sentiment data unused."}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "candidate_J":
                      {row["candidate"]: row["mean_selection_J"] for row in means}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
