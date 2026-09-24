"""Compare Q2 effective TV masking and separate temporal models on A2 valid."""
from __future__ import annotations

import argparse
import json
import re
import statistics

from src.q2_compare_step3 import ROOT, SEEDS, read_run, write_csv
from src.q2_compare_step4 import METRICS, flatten


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("tv", "all", "lr", "consistency"), required=True)
    parser.add_argument("--tv-tag", default="step7_20260924")
    parser.add_argument("--model-tag", default="step8_20260924")
    parser.add_argument("--lr-tag", default="step9_lr3e4_20260924")
    parser.add_argument("--cons-tag", default="step10_cons05_20260924")
    parser.add_argument("--output-tag", default="20260924")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", value)
           for value in (args.tv_tag, args.model_tag, args.lr_tag, args.cons_tag, args.output_tag)):
        parser.error("tags must contain only letters, digits, underscores, or hyphens")
    names = {
        "TV_old": lambda seed: f"q2_aug_TV_step4_20260924_seed_{seed}",
        "TV_effective": lambda seed: f"q2_aug_TV_eff_{args.tv_tag}_seed_{seed}",
        "TV_clean": lambda seed: f"q2_clean_TV_step6_20260924_seed_{seed}",
        "TAV_MLP": lambda seed: f"q2_aug_20260924_seed_{seed}",
        "B_early": lambda seed: f"q2_b_aug_2layer_step4_20260924_seed_{seed}",
    }
    if args.stage in ("all", "lr", "consistency"):
        names.update({
            "C_concat": lambda seed: f"q2_c_aug_1layer_concat_eff_{args.model_tag}_seed_{seed}",
        })
    if args.stage == "all":
        names.update({
            "C_gate": lambda seed: f"q2_c_aug_1layer_gate_eff_{args.model_tag}_seed_{seed}",
        })
    if args.stage in ("lr", "consistency"):
        names.update({
            "C_concat_lr": lambda seed: f"q2_c_aug_1layer_concat_eff_{args.lr_tag}_seed_{seed}",
        })
    if args.stage == "consistency":
        names.update({
            "C_concat_cons": lambda seed: f"q2_c_aug_1layer_concat_eff_{args.cons_tag}_seed_{seed}",
        })
    per_seed, groups = [], {}
    reference = None
    for candidate, make_name in names.items():
        for seed in SEEDS:
            path = ROOT / "results" / make_name(seed)
            summary, scenes = read_run(path)
            status = path / "exit_code.txt"
            if not status.exists() or status.read_text(encoding="utf-8-sig").strip() != "0":
                raise ValueError(f"Missing or failed exit status: {path}")
            if summary["seed"] != seed:
                raise ValueError(f"Wrong seed: {path}")
            if candidate.startswith("C_"):
                fusion = "reliability_gate" if candidate == "C_gate" else "concat"
                expected_lr = 3e-4 if candidate in ("C_concat_lr", "C_concat_cons") else 1e-3
                expected_consistency = 0.5 if candidate == "C_concat_cons" else 0.0
                if (summary["model"] != "C_separate_temporal_transformer" or
                        summary.get("architecture") != "separate" or
                        summary.get("fusion") != fusion or
                        summary.get("learning_rate", 1e-3) != expected_lr or
                        summary.get("consistency_weight", 0.0) != expected_consistency or
                        summary.get("effective_mask") is not True or
                        summary.get("layers") != 1 or summary.get("augment") is not True):
                    raise ValueError(f"Wrong separate model configuration: {path}")
            elif candidate.startswith("TV_"):
                if (summary["model"] != "A_clean_masked_mean_MLP" or
                        summary.get("modalities") != "TV" or
                        summary.get("augment") != (candidate != "TV_clean") or
                        summary.get("effective_mask", False) != (candidate == "TV_effective")):
                    raise ValueError(f"Wrong TV ablation configuration: {path}")
            elif candidate == "TAV_MLP":
                if summary["model"] != "A_clean_masked_mean_MLP" or not summary["augment"]:
                    raise ValueError(f"Wrong TAV MLP: {path}")
            elif (summary["model"] != "B_gated_temporal_transformer" or
                  summary.get("layers") != 2 or summary.get("mask_empty") is not True):
                raise ValueError(f"Wrong early Transformer: {path}")
            fingerprint = (summary["source_sha256"], summary["bert_sha256"],
                           tuple((r["modalities"], r["position"], r["ratio"], r["n"],
                                  r["masked_positions"], r["mean_actual_ratio"]) for r in scenes))
            if reference is None:
                reference = fingerprint
            elif fingerprint != reference:
                raise ValueError(f"Different source or validation panel: {path}")
            per_seed.append(flatten(candidate, seed, summary, scenes))
            for scene in scenes:
                group = "text_missing" if "T" in scene["modalities"] else "text_available"
                for key in (scene["modalities"], group):
                    groups.setdefault((candidate, key), []).append(scene)
    means = []
    for candidate in names:
        rows = [row for row in per_seed if row["candidate"] == candidate]
        entry = {"candidate": candidate, "seeds": len(rows),
                 "parameter_count": rows[0]["parameter_count"]}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            entry[f"mean_{metric}"] = statistics.mean(values)
            entry[f"sd_{metric}"] = statistics.stdev(values)
        means.append(entry)
    comparisons = [("TV_effective", "TV_old"), ("TV_effective", "TV_clean")]
    if args.stage == "all":
        comparisons += [("C_concat", "B_early"), ("C_concat", "TAV_MLP"),
                        ("C_concat", "TV_effective"), ("C_gate", "C_concat")]
    if args.stage == "lr":
        comparisons += [("C_concat_lr", "C_concat"), ("C_concat_lr", "TV_old")]
    if args.stage == "consistency":
        comparisons += [("C_concat_cons", "C_concat_lr"), ("C_concat_cons", "TV_old")]
    paired = []
    for treatment, control in comparisons:
        for seed in SEEDS:
            a = next(row for row in per_seed if row["candidate"] == treatment and row["seed"] == seed)
            b = next(row for row in per_seed if row["candidate"] == control and row["seed"] == seed)
            paired.append({"treatment": treatment, "control": control, "seed": seed,
                           **{f"delta_{metric}": float(a[metric]) - float(b[metric])
                              for metric in METRICS}})
    grouped = []
    for (candidate, group), rows in sorted(groups.items()):
        grouped.append({"candidate": candidate, "missing_group": group,
                        "scenario_seed_rows": len(rows),
                        **{f"mean_{metric}": statistics.mean(float(row[metric]) for row in rows)
                           for metric in ("accuracy", "macro_f1", "mae", "pearson")}})
    prefix = ("q2_step9_compare_lr" if args.stage == "lr" else
              "q2_step10_compare_consistency" if args.stage == "consistency" else
              f"q2_step8_compare_{args.stage}")
    out = ROOT / "results" / f"{prefix}_{args.output_tag}"
    if out.exists():
        raise FileExistsError(f"Use a new --output-tag: {out}")
    out.mkdir()
    write_csv(out / "all_seed_metrics.csv", per_seed)
    write_csv(out / "candidate_means.csv", means)
    write_csv(out / "paired_deltas.csv", paired)
    write_csv(out / "missing_group_means.csv", grouped)
    (out / "summary.json").write_text(json.dumps({
        "status": "VALIDATION_COMPARISON_ONLY", "stage": args.stage,
        "candidates": list(names), "seeds": SEEDS,
        "source_sha256": reference[0], "bert_sha256": reference[1],
        "same_54_scenarios": True,
        "note": "A2 test and A3/A4 unused; repeated valid comparisons are exploratory."
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "mean_J":
                      {row["candidate"]: row["mean_selection_J"] for row in means}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
