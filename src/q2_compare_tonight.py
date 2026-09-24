"""Compare the predeclared Q2 candidates on the unchanged A2 valid panel."""
from __future__ import annotations

import json
import statistics

from src.q2_compare_step3 import ROOT, SEEDS, read_run, write_csv
from src.q2_compare_step4 import METRICS, flatten


CONFIGS = {
    "C_current": ("concat", "step9_lr3e4_20260924", 3e-4, 0.0, "single"),
    "C_gate": ("gate", "tonight_gate_20260924", 3e-4, 0.0, "single"),
    "C_lr1e4": ("concat", "tonight_lr1e4_20260924", 1e-4, 0.0, "single"),
    "C_uni": ("concat", "tonight_uni_20260924", 3e-4, 0.1, "single"),
    "C_gate_uni": ("gate", "tonight_gate_uni_20260924", 3e-4, 0.1, "single"),
    "C_multi": ("concat", "tonight_multi_20260924", 3e-4, 0.0, "multi"),
}


def main() -> None:
    output = ROOT / "results" / "q2_tonight_20260924"
    per_seed = []
    groups = {}
    fingerprint = None
    for candidate, (short, tag, lr, aux, pattern) in CONFIGS.items():
        fusion = "reliability_gate" if short == "gate" else "concat"
        for seed in SEEDS:
            path = ROOT / "results" / f"q2_c_aug_1layer_{short}_eff_{tag}_seed_{seed}"
            summary, scenes = read_run(path)
            status = path / "exit_code.txt"
            if not status.exists() or status.read_text(encoding="utf-8-sig").strip() != "0":
                raise ValueError(f"Missing successful process exit code: {path}")
            if (summary["seed"] != seed or summary["architecture"] != "separate" or
                    summary["fusion"] != fusion or summary["learning_rate"] != lr or
                    summary.get("auxiliary_weight", 0.0) != aux or
                    summary.get("mask_pattern", "single") != pattern or
                    not summary["augment"] or not summary["effective_mask"]):
                raise ValueError(f"Candidate configuration differs: {path}")
            current = (summary["source_sha256"], summary["bert_sha256"],
                       tuple((row["modalities"], row["position"], row["ratio"], row["n"],
                              row["masked_positions"], row["mean_actual_ratio"],
                              row["newly_lost_observed_ratio"], row["unobservable_positions"])
                             for row in scenes))
            if fingerprint is None:
                fingerprint = current
            elif fingerprint != current:
                raise ValueError(f"Input or 54-case validation panel differs: {path}")
            per_seed.append(flatten(candidate, seed, summary, scenes))
            for scene in scenes:
                for group in (scene["modalities"],
                              "text_missing" if "T" in scene["modalities"] else "text_available"):
                    groups.setdefault((candidate, group), []).append(scene)
    means = []
    for candidate in CONFIGS:
        rows = [row for row in per_seed if row["candidate"] == candidate]
        entry = {"candidate": candidate, "seeds": len(rows), "parameter_count": rows[0]["parameter_count"]}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            entry[f"mean_{metric}"] = statistics.mean(values)
            entry[f"sd_{metric}"] = statistics.stdev(values)
        means.append(entry)
    base = next(row for row in means if row["candidate"] == "C_current")
    paired = []
    qualified = []
    for row in means[1:]:
        candidate = row["candidate"]
        improvements = 0
        for seed in SEEDS:
            a = next(r for r in per_seed if r["candidate"] == candidate and r["seed"] == seed)
            b = next(r for r in per_seed if r["candidate"] == "C_current" and r["seed"] == seed)
            delta = {metric: float(a[metric]) - float(b[metric]) for metric in METRICS}
            improvements += delta["selection_J"] < 0
            paired.append({"candidate": candidate, "seed": seed,
                           **{f"delta_{metric}": value for metric, value in delta.items()}})
        row["delta_mean_J"] = row["mean_selection_J"] - base["mean_selection_J"]
        row["delta_mean_masked_mae"] = row["mean_masked_mae"] - base["mean_masked_mae"]
        row["improved_seed_count"] = improvements
        row["qualifies"] = (row["delta_mean_J"] <= -.005 and improvements >= 2 and
                            row["delta_mean_masked_mae"] <= .01)
        if row["qualifies"]:
            qualified.append(row)
    base.update({"delta_mean_J": 0.0, "delta_mean_masked_mae": 0.0,
                 "improved_seed_count": 0, "qualifies": False})
    qualified.sort(key=lambda item: item["mean_selection_J"])
    selected = qualified[0]["candidate"] if qualified else "C_current"
    if qualified:
        # Preserve the smaller architecture when validation differences are tiny.
        close = [row for row in qualified
                 if row["mean_selection_J"] - qualified[0]["mean_selection_J"] < .002]
        selected = min(close, key=lambda item: (item["parameter_count"],
                                                item["mean_selection_J"]))["candidate"]
    for name, rows in (("all_seed_metrics.csv", per_seed), ("candidate_means.csv", means),
                       ("paired_deltas.csv", paired),
                       ("missing_group_means.csv", [
                           {"candidate": candidate, "missing_group": group,
                            "scenario_seed_rows": len(scenes),
                            **{f"mean_{metric}": statistics.mean(float(scene[metric]) for scene in scenes)
                               for metric in ("accuracy", "macro_f1", "mae", "pearson")}}
                           for (candidate, group), scenes in sorted(groups.items())])):
        path = output / name
        if path.exists():
            raise FileExistsError(path)
        write_csv(path, rows)
    payload = {"status": "VALIDATION_COMPARISON_ONLY", "selected": selected,
               "candidates": list(CONFIGS), "seeds": SEEDS, "source_sha256": fingerprint[0],
               "bert_sha256": fingerprint[1], "same_54_scenarios": True,
               "rule": "mean J <= current-0.005; >=2/3 seeds improve; masked MAE delta <=0.01",
               "test_used_for_selection": False, "attachment3_used_for_selection": False}
    (output / "selection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "mean_J": {r["candidate"]: r["mean_selection_J"] for r in means}},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
