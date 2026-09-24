"""Pack the selected three-modal Q2 weights, A3 predictions, and audit tables."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from src.q2_compare_step3 import ROOT, SEEDS
from src.q2_compare_tonight import CONFIGS


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    result = ROOT / "results/q2_tonight_20260924"
    choice = json.loads((result / "selection.json").read_text(encoding="utf-8"))["selected"]
    if choice not in CONFIGS:
        raise ValueError(f"Unknown selected candidate: {choice}")
    short, tag, _, _, _ = CONFIGS[choice]
    prefix = f"q2_c_aug_1layer_{short}_eff_{tag}"
    a3_name = ("q2_attachment3_three_modal_20260924" if choice == "C_current" else
               f"q2_attachment3_candidate_{prefix}")
    a3 = ROOT / "results" / a3_name
    summary = json.loads((a3 / "summary.json").read_text(encoding="utf-8"))
    if summary["sample_count"] != 30 or summary["labels_used"] or summary["a3_used_for_model_selection"]:
        raise ValueError("Invalid A3 prediction status")
    folder = ROOT / "submission/Q2_今晚交付_20260924"
    archive = folder.with_suffix(".zip")
    if folder.exists() or archive.exists():
        raise FileExistsError("Preserve the first package; use a new date or name")
    folder.mkdir(parents=True)

    def copy(source: Path, relative: str) -> None:
        destination = folder / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for name in ("实验协议.md", "candidate_means.csv", "paired_deltas.csv", "all_seed_metrics.csv",
                 "missing_group_means.csv",
                 "selection.json", "neutral_confusion.csv", "neutral_summary.json", "人工审查结论.md"):
        copy(result / name, f"results/{name}")
    for name in ("predictions.csv", "summary.json", "input_manifest.csv"):
        copy(a3 / name, f"results/attachment3/{name}")
    for seed in SEEDS:
        run = f"{prefix}_seed_{seed}"
        source = ROOT / "results" / run
        if (source / "exit_code.txt").read_text(encoding="utf-8-sig").strip() != "0":
            raise ValueError(f"Incomplete selected training run: {run}")
        for name in ("summary.json", "history.csv", "scenario_metrics.csv", "mask_history.csv", "exit_code.txt"):
            copy(source / name, f"results/{run}/{name}")
        copy(ROOT / "checkpoints" / run / "best.pt", f"checkpoints/{run}/best.pt")
    for name in ("q2_train_transformer.py", "q2_missing_patterns.py", "q2_train_baseline.py",
                 "q2_predict_attachment3.py", "q2_compare_tonight.py", "q2_neutral_diagnostic.py",
                 "q2_compare_step3.py", "q2_compare_step4.py", "q2_package_tonight.py",
                 "q2_valid_missing_audit.py",
                 "q2_locked_test_eval.py", "run_q2_transformer.ps1", "run_q2_tonight.ps1",
                 "data.py", "config.py", "__init__.py"):
        copy(ROOT / "src" / name, f"src/{name}")
    for name in ("__init__.py", "verify_raw_integrity.py"):
        copy(ROOT / "src/tools" / name, f"src/tools/{name}")
    copy(ROOT / "configs/base.json", "configs/base.json")
    copy(ROOT / "运行使用说明.md", "运行使用说明.md")

    with (a3 / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    if len(predictions) != 30:
        raise ValueError("Expected 30 A3 rows")
    columns = ("sample_id", "source_file", "polarity_code", "polarity", "intensity",
               "negative_prob", "neutral_prob", "positive_prob", "model_version")
    with (folder / "附件3_三模态预测.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in predictions:
            writer.writerow({"sample_id": row["sample_id"], "source_file": row["source_file"],
                             "polarity_code": row["polarity_code"], "polarity": row["polarity"],
                             "intensity": row["strength"], "negative_prob": row["p_negative"],
                             "neutral_prob": row["p_neutral"], "positive_prob": row["p_positive"],
                             "model_version": row["model_version"]})
    readme = (
        "# 第二问今晚三模态交付件\n\n"
        "先打开 `附件3_三模态预测.csv`：30 行无标签附件三预测。0/1/2 是负/中/正，"
        "intensity 在 [-3,3]。附件三没有真值，不能计算准确率。\n\n"
        f"依据训练前固定规则，附件二 valid 选出的版本：`{choice}`，三个种子等权集成。"
        "全部候选的完整指标见 `results/candidate_means.csv`，逐种子差值见 "
        "`results/paired_deltas.csv`，中性类误判见 `results/neutral_confusion.csv`。"
        "`results/人工审查结论.md` 写明改善与限制。\n\n"
        "附件二 test 在本轮优化前已揭示，本轮没有用它调模型，也不声称新候选有独立盲测成绩。"
        "原始数据、第一问特征和大型 BERT 不在此包；本机完整运行步骤见 `运行使用说明.md`。"
        "此包是第二问组件，正式提交前仍要按赛题核对总材料、容量、匿名与文件名。\n"
    )
    (folder / "README_交付说明.md").write_text(readme, encoding="utf-8")
    manifest = [f"{digest(path)}  {path.relative_to(folder).as_posix()}"
                for path in sorted(folder.rglob("*")) if path.is_file()]
    (folder / "SHA256SUMS.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(folder).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
    if archive.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("Q2 component package exceeds 50 MiB")
    print(json.dumps({"selected": choice, "archive": str(archive), "bytes": archive.stat().st_size,
                      "files": len(manifest) + 1}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
