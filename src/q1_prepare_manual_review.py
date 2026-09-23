"""Prepare a human-editable review sheet for Q1 without changing source outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/q1"
RESULTS = ROOT / "results"
COLUMNS = ("检查层级", "样本ID", "自动状态", "问题类型", "原视频", "给定文本", "待查词",
           "建议跳转秒", "自动起点秒", "自动终点秒", "对齐分数", "人工判定", "修正起点秒",
           "修正终点秒", "审查者", "审查日期", "备注")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("q1", "q1_v2"), default="q1")
    args = parser.parse_args()
    data = ROOT / "data/processed" / args.variant
    results = ROOT / "results" / ("q1_v2" if args.variant == "q1_v2" else "")
    with (results / "q1_review_queue.csv").open(encoding="utf-8-sig", newline="") as f:
        issues = list(csv.DictReader(f))
    by_id: dict[str, list[dict]] = {}
    for issue in issues:
        by_id.setdefault(issue["sample_id"], []).append(issue)
    rows = []
    for path in sorted(data.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        sid = meta["sample_id"]
        base = {"样本ID": sid, "自动状态": meta["quality_status"],
                "原视频": meta["video_path"], "给定文本": meta["transcript"]}
        rows.append(base | {"检查层级": "整段", "问题类型": "文本/音画/目标人物总核对"})
        for issue in by_id.get(sid, []):
            start = issue["start_s"]
            jump = round(max(0.0, float(start) - 0.5), 3) if start else ""
            rows.append(base | {"检查层级": "异常点", "问题类型": issue["reason"],
                                "待查词": issue["word"], "建议跳转秒": jump,
                                "自动起点秒": start, "自动终点秒": issue["end_s"],
                                "对齐分数": issue["score"]})
    target = results / "q1_manual_review_sheet.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"file": str(target), "clip_rows": 100,
                      "issue_rows": len(issues), "total_rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
