"""Read-only verification against the original archive pinned by independent review.

Run from 工程: python -m src.tools.verify_raw_integrity
Writes only results/codex_review/raw_integrity_<UTC timestamp>.json.
"""
import csv
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_ZIP = "f895003abeb3e972e69ef9c457032720cbcb8ecd3d2261a0e30bc15667604acb"
ENGINE = Path(__file__).resolve().parents[2]


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def file_hash(path):
    with path.open("rb") as f:
        return digest(f)


def main():
    started = datetime.now(timezone.utc)
    errors, entries = [], []
    archive = ENGINE.parent / "E题数据.zip"
    copied = ENGINE / "data/raw/E题数据.zip"
    extracted = ENGINE / "data/raw/extracted"
    hashes = {str(p): file_hash(p) for p in (archive, copied)}
    if any(h != EXPECTED_ZIP for h in hashes.values()):
        errors.append("Original archive or engineering copy does not match pinned SHA-256")
    if not errors:
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                target = (extracted / info.filename).resolve()
                if not target.is_relative_to(extracted.resolve()):
                    errors.append("Unsafe archive path: " + info.filename)
                    continue
                with zf.open(info) as src:
                    expected = digest(src)
                actual = file_hash(target) if target.is_file() else None
                entry = {"relative_path": target.relative_to(ENGINE).as_posix(),
                         "size_bytes": info.file_size, "archive_sha256": expected,
                         "disk_sha256": actual, "matches": expected == actual}
                entries.append(entry)
                if not entry["matches"]:
                    errors.append("Changed or missing extracted file: " + info.filename)
        expected_paths = {r["relative_path"] for r in entries}
        disk_paths = {p.relative_to(ENGINE).as_posix() for p in extracted.rglob("*") if p.is_file()}
        if disk_paths != expected_paths:
            errors.append("Extracted directory file set differs from archive")
        expected_manifest = {r["relative_path"]: r["archive_sha256"] for r in entries
                             if Path(r["relative_path"]).name != ".DS_Store"}
        expected_manifest[copied.relative_to(ENGINE).as_posix()] = EXPECTED_ZIP
        with (ENGINE / "results/input_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        actual_manifest = {r["relative_path"].replace("\\", "/"): r["sha256"].lower() for r in rows}
        if len(actual_manifest) != len(rows) or actual_manifest != expected_manifest:
            errors.append("Input manifest paths/hashes differ from pinned archive contents")
    result = {"timestamp_utc": started.isoformat(), "status": "FAIL" if errors else "PASS",
              "expected_zip_sha256": EXPECTED_ZIP, "archives": hashes,
              "checked_extracted_files": len(entries), "entries": entries, "errors": errors,
              "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
              "scope": "Archive and extracted bytes plus input manifest; not an OS permission lock"}
    output_dir = ENGINE / "results/codex_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / ("raw_integrity_" + started.strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"status": result["status"], "checked": len(entries), "errors": errors,
                      "report": str(output), "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
