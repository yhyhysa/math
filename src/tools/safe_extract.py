"""Safe extraction of the read-only E题 data archive into 工程/data/raw/extracted.

Guards:
  * refuses absolute paths / drive letters / .. traversal (zip-slip)
  * NEVER overwrites: an existing target is compared by *content*, never by size
  * writes a per-entry manifest (name, size, crc32, sha256, comparison result)

Existing-file policy (P1 fix, per E题/AGENTS.md `工程/data/raw/` is read-only):
  * target missing           -> extract
  * target exists, content identical (member CRC32 + SHA-256 both match) -> skip, status=skipped-existing-identical
  * target exists, content differs                                      -> ABORT with non-zero exit; nothing is overwritten
  * target exists but is unreadable / is a directory                    -> ABORT with non-zero exit

Usage (cwd = 工程/):
  python -m src.tools.safe_extract --zip data/raw/E题数据.zip --out data/raw/extracted
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
import zlib
from pathlib import Path


def is_within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def stream_sha256(stream, chunk_size: int = 1 << 22) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(chunk_size), b""):
        h.update(block)
    return h.hexdigest()


def file_sha256(path: Path, chunk_size: int = 1 << 22) -> str:
    with path.open("rb") as fh:
        return stream_sha256(fh, chunk_size)


def file_crc32(path: Path, chunk_size: int = 1 << 22) -> int:
    crc = 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            crc = zlib.crc32(block, crc)
    return crc & 0xFFFFFFFF


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--only", default=None, help="substring filter on entry name")
    ap.add_argument(
        "--ensure-hash-manifest",
        default=None,
        help="optional path to a JSON {relative_path: sha256} map to cross-check written files",
    )
    args = ap.parse_args(argv)

    zpath = Path(args.zip)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out.parent / "extract_manifest.json"

    t0 = time.time()
    entries_meta: list[dict] = []
    skipped_existing = 0
    written = 0
    rejected: list[dict] = []
    conflicts: list[dict] = []

    with zipfile.ZipFile(zpath) as zf:
        infos = sorted(zf.infolist(), key=lambda i: i.filename)
        for info in infos:
            name = info.filename
            if args.only and args.only not in name:
                continue
            if name.endswith("/"):
                entries_meta.append(
                    {"name": name, "kind": "dir", "size": 0, "crc32": None, "sha256": None, "status": "dir"}
                )
                continue
            target = out / name
            if not is_within(out, target):
                rejected.append({"name": name, "reason": "zip-slip outside output dir"})
                print(f"[REJECT] {name}: outside output dir", flush=True)
                continue
            if os.path.isabs(name) or ".." in Path(name).parts:
                rejected.append({"name": name, "reason": "absolute or traversal path"})
                print(f"[REJECT] {name}: absolute/traversal", flush=True)
                continue

            with zf.open(info) as src:
                member_sha = stream_sha256(src)
            member_crc = f"{info.CRC:08x}"

            if target.exists():
                if not target.is_file():
                    conflicts.append(
                        {"name": name, "reason": "existing target is not a regular file", "action": "aborted"}
                    )
                    print(f"[CONFLICT] {name}: existing target is not a regular file", flush=True)
                    break
                try:
                    existing_sha = file_sha256(target)
                    existing_crc = f"{file_crc32(target):08x}"
                except OSError as exc:
                    conflicts.append(
                        {"name": name, "reason": f"existing target unreadable: {exc}", "action": "aborted"}
                    )
                    print(f"[CONFLICT] {name}: unreadable existing target ({exc})", flush=True)
                    break
                identical = (existing_sha == member_sha) and (existing_crc == member_crc)
                if identical:
                    skipped_existing += 1
                    entries_meta.append(
                        {
                            "name": name,
                            "kind": "file",
                            "size": info.file_size,
                            "crc32": member_crc,
                            "sha256": member_sha,
                            "existing_sha256": existing_sha,
                            "existing_crc32": existing_crc,
                            "content_identical": True,
                            "status": "skipped-existing-identical",
                        }
                    )
                    continue
                conflicts.append(
                    {
                        "name": name,
                        "reason": "existing file content differs from archive member",
                        "action": "aborted-no-overwrite",
                        "existing_sha256": existing_sha,
                        "archive_sha256": member_sha,
                        "existing_size": int(target.stat().st_size),
                        "archive_size": int(info.file_size),
                    }
                )
                print(
                    f"[CONFLICT] {name}: existing content differs (existing {existing_sha[:12]}..., "
                    f"archive {member_sha[:12]}...); refusing to overwrite",
                    flush=True,
                )
                break

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "xb") as dst:
                while True:
                    chunk = src.read(1 << 22)
                    if not chunk:
                        break
                    dst.write(chunk)
            written_sha = file_sha256(target)
            if written_sha != member_sha:
                conflicts.append(
                    {
                        "name": name,
                        "reason": "written file hash does not match archive member",
                        "action": "aborted",
                        "written_sha256": written_sha,
                        "archive_sha256": member_sha,
                    }
                )
                print(f"[CONFLICT] {name}: written bytes do not match archive member", flush=True)
                break
            written += 1
            entries_meta.append(
                {
                    "name": name,
                    "kind": "file",
                    "size": info.file_size,
                    "crc32": member_crc,
                    "sha256": member_sha,
                    "content_identical": None,
                    "status": "written",
                }
            )
            if written % 25 == 0:
                print(f"[{written} files] {name}", flush=True)

    hash_manifest_check = None
    if args.ensure_hash_manifest:
        hm_path = Path(args.ensure_hash_manifest)
        if hm_path.is_file():
            expected_map = json.loads(hm_path.read_text(encoding="utf-8"))
            mismatched = []
            for rel, expected in expected_map.items():
                fp = out / rel
                if not fp.is_file():
                    mismatched.append({"relative_path": rel, "issue": "missing"})
                elif file_sha256(fp) != str(expected).lower():
                    mismatched.append({"relative_path": rel, "issue": "sha256 mismatch"})
            hash_manifest_check = {"manifest": str(hm_path), "checked": len(expected_map), "mismatched": mismatched}
            if mismatched:
                conflicts.append({"name": str(hm_path), "reason": "ensure-hash-manifest mismatch", "action": "aborted"})

    failed = bool(rejected or conflicts)
    payload = {
        "zip": str(zpath),
        "zip_size": zpath.stat().st_size,
        "zip_sha256": file_sha256(zpath),
        "out": str(out),
        "entries": entries_meta,
        "written": written,
        "skipped_existing_identical": skipped_existing,
        "rejected": rejected,
        "conflicts": conflicts,
        "overwrite_performed": False,
        "existing_file_policy": "content-hash compared (CRC32 + SHA-256); never overwritten, never size-only compared",
        "hash_manifest_check": hash_manifest_check,
        "status": "FAIL" if failed else "PASS",
        "elapsed_s": round(time.time() - t0, 2),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"DONE written={written} skipped_identical={skipped_existing} rejected={len(rejected)} "
        f"conflicts={len(conflicts)} elapsed={payload['elapsed_s']}s manifest={manifest_path}",
        flush=True,
    )
    if failed:
        print("RESULT: FAIL (no file was overwritten)", flush=True)
        return 2
    print("RESULT: PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
