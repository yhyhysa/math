"""Probe a pickle lazily: inspect container structure without materialising big arrays.

Reads with `pickle.load` but reports shapes/dtypes by walking the object graph and
freeing sub-objects where possible. For very large files use --light to only read
the top-level keys via a streaming unpickler.
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import pickletools
import sys
from pathlib import Path

import numpy as np


def describe(obj, depth: int = 0, max_items: int = 8, max_depth: int = 3) -> object:
    ind = "  " * depth
    if isinstance(obj, np.ndarray):
        return {"kind": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        out = {"kind": "dict", "n_keys": len(obj), "keys": list(map(str, obj.keys()))[:64]}
        if depth < max_depth:
            out["children"] = {
                str(k): describe(v, depth + 1, max_items, max_depth)
                for k, v in list(obj.items())[:max_items]
            }
        return out
    if isinstance(obj, (list, tuple)):
        out = {"kind": type(obj).__name__, "len": len(obj)}
        if depth < max_depth and len(obj):
            out["first"] = describe(obj[0], depth + 1, max_items, max_depth)
        return out
    if isinstance(obj, str):
        return {"kind": "str", "len": len(obj), "preview": obj[:120]}
    if isinstance(obj, (int, float, bool, np.integer, np.floating)):
        return {"kind": type(obj).__name__, "value": obj if not isinstance(obj, np.generic) else obj.item()}
    return {"kind": type(obj).__name__}


def light_scan(path: Path, limit: int = 200) -> dict:
    """Streaming scan of top-level opcodes so we can see keys without full load."""
    keys: list[dict] = []
    with open(path, "rb") as fh:
        buf = io.BytesIO(fh.read(1 << 22))
        try:
            for opcode, arg, pos in pickletools.genops(buf):
                if opcode.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE") and isinstance(arg, str):
                    keys.append({"opcode": opcode.name, "pos": pos, "value": arg[:80]})
                    if len(keys) >= limit:
                        break
        except Exception as exc:  # truncated stream, expected
            keys.append({"note": f"stream ended: {type(exc).__name__}"})
    return {"scanned_prefix_bytes": 1 << 22, "string_opcodes": keys[:limit]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pickle-byte-sample", type=int, default=0,
                    help="if >0, read only this many bytes and dump opcode strings")
    args = ap.parse_args(argv)

    report = {}
    for p in args.paths:
        path = Path(p)
        entry: dict = {"size_bytes": path.stat().st_size}
        if args.light or args.pickle_byte_sample:
            n = args.pickle_byte_sample or (1 << 22)
            with open(path, "rb") as fh:
                data = fh.read(n)
            entry["light"] = {"bytes_read": len(data)}
            strings = []
            try:
                for opcode, arg, pos in pickletools.genops(io.BytesIO(data)):
                    if opcode.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE") and isinstance(arg, str):
                        strings.append(arg[:60])
                        if len(strings) >= 80:
                            break
            except Exception as exc:
                entry["light"]["stream_note"] = type(exc).__name__
            entry["light"]["strings"] = strings
            report[str(path)] = entry
            print(json.dumps({str(path): entry}, ensure_ascii=False, indent=1)[:4000])
            continue
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        entry["structure"] = describe(obj)
        report[str(path)] = entry
        del obj
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
