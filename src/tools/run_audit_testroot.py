"""TEST-ONLY shim (no effect in a normal run).

Runs `src.audit` against a scratch engineering tree (`src/`, `configs/`, `results/`)
while the read-only data and the integrity-gate tree stay in the real `工程/`.
Used to verify failure semantics without touching the real `STATUS.md`.

Config rewriting rules (`configs/base.testroot.json` is written into the base dir):
  * the copy's engineering tree is `<base>/<engine_dir>` (same layout as the real tree),
    so `_engine_dir` == `<base>/工程` and `STATUS.md` is written *inside* the copy
  * `project_root`                       -> the base dir
  * path entries under the real engineering tree -> `<base>/<engine_dir>/<tail>` for outputs,
    or kept in the REAL tree for read-only inputs
  * everything else relative             -> resolved against `<base>/<engine_dir>`
  * `DSH_T0_TEST_DATA=1`                 -> read-only input paths point at the REAL engineering
    tree (original files are only ever read)

Every writable output path (results_dir, figures_dir, submission_dir, checkpoint_dir,
processed_dir, cache_dir) always resolves inside the copy.

When `DSH_T0_TESTROOT` is unset/empty this module simply calls `src.audit.main()`,
i.e. exactly the normal behaviour.

Usage (cwd = any scratch copy of `工程/`):
  $env:DSH_T0_TESTROOT = "<base dir copy>"
  $env:DSH_T0_TEST_DATA = "1"        # optional: read real inputs, write scratch outputs
  python -m src.tools.run_audit_testroot
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from src import audit as audit_mod
from src.config import DEFAULT_CONFIG

WRITABLE_KEYS = (
    "results_dir",
    "figures_dir",
    "submission_dir",
    "checkpoint_dir",
    "processed_dir",
    "cache_dir",
)
READONLY_INPUT_KEYS = (
    "raw_dir",
    "extracted_dir",
    "video_attachment1_root",
    "attachment2_root",
    "attachment3_root",
    "attachment4_root",
)
PATH_KEYS = WRITABLE_KEYS + READONLY_INPUT_KEYS


def _tail_under_engine(text: str, real_engine: Path, engine_name: str) -> str:
    """Return the path tail below the engineering directory (or below 工程/ if relative)."""
    lowered = text.lower()
    prefix = str(real_engine).lower()
    if lowered.startswith(prefix):
        return text[len(str(real_engine)):].lstrip("\\")
    for name in (engine_name, "工程"):
        if text.startswith(name + "\\"):
            return text[len(name) + 1:]
    return text


def rewrite_config(base_dir: Path) -> Path:
    """Rewrite the copy's config; return the path of the rewritten config."""
    real = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    engine_name = str(real.get("engine_dir", "工程"))
    real_engine = (Path(str(real["project_root"]).replace("/", "\\")) / engine_name).resolve()
    copy_engine = base_dir / engine_name
    use_real_data = os.environ.get("DSH_T0_TEST_DATA", "").strip() not in ("", "0", "false", "False")

    copy_cfg = json.loads((base_dir / "configs" / "base.json").read_text(encoding="utf-8"))
    copy_cfg["project_root"] = str(base_dir)
    for key in PATH_KEYS:
        raw = copy_cfg.get(key)
        if raw is None:
            continue
        text = str(raw).replace("/", "\\")
        tail = _tail_under_engine(text, real_engine, engine_name)
        if use_real_data and key in READONLY_INPUT_KEYS:
            copy_cfg[key] = str(real_engine / tail)          # read the genuine inputs
        else:
            copy_cfg[key] = str(copy_engine / tail)          # write inside the copy
    copy_cfg["python_exe"] = real.get("python_exe", copy_cfg.get("python_exe"))

    out_path = base_dir / "configs" / "base.testroot.json"
    out_path.write_text(json.dumps(copy_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    # machine-readable record of the effective paths, so callers never have to guess them
    paths_record = {
        "base_dir": str(base_dir),
        "copy_engine_dir": str(copy_engine),
        "status_md": str(copy_engine / "STATUS.md"),
        "results_dir": copy_cfg.get("results_dir"),
        "read_only_inputs_from_real_tree": use_real_data,
        "extracted_dir": copy_cfg.get("extracted_dir"),
        "config_in_use": str(out_path),
    }
    (base_dir / "testroot_paths.json").write_text(
        json.dumps(paths_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[shim] TEST-ONLY: project_root -> {base_dir}")
    print(f"[shim] real engine tree -> {real_engine}")
    print(f"[shim] copy engine tree -> {copy_engine}")
    print(f"[shim] read-only inputs from real tree: {use_real_data}")
    for key in ("results_dir", "extracted_dir"):
        print(f"[shim]   {key} -> {copy_cfg.get(key)}")
    print(f"[shim] STATUS.md will be written to -> {copy_engine / 'STATUS.md'}")
    print(f"[shim] config in use -> {out_path}")
    print(f"[shim] effective paths recorded in -> {base_dir / 'testroot_paths.json'}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    base_dir = os.environ.get("DSH_T0_TESTROOT", "").strip()
    if not base_dir:
        return audit_mod.main(argv)

    base = Path(base_dir).resolve()
    if not (base / "configs" / "base.json").is_file():
        print(f"[shim] base dir has no configs/base.json: {base}", file=sys.stderr)
        return 1
    out_path = rewrite_config(base)
    # Always force the rewritten config, otherwise a caller-supplied config would
    # silently send writes back into the real engineering tree.
    argv = [a for a in argv if a != "--config"]
    while "--config" in argv:
        argv.remove("--config")
    argv = [a for a in argv if not (a.endswith("base.json") or a.endswith("base.testroot.json"))]
    return audit_mod.main(["--config", str(out_path)] + argv)


if __name__ == "__main__":
    sys.exit(main())
