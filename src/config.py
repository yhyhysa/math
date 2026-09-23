"""Project-root and path resolution that never depends on the caller's PATH/cwd."""
from __future__ import annotations

import json
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.json"


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(cfg_path)
    cfg["_project_root"] = Path(cfg["project_root"]).resolve()
    engine_name = cfg.pop("engine_dir", "工程")
    cfg["_engine_dir"] = (cfg["_project_root"] / engine_name).resolve()
    for key in (
        "raw_dir",
        "extracted_dir",
        "processed_dir",
        "cache_dir",
        "checkpoint_dir",
        "results_dir",
        "figures_dir",
        "submission_dir",
        "video_attachment1_root",
        "attachment2_root",
        "attachment3_root",
        "attachment4_root",
    ):
        if key in cfg:
            rel = cfg[key]
            rel_p = Path(rel)
            if rel_p.is_absolute():
                cfg["_" + key] = rel_p
            else:
                parts = rel_p.parts
                if parts and parts[0] in (engine_name, "工程"):
                    rel_p = Path(*parts[1:])
                cfg["_" + key] = (cfg["_engine_dir"] / rel_p).resolve()
        else:
            cfg["_" + key] = None
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for key in (
        "results_dir",
        "figures_dir",
        "submission_dir",
        "checkpoint_dir",
        "processed_dir",
        "cache_dir",
    ):
        p = cfg.get("_" + key)
        if p:
            p.mkdir(parents=True, exist_ok=True)
