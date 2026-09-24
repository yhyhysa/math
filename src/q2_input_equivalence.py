"""Check that synthetic A/V zero rows and internal availability masks match on A2 valid."""
from __future__ import annotations

import argparse
import copy
import json
import re

import numpy as np
import torch

from src.config import load_config
from src.data import load_pickle
from src.q2_train_baseline import DATA_SHA256, content_mask, fit_stats, interval_mask, pooled, prepare, sha256
from src.q2_train_transformer import GatedTemporalTransformer, SeparateTemporalTransformer, as_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20260924")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("tag must contain only letters, digits, underscores, or hyphens")
    cfg = load_config()
    source = cfg["_attachment2_root"] / "aligned_50.pkl"
    if sha256(source) != DATA_SHA256:
        raise ValueError("Pinned A2 hash differs; stop")
    payload = load_pickle(source)
    valid_samples = payload.splits["valid"][:16]
    train = prepare(payload.splits["train"][:128])
    mean, std = fit_stats(train)
    original = prepare(valid_samples)
    for name in ("audio", "vision"):
        original["array"][name] = (original["array"][name] - mean[name]) / std[name]
        original["array"][name] *= original["observed"][name][..., None]
    result = {"status": "PASS", "source_sha256": DATA_SHA256, "valid_samples": len(valid_samples),
              "checks": []}
    for group in ("A", "V", "AV"):
        removed = {name: np.zeros((len(valid_samples), 50), dtype=bool)
                   for name in ("text", "audio", "vision")}
        zeroed_samples = []
        for row, sample in enumerate(valid_samples):
            block = interval_mask(content_mask(sample.arrays["text_bert"]), "middle", .3)
            clone = copy.copy(sample)
            clone.arrays = sample.arrays.copy()
            for short, name in (("A", "audio"), ("V", "vision")):
                if short in group:
                    array = sample.arrays[name].copy()
                    array[block] = 0
                    clone.arrays[name] = array
                    removed[name][row] = block
            zeroed_samples.append(clone)
        zeroed = prepare(zeroed_samples)
        for name in ("audio", "vision"):
            zeroed["array"][name] = (zeroed["array"][name] - mean[name]) / std[name]
            zeroed["array"][name] *= zeroed["observed"][name][..., None]
        pooled_mask = pooled(original, removed)
        pooled_zero = pooled(zeroed)
        pooled_error = float(np.max(np.abs(pooled_mask - pooled_zero)))
        masked_view = as_view(original, torch.device("cpu"))
        masked_view["available"] = masked_view["available"].clone()
        for short, index, name in (("A", 1, "audio"), ("V", 2, "vision")):
            if short in group:
                masked_view["available"][:, :, index] &= ~torch.as_tensor(removed[name])
        zeroed_view = as_view(zeroed, torch.device("cpu"))
        if not torch.equal(masked_view["available"], zeroed_view["available"]):
            raise AssertionError(f"Availability differs for {group}")
        prediction_errors = {}
        for name, model in (
            ("early", GatedTemporalTransformer(layers=1, mask_empty=True)),
            ("separate_concat", SeparateTemporalTransformer(layers=1, fusion="concat")),
            ("separate_gate", SeparateTemporalTransformer(layers=1, fusion="reliability_gate")),
        ):
            torch.manual_seed(77)
            model.eval()
            with torch.inference_mode():
                logits_mask, values_mask = model(masked_view)
                logits_zero, values_zero = model(zeroed_view)
            prediction_errors[name] = float(max((logits_mask - logits_zero).abs().max(),
                                                (values_mask - values_zero).abs().max()))
        if pooled_error > 1e-6 or any(error > 1e-6 for error in prediction_errors.values()):
            raise AssertionError(f"Zero row and mask disagree for {group}: {pooled_error}, {prediction_errors}")
        result["checks"].append({"group": group, "pooled_max_abs_error": pooled_error,
                                 "temporal_prediction_max_abs_error": prediction_errors,
                                 "masked_positions": int(sum(removed[name].sum() for name in ("audio", "vision")))})
    output = cfg["_results_dir"] / f"q2_step7_input_equivalence_{args.tag}.json"
    if output.exists():
        raise FileExistsError(f"Use a new --tag to preserve the old result: {output}")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
