"""Balanced one/two-block missing views for the Q2 train split only."""
from __future__ import annotations

import math

import numpy as np

from src.q2_train_baseline import MODALITY_SETS, RATIOS


NAMES = {"T": "text", "A": "audio", "V": "vision"}


def training_multi_block_view(data: dict, seed: int, epoch: int) -> tuple[dict[str, np.ndarray], dict]:
    content = data["content"]
    removed = {name: np.zeros_like(content) for name in NAMES.values()}
    counts = {group: 0 for group in MODALITY_SETS}
    two_blocks = fallback_clean = 0
    for row, valid in enumerate(content):
        positions = np.flatnonzero(valid)
        observed = {name: data["observed"][name][row] for name in NAMES.values()}
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, row, 2]))
        for _ in range(20):
            group = MODALITY_SETS[int(rng.integers(len(MODALITY_SETS)))]
            want_two = len(positions) >= 4 and rng.random() < .5
            ratios = RATIOS[:2] if want_two else RATIOS
            lengths = [min(len(positions), max(1, math.ceil(float(rng.choice(ratios)) * len(positions))))
                       for _ in range(2 if want_two else 1)]
            if want_two and len(positions) < sum(lengths) + 1:
                want_two = False
                lengths = lengths[:1]
            first = int(rng.integers(len(positions) - sum(lengths) - int(want_two) + 1))
            blocks = [positions[first:first + lengths[0]]]
            if want_two:
                second = int(rng.integers(first + lengths[0] + 1,
                                          len(positions) - lengths[1] + 1))
                blocks.append(positions[second:second + lengths[1]])
            chosen = np.concatenate(blocks)
            if not any(observed[NAMES[short]][chosen].any() for short in group):
                continue
            available = {name: mask.copy() for name, mask in observed.items()}
            for short in group:
                available[NAMES[short]][chosen] = False
            if not any(mask.any() for mask in available.values()):
                continue
            for short in group:
                removed[NAMES[short]][row, chosen] = True
            counts[group] += 1
            two_blocks += int(want_two)
            break
        else:
            fallback_clean += 1
    if any(np.any(mask & ~content) for mask in removed.values()):
        raise AssertionError("Training mask touched padding or special tokens")
    stats = {"epoch": epoch, "fallback_clean": fallback_clean,
             "effective_rows": len(content) - fallback_clean, "two_block_rows": two_blocks,
             **{f"group_{key}": value for key, value in counts.items()}}
    for name in removed:
        original = int(data["observed"][name].sum())
        lost = int((data["observed"][name] & removed[name]).sum())
        stats[f"{name}_newly_lost"] = lost
        stats[f"{name}_newly_lost_ratio"] = lost / original if original else None
    return removed, stats


def selfcheck() -> None:
    content = np.zeros((24, 50), dtype=bool)
    content[:, :20] = True
    data = {"content": content, "observed": {name: content.copy() for name in NAMES.values()}}
    removed, stats = training_multi_block_view(data, 19, 1)
    assert stats["effective_rows"] == 24 and stats["two_block_rows"] > 0
    assert sum(stats[f"group_{group}"] for group in MODALITY_SETS) == 24
    assert all(not np.any(mask & ~content) for mask in removed.values())
    combined = np.logical_or.reduce(tuple(removed.values()))
    assert any(np.diff(np.r_[False, row[:20], False].astype(int)).tolist().count(1) == 2
               for row in combined)
    print("PASS balanced one/two-block mask sampler")


if __name__ == "__main__":
    selfcheck()
