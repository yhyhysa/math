"""Small deterministic checks for the train-only Q2 augmentation path."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from src.q2_train_baseline import CoverageGate, masked_text_means, pooled, training_missing_view


def main() -> None:
    content = np.zeros((3, 50), dtype=bool)
    content[0, 1:11] = True
    content[1, 1] = True
    content[2, 1:6] = True
    observed = {"text": content.copy(), "audio": content.copy(), "vision": content.copy()}
    observed["audio"][1] = False
    observed["vision"][1] = False
    observed["vision"][2, 3] = False
    data = {"content": content, "observed": observed}
    first, stats = training_missing_view(data, 20260923, 1)
    second, same_stats = training_missing_view(data, 20260923, 1)
    assert stats == same_stats and all(np.array_equal(first[m], second[m]) for m in first)
    assert all(not np.any(first[m] & ~content) for m in first)
    assert sum(stats[f"group_{m}"] for m in ("T", "A", "V", "TA", "TV", "AV")) + \
        stats["fallback_clean"] == len(content)
    tv_removed, tv_stats = training_missing_view(data, 20260923, 1, "TV", True)
    assert tv_stats["group_A"] == tv_stats["group_TA"] == tv_stats["group_AV"] == 0
    assert tv_stats["effective_rows"] + tv_stats["fallback_clean"] == len(content)
    for row in range(len(content)):
        if any(mask[row].any() for mask in tv_removed.values()):
            assert any(np.any(observed[name][row] & tv_removed[name][row])
                       for name in ("text", "vision"))
    for row in range(len(content)):
        assert any(np.any(observed[m][row] & ~first[m][row]) for m in first)
        masked = np.flatnonzero(first["text"][row] | first["audio"][row] | first["vision"][row])
        if len(masked):
            assert np.array_equal(masked, np.arange(masked[0], masked[-1] + 1))

    data["array"] = {name: np.ones((3, 50, dim), dtype=np.float32)
                     for name, dim in (("text", 768), ("audio", 74), ("vision", 35))}
    full = pooled(data)
    for kept, missing_slices in (("T", (slice(768, 877),)),
                                 ("TA", (slice(842, 877),)),
                                 ("TV", (slice(768, 842),))):
        selected = pooled(data, keep_modalities=kept)
        assert selected.shape == full.shape and np.array_equal(selected[:, :768], full[:, :768])
        assert all(np.all(selected[:, cut] == 0) for cut in missing_slices)
        for short, fraction_col, flag_col in (("A", 878, 881), ("V", 879, 882)):
            if short not in kept:
                assert np.all(selected[:, [fraction_col, flag_col]] == 0)
    gate = CoverageGate()
    gate_input = torch.as_tensor(full)
    fused, weights = gate.fuse(gate_input)
    assert fused.shape == (3, 128) and weights.shape == (3, 3)
    assert torch.allclose(weights[1], torch.tensor([1., 0., 0.]))
    logits, estimates = gate(gate_input)
    assert logits.shape == (3, 3) and torch.isfinite(estimates).all()
    (logits.square().mean() + estimates.square().mean()).backward()
    assert all(parameter.grad is not None for parameter in gate.parameters())

    ids = np.zeros((1, 3, 50), dtype=np.int64)
    ids[0, 0, :4] = [101, 11, 22, 102]
    ids[0, 1, :4] = 1
    missing = np.zeros((1, 50), dtype=bool)
    missing[0, 1] = True
    one_content = np.zeros((1, 50), dtype=bool)
    one_content[0, 1:3] = True

    class FakeBert:
        def __call__(self, input_ids, attention_mask, token_type_ids):
            self.ids = input_ids.cpu().numpy()
            values = input_ids.float().unsqueeze(-1).expand(-1, -1, 768)
            return SimpleNamespace(last_hidden_state=values)

    fake = FakeBert()
    text = masked_text_means({"tb": ids, "content": one_content}, missing, fake,
                             torch.device("cpu"), batch_size=1)
    assert fake.ids[0, 1] == 103 and ids[0, 0, 1] == 11
    assert text.shape == (1, 768) and np.all(text == 22)
    print("Q2 augmentation self-check PASS: masks, modality ablation, coverage gate, masked text")


if __name__ == "__main__":
    main()
