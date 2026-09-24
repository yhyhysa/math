"""Q2 model A: clean or continuously masked training, same 54-case validation panel.

Run from 工程 with: python -m src.q2_train_baseline [--augment] [--smoke]
Only A2 train fits weights/statistics; only A2 valid selects the checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import BertModel, __version__ as transformers_version

from src.config import load_config
from src.data import load_pickle


BERT_REV = "86b5e0934494bd15c9632b12f734a8a67f723594"
BERT_SHA256 = "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"
DATA_SHA256 = "66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd"
MODALITY_SETS = ("T", "A", "V", "TA", "TV", "AV")
POSITIONS = ("front", "middle", "back")
RATIOS = (0.1, 0.3, 0.5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_mask(tb: np.ndarray) -> np.ndarray:
    ids, attention = tb[0], tb[1]
    valid = np.flatnonzero(attention == 1)
    if len(valid) < 3 or ids[valid[0]] != 101 or ids[valid[-1]] != 102:
        raise ValueError("Unexpected BERT special tokens or fewer than one content token")
    content = np.zeros(50, dtype=bool)
    content[valid[1:-1]] = True
    return content


def interval_mask(content: np.ndarray, position: str, ratio: float) -> np.ndarray:
    where = np.flatnonzero(content)
    length = min(len(where), max(1, math.ceil(ratio * len(where))))
    start = {"front": 0, "middle": (len(where) - length) // 2,
             "back": len(where) - length}[position]
    out = np.zeros(50, dtype=bool)
    out[where[start:start + length]] = True
    return out


def prepare(samples: list, mean: dict | None = None, std: dict | None = None) -> dict:
    """Convert samples once; exact zero rows are only treated as unobserved candidates."""
    tb = np.stack([s.arrays["text_bert"] for s in samples]).astype(np.int64)
    content = np.stack([content_mask(row) for row in tb])
    arrays = {m: np.stack([s.arrays[m] for s in samples]).astype(np.float32)
              for m in ("text", "audio", "vision")}
    for m, arr in arrays.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"Nonfinite {m} feature")
    observed = {"text": content.copy()}
    for m in ("audio", "vision"):
        observed[m] = content & np.any(np.abs(arrays[m]) > 1e-6, axis=2)
    if mean is not None:
        for m in ("audio", "vision"):
            arrays[m] = (arrays[m] - mean[m]) / std[m]
            arrays[m] *= observed[m][..., None]
    y_class = np.array([int(s.classification_label) for s in samples], dtype=np.int64)
    y_value = np.array([s.regression_label for s in samples], dtype=np.float32)
    if not np.isin(y_class, [0, 1, 2]).all() or not np.isfinite(y_value).all():
        raise ValueError("Invalid train/valid labels")
    return {"tb": tb, "content": content, "array": arrays, "observed": observed,
            "class": y_class, "value": y_value}


def fit_stats(train: dict) -> tuple[dict, dict]:
    means, stds = {}, {}
    for m in ("audio", "vision"):
        rows = train["array"][m][train["observed"][m]]
        means[m] = rows.mean(axis=0, dtype=np.float64).astype(np.float32)
        stds[m] = np.maximum(rows.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    return means, stds


def pooled(data: dict, missing: np.ndarray | dict[str, np.ndarray] | None = None,
           modalities: str = "", text_override: np.ndarray | None = None,
           keep_modalities: str = "TAV") -> np.ndarray:
    """Masked mean for T/A/V, followed by observed fractions and modality flags."""
    parts, fractions, flags = [], [], []
    for short, name in (("T", "text"), ("A", "audio"), ("V", "vision")):
        observed = data["observed"][name]
        if isinstance(missing, dict):
            keep = observed & ~missing[name]
        else:
            keep = observed & ~missing if missing is not None and short in modalities else observed
        if short not in keep_modalities:
            keep = np.zeros_like(keep)
        count = keep.sum(axis=1)
        if short == "T" and text_override is not None:
            mean = text_override
        else:
            mean = np.einsum("ntd,nt->nd", data["array"][name], keep,
                             optimize=True) / np.maximum(count[:, None], 1)
        parts.append(mean.astype(np.float32))
        fractions.append((count / data["content"].sum(axis=1)).astype(np.float32)[:, None])
        flags.append((count > 0).astype(np.float32)[:, None])
    return np.concatenate(parts + fractions + flags, axis=1).astype(np.float32)


def training_missing_view(data: dict, seed: int, epoch: int,
                          eligible_modalities: str = "TAV",
                          require_effective: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    """One reproducible contiguous train-only mask per row; never touches padding."""
    if eligible_modalities not in ("T", "TA", "TV", "TAV"):
        raise ValueError("eligible_modalities must match a supported model input")
    content = data["content"]
    removed = {name: np.zeros_like(content) for name in ("text", "audio", "vision")}
    names = {"T": "text", "A": "audio", "V": "vision"}
    counts = {key: 0 for key in MODALITY_SETS}
    fallback = 0
    effective_rows = 0
    singles = tuple(eligible_modalities)
    doubles = tuple(group for group in ("TA", "TV", "AV") if all(s in eligible_modalities for s in group))
    for row, valid in enumerate(content):
        positions = np.flatnonzero(valid)
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, row]))
        for _ in range(10):
            if require_effective:
                choices = singles if rng.random() < .75 or not doubles else doubles
                group = choices[int(rng.integers(len(choices)))]
            else:
                group = ("TAV"[int(rng.integers(3))] if rng.random() < .75
                         else ("TA", "TV", "AV")[int(rng.integers(3))])
            length = min(len(positions), max(1, math.ceil(RATIOS[int(rng.integers(3))] * len(positions))))
            start = int(rng.integers(len(positions) - length + 1))
            block = positions[start:start + length]
            available = {name: data["observed"][name][row].copy() for name in names.values()}
            lost = any(available[names[short]][block].any() for short in group)
            for short in group:
                available[names[short]][block] = False
            if any(mask.any() for mask in available.values()) and (lost or not require_effective):
                for short in group:
                    removed[names[short]][row, block] = True
                counts[group] += 1
                effective_rows += int(lost)
                break
        else:
            fallback += 1
    if any(np.any(mask & ~content) for mask in removed.values()):
        raise AssertionError("Training mask touched a special or padding position")
    stats = {"epoch": epoch, "fallback_clean": fallback,
             "effective_rows": effective_rows,
             **{f"group_{key}": value for key, value in counts.items()}}
    for name in removed:
        original = int(data["observed"][name].sum())
        stats[f"{name}_newly_lost"] = int((data["observed"][name] & removed[name]).sum())
        stats[f"{name}_newly_lost_ratio"] = stats[f"{name}_newly_lost"] / original if original else None
    return removed, stats


@torch.inference_mode()
def masked_text_means(data: dict, missing: np.ndarray, model: BertModel,
                      device: torch.device, batch_size: int = 64) -> np.ndarray:
    """Replace word IDs with [MASK], re-encode, then pool only unmasked content."""
    tb = data["tb"]
    ids = tb[:, 0].copy()
    ids[missing] = 103
    keep = data["content"] & ~missing
    output = np.empty((len(ids), 768), dtype=np.float32)
    for start in range(0, len(ids), batch_size):
        end = min(len(ids), start + batch_size)
        hidden = model(
            input_ids=torch.as_tensor(ids[start:end], device=device),
            attention_mask=torch.as_tensor(tb[start:end, 1], device=device),
            token_type_ids=torch.as_tensor(tb[start:end, 2], device=device),
        ).last_hidden_state
        weights = torch.as_tensor(keep[start:end], device=device)[..., None]
        output[start:end] = ((hidden * weights).sum(1) / weights.sum(1).clamp_min(1)).cpu().numpy()
    return output


class Baseline(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(128, 3)
        self.regressor = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(x)
        return self.classifier(hidden), 3 * torch.tanh(self.regressor(hidden).squeeze(-1))


class CoverageGate(nn.Module):
    """Fuse pooled modalities using their content coverage and availability."""

    def __init__(self):
        super().__init__()
        self.project = nn.ModuleList([nn.Sequential(nn.Linear(dim, 128), nn.ReLU())
                                      for dim in (768, 74, 35)])
        self.gate = nn.Linear(129, 1)
        self.body = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(128, 3)
        self.regressor = nn.Linear(128, 1)

    def fuse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != 883:
            raise ValueError("Expected 883 pooled features (text, audio, vision, coverage, flags)")
        hidden = torch.stack([layer(part) for layer, part in zip(
            self.project, (x[:, :768], x[:, 768:842], x[:, 842:877]))], dim=1)
        coverage, present = x[:, 877:880], x[:, 880:883] > 0.5
        scores = self.gate(torch.cat((hidden, coverage[..., None]), dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~present, -1e4)
        weights = torch.softmax(scores, dim=1) * present
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (weights[..., None] * hidden).sum(dim=1), weights

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused, _ = self.fuse(x)
        hidden = self.body(fused)
        return self.classifier(hidden), 3 * torch.tanh(self.regressor(hidden).squeeze(-1))


def metrics(labels: np.ndarray, values: np.ndarray,
            logits: np.ndarray, estimate: np.ndarray) -> dict:
    pred = logits.argmax(axis=1)
    f1 = []
    for label in range(3):
        tp = int(np.sum((pred == label) & (labels == label)))
        fp = int(np.sum((pred == label) & (labels != label)))
        fn = int(np.sum((pred != label) & (labels == label)))
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    pearson = float(np.corrcoef(values, estimate)[0, 1]) if np.std(estimate) > 1e-12 else None
    return {"accuracy": float(np.mean(pred == labels)), "macro_f1": float(np.mean(f1)),
            "f1_0": f1[0], "f1_1": f1[1], "f1_2": f1[2],
            "mae": float(np.mean(np.abs(values - estimate))), "pearson": pearson,
            "sign_conflict": float(np.mean(((pred == 0) & (estimate > 0)) |
                                           ((pred == 2) & (estimate < 0))))}


@torch.inference_mode()
def predict(model: Baseline, x: torch.Tensor, batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_values = [], []
    for block in x.split(batch_size):
        logits, values = model(block)
        all_logits.append(logits.cpu().numpy())
        all_values.append(values.cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_values)


def create_panel(valid: dict, bert_dir: Path, device: torch.device,
                 keep_modalities: str = "TAV") -> tuple[np.ndarray, list[dict]]:
    """Prepare the fixed panel once; the same inputs evaluate every epoch."""
    arrays, descriptors = [], []
    bert = None
    try:
        for position in POSITIONS:
            for ratio in RATIOS:
                missing = np.stack([interval_mask(c, position, ratio) for c in valid["content"]])
                text_mean = None
                for modset in MODALITY_SETS:
                    if "T" in modset and text_mean is None:
                        if bert is None:
                            bert = BertModel.from_pretrained(bert_dir, local_files_only=True).to(device).eval()
                            bert.requires_grad_(False)
                        text_mean = masked_text_means(valid, missing, bert, device)
                    arrays.append(pooled(valid, missing, modset,
                                         text_mean if "T" in modset else None,
                                         keep_modalities=keep_modalities))
                    original = sum(int(valid["observed"][m].sum()) for s, m in
                                   (("T", "text"), ("A", "audio"), ("V", "vision")) if s in modset)
                    newly_lost = sum(int((valid["observed"][m] & missing).sum()) for s, m in
                                     (("T", "text"), ("A", "audio"), ("V", "vision")) if s in modset)
                    descriptors.append({"modalities": modset, "position": position, "ratio": ratio,
                                        "n": len(valid["class"]), "masked_positions": int(missing.sum()),
                                        "mean_actual_ratio": float(np.mean(missing.sum(1) / valid["content"].sum(1))),
                                        "newly_lost_observed_ratio": newly_lost / original if original else None})
                print(f"panel prepared: {position} {ratio:.1f}", flush=True)
    finally:
        del bert
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(arrays), descriptors


def evaluate(model: Baseline, clean: torch.Tensor, panel: torch.Tensor,
             valid: dict, descriptors: list[dict]) -> tuple[dict, list[dict]]:
    n = len(valid["class"])
    c_logits, c_value = predict(model, clean)
    clean_metrics = metrics(valid["class"], valid["value"], c_logits, c_value)
    p_logits, p_value = predict(model, panel)
    rows = []
    for i, descriptor in enumerate(descriptors):
        chunk = slice(i * n, (i + 1) * n)
        result = metrics(valid["class"], valid["value"], p_logits[chunk], p_value[chunk])
        rows.append({**descriptor, **result,
                     "delta_accuracy": result["accuracy"] - clean_metrics["accuracy"],
                     "delta_macro_f1": result["macro_f1"] - clean_metrics["macro_f1"],
                     "delta_mae": result["mae"] - clean_metrics["mae"],
                     "delta_pearson": (result["pearson"] - clean_metrics["pearson"]
                                       if result["pearson"] is not None and clean_metrics["pearson"] is not None else None)})
    masked = {key: float(np.mean([row[key] for row in rows])) for key in
              ("accuracy", "macro_f1", "mae")}
    correlations = [row["pearson"] for row in rows if row["pearson"] is not None]
    masked["pearson"] = float(np.mean(correlations)) if correlations else None
    score = .25 * (clean_metrics["mae"] / 3 + 1 - clean_metrics["macro_f1"]) + \
            .75 * (masked["mae"] / 3 + 1 - masked["macro_f1"])
    return {"selection_J": score, "clean": clean_metrics, "masked_mean": masked}, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="128 train / 64 valid, one epoch")
    parser.add_argument("--augment", action="store_true",
                        help="train on a clean and one deterministic contiguous-missing view per row")
    parser.add_argument("--effective-mask", action="store_true",
                        help="sample only model-used modalities and require real observed feature loss")
    parser.add_argument("--modalities", choices=("T", "TA", "TV", "TAV"), default="TAV",
                        help="modalities used by this controlled ablation; T=text, A=audio, V=vision")
    parser.add_argument("--fusion", choices=("concat", "coverage_gate"), default="concat",
                        help="standard MLP concatenation or sample-level coverage gate")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--run-name", type=str, default=None,
                        help="result/checkpoint subdirectory name; use a new name to preserve prior runs")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    if args.effective_mask and not args.augment:
        parser.error("--effective-mask requires --augment")
    if args.run_name is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("run-name must contain only letters, digits, underscores, or hyphens")
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    root = cfg["_engine_dir"]
    source = cfg["_attachment2_root"] / "aligned_50.pkl"
    bert_dir = root / "models" / "bert-base-uncased" / BERT_REV
    if sha256(source) != DATA_SHA256 or sha256(bert_dir / "model.safetensors") != BERT_SHA256:
        raise ValueError("Pinned A2 or BERT SHA256 mismatch; training stopped")
    payload = load_pickle(source)
    train_samples = payload.splits["train"][:128] if args.smoke else payload.splits["train"]
    valid_samples = payload.splits["valid"][:64] if args.smoke else payload.splits["valid"]
    train = prepare(train_samples)
    valid = prepare(valid_samples)
    means, stds = fit_stats(train)
    for data in (train, valid):
        for m in ("audio", "vision"):
            data["array"][m] = (data["array"][m] - means[m]) / stds[m]
            data["array"][m] *= data["observed"][m][..., None]
    train_x = pooled(train, keep_modalities=args.modalities)
    valid_x = pooled(valid, keep_modalities=args.modalities)
    train_labels = train["class"].copy()
    train_values = train["value"].copy()
    del payload, train_samples, valid_samples
    if not args.augment:
        del train
    gc.collect()
    data_seconds = time.perf_counter() - started
    print(f"data prepared train={len(train_x)} valid={len(valid_x)} device={device}", flush=True)
    panel_x, descriptors = create_panel(valid, bert_dir, device, args.modalities)
    panel_seconds = time.perf_counter() - started - data_seconds
    if not np.isfinite(train_x).all() or not np.isfinite(valid_x).all() or not np.isfinite(panel_x).all():
        raise ValueError("Nonfinite model input")
    train_t = torch.as_tensor(train_x, device=device)
    valid_t = torch.as_tensor(valid_x, device=device)
    panel_t = torch.as_tensor(panel_x, device=device)
    labels_t = torch.as_tensor(train_labels, device=device)
    values_t = torch.as_tensor(train_values, device=device)
    counts = np.bincount(train_labels, minlength=3)
    if (counts == 0).any():
        raise ValueError("A class is absent from the selected training subset")
    class_weights = 1 / np.sqrt(counts)
    class_weights /= class_weights.mean()
    weights_t = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    model = (CoverageGate() if args.fusion == "coverage_gate" else Baseline(train_x.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    default_name = (("q2_smoke_augmented" if args.smoke else "q2_augmented")
                    if args.augment else
                    ("q2_smoke_baseline" if args.smoke else "q2_first_baseline"))
    run_name = args.run_name or default_name + ("" if args.modalities == "TAV" else f"_{args.modalities}") + \
        ("" if args.fusion == "concat" else "_gate")
    result_dir = root / "results" / run_name
    checkpoint_dir = root / "checkpoints" / run_name
    if (checkpoint_dir.exists() or
            (result_dir.exists() and any(
                file.name not in ("terminal.log", "exit_code.txt") for file in result_dir.iterdir()
            ))):
        raise FileExistsError(f"Use a new --run-name to preserve previous results: {run_name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_j, best_epoch, stale = float("inf"), 0, 0
    best_result, best_rows = None, None
    history = []
    mask_history = []
    max_epochs = 1 if args.smoke else args.epochs
    bert = None
    if args.augment:
        bert = BertModel.from_pretrained(bert_dir, local_files_only=True).to(device).eval()
        bert.requires_grad_(False)
    train_started = time.perf_counter()
    optimizer_steps = 0
    for epoch in range(1, max_epochs + 1):
        masked_t = None
        if args.augment:
            removed, mask_stats = training_missing_view(
                train, args.seed, epoch, args.modalities, args.effective_mask)
            mask_history.append(mask_stats)
            text_means = train_x[:, :768].copy()
            text_rows = np.flatnonzero(removed["text"].any(axis=1))
            if len(text_rows):
                subset = {"tb": train["tb"][text_rows], "content": train["content"][text_rows]}
                text_means[text_rows] = masked_text_means(
                    subset, removed["text"][text_rows], bert, device)
            masked_x = pooled(train, removed, text_override=text_means,
                              keep_modalities=args.modalities)
            if masked_x.shape != train_x.shape or not np.isfinite(masked_x).all():
                raise ValueError(f"Invalid augmented input at epoch {epoch}")
            masked_t = torch.as_tensor(masked_x, device=device)
            print(f"mask epoch={epoch} T={mask_stats['group_T']} A={mask_stats['group_A']} "
                  f"V={mask_stats['group_V']} TA={mask_stats['group_TA']} "
                  f"TV={mask_stats['group_TV']} AV={mask_stats['group_AV']} "
                  f"fallback={mask_stats['fallback_clean']} text_reencoded={len(text_rows)}",
                  flush=True)
        model.train()
        order = torch.randperm(len(train_t), device=device)
        loss_sum = class_sum = reg_sum = 0.0
        for indices in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            logits, estimate = model(train_t[indices])
            class_loss = nn.functional.cross_entropy(logits, labels_t[indices], weight=weights_t)
            reg_loss = nn.functional.huber_loss(estimate, values_t[indices], delta=1.0)
            if masked_t is not None:
                masked_logits, masked_estimate = model(masked_t[indices])
                class_loss = class_loss + nn.functional.cross_entropy(
                    masked_logits, labels_t[indices], weight=weights_t)
                reg_loss = reg_loss + nn.functional.huber_loss(
                    masked_estimate, values_t[indices], delta=1.0)
            loss = class_loss + reg_loss
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            loss_sum += float(loss.detach()) * len(indices)
            class_sum += float(class_loss.detach()) * len(indices)
            reg_sum += float(reg_loss.detach()) * len(indices)
        result, rows = evaluate(model, valid_t, panel_t, valid, descriptors)
        if result["selection_J"] < best_j - 1e-4:
            best_j, best_epoch, stale = result["selection_J"], epoch, 0
            best_result, best_rows = result, rows
            torch.save({"state_dict": model.state_dict(), "input_dim": train_x.shape[1],
                        "audio_mean": means["audio"], "audio_std": stds["audio"],
                        "vision_mean": means["vision"], "vision_std": stds["vision"],
                        "epoch": epoch, "seed": args.seed, "selection_J": best_j,
                        "data_sha256": DATA_SHA256, "bert_sha256": BERT_SHA256,
                        "augment": args.augment, "effective_mask": args.effective_mask,
                        "modalities": args.modalities,
                        "fusion": args.fusion},
                       checkpoint_dir / "best.pt")
        else:
            stale += 1
        record = {"epoch": epoch, "train_loss": loss_sum / len(train_t),
                  "train_class_loss": class_sum / len(train_t),
                  "train_reg_loss": reg_sum / len(train_t),
                  "selection_J": result["selection_J"], "best_epoch": best_epoch,
                  "clean_accuracy": result["clean"]["accuracy"],
                  "clean_macro_f1": result["clean"]["macro_f1"],
                  "clean_mae": result["clean"]["mae"],
                  "clean_pearson": result["clean"]["pearson"],
                  "masked_accuracy": result["masked_mean"]["accuracy"],
                  "masked_macro_f1": result["masked_mean"]["macro_f1"],
                  "masked_mae": result["masked_mean"]["mae"],
                  "masked_pearson": result["masked_mean"]["pearson"],
                  "elapsed_seconds": time.perf_counter() - train_started}
        history.append(record)
        clean_r = "NA" if record["clean_pearson"] is None else f"{record['clean_pearson']:.3f}"
        masked_r = "NA" if record["masked_pearson"] is None else f"{record['masked_pearson']:.3f}"
        print(f"epoch {epoch:02d}/{max_epochs} | loss={record['train_loss']:.4f} "
              f"class={record['train_class_loss']:.4f} reg={record['train_reg_loss']:.4f} "
              f"J={record['selection_J']:.4f} best={best_epoch} "
              f"time={record['elapsed_seconds']:.1f}s", flush=True)
        print(f"  valid clean:    Acc={record['clean_accuracy']:.3f} "
              f"F1={record['clean_macro_f1']:.3f} MAE={record['clean_mae']:.3f} "
              f"Pearson={clean_r}", flush=True)
        print(f"  valid masked54: Acc={record['masked_accuracy']:.3f} "
              f"F1={record['masked_macro_f1']:.3f} MAE={record['masked_mae']:.3f} "
              f"Pearson={masked_r}", flush=True)
        if stale >= args.patience:
            print(f"early stop at epoch {epoch}; best={best_epoch}", flush=True)
            break
    train_seconds = time.perf_counter() - train_started
    with (result_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with (result_dir / "scenario_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=best_rows[0].keys())
        writer.writeheader()
        writer.writerows(best_rows)
    if mask_history:
        with (result_dir / "mask_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=mask_history[0].keys())
            writer.writeheader()
            writer.writerows(mask_history)
    summary = {"status": "A_AUGMENTED_CANDIDATE" if args.augment else "FIRST_BASELINE_ONLY",
               "model": "A_coverage_gate" if args.fusion == "coverage_gate" else "A_clean_masked_mean_MLP",
               "augment": args.augment, "effective_mask": args.effective_mask,
               "modalities": args.modalities, "fusion": args.fusion,
               "train_n": len(train_t), "valid_n": len(valid_t), "scenarios": len(descriptors),
               "best_epoch": best_epoch, "epochs_ran": len(history), "best": best_result,
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "class_counts_train": counts.tolist(), "class_weights": class_weights.tolist(),
               "seed": args.seed, "device": str(device), "seconds": time.perf_counter() - started,
               "max_epochs": max_epochs, "patience": args.patience, "run_name": run_name,
               "optimizer_steps": optimizer_steps, "train_input_shape": list(train_x.shape),
               "train_input_bytes": int(train_x.nbytes), "data_seconds": data_seconds,
               "panel_seconds": panel_seconds, "train_and_evaluate_seconds": train_seconds,
               "source": str(source), "source_sha256": DATA_SHA256,
               "bert_snapshot": str(bert_dir), "bert_sha256": BERT_SHA256,
               "torch": torch.__version__, "transformers": transformers_version,
               "command": "python -m src.q2_train_baseline" +
                          (" --augment" if args.augment else "") +
                          (" --effective-mask" if args.effective_mask else "") +
                          (" --smoke" if args.smoke else "") +
                          f" --modalities {args.modalities}" +
                          f" --fusion {args.fusion}" +
                          f" --seed {args.seed} --epochs {args.epochs} --patience {args.patience} --run-name {run_name}",
               "checkpoint": str(checkpoint_dir / "best.pt"),
               "mask_sampling": ({"single_probability": 0.75, "double_probability": 0.25,
                                  "effective_only": args.effective_mask,
                                  "eligible_modalities": args.modalities if args.effective_mask else "TAV",
                                  "ratios": RATIOS, "seed_derivation": "[seed, epoch, train_row]",
                                  "text": "replace IDs with [MASK], frozen BERT recode"}
                                 if args.augment else None),
               "note": "A2 train fits weights/statistics, A2 valid selects epoch. A2 test/A3/Q1 unused."
               }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(json.dumps({"output": str(result_dir), "best_epoch": best_epoch,
                      "best_J": best_j, "seconds": summary["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
