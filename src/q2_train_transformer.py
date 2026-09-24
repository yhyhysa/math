"""Q2 model B: position-aware gated fusion and a small Transformer.

Run from 工程: python -m src.q2_train_transformer --selfcheck
or: python -m src.q2_train_transformer --smoke --layers 1 --run-name q2_b_try
Only A2 train fits weights/statistics; A2 valid selects the checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
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
from src.q2_train_baseline import (
    BERT_REV, BERT_SHA256, DATA_SHA256, MODALITY_SETS, POSITIONS, RATIOS,
    fit_stats, interval_mask, metrics, prepare, sha256, training_missing_view,
)
from src.q2_missing_patterns import training_multi_block_view

NAMES = ("text", "audio", "vision")
SHORT = {"T": "text", "A": "audio", "V": "vision"}


class GatedTemporalTransformer(nn.Module):
    def __init__(self, layers: int, mask_empty: bool = False):
        super().__init__()
        self.mask_empty = mask_empty
        self.project = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 128), nn.LayerNorm(128))
            for dim in (768, 74, 35)
        ])
        self.gate = nn.Linear(128, 1)
        self.empty = nn.Parameter(torch.zeros(128))
        self.presence = nn.Linear(3, 128, bias=False)
        self.position = nn.Embedding(50, 128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(128, 3)
        self.regressor = nn.Linear(128, 1)

    def forward(self, view: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        present = view["available"]
        content = view["content"]
        if present.shape[1:] != (50, 3) or content.shape != present.shape[:2]:
            raise ValueError("Expected (batch,50,3) availability and (batch,50) content")
        hidden = torch.stack([
            projector(view[name]) for projector, name in zip(self.project, NAMES)
        ], dim=2)
        scores = self.gate(torch.tanh(hidden)).squeeze(-1)
        scores = scores.masked_fill(~present, -1e4)
        weights = torch.softmax(scores, dim=-1) * present
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fused = (weights[..., None] * hidden).sum(dim=2)
        any_present = present.any(dim=-1)
        fused = torch.where(any_present[..., None], fused, self.empty)
        fused = fused + self.presence(present.float())
        fused = fused + self.position(torch.arange(50, device=fused.device))[None]
        active = content & any_present if self.mask_empty else content
        if self.mask_empty:
            # Preserve one learned empty token only when the whole clip is unobservable.
            empty_clips = ~active.any(dim=1)
            if empty_clips.any():
                active = active.clone()
                first_content = content.int().argmax(dim=1)
                active[empty_clips, first_content[empty_clips]] = True
        sequence = self.encoder(fused, src_key_padding_mask=~active)
        pooled = (sequence * active[..., None]).sum(dim=1) / active.sum(dim=1, keepdim=True).clamp_min(1)
        return self.classifier(pooled), 3 * torch.tanh(self.regressor(pooled).squeeze(-1))


class SeparateTemporalTransformer(nn.Module):
    """Encode each aligned modality in time before sample-level fusion."""

    def __init__(self, layers: int = 1, fusion: str = "concat", auxiliary: bool = False):
        super().__init__()
        if fusion not in ("concat", "reliability_gate"):
            raise ValueError("Unknown fusion")
        self.fusion = fusion
        width = 96
        self.project = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, width), nn.LayerNorm(width))
            for dim in (768, 74, 35)
        ])
        self.position = nn.Embedding(50, width)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=width, nhead=4, dim_feedforward=192,
                    dropout=.1, batch_first=True,
                ), num_layers=layers, enable_nested_tensor=False,
            ) for _ in NAMES
        ])
        if fusion == "reliability_gate":
            self.gate = nn.Linear(width + 1, 1)
            body_input = width + 3
        else:
            body_input = width * 3 + 3
        self.body = nn.Sequential(nn.Linear(body_input, 128), nn.ReLU(), nn.Dropout(.1))
        self.classifier = nn.Linear(128, 3)
        self.regressor = nn.Linear(128, 1)
        self.auxiliary = auxiliary
        if auxiliary:
            self.aux_heads = nn.ModuleList([nn.Linear(width, 4) for _ in NAMES])

    def forward(self, view: dict[str, torch.Tensor], return_aux: bool = False):
        content = view["content"]
        available = view["available"] & content[..., None]
        if available.shape[1:] != (50, 3) or content.shape != available.shape[:2]:
            raise ValueError("Expected aligned (batch, 50) and (batch, 50, 3) masks")
        position = self.position(torch.arange(50, device=content.device))[None]
        first_content = content.int().argmax(dim=1)
        vectors, coverage, present = [], [], []
        for index, name in enumerate(NAMES):
            active = available[:, :, index]
            safe = active.clone()
            empty = ~active.any(dim=1)
            if empty.any():
                safe[empty, first_content[empty]] = True
            hidden = self.encoder[index](self.project[index](view[name]) + position,
                                         src_key_padding_mask=~safe)
            count = active.sum(dim=1, keepdim=True)
            vectors.append((hidden * active[..., None]).sum(dim=1) / count.clamp_min(1))
            coverage.append(count / content.sum(dim=1, keepdim=True).clamp_min(1))
            present.append(~empty)
        cover = torch.cat(coverage, dim=1)
        if self.fusion == "reliability_gate":
            scores = torch.stack([
                self.gate(torch.cat((vector, fraction), dim=1)).squeeze(-1)
                for vector, fraction in zip(vectors, coverage)
            ], dim=1)
            flags = torch.stack(present, dim=1)
            scores = scores.masked_fill(~flags, -1e4)
            weights = torch.softmax(scores, dim=1) * flags
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            fused = torch.stack(vectors, dim=1).mul(weights[..., None]).sum(dim=1)
            joint = torch.cat((fused, cover), dim=1)
        else:
            joint = torch.cat((*vectors, cover), dim=1)
        shared = self.body(joint)
        logits = self.classifier(shared)
        estimate = 3 * torch.tanh(self.regressor(shared).squeeze(-1))
        if not return_aux:
            return logits, estimate
        if not self.auxiliary:
            raise ValueError("Auxiliary heads were not enabled")
        auxiliary = []
        for head, vector, flag in zip(self.aux_heads, vectors, present):
            output = head(vector)
            auxiliary.append((output[:, :3], 3 * torch.tanh(output[:, 3]), flag))
        return logits, estimate, auxiliary


def auxiliary_loss(rows, labels: torch.Tensor, values: torch.Tensor,
                   class_weights: torch.Tensor) -> torch.Tensor:
    losses = []
    for logits, estimate, present in rows:
        if present.any():
            classification = nn.functional.cross_entropy(
                logits, labels, weight=class_weights, reduction="none")
            regression = nn.functional.huber_loss(estimate, values, delta=1.0, reduction="none")
            losses.append(((classification + regression) * present).sum() / present.sum())
    return torch.stack(losses).mean() if losses else values.new_zeros(())


def selfcheck() -> None:
    torch.manual_seed(19)
    view = {name: torch.randn(3, 50, dim) for name, dim in zip(NAMES, (768, 74, 35))}
    view["available"] = torch.zeros(3, 50, 3, dtype=torch.bool)
    view["content"] = torch.zeros(3, 50, dtype=torch.bool)
    view["content"][:, :4] = True
    view["available"][:2, 0, 0] = True
    view["available"][:2, 2, 1] = True
    view["available"][:2, 3, 2] = True
    # Position 1 has no modality; positions 4..49 are padding; row 2 is wholly empty.
    for mask_empty in (False, True):
        model = GatedTemporalTransformer(layers=2, mask_empty=mask_empty)
        model.eval()
        with torch.no_grad():
            logits, values = model(view)
            changed = {key: val.clone() for key, val in view.items()}
            changed["text"][:, 4:] += 1000
            changed["audio"][:, 4:] -= 1000
            changed["vision"][:, 4:] += 1000
            changed_logits, changed_values = model(changed)
            with_extra_empty = {key: val.clone() for key, val in view.items()}
            with_extra_empty["content"][:2, 4] = True
            extra_logits, extra_values = model(with_extra_empty)
        assert logits.shape == (3, 3) and values.shape == (3,)
        assert torch.isfinite(logits).all() and torch.isfinite(values).all()
        assert (values.abs() <= 3).all()
        assert torch.allclose(logits, changed_logits, atol=1e-5)
        assert torch.allclose(values, changed_values, atol=1e-5)
        if mask_empty:
            assert torch.allclose(logits[:2], extra_logits[:2], atol=1e-5)
            assert torch.allclose(values[:2], extra_values[:2], atol=1e-5)
        model.train()
        train_logits, train_values = model(view)
        (train_logits.square().mean() + train_values.square().mean()).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print("PASS model-B old/fixed empty mask, padding isolation, gradients, range", flush=True)
    for fusion in ("concat", "reliability_gate"):
        model = SeparateTemporalTransformer(layers=1, fusion=fusion)
        model.eval()
        with torch.no_grad():
            logits, values = model(view)
            changed = {key: val.clone() for key, val in view.items()}
            changed["text"][:, 4:] += 1000
            changed["vision"][:, 4:] -= 1000
            changed_logits, changed_values = model(changed)
        assert logits.shape == (3, 3) and values.shape == (3,)
        assert torch.isfinite(logits).all() and torch.isfinite(values).all()
        assert (values.abs() <= 3).all()
        assert torch.allclose(logits, changed_logits, atol=1e-5)
        assert torch.allclose(values, changed_values, atol=1e-5)
        model.train()
        train_logits, train_values = model(view)
        (train_logits.square().mean() + train_values.square().mean()).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print("PASS separate temporal concat/gate, missing-position isolation, gradients, range", flush=True)
    model = SeparateTemporalTransformer(layers=1, fusion="reliability_gate", auxiliary=True)
    logits, values, rows = model(view, return_aux=True)
    labels = torch.tensor([0, 1, 2])
    weights = torch.ones(3)
    aux_value = auxiliary_loss(rows, labels, torch.zeros(3), weights)
    changed_rows = [(logit.clone(), value.clone(), flag) for logit, value, flag in rows]
    for logit, value, flag in changed_rows:
        logit[~flag] += 1000
        value[~flag] += 1000
    assert torch.allclose(aux_value, auxiliary_loss(changed_rows, labels, torch.zeros(3), weights))
    loss = logits.square().mean() + values.square().mean() + aux_value
    loss.backward()
    assert len(rows) == 3 and all(row[0].shape == (3, 3) for row in rows)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print("PASS auxiliary heads, unavailable-sample masking, gradients", flush=True)


@torch.inference_mode()
def recode_text(tb: np.ndarray, missing: np.ndarray, bert: BertModel,
                device: torch.device, batch_size: int = 64) -> np.ndarray:
    ids = tb[:, 0].copy()
    ids[missing] = 103
    out = np.empty((len(ids), 50, 768), dtype=np.float32)
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        hidden = bert(
            input_ids=torch.as_tensor(ids[start:end], device=device),
            attention_mask=torch.as_tensor(tb[start:end, 1], device=device),
            token_type_ids=torch.as_tensor(tb[start:end, 2], device=device),
        ).last_hidden_state
        out[start:end] = hidden.cpu().numpy()
    return out


def as_view(data: dict, device: torch.device) -> dict[str, torch.Tensor]:
    view = {name: torch.as_tensor(data["array"][name], device=device) for name in NAMES}
    view["available"] = torch.as_tensor(
        np.stack([data["observed"][name] for name in NAMES], axis=-1), device=device,
    )
    view["content"] = torch.as_tensor(data["content"], device=device)
    return view


def slice_view(view: dict[str, torch.Tensor], indices: torch.Tensor | slice) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in view.items()}


@torch.inference_mode()
def predict(model: nn.Module, view: dict[str, torch.Tensor],
            batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, values = [], []
    for start in range(0, len(view["content"]), batch_size):
        block = slice_view(view, slice(start, start + batch_size))
        q, y = model(block)
        logits.append(q.cpu().numpy())
        values.append(y.cpu().numpy())
    return np.concatenate(logits), np.concatenate(values)


def make_panel(valid: dict, valid_view: dict[str, torch.Tensor], bert: BertModel,
               device: torch.device) -> list[tuple[dict, torch.Tensor, torch.Tensor]]:
    """Nine frozen BERT recodes and 54 cheap mask specifications, kept on GPU."""
    panel = []
    n = len(valid["class"])
    for position in POSITIONS:
        for ratio in RATIOS:
            missing = np.stack([interval_mask(c, position, ratio) for c in valid["content"]])
            text = torch.as_tensor(recode_text(valid["tb"], missing, bert, device), device=device)
            for group in MODALITY_SETS:
                avail = valid_view["available"].clone()
                missing_t = torch.as_tensor(missing, device=device)
                for short in group:
                    avail[:, :, NAMES.index(SHORT[short])] &= ~missing_t
                original = sum(int(valid["observed"][SHORT[s]].sum()) for s in group)
                lost = sum(int((valid["observed"][SHORT[s]] & missing).sum()) for s in group)
                descriptor = {
                    "modalities": group, "position": position, "ratio": ratio, "n": n,
                    "masked_positions": int(missing.sum()),
                    "mean_actual_ratio": float(np.mean(missing.sum(1) / valid["content"].sum(1))),
                    "newly_lost_observed_ratio": lost / original if original else None,
                    "unobservable_positions": int((valid_view["content"] & ~avail.any(dim=-1)).sum().item()),
                }
                panel.append((descriptor, avail, text if "T" in group else valid_view["text"]))
            print(f"panel prepared: {position} {ratio:.1f}", flush=True)
    assert len(panel) == 54
    return panel


def masked_train_view(train: dict, clean: dict[str, torch.Tensor], removed: dict[str, np.ndarray],
                      bert: BertModel, device: torch.device) -> tuple[dict[str, torch.Tensor], int]:
    view = dict(clean)
    available = clean["available"].clone()
    for index, name in enumerate(NAMES):
        available[:, :, index] &= ~torch.as_tensor(removed[name], device=device)
    view["available"] = available
    rows = np.flatnonzero(removed["text"].any(axis=1))
    if len(rows):
        recoded = recode_text(train["tb"][rows], removed["text"][rows], bert, device)
        text = clean["text"].clone()
        text[rows] = torch.as_tensor(recoded, device=device)
        view["text"] = text
    return view, len(rows)


def evaluate(model: nn.Module, valid: dict, clean: dict[str, torch.Tensor],
             panel: list[tuple[dict, torch.Tensor, torch.Tensor]]) -> tuple[dict, list[dict]]:
    clean_logits, clean_values = predict(model, clean)
    clean_m = metrics(valid["class"], valid["value"], clean_logits, clean_values)
    rows = []
    for descriptor, available, text in panel:
        view = {**clean, "available": available, "text": text}
        logits, values = predict(model, view)
        m = metrics(valid["class"], valid["value"], logits, values)
        rows.append({**descriptor, **m,
                     "delta_accuracy": m["accuracy"] - clean_m["accuracy"],
                     "delta_macro_f1": m["macro_f1"] - clean_m["macro_f1"],
                     "delta_mae": m["mae"] - clean_m["mae"],
                     "delta_pearson": (m["pearson"] - clean_m["pearson"]
                                       if m["pearson"] is not None and clean_m["pearson"] is not None else None)})
    masked = {key: float(np.mean([row[key] for row in rows]))
              for key in ("accuracy", "macro_f1", "mae")}
    correlations = [row["pearson"] for row in rows if row["pearson"] is not None]
    masked["pearson"] = float(np.mean(correlations)) if correlations else None
    score = .25 * (clean_m["mae"] / 3 + 1 - clean_m["macro_f1"]) + \
            .75 * (masked["mae"] / 3 + 1 - masked["macro_f1"])
    return {"selection_J": score, "clean": clean_m, "masked_mean": masked}, rows


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true", help="synthetic model check; no input files")
    parser.add_argument("--smoke", action="store_true", help="128 train / 64 valid, one epoch")
    parser.add_argument("--augment", action="store_true", help="clean plus contiguous-missing train view")
    parser.add_argument("--effective-mask", action="store_true",
                        help="require each synthetic mask to remove observed model input")
    parser.add_argument("--architecture", choices=("early", "separate"), default="early")
    parser.add_argument("--fusion", choices=("concat", "reliability_gate"), default="concat",
                        help="sample-level fusion for separate temporal encoders")
    parser.add_argument("--mask-empty", action="store_true",
                        help="exclude fully unobservable content positions from attention and pooling")
    parser.add_argument("--layers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--auxiliary-weight", type=float, default=0.0)
    parser.add_argument("--mask-pattern", choices=("single", "multi"), default="single")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if not np.isfinite(args.consistency_weight) or args.consistency_weight < 0:
        parser.error("consistency-weight must be finite and nonnegative")
    if not np.isfinite(args.auxiliary_weight) or args.auxiliary_weight < 0:
        parser.error("auxiliary-weight must be finite and nonnegative")
    if args.consistency_weight and not args.augment:
        parser.error("consistency-weight requires --augment")
    if args.effective_mask and not args.augment:
        parser.error("--effective-mask requires --augment")
    if args.architecture == "early" and args.fusion != "concat":
        parser.error("--fusion applies only to --architecture separate")
    if args.auxiliary_weight and args.architecture != "separate":
        parser.error("auxiliary-weight applies only to --architecture separate")
    if args.mask_pattern == "multi" and (not args.augment or not args.effective_mask):
        parser.error("multi mask pattern requires --augment --effective-mask")
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
    family = "b" if args.architecture == "early" else "c"
    run_name = args.run_name or f"q2_{family}_{'aug' if args.augment else 'clean'}_{args.layers}layer_{args.fusion}_{args.seed}"
    result_dir = root / "results" / run_name
    checkpoint_dir = root / "checkpoints" / run_name
    if (checkpoint_dir.exists() or
            (result_dir.exists() and any(
                file.name not in ("terminal.log", "exit_code.txt") for file in result_dir.iterdir()
            ))):
        raise FileExistsError(f"Use a new --run-name to preserve previous results: {run_name}")
    if sha256(source) != DATA_SHA256 or sha256(bert_dir / "model.safetensors") != BERT_SHA256:
        raise ValueError("Pinned A2 or BERT SHA256 mismatch; training stopped")
    payload = load_pickle(source)
    train_samples = payload.splits["train"][:128] if args.smoke else payload.splits["train"]
    valid_samples = payload.splits["valid"][:64] if args.smoke else payload.splits["valid"]
    train = prepare(train_samples)
    valid = prepare(valid_samples)
    means, stds = fit_stats(train)
    for data in (train, valid):
        for name in ("audio", "vision"):
            data["array"][name] = (data["array"][name] - means[name]) / stds[name]
            data["array"][name] *= data["observed"][name][..., None]
    del payload, train_samples, valid_samples
    gc.collect()
    print(f"data prepared train={len(train['class'])} valid={len(valid['class'])} device={device}", flush=True)
    data_seconds = time.perf_counter() - started
    train_view = as_view(train, device)
    valid_view = as_view(valid, device)
    bert = BertModel.from_pretrained(bert_dir, local_files_only=True).to(device).eval()
    bert.requires_grad_(False)
    panel = make_panel(valid, valid_view, bert, device)
    if not args.augment:
        del bert
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    panel_seconds = time.perf_counter() - started - data_seconds
    if any(not np.isfinite(train["array"][name]).all() or not np.isfinite(valid["array"][name]).all()
           for name in NAMES):
        raise ValueError("Nonfinite model input")
    labels = torch.as_tensor(train["class"], device=device)
    values = torch.as_tensor(train["value"], device=device)
    counts = np.bincount(train["class"], minlength=3)
    if (counts == 0).any():
        raise ValueError("A class is absent from the selected training subset")
    weights = 1 / np.sqrt(counts)
    weights /= weights.mean()
    weights_t = torch.as_tensor(weights, device=device, dtype=torch.float32)
    model = (GatedTemporalTransformer(args.layers, mask_empty=args.mask_empty)
             if args.architecture == "early" else
             SeparateTemporalTransformer(args.layers, fusion=args.fusion,
                                         auxiliary=args.auxiliary_weight > 0)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True)
    best_j, best_epoch, stale = float("inf"), 0, 0
    best_result, best_rows = None, None
    history, mask_history = [], []
    max_epochs = 1 if args.smoke else args.epochs
    train_started = time.perf_counter()
    optimizer_steps = 0
    for epoch in range(1, max_epochs + 1):
        masked_view = None
        if args.augment:
            if args.mask_pattern == "multi":
                removed, mask_stats = training_multi_block_view(train, args.seed, epoch)
            else:
                removed, mask_stats = training_missing_view(
                    train, args.seed, epoch, "TAV", args.effective_mask)
            masked_view, text_count = masked_train_view(train, train_view, removed, bert, device)
            mask_history.append(mask_stats)
            print(f"mask epoch={epoch} T={mask_stats['group_T']} A={mask_stats['group_A']} "
                  f"V={mask_stats['group_V']} TA={mask_stats['group_TA']} "
                  f"TV={mask_stats['group_TV']} AV={mask_stats['group_AV']} "
                  f"two_blocks={mask_stats.get('two_block_rows', 0)} "
                  f"fallback={mask_stats['fallback_clean']} text_reencoded={text_count}", flush=True)
        model.train()
        order = torch.randperm(len(labels), device=device)
        loss_sum = class_sum = reg_sum = consistency_sum = auxiliary_sum = 0.0
        for indices in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            if args.auxiliary_weight:
                logits, estimate, clean_aux = model(slice_view(train_view, indices), return_aux=True)
                aux_loss = auxiliary_loss(clean_aux, labels[indices], values[indices], weights_t)
            else:
                logits, estimate = model(slice_view(train_view, indices))
                aux_loss = torch.zeros((), device=device)
            class_loss = nn.functional.cross_entropy(logits, labels[indices], weight=weights_t)
            reg_loss = nn.functional.huber_loss(estimate, values[indices], delta=1.0)
            consistency_loss = torch.zeros((), device=device)
            if masked_view is not None:
                if args.auxiliary_weight:
                    masked_logits, masked_estimate, masked_aux = model(
                        slice_view(masked_view, indices), return_aux=True)
                    aux_loss = (aux_loss + auxiliary_loss(
                        masked_aux, labels[indices], values[indices], weights_t)) / 2
                else:
                    masked_logits, masked_estimate = model(slice_view(masked_view, indices))
                class_loss = class_loss + nn.functional.cross_entropy(
                    masked_logits, labels[indices], weight=weights_t)
                reg_loss = reg_loss + nn.functional.huber_loss(masked_estimate, values[indices], delta=1.0)
                if args.consistency_weight:
                    consistency_loss = nn.functional.kl_div(
                        nn.functional.log_softmax(masked_logits, dim=-1),
                        nn.functional.softmax(logits.detach(), dim=-1), reduction="batchmean",
                    ) + nn.functional.huber_loss(masked_estimate / 3, estimate.detach() / 3, delta=1.0)
            loss = (class_loss + reg_loss + args.consistency_weight * consistency_loss
                    + args.auxiliary_weight * aux_loss)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            loss_sum += float(loss.detach()) * len(indices)
            class_sum += float(class_loss.detach()) * len(indices)
            reg_sum += float(reg_loss.detach()) * len(indices)
            consistency_sum += float(consistency_loss.detach()) * len(indices)
            auxiliary_sum += float(aux_loss.detach()) * len(indices)
        if masked_view is not None:
            del masked_view
        result, rows = evaluate(model, valid, valid_view, panel)
        if result["selection_J"] < best_j - 1e-4:
            best_j, best_epoch, stale = result["selection_J"], epoch, 0
            best_result, best_rows = result, rows
            torch.save({"state_dict": model.state_dict(), "layers": args.layers,
                        "audio_mean": means["audio"], "audio_std": stds["audio"],
                        "vision_mean": means["vision"], "vision_std": stds["vision"],
                        "epoch": epoch, "seed": args.seed, "selection_J": best_j,
                        "data_sha256": DATA_SHA256, "bert_sha256": BERT_SHA256,
                        "augment": args.augment,
                        "mask_empty": args.mask_empty if args.architecture == "early" else True,
                        "architecture": args.architecture, "fusion": args.fusion,
                        "effective_mask": args.effective_mask,
                        "learning_rate": args.learning_rate,
                        "consistency_weight": args.consistency_weight,
                        "auxiliary_weight": args.auxiliary_weight,
                        "mask_pattern": args.mask_pattern}, checkpoint_dir / "best.pt")
        else:
            stale += 1
        record = {"epoch": epoch, "train_loss": loss_sum / len(labels),
                  "train_class_loss": class_sum / len(labels),
                  "train_reg_loss": reg_sum / len(labels),
                  "train_consistency_loss": consistency_sum / len(labels),
                  "train_auxiliary_loss": auxiliary_sum / len(labels),
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
              f"cons={record['train_consistency_loss']:.4f} aux={record['train_auxiliary_loss']:.4f} "
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
    save_csv(result_dir / "history.csv", history)
    save_csv(result_dir / "scenario_metrics.csv", best_rows)
    if mask_history:
        save_csv(result_dir / "mask_history.csv", mask_history)
    summary = {"status": "TEMPORAL_CANDIDATE",
               "model": ("B_gated_temporal_transformer" if args.architecture == "early"
                         else "C_separate_temporal_transformer"),
               "layers": args.layers, "augment": args.augment,
               "mask_empty": args.mask_empty if args.architecture == "early" else True,
               "architecture": args.architecture, "fusion": args.fusion,
               "effective_mask": args.effective_mask,
               "train_n": len(labels), "valid_n": len(valid["class"]), "scenarios": len(panel),
               "best_epoch": best_epoch, "epochs_ran": len(history), "best": best_result,
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "class_counts_train": counts.tolist(), "class_weights": weights.tolist(),
               "seed": args.seed, "device": str(device), "seconds": time.perf_counter() - started,
               "max_epochs": max_epochs, "patience": args.patience,
               "learning_rate": args.learning_rate, "run_name": run_name,
               "consistency_weight": args.consistency_weight,
               "auxiliary_weight": args.auxiliary_weight, "mask_pattern": args.mask_pattern,
               "optimizer_steps": optimizer_steps, "data_seconds": data_seconds,
               "panel_seconds": panel_seconds, "train_and_evaluate_seconds": train_seconds,
               "source": str(source), "source_sha256": DATA_SHA256,
               "bert_snapshot": str(bert_dir), "bert_sha256": BERT_SHA256,
               "torch": torch.__version__, "transformers": transformers_version,
               "command": "python -m src.q2_train_transformer" +
                          (" --augment" if args.augment else "") +
                          (" --effective-mask" if args.effective_mask else "") +
                          (" --mask-empty" if args.mask_empty else "") +
                          (" --smoke" if args.smoke else "") +
                          f" --architecture {args.architecture} --fusion {args.fusion}" +
                          f" --layers {args.layers} --seed {args.seed} --epochs {args.epochs} "
                          f"--learning-rate {args.learning_rate} "
                          f"--consistency-weight {args.consistency_weight} "
                          f"--auxiliary-weight {args.auxiliary_weight} "
                          f"--mask-pattern {args.mask_pattern} "
                          f"--patience {args.patience} --run-name {run_name}",
               "checkpoint": str(checkpoint_dir / "best.pt"),
               "mask_sampling": ({"pattern": args.mask_pattern,
                                  "single_probability": (.75 if args.mask_pattern == "single" else None),
                                  "double_probability": (.25 if args.mask_pattern == "single" else None),
                                  "effective_only": args.effective_mask,
                                  "eligible_modalities": "TAV",
                                  "ratios": RATIOS, "seed_derivation": "[seed, epoch, train_row]",
                                  "text": "replace IDs with [MASK], frozen BERT recode"}
                                 if args.augment else None),
               "note": "A2 train fits weights/statistics, A2 valid selects epoch. A2 test/A3/Q1 unused."}
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(result_dir), "best_epoch": best_epoch,
                      "best_J": best_j, "seconds": summary["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
